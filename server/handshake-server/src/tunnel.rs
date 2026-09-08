//! The seam between a completed handshake and the WireGuard tunnel —
//! `PROTOCOL.md` §1 and §6.
//!
//! `PskInstaller` installs (or, on rekey, swaps) a peer's preshared key.
//! `WgCli` shells out to `wg set`; `DryRun` only logs and is the default when
//! `--wg-interface` is not given (tests, and dev hosts without WireGuard or
//! `CAP_NET_ADMIN`).

use std::collections::HashMap;
use std::io::Write;
use std::net::Ipv4Addr;
use std::process::{Command, Stdio};
use std::sync::Mutex;

use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;

use crate::wire::AlgoCode;

fn b64(bytes: &[u8]) -> String {
    B64.encode(bytes)
}

/// Decode a 32-byte WireGuard key from base64.
pub fn wg_key_from_base64(s: &str) -> Option<[u8; 32]> {
    let v = B64.decode(s.trim()).ok()?;
    v.try_into().ok()
}

// -------------------------------------------------------------------------
// tunnel settings
// -------------------------------------------------------------------------

#[derive(Clone)]
pub struct TunnelSettings {
    /// The server's own WireGuard public key (sent to the client in `ServerFinish`).
    pub server_wg_pubkey: [u8; 32],
    /// UDP port the server's WireGuard listens on.
    pub wg_port: u16,
    /// Tunnel network, assumed /24. Hosts are allocated from `.2` upward.
    pub subnet_base: [u8; 4],
}

impl TunnelSettings {
    /// Parse `10.8.0.0/24` (only /24 is supported for now).
    pub fn parse_cidr(cidr: &str) -> Result<[u8; 4], String> {
        let (addr, prefix) = cidr.split_once('/').ok_or("expected addr/prefix")?;
        if prefix.trim() != "24" {
            return Err("only /24 tunnel subnets are supported for now".into());
        }
        let octets: Vec<u8> = addr
            .split('.')
            .map(|o| o.parse::<u8>().map_err(|_| "bad octet".to_string()))
            .collect::<Result<_, _>>()?;
        let arr: [u8; 4] = octets.try_into().map_err(|_| "expected 4 octets".to_string())?;
        Ok(arr)
    }
}

// -------------------------------------------------------------------------
// per-peer address allocation
// -------------------------------------------------------------------------

/// Maps a peer's WireGuard public key to a stable tunnel address. A peer that
/// rekeys keeps the same address.
pub struct PeerAddresses {
    subnet_base: [u8; 4],
    inner: Mutex<Inner>,
}

struct Inner {
    assigned: HashMap<[u8; 32], Ipv4Addr>,
    next_host: u8,
}

impl PeerAddresses {
    pub fn new(subnet_base: [u8; 4]) -> Self {
        PeerAddresses {
            subnet_base,
            inner: Mutex::new(Inner {
                assigned: HashMap::new(),
                next_host: 2, // .1 is the server
            }),
        }
    }

    /// Return the peer's address, allocating one on first sight.
    pub fn assign(&self, peer_wg_pubkey: &[u8; 32]) -> Result<Ipv4Addr, InstallError> {
        let mut g = self.inner.lock().unwrap();
        if let Some(ip) = g.assigned.get(peer_wg_pubkey) {
            return Ok(*ip);
        }
        if g.next_host == 0 || g.next_host == 255 {
            return Err(InstallError::PoolExhausted);
        }
        let ip = Ipv4Addr::new(
            self.subnet_base[0],
            self.subnet_base[1],
            self.subnet_base[2],
            g.next_host,
        );
        g.next_host += 1;
        g.assigned.insert(*peer_wg_pubkey, ip);
        Ok(ip)
    }

    pub fn known(&self, peer_wg_pubkey: &[u8; 32]) -> bool {
        self.inner.lock().unwrap().assigned.contains_key(peer_wg_pubkey)
    }
}

// -------------------------------------------------------------------------
// PSK installer
// -------------------------------------------------------------------------

#[derive(Debug)]
pub enum InstallError {
    PoolExhausted,
    Spawn(String),
    WgFailed { code: Option<i32>, stderr: String },
}

impl std::fmt::Display for InstallError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InstallError::PoolExhausted => write!(f, "tunnel address pool exhausted"),
            InstallError::Spawn(e) => write!(f, "could not run `wg`: {e}"),
            InstallError::WgFailed { code, stderr } => {
                write!(f, "`wg set` failed (exit {code:?}): {}", stderr.trim())
            }
        }
    }
}
impl std::error::Error for InstallError {}

pub trait PskInstaller: Send + Sync {
    /// Install or replace the peer's preshared key. `first_install` is false on
    /// a rekey — the peer already exists and only the PSK changes.
    fn install(
        &self,
        peer_wg_pubkey: &[u8; 32],
        psk: &[u8; 32],
        assigned_ip: Ipv4Addr,
        algo: AlgoCode,
        first_install: bool,
    ) -> Result<(), InstallError>;

    /// Human-readable name for the startup banner.
    fn describe(&self) -> String;
}

/// Logs the action without touching WireGuard.
pub struct DryRun;

impl PskInstaller for DryRun {
    fn install(
        &self,
        peer_wg_pubkey: &[u8; 32],
        _psk: &[u8; 32],
        assigned_ip: Ipv4Addr,
        algo: AlgoCode,
        first_install: bool,
    ) -> Result<(), InstallError> {
        let verb = if first_install { "install" } else { "swap" };
        println!(
            "[dry-run] would {verb} PSK for wg peer {} ({}) at {assigned_ip}",
            b64(peer_wg_pubkey),
            algo.name(),
        );
        Ok(())
    }
    fn describe(&self) -> String {
        "dry-run (no --wg-interface)".into()
    }
}

/// `wg set <iface> peer <pubkey> preshared-key /dev/stdin [allowed-ips <ip>/32]`.
pub struct WgCli {
    pub interface: String,
}

impl WgCli {
    /// `wg show <iface> public-key` → 32 raw bytes.
    pub fn query_pubkey(interface: &str) -> Result<[u8; 32], InstallError> {
        let out = Command::new("wg")
            .args(["show", interface, "public-key"])
            .output()
            .map_err(|e| InstallError::Spawn(e.to_string()))?;
        if !out.status.success() {
            return Err(InstallError::WgFailed {
                code: out.status.code(),
                stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
            });
        }
        let s = String::from_utf8_lossy(&out.stdout);
        wg_key_from_base64(s.trim())
            .ok_or_else(|| InstallError::WgFailed { code: None, stderr: "unparseable public key".into() })
    }

    /// The argv `install` would run (for tests / logging).
    pub fn argv(&self, peer_wg_pubkey: &[u8; 32], assigned_ip: Ipv4Addr, first_install: bool) -> Vec<String> {
        let mut v = vec![
            "set".to_string(),
            self.interface.clone(),
            "peer".to_string(),
            b64(peer_wg_pubkey),
            "preshared-key".to_string(),
            "/dev/stdin".to_string(),
        ];
        if first_install {
            v.push("allowed-ips".to_string());
            v.push(format!("{assigned_ip}/32"));
        }
        v
    }
}

impl PskInstaller for WgCli {
    fn install(
        &self,
        peer_wg_pubkey: &[u8; 32],
        psk: &[u8; 32],
        assigned_ip: Ipv4Addr,
        _algo: AlgoCode,
        first_install: bool,
    ) -> Result<(), InstallError> {
        let mut child = Command::new("wg")
            .args(self.argv(peer_wg_pubkey, assigned_ip, first_install))
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| InstallError::Spawn(e.to_string()))?;

        child
            .stdin
            .take()
            .expect("piped")
            .write_all(format!("{}\n", b64(psk)).as_bytes())
            .map_err(|e| InstallError::Spawn(e.to_string()))?;

        let out = child
            .wait_with_output()
            .map_err(|e| InstallError::Spawn(e.to_string()))?;
        if !out.status.success() {
            return Err(InstallError::WgFailed {
                code: out.status.code(),
                stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
            });
        }
        Ok(())
    }
    fn describe(&self) -> String {
        format!("wg set {}", self.interface)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocator_is_stable_per_peer_and_skips_the_server() {
        let a = PeerAddresses::new([10, 8, 0, 0]);
        let p1 = [1u8; 32];
        let p2 = [2u8; 32];
        assert_eq!(a.assign(&p1).unwrap(), Ipv4Addr::new(10, 8, 0, 2));
        assert_eq!(a.assign(&p2).unwrap(), Ipv4Addr::new(10, 8, 0, 3));
        assert_eq!(a.assign(&p1).unwrap(), Ipv4Addr::new(10, 8, 0, 2)); // stable
    }

    #[test]
    fn wg_argv_matches_protocol() {
        let cli = WgCli { interface: "wg0".into() };
        let peer = [0xABu8; 32];
        let first = cli.argv(&peer, Ipv4Addr::new(10, 8, 0, 5), true);
        assert_eq!(first[0..3], ["set", "wg0", "peer"]);
        assert_eq!(first[3], B64.encode(peer));
        assert_eq!(first[4..6], ["preshared-key", "/dev/stdin"]);
        assert_eq!(first[6..8], ["allowed-ips", "10.8.0.5/32"]);

        // rekey: no allowed-ips re-set
        let rekey = cli.argv(&peer, Ipv4Addr::new(10, 8, 0, 5), false);
        assert_eq!(rekey.len(), 6);
    }

    #[test]
    fn cidr_parsing() {
        assert_eq!(TunnelSettings::parse_cidr("10.8.0.0/24").unwrap(), [10, 8, 0, 0]);
        assert!(TunnelSettings::parse_cidr("10.8.0.0/16").is_err());
    }

    #[test]
    fn dry_run_never_fails() {
        let d = DryRun;
        assert!(d.install(&[0u8; 32], &[0u8; 32], Ipv4Addr::new(10, 8, 0, 2), AlgoCode::MlKem768, true).is_ok());
    }
}
