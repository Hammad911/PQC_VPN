//! Multi-peer registry — Week 5.
//!
//! One `Registry`, keyed by WireGuard public key (the stable peer identity),
//! owns everything the server tracks per client: the PQC session, negotiated
//! algorithm, assigned tunnel address, and activity timestamps.
//!
//! Persisted to a JSON file — **without PSKs** (the server never re-uses a
//! stored PSK; a rekey derives a fresh one and WireGuard keeps the live one) —
//! so it survives a restart. On startup the caller reconciles it against the
//! live `wg0` peer list.

use std::collections::{HashMap, HashSet};
use std::io;
use std::net::Ipv4Addr;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;
use serde::{Deserialize, Serialize};

use crate::wire::AlgoCode;

pub const DEFAULT_MAX_PEERS: usize = 64;
const HOST_MIN: u8 = 2; // .1 is the server
const HOST_MAX: u8 = 254;
/// A /24 has room for hosts 2..=254.
pub const POOL_SIZE: usize = (HOST_MAX - HOST_MIN + 1) as usize;

#[derive(Debug, Clone)]
pub struct Peer {
    pub wg_pubkey: [u8; 32],
    pub assigned_ip: Ipv4Addr,
    pub session_id: [u8; 16],
    pub algo: AlgoCode,
    pub created: SystemTime,
    pub last_activity: SystemTime,
    pub rekeys: u32,
}

/// What a completed handshake/rekey needs to build `ServerFinish` and call `wg set`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeerSlot {
    pub session_id: [u8; 16],
    pub assigned_ip: Ipv4Addr,
}

#[derive(Debug, PartialEq, Eq)]
pub enum RegistryError {
    /// `--max-peers` reached.
    AtCapacity,
    /// Rekey for a wg pubkey the registry has no record of.
    UnknownPeer,
    /// Rekey `session_id` does not match the peer's current session.
    SessionMismatch,
    /// Rekey `algo` is weaker than the peer's in-force algorithm.
    Downgrade,
    /// Every address in the /24 is assigned.
    AddressPoolExhausted,
}

pub struct Registry {
    path: Option<PathBuf>,
    subnet_base: [u8; 4],
    max_peers: usize,
    inner: Mutex<HashMap<[u8; 32], Peer>>,
}

impl Registry {
    /// In-memory only (tests).
    pub fn new(subnet_base: [u8; 4], max_peers: usize) -> Self {
        Registry {
            path: None,
            subnet_base,
            max_peers: max_peers.min(POOL_SIZE),
            inner: Mutex::new(HashMap::new()),
        }
    }

    /// Load from `path` if it exists, else start empty. Persists to `path`.
    pub fn load(path: PathBuf, subnet_base: [u8; 4], max_peers: usize) -> io::Result<Self> {
        let mut map = HashMap::new();
        match std::fs::read(&path) {
            Ok(bytes) => {
                let file: RegistryFile = serde_json::from_slice(&bytes)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                for rec in file.peers {
                    match rec.into_peer() {
                        Ok(p) => {
                            map.insert(p.wg_pubkey, p);
                        }
                        Err(e) => eprintln!("registry: skipping bad peer record: {e}"),
                    }
                }
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(e),
        }
        Ok(Registry {
            path: Some(path),
            subnet_base,
            max_peers: max_peers.min(POOL_SIZE),
            inner: Mutex::new(map),
        })
    }

    /// A fresh `ClientHello`. If the peer is known, replace its session (new
    /// `session_id`, updated algorithm, same address, rekey count reset). If
    /// new, allocate an address.
    pub fn upsert_handshake(
        &self,
        wg_pubkey: [u8; 32],
        algo: AlgoCode,
        session_id: [u8; 16],
    ) -> Result<PeerSlot, RegistryError> {
        let mut map = self.inner.lock().unwrap();
        let now = SystemTime::now();

        if let Some(p) = map.get_mut(&wg_pubkey) {
            p.session_id = session_id;
            p.algo = algo;
            p.last_activity = now;
            p.rekeys = 0;
            return Ok(PeerSlot {
                session_id,
                assigned_ip: p.assigned_ip,
            });
        }

        if map.len() >= self.max_peers {
            return Err(RegistryError::AtCapacity);
        }
        let assigned_ip = self.alloc_ip(&map)?;
        map.insert(
            wg_pubkey,
            Peer {
                wg_pubkey,
                assigned_ip,
                session_id,
                algo,
                created: now,
                last_activity: now,
                rekeys: 0,
            },
        );
        Ok(PeerSlot { session_id, assigned_ip })
    }

    /// A `RekeyRequest`. The peer must exist, `claimed_session_id` must match its
    /// current session, and `algo` may not be weaker than the in-force one.
    pub fn rekey(
        &self,
        wg_pubkey: &[u8; 32],
        claimed_session_id: &[u8; 16],
        algo: AlgoCode,
    ) -> Result<PeerSlot, RegistryError> {
        let mut map = self.inner.lock().unwrap();
        let p = map.get_mut(wg_pubkey).ok_or(RegistryError::UnknownPeer)?;
        if p.session_id != *claimed_session_id {
            return Err(RegistryError::SessionMismatch);
        }
        if algo < p.algo {
            return Err(RegistryError::Downgrade);
        }
        p.algo = algo; // may escalate (REKEY_ESCALATES)
        p.rekeys += 1;
        p.last_activity = SystemTime::now();
        Ok(PeerSlot {
            session_id: p.session_id,
            assigned_ip: p.assigned_ip,
        })
    }

    fn alloc_ip(&self, map: &HashMap<[u8; 32], Peer>) -> Result<Ipv4Addr, RegistryError> {
        let taken: HashSet<u8> = map.values().map(|p| p.assigned_ip.octets()[3]).collect();
        for host in HOST_MIN..=HOST_MAX {
            if !taken.contains(&host) {
                let b = self.subnet_base;
                return Ok(Ipv4Addr::new(b[0], b[1], b[2], host));
            }
        }
        Err(RegistryError::AddressPoolExhausted)
    }

    pub fn pubkeys(&self) -> Vec<[u8; 32]> {
        self.inner.lock().unwrap().keys().copied().collect()
    }

    pub fn contains(&self, wg_pubkey: &[u8; 32]) -> bool {
        self.inner.lock().unwrap().contains_key(wg_pubkey)
    }

    pub fn remove(&self, wg_pubkey: &[u8; 32]) -> Option<Peer> {
        self.inner.lock().unwrap().remove(wg_pubkey)
    }

    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn touch(&self, wg_pubkey: &[u8; 32]) {
        if let Some(p) = self.inner.lock().unwrap().get_mut(wg_pubkey) {
            p.last_activity = SystemTime::now();
        }
    }

    /// Peers with no activity for longer than `timeout`.
    pub fn idle_peers(&self, timeout: Duration, now: SystemTime) -> Vec<[u8; 32]> {
        self.inner
            .lock()
            .unwrap()
            .values()
            .filter(|p| now.duration_since(p.last_activity).map(|d| d > timeout).unwrap_or(false))
            .map(|p| p.wg_pubkey)
            .collect()
    }

    /// Atomic write to the backing file. No-op for an in-memory registry.
    pub fn persist(&self) -> io::Result<()> {
        let Some(path) = &self.path else { return Ok(()) };
        let map = self.inner.lock().unwrap();
        let file = RegistryFile {
            schema_version: 1,
            peers: map.values().map(PeerRecord::from_peer).collect(),
        };
        let json = serde_json::to_vec_pretty(&file)?;
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, &json)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }
}

// ---- persistence records --------------------------------------------------

#[derive(Serialize, Deserialize)]
struct RegistryFile {
    schema_version: u32,
    peers: Vec<PeerRecord>,
}

#[derive(Serialize, Deserialize)]
struct PeerRecord {
    wg_pubkey_b64: String,
    assigned_ip: String,
    session_id_hex: String,
    algo: String,
    created_epoch: u64,
    last_activity_epoch: u64,
    rekeys: u32,
}

fn epoch(t: SystemTime) -> u64 {
    t.duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

impl PeerRecord {
    fn from_peer(p: &Peer) -> Self {
        PeerRecord {
            wg_pubkey_b64: B64.encode(p.wg_pubkey),
            assigned_ip: p.assigned_ip.to_string(),
            session_id_hex: hex::encode(p.session_id),
            algo: p.algo.name().to_string(),
            created_epoch: epoch(p.created),
            last_activity_epoch: epoch(p.last_activity),
            rekeys: p.rekeys,
        }
    }

    fn into_peer(self) -> Result<Peer, String> {
        let wg_pubkey: [u8; 32] = B64
            .decode(&self.wg_pubkey_b64)
            .map_err(|e| e.to_string())?
            .try_into()
            .map_err(|_| "wg_pubkey not 32 bytes".to_string())?;
        let session_id: [u8; 16] = hex::decode(&self.session_id_hex)
            .map_err(|e| e.to_string())?
            .try_into()
            .map_err(|_| "session_id not 16 bytes".to_string())?;
        let assigned_ip: Ipv4Addr = self.assigned_ip.parse().map_err(|_| "bad ip".to_string())?;
        let algo = AlgoCode::from_name(&self.algo).ok_or_else(|| format!("bad algo {}", self.algo))?;
        Ok(Peer {
            wg_pubkey,
            assigned_ip,
            session_id,
            algo,
            created: UNIX_EPOCH + Duration::from_secs(self.created_epoch),
            last_activity: UNIX_EPOCH + Duration::from_secs(self.last_activity_epoch),
            rekeys: self.rekeys,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sid(n: u8) -> [u8; 16] {
        [n; 16]
    }

    #[test]
    fn allocates_distinct_addresses_from_two() {
        let r = Registry::new([10, 8, 0, 0], DEFAULT_MAX_PEERS);
        let a = r.upsert_handshake([1; 32], AlgoCode::MlKem768, sid(1)).unwrap();
        let b = r.upsert_handshake([2; 32], AlgoCode::MlKem512, sid(2)).unwrap();
        assert_eq!(a.assigned_ip, Ipv4Addr::new(10, 8, 0, 2));
        assert_eq!(b.assigned_ip, Ipv4Addr::new(10, 8, 0, 3));
    }

    #[test]
    fn fresh_hello_from_known_peer_keeps_ip_new_session() {
        let r = Registry::new([10, 8, 0, 0], DEFAULT_MAX_PEERS);
        let a = r.upsert_handshake([7; 32], AlgoCode::MlKem512, sid(1)).unwrap();
        let b = r.upsert_handshake([7; 32], AlgoCode::MlKem1024, sid(2)).unwrap();
        assert_eq!(a.assigned_ip, b.assigned_ip);
        assert_ne!(a.session_id, b.session_id);
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn rekey_checks_peer_session_and_downgrade() {
        let r = Registry::new([10, 8, 0, 0], DEFAULT_MAX_PEERS);
        r.upsert_handshake([7; 32], AlgoCode::MlKem768, sid(1)).unwrap();

        assert_eq!(r.rekey(&[9; 32], &sid(1), AlgoCode::MlKem768), Err(RegistryError::UnknownPeer));
        assert_eq!(r.rekey(&[7; 32], &sid(2), AlgoCode::MlKem768), Err(RegistryError::SessionMismatch));
        assert_eq!(r.rekey(&[7; 32], &sid(1), AlgoCode::MlKem512), Err(RegistryError::Downgrade));

        // equal ok, escalation ok
        assert!(r.rekey(&[7; 32], &sid(1), AlgoCode::MlKem768).is_ok());
        let slot = r.rekey(&[7; 32], &sid(1), AlgoCode::MlKem1024).unwrap();
        assert_eq!(slot.session_id, sid(1));
    }

    #[test]
    fn capacity_is_enforced() {
        let r = Registry::new([10, 8, 0, 0], 2);
        r.upsert_handshake([1; 32], AlgoCode::MlKem768, sid(1)).unwrap();
        r.upsert_handshake([2; 32], AlgoCode::MlKem768, sid(2)).unwrap();
        assert_eq!(
            r.upsert_handshake([3; 32], AlgoCode::MlKem768, sid(3)),
            Err(RegistryError::AtCapacity)
        );
        // a known peer re-handshaking is not blocked by capacity
        assert!(r.upsert_handshake([1; 32], AlgoCode::MlKem768, sid(9)).is_ok());
    }

    #[test]
    fn persist_and_reload_round_trips() {
        let path = std::env::temp_dir().join(format!("pqcvpn-reg-{}.json", std::process::id()));
        let _ = std::fs::remove_file(&path);

        let r = Registry::load(path.clone(), [10, 8, 0, 0], DEFAULT_MAX_PEERS).unwrap();
        r.upsert_handshake([1; 32], AlgoCode::MlKem768, sid(1)).unwrap();
        r.upsert_handshake([2; 32], AlgoCode::MlKem1024, sid(2)).unwrap();
        r.rekey(&[1; 32], &sid(1), AlgoCode::MlKem1024).unwrap();
        r.persist().unwrap();

        let r2 = Registry::load(path.clone(), [10, 8, 0, 0], DEFAULT_MAX_PEERS).unwrap();
        assert_eq!(r2.len(), 2);
        assert!(r2.contains(&[1; 32]));
        // new peer should get .4 (.2 and .3 taken)
        let c = r2.upsert_handshake([3; 32], AlgoCode::MlKem512, sid(3)).unwrap();
        assert_eq!(c.assigned_ip, Ipv4Addr::new(10, 8, 0, 4));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn idle_sweep_selects_stale_peers() {
        let r = Registry::new([10, 8, 0, 0], DEFAULT_MAX_PEERS);
        r.upsert_handshake([1; 32], AlgoCode::MlKem768, sid(1)).unwrap();
        let future = SystemTime::now() + Duration::from_secs(3600);
        assert_eq!(r.idle_peers(Duration::from_secs(60), future), vec![[1u8; 32]]);
        assert!(r.idle_peers(Duration::from_secs(7200), future).is_empty());
    }
}
