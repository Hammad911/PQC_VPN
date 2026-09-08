//! PQC-VPN handshake server — `PROTOCOL.md` v1.
//!
//! Usage:
//!   handshake-server [--bind ADDR] [--identity PATH]
//!                    [--wg-interface NAME] [--wg-port N] [--tunnel-cidr CIDR]
//!                    [--wg-pubkey BASE64]
//!
//! defaults: --bind 0.0.0.0:51821  --identity ./server-identity.seed
//!           --wg-port 51820       --tunnel-cidr 10.8.0.0/24
//!
//! With `--wg-interface` the derived PSK is installed via `wg set` (needs
//! CAP_NET_ADMIN). Without it the server runs in dry-run: it completes
//! handshakes and logs what it would install, but touches no WireGuard.

use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};

use handshake_server::identity::ServerIdentity;
use handshake_server::server::{self, ServerContext};
use handshake_server::session::SessionTable;
use handshake_server::tunnel::{self, DryRun, PeerAddresses, PskInstaller, TunnelSettings, WgCli};

fn main() -> Result<()> {
    let mut bind = "0.0.0.0:51821".to_string();
    let mut identity_path = PathBuf::from("server-identity.seed");
    let mut wg_interface: Option<String> = None;
    let mut wg_port: u16 = 51820;
    let mut tunnel_cidr = "10.8.0.0/24".to_string();
    let mut wg_pubkey_b64: Option<String> = None;

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = || -> Result<String> { args.next().context("missing argument value") };
        match a.as_str() {
            "--bind" => bind = val()?,
            "--identity" => identity_path = PathBuf::from(val()?),
            "--wg-interface" => wg_interface = Some(val()?),
            "--wg-port" => wg_port = val()?.parse().context("--wg-port")?,
            "--tunnel-cidr" => tunnel_cidr = val()?,
            "--wg-pubkey" => wg_pubkey_b64 = Some(val()?),
            "-h" | "--help" => {
                println!("handshake-server [--bind ADDR] [--identity PATH] [--wg-interface NAME]");
                println!("                 [--wg-port N] [--tunnel-cidr CIDR] [--wg-pubkey BASE64]");
                return Ok(());
            }
            other => anyhow::bail!("unknown argument: {other}"),
        }
    }

    let identity = ServerIdentity::load_or_create(&identity_path)
        .with_context(|| format!("loading identity from {}", identity_path.display()))?;

    let vk = identity.verifying_key_bytes();
    let vk_hex = hex::encode(&vk);
    let pub_path = identity_path.with_extension("pub");
    std::fs::write(&pub_path, &vk_hex).ok();

    let subnet_base = TunnelSettings::parse_cidr(&tunnel_cidr).map_err(anyhow::Error::msg)?;

    let (installer, server_wg_pubkey): (Box<dyn PskInstaller>, [u8; 32]) = match &wg_interface {
        Some(iface) => {
            let pk = WgCli::query_pubkey(iface)
                .with_context(|| format!("`wg show {iface} public-key` (is the interface up?)"))?;
            (Box::new(WgCli { interface: iface.clone() }), pk)
        }
        None => {
            let pk = wg_pubkey_b64
                .as_deref()
                .and_then(tunnel::wg_key_from_base64)
                .unwrap_or([0u8; 32]);
            (Box::new(DryRun), pk)
        }
    };

    let ctx = Arc::new(ServerContext {
        identity,
        sessions: SessionTable::new(),
        addrs: PeerAddresses::new(subnet_base),
        tunnel: TunnelSettings {
            server_wg_pubkey,
            wg_port,
            subnet_base,
        },
        installer,
    });

    println!("identity seed  : {}", identity_path.display());
    println!("verifying key  : {} ({} bytes)", pub_path.display(), vk.len());
    println!("               : {vk_hex}");
    println!("psk install    : {}", ctx.installer.describe());
    println!(
        "tunnel         : {tunnel_cidr}, wg port {wg_port}, server wg pubkey {}",
        base64_or_none(&ctx.tunnel.server_wg_pubkey)
    );

    let listener = TcpListener::bind(&bind).with_context(|| format!("bind {bind}"))?;
    println!("listening on   : {}\n", listener.local_addr()?);

    server::serve(listener, ctx);
    Ok(())
}

fn base64_or_none(k: &[u8; 32]) -> String {
    if k.iter().all(|b| *b == 0) {
        "(none — dry-run)".into()
    } else {
        base64::Engine::encode(&base64::engine::general_purpose::STANDARD, k)
    }
}
