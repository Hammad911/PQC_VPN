//! PQC-VPN handshake server — `PROTOCOL.md` v1.
//!
//! Usage:
//!   handshake-server [--bind ADDR] [--identity PATH]
//!                    [--wg-interface NAME] [--wg-port N] [--tunnel-cidr CIDR]
//!                    [--wg-pubkey BASE64]
//!                    [--peers-file PATH] [--max-peers N] [--idle-timeout-secs N]
//!
//! defaults: --bind 0.0.0.0:51821  --identity ./server-identity.seed
//!           --wg-port 51820       --tunnel-cidr 10.8.0.0/24
//!           --peers-file <identity dir>/peers.json  --max-peers 64
//!           --idle-timeout-secs 0  (0 = no idle eviction)
//!
//! With `--wg-interface` the derived PSK is installed via `wg set` (needs
//! CAP_NET_ADMIN). Without it the server runs in dry-run.

use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};

use handshake_server::identity::ServerIdentity;
use handshake_server::registry::{Registry, DEFAULT_MAX_PEERS};
use handshake_server::server::{self, ServerContext};
use handshake_server::tunnel::{self, DryRun, PskInstaller, TunnelSettings, WgCli};

fn main() -> Result<()> {
    let mut bind = "0.0.0.0:51821".to_string();
    let mut identity_path = PathBuf::from("server-identity.seed");
    let mut wg_interface: Option<String> = None;
    let mut wg_port: u16 = 51820;
    let mut tunnel_cidr = "10.8.0.0/24".to_string();
    let mut wg_pubkey_b64: Option<String> = None;
    let mut peers_file: Option<PathBuf> = None;
    let mut max_peers = DEFAULT_MAX_PEERS;
    let mut idle_timeout_secs: u64 = 0;

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
            "--peers-file" => peers_file = Some(PathBuf::from(val()?)),
            "--max-peers" => max_peers = val()?.parse().context("--max-peers")?,
            "--idle-timeout-secs" => idle_timeout_secs = val()?.parse().context("--idle-timeout-secs")?,
            "-h" | "--help" => {
                println!("handshake-server [--bind ADDR] [--identity PATH] [--wg-interface NAME]");
                println!("                 [--wg-port N] [--tunnel-cidr CIDR] [--wg-pubkey BASE64]");
                println!("                 [--peers-file PATH] [--max-peers N] [--idle-timeout-secs N]");
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

    let peers_file = peers_file.unwrap_or_else(|| {
        identity_path
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."))
            .join("peers.json")
    });
    let registry = Registry::load(peers_file.clone(), subnet_base, max_peers)
        .with_context(|| format!("loading registry from {}", peers_file.display()))?;

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
        registry,
        tunnel: TunnelSettings { server_wg_pubkey, wg_port, subnet_base },
        installer,
    });

    println!("identity seed  : {}", identity_path.display());
    println!("verifying key  : {} ({} bytes)", pub_path.display(), vk.len());
    println!("               : {vk_hex}");
    println!("peers file     : {} ({} known)", peers_file.display(), ctx.registry.len());
    println!("psk install    : {}", ctx.installer.describe());
    println!("tunnel         : {tunnel_cidr}, wg port {wg_port}, max peers {max_peers}");
    if idle_timeout_secs > 0 {
        println!("idle eviction  : after {idle_timeout_secs}s");
    }

    // reconcile the registry against the live interface
    server::reconcile_wg_peers(&ctx);

    if idle_timeout_secs > 0 {
        server::spawn_idle_sweep(Arc::clone(&ctx), Duration::from_secs(idle_timeout_secs));
    }

    let listener = TcpListener::bind(&bind).with_context(|| format!("bind {bind}"))?;
    println!("listening on   : {}\n", listener.local_addr()?);

    server::serve(listener, ctx);
    Ok(())
}
