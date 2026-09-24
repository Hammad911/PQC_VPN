//! Throwaway handshake client for testing the server (the Week 3 "throwaway
//! client script"). Runs a full handshake, then prints a ready-to-use
//! `wg-quick` client config built from the derived PSK and the tunnel
//! parameters the server returned in `ServerFinish`.
//!
//! Usage:
//!   test-client <handshake-host:port> <verifying-key-hex | @path>
//!               [--algo 512|768|1024] [--wg-endpoint HOST]
//!
//! `--wg-endpoint` defaults to the handshake host (the WireGuard UDP port comes
//! from the server).
//!
//! Rekey (Week 6) — reuse the session id and wg keypair a first run printed to
//! exercise `RekeyRequest` against an *existing* peer, live:
//!   test-client <host:port> <vk-hex|@path> --rekey --session <hex>
//!               --wg-key <base64|@path> [--algo N]
//!
//! Bring a tunnel up from the first run's config, `ping` through it, then run
//! this against the same peer — the printed `wg set` line hot-swaps the PSK
//! without ever bringing the interface down, proving the rekey doesn't drop
//! the tunnel.

use std::net::TcpStream;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;
use rand::rngs::OsRng;
use x25519_dalek::{PublicKey, StaticSecret};

use handshake_server::client_handshake;
use handshake_server::kdf;
use handshake_server::wire::{AlgoCode, Message, RekeyRequest};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let hs_addr = args.next().context(
        "usage: test-client <host:port> <vk-hex|@path> [--algo N] [--wg-endpoint HOST] \
         [--rekey --session <hex> --wg-key <base64|@path>]",
    )?;
    let vk_arg = args.next().context("need the server verifying key (hex or @path)")?;

    let mut algo = AlgoCode::MlKem768;
    let mut wg_endpoint_host: Option<String> = None;
    let mut rekey = false;
    let mut session_arg: Option<String> = None;
    let mut wg_key_arg: Option<String> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--algo" => {
                algo = match args.next().as_deref() {
                    Some("512") => AlgoCode::MlKem512,
                    Some("768") => AlgoCode::MlKem768,
                    Some("1024") => AlgoCode::MlKem1024,
                    other => bail!("bad --algo {other:?} (use 512|768|1024)"),
                }
            }
            "--wg-endpoint" => wg_endpoint_host = args.next(),
            "--rekey" => rekey = true,
            "--session" => session_arg = args.next(),
            "--wg-key" => wg_key_arg = args.next(),
            other => bail!("unknown argument {other}"),
        }
    }

    let vk_hex = match vk_arg.strip_prefix('@') {
        Some(path) => std::fs::read_to_string(path)?.trim().to_string(),
        None => vk_arg,
    };
    let verifying_key = hex::decode(vk_hex.trim()).context("verifying key must be hex")?;

    // The WireGuard keypair for this client: fresh for a first handshake, or
    // the pinned one from an earlier run when rekeying the same peer.
    let wg_private = match &wg_key_arg {
        Some(arg) => {
            let b64 = match arg.strip_prefix('@') {
                Some(path) => std::fs::read_to_string(path)?.trim().to_string(),
                None => arg.clone(),
            };
            let bytes: [u8; 32] = B64
                .decode(b64.trim())
                .context("--wg-key must be base64")?
                .try_into()
                .map_err(|_| anyhow::anyhow!("--wg-key must decode to 32 bytes"))?;
            StaticSecret::from(bytes)
        }
        None => StaticSecret::random_from_rng(OsRng),
    };
    let wg_public = PublicKey::from(&wg_private);

    let claimed_session: Option<[u8; 16]> = match &session_arg {
        Some(hex_str) => Some(
            hex::decode(hex_str.trim())
                .context("--session must be hex")?
                .try_into()
                .map_err(|_| anyhow::anyhow!("--session must decode to 16 bytes"))?,
        ),
        None => None,
    };

    if rekey && (wg_key_arg.is_none() || claimed_session.is_none()) {
        bail!("--rekey needs both --session <hex> and --wg-key <base64|@path> (from the first run)");
    }

    let client_nonce: [u8; 32] = rand::random();
    let state = client_handshake::start(algo, client_nonce, wg_public.to_bytes());

    let mut stream = TcpStream::connect(&hs_addr).with_context(|| format!("connect {hs_addr}"))?;
    stream.set_read_timeout(Some(Duration::from_secs(10)))?;
    stream.set_write_timeout(Some(Duration::from_secs(10)))?;

    if let Some(session_id) = claimed_session {
        println!("-> RekeyRequest session={} ({})", hex::encode(session_id), algo.name());
        Message::RekeyRequest(RekeyRequest { session_id, hello: state.hello().clone() })
            .write(&mut stream)?;
    } else {
        println!("-> ClientHello ({})", algo.name());
        Message::ClientHello(state.hello().clone()).write(&mut stream)?;
    }

    let server_hello = match Message::read(&mut stream)? {
        Message::ServerHello(sh) => sh,
        Message::Error(e) => bail!("server returned Error {:#04x}: {}", e.code, e.message),
        other => bail!("expected ServerHello, got {other:?}"),
    };
    println!("<- ServerHello  session={}", hex::encode(server_hello.session_id));

    let (keys, finish, transcript) = state
        .finish(&server_hello, &verifying_key)
        .context("verifying ServerHello / deriving keys")?;
    println!("   server signature verified, keys derived");

    println!("-> ClientFinish");
    Message::ClientFinish(finish.clone()).write(&mut stream)?;

    let sf = match Message::read(&mut stream)? {
        Message::ServerFinish(sf) => sf,
        Message::Error(e) => bail!("server returned Error {:#04x}: {}", e.code, e.message),
        other => bail!("expected ServerFinish, got {other:?}"),
    };
    if sf.session_id != finish.session_id {
        bail!("ServerFinish session id mismatch");
    }
    if !kdf::verify_server_tag(&keys.confirm_key, &transcript, &sf.server_tag) {
        bail!("ServerFinish tag did not verify — server derived a different secret");
    }
    println!("<- ServerFinish verified\n");

    let assigned_ip = std::net::Ipv4Addr::from(sf.assigned_ip);
    let endpoint_host = wg_endpoint_host.unwrap_or_else(|| {
        hs_addr.rsplit_once(':').map(|(h, _)| h.to_string()).unwrap_or(hs_addr.clone())
    });

    if rekey {
        println!("REKEY OK");
        println!("  algorithm      : {}", algo.name());
        println!("  session id     : {}", hex::encode(finish.session_id));
        println!("  new PSK        : {}  (base64: {})", hex::encode(keys.psk), B64.encode(keys.psk));
        println!("  assigned IP    : {assigned_ip}  (unchanged)");
        println!();
        println!("Hot-swap the PSK on the client's already-up interface — no `wg-quick down`,");
        println!("the tunnel keeps passing traffic through the rekey:");
        println!();
        println!(
            "  wg set <iface> peer {} preshared-key <(echo {})",
            B64.encode(sf.server_wg_pubkey),
            B64.encode(keys.psk)
        );
        return Ok(());
    }

    println!("HANDSHAKE OK");
    println!("  algorithm     : {}", algo.name());
    println!("  session id    : {}", hex::encode(finish.session_id));
    println!("  derived PSK    : {}  (base64: {})", hex::encode(keys.psk), B64.encode(keys.psk));
    println!("  assigned IP    : {assigned_ip}");
    println!("  server wg pub  : {}", B64.encode(sf.server_wg_pubkey));
    println!("  wg endpoint    : {endpoint_host}:{}", sf.wg_port);
    println!();
    println!("----- wg-quick config (save as pqc0.conf) -----");
    println!("[Interface]");
    println!("PrivateKey = {}", B64.encode(wg_private.to_bytes()));
    println!("Address    = {assigned_ip}/24");
    println!();
    println!("[Peer]");
    println!("PublicKey    = {}", B64.encode(sf.server_wg_pubkey));
    println!("PresharedKey = {}", B64.encode(keys.psk));
    println!("Endpoint     = {endpoint_host}:{}", sf.wg_port);
    println!("AllowedIPs   = {}.0/24", trim_last_octet(&assigned_ip));
    println!("PersistentKeepalive = 25");
    println!();
    println!("Rekey this peer later with:");
    println!(
        "  test-client {hs_addr} <vk> --rekey --session {} --wg-key {}",
        hex::encode(finish.session_id),
        B64.encode(wg_private.to_bytes())
    );
    Ok(())
}

fn trim_last_octet(ip: &std::net::Ipv4Addr) -> String {
    let o = ip.octets();
    format!("{}.{}.{}", o[0], o[1], o[2])
}
