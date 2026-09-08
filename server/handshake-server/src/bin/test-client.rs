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

use std::net::TcpStream;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;
use rand::rngs::OsRng;
use x25519_dalek::{PublicKey, StaticSecret};

use handshake_server::client_handshake;
use handshake_server::kdf;
use handshake_server::wire::{AlgoCode, Message};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let hs_addr = args
        .next()
        .context("usage: test-client <host:port> <vk-hex|@path> [--algo N] [--wg-endpoint HOST]")?;
    let vk_arg = args.next().context("need the server verifying key (hex or @path)")?;

    let mut algo = AlgoCode::MlKem768;
    let mut wg_endpoint_host: Option<String> = None;
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
            other => bail!("unknown argument {other}"),
        }
    }

    let vk_hex = match vk_arg.strip_prefix('@') {
        Some(path) => std::fs::read_to_string(path)?.trim().to_string(),
        None => vk_arg,
    };
    let verifying_key = hex::decode(vk_hex.trim()).context("verifying key must be hex")?;

    // Real WireGuard keypair for this client.
    let wg_private = StaticSecret::random_from_rng(OsRng);
    let wg_public = PublicKey::from(&wg_private);

    let client_nonce: [u8; 32] = rand::random();
    let state = client_handshake::start(algo, client_nonce, wg_public.to_bytes());

    let mut stream = TcpStream::connect(&hs_addr).with_context(|| format!("connect {hs_addr}"))?;
    stream.set_read_timeout(Some(Duration::from_secs(10)))?;
    stream.set_write_timeout(Some(Duration::from_secs(10)))?;

    println!("-> ClientHello ({})", algo.name());
    Message::ClientHello(state.hello().clone()).write(&mut stream)?;

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
    Ok(())
}

fn trim_last_octet(ip: &std::net::Ipv4Addr) -> String {
    let o = ip.octets();
    format!("{}.{}.{}", o[0], o[1], o[2])
}
