//! Throwaway handshake client for testing the server (the Week 3 "throwaway
//! client script", pulled forward). Runs a full handshake and prints the
//! derived PSK.
//!
//! Usage:
//!   test-client <host:port> <verifying-key-hex | @path> [--algo 512|768|1024]

use std::net::TcpStream;
use std::time::Duration;

use anyhow::{bail, Context, Result};

use handshake_server::client_handshake;
use handshake_server::kdf;
use handshake_server::wire::{AlgoCode, Message};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let addr = args
        .next()
        .context("usage: test-client <host:port> <vk-hex|@path> [--algo 512|768|1024]")?;
    let vk_arg = args.next().context("need the server verifying key (hex or @path)")?;

    let mut algo = AlgoCode::MlKem768;
    while let Some(a) = args.next() {
        if a == "--algo" {
            algo = match args.next().as_deref() {
                Some("512") => AlgoCode::MlKem512,
                Some("768") => AlgoCode::MlKem768,
                Some("1024") => AlgoCode::MlKem1024,
                other => bail!("bad --algo {other:?} (use 512|768|1024)"),
            };
        } else {
            bail!("unknown argument {a}");
        }
    }

    let vk_hex = match vk_arg.strip_prefix('@') {
        Some(path) => std::fs::read_to_string(path)?.trim().to_string(),
        None => vk_arg,
    };
    let verifying_key = hex::decode(vk_hex.trim()).context("verifying key must be hex")?;

    let client_nonce: [u8; 32] = rand::random();
    let client_wg_pubkey: [u8; 32] = rand::random(); // stand-in for a real WG pubkey
    let state = client_handshake::start(algo, client_nonce, client_wg_pubkey);

    let mut stream = TcpStream::connect(&addr).with_context(|| format!("connect {addr}"))?;
    stream.set_read_timeout(Some(Duration::from_secs(10)))?;
    stream.set_write_timeout(Some(Duration::from_secs(10)))?;

    println!("-> ClientHello ({})", algo.name());
    Message::ClientHello(state.hello().clone()).write(&mut stream)?;

    let server_hello = match Message::read(&mut stream)? {
        Message::ServerHello(sh) => sh,
        Message::Error(e) => bail!("server returned Error {:#04x}: {}", e.code, e.message),
        other => bail!("expected ServerHello, got {other:?}"),
    };
    println!(
        "<- ServerHello  session={}  sig={} B",
        hex::encode(server_hello.session_id),
        server_hello.signature.len()
    );

    let (keys, finish, transcript) = state
        .finish(&server_hello, &verifying_key)
        .context("verifying ServerHello / deriving keys")?;
    println!("   server signature verified, keys derived");

    println!("-> ClientFinish");
    Message::ClientFinish(finish.clone()).write(&mut stream)?;

    match Message::read(&mut stream)? {
        Message::ServerFinish(sf) => {
            if sf.session_id != finish.session_id {
                bail!("ServerFinish session id mismatch");
            }
            if !kdf::verify_server_tag(&keys.confirm_key, &transcript, &sf.server_tag) {
                bail!("ServerFinish tag did not verify — server derived a different secret");
            }
            println!("<- ServerFinish verified");
        }
        Message::Error(e) => bail!("server returned Error {:#04x}: {}", e.code, e.message),
        other => bail!("expected ServerFinish, got {other:?}"),
    }

    println!("\nHANDSHAKE OK");
    println!("  algorithm    : {}", algo.name());
    println!("  session id   : {}", hex::encode(finish.session_id));
    println!("  derived PSK   : {}", hex::encode(keys.psk));
    println!("  (this PSK is what both ends would load with `wg set ... preshared-key`)");
    Ok(())
}
