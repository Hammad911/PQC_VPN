//! PQC-VPN handshake server — `PROTOCOL.md` v1.
//!
//! Usage:
//!   handshake-server [--bind ADDR] [--identity PATH]
//!   defaults: --bind 0.0.0.0:51821  --identity ./server-identity.seed

use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};

use handshake_server::identity::ServerIdentity;
use handshake_server::server;
use handshake_server::session::SessionTable;

fn main() -> Result<()> {
    let mut bind = "0.0.0.0:51821".to_string();
    let mut identity_path = PathBuf::from("server-identity.seed");
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--bind" => bind = args.next().context("--bind needs a value")?,
            "--identity" => {
                identity_path = PathBuf::from(args.next().context("--identity needs a value")?)
            }
            "-h" | "--help" => {
                println!("handshake-server [--bind ADDR] [--identity PATH]");
                return Ok(());
            }
            other => anyhow::bail!("unknown argument: {other}"),
        }
    }

    let identity = Arc::new(
        ServerIdentity::load_or_create(&identity_path)
            .with_context(|| format!("loading identity from {}", identity_path.display()))?,
    );
    let sessions = Arc::new(SessionTable::new());

    let vk = identity.verifying_key_bytes();
    let vk_hex = hex::encode(&vk);
    let pub_path = identity_path.with_extension("pub");
    std::fs::write(&pub_path, &vk_hex).ok();

    println!("identity seed  : {}", identity_path.display());
    println!("verifying key  : {} ({} bytes)", pub_path.display(), vk.len());
    println!("               : {vk_hex}");

    let listener = TcpListener::bind(&bind).with_context(|| format!("bind {bind}"))?;
    println!("listening on   : {}", listener.local_addr()?);
    println!("note: PSK installation (wg set) is stubbed until Week 3\n");

    server::serve(listener, identity, sessions);
    Ok(())
}
