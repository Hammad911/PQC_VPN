//! TCP server loop, per-connection handler, and multi-peer housekeeping.
//!
//! Blocking, one thread per connection, one handshake (or rekey) per connection.
//! A completed handshake updates the [`Registry`] and installs the derived PSK
//! into WireGuard.

use std::collections::HashSet;
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;

use crate::error::HandshakeError;
use crate::handshake;
use crate::identity::ServerIdentity;
use crate::kdf;
use crate::registry::{PeerSlot, Registry, RegistryError};
use crate::tunnel::{InstallError, PskInstaller, TunnelSettings};
use crate::wire::{ClientHello, ErrorMsg, FrameError, Message, ServerFinish};

pub const CONN_TIMEOUT: Duration = Duration::from_secs(10);
const SWEEP_INTERVAL: Duration = Duration::from_secs(60);

/// Everything a connection handler needs, shared across threads.
pub struct ServerContext {
    pub identity: ServerIdentity,
    pub registry: Registry,
    pub installer: Box<dyn PskInstaller>,
    pub tunnel: TunnelSettings,
}

fn b64(k: &[u8; 32]) -> String {
    B64.encode(k)
}

/// Remove `wg0` peers the registry does not know about (e.g. leftovers from a
/// previous run). Registered peers missing from `wg0` are kept — their clients
/// re-handshake.
pub fn reconcile_wg_peers(ctx: &ServerContext) {
    let installed = match ctx.installer.list_installed_peers() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("reconcile: could not list wg peers: {e}");
            return;
        }
    };
    let known: HashSet<[u8; 32]> = ctx.registry.pubkeys().into_iter().collect();
    let installed_set: HashSet<[u8; 32]> = installed.iter().copied().collect();

    let mut removed = 0;
    for pk in &installed {
        if !known.contains(pk) {
            match ctx.installer.remove_peer(pk) {
                Ok(()) => removed += 1,
                Err(e) => eprintln!("reconcile: failed to remove {}: {e}", b64(pk)),
            }
        }
    }
    let absent = known.iter().filter(|k| !installed_set.contains(*k)).count();
    println!(
        "reconcile: {} known / {} on wg0 / {removed} stale removed / {absent} awaiting re-handshake",
        known.len(),
        installed.len()
    );
}

/// Background thread: every minute, evict peers whose last activity *and* last
/// WireGuard handshake are both older than `timeout`.
pub fn spawn_idle_sweep(ctx: Arc<ServerContext>, timeout: Duration) {
    std::thread::spawn(move || loop {
        std::thread::sleep(SWEEP_INTERVAL);
        let wg_hs = ctx.installer.last_handshakes().unwrap_or_default();
        let now = SystemTime::now();
        let mut evicted = 0;
        for pk in ctx.registry.idle_peers(timeout, now) {
            // If WireGuard still shows a recent handshake, the tunnel is live —
            // keep it and just refresh our clock.
            if let Some(hs) = wg_hs.get(&pk) {
                if now.duration_since(*hs).map(|d| d <= timeout).unwrap_or(false) {
                    ctx.registry.touch(&pk);
                    continue;
                }
            }
            ctx.registry.remove(&pk);
            let _ = ctx.installer.remove_peer(&pk);
            println!("idle-sweep: evicted peer {}", b64(&pk));
            evicted += 1;
        }
        if evicted > 0 {
            let _ = ctx.registry.persist();
        }
    });
}

/// Accept loop. Returns only if the listener errors fatally.
pub fn serve(listener: TcpListener, ctx: Arc<ServerContext>) {
    for stream in listener.incoming() {
        match stream {
            Ok(s) => {
                let ctx = Arc::clone(&ctx);
                std::thread::spawn(move || {
                    if let Err(e) = handle_connection(s, &ctx) {
                        eprintln!("connection error: {e}");
                    }
                });
            }
            Err(e) => eprintln!("accept error: {e}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConnResult {
    Handshake { session_id: [u8; 16] },
    Rekey { session_id: [u8; 16] },
    Rejected { code: u8 },
}

fn registry_reject_code(e: &RegistryError) -> (u8, &'static str) {
    match e {
        RegistryError::AtCapacity => (0x09, "server at peer capacity"),
        RegistryError::UnknownPeer | RegistryError::SessionMismatch => {
            (0x04, "unknown session for this peer")
        }
        RegistryError::Downgrade => (0x05, "rekey may not downgrade the algorithm"),
        RegistryError::AddressPoolExhausted => (0x08, "tunnel address pool exhausted"),
    }
}

pub fn handle_connection(mut stream: TcpStream, ctx: &ServerContext) -> Result<ConnResult, ConnError> {
    stream.set_read_timeout(Some(CONN_TIMEOUT))?;
    stream.set_write_timeout(Some(CONN_TIMEOUT))?;

    // --- message 1: ClientHello or RekeyRequest ---
    let (hello, rekey_session): (ClientHello, Option<[u8; 16]>) = match Message::read(&mut stream) {
        Ok(Message::ClientHello(h)) => (h, None),
        Ok(Message::RekeyRequest(r)) => (r.hello, Some(r.session_id)),
        Ok(other) => {
            reject(&mut stream, 0x02, "expected ClientHello or RekeyRequest")?;
            return Err(ConnError::Protocol(format!("unexpected first message: {other:?}")));
        }
        Err(e) => {
            let _ = reject(&mut stream, 0x02, &e.to_string());
            return Err(e.into());
        }
    };

    let session_id = rekey_session.unwrap_or_else(|| rand::random());
    let server_nonce: [u8; 32] = rand::random();

    let outcome = match handshake::respond(&ctx.identity, &hello, session_id, server_nonce) {
        Ok(o) => o,
        Err(e) => {
            let code = e.protocol_code();
            reject(&mut stream, code, &e.to_string())?;
            return Ok(ConnResult::Rejected { code });
        }
    };

    Message::ServerHello(outcome.server_hello.clone()).write(&mut stream)?;

    // --- message 3: ClientFinish ---
    let fin = match Message::read(&mut stream)? {
        Message::ClientFinish(f) => f,
        other => {
            reject(&mut stream, 0x02, "expected ClientFinish")?;
            return Err(ConnError::Protocol(format!("expected ClientFinish, got {other:?}")));
        }
    };
    if fin.session_id != session_id
        || !kdf::verify_client_tag(&outcome.keys.confirm_key, &outcome.transcript, &fin.client_tag)
    {
        reject(&mut stream, HandshakeError::AuthFailed.protocol_code(), "client finish tag did not verify")?;
        return Ok(ConnResult::Rejected { code: 0x06 });
    }

    // --- update the registry ---
    let (slot, is_rekey): (PeerSlot, bool) = if let Some(sid) = rekey_session {
        match ctx.registry.rekey(&hello.client_wg_pubkey, &sid, hello.algo) {
            Ok(s) => (s, true),
            Err(e) => {
                let (code, msg) = registry_reject_code(&e);
                reject(&mut stream, code, msg)?;
                return Ok(ConnResult::Rejected { code });
            }
        }
    } else {
        match ctx.registry.upsert_handshake(hello.client_wg_pubkey, hello.algo, session_id) {
            Ok(s) => (s, false),
            Err(e) => {
                let (code, msg) = registry_reject_code(&e);
                reject(&mut stream, code, msg)?;
                return Ok(ConnResult::Rejected { code });
            }
        }
    };

    // --- install the PSK ---
    if let Err(e) = ctx.installer.install(
        &hello.client_wg_pubkey,
        &outcome.keys.psk,
        slot.assigned_ip,
        hello.algo,
    ) {
        reject(&mut stream, 0x08, &e.to_string())?;
        return Err(ConnError::Install(e));
    }
    if let Err(e) = ctx.registry.persist() {
        eprintln!("registry persist failed: {e}");
    }

    let peer_hex = b64(&hello.client_wg_pubkey);
    let kind = if is_rekey { "REKEY    " } else { "HANDSHAKE" };
    println!(
        "{kind}  session={}  algo={}  peer {peer_hex} @ {}  (peers: {})",
        hex::encode(slot.session_id),
        hello.algo.name(),
        slot.assigned_ip,
        ctx.registry.len(),
    );

    // --- message 4: ServerFinish ---
    let server_tag = kdf::server_tag(&outcome.keys.confirm_key, &outcome.transcript);
    Message::ServerFinish(ServerFinish {
        session_id: slot.session_id,
        server_tag,
        server_wg_pubkey: ctx.tunnel.server_wg_pubkey,
        assigned_ip: slot.assigned_ip.octets(),
        wg_port: ctx.tunnel.wg_port,
    })
    .write(&mut stream)?;

    Ok(if is_rekey {
        ConnResult::Rekey { session_id: slot.session_id }
    } else {
        ConnResult::Handshake { session_id: slot.session_id }
    })
}

fn reject<W: std::io::Write>(w: &mut W, code: u8, message: &str) -> std::io::Result<()> {
    Message::Error(ErrorMsg { code, message: message.to_string() }).write(w)
}

#[derive(Debug)]
pub enum ConnError {
    Io(std::io::Error),
    Frame(FrameError),
    Protocol(String),
    Install(InstallError),
}

impl std::fmt::Display for ConnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConnError::Io(e) => write!(f, "io: {e}"),
            ConnError::Frame(e) => write!(f, "frame: {e}"),
            ConnError::Protocol(m) => write!(f, "protocol: {m}"),
            ConnError::Install(e) => write!(f, "psk install: {e}"),
        }
    }
}
impl std::error::Error for ConnError {}
impl From<std::io::Error> for ConnError {
    fn from(e: std::io::Error) -> Self {
        ConnError::Io(e)
    }
}
impl From<FrameError> for ConnError {
    fn from(e: FrameError) -> Self {
        ConnError::Frame(e)
    }
}
