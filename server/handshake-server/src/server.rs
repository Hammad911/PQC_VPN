//! TCP server loop and per-connection handler — `PROTOCOL.md` §2.
//!
//! Blocking, one thread per connection, one handshake (or rekey) per connection.
//! Week 2: completes the authenticated handshake and records the PSK; the
//! `wg set` install is a logged stub until Week 3.

use std::io::Write;
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use crate::error::HandshakeError;
use crate::handshake;
use crate::identity::ServerIdentity;
use crate::kdf;
use crate::session::{Session, SessionTable};
use crate::wire::{ClientHello, ErrorMsg, FrameError, Message, ServerFinish};

pub const CONN_TIMEOUT: Duration = Duration::from_secs(10);

/// Accept loop. Never returns unless the listener errors fatally.
pub fn serve(listener: TcpListener, identity: Arc<ServerIdentity>, sessions: Arc<SessionTable>) {
    for stream in listener.incoming() {
        match stream {
            Ok(s) => {
                let identity = Arc::clone(&identity);
                let sessions = Arc::clone(&sessions);
                std::thread::spawn(move || {
                    if let Err(e) = handle_connection(s, &identity, &sessions) {
                        eprintln!("connection error: {e}");
                    }
                });
            }
            Err(e) => eprintln!("accept error: {e}"),
        }
    }
}

/// Outcome of one connection, for logging / tests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConnResult {
    Handshake { session_id: [u8; 16] },
    Rekey { session_id: [u8; 16] },
    Rejected { code: u8 },
}

pub fn handle_connection(
    mut stream: TcpStream,
    identity: &ServerIdentity,
    sessions: &SessionTable,
) -> Result<ConnResult, ConnError> {
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

    // --- rekey policy (PROTOCOL.md §6) ---
    if let Some(sid) = rekey_session {
        match sessions.in_force_algo(&sid) {
            None => {
                reject(&mut stream, 0x04, "unknown session id")?;
                return Ok(ConnResult::Rejected { code: 0x04 });
            }
            Some(in_force) if hello.algo < in_force => {
                reject(&mut stream, 0x05, "rekey may not downgrade the algorithm")?;
                return Ok(ConnResult::Rejected { code: 0x05 });
            }
            Some(_) => {} // equal, or stronger (REKEY_ESCALATES default on)
        }
    }

    let session_id = rekey_session.unwrap_or_else(|| rand::random());
    let server_nonce: [u8; 32] = rand::random();

    let outcome = match handshake::respond(identity, &hello, session_id, server_nonce) {
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

    // --- record + (stubbed) PSK install ---
    let peer_hex = hex::encode(hello.client_wg_pubkey);
    let sid_hex = hex::encode(session_id);
    let result = if rekey_session.is_some() {
        sessions.apply_rekey(&session_id, hello.algo, outcome.keys.psk);
        println!(
            "REKEY      session={sid_hex}  algo={}  -> would swap PSK for wg peer {peer_hex}",
            hello.algo.name()
        );
        ConnResult::Rekey { session_id }
    } else {
        sessions.insert(Session {
            session_id,
            peer_wg_pubkey: hello.client_wg_pubkey,
            in_force_algo: hello.algo,
            psk: outcome.keys.psk,
            created: SystemTime::now(),
            rekeys: 0,
        });
        println!(
            "HANDSHAKE  session={sid_hex}  algo={}  -> would install PSK for wg peer {peer_hex}  (sessions: {})",
            hello.algo.name(),
            sessions.len()
        );
        ConnResult::Handshake { session_id }
    };

    // --- message 4: ServerFinish ---
    let server_tag = kdf::server_tag(&outcome.keys.confirm_key, &outcome.transcript);
    Message::ServerFinish(ServerFinish { session_id, server_tag }).write(&mut stream)?;

    Ok(result)
}

fn reject<W: Write>(w: &mut W, code: u8, message: &str) -> std::io::Result<()> {
    Message::Error(ErrorMsg { code, message: message.to_string() }).write(w)
}

#[derive(Debug)]
pub enum ConnError {
    Io(std::io::Error),
    Frame(FrameError),
    Protocol(String),
}

impl std::fmt::Display for ConnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConnError::Io(e) => write!(f, "io: {e}"),
            ConnError::Frame(e) => write!(f, "frame: {e}"),
            ConnError::Protocol(m) => write!(f, "protocol: {m}"),
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
