//! Real network I/O for the handshake.
//!
//! This is the Week 6 deliverable itself
//! (`TEAM_TIMELINE_PROPOSAL.md`, Member 1 Week 6): *"Integrate `protocol/`
//! client side against Member 2's server ... First real handshake over the
//! network, Rust client <-> Rust-or-mock server."* Everything in
//! [`wire`](super::wire), [`client`](super::client) and
//! [`tags`](super::tags) is pure logic with no I/O, exercised against
//! `server/handshake-vectors.json` and an in-process simulated server; this
//! module is the one place that opens a real `TcpStream` and drives that
//! logic against whatever is actually listening on the other end —
//! `server/handshake-server` today, a mock in a test, a mobile port's own
//! socket layer later.

use std::io;
use std::net::{Ipv4Addr, SocketAddr, TcpStream, ToSocketAddrs};
use std::time::Duration;

use crate::crypto::MlKemLevel;
use crate::state::TunnelConfig;

use super::client::{ClientHandshake, HandshakeError};
use super::tags;
use super::wire::{FrameError, Message};

/// How long to wait for any single read or write during the handshake.
/// Matches the server's own idle timeout (`server/PROTOCOL.md` §2) — if the
/// server hasn't replied by then, something else is wrong and retrying
/// blindly won't help.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(10);

/// Everything that can go wrong running the handshake over the network.
#[derive(Debug)]
pub enum ProtocolError {
    /// A transport-level failure (connect, read, write, timeout).
    Io(io::Error),
    /// A frame did not decode (`server/PROTOCOL.md` §3).
    Wire(super::wire::WireError),
    /// The handshake logic itself rejected the exchange (bad signature,
    /// algorithm mismatch, ...).
    Handshake(HandshakeError),
    /// The server replied with an `Error` message (`PROTOCOL.md` §4.7).
    Server { code: u8, message: String },
    /// A structurally valid message arrived, but not the one expected next.
    UnexpectedMessage(&'static str),
}

impl std::fmt::Display for ProtocolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProtocolError::Io(e) => write!(f, "network error: {e}"),
            ProtocolError::Wire(e) => write!(f, "malformed message: {e}"),
            ProtocolError::Handshake(e) => write!(f, "{e}"),
            ProtocolError::Server { code, message } => {
                write!(f, "server rejected the handshake ({code:#04x}): {message}")
            }
            ProtocolError::UnexpectedMessage(what) => write!(f, "{what}"),
        }
    }
}

impl std::error::Error for ProtocolError {}

impl From<io::Error> for ProtocolError {
    fn from(e: io::Error) -> Self {
        ProtocolError::Io(e)
    }
}

impl From<FrameError> for ProtocolError {
    fn from(e: FrameError) -> Self {
        match e {
            FrameError::Io(e) => ProtocolError::Io(e),
            FrameError::Wire(e) => ProtocolError::Wire(e),
        }
    }
}

impl From<HandshakeError> for ProtocolError {
    fn from(e: HandshakeError) -> Self {
        ProtocolError::Handshake(e)
    }
}

/// Runs one full handshake against `addr` and returns everything
/// [`TunnelHandle::bring_up`](crate::state::TunnelHandle::bring_up) needs.
///
/// `pinned_verifying_key` is the server's 1952-byte ML-DSA-65 public key,
/// shipped with the client (`server/PROTOCOL.md` §5.2) — this function
/// never fetches it and never trusts one it wasn't given. `client_wg_pubkey`
/// is this client's own WireGuard public key, which becomes the server's
/// peer identity for the session (`PROTOCOL.md` §4.2, §8 item 5).
///
/// Four messages, one TCP connection, matching `PROTOCOL.md` §1's diagram:
/// `ClientHello -> ServerHello -> ClientFinish -> ServerFinish`. The
/// connection is closed on return (success or failure) either way, per
/// `PROTOCOL.md` §2 ("one handshake per TCP connection, then close").
pub fn connect<A: ToSocketAddrs>(
    addr: A,
    algo: MlKemLevel,
    client_wg_pubkey: [u8; 32],
    pinned_verifying_key: &[u8],
    timeout: Duration,
) -> Result<TunnelConfig, ProtocolError> {
    let mut stream = TcpStream::connect(addr)?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;
    let peer_ip = stream.peer_addr()?.ip();

    let handshake = ClientHandshake::start(algo, client_wg_pubkey);

    Message::ClientHello(handshake.hello().clone()).write(&mut stream)?;

    let server_hello = match Message::read(&mut stream)? {
        Message::ServerHello(sh) => sh,
        Message::Error(e) => {
            return Err(ProtocolError::Server { code: e.code, message: e.message });
        }
        _ => return Err(ProtocolError::UnexpectedMessage("expected ServerHello")),
    };

    let (keys, finish, transcript) = handshake.finish(&server_hello, pinned_verifying_key)?;

    Message::ClientFinish(finish.clone()).write(&mut stream)?;

    let server_finish = match Message::read(&mut stream)? {
        Message::ServerFinish(sf) => sf,
        Message::Error(e) => {
            return Err(ProtocolError::Server { code: e.code, message: e.message });
        }
        _ => return Err(ProtocolError::UnexpectedMessage("expected ServerFinish")),
    };

    if server_finish.session_id != finish.session_id {
        return Err(ProtocolError::UnexpectedMessage("ServerFinish session id did not match ClientFinish"));
    }

    if !tags::verify_server_tag(&keys.confirm_key, &transcript, &server_finish.server_tag) {
        // The server proved nothing about its own derived secret — never
        // trust the tunnel parameters that follow.
        return Err(ProtocolError::Handshake(HandshakeError::AuthFailed));
    }

    Ok(TunnelConfig {
        assigned_ip: Ipv4Addr::from(server_finish.assigned_ip),
        server_wg_pubkey: server_finish.server_wg_pubkey,
        server_endpoint: SocketAddr::new(peer_ip, server_finish.wg_port),
        psk: keys.psk,
    })
}

#[cfg(test)]
mod tests {
    use std::net::TcpListener;
    use std::sync::Arc;

    use handshake_server::identity::ServerIdentity;
    use handshake_server::registry::{Registry, DEFAULT_MAX_PEERS};
    use handshake_server::server::{self, ServerContext};
    use handshake_server::tunnel::{DryRun, TunnelSettings};

    use super::*;

    /// The Week 6 proof: a real `core::protocol::connect` client, over a
    /// real loopback `TcpStream`, against the real
    /// `server/handshake-server` connection handler (dry-run PSK install,
    /// so it needs no `CAP_NET_ADMIN` / no actual WireGuard interface) —
    /// "Rust client <-> Rust-or-mock server" from the delivery plan.
    #[test]
    fn real_handshake_over_a_loopback_socket() {
        let identity = ServerIdentity::generate();
        let pinned_vk = identity.verifying_key_bytes();

        let ctx = Arc::new(ServerContext {
            identity,
            registry: Registry::new([10, 8, 0, 0], DEFAULT_MAX_PEERS),
            installer: Box::new(DryRun),
            tunnel: TunnelSettings {
                server_wg_pubkey: [0x5a; 32],
                wg_port: 51820,
                subnet_base: [10, 8, 0, 0],
            },
        });

        let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback listener");
        let addr = listener.local_addr().unwrap();

        let server_ctx = Arc::clone(&ctx);
        let server_thread = std::thread::spawn(move || {
            let (stream, _) = listener.accept().expect("accept one connection");
            server::handle_connection(stream, &server_ctx).expect("server-side handshake")
        });

        let client_wg_pubkey = [0x11u8; 32];
        let tunnel = connect(
            addr,
            MlKemLevel::MlKem768,
            client_wg_pubkey,
            &pinned_vk,
            Duration::from_secs(5),
        )
        .expect("client-side handshake over the network");

        server_thread.join().expect("server thread panicked");

        assert_eq!(tunnel.server_wg_pubkey, [0x5a; 32]);
        assert_eq!(tunnel.server_endpoint.port(), 51820);
        assert_eq!(tunnel.assigned_ip, Ipv4Addr::new(10, 8, 0, 2));
        assert_eq!(tunnel.psk.len(), 32);
        assert!(ctx.registry.contains(&client_wg_pubkey));
    }

    #[test]
    fn wrong_pinned_key_over_the_network_is_rejected_client_side() {
        let identity = ServerIdentity::generate();
        let attacker_vk = ServerIdentity::generate().verifying_key_bytes();

        let ctx = Arc::new(ServerContext {
            identity,
            registry: Registry::new([10, 8, 0, 0], DEFAULT_MAX_PEERS),
            installer: Box::new(DryRun),
            tunnel: TunnelSettings {
                server_wg_pubkey: [0x5a; 32],
                wg_port: 51820,
                subnet_base: [10, 8, 0, 0],
            },
        });

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        let server_thread = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            // The server completes its side; only the client's signature
            // check should fail.
            let _ = server::handle_connection(stream, &ctx);
        });

        let result = connect(
            addr,
            MlKemLevel::MlKem768,
            [0x22u8; 32],
            &attacker_vk,
            Duration::from_secs(5),
        );

        server_thread.join().unwrap();

        assert!(matches!(
            result,
            Err(ProtocolError::Handshake(HandshakeError::AuthFailed))
        ));
    }
}
