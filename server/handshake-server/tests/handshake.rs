//! End-to-end handshake tests.
//!
//! `in_process_*` exercise the crypto/protocol logic directly.
//! `over_tcp_*` run a real `TcpListener` + `handle_connection` (with a dry-run
//! PSK installer), so `Message::read`/`write` and the registry are exercised.

use std::collections::HashMap;
use std::net::{Ipv4Addr, TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use handshake_server::client_handshake;
use handshake_server::handshake;
use handshake_server::identity::ServerIdentity;
use handshake_server::kdf;
use handshake_server::registry::{Registry, DEFAULT_MAX_PEERS};
use handshake_server::server::{self, ConnResult, ServerContext};
use handshake_server::tunnel::{DryRun, InstallError, PskInstaller, TunnelSettings};
use handshake_server::wire::{AlgoCode, Message, RekeyRequest};
use handshake_server::HandshakeError;

const ALL: [AlgoCode; 3] = [AlgoCode::MlKem512, AlgoCode::MlKem768, AlgoCode::MlKem1024];

fn test_ctx() -> Arc<ServerContext> {
    Arc::new(ServerContext {
        identity: ServerIdentity::generate(),
        registry: Registry::new([10, 8, 0, 0], DEFAULT_MAX_PEERS),
        installer: Box::new(DryRun),
        tunnel: TunnelSettings {
            server_wg_pubkey: [0x5a; 32],
            wg_port: 51820,
            subnet_base: [10, 8, 0, 0],
        },
    })
}

#[test]
fn in_process_handshake_all_levels_agree_on_psk() {
    let identity = ServerIdentity::generate();
    let vk = identity.verifying_key_bytes();

    for algo in ALL {
        let state = client_handshake::start(algo, [1u8; 32], [2u8; 32]);
        let outcome =
            handshake::respond(&identity, state.hello(), [9u8; 16], [8u8; 32]).expect("respond");

        let (client_keys, finish, transcript) =
            state.finish(&outcome.server_hello, &vk).expect("client finish");

        assert_eq!(client_keys.psk, outcome.keys.psk, "{algo:?}: PSK mismatch");
        assert_eq!(client_keys.confirm_key, outcome.keys.confirm_key);
        assert!(kdf::verify_client_tag(
            &outcome.keys.confirm_key,
            &outcome.transcript,
            &finish.client_tag
        ));
        let server_tag = kdf::server_tag(&outcome.keys.confirm_key, &outcome.transcript);
        assert!(kdf::verify_server_tag(&client_keys.confirm_key, &transcript, &server_tag));
    }
}

#[test]
fn wrong_pinned_key_is_rejected() {
    let identity = ServerIdentity::generate();
    let attacker = ServerIdentity::generate();
    let state = client_handshake::start(AlgoCode::MlKem768, [1u8; 32], [2u8; 32]);
    let outcome = handshake::respond(&identity, state.hello(), [0u8; 16], [0u8; 32]).unwrap();
    assert!(matches!(
        state.finish(&outcome.server_hello, &attacker.verifying_key_bytes()),
        Err(HandshakeError::AuthFailed)
    ));
}

#[test]
fn tampered_server_hello_fails_auth() {
    let identity = ServerIdentity::generate();
    let state = client_handshake::start(AlgoCode::MlKem768, [1u8; 32], [2u8; 32]);
    let mut outcome = handshake::respond(&identity, state.hello(), [0u8; 16], [0u8; 32]).unwrap();
    outcome.server_hello.mlkem_ciphertext[0] ^= 0x01;
    assert!(matches!(
        state.finish(&outcome.server_hello, &identity.verifying_key_bytes()),
        Err(HandshakeError::AuthFailed)
    ));
}

/// Run one full handshake over a loopback socket. Returns the assigned IP.
fn tcp_handshake(ctx: &Arc<ServerContext>, wg_pub: [u8; 32], algo: AlgoCode) -> [u8; 4] {
    let vk = ctx.identity.verifying_key_bytes();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();

    let srv_ctx = Arc::clone(ctx);
    let server = std::thread::spawn(move || {
        let (stream, _) = listener.accept().unwrap();
        server::handle_connection(stream, &srv_ctx).unwrap()
    });

    let state = client_handshake::start(algo, rand_nonce(), wg_pub);
    let mut stream = TcpStream::connect(addr).unwrap();
    stream.set_read_timeout(Some(Duration::from_secs(5))).unwrap();

    Message::ClientHello(state.hello().clone()).write(&mut stream).unwrap();
    let sh = match Message::read(&mut stream).unwrap() {
        Message::ServerHello(sh) => sh,
        other => panic!("expected ServerHello, got {other:?}"),
    };
    let (keys, finish, transcript) = state.finish(&sh, &vk).unwrap();
    Message::ClientFinish(finish.clone()).write(&mut stream).unwrap();

    let assigned = match Message::read(&mut stream).unwrap() {
        Message::ServerFinish(sf) => {
            assert_eq!(sf.session_id, finish.session_id);
            assert!(kdf::verify_server_tag(&keys.confirm_key, &transcript, &sf.server_tag));
            assert_eq!(sf.server_wg_pubkey, [0x5a; 32]);
            assert_eq!(sf.wg_port, 51820);
            sf.assigned_ip
        }
        other => panic!("expected ServerFinish, got {other:?}"),
    };
    match server.join().unwrap() {
        ConnResult::Handshake { session_id } => assert_eq!(session_id, finish.session_id),
        other => panic!("expected Handshake, got {other:?}"),
    }
    assigned
}

fn rand_nonce() -> [u8; 32] {
    let mut n = [0u8; 32];
    for (i, b) in n.iter_mut().enumerate() {
        *b = (i as u8).wrapping_mul(37).wrapping_add(1);
    }
    n
}

#[test]
fn over_tcp_two_peers_get_distinct_addresses_and_sessions() {
    let ctx = test_ctx();
    let ip1 = tcp_handshake(&ctx, [0x11; 32], AlgoCode::MlKem768);
    let ip2 = tcp_handshake(&ctx, [0x22; 32], AlgoCode::MlKem1024);

    assert_eq!(ip1, [10, 8, 0, 2]);
    assert_eq!(ip2, [10, 8, 0, 3]);
    assert_eq!(ctx.registry.len(), 2);
    assert!(ctx.registry.contains(&[0x11; 32]));
    assert!(ctx.registry.contains(&[0x22; 32]));

    // same peer re-handshakes -> same address, still one entry
    let ip1_again = tcp_handshake(&ctx, [0x11; 32], AlgoCode::MlKem512);
    assert_eq!(ip1_again, ip1);
    assert_eq!(ctx.registry.len(), 2);
}

// ---------------------------------------------------------------------------
// Week 6 — live rekey: RekeyRequest over real TCP, proving the PSK swap never
// touches the WireGuard peer entry (no `remove_peer` call = no dropped tunnel).
// ---------------------------------------------------------------------------

/// Tracks every `install`/`remove_peer` call so a test can assert the tunnel
/// was never torn down across a rekey.
#[derive(Default, Clone)]
struct RecordingInstaller {
    installs: Arc<Mutex<Vec<([u8; 32], [u8; 32], Ipv4Addr, AlgoCode)>>>,
    removals: Arc<Mutex<Vec<[u8; 32]>>>,
}

impl PskInstaller for RecordingInstaller {
    fn install(
        &self,
        peer_wg_pubkey: &[u8; 32],
        psk: &[u8; 32],
        assigned_ip: Ipv4Addr,
        algo: AlgoCode,
    ) -> Result<(), InstallError> {
        self.installs.lock().unwrap().push((*peer_wg_pubkey, *psk, assigned_ip, algo));
        Ok(())
    }
    fn list_installed_peers(&self) -> Result<Vec<[u8; 32]>, InstallError> {
        Ok(self.installs.lock().unwrap().iter().map(|(pk, ..)| *pk).collect())
    }
    fn remove_peer(&self, peer_wg_pubkey: &[u8; 32]) -> Result<(), InstallError> {
        self.removals.lock().unwrap().push(*peer_wg_pubkey);
        Ok(())
    }
    fn last_handshakes(&self) -> Result<HashMap<[u8; 32], SystemTime>, InstallError> {
        Ok(HashMap::new())
    }
    fn describe(&self) -> String {
        "recording (test)".into()
    }
}

fn recording_ctx() -> (Arc<ServerContext>, RecordingInstaller) {
    let rec = RecordingInstaller::default();
    let ctx = Arc::new(ServerContext {
        identity: ServerIdentity::generate(),
        registry: Registry::new([10, 8, 0, 0], DEFAULT_MAX_PEERS),
        installer: Box::new(rec.clone()),
        tunnel: TunnelSettings {
            server_wg_pubkey: [0x5a; 32],
            wg_port: 51820,
            subnet_base: [10, 8, 0, 0],
        },
    });
    (ctx, rec)
}

/// One connection over loopback: `ClientHello` if `rekey_session` is `None`,
/// else `RekeyRequest` claiming that session. Returns the server's
/// authoritative [`ConnResult`] plus, on a successful finish, the assigned IP
/// and derived PSK the client saw.
fn tcp_connect(
    ctx: &Arc<ServerContext>,
    wg_pub: [u8; 32],
    algo: AlgoCode,
    rekey_session: Option<[u8; 16]>,
) -> (ConnResult, Option<([u8; 4], [u8; 32])>) {
    let vk = ctx.identity.verifying_key_bytes();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();

    let srv_ctx = Arc::clone(ctx);
    let server = std::thread::spawn(move || {
        let (stream, _) = listener.accept().unwrap();
        server::handle_connection(stream, &srv_ctx).unwrap()
    });

    let state = client_handshake::start(algo, rand_nonce(), wg_pub);
    let mut stream = TcpStream::connect(addr).unwrap();
    stream.set_read_timeout(Some(Duration::from_secs(5))).unwrap();

    if let Some(session_id) = rekey_session {
        Message::RekeyRequest(RekeyRequest { session_id, hello: state.hello().clone() })
            .write(&mut stream)
            .unwrap();
    } else {
        Message::ClientHello(state.hello().clone()).write(&mut stream).unwrap();
    }

    let sh = match Message::read(&mut stream).unwrap() {
        Message::ServerHello(sh) => sh,
        Message::Error(_) => return (server.join().unwrap(), None),
        other => panic!("expected ServerHello, got {other:?}"),
    };
    let (keys, finish, transcript) = state.finish(&sh, &vk).unwrap();
    Message::ClientFinish(finish.clone()).write(&mut stream).unwrap();

    let out = match Message::read(&mut stream).unwrap() {
        Message::ServerFinish(sf) => {
            assert_eq!(sf.session_id, finish.session_id);
            assert!(kdf::verify_server_tag(&keys.confirm_key, &transcript, &sf.server_tag));
            Some((sf.assigned_ip, keys.psk))
        }
        Message::Error(_) => None,
        other => panic!("expected ServerFinish or Error, got {other:?}"),
    };
    (server.join().unwrap(), out)
}

#[test]
fn over_tcp_rekey_swaps_psk_without_removing_peer() {
    let (ctx, rec) = recording_ctx();
    let wg_pub = [0x33; 32];

    let (result, out) = tcp_connect(&ctx, wg_pub, AlgoCode::MlKem768, None);
    let session_id = match result {
        ConnResult::Handshake { session_id } => session_id,
        other => panic!("expected Handshake, got {other:?}"),
    };
    let (ip1, psk1) = out.expect("first handshake should succeed");

    // The agent triggers rekey-now and escalates, per REKEY_ESCALATES.
    let (result, out) = tcp_connect(&ctx, wg_pub, AlgoCode::MlKem1024, Some(session_id));
    match result {
        ConnResult::Rekey { session_id: sid } => assert_eq!(sid, session_id),
        other => panic!("expected Rekey, got {other:?}"),
    }
    let (ip2, psk2) = out.expect("rekey should succeed");

    assert_eq!(ip1, ip2, "rekey must not move the peer's tunnel address");
    assert_ne!(psk1, psk2, "rekey must derive a fresh PSK");
    assert_eq!(ctx.registry.len(), 1, "rekey must not create a second peer entry");

    let installs = rec.installs.lock().unwrap();
    assert_eq!(installs.len(), 2, "one wg set per handshake/rekey");
    assert!(installs.iter().all(|(pk, _, ip, _)| *pk == wg_pub && *ip == Ipv4Addr::from(ip1)));
    assert_eq!(installs[0].3, AlgoCode::MlKem768);
    assert_eq!(installs[1].3, AlgoCode::MlKem1024);
    assert_ne!(installs[0].1, installs[1].1, "the two `wg set` calls used different PSKs");

    assert!(
        rec.removals.lock().unwrap().is_empty(),
        "a rekey must never remove the wg peer — that would drop the tunnel"
    );
}

#[test]
fn over_tcp_rekey_rejects_unknown_session() {
    let (ctx, rec) = recording_ctx();
    // No prior handshake for this peer at all.
    let (result, out) = tcp_connect(&ctx, [0x44; 32], AlgoCode::MlKem768, Some([0xAB; 16]));
    assert_eq!(result, ConnResult::Rejected { code: 0x04 });
    assert!(out.is_none());
    assert!(rec.installs.lock().unwrap().is_empty());
}

#[test]
fn over_tcp_rekey_rejects_session_mismatch_for_known_peer() {
    let (ctx, rec) = recording_ctx();
    let wg_pub = [0x55; 32];
    let (result, _) = tcp_connect(&ctx, wg_pub, AlgoCode::MlKem768, None);
    let session_id = match result {
        ConnResult::Handshake { session_id } => session_id,
        other => panic!("expected Handshake, got {other:?}"),
    };

    let mut wrong_session = session_id;
    wrong_session[0] ^= 0xFF;
    let (result, out) = tcp_connect(&ctx, wg_pub, AlgoCode::MlKem768, Some(wrong_session));
    assert_eq!(result, ConnResult::Rejected { code: 0x04 });
    assert!(out.is_none());
    assert_eq!(rec.installs.lock().unwrap().len(), 1, "only the original handshake installed");
    assert!(rec.removals.lock().unwrap().is_empty());
}

#[test]
fn over_tcp_rekey_rejects_downgrade() {
    let (ctx, rec) = recording_ctx();
    let wg_pub = [0x66; 32];
    let (result, _) = tcp_connect(&ctx, wg_pub, AlgoCode::MlKem1024, None);
    let session_id = match result {
        ConnResult::Handshake { session_id } => session_id,
        other => panic!("expected Handshake, got {other:?}"),
    };

    let (result, out) = tcp_connect(&ctx, wg_pub, AlgoCode::MlKem512, Some(session_id));
    assert_eq!(result, ConnResult::Rejected { code: 0x05 });
    assert!(out.is_none());
    assert_eq!(rec.installs.lock().unwrap().len(), 1, "the downgrade must not be installed");
    assert!(rec.removals.lock().unwrap().is_empty());
}
