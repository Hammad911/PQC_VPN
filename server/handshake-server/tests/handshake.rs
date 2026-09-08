//! End-to-end handshake tests.
//!
//! `in_process_*` exercise the crypto/protocol logic directly.
//! `over_tcp_*` run a real `TcpListener` + `handle_connection` (with a dry-run
//! PSK installer), so `Message::read`/`write` and the registry are exercised.

use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::time::Duration;

use handshake_server::client_handshake;
use handshake_server::handshake;
use handshake_server::identity::ServerIdentity;
use handshake_server::kdf;
use handshake_server::registry::{Registry, DEFAULT_MAX_PEERS};
use handshake_server::server::{self, ConnResult, ServerContext};
use handshake_server::tunnel::{DryRun, TunnelSettings};
use handshake_server::wire::{AlgoCode, Message};
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
