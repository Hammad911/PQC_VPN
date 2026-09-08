//! Generates `server/handshake-vectors.json` — deterministic fixtures for the
//! wire protocol, so Member 1 can verify the client side of `core/protocol/`
//! against the exact same bytes (same pattern as `contracts/*_vectors.json`).
//!
//! Only the deterministic layers are covered: framing, the signed transcript,
//! the key schedule, and the finish MACs. The KEM/DH/signature roundtrip is
//! non-deterministic (fresh randomness) and is covered by `core`'s own tests,
//! `crypto-spike`, and the integration test.
//!
//! Usage:  emit-vectors [output-path]      (default: server/handshake-vectors.json)

use std::path::PathBuf;

use sha2::{Digest, Sha256};
use serde_json::json;

use handshake_server::kdf;
use handshake_server::transcript::{self, TranscriptInputs};
use handshake_server::wire::{
    AlgoCode, ClientFinish, ClientHello, ErrorMsg, Message, RekeyRequest, ServerFinish, ServerHello,
};

fn frame_hex(m: &Message) -> String {
    let mut buf = Vec::new();
    m.write(&mut buf).expect("in-memory write");
    hex::encode(buf)
}

fn main() {
    let out = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("server/handshake-vectors.json"));

    // ---- fixed inputs ----
    let client_nonce = [0x11u8; 32];
    let server_nonce = [0x22u8; 32];
    let client_x25519 = [0x33u8; 32];
    let server_x25519 = [0x55u8; 32];
    let client_mlkem_pub = vec![0x44u8; AlgoCode::MlKem768.mlkem_pub_len()];
    let ciphertext = vec![0x66u8; AlgoCode::MlKem768.ciphertext_len()];
    let hybrid_secret = [0x77u8; 32];
    let session_id = [0xABu8; 16];

    // ---- transcript ----
    let t = transcript::build(&TranscriptInputs {
        algo: AlgoCode::MlKem768,
        client_nonce: &client_nonce,
        server_nonce: &server_nonce,
        client_x25519_pub: &client_x25519,
        client_mlkem_pub: &client_mlkem_pub,
        server_x25519_pub: &server_x25519,
        mlkem_ciphertext: &ciphertext,
    });
    let t_sha = hex::encode(Sha256::digest(&t));

    // ---- key schedule ----
    let keys = kdf::derive(&hybrid_secret, &client_nonce, &server_nonce);
    let client_tag = kdf::client_tag(&keys.confirm_key, &t);
    let server_tag = kdf::server_tag(&keys.confirm_key, &t);

    // ---- framing samples ----
    let sig_stub = vec![0x99u8; handshake_server::wire::SIGNATURE_LEN];
    let framing = json!([
        {
            "name": "client_hello_mlkem768",
            "frame_hex": frame_hex(&Message::ClientHello(ClientHello {
                algo: AlgoCode::MlKem768,
                client_nonce,
                client_wg_pubkey: [0x01u8; 32],
                client_x25519_pub: client_x25519,
                client_mlkem_pub: client_mlkem_pub.clone(),
            })),
        },
        {
            "name": "server_hello_mlkem768",
            "frame_hex": frame_hex(&Message::ServerHello(ServerHello {
                algo: AlgoCode::MlKem768,
                session_id,
                server_nonce,
                server_x25519_pub: server_x25519,
                mlkem_ciphertext: ciphertext.clone(),
                signature: sig_stub.clone(),
            })),
        },
        {
            "name": "client_finish",
            "frame_hex": frame_hex(&Message::ClientFinish(ClientFinish {
                session_id,
                client_tag,
            })),
        },
        {
            "name": "server_finish",
            "frame_hex": frame_hex(&Message::ServerFinish(ServerFinish {
                session_id,
                server_tag,
                server_wg_pubkey: [0x5au8; 32],
                assigned_ip: [10, 8, 0, 2],
                wg_port: 51820,
            })),
        },
        {
            "name": "rekey_request_mlkem1024",
            "frame_hex": frame_hex(&Message::RekeyRequest(RekeyRequest {
                session_id,
                hello: ClientHello {
                    algo: AlgoCode::MlKem1024,
                    client_nonce,
                    client_wg_pubkey: [0x01u8; 32],
                    client_x25519_pub: client_x25519,
                    client_mlkem_pub: vec![0x44u8; AlgoCode::MlKem1024.mlkem_pub_len()],
                },
            })),
        },
        {
            "name": "error_algo_mismatch",
            "frame_hex": frame_hex(&Message::Error(ErrorMsg {
                code: 0x05,
                message: "rekey may not downgrade the algorithm".into(),
            })),
        },
    ]);

    let doc = json!({
        "schema_version": 1,
        "generated_by": "cargo run -p handshake-server --bin emit-vectors",
        "protocol": "server/PROTOCOL.md v1 DRAFT",
        "note": "Deterministic layers only. The KEM/DH/signature roundtrip uses fresh randomness and is covered by core's tests, crypto-spike, and the integration test.",
        "framing": framing,
        "transcript": {
            "algo": "ML-KEM-768",
            "client_nonce_hex": hex::encode(client_nonce),
            "server_nonce_hex": hex::encode(server_nonce),
            "client_x25519_pub_hex": hex::encode(client_x25519),
            "client_mlkem_pub_hex": hex::encode(&client_mlkem_pub),
            "server_x25519_pub_hex": hex::encode(server_x25519),
            "mlkem_ciphertext_hex": hex::encode(&ciphertext),
            "transcript_hex": hex::encode(&t),
            "transcript_sha256": t_sha,
        },
        "kdf": {
            "hybrid_secret_hex": hex::encode(hybrid_secret),
            "client_nonce_hex": hex::encode(client_nonce),
            "server_nonce_hex": hex::encode(server_nonce),
            "psk_hex": hex::encode(keys.psk),
            "confirm_key_hex": hex::encode(keys.confirm_key),
        },
        "mac": {
            "confirm_key_hex": hex::encode(keys.confirm_key),
            "transcript_hex": hex::encode(&t),
            "client_tag_hex": hex::encode(client_tag),
            "server_tag_hex": hex::encode(server_tag),
        },
    });

    let text = serde_json::to_string_pretty(&doc).unwrap();
    std::fs::write(&out, text + "\n").expect("write vectors file");
    eprintln!("wrote {}", out.display());
}
