//! PQC crypto spike for the RL-PQC-VPN server (Member 2, Week 1).
//!
//! Throwaway validation binary. **This is not the server.**
//!
//! Member 1 has already ported the cryptographic layer to Rust in `core/crypto/`
//! (pure-Rust `ml-kem` / `ml-dsa` / `x25519-dalek`, no liboqs). This binary
//! exercises that crate exactly the way Member 2's handshake responder will, and
//! prints the encoded byte size of every value that has to cross the wire - the
//! numbers the Week 1 wire-protocol document needs in order to define message
//! framing.
//!
//! Run:  cargo run -p crypto-spike     (from the workspace root)

use ml_kem::{EncodedSizeUser, KemCore, MlKem512, MlKem768, MlKem1024};

use vpn_core::crypto::MlKemLevel;
use vpn_core::crypto::auth::ServerAuthenticator;
use vpn_core::crypto::hybrid_kem::{
    build_handshake_transcript, client_decapsulate, generate_client_keypairs, run_local_authenticated_handshake,
    server_encapsulate,
};

fn hex8(bytes: &[u8]) -> String {
    bytes.iter().take(8).map(|b| format!("{b:02x}")).collect()
}

/// Run one full authenticated hybrid handshake for parameter set `K`:
/// client keygen -> server encapsulation -> server signs the transcript ->
/// client verifies -> client decapsulation -> both sides must agree.
/// Then report every wire size.
fn exercise<K>(level: MlKemLevel)
where
    K: KemCore,
    K::EncapsulationKey: EncodedSizeUser,
{
    let name = level.name();

    // Client generates its ephemeral X25519 + ML-KEM keypairs.
    let client_keys = generate_client_keypairs::<K>();

    // Server: long-term ML-DSA-65 identity, plus this-handshake encapsulation.
    let server_identity = ServerAuthenticator::generate();
    let server_result =
        server_encapsulate::<K>(client_keys.x25519_public(), client_keys.mlkem_public())
            .expect("server encapsulation failed");

    // Server signs the exact public transcript.
    let transcript = build_handshake_transcript::<K>(
        name.as_bytes(),
        &client_keys,
        &server_result.server_x25519_public,
        &server_result.ciphertext,
    );
    let signature = server_identity.sign(&transcript);

    // Client verifies the signature before trusting anything.
    let authentic =
        ServerAuthenticator::verify(server_identity.verifying_key(), &transcript, &signature);
    assert!(authentic, "{name}: server signature did not verify");

    // Client derives the shared secret and both sides must match.
    let client_secret = client_decapsulate::<K>(
        &client_keys,
        &server_result.server_x25519_public,
        &server_result.ciphertext,
    )
    .expect("client decapsulation failed");
    assert_eq!(
        client_secret, server_result.hybrid_secret,
        "{name}: client and server hybrid secrets differ"
    );

    let x25519_pub = client_keys.x25519_public().as_bytes().len();
    let mlkem_pub = client_keys.mlkem_public().as_bytes().len();
    let ciphertext = server_result.ciphertext.len();
    let sig_len = signature.encode().len();

    println!("  {name}");
    println!(
        "      hybrid secret            {:>5} B   SHA-256, first8={}",
        client_secret.len(),
        hex8(&client_secret)
    );
    println!("      client X25519 public     {x25519_pub:>5} B");
    println!("      server X25519 public     {x25519_pub:>5} B");
    println!("      client ML-KEM public     {mlkem_pub:>5} B");
    println!("      ML-KEM ciphertext        {ciphertext:>5} B");
    println!("      ML-DSA-65 signature      {sig_len:>5} B");
    println!("      signed transcript        {:>5} B", transcript.len());
    println!();
}

fn main() {
    println!("crypto-spike - exercises core::crypto (Member 1's Rust port)\n");
    println!("Authenticated hybrid handshake  (X25519 + ML-KEM, signed with ML-DSA-65):\n");

    exercise::<MlKem512>(MlKemLevel::MlKem512);
    exercise::<MlKem768>(MlKemLevel::MlKem768);
    exercise::<MlKem1024>(MlKemLevel::MlKem1024);

    // Also drive core's own all-in-one helper for each level.
    for level in [
        MlKemLevel::MlKem512,
        MlKemLevel::MlKem768,
        MlKemLevel::MlKem1024,
    ] {
        let secret = run_local_authenticated_handshake(level)
            .expect("run_local_authenticated_handshake failed");
        assert_eq!(secret.len(), 32);
    }
    println!("core::crypto::run_local_authenticated_handshake - all 3 levels OK\n");

    println!("All checks passed. Member 2 can build on core::crypto.");
}
