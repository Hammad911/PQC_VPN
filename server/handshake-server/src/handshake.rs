//! Server side of the handshake: `ClientHello` → `ServerHello`.
//!
//! All cryptography is `vpn_core::crypto` (Member 1's port). This module only
//! reconstructs the client's public values from wire bytes, calls
//! `core::server_encapsulate`, signs the transcript, and derives the session
//! keys.

use ml_kem::{Encoded, EncodedSizeUser, KemCore, MlKem512, MlKem768, MlKem1024};
use x25519_dalek::PublicKey;

use crate::error::HandshakeError;
use crate::identity::ServerIdentity;
use crate::kdf::{self, SessionKeys};
use crate::transcript::{self, TranscriptInputs};
use crate::wire::{AlgoCode, ClientHello, ServerHello};

pub struct HandshakeOutcome {
    pub server_hello: ServerHello,
    pub keys: SessionKeys,
    /// The signed transcript — kept so the caller can compute the finish MACs.
    pub transcript: Vec<u8>,
}

/// Produce a `ServerHello` in response to `hello`.
///
/// `session_id` and `server_nonce` are supplied by the caller so this function
/// stays deterministic for test vectors; production passes fresh random values.
pub fn respond(
    identity: &ServerIdentity,
    hello: &ClientHello,
    session_id: [u8; 16],
    server_nonce: [u8; 32],
) -> Result<HandshakeOutcome, HandshakeError> {
    let client_x = PublicKey::from(hello.client_x25519_pub);

    let (server_x25519_pub, ciphertext, hybrid_secret) = match hello.algo {
        AlgoCode::MlKem512 => encapsulate_for::<MlKem512>(&client_x, &hello.client_mlkem_pub)?,
        AlgoCode::MlKem768 => encapsulate_for::<MlKem768>(&client_x, &hello.client_mlkem_pub)?,
        AlgoCode::MlKem1024 => encapsulate_for::<MlKem1024>(&client_x, &hello.client_mlkem_pub)?,
    };

    let transcript = transcript::build(&TranscriptInputs {
        algo: hello.algo,
        client_nonce: &hello.client_nonce,
        server_nonce: &server_nonce,
        client_x25519_pub: &hello.client_x25519_pub,
        client_mlkem_pub: &hello.client_mlkem_pub,
        server_x25519_pub: &server_x25519_pub,
        mlkem_ciphertext: &ciphertext,
    });

    let signature = identity.sign(&transcript);
    let keys = kdf::derive(&hybrid_secret, &hello.client_nonce, &server_nonce);

    Ok(HandshakeOutcome {
        server_hello: ServerHello {
            algo: hello.algo,
            session_id,
            server_nonce,
            server_x25519_pub,
            mlkem_ciphertext: ciphertext,
            signature,
        },
        keys,
        transcript,
    })
}

fn encapsulate_for<K>(
    client_x: &PublicKey,
    client_mlkem_pub: &[u8],
) -> Result<([u8; 32], Vec<u8>, [u8; 32]), HandshakeError>
where
    K: KemCore,
    K::EncapsulationKey: EncodedSizeUser,
{
    let enc = Encoded::<K::EncapsulationKey>::try_from(client_mlkem_pub)
        .map_err(|_| HandshakeError::Crypto("client ML-KEM public key wrong length".into()))?;
    let ek = K::EncapsulationKey::from_bytes(&enc);

    let r = vpn_core::crypto::hybrid_kem::server_encapsulate::<K>(client_x, &ek)
        .map_err(HandshakeError::Crypto)?;

    Ok((
        r.server_x25519_public.to_bytes(),
        r.ciphertext.to_vec(),
        r.hybrid_secret,
    ))
}
