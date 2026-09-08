//! The byte string the server signs — `PROTOCOL.md` §5.1.
//!
//! This is `core::crypto::hybrid_kem::build_handshake_transcript` extended with
//! the two handshake nonces (inserted right after the algorithm name). Both the
//! server and the test client build it from this one function so they cannot
//! drift; unify into `core` after the freeze meeting (`PROTOCOL.md` §8, Q1).

use crate::wire::{AlgoCode, PROTOCOL_LABEL};

/// Everything that goes into the transcript, as raw wire bytes.
pub struct TranscriptInputs<'a> {
    pub algo: AlgoCode,
    pub client_nonce: &'a [u8; 32],
    pub server_nonce: &'a [u8; 32],
    pub client_x25519_pub: &'a [u8; 32],
    pub client_mlkem_pub: &'a [u8],
    pub server_x25519_pub: &'a [u8; 32],
    pub mlkem_ciphertext: &'a [u8],
}

/// `label ‖ algo_name ‖ client_nonce ‖ server_nonce ‖ client_x25519_pub ‖
///  client_mlkem_pub ‖ server_x25519_pub ‖ mlkem_ciphertext`
pub fn build(i: &TranscriptInputs) -> Vec<u8> {
    let name = i.algo.transcript_name();
    let mut t = Vec::with_capacity(
        PROTOCOL_LABEL.len()
            + name.len()
            + 32 * 4
            + i.client_mlkem_pub.len()
            + i.mlkem_ciphertext.len(),
    );
    t.extend_from_slice(PROTOCOL_LABEL);
    t.extend_from_slice(name);
    t.extend_from_slice(i.client_nonce);
    t.extend_from_slice(i.server_nonce);
    t.extend_from_slice(i.client_x25519_pub);
    t.extend_from_slice(i.client_mlkem_pub);
    t.extend_from_slice(i.server_x25519_pub);
    t.extend_from_slice(i.mlkem_ciphertext);
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transcript_layout_is_stable() {
        let t = build(&TranscriptInputs {
            algo: AlgoCode::MlKem768,
            client_nonce: &[1; 32],
            server_nonce: &[2; 32],
            client_x25519_pub: &[3; 32],
            client_mlkem_pub: &[4; 1184],
            server_x25519_pub: &[5; 32],
            mlkem_ciphertext: &[6; 1088],
        });
        assert_eq!(t.len(), 20 + 10 + 32 + 32 + 32 + 1184 + 32 + 1088);
        assert_eq!(&t[..20], PROTOCOL_LABEL);
        assert_eq!(&t[20..30], b"ML-KEM-768");
    }
}
