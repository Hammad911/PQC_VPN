//! The byte string the server signs — `PROTOCOL.md` §5.1.
//!
//! A thin wire-bytes adapter over
//! `vpn_core::crypto::hybrid_kem::build_handshake_transcript_from_bytes`, the
//! same function the client-side typed builder delegates to — so the server,
//! the test client, and Member 1's `core::protocol` client cannot drift
//! (`PROTOCOL.md` §8 Q1, resolved).

use vpn_core::crypto::hybrid_kem::build_handshake_transcript_from_bytes;

use crate::wire::AlgoCode;

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
    build_handshake_transcript_from_bytes(
        i.algo.transcript_name(),
        i.client_nonce,
        i.server_nonce,
        i.client_x25519_pub,
        i.client_mlkem_pub,
        i.server_x25519_pub,
        i.mlkem_ciphertext,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wire::PROTOCOL_LABEL;

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
