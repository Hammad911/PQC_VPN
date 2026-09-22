//! `ClientFinish` / `ServerFinish` confirmation MACs — `server/PROTOCOL.md`
//! §5.3.
//!
//! ```text
//! client_tag = HMAC-SHA256(confirm_key, "client finished" ‖ transcript)
//! server_tag = HMAC-SHA256(confirm_key, "server finished" ‖ transcript)
//! ```
//!
//! `confirm_key` itself comes from [`crate::crypto::derive_session_keys`],
//! already shared with the server (`server/PROTOCOL.md` §8 Q2). Until this
//! module, the two HMAC labels above only existed in
//! `server/handshake-server/src/kdf.rs` — this is the "finish MACs" move to
//! `core::protocol` that document explicitly calls for at Week 6, so the
//! client can build and check its own confirmation tags instead of trusting
//! the server unauthenticated.

use hmac::{Hmac, Mac};
use sha2::Sha256;

type HmacSha256 = Hmac<Sha256>;

const LABEL_CLIENT_FINISHED: &[u8] = b"client finished";
const LABEL_SERVER_FINISHED: &[u8] = b"server finished";

fn tag(confirm_key: &[u8; 32], label: &[u8], transcript: &[u8]) -> [u8; 32] {
    let mut mac = HmacSha256::new_from_slice(confirm_key).expect("HMAC accepts any key length");
    mac.update(label);
    mac.update(transcript);
    mac.finalize().into_bytes().into()
}

/// Computes the tag the client sends in `ClientFinish`.
pub fn client_tag(confirm_key: &[u8; 32], transcript: &[u8]) -> [u8; 32] {
    tag(confirm_key, LABEL_CLIENT_FINISHED, transcript)
}

/// Constant-time check of a `ServerFinish.server_tag` the peer sent us.
pub fn verify_server_tag(confirm_key: &[u8; 32], transcript: &[u8], claimed: &[u8; 32]) -> bool {
    let mut mac = HmacSha256::new_from_slice(confirm_key).expect("HMAC accepts any key length");
    mac.update(LABEL_SERVER_FINISHED);
    mac.update(transcript);
    mac.verify_slice(claimed).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode_hex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    #[test]
    fn client_and_server_tags_differ_for_the_same_inputs() {
        let confirm_key = [3u8; 32];
        let transcript = b"transcript bytes";

        let c = client_tag(&confirm_key, transcript);
        assert_ne!(c, [0u8; 32]);
        assert!(!verify_server_tag(&confirm_key, transcript, &c));
    }

    /// Parity with `server/handshake-vectors.json`'s `mac` section — the
    /// server side (`server/handshake-server/src/kdf.rs`) produces these
    /// exact bytes for the same `confirm_key`/`transcript`/label inputs.
    /// `confirm_key` and both tags are copied verbatim from that file's
    /// `mac.confirm_key_hex` / `mac.client_tag_hex` / `mac.server_tag_hex`;
    /// the transcript is rebuilt field-by-field the same way
    /// `hybrid_kem.rs`'s own vector test does, matching `mac.transcript_hex`
    /// (independently confirmed byte-for-byte before writing this test).
    #[test]
    fn tags_match_the_frozen_vector() {
        let confirm_key: [u8; 32] =
            decode_hex("a0c383a95d5cc667a44a6d3cb2b16cf9c079597e8a6ea19e6b04408ca934615f")
                .try_into()
                .unwrap();

        let mut transcript = decode_hex("5051432d56504e2d48414e445348414b452d76314d4c2d4b454d2d373638");
        transcript.extend(std::iter::repeat(0x11u8).take(32)); // client_nonce
        transcript.extend(std::iter::repeat(0x22u8).take(32)); // server_nonce
        transcript.extend(std::iter::repeat(0x33u8).take(32)); // client_x25519_pub
        transcript.extend(std::iter::repeat(0x44u8).take(1184)); // client_mlkem_pub
        transcript.extend(std::iter::repeat(0x55u8).take(32)); // server_x25519_pub
        transcript.extend(std::iter::repeat(0x66u8).take(1088)); // mlkem_ciphertext
        assert_eq!(transcript.len(), 2430);

        let expected_client_tag: [u8; 32] =
            decode_hex("17eb8450e6c8c76591b8706bf1363ac5d5d924002c8855d14d9ab7acbcc0b9ff")
                .try_into()
                .unwrap();
        let expected_server_tag: [u8; 32] =
            decode_hex("89bb0c7af045ac12f8b241c0655fa1eef9bb7b63f7ca67e60bf074b60f81794a")
                .try_into()
                .unwrap();

        assert_eq!(client_tag(&confirm_key, &transcript), expected_client_tag);
        assert!(verify_server_tag(&confirm_key, &transcript, &expected_server_tag));
    }
}
