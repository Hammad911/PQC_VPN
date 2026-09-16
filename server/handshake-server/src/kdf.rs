//! Key schedule — `PROTOCOL.md` §5.3.
//!
//! ```text
//! ikm  = hybrid_secret                       (32 B, from core::crypto)
//! salt = client_nonce ‖ server_nonce         (64 B)
//! psk         = HKDF-SHA256(ikm, salt, "pqc-vpn wg psk",  32)
//! confirm_key = HKDF-SHA256(ikm, salt, "pqc-vpn confirm", 32)
//!
//! client_tag = HMAC-SHA256(confirm_key, "client finished" ‖ transcript)
//! server_tag = HMAC-SHA256(confirm_key, "server finished" ‖ transcript)
//! ```
//!
//! The HKDF half is `vpn_core::crypto::derive_session_keys`, shared with the
//! client (`PROTOCOL.md` §8 Q2, resolved). The finish MACs stay here until
//! `core::protocol` (Week 6) needs them on the client side too.

use hmac::{Hmac, Mac};
use sha2::Sha256;

type HmacSha256 = Hmac<Sha256>;

const LABEL_CLIENT_FINISHED: &[u8] = b"client finished";
const LABEL_SERVER_FINISHED: &[u8] = b"server finished";

#[derive(Clone)]
pub struct SessionKeys {
    /// Installed as the WireGuard preshared key.
    pub psk: [u8; 32],
    /// Keys the `ClientFinish` / `ServerFinish` MACs.
    pub confirm_key: [u8; 32],
}

pub fn derive(
    hybrid_secret: &[u8; 32],
    client_nonce: &[u8; 32],
    server_nonce: &[u8; 32],
) -> SessionKeys {
    let keys = vpn_core::crypto::derive_session_keys(hybrid_secret, client_nonce, server_nonce);
    SessionKeys {
        psk: *keys.psk,
        confirm_key: *keys.confirm_key,
    }
}

fn tag(confirm_key: &[u8; 32], label: &[u8], transcript: &[u8]) -> [u8; 32] {
    let mut mac = HmacSha256::new_from_slice(confirm_key).expect("HMAC key any length");
    mac.update(label);
    mac.update(transcript);
    mac.finalize().into_bytes().into()
}

pub fn client_tag(confirm_key: &[u8; 32], transcript: &[u8]) -> [u8; 32] {
    tag(confirm_key, LABEL_CLIENT_FINISHED, transcript)
}

pub fn server_tag(confirm_key: &[u8; 32], transcript: &[u8]) -> [u8; 32] {
    tag(confirm_key, LABEL_SERVER_FINISHED, transcript)
}

/// Constant-time comparison via `hmac`'s `verify_slice`.
pub fn verify_client_tag(confirm_key: &[u8; 32], transcript: &[u8], claimed: &[u8; 32]) -> bool {
    let mut mac = HmacSha256::new_from_slice(confirm_key).expect("HMAC key any length");
    mac.update(LABEL_CLIENT_FINISHED);
    mac.update(transcript);
    mac.verify_slice(claimed).is_ok()
}

pub fn verify_server_tag(confirm_key: &[u8; 32], transcript: &[u8], claimed: &[u8; 32]) -> bool {
    let mut mac = HmacSha256::new_from_slice(confirm_key).expect("HMAC key any length");
    mac.update(LABEL_SERVER_FINISHED);
    mac.update(transcript);
    mac.verify_slice(claimed).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kdf_is_deterministic_and_separated() {
        let k1 = derive(&[7; 32], &[1; 32], &[2; 32]);
        let k2 = derive(&[7; 32], &[1; 32], &[2; 32]);
        assert_eq!(k1.psk, k2.psk);
        assert_eq!(k1.confirm_key, k2.confirm_key);
        assert_ne!(k1.psk, k1.confirm_key);
    }

    #[test]
    fn nonces_change_the_keys() {
        let a = derive(&[7; 32], &[1; 32], &[2; 32]);
        let b = derive(&[7; 32], &[1; 32], &[9; 32]);
        assert_ne!(a.psk, b.psk);
    }

    #[test]
    fn tags_verify_and_labels_differ() {
        let ck = [3u8; 32];
        let t = b"transcript bytes";
        let c = client_tag(&ck, t);
        let s = server_tag(&ck, t);
        assert_ne!(c, s);
        assert!(verify_client_tag(&ck, t, &c));
        assert!(verify_server_tag(&ck, t, &s));
        assert!(!verify_client_tag(&ck, t, &s));
    }
}
