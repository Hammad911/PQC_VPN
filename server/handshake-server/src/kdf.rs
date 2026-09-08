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
//! Lives in this crate until the freeze meeting decides whether it belongs in
//! `core` (`PROTOCOL.md` §8, Q2).

use hkdf::Hkdf;
use hmac::{Hmac, Mac};
use sha2::Sha256;

type HmacSha256 = Hmac<Sha256>;

const INFO_PSK: &[u8] = b"pqc-vpn wg psk";
const INFO_CONFIRM: &[u8] = b"pqc-vpn confirm";
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
    let mut salt = [0u8; 64];
    salt[..32].copy_from_slice(client_nonce);
    salt[32..].copy_from_slice(server_nonce);

    let hk = Hkdf::<Sha256>::new(Some(&salt), hybrid_secret);
    let mut psk = [0u8; 32];
    let mut confirm_key = [0u8; 32];
    hk.expand(INFO_PSK, &mut psk).expect("32 <= 255*32");
    hk.expand(INFO_CONFIRM, &mut confirm_key).expect("32 <= 255*32");
    SessionKeys { psk, confirm_key }
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
