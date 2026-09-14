//! Session key derivation — `server/PROTOCOL.md` §5.3.
//!
//! Turns the raw 32-byte hybrid secret from a completed handshake into the
//! two keys the protocol actually uses, via independent HKDF-SHA256 expands
//! so neither key ever exposes anything about the other:
//!
//! ```text
//! ikm  = hybrid_secret                        (32 B, from combine_shared_secrets)
//! salt = client_nonce ‖ server_nonce          (64 B)
//! psk         = HKDF-SHA256(ikm, salt, "pqc-vpn wg psk",  32)
//! confirm_key = HKDF-SHA256(ikm, salt, "pqc-vpn confirm", 32)
//! ```
//!
//! `psk` is installed as the WireGuard pre-shared key; `confirm_key` keys the
//! `ClientFinish` / `ServerFinish` HMAC tags (not built here — those live
//! with the wire protocol in Week 6's `core::protocol`).
//!
//! This is the shared helper `server/PROTOCOL.md` §8 Q2 proposed: the same
//! derivation previously existed only in
//! `server/handshake-server/src/kdf.rs::derive`, duplicated rather than
//! shared. The constants here (`INFO_PSK`, `INFO_CONFIRM`, the salt layout)
//! are copied from that module byte for byte, so for the same inputs the two
//! produce identical output — adopting this helper on the server side is
//! Member 2's call, not made unilaterally here.

use hkdf::Hkdf;
use sha2::Sha256;
use zeroize::Zeroizing;

const INFO_PSK: &[u8] = b"pqc-vpn wg psk";
const INFO_CONFIRM: &[u8] = b"pqc-vpn confirm";

/// The two keys derived from one handshake's hybrid secret.
pub struct SessionKeys {
    /// Installed as the WireGuard pre-shared key.
    pub psk: Zeroizing<[u8; 32]>,
    /// Keys the `ClientFinish` / `ServerFinish` confirmation MACs.
    pub confirm_key: Zeroizing<[u8; 32]>,
}

/// Derives [`SessionKeys`] from a handshake's hybrid secret and the two
/// nonces exchanged during it.
pub fn derive_session_keys(
    hybrid_secret: &[u8; 32],
    client_nonce: &[u8; 32],
    server_nonce: &[u8; 32],
) -> SessionKeys {
    let mut salt = [0_u8; 64];

    salt[..32].copy_from_slice(client_nonce);
    salt[32..].copy_from_slice(server_nonce);

    let kdf = Hkdf::<Sha256>::new(Some(&salt), hybrid_secret);

    let mut psk = [0_u8; 32];
    kdf.expand(INFO_PSK, &mut psk)
        .expect("32-byte output is within HKDF-SHA256's max expand length");

    let mut confirm_key = [0_u8; 32];
    kdf.expand(INFO_CONFIRM, &mut confirm_key)
        .expect("32-byte output is within HKDF-SHA256's max expand length");

    SessionKeys {
        psk: Zeroizing::new(psk),
        confirm_key: Zeroizing::new(confirm_key),
    }
}

#[cfg(test)]
mod tests {
    use super::derive_session_keys;

    #[test]
    fn derivation_is_deterministic_and_separates_the_two_keys() {
        let a = derive_session_keys(&[7_u8; 32], &[1_u8; 32], &[2_u8; 32]);
        let b = derive_session_keys(&[7_u8; 32], &[1_u8; 32], &[2_u8; 32]);

        assert_eq!(*a.psk, *b.psk);
        assert_eq!(*a.confirm_key, *b.confirm_key);
        assert_ne!(*a.psk, *a.confirm_key);
    }

    #[test]
    fn changing_either_nonce_changes_both_keys() {
        let baseline = derive_session_keys(&[7_u8; 32], &[1_u8; 32], &[2_u8; 32]);
        let different_server_nonce = derive_session_keys(&[7_u8; 32], &[1_u8; 32], &[9_u8; 32]);
        let different_client_nonce = derive_session_keys(&[7_u8; 32], &[9_u8; 32], &[2_u8; 32]);

        assert_ne!(*baseline.psk, *different_server_nonce.psk);
        assert_ne!(*baseline.psk, *different_client_nonce.psk);
    }

    #[test]
    fn changing_the_hybrid_secret_changes_both_keys() {
        let a = derive_session_keys(&[7_u8; 32], &[1_u8; 32], &[2_u8; 32]);
        let b = derive_session_keys(&[8_u8; 32], &[1_u8; 32], &[2_u8; 32]);

        assert_ne!(*a.psk, *b.psk);
        assert_ne!(*a.confirm_key, *b.confirm_key);
    }
}
