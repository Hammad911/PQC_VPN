use ::ml_dsa::{B32, Generate, MlDsa65, Signature, Signer, SigningKey, Verifier, VerifyingKey};
use zeroize::Zeroizing;

/// ML-DSA-65 signature produced by the server.
pub type ServerSignature = Signature<MlDsa65>;

/// Public key used by clients to verify the server.
pub type ServerVerifyingKey = VerifyingKey<MlDsa65>;

/// Length in bytes of a persisted identity seed (FIPS 204 ξ).
pub const SEED_LEN: usize = 32;

/// Owns the server's long-term ML-DSA-65 private identity key.
pub struct ServerAuthenticator {
    signing_key: SigningKey<MlDsa65>,
}

impl ServerAuthenticator {
    /// Generates a new server identity.
    pub fn generate() -> Self {
        Self {
            signing_key: SigningKey::<MlDsa65>::generate(),
        }
    }

    /// Rebuilds a server identity from a previously saved seed
    /// (`to_seed_bytes`). The public key, and therefore what clients have
    /// pinned, is identical every time the same seed is loaded.
    ///
    /// Added for the Week 4 checkpoint (`server/PROTOCOL.md` §8 Q7): before
    /// this, `server/handshake-server`'s `identity` module could not persist
    /// through `ServerAuthenticator` and used `ml-dsa` directly instead.
    pub fn from_seed_bytes(seed: &[u8; SEED_LEN]) -> Self {
        Self {
            signing_key: SigningKey::<MlDsa65>::from_seed(&B32::from(*seed)),
        }
    }

    /// Returns the 32-byte seed this identity was generated or loaded from,
    /// wrapped so it is zeroized when dropped. Save it and pass it back to
    /// [`from_seed_bytes`](Self::from_seed_bytes) to reload the same identity.
    pub fn to_seed_bytes(&self) -> Zeroizing<[u8; SEED_LEN]> {
        let seed = self.signing_key.to_seed();

        let mut out = [0_u8; SEED_LEN];
        out.copy_from_slice(&seed);

        Zeroizing::new(out)
    }

    /// Returns the public part of the server identity.
    pub fn verifying_key(&self) -> &ServerVerifyingKey {
        self.signing_key.as_ref()
    }

    /// Signs the provided handshake bytes.
    pub fn sign(&self, message: &[u8]) -> ServerSignature {
        self.signing_key.sign(message)
    }

    /// Verifies a signature using a trusted server public key.
    pub fn verify(
        verifying_key: &ServerVerifyingKey,
        message: &[u8],
        signature: &ServerSignature,
    ) -> bool {
        verifying_key.verify(message, signature).is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::ServerAuthenticator;

    #[test]
    fn ml_dsa_65_sign_and_verify() {
        let authenticator = ServerAuthenticator::generate();

        let message = b"test server public key";

        let signature = authenticator.sign(message);

        let valid = ServerAuthenticator::verify(authenticator.verifying_key(), message, &signature);

        assert!(valid);
    }

    #[test]
    fn modified_message_is_rejected() {
        let authenticator = ServerAuthenticator::generate();

        let original_message = b"real handshake public-key data";

        let modified_message = b"attacker handshake public-key data";

        let signature = authenticator.sign(original_message);

        let valid = ServerAuthenticator::verify(
            authenticator.verifying_key(),
            modified_message,
            &signature,
        );

        assert!(!valid);
    }

    #[test]
    fn seed_round_trip_reproduces_the_same_identity() {
        let original = ServerAuthenticator::generate();

        let seed = original.to_seed_bytes();

        let reloaded = ServerAuthenticator::from_seed_bytes(&seed);

        // Same seed must yield the same public key - otherwise reloading a
        // persisted identity would silently lock out every pinned client.
        assert_eq!(
            original.verifying_key().encode(),
            reloaded.verifying_key().encode()
        );

        let message = b"reloaded identity signs the same way";

        let signature = reloaded.sign(message);

        assert!(ServerAuthenticator::verify(
            original.verifying_key(),
            message,
            &signature
        ));
    }

    #[test]
    fn attacker_identity_key_is_rejected() {
        // Genuine VPN server identity.
        let real_server = ServerAuthenticator::generate();

        // A separate identity controlled by an attacker.
        let attacker = ServerAuthenticator::generate();

        let handshake_data = b"genuine server handshake data";

        // The real server signs the handshake.
        let signature = real_server.sign(handshake_data);

        // Attempt verification using the attacker's public key.
        let valid =
            ServerAuthenticator::verify(attacker.verifying_key(), handshake_data, &signature);

        assert!(!valid);
    }
}
