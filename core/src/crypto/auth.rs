use ::ml_dsa::{Generate, MlDsa65, Signature, Signer, SigningKey, Verifier, VerifyingKey};

/// ML-DSA-65 signature produced by the server.
pub type ServerSignature = Signature<MlDsa65>;

/// Public key used by clients to verify the server.
pub type ServerVerifyingKey = VerifyingKey<MlDsa65>;

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
