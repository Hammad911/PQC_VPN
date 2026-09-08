//! The server's long-term ML-DSA-65 identity key — `PROTOCOL.md` §5.2.
//!
//! Persisted as its **32-byte seed** (FIPS 204 ξ) so it survives restarts. The
//! public half is pinned in every client, so regenerating it would lock all
//! clients out.
//!
//! `core::crypto::auth::ServerAuthenticator` wraps the same key type but exposes
//! no load/save. Using `ml-dsa` directly here; `PROTOCOL.md` §8 asks Member 1 to
//! add persistence to `ServerAuthenticator` after the freeze, at which point
//! this module becomes a thin wrapper.

use std::fs;
use std::io;
use std::path::Path;

use ml_dsa::{
    B32, Generate, Keypair, MlDsa65, Signature, Signer, SigningKey, Verifier, VerifyingKey,
};

pub const SEED_LEN: usize = 32;
pub const VERIFYING_KEY_LEN: usize = 1952; // ML-DSA-65, FIPS 204
pub const SIGNATURE_LEN: usize = 3309;

/// Owns the server's long-term signing key.
pub struct ServerIdentity {
    signing_key: SigningKey<MlDsa65>,
}

impl ServerIdentity {
    /// Fresh random identity (not persisted).
    pub fn generate() -> Self {
        ServerIdentity {
            signing_key: SigningKey::<MlDsa65>::generate(),
        }
    }

    fn from_seed_bytes(seed: &[u8; SEED_LEN]) -> Self {
        let seed = B32::from(*seed);
        ServerIdentity {
            signing_key: SigningKey::<MlDsa65>::from_seed(&seed),
        }
    }

    fn seed_bytes(&self) -> [u8; SEED_LEN] {
        let seed = self.signing_key.to_seed();
        let mut out = [0u8; SEED_LEN];
        out.copy_from_slice(&seed);
        out
    }

    /// Load the identity seed from `path`; if absent, generate a new one and
    /// persist it (mode 0600).
    pub fn load_or_create(path: &Path) -> io::Result<Self> {
        match fs::read(path) {
            Ok(bytes) => {
                let seed: [u8; SEED_LEN] = bytes.as_slice().try_into().map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidData, "identity seed must be 32 bytes")
                })?;
                Ok(Self::from_seed_bytes(&seed))
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                let id = Self::generate();
                id.save(path)?;
                Ok(id)
            }
            Err(e) => Err(e),
        }
    }

    pub fn save(&self, path: &Path) -> io::Result<()> {
        if let Some(dir) = path.parent() {
            if !dir.as_os_str().is_empty() {
                fs::create_dir_all(dir)?;
            }
        }
        fs::write(path, self.seed_bytes())?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
        }
        Ok(())
    }

    pub fn verifying_key(&self) -> VerifyingKey<MlDsa65> {
        self.signing_key.verifying_key()
    }

    /// The bytes a client pins (1952 B for ML-DSA-65).
    pub fn verifying_key_bytes(&self) -> Vec<u8> {
        self.verifying_key().encode().to_vec()
    }

    /// Sign `message`, returning the 3309-byte ML-DSA-65 signature.
    pub fn sign(&self, message: &[u8]) -> Vec<u8> {
        let sig: Signature<MlDsa65> = self.signing_key.sign(message);
        sig.encode().to_vec()
    }
}

/// Verify a server signature against pinned verifying-key bytes.
/// Used by the client (and the test client).
pub fn verify(verifying_key_bytes: &[u8], message: &[u8], signature_bytes: &[u8]) -> bool {
    let Ok(vk_arr) = <[u8; VERIFYING_KEY_LEN]>::try_from(verifying_key_bytes) else {
        return false;
    };
    let vk = VerifyingKey::<MlDsa65>::decode(&vk_arr.into());
    let Ok(sig_arr) = <[u8; SIGNATURE_LEN]>::try_from(signature_bytes) else {
        return false;
    };
    let Some(sig) = Signature::<MlDsa65>::decode(&sig_arr.into()) else {
        return false;
    };
    vk.verify(message, &sig).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sign_verify_round_trip() {
        let id = ServerIdentity::generate();
        let msg = b"handshake transcript";
        let sig = id.sign(msg);
        assert_eq!(sig.len(), SIGNATURE_LEN);
        assert_eq!(id.verifying_key_bytes().len(), VERIFYING_KEY_LEN);
        assert!(verify(&id.verifying_key_bytes(), msg, &sig));
        assert!(!verify(&id.verifying_key_bytes(), b"tampered", &sig));
    }

    #[test]
    fn persists_and_reloads_the_same_key() {
        let path = std::env::temp_dir().join(format!(
            "pqcvpn-test-identity-{}.seed",
            std::process::id()
        ));
        let _ = fs::remove_file(&path);

        let a = ServerIdentity::load_or_create(&path).unwrap();
        let b = ServerIdentity::load_or_create(&path).unwrap();
        assert_eq!(a.verifying_key_bytes(), b.verifying_key_bytes());

        let msg = b"same key still signs";
        let sig = a.sign(msg);
        assert!(verify(&b.verifying_key_bytes(), msg, &sig));

        let _ = fs::remove_file(&path);
    }
}
