use super::algorithm::MlKemLevel;
use super::auth::ServerAuthenticator;
use ::kem::{Decapsulate, Encapsulate};
use ::ml_kem::{EncodedSizeUser, KemCore, MlKem512, MlKem768, MlKem1024};
use sha2::{Digest, Sha256};
use x25519_dalek::{PublicKey, StaticSecret};

pub type MlKem768PublicKey = <MlKem768 as KemCore>::EncapsulationKey;
pub type MlKem512PublicKey = <MlKem512 as KemCore>::EncapsulationKey;

pub type MlKem1024PublicKey = <MlKem1024 as KemCore>::EncapsulationKey;

pub type MlKem768Ciphertext = ::ml_kem::Ciphertext<MlKem768>;
pub type MlKem512Ciphertext = ::ml_kem::Ciphertext<MlKem512>;

pub type MlKem1024Ciphertext = ::ml_kem::Ciphertext<MlKem1024>;

/// Contains the client's classical and post-quantum keypairs.
/// Contains a client's classical and post-quantum keypairs.
///
/// K represents the selected ML-KEM parameter set.
pub struct HybridClientKeys<K: KemCore> {
    x25519_private: StaticSecret,
    x25519_public: PublicKey,
    mlkem_private: K::DecapsulationKey,
    mlkem_public: K::EncapsulationKey,
}

/// Client keys using the ML-KEM-768 parameter set.
pub type HybridClientKeys768 = HybridClientKeys<MlKem768>;
pub type HybridClientKeys512 = HybridClientKeys<MlKem512>;

pub type HybridClientKeys1024 = HybridClientKeys<MlKem1024>;

impl<K: KemCore> HybridClientKeys<K> {
    /// Returns the client's X25519 public key.
    pub fn x25519_public(&self) -> &PublicKey {
        &self.x25519_public
    }

    /// Returns the client's ML-KEM public key.
    pub fn mlkem_public(&self) -> &K::EncapsulationKey {
        &self.mlkem_public
    }
}
/// Generates client X25519 and ML-KEM keys for parameter set K.
pub fn generate_client_keypairs<K: KemCore>() -> HybridClientKeys<K> {
    let mut rng = rand::thread_rng();

    let x25519_private = StaticSecret::random_from_rng(rand::rngs::OsRng);

    let x25519_public = PublicKey::from(&x25519_private);

    let (mlkem_private, mlkem_public) = K::generate(&mut rng);

    HybridClientKeys {
        x25519_private,
        x25519_public,
        mlkem_private,
        mlkem_public,
    }
}

/// Generates client keys using ML-KEM-512.
pub fn generate_client_keypairs_512() -> HybridClientKeys512 {
    generate_client_keypairs::<MlKem512>()
}

/// Generates client keys using ML-KEM-768.
pub fn generate_client_keypairs_768() -> HybridClientKeys768 {
    generate_client_keypairs::<MlKem768>()
}

/// Generates client keys using ML-KEM-1024.
pub fn generate_client_keypairs_1024() -> HybridClientKeys1024 {
    generate_client_keypairs::<MlKem1024>()
}

/// Information produced by the server during encapsulation.
///
/// K determines the ML-KEM ciphertext type.
pub struct ServerEncapsulation<K: KemCore> {
    pub server_x25519_public: PublicKey,
    pub ciphertext: ::ml_kem::Ciphertext<K>,
    pub hybrid_secret: [u8; 32],
}

/// Server result using ML-KEM-512.
pub type ServerEncapsulation512 = ServerEncapsulation<MlKem512>;

/// Server result using ML-KEM-768.
pub type ServerEncapsulation768 = ServerEncapsulation<MlKem768>;

/// Server result using ML-KEM-1024.
pub type ServerEncapsulation1024 = ServerEncapsulation<MlKem1024>;
/// Performs the client side of an X25519 + ML-KEM-768 exchange.
/// Performs the client side of a hybrid exchange using K.
pub fn client_decapsulate<K: KemCore>(
    client_keys: &HybridClientKeys<K>,
    server_x25519_public: &PublicKey,
    ciphertext: &::ml_kem::Ciphertext<K>,
) -> Result<[u8; 32], String> {
    let client_x25519_secret = client_keys
        .x25519_private
        .diffie_hellman(server_x25519_public);

    let client_mlkem_secret = client_keys
        .mlkem_private
        .decapsulate(ciphertext)
        .map_err(|_| String::from("ML-KEM decapsulation failed"))?;

    let hybrid_secret = combine_shared_secrets(
        client_x25519_secret.as_bytes(),
        client_mlkem_secret.as_ref(),
    );

    Ok(hybrid_secret)
}

/// Performs client decapsulation using ML-KEM-512.
pub fn client_decapsulate_512(
    client_keys: &HybridClientKeys512,
    server_x25519_public: &PublicKey,
    ciphertext: &MlKem512Ciphertext,
) -> Result<[u8; 32], String> {
    client_decapsulate::<MlKem512>(client_keys, server_x25519_public, ciphertext)
}

/// Performs client decapsulation using ML-KEM-768.
pub fn client_decapsulate_768(
    client_keys: &HybridClientKeys768,
    server_x25519_public: &PublicKey,
    ciphertext: &MlKem768Ciphertext,
) -> Result<[u8; 32], String> {
    client_decapsulate::<MlKem768>(client_keys, server_x25519_public, ciphertext)
}

/// Performs client decapsulation using ML-KEM-1024.
pub fn client_decapsulate_1024(
    client_keys: &HybridClientKeys1024,
    server_x25519_public: &PublicKey,
    ciphertext: &MlKem1024Ciphertext,
) -> Result<[u8; 32], String> {
    client_decapsulate::<MlKem1024>(client_keys, server_x25519_public, ciphertext)
}
/// Performs the server side of a hybrid exchange using K.
pub fn server_encapsulate<K: KemCore>(
    client_x25519_public: &PublicKey,
    client_mlkem_public: &K::EncapsulationKey,
) -> Result<ServerEncapsulation<K>, String> {
    let server_x25519_private = StaticSecret::random_from_rng(rand::rngs::OsRng);

    let server_x25519_public = PublicKey::from(&server_x25519_private);

    let server_x25519_secret = server_x25519_private.diffie_hellman(client_x25519_public);

    let mut rng = rand::thread_rng();

    let (ciphertext, server_mlkem_secret) = client_mlkem_public
        .encapsulate(&mut rng)
        .map_err(|_| String::from("ML-KEM encapsulation failed"))?;

    let hybrid_secret = combine_shared_secrets(
        server_x25519_secret.as_bytes(),
        server_mlkem_secret.as_ref(),
    );

    Ok(ServerEncapsulation {
        server_x25519_public,
        ciphertext,
        hybrid_secret,
    })
}

/// Performs server encapsulation using ML-KEM-512.
pub fn server_encapsulate_512(
    client_x25519_public: &PublicKey,
    client_mlkem_public: &MlKem512PublicKey,
) -> Result<ServerEncapsulation512, String> {
    server_encapsulate::<MlKem512>(client_x25519_public, client_mlkem_public)
}

/// Performs server encapsulation using ML-KEM-768.
pub fn server_encapsulate_768(
    client_x25519_public: &PublicKey,
    client_mlkem_public: &MlKem768PublicKey,
) -> Result<ServerEncapsulation768, String> {
    server_encapsulate::<MlKem768>(client_x25519_public, client_mlkem_public)
}

/// Performs server encapsulation using ML-KEM-1024.
pub fn server_encapsulate_1024(
    client_x25519_public: &PublicKey,
    client_mlkem_public: &MlKem1024PublicKey,
) -> Result<ServerEncapsulation1024, String> {
    server_encapsulate::<MlKem1024>(client_x25519_public, client_mlkem_public)
}
/// Combines the classical and post-quantum shared secrets.
///
/// This reproduces the Python formula:
/// SHA256(x25519_secret || mlkem_secret)
pub fn combine_shared_secrets(x25519_secret: &[u8], mlkem_secret: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();

    // Add the X25519 secret first.
    hasher.update(x25519_secret);

    // Add the ML-KEM secret immediately after it.
    hasher.update(mlkem_secret);

    // SHA-256 always returns 32 bytes.
    hasher.finalize().into()
}

/// Builds the handshake bytes signed by the server.
///
/// K determines the selected ML-KEM parameter set.
pub fn build_handshake_transcript<K: KemCore>(
    algorithm_name: &[u8],
    client_keys: &HybridClientKeys<K>,
    server_x25519_public: &PublicKey,
    ciphertext: &::ml_kem::Ciphertext<K>,
) -> Vec<u8> {
    const PROTOCOL_LABEL: &[u8] = b"PQC-VPN-HANDSHAKE-v1";

    let client_mlkem_public = client_keys.mlkem_public.as_bytes();

    let mut transcript = Vec::new();

    transcript.extend_from_slice(PROTOCOL_LABEL);
    transcript.extend_from_slice(algorithm_name);

    transcript.extend_from_slice(client_keys.x25519_public.as_bytes());

    transcript.extend_from_slice(client_mlkem_public.as_ref());

    transcript.extend_from_slice(server_x25519_public.as_bytes());

    transcript.extend_from_slice(ciphertext.as_ref());

    transcript
}

/// Builds a signed transcript for ML-KEM-512.
pub fn build_handshake_transcript_512(
    client_keys: &HybridClientKeys512,
    server_x25519_public: &PublicKey,
    ciphertext: &MlKem512Ciphertext,
) -> Vec<u8> {
    build_handshake_transcript::<MlKem512>(
        b"ML-KEM-512",
        client_keys,
        server_x25519_public,
        ciphertext,
    )
}

/// Builds a signed transcript for ML-KEM-768.
pub fn build_handshake_transcript_768(
    client_keys: &HybridClientKeys768,
    server_x25519_public: &PublicKey,
    ciphertext: &MlKem768Ciphertext,
) -> Vec<u8> {
    build_handshake_transcript::<MlKem768>(
        b"ML-KEM-768",
        client_keys,
        server_x25519_public,
        ciphertext,
    )
}

/// Builds a signed transcript for ML-KEM-1024.
pub fn build_handshake_transcript_1024(
    client_keys: &HybridClientKeys1024,
    server_x25519_public: &PublicKey,
    ciphertext: &MlKem1024Ciphertext,
) -> Vec<u8> {
    build_handshake_transcript::<MlKem1024>(
        b"ML-KEM-1024",
        client_keys,
        server_x25519_public,
        ciphertext,
    )
}

/// Runs one local authenticated hybrid handshake for K.
///
/// This helper simulates both client and server so the complete crypto
/// flow can be exercised before real networking is connected.
fn run_local_handshake_for<K: KemCore>(algorithm_name: &[u8]) -> Result<[u8; 32], String> {
    let authenticator = ServerAuthenticator::generate();

    let client_keys = generate_client_keypairs::<K>();

    let server_result =
        server_encapsulate::<K>(client_keys.x25519_public(), client_keys.mlkem_public())?;

    let transcript = build_handshake_transcript::<K>(
        algorithm_name,
        &client_keys,
        &server_result.server_x25519_public,
        &server_result.ciphertext,
    );

    let signature = authenticator.sign(&transcript);

    let authenticated =
        ServerAuthenticator::verify(authenticator.verifying_key(), &transcript, &signature);

    if !authenticated {
        return Err(String::from("Server handshake authentication failed"));
    }

    let client_secret = client_decapsulate::<K>(
        &client_keys,
        &server_result.server_x25519_public,
        &server_result.ciphertext,
    )?;

    if client_secret != server_result.hybrid_secret {
        return Err(String::from(
            "Client and server hybrid secrets do not match",
        ));
    }

    Ok(client_secret)
}

/// Selects and runs a local authenticated hybrid handshake.
pub fn run_local_authenticated_handshake(level: MlKemLevel) -> Result<[u8; 32], String> {
    match level {
        MlKemLevel::MlKem512 => run_local_handshake_for::<MlKem512>(b"ML-KEM-512"),

        MlKemLevel::MlKem768 => run_local_handshake_for::<MlKem768>(b"ML-KEM-768"),

        MlKemLevel::MlKem1024 => run_local_handshake_for::<MlKem1024>(b"ML-KEM-1024"),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        build_handshake_transcript_512, build_handshake_transcript_768,
        build_handshake_transcript_1024, client_decapsulate_512, client_decapsulate_768,
        client_decapsulate_1024, generate_client_keypairs_512, generate_client_keypairs_768,
        generate_client_keypairs_1024, run_local_authenticated_handshake, server_encapsulate_512,
        server_encapsulate_768, server_encapsulate_1024,
    };

    use crate::crypto::MlKemLevel;
    use crate::crypto::auth::ServerAuthenticator;

    #[test]
    fn hybrid_x25519_ml_kem_768_secrets_match() {
        // Client generates its two keypairs.
        let client_keys = generate_client_keypairs_768();

        // Server uses only the client's public keys.
        let server_result =
            server_encapsulate_768(client_keys.x25519_public(), client_keys.mlkem_public())
                .expect("Server encapsulation failed");

        // Client uses its private keys and the server's response.
        let client_hybrid_secret = client_decapsulate_768(
            &client_keys,
            &server_result.server_x25519_public,
            &server_result.ciphertext,
        )
        .expect("Client decapsulation failed");

        assert_eq!(client_hybrid_secret, server_result.hybrid_secret);

        assert_eq!(client_hybrid_secret.len(), 32);
    }

    #[test]
    fn modified_ciphertext_does_not_produce_server_secret() {
        let client_keys = generate_client_keypairs_768();

        let server_result =
            server_encapsulate_768(client_keys.x25519_public(), client_keys.mlkem_public())
                .expect("Server encapsulation failed");

        // Simulate an attacker changing one bit of the ciphertext.
        let mut modified_ciphertext = server_result.ciphertext;

        modified_ciphertext[0] ^= 0x01;

        let client_result = client_decapsulate_768(
            &client_keys,
            &server_result.server_x25519_public,
            &modified_ciphertext,
        );

        match client_result {
            Ok(client_secret) => {
                assert_ne!(client_secret, server_result.hybrid_secret);
            }

            Err(_) => {
                // Rejecting the modified ciphertext is also safe behavior.
            }
        }
    }

    #[test]
    fn authenticated_hybrid_handshake_succeeds() {
        // Long-term server identity.
        let authenticator = ServerAuthenticator::generate();

        // Client creates its temporary handshake keys.
        let client_keys = generate_client_keypairs_768();

        // Server performs the hybrid encapsulation.
        let server_result =
            server_encapsulate_768(client_keys.x25519_public(), client_keys.mlkem_public())
                .expect("Server encapsulation failed");

        // Server constructs the exact public handshake transcript.
        let transcript = build_handshake_transcript_768(
            &client_keys,
            &server_result.server_x25519_public,
            &server_result.ciphertext,
        );

        // Server signs the real transcript.
        let signature = authenticator.sign(&transcript);

        // Client verifies the signature using the trusted server key.
        let authenticated =
            ServerAuthenticator::verify(authenticator.verifying_key(), &transcript, &signature);

        assert!(authenticated);

        // Only after authentication does the client derive the secret.
        let client_secret = client_decapsulate_768(
            &client_keys,
            &server_result.server_x25519_public,
            &server_result.ciphertext,
        )
        .expect("Client decapsulation failed");

        assert_eq!(client_secret, server_result.hybrid_secret);
    }

    #[test]
    fn modified_handshake_transcript_is_rejected() {
        let authenticator = ServerAuthenticator::generate();

        let client_keys = generate_client_keypairs_768();

        let server_result =
            server_encapsulate_768(client_keys.x25519_public(), client_keys.mlkem_public())
                .expect("Server encapsulation failed");

        let transcript = build_handshake_transcript_768(
            &client_keys,
            &server_result.server_x25519_public,
            &server_result.ciphertext,
        );

        // Server signs the genuine transcript.
        let signature = authenticator.sign(&transcript);

        // Simulate an attacker modifying one transcript byte.
        let mut modified_transcript = transcript.clone();

        let last_index = modified_transcript.len() - 1;

        modified_transcript[last_index] ^= 0x01;

        // The original signature must not verify modified data.
        let accepted = ServerAuthenticator::verify(
            authenticator.verifying_key(),
            &modified_transcript,
            &signature,
        );

        assert!(!accepted);
    }

    #[test]
    fn hybrid_x25519_ml_kem_512_secrets_match() {
        let client_keys = generate_client_keypairs_512();

        let server_result =
            server_encapsulate_512(client_keys.x25519_public(), client_keys.mlkem_public())
                .expect("ML-KEM-512 server encapsulation failed");

        let client_secret = client_decapsulate_512(
            &client_keys,
            &server_result.server_x25519_public,
            &server_result.ciphertext,
        )
        .expect("ML-KEM-512 client decapsulation failed");

        assert_eq!(client_secret, server_result.hybrid_secret);

        assert_eq!(client_secret.len(), 32);
    }

    #[test]
    fn hybrid_x25519_ml_kem_1024_secrets_match() {
        let client_keys = generate_client_keypairs_1024();

        let server_result =
            server_encapsulate_1024(client_keys.x25519_public(), client_keys.mlkem_public())
                .expect("ML-KEM-1024 server encapsulation failed");

        let client_secret = client_decapsulate_1024(
            &client_keys,
            &server_result.server_x25519_public,
            &server_result.ciphertext,
        )
        .expect("ML-KEM-1024 client decapsulation failed");

        assert_eq!(client_secret, server_result.hybrid_secret);

        assert_eq!(client_secret.len(), 32);
    }

    #[test]
    fn authenticated_ml_kem_512_handshake_succeeds() {
        let authenticator = ServerAuthenticator::generate();

        let client_keys = generate_client_keypairs_512();

        let server_result =
            server_encapsulate_512(client_keys.x25519_public(), client_keys.mlkem_public())
                .expect("ML-KEM-512 server encapsulation failed");

        let transcript = build_handshake_transcript_512(
            &client_keys,
            &server_result.server_x25519_public,
            &server_result.ciphertext,
        );

        let signature = authenticator.sign(&transcript);

        let authenticated =
            ServerAuthenticator::verify(authenticator.verifying_key(), &transcript, &signature);

        assert!(authenticated);

        let client_secret = client_decapsulate_512(
            &client_keys,
            &server_result.server_x25519_public,
            &server_result.ciphertext,
        )
        .expect("ML-KEM-512 client decapsulation failed");

        assert_eq!(client_secret, server_result.hybrid_secret);
    }

    #[test]
    fn authenticated_ml_kem_1024_handshake_succeeds() {
        let authenticator = ServerAuthenticator::generate();

        let client_keys = generate_client_keypairs_1024();

        let server_result =
            server_encapsulate_1024(client_keys.x25519_public(), client_keys.mlkem_public())
                .expect("ML-KEM-1024 server encapsulation failed");

        let transcript = build_handshake_transcript_1024(
            &client_keys,
            &server_result.server_x25519_public,
            &server_result.ciphertext,
        );

        let signature = authenticator.sign(&transcript);

        let authenticated =
            ServerAuthenticator::verify(authenticator.verifying_key(), &transcript, &signature);

        assert!(authenticated);

        let client_secret = client_decapsulate_1024(
            &client_keys,
            &server_result.server_x25519_public,
            &server_result.ciphertext,
        )
        .expect("ML-KEM-1024 client decapsulation failed");

        assert_eq!(client_secret, server_result.hybrid_secret);
    }

    #[test]
    fn runtime_selection_runs_all_ml_kem_levels() {
        let levels = [
            MlKemLevel::MlKem512,
            MlKemLevel::MlKem768,
            MlKemLevel::MlKem1024,
        ];

        for level in levels {
            let secret =
                run_local_authenticated_handshake(level).expect("Selected hybrid handshake failed");

            assert_eq!(secret.len(), 32);
        }
    }
}
