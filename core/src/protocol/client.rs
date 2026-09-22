//! Client side of the handshake state machine — `server/PROTOCOL.md`
//! §4.2–§4.5.
//!
//! This is the piece `server/PROTOCOL.md` names directly: *"For Member 1:
//! build the client side of `core/protocol/` against this document and
//! verify it against `server/handshake-vectors.json`"*. Everything here
//! runs against the same `crate::crypto` primitives the server crate calls
//! through `vpn_core` — there is exactly one implementation of the KEM, the
//! signature check and the transcript, shared by both ends, so client and
//! server cannot silently drift apart the way two independent ports could.

use ::ml_kem::{Ciphertext, EncodedSizeUser, MlKem512, MlKem768, MlKem1024};
use x25519_dalek::PublicKey;

use crate::crypto::hybrid_kem as hk;
use crate::crypto::{MlKemLevel, SessionKeys, ServerSignature, ServerVerifyingKey, derive_session_keys};

use super::tags;
use super::wire::{ClientFinish, ClientHello, ServerHello};

/// The server's ML-DSA-65 public key is always this many bytes (FIPS 204,
/// `server/PROTOCOL.md` §5.2). The client ships it pinned — never fetched
/// at runtime, never TOFU.
pub const VERIFYING_KEY_LEN: usize = 1952;

/// Failure verifying a `ServerHello` or deriving the session keys from it.
#[derive(Debug)]
pub enum HandshakeError {
    /// `ServerHello.algo` did not match what the client requested. The
    /// server must never silently downgrade (`PROTOCOL.md` §4.3) — a
    /// mismatch here means something on the wire is wrong or hostile, not
    /// that the client should quietly accept the server's choice.
    AlgoMismatch,
    /// The pinned server signature did not verify over the transcript —
    /// the whole MitM defence (`PROTOCOL.md` §5.2). Never derive a secret
    /// or send `ClientFinish` past this.
    AuthFailed,
    /// A `core::crypto` operation failed (bad ciphertext encoding, etc).
    Crypto(String),
}

impl std::fmt::Display for HandshakeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HandshakeError::AlgoMismatch => write!(f, "server accepted a different algorithm than requested"),
            HandshakeError::AuthFailed => write!(f, "server signature did not verify against the pinned key"),
            HandshakeError::Crypto(m) => write!(f, "crypto error: {m}"),
        }
    }
}

impl std::error::Error for HandshakeError {}

/// A client mid-handshake: holds the ephemeral keys needed to finish it.
///
/// One instance is good for exactly one handshake (or rekey) — `start`
/// generates fresh keys and a fresh nonce every time, as the protocol
/// requires (`PROTOCOL.md` §7: nonces make every transcript unique).
pub enum ClientHandshake {
    K512(hk::HybridClientKeys512, ClientHello),
    K768(hk::HybridClientKeys768, ClientHello),
    K1024(hk::HybridClientKeys1024, ClientHello),
}

impl ClientHandshake {
    /// Generates ephemeral X25519 + ML-KEM keys for `algo` and builds the
    /// `ClientHello` to send.
    pub fn start(algo: MlKemLevel, client_wg_pubkey: [u8; 32]) -> Self {
        let client_nonce = hk::generate_nonce();

        macro_rules! build {
            ($gen:path, $variant:ident) => {{
                let keys = $gen();
                let hello = ClientHello {
                    algo,
                    client_nonce,
                    client_wg_pubkey,
                    client_x25519_pub: keys.x25519_public().to_bytes(),
                    client_mlkem_pub: keys.mlkem_public().as_bytes().to_vec(),
                };
                ClientHandshake::$variant(keys, hello)
            }};
        }

        match algo {
            MlKemLevel::MlKem512 => build!(hk::generate_client_keypairs_512, K512),
            MlKemLevel::MlKem768 => build!(hk::generate_client_keypairs_768, K768),
            MlKemLevel::MlKem1024 => build!(hk::generate_client_keypairs_1024, K1024),
        }
    }

    /// The `ClientHello` to send first.
    pub fn hello(&self) -> &ClientHello {
        match self {
            ClientHandshake::K512(_, h)
            | ClientHandshake::K768(_, h)
            | ClientHandshake::K1024(_, h) => h,
        }
    }

    /// Verifies the server's signature against `pinned_verifying_key`,
    /// decapsulates the hybrid secret, and derives the session keys.
    ///
    /// Returns the derived [`SessionKeys`], the `ClientFinish` to send, and
    /// the signed transcript (needed again to verify `ServerFinish`).
    /// Verification happens **before** any secret is derived — an
    /// unauthenticated `ServerHello` never reaches the KEM decapsulation
    /// step.
    pub fn finish(
        &self,
        server_hello: &ServerHello,
        pinned_verifying_key: &[u8],
    ) -> Result<(SessionKeys, ClientFinish, Vec<u8>), HandshakeError> {
        let hello = self.hello();
        if server_hello.algo != hello.algo {
            return Err(HandshakeError::AlgoMismatch);
        }

        let transcript = hk::build_handshake_transcript_from_bytes(
            hello.algo.name().as_bytes(),
            &hello.client_nonce,
            &server_hello.server_nonce,
            &hello.client_x25519_pub,
            &hello.client_mlkem_pub,
            &server_hello.server_x25519_pub,
            &server_hello.mlkem_ciphertext,
        );

        if !verify_server_signature(pinned_verifying_key, &transcript, &server_hello.signature) {
            return Err(HandshakeError::AuthFailed);
        }

        let server_x = PublicKey::from(server_hello.server_x25519_pub);
        let ct_bytes = server_hello.mlkem_ciphertext.as_slice();
        let bad_ct = || HandshakeError::Crypto("server ciphertext has the wrong length".into());

        let hybrid_secret = match self {
            ClientHandshake::K512(keys, _) => {
                let ct = Ciphertext::<MlKem512>::try_from(ct_bytes).map_err(|_| bad_ct())?;
                hk::client_decapsulate_512(keys, &server_x, &ct).map_err(HandshakeError::Crypto)?
            }
            ClientHandshake::K768(keys, _) => {
                let ct = Ciphertext::<MlKem768>::try_from(ct_bytes).map_err(|_| bad_ct())?;
                hk::client_decapsulate_768(keys, &server_x, &ct).map_err(HandshakeError::Crypto)?
            }
            ClientHandshake::K1024(keys, _) => {
                let ct = Ciphertext::<MlKem1024>::try_from(ct_bytes).map_err(|_| bad_ct())?;
                hk::client_decapsulate_1024(keys, &server_x, &ct).map_err(HandshakeError::Crypto)?
            }
        };

        let keys = derive_session_keys(&hybrid_secret, &hello.client_nonce, &server_hello.server_nonce);
        let client_tag = tags::client_tag(&keys.confirm_key, &transcript);

        Ok((
            keys,
            ClientFinish {
                session_id: server_hello.session_id,
                client_tag,
            },
            transcript,
        ))
    }
}

/// Verifies a server signature against pinned verifying-key bytes
/// (`server/PROTOCOL.md` §5.2). `verifying_key_bytes` must be exactly
/// [`VERIFYING_KEY_LEN`] bytes and `signature_bytes` exactly
/// `wire::SIGNATURE_LEN` bytes — a wrong length is treated as "does not
/// verify" rather than a panic, since both come straight off the wire.
pub fn verify_server_signature(
    verifying_key_bytes: &[u8],
    message: &[u8],
    signature_bytes: &[u8],
) -> bool {
    let Ok(vk_arr) = <[u8; VERIFYING_KEY_LEN]>::try_from(verifying_key_bytes) else {
        return false;
    };
    let vk = ServerVerifyingKey::decode(&vk_arr.into());

    let Ok(sig_arr) = <[u8; super::wire::SIGNATURE_LEN]>::try_from(signature_bytes) else {
        return false;
    };
    let Some(sig) = ServerSignature::decode(&sig_arr.into()) else {
        return false;
    };

    crate::crypto::ServerAuthenticator::verify(&vk, message, &sig)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::ServerAuthenticator;
    use crate::protocol::wire::ServerHello;

    const ALL: [MlKemLevel; 3] = [MlKemLevel::MlKem512, MlKemLevel::MlKem768, MlKemLevel::MlKem1024];

    /// Builds a genuine `ServerHello` the way `server/handshake-server`'s
    /// `handshake::respond` does (same `Encoded<K::EncapsulationKey>` ->
    /// `from_bytes` reconstruction), so these tests exercise the real
    /// client against real (if locally simulated) server crypto rather than
    /// hand-rolled fixtures.
    /// `response_algo` is what the returned `ServerHello` claims (its
    /// `algo` field and the transcript's algorithm-name bytes) — normally
    /// equal to `client_hello.algo`, but deliberately different in
    /// [`server_algo_mismatch_is_rejected_before_touching_crypto`]. The
    /// actual KEM encapsulation always runs at `client_hello.algo`, since
    /// that is the only key size the client's `client_mlkem_pub` bytes
    /// actually are — a real server response might *claim* the wrong
    /// algorithm, but it cannot retroactively make the client's ML-KEM
    /// public key a different size.
    fn server_side_hello(
        response_algo: MlKemLevel,
        client_hello: &ClientHello,
        authenticator: &ServerAuthenticator,
        session_id: [u8; 16],
        server_nonce: [u8; 32],
    ) -> (ServerHello, [u8; 32]) {
        use ::ml_kem::{Encoded, KemCore};

        let client_x25519_pub = PublicKey::from(client_hello.client_x25519_pub);

        macro_rules! respond {
            ($encap:path, $K:ty) => {{
                let enc = Encoded::<<$K as KemCore>::EncapsulationKey>::try_from(
                    client_hello.client_mlkem_pub.as_slice(),
                )
                .expect("test-generated ClientHello has a correctly sized key");
                let client_mlkem_pub = <$K as KemCore>::EncapsulationKey::from_bytes(&enc);

                let server_result = $encap(&client_x25519_pub, &client_mlkem_pub).unwrap();

                let transcript = hk::build_handshake_transcript_from_bytes(
                    response_algo.name().as_bytes(),
                    &client_hello.client_nonce,
                    &server_nonce,
                    &client_hello.client_x25519_pub,
                    &client_hello.client_mlkem_pub,
                    server_result.server_x25519_public.as_bytes(),
                    server_result.ciphertext.as_ref(),
                );
                let signature = authenticator.sign(&transcript).encode().to_vec();

                (
                    ServerHello {
                        algo: response_algo,
                        session_id,
                        server_nonce,
                        server_x25519_pub: *server_result.server_x25519_public.as_bytes(),
                        mlkem_ciphertext: server_result.ciphertext.to_vec(),
                        signature,
                    },
                    server_result.hybrid_secret,
                )
            }};
        }

        match client_hello.algo {
            MlKemLevel::MlKem512 => respond!(hk::server_encapsulate_512, ::ml_kem::MlKem512),
            MlKemLevel::MlKem768 => respond!(hk::server_encapsulate_768, ::ml_kem::MlKem768),
            MlKemLevel::MlKem1024 => respond!(hk::server_encapsulate_1024, ::ml_kem::MlKem1024),
        }
    }

    #[test]
    fn full_handshake_agrees_with_the_server_side_secret_at_every_level() {
        for algo in ALL {
            let authenticator = ServerAuthenticator::generate();
            let vk = authenticator.verifying_key().encode().to_vec();

            let client = ClientHandshake::start(algo, [7u8; 32]);
            let (server_hello, server_secret) =
                server_side_hello(algo, client.hello(), &authenticator, [9u8; 16], [8u8; 32]);

            let (keys, finish, _transcript) = client
                .finish(&server_hello, &vk)
                .unwrap_or_else(|e| panic!("{algo:?}: finish failed: {e}"));

            assert_eq!(finish.session_id, server_hello.session_id);

            // The client derived the same PSK the server would derive from
            // its own hybrid_secret (same nonces, same derive_session_keys).
            let server_keys =
                derive_session_keys(&server_secret, &client.hello().client_nonce, &server_hello.server_nonce);
            assert_eq!(*keys.psk, *server_keys.psk);
        }
    }

    #[test]
    fn wrong_pinned_key_is_rejected() {
        let authenticator = ServerAuthenticator::generate();
        let attacker = ServerAuthenticator::generate();
        let attacker_vk = attacker.verifying_key().encode().to_vec();

        let client = ClientHandshake::start(MlKemLevel::MlKem768, [1u8; 32]);
        let (server_hello, _) =
            server_side_hello(MlKemLevel::MlKem768, client.hello(), &authenticator, [0u8; 16], [0u8; 32]);

        assert!(matches!(
            client.finish(&server_hello, &attacker_vk),
            Err(HandshakeError::AuthFailed)
        ));
    }

    #[test]
    fn tampered_ciphertext_fails_authentication() {
        let authenticator = ServerAuthenticator::generate();
        let vk = authenticator.verifying_key().encode().to_vec();

        let client = ClientHandshake::start(MlKemLevel::MlKem768, [1u8; 32]);
        let (mut server_hello, _) =
            server_side_hello(MlKemLevel::MlKem768, client.hello(), &authenticator, [0u8; 16], [0u8; 32]);
        server_hello.mlkem_ciphertext[0] ^= 0x01;

        // The signature was over the *original* ciphertext, so tampering
        // with it is caught by the signature check, not the KEM.
        assert!(matches!(
            client.finish(&server_hello, &vk),
            Err(HandshakeError::AuthFailed)
        ));
    }

    #[test]
    fn server_algo_mismatch_is_rejected_before_touching_crypto() {
        let authenticator = ServerAuthenticator::generate();
        let vk = authenticator.verifying_key().encode().to_vec();

        let client = ClientHandshake::start(MlKemLevel::MlKem512, [1u8; 32]);
        // Server (or an attacker) replies at a different level than requested.
        let (server_hello, _) =
            server_side_hello(MlKemLevel::MlKem768, client.hello(), &authenticator, [0u8; 16], [0u8; 32]);

        assert!(matches!(
            client.finish(&server_hello, &vk),
            Err(HandshakeError::AlgoMismatch)
        ));
    }
}
