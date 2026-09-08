//! Client side of the handshake — used by the `test-client` binary and the
//! vector emitter. Member 1's real client lives in `core/protocol/` and is
//! verified against the same `server/handshake-vectors.json` fixtures.
//!
//! All crypto is `vpn_core::crypto`.

use ml_kem::{Ciphertext, EncodedSizeUser, MlKem512, MlKem768, MlKem1024};
use x25519_dalek::PublicKey;

use vpn_core::crypto::hybrid_kem as ck;

use crate::error::HandshakeError;
use crate::identity;
use crate::kdf::{self, SessionKeys};
use crate::transcript::{self, TranscriptInputs};
use crate::wire::{AlgoCode, ClientFinish, ClientHello, ServerHello};

/// A client mid-handshake: holds the ephemeral keys needed to finish.
pub enum ClientState {
    K512(ck::HybridClientKeys512, ClientHello),
    K768(ck::HybridClientKeys768, ClientHello),
    K1024(ck::HybridClientKeys1024, ClientHello),
}

/// Generate ephemeral keys and the `ClientHello` for `algo`.
pub fn start(algo: AlgoCode, client_nonce: [u8; 32], client_wg_pubkey: [u8; 32]) -> ClientState {
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
            ClientState::$variant(keys, hello)
        }};
    }
    match algo {
        AlgoCode::MlKem512 => build!(ck::generate_client_keypairs_512, K512),
        AlgoCode::MlKem768 => build!(ck::generate_client_keypairs_768, K768),
        AlgoCode::MlKem1024 => build!(ck::generate_client_keypairs_1024, K1024),
    }
}

impl ClientState {
    pub fn hello(&self) -> &ClientHello {
        match self {
            ClientState::K512(_, h) | ClientState::K768(_, h) | ClientState::K1024(_, h) => h,
        }
    }

    /// Verify the server's signature, derive keys, and produce `ClientFinish`.
    /// Returns the session keys, the finish message, and the signed transcript
    /// (needed to check `ServerFinish`).
    pub fn finish(
        &self,
        server_hello: &ServerHello,
        pinned_verifying_key: &[u8],
    ) -> Result<(SessionKeys, ClientFinish, Vec<u8>), HandshakeError> {
        let hello = self.hello();
        if server_hello.algo != hello.algo {
            return Err(HandshakeError::AlgoMismatch);
        }

        let transcript = transcript::build(&TranscriptInputs {
            algo: hello.algo,
            client_nonce: &hello.client_nonce,
            server_nonce: &server_hello.server_nonce,
            client_x25519_pub: &hello.client_x25519_pub,
            client_mlkem_pub: &hello.client_mlkem_pub,
            server_x25519_pub: &server_hello.server_x25519_pub,
            mlkem_ciphertext: &server_hello.mlkem_ciphertext,
        });

        if !identity::verify(pinned_verifying_key, &transcript, &server_hello.signature) {
            return Err(HandshakeError::AuthFailed);
        }

        let server_x = PublicKey::from(server_hello.server_x25519_pub);
        let ct_bytes = server_hello.mlkem_ciphertext.as_slice();
        let bad_ct = || HandshakeError::Crypto("server ciphertext wrong length".into());

        let hybrid_secret = match self {
            ClientState::K512(k, _) => {
                let ct = Ciphertext::<MlKem512>::try_from(ct_bytes).map_err(|_| bad_ct())?;
                ck::client_decapsulate_512(k, &server_x, &ct).map_err(HandshakeError::Crypto)?
            }
            ClientState::K768(k, _) => {
                let ct = Ciphertext::<MlKem768>::try_from(ct_bytes).map_err(|_| bad_ct())?;
                ck::client_decapsulate_768(k, &server_x, &ct).map_err(HandshakeError::Crypto)?
            }
            ClientState::K1024(k, _) => {
                let ct = Ciphertext::<MlKem1024>::try_from(ct_bytes).map_err(|_| bad_ct())?;
                ck::client_decapsulate_1024(k, &server_x, &ct).map_err(HandshakeError::Crypto)?
            }
        };

        let keys = kdf::derive(&hybrid_secret, &hello.client_nonce, &server_hello.server_nonce);
        let client_tag = kdf::client_tag(&keys.confirm_key, &transcript);
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
