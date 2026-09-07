#[cfg(test)]
mod tests {
    use rand::rngs::OsRng;
    use x25519_dalek::{PublicKey, StaticSecret};

    #[test]
    fn x25519_shared_secrets_match() {
        // Client generates a private key and its corresponding public key.
        let client_private = StaticSecret::random_from_rng(OsRng);
        let client_public = PublicKey::from(&client_private);

        // Server independently generates its private and public keys.
        let server_private = StaticSecret::random_from_rng(OsRng);
        let server_public = PublicKey::from(&server_private);

        // Client combines its private key with the server's public key.
        let client_shared_secret = client_private.diffie_hellman(&server_public);

        // Server combines its private key with the client's public key.
        let server_shared_secret = server_private.diffie_hellman(&client_public);

        // Both sides must calculate exactly the same secret.
        assert_eq!(
            client_shared_secret.as_bytes(),
            server_shared_secret.as_bytes()
        );
    }
}
