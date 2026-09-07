#[cfg(test)]
mod tests {
    use ::kem::{Decapsulate, Encapsulate};
    use ::ml_kem::{KemCore, MlKem512, MlKem768, MlKem1024};

    #[test]
    fn ml_kem_512_round_trip() {
        let mut rng = rand::thread_rng();

        let (client_private_key, client_public_key) = MlKem512::generate(&mut rng);

        let (ciphertext, server_shared_secret) = client_public_key
            .encapsulate(&mut rng)
            .expect("ML-KEM-512 encapsulation failed");

        let client_shared_secret = client_private_key
            .decapsulate(&ciphertext)
            .expect("ML-KEM-512 decapsulation failed");

        assert_eq!(client_shared_secret, server_shared_secret);
    }

    #[test]
    fn ml_kem_768_round_trip() {
        let mut rng = rand::thread_rng();

        let (client_private_key, client_public_key) = MlKem768::generate(&mut rng);

        let (ciphertext, server_shared_secret) = client_public_key
            .encapsulate(&mut rng)
            .expect("ML-KEM-768 encapsulation failed");

        let client_shared_secret = client_private_key
            .decapsulate(&ciphertext)
            .expect("ML-KEM-768 decapsulation failed");

        assert_eq!(client_shared_secret, server_shared_secret);
    }

    #[test]
    fn ml_kem_1024_round_trip() {
        let mut rng = rand::thread_rng();

        let (client_private_key, client_public_key) = MlKem1024::generate(&mut rng);

        let (ciphertext, server_shared_secret) = client_public_key
            .encapsulate(&mut rng)
            .expect("ML-KEM-1024 encapsulation failed");

        let client_shared_secret = client_private_key
            .decapsulate(&ciphertext)
            .expect("ML-KEM-1024 decapsulation failed");

        assert_eq!(client_shared_secret, server_shared_secret);
    }
}
