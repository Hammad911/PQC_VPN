# client/vpn_daemon/hybrid_kem.py
"""
Hybrid key exchange: X25519 (classical) + ML-KEM-768 (post-quantum)
Secure even if one algorithm is broken.
"""
import oqs
import os
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey
)
from cryptography.hazmat.primitives.serialization import (
    Encoding, PublicFormat, PrivateFormat, NoEncryption
)
import hashlib

class HybridKEM:
    """
    Combines X25519 + ML-KEM-768 for hybrid post-quantum key exchange.
    Shared secret = SHA256(x25519_secret || mlkem_secret)
    """

    def generate_client_keypairs(self, mlkem_algorithm: str = "ML-KEM-768"):
        # Classical keypair
        x25519_private = X25519PrivateKey.generate()
        x25519_public  = x25519_private.public_key().public_bytes(
            Encoding.Raw, PublicFormat.Raw
        )

        # Post-quantum keypair
        kem = oqs.KeyEncapsulation(mlkem_algorithm)
        mlkem_public  = kem.generate_keypair()
        mlkem_private = kem.export_secret_key()

        return {
            "x25519_private": x25519_private,
            "x25519_public":  x25519_public,
            "mlkem_public":   mlkem_public,
            "mlkem_private":  mlkem_private,
        }

    def server_encapsulate(self, x25519_client_pub: bytes,
                            mlkem_client_pub: bytes,
                            server_x25519_private,
                            mlkem_algorithm: str = "ML-KEM-768"):
        # Classical DH
        from cryptography.hazmat.primitives.asymmetric.x25519 import (
            X25519PublicKey
        )
        client_x25519 = X25519PublicKey.from_public_bytes(
            x25519_client_pub
        )
        x25519_secret = server_x25519_private.exchange(client_x25519)

        # PQ encapsulation
        with oqs.KeyEncapsulation(mlkem_algorithm) as kem:
            ciphertext, mlkem_secret = kem.encap_secret(mlkem_client_pub)

        # Combine both secrets
        hybrid_secret = hashlib.sha256(
            x25519_secret + mlkem_secret
        ).digest()

        return ciphertext, hybrid_secret

    def client_decapsulate(self, x25519_private,
                            server_x25519_pub: bytes,
                            ciphertext: bytes,
                            mlkem_private: bytes,
                            mlkem_algorithm: str = "ML-KEM-768"):
        # Classical DH
        from cryptography.hazmat.primitives.asymmetric.x25519 import (
            X25519PublicKey
        )
        server_pub = X25519PublicKey.from_public_bytes(server_x25519_pub)
        x25519_secret = x25519_private.exchange(server_pub)

        # PQ decapsulation
        with oqs.KeyEncapsulation(mlkem_algorithm,
                                   secret_key=mlkem_private) as kem:
            mlkem_secret = kem.decap_secret(ciphertext)

        # Combine — same formula as server
        hybrid_secret = hashlib.sha256(
            x25519_secret + mlkem_secret
        ).digest()

        return hybrid_secret