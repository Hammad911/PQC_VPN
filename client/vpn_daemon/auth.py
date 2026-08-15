# client/vpn_daemon/auth.py
"""
ML-DSA-65 based server authentication.
Prevents MitM from substituting their own ML-KEM public key.
"""
import oqs

class ServerAuthenticator:
    def generate_server_identity(self):
        """Run once on server — generates long-term identity keys."""
        with oqs.Signature("ML-DSA-65") as signer:
            public_key  = signer.generate_keypair()
            private_key = signer.export_secret_key()
        return public_key, private_key

    def sign_public_key(self, mlkem_public_key: bytes,
                         dsa_private_key: bytes) -> bytes:
        """Server signs its ML-KEM public key."""
        with oqs.Signature("ML-DSA-65",
                            secret_key=dsa_private_key) as signer:
            return signer.sign(mlkem_public_key)

    def verify_server_public_key(self, mlkem_public_key: bytes,
                                  signature: bytes,
                                  known_server_dsa_pub: bytes) -> bool:
        """Client verifies server's ML-KEM key is genuine."""
        with oqs.Signature("ML-DSA-65") as verifier:
            return verifier.verify(
                mlkem_public_key, signature, known_server_dsa_pub
            )