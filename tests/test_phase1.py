# tests/test_phase1.py
"""
Phase 1 integration test.
Run with: pytest tests/test_phase1.py -v
All tests must pass before moving to Phase 2.
"""
import pytest

def test_liboqs_installed():
    import oqs
    assert oqs is not None

def test_mlkem_512_roundtrip():
    import oqs
    with oqs.KeyEncapsulation("ML-KEM-512") as client:
        pub = client.generate_keypair()
        with oqs.KeyEncapsulation("ML-KEM-512") as server:
            ct, s_secret = server.encap_secret(pub)
        c_secret = client.decap_secret(ct)
    assert c_secret == s_secret

def test_mlkem_768_roundtrip():
    import oqs
    with oqs.KeyEncapsulation("ML-KEM-768") as client:
        pub = client.generate_keypair()
        with oqs.KeyEncapsulation("ML-KEM-768") as server:
            ct, s_secret = server.encap_secret(pub)
        c_secret = client.decap_secret(ct)
    assert c_secret == s_secret

def test_mlkem_1024_roundtrip():
    import oqs
    with oqs.KeyEncapsulation("ML-KEM-1024") as client:
        pub = client.generate_keypair()
        with oqs.KeyEncapsulation("ML-KEM-1024") as server:
            ct, s_secret = server.encap_secret(pub)
        c_secret = client.decap_secret(ct)
    assert c_secret == s_secret

def test_mldsa_sign_verify():
    import oqs
    message = b"test server public key"
    with oqs.Signature("ML-DSA-65") as signer:
        pub = signer.generate_keypair()
        priv = signer.export_secret_key()
        sig = signer.sign(message)
    with oqs.Signature("ML-DSA-65") as verifier:
        valid = verifier.verify(message, sig, pub)
    assert valid

def test_hybrid_kem():
    import sys
    sys.path.insert(0, ".")
    from client.vpn_daemon.hybrid_kem import HybridKEM
    kem = HybridKEM()
    client_keys = kem.generate_client_keypairs()
    from cryptography.hazmat.primitives.asymmetric.x25519 import (
        X25519PrivateKey
    )
    from cryptography.hazmat.primitives.serialization import (
        Encoding, PublicFormat
    )
    server_priv = X25519PrivateKey.generate()
    server_pub  = server_priv.public_key().public_bytes(
        Encoding.Raw, PublicFormat.Raw
    )
    ct, server_secret = kem.server_encapsulate(
        client_keys["x25519_public"],
        client_keys["mlkem_public"],
        server_priv
    )
    client_secret = kem.client_decapsulate(
        client_keys["x25519_private"],
        server_pub,
        ct,
        client_keys["mlkem_private"]
    )
    assert client_secret == server_secret

def test_pytorch_mps():
    import torch
    assert torch.backends.mps.is_available(), \
        "MPS not available — check PyTorch M2 install"

def test_psutil_reads():
    import psutil
    cpu = psutil.cpu_percent(interval=0.1)
    ram = psutil.virtual_memory().percent
    net = psutil.net_io_counters()
    assert 0 <= cpu <= 100
    assert 0 <= ram <= 100
    assert net.bytes_sent >= 0

def test_gymnasium_import():
    import gymnasium as gym
    env = gym.make("CartPole-v1")
    obs, _ = env.reset()
    assert obs is not None

def test_stable_baselines():
    from stable_baselines3 import PPO
    import gymnasium as gym
    model = PPO("MlpPolicy", "CartPole-v1", verbose=0)
    model.learn(total_timesteps=100)
    assert model is not None