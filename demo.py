#!/usr/bin/env python3
"""
End-to-end demo: device state -> trained RL agent picks a PQC
algorithm -> a real hybrid X25519+ML-KEM handshake runs with that
algorithm, timed.

Ties Phase 1 (client/vpn_daemon/) and Phase 2 (client/rl_agent/)
together into one runnable script.

What's genuinely live: the Layer 1 Z-score anomaly check, the agent's
forward pass, and the crypto handshake itself (real key generation,
encapsulation, decapsulation — timed as it happens).

What's staged, and printed as [STAGED]: CPU/RAM/connection-type/session
-age per scenario. Two reasons: (1) a laptop demo can't actually hop
onto cellular or run a 45-minute session live, and (2) the trained
policy's ML-KEM-768 decision boundary sits close enough to CPU load
that whatever happens to be running in the background (browser, Zoom,
whatever) can flip the outcome — see PROGRESS.md for why. Staging
those inputs makes the demo reproducible instead of leaving it to
chance which tier shows up. The four scenarios below were verified
against the trained model to be stable under +-0.15 CPU / +-0.1 RAM
jitter before being wired in here.

Run with: python demo.py
"""
import time

import numpy as np
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from stable_baselines3 import PPO

from client.rl_agent.anomaly_detector import ZScoreBaseline
from client.rl_agent.state_observer import StateObserver
from client.rl_agent.vpn_env import (
    CPU_LOAD, RAM_AVAIL, CONN_TYPE, TIME_SINCE_REKEY,
)
from client.vpn_daemon.algo_registry import ACTIVE_ACTIONS
from client.vpn_daemon.auth import ServerAuthenticator
from client.vpn_daemon.hybrid_kem import HybridKEM

MODEL_PATH = "client/rl_agent/models/ppo_vpn_agent.zip"
ACTION_INDEX = {i: k for i, k in enumerate(ACTIVE_ACTIONS.keys())}


def banner(text):
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def run_handshake(algo_name: str):
    kem = HybridKEM()
    t0 = time.perf_counter()

    client_keys = kem.generate_client_keypairs(mlkem_algorithm=algo_name)
    server_priv = X25519PrivateKey.generate()
    server_pub = server_priv.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)

    ciphertext, server_secret = kem.server_encapsulate(
        client_keys["x25519_public"], client_keys["mlkem_public"],
        server_priv, mlkem_algorithm=algo_name,
    )
    client_secret = kem.client_decapsulate(
        client_keys["x25519_private"], server_pub, ciphertext,
        client_keys["mlkem_private"], mlkem_algorithm=algo_name,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert client_secret == server_secret, "handshake mismatch!"
    return {
        "elapsed_ms": elapsed_ms,
        "ciphertext_bytes": len(ciphertext),
        "mlkem_pub_bytes": len(client_keys["mlkem_public"]),
        "shared_secret": client_secret,
    }


def demo_server_auth():
    banner("SERVER AUTHENTICATION — ML-DSA-65 (real, live)")
    auth = ServerAuthenticator()
    t0 = time.perf_counter()
    server_pub, server_priv = auth.generate_server_identity()
    dummy_mlkem_pub = b"placeholder-mlkem-public-key-bytes"
    sig = auth.sign_public_key(dummy_mlkem_pub, server_priv)
    valid = auth.verify_server_public_key(dummy_mlkem_pub, sig, server_pub)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"server identity generated, key signed, signature verified: {valid}")
    print(f"took {elapsed_ms:.2f} ms")

    forged_pub = b"attacker-substituted-key-bytesss!!"
    rejected = not auth.verify_server_public_key(forged_pub, sig, server_pub)
    print(f"MitM substitution correctly rejected: {rejected}")


def run_round(label, model, observer, detector, *, staged, inject_anomaly=False):
    """staged: dict of {index: value} overriding the live-read state,
    e.g. {CPU_LOAD: 0.2, CONN_TYPE: 1.0}."""
    banner(f"SCENARIO: {label}")

    if inject_anomaly:
        # calibrate a normal baseline, then feed one outlier — this is the
        # real Layer 1 Z-score detector reacting, not a hardcoded number
        for _ in range(25):
            detector.update(latency=0.1, packet_rate=0.1, packet_size=0.1)
        result = detector.update(latency=0.1, packet_rate=0.1, packet_size=50.0)
    else:
        result = detector.update(latency=0.1, packet_rate=0.1, packet_size=0.12)

    state = observer.read_state(threat_score=result["threat_score"])

    STATE_NAMES = {
        CPU_LOAD: "cpu", RAM_AVAIL: "ram_avail",
        CONN_TYPE: "conn_type", TIME_SINCE_REKEY: "time_since_rekey",
    }
    staged_notes = []
    for idx, value in staged.items():
        state[idx] = value
        staged_notes.append(f"{STATE_NAMES[idx]}={value:.2f}")
    print(f"[STAGED] {', '.join(staged_notes)}")
    print(
        f"state  cpu={state[0]:.2f}  ram_avail={state[1]:.2f}  "
        f"latency={state[2]:.2f} (live ping)  upload={state[3]:.2f}  "
        f"conn_type={state[4]:.2f}  time_since_rekey={state[5]:.2f}  "
        f"threat={state[6]:.2f} (live Z-score)"
    )

    action, _ = model.predict(state, deterministic=True)
    algo_key = ACTION_INDEX[int(action)]
    algo = ACTIVE_ACTIONS[algo_key]
    print(f"agent decision -> {algo['name']}  "
          f"(security={algo['security']}, cpu_cost={algo['cpu_cost']})")

    if algo["name"] == "rekey-now":
        observer.mark_rekey()
        print("agent triggered immediate rekey — running fresh handshake at ML-KEM-768")
        result = run_handshake("ML-KEM-768")
    else:
        result = run_handshake(algo["name"])

    print(
        f"handshake complete in {result['elapsed_ms']:.2f} ms  "
        f"(ML-KEM pubkey {result['mlkem_pub_bytes']}B, "
        f"ciphertext {result['ciphertext_bytes']}B)"
    )
    print(f"shared secret (first 8 bytes): {result['shared_secret'][:8].hex()}")


def main():
    banner("RL-PQC-VPN — live demo")
    print("Loading trained PPO agent...")
    model = PPO.load(MODEL_PATH)
    observer = StateObserver()
    detector = ZScoreBaseline(window=20)

    demo_server_auth()

    run_round(
        "idle laptop, fresh session, trusted wifi",
        model, observer, detector,
        staged={CPU_LOAD: 0.2, RAM_AVAIL: 0.7, CONN_TYPE: 0.0, TIME_SINCE_REKEY: 0.0},
    )
    run_round(
        "on cellular, 45 minutes into the session, calm traffic",
        model, observer, detector,
        staged={CPU_LOAD: 0.15, RAM_AVAIL: 0.75, CONN_TYPE: 1.0, TIME_SINCE_REKEY: 0.7},
    )
    run_round(
        "cellular + anomalous traffic spike (Layer 1 Z-score fires live)",
        model, observer, detector,
        staged={CPU_LOAD: 0.2, RAM_AVAIL: 0.7, CONN_TYPE: 1.0, TIME_SINCE_REKEY: 0.3},
        inject_anomaly=True,
    )
    run_round(
        "post-rekey, session fresh again",
        model, observer, detector,
        staged={CPU_LOAD: 0.2, RAM_AVAIL: 0.7, CONN_TYPE: 0.0, TIME_SINCE_REKEY: 0.0},
    )

    banner("done")


if __name__ == "__main__":
    main()
