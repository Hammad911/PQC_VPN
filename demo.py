#!/usr/bin/env python3
"""
End-to-end demo: device state -> trained RL agent picks a PQC
algorithm -> a real hybrid X25519+ML-KEM handshake runs with that
algorithm, timed.

Ties Phase 1 (client/vpn_daemon/) and Phase 2 (client/rl_agent/)
together into one runnable script.

What's genuinely live: the anomaly combiner (Layer 1's Z-score baseline,
always on, plus the Layer 2 rule-based signatures below 70% CPU — this
laptop's own CPU load decides whether Layer 2 runs, same as it would in
production), the agent's forward pass, and the crypto handshake itself
(real key generation, encapsulation, decapsulation — timed as it happens).

What's staged, and printed as [STAGED]: CPU/RAM/connection-type/session
-age, and in two scenarios the threat score. A laptop demo can't hop
onto a cellular network, run a 45-minute session, or arrange a genuine
attack on cue, so those conditions are set explicitly.

Staging used to carry a second reason — the 50k model's ML-KEM-768
boundary was so close to ambient CPU load that whatever ran in the
background could flip the outcome. That is no longer true: after the
Week 2 retraining the five scenarios below clear their decision
boundaries by margins of 0.04 to 0.33 in reward terms.

Each scenario was checked against the reward function before being
wired in, and the agent's choice is printed next to the analytically
optimal one so the demo shows whether a decision is *right*, not just
what it was. Between them the five cover all four actions.

Run with: python demo.py
"""
import time

import numpy as np
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from stable_baselines3 import PPO

from client.rl_agent.anomaly_detector import AnomalyCombiner
from client.rl_agent.state_observer import StateObserver
from client.rl_agent.vpn_env import (
    CPU_LOAD, RAM_AVAIL, CONN_TYPE, TIME_SINCE_REKEY, THREAT, optimal_action,
)
from client.vpn_daemon.algo_registry import ACTIVE_ACTIONS
from client.vpn_daemon.auth import ServerAuthenticator
from client.vpn_daemon.hybrid_kem import HybridKEM

MODEL_PATH = "client/rl_agent/models/ppo_vpn_agent.zip"
ACTION_INDEX = {i: k for i, k in enumerate(ACTIVE_ACTIONS.keys())}

# Benign per-tick Layer 2 inputs for scenarios that aren't specifically
# injecting a Layer 2 signature — keeps every run_round() call below every
# signature's trigger line (see anomaly_detector.py for the thresholds).
NORMAL_LAYER2 = dict(
    distinct_ports=2, connection_attempts=2, failed_attempts=0,
    retransmit_rate=0.0, latency_ms=10.0,
    upload_bytes_per_sec=1_000.0, download_bytes_per_sec=1_000.0,
    dns_server="1.1.1.1",
)
# A synthetic port-scan pattern (PortScanSignature's own trigger example) —
# high fanout, mostly-failed connections.
PORT_SCAN_LAYER2 = {**NORMAL_LAYER2,
                     "distinct_ports": 20, "connection_attempts": 20, "failed_attempts": 18}


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


def run_round(label, model, observer, detector, *, staged, current_algo,
              inject_anomaly=False, inject_port_scan=False):
    """staged: dict of {index: value} overriding the live-read state,
    e.g. {CPU_LOAD: 0.2, CONN_TYPE: 1.0}.

    current_algo: the KEM currently in force for this session. A rekey
    rotates key material at the same strength, so this is what a
    `rekey-now` decision re-runs the handshake with. Returns the algorithm
    in force after this round.
    """
    banner(f"SCENARIO: {label}")

    # One live CPU sample, shared between the combiner's Layer 2 gate and
    # the state vector's CPU_LOAD dimension — see state_observer.cpu_load().
    cpu = observer.cpu_load()

    if inject_anomaly:
        # calibrate a normal baseline, then feed one outlier — this is the
        # real Layer 1 Z-score detector reacting, not a hardcoded number
        for _ in range(25):
            detector.update(cpu_load=cpu, latency=0.1, packet_rate=0.1,
                             packet_size=0.1, **NORMAL_LAYER2)
        result = detector.update(cpu_load=cpu, latency=0.1, packet_rate=0.1,
                                  packet_size=50.0, **NORMAL_LAYER2)
    elif inject_port_scan:
        # the real Layer 2 PortScanSignature reacting, gated live by this
        # laptop's own current CPU load (see the [STAGED] cpu note below —
        # only the printed/agent-facing CPU_LOAD is staged, the gate always
        # uses the real reading).
        result = detector.update(cpu_load=cpu, latency=0.1, packet_rate=0.1,
                                  packet_size=0.12, **PORT_SCAN_LAYER2)
    else:
        result = detector.update(cpu_load=cpu, latency=0.1, packet_rate=0.1,
                                  packet_size=0.12, **NORMAL_LAYER2)

    state = observer.read_state(threat_score=result["threat_score"], cpu_load=cpu)

    STATE_NAMES = {
        CPU_LOAD: "cpu", RAM_AVAIL: "ram_avail", CONN_TYPE: "conn_type",
        TIME_SINCE_REKEY: "time_since_rekey", THREAT: "threat",
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
        f"threat={state[6]:.2f} (live combiner: Layer 1 + Layer 2 below 70% CPU)"
    )

    action, _ = model.predict(state, deterministic=True)
    algo_key = ACTION_INDEX[int(action)]
    algo = ACTIVE_ACTIONS[algo_key]

    # Print the analytically optimal action next to the agent's, so the demo
    # shows whether the decision is *right* rather than only what it was.
    best = optimal_action(state)
    best_name = ACTIVE_ACTIONS[ACTION_INDEX[best]]["name"]
    verdict = "matches the optimal choice" if best == int(action) else \
              f"DIFFERS from the optimal choice ({best_name})"
    print(f"agent decision -> {algo['name']}  "
          f"(security={algo['security']}, cpu_cost={algo['cpu_cost']})")
    print(f"               -> {verdict}")

    if algo["name"] == "rekey-now":
        observer.mark_rekey()
        # A rekey rotates key material without changing strength — the tier
        # stays whatever the agent last selected. See INTERFACE_FREEZE_PROPOSAL.md;
        # Member 2's server relies on this to swap a peer's PSK without
        # renegotiating the algorithm.
        print(f"agent triggered immediate rekey — fresh handshake at the "
              f"algorithm in force ({current_algo})")
        result = run_handshake(current_algo)
        next_algo = current_algo
    else:
        result = run_handshake(algo["name"])
        next_algo = algo["name"]

    print(
        f"handshake complete in {result['elapsed_ms']:.2f} ms  "
        f"(ML-KEM pubkey {result['mlkem_pub_bytes']}B, "
        f"ciphertext {result['ciphertext_bytes']}B)"
    )
    print(f"shared secret (first 8 bytes): {result['shared_secret'][:8].hex()}")
    return next_algo


def main():
    banner("RL-PQC-VPN — live demo")
    print("Loading trained PPO agent...")
    model = PPO.load(MODEL_PATH)
    observer = StateObserver()
    detector = AnomalyCombiner(window=20)

    demo_server_auth()

    # The session has to be established with something before the agent's
    # first decision; ML-KEM-768 is hybrid_kem.py's default.
    current_algo = "ML-KEM-768"

    # Five scenarios chosen so that each of the four actions is the correct
    # answer somewhere, and every one was checked against the reward function
    # before being wired in — the agent agrees with the optimal choice on all
    # five, with margins from 0.04 to 0.33. The earlier four-scenario script
    # was built around the 50k smoke model and encoded two of its mistakes as
    # if they were the expected behaviour.
    current_algo = run_round(
        "idle laptop, fresh session, trusted wifi",
        model, observer, detector, current_algo=current_algo,
        staged={CPU_LOAD: 0.2, RAM_AVAIL: 0.7, CONN_TYPE: 0.0, TIME_SINCE_REKEY: 0.0},
    )
    current_algo = run_round(
        "cellular + anomalous traffic spike (Layer 1 Z-score fires live)",
        model, observer, detector, current_algo=current_algo,
        staged={CPU_LOAD: 0.2, RAM_AVAIL: 0.7, CONN_TYPE: 1.0, TIME_SINCE_REKEY: 0.3},
        inject_anomaly=True,
    )
    current_algo = run_round(
        "idle but RAM-rich device, stale session, confirmed threat",
        model, observer, detector, current_algo=current_algo,
        staged={CPU_LOAD: 0.1, RAM_AVAIL: 0.95, CONN_TYPE: 1.0,
                TIME_SINCE_REKEY: 0.9, THREAT: 1.0},
    )
    current_algo = run_round(
        "same threat, but the device is now loaded and low on RAM",
        model, observer, detector, current_algo=current_algo,
        staged={CPU_LOAD: 0.9, RAM_AVAIL: 0.3, CONN_TYPE: 1.0,
                TIME_SINCE_REKEY: 1.0, THREAT: 1.0},
    )
    current_algo = run_round(
        "post-rekey, session fresh again",
        model, observer, detector, current_algo=current_algo,
        staged={CPU_LOAD: 0.3, RAM_AVAIL: 0.65, CONN_TYPE: 0.0,
                TIME_SINCE_REKEY: 0.0, THREAT: 0.1},
    )
    # Sixth scenario, appended rather than interleaved so it can't perturb
    # the current_algo chain the five reward-checked scenarios above rely
    # on. THREAT is deliberately left unstaged — the combiner's real Layer 2
    # PortScanSignature is what drives it, gated live by this machine's
    # actual (low, idle-demo) CPU load, not a hardcoded number.
    run_round(
        "reconnaissance sweep against an idle host (Layer 2 port-scan signature fires live)",
        model, observer, detector, current_algo=current_algo,
        staged={CPU_LOAD: 0.2, RAM_AVAIL: 0.7, CONN_TYPE: 0.5, TIME_SINCE_REKEY: 0.2},
        inject_port_scan=True,
    )

    banner("done")


if __name__ == "__main__":
    main()
