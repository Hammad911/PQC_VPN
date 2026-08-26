"""
Generates the machine-readable form of the interfaces frozen in
INTERFACE_FREEZE_PROPOSAL.md, so Members 1 and 2 consume them as data
instead of re-typing them into Rust.

The point is drift. The algorithm table currently exists in
algo_registry.py, in the freeze proposal's prose, and would shortly exist
again in Member 1's Rust core and Member 2's server — four copies, three
of them hand-maintained. Generating from the Python source of truth means
a change there either propagates or fails a test, rather than silently
disagreeing at integration time.

Writes contracts/algo_registry.json and contracts/state_vector.json, then
delegates to export_onnx for the model artifact and its test vectors.

Run with: python -m client.rl_agent.export_contracts
"""
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from client.rl_agent import export_onnx  # noqa: E402
from client.rl_agent import decision_gate  # noqa: E402
from client.rl_agent.state_observer import (  # noqa: E402
    CONNECTION_TYPE_SCORE,
    LATENCY_CAP_MS,
    REKEY_INTERVAL_CAP_SEC,
    UPLOAD_CAP_BYTES_PER_SEC,
)
from client.rl_agent.vpn_env import (  # noqa: E402
    CONN_TYPE,
    CPU_LOAD,
    LATENCY,
    RAM_AVAIL,
    STATE_DIM,
    THREAT,
    TIME_SINCE_REKEY,
    UPLOAD,
)
from client.vpn_daemon.algo_registry import ACTIVE_ACTIONS, ALGORITHMS  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACTS_DIR = REPO_ROOT / "contracts"

GENERATOR = "python -m client.rl_agent.export_contracts"

# Decided at the Week 1 sync and recorded here because Member 2's Week 6 task
# (accept a fresh handshake for an existing peer) needs it pinned: a rekey
# rotates key material without changing strength. "When to rotate" and "how
# strong" stay independent decisions, matching the proposal's stated reason to
# rekey — stale session keys sitting in RAM, not a threat escalation.
REKEY_SEMANTICS = "re-run the handshake using the algorithm currently in force"


def build_algo_registry() -> dict:
    actions = []
    for action_index, (registry_key, algo) in enumerate(ACTIVE_ACTIONS.items()):
        is_rekey = algo["name"] == "rekey-now"
        entry = {
            "action_index": action_index,
            "registry_key": registry_key,
            "name": algo["name"],
            "standard": algo["standard"],
            "security": algo["security"],
            "cpu_cost": algo["cpu_cost"],
            "kind": "rekey" if is_rekey else "kem",
            # hybrid_kem.py passes `name` straight through to liboqs, so for the
            # KEM actions the name IS the liboqs algorithm identifier.
            "liboqs_id": None if is_rekey else algo["name"],
        }
        if is_rekey:
            entry["semantics"] = REKEY_SEMANTICS
        actions.append(entry)

    inactive = [
        {
            "registry_key": key,
            "name": algo["name"],
            "standard": algo["standard"],
            "reason_disabled": "draft NIST standard, not finalized",
        }
        for key, algo in ALGORITHMS.items()
        if not algo["active"]
    ]

    return {
        "schema_version": 1,
        "generated_by": GENERATOR,
        "source_of_truth": "client/vpn_daemon/algo_registry.py",
        "action_space_size": len(actions),
        "notes": [
            "action_index is the contiguous 0..N-1 index the RL policy emits; "
            "registry_key is the algo_registry.py key and is NOT contiguous "
            "(key 3 is the disabled HQC-256).",
            "Read action_space_size from the exported ONNX output shape rather "
            "than hardcoding it, so enabling HQC-256 later is a re-export "
            "instead of a Rust code change.",
        ],
        "actions": actions,
        "inactive": inactive,
    }


def build_state_vector() -> dict:
    fields = [
        (CPU_LOAD, "CPU_LOAD", "0=idle, 1=saturated",
         "Current CPU utilization", "StateObserver (psutil.cpu_percent)", None),
        (RAM_AVAIL, "RAM_AVAIL", "0=none free, 1=all free",
         "Fraction of RAM available", "StateObserver (psutil.virtual_memory)", None),
        (LATENCY, "LATENCY", "0=fast, 1=slow/unreachable",
         "ICMP round-trip; saturates to 1.0 if the host is unreachable",
         "StateObserver (single ping)", {"latency_cap_ms": LATENCY_CAP_MS}),
        (UPLOAD, "UPLOAD", "0=idle, 1=saturated",
         "Upload throughput", "StateObserver (psutil.net_io_counters delta)",
         {"upload_cap_bytes_per_sec": UPLOAD_CAP_BYTES_PER_SEC}),
        (CONN_TYPE, "CONN_TYPE", "0=wired, 0.5=wifi/unknown, 1=cellular",
         "Best-effort classification from interface names", "StateObserver",
         {"scores": CONNECTION_TYPE_SCORE}),
        (TIME_SINCE_REKEY, "TIME_SINCE_REKEY", "0=just rekeyed, 1=at or beyond cap",
         "Wall-clock since last key rotation; reset via mark_rekey()",
         "StateObserver", {"rekey_interval_cap_sec": REKEY_INTERVAL_CAP_SEC}),
        (THREAT, "THREAT", "0=none, 1=confirmed anomaly",
         "Combined anomaly score across whichever detection layers the CPU gate "
         "allows; the consumer never needs to know which are active",
         "anomaly_detector.py (Layer 1 today; Layers 2/3 land Weeks 5-8)", None),
    ]

    return {
        "schema_version": 1,
        "generated_by": GENERATOR,
        "source_of_truth": "client/rl_agent/vpn_env.py, client/rl_agent/state_observer.py",
        "dim": STATE_DIM,
        "dtype": "float32",
        "range": [0.0, 1.0],
        "notes": [
            "Index order is frozen. Both VPNEnv and StateObserver clip to [0,1].",
            "The normalization caps below are reasoned first-pass choices, not "
            "calibrated against real traffic. They may be retuned; that changes "
            "the raw-metric mapping, not the interface.",
        ],
        "fields": [
            {
                "index": index,
                "name": name,
                "range": range_desc,
                "semantics": semantics,
                "produced_by": producer,
                **({"normalization": norm} if norm else {}),
            }
            for index, name, range_desc, semantics, producer, norm in fields
        ],
    }


# Scenario names double as the failure message a Rust test prints, so they say
# what the tick sequence is exercising rather than "case 3".
_GATE_SCENARIOS = [
    ("steady state: policy agrees with the algorithm in force", 0, [
        [0.97, 0.02, 0.005, 0.005]] * 4),
    ("single-tick noise spike is absorbed, no handshake", 0, [
        [0.97, 0.02, 0.005, 0.005],
        [0.02, 0.95, 0.02, 0.01],
        [0.97, 0.02, 0.005, 0.005],
        [0.97, 0.02, 0.005, 0.005]]),
    ("sustained escalation is confirmed on the third tick", 0, [
        [0.97, 0.02, 0.005, 0.005],
        [0.02, 0.95, 0.02, 0.01],
        [0.02, 0.95, 0.02, 0.01],
        [0.02, 0.95, 0.02, 0.01],
        [0.02, 0.95, 0.02, 0.01]]),
    ("alternating challengers never accumulate a streak", 0, [
        [0.02, 0.95, 0.02, 0.01],
        [0.02, 0.02, 0.95, 0.01],
        [0.02, 0.95, 0.02, 0.01],
        [0.02, 0.02, 0.95, 0.01]]),
    ("ambiguous state: margin below threshold blocks the change", 0, [
        [0.45, 0.52, 0.02, 0.01]] * 5),
    ("rekey fires immediately, then the cooldown holds", 0, [
        [0.05, 0.05, 0.10, 0.80],
        [0.05, 0.05, 0.10, 0.80],
        [0.05, 0.05, 0.10, 0.80]]),
    ("rekey escalates to the policy's top-ranked KEM", 0, [
        [0.02, 0.10, 0.28, 0.60]]),
    ("rekey never downgrades the algorithm in force", 2, [
        [0.30, 0.05, 0.05, 0.60]]),
]


def build_decision_gate_vectors() -> dict:
    cases = []
    for name, initial, ticks in _GATE_SCENARIOS:
        gate = decision_gate.DecisionGate(in_force=initial)
        steps = []
        for probs in ticks:
            d = gate.update(probs)
            steps.append({
                "probs": [round(float(x), 6) for x in probs],
                "expect": {
                    "in_force": d.in_force,
                    "in_force_name": d.in_force_name,
                    "change_algorithm": bool(d.change_algorithm),
                    "rekey": bool(d.rekey),
                },
            })
        cases.append({"name": name, "initial_in_force": initial, "ticks": steps})

    return {
        "schema_version": 1,
        "generated_by": GENERATOR,
        "source_of_truth": "client/rl_agent/decision_gate.py",
        "purpose": (
            "Verify a port of the decision gate tick-for-tick. Feed each "
            "case's `probs` (softmax over the ONNX logits) to the gate in "
            "order, starting from `initial_in_force`, and assert the three "
            "`expect` fields after every tick. The gate is stateful, so a "
            "case only means anything replayed in sequence from a fresh gate."
        ),
        "constants": {
            "tick_seconds": decision_gate.TICK_SECONDS,
            "confirm_ticks": decision_gate.CONFIRM_TICKS,
            "min_margin": decision_gate.MIN_MARGIN,
            "rekey_cooldown_ticks": decision_gate.REKEY_COOLDOWN_TICKS,
            "rekey_escalates": decision_gate.REKEY_ESCALATES,
        },
        "cases": cases,
    }


def build_manifest(artifacts: list[str]) -> dict:
    """Checksums for everything a consumer copies out of this repo.

    The hazard this closes: `ppo_vpn_agent.onnx`, `policy_test_vectors.json`
    and `algo_registry.json` only agree with each other if they came from the
    same export. Once Member 1 copies them into a Rust crate's assets, nothing
    downstream can tell a matched set from a stale one — and a stale policy
    paired with a current registry fails silently, as wrong algorithm choices
    rather than an error. Hashing them together means the Rust build can
    assert the set is coherent before it ships.
    """
    entries = {}
    for rel in artifacts:
        path = REPO_ROOT / rel
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entries[rel] = {"sha256": digest, "bytes": path.stat().st_size}

    return {
        "schema_version": 1,
        "generated_by": GENERATOR,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": _git_commit(),
        "note": (
            "Verify every artifact against these digests after copying them "
            "into another repo. A mismatch means the set is not the one this "
            "manifest describes; re-run the generator rather than guessing "
            "which file is stale."
        ),
        "artifacts": entries,
    }


def _git_commit() -> dict | None:
    """HEAD, plus whether the tree was dirty when this ran.

    The commit alone would be misleading: these files are normally generated
    *before* the commit that ships them, so HEAD is the previous one. The
    dirty flag says so out loud instead of letting a consumer assume the
    artifacts correspond to that commit's contents.
    """
    def _run(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", "-C", str(REPO_ROOT), *args],
                                 capture_output=True, text=True, timeout=5)
            return out.stdout if out.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    head = _run("rev-parse", "HEAD")
    if head is None:
        return None
    status = _run("status", "--porcelain")
    return {
        "head": head.strip(),
        "tree_dirty": bool(status and status.strip()),
        "note": ("generated from the working tree; when tree_dirty is true the "
                 "artifacts do not correspond to `head` but to the commit made "
                 "immediately after it"),
    }


def write(name: str, payload: dict) -> None:
    CONTRACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = CONTRACTS_DIR / name
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {path.relative_to(REPO_ROOT)}")


def main() -> int:
    write("algo_registry.json", build_algo_registry())
    write("state_vector.json", build_state_vector())
    write("decision_gate_vectors.json", build_decision_gate_vectors())
    rc = export_onnx.main()
    if rc != 0:
        return rc

    # Last, so it hashes the files this run just wrote.
    write("manifest.json", build_manifest([
        "contracts/algo_registry.json",
        "contracts/state_vector.json",
        "contracts/policy_test_vectors.json",
        "contracts/decision_gate_vectors.json",
        "client/rl_agent/models/ppo_vpn_agent.onnx",
    ]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
