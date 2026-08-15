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
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from client.rl_agent import export_onnx  # noqa: E402
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


def write(name: str, payload: dict) -> None:
    CONTRACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = CONTRACTS_DIR / name
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {path.relative_to(REPO_ROOT)}")


def main() -> int:
    write("algo_registry.json", build_algo_registry())
    write("state_vector.json", build_state_vector())
    return export_onnx.main()


if __name__ == "__main__":
    raise SystemExit(main())
