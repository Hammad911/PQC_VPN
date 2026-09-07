# PQC-VPN

A VPN client that uses reinforcement learning to pick a post-quantum key
encapsulation mechanism (ML-KEM-512/768/1024, NIST FIPS 203) per network
state, with rule- and anomaly-based threat detection layered on top.

This repo is the single shared workspace for the whole team. Start with:

- **[`TEAM_TIMELINE_PROPOSAL.md`](TEAM_TIMELINE_PROPOSAL.md)** — the 3-month
  delivery plan: ownership split, frozen interface contracts, and the
  week-by-week schedule for all three members.
- **[`PROGRESS.md`](PROGRESS.md)** — running log of what's shipped each week.
- **[`contracts/`](contracts/)** — the frozen, checksummed interfaces between
  the Rust core and the RL agent (`algo_registry.json`, `state_vector.json`,
  policy/decision-gate test vectors, `INTEGRATION.md`, `DECISIONS.md`).

## Layout

- **`core/`** — Rust workspace crate: hybrid PQC/classical KEM, auth, key
  storage (`core/src/crypto/`), with `anomaly/`, `protocol/`, `rl/`, and
  `state/` as the integration points for the RL policy and anomaly detection.
- **`desktop/`** — Tauri + React desktop client.
- **`mobile-bindings/`** — uniffi bindings for a future mobile client.
- **`client/rl_agent/`** — the PPO-trained RL policy, its Gymnasium
  simulator (`vpn_env.py`), the decision gate (debounce/hysteresis layer
  between raw policy output and actual handshake decisions), and ONNX export.
- **`client/vpn_daemon/`, `client/api/`** — the Python-side client daemon.
- **`server/`** — WireGuard + PQC server (in progress).
- **`tests/`** — Python test suite (`pytest tests/`).

## Building

Rust workspace (`core/`, `desktop/src-tauri/`):
```
cargo build --workspace
```

Python side (RL agent, decision gate, tests):
```
pip install -r requirements.txt
pytest tests/ -v
```
