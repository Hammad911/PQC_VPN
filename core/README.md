# `core/` — shared VPN logic (Member 1)

All logic that a desktop **and** a future mobile client would share lives
here. This crate is the reason mobile is a later *option* rather than a
later *rewrite* (`RL-PQC-VPN_3-Month-Delivery-Plan.pdf`, Section 3).

## The one rule

**Nothing in `core/` calls a platform API.** No `sysinfo`, no `psutil`
equivalent, no shelling out to `ping` or `wg`, no OS-specific network or
filesystem calls. Every platform fact enters through a trait that the
`desktop/` crate implements today and an `ios/` / `android/` crate would
implement later.

This is enforced in code review, not just at compile time — "it builds" is
not the same as "it has no platform leakage" (the Week 12 checkpoint checks
the latter).

## Module map

| Module | Contents | Status |
|---|---|---|
| `crypto/` | Hybrid X25519 + ML-KEM (512/768/1024), ML-DSA-65 server auth, the zeroizing key store, the signed handshake transcript | **done** (plan Weeks 2–3) |
| `state/` | `DeviceState` + `TunnelHandle` traits, `StatePipeline` (builds the frozen 7-dim RL state vector), `NormalizationCaps`, mock implementations | **traits done** (plan Week 3) |
| `protocol/` | Client side of the handshake wire protocol (`server/PROTOCOL.md`) | stub — plan Week 6 |
| `rl/` | ONNX inference over the trained policy + the decision gate (`contracts/`) | stub — plan Week 7 |
| `anomaly/` | Layers 1–3 + the CPU-gated combiner (`client/rl_agent/anomaly_detector.py`) | stub — plan Week 8 |

## The trait boundary (`state/`)

```
        desktop/ (or a future mobile crate)
        implements:  DeviceState        TunnelHandle
                          │                   │
        ┌─────────────────┼───────────────────┼─────────────────┐
        │ core/           ▼                   ▼                  │
        │           StatePipeline        core::protocol         │
        │        (normalizes to the      (handshake, Week 6)    │
        │         7-dim [0,1] vector)          │                │
        │                 │                   ▼                 │
        │                 ▼              TunnelConfig ──────────▶│ TunnelHandle
        │            core::rl (Week 7)                           │
        └───────────────────────────────────────────────────────┘
```

- **`DeviceState`** returns *raw* readings (CPU fraction, RAM fraction,
  latency, upload rate, link type). `StatePipeline` — not the platform —
  applies the normalization caps (300 ms latency, 5 MB/s upload, 1 h rekey
  interval), so every port scales inputs the way the policy was trained.
  The rekey timer and the anomaly `threat` score are *not* in the trait:
  the timer lives in `StatePipeline` (`mark_rekey()`), the threat score
  comes from `core::anomaly`.
- **`TunnelHandle`** controls the local WireGuard interface: `bring_up`
  from a `TunnelConfig` (learned from `ServerFinish`), `rotate_psk` on a
  rekey without dropping the tunnel, `tear_down`, `status`.

`cargo build -p core --features mock` exposes `core::state::mock`
(`MockDeviceState`, `MockTunnelHandle`) so the Weeks 5–9 wiring can proceed
before the real platform implementations exist.

## Deliberate deviations from the plan text

- **Pure-Rust crypto, not `liboqs-rust`.** `crypto/` uses the `ml-kem` /
  `ml-dsa` / `x25519-dalek` crates. No C dependency, and it lets
  `server/handshake-server` share this exact crate instead of linking a
  separate C library.
- **The package is named `core`.** This shadows Rust's built-in `core`, so
  other workspace members import it as
  `vpn_core = { path = "../../core", package = "core" }`. Renaming it is on
  the agenda for the Week 4 checkpoint.

## Build & test

```bash
cargo test -p core                       # unit tests (crypto + state)
cargo test --workspace --exclude desktop # core + the server crates
cargo build -p core --features mock      # expose the test doubles
cargo clippy -p core --all-targets
```

## Parity with the Python originals

`crypto/` is a port of `client/vpn_daemon/{hybrid_kem,auth,key_store}.py`.
KEM and signature key generation are not reproducible across
implementations, so those are covered by round-trip tests plus
`server/handshake-vectors.json`. The one deterministic piece —
`combine_shared_secrets` (`SHA-256(x25519 ‖ mlkem)`) — is pinned against
CPython-computed digests in `hybrid_kem.rs`.
