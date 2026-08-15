# RL State/Action Interface — Freeze Proposal (Member 3)

Proposed for sign-off at the end-of-Week-1 sync (`TEAM_TIMELINE_PROPOSAL.md`,
Section 2, item 2). This interface already exists in code and has been
stable since Phase 2 was built — nothing here is a redesign. The purpose
of this document is to make it an explicit, reviewable contract so
Member 1 can build the Rust-side ONNX consumer (`core/rl/`, Week 7)
against it without waiting on Member 3's training work (Weeks 2–3) to
finish first.

Source of truth for everything below: `client/rl_agent/vpn_env.py`,
`client/rl_agent/state_observer.py`, `client/vpn_daemon/algo_registry.py`.
If this document and the code ever disagree, the code wins and this
document is stale — flag it for correction.

---

## 1. State vector contract

7-dim `float32` vector, every value clipped to `[0.0, 1.0]`. Index order
is fixed and exposed as named constants in `vpn_env.py`
(`CPU_LOAD, RAM_AVAIL, LATENCY, UPLOAD, CONN_TYPE, TIME_SINCE_REKEY, THREAT
= range(7)`); `state_observer.py::StateObserver.read_state()` produces a
vector in this exact order at inference time.

| Idx | Name | Range | Semantics | Produced by |
|---|---|---|---|---|
| 0 | `CPU_LOAD` | 0=idle, 1=saturated | Current CPU utilization | `StateObserver` (`psutil.cpu_percent`) |
| 1 | `RAM_AVAIL` | 0=none free, 1=all free | Fraction of RAM available (not used) | `StateObserver` (`psutil.virtual_memory`) |
| 2 | `LATENCY` | 0=fast, 1=slow/unreachable | ICMP round-trip, normalized against a 300ms cap; saturates to 1.0 if host is unreachable | `StateObserver` (single ping) |
| 3 | `UPLOAD` | 0=idle, 1=saturated | Upload throughput, normalized against a 5MB/s cap | `StateObserver` (`psutil.net_io_counters` delta) |
| 4 | `CONN_TYPE` | 0=wired, 0.5=wifi/unknown, 1=cellular | Best-effort classification from interface names | `StateObserver` |
| 5 | `TIME_SINCE_REKEY` | 0=just rekeyed, 1=≥1hr | Wall-clock since last key rotation, normalized against a 1hr cap | `StateObserver` (resets via `mark_rekey()`) |
| 6 | `THREAT` | 0=none, 1=confirmed anomaly | Combined anomaly score | **Not** `StateObserver` — passed in from the anomaly detection pipeline (`anomaly_detector.py`, Layer 1 today; Layers 2/3 land Weeks 5–8 and feed the same slot) |

**Invariants Member 1 can rely on:**
- Index order will not change.
- dtype is always `float32`; values are always in `[0, 1]` (both
  `VPNEnv` and `StateObserver` clip explicitly).
- Dim 6 (`THREAT`) is the one slot whose *upstream producer* will change
  shape internally (Layer 1 → Layer 1+2 → Layer 1+2+3) over the next two
  months, but its position, dtype, and range in the vector do not change.
  Member 1's consumer never needs to know which anomaly layers are
  currently active — that's fully absorbed by Member 3's combiner before
  it reaches this vector.

**What is not frozen:** the normalization caps (300ms latency, 5MB/s
upload, 1hr rekey interval) are flagged in `state_observer.py`'s own
docstring as reasoned first-pass choices, not calibrated against real
traffic. They may be retuned later. This does not change the interface
(still `[0,1]` float32 at the same index) — only the mapping from raw
metric to normalized value. Not a blocker for freezing the shape.

---

## 2. Action space contract

Sourced dynamically from `algo_registry.ACTIVE_ACTIONS` — this registry
is the single source of truth; `VPNEnv`'s action space is built from it
(`spaces.Discrete(len(_ACTION_TO_ALGO_KEY))`), so the env and registry
cannot drift apart.

Currently 4 active actions, in registry-key order:

| Action idx | Registry key | Name | Security | CPU cost |
|---|---|---|---|---|
| 0 | 0 | ML-KEM-512 | 0.70 | 0.5 |
| 1 | 1 | ML-KEM-768 | 0.85 | 1.0 |
| 2 | 2 | ML-KEM-1024 | 1.00 | 2.0 |
| 3 | 4 | rekey-now | 1.00 | 3.0 |

(Registry key 3, HQC-256, exists but is `active: False` — draft NIST
standard, not finalized — and is excluded from `ACTIVE_ACTIONS`, hence
the gap between keys 2 and 4 above.)

**Discussion point for the freeze meeting (not a unilateral decision):**
if HQC-256 is enabled later, the action count silently becomes 5. The
proposal is that Member 1's Rust consumer read the action count from the
exported ONNX model's output shape (available from Week 4 onward)
instead of hardcoding `4`, so enabling HQC-256 in the registry later
does not require a corresponding Rust code change — only a re-export and
a re-deploy of the model artifact. Needs explicit agreement at the sync,
since it affects how Member 1 writes the `core/rl/` inference wrapper.

---

## 3. What "frozen" means here

- **Frozen:** the *shape and semantics* of the state vector (7 floats,
  this order, this meaning) and the *shape* of the action space (N
  discrete actions, each mapping to an algorithm/rekey decision via the
  registry).
- **Not frozen, and not expected to be:** the trained policy's weights.
  Training continues through Week 3 (longer runs, exploration tuning —
  see `PROGRESS.md`), and the model is re-exported to ONNX in Week 4.
  Member 1 only needs the interface to be stable to start building the
  inference wrapper now; a better-trained model later is a drop-in
  artifact swap, not an interface change.

---

## 4. Forward pointer to Week 4 (ONNX export)

For Member 1's planning: the exported model will accept a `(1, 7)`
`float32` input tensor (batch size 1, matching the state vector above)
and produce output over the action space defined in Section 2 — exact
output shape (raw logits vs. probabilities vs. argmax) to be confirmed
when the export happens in Week 4, alongside a PyTorch-vs-ONNX numerical
parity check before it's handed off.
