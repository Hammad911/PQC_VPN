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

### 2.1 What `rekey-now` actually does (resolved Week 2)

The action space says "rekey-now" without stating which algorithm the
resulting handshake uses. That gap mattered: `demo.py` was hardcoding
ML-KEM-768 on rekey while the registry lists the action with
`security: 1.00`, and Member 2's Week 6 task (accept a fresh handshake
for an *existing* peer and swap its PSK) cannot be implemented against
an undefined answer.

**Resolved: a rekey re-runs the handshake using the algorithm currently
in force.** It rotates key material without changing strength.

Rationale: "when to rotate" and "how strong" stay independent decisions.
The proposal's stated justification for rekeying is stale session keys
sitting in RAM — a key-lifetime concern, not a threat escalation — so
rekeying should not silently change the negotiated tier in either
direction. It also means the server never has to renegotiate an
algorithm mid-session to honour a rekey.

Consequences for consumers:
- The client tracks the in-force algorithm across ticks; `rekey-now`
  reads it rather than choosing one. `demo.py` now does this.
- Member 2's server swaps the PSK for an existing peer at the same
  algorithm the peer already negotiated.
- The `security: 1.00` on registry key 4 describes the *value of
  rotating*, not a KEM strength. `contracts/algo_registry.json` marks
  this action `"kind": "rekey"` with `"liboqs_id": null` so it cannot be
  mistaken for a KEM identifier.

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

## 4. ONNX export — shipped in Week 2, not Week 4

This section previously said the output shape would be "confirmed when
the export happens in Week 4". The export was pulled forward to Week 2
instead, because Member 1's Week 7 work is the top blocking dependency in
`TEAM_TIMELINE_PROPOSAL.md`'s own risk table and the interface — unlike
the weights — is already stable. Confirmed shapes:

| | |
|---|---|
| Artifact | `client/rl_agent/models/ppo_vpn_agent.onnx` (self-contained, no sidecar weights) |
| Input `state` | `float32[batch, 7]`, values in `[0,1]`, index order per Section 1 |
| Output `action_logits` | `float32[batch, 4]` — **raw logits; `argmax` is the chosen action** |
| Batch dimension | dynamic (1 row on-device, N rows for offline evaluation) |
| Parameters | 4,932 — policy network only; the value network is training-only |
| Opset | 18 |

Raw logits rather than probabilities or a baked-in argmax: `argmax` is
identical over logits and softmax, and leaving them raw lets the consumer
apply its own temperature later without a re-export.

The PyTorch-vs-ONNX parity check runs as part of the export and fails it
on divergence — currently `max|logit diff| = 7.6e-06` over 63 states with
63/63 argmax agreement. Those 63 states, with their expected logits, ship
as `contracts/policy_test_vectors.json` so the Rust `ort` wrapper can be
verified against the same fixture.

**Weights are still not frozen.** Training continues through Week 3 and
the model will be re-exported; that is an artifact swap requiring no
consumer change, which is precisely why shipping the interface early is
safe. See `contracts/README.md` for the consumer-side details.

---

## 5. Decision gating — added Week 3

The sections above define what the policy *is*. Week 3 added the rule for how
a client should *act* on it, because measurement showed the two are not the
same thing.

The policy is a pure function of a state built from noisy sensors, and a
change of algorithm costs a full handshake. Acting on the raw argmax every
tick therefore pays a handshake for sensor noise: measured over 133 simulated
client-hours at +/-0.05 observation noise, **71.0 handshakes/hour acting on
the argmax directly versus 34.5 through the gate**.

| | |
|---|---|
| Reference implementation | `client/rl_agent/decision_gate.py` |
| Test vectors | `contracts/decision_gate_vectors.json` (8 tick sequences) |
| Input | softmax over the ONNX logits (not raw logits — the margin test compares probability mass) |
| Output | `in_force`, `change_algorithm`, `rekey` |
| Rule | a challenger must be argmax for `confirm_ticks = 3` consecutive ticks *and* beat the incumbent by `min_margin = 0.15` |
| Rekey | fires immediately (it is an event, not a mode), rate-limited by `rekey_cooldown_ticks = 12` |
| Cost | legitimate escalations delayed by a median 2 ticks (10s), p95 4 ticks (20s) |

The constants are sized from Week 3's jitter measurement, not chosen by feel:
the per-tick unjustified flip rate is 0.13% overall and 2.9% on
ML-KEM-1024-optimal states, and requiring three consecutive wins pushes the
expected spurious-change interval past a typical session length.

**This is frozen in the same sense as the rest of the document** — the rule and
its constants are the contract; that they were derived from a measurement does
not make them negotiable per-client, or the desktop and mobile ports would
debounce differently.

### Open item for the Week 4 checkpoint

Section 2.1 pinned a rekey as "re-run the handshake at the algorithm currently
in force". Week 3 found that under this rule, high-security-need sessions
re-handshake at whatever they already had — often ML-KEM-512 — leaving a mean
security shortfall of 0.226 against what the situation calls for, because the
policy asks for `rekey-now` on 75% of those states.

Proposed amendment: rekey at the **stronger** of {algorithm in force, the
policy's top-ranked KEM}, never weaker. This cuts the shortfall to 0.022
(oracle floor 0.012) using information already in the logits, so it costs no
interface change and no extra inference. Escalating during a handshake you are
performing anyway is free; downgrading on one noisy tick is not, which is why
it is a `max` and downgrades still go through the confirm-ticks path.

It is implemented behind `rekey_escalates` (default on; `False` gives exactly
the Week 2 semantics) and **needs Member 2's sign-off**, since it means a rekey
can arrive at a different algorithm than the one in force. Raised here rather
than changed unilaterally — see `PROGRESS.md` Week 3 for the underlying
reward-model defect that produces the behaviour.
