# RL-PQC-VPN — Full Technical Briefing

Read this end to end and you should be able to answer almost anything
asked about the current state of the project — what exists, why it was
built that way, what broke and got fixed, and what's still missing.
Everything here describes code that actually runs; nothing is aspirational
unless explicitly marked as such under "Not Built Yet."

---

## 1. Executive Summary

The project is `RL-PQC-VPN`: a VPN that (a) replaces classical key exchange
with a hybrid post-quantum handshake, and (b) uses a reinforcement learning
agent to decide, in real time, which PQC algorithm strength to use and
when to force a key rotation, based on live device/network conditions.

Two of the proposal's phases are implemented and tested:

| Phase | Scope | Status |
|---|---|---|
| **Phase 1 — Core Cryptographic Layer** | Hybrid X25519+ML-KEM key exchange, ML-DSA-65 server auth, secure key storage | Complete, 10/10 tests passing |
| **Phase 2 — RL Decision Engine** | MDP-formalized Gymnasium environment, live state observer, Layer 1 anomaly detection, trained PPO agent | Initial implementation complete, 9/9 tests passing |

**Combined: 19/19 automated tests passing.** There is also a live,
runnable demo (`demo.py`) that chains both phases together: real device
state → agent decision → real timed cryptographic handshake.

Not yet built: WireGuard integration, anomaly detection Layers 2/3,
the FastAPI daemon, the Tauri/React desktop app, and baseline evaluation.
Section 9 below lists these precisely so nothing gets overclaimed.

---

## 2. What the Proposal Committed To (recap)

From `RL_PQC_VPN_Proposal.pdf`:

- **Tunnel:** WireGuard, with a hybrid X25519 + ML-KEM-768 handshake
  injected as the PSK. Server authenticated via ML-DSA-65.
- **RL agent:** PPO, 64×64 hidden layers, ~4,932 parameters, under 20KB
  on disk. Observes a 7-dim state vector (CPU load, RAM available,
  network latency, upload volume, connection type, time since last key
  exchange, threat score). Picks one of 4 actions: ML-KEM-512, ML-KEM-768,
  ML-KEM-1024, or an immediate rekey.
- **Anomaly detection:** 3 layers, gated by available CPU headroom
  (Z-score baseline always on; rule-based signatures below 70% CPU;
  Isolation Forest below 40% CPU).
- **Desktop app:** Tauri + React, FastAPI daemon, WebSocket live updates.
- **Evaluation:** trained agent vs. three baselines (classical WireGuard,
  static ML-KEM-768, rule-based policy).

Everything built so far tracks this spec directly — the action space, the
state dimensions, and the network architecture all match what's on paper
(see Section 8 for the parameter-count verification).

---

## 3. Repository Map

```
PQC_VPN/
├── RL_PQC_VPN_Proposal.pdf        original proposal
├── PROGRESS.md                     technical changelog (what/why, chronological)
├── PROJECT_BRIEFING.md             this document
├── requirements.txt                pinned dependencies
├── demo.py                         live end-to-end demo script
├── contracts/                      generated interface artifacts for M1/M2
│   ├── algo_registry.json
│   ├── state_vector.json
│   └── policy_test_vectors.json
├── client/
│   ├── vpn_daemon/                 Phase 1 — cryptography
│   │   ├── hybrid_kem.py
│   │   ├── auth.py
│   │   ├── key_store.py
│   │   └── algo_registry.py
│   ├── rl_agent/                   Phase 2 — RL decision engine
│   │   ├── vpn_env.py
│   │   ├── state_observer.py
│   │   ├── anomaly_detector.py
│   │   ├── train.py                training + the Week 2 sweep
│   │   ├── evaluate.py             return vs baselines and the oracle
│   │   ├── analyze_boundary.py     decision-boundary diagnostics
│   │   ├── export_onnx.py          policy export + parity check
│   │   ├── export_contracts.py     regenerates contracts/
│   │   └── models/
│   │       ├── ppo_vpn_agent.zip   the trained policy
│   │       └── ppo_vpn_agent.onnx  what Member 1's Rust client loads
│   └── api/                        not yet built (FastAPI daemon)
├── server/                         not yet built
├── dashboard/                      not yet built (Tauri/React)
├── notebooks/                      unused so far
└── tests/
    ├── test_phase1.py
    ├── test_phase2.py
    └── test_phase3.py              contracts, ONNX, curriculum regressions
```

---

## 4. File-by-File Walkthrough

### 4.1 `client/vpn_daemon/hybrid_kem.py` — Hybrid PQC key exchange

The core cryptographic primitive. Implements `HybridKEM` with three methods:

- `generate_client_keypairs(mlkem_algorithm="ML-KEM-768")` — generates an
  X25519 keypair (classical) and an ML-KEM keypair (post-quantum) together.
- `server_encapsulate(...)` — server does classical Diffie-Hellman with the
  client's X25519 public key, and ML-KEM encapsulation against the
  client's ML-KEM public key. Combines both secrets:
  `SHA256(x25519_secret || mlkem_secret)`.
- `client_decapsulate(...)` — client mirrors the process and arrives at
  the identical combined secret.

**Why hybrid, not PQC-only:** if ML-KEM is ever found to be broken (it's a
newer, less battle-tested primitive than X25519), the classical half still
holds the line. If a quantum computer breaks X25519, the PQC half holds
the line. You need both broken simultaneously to compromise the session.

**Recent change:** originally hardcoded to `"ML-KEM-768"`. Added an
`mlkem_algorithm` parameter (default-preserving, so existing tests still
pass unmodified) so the RL agent's chosen algorithm can actually be used
in a real handshake — this is what makes `demo.py` possible.

### 4.2 `client/vpn_daemon/auth.py` — Server authentication

`ServerAuthenticator` using ML-DSA-65 (NIST FIPS 204):

- `generate_server_identity()` — long-term signing keypair, generated once.
- `sign_public_key(mlkem_public_key, dsa_private_key)` — server signs its
  ephemeral ML-KEM public key with its long-term identity key.
- `verify_server_public_key(...)` — client checks the signature before
  trusting the server's ML-KEM key.

**Why this exists:** without it, a man-in-the-middle could hand the
client their own ML-KEM public key and silently decrypt everything. Signing
the ephemeral key with a long-term identity closes that gap — the same
role a TLS certificate plays.

### 4.3 `client/vpn_daemon/key_store.py` — Secure in-memory key handling

`SecureKeyStore`: keys live only in memory (`bytearray`, never written to
disk), with an explicit `wipe()` that overwrites the bytes with zeros
(via `ctypes.memset`) before deletion, and `wipe_all()`.

**Why it matters operationally:** this is also the reason the RL agent's
rekey action exists at all — session keys sitting in RAM are the attack
surface for memory-scraping and side-channel attacks. Rotating keys limits
the blast radius if one is ever extracted.

### 4.4 `client/vpn_daemon/algo_registry.py` — The action space

A dict of 5 algorithms, each with `name`, `standard`, `active` flag,
`security` score (0–1), and `cpu_cost`:

| Key | Algorithm | Active | Security | CPU cost |
|---|---|---|---|---|
| 0 | ML-KEM-512 | ✅ | 0.70 | 0.5 |
| 1 | ML-KEM-768 | ✅ | 0.85 | 1.0 |
| 2 | ML-KEM-1024 | ✅ | 1.00 | 2.0 |
| 3 | HQC-256 | ❌ (disabled) | 0.90 | 1.5 |
| 4 | rekey-now | ✅ | 1.00 | 3.0 |

`ACTIVE_ACTIONS` filters to only the enabled ones — this is the single
source of truth the RL environment's action space is built from, so the
registry and the agent can never drift out of sync. HQC-256 is defined but
disabled because it's still a NIST 2025 draft standard, not finalized.

### 4.5 `client/rl_agent/vpn_env.py` — The MDP, as a Gymnasium environment

`VPNEnv(gym.Env)` — the formal Markov Decision Process from the proposal's
"academic contribution" section:

- **Observation space:** `Box(0, 1, shape=(7,))` — matches the proposal's
  7 state variables exactly (indices exposed as named constants:
  `CPU_LOAD, RAM_AVAIL, LATENCY, UPLOAD, CONN_TYPE, TIME_SINCE_REKEY, THREAT`).
- **Action space:** `Discrete(4)`, built dynamically from
  `algo_registry.ACTIVE_ACTIONS`.
- **This is a simulator, not a live connection.** A real device only gives
  one trajectory — not enough to train a policy from scratch — so
  `reset()` samples a random initial state and `step()` evolves it with a
  random walk plus occasional threat spikes (5% chance per step). The live
  counterpart for actual deployment is `state_observer.py` (below).
- **Reward function** (`_reward`): the part that took two rounds of
  debugging to get right — see Section 6 for the full story. Current
  logic: compute `security_need` from threat/connection-type/session-age,
  then heavily penalize choosing an algorithm *weaker* than what's needed
  (`shortfall`, weighted 3×) and lightly penalize cost when *stronger* than
  needed. The `rekey-now` action is scored separately, tied to the
  proposal's actual stated justification for rekeying — stale keys in RAM
  — not threat alone.

Validated against Gymnasium's own `check_env()` utility, which checks
observation/action space conformance, reset/step contracts, and dtype
correctness.

### 4.6 `client/rl_agent/state_observer.py` — Live state reader

`StateObserver` — the deployment-time counterpart to `VPNEnv`'s simulator.
Reads real signals:

- **CPU load / RAM available:** `psutil`.
- **Latency:** an actual single ICMP ping (subprocess), normalized against
  a 300ms cap; degrades to worst-case (1.0) if the host is unreachable
  rather than crashing.
- **Upload volume:** delta of `psutil.net_io_counters()` between calls,
  normalized against a 5MB/s cap.
- **Connection type:** best-effort classification from active network
  interface names (no cross-platform API gives this reliably without
  elevated privileges).
- **Time since last rekey:** wall-clock since `mark_rekey()` was last
  called, normalized against a 1-hour cap.
- **Threat score:** *not* computed here — passed in from
  `anomaly_detector.py`, keeping the two subsystems decoupled.

All normalization caps are explicitly flagged in the file's own docstring
as reasoned first-pass choices, not calibrated against real traffic data
yet — the first thing to revisit once real traces exist.

### 4.7 `client/rl_agent/anomaly_detector.py` — Anomaly detection, Layer 1 only

`ZScoreBaseline` — rolling mean/std over latency, packet rate, and packet
size (default window: 50 samples). Flags a sample if any metric's Z-score
exceeds 3.0. Returns `not anomalous` until it has enough history to judge
(can't compute a meaningful baseline from 1–2 samples).

This is Layer 1 of the proposal's 3-layer pipeline — the "always active,
all devices" layer. Layers 2 (rule-based signatures, gated below 70% CPU)
and 3 (Isolation Forest, gated below 40% CPU) are **not built yet.**

### 4.8 `client/rl_agent/train.py` — PPO training script

Trains `PPO("MlpPolicy", VPNEnv, ...)` from Stable-Baselines3, with
`policy_kwargs=dict(net_arch=[64, 64])` — a *shared* trunk feeding both
policy and value heads, matching the proposal's "two hidden layers of 64
neurons each" description (SB3's default is actually two *separate*
64×64 networks, which would roughly double the parameter count — this had
to be set explicitly). Trains for 50,000 timesteps across 4 parallel
environments, saves to `client/rl_agent/models/ppo_vpn_agent.zip`.

### 4.9 `client/rl_agent/models/ppo_vpn_agent.zip` — the trained model

Output of the current training run. See Section 6 for what's actually
been verified about its behavior, and Section 9 for what evaluation still
needs to happen before any performance claim can be made about it.

### 4.10 `demo.py` — Live, runnable, end-to-end demo

Chains everything together: loads the trained model, then for each of
four scenarios — reads/stages a state, runs it through the anomaly
detector and the agent, and then executes a **real, timed** hybrid
handshake using whichever algorithm the agent picked. Also demos ML-DSA-65
server auth including a MitM-substitution rejection check. See Section 7
for the full walkthrough and what's genuinely live vs. staged.

### 4.11 `tests/test_phase1.py` — 10 tests

ML-KEM-512/768/1024 roundtrips, ML-DSA-65 sign/verify, the full hybrid KEM
handshake, and environment sanity checks (PyTorch MPS backend, psutil,
Gymnasium import, Stable-Baselines3 smoke test).

### 4.12 `tests/test_phase2.py` — 9 tests

Registry/action-space consistency, `VPNEnv` Gymnasium conformance,
reset/step shape and bounds, episode truncation, `StateObserver` output
range and rekey-timer reset, the anomaly detector actually flagging an
injected outlier, the trained model loading and producing a valid action,
and a hard assertion that the policy network's parameter count equals
exactly 4,932 (pins the architecture to the proposal's spec).

### 4.13 `requirements.txt`

Pinned versions from the working venv. Note: `liboqs-python` isn't on
PyPI — it has to be built from source per the Open Quantum Safe project's
instructions, flagged explicitly in the file so a plain `pip install -r`
doesn't silently fail on it.

### 4.14 `.gitignore` (fixed, not new)

Had a comment claiming model files under `client/rl_agent/models/` were
tracked, but the actual `*.zip` / `**/*.zip` rules were still silently
excluding them — meaning the trained model would never actually get
committed despite the stated intent. Added a negation rule
(`!client/rl_agent/models/**/*.zip`) so it can be.

### 4.15 `PROGRESS.md`

The technical changelog — same facts as this document, organized
chronologically by what was done and why, including the full reward-bug
debugging narrative. Treat this document (`PROJECT_BRIEFING.md`) as the
one to read for the full picture; `PROGRESS.md` as the shorter one to hand
over as a leave-behind.

---

## 5. Testing Summary

```
pytest tests/ -v
```

**45/45 passing.** Breakdown:

- Phase 1 (`test_phase1.py`): 10/10 — cryptographic correctness (every
  algorithm roundtrips to the same shared secret on both sides) plus
  environment/dependency sanity checks.
- Phase 2 (`test_phase2.py`): 9/9 — environment API conformance, live
  state reading, anomaly detection, and the trained model producing valid
  output.
- Phase 3 (`test_phase3.py`): 26/26 — added in Week 2. Three groups:
  the artifacts Members 1 and 2 build against (generated `contracts/*.json`
  must agree with the Python source of truth; the exported ONNX must be
  self-contained, match its committed test vectors, and accept a dynamic
  batch); the `curriculum` and `cpu_relax` training knobs pinned as
  provable no-ops by default, so the environment still produces the exact
  state sequence the frozen contract describes; and floors on the shipped
  policy's quality (macro-recall, per-action recall, regret, decisiveness)
  so a regression to the under-trained model cannot pass.

One real bug was caught and fixed *while verifying this*, not before: the
venv's `torch` install was missing `libtorch_cpu.dylib`, silently failing
2 of the 10 Phase 1 tests. Reinstalling cleanly fixed it — a good example
of why "run the tests and read the output" matters more than "the code
looks right."

---

## 6. The Reward-Design Debugging Story (full detail)

This is worth being able to explain in your own words if asked "does the
RL agent actually work" — it's the most substantive engineering finding
in Phase 2.

### Bug 1 — degenerate policy from a symmetric reward

**First reward draft:** `reward = 1 - |algorithm.security - security_need|`
— i.e., reward for being *close* to what the situation needs, in either
direction.

**How it was caught:** trained a model, then — instead of trusting that
training completing without errors meant it worked — queried the trained
policy against 2,000 random states. It picked **ML-KEM-512 for all 2,000**,
including states with `threat = 1.0` (maximum). A policy that ignores its
input entirely is not a working policy.

**Root cause:** with security-need computed as a blend of threat,
connection type, and session age, its *average* value across a uniform
random state distribution lands near 0.5. Among the three available
algorithms' security scores (0.70, 0.85, 1.00), 0.70 is closest to that
average. A symmetric "distance to need" reward therefore taught the agent
to minimize distance to the *average* case rather than adapt to the
*actual* case per state — technically optimal for the reward as written,
just not the reward that was intended.

**Fix:** made the reward asymmetric. Under-provisioning (picking something
weaker than the situation needs) is a real security risk and gets
penalized 3× as hard as the cost of over-provisioning. This mirrors how
security actually works in practice — falling short is dangerous, using
more than strictly necessary just costs some efficiency.

### Bug 2 — reward scale mismatch starving out two of the four actions

After fixing Bug 1, re-checked the distribution: the policy now used
ML-KEM-512 and occasionally `rekey-now`, but **never** ML-KEM-768 or
ML-KEM-1024.

**Root cause:** the `rekey-now` branch's reward topped out around 2.0,
while the algorithm-choice branch topped out around 1.0. Any time threat
was even slightly nonzero, rekeying was more rewarding than escalating to
a stronger algorithm — so the agent never had a reason to learn when
ML-KEM-768/1024 were the right call.

**Fix:** rescaled the rekey reward to the same order of magnitude, and
tied its justification specifically to the proposal's own stated reason to
rekey (stale session keys sitting in RAM — `time_since_rekey`), rather
than threat alone.

### Verifying the fix, two ways

1. **Offline, exhaustive, no training involved:** brute-forced the reward
   function itself over a 5-value grid across the 5 relevant state
   dimensions (7,776 combinations) and checked which action is
   analytically optimal at each point. Result: all four actions are
   optimal *somewhere* — ML-KEM-512 in ~86% of the grid, ML-KEM-768 in
   ~11%, ML-KEM-1024 in ~1.3%, rekey-now in ~1%. This confirms the reward
   *function* is sound; the rarer actions just have narrow optimal
   regions, which is a training/exploration problem, not a design flaw.

2. **On the actual trained policy:** bucketed 4,000 random states by
   computed security need and checked what the trained model picked in
   each bucket. Clear, monotonic trend: ML-KEM-512 dominant at low need,
   ML-KEM-768 share rising through the middle bands, rekey-now dominant at
   high need.

### The honest caveat that remains

The ML-KEM-768 boundary is narrow enough (only ~11% of the state space is
where it's actually optimal) that a 50,000-timestep run on a small network
hasn't carved out a *wide margin* around it — under realistic background
CPU load (30–60%, i.e. a normal laptop with other things running), the
decision can flip from ML-KEM-768 to `rekey-now` because rekey's reward
doesn't depend on CPU cost at all while the algorithm-choice reward does.
ML-KEM-1024 (optimal in only ~1.3% of the grid) essentially never gets
hit by the current policy at all.

This is *why* `demo.py` stages CPU/RAM inputs explicitly for the scenarios
where the specific tier matters (see Section 7) — not to hide the issue,
but because leaving it to whatever's running on the laptop at demo time
would make the outcome a coin flip, and that's a bad thing to discover live
in front of a supervisor. The fix going forward is more training steps,
an entropy/exploration bonus targeted at under-visited actions, or a
training curriculum that oversamples the narrow high-need region — not
something to paper over by picking a longer run because it happened to
look better on one random seed.

---

## 7. Live Demo Walkthrough (`demo.py`)

Run with:

```bash
source venv/bin/activate
python demo.py
```

**What's genuinely live, computed fresh every run:**
- The Layer 1 Z-score anomaly check (25 calibration samples, then one
  deliberately spiked sample — the detector actually reacts, it's not a
  hardcoded "anomalous=True").
- The agent's forward pass (real neural network inference).
- The full cryptographic handshake — real key generation, encapsulation,
  decapsulation, timed as it happens, with an assertion that both sides
  land on the identical shared secret.
- Network latency (an actual ICMP ping).

**What's staged, and clearly printed as `[STAGED]`:** CPU load, RAM
available, connection type, and session age, per scenario. Two reasons:
(1) a short demo can't actually hop onto a cellular network or run a
45-minute session live, and (2) as explained in Section 6, the ML-KEM-768
decision boundary is sensitive enough to real CPU noise that staging it
makes the outcome reproducible instead of leaving it to chance.

**The four scenarios, verified stable under ±0.15 CPU / ±0.1 RAM jitter
before being wired in:**

1. **Idle laptop, fresh session, trusted wifi** → agent picks
   **ML-KEM-512** (nothing warrants more).
2. **Cellular, 45 minutes into the session, calm traffic** → agent
   escalates to **ML-KEM-768**.
3. **Cellular + anomalous traffic spike** (Layer 1 fires live) → agent
   triggers an **immediate rekey**, then runs a fresh handshake.
4. **Post-rekey, session fresh again** → agent drops back to
   **ML-KEM-512** — showing it's not a one-way ratchet, it actually
   responds to the state resetting.

Also demos ML-DSA-65 server authentication standalone, including a check
that a forged/substituted public key is correctly rejected — the MitM
defense actually working, not just asserted.

---

## 8. Architecture Validation Against the Proposal

Two concrete, checkable claims from the proposal were verified against
the real implementation rather than taken on faith:

- **Action space:** proposal specifies "apply ML-KEM-512, apply ML-KEM-768,
  apply ML-KEM-1024, or trigger an immediate key re-exchange." Confirmed
  exact match via `algo_registry.ACTIVE_ACTIONS` and pinned in
  `test_phase2.py::test_algo_registry_matches_proposal_action_space`.

- **Network size:** proposal specifies "~4,932 parameters, under 20KB on
  disk." The full actor-critic network (policy + value, needed for
  training) is 9,669 parameters — but the value network is training-only;
  it's never needed for on-device inference. The **policy-only** network
  (the part actually shipped) comes out to **exactly 4,932 parameters**
  (448+64+4096+64+256+4), which at 4 bytes/param (fp32) is ≈19.7KB —
  matching "under 20KB" almost to the byte. This is pinned in
  `test_phase2.py::test_ppo_policy_param_count_matches_proposal_spec`, so
  it can't silently drift if the architecture changes later.

---

## 9. Not Built Yet (be precise about this if asked)

- **Anomaly detection Layers 2 and 3** — rule-based signature checks
  (CPU-gated below 70%) and Isolation Forest (CPU-gated below 40%). Only
  Layer 1 (Z-score) exists.
- **WireGuard integration** — no `wg set` PSK injection yet; the hybrid
  handshake output isn't wired into an actual tunnel.
- **FastAPI daemon + WebSocket** — the localhost:8000 API and live state
  push described in proposal section 3.4 don't exist yet.
- **Tauri/React desktop application** — no UI at all yet; everything runs
  from the CLI (`demo.py`, `pytest`).
- **Baseline evaluation** — the proposal's Expected Outcomes call for the
  trained agent to be benchmarked against classical WireGuard, static
  ML-KEM-768, and a rule-based policy across defined scenarios. That
  comparison hasn't been run. The current model is a **smoke-tested**
  pipeline (proven to work end-to-end), not a converged, evaluated policy
  — see Section 6 for exactly what's been verified and what hasn't.
- **Windows support** — explicitly phase 2 in the proposal's own delivery
  timeline; not started.

---

## 10. Anticipated Questions

**"Is the RL agent actually good?"**
Not evaluated yet. What's proven: the pipeline works end-to-end (state →
decision → real crypto), and the decision-making is genuinely
state-dependent with a verified-correct reward design (Section 6). What's
not proven: that it outperforms a static or rule-based baseline — that's
the next milestone, and the proposal itself scopes it as a later
deliverable.

**"Why does the demo stage some inputs?"**
Two honest reasons, both explained in Section 7: some conditions (cellular
network, a 45-minute session) can't be reproduced live in a short demo,
and the ML-KEM-768 decision boundary is currently sensitive to background
CPU load in a way that would make the outcome nondeterministic if left
live. Staging those specific inputs makes the demo reproducible; everything
else (the anomaly detector, the agent's inference, the crypto handshake) is
computed fresh every run.

**"What would you do next?"**
In priority order: (1) more training / better exploration to widen the
ML-KEM-768 and ML-KEM-1024 decision margins, (2) WireGuard PSK injection
to get a real tunnel running end-to-end, (3) Layers 2/3 of anomaly
detection, (4) baseline evaluation now that there's something to compare
against.

**"Did everything just work first try?"**
No — and that's worth saying plainly rather than smoothing over. A broken
`torch` install silently failed 2 of 10 Phase 1 tests until it was
reinstalled. The first RL reward design produced a policy that ignored its
input entirely. The fix for that then created a second, more subtle bug
where two of the four actions were never selected. All three were caught
by actually running and querying the code, not by inspection.

---

## 11. How to Reproduce Everything

```bash
source venv/bin/activate
pip install -r requirements.txt   # liboqs-python must be built separately, see file comment
pytest tests/ -v                  # 19/19
python demo.py                    # live end-to-end walkthrough
python -m client.rl_agent.train   # retrain the PPO agent from scratch
```
