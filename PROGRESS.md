# RL-PQC-VPN — Progress Report

Status against the project proposal (`RL_PQC_VPN_Proposal.pdf`). This
reflects what is actually implemented and passing tests as of this
submission, organized by phase rather than by date.

## Phase 1 — Core Cryptographic Layer (complete)

Implements proposal section 3.1 (hybrid PQC key exchange) and 3.2's
authentication requirement.

| File | What it does |
|---|---|
| `client/vpn_daemon/hybrid_kem.py` | Hybrid X25519 + ML-KEM-768 key exchange. Shared secret = SHA-256(x25519_secret \|\| mlkem_secret), so the session stays secure even if one of the two algorithms is later broken. |
| `client/vpn_daemon/auth.py` | ML-DSA-65 (FIPS 204) server authentication — server signs its ML-KEM public key, client verifies before accepting it, closing the MitM substitution gap. |
| `client/vpn_daemon/key_store.py` | In-memory key store with explicit `wipe()`/`wipe_all()` that zeroes key bytes before deallocation — no private key ever touches disk. |
| `client/vpn_daemon/algo_registry.py` | The four-action registry (ML-KEM-512/768/1024, rekey-now) the RL agent selects from. HQC-256 is defined but disabled pending NIST finalization, per the proposal's algorithm table. |

**Tests:** `tests/test_phase1.py` — 10/10 passing. Covers ML-KEM-512/768/1024
roundtrips, ML-DSA-65 sign/verify, the full hybrid KEM handshake, and
environment sanity checks (PyTorch MPS backend, psutil, Gymnasium,
Stable-Baselines3).

## Phase 2 — RL Decision Engine (initial implementation)

Implements the MDP formulation from proposal section 7: state space =
device/network conditions, action space = NIST algorithm choice or
rekey, reward = security strength earned vs. computational cost paid.

| File | What it does |
|---|---|
| `client/rl_agent/vpn_env.py` | Custom Gymnasium environment (`VPNEnv`) — the 7-dim state space and 4-action space wired directly to `algo_registry.ACTIVE_ACTIONS`, so the registry stays the single source of truth. Validated against Gymnasium's `check_env`. |
| `client/rl_agent/state_observer.py` | Live state reader — real `psutil` CPU/RAM/traffic readings, an actual ICMP ping for latency, and interface-name heuristics for connection type. This is the deployment-time counterpart to `vpn_env.py`'s simulator. |
| `client/rl_agent/anomaly_detector.py` | Layer 1 of the anomaly pipeline (proposal 3.3) — rolling Z-score baseline over latency/packet-rate/packet-size, always active. |
| `client/rl_agent/train.py` | Trains PPO (Stable-Baselines3) on `VPNEnv`. |
| `client/rl_agent/models/ppo_vpn_agent.zip` | Saved model from an initial training run (50k timesteps). |

**Architecture check against the proposal:** the proposal specifies a PPO
policy with two 64-neuron hidden layers, ~4,932 parameters, under 20KB on
disk. The policy-only network here (the part actually needed for on-device
inference — the value network is training-only) comes out to exactly
**4,932 parameters** (≈19.7KB in fp32), confirming the 64×64 architecture
matches spec. `tests/test_phase2.py::test_ppo_policy_param_count_matches_proposal_spec`
pins this.

**Tests:** `tests/test_phase2.py` — 9/9 passing. Covers the env against the
Gymnasium API contract, reset/step shape and bounds checks, episode
truncation, the state observer's output range, rekey-timer reset, the
anomaly detector flagging an injected outlier, and the trained model
loading and producing a valid action.

**Combined: 19/19 tests passing** (`pytest tests/ -v`).

**Live demo:** `demo.py` chains the two phases together — real device state
(CPU, RAM, a live ping, a live Z-score anomaly check) feeds the trained
agent, which picks an algorithm, which then runs a real, timed hybrid
handshake. Two of the seven state dimensions (connection type, session age)
are staged per scenario and printed as `[STAGED]`, since a short demo can't
actually hop onto cellular or run a 45-minute session — everything else is
live. Run with `python demo.py`.

## Reward-shaping bugs found and fixed while building the demo

Worth documenting because they're the kind of thing that only shows up once
you actually test the trained policy against varied states, not just check
that training runs without crashing:

1. **First reward draft was symmetric** (`1 - |algo.security - need|`).
   Checked the trained policy against 2,000 random states and found it
   picked ML-KEM-512 100% of the time, including at maximum threat — a
   completely degenerate, state-blind policy. Root cause: with a uniform
   state distribution, the average security need lands near 0.5, and 0.7
   (ML-KEM-512's security level) is the closest available constant to that
   average — so the symmetric reward taught the agent to minimize distance
   to the *average* case rather than adapt to the *actual* case. Fixed by
   making the reward asymmetric: heavily penalize under-provisioning
   (`shortfall = max(0, need - algo.security)`, weighted 3x) and only
   lightly penalize over-provisioning (via cost). This matches how security
   actually works — falling short is a real risk, using more than strictly
   necessary just costs efficiency.

2. **`rekey-now`'s reward was on a different scale** than the three
   algorithm-choice actions (max ~2.0 vs. max ~1.0), so it dominated
   whenever threat was merely nonzero, and the agent never learned to pick
   ML-KEM-768 or ML-KEM-1024 at all. Rescaled it to the same range and tied
   it to the proposal's actual stated justification for rekeying (stale
   session keys sitting in RAM — `time_since_rekey`), not threat alone.

**Current state, verified two ways:**
- *Offline, exhaustively*: brute-forced the reward function over a
  5-value grid across all 5 relevant state dimensions (7,776 combinations).
  All four actions are the analytically optimal choice somewhere — ML-KEM-512
  in ~86%, ML-KEM-768 in ~11%, rekey-now in ~1%, ML-KEM-1024 in ~1.3% of the
  grid — so the reward function itself is sound; the rarer actions just have
  narrow optimal regions.
- *On the actual trained policy*: bucketing 4,000 random states by
  computed security need shows a clear, monotonic trend — ML-KEM-512
  dominant at low need, ML-KEM-768 share rising through the middle, rekey-now
  dominant at high need. The ML-KEM-768 boundary is still narrow enough that
  it can flip to rekey-now under realistic background CPU load (30-60%,
  i.e. a normal dev laptop) rather than holding a wide margin — a 50k-step
  run on a small network is not enough to carve out a robust boundary for an
  action whose optimal region is only ~11% of the state space. More
  training steps, reward curriculum, or explicit exploration bonuses for
  under-visited actions are the next thing to try here — not something to
  paper over with a longer run picked because it happened to look better on
  one random seed.

## Important caveat on the RL training run

The 50k-timestep run bundled here is a **smoke run** proving the pipeline
end-to-end (env → PPO → saved model → reloaded and queried) — it is not a
converged, evaluated policy. The proposal's Expected Outcomes call for the
trained agent to be benchmarked against three baselines (classical
WireGuard, static ML-KEM-768, rule-based policy) across defined scenarios.
That evaluation hasn't been run yet and no performance claims are being
made about this model.

The training environment (`VPNEnv`) is also a **simulator** with
hand-designed reward weights and state-transition dynamics — reasonable
first-pass choices, not fit to any measured traffic or threat data. That's
flagged directly in the file's docstring as the first thing to revisit
once real traces are available.

## Fixed along the way

- `torch` was broken in the venv (missing `libtorch_cpu.dylib`, blocking
  2 of the 10 Phase 1 tests) — reinstalled clean.
- `.gitignore` had a comment stating model files under
  `client/rl_agent/models/` should be tracked, but the actual `*.zip` /
  `**/*.zip` rules were still silently excluding them. Added a negation
  rule so the trained model can actually be committed.
- Added `requirements.txt` (wasn't present) with pinned versions from the
  working venv; noted that `liboqs-python` isn't on PyPI and has to be
  built from source per the OQS project's instructions.

## Not yet built (per the proposal's remaining scope)

- Anomaly detection Layers 2 (rule-based signatures, CPU-gated <70%) and
  3 (Isolation Forest, CPU-gated <40%).
- WireGuard PSK injection (`wg set`) tying the hybrid KEM output into an
  actual tunnel.
- FastAPI daemon + WebSocket state pushes (section 3.4).
- Tauri/React desktop application and dashboard.
- Baseline comparison and evaluation (latency overhead, CPU usage,
  security level vs. the three defined baselines).
- Windows support (explicitly phase 2 in the proposal's own timeline).

## Week 1 (Member 3 track, 2026-08-05) — quantifying the decision-boundary weak point

Per `TEAM_TIMELINE_PROPOSAL.md`, Member 3's Week 1 task is to "start
addressing the known weak point" from Section 6 above — not to fix it
yet (that's Week 2's entropy-bonus/longer-training work), but to turn
"we noticed this in the demo" into actual numbers. New script:
`client/rl_agent/analyze_boundary.py` (`python -m
client.rl_agent.analyze_boundary`). It reads the existing trained model
only — no retraining, no changes to `vpn_env.py`.

**Method:** fixed a synthetic state deliberately inside ML-KEM-768's
narrow analytically-optimal region (moderate threat=0.5, cellular
connection, mid session age=0.5, ram_avail=0.7), then swept `CPU_LOAD`
and read the policy's actual action *probabilities* (via
`model.policy.get_distribution(...).distribution.probs`), not just the
argmax — so the margin between the top and runner-up action is a real
number, not a guess.

**Finding 1 — the ML-KEM-768 window is real but thin.** In this
scenario, ML-KEM-768 is the top action only for `cpu ∈ [0.32, 0.46]`
across the sampled grid, flipping away at `cpu=0.48` — so the window is
14 percentage points wide measured to the last winning sample, or 16
measured to the flip point (the sweep steps in 0.02 increments through
this range, so the true edge sits somewhere between). The margin over
the runner-up peaks at just 0.030 (probability mass) around `cpu=0.38`
and shrinks to 0.006 and 0.004 at the two edges of the window. Below `cpu≈0.32` the
policy prefers `rekey-now` instead (with a much more decisive margin —
up to 0.164 at `cpu=0`); above `cpu≈0.48` it drops to ML-KEM-512. A
±0.05 CPU perturbation from ordinary background load is enough to flip
the decision at either edge — consistent with what `demo.py` already
had to work around by staging CPU/RAM per scenario.

**Finding 2 — ML-KEM-1024 is more unreachable than previously
characterized.** `PROGRESS.md`'s existing brute-force reward-grid
analysis found ML-KEM-1024 analytically optimal in ~1.3% of the state
space — a statement about the reward function. Checking the *trained
policy* itself against 4,000 random states: ML-KEM-1024 is never the
argmax choice — **0/4000 (0.00%)** — and its probability mass never
exceeds **0.1576** anywhere in that sample (mean 0.095), while the
other three actions each reach 0.6–0.79 somewhere. The network hasn't
just under-visited this action, it currently has no state where it can
structurally win. This is a materially larger gap than "narrow optimal
region" alone implies, and sharpens what Week 2's exploration bonus
needs to target — it needs to lift ML-KEM-1024's probability ceiling,
not just its visit frequency.

**Finding 3 — 2D (CPU, RAM) sweep.** Mapping the same scenario over a
CPU×RAM grid shows `rekey-now` dominating most of the low-CPU /
low-RAM-available region, and the ML-KEM-768/1024-vs-512 boundary only
opening up once both CPU and available RAM are moderate-to-high. This
means the fragile boundary isn't purely a 1D CPU effect — RAM pressure
shifts it too, which the original demo-staging workaround (CPU/RAM
jitter-checked together) already implicitly accounted for without it
being written down as a number until now.

**Baseline for Week 2/3:** these numbers (14–16-point-wide ML-KEM-768
window, ≤0.030 margin, 0.00% ML-KEM-1024 reachability, 0.1576 probability
ceiling) are the explicit "before" state. `analyze_boundary.py` prints all
four together in a `BASELINE SUMMARY` block at the end of its run, so a
post-retraining re-run can be diffed against this section directly rather
than the numbers having to be recomputed by hand. Week 2's longer training run
+ targeted exploration bonus should be checked against them directly —
e.g. re-running `analyze_boundary.py` after retraining and confirming
the window widens, the margin grows, and ML-KEM-1024 starts winning on
at least some non-trivial slice of state space — rather than relying on
a qualitative "seems better" impression.

## How to reproduce

```bash
source venv/bin/activate
pip install -r requirements.txt   # liboqs-python must be built separately
pytest tests/ -v                  # 19/19
python -m client.rl_agent.train   # retrain PPO from scratch
```
