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
  6-value grid across all 5 relevant state dimensions (7,776 combinations).
  All four actions are the analytically optimal choice somewhere — ML-KEM-512
  in ~86%, ML-KEM-768 in ~11%, rekey-now in ~1%, ML-KEM-1024 in ~1.3% of the
  grid — so the reward function itself is sound; the rarer actions just have
  narrow optimal regions.

  **Correction (Week 2):** those percentages are properties of the *grid*,
  not of the state space, and the grid overstates the rare actions. A
  product grid places 1/6 of its mass at exactly 1.0 in every dimension,
  which massively over-represents the corner where threat, connection type
  and session age are simultaneously maximal — precisely where ML-KEM-1024
  wins. Measured against the continuous uniform distribution the environment
  actually samples from, the true figures are ML-KEM-512 **90.5%**,
  ML-KEM-768 **8.9%**, ML-KEM-1024 **0.39%**, rekey-now **0.14%**. So
  ML-KEM-1024's optimal region is ~3.4x smaller than previously recorded and
  rekey-now's is ~7x smaller. The grid numbers reproduce exactly
  (86.47 / 11.32 / 1.31 / 0.90) — they were not wrong, they were answering a
  different question. This matters because it is the correct denominator for
  judging whether the agent is under-selecting an action.
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
characterized.** The brute-force reward-grid analysis above found
ML-KEM-1024 analytically optimal in ~1.3% of the grid — a statement about
the reward function, and one the Week 2 correction above revises down to
0.39% of the actual state space. Checking the *trained
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

## Week 2 (Member 3 track, 2026-08-15) — unblocking Members 1 & 2, and retraining

Two workstreams. The assigned task was the retraining; the contracts work
was added because Member 1's Week 7 dependency on the ONNX policy is the
top item in `TEAM_TIMELINE_PROPOSAL.md`'s own risk table, and the
interface — unlike the weights — was already stable enough to ship.

### Shipped for Members 1 and 2 (pulled forward from Week 4)

- **`client/rl_agent/models/ppo_vpn_agent.onnx`** — the policy, exported
  and ready for Member 1's `ort` wrapper. Input `float32[batch,7]`, output
  `float32[batch,4]` raw logits with `argmax` as the chosen action, dynamic
  batch, 4,932 parameters.

  One real bug caught here: torch's exporter writes weights to a sibling
  `.onnx.data` file by default, so the `.onnx` alone is a 2.2 KB graph with
  no weights in it. That loads fine as long as both files sit together and
  fails the moment someone copies just the model into a Rust crate's
  assets. The export now inlines the weights and asserts there are no
  external references left, producing one self-contained 29 KB file.

- **`contracts/`** — `algo_registry.json`, `state_vector.json`, and
  `policy_test_vectors.json`, all *generated* from the Python source of
  truth by `python -m client.rl_agent.export_contracts`. The algorithm
  table was about to exist in four places (Python, the freeze proposal's
  prose, Member 1's Rust core, Member 2's server), three of them
  hand-maintained. `tests/test_phase3.py` fails if the generated files and
  the Python disagree, so drift is caught by CI rather than at integration.

- **PyTorch-vs-ONNX parity check**, run as part of every export and
  currently `max|logit diff| = 9.5e-07` with 31/31 argmax agreement. The 31
  states and their expected logits ship as the test-vector fixture so the
  Rust side can run the same check — the same technique Member 1 is already
  using to validate the Phase 1 crypto port.

- **`rekey-now` semantics pinned.** The action space never said which
  algorithm a rekey handshake uses; `demo.py` was hardcoding ML-KEM-768
  while the registry lists the action with `security: 1.00`. Resolved: a
  rekey re-runs the handshake at the algorithm currently in force —
  rotation and strength stay independent decisions, matching the stated
  reason to rekey (stale keys in RAM, not threat escalation). Member 2's
  Week 6 task is server-side rekey for an existing peer and could not have
  been implemented against an undefined answer. `demo.py` now threads the
  in-force algorithm through and rekeys at it.

### Evaluation harness (`client/rl_agent/evaluate.py`)

New, and it changed the interpretation of the problem. Scoring the policy
by how often it agrees with the single-step reward optimum makes it look
broken — 57% agreement, where a constant "always ML-KEM-512" policy scores
91%. On **episode return**, which is what PPO actually optimises, the same
policy is comfortably ahead of every constant baseline:

| Policy | Mean return (120 episodes x 200 steps) |
|---|---|
| greedy oracle (reference, reads the reward function) | 162.30 ± 1.06 |
| **trained PPO (50k smoke run)** | **155.65 ± 1.22** |
| rule-based baseline | 149.25 ± 1.61 |
| static ML-KEM-768 | 132.39 ± 3.76 |
| always ML-KEM-1024 | 99.57 ± 3.23 |
| always ML-KEM-512 | 98.83 ± 7.40 |

**Success criteria therefore key off return; greedy agreement is a
diagnostic only.** Three of these baselines are the ones the proposal's
Expected Outcomes call for in Week 9, so that milestone now starts from
working code.

What return does *not* excuse, and an earlier draft of this section got
wrong: it is tempting to read the policy's heavy `rekey-now` use (14% of
states against a greedy optimum of 0.16%) as long-run reasoning that
per-step scoring cannot see — rekeying does reset the session timer and
cut threat by 0.5. **The measurement contradicts that.** The greedy
per-step oracle rekeys ~0.1% of the time and still scores 162.30 against
the trained policy's 155.65, so on this environment the myopic policy is
the better one over full episodes. The over-rekeying is a training
artifact, not a strategy. That also says something useful about the
problem: if a myopic oracle is near-optimal, this is close to a
contextual bandit, which is what justifies the short-horizon settings in
the training section below rather than them being a tuning hack.

Two further gaps. ML-KEM-1024 is selected in 0.00% of states, and on
targeted high-need states where it is optimal in 23% of cases the policy
picks `rekey-now` 75% of the time instead. And the margin over a
four-line rule-based baseline is only +4.3% — worth stating plainly ahead
of Week 9 rather than discovering it there.

### The Week 1 boundary metric was measuring the wrong thing

`analyze_boundary.py`'s `BASE_STATE` is documented as sitting "inside
ML-KEM-768's narrow optimal region". It does not. Its `security_need` is
**0.650**, below ML-KEM-512's security of 0.70, so both 512 and 768 have
zero shortfall and 512 wins on cost at *every* CPU value on that sweep.

So Week 1's headline — "ML-KEM-768 is the top action only for
cpu ∈ [0.32, 0.46], peak margin 0.030" — describes a band where the
policy is **wrong**, and the Week 2 target derived from it ("widen the
window, grow the margin") would have been a target for making it more
wrong. Caught before the retraining was scored against it.

Two replacement probes are now swept alongside it, each verified so that
the named algorithm is optimal across the *whole* CPU range, with the
oracle's choice printed next to the policy's:

| probe | `need` | oracle over cpu∈[0,1] | committed policy |
|---|---|---|---|
| `PROBE_768` | 0.805 | ML-KEM-768 everywhere | **0.0% agreement** — picks `rekey-now`/512 |
| `PROBE_1024` | 0.942 | ML-KEM-1024 everywhere | **0.0% agreement** — picks `rekey-now`/768 |

The true "before" is therefore worse than Week 1 recorded. `BASE_STATE`
is kept for continuity but now prints a warning explaining why its window
number should not be used as a target.

### Training changes

- **Curriculum.** `VPNEnv(curriculum=p)` draws a fraction `p` of episode
  resets from a high-security-need / low-resource-pressure mixture
  (Beta(4,1) on threat, connection type and session age; Beta(1,3) on CPU,
  Beta(3,1) on available RAM). At `p=0.3` the share of states where
  ML-KEM-1024 is optimal goes from **0.39% to 7.2%** — an 18x increase in
  learning signal — while 70% of training stays on the true deployment
  distribution so the policy is not warped everywhere else. `p=0.0` is the
  default and short-circuits before touching the RNG, so the default
  environment produces a byte-identical state sequence to the pre-change
  version; verified against the committed file, and pinned by
  `test_phase3.py::test_default_env_reset_distribution_is_unchanged`.
- **Entropy bonus.** `ent_coef` was sitting at SB3's default of 0.0 — there
  was no exploration bonus at all. Now swept over {0, 0.005, 0.01, 0.02}.
- **Seeding bug.** All four vector envs were constructed with the *same*
  seed, so they generated four correlated copies of the same trajectory
  noise instead of independent experience. Now offset per rank.
- **Model selection.** Added `EvalCallback`/checkpointing/tensorboard.
  Evaluation always runs on `curriculum=0.0` even when training uses a
  curriculum, so a model cannot win selection by overfitting the
  oversampled corner. Both the best-by-eval and final models are kept,
  because comparing them is how you spot an early fluke.
- **Unchanged on purpose:** `net_arch=[64,64]`, `STATE_DIM`, and the active
  action count — the three things the 4,932-parameter pin depends on. The
  reward function and the registry's security/cost values were also left
  alone, so any improvement below is attributable to training, not to
  moving the target.

### What actually fixed it — an ablation, not a bundle

Two measurements reframed the problem before any of this was run:

- **The curriculum washes out.** At `episode_len=200` it lifts the share of
  ML-KEM-1024-optimal *reset* states from 0.39% to 7.2%, but only 0.57% of
  *visited* states — the episode dynamics erase the bias within a few steps.
- **`CPU_LOAD` is a one-way ratchet.** Every action adds load and nothing
  removes it, against zero-mean drift, so CPU saturates at 1.0 by about step
  10 and stays there: mean 0.994 over an episode, 96.5% of steps above 0.95.
  The 30-60% band Week 1 found fragile is a band the agent is almost never
  trained in. `cpu_relax` mean-reverts toward the episode's starting load and
  is off by default.

Seven configurations x 3 seeds x 300k timesteps, each adding one lever:

| config | macro-recall | ML-KEM-1024 recall | regret | seeds passing |
|---|---|---|---|---|
| A status quo | 48.1 ± 1.5 | 0.0% | 0.0812 | 0/3 |
| B + curriculum (`ep=200`) | 48.6 ± 0.3 | 0.0% | 0.0739 | 0/3 |
| C + short horizon (γ=0.9) | 59.8 ± 0.5 | 0.0% | 0.0182 | 0/3 |
| **D + γ=0.5** | **84.0 ± 2.1** | **44.8%** | 0.0006 | **3/3** |
| E + `cpu_relax=0.5` | 85.3 ± 2.6 | 50.8% | 0.0007 | 3/3 |
| **F `ep=4`, full curriculum** | **87.6 ± 0.9** | **55.0%** | 0.0006 | **3/3** |
| G + `ent_coef=0.01` | 86.6 ± 1.1 | 52.7% | 0.0005 | 3/3 |

Read A→B→C→D: **the assigned Week 2 levers are not the ones that work.**
The curriculum alone changes nothing (A ≈ B, and both leave ML-KEM-1024 at
0.0% even at 300k with decisiveness 0.96, so this is a converged policy, not
an undertrained one). Shortening the horizon helps accuracy but still never
reaches ML-KEM-1024. **The discount factor is the decisive lever** — C→D, one
change, takes macro-recall 59.8 → 84.0 and ML-KEM-1024 recall 0% → 44.8%.
And the entropy bonus the timeline named as *the* mechanism makes things
slightly **worse** (F → G), which is consistent with what it actually does:
raising a probability ceiling by flattening the softmax is not the same as
improving a decision.

Why γ matters so much: the reward difference between ML-KEM-1024 and
ML-KEM-768 where the former wins is ~0.075 per step, against a return
standard deviation of ~5.8 over a 200-step episode at γ=0.99. The signal is
buried. Shortening the effective horizon is what makes it visible. This is
also a modelling stance worth stating rather than hiding: it says the
decision is close to a per-tick reactive one, which matches the proposal's
own 5-second control loop and is corroborated by the myopic oracle being
near-optimal on this environment.

### Results — promoted model `F_ep4_full_s1`

| metric | 50k smoke model | **promoted** | reference |
|---|---|---|---|
| Mean episode return | 155.65 | **164.31** | greedy oracle 162.30 |
| …vs best baseline (rule-based) | +4.3% | **+10.1%** | |
| Macro-recall over the 4 actions | 45.4% | **88.5%** | 100% |
| ML-KEM-1024 recall (targeted) | 0.0% | **86.5%** | 100% |
| ML-KEM-1024 argmax share | 0.00% | **0.31%** | oracle 0.47% |
| ML-KEM-1024 probability ceiling | 0.1576 | **0.9999** | 1.0 |
| ML-KEM-768 recall | 22.5% | **94.4%** | 100% |
| Mean per-step regret vs oracle | 0.1009 | **0.0007** | 0 |
| Oracle agreement | 56.1% | **99.3%** | 100% |
| `PROBE_768` agreement over cpu∈[0,1] | 0.0% | **100.0%** | 100% |
| `PROBE_1024` agreement over cpu∈[0,1] | 0.0% | **75.6%** | 100% |

**The trained policy now exceeds the greedy oracle** (164.31 vs 162.30). The
50k model sat 4.1% below it. That is the sequential advantage RL is supposed
to provide finally showing up, and it is the cleanest single answer to "does
the agent beat a rule-based policy" — +10.1%, up from +4.3%.

All acceptance criteria set at the start of the week are met, except the one
that had to be withdrawn because it was measuring the wrong thing (the
`BASE_STATE` window; replaced by the two probe agreements above).

`tests/test_phase3.py` pins macro-recall ≥ 75%, ML-KEM-1024 recall ≥ 55%,
argmax share within [0.10%, 0.85%], regret ≤ 0.010 and decisiveness ≥ 0.60 —
all comfortably below what was achieved so seed noise doesn't fail the suite,
and all far above the smoke model so a regression to it cannot pass. The
share bound is deliberately two-sided: over-selecting ML-KEM-1024 is as wrong
as never selecting it, and a one-sided "nonzero" target is trivially gamed.

### Honest caveats

- The evaluation environment is the same simulator the agent trains in, with
  hand-designed reward weights. Every number above is "correct with respect
  to a reward function we wrote", not "better VPN security". That does not
  change until there are real traffic traces.
- `cpu_relax` and the short horizon change the simulator, not the agent's
  interface. The CPU ratchet looks like a simulator bug rather than a
  modelling choice; it is fixed here only for training runs, and the default
  dynamics are untouched and still pinned by a golden-trajectory test.
- ML-KEM-1024 recall is 86.5% on targeted states but 63% on a uniform sample,
  and `PROBE_1024` agreement is 75.6%, not 100%. The rarest action is
  learned, not mastered.
- The demo's five scenarios are staged. They are now each verified
  analytically optimal before being wired in, which the previous four were
  not — two of them encoded the smoke model's mistakes as expected behaviour.

## How to reproduce

```bash
source venv/bin/activate
pip install -r requirements.txt   # liboqs-python must be built separately
pytest tests/ -v                  # 34/34

python demo.py                            # end-to-end walkthrough
python -m client.rl_agent.evaluate        # return vs baselines and the oracle
python -m client.rl_agent.analyze_boundary  # the four before/after metrics
python -m client.rl_agent.export_contracts  # regenerate contracts/ + the .onnx

python -m client.rl_agent.train             # single run, Week 2 defaults
python -m client.rl_agent.train --sweep --promote   # the Week 2 experiment
```

Note on `pip install`: installing `onnx`/`onnxruntime` pulls numpy 2.x,
which makes stable_baselines3 emit deprecation warnings when loading a
saved model. `requirements.txt` pins numpy back to 1.26.4 and orders the
entries so a plain install ends up on the pin.
