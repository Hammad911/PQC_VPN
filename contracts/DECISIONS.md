# Decisions from the Week 4 checkpoint — resolved

Two amendments to frozen Week 1/2 contracts, proposed and measured on
Member 3's side, brought to the Week 4 checkpoint for Member 2's (and, for
the second, the whole team's) sign-off.

| # | Decision | Resolution |
|---|---|---|
| 1 | `REKEY_ESCALATES` | **Approved as shipped.** Default stays `True`; the server already accepts an equal-or-stronger rekey algorithm (`server/PROTOCOL.md` §4.6, §6). |
| 2 | ML-KEM-1024 dead band | **Option B chosen, retrain deferred** to Member 3's Week 11 tuning pass, after Member 1's Week 7 ONNX integration. Option A stays future work. |

The original write-ups follow unchanged below the resolution notes, so the
measured trade-offs behind each decision stay on record.

---

## Decision 1 — `REKEY_ESCALATES`: should a rekey be allowed to raise the algorithm in force?

> **Resolution (Week 4 checkpoint): approved as shipped.** No code change:
> `DecisionGate(rekey_escalates=True)` stays the default, and
> `contracts/algo_registry.json`'s rekey semantics now state it. The server
> side was already written to the approved rule (`registry.rs` rejects only a
> downgrade). The tighter-margin variant was not requested.

**What Week 2 froze:** `algo_registry.json` defines `rekey-now` as "re-run
the handshake using the algorithm currently in force" — rotation and
strength are independent decisions.

**What Week 3 measured:** on high-security-need states the policy asks for
`rekey-now` 75% of the time, including 2,034 states (of a 6,000-state
sample) where the oracle wants ML-KEM-1024. Under the literal rule those
sessions re-handshake at whatever they already had — usually ML-KEM-512 —
leaving a mean security shortfall of **0.226** against what the situation
calls for, on 100% of them.

**The proposed amendment:** when a rekey fires, rekey at the *stronger* of
{algorithm currently in force, the policy's top-ranked KEM} — never weaker.
The information is already in the logits Member 1 receives; no interface
change, no extra inference.

| what the client rekeys at | mean security shortfall (high-need states) | states left short |
|---|---|---|
| algorithm in force (literal Week 2 rule) | 0.2260 | 100.0% |
| **policy's top-ranked KEM (proposed)** | **0.0222** | 36.5% |
| oracle's choice (floor) | 0.0115 | 20.2% |

Stated with the same honesty the measurement was reported with: on full
simulated sessions (not just the high-need slice) the effect is much
smaller — mean shortfall 0.0083 → 0.0076 — because the simulator's CPU
ratchet rarely keeps a session in the region where this applies. Both
numbers live in `client/rl_agent/models/decision_gate_sizing.json`.

**Implementation status:** shipped as `DecisionGate(rekey_escalates=True)`
(default), reference-implemented and vector-tested
(`contracts/decision_gate_vectors.json`, `tests/test_phase4.py`). Two
correctness bugs found in the escalation path during this week's review have
been fixed independent of this decision (they'd need fixing whichever way
the vote goes): a stale confirm-streak that could let a challenger downgrade
the algorithm one tick after an escalation, and an escalation-delay
measurement bug in `simulate()` that undercounted its own sample by ~97%
(the headline median-2/p95-4-tick numbers in `INTEGRATION.md` held up
against the corrected estimator; they were not silently wrong, just
under-sampled).

**The options that were put to Member 2** (approve as shipped was chosen):
- **Approve as shipped** — no code change, default stays `True`.
- **Reject** — flip the default to `False`, Week 2 semantics stand exactly.
- **Approve with a tighter margin** — the escalation path currently bypasses
  `MIN_MARGIN`/`CONFIRM_TICKS` entirely (deliberate: the handshake is
  happening anyway, so choosing which algorithm to use during it is free).
  If that trade isn't acceptable, the fix is a `min_margin`-gated variant of
  the escalation check, scoped separately since it changes the measured
  numbers above.

---

## Decision 2 — the ML-KEM-1024 dead band: how to fix the reward, not just mitigate it

> **Resolution (Week 4 checkpoint): Option B chosen, retrain deferred.**
> Implementing it means retraining and re-exporting the ONNX policy, and
> Member 1 builds the `ort` wrapper against the current artifact in Week 7. So
> the reward patch plus retrain/re-export is scheduled for **Member 3's Week 11
> tuning pass** (`TEAM_TIMELINE_PROPOSAL.md`), after the Week 8 vertical slice is
> working. Member 1 then swaps in the new `.onnx` + `policy_test_vectors.json` +
> `manifest.json` together, with no interface change. Until then the shipped
> mitigation (Decision 1) covers the practical harm, and
> `test_phase4.py`'s 0.45 high-need floor stays as the regression guard.
> Option A (in-force algorithm as environment state) is recorded as future work.

**Root cause (PROGRESS.md, Week 3):** `action_reward`'s `rekey-now` branch
scores purely on urgency (`0.5*threat + 0.5*time_since_rekey`) and never
references the algorithm actually in force, because **the environment does
not track that anywhere** — not in the 7-dim observed state, and not
internally either. `VPNEnv._evolve_state` mutates `TIME_SINCE_REKEY` and
`THREAT` on a rekey but has no notion of "current algorithm." The reward
function is a pure function of `(state, action)` with no memory, so it
literally cannot charge a rekey for leaving a weak algorithm in place — it
credits the action as if it satisfies security need, when the frozen
semantics say it does nothing of the kind. Ruled out as a training-coverage
problem: the `curriculum_pressure="spread"` ablation moved high-need
agreement 49.9% → 52.7%, inside seed noise (sd 3.6).

Two ways to actually fix this, not just mitigate it (Decision 1's escalation
rule is the mitigation already shipped — it improves the *client's* rekey
choice without touching what the agent is trained to optimize):

**Option A — give the agent (and the reward) memory of the in-force algorithm.**
Track it as environment state during rollouts (`VPNEnv` already tracks
`TIME_SINCE_REKEY` the same way) and pass it into `action_reward` so the
`rekey-now` branch can charge the real shortfall instead of urgency alone.
This is the clean fix, but it is not free: `action_reward`/`action_rewards_batch`
are pure single-state functions used everywhere — the oracle, the bucket
analysis, every baseline in `evaluate.py` — so giving them history changes
what "optimal action" means across the whole verification suite, not just
training. It likely also means retraining and re-exporting the ONNX policy
Member 1 is about to build the Rust `ort` wrapper against, in the same week
that wrapper work is scheduled to start.

**Option B — a contract-preserving reward patch.** Add an explicit penalty
to the `rekey-now` branch keyed on `security_need > 0.90` (the same
threshold the Week 3 bucket table and the `test_phase4.py` floor test
already use), pushing escalation actions to dominate specifically in the
band where the defect lives. No state change, no retrain-breaks-the-frozen-
interface risk. Costs honesty: it doesn't know the real in-force algorithm
either, so it's tuned to assume the worst above the threshold rather than
actually pricing the shortfall — a calibrated band-aid, not the fix.

**Recommendation that was brought to the sync** (adopted, with the retrain timed
as in the resolution above): ship Option B now, if the team
wants a training-side improvement, since Decision 1's escalation rule
already recovers most of the practical harm; save Option A for a deliberate
retrain/re-export cycle instead of one landing in the same week Member 1
starts consuming the current ONNX contract. Whichever is chosen, it is
Member 3 execution, not something needing Member 2 sign-off the way
Decision 1 does — recorded here because it's the item `test_phase4.py`
itself flags as "a Week 4 checkpoint decision rather than a unilateral one."

**Housekeeping done regardless of the outcome:** `train.py`'s
`MIN_HIGH_NEED_AGREEMENT` promotion gate was set to 0.75 against a
comment claiming the Week 2 model scored 0.64 there — it never scored above
0.551 in any recorded run (`models/sweep_results_week3.json`), so the gate
was permanently unreachable and its `pool = eligible or results` fallback
silently disabled the other two promotion gates as well. Recalibrated to
0.45, matching the floor `test_phase4.py` already pins the defect at, so it
rejects an actual regression instead of rejecting every run including the
best one. Provisional like everything else here — raise it once whichever
option above is implemented and measured.
