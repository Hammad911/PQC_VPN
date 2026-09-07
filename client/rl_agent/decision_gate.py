"""
The rule that turns per-tick policy output into a stable stream of decisions
Member 1's client can act on. Reference implementation, to be ported to Rust.

Why this exists. The ONNX policy is a pure function: 7 floats in, 4 logits
out, argmax is the answer. That is the whole contract, and it is correct —
but it is not sufficient to drive a VPN client, because a *change* of
algorithm costs a full handshake. Running argmax directly on every 5-second
tick means any observation noise that flips the argmax also triggers a
renegotiation, and the state vector is noisy by construction:
`psutil.cpu_percent` over a 0.1s window moves several points between
consecutive calls on an idle machine, and the ICMP latency probe is worse.

Week 3's jitter measurement (`verify_policy.py`) put numbers on it. Under
+/-0.05 noise on CPU and RAM, 0.13% of ticks flip the decision without the
underlying optimum having moved. That looks negligible until it is a rate:
at one tick every 5 seconds it is a spurious handshake roughly every 65
minutes per client on average, and on the high-security states where
ML-KEM-1024 is correct the same rate is 2.9%, i.e. every ~3 minutes. Those
are the sessions least able to afford churn.

So the client needs a debounce, and it belongs here rather than in Member 1's
Rust: it is a property of the policy's behaviour, which is measured on this
side, and pinning it as a shared contract means the client and any future
mobile port debounce identically instead of each inventing a rule.

The design is deliberately dull — integer counters and float comparisons, no
state beyond a few scalars, no allocation. It is meant to be transliterated
into Rust in an afternoon and verified against
`contracts/decision_gate_vectors.json` the same way the ONNX wrapper is
verified against `policy_test_vectors.json`.

Run `python -m client.rl_agent.decision_gate` to see the churn measurement
that sizes the constants below.
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# The action tables come from the registry, which defines the index ordering
# the frozen contract guarantees. Importing them from there rather than
# re-deriving them keeps this module on the stdlib — no numpy, no gymnasium —
# which is the property that makes the Rust port a transliteration.
# ACTION_SECURITY is used only to ensure a rekey never *lowers* the algorithm
# in force.
from client.vpn_daemon.algo_registry import (  # noqa: E402
    ACTION_NAMES, ACTION_SECURITY, KEM_ACTIONS, REKEY_ACTION_IDX,
)

# One tick = one agent loop = 5 seconds (proposal section 3.3).
TICK_SECONDS = 5

# A challenger must win this many consecutive ticks before the in-force
# algorithm changes. Sized from the measured per-tick unjustified flip rate:
# the worst case is 2.9% on ML-KEM-1024-optimal states at +/-0.05 noise, and
# requiring three in a row takes the expected spurious-change interval from
# minutes to well beyond a session's length. Three rather than two because two
# still leaves a visible rate on the high-need states; more than three starts
# to delay *legitimate* escalation, which is a security cost, not just a
# latency one. The simulate() function below reports both sides of that
# trade-off so the choice can be re-checked rather than trusted.
CONFIRM_TICKS = 3

# ...and it must beat the in-force algorithm by this much probability mass.
# The policy is decisive where it is confident (median margin ~1.0 on correct
# decisions), so in practice this only bites in genuinely ambiguous states —
# which is exactly where a handshake should not be spent.
MIN_MARGIN = 0.15

# rekey-now is an event, not a mode: it does not change which algorithm is in
# force, so it is not debounced by CONFIRM_TICKS — delaying a rekey by three
# ticks would defeat its purpose. It is rate-limited instead, because a policy
# that wants to rekey usually wants to rekey again on the next tick too (the
# state barely moves in 5 seconds), and each one is a full handshake.
REKEY_COOLDOWN_TICKS = 12          # 60 seconds

# When the policy asks for a rekey, rekey at the strongest of {algorithm in
# force, the policy's highest-ranked KEM} rather than blindly at the one in
# force.
#
# This is a PROPOSED AMENDMENT to the rekey semantics frozen in Week 2 ("re-run
# the handshake using the algorithm currently in force") and needs Member 2's
# sign-off at the Week 4 checkpoint. Set False to get exactly the Week 2
# behaviour. It is defaulted True because the measurement behind it is large:
#
# Week 3 found the policy asks for rekey-now on 75% of high-security-need
# states, including 2,034 where the oracle wants ML-KEM-1024. Under the literal
# Week 2 rule those sessions rekey at whatever they had — ML-KEM-512 in the
# common case — leaving a mean security shortfall of 0.226 against what the
# situation calls for, on 100% of them. Rekeying at the policy's top-ranked KEM
# instead cuts that to 0.0222, against an oracle floor of 0.0115. The
# information needed is already in the logits Member 1 receives, so this costs
# no interface change and no extra inference.
#
# The `max` is deliberate: escalating during a handshake you are performing
# anyway is free, but *downgrading* on a single noisy tick is a security
# regression, so downgrades still have to go through the confirm-ticks path.
REKEY_ESCALATES = True

# Why the gate decided what it did, for the client's session log. Fixed
# strings, not f-strings: `update` runs once per session per tick, and the
# module's whole claim is that a tick costs a few comparisons and no
# allocation. A Rust port makes these `&'static str`. The varying quantities
# a formatted message would have carried (the margin, the streak position)
# are already in the caller's hands — it passed the probabilities in.
REASON_AGREES = "policy agrees with in-force"
REASON_LOW_MARGIN = "challenger margin below the minimum"
REASON_ON_STREAK = "challenger on streak, not yet confirmed"
REASON_CONFIRMED = "challenger confirmed for the full streak"
REASON_REKEY = "rekey requested and cooldown clear"
REASON_REKEY_ESCALATED = "rekey requested; escalating to the policy's top-ranked KEM"
REASON_REKEY_SUPPRESSED = "rekey suppressed, cooldown still running"


@dataclass
class Decision:
    """What the client should do this tick."""
    in_force: int                  # action index of the algorithm now in force
    change_algorithm: bool         # renegotiate to `in_force` with a handshake
    rekey: bool                    # re-run the handshake at `in_force`
    reason: str

    @property
    def in_force_name(self) -> str:
        return ACTION_NAMES[self.in_force]


@dataclass
class DecisionGate:
    """Debounces raw policy output into stable client actions.

    Construct one per VPN session. `initial` is the algorithm the session
    negotiated at connect time; the gate never returns a decision that
    contradicts it without an explicit change_algorithm.
    """
    in_force: int = 0
    confirm_ticks: int = CONFIRM_TICKS
    min_margin: float = MIN_MARGIN
    rekey_cooldown_ticks: int = REKEY_COOLDOWN_TICKS
    rekey_escalates: bool = REKEY_ESCALATES

    _candidate: int = field(default=-1, init=False)
    _streak: int = field(default=0, init=False)
    _rekey_blocked_for: int = field(default=0, init=False)

    def update(self, probs) -> Decision:
        """Feed one tick of policy output. `probs` is the softmax over the
        ONNX logits — softmax rather than raw logits because the margin test
        is a probability-mass comparison, and the consumer already needs a
        softmax for nothing else, so this is the one place it is required.
        """
        if self._rekey_blocked_for > 0:
            self._rekey_blocked_for -= 1

        top = max(range(len(probs)), key=lambda i: probs[i])

        # rekey: fires immediately, subject only to the cooldown.
        if top == REKEY_ACTION_IDX:
            # A rekey request does not disturb an algorithm change in
            # progress — they are independent decisions, per the registry's
            # rekey semantics ("re-run the handshake at the algorithm
            # currently in force").
            if self._rekey_blocked_for == 0:
                self._rekey_blocked_for = self.rekey_cooldown_ticks
                reason = REASON_REKEY
                if self.rekey_escalates:
                    best_kem = max(KEM_ACTIONS, key=lambda i: probs[i])
                    if ACTION_SECURITY[best_kem] > ACTION_SECURITY[self.in_force]:
                        self.in_force = best_kem
                        reason = REASON_REKEY_ESCALATED
                        # A streak banked against the pre-escalation incumbent
                        # must not carry over: it would let a challenger
                        # confirm against the new (stronger) in_force after
                        # fewer than confirm_ticks ticks, including one that
                        # downgrades it — the one thing this gate promises
                        # never happens on a single noisy tick.
                        self._candidate, self._streak = -1, 0
                return Decision(self.in_force, False, True, reason)
            return Decision(self.in_force, False, False, REASON_REKEY_SUPPRESSED)

        # algorithm choice: needs a streak and a margin.
        if top == self.in_force:
            self._candidate, self._streak = -1, 0
            return Decision(self.in_force, False, False, REASON_AGREES)

        if probs[top] - probs[self.in_force] < self.min_margin:
            self._candidate, self._streak = -1, 0
            return Decision(self.in_force, False, False, REASON_LOW_MARGIN)

        if top == self._candidate:
            self._streak += 1
        else:
            self._candidate, self._streak = top, 1

        if self._streak >= self.confirm_ticks:
            self.in_force = top
            self._candidate, self._streak = -1, 0
            return Decision(self.in_force, True, False, REASON_CONFIRMED)

        return Decision(self.in_force, False, False, REASON_ON_STREAK)


# ------------------------------------------------------- sizing measurement

def simulate(noise: float = 0.05, n_sessions: int = 400, ticks: int = 240,
             seed: int = 21, confirm_ticks: int = CONFIRM_TICKS,
             rekey_escalates: bool = REKEY_ESCALATES, curriculum: float = 0.0,
             model=None):
    """Measure what the gate costs and what it buys, on noisy trajectories.

    Both sides matter and reporting only one would be misleading: the gate
    suppresses spurious handshakes (the win) but also delays legitimate
    escalations by up to `confirm_ticks` (the cost, in seconds spent on a
    weaker algorithm than the situation calls for).

    Sessions are advanced in lockstep so the policy runs as one batched
    forward pass per tick rather than one per session-tick — the same
    measurement, ~200x faster.

    `ticks=240` is a 20-minute session at 5s/tick.
    """
    import numpy as np

    from client.rl_agent.evaluate import policy_probs
    from client.rl_agent.vpn_env import (
        SENSOR_DIMS, VPNEnv, action_rewards_batch, security_need_batch,
        security_shortfall_batch,
    )

    if model is None:
        model = _load_model()
    rng = np.random.default_rng(seed)

    envs = [VPNEnv(episode_len=ticks, seed=seed + s, curriculum=curriculum)
            for s in range(n_sessions)]
    states = np.array([e.reset(seed=seed + s)[0] for s, e in enumerate(envs)])
    gates = [DecisionGate(in_force=0, confirm_ticks=confirm_ticks,
                          rekey_escalates=rekey_escalates)
             for _ in range(n_sessions)]
    raw_in_force = np.zeros(n_sessions, dtype=int)

    # Only the measured dimensions carry observation noise; SENSOR_DIMS names
    # which those are, beside the state indices themselves.
    noise_mask = np.zeros(states.shape[1], dtype=np.float32)
    noise_mask[list(SENSOR_DIMS)] = 1.0

    raw_changes = gated_changes = raw_rekeys = gated_rekeys = 0
    delay_ticks: list[int] = []
    pending_since = [None] * n_sessions
    # In-force algorithm per session per tick, for the shortfall computed in
    # one vectorised pass at the end rather than element-by-element in here.
    gated_hist = np.zeros((ticks, n_sessions), dtype=int)
    literal_hist = np.zeros((ticks, n_sessions), dtype=int)
    need_hist = np.zeros((ticks, n_sessions), dtype=np.float64)

    for t in range(ticks):
        obs = np.clip(
            states + rng.uniform(-noise, noise, size=states.shape).astype(np.float32) * noise_mask,
            0.0, 1.0).astype(np.float32)
        probs = policy_probs(model, obs)
        top = probs.argmax(axis=1)
        # One reward matrix per tick: `optimal_action_batch` is its argmax and
        # recomputes `security_need_batch` inside itself, so calling all three
        # built the same quantities three times.
        rewards = action_rewards_batch(states)
        want = rewards.argmax(axis=1)
        need = security_need_batch(states)
        need_hist[t] = need

        for i in range(n_sessions):
            # ungated: act on argmax directly
            if top[i] == REKEY_ACTION_IDX:
                raw_rekeys += 1
            elif top[i] != raw_in_force[i]:
                raw_changes += 1
                raw_in_force[i] = top[i]

            d = gates[i].update(probs[i])
            gated_changes += d.change_algorithm
            gated_rekeys += d.rekey

            # Delay must be read against the pending state carried in from
            # *before* this tick's update — `gates[i].in_force` above already
            # reflects the change `update()` just made, so checking
            # `pending_since[i]` after also refreshing it against the new
            # in_force would clear it in the same tick it should be read,
            # undercounting the sample by nearly two orders of magnitude.
            if d.change_algorithm and pending_since[i] is not None:
                delay_ticks.append(t - pending_since[i])

            if want[i] != REKEY_ACTION_IDX and want[i] != gates[i].in_force:
                if pending_since[i] is None:
                    pending_since[i] = t
            else:
                pending_since[i] = None

            # security actually delivered this tick, gated vs the literal
            # Week 2 rekey rule (which never escalates during a rekey)
            gated_hist[t, i] = gates[i].in_force
            literal_hist[t, i] = raw_in_force[i]

            # `advance` rather than `step`: identical transition and RNG
            # draws, without scoring a reward this loop does not read.
            states[i] = envs[i].advance(int(top[i]))

    tick_hours = n_sessions * ticks * TICK_SECONDS / 3600
    med_delay = float(np.median(delay_ticks)) if delay_ticks else 0.0
    p95_delay = float(np.percentile(delay_ticks, 95)) if delay_ticks else 0.0
    sg = float(security_shortfall_batch(need_hist, gated_hist).mean())
    sl = float(security_shortfall_batch(need_hist, literal_hist).mean())

    label = "high-need sessions" if curriculum else "deployment distribution"
    print(f"\n--- decision gate, {label}, +/-{noise:.2f} observation noise, "
          f"confirm_ticks={confirm_ticks}, rekey_escalates={rekey_escalates} ---")
    print(f"  {n_sessions} sessions x {ticks} ticks = {tick_hours:.1f} client-hours")
    print(f"  algorithm changes : raw {raw_changes:>6}  gated {gated_changes:>6}  "
          f"({100*(1 - gated_changes/max(raw_changes,1)):.1f}% suppressed)")
    print(f"  rekeys            : raw {raw_rekeys:>6}  gated {gated_rekeys:>6}  "
          f"({100*(1 - gated_rekeys/max(raw_rekeys,1)):.1f}% suppressed)")
    print(f"  handshakes/hour   : raw {(raw_changes+raw_rekeys)/tick_hours:>6.1f}  "
          f"gated {(gated_changes+gated_rekeys)/tick_hours:>6.1f}")
    print(f"  escalation delay  : median {med_delay:.0f} ticks "
          f"({med_delay*TICK_SECONDS:.0f}s), p95 {p95_delay:.0f} ticks "
          f"({p95_delay*TICK_SECONDS:.0f}s)")
    print(f"  mean security shortfall: gated {sg:.4f}  ungated/literal-rekey {sl:.4f}")
    return {
        "noise": noise, "confirm_ticks": confirm_ticks,
        "rekey_escalates": rekey_escalates, "client_hours": tick_hours,
        "raw_changes": raw_changes, "gated_changes": gated_changes,
        "raw_rekeys": raw_rekeys, "gated_rekeys": gated_rekeys,
        "raw_handshakes_per_hour": (raw_changes + raw_rekeys) / tick_hours,
        "gated_handshakes_per_hour": (gated_changes + gated_rekeys) / tick_hours,
        "escalation_delay_median_ticks": med_delay,
        "escalation_delay_p95_ticks": p95_delay,
        "mean_shortfall_gated": sg, "mean_shortfall_ungated": sl,
        "curriculum": curriculum,
    }


def _load_model():
    """Load the promoted policy, with the same missing-model message every
    other entry point prints rather than an SB3 stack trace on a fresh clone."""
    from stable_baselines3 import PPO

    from client.rl_agent.evaluate import MODEL_PATH

    if not MODEL_PATH.exists():
        raise SystemExit(
            f"no trained model at {MODEL_PATH} — "
            f"run `python -m client.rl_agent.train` first"
        )
    return PPO.load(MODEL_PATH, device="cpu")


def main() -> int:
    import json

    # Loaded once and passed down: every simulate() below scores the same
    # promoted policy.
    model = _load_model()
    results = [simulate(noise=n, model=model) for n in (0.02, 0.05, 0.10)]
    print("\n=== confirm_ticks sensitivity at +/-0.05 ===")
    results += [simulate(noise=0.05, confirm_ticks=ct, n_sessions=200, model=model)
                for ct in (1, 2, 4)]
    # The escalation rule only has anything to do on high-security-need
    # sessions, which are ~0.5% of the deployment distribution — so it is
    # ablated there, where the effect is visible, as well as above where it
    # is correctly shown to be near-invisible.
    print("\n=== rekey_escalates ablation ===")
    for curr in (0.0, 1.0):
        for esc in (False, True):
            results.append(simulate(noise=0.05, rekey_escalates=esc,
                                    n_sessions=200, curriculum=curr, model=model))
    out = Path(__file__).resolve().parent / "models" / "decision_gate_sizing.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
