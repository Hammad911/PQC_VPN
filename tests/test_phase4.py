"""
Week 3 (Member 3 track): decision-stability contract and the robustness
floors the Week 3 verification established.

Two groups of tests here, with different jobs.

The `decision_gate` group pins behaviour that Member 1's Rust port has to
reproduce exactly. These are cheap, deterministic, and they are the ones that
must never be relaxed — the generated `contracts/decision_gate_vectors.json`
is only worth shipping if the Python it was generated from is itself pinned.

The `week3_floors` group pins measured properties of the trained policy.
Those are set below what was actually achieved so seed noise cannot fail the
suite, and above the pre-Week-2 behaviour so a regression to it cannot pass.
One of them (`high_need_agreement`) pins a KNOWN DEFECT at its current value
rather than at a target — see the comment there.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# decision_gate is stdlib-only, so it imports here rather than inside each
# test. The model-backed tests below still defer their imports: those pull
# in torch/SB3, which the `model` fixture skips on when unavailable.
from client.rl_agent.decision_gate import (  # noqa: E402
    CONFIRM_TICKS, REKEY_COOLDOWN_TICKS, DecisionGate,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = REPO_ROOT / "contracts"
MODEL_PATH = REPO_ROOT / "client" / "rl_agent" / "models" / "ppo_vpn_agent.zip"


# --------------------------------------------------------- decision gate

def test_gate_vectors_replay_against_the_python_source():
    """The shipped vectors must reproduce from the code they document.

    This is the same drift guard `contracts/` already applies to the algorithm
    registry: a hand-edit to the JSON, or a change to decision_gate.py without
    a regenerate, fails here rather than at integration.
    """
    payload = json.loads((CONTRACTS / "decision_gate_vectors.json").read_text())
    assert payload["cases"], "no cases in the shipped vector file"

    for case in payload["cases"]:
        gate = DecisionGate(in_force=case["initial_in_force"])
        for i, tick in enumerate(case["ticks"]):
            d = gate.update(tick["probs"])
            want = tick["expect"]
            assert d.in_force == want["in_force"], f"{case['name']} tick {i}"
            assert d.change_algorithm == want["change_algorithm"], f"{case['name']} tick {i}"
            assert d.rekey == want["rekey"], f"{case['name']} tick {i}"


def test_gate_constants_in_vectors_match_the_module():
    from client.rl_agent import decision_gate as dg

    c = json.loads((CONTRACTS / "decision_gate_vectors.json").read_text())["constants"]
    assert c["confirm_ticks"] == dg.CONFIRM_TICKS
    assert c["min_margin"] == pytest.approx(dg.MIN_MARGIN)
    assert c["rekey_cooldown_ticks"] == dg.REKEY_COOLDOWN_TICKS
    assert c["tick_seconds"] == dg.TICK_SECONDS
    assert c["rekey_escalates"] == dg.REKEY_ESCALATES


def test_single_tick_flip_does_not_trigger_a_handshake():
    """The whole reason the gate exists: one noisy tick must cost nothing."""
    gate = DecisionGate(in_force=0)
    gate.update([0.97, 0.02, 0.005, 0.005])
    d = gate.update([0.02, 0.95, 0.02, 0.01])
    assert not d.change_algorithm
    assert d.in_force == 0


def test_sustained_challenger_is_confirmed_after_confirm_ticks():
    gate = DecisionGate(in_force=0)
    changes = [gate.update([0.02, 0.95, 0.02, 0.01]).change_algorithm
               for _ in range(CONFIRM_TICKS + 1)]
    assert changes == [False] * (CONFIRM_TICKS - 1) + [True, False]
    assert gate.in_force == 1


def test_alternating_challengers_never_confirm():
    """Two challengers trading places is noise, not a decision."""
    gate = DecisionGate(in_force=0)
    for i in range(10):
        probs = [0.02, 0.95, 0.02, 0.01] if i % 2 else [0.02, 0.02, 0.95, 0.01]
        assert not gate.update(probs).change_algorithm
    assert gate.in_force == 0


def test_low_margin_challenger_is_rejected_however_long_it_persists():
    gate = DecisionGate(in_force=0)
    for _ in range(20):
        assert not gate.update([0.45, 0.52, 0.02, 0.01]).change_algorithm
    assert gate.in_force == 0


def test_rekey_fires_immediately_then_respects_cooldown():
    gate = DecisionGate(in_force=1)
    probs = [0.05, 0.10, 0.05, 0.80]
    assert gate.update(probs).rekey
    fired = [gate.update(probs).rekey for _ in range(REKEY_COOLDOWN_TICKS)]
    assert not any(fired[:-1])
    assert fired[-1], "cooldown should expire after REKEY_COOLDOWN_TICKS"


def test_rekey_escalates_to_the_top_ranked_kem():
    gate = DecisionGate(in_force=0, rekey_escalates=True)
    d = gate.update([0.02, 0.10, 0.28, 0.60])
    assert d.rekey and d.in_force == 2


def test_escalation_resets_the_confirm_streak():
    """A streak banked against the pre-escalation incumbent must not carry
    over. Without resetting it, two ticks favouring a competing algorithm
    followed by one post-escalation tick would confirm a change after a
    single tick against the newly-escalated in_force instead of the full
    `confirm_ticks`, including a one-tick downgrade of the algorithm the
    escalation just raised."""
    gate = DecisionGate(in_force=0, rekey_escalates=True)
    challenger = [0.10, 0.85, 0.02, 0.03]  # builds a streak toward action 1
    gate.update(challenger)
    gate.update(challenger)  # streak == CONFIRM_TICKS - 1, uninterrupted

    d = gate.update([0.02, 0.05, 0.28, 0.65])  # rekey escalates in_force to 2
    assert d.rekey and d.in_force == 2

    d = gate.update(challenger)
    assert not d.change_algorithm, "stale streak let a challenger confirm in one tick"
    assert gate.in_force == 2


def test_non_escalating_rekey_also_resets_the_confirm_streak():
    """The fix above only covered the escalating branch. A rekey tick's
    argmax is rekey-now, not a KEM, whether or not it ends up escalating --
    the streak building toward a challenger did not get a consecutive tick in
    its favour either way. Reproduced here with `in_force` already the
    strongest KEM, so `rekey_escalates=True` has nothing to escalate to and
    takes the same code path as `rekey_escalates=False`: without the fix, a
    stale streak survives and confirms one tick later, downgrading `in_force`
    right after the module's own no-downgrade rekey."""
    gate = DecisionGate(in_force=2, rekey_escalates=True)
    challenger = [0.85, 0.02, 0.03, 0.10]  # builds a streak toward action 0
    gate.update(challenger)
    gate.update(challenger)  # streak == CONFIRM_TICKS - 1, uninterrupted

    d = gate.update([0.02, 0.03, 0.05, 0.90])  # rekey fires; nothing stronger to escalate to
    assert d.rekey and not d.change_algorithm and d.in_force == 2

    d = gate.update(challenger)
    assert not d.change_algorithm, "stale streak let a challenger confirm across a rekey"
    assert gate.in_force == 2


def test_rekey_escalation_disabled_still_resets_the_confirm_streak():
    """Same gap, hit via the literal contract option `rekey_escalates=False`
    (one of the two choices `contracts/DECISIONS.md` puts to Member 2) rather
    than the already-strongest special case above."""
    gate = DecisionGate(in_force=1, rekey_escalates=False)
    challenger = [0.02, 0.02, 0.85, 0.11]  # builds a streak toward action 2
    gate.update(challenger)
    gate.update(challenger)

    d = gate.update([0.05, 0.05, 0.05, 0.85])  # rekey at whatever is in force
    assert d.rekey and not d.change_algorithm and d.in_force == 1

    d = gate.update(challenger)
    assert not d.change_algorithm, "stale streak let a challenger confirm across a rekey"
    assert gate.in_force == 1


def test_rekey_never_downgrades_the_algorithm_in_force():
    """Escalating during a handshake you are doing anyway is free; downgrading
    on one noisy tick is a security regression, so it must go through the
    confirm-ticks path like any other change."""
    gate = DecisionGate(in_force=2, rekey_escalates=True)
    d = gate.update([0.30, 0.05, 0.05, 0.60])
    assert d.rekey and d.in_force == 2


def test_rekey_escalation_can_be_disabled_for_week2_semantics():
    gate = DecisionGate(in_force=0, rekey_escalates=False)
    d = gate.update([0.02, 0.10, 0.28, 0.60])
    assert d.rekey and d.in_force == 0


# -------------------------------------------------------------- manifest

def test_manifest_digests_match_the_shipped_artifacts():
    """Catches a stale artifact set — the failure mode that is otherwise
    silent, because a stale policy paired with a current registry produces
    wrong algorithm choices rather than an error."""
    import hashlib

    manifest = json.loads((CONTRACTS / "manifest.json").read_text())
    assert manifest["artifacts"], "manifest lists no artifacts"

    for rel, entry in manifest["artifacts"].items():
        path = REPO_ROOT / rel
        assert path.exists(), f"{rel} is in the manifest but missing on disk"
        blob = path.read_bytes()
        assert hashlib.sha256(blob).hexdigest() == entry["sha256"], (
            f"{rel} does not match its manifest digest — re-run "
            f"`python -m client.rl_agent.export_contracts`")
        assert len(blob) == entry["bytes"]


# ------------------------------------------------------- curriculum knob

def test_curriculum_pressure_defaults_to_the_week2_draw():
    """`spread` must be opt-in: the Week 2 model was trained under `low`, and
    a silent change here would invalidate every number it is pinned against."""
    from client.rl_agent.vpn_env import VPNEnv

    a = VPNEnv(seed=7, curriculum=1.0)
    b = VPNEnv(seed=7, curriculum=1.0, curriculum_pressure="low")
    sa = np.array([a.reset()[0] for _ in range(200)])
    sb = np.array([b.reset()[0] for _ in range(200)])
    np.testing.assert_array_equal(sa, sb)


def test_curriculum_pressure_spread_covers_the_decision_boundary():
    """The point of the mode: `low` puts only ~20% of its high-need resets in
    the pressure band where ML-KEM-1024 gives way to rekey-now, so the agent
    sees almost no counterexample telling it where the boundary is."""
    from client.rl_agent.vpn_env import (VPNEnv, resource_pressure_batch,
                                         security_need_batch)

    def in_band(mode):
        env = VPNEnv(seed=3, curriculum=1.0, curriculum_pressure=mode)
        s = np.array([env.reset()[0] for _ in range(4000)])
        hi = security_need_batch(s) >= 0.90
        p = resource_pressure_batch(s)[hi]
        return float(((p >= 0.35) & (p <= 0.60)).mean())

    assert in_band("spread") > in_band("low") * 1.5


@pytest.mark.parametrize("bad", ["high", "", "LOW", None])
def test_curriculum_pressure_rejects_unknown_modes(bad):
    from client.rl_agent.vpn_env import VPNEnv

    with pytest.raises(ValueError):
        VPNEnv(curriculum_pressure=bad)


# ------------------------------------------------------- week 3 floors

@pytest.fixture(scope="module")
def model():
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import PPO

    if not MODEL_PATH.exists():
        pytest.skip("no trained model")
    return PPO.load(MODEL_PATH, device="cpu")


def test_every_bucket_below_the_high_need_band_picks_the_right_modal_action(model):
    """Week 3's bucket check, as an assertion. The top bucket is excluded and
    tested separately below — it is the one known defect, and folding it in
    here would either fail the suite permanently or force the floor so low it
    stops catching anything."""
    from client.rl_agent.verify_policy import bucket_by_security_need

    buckets = bucket_by_security_need(model, n=8000)
    for b in buckets:
        if b["lo"] >= 0.90:
            continue
        assert b["modal_match"], (
            f"bucket [{b['lo']:.2f},{b['hi']:.2f}) modal action is "
            f"{b['policy_modal']}, oracle says {b['oracle_modal']}")
        assert b["agreement"] >= 0.85


def test_high_need_agreement_does_not_regress_below_measured(model):
    """KNOWN DEFECT, pinned at its measured value, not at a target.

    Above security_need 0.90 the policy asks for rekey-now on ~75% of states,
    including ones where the oracle wants ML-KEM-1024, and scores ~51%
    agreement. Week 3 established this is not a training-coverage problem —
    the curriculum-spread ablation moved it 49.9% -> 52.7%, inside seed noise.
    The cause is in the reward model (see PROGRESS.md Week 3), and fixing it
    changes what the agent optimises, which is a Week 4 checkpoint decision
    rather than a unilateral one.

    So this floor exists to stop it getting *worse* while that is pending. It
    should be raised, not deleted, once the reward question is settled.
    """
    from client.rl_agent.evaluate import high_need_metrics, high_need_states

    states = high_need_states(2000, seed=4244)
    m = high_need_metrics(model, states)
    assert m["high_need_agreement"] >= 0.45


def test_correct_decisions_are_held_with_a_robust_margin(model):
    """'All four actions are reachable' (Week 2) does not imply the decisions
    are stable — reachability is about the argmax, fragility about the gap
    underneath it. Measured: 0.20% of correct decisions sit below the
    threshold."""
    from client.rl_agent.verify_policy import robust_margin

    m = robust_margin(model, n=8000)
    assert m["fragile_share"] <= 0.05
    assert m["macro_recall"] >= 0.75


def test_unjustified_flip_rate_stays_low_under_sensor_noise(model):
    """The property the decision gate is sized against. 'Unjustified' means
    the policy changed its mind while the oracle did not; a flip the oracle
    also makes is correct tracking of a real boundary."""
    from client.rl_agent.verify_policy import jitter_stability

    levels = jitter_stability(model, n=1500, n_perturb=4)
    by_jitter = {round(lv["jitter"], 2): lv for lv in levels}
    assert by_jitter[0.05]["unjustified_rate"] <= 0.01
    assert by_jitter[0.15]["unjustified_rate"] <= 0.02
