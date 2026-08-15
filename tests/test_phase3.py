"""
Week 2 regression tests.

Two things these guard that the Phase 1/2 suites do not:

1. The artifacts Members 1 and 2 build against (contracts/*.json, the
   exported ONNX policy) must stay in agreement with the Python source of
   truth. A hand-edited or stale contract file is exactly the kind of thing
   that surfaces as a mysterious integration bug in week 8, so it fails here
   instead.

2. The `curriculum` parameter added to VPNEnv must remain opt-in. Its whole
   justification is that it changes training without touching the frozen
   state/action interface, and that claim needs a test, not a comment.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from client.rl_agent.vpn_env import (  # noqa: E402
    CONN_TYPE, CPU_LOAD, LATENCY, RAM_AVAIL, STATE_DIM, THREAT,
    TIME_SINCE_REKEY, UPLOAD, VPNEnv, action_reward, action_rewards,
    action_rewards_batch, optimal_action, optimal_action_batch,
)
from client.vpn_daemon.algo_registry import ACTIVE_ACTIONS, ALGORITHMS  # noqa: E402

CONTRACTS = REPO_ROOT / "contracts"
ONNX_PATH = REPO_ROOT / "client" / "rl_agent" / "models" / "ppo_vpn_agent.onnx"

EXPECTED_POLICY_PARAMS = 4932

# The first three states VPNEnv(seed=0) produces. Pinned so that any change to
# the default reset distribution — including one accidentally introduced by a
# curriculum change — fails loudly rather than silently retraining against a
# different environment than the one the frozen contract describes.
GOLDEN_RESET_STATES = np.array([
    [0.63696170, 0.26978672, 0.04097353, 0.01652764, 0.81327021, 0.91275555, 0.60663575],
    [0.72949654, 0.54362500, 0.93507242, 0.81585354, 0.00273850, 0.85740429, 0.03358557],
    [0.72965544, 0.17565562, 0.86317891, 0.54146123, 0.29971188, 0.42268723, 0.02831967],
], dtype=np.float32)


def _load(name):
    path = CONTRACTS / name
    if not path.exists():
        pytest.skip(f"{name} not generated — run `python -m client.rl_agent.export_contracts`")
    return json.loads(path.read_text())


# ------------------------------------------------------- curriculum is opt-in

def test_default_env_reset_distribution_is_unchanged():
    env = VPNEnv(seed=0)
    states = [env.reset(seed=0)[0]]
    states += [env.reset()[0] for _ in range(2)]
    np.testing.assert_allclose(np.array(states), GOLDEN_RESET_STATES, rtol=0, atol=1e-7)


def test_curriculum_zero_matches_default():
    """Explicit 0.0 must not consume RNG draws that the default path doesn't."""
    default = VPNEnv(seed=7)
    explicit = VPNEnv(seed=7, curriculum=0.0)
    a = [default.reset(seed=7)[0]] + [default.reset()[0] for _ in range(5)]
    b = [explicit.reset(seed=7)[0]] + [explicit.reset()[0] for _ in range(5)]
    np.testing.assert_array_equal(np.array(a), np.array(b))


def test_curriculum_shifts_states_toward_high_security_need():
    def mean_need(curriculum):
        env = VPNEnv(seed=3, curriculum=curriculum)
        s = np.array([env.reset()[0] for _ in range(4000)])
        return np.clip(0.4 * s[:, THREAT] + 0.3 * s[:, CONN_TYPE]
                       + 0.3 * s[:, TIME_SINCE_REKEY], 0, 1).mean()

    assert mean_need(0.5) > mean_need(0.0) + 0.05


def test_curriculum_preserves_observation_contract():
    env = VPNEnv(seed=1, curriculum=1.0)
    for _ in range(200):
        state, _ = env.reset()
        assert state.shape == (STATE_DIM,)
        assert state.dtype == np.float32
        assert env.observation_space.contains(state)


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_curriculum_rejects_out_of_range(bad):
    with pytest.raises(ValueError):
        VPNEnv(curriculum=bad)


# ------------------------------------------------- reward helpers agree w/ env

def test_action_reward_matches_env_reward():
    env = VPNEnv(seed=0)
    rng = np.random.default_rng(11)
    for state in rng.uniform(0, 1, size=(200, STATE_DIM)).astype(np.float32):
        for action in range(env.action_space.n):
            assert action_reward(state, action) == env._reward(state, action)


def test_optimal_action_is_argmax_of_action_rewards():
    rng = np.random.default_rng(12)
    for state in rng.uniform(0, 1, size=(200, STATE_DIM)).astype(np.float32):
        assert optimal_action(state) == int(np.argmax(action_rewards(state)))


def test_batched_rewards_match_scalar_rewards():
    """The vectorised oracle is a second implementation of the reward algebra,
    used to build evaluation sets over millions of states. If it drifts from
    the scalar version, every metric silently starts grading the agent against
    a reward it was never trained on — so pin them equal."""
    rng = np.random.default_rng(13)
    states = rng.uniform(0, 1, size=(3000, STATE_DIM)).astype(np.float32)

    batched = action_rewards_batch(states)
    scalar = np.array([action_rewards(s) for s in states])
    np.testing.assert_allclose(batched, scalar, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(optimal_action_batch(states),
                                  np.array([optimal_action(s) for s in states]))


# --------------------------------------------------------------- cpu_relax

def test_cpu_relax_defaults_to_original_dynamics():
    """cpu_relax changes the simulator's transition kernel, so its default must
    be a provable no-op — the golden trajectory test above depends on it."""
    plain = VPNEnv(episode_len=30, seed=5)
    explicit = VPNEnv(episode_len=30, seed=5, cpu_relax=0.0)
    a, _ = plain.reset(seed=5)
    b, _ = explicit.reset(seed=5)
    traj_a, traj_b = [a], [b]
    for action in [0, 1, 2, 3] * 5:
        traj_a.append(plain.step(action)[0])
        traj_b.append(explicit.step(action)[0])
    np.testing.assert_array_equal(np.array(traj_a), np.array(traj_b))


def test_cpu_relax_prevents_cpu_saturation():
    """Without it, CPU_LOAD ratchets to 1.0 within ~10 steps and stays there,
    so training never visits the 30-60% band the Week 1 analysis flagged."""
    def mean_cpu(cpu_relax):
        env = VPNEnv(episode_len=100, seed=2, cpu_relax=cpu_relax)
        env.reset(seed=2)
        seen = [env.step(a % 4)[0][CPU_LOAD] for a in range(100)]
        return float(np.mean(seen))

    assert mean_cpu(0.0) > 0.9        # the ratchet, as it stands today
    assert mean_cpu(0.5) < 0.7        # mean-reverting keeps the band reachable


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_cpu_relax_rejects_out_of_range(bad):
    with pytest.raises(ValueError):
        VPNEnv(cpu_relax=bad)


# --------------------------------------------------------- contracts vs source

def test_algo_registry_contract_matches_python_source():
    contract = _load("algo_registry.json")
    assert contract["action_space_size"] == len(ACTIVE_ACTIONS)

    expected = list(ACTIVE_ACTIONS.items())
    assert len(contract["actions"]) == len(expected)

    for action_index, ((registry_key, algo), entry) in enumerate(
        zip(expected, contract["actions"])
    ):
        assert entry["action_index"] == action_index
        assert entry["registry_key"] == registry_key
        assert entry["name"] == algo["name"]
        assert entry["security"] == algo["security"]
        assert entry["cpu_cost"] == algo["cpu_cost"]
        # rekey-now is not a KEM and must not be handed to liboqs as one
        if algo["name"] == "rekey-now":
            assert entry["kind"] == "rekey" and entry["liboqs_id"] is None
        else:
            assert entry["kind"] == "kem" and entry["liboqs_id"] == algo["name"]


def test_algo_registry_contract_lists_disabled_algorithms():
    contract = _load("algo_registry.json")
    inactive = {e["name"] for e in contract["inactive"]}
    assert inactive == {a["name"] for a in ALGORITHMS.values() if not a["active"]}


def test_state_vector_contract_matches_index_constants():
    contract = _load("state_vector.json")
    assert contract["dim"] == STATE_DIM
    assert contract["dtype"] == "float32"

    expected = {
        CPU_LOAD: "CPU_LOAD", RAM_AVAIL: "RAM_AVAIL", LATENCY: "LATENCY",
        UPLOAD: "UPLOAD", CONN_TYPE: "CONN_TYPE",
        TIME_SINCE_REKEY: "TIME_SINCE_REKEY", THREAT: "THREAT",
    }
    assert {f["index"]: f["name"] for f in contract["fields"]} == expected


# ------------------------------------------------------------- ONNX artifact

def test_onnx_model_is_self_contained():
    """A .onnx that depends on a sibling weights file breaks the moment
    Member 1 copies just the model into a Rust crate's assets."""
    onnx = pytest.importorskip("onnx")
    if not ONNX_PATH.exists():
        pytest.skip("no exported model — run `python -m client.rl_agent.export_onnx`")

    model = onnx.load(str(ONNX_PATH))
    external = [
        init.name for init in model.graph.initializer
        if init.HasField("data_location")
        and init.data_location != onnx.TensorProto.DEFAULT
    ]
    assert not external, f"weights stored outside the .onnx: {external}"
    assert not ONNX_PATH.with_suffix(".onnx.data").exists()


def test_onnx_policy_param_count_matches_proposal_spec():
    onnx = pytest.importorskip("onnx")
    if not ONNX_PATH.exists():
        pytest.skip("no exported model")

    model = onnx.load(str(ONNX_PATH))
    total = sum(onnx.numpy_helper.to_array(i).size for i in model.graph.initializer)
    assert total == EXPECTED_POLICY_PARAMS


def test_onnx_matches_committed_test_vectors():
    """The exact check Member 1's Rust `ort` wrapper should perform."""
    ort = pytest.importorskip("onnxruntime")
    if not ONNX_PATH.exists():
        pytest.skip("no exported model")

    vectors = _load("policy_test_vectors.json")
    states = np.array([v["state"] for v in vectors["vectors"]], dtype=np.float32)
    expected = np.array([v["action_logits"] for v in vectors["vectors"]], dtype=np.float32)

    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    actual = session.run(["action_logits"], {"state": states})[0]

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=vectors["tolerance"])
    np.testing.assert_array_equal(
        actual.argmax(axis=1),
        np.array([v["action_index"] for v in vectors["vectors"]]),
    )


# --------------------------------------------------- promoted policy quality

# Floors for the shipped model, set below what the Week 2 run F_ep4_full_s1
# actually achieved so ordinary seed//eval noise doesn't fail the suite, but
# far above the 50k smoke model so a regression to it cannot pass.
# Achieved -> floor:
#   macro-recall            88.5% -> 75%
#   ML-KEM-1024 recall      86.5% -> 55%     (smoke model: 0.0%)
#   ML-KEM-1024 argmax      0.31% -> [0.10%, 0.85%]   (oracle 0.47%, smoke 0.00%)
#   mean regret            0.0007 -> 0.010   (smoke model: 0.1009)
#   decisiveness             0.99 -> 0.60    (smoke model: 0.487)
WEEK2_FLOORS = {
    "macro_recall": 0.75,
    "mlkem1024_recall": 0.55,
    "mlkem1024_share": (0.0010, 0.0085),
    "regret": 0.010,
    "decisiveness": 0.60,
}

MODEL_ZIP = REPO_ROOT / "client" / "rl_agent" / "models" / "ppo_vpn_agent.zip"


@pytest.fixture(scope="module")
def promoted_metrics():
    if not MODEL_ZIP.exists():
        pytest.skip("no trained model — run `python -m client.rl_agent.train`")
    pytest.importorskip("stable_baselines3")

    from stable_baselines3 import PPO

    from client.rl_agent.evaluate import (
        balanced_states, policy_metrics, sample_states,
    )

    model = PPO.load(MODEL_ZIP, device="cpu")
    uniform = sample_states(8000, seed=4242, curriculum=0.0)
    balanced = balanced_states(per_class=800, seed=4243)
    return policy_metrics(model, uniform, balanced)


def test_all_four_actions_are_reachable(promoted_metrics):
    """The Week 1 defect in one line: ML-KEM-1024 was never selected in any
    state, so the action existed in the space but not in the policy."""
    shares = promoted_metrics["action_share"]
    unreachable = [name for name, s in shares.items() if s == 0.0]
    assert not unreachable, f"actions never selected: {unreachable}"


def test_macro_recall_meets_week2_floor(promoted_metrics):
    assert promoted_metrics["macro_recall"] >= WEEK2_FLOORS["macro_recall"]


def test_mlkem1024_recall_meets_week2_floor(promoted_metrics):
    assert (promoted_metrics["recall"]["ML-KEM-1024"]
            >= WEEK2_FLOORS["mlkem1024_recall"])


def test_mlkem1024_is_neither_starved_nor_over_selected(promoted_metrics):
    """Two-sided on purpose. The oracle picks ML-KEM-1024 in ~0.42% of states,
    so a policy that picks it constantly is as wrong as one that never does —
    and a one-sided 'nonzero' target would pass such a policy."""
    lo, hi = WEEK2_FLOORS["mlkem1024_share"]
    assert lo <= promoted_metrics["mlkem1024_share"] <= hi


def test_mean_regret_below_week2_ceiling(promoted_metrics):
    assert promoted_metrics["regret"] <= WEEK2_FLOORS["regret"]


def test_policy_is_not_degenerately_flat(promoted_metrics):
    """Guards the cheap way of faking the reachability metrics: a large entropy
    bonus raises every action's probability ceiling by flattening the softmax,
    without improving a single decision."""
    assert promoted_metrics["decisiveness"] >= WEEK2_FLOORS["decisiveness"]


def test_onnx_accepts_dynamic_batch():
    ort = pytest.importorskip("onnxruntime")
    if not ONNX_PATH.exists():
        pytest.skip("no exported model")

    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    for batch in (1, 5, 64):
        out = session.run(
            ["action_logits"],
            {"state": np.zeros((batch, STATE_DIM), dtype=np.float32)},
        )[0]
        assert out.shape == (batch, len(ACTIVE_ACTIONS))
