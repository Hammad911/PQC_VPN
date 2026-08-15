"""
Exports the trained PPO policy to ONNX for the Rust client to consume,
and writes the test vectors Member 1 needs to prove their `ort` wrapper
computes the same thing this one does.

Scheduled for Week 4 in TEAM_TIMELINE_PROPOSAL.md; pulled forward to
Week 2 because Member 1's Week 7 task (core/rl/ inference) is the top
blocking dependency in that document's own risk table. The weights
exported here are not final — training continues through Week 3 — but
the *interface* is, so Member 1 can build and test against a real file
now and take a re-exported artifact later without touching their code.

Only the policy path is exported. The value network is training-only and
never runs on-device, which is also why the proposal's "~4,932 parameters,
under 20KB" figure refers to the policy alone.

Run with: python -m client.rl_agent.export_onnx
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from client.rl_agent.vpn_env import STATE_DIM  # noqa: E402
from client.vpn_daemon.algo_registry import ACTIVE_ACTIONS  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_ROOT / "client" / "rl_agent" / "models" / "ppo_vpn_agent.zip"
ONNX_PATH = REPO_ROOT / "client" / "rl_agent" / "models" / "ppo_vpn_agent.onnx"
VECTORS_PATH = REPO_ROOT / "contracts" / "policy_test_vectors.json"

OPSET = 18
# float32 on two different runtimes will not agree bit-for-bit; this is the
# tolerance Member 1's Rust parity test should use too.
PARITY_ATOL = 1e-5
# Pinned by the proposal and by test_phase2.py; asserted again here so a bad
# export can't ship a model of the wrong shape.
EXPECTED_POLICY_PARAMS = 4932

_ACTION_NAMES = [ACTIVE_ACTIONS[k]["name"] for k in ACTIVE_ACTIONS]


class PolicyLogits(nn.Module):
    """The on-device inference path, and nothing else.

    SB3's `policy.forward()` returns a (actions, values, log_probs) tuple and
    samples from a distribution — neither is what we want frozen into an
    artifact. This exposes the deterministic part: observation -> raw action
    logits, where argmax is the chosen action. Raw logits rather than
    softmax probabilities because argmax is identical either way and it
    leaves the consumer free to apply its own temperature if it ever wants to.
    """

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.policy.extract_features(obs)
        latent_pi = self.policy.mlp_extractor.forward_actor(features)
        return self.policy.action_net(latent_pi)


def build_sample_states(n_random: int = 24, seed: int = 7) -> np.ndarray:
    """Test-vector states: random coverage plus the specific corners that
    have actually caused trouble, so a parity break shows up on the cases we
    know are sensitive rather than only on average."""
    rng = np.random.default_rng(seed)
    states = [rng.uniform(0.0, 1.0, size=(n_random, STATE_DIM)).astype(np.float32)]

    # Hand-picked edges: all-zero, all-one, and the ML-KEM-768 boundary the
    # Week 1 analysis found flips under a ~0.05 CPU perturbation.
    edges = [
        np.zeros(STATE_DIM, dtype=np.float32),
        np.ones(STATE_DIM, dtype=np.float32),
    ]
    for cpu in (0.30, 0.32, 0.38, 0.46, 0.48):
        s = np.array([cpu, 0.7, 0.2, 0.2, 1.0, 0.5, 0.5], dtype=np.float32)
        edges.append(s)
    states.append(np.stack(edges))

    # A uniform sample is ~90% ML-KEM-512 states, so on its own it produces a
    # fixture that never exercises the rare actions — precisely the ones most
    # likely to expose an indexing bug in a reimplementation. Add states drawn
    # so each action is the optimal one, to cover every output index.
    from client.rl_agent.evaluate import balanced_states
    states.append(balanced_states(per_class=8, seed=seed))

    # Round HERE, before anything is computed from these states. The fixture
    # serialises states at 6dp for readability; if the logits were computed
    # from unrounded inputs, feeding the file's own `state` back in would
    # reproduce its `action_logits` only to ~7e-5 — outside the stated 1e-5
    # tolerance. Member 1's parity test would then fail against a correct
    # implementation. Rounding up front keeps the fixture self-consistent.
    return np.round(np.concatenate(states), 6).astype(np.float32)


def export(model: PPO, states: np.ndarray) -> np.ndarray:
    wrapper = PolicyLogits(model.policy).eval()
    dummy = torch.zeros(1, STATE_DIM, dtype=torch.float32)

    ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy,
        str(ONNX_PATH),
        input_names=["state"],
        output_names=["action_logits"],
        # Batch stays dynamic so the same artifact serves one-at-a-time
        # on-device inference and batched offline evaluation.
        dynamic_shapes={"obs": {0: torch.export.Dim("batch")}},
        opset_version=OPSET,
    )
    _inline_external_weights()

    with torch.no_grad():
        return wrapper(torch.from_numpy(states)).numpy()


def _inline_external_weights() -> None:
    """Force the weights back into the .onnx itself.

    torch's exporter spills tensors into a sibling `<name>.onnx.data` file by
    default. That works only as long as both files stay side by side, so the
    first person to copy just the .onnx into a Rust crate's assets gets a model
    that loads and then fails on missing weights. Member 1 should be able to
    treat this as one self-contained artifact, so collapse it back.
    """
    model = onnx.load(str(ONNX_PATH))  # resolves the sidecar while it exists
    onnx.save_model(model, str(ONNX_PATH), save_as_external_data=False)

    sidecar = ONNX_PATH.with_suffix(".onnx.data")
    if sidecar.exists():
        sidecar.unlink()

    reloaded = onnx.load(str(ONNX_PATH))
    n_params = sum(
        onnx.numpy_helper.to_array(init).size for init in reloaded.graph.initializer
    )
    external = [
        init.name for init in reloaded.graph.initializer
        if init.HasField("data_location") and init.data_location != onnx.TensorProto.DEFAULT
    ]
    assert not external, f"weights still external after inlining: {external}"
    assert n_params == EXPECTED_POLICY_PARAMS, (
        f"exported policy has {n_params} parameters, expected {EXPECTED_POLICY_PARAMS} "
        "— architecture drifted from the proposal spec"
    )

    print(f"wrote {ONNX_PATH.relative_to(REPO_ROOT)}  "
          f"({ONNX_PATH.stat().st_size / 1024:.1f} KB, {n_params} params, self-contained)")


def check_parity(states: np.ndarray, torch_logits: np.ndarray) -> np.ndarray:
    """A silent export bug would be invisible until it showed up as weird
    behaviour on-device, so fail loudly here instead."""
    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    onnx_logits = session.run(["action_logits"], {"state": states})[0]

    max_diff = float(np.abs(onnx_logits - torch_logits).max())
    torch_actions = torch_logits.argmax(axis=1)
    onnx_actions = onnx_logits.argmax(axis=1)
    agree = int((torch_actions == onnx_actions).sum())

    print(f"parity over {len(states)} states: max|logit diff| = {max_diff:.3e}, "
          f"argmax agreement {agree}/{len(states)}")

    assert max_diff < PARITY_ATOL, f"ONNX/PyTorch logits diverge by {max_diff:.3e}"
    assert agree == len(states), "ONNX and PyTorch disagree on the chosen action"
    return onnx_logits


def write_test_vectors(states: np.ndarray, logits: np.ndarray) -> None:
    """What Member 1's Rust test loads. Same idea as the Phase 1 crypto test
    vectors their Week 2 port is checked against: same input, same expected
    output, across two independent implementations."""
    payload = {
        "schema_version": 1,
        "generated_by": "python -m client.rl_agent.export_onnx",
        "source_of_truth": "client/rl_agent/models/ppo_vpn_agent.onnx",
        "description": (
            "Feed `state` to the exported ONNX model and compare `action_logits` "
            "within `tolerance`; `action_index` must match exactly."
        ),
        "tolerance": PARITY_ATOL,
        "state_dim": STATE_DIM,
        "action_names": _ACTION_NAMES,
        "vectors": [
            {
                "state": [round(float(x), 6) for x in state],
                "action_logits": [float(x) for x in row],
                "action_index": int(row.argmax()),
                "action_name": _ACTION_NAMES[int(row.argmax())],
            }
            for state, row in zip(states, logits)
        ],
    }

    VECTORS_PATH.parent.mkdir(parents=True, exist_ok=True)
    VECTORS_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {VECTORS_PATH.relative_to(REPO_ROOT)}  ({len(payload['vectors'])} vectors)")


def main() -> int:
    if not MODEL_PATH.exists():
        print(f"no trained model at {MODEL_PATH} — run `python -m client.rl_agent.train` first")
        return 1

    model = PPO.load(MODEL_PATH, device="cpu")
    states = build_sample_states()

    torch_logits = export(model, states)
    onnx_logits = check_parity(states, torch_logits)
    write_test_vectors(states, onnx_logits)

    counts = np.bincount(onnx_logits.argmax(axis=1), minlength=len(_ACTION_NAMES))
    print("actions covered by the test vectors: "
          + ", ".join(f"{n}={c}" for n, c in zip(_ACTION_NAMES, counts)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
