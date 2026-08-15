# contracts/ — generated interface artifacts

Machine-readable form of the interfaces frozen in
`INTERFACE_FREEZE_PROPOSAL.md`, for Members 1 and 2 to consume as data
rather than re-typing into Rust.

**Everything here is generated. Do not hand-edit.** Regenerate with:

```bash
python -m client.rl_agent.export_contracts
```

`tests/test_phase3.py` fails if these files disagree with the Python
source of truth, so a stale checkout is caught rather than discovered at
integration.

| File | Who needs it | What it is |
|---|---|---|
| `algo_registry.json` | Member 1 (`core/rl/`, `core/crypto/`), Member 2 (server) | The action space. Maps the policy's `action_index` → algorithm name / liboqs identifier / security / CPU cost. |
| `state_vector.json` | Member 1 (`core/state/`) | The 7-dim observation contract: index order, ranges, semantics, and the normalization caps `StateObserver` currently applies. |
| `policy_test_vectors.json` | Member 1 (`core/rl/`) | 31 states with the exact logits the Python model produces, for verifying the Rust `ort` wrapper. |
| `../client/rl_agent/models/ppo_vpn_agent.onnx` | Member 1 (`core/rl/`) | The policy itself. Self-contained single file — no sidecar weights. |

## The model artifact

- Input `state`: `float32[batch, 7]`, values in `[0,1]`, index order per
  `state_vector.json`.
- Output `action_logits`: `float32[batch, 4]`. **`argmax` is the chosen
  action**, indexing into `algo_registry.json`'s `actions[].action_index`.
  Raw logits, not softmax — argmax is identical either way.
- Batch dimension is dynamic: one row on-device, many rows for offline
  evaluation, same file.
- Policy network only (4,932 parameters). The value network is
  training-only and never runs on-device.

Read the action count from the output shape rather than hardcoding `4`.
If HQC-256 is enabled later the count becomes 5, and that should be a
re-export plus redeploy, not a Rust code change.

## Parity check for the Rust side

`policy_test_vectors.json` carries a `tolerance` field (`1e-5`). For each
vector, feed `state` to the ONNX model and assert `action_logits` matches
within tolerance and `action_index` matches exactly. Two float32 runtimes
will not agree bit-for-bit, which is why the tolerance is there; the
chosen action should never differ.

This mirrors how the Phase 1 crypto port is verified — same inputs, same
expected outputs, two independent implementations.

## Stability

The **shapes and semantics** here are frozen. The **weights** are not:
training continues through Week 3 and the model will be re-exported. That
is a drop-in artifact swap and requires no consumer change — which is the
whole reason this was shipped in Week 2 rather than Week 4.

Current weights are from the Week 2 run `F_ep4_full_s1` (300k timesteps),
which agrees with the analytically optimal action on 99.3% of states and
selects all four actions. `PROGRESS.md` documents what has and has not
been established about it — in particular, "optimal" there means optimal
with respect to a hand-designed reward in a simulator, not a measured
security outcome.
