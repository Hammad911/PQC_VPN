# Integrating the RL agent — for Members 1 and 2

Everything in this directory is generated. Nothing here needs Python at
runtime: the client loads one ONNX file and reads four JSON files.

```bash
python -m client.rl_agent.export_contracts    # regenerates all of it
```

## Start here

| You are | You need | Ignore |
|---|---|---|
| **Member 1** (`core/rl/`, `core/crypto/`, `core/state/`) | all of it | — |
| **Member 2** (server) | `algo_registry.json` only | the model, the state vector, the gate |

Member 2's entire dependency on this track is: which algorithm identifiers can
arrive in a handshake, and what a rekey means. Both are in `algo_registry.json`.
If the server ever needs to know *why* the client chose an algorithm, something
has gone wrong with the layering — the choice is the client's alone.

## Verify the set is coherent — do this first

`manifest.json` carries a SHA-256 of every artifact plus the git commit they
came from. After copying files into another repo, check them:

```bash
python - <<'EOF'
import hashlib, json, pathlib
m = json.loads(pathlib.Path("contracts/manifest.json").read_text())
for rel, e in m["artifacts"].items():
    got = hashlib.sha256(pathlib.Path(rel).read_bytes()).hexdigest()
    print(("ok  " if got == e["sha256"] else "STALE"), rel)
EOF
```

This exists because the artifacts are only correct *as a set*. A stale
`ppo_vpn_agent.onnx` paired with a current `algo_registry.json` does not throw
— it silently returns action indices that mean something different from what
the registry says. That is the worst kind of integration bug, so it gets a
checksum rather than a convention.

## Step 1 — the state vector

Build `float32[7]`, every element clipped to `[0,1]`, in the index order given
by `state_vector.json`. The order is frozen; read the indices from the file
rather than hardcoding them.

Index 6 (`THREAT`) comes from the anomaly detector, not from device metrics.
Until Layers 2 and 3 land (Weeks 5–8) Layer 1 supplies it, and the consumer
never needs to know which layers are active — it is always one float in
`[0,1]`.

`state_vector.json` also carries the normalization caps `StateObserver` uses
(`latency_cap_ms`, `upload_cap_bytes_per_sec`, `rekey_interval_cap_sec`). Use
those numbers, do not invent your own — the policy was trained against states
normalized this way, and feeding it differently-scaled inputs is the kind of
bug that produces plausible-looking wrong answers.

## Step 2 — run the policy

Load `client/rl_agent/models/ppo_vpn_agent.onnx` with the `ort` crate. One
input `state: float32[batch, 7]`, one output `action_logits: float32[batch, 4]`.
Batch is dynamic; on-device you will always pass 1.

Read the action count from the output shape, not from a constant. If HQC-256
is enabled later the count becomes 5, and that should be a re-export and a
redeploy, not a Rust change.

**Verify the port before building on it.** `policy_test_vectors.json` has 63
states with the exact logits the Python model produces, and a `tolerance`
field (`1e-5`). For each vector, assert the logits match within tolerance and
the argmax matches exactly. Two float32 runtimes will not agree bit-for-bit —
that is what the tolerance is for — but the chosen action must never differ.
Same technique as the Phase 1 crypto port.

## Step 3 — do NOT act on the argmax directly

This is the part that is easy to get wrong, and it is new as of Week 3.

The policy is a pure function of the current state, and the state is noisy:
`psutil.cpu_percent` over a 0.1s window moves several points between
consecutive calls on an idle machine, and the latency probe is worse. Acting
on the raw argmax every tick means acting on that noise — and a change of
algorithm costs a full handshake.

Measured, on 133 client-hours of simulated sessions at ±0.05 observation
noise: **71.0 handshakes/hour acting on the raw argmax, 34.5 through the
gate.** Roughly half of all renegotiations were the client arguing with sensor
noise.

So put the decision gate between the policy and the crypto layer.
`client/rl_agent/decision_gate.py` is the reference implementation — about 40
lines of counters and comparisons, designed to be transliterated rather than
called. Port it and verify against `decision_gate_vectors.json`, which replays
tick-by-tick exactly like the policy vectors do:

```
for case in cases:
    gate = DecisionGate(in_force = case.initial_in_force)
    for tick in case.ticks:
        d = gate.update(softmax(tick.probs))
        assert d.in_force == tick.expect.in_force
        assert d.change_algorithm == tick.expect.change_algorithm
        assert d.rekey == tick.expect.rekey
```

The gate is stateful, so a case only means anything replayed in order from a
fresh gate. Note it takes **softmax'd** probabilities, not raw logits — the
margin test compares probability mass. This is the one place a softmax is
required; argmax alone never needs it.

What the gate costs, stated plainly: it delays a legitimate escalation by a
median of 2 ticks (10s) and p95 of 4 ticks (20s). That is the price of not
thrashing, and `python -m client.rl_agent.decision_gate` re-derives both sides
of the trade so you can re-check the constants rather than trusting them.

## Step 4 — act on the decision

The gate returns three things per tick:

- `in_force` — the algorithm that should be in use. Look up `action_index` in
  `algo_registry.json` for its `liboqs_id`.
- `change_algorithm` — renegotiate to `in_force` with a fresh handshake.
- `rekey` — re-run the handshake at `in_force`. This does not raise or lower
  the algorithm on its own, but `in_force` may already have been raised by
  this same tick's escalation rule — see the open item just below.

`change_algorithm` and `rekey` are never both true in the same tick.

### One open item for the Week 4 checkpoint

`algo_registry.json` says a rekey means "re-run the handshake using the
algorithm currently in force". The reference gate defaults to a slightly
stronger rule: rekey at the **stronger** of {in force, the policy's
top-ranked KEM}, never weaker.

Why: Week 3 measured that on high-security-need states the policy asks for
rekey-now 75% of the time, including on states where the oracle wants
ML-KEM-1024. Under the literal rule those sessions re-handshake at whatever
they already had — often ML-KEM-512 — leaving a mean security shortfall of
0.226 against what the situation calls for. Rekeying at the policy's
top-ranked KEM instead cuts that to 0.022 (oracle floor 0.012), using
information already present in the logits.

Honest caveat: on full simulated sessions the effect is much smaller (mean
shortfall 0.0083 → 0.0076), because the simulator's CPU dynamics rarely keep a
session in the region where it applies. The large number is a statement about
decision quality on those states; the small one is what a session average
looks like in this simulator.

**This needs Member 2's sign-off**, since it means a rekey can arrive at a
different algorithm than the one in force. Set `rekey_escalates = false` for
exactly the Week 2 semantics. Flagged for the Week 4 checkpoint rather than
changed unilaterally.

## What is frozen and what is not

**Frozen** — the state vector's shape, order and semantics; the action space's
shape; the meaning of each action index; the gate's rule.

**Not frozen** — the policy weights. Retraining continues, and a re-export is
a drop-in artifact swap requiring no consumer change. That is the whole reason
the interface shipped in Week 2 rather than Week 4. When weights change, the
`.onnx`, `policy_test_vectors.json` and `manifest.json` all change together;
re-copy all three.

**Not calibrated** — the normalization caps and the reward weights behind the
policy are reasoned first-pass choices, not fit to real traffic. Every claim
about the agent's quality is "correct with respect to a reward function we
wrote in a simulator", not "measured to improve VPN security".
