"""
Week 3 re-verification (Member 3 track): confirm the retrained policy's
margins are ROBUST, not merely correct-on-average.

Week 2 established that all four actions are reachable and that the policy
beats every baseline on episode return. Neither of those says the decisions
are *stable*. A policy can hit 99.3% oracle agreement while holding most of
those decisions by a hair, so that ordinary sensor noise flips them — which
on a real device means the client renegotiating a handshake every 5-second
tick instead of settling. That failure mode is invisible to accuracy and to
return, and it is the one that actually reaches Member 1's client.

Three checks, all reading the committed model only — no retraining:

  1. bucket-by-security-need distribution — the technique used to validate
     the original reward fix, now applied per `security_need` bucket so a
     failure localises to a band instead of averaging away.
  2. jitter-stability — perturb the state and count decision flips. The
     metric that matters is the UNJUSTIFIED flip rate: the policy changing
     its mind while the oracle does not. A flip the oracle also makes is
     correct behaviour near a real boundary, not fragility, and counting it
     as failure would penalise the policy for being right.
  3. robust margin — of the decisions the policy gets right, how many does
     it hold with enough probability mass to survive noise.

Check 2 also sizes the decision-stability rule that ships to Member 1 in
`contracts/` (see decision_gate.py) — the flip rates here are what set the
hysteresis threshold, so this is not a report-only diagnostic.

Run with: python -m client.rl_agent.verify_policy
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stable_baselines3 import PPO  # noqa: E402

from client.rl_agent.vpn_env import (  # noqa: E402
    CPU_LOAD, RAM_AVAIL, STATE_DIM,
    action_rewards_batch, optimal_action_batch, security_need_batch,
)
from client.vpn_daemon.algo_registry import ACTIVE_ACTIONS  # noqa: E402

MODEL_PATH = Path(__file__).resolve().parent / "models" / "ppo_vpn_agent.zip"
RESULTS_PATH = Path(__file__).resolve().parent / "models" / "week3_verification.json"

ACTION_NAMES = [ACTIVE_ACTIONS[k]["name"] for k in ACTIVE_ACTIONS]
N_ACTIONS = len(ACTION_NAMES)

# The jitter magnitudes the original verification used (PROJECT_BRIEFING.md
# section on the demo scenarios), plus a wider band to find where it breaks.
JITTER_LEVELS = (0.05, 0.10, 0.15)

# A decision held with less probability margin than this is counted "fragile"
# even when it is correct. Not a tuned constant — it is the point below which
# a single float32 rounding difference between the Python and Rust runtimes
# could plausibly reorder the top two logits, which is the property Member 1
# actually depends on.
FRAGILE_MARGIN = 0.10


def policy_probs(model: PPO, states: np.ndarray) -> np.ndarray:
    """Action probabilities for an (N, 7) batch."""
    obs_tensor, _ = model.policy.obs_to_tensor(states.astype(np.float32))
    with torch.no_grad():
        dist = model.policy.get_distribution(obs_tensor)
    return dist.distribution.probs.numpy()


def sample_states(n: int, rng: np.random.Generator) -> np.ndarray:
    """The deployment distribution: uniform over the observation box.

    Deliberately NOT the training curriculum. The curriculum oversamples the
    high-need corner to create learning signal; verifying on it would be
    grading the policy on its own study notes.
    """
    return rng.uniform(0.0, 1.0, size=(n, STATE_DIM)).astype(np.float32)


# ------------------------------------------------ check 1: need buckets

def bucket_by_security_need(model: PPO, n: int = 40_000, n_buckets: int = 10,
                            seed: int = 11):
    """Policy vs oracle action distribution, bucketed by security_need.

    Reported per bucket rather than pooled because the pooled number is
    dominated by the low-need buckets: ~76% of a uniform sample has
    need <= 0.7, where ML-KEM-512 is trivially correct. A policy that is
    perfect there and useless above 0.85 still scores >90% pooled.
    """
    rng = np.random.default_rng(seed)
    states = sample_states(n, rng)
    need = security_need_batch(states)
    oracle = optimal_action_batch(states)
    chosen = policy_probs(model, states).argmax(axis=1)
    rewards = action_rewards_batch(states)
    regret = rewards[np.arange(n), oracle] - rewards[np.arange(n), chosen]

    edges = np.linspace(0.0, 1.0, n_buckets + 1)
    idx = np.clip(np.digitize(need, edges[1:-1]), 0, n_buckets - 1)

    print("\n--- check 1: action distribution bucketed by security_need ---")
    print("  'modal' = most-chosen action in the bucket. The policy's modal")
    print("  action must match the oracle's, and per-bucket regret must stay")
    print("  small, or the pooled average is hiding a dead band.")
    header = (f"  {'need bucket':>13} | {'n':>6} | {'policy modal':>13} | "
              f"{'oracle modal':>13} | {'agree':>6} | {'regret':>8}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    buckets = []
    for b in range(n_buckets):
        m = idx == b
        count = int(m.sum())
        if count == 0:
            continue
        p_modal = int(np.bincount(chosen[m], minlength=N_ACTIONS).argmax())
        o_modal = int(np.bincount(oracle[m], minlength=N_ACTIONS).argmax())
        agree = float((chosen[m] == oracle[m]).mean())
        breg = float(regret[m].mean())
        ok = "ok " if p_modal == o_modal else "MISS"
        buckets.append({
            "lo": float(edges[b]), "hi": float(edges[b + 1]), "n": count,
            "policy_modal": ACTION_NAMES[p_modal],
            "oracle_modal": ACTION_NAMES[o_modal],
            "modal_match": p_modal == o_modal,
            "agreement": agree, "mean_regret": breg,
        })
        print(f"  [{edges[b]:.2f},{edges[b+1]:.2f}) | {count:>6} | "
              f"{ACTION_NAMES[p_modal]:>13} | {ACTION_NAMES[o_modal]:>13} | "
              f"{100*agree:>5.1f}% | {breg:>8.4f}  {ok}")

    worst = min(buckets, key=lambda b: b["agreement"])
    print(f"  worst bucket: [{worst['lo']:.2f},{worst['hi']:.2f}) at "
          f"{100*worst['agreement']:.1f}% agreement, regret {worst['mean_regret']:.4f}")
    return buckets


# --------------------------------------------- check 2: jitter stability

def jitter_stability(model: PPO, n: int = 8_000, n_perturb: int = 16,
                     seed: int = 12, dims=(CPU_LOAD, RAM_AVAIL)):
    """Flip rate under bounded observation noise, split justified/unjustified.

    `dims` defaults to CPU/RAM because those are the two the original demo
    staging was jitter-checked over, and they are the noisiest readings on a
    real device — psutil.cpu_percent over a 0.1s window moves several points
    between consecutive calls on an idle machine.
    """
    rng = np.random.default_rng(seed)
    base = sample_states(n, rng)
    base_policy = policy_probs(model, base).argmax(axis=1)
    base_oracle = optimal_action_batch(base)

    print(f"\n--- check 2: jitter stability over "
          f"{', '.join(('CPU_LOAD','RAM_AVAIL','LATENCY','UPLOAD','CONN_TYPE','TIME_SINCE_REKEY','THREAT')[d] for d in dims)} ---")
    print("  'unjustified' = the policy changed its decision while the oracle")
    print("  did not. Those are the flips that are pure noise-chasing; a flip")
    print("  the oracle also makes is the policy correctly tracking a real")
    print("  boundary, so counting it as instability would penalise being right.")
    header = (f"  {'jitter':>7} | {'flip rate':>10} | {'unjustified':>12} | "
              f"{'worst action':>13} | {'its unjust.':>12}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    levels = []
    for j in JITTER_LEVELS:
        flips = np.zeros(n, dtype=np.int64)
        unjust = np.zeros(n, dtype=np.int64)
        for _ in range(n_perturb):
            noise = np.zeros_like(base)
            for d in dims:
                noise[:, d] = rng.uniform(-j, j, size=n)
            pert = np.clip(base + noise, 0.0, 1.0).astype(np.float32)
            p = policy_probs(model, pert).argmax(axis=1)
            o = optimal_action_batch(pert)
            moved = p != base_policy
            flips += moved
            unjust += moved & (o == base_oracle)

        flip_rate = float(flips.sum() / (n * n_perturb))
        unjust_rate = float(unjust.sum() / (n * n_perturb))

        # Per-action, keyed on what the oracle wants at the base state: the
        # question is "when 1024 is the right answer, does the policy hold it",
        # not "when the policy happens to say 1024".
        per_action = {}
        for a, name in enumerate(ACTION_NAMES):
            m = base_oracle == a
            if not m.any():
                continue
            per_action[name] = {
                "n": int(m.sum()),
                "flip_rate": float(flips[m].sum() / (m.sum() * n_perturb)),
                "unjustified_rate": float(unjust[m].sum() / (m.sum() * n_perturb)),
            }
        worst = max(per_action.items(), key=lambda kv: kv[1]["unjustified_rate"])
        levels.append({"jitter": j, "flip_rate": flip_rate,
                       "unjustified_rate": unjust_rate, "per_action": per_action})
        print(f"  +/-{j:>4.2f} | {100*flip_rate:>9.2f}% | {100*unjust_rate:>11.2f}% | "
              f"{worst[0]:>13} | {100*worst[1]['unjustified_rate']:>11.2f}%")

    print("\n  per-action unjustified flip rate (grouped by the oracle's choice"
          " at the un-jittered state):")
    for j_result in levels:
        parts = ", ".join(f"{k} {100*v['unjustified_rate']:.1f}%"
                          for k, v in j_result["per_action"].items())
        print(f"    +/-{j_result['jitter']:.2f}: {parts}")
    return levels


# ------------------------------------------------ check 3: robust margin

def robust_margin(model: PPO, n: int = 40_000, seed: int = 13):
    """Of the decisions the policy gets right, how many are held decisively.

    Correct-but-fragile is the state Week 1 found and Week 2 was supposed to
    fix; "all four actions reachable" does not by itself establish that it
    was fixed, because reachability is about the argmax and fragility is
    about the gap under it.
    """
    rng = np.random.default_rng(seed)
    states = sample_states(n, rng)
    probs = policy_probs(model, states)
    chosen = probs.argmax(axis=1)
    oracle = optimal_action_batch(states)

    top2 = np.sort(probs, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    correct = chosen == oracle

    print("\n--- check 3: robust margin on correct decisions ---")
    print(f"  a decision is 'fragile' if its margin over the runner-up is below")
    print(f"  {FRAGILE_MARGIN:.2f} — close enough that float32 differences between the")
    print("  Python and Rust runtimes could plausibly reorder the top two.")
    header = (f"  {'oracle action':>13} | {'n':>6} | {'recall':>7} | "
              f"{'median margin':>13} | {'fragile':>8}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    per_action = {}
    for a, name in enumerate(ACTION_NAMES):
        m = oracle == a
        if not m.any():
            continue
        rec = float(correct[m].mean())
        good = m & correct
        med = float(np.median(margin[good])) if good.any() else 0.0
        frag = float((margin[good] < FRAGILE_MARGIN).mean()) if good.any() else 1.0
        per_action[name] = {"n": int(m.sum()), "recall": rec,
                            "median_margin": med, "fragile_share": frag}
        print(f"  {name:>13} | {int(m.sum()):>6} | {100*rec:>6.1f}% | "
              f"{med:>13.4f} | {100*frag:>7.2f}%")

    overall_fragile = float((margin[correct] < FRAGILE_MARGIN).mean())
    macro_recall = float(np.mean([v["recall"] for v in per_action.values()]))
    print(f"  macro-recall {100*macro_recall:.1f}%, "
          f"overall fragile share {100*overall_fragile:.2f}%")
    return {"per_action": per_action, "macro_recall": macro_recall,
            "fragile_share": overall_fragile}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(MODEL_PATH))
    ap.add_argument("--quick", action="store_true",
                    help="smaller samples, for a fast sanity run")
    args = ap.parse_args()

    path = Path(args.model)
    if not path.exists():
        print(f"no trained model at {path} — run `python -m client.rl_agent.train` first")
        return 1

    model = PPO.load(path, device="cpu")
    print(f"model: {path}")

    scale = 0.1 if args.quick else 1.0
    buckets = bucket_by_security_need(model, n=int(40_000 * scale))
    jitter = jitter_stability(model, n=int(8_000 * scale),
                              n_perturb=4 if args.quick else 16)
    margins = robust_margin(model, n=int(40_000 * scale))

    print("\n=== WEEK 3 VERIFICATION SUMMARY ===")
    bad_buckets = [b for b in buckets if not b["modal_match"]]
    print(f"  buckets with a wrong modal action : {len(bad_buckets)} of {len(buckets)}")
    print(f"  worst bucket agreement            : "
          f"{100*min(b['agreement'] for b in buckets):.1f}%")
    for lv in jitter:
        print(f"  unjustified flip rate +/-{lv['jitter']:.2f}      : "
              f"{100*lv['unjustified_rate']:.2f}%")
    print(f"  macro-recall                      : {100*margins['macro_recall']:.1f}%")
    print(f"  fragile share of correct decisions: {100*margins['fragile_share']:.2f}%")

    payload = {"model": str(path), "buckets": buckets,
               "jitter": jitter, "margins": margins,
               "fragile_margin_threshold": FRAGILE_MARGIN}
    RESULTS_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
