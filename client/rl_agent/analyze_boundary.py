"""
Week 1 diagnostic (Member 3 track): quantify the known weak point flagged
in PROJECT_BRIEFING.md Section 6 — the trained policy's ML-KEM-768 /
ML-KEM-1024 decision boundary is narrow and can collapse under realistic
background CPU load (30-60%, i.e. a normal laptop with other things
running).

This is the same verification technique already used to validate the
reward fix (bucket outcomes and look at the actual distribution the
policy produces), aimed specifically at the CPU-load confound that
technique didn't isolate before. It reads the existing trained model —
it does NOT retrain anything or touch vpn_env.py / train.py.

Produces the "before" numbers that Week 2's entropy-bonus / longer
training fix needs to demonstrably improve on, and that Week 3's
re-verification checks against.

Run with: python -m client.rl_agent.analyze_boundary
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stable_baselines3 import PPO  # noqa: E402

from client.rl_agent.vpn_env import (  # noqa: E402
    CPU_LOAD, RAM_AVAIL, LATENCY, UPLOAD, CONN_TYPE, TIME_SINCE_REKEY, THREAT,
)
from client.vpn_daemon.algo_registry import ACTIVE_ACTIONS  # noqa: E402

MODEL_PATH = Path(__file__).resolve().parent / "models" / "ppo_vpn_agent.zip"
_ACTION_NAMES = [ACTIVE_ACTIONS[k]["name"] for k in ACTIVE_ACTIONS]

# Deliberately placed inside ML-KEM-768's narrow optimal region per the
# brute-force reward-grid analysis in PROGRESS.md: moderate threat,
# cellular connection, mid session age — the same kind of scenario
# demo.py stages for its "cellular, 45 minutes in" case.
BASE_STATE = np.array(
    [0.3, 0.7, 0.2, 0.2, 1.0, 0.5, 0.5], dtype=np.float32
)  # cpu, ram_avail, latency, upload, conn_type, time_since_rekey, threat


def action_probs(model: PPO, state: np.ndarray) -> np.ndarray:
    obs_tensor, _ = model.policy.obs_to_tensor(state[None, :])
    with torch.no_grad():
        dist = model.policy.get_distribution(obs_tensor)
    return dist.distribution.probs[0].numpy()


def sweep_cpu(model: PPO, cpu_values: np.ndarray):
    print("\n--- 1D sweep: CPU_LOAD, everything else fixed ---")
    print(f"fixed state: {dict(zip(['ram_avail','latency','upload','conn_type','time_since_rekey','threat'], BASE_STATE[[RAM_AVAIL, LATENCY, UPLOAD, CONN_TYPE, TIME_SINCE_REKEY, THREAT]]))}")
    header = f"{'cpu':>6} | {'chosen':>12} | " + " | ".join(f"{n:>12}" for n in _ACTION_NAMES) + f" | {'margin':>7}"
    print(header)
    print("-" * len(header))

    prev_choice = None
    flip_points = []
    rows = []
    for cpu in cpu_values:
        s = BASE_STATE.copy()
        s[CPU_LOAD] = cpu
        probs = action_probs(model, s)
        top2 = np.argsort(probs)[::-1][:2]
        margin = probs[top2[0]] - probs[top2[1]]
        chosen = _ACTION_NAMES[top2[0]]
        rows.append((cpu, chosen, probs, margin))

        if prev_choice is not None and chosen != prev_choice:
            flip_points.append((cpu, prev_choice, chosen))
        prev_choice = chosen

        probs_str = " | ".join(f"{p:>12.3f}" for p in probs)
        print(f"{cpu:>6.2f} | {chosen:>12} | {probs_str} | {margin:>7.3f}")

    print("\nflips detected:")
    if not flip_points:
        print("  none in this range")
    for cpu, before, after in flip_points:
        print(f"  at cpu~={cpu:.2f}: {before} -> {after}")

    return rows


def window_stats(rows, action_name: str):
    """Width of the cpu band where `action_name` is the argmax, and the peak
    margin it holds over the runner-up inside that band. These are two of the
    four Week-2 comparison metrics in PROGRESS.md — retraining should widen
    the band and grow the margin."""
    won = [(cpu, margin) for cpu, chosen, _, margin in rows if chosen == action_name]
    if not won:
        return None

    lo, hi = won[0][0], won[-1][0]
    # The first sampled cpu *above* the band is where the policy actually
    # flips away, so the band's true upper edge lies between hi and that
    # point; report both rather than picking one silently.
    after = [cpu for cpu, chosen, _, _ in rows if cpu > hi and chosen != action_name]
    flip_out = after[0] if after else None
    peak_cpu, peak_margin = max(won, key=lambda t: t[1])

    return {
        "lo": lo,
        "hi": hi,
        "flip_out": flip_out,
        "width_sampled": hi - lo,
        "width_to_flip": None if flip_out is None else flip_out - lo,
        "peak_margin": peak_margin,
        "peak_cpu": peak_cpu,
        "edge_margins": (won[0][1], won[-1][1]),
    }


def sweep_cpu_ram_grid(model: PPO, cpu_values: np.ndarray, ram_values: np.ndarray):
    print("\n--- 2D sweep: (CPU_LOAD, RAM_AVAIL) -> chosen action ---")
    header = "cpu\\ram | " + " | ".join(f"{r:>5.2f}" for r in ram_values)
    print(header)
    print("-" * len(header))
    for cpu in cpu_values:
        row_labels = []
        for ram in ram_values:
            s = BASE_STATE.copy()
            s[CPU_LOAD] = cpu
            s[RAM_AVAIL] = ram
            probs = action_probs(model, s)
            chosen = _ACTION_NAMES[int(np.argmax(probs))]
            row_labels.append(chosen[:5])
        print(f"{cpu:>7.2f} | " + " | ".join(f"{lbl:>5}" for lbl in row_labels))


def ml_kem_1024_reachability(model: PPO, n_samples: int = 4000, seed: int = 0):
    print("\n--- ML-KEM-1024 reachability over random states ---")
    rng = np.random.default_rng(seed)
    states = rng.uniform(0.0, 1.0, size=(n_samples, 7)).astype(np.float32)
    obs_tensor, _ = model.policy.obs_to_tensor(states)
    with torch.no_grad():
        dist = model.policy.get_distribution(obs_tensor)
    probs = dist.distribution.probs.numpy()
    chosen = probs.argmax(axis=1)

    # Argmax counts alone can't distinguish "this action never wins but is a
    # close second" from "this action is structurally unreachable" — the max
    # probability ceiling is what separates them, and it's the metric Week 2's
    # exploration bonus has to move.
    header = f"  {'action':>12} | {'argmax':>14} | {'p_max':>7} | {'p_mean':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    stats = {}
    for i, name in enumerate(_ACTION_NAMES):
        count = int((chosen == i).sum())
        p_max, p_mean = float(probs[:, i].max()), float(probs[:, i].mean())
        stats[name] = {
            "count": count,
            "share": count / n_samples,
            "p_max": p_max,
            "p_mean": p_mean,
        }
        share = f"{count:>5} / {n_samples}"
        print(f"  {name:>12} | {share:>14} | {p_max:>7.4f} | {p_mean:>7.4f}")

    return stats


def print_baseline_summary(rows, reach_stats, target="ML-KEM-768", starved="ML-KEM-1024"):
    """The four numbers PROGRESS.md pins as the Week-1 'before' state, printed
    together so a post-retraining re-run can be diffed against them directly."""
    print("\n=== BASELINE SUMMARY (compare against this after Week 2 retraining) ===")

    w = window_stats(rows, target)
    if w is None:
        print(f"  {target} window        : never the top action in this sweep")
    else:
        flip = "n/a" if w["flip_out"] is None else f"{w['flip_out']:.2f}"
        width_to_flip = "n/a" if w["width_to_flip"] is None else f"{w['width_to_flip']:.2f}"
        print(f"  {target} window        : cpu in [{w['lo']:.2f}, {w['hi']:.2f}] sampled, flips out at {flip}")
        print(f"  {target} window width  : {w['width_sampled']:.2f} sampled ({width_to_flip} counting to the flip point)")
        print(f"  {target} peak margin   : {w['peak_margin']:.4f} at cpu={w['peak_cpu']:.2f}")
        print(f"  {target} edge margins  : {w['edge_margins'][0]:.4f} (low edge), {w['edge_margins'][1]:.4f} (high edge)")

    s = reach_stats[starved]
    print(f"  {starved} argmax share: {100*s['share']:.2f}%")
    print(f"  {starved} prob ceiling: {s['p_max']:.4f} (mean {s['p_mean']:.4f})")
    print("\n  Targets for Week 2: wider window, larger peak margin, and a nonzero")
    print(f"  argmax share for {starved} — the prob ceiling must rise, not just its visit count.")


def main():
    if not MODEL_PATH.exists():
        print(f"no trained model at {MODEL_PATH} — run `python -m client.rl_agent.train` first")
        return

    model = PPO.load(MODEL_PATH)

    cpu_values = np.concatenate([
        np.arange(0.0, 0.3, 0.1),
        np.arange(0.3, 0.6, 0.02),
        np.arange(0.6, 0.85, 0.1),
    ])
    rows = sweep_cpu(model, cpu_values)

    coarse_cpu = np.arange(0.0, 0.85, 0.1)
    ram_values = np.arange(0.3, 1.0, 0.15)
    sweep_cpu_ram_grid(model, coarse_cpu, ram_values)

    reach_stats = ml_kem_1024_reachability(model)
    print_baseline_summary(rows, reach_stats)


if __name__ == "__main__":
    main()
