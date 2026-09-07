"""
Scores the trained policy against baselines on episode return.

Why return and not accuracy: it is tempting to measure the policy by how
often it agrees with `optimal_action` (the single-step reward maximiser),
but that metric is actively misleading here. A constant "always ML-KEM-512"
policy agrees with the greedy optimum ~90% of the time and still performs
far worse over an episode, because actions change future state — rekeying
resets the session timer and cuts threat, which pays off later and greedy
scoring cannot see. Return is what PPO optimises and what a VPN user
actually experiences, so it decides; agreement is reported alongside as a
diagnostic only.

Three of the four baselines here (classical/no-PQC stand-in, static
ML-KEM-768, rule-based) are what the proposal's Expected Outcomes call for
in the Week 9 evaluation. Building the harness now means that milestone
starts from working code.

Run with: python -m client.rl_agent.evaluate
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stable_baselines3 import PPO  # noqa: E402

from client.rl_agent.vpn_env import (  # noqa: E402
    ACTION_NAMES, CONN_TYPE, N_ACTIONS, REKEY_ACTION_IDX,
    STATE_DIM, THREAT, TIME_SINCE_REKEY, VPNEnv, action_rewards,
    action_rewards_batch, optimal_action, optimal_action_batch,
    security_need_batch,
)

MODEL_PATH = Path(__file__).resolve().parent / "models" / "ppo_vpn_agent.zip"

# The band Week 3's verification isolated the remaining defect in. Named once
# because the sampler, the metric and the test that pins it all have to agree
# on where "high need" starts.
HIGH_NEED_FLOOR = 0.90


# ---------------------------------------------------------------- baselines

def constant_policy(action_idx: int):
    return lambda state: action_idx


def greedy_policy(state: np.ndarray) -> int:
    """Upper reference, not a real baseline — it reads the reward function
    directly, which a deployed agent cannot do."""
    return optimal_action(state)


def rule_based_policy(state: np.ndarray) -> int:
    """The proposal's third baseline: escalate on obvious signals, no
    learning. Deliberately simple — this is the 'would a few if-statements
    have done the job' control that the RL agent has to beat to justify
    itself."""
    if state[THREAT] > 0.7 or state[TIME_SINCE_REKEY] > 0.9:
        return REKEY_ACTION_IDX
    if state[THREAT] > 0.4 or state[CONN_TYPE] > 0.75:
        return 1                                  # ML-KEM-768
    return 0                                      # ML-KEM-512


def model_policy(model: PPO):
    return lambda state: int(model.predict(state, deterministic=True)[0])


# --------------------------------------------------------------- evaluation

def rollout(policy_fn, n_episodes: int, episode_len: int, seed0: int,
            curriculum: float = 0.0) -> np.ndarray:
    """Total undiscounted return per episode. Every policy sees the same
    episode seeds, so differences are policy differences, not luck."""
    returns = []
    for i in range(n_episodes):
        env = VPNEnv(episode_len=episode_len, seed=seed0 + i, curriculum=curriculum)
        state, _ = env.reset(seed=seed0 + i)
        total, done = 0.0, False
        while not done:
            state, reward, terminated, truncated, _ = env.step(policy_fn(state))
            total += reward
            done = terminated or truncated
        returns.append(total)
    return np.array(returns)


def sample_states(n: int, seed: int, curriculum: float = 0.0) -> np.ndarray:
    env = VPNEnv(seed=seed, curriculum=curriculum)
    return np.array([env.reset()[0] for _ in range(n)], dtype=np.float32)


def balanced_states(per_class: int = 1500, seed: int = 0,
                    batch: int = 200_000) -> np.ndarray:
    """Equal numbers of states per oracle-optimal action, by rejection sampling.

    ML-KEM-1024 is optimal in 0.39% of the state space and rekey-now in 0.14%,
    so a uniform sample of any practical size gives a recall estimate for them
    with enormous variance — 4,000 uniform states contain ~16 ML-KEM-1024 cases.
    Balancing makes the per-action recalls directly comparable and low-variance,
    which is what lets macro-recall be a trustworthy selection metric.
    """
    rng = np.random.default_rng(seed)
    pools: list[list[np.ndarray]] = [[] for _ in range(N_ACTIONS)]

    while min(len(p) for p in pools) < per_class:
        cand = rng.uniform(0.0, 1.0, size=(batch, STATE_DIM)).astype(np.float32)
        best = optimal_action_batch(cand)
        for i in range(N_ACTIONS):
            if len(pools[i]) < per_class:
                pools[i].extend(cand[best == i][: per_class - len(pools[i])])

    return np.concatenate([np.array(p[:per_class]) for p in pools]).astype(np.float32)


def high_need_states(n: int = 4000, seed: int = 0,
                     need_floor: float = HIGH_NEED_FLOOR,
                     batch: int = 400_000) -> np.ndarray:
    """States where security_need >= `need_floor`, by rejection sampling.

    Week 3's verification found the policy's one real remaining defect lives
    here: above need 0.90 it switches from ML-KEM-1024 to rekey-now at
    resource pressure ~0.35 when the true crossover is ~0.50, so it is
    confidently wrong across that band. The band is ~0.5% of a uniform
    sample, which is why every Week 2 aggregate — 99.3% agreement, 0.0007
    regret — stayed clean while it was broken. Scoring it explicitly is what
    stops that from happening again.
    """
    rng = np.random.default_rng(seed)
    kept: list[np.ndarray] = []
    total = 0
    while total < n:
        cand = rng.uniform(0.0, 1.0, size=(batch, STATE_DIM)).astype(np.float32)
        hit = cand[security_need_batch(cand) >= need_floor]
        kept.append(hit)
        total += len(hit)
    return np.concatenate(kept)[:n]


def policy_probs(model: PPO, states: np.ndarray) -> np.ndarray:
    """Action probabilities for an (N, 7) batch.

    The one place this three-line SB3 incantation lives. It had drifted into
    five near-copies that disagreed on whether to `.cpu()` and whether to cast
    the input, which meant a device or dtype fix had five places to reach.
    """
    import torch
    obs, _ = model.policy.obs_to_tensor(np.asarray(states, dtype=np.float32))
    with torch.no_grad():
        return model.policy.get_distribution(obs).distribution.probs.cpu().numpy()


def regret_batch(rewards: np.ndarray, chosen: np.ndarray,
                 best: np.ndarray | None = None) -> np.ndarray:
    """Per-state reward given up by taking `chosen` instead of the optimum.

    Takes the already-built (N, n_actions) reward matrix rather than states,
    because every caller needs the matrix for something else too and building
    it twice is the expensive half.
    """
    if best is None:
        best = rewards.argmax(axis=1)
    idx = np.arange(len(rewards))
    return rewards[idx, best] - rewards[idx, chosen]


def high_need_metrics(model: PPO, high_need: np.ndarray) -> dict:
    """Agreement and regret restricted to the high-security-need band.

    Its own function rather than a branch inside `policy_metrics`: the band is
    ~0.5% of a uniform sample, so it is the one metric that does not follow
    from the others, and a caller that wants only it should not have to supply
    two unrelated evaluation sets to get it.
    """
    rewards = action_rewards_batch(high_need)
    chosen = policy_probs(model, high_need).argmax(axis=1)
    best = rewards.argmax(axis=1)
    return {
        "high_need_agreement": float((chosen == best).mean()),
        "high_need_regret": float(regret_batch(rewards, chosen, best).mean()),
    }


def policy_metrics(model: PPO, uniform: np.ndarray, balanced: np.ndarray) -> dict:
    """The bundle Week 2 is judged on.

    No single number here is sufficient on its own, and each one catches a
    different way of faking success: macro-recall catches a policy that
    ignores the rare actions, the two-sided ML-KEM-1024 share catches one that
    over-selects them, regret catches one that is right about rare cases but
    wrong about common ones, and decisiveness catches a policy flattened by an
    over-large entropy bonus (which raises the probability ceiling without
    improving any decision).
    """
    pu = policy_probs(model, uniform)
    chosen_u = pu.argmax(axis=1)
    rewards_u = action_rewards_batch(uniform)
    best_u = rewards_u.argmax(axis=1)
    regret = float(regret_batch(rewards_u, chosen_u, best_u).mean())

    pb = policy_probs(model, balanced)
    chosen_b = pb.argmax(axis=1)
    best_b = optimal_action_batch(balanced)
    recall = {
        name: float((chosen_b[best_b == i] == i).mean()) if (best_b == i).any() else 0.0
        for i, name in enumerate(ACTION_NAMES)
    }

    return {
        "macro_recall": float(np.mean(list(recall.values()))),
        "recall": recall,
        "regret": regret,
        "agreement": float((chosen_u == best_u).mean()),
        "decisiveness": float(pu.max(axis=1).mean()),
        "mlkem1024_share": float((chosen_u == 2).mean()),
        "mlkem1024_pmax": float(pu[:, 2].max()),
        "action_share": {n: float((chosen_u == i).mean())
                         for i, n in enumerate(ACTION_NAMES)},
    }


def action_report(model: PPO, states: np.ndarray) -> dict:
    """Where the policy's choices sit relative to the greedy optimum, and how
    much per-step reward that costs."""
    chosen = np.array([int(model.predict(s, deterministic=True)[0]) for s in states])
    best = np.array([optimal_action(s) for s in states])
    rewards = np.array([action_rewards(s) for s in states])

    gap = regret_batch(rewards, chosen, best)

    return {
        "chosen": chosen,
        "best": best,
        "agreement": float((chosen == best).mean()),
        "mean_gap": float(gap.mean()),
        "recall": {
            name: (float((chosen[best == i] == i).mean()) if (best == i).any() else None,
                   int((best == i).sum()))
            for i, name in enumerate(ACTION_NAMES)
        },
    }


# ------------------------------------------------------------------ reports

def report_returns(model: PPO, episodes: int, episode_len: int, seed0: int) -> None:
    print(f"\n=== episode return over {episodes} episodes x {episode_len} steps ===")
    print("(same episode seeds for every policy; higher is better)\n")

    policies = [
        ("trained PPO", model_policy(model)),
        ("rule-based baseline", rule_based_policy),
        ("static ML-KEM-768", constant_policy(1)),
        ("always ML-KEM-512", constant_policy(0)),
        ("always ML-KEM-1024", constant_policy(2)),
        ("greedy oracle (reference)", greedy_policy),
    ]

    results = {}
    for name, fn in policies:
        r = rollout(fn, episodes, episode_len, seed0)
        results[name] = r
        # standard error, so "better" can be distinguished from "noisier"
        sem = r.std(ddof=1) / np.sqrt(len(r))
        print(f"  {name:>26}: {r.mean():8.2f}  +/- {1.96*sem:5.2f} (95% CI)")

    trained = results["trained PPO"].mean()
    oracle = results["greedy oracle (reference)"].mean()
    best_baseline = max(v.mean() for k, v in results.items()
                        if k not in ("trained PPO", "greedy oracle (reference)"))
    print(f"\n  trained vs best baseline : {trained - best_baseline:+.2f} "
          f"({100*(trained/best_baseline - 1):+.1f}%)")
    print(f"  trained vs greedy oracle : {trained - oracle:+.2f} "
          f"({100*(trained/oracle - 1):+.1f}%)")


def report_actions(model: PPO, n_states: int, seed: int) -> None:
    print(f"\n=== action selection over {n_states} states (uniform / deployment distribution) ===")
    states = sample_states(n_states, seed)
    rep = action_report(model, states)

    print(f"\n  policy vs greedy-optimal action distribution:")
    for i, name in enumerate(ACTION_NAMES):
        pol = 100 * (rep["chosen"] == i).mean()
        opt = 100 * (rep["best"] == i).mean()
        print(f"    {name:>12}: policy {pol:6.2f}%   greedy-optimal {opt:6.2f}%")

    print(f"\n  recall (of states where the action is greedy-optimal, "
          f"how often the policy picks it):")
    for name, (rec, n) in rep["recall"].items():
        shown = "     n/a" if rec is None else f"{100*rec:6.2f}%"
        print(f"    {name:>12}: {shown}  (n={n})")

    print(f"\n  greedy agreement: {100*rep['agreement']:.2f}%  "
          f"(diagnostic only — see module docstring)")
    print(f"  mean per-step reward gap vs greedy: {rep['mean_gap']:.4f}")


def report_targeted(model: PPO, n_states: int, seed: int) -> None:
    """The states ML-KEM-1024 exists for. Its share of the uniform
    distribution is so small (~0.4%) that a uniform sample says almost
    nothing about whether the policy can select it at all."""
    print(f"\n=== targeted high-need / low-resource-pressure states (n={n_states}) ===")
    states = sample_states(n_states, seed, curriculum=1.0)
    rep = action_report(model, states)

    for i, name in enumerate(ACTION_NAMES):
        pol = 100 * (rep["chosen"] == i).mean()
        opt = 100 * (rep["best"] == i).mean()
        print(f"    {name:>12}: policy {pol:6.2f}%   greedy-optimal {opt:6.2f}%")

    rec, n = rep["recall"]["ML-KEM-1024"]
    if rec is None:
        print("\n  ML-KEM-1024 is not greedy-optimal anywhere in this sample")
    else:
        print(f"\n  ML-KEM-1024 recall here: {100*rec:.2f}% of {n} states where it is optimal")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--episodes", type=int, default=150)
    parser.add_argument("--episode-len", type=int, default=200)
    parser.add_argument("--states", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()

    if not args.model.exists():
        print(f"no trained model at {args.model} — run `python -m client.rl_agent.train` first")
        return 1

    print(f"model: {args.model}")
    model = PPO.load(args.model, device="cpu")

    report_returns(model, args.episodes, args.episode_len, args.seed)
    report_actions(model, args.states, args.seed)
    report_targeted(model, args.states, args.seed + 1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
