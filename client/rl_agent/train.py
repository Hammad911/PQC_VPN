# client/rl_agent/train.py
"""
Trains the PPO agent on VPNEnv and saves it to client/rl_agent/models/.

Architecture matches proposal section 3.2: two 64-neuron hidden layers
feeding the policy head (not SB3's default of two separate 64x64
networks), which is what gets the parameter count in the proposal's
ballpark. Do not change `net_arch` — test_phase2.py pins the resulting
policy at exactly 4,932 parameters.

Week 2 changes, all aimed at the ML-KEM-1024 reachability problem
documented in PROGRESS.md:

  * `ent_coef` was left at SB3's default of 0.0, i.e. no exploration
    bonus at all. It is now a swept parameter.
  * Training runs on a curriculum (`VPNEnv(curriculum=...)`) that
    oversamples the high-security-need corner. Under the plain uniform
    reset, ML-KEM-1024 is the best action in 0.39% of states, so a 50k
    run saw a few hundred such states in total.
  * The four parallel envs all received the *same* seed, so their noise
    was correlated. They are now offset per rank.
  * Evaluation runs on `curriculum=0.0` — the deployment distribution —
    even when training uses a curriculum, so model selection can't be
    won by overfitting the oversampled corner.

Run with:
    python -m client.rl_agent.train              # single run, sensible defaults
    python -m client.rl_agent.train --sweep      # the Week 2 experiment
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import EvalCallback  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

from client.rl_agent.evaluate import (  # noqa: E402
    balanced_states, high_need_states, policy_metrics, sample_states,
)
from client.rl_agent.vpn_env import VPNEnv  # noqa: E402
from client.vpn_daemon.algo_registry import ACTIVE_ACTIONS  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODEL_DIR / "ppo_vpn_agent"
RUNS_DIR = MODEL_DIR / "runs"
SWEEP_RESULTS = MODEL_DIR / "sweep_results.json"

ACTION_NAMES = [ACTIVE_ACTIONS[k]["name"] for k in ACTIVE_ACTIONS]
MLKEM_1024_IDX = ACTION_NAMES.index("ML-KEM-1024")

EPISODE_LEN = 200
N_ENVS = 8
# n_envs * n_steps is the rollout size; 8 x 1024 keeps it at the 8192 the
# previous 4 x 2048 default produced, so more envs buys throughput without
# changing the learning dynamics.
N_STEPS = 1024


def count_params(model) -> int:
    return sum(p.numel() for p in model.policy.parameters())


def make_vec_env(seed: int, curriculum: float, episode_len: int = EPISODE_LEN,
                 cpu_relax: float = 0.0, n_envs: int = N_ENVS,
                 curriculum_pressure: str = "low") -> DummyVecEnv:
    def factory(rank: int):
        def _init():
            # offset per rank — identical seeds across envs meant four copies
            # of the same trajectory noise
            return Monitor(VPNEnv(episode_len=episode_len, seed=seed + 1000 * rank,
                                  curriculum=curriculum, cpu_relax=cpu_relax,
                                  curriculum_pressure=curriculum_pressure))
        return _init

    return DummyVecEnv([factory(i) for i in range(n_envs)])


# Fixed evaluation sets, built once. Selection uses these and never the
# training distribution, so a curriculum cannot win by being oversampled.
_EVAL_SETS: dict[str, np.ndarray] = {}


def eval_sets() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not _EVAL_SETS:
        _EVAL_SETS["uniform"] = sample_states(20_000, seed=4242, curriculum=0.0)
        _EVAL_SETS["balanced"] = balanced_states(per_class=1500, seed=4243)
        # Week 3: the band where the ML-KEM-1024 / rekey-now boundary lives.
        # Held out from training the same way the other two are.
        _EVAL_SETS["high_need"] = high_need_states(4000, seed=4244)
    return _EVAL_SETS["uniform"], _EVAL_SETS["balanced"], _EVAL_SETS["high_need"]


# A policy can look good on any single metric while being useless. These gates
# reject the two cheap ways of faking it before macro-recall is even consulted:
# a policy flattened by too much entropy (low decisiveness, high probability
# ceiling but no better decisions), and one that simply over-selects the rare
# action (high recall, terrible everywhere else).
MIN_DECISIVENESS = 0.60
MLKEM1024_SHARE_RANGE = (0.0010, 0.0090)   # oracle value is ~0.0040
# Week 3 adds a third gate. The promoted Week 2 model scores 0.64 here while
# clearing both gates above comfortably, which is exactly the blind spot: the
# band is 0.5% of the state space, so being wrong across all of it moves no
# aggregate. A model that regresses to that behaviour is now ineligible.
MIN_HIGH_NEED_AGREEMENT = 0.75


def passes_gates(m: dict) -> bool:
    lo, hi = MLKEM1024_SHARE_RANGE
    return (m["decisiveness"] >= MIN_DECISIVENESS
            and lo <= m["mlkem1024_share"] <= hi
            and m.get("high_need_agreement", 0.0) >= MIN_HIGH_NEED_AGREEMENT)


def score_model(model: PPO) -> dict:
    uniform, balanced, high_need = eval_sets()
    m = policy_metrics(model, uniform, balanced, high_need)
    m["gates_pass"] = passes_gates(m)
    return m


def train_one(*, total_timesteps: int, seed: int, curriculum: float, ent_coef: float,
              run_name: str, episode_len: int = EPISODE_LEN, gamma: float = 0.99,
              cpu_relax: float = 0.0, curriculum_pressure: str = "low",
              verbose: int = 0) -> dict:
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    env = make_vec_env(seed, curriculum, episode_len=episode_len, cpu_relax=cpu_relax,
                       curriculum_pressure=curriculum_pressure)
    # Selection happens on the deployment distribution, never the curriculum —
    # and with the original CPU dynamics, since cpu_relax is a training aid.
    eval_env = make_vec_env(seed + 77, curriculum=0.0, episode_len=episode_len,
                            cpu_relax=0.0, n_envs=1)

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[64, 64]),
        n_steps=N_STEPS,
        ent_coef=ent_coef,
        gamma=gamma,
        seed=seed,
        verbose=verbose,
        tensorboard_log=str(run_dir / "tb"),
    )

    # ~10 evaluations per run regardless of length, so a short smoke run still
    # produces a best_model and the number below is never -inf.
    # 50 episodes rather than 30: selection noise matters here, because an
    # early lucky evaluation can otherwise be crowned "best" over a genuinely
    # better-trained later checkpoint.
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir),
        n_eval_episodes=50,
        eval_freq=max(1, total_timesteps // (N_ENVS * 10)),
        deterministic=True,
        verbose=0,
    )

    t0 = time.perf_counter()
    model.learn(total_timesteps=total_timesteps, callback=eval_cb, progress_bar=False)
    elapsed = time.perf_counter() - t0

    # Keep the final model too, not just the best-by-eval one — comparing the
    # two is how you notice that "best" was an early fluke rather than the
    # end of a real learning curve.
    final_path = run_dir / "final.zip"
    model.save(final_path.with_suffix(""))

    best_path = run_dir / "best_model.zip"
    best = PPO.load(best_path, device="cpu") if best_path.exists() else model

    # Score both, and keep whichever is better on macro-recall. The best-by-return
    # checkpoint and the final one disagree often enough that taking either on
    # faith is a mistake — the previous train.py always took the final one.
    best_metrics = score_model(best)
    final_metrics = score_model(model)
    if final_metrics["macro_recall"] > best_metrics["macro_recall"]:
        best_metrics, chosen_path = final_metrics, final_path
    else:
        chosen_path = best_path if best_path.exists() else final_path

    result = {
        "run_name": run_name,
        "seed": seed,
        "curriculum": curriculum,
        "ent_coef": ent_coef,
        "episode_len": episode_len,
        "gamma": gamma,
        "cpu_relax": cpu_relax,
        "curriculum_pressure": curriculum_pressure,
        "total_timesteps": total_timesteps,
        "eval_mean_return": float(eval_cb.best_mean_reward),
        "train_seconds": round(elapsed, 1),
        "model_path": str(chosen_path),
        **best_metrics,
    }

    env.close()
    eval_env.close()
    return result


def describe(result: dict) -> str:
    gate = "PASS" if result["gates_pass"] else "fail"
    return (f"  {result['run_name']:<34} macro {100*result['macro_recall']:5.1f}%  "
            f"1024 recall {100*result['recall']['ML-KEM-1024']:5.1f}%  "
            f"share {100*result['mlkem1024_share']:5.3f}%  "
            f"regret {result['regret']:.4f}  "
            f"hi-need {100*result.get('high_need_agreement', 0.0):5.1f}%  "
            f"[{gate}, {result['train_seconds']:.0f}s]")


# An ablation ladder rather than a single candidate: each row adds one lever, so
# the write-up can attribute the effect instead of asserting that a bundle of
# changes happened to work.
#
# label                     episode_len  curriculum  gamma  cpu_relax  ent_coef  pressure
LADDERS = {
    "week2": [
        ("A_status_quo",           200,   0.0,        0.99,  0.0,       0.0,      "low"),
        ("B_curriculum_only",      200,   0.3,        0.99,  0.0,       0.0,      "low"),
        ("C_short_horizon",          8,   0.5,        0.90,  0.0,       0.0,      "low"),
        ("D_short_lowgamma",         8,   0.5,        0.50,  0.0,       0.0,      "low"),
        ("E_plus_cpu_relax",         8,   0.5,        0.50,  0.5,       0.0,      "low"),
        ("F_ep4_full",               4,   1.0,        0.50,  0.5,       0.0,      "low"),
        ("G_ep4_full_entropy",       4,   1.0,        0.50,  0.5,       0.01,     "low"),
    ],
    # Week 3. F is re-run rather than compared against its recorded Week 2
    # numbers, because H adds the high-need eval set and a fair A/B needs both
    # arms scored by the same code on the same seeds.
    "week3": [
        ("F_control",                4,   1.0,        0.50,  0.5,       0.0,      "low"),
        ("H_pressure_spread",        4,   1.0,        0.50,  0.5,       0.0,      "spread"),
    ],
}

SWEEP_CONFIGS = LADDERS["week2"]


def sweep(total_timesteps: int, seeds: list[int],
          configs: list[tuple] | None = None,
          results_path: Path = SWEEP_RESULTS) -> list[dict]:
    configs = configs if configs is not None else SWEEP_CONFIGS
    results = []
    total = len(configs) * len(seeds)
    for label, ep_len, curr, gamma, relax, ent, pressure in configs:
        for seed in seeds:
            name = f"{label}_s{seed}"
            print(f"[{len(results)+1}/{total}] {name} ...", flush=True)
            r = train_one(total_timesteps=total_timesteps, seed=seed, run_name=name,
                          curriculum=curr, ent_coef=ent, episode_len=ep_len,
                          gamma=gamma, cpu_relax=relax,
                          curriculum_pressure=pressure)
            r["config"] = label
            results.append(r)
            print(describe(r), flush=True)
            results_path.write_text(json.dumps(results, indent=2) + "\n")

    return results


def summarise(results: list[dict], configs: list[tuple] | None = None) -> dict:
    configs = configs if configs is not None else SWEEP_CONFIGS
    print("\n=== sweep summary (mean +/- sd over seeds) ===")
    print(f"  {'config':<22} {'macro':>14} {'1024 recall':>12} {'1024 share':>12} "
          f"{'hi-need':>9} {'regret':>9} {'passing':>9}")
    by_cfg: dict[str, list[dict]] = {}
    for r in results:
        by_cfg.setdefault(r["config"], []).append(r)

    rows = []
    for label, *_ in configs:
        runs = by_cfg.get(label, [])
        if not runs:
            continue
        macro = np.array([r["macro_recall"] for r in runs])
        rec = np.array([r["recall"]["ML-KEM-1024"] for r in runs])
        share = np.array([r["mlkem1024_share"] for r in runs])
        regret = np.array([r["regret"] for r in runs])
        hi = np.array([r.get("high_need_agreement", 0.0) for r in runs])
        n_pass = sum(r["gates_pass"] for r in runs)
        sd = lambda a: a.std(ddof=1) if len(a) > 1 else 0.0  # noqa: E731

        rows.append({
            "config": label, "n_seeds": len(runs),
            "macro_recall_mean": float(macro.mean()), "macro_recall_sd": float(sd(macro)),
            "mlkem1024_recall_mean": float(rec.mean()),
            "mlkem1024_share_mean": float(share.mean()),
            "high_need_agreement_mean": float(hi.mean()),
            "high_need_agreement_sd": float(sd(hi)),
            "regret_mean": float(regret.mean()),
            "seeds_passing_gates": int(n_pass),
        })
        print(f"  {label:<22} {100*macro.mean():6.1f} +/-{100*sd(macro):4.1f} "
              f"{100*rec.mean():10.1f}% {100*share.mean():11.3f}% "
              f"{100*hi.mean():8.1f}% {regret.mean():9.4f} {n_pass:>6}/{len(runs)}")

    # Rank on macro-recall, but only among runs that clear the gates — a run
    # that fakes it by flattening or over-selecting is not eligible at all.
    eligible = [r for r in results if r["gates_pass"]]
    pool = eligible or results
    # Rank on macro-recall plus high-need agreement, weighted equally. Macro
    # alone would have promoted the Week 2 model again: it is 88.5% macro and
    # 64% high-need, and the defect Week 3 exists to fix lives entirely in the
    # second number.
    best_run = max(pool, key=lambda r: r["macro_recall"]
                   + r.get("high_need_agreement", 0.0))
    print(f"\n{len(eligible)}/{len(results)} runs cleared the gates")
    print(f"best run: {best_run['run_name']}  macro-recall "
          f"{100*best_run['macro_recall']:.1f}%  "
          f"1024 recall {100*best_run['recall']['ML-KEM-1024']:.1f}%"
          + ("" if eligible else "   (NO run cleared the gates — see write-up)"))
    return {"per_config": rows, "best_run": best_run,
            "n_eligible": len(eligible), "n_runs": len(results)}


def promote(model_path: str) -> None:
    """Install a swept model as THE model the demo, tests and ONNX export use."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(model_path, MODEL_PATH.with_suffix(".zip"))
    print(f"promoted {model_path} -> {MODEL_PATH}.zip")


def train(total_timesteps: int = 1_000_000, seed: int = 0,
          curriculum: float = 0.3, ent_coef: float = 0.01) -> PPO:
    """Single run with the Week 2 defaults."""
    env = make_vec_env(seed, curriculum)
    model = PPO("MlpPolicy", env, policy_kwargs=dict(net_arch=[64, 64]),
                n_steps=N_STEPS, ent_coef=ent_coef, seed=seed, verbose=1)
    print(f"policy parameter count: {count_params(model)}")
    model.learn(total_timesteps=total_timesteps)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"saved model to {MODEL_PATH}.zip")
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the PPO VPN agent")
    parser.add_argument("--sweep", action="store_true",
                        help="run an ablation ladder (see --ladder)")
    parser.add_argument("--ladder", choices=sorted(LADDERS), default="week2",
                        help="which ablation ladder to run under --sweep")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--promote", action="store_true",
                        help="after a sweep, install the best run as the shipped model")
    args = parser.parse_args()

    if not args.sweep:
        train(total_timesteps=args.timesteps)
        return 0

    configs = LADDERS[args.ladder]
    results_path = (SWEEP_RESULTS if args.ladder == "week2"
                    else MODEL_DIR / f"sweep_results_{args.ladder}.json")
    results = sweep(args.timesteps, args.seeds, configs, results_path)
    summary = summarise(results, configs)
    results_path.write_text(json.dumps(
        {"runs": results, "summary": summary}, indent=2) + "\n")
    print(f"\nwrote {results_path}")

    if args.promote:
        promote(summary["best_run"]["model_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
