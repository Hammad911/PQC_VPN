# client/rl_agent/train.py
"""
Trains the PPO agent on VPNEnv and saves it to client/rl_agent/models/.

Architecture matches proposal section 3.2: a single shared trunk of
two 64-neuron hidden layers feeding both the policy and value heads
(not SB3's default of two separate 64x64 networks), which is what
gets the parameter count in the proposal's ballpark.

Run with: python -m client.rl_agent.train
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from client.rl_agent.vpn_env import VPNEnv

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODEL_DIR / "ppo_vpn_agent"


def count_params(model) -> int:
    return sum(p.numel() for p in model.policy.parameters())


def train(total_timesteps: int = 50_000, seed: int = 0) -> PPO:
    env = make_vec_env(lambda: VPNEnv(episode_len=200, seed=seed), n_envs=4)

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[64, 64]),
        verbose=1,
        seed=seed,
    )
    print(f"policy parameter count: {count_params(model)}")

    model.learn(total_timesteps=total_timesteps)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"saved model to {MODEL_PATH}.zip")
    return model


if __name__ == "__main__":
    train()
