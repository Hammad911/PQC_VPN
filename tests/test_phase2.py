# tests/test_phase2.py
"""
Phase 2 integration test — RL agent + state observer + anomaly detector.
Run with: pytest tests/test_phase2.py -v
"""
import sys

import numpy as np
import pytest

sys.path.insert(0, ".")


def test_algo_registry_matches_proposal_action_space():
    from client.vpn_daemon.algo_registry import ACTIVE_ACTIONS
    names = {a["name"] for a in ACTIVE_ACTIONS.values()}
    assert names == {"ML-KEM-512", "ML-KEM-768", "ML-KEM-1024", "rekey-now"}


def test_vpn_env_conforms_to_gymnasium_api():
    from gymnasium.utils.env_checker import check_env
    from client.rl_agent.vpn_env import VPNEnv
    env = VPNEnv(seed=0)
    check_env(env, skip_render_check=True)


def test_vpn_env_reset_and_step_shapes():
    from client.rl_agent.vpn_env import VPNEnv
    env = VPNEnv(episode_len=10, seed=1)
    obs, info = env.reset(seed=1)
    assert obs.shape == (7,)
    assert np.all(obs >= 0.0) and np.all(obs <= 1.0)

    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert obs.shape == (7,)
    assert np.all(obs >= 0.0) and np.all(obs <= 1.0)
    assert isinstance(reward, float)
    assert "algo" in info


def test_vpn_env_episode_truncates_at_episode_len():
    from client.rl_agent.vpn_env import VPNEnv
    env = VPNEnv(episode_len=5, seed=2)
    env.reset(seed=2)
    truncated = False
    steps = 0
    while not truncated and steps < 100:
        _, _, _, truncated, _ = env.step(env.action_space.sample())
        steps += 1
    assert steps == 5


def test_state_observer_produces_valid_vector():
    from client.rl_agent.state_observer import StateObserver, STATE_DIM
    observer = StateObserver()
    state = observer.read_state(threat_score=0.3)
    assert state.shape == (STATE_DIM,)
    assert np.all(state >= 0.0) and np.all(state <= 1.0)


def test_state_observer_mark_rekey_resets_timer():
    from client.rl_agent.state_observer import StateObserver
    from client.rl_agent.vpn_env import TIME_SINCE_REKEY
    import time
    observer = StateObserver()
    time.sleep(0.05)
    observer.mark_rekey()
    state = observer.read_state()
    assert state[TIME_SINCE_REKEY] < 0.01


def test_anomaly_detector_flags_injected_outlier():
    from client.rl_agent.anomaly_detector import ZScoreBaseline
    detector = ZScoreBaseline(window=20)
    for _ in range(20):
        result = detector.update(latency=0.1, packet_rate=0.1, packet_size=0.1)
    assert result["ready"]
    assert not result["anomalous"]

    spike = detector.update(latency=50.0, packet_rate=0.1, packet_size=0.1)
    assert spike["anomalous"]
    assert spike["threat_score"] > 0.5


def test_trained_ppo_agent_loads_and_predicts_valid_action():
    from pathlib import Path
    from stable_baselines3 import PPO
    from client.rl_agent.vpn_env import VPNEnv

    model_path = Path("client/rl_agent/models/ppo_vpn_agent.zip")
    if not model_path.exists():
        pytest.skip("no trained model on disk yet — run client/rl_agent/train.py")

    model = PPO.load(model_path)
    env = VPNEnv(seed=3)
    obs, _ = env.reset(seed=3)
    action, _ = model.predict(obs, deterministic=True)
    assert env.action_space.contains(int(action))


def test_ppo_policy_param_count_matches_proposal_spec():
    from stable_baselines3 import PPO
    from client.rl_agent.vpn_env import VPNEnv

    env = VPNEnv(seed=0)
    model = PPO("MlpPolicy", env, policy_kwargs=dict(net_arch=[64, 64]), verbose=0)

    policy_only_params = sum(
        p.numel()
        for name, p in model.policy.named_parameters()
        if "value_net" not in name
    )
    assert policy_only_params == 4932
