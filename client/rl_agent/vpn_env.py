# client/rl_agent/vpn_env.py
"""
Gymnasium environment formalising the MDP from proposal section 7:
state = 7-dim device/network vector, action = algorithm choice or
rekey, reward = security strength earned vs. computational cost paid.

This env is a SIMULATOR used to train the PPO agent offline. It does
not talk to psutil or a real VPN session — a live device only ever
gives you one trajectory, which isn't enough to train a policy from
scratch. StateObserver (state_observer.py) is the live counterpart
used at inference time once a policy is trained here.

The reward weights below are an initial, reasoned design (not fit to
any data yet) — the first thing to revisit once we have real traffic
traces or a labelled threat dataset to calibrate against.
"""
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from client.vpn_daemon.algo_registry import ACTIVE_ACTIONS  # noqa: E402

STATE_DIM = 7
MAX_CPU_COST = max(a["cpu_cost"] for a in ACTIVE_ACTIONS.values())

# action index (0..N-1, contiguous, gym-friendly) -> algo_registry key
_ACTION_TO_ALGO_KEY = {i: k for i, k in enumerate(ACTIVE_ACTIONS.keys())}
_REKEY_ACTION_IDX = next(
    i for i, k in _ACTION_TO_ALGO_KEY.items()
    if ACTIVE_ACTIONS[k]["name"] == "rekey-now"
)

# state vector indices, matching StateObserver.read_state() ordering
CPU_LOAD, RAM_AVAIL, LATENCY, UPLOAD, CONN_TYPE, TIME_SINCE_REKEY, THREAT = range(7)


class VPNEnv(gym.Env):
    """One episode = one simulated session, `episode_len` decision ticks
    (each tick represents one 5-second agent loop per proposal 3.3)."""

    metadata = {"render_modes": []}

    def __init__(self, episode_len: int = 200, seed: int | None = None):
        super().__init__()
        self.episode_len = episode_len
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(STATE_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(_ACTION_TO_ALGO_KEY))
        self._rng = np.random.default_rng(seed)
        self._state = None
        self._t = 0

    def _random_initial_state(self) -> np.ndarray:
        return self._rng.uniform(0.0, 1.0, size=STATE_DIM).astype(np.float32)

    def _evolve_state(self, action_idx: int) -> np.ndarray:
        """Random-walk the state, with an occasional threat spike, and
        apply the effect of the chosen action (rekey resets the rekey
        timer; any action nudges CPU load by its cpu_cost)."""
        s = self._state.copy()

        drift = self._rng.normal(0.0, 0.03, size=STATE_DIM).astype(np.float32)
        s = np.clip(s + drift, 0.0, 1.0)

        if self._rng.random() < 0.05:
            s[THREAT] = np.clip(s[THREAT] + self._rng.uniform(0.4, 1.0), 0.0, 1.0)

        algo_key = _ACTION_TO_ALGO_KEY[action_idx]
        cpu_cost = ACTIVE_ACTIONS[algo_key]["cpu_cost"] / MAX_CPU_COST
        s[CPU_LOAD] = np.clip(s[CPU_LOAD] + 0.1 * cpu_cost, 0.0, 1.0)

        if action_idx == _REKEY_ACTION_IDX:
            s[TIME_SINCE_REKEY] = 0.0
            s[THREAT] = max(0.0, s[THREAT] - 0.5)
        else:
            s[TIME_SINCE_REKEY] = np.clip(s[TIME_SINCE_REKEY] + 0.02, 0.0, 1.0)

        return s.astype(np.float32)

    def _reward(self, prev_state: np.ndarray, action_idx: int) -> float:
        algo_key = _ACTION_TO_ALGO_KEY[action_idx]
        algo = ACTIVE_ACTIONS[algo_key]

        security_need = np.clip(
            0.4 * prev_state[THREAT]
            + 0.3 * prev_state[CONN_TYPE]
            + 0.3 * prev_state[TIME_SINCE_REKEY],
            0.0,
            1.0,
        )
        resource_pressure = np.clip(
            0.5 * prev_state[CPU_LOAD] + 0.5 * (1.0 - prev_state[RAM_AVAIL]), 0.0, 1.0
        )

        if algo["name"] == "rekey-now":
            # Scaled to roughly the same range as the algorithm-choice
            # branch below (max ~1.0). An earlier version scored this
            # branch up to 2.0, which made rekeying dominate over
            # escalating to a stronger algorithm any time threat was
            # merely nonzero — the agent never learned to pick
            # ML-KEM-768/1024 at all. Justification is tied to the
            # proposal's actual stated reason to rekey (stale session
            # keys sitting in RAM), not threat alone.
            urgency = 0.5 * prev_state[THREAT] + 0.5 * prev_state[TIME_SINCE_REKEY]
            reward = urgency - 0.5 * (1.0 - urgency) - 0.3
        else:
            # Asymmetric on purpose: falling short of the security the
            # situation calls for is a real risk and must dominate the
            # reward; picking *more* security than strictly needed only
            # costs efficiency, which matters less. A symmetric "closest
            # to need" reward (tried first) let the agent settle on
            # always picking the cheapest algorithm, since it's closest
            # to the *average* need across a uniform state distribution
            # rather than adapting per-state.
            shortfall = max(0.0, security_need - algo["security"])
            cost_penalty = (algo["cpu_cost"] / MAX_CPU_COST) * resource_pressure
            reward = 1.0 - 3.0 * shortfall - cost_penalty

        return float(reward)

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._state = self._random_initial_state()
        self._t = 0
        return self._state, {}

    def step(self, action_idx: int):
        assert self.action_space.contains(action_idx)
        reward = self._reward(self._state, action_idx)
        self._state = self._evolve_state(action_idx)
        self._t += 1

        terminated = False
        truncated = self._t >= self.episode_len
        info = {"algo": ACTIVE_ACTIONS[_ACTION_TO_ALGO_KEY[action_idx]]["name"]}

        return self._state, reward, terminated, truncated, info
