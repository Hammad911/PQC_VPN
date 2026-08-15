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


def action_reward(state: np.ndarray, action_idx: int) -> float:
    """Reward for taking `action_idx` in `state`.

    Module-level rather than a method because evaluation and tests need to
    score actions the agent didn't take. Keeping one implementation means the
    baselines in evaluate.py are scored by exactly the same function the agent
    is trained on, instead of a copy that can drift away from it.
    """
    algo_key = _ACTION_TO_ALGO_KEY[action_idx]
    algo = ACTIVE_ACTIONS[algo_key]

    security_need = np.clip(
        0.4 * state[THREAT]
        + 0.3 * state[CONN_TYPE]
        + 0.3 * state[TIME_SINCE_REKEY],
        0.0,
        1.0,
    )
    resource_pressure = np.clip(
        0.5 * state[CPU_LOAD] + 0.5 * (1.0 - state[RAM_AVAIL]), 0.0, 1.0
    )

    if algo["name"] == "rekey-now":
        # Scaled to roughly the same range as the algorithm-choice branch
        # below (max ~1.0). An earlier version scored this branch up to 2.0,
        # which made rekeying dominate over escalating to a stronger
        # algorithm any time threat was merely nonzero — the agent never
        # learned to pick ML-KEM-768/1024 at all. Justification is tied to
        # the proposal's actual stated reason to rekey (stale session keys
        # sitting in RAM), not threat alone.
        urgency = 0.5 * state[THREAT] + 0.5 * state[TIME_SINCE_REKEY]
        reward = urgency - 0.5 * (1.0 - urgency) - 0.3
    else:
        # Asymmetric on purpose: falling short of the security the situation
        # calls for is a real risk and must dominate the reward; picking
        # *more* security than strictly needed only costs efficiency, which
        # matters less. A symmetric "closest to need" reward (tried first)
        # let the agent settle on always picking the cheapest algorithm,
        # since it's closest to the *average* need across a uniform state
        # distribution rather than adapting per-state.
        shortfall = max(0.0, security_need - algo["security"])
        cost_penalty = (algo["cpu_cost"] / MAX_CPU_COST) * resource_pressure
        reward = 1.0 - 3.0 * shortfall - cost_penalty

    return float(reward)


def action_rewards(state: np.ndarray) -> np.ndarray:
    """Reward for every action in `state`, in action-index order."""
    return np.array(
        [action_reward(state, a) for a in range(len(_ACTION_TO_ALGO_KEY))],
        dtype=np.float64,
    )


def action_rewards_batch(states: np.ndarray) -> np.ndarray:
    """Vectorised `action_rewards` over an (N, 7) batch, returning (N, n_actions).

    A second implementation of the same algebra, which is a drift risk — so
    `test_phase3.py::test_batched_rewards_match_scalar_rewards` pins the two
    equal elementwise. It exists because building a balanced evaluation set
    needs the oracle over millions of candidate states, and the scalar path
    (a Python loop, four calls per state) is far too slow for that.
    """
    states = np.asarray(states, dtype=np.float64)
    need = np.clip(
        0.4 * states[:, THREAT]
        + 0.3 * states[:, CONN_TYPE]
        + 0.3 * states[:, TIME_SINCE_REKEY],
        0.0, 1.0,
    )
    pressure = np.clip(
        0.5 * states[:, CPU_LOAD] + 0.5 * (1.0 - states[:, RAM_AVAIL]), 0.0, 1.0
    )

    columns = []
    for action_idx in range(len(_ACTION_TO_ALGO_KEY)):
        algo = ACTIVE_ACTIONS[_ACTION_TO_ALGO_KEY[action_idx]]
        if algo["name"] == "rekey-now":
            urgency = 0.5 * states[:, THREAT] + 0.5 * states[:, TIME_SINCE_REKEY]
            columns.append(urgency - 0.5 * (1.0 - urgency) - 0.3)
        else:
            shortfall = np.maximum(0.0, need - algo["security"])
            cost_penalty = (algo["cpu_cost"] / MAX_CPU_COST) * pressure
            columns.append(1.0 - 3.0 * shortfall - cost_penalty)

    return np.stack(columns, axis=1)


def optimal_action_batch(states: np.ndarray) -> np.ndarray:
    return action_rewards_batch(states).argmax(axis=1)


def optimal_action(state: np.ndarray) -> int:
    """The single-step reward-maximising action.

    A useful diagnostic and the oracle baseline in evaluate.py, but NOT the
    optimal policy: actions change future state (rekey resets the session
    timer and cuts threat), so a discounted-return maximiser can rationally
    disagree with this. Compare policies on episode return, not on agreement
    with this function.
    """
    return int(action_rewards(state).argmax())


class VPNEnv(gym.Env):
    """One episode = one simulated session, `episode_len` decision ticks
    (each tick represents one 5-second agent loop per proposal 3.3)."""

    metadata = {"render_modes": []}

    # curriculum/cpu_relax are keyword-only: they are training knobs, not part
    # of the interface, and positionally `VPNEnv(200, 0, 0.5)` would silently
    # mean "curriculum=0.5" to a reader expecting the old two-arg signature.
    def __init__(self, episode_len: int = 200, seed: int | None = None, *,
                 curriculum: float = 0.0, cpu_relax: float = 0.0):
        super().__init__()
        if not 0.0 <= curriculum <= 1.0:
            raise ValueError(f"curriculum must be in [0,1], got {curriculum}")
        if not 0.0 <= cpu_relax <= 1.0:
            raise ValueError(f"cpu_relax must be in [0,1], got {cpu_relax}")
        self.episode_len = episode_len
        self.curriculum = float(curriculum)
        self.cpu_relax = float(cpu_relax)
        self._cpu_baseline = 0.0
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(STATE_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(_ACTION_TO_ALGO_KEY))
        self._rng = np.random.default_rng(seed)
        self._state = None
        self._t = 0

    def _high_need_initial_state(self) -> np.ndarray:
        """An initial state from the corner where the strong algorithms are
        actually the right answer: high security need, low resource pressure.

        Why this exists: under the plain uniform reset below, ML-KEM-1024 is
        the analytically optimal action in only 0.39% of the state space
        (measured continuously — PROGRESS.md's earlier ~1.3% came from a
        coarse product grid that over-weights the all-dimensions-maximal
        corner). At 50k timesteps that is a few hundred relevant states in
        all of training, which is why the smoke-run policy never learned the
        action at all. Drawing a fraction of episodes from here raises the
        share of ML-KEM-1024-optimal states to ~22%, so the signal exists to
        be learned from.

        LATENCY and UPLOAD stay uniform — they do not enter the reward.
        """
        s = self._rng.uniform(0.0, 1.0, size=STATE_DIM)
        # the three drivers of security_need, biased high
        s[THREAT] = self._rng.beta(4.0, 1.0)
        s[CONN_TYPE] = self._rng.beta(4.0, 1.0)
        s[TIME_SINCE_REKEY] = self._rng.beta(4.0, 1.0)
        # low resource pressure, so paying for a stronger algorithm is viable
        s[CPU_LOAD] = self._rng.beta(1.0, 3.0)
        s[RAM_AVAIL] = self._rng.beta(3.0, 1.0)
        return s.astype(np.float32)

    def _random_initial_state(self) -> np.ndarray:
        # Short-circuits before touching the RNG when curriculum is 0, so the
        # default env draws the exact same sequence it did before this
        # parameter existed — the frozen contract and the existing tests are
        # unaffected, and a curriculum run is opt-in only.
        if self.curriculum > 0.0 and self._rng.random() < self.curriculum:
            return self._high_need_initial_state()
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

        # The line above is a one-way ratchet: every action adds CPU load and
        # nothing ever removes it, while the drift above is zero-mean. CPU
        # therefore saturates at 1.0 within ~10 steps and stays there for the
        # rest of the episode (measured: mean 0.994, 96.5% of steps above
        # 0.95). That is why the policy is fragile in the 30-60% CPU band the
        # Week 1 analysis flagged — training almost never visits it. A real
        # device sheds load when work finishes, so mean-revert toward the
        # episode's starting load. Default 0.0 leaves the original dynamics
        # untouched; this is opt-in for training runs.
        if self.cpu_relax > 0.0:
            s[CPU_LOAD] = np.clip(
                s[CPU_LOAD] - self.cpu_relax * (s[CPU_LOAD] - self._cpu_baseline),
                0.0, 1.0,
            )

        if action_idx == _REKEY_ACTION_IDX:
            s[TIME_SINCE_REKEY] = 0.0
            s[THREAT] = max(0.0, s[THREAT] - 0.5)
        else:
            s[TIME_SINCE_REKEY] = np.clip(s[TIME_SINCE_REKEY] + 0.02, 0.0, 1.0)

        return s.astype(np.float32)

    def _reward(self, prev_state: np.ndarray, action_idx: int) -> float:
        return action_reward(prev_state, action_idx)

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._state = self._random_initial_state()
        self._cpu_baseline = float(self._state[CPU_LOAD])
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
