"""
Gymnasium wrapper around the pure-numpy SlicingEnv (Phase 1).

Thin adapter: it DELEGATES all dynamics to SlicingEnv (no reimplementation), so the smoke-test
parity check can prove the wrapper reproduces `run_episode` bit-for-bit. A safety `shield_fn` is
applied by PROJECTION inside step() -- the executed action is shield(agent_action), which is the
literal deployment model (a runtime safety filter over a fixed controller) and keeps PPO/DQN on an
identical, unmodified SB3 code path.

Normalization is intentionally NOT done here (it would break parity); do it at the VecEnv layer with
VecNormalize(norm_obs=True, norm_reward=False).
"""
from __future__ import annotations

import os
import sys

import numpy as np

# make `slicing_env` importable regardless of cwd / subprocess worker
_ENV_DIR = os.path.dirname(os.path.abspath(__file__))
if _ENV_DIR not in sys.path:
    sys.path.insert(0, _ENV_DIR)

import gymnasium as gym
from gymnasium import spaces

from slicing_env import (  # noqa: E402
    SlicingEnv, EnvConfig, SLICES,
    shield_none, shield_static, shield_dynamic_oracle, shield_dynamic_oracle_margin,
)

# Generous, fixed observation bounds. They are containment guarantees for check_env, NOT a
# normalizer (VecNormalize handles scaling). We must NOT clip obs to these bounds -- clipping would
# alter values vs the raw env and break the parity test. Backlog can grow large under a starving
# policy, so its ceiling is deliberately high.
OBS_LOW = np.array([0, 0, 0, 0, 0, 0, 0.0, 0.0], dtype=np.float32)
OBS_HIGH = np.array([1000, 1000, 1000, 100000, 100000, 100000, 5.0, 1.0], dtype=np.float32)


def make_shield(name: str, env_cfg: EnvConfig | None = None, static_floor: int = 0,
                reliability: float = 0.99):
    """Resolve a shield name -> shield_fn(inner_env, action_idx) -> action_idx."""
    if name == "none":
        return shield_none
    if name == "static":
        return shield_static(static_floor)
    if name == "oracle_min":
        return shield_dynamic_oracle
    if name == "oracle_margin":
        return shield_dynamic_oracle_margin(reliability)
    raise ValueError(f"unknown shield: {name}")


class SlicingGymEnv(gym.Env):
    """gymnasium.Env adapter for SlicingEnv with an optional action-projection shield."""

    metadata = {"render_modes": []}

    def __init__(self, cfg: EnvConfig | None = None, regime: str = "balanced",
                 shield_fn=None, seed: int | None = None):
        super().__init__()
        self.cfg = cfg or EnvConfig()
        self.regime = regime
        self.shield_fn = shield_fn
        self._seed = seed
        self.inner = SlicingEnv(self.cfg, regime=regime, seed=seed or 0)
        self.action_space = spaces.Discrete(self.inner.n_actions)
        self.observation_space = spaces.Box(low=OBS_LOW, high=OBS_HIGH, shape=(8,), dtype=np.float32)
        self.last_executed_action = None

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        s = seed if seed is not None else self._seed
        obs, info = self.inner.reset(seed=s)
        return obs.astype(np.float32), dict(info)

    def step(self, action):
        agent_a = int(action)
        a = agent_a
        if self.shield_fn is not None:
            a = int(self.shield_fn(self.inner, a))   # project BEFORE inner.step (matches run_episode)
        self.last_executed_action = a
        obs, reward, done, _, comp = self.inner.step(a)
        # no true terminal state -> pure time limit -> truncated, not terminated (correct bootstrap)
        terminated = False
        truncated = bool(done)
        info = dict(comp)
        info["agent_action"] = agent_a
        info["executed_action"] = a
        info["shield_corrected"] = bool(a != agent_a)
        return obs.astype(np.float32), float(reward), terminated, truncated, info
