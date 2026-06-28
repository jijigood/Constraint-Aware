"""Constraint-aware Gym wrapper with p_min/P_max state augmentation."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from safe_oran.constraints import DeterministicSolver, Verifier
from safe_oran.constraints.spec import ConstraintSpec
from safe_oran.constraints.z_source import ZCache
from safe_oran.shield import project_to_min_urllc


class ConstraintAwareWrapper(gym.Wrapper):
    """V4 unified wrapper for M1-M6 style baselines."""

    def __init__(
        self,
        env: gym.Env,
        solver: DeterministicSolver | None = None,
        verifier: Verifier | None = None,
        z_cache: ZCache | None = None,
        *,
        use_shield: bool = True,
        static_p_min: int | None = None,
        use_state_aug: bool = True,
        proj_penalty: float = 0.0,
        z_mode: str = "oracle",
        verifier_on: bool = True,
        scenario: str = "high_urllc",
        sla_schedule: dict[int, float] | None = None,
    ):
        super().__init__(env)
        self.solver = solver or DeterministicSolver(self._cfg())
        self.verifier = verifier or Verifier()
        self.z_cache = z_cache or ZCache()
        self.use_shield = bool(use_shield)
        self.static_p_min = static_p_min
        self.use_state_aug = bool(use_state_aug)
        self.proj_penalty = float(proj_penalty)
        self.z_mode = z_mode
        self.verifier_on = bool(verifier_on)
        self.scenario = scenario
        self.sla_schedule = sla_schedule or {0: 0.99}
        if 0 not in self.sla_schedule:
            raise ValueError("sla_schedule must include t=0")

        base_space = env.observation_space
        if self.use_state_aug:
            self.observation_space = gym.spaces.Box(
                low=np.append(base_space.low, 0.0).astype(np.float32),
                high=np.append(base_space.high, 1.0).astype(np.float32),
                dtype=np.float32,
            )
        else:
            self.observation_space = base_space

        self.current_sla = float(self.sla_schedule[0])
        self.z_current = self.z_cache.get(self.scenario, 0, self.current_sla)
        self.p_min = 0
        self.prev_g = 0.5
        self.prev_regime = self.scenario

    def _inner(self):
        env = self.env
        while hasattr(env, "env") and not hasattr(env, "inner"):
            env = env.env
        return env.inner if hasattr(env, "inner") else env

    def _cfg(self):
        inner = self._inner() if hasattr(self, "env") else None
        return getattr(inner, "cfg", getattr(self.env, "cfg", None)) if inner is not None else None

    def _pmax(self) -> int:
        return int(self._cfg().n_prb)

    def _obs(self, obs_raw: np.ndarray) -> np.ndarray:
        obs = obs_raw.astype(np.float32)
        if self.use_state_aug:
            return np.append(obs, self.p_min / max(self._pmax(), 1)).astype(np.float32)
        return obs

    def _state(self) -> dict[str, Any]:
        inner = self._inner()
        cfg = inner.cfg
        return {
            "t": int(inner.t),
            "regime": getattr(inner, "regime", self.scenario),
            "demand": {s: float(inner._pending[s]) for s in ("embb", "urllc", "mmtc")},
            "backlog": {s: float(inner.backlog[s]) for s in ("embb", "urllc", "mmtc")},
            "channel": float(getattr(inner, "_last_channel", cfg.channel_mean)),
        }

    def _should_update_z(self, state: dict[str, Any], t: int) -> bool:
        sla_changed = t in self.sla_schedule
        regime_changed = state["regime"] != self.prev_regime
        d_u = state["demand"]["urllc"]
        b_u = state["backlog"]["urllc"]
        backlog_trigger = d_u > 0 and b_u / max(d_u, 1e-6) > 0.5
        g = float(state["channel"])
        channel_cross = (self.prev_g >= 0.3 and g < 0.3) or (self.prev_g < 0.3 and g >= 0.3)
        periodic = t % 200 == 0
        return any([sla_changed, regime_changed, backlog_trigger, channel_cross, periodic])

    def _update_z(self, state: dict[str, Any], t: int) -> None:
        new_z = self.z_cache.get(self.scenario, t, self.current_sla)
        retrieved_ids = new_z.get("retrieved_ids", new_z.get("rag_evidence_ids", []))
        if self.z_mode == "oracle" or not self.verifier_on:
            new_z["verified"] = True
            self.z_current = new_z
            return
        result = self.verifier.verify(new_z, retrieved_ids, state, z_mode=self.z_mode)
        if result.passed and result.spec is not None:
            self.z_current = result.spec.to_dict()

    def _compute_p_min(self, state: dict[str, Any]) -> tuple[int, bool, str]:
        if not self.use_shield:
            return 0, False, "shield_disabled"
        if self.static_p_min is not None:
            return int(self.static_p_min), False, "static"
        retrieved_ids = self.z_current.get("retrieved_ids", self.z_current.get("rag_evidence_ids", []))
        if self.verifier_on:
            result = self.verifier.verify(self.z_current, retrieved_ids, state, z_mode=self.z_mode)
            if not result.passed:
                z = self.verifier.fail_closed_spec(self.current_sla)
                out = self.solver.solve(z, state)
                return out.p_min, out.infeasible, f"fail_closed:{result.reason}"
            spec = result.spec or ConstraintSpec.from_mapping(self.z_current)
        else:
            spec = ConstraintSpec.from_mapping(self.z_current)
        out = self.solver.solve(spec, state)
        return out.p_min, out.infeasible, out.reason

    def reset(self, **kwargs):
        obs_raw, info = self.env.reset(**kwargs)
        self.current_sla = float(self.sla_schedule[0])
        self.z_current = self.z_cache.get(self.scenario, 0, self.current_sla)
        self.prev_g = 0.5
        self.prev_regime = self.scenario
        state = self._state()
        self.p_min, _, _ = self._compute_p_min(state)
        return self._obs(obs_raw), info

    def step(self, action_idx: int):
        state_t = self._state()
        t = int(state_t["t"])
        if t in self.sla_schedule:
            self.current_sla = float(self.sla_schedule[t])
        if self._should_update_z(state_t, t):
            self._update_z(state_t, t)

        self.prev_g = float(state_t["channel"])
        self.prev_regime = state_t["regime"]
        p_min_action, infeasible, pmin_reason = self._compute_p_min(state_t)

        raw_idx = int(action_idx)
        if self.use_shield:
            safe_idx, d_proj = project_to_min_urllc(self._inner(), raw_idx, p_min_action)
        else:
            safe_idx, d_proj = raw_idx, 0

        obs_raw, reward, terminated, truncated, info = self.env.step(safe_idx)
        if self.proj_penalty > 0:
            reward -= self.proj_penalty * d_proj

        if self.use_state_aug:
            state_next = self._state()
            self.p_min, _, _ = self._compute_p_min(state_next)
        else:
            self.p_min = p_min_action

        info.update({
            "raw_idx": raw_idx,
            "safe_idx": int(safe_idx),
            "agent_action": raw_idx,
            "executed_action": int(safe_idx),
            "shield_corrected": bool(safe_idx != raw_idx),
            "p_min": int(p_min_action),
            "p_min_next": int(self.p_min),
            "D_proj": int(d_proj),
            "infeasible": bool(infeasible),
            "pmin_reason": pmin_reason,
            "z_current": self.z_current,
            "sla": self.current_sla,
            "z_mode": self.z_mode,
            "verifier_on": self.verifier_on,
        })
        return self._obs(obs_raw), float(reward), terminated, truncated, info

