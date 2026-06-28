"""Projection from a raw PRB action to a minimum URLLC reservation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _action_table(env) -> np.ndarray:
    if hasattr(env, "actions"):
        return env.actions
    if hasattr(env, "inner") and hasattr(env.inner, "actions"):
        return env.inner.actions
    raise AttributeError("env must expose actions or inner.actions")


def _n_prb(env) -> int:
    if hasattr(env, "cfg"):
        return int(env.cfg.n_prb)
    if hasattr(env, "inner") and hasattr(env.inner, "cfg"):
        return int(env.inner.cfg.n_prb)
    return 100


def project_to_min_urllc(env, action_idx: int, min_urllc: int) -> tuple[int, int]:
    """Return ``(safe_action_idx, L1_projection_distance)``."""
    actions = _action_table(env)
    floor = int(min(_n_prb(env), max(0, min_urllc)))
    raw_idx = int(action_idx)
    raw = actions[raw_idx]
    if raw[1] >= floor:
        return raw_idx, 0
    mask = actions[:, 1] >= floor
    cand = actions[mask]
    distances = np.abs(cand - raw).sum(axis=1)
    best = cand[int(np.argmin(distances))]
    safe_idx = int(np.where((actions == best).all(axis=1))[0][0])
    return safe_idx, int(np.abs(best - raw).sum())


@dataclass
class ReservationShield:
    env: object

    def project(self, action_idx: int, min_urllc: int) -> tuple[int, int]:
        return project_to_min_urllc(self.env, action_idx, min_urllc)

