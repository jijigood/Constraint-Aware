"""Deterministic safety solver: typed spec + state -> URLLC PRB floor."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from safe_oran.envs.legacy import EnvConfig

from .spec import ConstraintSpec, DEFAULT_FORMULA_ID, fallback_spec, oracle_spec


@dataclass(frozen=True)
class ReservationResult:
    p_min: int
    infeasible: bool
    reason: str
    meta: dict[str, Any] = field(default_factory=dict)


def _cfg(cfg: EnvConfig | None = None) -> EnvConfig:
    return cfg or EnvConfig()


def _urllc_load(state: dict[str, Any]) -> tuple[float, float]:
    if "demand" in state and "backlog" in state:
        return float(state["demand"]["urllc"]), float(state["backlog"]["urllc"])
    if "d_u" in state or "b_u" in state:
        return float(state.get("d_u", 0.0)), float(state.get("b_u", 0.0))
    if "d" in state and "b" in state:
        d, b = state["d"], state["b"]
        return float(d.get("u", d.get("urllc", 0.0))), float(b.get("u", b.get("urllc", 0.0)))
    raise KeyError("state must contain URLLC demand/backlog")


def _state_channel(state: dict[str, Any], cfg: EnvConfig) -> float:
    for key in ("channel", "g_t", "g", "last_channel"):
        if key in state:
            return float(state[key])
    return float(cfg.channel_mean)


def pessimistic_channel(cfg: EnvConfig, reliability: float) -> float:
    """Legacy oracle-margin channel model, preserved bit-for-bit in intent."""
    z = 1.0 + 1.5 * float(reliability)
    return float(max(0.2, cfg.channel_mean - cfg.channel_amp - z * cfg.channel_noise))


def margin_channel(spec: ConstraintSpec, state: dict[str, Any], cfg: EnvConfig) -> float:
    policy = spec.channel_margin_policy
    if policy == "nominal":
        return max(0.2, _state_channel(state, cfg))
    if policy == "pessimistic_quantile":
        return pessimistic_channel(cfg, spec.reliability_target)
    if policy == "worst_case":
        return 0.2
    raise ValueError(f"unsupported channel_margin_policy: {policy}")


def _ceil_to_prb_step(raw_prb: float, cfg: EnvConfig) -> tuple[int, bool]:
    infeasible = raw_prb > cfg.n_prb + 1e-9
    capped_ceiled = int(min(cfg.n_prb, math.ceil(raw_prb)))
    quantized = int(math.ceil(capped_ceiled / cfg.prb_step) * cfg.prb_step)
    return int(min(cfg.n_prb, quantized)), bool(infeasible)


class DeterministicSolver:
    """Formula whitelist dispatcher for executable safety constraints."""

    def __init__(self, cfg: EnvConfig | None = None):
        self.cfg = _cfg(cfg)

    def solve(self, spec: ConstraintSpec | dict[str, Any], state: dict[str, Any]) -> ReservationResult:
        z = spec if isinstance(spec, ConstraintSpec) else ConstraintSpec.from_mapping(spec)
        if z.formula_id != DEFAULT_FORMULA_ID:
            raise ValueError(f"unsupported formula_id: {z.formula_id}")

        d_u, b_u = _urllc_load(state)
        g_pess = margin_channel(z, state, self.cfg)
        se_u = float(self.cfg.se["urllc"])
        raw_prb = (d_u + b_u) / max(se_u * g_pess, 1e-6)
        p_min, infeasible = _ceil_to_prb_step(raw_prb, self.cfg)
        return ReservationResult(
            p_min=p_min,
            infeasible=infeasible,
            reason="deterministic_solver",
            meta={
                "d_urllc": d_u,
                "b_urllc": b_u,
                "se_urllc": se_u,
                "g_pess": g_pess,
                "raw_prb": raw_prb,
                "formula_id": z.formula_id,
                "channel_margin_policy": z.channel_margin_policy,
                "reliability_target": z.reliability_target,
            },
        )

    def compute(self, spec: ConstraintSpec | dict[str, Any], state: dict[str, Any]) -> tuple[int, bool]:
        result = self.solve(spec, state)
        return result.p_min, result.infeasible

    def fallback_compute(self, state: dict[str, Any], reliability: float = 0.99) -> tuple[int, bool]:
        result = self.solve(fallback_spec(reliability), state)
        return result.p_min, result.infeasible


def oracle_reservation(
    state: dict[str, Any],
    cfg: EnvConfig | None = None,
    reliability: float = 0.99,
) -> int:
    """Compatibility helper matching the legacy oracle-margin reservation."""
    return DeterministicSolver(_cfg(cfg)).solve(oracle_spec(reliability), state).p_min
