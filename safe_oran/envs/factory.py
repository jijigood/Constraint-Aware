"""Factories that bind V4 scenario/baseline configs to legacy env dynamics."""

from __future__ import annotations

from safe_oran.constraints import DeterministicSolver, Verifier, ZCache
from safe_oran.experiments.configs import BASELINE_CONFIGS, SCENARIOS
from safe_oran.rl import ConstraintAwareWrapper

from .legacy import EnvConfig, SlicingGymEnv


def make_legacy_env(scenario: str, seed: int = 0, cfg: EnvConfig | None = None) -> SlicingGymEnv:
    """Build an unshielded legacy Gym env for a named V4 scenario."""
    if scenario not in SCENARIOS:
        raise KeyError(f"unknown scenario: {scenario}")
    env_cfg = cfg or EnvConfig()
    regime = SCENARIOS[scenario]["legacy_regime"]
    return SlicingGymEnv(env_cfg, regime=regime, shield_fn=None, seed=seed)


def make_constraint_env(
    method: str,
    scenario: str,
    seed: int = 0,
    *,
    cfg: EnvConfig | None = None,
    z_cache: ZCache | None = None,
) -> ConstraintAwareWrapper:
    """Build a V4 wrapped env without invoking LLM/RAG services."""
    if method not in BASELINE_CONFIGS:
        raise KeyError(f"unknown baseline method: {method}")
    if scenario not in SCENARIOS:
        raise KeyError(f"unknown scenario: {scenario}")
    env_cfg = cfg or EnvConfig()
    base = make_legacy_env(scenario, seed=seed, cfg=env_cfg)
    bl_cfg = dict(BASELINE_CONFIGS[method])
    return ConstraintAwareWrapper(
        base,
        solver=DeterministicSolver(env_cfg),
        verifier=Verifier(),
        z_cache=z_cache or ZCache(),
        scenario=scenario,
        sla_schedule=SCENARIOS[scenario]["sla_schedule"],
        **bl_cfg,
    )

