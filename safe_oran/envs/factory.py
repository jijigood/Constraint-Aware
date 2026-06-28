"""Factories that bind V4 scenario/baseline configs to legacy env dynamics."""

from __future__ import annotations

from dataclasses import replace

from safe_oran.constraints import DeterministicSolver, Verifier, ZCache
from safe_oran.experiments.configs import BASELINE_CONFIGS, SCENARIOS
from safe_oran.rl import ConstraintAwareWrapper

from .legacy import EnvConfig, SlicingGymEnv
from .scenario import ScenarioGymEnv


def scenario_env_config(cfg: EnvConfig | None, scenario_cfg: dict) -> EnvConfig:
    env_cfg = cfg or EnvConfig()
    if "episode_len" in scenario_cfg:
        env_cfg = replace(env_cfg, episode_len=int(scenario_cfg["episode_len"]))
    return env_cfg


def make_legacy_env(scenario: str, seed: int = 0, cfg: EnvConfig | None = None) -> SlicingGymEnv:
    """Build an unshielded legacy Gym env for a named V4 scenario."""
    if scenario not in SCENARIOS:
        raise KeyError(f"unknown scenario: {scenario}")
    scenario_cfg = SCENARIOS[scenario]
    env_cfg = scenario_env_config(cfg, scenario_cfg)
    regime = scenario_cfg["legacy_regime"]
    return ScenarioGymEnv(env_cfg, regime=regime, scenario_cfg=scenario_cfg, shield_fn=None, seed=seed)


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
    scenario_cfg = SCENARIOS[scenario]
    env_cfg = scenario_env_config(cfg, scenario_cfg)
    base = make_legacy_env(scenario, seed=seed, cfg=env_cfg)
    bl_cfg = dict(BASELINE_CONFIGS[method])
    return ConstraintAwareWrapper(
        base,
        solver=DeterministicSolver(env_cfg),
        verifier=Verifier(),
        z_cache=z_cache or ZCache(),
        scenario=scenario,
        sla_schedule=scenario_cfg["sla_schedule"],
        **bl_cfg,
    )
