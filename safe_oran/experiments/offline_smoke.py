"""Offline validation for the V4 refactor.

No LLM calls, no DRL training. This checks compatibility, solver equivalence,
counterfactual reconstruction, wrapper timing, and fail-closed behavior.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from safe_oran.constraints import DeterministicSolver, Verifier, oracle_spec
from safe_oran.constraints.z_source import ZCache
from safe_oran.envs.factory import make_constraint_env
from safe_oran.envs.legacy import EnvConfig, PROJECT_ROOT, SlicingGymEnv, ensure_legacy_paths
from safe_oran.rl import ConstraintAwareWrapper


def _legacy_oracle_formula(state: dict, cfg: EnvConfig | None = None, reliability: float = 0.99) -> int:
    cfg = cfg or EnvConfig()
    z = 1.0 + 1.5 * reliability
    g_pess = max(0.2, cfg.channel_mean - cfg.channel_amp - z * cfg.channel_noise)
    need = (state["demand"]["urllc"] + state["backlog"]["urllc"]) / max(cfg.se["urllc"] * g_pess, 1e-6)
    return int(min(cfg.n_prb, math.ceil(math.ceil(need) / cfg.prb_step) * cfg.prb_step))


def _state_files() -> list[Path]:
    return [
        PROJECT_ROOT / "04_results" / "phase2a" / "states_cross.json",
        PROJECT_ROOT / "04_results" / "phase2a" / "states_high_embb.json",
        PROJECT_ROOT / "04_results" / "phase2a" / "states_high_urllc.json",
    ]


def check_solver_equivalence() -> dict:
    solver = DeterministicSolver(EnvConfig())
    checked = 0
    for path in _state_files():
        data = json.loads(path.read_text())
        for state in data["states"]:
            new = solver.solve(oracle_spec(0.99), state).p_min
            old = _legacy_oracle_formula(state, reliability=0.99)
            if new != old:
                raise AssertionError(f"solver mismatch {path.name}: new={new} old={old}")
            checked += 1
    return {"states": checked, "passed": True}


def check_counterfactual_reproduction() -> dict:
    ensure_legacy_paths(include_rag=True)
    import counterfactual as cf  # noqa: PLC0415

    out = {}
    for path in _state_files():
        data = json.loads(path.read_text())
        res = cf.reproduce_check(data["states"])
        if not res["passed"]:
            raise AssertionError(f"counterfactual reproduction failed: {path.name} {res}")
        out[data["set"]] = res
    return out


def check_wrapper() -> dict:
    cfg = EnvConfig()
    env = make_constraint_env("M5_constraint_aware", "high_urllc", seed=123, cfg=cfg)
    obs, _ = env.reset(seed=123)
    if obs.shape[0] != 9:
        raise AssertionError(f"augmented obs dim {obs.shape[0]} != 9")
    for _ in range(20):
        obs, _, terminated, truncated, info = env.step(env.action_space.sample())
        if not isinstance(info["p_min"], int):
            raise AssertionError("p_min is not int")
        expected = info["p_min_next"] / cfg.n_prb
        if abs(float(obs[-1]) - expected) > 1e-6:
            raise AssertionError(f"obs p_min mismatch: {obs[-1]} vs {expected}")
        if terminated or truncated:
            obs, _ = env.reset()

    static_env = ConstraintAwareWrapper(
        SlicingGymEnv(cfg, regime="high_embb", shield_fn=None, seed=321),
        solver=DeterministicSolver(cfg),
        verifier=Verifier(),
        z_cache=ZCache(),
        scenario="high_embb",
        use_shield=True,
        static_p_min=50,
        use_state_aug=False,
    )
    obs_static, _ = static_env.reset(seed=321)
    if obs_static.shape[0] != 8:
        raise AssertionError(f"static obs dim {obs_static.shape[0]} != 8")

    none_env = ConstraintAwareWrapper(
        SlicingGymEnv(cfg, regime="high_embb", shield_fn=None, seed=654),
        use_shield=False,
        use_state_aug=False,
    )
    obs_none, _ = none_env.reset(seed=654)
    if obs_none.shape[0] != 8:
        raise AssertionError(f"no-shield obs dim {obs_none.shape[0]} != 8")
    return {"aug_dim": 9, "base_dim": 8, "steps": 20, "passed": True}


def check_factory() -> dict:
    for method in ("M1_vanilla", "M2_static", "M3_dynamic_no_aug", "M5_constraint_aware", "M6_full_cer"):
        env = make_constraint_env(method, "S4_sla_upgrade", seed=777)
        obs, _ = env.reset(seed=777)
        expected = 9 if env.use_state_aug else 8
        if obs.shape[0] != expected:
            raise AssertionError(f"{method} obs dim {obs.shape[0]} != {expected}")
    return {"methods": 5, "scenario": "S4_sla_upgrade", "passed": True}


def check_fail_closed() -> dict:
    verifier = Verifier()
    bad_spec = {
        "formula_id": "load_backlog_over_spectral_efficiency",
        "reliability_target": 1.2,
        "channel_margin_policy": "nominal",
        "service_rule": "serve_offered_plus_backlog",
        "priority_rank": 1,
        "urllc_min_prb": 10,
    }
    result = verifier.verify(bad_spec, [], {}, z_mode="cer")
    if result.passed:
        raise AssertionError("direct numeric / out-of-range spec incorrectly passed verifier")
    state = json.loads(_state_files()[0].read_text())["states"][0]
    fallback = verifier.fail_closed_spec(0.99)
    p_min = DeterministicSolver(EnvConfig()).solve(fallback, state).p_min
    if p_min == bad_spec["urllc_min_prb"]:
        raise AssertionError("fail-closed path reused direct numeric PRB")
    return {"verifier_reason": result.reason, "fallback_p_min": p_min, "passed": True}


def main() -> int:
    report = {
        "solver_equivalence": check_solver_equivalence(),
        "counterfactual_reproduction": check_counterfactual_reproduction(),
        "wrapper": check_wrapper(),
        "factory": check_factory(),
        "fail_closed": check_fail_closed(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    print("V4 offline smoke PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
