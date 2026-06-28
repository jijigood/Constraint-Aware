"""
Shared helpers for Phase 1 DRL baselines: env factories, VecNormalize stat loading, deterministic
component-metric eval (manual normalization for full seed control), bootstrap CIs, and the honest
`paper_usable` gate. Imported by train_baselines.py and eval_baselines.py.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

# project paths
DRL_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(DRL_DIR)
PROJ = os.path.dirname(CODE_DIR)
ENV_DIR = os.path.join(CODE_DIR, "env")
sys.path.insert(0, ENV_DIR)

from slicing_env import EnvConfig, SLICES, jain  # noqa: E402
from slicing_gym_env import SlicingGymEnv, make_shield  # noqa: E402

from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

PHASE0_JSON = os.path.join(PROJ, "04_results", "phase0_headroom.json")
SCHEMA_VERSION = "safe_drl_v1"
REGIMES = ["high_embb", "high_urllc"]
SHIELDS = ["none", "static", "oracle_margin"]
RELIABILITY = 0.99


def static_floor(regime: str) -> int:
    """Best static URLLC PRB reservation for a regime, read from the Phase-0 oracle result."""
    with open(PHASE0_JSON) as f:
        d = json.load(f)
    return int(d["regimes"][regime]["safe_static_p"])


def make_env_fn(regime: str, shield_name: str, floor: int, seed: int, reliability: float = RELIABILITY):
    """Module-level factory -> callable building a Monitor(SlicingGymEnv) (cloudpicklable for VecEnv)."""
    def _f():
        sfn = make_shield(shield_name, static_floor=floor, reliability=reliability)
        env = SlicingGymEnv(EnvConfig(), regime=regime, shield_fn=sfn, seed=seed)
        return Monitor(env)
    return _f


def load_norm_stats(vecnorm_path: str):
    """Load VecNormalize obs stats so we can normalize obs manually during eval (exact SB3 formula)."""
    dummy = DummyVecEnv([make_env_fn("high_embb", "none", 0, 0)])
    vn = VecNormalize.load(vecnorm_path, dummy)
    mean = vn.obs_rms.mean.astype(np.float64)
    var = vn.obs_rms.var.astype(np.float64)
    clip = float(vn.clip_obs)
    eps = float(vn.epsilon)
    dummy.close()

    def norm(o):
        return np.clip((o.astype(np.float64) - mean) / np.sqrt(var + eps), -clip, clip).astype(np.float32)

    return norm


def _agg(comps: list[dict]) -> dict:
    """Aggregate per-step comp dicts into the same component metrics as run_episode()."""
    prb = np.array([[c["prb_embb"], c["prb_urllc"], c["prb_mmtc"]] for c in comps])
    return {
        "reward": float(np.mean([c["reward"] for c in comps])),
        "urllc_violation_rate": float(np.mean([c["urllc_violation"] for c in comps])),
        "embb_sla_rate": float(np.mean([c["embb_ok"] for c in comps])),
        "mmtc_sla_rate": float(np.mean([c["mmtc_ok"] for c in comps])),
        "fairness": jain(prb.mean(axis=0)),
        "shield_correction_rate": float(np.mean([c.get("shield_corrected", False) for c in comps])),
    }


def deterministic_eval(model, norm_fn, regime: str, shield_name: str, floor: int,
                       n_episodes: int = 20, seed0: int = 100_000, reliability: float = RELIABILITY) -> dict:
    """Run n_episodes of the trained model (deterministic) on the (regime, shield); return per-episode
    component metrics + their means. Manual obs normalization gives exact per-episode seed control."""
    sfn = make_shield(shield_name, static_floor=floor, reliability=reliability)
    per_ep = []
    for ep in range(n_episodes):
        env = SlicingGymEnv(EnvConfig(), regime=regime, shield_fn=sfn, seed=seed0 + ep)
        obs, _ = env.reset(seed=seed0 + ep)
        comps = []
        term = trunc = False
        while not (term or trunc):
            a, _ = model.predict(norm_fn(obs), deterministic=True)
            obs, r, term, trunc, info = env.step(int(a))
            comps.append(info)
        per_ep.append(_agg(comps))
    keys = per_ep[0].keys()
    means = {k: float(np.mean([e[k] for e in per_ep])) for k in keys}
    return {"n_episodes": n_episodes, "means": means, "per_episode": per_ep}


def bootstrap_ci(values, n_boot: int = 2000, alpha: float = 0.05, rng_seed: int = 0):
    """95% bootstrap CI of the mean (honest about small N across seeds)."""
    v = np.asarray(values, dtype=float)
    if len(v) == 1:
        return float(v[0]), float(v[0]), float(v[0])
    rng = np.random.default_rng(rng_seed)
    boots = [float(np.mean(rng.choice(v, size=len(v), replace=True))) for _ in range(n_boot)]
    return float(np.mean(v)), float(np.percentile(boots, 100 * alpha / 2)), float(np.percentile(boots, 100 * (1 - alpha / 2)))


def compute_paper_usable(ev: dict) -> bool:
    """Honest gate from RUNTIME evidence -- never a CLI flag."""
    return bool(
        ev.get("total_timesteps_consumed", 0) >= 0.99 * ev.get("total_timesteps_requested", 1)
        and ev.get("checkpoint_exists")
        and int(ev.get("checkpoint_bytes", 0)) > 0
        and int(ev.get("n_eval_episodes", 0)) >= 20
        and bool(ev.get("lib")) and bool(ev.get("device")) and ev.get("seed") is not None
    )
