"""Shared helpers for Phase3 M3/M5 training and evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from safe_oran.envs.factory import make_constraint_env
from safe_oran.envs.legacy import PROJECT_ROOT
from safe_oran.experiments.configs import SCENARIOS

METHODS = ("M3_dynamic_no_aug", "M5_constraint_aware")
SCENARIOS_PHASE3 = ("S3_channel_decay", "S4_sla_upgrade", "S5_combined")
OUT_DIR = PROJECT_ROOT / "04_results" / "phase3_m3_m5"
RUNS_DIR = OUT_DIR / "runs"
MODEL_DIR = PROJECT_ROOT / "02_models" / "phase3_m3_m5"


def make_env_fn(method: str, scenario: str, seed: int):
    def _init():
        return make_constraint_env(method, scenario, seed=seed)

    return _init


def norm_from_vecnormalize(vecnorm_path: str, method: str, scenario: str):
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    dummy = DummyVecEnv([make_env_fn(method, scenario, 0)])
    vn = VecNormalize.load(vecnorm_path, dummy)
    mean = vn.obs_rms.mean.astype(np.float64)
    var = vn.obs_rms.var.astype(np.float64)
    clip = float(vn.clip_obs)
    eps = float(vn.epsilon)
    dummy.close()

    def norm(obs):
        return np.clip((obs.astype(np.float64) - mean) / np.sqrt(var + eps), -clip, clip).astype(np.float32)

    return norm


def aggregate_infos(infos: list[dict[str, Any]], scenario: str) -> dict[str, Any]:
    if not infos:
        return {}
    dproj = np.asarray([i.get("D_proj", 0) for i in infos], dtype=float)
    p_min = np.asarray([i.get("p_min", 0) for i in infos], dtype=float)
    rewards = np.asarray([i.get("reward", 0.0) for i in infos], dtype=float)
    violation = np.asarray([int(bool(i.get("urllc_violation", False))) for i in infos], dtype=float)
    corrected = np.asarray([int(bool(i.get("shield_corrected", False))) for i in infos], dtype=float)
    return {
        "reward": float(rewards.mean()),
        "urllc_violation_rate": float(violation.mean()),
        "mean_D_proj": float(dproj.mean()),
        "p95_D_proj": float(np.percentile(dproj, 95)),
        "shield_correction_rate": float(corrected.mean()),
        "mean_p_min": float(p_min.mean()),
        "p95_p_min": float(np.percentile(p_min, 95)),
        "adaptation_delay": adaptation_delay(infos, scenario),
    }


def adaptation_delay(infos: list[dict[str, Any]], scenario: str, stable_window: int = 10) -> int | None:
    schedule = SCENARIOS[scenario]["sla_schedule"]
    change_points = sorted(t for t in schedule if t > 0)
    if not change_points:
        return None
    t0 = change_points[0]
    for idx, info in enumerate(infos):
        t = int(info.get("t", idx))
        if t < t0:
            continue
        window = [x for x in infos[idx: idx + stable_window] if int(x.get("t", 0)) >= t0]
        if len(window) < stable_window:
            break
        if all(int(x.get("D_proj", 0)) == 0 for x in window):
            return int(t - t0)
    return None


def evaluate_model(model, norm_fn, method: str, scenario: str, *, n_episodes: int = 5, seed0: int = 100_000):
    per_episode = []
    for ep in range(n_episodes):
        env = make_constraint_env(method, scenario, seed=seed0 + ep)
        obs, _ = env.reset(seed=seed0 + ep)
        infos = []
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(norm_fn(obs), deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            info = dict(info)
            info["reward"] = float(reward)
            infos.append(info)
        per_episode.append(aggregate_infos(infos, scenario))
    keys = per_episode[0].keys()
    means = {}
    for key in keys:
        values = [ep[key] for ep in per_episode if ep[key] is not None]
        means[key] = None if not values else float(np.mean(values))
    return {"n_episodes": n_episodes, "means": means, "per_episode": per_episode}


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))

