"""Replay S6 M5 policies under different cached RAG/CER z sources."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from safe_oran.envs.legacy import PROJECT_ROOT
from safe_oran.experiments.build_m6_rag_ablation_caches import ABLATION_ARMS, OUT_DIR, SCENARIO
from safe_oran.experiments.m3_m5_common import MODEL_DIR, aggregate_infos, make_env_fn, norm_from_vecnormalize, write_json

RUNS_DIR = OUT_DIR / "runs"
SEEDS = (42, 43, 44)
EVENT_STEPS = (0, 250)


def _load_model(seed: int):
    from stable_baselines3 import PPO

    tag = f"ppo_M5_constraint_aware_{SCENARIO}_s{seed}"
    model_path = MODEL_DIR / f"{tag}.zip"
    vecnorm_path = MODEL_DIR / f"{tag}_vecnorm.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"missing M5 checkpoint: {model_path}")
    if not vecnorm_path.exists():
        raise FileNotFoundError(f"missing M5 VecNormalize state: {vecnorm_path}")
    return PPO.load(str(model_path), device="cpu"), vecnorm_path, model_path


def _method_for_arm(arm: str) -> str:
    return "M5_constraint_aware" if arm == "oracle_z" else "M6_field_CER_z"


def _cache_for_arm(arm: str) -> str | None:
    if arm == "oracle_z":
        return None
    return str(OUT_DIR / f"z_cache__{arm}.json")


def _eval_one(arm: str, seed: int, n_eval_episodes: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model, vecnorm_path, model_path = _load_model(seed)
    method = _method_for_arm(arm)
    z_cache_path = _cache_for_arm(arm)
    norm_fn = norm_from_vecnormalize(str(vecnorm_path), method, SCENARIO, z_cache_path)
    per_episode = []
    trace_rows = []
    for ep in range(n_eval_episodes):
        eval_seed = 100_000 + ep
        env = make_env_fn(method, SCENARIO, eval_seed, z_cache_path)()
        obs, _ = env.reset(seed=eval_seed)
        infos: list[dict[str, Any]] = []
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(norm_fn(obs), deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            info = dict(info)
            info["reward"] = float(reward)
            infos.append(info)
            if int(info.get("t", -1)) in EVENT_STEPS:
                trace_rows.append({
                    "scenario": SCENARIO,
                    "seed": seed,
                    "episode": ep,
                    "t_event": int(info.get("t", -1)),
                    "retrieval_arm": arm,
                    "p_min": int(info.get("p_min", 0)),
                    "p_min_oracle": int(info.get("p_min_oracle", 0)),
                    "delta_p_min": int(info.get("delta_p_min_vs_oracle", 0)),
                    "under_reservation": max(0, -int(info.get("delta_p_min_vs_oracle", 0))),
                    "over_reservation": max(0, int(info.get("delta_p_min_vs_oracle", 0))),
                    "fallback_used": str(bool(info.get("z_fallback", False))).lower(),
                    "unsafe_under_reservation": str(bool(info.get("unsafe_under_reservation", False))).lower(),
                    "D_proj": int(info.get("D_proj", 0)),
                    "reward": float(reward),
                    "violation": str(bool(info.get("urllc_violation", False))).lower(),
                    "z_mode": info.get("z_mode", ""),
                    "verifier_on": str(bool(info.get("verifier_on", False))).lower(),
                })
        per_episode.append(aggregate_infos(infos, SCENARIO))
        env.close()
    keys = per_episode[0].keys()
    means = {}
    for key in keys:
        vals = [ep[key] for ep in per_episode if ep[key] is not None]
        means[key] = None if not vals else float(sum(vals) / len(vals))
    run = {
        "schema_version": "safe_oran_m6_rag_ablation",
        "kind": "m6_rag_ablation_eval_run",
        "scenario": SCENARIO,
        "retrieval_arm": arm,
        "policy_method": "M5_constraint_aware",
        "env_method": method,
        "seed": seed,
        "quick": False,
        "evidence": {
            "checkpoint_path": str(model_path),
            "checkpoint_exists": model_path.exists(),
            "checkpoint_bytes": model_path.stat().st_size,
            "vecnorm_path": str(vecnorm_path),
            "z_cache_path": z_cache_path or "",
            "z_cache_exists": bool(z_cache_path and Path(z_cache_path).exists()),
            "n_eval_episodes": n_eval_episodes,
            "no_training": True,
            "no_llm_calls": True,
        },
        "eval_metrics": means,
        "eval_per_episode": per_episode,
        "paper_usable": bool(n_eval_episodes >= 5 and model_path.exists() and model_path.stat().st_size > 0),
    }
    return run, trace_rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-episodes", type=int, default=5)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = ap.parse_args()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    outputs = []
    replay_trace: list[dict[str, Any]] = []
    for arm in ABLATION_ARMS:
        cache_path = _cache_for_arm(arm)
        if cache_path and not Path(cache_path).exists():
            raise SystemExit(f"missing z-cache for {arm}: {cache_path}")
        for seed in args.seeds:
            run, rows = _eval_one(arm, int(seed), args.eval_episodes)
            out_path = RUNS_DIR / f"ppo_M5policy_{arm}_{SCENARIO}_s{seed}.json"
            write_json(out_path, run)
            outputs.append(str(out_path))
            replay_trace.extend(rows)

    trace_path = OUT_DIR / "m6_replay_trace.csv"
    if replay_trace:
        with trace_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(replay_trace[0].keys()), lineterminator="\n")
            writer.writeheader()
            writer.writerows(replay_trace)
    print(json.dumps({
        "n_runs": len(outputs),
        "runs": outputs,
        "replay_trace": str(trace_path),
        "n_eval_episodes": args.eval_episodes,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
