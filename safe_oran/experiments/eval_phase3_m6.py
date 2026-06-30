"""Re-evaluate S6 M5/M6 checkpoints for the Phase3 M6 closed-loop comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from safe_oran.envs.legacy import PROJECT_ROOT
from safe_oran.experiments.m3_m5_common import (
    MODEL_DIR,
    RUNS_DIR,
    evaluate_model,
    norm_from_vecnormalize,
    write_json,
)

OUT_DIR = PROJECT_ROOT / "04_results" / "phase3_m6"
RUNS_OUT_DIR = OUT_DIR / "runs"
SCENARIO = "S6_moderate_decay"
METHODS = ("M5_constraint_aware", "M6_field_CER_z")
SEEDS = (42, 43, 44)
DEFAULT_Z_CACHE = OUT_DIR / "cer_z_cache__Qwen3-4B__bge.json"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def _load_model(method: str, seed: int):
    from stable_baselines3 import PPO

    tag = f"ppo_{method}_{SCENARIO}_s{seed}"
    model_path = MODEL_DIR / f"{tag}.zip"
    vecnorm_path = MODEL_DIR / f"{tag}_vecnorm.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"missing checkpoint: {model_path}")
    if not vecnorm_path.exists():
        raise FileNotFoundError(f"missing VecNormalize state: {vecnorm_path}")
    return PPO.load(str(model_path), device="cpu"), model_path, vecnorm_path


def evaluate_one(method: str, seed: int, z_cache_path: str, n_eval_episodes: int) -> dict[str, Any]:
    model, model_path, vecnorm_path = _load_model(method, seed)
    use_z_cache = z_cache_path if method == "M6_field_CER_z" else None
    norm_fn = norm_from_vecnormalize(str(vecnorm_path), method, SCENARIO, use_z_cache)
    ev = evaluate_model(
        model,
        norm_fn,
        method,
        SCENARIO,
        n_episodes=n_eval_episodes,
        z_cache_path=use_z_cache,
    )
    source = _load_json(RUNS_DIR / f"ppo_{method}_{SCENARIO}_s{seed}.json")
    source_evidence = source.get("evidence", {})
    evidence = {
        "seed": seed,
        "checkpoint_path": str(model_path),
        "checkpoint_exists": model_path.exists(),
        "checkpoint_bytes": model_path.stat().st_size,
        "vecnorm_path": str(vecnorm_path),
        "n_eval_episodes": ev["n_episodes"],
        "source_run_path": str(RUNS_DIR / f"ppo_{method}_{SCENARIO}_s{seed}.json"),
        "source_total_timesteps_requested": source_evidence.get("total_timesteps_requested"),
        "source_total_timesteps_consumed": source_evidence.get("total_timesteps_consumed"),
        "z_cache_path": use_z_cache or "",
        "z_cache_exists": bool(use_z_cache and Path(use_z_cache).exists()),
    }
    paper_usable = bool(
        source.get("paper_usable", False)
        and evidence["checkpoint_exists"]
        and evidence["checkpoint_bytes"] > 0
        and ev["n_episodes"] >= 5
    )
    return {
        "schema_version": "safe_oran_phase3_m6",
        "kind": "phase3_m6_eval_run",
        "algo": "ppo",
        "method": method,
        "scenario": SCENARIO,
        "seed": seed,
        "quick": bool(source.get("quick", False)),
        "evidence": evidence,
        "train_metrics": source.get("train_metrics", {}),
        "eval_metrics": ev["means"],
        "eval_per_episode": ev["per_episode"],
        "paper_usable": paper_usable,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--z-cache", default=str(DEFAULT_Z_CACHE))
    ap.add_argument("--eval-episodes", type=int, default=5)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = ap.parse_args()

    z_cache_path = Path(args.z_cache)
    if not z_cache_path.exists():
        raise SystemExit(f"missing M6 z-cache: {z_cache_path}")

    RUNS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = []
    for method in METHODS:
        for seed in args.seeds:
            run = evaluate_one(method, int(seed), str(z_cache_path), args.eval_episodes)
            out_path = RUNS_OUT_DIR / f"ppo_{method}_{SCENARIO}_s{seed}.json"
            write_json(out_path, run)
            outputs.append(str(out_path))
    print(json.dumps({
        "runs": outputs,
        "n_runs": len(outputs),
        "z_cache": str(z_cache_path),
        "n_eval_episodes": args.eval_episodes,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
