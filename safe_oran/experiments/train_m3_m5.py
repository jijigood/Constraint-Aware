"""Train one Phase3 M3/M5 PPO run."""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from safe_oran.experiments.m3_m5_common import (
    MODEL_DIR,
    RUNS_DIR,
    SCENARIOS_PHASE3,
    TRAIN_METHODS,
    evaluate_model,
    make_env_fn,
    norm_from_vecnormalize,
    write_json,
)

os.environ.setdefault("OMP_NUM_THREADS", "1")
torch.set_num_threads(1)


class TrainMetricsCallback:
    """Small callback object compatible with SB3 BaseCallback via inheritance."""

    def __init__(self, rec_every: int = 5000):
        from stable_baselines3.common.callbacks import BaseCallback

        class _Callback(BaseCallback):
            def __init__(self, outer):
                super().__init__()
                self.outer = outer

            def _on_step(self) -> bool:
                self.outer.on_step(self)
                return True

        self.callback = _Callback(self)
        self.rec_every = rec_every
        self._last_rec = 0
        self.cum_steps = 0
        self.cum_viol = 0
        self.cum_corr = 0
        self.cum_dproj = 0.0
        self.cum_fallback = 0
        self.cum_unsafe_under = 0
        self.cum_delta_abs = 0.0
        self.cum_delta_under = 0.0
        self.cum_delta_over = 0.0
        self.cum_parity = 0
        self.history = []

    def on_step(self, cb) -> None:
        for info in cb.locals["infos"]:
            if "urllc_violation" in info:
                self.cum_steps += 1
                self.cum_viol += int(bool(info.get("urllc_violation", False)))
                self.cum_corr += int(bool(info.get("shield_corrected", False)))
                self.cum_dproj += float(info.get("D_proj", 0.0))
                self.cum_fallback += int(bool(info.get("z_fallback", False)))
                self.cum_unsafe_under += int(bool(info.get("unsafe_under_reservation", False)))
                delta = float(info.get("delta_p_min_vs_oracle", 0.0))
                self.cum_delta_abs += abs(delta)
                self.cum_delta_under += max(0.0, -delta)
                self.cum_delta_over += max(0.0, delta)
                self.cum_parity += int(delta == 0.0)
        if cb.num_timesteps - self._last_rec >= self.rec_every:
            self._last_rec = cb.num_timesteps
            self.history.append([
                int(cb.num_timesteps),
                int(self.cum_viol),
                int(self.cum_steps),
                float(self.cum_dproj),
            ])


def build_model(env, seed: int, quick: bool):
    from stable_baselines3 import PPO

    return PPO(
        "MlpPolicy",
        env,
        device="cpu",
        seed=seed,
        verbose=0,
        n_steps=128 if quick else 512,
        batch_size=128 if quick else 512,
        n_epochs=5 if quick else 10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        learning_rate=3e-4,
        policy_kwargs=dict(net_arch=[64, 64]),
    )


def main() -> int:
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=TRAIN_METHODS)
    ap.add_argument("--scenario", required=True, choices=SCENARIOS_PHASE3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--eval-episodes", type=int, default=5)
    ap.add_argument("--z-cache", default=None, help="Optional event-level symbolic z-cache JSON.")
    args = ap.parse_args()

    if args.quick:
        args.timesteps = min(args.timesteps, 2_000)
        args.eval_episodes = min(args.eval_episodes, 2)

    set_random_seed(args.seed)
    tag = f"ppo_{args.method}_{args.scenario}_s{args.seed}"
    ckpt = MODEL_DIR / f"{tag}.zip"
    vecnorm_path = MODEL_DIR / f"{tag}_vecnorm.pkl"
    out_path = RUNS_DIR / f"{tag}.json"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    def _env():
        return Monitor(make_env_fn(args.method, args.scenario, args.seed, args.z_cache)())

    venv = VecNormalize(DummyVecEnv([_env]), norm_obs=True, norm_reward=False, clip_obs=10.0)
    observation_dim = int(venv.observation_space.shape[0])
    callback = TrainMetricsCallback(rec_every=500 if args.quick else 5000)
    model = build_model(venv, args.seed, args.quick)
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=callback.callback, progress_bar=False)
    consumed = int(model.num_timesteps)
    model.save(str(ckpt))
    venv.save(str(vecnorm_path))
    venv.close()

    reloaded = PPO.load(str(ckpt), device="cpu")
    norm_fn = norm_from_vecnormalize(str(vecnorm_path), args.method, args.scenario, args.z_cache)
    ev = evaluate_model(
        reloaded,
        norm_fn,
        args.method,
        args.scenario,
        n_episodes=args.eval_episodes,
        z_cache_path=args.z_cache,
    )
    evidence = {
        "total_timesteps_requested": args.timesteps,
        "total_timesteps_consumed": consumed,
        "checkpoint_path": str(ckpt),
        "checkpoint_exists": ckpt.exists(),
        "checkpoint_bytes": ckpt.stat().st_size if ckpt.exists() else 0,
        "vecnorm_path": str(vecnorm_path),
        "n_eval_episodes": ev["n_episodes"],
        "wall_seconds": round(time.time() - t0, 1),
        "seed": args.seed,
        "z_cache_path": args.z_cache or "",
        "z_cache_exists": bool(args.z_cache and os.path.exists(args.z_cache)),
        "observation_dim": observation_dim,
    }
    result = {
        "schema_version": "safe_oran_phase3_m3_m5",
        "kind": "phase3_m3_m5_run",
        "algo": "ppo",
        "method": args.method,
        "scenario": args.scenario,
        "seed": args.seed,
        "quick": bool(args.quick),
        "evidence": evidence,
        "train_metrics": {
            "history": callback.history,
            "train_violation_rate": callback.cum_viol / max(callback.cum_steps, 1),
            "train_shield_correction_rate": callback.cum_corr / max(callback.cum_steps, 1),
            "train_mean_D_proj": callback.cum_dproj / max(callback.cum_steps, 1),
            "train_fallback_rate": callback.cum_fallback / max(callback.cum_steps, 1),
            "train_unsafe_under_reservation_rate": callback.cum_unsafe_under / max(callback.cum_steps, 1),
            "train_p_min_parity_rate": callback.cum_parity / max(callback.cum_steps, 1),
            "train_mean_abs_delta_p_min_vs_oracle": callback.cum_delta_abs / max(callback.cum_steps, 1),
            "train_mean_under_reservation_prb": callback.cum_delta_under / max(callback.cum_steps, 1),
            "train_mean_over_reservation_prb": callback.cum_delta_over / max(callback.cum_steps, 1),
        },
        "eval_metrics": ev["means"],
        "eval_per_episode": ev["per_episode"],
        "paper_usable": bool(
            not args.quick
            and consumed >= 0.99 * args.timesteps
            and evidence["checkpoint_exists"]
            and evidence["checkpoint_bytes"] > 0
            and ev["n_episodes"] >= 5
        ),
    }
    write_json(out_path, result)
    print(f"[done] {tag} -> {out_path}")
    print(result["eval_metrics"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
