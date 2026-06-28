"""
Phase 1 DRL trainer -- ONE run per invocation (algo x regime x shield x seed), so run_phase1.sh can
fan runs across the 48 cores. PPO/DQN on CPU, VecNormalize(obs only), a TrainLogCallback that records
TRAINING-TIME cumulative URLLC violations (the safe-exploration metric), then a deterministic 20-ep
eval from the reloaded checkpoint. Writes a per-run JSON with an honest, runtime-evidence paper_usable.

Example:
  .venv/bin/python 01_code/drl/train_baselines.py --algo ppo --regime high_urllc --shield none --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch

torch.set_num_threads(1)

DRL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DRL_DIR)

import drl_common as C  # noqa: E402
from stable_baselines3 import PPO, DQN  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.utils import safe_mean, set_random_seed  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize  # noqa: E402

ALGOS = {"ppo": PPO, "dqn": DQN}


class TrainLogCallback(BaseCallback):
    """Cumulative training-time URLLC violations + shield corrections + reward curve."""

    def __init__(self, rec_every: int = 5000):
        super().__init__()
        self.cum_viol = self.cum_steps = self.cum_corr = 0
        self.history = []
        self.rec_every = rec_every
        self._last_rec = 0

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "urllc_violation" in info:
                self.cum_viol += int(info["urllc_violation"])
                self.cum_corr += int(info.get("shield_corrected", False))
                self.cum_steps += 1
        if self.num_timesteps - self._last_rec >= self.rec_every:
            self._last_rec = self.num_timesteps
            erm = (safe_mean([e["r"] for e in self.model.ep_info_buffer])
                   if len(self.model.ep_info_buffer) > 0 else float("nan"))
            self.history.append([int(self.num_timesteps), int(self.cum_viol),
                                 int(self.cum_steps), float(erm)])
        return True


def build_venv(algo, regime, shield, floor, seed, n_envs):
    fns = [C.make_env_fn(regime, shield, floor, seed + i) for i in range(n_envs)]
    base = SubprocVecEnv(fns) if (algo == "ppo" and n_envs > 1) else DummyVecEnv(fns)
    base.seed(seed)
    return VecNormalize(base, norm_obs=True, norm_reward=False, clip_obs=10.0)


def make_model(algo, venv, seed):
    if algo == "ppo":
        return PPO("MlpPolicy", venv, device="cpu", seed=seed, verbose=0,
                   n_steps=512, batch_size=512, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                   clip_range=0.2, ent_coef=0.01, learning_rate=3e-4, vf_coef=0.5,
                   max_grad_norm=0.5, policy_kwargs=dict(net_arch=[64, 64]))
    return DQN("MlpPolicy", venv, device="cpu", seed=seed, verbose=0,
               learning_rate=1e-3, buffer_size=100_000, learning_starts=MK.learning_starts,
               batch_size=128, gamma=0.99, train_freq=4, gradient_steps=1,
               target_update_interval=1_000, exploration_fraction=0.2,
               exploration_final_eps=0.05, policy_kwargs=dict(net_arch=[128, 128]))


class MK:  # mutable knob set by main() (keeps make_model signature clean)
    learning_starts = 5_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True, choices=list(ALGOS))
    ap.add_argument("--regime", required=True, choices=C.REGIMES)
    ap.add_argument("--shield", required=True, choices=C.SHIELDS)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--models-dir", default=os.path.join(C.PROJ, "02_models"))
    ap.add_argument("--out-dir", default=os.path.join(C.PROJ, "04_results", "phase1", "runs"))
    args = ap.parse_args()

    if args.quick:
        args.timesteps = min(args.timesteps, 30_000)
        args.n_envs = min(args.n_envs, 4)
        MK.learning_starts = 1_000
    if args.algo == "dqn":
        args.n_envs = 1

    set_random_seed(args.seed)
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    floor = C.static_floor(args.regime)
    tag = f"{args.algo}_{args.regime}_{args.shield}_s{args.seed}"
    ckpt = os.path.join(args.models_dir, f"{tag}.zip")
    vnorm = os.path.join(args.models_dir, f"{tag}_vecnorm.pkl")
    print(f"[train] {tag} timesteps={args.timesteps} n_envs={args.n_envs} floor={floor}")

    t0 = time.time()
    venv = build_venv(args.algo, args.regime, args.shield, floor, args.seed, args.n_envs)
    model = make_model(args.algo, venv, args.seed)
    cb = TrainLogCallback()
    model.learn(total_timesteps=args.timesteps, callback=cb, progress_bar=False)
    consumed = int(model.num_timesteps)
    model.save(ckpt)
    venv.save(vnorm)
    venv.close()
    wall = time.time() - t0

    # reload from disk (proves the checkpoint is real + loadable) and eval deterministically
    model = ALGOS[args.algo].load(ckpt, device="cpu")
    norm_fn = C.load_norm_stats(vnorm)
    ev_eval = C.deterministic_eval(model, norm_fn, args.regime, args.shield, floor, n_episodes=20)

    lib = {"stable_baselines3": __import__("stable_baselines3").__version__,
           "gymnasium": __import__("gymnasium").__version__,
           "torch": torch.__version__, "numpy": np.__version__}
    evidence = {
        "total_timesteps_requested": args.timesteps, "total_timesteps_consumed": consumed,
        "checkpoint_path": ckpt, "checkpoint_exists": os.path.exists(ckpt),
        "checkpoint_bytes": os.path.getsize(ckpt) if os.path.exists(ckpt) else 0,
        "vecnorm_path": vnorm, "n_eval_episodes": ev_eval["n_episodes"], "wall_seconds": round(wall, 1),
        "train_cum_urllc_violations": cb.cum_viol, "train_steps": cb.cum_steps,
        "train_shield_corrections": cb.cum_corr, "lib": lib, "device": "cpu", "seed": args.seed,
    }
    result = {
        "schema_version": C.SCHEMA_VERSION, "kind": "drl_baseline_run",
        "algo": args.algo, "regime": args.regime, "shield": args.shield, "seed": args.seed,
        "static_floor": floor, "device": "cpu", "lib": lib, "evidence": evidence,
        "train_metrics": {
            "reward_curve": [[h[0], h[3]] for h in cb.history],
            "safety_curve": [[h[0], h[1], h[2]] for h in cb.history],
            "train_violation_rate": cb.cum_viol / max(cb.cum_steps, 1),
            "train_shield_correction_rate": cb.cum_corr / max(cb.cum_steps, 1),
            "final_train_ep_rew": cb.history[-1][3] if cb.history else None,
        },
        "eval_metrics": ev_eval["means"], "eval_per_episode": ev_eval["per_episode"],
        "paper_usable": C.compute_paper_usable(evidence),
    }
    out = os.path.join(args.out_dir, f"{tag}.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    m = ev_eval["means"]
    print(f"[done] {tag} wall={wall:.0f}s consumed={consumed} "
          f"eval_reward={m['reward']:.3f} urllc_viol={m['urllc_violation_rate']:.3f} "
          f"train_viol_rate={result['train_metrics']['train_violation_rate']:.3f} "
          f"paper_usable={result['paper_usable']} -> {out}")


if __name__ == "__main__":
    main()
