"""
Phase 1 smoke test (the user's #1 safeguard): validate the Gymnasium wrapper BEFORE any training.
Exits non-zero on any failure. Four blocks:
  1. SB3 check_env compliance.
  2. Observation containment + no-NaN across regimes x policies; print per-dim obs min/max.
  3. BIT-FOR-BIT parity vs raw run_episode over {regimes} x {policies} x {shields} x {seeds}.
  4. VecEnv sanity (SubprocVecEnv with a shield closure -> catches pickling issues).

Run:  ~/safe_drl_oran/.venv/bin/python 01_code/smoke_test.py
"""
from __future__ import annotations

import os
import sys
import traceback

import numpy as np

ENV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "env")
sys.path.insert(0, ENV_DIR)

from slicing_env import (  # noqa: E402
    EnvConfig, SLICES, POLICIES, run_episode, jain,
    shield_none, shield_static, shield_dynamic_oracle, shield_dynamic_oracle_margin,
)
from slicing_gym_env import SlicingGymEnv  # noqa: E402

REGIMES = ["balanced", "high_embb", "high_urllc", "bursty"]
POLICY_NAMES = ["random", "throughput_greedy", "reward_greedy"]
SHIELDS = {
    "none": shield_none,
    "static50": shield_static(50),
    "oracle_min": shield_dynamic_oracle,
    "oracle_margin": shield_dynamic_oracle_margin(0.99),
}
SEEDS = [42, 43, 44]


# ---- module-level factory so SubprocVecEnv can cloudpickle it (block 4) ----
def make_env_fn(regime, seed):
    def _f():
        return SlicingGymEnv(EnvConfig(), regime=regime,
                             shield_fn=shield_dynamic_oracle_margin(0.99), seed=seed)
    return _f


def block1_check_env() -> list[str]:
    from stable_baselines3.common.env_checker import check_env
    errs = []
    for regime in REGIMES:
        try:
            check_env(SlicingGymEnv(EnvConfig(), regime=regime, seed=0), warn=True,
                      skip_render_check=True)
        except Exception as e:  # noqa: BLE001
            errs.append(f"check_env[{regime}] raised: {e}")
    return errs


def block2_containment() -> list[str]:
    errs = []
    gmin = np.full(8, np.inf)
    gmax = np.full(8, -np.inf)
    for regime in REGIMES:
        for pname in POLICY_NAMES:
            env = SlicingGymEnv(EnvConfig(), regime=regime, seed=7)
            rng = np.random.default_rng(123)
            for ep in range(5):
                obs, _ = env.reset(seed=7 + ep)
                steps, trunc_count = 0, 0
                term = trunc = False
                while not (term or trunc):
                    if not env.observation_space.contains(obs):
                        bad = np.where((obs < env.observation_space.low) |
                                       (obs > env.observation_space.high))[0]
                        errs.append(f"OOB obs [{regime}/{pname}] dims={bad.tolist()} obs={obs.tolist()}")
                    if not np.isfinite(obs).all():
                        errs.append(f"non-finite obs [{regime}/{pname}]: {obs.tolist()}")
                    gmin = np.minimum(gmin, obs)
                    gmax = np.maximum(gmax, obs)
                    a = POLICIES[pname](env.inner, obs, rng)
                    obs, r, term, trunc, info = env.step(a)
                    steps += 1
                    trunc_count += int(trunc)
                    if not np.isfinite(r):
                        errs.append(f"non-finite reward [{regime}/{pname}]")
                    if not (isinstance(term, bool) and isinstance(trunc, bool)):
                        errs.append(f"term/trunc not bool [{regime}/{pname}]")
                if steps != EnvConfig().episode_len:
                    errs.append(f"episode len {steps} != {EnvConfig().episode_len} [{regime}/{pname}]")
                if trunc_count != 1:
                    errs.append(f"truncated fired {trunc_count}x (expect 1) [{regime}/{pname}]")
    labels = ["d_embb", "d_urllc", "d_mmtc", "bk_embb", "bk_urllc", "bk_mmtc", "chan", "t_frac"]
    print("  per-dim obs range (min / max):")
    for i, lab in enumerate(labels):
        print(f"    {lab:<8} {gmin[i]:>10.2f} / {gmax[i]:>12.2f}")
    return errs


def _wrapped_episode(regime, pname, shield_fn, seed):
    """Replicate run_episode's loop through the WRAPPER (policy emits raw action; wrapper shields)."""
    env = SlicingGymEnv(EnvConfig(), regime=regime, shield_fn=shield_fn, seed=seed)
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed + 9999)   # MUST match run_episode
    rew, viol, embb_ok, mmtc_ok, prb = [], [], [], [], []
    term = trunc = False
    while not (term or trunc):
        a = POLICIES[pname](env.inner, obs, rng)
        obs, r, term, trunc, info = env.step(a)
        rew.append(r)
        viol.append(info["urllc_violation"])
        embb_ok.append(info["embb_ok"])
        mmtc_ok.append(info["mmtc_ok"])
        prb.append([info["prb_embb"], info["prb_urllc"], info["prb_mmtc"]])
    prb = np.array(prb)
    return {
        "reward": float(np.mean(rew)),
        "urllc_violation_rate": float(np.mean(viol)),
        "embb_sla_rate": float(np.mean(embb_ok)),
        "mmtc_sla_rate": float(np.mean(mmtc_ok)),
        "fairness": jain(prb.mean(axis=0)),
    }


def block3_parity() -> list[str]:
    errs = []
    worst = 0.0
    n = 0
    for regime in REGIMES:
        for pname in POLICY_NAMES:
            for sname, sfn in SHIELDS.items():
                for seed in SEEDS:
                    raw = run_episode(EnvConfig(), regime, POLICIES[pname], sfn, seed)
                    wrp = _wrapped_episode(regime, pname, sfn, seed)
                    for k in raw:
                        d = abs(raw[k] - wrp[k])
                        worst = max(worst, d)
                        if d >= 1e-9:
                            errs.append(f"parity MISS [{regime}/{pname}/{sname}/s{seed}] {k}: "
                                        f"raw={raw[k]:.10f} wrp={wrp[k]:.10f} d={d:.2e}")
                    n += 1
    print(f"  parity cells checked: {n}  worst |raw-wrapped| = {worst:.2e}  (threshold 1e-9)")
    return errs


def block4_vecenv() -> list[str]:
    from stable_baselines3.common.vec_env import SubprocVecEnv
    errs = []
    try:
        venv = SubprocVecEnv([make_env_fn("high_urllc", 100 + i) for i in range(4)])
        venv.reset()
        for _ in range(50):
            acts = [venv.action_space.sample() for _ in range(4)]
            obs, rews, dones, infos = venv.step(np.array(acts))
            if obs.shape != (4, 8):
                errs.append(f"vec obs shape {obs.shape} != (4,8)")
                break
        venv.close()
    except Exception as e:  # noqa: BLE001
        errs.append(f"SubprocVecEnv failed: {e}\n{traceback.format_exc()}")
    return errs


def main():
    print("=" * 70)
    print("PHASE 1 SMOKE TEST")
    print("=" * 70)
    all_errs = []
    for name, fn in [("1. check_env", block1_check_env),
                     ("2. containment/no-NaN", block2_containment),
                     ("3. bit-for-bit parity", block3_parity),
                     ("4. VecEnv sanity", block4_vecenv)]:
        print(f"\n[{name}]")
        errs = fn()
        if errs:
            print(f"  FAIL ({len(errs)}):")
            for e in errs[:10]:
                print(f"    - {e}")
            all_errs += errs
        else:
            print("  PASS")
    print("\n" + "=" * 70)
    if all_errs:
        print(f"SMOKE TEST FAILED ({len(all_errs)} issue(s)) -- DO NOT TRAIN")
        print("=" * 70)
        sys.exit(1)
    print("SMOKE TEST PASSED -- wrapper is faithful; safe to train")
    print("=" * 70)


if __name__ == "__main__":
    main()
