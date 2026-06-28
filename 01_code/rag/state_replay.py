"""
Phase 2a state replay (runs in the SB3 venv: ~/safe_drl_oran/.venv).

Rolls the Phase-1 trained PPO-none policy (the realistic unshielded controller a runtime shield would
wrap) on HELD-OUT seeds and dumps per-step states to JSON. Three sets:
  - in-regime high_embb (static floor 50)
  - in-regime high_urllc (static floor 60)
  - CROSS-REGIME: policy trained on high_embb, rolled out in high_urllc, static floor 50 (decisive)
States are stratified by URLLC demand so high-load (hard) steps are well represented -- otherwise the
gate is dominated by trivially-safe low-load steps.

The dump records the REALIZED channel g per step; the offline counterfactual forces that g (monkeypatch),
so no RNG serialization is needed. logged_comp is stored so counterfactual.py can prove bit-for-bit
reproduction with the logged action.

Run:  ~/safe_drl_oran/.venv/bin/python 01_code/rag/state_replay.py
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(RAG_DIR)
sys.path.insert(0, os.path.join(CODE_DIR, "drl"))
sys.path.insert(0, os.path.join(CODE_DIR, "env"))

import drl_common as C  # noqa: E402
from slicing_env import EnvConfig, SLICES, SlicingEnv  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

OUT_DIR = os.path.join(C.PROJ, "04_results", "phase2a")
# (set_name, policy_train_regime, eval_regime, static_floor)
SETS = [
    ("high_embb", "high_embb", "high_embb", 50),
    ("high_urllc", "high_urllc", "high_urllc", 60),
    ("cross", "high_embb", "high_urllc", 50),
]
N_PER_SET = 300
N_EPISODES = 8
SEED0 = 200_000


def min_prb_realized(cfg, d_urllc, bk_urllc, g):
    return int(min(cfg.n_prb, math.ceil((d_urllc + bk_urllc) / max(cfg.se["urllc"] * g, 1e-6))))


def roll_states(policy_tag, eval_regime, static_floor, set_name):
    cfg = EnvConfig()
    model = PPO.load(os.path.join(C.PROJ, "02_models", f"{policy_tag}.zip"), device="cpu")
    norm_fn = C.load_norm_stats(os.path.join(C.PROJ, "02_models", f"{policy_tag}_vecnorm.pkl"))
    states = []
    for ep in range(N_EPISODES):
        seed = SEED0 + ep
        env = SlicingEnv(cfg, regime=eval_regime, seed=seed)
        obs, _ = env.reset(seed=seed)
        done = False
        while not done:
            a, _ = model.predict(norm_fn(obs), deterministic=True)
            a = int(a)
            pre = {
                "set": set_name, "regime": eval_regime, "static_floor": static_floor, "t": int(env.t),
                "demand": {s: float(env._pending[s]) for s in SLICES},
                "backlog": {s: float(env.backlog[s]) for s in SLICES},
                "last_channel": float(getattr(env, "_last_channel", cfg.channel_mean)),
                "obs": [float(x) for x in obs],
                "logged_action_idx": a,
                "logged_alloc": [int(x) for x in env.actions[a]],
            }
            obs, r, done, _, comp = env.step(a)
            g = float(comp["channel"])
            pre["channel"] = g
            pre["min_prb_urllc_realized"] = min_prb_realized(
                cfg, pre["demand"]["urllc"], pre["backlog"]["urllc"], g)
            pre["logged_comp"] = {
                "reward": float(comp["reward"]), "urllc_violation": bool(comp["urllc_violation"]),
                "prb_embb": int(comp["prb_embb"]), "prb_urllc": int(comp["prb_urllc"]),
                "prb_mmtc": int(comp["prb_mmtc"]), "channel": g,
            }
            states.append(pre)
    return states


def stratified_sample(states, n):
    """Systematic sample after sorting by URLLC demand -> uniform coverage of the load range."""
    if len(states) <= n:
        return states
    order = sorted(range(len(states)), key=lambda i: states[i]["demand"]["urllc"])
    idx = np.linspace(0, len(order) - 1, n).round().astype(int)
    return [states[order[i]] for i in idx]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cfg = EnvConfig()
    summary = {}
    for set_name, train_regime, eval_regime, floor in SETS:
        tag = f"ppo_{train_regime}_none_s42"
        allst = roll_states(tag, eval_regime, floor, set_name)
        sampled = stratified_sample(allst, N_PER_SET)
        hard = sum(1 for s in sampled if s["min_prb_urllc_realized"] > floor)
        out = os.path.join(OUT_DIR, f"states_{set_name}.json")
        with open(out, "w") as f:
            json.dump({"set": set_name, "policy_tag": tag, "eval_regime": eval_regime,
                       "static_floor": floor, "n": len(sampled),
                       "se_urllc": cfg.se["urllc"], "n_prb": cfg.n_prb, "prb_step": cfg.prb_step,
                       "states": sampled}, f)
        d_urllc = [s["demand"]["urllc"] for s in sampled]
        summary[set_name] = {"n": len(sampled), "hard_states_gt_floor": hard,
                             "urllc_demand_min": round(min(d_urllc), 1),
                             "urllc_demand_max": round(max(d_urllc), 1)}
        print(f"[{set_name}] policy={tag} eval={eval_regime} floor={floor} "
              f"-> {len(sampled)} states ({hard} hard, demand {min(d_urllc):.0f}-{max(d_urllc):.0f} Mbps) -> {out}")
    with open(os.path.join(OUT_DIR, "states_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("done.")


if __name__ == "__main__":
    main()
