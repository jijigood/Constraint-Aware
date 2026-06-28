"""
Phase 2a one-step counterfactual scorer (numpy + slicing_env only; runs in either venv).

Given a logged held-out state and a URLLC PRB reservation, reconstruct the env, FORCE the realized
channel (monkeypatch -> no RNG serialization), project the logged action to satisfy the reservation
via the existing `_project_to_min_urllc` shield, step ONCE, and return the true env outcome. This is
a faithful proxy (it IS the env, not a surrogate) for the INSTANTANEOUS safety/reward of a reservation;
it deliberately omits backlog feedback, which is exactly why closed-loop is the post-gate step.

Self-test: reconstruct + step with the LOGGED action must reproduce logged_comp bit-for-bit.

Run self-test:  python 01_code/rag/counterfactual.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(RAG_DIR))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(os.path.dirname(RAG_DIR), "env"))

from slicing_env import EnvConfig, SLICES, SlicingEnv, jain  # noqa: E402
from safe_oran.shield import project_to_min_urllc  # noqa: E402

PHASE2A = os.path.join(os.path.dirname(os.path.dirname(RAG_DIR)), "04_results", "phase2a")


def reconstruct_env(state) -> SlicingEnv:
    env = SlicingEnv(EnvConfig(), regime=state["regime"], seed=0)
    env.t = int(state["t"])
    env.backlog = {s: float(state["backlog"][s]) for s in SLICES}
    env._pending = {s: float(state["demand"][s]) for s in SLICES}
    env._last_channel = float(state["last_channel"])
    g = float(state["channel"])
    env.channel = lambda: g          # force the realized channel for this step
    return env


def score_reservation(state, reservation: int) -> dict:
    env = reconstruct_env(state)
    a_exec, _ = project_to_min_urllc(env, int(state["logged_action_idx"]), int(reservation))
    _, reward, _, _, comp = env.step(a_exec)
    return {
        "urllc_violation": int(bool(comp["urllc_violation"])),
        "reward": float(reward),
        "prb_urllc": int(comp["prb_urllc"]),
        "shield_corrected": int(a_exec != int(state["logged_action_idx"])),
    }


def score_reservations(states, reservations) -> dict:
    per = [score_reservation(s, r) for s, r in zip(states, reservations)]
    prb = np.array([p["prb_urllc"] for p in per])
    return {
        "per_state": per,
        "agg": {
            "urllc_violation_rate": float(np.mean([p["urllc_violation"] for p in per])),
            "reward": float(np.mean([p["reward"] for p in per])),
            "mean_prb_urllc": float(prb.mean()),
            "shield_correction_rate": float(np.mean([p["shield_corrected"] for p in per])),
        },
    }


def reproduce_check(states, tol: float = 1e-9) -> dict:
    """Reconstruct + step with the LOGGED action; must reproduce logged_comp exactly."""
    max_rd = 0.0
    mism = 0
    for s in states:
        env = reconstruct_env(s)
        _, reward, _, _, comp = env.step(int(s["logged_action_idx"]))
        lc = s["logged_comp"]
        rd = abs(reward - lc["reward"])
        max_rd = max(max_rd, rd)
        if (rd >= tol or int(comp["urllc_violation"]) != int(lc["urllc_violation"])
                or int(comp["prb_urllc"]) != int(lc["prb_urllc"])
                or abs(comp["channel"] - lc["channel"]) >= tol):
            mism += 1
    return {"n": len(states), "mismatches": mism, "max_reward_diff": max_rd,
            "passed": mism == 0 and max_rd < tol}


def _load_all_states():
    out = {}
    for p in sorted(glob.glob(os.path.join(PHASE2A, "states_*.json"))):
        if p.endswith("states_summary.json"):
            continue
        with open(p) as f:
            d = json.load(f)
        out[d["set"]] = d
    return out


if __name__ == "__main__":
    sets = _load_all_states()
    if not sets:
        print("no states_*.json found -- run state_replay.py first"); sys.exit(1)
    print("=" * 64)
    print("counterfactual self-test (reconstruct + logged action == logged comp)")
    print("=" * 64)
    ok = True
    for name, d in sets.items():
        res = reproduce_check(d["states"])
        flag = "PASS" if res["passed"] else "FAIL"
        ok &= res["passed"]
        print(f"  [{flag}] {name:<11} n={res['n']:<4} mismatches={res['mismatches']:<4} "
              f"max_reward_diff={res['max_reward_diff']:.2e}")
    print("=" * 64)
    print("SELF-TEST PASSED -- reconstruction is faithful" if ok else "SELF-TEST FAILED")
    sys.exit(0 if ok else 1)
