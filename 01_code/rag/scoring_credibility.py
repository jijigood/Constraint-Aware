"""
Phase 2a pre-gate: adversarial scoring-credibility check (NO LLM, offline, paper_usable=false).
Mirrors route_b_reward_credibility.py's discipline: prove the one-step counterfactual SCORER orders
hand-built reservations correctly (and has teeth) BEFORE spending any LLM generations -- so a later GATE
PASS reflects constraint quality, not a scorer artifact.

Template reservations per state: {0, oracle-10, oracle, oracle+30, static_floor}.
Checks: monotone safety; over-reservation costs reward; cross-set static failure reproduced;
discriminating-power control (a "broken" scorer that ignores the reservation must FAIL monotonicity).

Run:  python 01_code/rag/scoring_credibility.py
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(RAG_DIR))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(os.path.dirname(RAG_DIR), "env"))
sys.path.insert(0, RAG_DIR)

from slicing_env import EnvConfig  # noqa: E402
import counterfactual as CF  # noqa: E402
from safe_oran.constraints.solver import oracle_reservation as _v4_oracle_reservation  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(RAG_DIR)), "04_results", "phase2a", "scoring_credibility.json")
RELIABILITY = 0.99


def oracle_reservation(state, cfg=None, reliability=RELIABILITY) -> int:
    """Compatibility shim for the refactored deterministic solver."""
    return _v4_oracle_reservation(state, cfg=cfg or EnvConfig(), reliability=reliability)


def templates(state):
    o = oracle_reservation(state)
    return {"zero": 0, "oracle_minus10": max(0, o - 10), "oracle": o,
            "oracle_plus30": min(100, o + 30), "static": int(state["static_floor"])}


def landscape(states, broken=False):
    """Return {template: {viol, reward}} aggregated over states. broken -> ignore reservation."""
    out = {}
    names = ["zero", "oracle_minus10", "oracle", "oracle_plus30", "static"]
    for name in names:
        viols, rews = [], []
        for s in states:
            r = templates(s)[name]
            if broken:
                # broken scorer: always step the logged action, ignoring the reservation
                sc = CF.score_reservation(s, s["logged_alloc"][1])  # reservation = logged urllc prb -> no projection
            else:
                sc = CF.score_reservation(s, r)
            viols.append(sc["urllc_violation"]); rews.append(sc["reward"])
        out[name] = {"viol": float(np.mean(viols)), "reward": float(np.mean(rews))}
    return out


def main():
    sets = CF._load_all_states()
    if not sets:
        print("run state_replay.py first"); sys.exit(1)

    land = {name: landscape(d["states"]) for name, d in sets.items()}
    broken_cross = landscape(sets["cross"]["states"], broken=True)

    EPS_V, EPS_R = 0.01, 0.02
    c = land["cross"]; u = land["high_urllc"]; e = land["high_embb"]
    checks = {}
    # 1. monotone safety on the regimes where the reservation bites (cross + high_urllc)
    def monotone(L):
        return (L["zero"]["viol"] > L["oracle"]["viol"] + EPS_V
                and L["oracle_minus10"]["viol"] >= L["oracle"]["viol"] - EPS_V
                and L["oracle_plus30"]["viol"] <= L["oracle"]["viol"] + EPS_V)
    checks["monotone_safety_cross"] = monotone(c)
    checks["monotone_safety_high_urllc"] = monotone(u)
    # 2. over-reservation costs reward where URLLC needs little (high_embb)
    checks["over_reservation_costs_reward"] = e["oracle_plus30"]["reward"] < e["oracle"]["reward"] - EPS_R
    # 3. cross-set static failure reproduced (the Phase-1 motivation)
    checks["cross_static_fails_vs_oracle"] = c["static"]["viol"] > c["oracle"]["viol"] + 0.20
    # 4. discriminating-power control: broken scorer must NOT be monotone on cross
    checks["discriminating_control_has_teeth"] = not (broken_cross["zero"]["viol"] > broken_cross["oracle"]["viol"] + EPS_V)

    PASS = all(checks.values())
    result = {"schema_version": "safe_drl_v1", "kind": "scoring_credibility", "paper_usable": False,
              "reliability": RELIABILITY, "landscape": land, "broken_cross": broken_cross,
              "checks": checks, "PASS": PASS}
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)

    print("=" * 74)
    print("SCORING-CREDIBILITY (no LLM) — reservation landscape: viol / reward")
    print("=" * 74)
    for name in ["high_embb", "high_urllc", "cross"]:
        L = land[name]
        print(f"\n# {name}")
        print(f"  {'template':<16}{'viol':>8}{'reward':>9}")
        for t in ["zero", "oracle_minus10", "oracle", "oracle_plus30", "static"]:
            print(f"  {t:<16}{L[t]['viol']:>8.3f}{L[t]['reward']:>9.3f}")
    print(f"\n# discriminating control (cross, broken scorer ignores reservation):")
    print(f"  zero.viol={broken_cross['zero']['viol']:.3f}  oracle.viol={broken_cross['oracle']['viol']:.3f}  (should be ~equal)")
    print("\n--- checks ---")
    for k, v in checks.items():
        print(f"  [{'x' if v else ' '}] {k}")
    print(f"\n  ==> SCORING-CREDIBILITY: {'PASS — scorer is trustworthy; safe to spend LLM calls' if PASS else 'FAIL — fix before LLM'}")
    print(f"artifact: {OUT}")
    sys.exit(0 if PASS else 1)


if __name__ == "__main__":
    main()
