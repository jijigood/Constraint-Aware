"""Phase 2c-v2 downstream control replay (G4 keystone, deterministic, no GPU/LLM).

Second half of the dual-signal G4. Over the 900 saved Phase 2a states it asks:
once the verifier/solver/shield are in the loop, what does the *control outcome*
look like for different ways of producing the URLLC reservation?

It grounds the synthetic benchmark (``run_phase2c_v2``) in the real env and
establishes two things honestly:

1. **Why fixed numbers are unsafe** — a fixed numeric floor (``static`` = the
   Phase 2a static reservation, the analogue of a direct-numeric LLM output)
   under-reserves under load and violates URLLC, reproducing the Phase 2a result.
2. **Safety-by-construction of symbolic constraints** — load-aware symbolic z_k
   (``oracle`` and the CER-compiled arm) never under-reserve on these states
   (``under_reservation = 0``), so URLLC violation ~= 0, while remaining
   resource-efficient. Retrieval quality therefore traverses to control as
   *efficiency*, not as safety, because the shield is safe by construction here.

Reuses ``counterfactual.score_reservation`` (bit-for-bit faithful one-step env
replay) and ``project_to_min_urllc`` for the L1 projection distance ``D_proj``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from safe_oran.constraints import DeterministicSolver
from safe_oran.envs.legacy import EnvConfig, PROJECT_ROOT, ensure_legacy_paths
from safe_oran.experiments.run_phase2c_mini_cer import MiniRetriever, MiniSample
from safe_oran.rag.cer_benchmark import GENERIC_INTENT, build_corpus_v2, produce_spec_v2
from safe_oran.shield import project_to_min_urllc

# Make the legacy one-step counterfactual importable (01_code/rag/counterfactual.py).
ensure_legacy_paths(include_rag=True)
import counterfactual  # noqa: E402

OUT_DIR = PROJECT_ROOT / "04_results" / "phase2c_v2"
P2A = PROJECT_ROOT / "04_results" / "phase2a"
SETS = ("cross", "high_urllc", "high_embb")
HARD_SETS = ("cross", "high_urllc")
# Symbolic arms compile a load-aware reservation; `static` is the fixed-number baseline.
ARMS = ("static", "oracle", "ordinary_rag", "field_cer")


def _derive_summary(state: dict[str, Any]) -> tuple[str, str]:
    """Map a raw saved state to a (category, state_summary) using the v2 vocabulary."""
    g = float(state["channel"])
    b = float(state["backlog"]["urllc"])
    d = float(state["demand"]["urllc"])
    if g < 0.45:
        category, scene = "degraded", "channel degraded weak radio low gain"
    elif b > 50.0:
        category, scene = "burst", "traffic burst offered load plus backlog high"
    else:
        category, scene = "normal", "stable channel nominal load"
    return category, f"{scene} d_urllc={d:.1f} backlog_urllc={b:.1f} channel={g:.2f}"


def _as_sample(state: dict[str, Any]) -> MiniSample:
    category, summary = _derive_summary(state)
    return MiniSample(
        sample_id=f"{state['set']}_{state['t']}",
        category=category,
        intent=GENERIC_INTENT,
        state_summary=summary,
        state=state,
        gold_evidence_ids=(),
        gold_spec={},
        gold_p_min=int(state["min_prb_urllc_realized"]),
    )


def _reservation(arm: str, state: dict[str, Any], retriever: MiniRetriever, solver: DeterministicSolver) -> int:
    if arm == "static":
        return int(state["static_floor"])
    if arm == "oracle":
        return int(state["min_prb_urllc_realized"])
    sample = _as_sample(state)
    rag_arm = "ordinary_rag_intent_only" if arm == "ordinary_rag" else "field_aware_cer"
    spec, _ = produce_spec_v2(rag_arm, sample, retriever)
    return int(solver.solve(spec, state).p_min)


def _d_proj(state: dict[str, Any], reservation: int) -> int:
    env = counterfactual.reconstruct_env(state)
    _, dist = project_to_min_urllc(env, int(state["logged_action_idx"]), int(reservation))
    return int(dist)


def _score_state(arm: str, state: dict[str, Any], retriever, solver) -> dict[str, Any]:
    reservation = _reservation(arm, state, retriever, solver)
    out = counterfactual.score_reservation(state, reservation)
    gold = int(state["min_prb_urllc_realized"])
    return {
        "reservation": reservation,
        "gold_reservation": gold,
        "urllc_violation": out["urllc_violation"],
        "reward": out["reward"],
        "prb_urllc": out["prb_urllc"],
        "shield_corrected": out["shield_corrected"],
        "d_proj": _d_proj(state, reservation),
        "under_reservation_prb": max(0, gold - reservation),
        "over_reservation_prb": max(0, reservation - gold),
    }


def _agg(rows: list[dict[str, Any]]) -> dict[str, Any]:
    arr = lambda k: np.asarray([r[k] for r in rows], dtype=float)
    return {
        "n": len(rows),
        "urllc_violation_rate": float(arr("urllc_violation").mean()),
        "unsafe_pass_rate": float(arr("urllc_violation").mean()),  # all symbolic specs are valid; static is a number
        "reward": float(arr("reward").mean()),
        "mean_d_proj": float(arr("d_proj").mean()),
        "shield_correction_rate": float(arr("shield_corrected").mean()),
        "mean_under_reservation_prb": float(arr("under_reservation_prb").mean()),
        "mean_over_reservation_prb": float(arr("over_reservation_prb").mean()),
        "under_reservation_rate": float((arr("under_reservation_prb") > 0).mean()),
    }


def run() -> dict[str, Any]:
    states_by_set = {}
    for s in SETS:
        d = json.loads((P2A / f"states_{s}.json").read_text())
        states_by_set[s] = d["states"]

    retriever = MiniRetriever(build_corpus_v2())
    solver = DeterministicSolver(EnvConfig())

    # Parity self-test: reconstruction must reproduce logged outcomes bit-for-bit.
    parity = {s: counterfactual.reproduce_check(states_by_set[s]) for s in SETS}
    parity_ok = all(p["passed"] for p in parity.values())

    per_arm: dict[str, Any] = {}
    for arm in ARMS:
        by_set: dict[str, Any] = {}
        all_rows: list[dict[str, Any]] = []
        hard_rows: list[dict[str, Any]] = []
        for s in SETS:
            rows = [_score_state(arm, st, retriever, solver) for st in states_by_set[s]]
            by_set[s] = _agg(rows)
            all_rows.extend(rows)
            if s in HARD_SETS:
                hard_rows.extend(rows)
        per_arm[arm] = {"by_set": by_set, "all": _agg(all_rows), "hard": _agg(hard_rows)}

    gate = _gate(per_arm)
    summary = {
        "schema_version": "safe_oran_phase2c_v2_downstream",
        "kind": "phase2c_v2_downstream_control_replay",
        "claim_scope": (
            "one-step counterfactual control replay over 900 saved Phase2a states; "
            "real env, deterministic, no DRL retrain"
        ),
        "paper_usable": True,
        "parity_self_test": {"passed": parity_ok, "per_set": parity},
        "n_states": sum(len(v) for v in states_by_set.values()),
        "arms": per_arm,
        "gate": gate,
    }
    return summary


def _gate(per_arm: dict[str, Any]) -> dict[str, Any]:
    static_h = per_arm["static"]["hard"]
    oracle_h = per_arm["oracle"]["hard"]
    cer_h = per_arm["field_cer"]["hard"]
    checks = {
        "static_is_unsafe": static_h["urllc_violation_rate"] > 0.1,  # fixed number under-reserves under load
        "oracle_is_safe": oracle_h["urllc_violation_rate"] <= 0.05,
        "cer_safe_by_construction": cer_h["under_reservation_rate"] <= 1e-9
        and cer_h["urllc_violation_rate"] <= 0.05,
        "cer_matches_oracle_efficiency": cer_h["mean_over_reservation_prb"] <= oracle_h["mean_over_reservation_prb"] + 5.0,
    }
    return {"checks": checks, "PASS": all(checks.values())}


def main() -> int:
    summary = run()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "downstream.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    brief = {
        "parity_passed": summary["parity_self_test"]["passed"],
        "gate": summary["gate"],
        "hard_set_violation": {a: round(summary["arms"][a]["hard"]["urllc_violation_rate"], 4) for a in ARMS},
        "hard_set_over_prb": {a: round(summary["arms"][a]["hard"]["mean_over_reservation_prb"], 2) for a in ARMS},
        "hard_set_reward": {a: round(summary["arms"][a]["hard"]["reward"], 4) for a in ARMS},
    }
    print(json.dumps(brief, indent=2, sort_keys=True))
    if not summary["parity_self_test"]["passed"]:
        raise SystemExit("Phase2c-v2 downstream parity self-test FAILED")
    if not summary["gate"]["PASS"]:
        raise SystemExit("Phase2c-v2 downstream G4 gate FAILED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
