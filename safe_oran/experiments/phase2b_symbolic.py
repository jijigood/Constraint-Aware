"""Deterministic symbolic-z producers and scorer for Phase2b-v1."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from safe_oran.constraints import ConstraintSpec, DeterministicSolver, Verifier, oracle_spec
from safe_oran.envs.legacy import EnvConfig, PROJECT_ROOT, ensure_legacy_paths


PHASE2A_DIR = PROJECT_ROOT / "04_results" / "phase2a"
PHASE2B_DIR = PROJECT_ROOT / "04_results" / "phase2b_v1"
SETS_ORDER = ("cross", "high_urllc", "high_embb")


@dataclass
class SymbolicOutput:
    raw_spec: dict[str, Any]
    source: str
    retrieved_ids: list[str] = field(default_factory=list)
    used_fallback: bool = False
    verifier_passed: bool | None = None
    verifier_reason: str = ""
    unsafe_numeric_output: bool = False
    unsafe_numeric_passed: bool = False
    solver_error: str = ""


class BaseSymbolicProducer:
    name = "base"

    def produce(self, state: dict[str, Any]) -> SymbolicOutput:
        raise NotImplementedError


class StaticArm(BaseSymbolicProducer):
    name = "static"

    def produce(self, state: dict[str, Any]) -> SymbolicOutput:
        return SymbolicOutput(raw_spec={}, source=self.name)


class DirectNumericLegacyRejected(BaseSymbolicProducer):
    name = "direct_numeric_legacy_rejected"

    def produce(self, state: dict[str, Any]) -> SymbolicOutput:
        return SymbolicOutput(
            raw_spec={
                "urllc_min_prb": int(state["static_floor"]),
                "reliability_target": 0.99,
                "reason": "legacy direct numeric output, kept only for verifier rejection",
                "citations": [],
            },
            source=self.name,
            unsafe_numeric_output=True,
        )


class OracleZ(BaseSymbolicProducer):
    name = "oracle_z"

    def produce(self, state: dict[str, Any]) -> SymbolicOutput:
        del state
        return SymbolicOutput(raw_spec=oracle_spec(0.99).to_dict(), source=self.name)


class TemplateSymbolicZ(BaseSymbolicProducer):
    name = "template_symbolic_z"

    def produce(self, state: dict[str, Any]) -> SymbolicOutput:
        del state
        return SymbolicOutput(
            raw_spec=ConstraintSpec(
                formula_id="load_backlog_over_spectral_efficiency",
                reliability_target=0.99,
                channel_margin_policy="pessimistic_quantile",
                service_rule="serve_offered_plus_backlog",
                priority_rank=1,
                citations=[],
                retrieved_ids=[],
                verified=False,
            ).to_dict(),
            source=self.name,
        )


class TemplateSymbolicZVerifierOn(TemplateSymbolicZ):
    name = "template_symbolic_z_vrf_on"


class NoisySymbolicZVerifierOn(BaseSymbolicProducer):
    name = "noisy_symbolic_z_vrf_on"

    def produce(self, state: dict[str, Any]) -> SymbolicOutput:
        del state
        return SymbolicOutput(
            raw_spec={
                "formula_id": "load_backlog_over_spectral_efficiency",
                "reliability_target": 1.2,
                "channel_margin_policy": "nominal",
                "service_rule": "serve_offered_plus_backlog",
                "priority_rank": 1,
                "citations": [],
                "retrieved_ids": [],
            },
            source=self.name,
        )


class NoisySymbolicZVerifierOff(NoisySymbolicZVerifierOn):
    name = "noisy_symbolic_z_vrf_off"


def load_saved_states(smoke: int = 0) -> dict[str, list[dict[str, Any]]]:
    states = {}
    for name in SETS_ORDER:
        data = json.loads((PHASE2A_DIR / f"states_{name}.json").read_text())
        rows = data["states"]
        states[name] = rows[:smoke] if smoke else rows
    return states


def _counterfactual_module():
    ensure_legacy_paths(include_rag=True)
    import counterfactual as cf  # noqa: PLC0415

    return cf


def _legacy_oracle_pmin(state: dict[str, Any], solver: DeterministicSolver) -> int:
    return solver.solve(oracle_spec(0.99), state).p_min


def _resolve_pmin(
    producer: BaseSymbolicProducer,
    state: dict[str, Any],
    solver: DeterministicSolver,
    verifier: Verifier,
) -> tuple[int, SymbolicOutput]:
    out = producer.produce(state)
    if producer.name == "static":
        return int(state["static_floor"]), out

    verifier_on = producer.name.endswith("_vrf_on") or producer.name in {
        "direct_numeric_legacy_rejected",
        "oracle_z",
    }
    if verifier_on:
        result = verifier.verify(out.raw_spec, out.raw_spec.get("retrieved_ids", []), state, z_mode="template")
        out.verifier_passed = result.passed
        out.verifier_reason = result.reason
        if out.unsafe_numeric_output and result.passed:
            out.unsafe_numeric_passed = True
        if not result.passed:
            out.used_fallback = True
            spec = verifier.fail_closed_spec(0.99)
        else:
            spec = result.spec
    else:
        out.verifier_passed = None
        spec = ConstraintSpec.from_mapping(out.raw_spec)

    try:
        return solver.solve(spec, state).p_min, out
    except Exception as exc:  # noqa: BLE001
        out.solver_error = str(exc)
        out.used_fallback = True
        return solver.solve(verifier.fail_closed_spec(0.99), state).p_min, out


def _metrics_for_arm(
    set_name: str,
    states: list[dict[str, Any]],
    producer: BaseSymbolicProducer,
    solver: DeterministicSolver,
    verifier: Verifier,
) -> dict[str, Any]:
    cf = _counterfactual_module()
    reservations, meta = [], []
    oracle_pmins = [_legacy_oracle_pmin(s, solver) for s in states]
    for state in states:
        p_min, out = _resolve_pmin(producer, state, solver, verifier)
        reservations.append(int(p_min))
        meta.append(out)
    scored = cf.score_reservations(states, reservations)
    per = scored["per_state"]
    p = np.asarray(reservations, dtype=float)
    oracle = np.asarray(oracle_pmins, dtype=float)
    delta = p - oracle
    numeric_outputs = [m for m in meta if m.unsafe_numeric_output]
    return {
        "set": set_name,
        "arm": producer.name,
        "n": len(states),
        "urllc_violation_rate": float(np.mean([x["urllc_violation"] for x in per])),
        "reward": float(np.mean([x["reward"] for x in per])),
        "mean_p_min": float(np.mean(p)),
        "p95_p_min": float(np.percentile(p, 95)),
        "mean_prb_urllc_executed": float(np.mean([x["prb_urllc"] for x in per])),
        "spec_validity": float(np.mean([m.verifier_passed is not False for m in meta])),
        "verifier_rejection_rate": float(np.mean([m.verifier_passed is False for m in meta])),
        "fallback_rate": float(np.mean([m.used_fallback for m in meta])),
        "unsafe_numeric_pass_rate": float(
            np.mean([m.unsafe_numeric_passed for m in numeric_outputs]) if numeric_outputs else 0.0
        ),
        "mean_abs_delta_p_min_vs_oracle": float(np.mean(np.abs(delta))),
        "under_reservation_rate": float(np.mean(delta < 0)),
        "mean_under_reservation_prb": float(np.mean(np.maximum(0.0, -delta))),
        "mean_over_reservation_prb": float(np.mean(np.maximum(0.0, delta))),
        "solver_error_rate": float(np.mean([bool(m.solver_error) for m in meta])),
        "verifier_reasons": sorted({m.verifier_reason for m in meta if m.verifier_reason}),
    }


def evaluate_phase2b(smoke: int = 0) -> dict[str, Any]:
    states_by_set = load_saved_states(smoke=smoke)
    solver = DeterministicSolver(EnvConfig())
    verifier = Verifier()
    producers: list[BaseSymbolicProducer] = [
        StaticArm(),
        DirectNumericLegacyRejected(),
        OracleZ(),
        TemplateSymbolicZ(),
        TemplateSymbolicZVerifierOn(),
        NoisySymbolicZVerifierOn(),
        NoisySymbolicZVerifierOff(),
    ]
    sets = {}
    for set_name, states in states_by_set.items():
        sets[set_name] = {
            "n": len(states),
            "arms": {
                producer.name: _metrics_for_arm(set_name, states, producer, solver, verifier)
                for producer in producers
            },
        }
    return {
        "schema_version": "safe_oran_phase2b_v1",
        "kind": "phase2b_v1_symbolic_offline_gate",
        "paper_usable": not bool(smoke),
        "smoke": bool(smoke),
        "claim_scope": "symbolic-z path only; no real CER/RAG retrieval is evaluated",
        "sets": sets,
    }


def compute_gate(summary: dict[str, Any]) -> dict[str, Any]:
    checks = {}
    for set_name, set_rec in summary["sets"].items():
        arms = set_rec["arms"]
        checks[f"{set_name}_direct_numeric_unsafe_pass_zero"] = (
            arms["direct_numeric_legacy_rejected"]["unsafe_numeric_pass_rate"] == 0.0
        )
        checks[f"{set_name}_oracle_z_matches_oracle_pmin"] = (
            arms["oracle_z"]["mean_abs_delta_p_min_vs_oracle"] == 0.0
        )
        checks[f"{set_name}_template_vrf_matches_oracle_pmin"] = (
            arms["template_symbolic_z_vrf_on"]["mean_abs_delta_p_min_vs_oracle"] == 0.0
        )
        checks[f"{set_name}_noisy_vrf_on_fallbacks"] = (
            arms["noisy_symbolic_z_vrf_on"]["fallback_rate"] == 1.0
        )
    return {
        "PASS": bool(all(checks.values())),
        "checks": checks,
        "verdict": (
            "PASS: symbolic-z verifier/solver path is deterministic and direct numeric outputs are fail-closed."
            if all(checks.values())
            else "FAIL: symbolic-z offline gate did not satisfy one or more safety checks."
        ),
    }

