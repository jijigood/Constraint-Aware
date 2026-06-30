"""Write a compact evidence story that connects Phase2a, Phase2b, Phase2c, and Phase3."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from safe_oran.envs.legacy import PROJECT_ROOT

REPORT_PATH = PROJECT_ROOT / "06_reports" / "LETTER_EVIDENCE_STORY.md"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def _fmt(x: Any, digits: int = 4) -> str:
    if x is None or x == "":
        return "n/a"
    return f"{float(x):.{digits}f}"


def _phase3_rows() -> list[dict[str, str]]:
    path = PROJECT_ROOT / "04_results" / "phase3_m3_m5" / "phase3_table.csv"
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def _delta_rows() -> list[dict[str, str]]:
    path = PROJECT_ROOT / "04_results" / "phase3_m3_m5" / "paired_seed_delta.csv"
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def _m6_rows() -> list[dict[str, str]]:
    path = PROJECT_ROOT / "04_results" / "phase3_m6" / "m6_table.csv"
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def _delta(rows: list[dict[str, str]], scenario: str, metric: str) -> str:
    rec = next((r for r in rows if r.get("scenario") == scenario and r.get("metric") == metric), None)
    if not rec or not rec.get("mean_delta"):
        return "n/a"
    return f"{float(rec['mean_delta']):.4f} ± {float(rec.get('std_delta') or 0.0):.4f}"


def _mini_cer_line(summary: dict[str, Any], arm: str) -> str:
    rec = summary.get("arms", {}).get(arm, {})
    return (
        f"| `{arm}` | {_fmt(rec.get('evidence_recall_at5'))} | "
        f"{_fmt(rec.get('field_accuracy_mean'))} | "
        f"{_fmt(rec.get('mean_abs_delta_p_min'))} | "
        f"{_fmt(rec.get('spec_validity'))} |"
    )


def main() -> int:
    phase2a = _load(PROJECT_ROOT / "04_results" / "phase2a" / "analysis.json")
    phase2b = _load(PROJECT_ROOT / "04_results" / "phase2b_v1" / "analysis.json")
    phase2c = _load(PROJECT_ROOT / "04_results" / "phase2c_mini_cer" / "summary.json")
    phase3_m6 = _load(PROJECT_ROOT / "04_results" / "phase3_m6" / "summary.json")
    phase3_rows = _phase3_rows()
    delta_rows = _delta_rows()
    m6_rows = _m6_rows()
    p2a_cross = phase2a.get("gate", {}).get("cross_summary", {})
    p2b_gate = phase2b.get("gate", {})
    p2c_gate = phase2c.get("gate", {})

    phase3_text = "\n".join(
        f"| {r['scenario']} | {r['method']} | {r['reward']} | {r['urllc_violation_rate']} | "
        f"{r['mean_D_proj']} | {r['shield_correction_rate']} | {r['adaptation_delay']} |"
        for r in phase3_rows
    )
    mini_cer_text = "\n".join(_mini_cer_line(phase2c, arm) for arm in (
        "no_retrieval",
        "ordinary_rag_intent_only",
        "state_aware_rag",
        "field_aware_cer",
        "cer_verifier_solver",
    ))
    m6_text = ""
    if m6_rows:
        m6_table = "\n".join(
            f"| {r['method']} | {r['reward']} | {r['urllc_violation_rate']} | {r['mean_D_proj']} | "
            f"{r['shield_correction_rate']} | {r['fallback_rate']} | {r['unsafe_under_reservation_rate']} | "
            f"{r['p_min_parity_rate']} |"
            for r in m6_rows
        )
        m6_verdict = "passed" if phase3_m6.get("gates", {}).get("passed") else "boundary result"
        m6_text = f"""

### Phase3-M6: closed-loop CER-z completed

Gate: `{m6_verdict}`.

| Method | Reward | Violation | Mean D_proj | Shield correction | Fallback | Unsafe under-rsv | p_min parity |
|---|---:|---:|---:|---:|---:|---:|---:|
{m6_table}

This closes the final system loop: the S6 controller replaces Oracle-z with cached real-LLM field-CER-z while keeping the same verified solver/shield/DRL path.
"""

    text = f"""# Letter Evidence Story

## Core Claim
Language models can help interpret SLA and policy evidence, but they should not directly output safety-critical PRB numbers. The safer design is:

`SLA / intent / evidence -> symbolic z_k -> verifier -> solver -> p_min -> shield -> DRL`

The DRL policy receives only the verified scalar constraint strength `p_min/Pmax`, not natural language, JSON fields, or citations.

## Evidence Chain

### Phase2a: direct numeric LLM control is unsafe / unreliable

The direct numeric path `LLM/RAG -> urllc_min_prb` was a pre-registered NO-GO:

- Cross-regime violation: static `{_fmt(p2a_cross.get('V_static'))}`, no-RAG `{_fmt(p2a_cross.get('V_norag'))}`, RAG `{_fmt(p2a_cross.get('V_rag'))}`, oracle `{_fmt(p2a_cross.get('V_oracle'))}`.
- RAG did not add value over no-RAG, and the reward/safety gap to oracle remained large.

### Phase2b-v1: symbolic-z verified compilation works

The symbolic path gate is `{p2b_gate.get('verdict', 'n/a')}`.

- Direct numeric outputs are rejected before entering the safety path.
- Oracle/template symbolic specs reproduce the deterministic solver reservation.
- This phase proves symbolic compilation mechanics, not real CER/RAG retrieval.

### Phase2c: mini-CER shows field-aware retrieval is useful

Gate: `{p2c_gate.get('PASS', 'n/a')}`.

| Arm | Recall@5 | Field accuracy | Mean abs delta p_min | Spec validity |
|---|---:|---:|---:|---:|
{mini_cer_text}

This supports the narrower retrieval claim: CER is useful when it is evaluated as field-level evidence routing for symbolic constraint construction, rather than as ordinary QA-style RAG.

### Phase3: p_min/Pmax reduces projection burden when the regime is not saturated

| Scenario | Method | Reward | Violation | Mean D_proj | Shield correction | Adaptation delay |
|---|---|---:|---:|---:|---:|---:|
{phase3_text}

Paired deltas are computed as `M5 - M3`:

- S4 `Delta D_proj`: {_delta(delta_rows, 'S4_sla_upgrade', 'mean_D_proj')}; `Delta violation`: {_delta(delta_rows, 'S4_sla_upgrade', 'urllc_violation_rate')}.
- S6 `Delta D_proj`: {_delta(delta_rows, 'S6_moderate_decay', 'mean_D_proj')}; `Delta violation`: {_delta(delta_rows, 'S6_moderate_decay', 'urllc_violation_rate')}.
- S5 `Delta adaptation delay`: {_delta(delta_rows, 'S5_combined', 'adaptation_delay')}.
- S3 is the boundary case: the constraint saturates near the resource ceiling, so M5 has little room to improve and should be reported as a near-infeasible regime.
{m6_text}

## Paper Wording

Use this claim:

> We propose a verifiable symbolic constraint compilation framework for safe O-RAN slicing control. Direct numeric LLM constraint generation is unreliable; symbolic-z verified compilation safely produces `p_min`; and exposing `p_min/Pmax` to the DRL policy reduces shield projection under dynamic, non-saturated constraints, with reward/safety trade-offs reported explicitly.

Avoid this claim:

> The system is a real O-RAN deployment or a fully validated real-world CER/RAG controller.

Phase2c is a controlled mini benchmark for field-level retrieval. Real retrieval over external standards and field attribution remains Phase2c-v2 / future work.
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(text)
    print(str(REPORT_PATH))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
