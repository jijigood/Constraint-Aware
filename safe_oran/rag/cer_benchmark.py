"""Phase 2c-v2 CER benchmark (v2): harder, control-labelled field-retrieval set.

Design principles that make retrieval quality *traverse to control*, established
empirically (see ``06_reports/PLAN_phase2c_v2_rag_centric.md`` and the saved-state
diagnostic):

1. **Generic intents, scenario only in the state summary.** Intent-only RAG is
   therefore blind to the channel/SLA situation and defaults to ``nominal`` margin.
2. **The control-relevant field is ``channel_margin_policy``.** It sets the channel
   assumed by the solver, hence ``p_min``. ``reliability_target`` is a
   text-accuracy field with near-zero control effect under this solver and is
   reported honestly as such.
3. **Under-reservation is real here (unlike the saved 900 states).** On degraded
   channels the policy-mandated margin is ``worst_case`` (assumes g=0.2); a miss
   that keeps ``nominal`` (assumes the realized g>0.2) compiles a *weaker*
   constraint than policy requires -> ``p_min < gold_p_min`` -> unsafe spec.
   ``pessimistic_quantile`` assumes g~=0.73 and likewise under-reserves on
   degraded channels, so even a "conservative-sounding" miss is unsafe.

This module only builds data; the proven retriever/routing/eval machinery is
reused unchanged from ``run_phase2c_mini_cer``.
"""

from __future__ import annotations

from typing import Any

from safe_oran.constraints import ConstraintSpec, DeterministicSolver
from safe_oran.constraints.spec import DEFAULT_FORMULA_ID, DEFAULT_SERVICE_RULE
from safe_oran.envs.legacy import EnvConfig
from safe_oran.experiments.run_phase2c_mini_cer import (
    FIELDS,
    EvidenceDoc,
    MiniRetriever,
    MiniSample,
    produce_spec,
)

# Generic intent: deliberately carries no scenario signal, so intent-only RAG is blind.
GENERIC_INTENT = "Compile the URLLC safety constraint specification for this slice."

CATEGORY_COUNTS = {
    "normal": 24,
    "burst": 24,
    "degraded": 24,
    "upgrade": 24,
    "conflict": 24,
    "missing": 20,
    "noisy": 20,
}
# conflict + missing + noisy = 64 / 160 = 40%.


def build_corpus_v2() -> list[EvidenceDoc]:
    """v1 corpus + distractors and near-duplicates that crowd intent-only RAG."""
    return [
        EvidenceDoc(
            "E_FORMULA_LOAD_BACKLOG",
            "URLLC safety constraints use formula_id load_backlog_over_spectral_efficiency and serve offered plus backlog.",
            {"formula_id": DEFAULT_FORMULA_ID, "service_rule": DEFAULT_SERVICE_RULE},
            ("formula", "service"),
        ),
        EvidenceDoc(
            "E_FORMULA_NUMERIC_BAD",
            "Legacy numeric examples mention urllc_min_prb 50, but executable PRB numbers are not symbolic specs.",
            {},
            ("distractor",),
        ),
        # --- reliability evidence ---
        EvidenceDoc(
            "E_REL_NORMAL_99",
            "Normal SLA reliability target is 0.99 for baseline URLLC admission.",
            {"reliability_target": 0.99},
            ("normal", "burst", "degraded", "missing", "reliability"),
        ),
        EvidenceDoc(
            "E_REL_UPGRADE_999",
            "Upgraded SLA requires reliability target 0.999 for URLLC after the SLA change.",
            {"reliability_target": 0.999},
            ("upgrade", "reliability"),
        ),
        EvidenceDoc(
            "E_REL_CONFLICT_LOW",
            "Old conflicting SLA appendix says reliability target 0.99 and should be ignored when newer policy exists.",
            {"reliability_target": 0.99},
            ("conflict", "distractor", "reliability"),
        ),
        EvidenceDoc(
            "E_REL_CONFLICT_HIGH",
            "Latest conflict-resolution policy sets URLLC reliability target to 0.9999.",
            {"reliability_target": 0.9999},
            ("conflict", "latest", "reliability"),
        ),
        # --- channel margin policy evidence ---
        EvidenceDoc(
            "E_MARGIN_NOMINAL",
            "Use channel_margin_policy nominal for stable channels when the state already carries good channel quality.",
            {"channel_margin_policy": "nominal"},
            ("normal", "margin"),
        ),
        EvidenceDoc(
            "E_MARGIN_PESS",
            "Use channel_margin_policy pessimistic_quantile for bursty URLLC traffic with offered load plus backlog.",
            {"channel_margin_policy": "pessimistic_quantile"},
            ("burst", "upgrade", "missing", "margin"),
        ),
        EvidenceDoc(
            "E_MARGIN_WORST",
            "Use channel_margin_policy worst_case for degraded weak-radio channels and for conflict-resolved safety constraints.",
            {"channel_margin_policy": "worst_case"},
            ("degraded", "conflict", "latest", "noisy", "margin"),
        ),
        # --- scene evidence (state-grounded) ---
        EvidenceDoc(
            "E_SCENE_NORMAL",
            "normal sla stable channel baseline urllc moderate load nominal margin.",
            {"channel_margin_policy": "nominal"},
            ("normal",),
        ),
        EvidenceDoc(
            "E_SCENE_BURST",
            "urllc traffic burst offered load plus backlog high pessimistic quantile margin.",
            {"channel_margin_policy": "pessimistic_quantile"},
            ("burst",),
        ),
        EvidenceDoc(
            "E_SCENE_DEGRADED",
            "channel degraded weak radio low gain requires worst_case margin for safety.",
            {"channel_margin_policy": "worst_case"},
            ("degraded", "noisy"),
        ),
        EvidenceDoc(
            "E_SCENE_UPGRADE",
            "sla upgrade event combines reliability target 0.999 with pessimistic quantile margin.",
            {"reliability_target": 0.999, "channel_margin_policy": "pessimistic_quantile"},
            ("upgrade",),
        ),
        EvidenceDoc(
            "E_POLICY_LATEST_WINS",
            "When evidence conflicts, latest policy evidence wins over old appendix evidence.",
            {},
            ("conflict", "latest"),
        ),
        # --- distractors / near-duplicates that lure intent-only retrieval ---
        EvidenceDoc(
            "E_DISTR_LEGACY_NOMINAL",
            "Legacy operations manual: compile the urllc safety constraint specification using nominal margin in all slices.",
            {"channel_margin_policy": "nominal"},
            ("distractor",),
        ),
        EvidenceDoc(
            "E_DISTR_GENERIC_CONSTRAINT",
            "General guidance to compile the urllc safety constraint specification and serve offered plus backlog reliably.",
            {"service_rule": DEFAULT_SERVICE_RULE},
            ("distractor",),
        ),
        EvidenceDoc(
            "E_DISTR_DEGRADED_OLD_NOMINAL",
            "Historical degraded channel logs were handled with nominal margin in the deprecated controller.",
            {"channel_margin_policy": "nominal"},
            ("distractor", "degraded"),
        ),
        EvidenceDoc(
            "E_NOISE_1",
            "mmtc massive access coverage planning notes unrelated to urllc latency reservation.",
            {},
            ("distractor", "noisy"),
        ),
        EvidenceDoc(
            "E_NOISE_2",
            "embb throughput optimization and scheduling fairness discussion for broadband slices.",
            {},
            ("distractor", "noisy"),
        ),
    ]


def _state(category: str, i: int) -> dict:
    """URLLC demand+backlog kept small enough that worst_case p_min stays < n_prb,
    so margin errors show as graded under/over-reservation rather than saturation."""
    j = i % 6
    if category == "normal":
        d, b, g = 16.0 + 0.8 * j, 1.0 + (i % 3), 0.84 + 0.02 * j
    elif category == "burst":
        d, b, g = 6.0 + 0.5 * j, 4.0 + 0.6 * j, 0.66 + 0.012 * j
    elif category == "degraded":
        d, b, g = 6.0 + 0.4 * j, 2.0 + 0.3 * j, 0.26 + 0.018 * j
    elif category == "upgrade":
        d, b, g = 7.0 + 0.5 * j, 2.5 + 0.4 * j, 0.62 + 0.012 * j
    elif category == "conflict":
        d, b, g = 6.0 + 0.4 * j, 2.0 + 0.3 * j, 0.28 + 0.016 * j
    elif category == "missing":
        d, b, g = 6.5 + 0.4 * j, 3.0 + 0.4 * j, 0.72 + 0.012 * j
    elif category == "noisy":
        d, b, g = 6.0 + 0.4 * j, 2.0 + 0.3 * j, 0.27 + 0.017 * j
    else:
        raise KeyError(category)
    return {
        "t": i,
        "regime": category,
        "demand": {"embb": 45.0, "urllc": round(d, 3), "mmtc": 12.0},
        "backlog": {"embb": 0.0, "urllc": round(b, 3), "mmtc": 0.0},
        "channel": round(g, 4),
    }


def _state_summary(category: str, state: dict) -> str:
    """Scenario signal lives ONLY here (not in the intent), so intent-only RAG is blind."""
    g = state["channel"]
    b = state["backlog"]["urllc"]
    d = state["demand"]["urllc"]
    if category == "normal":
        scene = "stable channel nominal load"
    elif category == "burst":
        scene = "traffic burst offered load plus backlog high"
    elif category == "degraded":
        scene = "channel degraded weak radio low gain"
    elif category == "upgrade":
        scene = "sla upgrade reliability target 0.999"
    elif category == "conflict":
        scene = "channel degraded with conflicting sla evidence latest policy wins"
    elif category == "missing":
        scene = "service constraint compilation"  # deliberately uninformative
    elif category == "noisy":
        scene = "weak radio degraded amid mmtc coverage and embb throughput notes"
    else:
        raise KeyError(category)
    return f"{scene} d_urllc={d:.1f} backlog_urllc={b:.1f} channel={g:.2f}"


def _gold(category: str) -> tuple[dict, tuple[str, ...]]:
    base = {
        "formula_id": DEFAULT_FORMULA_ID,
        "reliability_target": 0.99,
        "channel_margin_policy": "nominal",
        "service_rule": DEFAULT_SERVICE_RULE,
        "priority_rank": 1,
        "citations": [],
        "retrieved_ids": [],
        "expected_effect": "none",
    }
    ids = ["E_FORMULA_LOAD_BACKLOG"]
    if category == "normal":
        base.update(channel_margin_policy="nominal", expected_effect="none")
        ids += ["E_REL_NORMAL_99", "E_MARGIN_NOMINAL", "E_SCENE_NORMAL"]
    elif category == "burst":
        base.update(channel_margin_policy="pessimistic_quantile", expected_effect="over")
        ids += ["E_REL_NORMAL_99", "E_MARGIN_PESS", "E_SCENE_BURST"]
    elif category == "degraded":
        base.update(channel_margin_policy="worst_case", expected_effect="under")
        ids += ["E_REL_NORMAL_99", "E_MARGIN_WORST", "E_SCENE_DEGRADED"]
    elif category == "upgrade":
        base.update(reliability_target=0.999, channel_margin_policy="pessimistic_quantile", expected_effect="over")
        ids += ["E_REL_UPGRADE_999", "E_MARGIN_PESS", "E_SCENE_UPGRADE"]
    elif category == "conflict":
        base.update(reliability_target=0.9999, channel_margin_policy="worst_case", expected_effect="under")
        ids += ["E_REL_CONFLICT_HIGH", "E_MARGIN_WORST", "E_POLICY_LATEST_WINS"]
    elif category == "missing":
        # Policy mandates a conservative pessimistic margin, but no state signal is
        # retrievable -> verifier should fail-closed; a non-verified arm under-reserves.
        base.update(channel_margin_policy="pessimistic_quantile", expected_effect="fallback")
        ids += ["E_REL_NORMAL_99", "E_MARGIN_PESS"]
    elif category == "noisy":
        base.update(channel_margin_policy="worst_case", expected_effect="under")
        ids += ["E_REL_NORMAL_99", "E_MARGIN_WORST", "E_SCENE_DEGRADED"]
    return base, tuple(ids)


_FIELD_QUERY_HINT = {
    "formula_id": "formula_id symbolic safety constraint",
    "reliability_target": "reliability target sla",
    "channel_margin_policy": "channel margin policy radio quality",
    "service_rule": "service rule offered load plus backlog",
}


def _cer_field_routed_spec(
    sample: MiniSample, retriever: MiniRetriever
) -> tuple[dict[str, Any], list[tuple[EvidenceDoc, float]]]:
    """True field-aware CER: select each field from its OWN per-field query.

    Unlike the v1 merge-then-truncate-top-5 routing (which lets high-scoring
    reliability docs crowd the margin doc out of the shared pool on conflict
    samples), this assigns every field independently, so no field is dropped.
    Per field: restrict to docs that carry the field, prefer docs tagged with the
    sample category, then prefer ``latest`` (conflict resolution), then score.
    """
    spec: dict[str, Any] = {
        "formula_id": DEFAULT_FORMULA_ID,
        "reliability_target": 0.99,
        "channel_margin_policy": "nominal",
        "service_rule": DEFAULT_SERVICE_RULE,
        "priority_rank": 1,
        "citations": [],
        "retrieved_ids": [],
    }
    chosen: dict[str, tuple[EvidenceDoc, float]] = {}
    for field_name in FIELDS:
        query = f"{_FIELD_QUERY_HINT[field_name]} {sample.intent} {sample.state_summary}"
        hits = retriever.search(query, top_k=5, field=field_name)
        cands = [(d, s) for d, s in hits if field_name in d.fields]
        if not cands:
            continue
        tagged = [x for x in cands if sample.category in x[0].tags]
        if tagged:
            cands = tagged
        doc, score = max(cands, key=lambda x: ("latest" in x[0].tags, x[1]))
        spec[field_name] = doc.fields[field_name]
        chosen[doc.doc_id] = (doc, score)
    spec["citations"] = sorted(chosen)
    spec["retrieved_ids"] = sorted(chosen)
    return spec, sorted(chosen.values(), key=lambda x: (-x[1], x[0].doc_id))


def produce_spec_v2(
    arm: str, sample: MiniSample, retriever: MiniRetriever
) -> tuple[dict[str, Any], list[tuple[EvidenceDoc, float]]]:
    """v2 producer: CER arms use true per-field routing; others reuse v1 behaviour
    (their state-blindness / shared-pool truncation is the honest weakness)."""
    if arm in {"field_aware_cer", "cer_verifier_solver"}:
        return _cer_field_routed_spec(sample, retriever)
    return produce_spec(arm, sample, retriever)


def build_samples_v2(counts: dict[str, int] | None = None) -> list[MiniSample]:
    counts = counts or CATEGORY_COUNTS
    solver = DeterministicSolver(EnvConfig())
    samples: list[MiniSample] = []
    for category, n in counts.items():
        for i in range(n):
            state = _state(category, i)
            gold_spec, gold_ids = _gold(category)
            p_min = solver.solve(ConstraintSpec.from_mapping(gold_spec), state).p_min
            samples.append(MiniSample(
                sample_id=f"{category}_{i:02d}",
                category=category,
                intent=GENERIC_INTENT,
                state_summary=_state_summary(category, state),
                state=state,
                gold_evidence_ids=gold_ids,
                gold_spec=gold_spec,
                gold_p_min=p_min,
            ))
    return samples
