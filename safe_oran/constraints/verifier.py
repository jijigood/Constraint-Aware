"""Mechanical verifier for LLM/RAG-compiled constraint specs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .spec import (
    ALLOWED_FORMULA_IDS,
    ALLOWED_MARGIN_POLICIES,
    ALLOWED_SERVICE_RULES,
    ConstraintSpec,
    fallback_spec,
)


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    reason: str
    spec: ConstraintSpec | None = None


class Verifier:
    """Fail-closed checks before a symbolic spec reaches the solver."""

    def __init__(self, *, require_citations: bool = False):
        self.require_citations = require_citations

    def verify(
        self,
        spec: ConstraintSpec | dict[str, Any],
        retrieved_ids: list[str] | None = None,
        state: dict[str, Any] | None = None,
        z_mode: str = "oracle",
    ) -> VerificationResult:
        del state  # reserved for future feasibility/monotonicity checks
        raw = spec if isinstance(spec, dict) else spec.to_dict()
        if "urllc_min_prb" in raw:
            return VerificationResult(False, "direct_numeric_prb_is_not_a_symbolic_spec")
        try:
            z = spec if isinstance(spec, ConstraintSpec) else ConstraintSpec.from_mapping(raw)
        except Exception as exc:  # noqa: BLE001
            return VerificationResult(False, f"schema_parse_error:{exc}")

        if z.formula_id not in ALLOWED_FORMULA_IDS:
            return VerificationResult(False, f"formula_not_whitelisted:{z.formula_id}", z)
        if z.channel_margin_policy not in ALLOWED_MARGIN_POLICIES:
            return VerificationResult(False, f"margin_policy_not_allowed:{z.channel_margin_policy}", z)
        if z.service_rule not in ALLOWED_SERVICE_RULES:
            return VerificationResult(False, f"service_rule_not_allowed:{z.service_rule}", z)
        if not (0.0 < float(z.reliability_target) <= 1.0):
            return VerificationResult(False, f"reliability_out_of_range:{z.reliability_target}", z)
        if not (1 <= int(z.priority_rank) <= 5):
            return VerificationResult(False, f"priority_rank_out_of_range:{z.priority_rank}", z)

        valid_ids = set(str(x) for x in (retrieved_ids if retrieved_ids is not None else z.retrieved_ids))
        if self.require_citations and z_mode != "oracle" and not z.citations:
            return VerificationResult(False, "missing_required_citations", z)
        if z.citations and valid_ids:
            bad = [c for c in z.citations if c not in valid_ids]
            if bad:
                return VerificationResult(False, f"citation_not_retrieved:{bad}", z)

        return VerificationResult(True, "ok", ConstraintSpec.from_mapping({**z.to_dict(), "verified": True}))

    def semantic_verify(
        self,
        spec: ConstraintSpec | dict[str, Any],
        retrieved_ids: list[str] | None,
        state: dict[str, Any],
        z_mode: str = "oracle",
    ) -> tuple[bool, str]:
        result = self.verify(spec, retrieved_ids, state, z_mode=z_mode)
        return result.passed, result.reason

    def fail_closed_spec(self, reliability: float = 0.99) -> ConstraintSpec:
        return fallback_spec(reliability)

