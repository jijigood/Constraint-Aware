"""Typed symbolic constraint specification used by V4/Phase-2b."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

DEFAULT_FORMULA_ID = "load_backlog_over_spectral_efficiency"
DEFAULT_SERVICE_RULE = "serve_offered_plus_backlog"

ALLOWED_FORMULA_IDS = {DEFAULT_FORMULA_ID}
ALLOWED_MARGIN_POLICIES = {"nominal", "pessimistic_quantile", "worst_case"}
ALLOWED_SERVICE_RULES = {DEFAULT_SERVICE_RULE}


@dataclass(frozen=True)
class ConstraintSpec:
    """Symbolic z_k; it intentionally contains no safety-critical PRB number."""

    formula_id: str = DEFAULT_FORMULA_ID
    reliability_target: float = 0.99
    channel_margin_policy: str = "pessimistic_quantile"
    service_rule: str = DEFAULT_SERVICE_RULE
    priority_rank: int = 1
    citations: list[str] = field(default_factory=list)
    retrieved_ids: list[str] = field(default_factory=list)
    verified: bool = False

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "ConstraintSpec":
        return cls(
            formula_id=str(data.get("formula_id", DEFAULT_FORMULA_ID)),
            reliability_target=float(data.get("reliability_target", 0.99)),
            channel_margin_policy=str(data.get("channel_margin_policy", "pessimistic_quantile")),
            service_rule=str(data.get("service_rule", DEFAULT_SERVICE_RULE)),
            priority_rank=int(data.get("priority_rank", 1)),
            citations=[str(x) for x in data.get("citations", [])],
            retrieved_ids=[str(x) for x in data.get("retrieved_ids", data.get("rag_evidence_ids", []))],
            verified=bool(data.get("verified", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def oracle_spec(reliability: float = 0.99, *, verified: bool = True) -> ConstraintSpec:
    """Gold/oracle symbolic spec that reproduces the legacy oracle-margin shield."""
    return ConstraintSpec(
        reliability_target=float(reliability),
        channel_margin_policy="pessimistic_quantile",
        priority_rank=1,
        citations=[],
        retrieved_ids=[],
        verified=verified,
    )


def fallback_spec(reliability: float = 0.99) -> ConstraintSpec:
    """Fail-closed default used when a generated spec is invalid."""
    return ConstraintSpec(
        reliability_target=float(reliability),
        channel_margin_policy="pessimistic_quantile",
        priority_rank=1,
        citations=[],
        retrieved_ids=[],
        verified=True,
    )

