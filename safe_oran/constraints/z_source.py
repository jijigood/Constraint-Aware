"""Constraint-spec cache/source for oracle, CER, and noisy modes."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .spec import fallback_spec, oracle_spec


class ZCache:
    """Simple event cache keyed by ``scenario|sla=<target>|t=<event>``."""

    def __init__(self, cache_path: str | None = None, entries: dict[str, dict[str, Any]] | None = None):
        if entries is not None:
            self._cache = copy.deepcopy(entries)
        elif cache_path:
            self._cache = json.loads(Path(cache_path).read_text())
        else:
            self._cache = default_oracle_entries()

    @staticmethod
    def make_key(scenario: str, sla: float, t_event: int) -> str:
        return f"{scenario}|sla={sla:.4f}|t={int(t_event)}"

    @staticmethod
    def parse_key(key: str) -> tuple[str, float, int]:
        scenario, sla_part, t_part = key.split("|")
        return scenario, float(sla_part.replace("sla=", "")), int(t_part.replace("t=", ""))

    def get(self, scenario: str, t: int, sla_target: float) -> dict[str, Any]:
        candidates = {}
        for key, val in self._cache.items():
            try:
                sc, sla, t_event = self.parse_key(key)
            except Exception:  # noqa: BLE001
                continue
            if sc == scenario and abs(sla - float(sla_target)) < 1e-5:
                candidates[t_event] = val
        if not candidates:
            return fallback_spec(sla_target).to_dict()
        valid = {te: z for te, z in candidates.items() if te <= int(t)}
        if valid:
            return copy.deepcopy(valid[max(valid)])
        return copy.deepcopy(candidates[min(candidates)])


def default_oracle_entries() -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    scenarios = {
        "S1_stable": [(0.99, 0, "nominal")],
        "S2_urllc_burst": [(0.99, 0, "nominal"), (0.99, 100, "pessimistic_quantile")],
        # S3 keeps the symbolic rule fixed; the deterministic solver's p_min
        # should move because the state channel decays.
        "S3_channel_decay": [(0.99, 0, "nominal")],
        "S4_sla_upgrade": [(0.99, 0, "nominal"), (0.9999, 100, "pessimistic_quantile")],
        "S5_combined": [(0.99, 0, "nominal"), (0.99, 100, "pessimistic_quantile"), (0.9999, 150, "pessimistic_quantile")],
        "balanced": [(0.99, 0, "pessimistic_quantile")],
        "high_embb": [(0.99, 0, "pessimistic_quantile")],
        "high_urllc": [(0.99, 0, "pessimistic_quantile")],
        "bursty": [(0.99, 0, "pessimistic_quantile")],
    }
    for scenario, specs in scenarios.items():
        for sla, t_event, margin in specs:
            z = oracle_spec(sla).to_dict()
            z["channel_margin_policy"] = margin
            entries[ZCache.make_key(scenario, sla, t_event)] = z
    return entries
