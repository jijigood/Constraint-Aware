"""Aggregate Phase3 M3/M5 run JSONs."""

from __future__ import annotations

import argparse
import glob
import json

import numpy as np

from safe_oran.experiments.m3_m5_common import OUT_DIR, RUNS_DIR, write_json


def _agg(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return {"mean": None, "std": None, "n": 0}
    arr = np.asarray(vals, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std()), "n": int(len(arr))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(RUNS_DIR))
    args = ap.parse_args()

    runs = [json.load(open(path)) for path in sorted(glob.glob(f"{args.runs_dir}/*.json"))]
    groups = {}
    for run in runs:
        groups.setdefault((run["method"], run["scenario"]), []).append(run)
    summary = {
        "schema_version": "safe_oran_phase3_m3_m5",
        "kind": "phase3_m3_m5_summary",
        "n_runs": len(runs),
        "groups": {},
    }
    metric_keys = [
        "reward",
        "urllc_violation_rate",
        "mean_D_proj",
        "p95_D_proj",
        "shield_correction_rate",
        "mean_p_min",
        "p95_p_min",
        "adaptation_delay",
    ]
    for (method, scenario), rs in groups.items():
        rec = {"n_runs": len(rs), "all_paper_usable": all(r.get("paper_usable", False) for r in rs)}
        for key in metric_keys:
            rec[key] = _agg([r["eval_metrics"].get(key) for r in rs])
        rec["train_violation_rate"] = _agg([r["train_metrics"].get("train_violation_rate") for r in rs])
        rec["train_mean_D_proj"] = _agg([r["train_metrics"].get("train_mean_D_proj") for r in rs])
        summary["groups"][f"{method}|{scenario}"] = rec
    write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

