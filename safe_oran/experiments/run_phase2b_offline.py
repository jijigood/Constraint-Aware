"""Run Phase2b-v1 symbolic-z offline gate.

Examples:
  .venv/bin/python -m safe_oran.experiments.run_phase2b_offline --smoke 5
  .venv/bin/python -m safe_oran.experiments.run_phase2b_offline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from safe_oran.envs.legacy import PROJECT_ROOT
from safe_oran.experiments.phase2b_symbolic import PHASE2B_DIR, compute_gate, evaluate_phase2b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0, help="Use the first N states per set.")
    ap.add_argument("--out-dir", default=str(PHASE2B_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = evaluate_phase2b(smoke=args.smoke)
    analysis = {
        "schema_version": summary["schema_version"],
        "kind": "phase2b_v1_analysis",
        "smoke": bool(args.smoke),
        "gate": compute_gate(summary),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    (out_dir / "analysis.json").write_text(json.dumps(analysis, indent=2, sort_keys=True))
    print(json.dumps(analysis["gate"], indent=2, sort_keys=True))
    print(f"summary -> {out_dir / 'summary.json'}")
    print(f"analysis -> {out_dir / 'analysis.json'}")
    return 0 if analysis["gate"]["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
