"""Compatibility entrypoint for the refactored V4 offline smoke test.

Run:
    .venv/bin/python 01_code/smoke_v4.py
"""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from safe_oran.experiments.offline_smoke import main


if __name__ == "__main__":
    raise SystemExit(main())
