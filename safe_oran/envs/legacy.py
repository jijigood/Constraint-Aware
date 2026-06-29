"""Compatibility imports for the existing pure-numpy and Gym slicing envs."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = PROJECT_ROOT / "01_code"
ENV_DIR = CODE_DIR / "env"
DRL_DIR = CODE_DIR / "drl"
RAG_DIR = CODE_DIR / "rag"


def ensure_legacy_paths(*, include_drl: bool = False, include_rag: bool = False) -> None:
    """Make legacy modules importable without turning ``01_code`` into a package."""
    paths = [ENV_DIR]
    if include_drl:
        paths.append(DRL_DIR)
    if include_rag:
        paths.append(RAG_DIR)
    for path in paths:
        sp = str(path)
        if sp not in sys.path:
            sys.path.insert(0, sp)


ensure_legacy_paths()

from slicing_env import EnvConfig, SLICES, SlicingEnv  # noqa: E402

try:  # gymnasium is only needed for DRL training (runs in .venv); the pure-numpy
    from slicing_gym_env import SlicingGymEnv  # noqa: E402
    # constraint/solver/RAG path must stay importable in gym-less envs (e.g. dify_vllm_uv310).
except ModuleNotFoundError:  # pragma: no cover - depends on the active venv
    SlicingGymEnv = None

__all__ = [
    "PROJECT_ROOT",
    "CODE_DIR",
    "ENV_DIR",
    "DRL_DIR",
    "RAG_DIR",
    "EnvConfig",
    "SLICES",
    "SlicingEnv",
    "SlicingGymEnv",
    "ensure_legacy_paths",
]

