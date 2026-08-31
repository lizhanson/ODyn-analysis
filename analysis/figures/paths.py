"""Portable paths shared by figure notebooks and command-line analyses."""

from __future__ import annotations

import os
from pathlib import Path


# This module lives at <repo>/analysis/figures/paths.py.
REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_path(*parts: str) -> Path:
    """Return a path relative to the checked-out repository."""
    return REPO_ROOT.joinpath(*parts)


def output_root(value=None) -> Path:
    """Where generated figures and tables go, deliberately outside the repository.

    Outputs are large and regenerate from the source data, so they are never
    committed. Writing them outside the working tree entirely is stronger than
    a .gitignore rule: a stray `git add -A` cannot pick them up, a synced
    folder never tries to replicate them, and `du` on the repository reports
    the code rather than the pictures.

    Set ODYN_OUTPUT_ROOT to move them elsewhere.
    """
    configured = value if value is not None else os.environ.get("ODYN_OUTPUT_ROOT")
    if configured is not None and str(configured).strip():
        return Path(configured).expanduser()
    return Path.home() / "odyn_scratch" / "odyn-analysis-outputs"


def imaging_root(value=None) -> Path:
    """Resolve the external ImagingData root without machine-specific paths."""
    configured = value if value is not None else os.environ.get("ODYN_IMAGING_ROOT")
    if configured is None or not str(configured).strip():
        raise RuntimeError(
            "ImagingData root is not configured. Set ODYN_IMAGING_ROOT or pass "
            "--imaging-root to the command-line analysis."
        )
    return Path(configured).expanduser()
