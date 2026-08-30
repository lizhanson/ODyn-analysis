"""Portable paths shared by figure notebooks and command-line analyses."""

from __future__ import annotations

import os
from pathlib import Path


# This module lives at <repo>/analysis/figures/paths.py.
REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_path(*parts: str) -> Path:
    """Return a path relative to the checked-out repository."""
    return REPO_ROOT.joinpath(*parts)


def imaging_root(value=None) -> Path:
    """Resolve the external ImagingData root without machine-specific paths."""
    configured = value if value is not None else os.environ.get("ODYN_IMAGING_ROOT")
    if configured is None or not str(configured).strip():
        raise RuntimeError(
            "ImagingData root is not configured. Set ODYN_IMAGING_ROOT or pass "
            "--imaging-root to the command-line analysis."
        )
    return Path(configured).expanduser()
