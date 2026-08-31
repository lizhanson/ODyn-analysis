"""Shared path resolution and loading for the exploratory pass.

Read-only on the imaging share.  Nothing here writes into the repository or
into analysis/figures, which is being restructured concurrently.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np

from ..figures.paths import REPO_ROOT, imaging_root as _imaging_root

REPO = REPO_ROOT
MANIFEST = REPO / "analysis/stage0/ketxyl_16odor_session_manifest.csv"
ODOR_DICTIONARY = REPO / "analysis/stage0/odor_dictionary.csv"
ODOR_PANELS = REPO / "analysis/stage0/odor_panels.csv"


def IMAGING_ROOT_or_default():
    """ImagingData root from ODYN_IMAGING_ROOT, as the figure code resolves it."""
    return _imaging_root()


IMAGING_ROOT = Path(os.environ.get("ODYN_IMAGING_ROOT",
                                   "/Volumes/MossLab/ImagingData"))

BLANK_ODOR = 0
MIXTURE_PAIRS = ((17, 18), (31, 32), (39, 40))


def manifest_rows(objective=None):
    with MANIFEST.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["line"] = row["population"].split("-")[0]
        row["cohort"] = (f"{row['line']} {row['objective']}"
                         + (f" {row['depth_class']}"
                            if row["depth_class"] not in ("", "na") else ""))
    if objective:
        rows = [r for r in rows if r["objective"].lower() == objective.lower()]
    return rows


def experiment_dir(row) -> Path:
    return IMAGING_ROOT / row["date"] / row["mouse"] / row["exp"]


def output_dir(row) -> Path:
    return experiment_dir(row) / "processed" / "python"


def find_grouped(row) -> Path | None:
    """Newest grouped product, preferring the per-trial/median rebuild."""
    objective = row["objective"].lower()
    candidates = list(output_dir(row).glob(f"*_{objective}_grouped.h5"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: ("pertrial_median" in p.name,
                                          p.stat().st_mtime_ns))


def find_source(row, grouped_path=None) -> Path | None:
    """The extraction round the grouped product was built from."""
    import h5py

    if grouped_path is not None:
        with h5py.File(grouped_path, "r") as handle:
            for key in ("source_round", "source_component_round"):
                value = handle.attrs.get(key)
                if value is None:
                    continue
                candidate = Path(value.decode() if isinstance(value, bytes)
                                 else str(value))
                if candidate.exists():
                    return candidate
    rounds = [p for p in output_dir(row).glob("*_processed_*.h5")
              if "masks_processed" not in p.name and "grouped" not in p.name]
    return max(rounds, key=lambda p: p.stat().st_mtime_ns) if rounds else None


def find_auxiliary(row) -> Path | None:
    candidates = list((output_dir(row) / "aux").glob("*_auxiliary.h5"))
    return max(candidates, key=lambda p: p.stat().st_mtime_ns) if candidates else None


def decode(values) -> tuple[str, ...]:
    return tuple(v.decode() if isinstance(v, bytes) else str(v) for v in values)


def populations(row) -> tuple[str, ...]:
    return ("units",) if row["objective"].lower() == "10x" else (
        "groups", "somas", "processes")


def odor_groups() -> dict[int, str]:
    """Map odor_id to the reporting group used throughout the EDA."""
    with ODOR_DICTIONARY.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    mapping = {}
    for row in rows:
        odor = int(row["odor_id"])
        role = row["role"]
        odor_class = (row["odor_class"] or "").strip()
        if role == "control":
            mapping[odor] = "blank"
        elif role == "mix":
            mapping[odor] = "mixture"
        elif odor_class == "I":
            mapping[odor] = "class I single"
        elif odor_class == "I and II":
            # Kept separate rather than forced into either class: this odor
            # would otherwise silently decide the class I contrast.
            mapping[odor] = "class I/II single"
        else:
            mapping[odor] = "other single"
    return mapping


def mixture_components(panel="panel_16") -> dict[int, tuple[int, ...]]:
    """Component odor ids for each mixture, read from the panel table."""
    with ODOR_PANELS.open(newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r["panel"] == panel]
    output = {}
    for row in rows:
        odor = int(row["odor_id"])
        parts = tuple(
            int(float(row[key])) for key in
            ("component_a_odor_id", "component_b_odor_id")
            if row.get(key) not in (None, "")
        )
        if len(parts) > 1:
            output[odor] = parts
    return output


def panel_odors(panel="panel_16") -> tuple[int, ...]:
    with ODOR_PANELS.open(newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r["panel"] == panel]
    return tuple(sorted({int(r["odor_id"]) for r in rows}))


def time_axis(source_path) -> np.ndarray:
    import h5py

    with h5py.File(source_path, "r") as handle:
        return handle["traces/time_s"][:]
