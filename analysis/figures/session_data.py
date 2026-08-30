"""Load the final 10x/20x grouped products into a common representation."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ResponseSession:
    group_id: int
    mouse: str
    line: str
    objective: str
    depth_class: str
    population: str
    path: Path
    unit_ids: np.ndarray
    trial_ids: np.ndarray
    odor_ids: np.ndarray
    states: np.ndarray
    state_levels: tuple[str, ...]
    mean_z: np.ndarray             # unit x trial
    z: np.ndarray                  # unit x trial x frame


def _decode(values) -> tuple[str, ...]:
    return tuple(v.decode() if isinstance(v, bytes) else str(v) for v in values)


def manifest(path) -> list[dict]:
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def experiment_dir(row, imaging_root) -> Path:
    return Path(imaging_root) / row["date"] / row["mouse"] / row["exp"]


def latest_grouped_file(row, imaging_root) -> Path | None:
    """Prefer the current per-trial/median rebuild over older grouped products."""
    output = experiment_dir(row, imaging_root) / "processed" / "python"
    objective = row["objective"].lower()
    candidates = list(output.glob(f"*_{objective}_grouped.h5"))
    if not candidates:
        return None
    def rank(path):
        return ("pertrial_median" in path.name, path.stat().st_mtime_ns)
    return max(candidates, key=rank)


def load_grouped(row, path, *, population=None) -> ResponseSession:
    """Load 10x units or one of the 20x groups/somas/processes populations."""
    import h5py

    path = Path(path)
    objective = row["objective"].lower()
    if population is None:
        population = "units" if objective == "10x" else "groups"
    with h5py.File(path, "r") as handle:
        root = handle[population]
        trials = {
            name: handle[name][:]
            for name in ("trial_id", "odor_id", "state")
        }
        response = root["responses/odor/mean_z"][:]
        z = root["z"][:]
        unit_ids = root["unit_id"][:]
        levels = _decode(handle["state_levels"][:])
    return ResponseSession(
        group_id=int(row["group_id"]), mouse=row["mouse"],
        line=row["population"].split("-")[0], objective=objective,
        depth_class=row.get("depth_class", ""), population=population,
        path=path, unit_ids=np.asarray(unit_ids), trial_ids=trials["trial_id"],
        odor_ids=trials["odor_id"], states=trials["state"],
        state_levels=levels, mean_z=response, z=z,
    )


def available_sessions(manifest_path, imaging_root, *, objective=None):
    output = []
    for row in manifest(manifest_path):
        if objective and row["objective"].lower() != objective.lower():
            continue
        path = latest_grouped_file(row, imaging_root)
        output.append({**row, "grouped_path": str(path) if path else None,
                       "available": path is not None})
    return output
