"""Stage 1/2: one pass over the traces produces the signed feature table.

Rows are unit x odor x state x population x session.  The same table answers
the diagnostic question (how often does a 4 s mean cancel a real response) and
supplies every downstream breadth, mass, balance and recruitment metric.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np

from .common import (BLANK_ODOR, decode, find_grouped, find_source,
                    manifest_rows, odor_groups, time_axis)
from .signed import signed_features

BIN_S = 0.5
BIN_RANGE = (-5.0, 14.0)

SCALAR_FIELDS = (
    "mean_z", "peak_positive_z", "peak_negative_z", "threshold_positive",
    "threshold_negative", "excitation_area", "suppression_area",
    "excited", "suppressed", "biphasic",
    "excitation_area_onset", "suppression_area_onset",
    "excitation_area_sustained", "suppression_area_sustained",
    "excitation_area_offset", "suppression_area_offset",
    "excited_onset", "suppressed_onset", "excited_sustained",
    "suppressed_sustained", "excited_offset", "suppressed_offset",
    "raw_mean_baseline", "raw_mean_onset", "raw_mean_sustained",
    "raw_mean_offset", "raw_mean_late",
)


def population_names(row, handle):
    wanted = ("units",) if row["objective"].lower() == "10x" else (
        "somas", "processes", "groups")
    return [name for name in wanted if name in handle]


def binned_trials(z, time_s, *, bin_s=BIN_S, span=BIN_RANGE):
    """unit x trial x time-bin means: the compact re-representation.

    Every downstream analysis -- window means, crossnobis on any window,
    time-resolved and spatiotemporal geometry, threshold sweeps -- is a
    combination of these bins, so the traces are read from the share once.
    """
    time_s = np.asarray(time_s, float)
    edges = np.arange(span[0], span[1] + bin_s, bin_s)
    stack, centres = [], []
    for start in edges[:-1]:
        mask = (time_s >= start) & (time_s < start + bin_s)
        if not mask.any():
            continue
        stack.append(np.nanmean(z[:, :, mask], axis=2))
        centres.append(start + bin_s/2)
    return np.stack(stack, axis=2).astype(np.float32), np.asarray(centres)


def session_features(row, *, tail_probability=.05, skip_groups=True,
                     cache_path=None):
    import h5py
    import pandas as pd

    grouped = find_grouped(row)
    if grouped is None:
        raise FileNotFoundError("no grouped product")
    source = find_source(row, grouped)
    if source is None:
        raise FileNotFoundError("no extraction round for the time axis")
    time_s = time_axis(source)
    groups_map = odor_groups()

    frames, caches = [], {}
    with h5py.File(grouped, "r") as handle:
        odor_ids = handle["odor_id"][:]
        states = handle["state"][:]
        levels = decode(handle["state_levels"][:])
        names = [n for n in population_names(row, handle)
                 if not (skip_groups and n == "groups")]
        for population in names:
            unit_ids = decode(handle[f"{population}/unit_id"][:])
            z = handle[f"{population}/z"][:]
            if z.shape[2] != time_s.size:
                raise ValueError(
                    f"{population}: {z.shape[2]} frames but time axis has "
                    f"{time_s.size}")
            if cache_path is not None:
                binned, centres = binned_trials(z, time_s)
                caches[population] = {
                    "binned": binned, "bin_centre_s": centres,
                    "unit_id": np.asarray(unit_ids, dtype=object),
                }
            for code, state_name in enumerate(levels):
                if not (states == code).any():
                    continue
                odor_levels = np.unique(odor_ids[states == code])
                result = signed_features(
                    z, time_s, odor_ids, states, code, odor_levels,
                    tail_probability=tail_probability)
                n_unit, n_odor = result["mean_z"].shape
                frame = pd.DataFrame({
                    "group_id": int(row["group_id"]), "mouse": row["mouse"],
                    "line": row["line"], "objective": row["objective"],
                    "depth_class": row["depth_class"], "cohort": row["cohort"],
                    "population": population, "state": state_name,
                    "unit_id": np.repeat(unit_ids, n_odor),
                    "odor_id": np.tile(odor_levels, n_unit),
                    "trials_per_odor": np.tile(result["trials_per_odor"], n_unit),
                })
                frame["odor_group"] = frame.odor_id.map(groups_map)
                frame["is_blank"] = frame.odor_id == BLANK_ODOR
                for field in SCALAR_FIELDS:
                    frame[field] = np.asarray(result[field]).ravel()
                frames.append(frame)
            del z
            gc.collect()
    if cache_path is not None:
        payload = {"odor_id": odor_ids, "state": states,
                   "state_levels": np.asarray(levels, dtype=object),
                   "time_s": time_s}
        for population, item in caches.items():
            for name, value in item.items():
                payload[f"{population}/{name}"] = value
        np.savez_compressed(cache_path, **payload)
    return pd.concat(frames, ignore_index=True)


def main(argv=None):
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tail-probability", type=float, default=.05)
    parser.add_argument("--objective", default=None)
    parser.add_argument("--include-groups", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    directory = args.output_dir / "features"
    trials = args.output_dir / "trials"
    directory.mkdir(parents=True, exist_ok=True)
    trials.mkdir(parents=True, exist_ok=True)

    failures = []
    for row in manifest_rows(args.objective):
        target = directory / f"group{int(row['group_id'])}_features.csv.gz"
        cache = trials / f"group{int(row['group_id'])}_trials.npz"
        if target.exists() and cache.exists() and not args.overwrite:
            print(f"  {row['group_id']:>4} cached", flush=True)
            continue
        try:
            table = session_features(
                row, tail_probability=args.tail_probability,
                skip_groups=not args.include_groups, cache_path=cache)
            table.to_csv(target, index=False, compression="gzip")
            print(f"  {row['group_id']:>4} {len(table):>7,} rows "
                  f"({table.population.nunique()} populations)", flush=True)
        except Exception as error:
            failures.append({"group_id": int(row["group_id"]),
                             "cohort": row["cohort"],
                             "error": f"{type(error).__name__}: {error}"})
            print(f"  {row['group_id']:>4} FAILED {failures[-1]['error']}",
                  flush=True)
    if failures:
        pd.DataFrame(failures).to_csv(
            args.output_dir / "stage1_failures.csv", index=False)
    print(f"\n{len(failures)} failures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
