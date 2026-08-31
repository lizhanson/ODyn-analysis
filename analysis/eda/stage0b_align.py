"""Can each grouped product be aligned to its auxiliary product?

The grouped files store trial_id as a positional arange, not the database id,
so the only available key is the odor/state sequence itself.  Joining by
position would silently misattribute arousal covariates wherever imaging
trials were dropped, so alignment is verified before any state analysis.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .common import decode, find_auxiliary, find_grouped, manifest_rows


def contiguous_offset(short, long):
    """Offset k with long[k:k+len(short)] == short, or None."""
    n, m = len(short), len(long)
    for k in range(m - n + 1):
        if np.array_equal(long[k:k+n], short):
            return k
    return None


def greedy_subsequence(short, long):
    """Indices of long, in order, matching short one-for-one; None if impossible."""
    indices, cursor = [], 0
    for value in short:
        while cursor < len(long) and long[cursor] != value:
            cursor += 1
        if cursor == len(long):
            return None
        indices.append(cursor)
        cursor += 1
    return np.asarray(indices)


def check(row) -> dict:
    import h5py

    record = {"group_id": int(row["group_id"]), "mouse": row["mouse"],
              "cohort": row["cohort"]}
    grouped, auxiliary = find_grouped(row), find_auxiliary(row)
    if grouped is None or auxiliary is None:
        record["alignment"] = "missing product"
        return record
    with h5py.File(grouped, "r") as handle:
        g_odor = handle["odor_id"][:]
        g_state = handle["state"][:]
        g_levels = decode(handle["state_levels"][:])
        g_trial = handle["trial_id"][:]
    with h5py.File(auxiliary, "r") as handle:
        a_odor = handle["trials/odor_id"][:]
        a_state = handle["trials/state"][:]
        a_levels = decode(handle["trials/state_levels"][:])
        a_trial = handle["trials/trial_id"][:]

    record["n_grouped"] = len(g_odor)
    record["n_aux"] = len(a_odor)
    record["grouped_trial_id_is_positional"] = bool(
        np.array_equal(g_trial, np.arange(len(g_trial))))
    record["state_levels_match"] = g_levels == a_levels
    # Compare on the (odor, state-name) pair so a state relabelling cannot
    # produce a spurious match.
    g_key = np.array([f"{o}|{g_levels[s]}" for o, s in zip(g_odor, g_state)])
    a_key = np.array([f"{o}|{a_levels[s]}" for o, s in zip(a_odor, a_state)])

    offset = contiguous_offset(g_key, a_key)
    if offset is not None:
        record["alignment"] = "contiguous"
        record["offset"] = int(offset)
        record["n_dropped"] = int(len(a_key) - len(g_key))
        return record
    indices = greedy_subsequence(g_key, a_key)
    if indices is not None:
        record["alignment"] = "scattered"
        record["offset"] = int(indices[0])
        record["n_dropped"] = int(len(a_key) - len(g_key))
        record["max_gap"] = int(np.max(np.diff(indices))) if len(indices) > 1 else 0
        return record
    record["alignment"] = "FAILED"
    record["n_dropped"] = int(len(a_key) - len(g_key))
    return record


def main(argv=None):
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for row in manifest_rows():
        try:
            records.append(check(row))
        except Exception as error:
            records.append({"group_id": int(row["group_id"]),
                            "cohort": row["cohort"],
                            "alignment": f"{type(error).__name__}: {error}"})
        print(f"  {records[-1]['group_id']:>4} {records[-1].get('alignment')}",
              flush=True)
    table = pd.DataFrame(records)
    table.to_csv(args.output_dir / "stage0b_alignment.csv", index=False)
    print()
    print(table.alignment.value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
