"""Is mineral oil a null stimulus?  Population time course, blank vs odor."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
from .common import (decode, find_grouped, find_source, manifest_rows, time_axis)

WINDOWS = [("base", -4., -1.), ("onset", 0., 1.), ("sustained", 1., 4.),
           ("offset", 4., 8.), ("late", 8., 14.)]


def profile(row, population):
    import h5py
    grouped = find_grouped(row)
    if grouped is None:
        return []
    time_s = time_axis(find_source(row, grouped))
    with h5py.File(grouped, "r") as handle:
        if population not in handle:
            return []
        odor = handle["odor_id"][:]; state = handle["state"][:]
        levels = decode(handle["state_levels"][:])
        z = handle[f"{population}/z"][:]
    output = []
    for code, name in enumerate(levels):
        if not (state == code).any():
            continue
        for label, is_blank in (("blank", True), ("odor", False)):
            mask = (state == code) & ((odor == 0) if is_blank else (odor != 0))
            if not mask.any():
                continue
            # Median across trials per unit, then median across units: the
            # population centre, not an average dominated by a few loud units.
            unit_trace = np.nanmedian(z[:, mask, :], axis=1)
            centre = np.nanmedian(unit_trace, axis=0)
            record = {"group_id": int(row["group_id"]), "mouse": row["mouse"],
                      "cohort": row["cohort"], "population": population,
                      "state": name, "stimulus": label,
                      "n_trial": int(mask.sum()), "n_unit": z.shape[0]}
            for window, start, stop in WINDOWS:
                selected = (time_s >= start) & (time_s < stop)
                record[window] = float(np.nanmean(centre[selected]))
            output.append(record)
    return output


def main(argv=None):
    import pandas as pd
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--groups", nargs="*", type=int, default=[])
    args = parser.parse_args(argv)
    rows = []
    for row in manifest_rows():
        if args.groups and int(row["group_id"]) not in args.groups:
            continue
        population = "units" if row["objective"].lower() == "10x" else "somas"
        try:
            rows.extend(profile(row, population))
            print(f"  {row['group_id']:>4} ok", flush=True)
        except Exception as error:
            print(f"  {row['group_id']:>4} {type(error).__name__}: {error}",
                  flush=True)
    table = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "blank_profile.csv", index=False)
    print(f"\nwrote {len(table)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
