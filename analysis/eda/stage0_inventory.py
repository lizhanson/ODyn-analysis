"""Stage 0: what data actually exist, and what is estimable from them.

Trials per odor per state is the gate on every later analysis, so it is
recorded per session rather than assumed from the protocol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .common import (BLANK_ODOR, decode, find_auxiliary, find_grouped,
                    find_source, manifest_rows, populations, time_axis)


def session_inventory(row) -> dict:
    import h5py

    record = {
        "group_id": int(row["group_id"]), "mouse": row["mouse"],
        "date": row["date"], "exp": row["exp"], "line": row["line"],
        "objective": row["objective"], "depth_class": row["depth_class"],
        "cohort": row["cohort"], "depth_um": row["depth_um"],
        "metadata_flag": row["metadata_flag"],
    }
    grouped = find_grouped(row)
    record["grouped"] = grouped.name if grouped else ""
    record["pertrial_median"] = bool(grouped and "pertrial_median" in grouped.name)
    if grouped is None:
        record["status"] = "no grouped product"
        return record

    with h5py.File(grouped, "r") as handle:
        odor = handle["odor_id"][:]
        state = handle["state"][:]
        levels = decode(handle["state_levels"][:])
        record["baseline_sd_mode"] = str(handle.attrs.get("baseline_sd_mode", ""))
        available = [key for key in populations(row) if key in handle]
        record["populations"] = "|".join(available)
        for key in available:
            record[f"n_{key}"] = int(handle[f"{key}/unit_id"].shape[0])
            record[f"{key}_z_shape"] = str(tuple(handle[f"{key}/z"].shape))
        record["n_trial"] = int(len(odor))

    record["states"] = "|".join(levels)
    counts = {}
    for code, name in enumerate(levels):
        selected = state == code
        record[f"n_trial_{name}"] = int(selected.sum())
        if not selected.any():
            continue
        odors, per_odor = np.unique(odor[selected], return_counts=True)
        counts[name] = dict(zip(odors.tolist(), per_odor.tolist()))
        real = per_odor[odors != BLANK_ODOR]
        record[f"n_odor_{name}"] = int((odors != BLANK_ODOR).sum())
        record[f"trials_per_odor_min_{name}"] = int(real.min()) if real.size else 0
        record[f"trials_per_odor_median_{name}"] = float(np.median(real)) if real.size else 0
        record[f"trials_per_odor_max_{name}"] = int(real.max()) if real.size else 0
        record[f"n_blank_{name}"] = int(counts[name].get(BLANK_ODOR, 0))
    record["trial_counts_json"] = json.dumps(counts)
    record["has_pre_and_post"] = bool({"pre", "post"} <= set(levels)
                                      and record.get("n_trial_pre", 0) > 0
                                      and record.get("n_trial_post", 0) > 0)

    source = find_source(row, grouped)
    record["source"] = source.name if source else ""
    if source is not None:
        try:
            time_s = time_axis(source)
            record["time_min_s"] = float(time_s.min())
            record["time_max_s"] = float(time_s.max())
            record["n_frame"] = int(time_s.size)
            record["frame_rate_hz"] = float(1.0 / np.median(np.diff(time_s)))
            # Non-odor time available inside each trial epoch: the pre-odor
            # baseline plus everything after the 4 s odor and its 4 s post
            # window.  This is what a spontaneous-event analysis could use.
            record["pre_odor_s"] = float(-time_s.min())
            record["after_post_odor_s"] = float(max(0.0, time_s.max() - 8.0))
        except Exception as error:
            record["time_error"] = f"{type(error).__name__}: {error}"

    auxiliary = find_auxiliary(row)
    record["auxiliary"] = auxiliary.name if auxiliary else ""
    if auxiliary is not None:
        with h5py.File(auxiliary, "r") as handle:
            for name in ("pupil_available", "respiration_available",
                         "treadmill_available"):
                record[name] = int(handle.attrs.get(name, 0))
            record["aux_n_trial"] = int(handle.attrs.get("n_trial", 0))
            record["aux_n_frame"] = int(handle.attrs.get("n_frame", 0))
            record["aux_frame_rate_hz"] = float(handle.attrs.get("frame_rate_hz", np.nan))
            if "treadmill/speed" in handle:
                speed = handle["treadmill/speed"][:]
                record["aux_speed_shape"] = str(tuple(speed.shape))
                record["median_speed_cm_s"] = float(np.nanmedian(np.abs(speed)))
            if "acquisition/time_from_odor_s" in handle:
                t = handle["acquisition/time_from_odor_s"][:]
                record["aux_time_min_s"] = float(np.nanmin(t))
                record["aux_time_max_s"] = float(np.nanmax(t))
    record["status"] = "ok"
    return record


def main(argv=None):
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--objective", default=None)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for row in manifest_rows(args.objective):
        try:
            records.append(session_inventory(row))
        except Exception as error:
            records.append({
                "group_id": int(row["group_id"]), "mouse": row["mouse"],
                "cohort": row["cohort"],
                "status": f"{type(error).__name__}: {error}",
            })
        print(f"  {records[-1]['group_id']:>4} {records[-1].get('status','?')}",
              flush=True)
    table = pd.DataFrame(records)
    table.to_csv(args.output_dir / "stage0_inventory.csv", index=False)
    print(f"\nWrote {args.output_dir / 'stage0_inventory.csv'} ({len(table)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
