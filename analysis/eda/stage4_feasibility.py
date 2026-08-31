"""Stage 4 feasibility: is there enough arousal variation to regress against?

Reports, per session and state, how much the animal actually ran and how much
the pupil actually moved, and how many spontaneous running onsets fall in the
odor-free part of each trial epoch.  A state regression is only worth building
where these are non-trivial.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .arousal import (align_to_auxiliary, continuous_channels, detect_onsets,
                     odor_free_mask, trial_covariates)
from .common import manifest_rows

RUNNING_THRESHOLD_CM_S = 1.0


def session_report(row):
    alignment = align_to_auxiliary(row)
    if alignment is None:
        return [{"group_id": int(row["group_id"]), "cohort": row["cohort"],
                 "status": "no alignment"}]
    covariates = trial_covariates(row, alignment)
    channels = continuous_channels(row, alignment)
    if covariates is None or "speed" not in channels:
        return [{"group_id": int(row["group_id"]), "cohort": row["cohort"],
                 "status": "no treadmill"}]

    speed = np.abs(np.asarray(channels["speed"], float))
    pupil = np.asarray(channels.get("diameter_px", np.full(speed.shape, np.nan)),
                       float)
    free = odor_free_mask(channels["time_from_odor_s"])
    output = []
    for state in ("pre", "post"):
        rows = (covariates.state == state).to_numpy()
        if not rows.any():
            continue
        state_speed, state_free = speed[rows], free[rows]
        state_pupil = pupil[rows]
        spontaneous = state_speed.copy()
        spontaneous[~state_free] = np.nan
        onsets = detect_onsets(state_speed, state_free,
                               threshold=RUNNING_THRESHOLD_CM_S,
                               min_quiet_samples=20, min_run_samples=10)
        # Pupil is only interpretable where the eye was actually tracked.
        finite_pupil = state_pupil[np.isfinite(state_pupil)]
        record = {
            "group_id": int(row["group_id"]), "mouse": row["mouse"],
            "cohort": row["cohort"], "state": state, "status": "ok",
            "alignment_mode": alignment["mode"],
            "n_trial": int(rows.sum()),
            "running_fraction_all": float(np.nanmean(
                state_speed > RUNNING_THRESHOLD_CM_S)),
            "running_fraction_odor_free": float(np.nanmean(
                spontaneous > RUNNING_THRESHOLD_CM_S)),
            "trials_with_running": float(np.mean(np.nanmax(
                state_speed, axis=1) > RUNNING_THRESHOLD_CM_S)),
            "median_speed_cm_s": float(np.nanmedian(state_speed)),
            "p90_speed_cm_s": float(np.nanpercentile(state_speed, 90)),
            "n_spontaneous_run_onsets": int(len(onsets)),
            "odor_free_seconds": float(np.sum(state_free)
                                       / max(1, rows.sum()) * rows.sum()
                                       / _rate(channels)),
            "pupil_coverage": float(np.mean(np.isfinite(state_pupil))),
            "pupil_median_px": float(np.median(finite_pupil))
            if finite_pupil.size else np.nan,
            "pupil_cv": float(np.std(finite_pupil)/np.mean(finite_pupil))
            if finite_pupil.size and np.mean(finite_pupil) > 0 else np.nan,
            "pupil_trial_range_px": float(np.nanmax(np.nanmedian(
                state_pupil, axis=1)) - np.nanmin(np.nanmedian(
                    state_pupil, axis=1))) if finite_pupil.size else np.nan,
        }
        output.append(record)
    return output


def _rate(channels):
    time = np.asarray(channels["time_from_odor_s"], float)
    return 1.0/np.nanmedian(np.diff(time[0]))


def main(argv=None):
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in manifest_rows():
        try:
            rows.extend(session_report(row))
        except Exception as error:
            rows.append({"group_id": int(row["group_id"]),
                         "cohort": row["cohort"],
                         "status": f"{type(error).__name__}: {error}"})
        print(f"  {row['group_id']:>4} {rows[-1].get('status')}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(args.output_dir / "stage4_feasibility.csv", index=False)
    print(f"\nwrote {len(table)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
