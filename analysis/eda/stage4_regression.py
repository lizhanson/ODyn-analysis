"""Stage 4: does trial-to-trial arousal predict the neural response?

Odor identity is removed from both sides first, within state, so the estimate
comes from deviations around each odor's own mean rather than from the fact
that odors drive both locomotion and the neural response.  Running roughly
doubles at odor onset, so the raw correlation is confounded by construction and
is reported only as the contrast.

Awake trials only: locomotion is identically zero under ket/xyl.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .arousal import (align_to_auxiliary, continuous_channels, trial_covariates,
                     within_odor_deviation)
from .common import manifest_rows

ODOR_WINDOW = (0.0, 4.0)
BASELINE_WINDOW = (-5.0, -1.0)
MIN_TRIALS = 20
MIN_PUPIL_COVERAGE = 0.5


def window_mean(values, time_from_odor, window):
    """Per-trial mean of a trial x frame channel over one window."""
    values = np.asarray(values, float)
    time = np.asarray(time_from_odor, float)
    mask = (time >= window[0]) & (time < window[1])
    output = np.full(values.shape[0], np.nan)
    for trial in range(values.shape[0]):
        selected = mask[trial]
        if selected.any():
            output[trial] = np.nanmean(values[trial][selected])
    return output


def standardized_slope(y, x):
    """Correlation-scale slope of y on x, ignoring non-finite pairs."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    good = np.isfinite(y) & np.isfinite(x)
    if good.sum() < 8:
        return np.nan
    y, x = y[good], x[good]
    if np.std(x) <= 0 or np.std(y) <= 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def session_regression(row, cache_path):
    alignment = align_to_auxiliary(row)
    if alignment is None:
        return []
    covariates = trial_covariates(row, alignment)
    channels = continuous_channels(row, alignment)
    if covariates is None or "speed" not in channels:
        return []

    data = np.load(cache_path, allow_pickle=True)
    levels = [str(x) for x in data["state_levels"]]
    if "pre" not in levels:
        return []
    awake = data["state"] == levels.index("pre")
    if awake.sum() < MIN_TRIALS:
        return []
    if len(covariates) != len(data["odor_id"]):
        raise ValueError("alignment produced the wrong number of trials")

    time = np.asarray(channels["time_from_odor_s"], float)
    speed = np.abs(np.asarray(channels["speed"], float))
    running = window_mean(speed, time, ODOR_WINDOW)[awake]
    pupil_raw = channels.get("diameter_px")
    coverage = float(np.mean(np.isfinite(pupil_raw))) if pupil_raw is not None else 0.
    if pupil_raw is not None:
        pupil = (window_mean(pupil_raw, time, ODOR_WINDOW)
                 - window_mean(pupil_raw, time, BASELINE_WINDOW))[awake]
    else:
        pupil = np.full(awake.sum(), np.nan)

    odor = data["odor_id"][awake]
    state = np.array(["pre"]*int(awake.sum()))
    running_deviation = within_odor_deviation(running, odor, state)
    pupil_deviation = within_odor_deviation(pupil, odor, state)

    output = []
    populations = sorted({k.split("/")[0] for k in data.files if "/" in k})
    for population in populations:
        binned = data[f"{population}/binned"][:, awake, :]
        centres = data[f"{population}/bin_centre_s"]
        mask = (centres >= ODOR_WINDOW[0]) & (centres < ODOR_WINDOW[1])
        response = np.nanmean(binned[:, :, mask], axis=2)      # unit x trial
        unit_ids = data[f"{population}/unit_id"]
        for index in range(response.shape[0]):
            deviation = within_odor_deviation(response[index], odor, state)
            output.append({
                "group_id": int(row["group_id"]), "mouse": row["mouse"],
                "line": row["line"], "cohort": row["cohort"],
                "depth_class": row["depth_class"], "population": population,
                "unit_id": str(unit_ids[index]),
                "n_awake_trial": int(awake.sum()),
                "alignment_mode": alignment["mode"],
                "pupil_coverage": coverage,
                "running_r": standardized_slope(deviation, running_deviation),
                "pupil_r": (standardized_slope(deviation, pupil_deviation)
                            if coverage >= MIN_PUPIL_COVERAGE else np.nan),
                # Uncorrected, for comparison only: odor identity is left in,
                # so this is inflated by odors driving both signals.
                "running_r_uncorrected": standardized_slope(
                    response[index], running),
            })
    return output


def main(argv=None):
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    trials = args.output_dir / "trials"
    rows, failures = [], []
    for row in manifest_rows():
        cache = trials / f"group{int(row['group_id'])}_trials.npz"
        if not cache.exists():
            continue
        try:
            new = session_regression(row, cache)
            rows.extend(new)
            print(f"  {row['group_id']:>4} {len(new):>5} units", flush=True)
        except Exception as error:
            failures.append({"group_id": int(row["group_id"]),
                             "error": f"{type(error).__name__}: {error}"})
            print(f"  {row['group_id']:>4} FAILED {failures[-1]['error']}",
                  flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(args.output_dir / "stage4_regression.csv.gz", index=False,
                 compression="gzip")
    print(f"\nwrote {len(table):,} unit rows, {len(failures)} failures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
