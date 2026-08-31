"""Time-resolved and odor-specific running coupling.

The stage 4 estimate averaged 0-4 s, which cannot see a coupling that changes
sign within the response, nor one confined to particular odors.  Three views
are computed from the cached bins:

A  coupling of each 0.5 s bin to the odor-window running deviation, so a
   suppressive phase at a specific latency would be visible;
B  the same, computed separately within odor groups, so a coupling confined to
   the reciprocal pair 17/18 would be visible;
C  coupling split by whether that unit-odor response was excited or suppressed,
   which is the form the arousal-suppression hypothesis actually predicts.

Odor identity is removed within every subset before correlating, so nothing
here is carried by odors driving both signals.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .arousal import (align_to_auxiliary, continuous_channels, trial_covariates,
                     within_odor_deviation)
from .common import manifest_rows, odor_groups
from .stage4_regression import window_mean

ODOR_WINDOW = (0.0, 4.0)
MIN_TRIALS = 20
MIN_GROUP_TRIALS = 8


def bin_channel(values, time_from_odor, centres, width=0.5):
    """Trial x bin means of a trial x frame channel on the neural bin grid."""
    values = np.asarray(values, float)
    time = np.asarray(time_from_odor, float)
    output = np.full((values.shape[0], len(centres)), np.nan)
    for index, centre in enumerate(centres):
        mask = (time >= centre - width/2) & (time < centre + width/2)
        for trial in range(values.shape[0]):
            selected = mask[trial]
            if selected.any():
                output[trial, index] = np.nanmean(values[trial][selected])
    return output


def columnwise_correlation(deviation, covariate):
    """Correlation over trials for every unit x bin against one trial vector.

    `deviation` is unit x trial x bin and already odor-centred; `covariate` is
    a trial vector.  NaNs are handled by zero-filling paired with a validity
    mask so each cell uses only the trials both sides observed.
    """
    x = np.asarray(deviation, float)
    r = np.asarray(covariate, float)
    valid = np.isfinite(x) & np.isfinite(r)[None, :, None]
    xf = np.where(valid, x, 0.)
    rf = np.where(valid, r[None, :, None], 0.)
    n = valid.sum(axis=1)
    sx, sr = xf.sum(axis=1), rf.sum(axis=1)
    sxx = (xf*xf).sum(axis=1)
    srr = (rf*rf).sum(axis=1)
    sxr = (xf*rf).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        numerator = sxr - sx*sr/n
        denominator = np.sqrt((sxx - sx*sx/n)*(srr - sr*sr/n))
        result = np.where((n >= 8) & (denominator > 0), numerator/denominator,
                          np.nan)
    return result


def centre_by_odor(response, odor, states):
    """Odor-centred deviation for a unit x trial x bin array."""
    output = np.full(response.shape, np.nan)
    for index in range(response.shape[0]):
        for b in range(response.shape[2]):
            output[index, :, b] = within_odor_deviation(
                response[index, :, b], odor, states)
    return output


def session_rows(row, cache_path, features):
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
    if awake.sum() < MIN_TRIALS or len(covariates) != len(data["odor_id"]):
        return []

    time = np.asarray(channels["time_from_odor_s"], float)
    speed = np.abs(np.asarray(channels["speed"], float))
    running = window_mean(speed, time, ODOR_WINDOW)[awake]
    odor = data["odor_id"][awake]
    states = np.array(["pre"]*int(awake.sum()))
    running_deviation = within_odor_deviation(running, odor, states)
    groups = odor_groups()

    output = []
    for population in sorted({k.split("/")[0] for k in data.files if "/" in k}):
        binned = data[f"{population}/binned"][:, awake, :]
        centres = data[f"{population}/bin_centre_s"]
        unit_ids = np.asarray([str(u) for u in data[f"{population}/unit_id"]])
        deviation = centre_by_odor(binned, odor, states)
        common = {"group_id": int(row["group_id"]), "mouse": row["mouse"],
                  "line": row["line"], "cohort": row["cohort"],
                  "population": population}

        # A: time course against the odor-window running deviation.
        coupling = columnwise_correlation(deviation, running_deviation)
        for index, centre in enumerate(centres):
            output.append(common | {"analysis": "time_course",
                                    "subset": "all", "bin_centre_s": float(centre),
                                    "median_r": float(np.nanmedian(coupling[:, index])),
                                    "n_unit": int(np.isfinite(coupling[:, index]).sum())})

        # B: the same, within odor groups, with 17/18 kept separate.
        label = np.where(np.isin(odor, (17, 18)), "pair 17/18",
                         np.array([groups.get(int(o), "other") for o in odor]))
        for name in np.unique(label):
            rows_in = label == name
            if rows_in.sum() < MIN_GROUP_TRIALS:
                continue
            sub = centre_by_odor(binned[:, rows_in, :], odor[rows_in],
                                 states[rows_in])
            sub_running = within_odor_deviation(running[rows_in], odor[rows_in],
                                                states[rows_in])
            group_coupling = columnwise_correlation(sub, sub_running)
            for index, centre in enumerate(centres):
                output.append(common | {
                    "analysis": "odor_group", "subset": str(name),
                    "bin_centre_s": float(centre),
                    "median_r": float(np.nanmedian(group_coupling[:, index])),
                    "n_unit": int(np.isfinite(group_coupling[:, index]).sum()),
                    "n_trial": int(rows_in.sum())})

        # C: split by the sign of that unit's response, from the feature table.
        table = features[(features.group_id == int(row["group_id"]))
                         & (features.population == population)
                         & (features.state == "pre")]
        if len(table):
            mask = (centres >= ODOR_WINDOW[0]) & (centres < ODOR_WINDOW[1])
            window_response = np.nanmean(deviation[:, :, mask], axis=2)
            unit_index = {unit: i for i, unit in enumerate(unit_ids)}
            for kind, column in (("excited", "excited"),
                                 ("suppressed", "suppressed")):
                selected = table[table[column]]
                values = []
                for unit, odor_id in zip(selected.unit_id, selected.odor_id):
                    i = unit_index.get(str(unit))
                    if i is None:
                        continue
                    trials = odor == odor_id
                    if trials.sum() < 4:
                        continue
                    a = window_response[i][trials]
                    b = running_deviation[trials]
                    good = np.isfinite(a) & np.isfinite(b)
                    if good.sum() < 4 or np.std(a[good]) == 0 or np.std(b[good]) == 0:
                        continue
                    values.append(np.corrcoef(a[good], b[good])[0, 1])
                if values:
                    output.append(common | {
                        "analysis": "response_sign", "subset": kind,
                        "bin_centre_s": np.nan,
                        "median_r": float(np.nanmedian(values)),
                        "n_unit": len(values)})
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    from .stage2_summarise import load_features
    features = load_features(args.output_dir / "features")
    features = features[~features.is_blank]
    rows, failures = [], []
    for row in manifest_rows():
        cache = args.output_dir / "trials" / f"group{int(row['group_id'])}_trials.npz"
        if not cache.exists():
            continue
        try:
            new = session_rows(row, cache, features)
            rows.extend(new)
            print(f"  {row['group_id']:>4} {len(new):>5} rows", flush=True)
        except Exception as error:
            failures.append({"group_id": int(row["group_id"]),
                             "error": f"{type(error).__name__}: {error}"})
            print(f"  {row['group_id']:>4} FAILED {failures[-1]['error']}",
                  flush=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "stage6_arousal_time.csv",
                              index=False)
    print(f"\nwrote {len(rows):,} rows, {len(failures)} failures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
