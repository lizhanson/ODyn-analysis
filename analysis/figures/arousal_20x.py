"""Within-odor awake arousal associations for 20x cellular populations."""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from .cellular_20x import TemporalWindows, _common, _window, load_population


def find_auxiliary(row, imaging_root):
    directory = (Path(imaging_root) / row["date"] / row["mouse"] / row["exp"] /
                 "processed/python/aux")
    matches = sorted(directory.glob(f"group{int(row['group_id'])}_*_auxiliary.h5"))
    return matches[-1] if matches else None


def _rank_correlation(a, b):
    from scipy.stats import rankdata
    a, b = np.asarray(a, float), np.asarray(b, float)
    valid = np.isfinite(a) & np.isfinite(b)
    if np.sum(valid) < 8:
        return np.nan
    ar, br = rankdata(a[valid]), rankdata(b[valid])
    if np.std(ar) == 0 or np.std(br) == 0:
        return np.nan
    return float(np.corrcoef(ar, br)[0, 1])


def _within_odor(values, odors):
    values, odors = np.asarray(values, float), np.asarray(odors)
    output = np.full_like(values, np.nan)
    for odor in np.unique(odors):
        selected = odors == odor
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            center = np.nanmedian(values[..., selected], axis=-1, keepdims=True)
        output[..., selected] = values[..., selected] - center
    return output


def arousal_association_table(grouped_path, aux_path, row, population, *,
                              windows=TemporalWindows(), minimum_pupil_coverage=.7):
    """Correlate within-odor neural deviations with pupil and running deviations."""
    import h5py

    data = load_population(grouped_path, population)
    with h5py.File(aux_path) as handle:
        aux_trial = handle["trials/trial_id"][:]
        aux_time = handle["acquisition/time_from_odor_s"][:]
        pupil = handle["pupil/diameter_px"][:]
        pupil_valid = (handle["pupil/alignment_valid"][:] &
                       ~handle["pupil/blink"][:] & ~handle["pupil/clipped"][:])
        speed = np.abs(handle["treadmill/speed"][:])
    lookup = {int(trial): index for index, trial in enumerate(aux_trial)}
    paired = [(index, lookup.get(int(trial))) for index, trial in enumerate(data["trial_id"])]
    paired = [(a, b) for a, b in paired if b is not None]
    if not paired:
        return pd.DataFrame()
    imaging_index = np.asarray([a for a, _ in paired]); aux_index = np.asarray([b for _, b in paired])
    state = data["state"][imaging_index]
    pre_code = list(data["state_levels"]).index("pre")
    awake = state == pre_code
    imaging_index, aux_index = imaging_index[awake], aux_index[awake]
    odors = data["odor_id"][imaging_index]

    neural_time = data["time_s"]
    neural_mask = _window(neural_time, windows.odor)
    neural = data["z"][:, imaging_index, :][:, :, neural_mask]
    negative_auc = np.trapezoid(np.maximum(-neural, 0), neural_time[neural_mask], axis=2)
    positive_auc = np.trapezoid(np.maximum(neural, 0), neural_time[neural_mask], axis=2)

    pupil_delta, running, coverage = [], [], []
    for index in aux_index:
        time = aux_time[index]
        baseline = _window(time, (-5., 0.)); odor_window = _window(time, windows.odor)
        valid_base = baseline & pupil_valid[index]; valid_odor = odor_window & pupil_valid[index]
        possible = max(1, int(np.sum(baseline) + np.sum(odor_window)))
        coverage.append((np.sum(valid_base) + np.sum(valid_odor)) / possible)
        pupil_delta.append(np.nanmedian(pupil[index, valid_odor]) -
                           np.nanmedian(pupil[index, valid_base])
                           if np.any(valid_base) and np.any(valid_odor) else np.nan)
        running.append(np.nanmean(speed[index, odor_window]))
    pupil_delta, running, coverage = map(np.asarray, (pupil_delta, running, coverage))
    pupil_delta[coverage < minimum_pupil_coverage] = np.nan

    negative_residual = _within_odor(negative_auc, odors)
    positive_residual = _within_odor(positive_auc, odors)
    pupil_residual = _within_odor(pupil_delta, odors)
    running_residual = _within_odor(running, odors)
    common = _common(row, population)
    rows = []
    for index, unit_id in enumerate(data["unit_id"]):
        rows.append(common | {
            "unit_id": unit_id, "n_awake_trials": len(odors),
            "median_pupil_coverage": float(np.nanmedian(coverage)),
            "suppression_vs_pupil_rho": _rank_correlation(
                negative_residual[index], pupil_residual),
            "suppression_vs_running_rho": _rank_correlation(
                negative_residual[index], running_residual),
            "excitation_vs_pupil_rho": _rank_correlation(
                positive_residual[index], pupil_residual),
            "excitation_vs_running_rho": _rank_correlation(
                positive_residual[index], running_residual),
        })
    return pd.DataFrame(rows)
