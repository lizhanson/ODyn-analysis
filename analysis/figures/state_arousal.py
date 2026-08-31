"""Trial-resolved awake-state covariation between neural activity and arousal.

Two deliberately simple analyses live here:

1. Baseline covariation asks whether detrended pre-odor fluorescence varies
   with pupil diameter, treadmill speed, or respiration across awake trials.
2. Response-distribution evolution asks whether high-pupil-dilation or
   high-running trials change the negative and positive tails of the neural
   response distribution during odor delivery.

Neither analysis assigns pupil or running to a particular neuromodulator.
The distribution analysis can remove each odor's median response time course
before pooling trials, so an association is not produced solely because a
particular odor both evokes movement and recruits a particular neural pattern.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from .population_metrics import (_common, _decode, _source_path, _window,
                                 load_population)


def _population(source, population):
    """Accept an already loaded population to avoid rereading large H5 arrays."""
    return source if isinstance(source, dict) else load_population(source, population)


def load_baseline_population(grouped_path, population):
    """Load only trial metadata and baseline F, avoiding the large z array."""
    import h5py

    grouped_path = Path(grouped_path)
    with h5py.File(grouped_path) as handle:
        source = _source_path(grouped_path, handle)
        root = handle[population]
        data = {
            "unit_id": _decode(root["unit_id"][:]),
            "baseline_mean": root["baseline_mean"][:],
            "odor_id": handle["odor_id"][:], "state": handle["state"][:],
            "state_levels": _decode(handle["state_levels"][:]),
            "trial_id": handle["trial_id"][:],
        }
    with h5py.File(source) as handle:
        source_trial_ids = handle["trials/trial_id"][:]
    if len(source_trial_ids) != data["baseline_mean"].shape[1]:
        raise ValueError("source/grouped trial counts do not align")
    data["trial_id"] = source_trial_ids
    return data


def find_auxiliary(row, imaging_root):
    directory = (Path(imaging_root) / row["date"] / row["mouse"] / row["exp"] /
                 "processed/python/aux")
    matches = sorted(directory.glob(f"group{int(row['group_id'])}_*_auxiliary.h5"))
    return matches[-1] if matches else None


def _rank_correlation(a, b, minimum=8):
    from scipy.stats import rankdata

    a, b = np.asarray(a, float), np.asarray(b, float)
    valid = np.isfinite(a) & np.isfinite(b)
    if np.sum(valid) < minimum:
        return np.nan
    ar, br = rankdata(a[valid]), rankdata(b[valid])
    if np.nanstd(ar) == 0 or np.nanstd(br) == 0:
        return np.nan
    return float(np.corrcoef(ar, br)[0, 1])


def _partial_rank_correlation(a, b, controls, minimum=8):
    """Spearman partial correlation after linearly removing ranked controls."""
    from scipy.stats import rankdata

    a, b = np.asarray(a, float), np.asarray(b, float)
    controls = np.column_stack([np.asarray(value, float) for value in controls])
    valid = np.isfinite(a) & np.isfinite(b) & np.isfinite(controls).all(axis=1)
    if np.sum(valid) < minimum:
        return np.nan
    ar, br = rankdata(a[valid]), rankdata(b[valid])
    ranked_controls = np.column_stack([
        rankdata(controls[valid, column]) for column in range(controls.shape[1])])
    design = np.column_stack([np.ones(np.sum(valid)), ranked_controls])
    ar = ar-design @ np.linalg.lstsq(design, ar, rcond=None)[0]
    br = br-design @ np.linalg.lstsq(design, br, rcond=None)[0]
    if np.std(ar) == 0 or np.std(br) == 0:
        return np.nan
    return float(np.corrcoef(ar, br)[0, 1])


def _zscore(values):
    values = np.asarray(values, float)
    if not np.any(np.isfinite(values)):
        return np.full_like(values, np.nan)
    center = np.nanmedian(values)
    scale = np.nanstd(values)
    return ((values - center) / scale if np.isfinite(scale) and scale > 0
            else np.full_like(values, np.nan))


def _linear_detrend_rows(values, trial_order):
    """Remove a per-unit linear trial-order trend without filling missing data."""
    values = np.asarray(values, float)
    x = np.asarray(trial_order, float)
    output = np.full_like(values, np.nan)
    for unit in range(values.shape[0]):
        valid = np.isfinite(values[unit]) & np.isfinite(x)
        if np.sum(valid) < 3:
            continue
        design = np.column_stack([np.ones(np.sum(valid)), x[valid] - np.mean(x[valid])])
        coefficients = np.linalg.lstsq(design, values[unit, valid], rcond=None)[0]
        output[unit, valid] = values[unit, valid] - design @ coefficients
    return output


def _residualize(values, covariate):
    """Remove a linear covariate effect while retaining the original center."""
    values, covariate = np.asarray(values, float), np.asarray(covariate, float)
    output = np.full_like(values, np.nan)
    valid = np.isfinite(values) & np.isfinite(covariate)
    if np.sum(valid) < 3:
        return output
    x = covariate[valid] - np.mean(covariate[valid])
    design = np.column_stack([np.ones(np.sum(valid)), x])
    fit = design @ np.linalg.lstsq(design, values[valid], rcond=None)[0]
    output[valid] = values[valid] - fit + np.nanmedian(values[valid])
    return output


def _odor_center(values, odors):
    """Remove each unit x odor median; supports scalar or time-resolved values."""
    values, odors = np.asarray(values, float), np.asarray(odors)
    output = np.full_like(values, np.nan)
    # values is unit x trial [x time]
    for odor in np.unique(odors):
        selected = odors == odor
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            center = np.nanmedian(values[:, selected, ...], axis=1, keepdims=True)
        output[:, selected, ...] = values[:, selected, ...] - center
    return output


def _matched_awake(data, aux_path):
    """Return aligned awake imaging/auxiliary indices and loaded aux arrays."""
    import h5py

    with h5py.File(aux_path) as handle:
        aux = {
            "trial_id": handle["trials/trial_id"][:],
            "time_s": handle["acquisition/time_from_odor_s"][:],
            "pupil": handle["pupil/diameter_px"][:],
            "pupil_valid": (handle["pupil/alignment_valid"][:] &
                            ~handle["pupil/blink"][:] &
                            ~handle["pupil/clipped"][:]),
            "speed": np.abs(handle["treadmill/speed"][:]),
            "respiration": handle["respiration/sniff_frequency_hz"][:],
        }
    lookup = {int(trial): index for index, trial in enumerate(aux["trial_id"])}
    pairs = [(index, lookup.get(int(trial)))
             for index, trial in enumerate(data["trial_id"])]
    pairs = [(a, b) for a, b in pairs if b is not None]
    pre_code = list(data["state_levels"]).index("pre")
    pairs = [(a, b) for a, b in pairs if data["state"][a] == pre_code]
    return (np.asarray([a for a, _ in pairs], int),
            np.asarray([b for _, b in pairs], int), aux)


def _aux_window_features(aux, indices, limits, *, minimum_pupil_coverage=.7):
    pupil, speed, respiration = [], [], []
    pupil_coverage = []
    for index in indices:
        selected = _window(aux["time_s"][index], limits)
        pupil_valid = (selected & aux["pupil_valid"][index] &
                       np.isfinite(aux["pupil"][index]))
        pupil_coverage.append(np.sum(pupil_valid) / max(1, np.sum(selected)))
        pupil.append(np.nanmedian(aux["pupil"][index, pupil_valid])
                     if np.any(pupil_valid) else np.nan)
        speed.append(np.nanmean(aux["speed"][index, selected]))
        respiratory_values = aux["respiration"][index, selected]
        respiration.append(np.nanmedian(respiratory_values)
                           if np.any(np.isfinite(respiratory_values)) else np.nan)
    pupil = np.asarray(pupil)
    pupil_coverage = np.asarray(pupil_coverage)
    pupil[pupil_coverage < minimum_pupil_coverage] = np.nan
    return {
        "pupil": pupil,
        "speed": np.asarray(speed),
        "respiration": np.asarray(respiration),
        "pupil_coverage": pupil_coverage,
    }


def baseline_covariation_tables(grouped_path, aux_path, row, population,
                                *, baseline=(-5., 0.),
                                minimum_pupil_coverage=.7):
    """Detrended F0 associations with awake pre-odor auxiliary measurements.

    Auxiliary variables are z-scored relative to all valid awake pre-odor
    trials in the session. F0 is linearly detrended separately for every unit,
    then z-scored across awake trials. Both unit-level correlations and a
    session-level population-median correlation are returned.
    """
    data = _population(grouped_path, population)
    imaging_index, aux_index, aux = _matched_awake(data, aux_path)
    if len(imaging_index) < 8:
        return pd.DataFrame(), pd.DataFrame()
    order = np.arange(len(imaging_index), dtype=float)
    f0 = data["baseline_mean"][:, imaging_index]
    f0_residual = _linear_detrend_rows(f0, order)
    f0_z = np.vstack([_zscore(row_values) for row_values in f0_residual])
    features = _aux_window_features(
        aux, aux_index, baseline,
        minimum_pupil_coverage=minimum_pupil_coverage)
    feature_z = {name: _zscore(features[name])
                 for name in ("pupil", "speed", "respiration")}
    common = _common(row, population)
    rows = []
    for unit, unit_id in enumerate(data["unit_id"]):
        rows.append(common | {
            "unit_id": unit_id, "n_awake_trials": len(imaging_index),
            "median_pupil_coverage": float(np.nanmedian(features["pupil_coverage"])),
            **{f"f0_vs_{name}_rho": _rank_correlation(f0_z[unit], values)
               for name, values in feature_z.items()},
        })
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        population_f0 = np.nanmedian(f0_z, axis=0)
    session = common | {
        "n_units": f0.shape[0], "n_awake_trials": len(imaging_index),
        "median_pupil_coverage": float(np.nanmedian(features["pupil_coverage"])),
        **{f"population_f0_vs_{name}_rho": _rank_correlation(population_f0, values)
           for name, values in feature_z.items()},
        "population_f0_vs_speed_given_respiration_rho": _partial_rank_correlation(
            population_f0, feature_z["speed"], [feature_z["respiration"]]),
        "population_f0_vs_respiration_given_speed_rho": _partial_rank_correlation(
            population_f0, feature_z["respiration"], [feature_z["speed"]]),
    }
    return pd.DataFrame(rows), pd.DataFrame([session])


def response_distribution_table(grouped_path, aux_path, row, population, *,
                                response=(-2., 8.), baseline=(-5., 0.),
                                tertile=.30, minimum_pupil_coverage=.7,
                                odor_center=True):
    """Time-resolved response quantiles on high/low arousal awake trials.

    Trial classes use the upper/lower ``tertile`` of odor-evoked pupil change
    or odor-window speed. Quantiles are first computed within a session across
    its unit x trial observations, which prevents a session with many ROIs from
    directly overwhelming later mouse-level summaries.
    """
    data = _population(grouped_path, population)
    imaging_index, aux_index, aux = _matched_awake(data, aux_path)
    if len(imaging_index) < 10:
        return pd.DataFrame()
    pre = _aux_window_features(aux, aux_index, baseline,
                               minimum_pupil_coverage=minimum_pupil_coverage)
    during = _aux_window_features(aux, aux_index, (0., 4.),
                                  minimum_pupil_coverage=minimum_pupil_coverage)
    # Residual pupil dilation avoids the mathematical coupling whereby trials
    # beginning with a small pupil have more room for a large raw difference.
    arousal = {
        "pupil_dilation": _residualize(during["pupil"], pre["pupil"]),
        "running": during["speed"],
    }
    time_s = np.asarray(data["time_s"], float)
    time_mask = _window(time_s, response)
    response_time = time_s[time_mask]
    traces = np.asarray(data["z"][:, imaging_index, :][:, :, time_mask], float)
    pre_response = _window(response_time, (-2., 0.))
    if np.any(pre_response):
        traces = traces - np.nanmean(
            traces[..., pre_response], axis=-1, keepdims=True)
    odors = data["odor_id"][imaging_index]
    if odor_center:
        traces = _odor_center(traces, odors)
    common = _common(row, population)
    # Interpolate the auxiliary driver onto the imaging-frame time axis. Pupil
    # is standardized to all valid awake baseline samples; speed remains in its
    # native units so zero retains a clear interpretation.
    pupil_baseline_samples = []
    for index in aux_index:
        selected = (_window(aux["time_s"][index], baseline) &
                    aux["pupil_valid"][index])
        pupil_baseline_samples.extend(aux["pupil"][index, selected])
    pupil_baseline_samples = np.asarray(pupil_baseline_samples, float)
    pupil_scale = np.nanstd(pupil_baseline_samples)
    driver_traces = {"pupil_dilation": [], "running": []}
    for index in aux_index:
        source_time = aux["time_s"][index]
        valid_pupil = aux["pupil_valid"][index] & np.isfinite(aux["pupil"][index])
        trial_pre = (_window(source_time, baseline) & valid_pupil)
        trial_center = (np.nanmedian(aux["pupil"][index, trial_pre])
                        if np.any(trial_pre) else np.nan)
        if np.sum(valid_pupil) >= 2 and pupil_scale > 0 and np.isfinite(trial_center):
            pupil_trace = np.interp(
                response_time, source_time[valid_pupil],
                (aux["pupil"][index, valid_pupil]-trial_center)/pupil_scale,
                left=np.nan, right=np.nan)
        else:
            pupil_trace = np.full(len(response_time), np.nan)
        valid_speed = np.isfinite(source_time) & np.isfinite(aux["speed"][index])
        speed_trace = (np.interp(response_time, source_time[valid_speed],
                                 aux["speed"][index, valid_speed],
                                 left=np.nan, right=np.nan)
                       if np.sum(valid_speed) >= 2 else
                       np.full(len(response_time), np.nan))
        driver_traces["pupil_dilation"].append(pupil_trace)
        driver_traces["running"].append(speed_trace)
    driver_traces = {name: np.asarray(values)
                     for name, values in driver_traces.items()}
    rows = []
    for arousal_name, arousal_values in arousal.items():
        valid = np.isfinite(arousal_values)
        if np.sum(valid) < 10:
            continue
        low_cut = np.nanquantile(arousal_values[valid], tertile)
        high_cut = np.nanquantile(arousal_values[valid], 1-tertile)
        classes = {
            "low": valid & (arousal_values <= low_cut),
            "high": valid & (arousal_values >= high_cut),
        }
        for class_name, selected in classes.items():
            if np.sum(selected) < 3:
                continue
            flattened = traces[:, selected, :].reshape(-1, traces.shape[-1])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                quantiles = np.nanquantile(
                    flattened, [.10, .25, .50, .75, .90], axis=0)
                driver = np.nanmedian(driver_traces[arousal_name][selected], axis=0)
            for frame, time in enumerate(response_time):
                rows.append(common | {
                    "arousal_measure": arousal_name,
                    "arousal_class": class_name,
                    "odor_centered": bool(odor_center),
                    "time_s": float(time),
                    "n_trials": int(np.sum(selected)),
                    "n_units": int(traces.shape[0]),
                    "q10_z": float(quantiles[0, frame]),
                    "q25_z": float(quantiles[1, frame]),
                    "median_z": float(quantiles[2, frame]),
                    "q75_z": float(quantiles[3, frame]),
                    "q90_z": float(quantiles[4, frame]),
                    "iqr_z": float(quantiles[3, frame]-quantiles[1, frame]),
                    "central80_width_z": float(quantiles[4, frame]-quantiles[0, frame]),
                    "driver_value": float(driver[frame]),
                    "driver_unit": ("baseline SD" if arousal_name == "pupil_dilation"
                                    else "speed"),
                })
    return pd.DataFrame(rows)
