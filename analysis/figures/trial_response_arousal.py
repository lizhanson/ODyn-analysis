"""Trial-level response-distribution and auxiliary-state summaries.

Every row represents one imaging trial and one population/compartment. Neural
width is the q90-q10 spread across ROIs after each ROI is averaged over the
four-second odor period. Pupil metrics are standardized using the distribution
of valid awake pre-odor pupil samples from that session.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .population_metrics import _common, _window
from .state_arousal import _matched_awake


ALPHA_ODORS = (4, 10, 17, 18)


def trial_response_arousal_table(data, aux_path, row, population, *,
                                 baseline=(-5., 0.), odor_window=(0., 4.),
                                 minimum_pupil_coverage=.7):
    imaging_index, aux_index, aux = _matched_awake(data, aux_path)
    if len(imaging_index) == 0:
        return pd.DataFrame()
    odor_mask = _window(data["time_s"], odor_window)
    trial_response = np.nanmean(
        data["z"][:, imaging_index, :][:, :, odor_mask], axis=2)

    baseline_samples = []
    for index in aux_index:
        selected = (_window(aux["time_s"][index], baseline) &
                    aux["pupil_valid"][index] &
                    np.isfinite(aux["pupil"][index]))
        baseline_samples.extend(aux["pupil"][index, selected])
    baseline_samples = np.asarray(baseline_samples, float)
    pupil_center = np.nanmedian(baseline_samples)
    pupil_scale = np.nanstd(baseline_samples)
    common = _common(row, population)
    rows = []
    for local, (imaging_trial, aux_trial) in enumerate(zip(imaging_index, aux_index)):
        time = aux["time_s"][aux_trial]
        base = _window(time, baseline)
        odor = _window(time, odor_window)
        valid_base = base & aux["pupil_valid"][aux_trial] & np.isfinite(aux["pupil"][aux_trial])
        valid_odor = odor & aux["pupil_valid"][aux_trial] & np.isfinite(aux["pupil"][aux_trial])
        coverage = np.sum(valid_odor)/max(1, np.sum(odor))
        pupil_base = (np.nanmean(aux["pupil"][aux_trial, valid_base])
                      if np.any(valid_base) else np.nan)
        pupil_odor = (np.nanmean(aux["pupil"][aux_trial, valid_odor])
                      if np.any(valid_odor) and coverage >= minimum_pupil_coverage else np.nan)
        response = trial_response[:, local]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            quantiles = np.nanquantile(response, [.10, .25, .50, .75, .90])
        odor_id = int(data["odor_id"][imaging_trial])
        alpha_class = ("components 4/10" if odor_id in (4, 10) else
                       "mixtures 17/18" if odor_id in (17, 18) else "other")
        rows.append(common | {
            "trial_id": int(data["trial_id"][imaging_trial]),
            "trial_order_awake": int(local), "odor_id": odor_id,
            "alpha_class": alpha_class, "n_units": int(len(response)),
            "q10_z": float(quantiles[0]), "q25_z": float(quantiles[1]),
            "median_z": float(quantiles[2]), "q75_z": float(quantiles[3]),
            "q90_z": float(quantiles[4]),
            "response_width_z": float(quantiles[4]-quantiles[0]),
            "pupil_odor_z": ((pupil_odor-pupil_center)/pupil_scale
                              if np.isfinite(pupil_odor) and pupil_scale > 0 else np.nan),
            "pupil_change_z": ((pupil_odor-pupil_base)/pupil_scale
                                if np.isfinite(pupil_odor) and np.isfinite(pupil_base)
                                and pupil_scale > 0 else np.nan),
            "pupil_odor_coverage": float(coverage),
            "running_speed": float(np.nanmean(aux["speed"][aux_trial, odor])),
            "respiration_hz": float(np.nanmedian(aux["respiration"][aux_trial, odor]))
            if np.any(np.isfinite(aux["respiration"][aux_trial, odor])) else np.nan,
        })
    return pd.DataFrame(rows)


def within_odor_family_correlations(table):
    """Session correlations after removing each odor's median trial response."""
    from scipy.stats import spearmanr

    rows = []
    keys = ["group_id", "mouse", "line", "depth_class", "cohort", "compartment"]
    for key, session in table.groupby(keys, dropna=False):
        session = session.copy()
        for column in ("q10_z", "q90_z", "response_width_z", "pupil_odor_z",
                       "pupil_change_z", "running_speed"):
            session[column+"_residual"] = session[column]-session.groupby("odor_id")[column].transform("median")
        for family in ("components 4/10", "mixtures 17/18", "other"):
            subset = session[session.alpha_class == family]
            for arousal in ("pupil_odor_z", "pupil_change_z", "running_speed"):
                for neural in ("q10_z", "q90_z", "response_width_z"):
                    a = subset[arousal+"_residual"]
                    b = subset[neural+"_residual"]
                    valid = np.isfinite(a) & np.isfinite(b)
                    rho = (spearmanr(a[valid], b[valid]).statistic
                           if np.sum(valid) >= 8 else np.nan)
                    rows.append(dict(zip(keys, key)) | {
                        "odor_family": family, "arousal_metric": arousal,
                        "neural_metric": neural, "rho": float(rho),
                        "n_trials": int(np.sum(valid)),
                    })
    return pd.DataFrame(rows)
