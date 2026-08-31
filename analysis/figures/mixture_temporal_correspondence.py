"""Temporal mixture separation and signed population recruitment at 10x.

The cross-population question is deliberately phrased at the odor-family x
time-bin level because Thy1 and DA populations were recorded in different
animals. It asks whether the emergence of a reproducible Thy1 reciprocal-mix
difference accompanies stronger DA excitation, suppression, or distribution
broadening; it is not a trialwise or causal coupling analysis.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .population_metrics import _common, _window


MIXTURE_PAIRS = ((17, 18), (31, 32), (39, 40))


def crossvalidated_pattern_energy(x, labels, *, repeats=100, seed=0):
    """Unwhitened split-half squared pattern distance per feature.

    Unlike crossnobis, this does not divide by a feature-wise noise estimate.
    It therefore retains response scale and avoids unstable variance weights at
    low repeat counts. Its signed square root has the units of the input z
    responses; negative estimates mean the reliable distance is unresolved.
    """
    x, labels = np.asarray(x, float), np.asarray(labels)
    levels = np.unique(labels)
    if len(levels) != 2:
        raise ValueError("exactly two odor labels are required")
    indices = [np.flatnonzero(labels == level) for level in levels]
    if min(map(len, indices)) < 2:
        return np.nan
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(int(repeats)):
        halves = []
        for index in indices:
            shuffled = rng.permutation(index)
            cut = len(shuffled)//2
            halves.append((shuffled[:cut], shuffled[cut:]))
        delta_a = (np.nanmean(x[halves[0][0]], axis=0)-
                   np.nanmean(x[halves[1][0]], axis=0))
        delta_b = (np.nanmean(x[halves[0][1]], axis=0)-
                   np.nanmean(x[halves[1][1]], axis=0))
        values.append(np.nanmean(delta_a*delta_b))
    return float(np.nanmean(values))


def session_temporal_mixture_table(data, row, population="units", *,
                                   bins=((0., 1.), (1., 2.), (2., 3.), (3., 4.)),
                                   minimum_trials=2, repeats=100):
    common = _common(row, population)
    rows = []
    for state_code, state_name in enumerate(data["state_levels"]):
        for pair_index, pair in enumerate(MIXTURE_PAIRS):
            selected_trials = ((data["state"] == state_code) &
                               np.isin(data["odor_id"], pair))
            labels = data["odor_id"][selected_trials]
            counts = [int(np.sum(labels == odor)) for odor in pair]
            if min(counts) < int(minimum_trials):
                continue
            for bin_index, (start, stop) in enumerate(bins):
                selected_time = _window(data["time_s"], (start, stop))
                response = np.nanmean(
                    data["z"][:, selected_trials, :][:, :, selected_time], axis=2).T
                energy = crossvalidated_pattern_energy(
                    response, labels, repeats=repeats,
                    seed=int(row["group_id"])*100+pair_index*10+bin_index)
                centroids = []
                for odor in pair:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        centroids.append(np.nanmedian(response[labels == odor], axis=0))
                population = np.concatenate(centroids)
                quantiles = np.nanquantile(population, [.10, .25, .50, .75, .90])
                biased_rms = float(np.sqrt(np.nanmean((centroids[0]-centroids[1])**2)))
                rows.append(common | {
                    "state": state_name, "pair": f"{pair[0]}-{pair[1]}",
                    "bin_start_s": float(start), "bin_stop_s": float(stop),
                    "n_a": counts[0], "n_b": counts[1],
                    "cv_pattern_energy_z2": energy,
                    "signed_cv_pattern_rms_z": (np.sign(energy)*np.sqrt(abs(energy))
                                                if np.isfinite(energy) else np.nan),
                    "centroid_difference_rms_z": biased_rms,
                    "q10_z": float(quantiles[0]), "q25_z": float(quantiles[1]),
                    "median_z": float(quantiles[2]), "q75_z": float(quantiles[3]),
                    "q90_z": float(quantiles[4]),
                    "central80_width_z": float(quantiles[4]-quantiles[0]),
                })
    return pd.DataFrame(rows)


def family_time_correspondence(table, *, state="pre"):
    """Descriptive correlations after equal-weight mouse aggregation."""
    from scipy.stats import spearmanr

    selected = table[table.state == state]
    keys = ["line", "mouse", "pair", "bin_start_s"]
    mouse = selected.groupby(keys, as_index=False).median(numeric_only=True)
    population = mouse.groupby(["line", "pair", "bin_start_s"], as_index=False).median(
        numeric_only=True)
    thy1 = population[population.line == "Thy1"][[
        "pair", "bin_start_s", "signed_cv_pattern_rms_z"]].rename(
        columns={"signed_cv_pattern_rms_z": "thy1_signed_cv_pattern_rms_z"})
    rows = []
    for line in ("TH", "DAT"):
        joined = thy1.merge(population[population.line == line],
                            on=["pair", "bin_start_s"], how="inner")
        for metric in ("q10_z", "q90_z", "central80_width_z",
                       "centroid_difference_rms_z"):
            valid = (np.isfinite(joined.thy1_signed_cv_pattern_rms_z) &
                     np.isfinite(joined[metric]))
            rho = (spearmanr(joined.loc[valid, "thy1_signed_cv_pattern_rms_z"],
                             joined.loc[valid, metric]).statistic
                   if np.sum(valid) >= 5 else np.nan)
            rows.append({"state": state, "da_line": line, "da_metric": metric,
                         "rho_across_family_time_cells": float(rho),
                         "n_family_time_cells": int(np.sum(valid))})
    return pd.DataFrame(rows), population
