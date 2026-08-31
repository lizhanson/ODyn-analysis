"""Sensitivity of signed odor responses to tonic fluorescence and SNR.

The grouped products contain the same trace in raw fluorescence and in the
per-trial z-score used by the main analysis.  This module measures negative
AUC from median odor waveforms in three units (z, raw delta-F, and delta-F/F0)
and repeats the state comparison after removing the lowest post-anesthesia F0
or SNR quartile within each session.  The filtering is deliberately a
sensitivity analysis, not an exclusion rule for the main figures.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .population_metrics import _common, _decode, _source_path, _window


def load_raw_population(grouped_path, population):
    """Load raw and z-scored grouped traces for one population."""
    import h5py

    grouped_path = Path(grouped_path)
    with h5py.File(grouped_path) as handle:
        root = handle[population]
        source = _source_path(grouped_path, handle)
        data = {
            "unit_id": _decode(root["unit_id"][:]),
            "raw": root["raw"][:],
            "z": root["z"][:],
            "baseline_mean": root["baseline_mean"][:],
            "normalization_sd": root["normalization_sd"][:],
            "odor_id": handle["odor_id"][:],
            "state": handle["state"][:],
            "state_levels": _decode(handle["state_levels"][:]),
        }
    with h5py.File(source) as handle:
        data["time_s"] = handle["traces/time_s"][:]
    return data


def _negative_auc(waveform, time_s, odor_mask):
    odor = waveform[..., odor_mask]
    return np.trapezoid(np.maximum(-odor, 0), time_s[odor_mask], axis=-1)


def f0_sensitivity_table(data, row, population, *, odor_window=(0., 4.),
                         blank_odor=0, reducer="median"):
    """Return one row per unit, odor, and state with matched response metrics."""
    function = {"median": np.nanmedian, "mean": np.nanmean}[reducer]
    time_s = np.asarray(data["time_s"], float)
    odor_mask = _window(time_s, odor_window)
    common = _common(row, population)
    rows = []
    for state_code, state_name in enumerate(data["state_levels"]):
        in_state = data["state"] == state_code
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            unit_f0 = np.nanmedian(data["baseline_mean"][:, in_state], axis=1)
            unit_noise = np.nanmedian(data["normalization_sd"][:, in_state], axis=1)
        unit_snr = np.divide(unit_f0, unit_noise,
                             out=np.full_like(unit_f0, np.nan, dtype=float),
                             where=unit_noise > 0)
        f0_cut = np.nanquantile(unit_f0, .25)
        snr_cut = np.nanquantile(unit_snr, .25)
        for odor in np.unique(data["odor_id"][in_state]):
            if int(odor) == int(blank_odor):
                continue
            selected = in_state & (data["odor_id"] == odor)
            baseline = data["baseline_mean"][:, selected, None]
            delta = data["raw"][:, selected, :] - baseline
            dff = np.divide(delta, baseline,
                            out=np.full_like(delta, np.nan, dtype=float),
                            where=baseline > 0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                z_wave = function(data["z"][:, selected, :], axis=1)
                df_wave = function(delta, axis=1)
                dff_wave = function(dff, axis=1)
            measurements = {
                "negative_auc_z_s": _negative_auc(z_wave, time_s, odor_mask),
                "negative_auc_df_s": _negative_auc(df_wave, time_s, odor_mask),
                "negative_auc_dff_s": _negative_auc(dff_wave, time_s, odor_mask),
            }
            for index, unit_id in enumerate(data["unit_id"]):
                rows.append(common | {
                    "unit_id": unit_id, "state": state_name,
                    "odor_id": int(odor), "n_trials": int(np.sum(selected)),
                    "f0": float(unit_f0[index]), "snr": float(unit_snr[index]),
                    "retain_f0_q25": bool(unit_f0[index] >= f0_cut),
                    "retain_snr_q25": bool(unit_snr[index] >= snr_cut),
                    **{name: float(value[index])
                       for name, value in measurements.items()},
                })
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    # Filters must be defined by post-anesthesia quality and then applied to
    # both states for the same unit, preserving a paired comparison.
    keys = ["group_id", "compartment", "unit_id"]
    post = table[table.state == "post"][keys + ["retain_f0_q25", "retain_snr_q25"]]
    post = post.groupby(keys, as_index=False).first().rename(columns={
        "retain_f0_q25": "adequate_post_f0",
        "retain_snr_q25": "adequate_post_snr",
    })
    return table.drop(columns=["retain_f0_q25", "retain_snr_q25"]).merge(
        post, on=keys, how="left", validate="many_to_one")


def session_sensitivity_summary(table):
    """Nested-analysis input: one state summary per session and filter."""
    metrics = ("negative_auc_z_s", "negative_auc_df_s", "negative_auc_dff_s")
    keys = ["group_id", "mouse", "line", "depth_class", "cohort",
            "compartment", "state"]
    rows = []
    filters = {
        "all units": np.ones(len(table), dtype=bool),
        "exclude lowest post-F0 quartile": table.adequate_post_f0.fillna(False),
        "exclude lowest post-SNR quartile": table.adequate_post_snr.fillna(False),
    }
    for label, keep in filters.items():
        for key, group in table[keep].groupby(keys, dropna=False):
            row = dict(zip(keys, key)) | {"sensitivity_set": label,
                                          "n_units": group.unit_id.nunique()}
            for metric in metrics:
                # q75 emphasizes the suppression-rich tail while remaining
                # substantially more stable than an extreme maximum.
                row[metric + "_median"] = float(group[metric].median())
                row[metric + "_q75"] = float(group[metric].quantile(.75))
            rows.append(row)
    return pd.DataFrame(rows)


def f0_change_associations(table):
    """Session-level association between F0 loss and suppression change."""
    from scipy.stats import spearmanr

    unit_keys = ["group_id", "mouse", "line", "depth_class", "cohort",
                 "compartment", "unit_id", "state"]
    metrics = ("negative_auc_z_s", "negative_auc_df_s", "negative_auc_dff_s")
    unit = table.groupby(unit_keys, dropna=False).agg(
        f0=("f0", "first"), snr=("snr", "first"),
        **{metric: (metric, "median") for metric in metrics}).reset_index()
    wide = unit.pivot(index=unit_keys[:-1], columns="state",
                      values=["f0", "snr", *metrics])
    rows = []
    session_keys = ["group_id", "mouse", "line", "depth_class", "cohort",
                    "compartment"]
    for key, group in wide.groupby(level=session_keys, dropna=False):
        f0_ratio = np.log2(group[("f0", "post")] / group[("f0", "pre")])
        for metric in metrics:
            change = group[(metric, "post")] - group[(metric, "pre")]
            valid = np.isfinite(f0_ratio) & np.isfinite(change)
            rho = spearmanr(f0_ratio[valid], change[valid]).statistic \
                if np.sum(valid) >= 8 else np.nan
            rows.append(dict(zip(session_keys, key)) | {
                "metric": metric, "rho_f0_change_vs_suppression_change": rho,
                "n_units": int(np.sum(valid)),
            })
    return pd.DataFrame(rows)
