"""Unit, odor and session summaries built on the signed feature table.

Mineral oil is not a null stimulus here: it drives most of the population, and
it drives it harder under anesthesia.  So every responsivity measure taken
against the pre-odor baseline is partly a measure of the delivery response,
and is reported next to two references that a shared delivery response cannot
inflate: the odor-minus-blank contrast, and across-odor modulation, which is
invariant to any constant added to every odor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

UNIT_KEYS = ["group_id", "mouse", "line", "cohort", "depth_class",
             "population", "state", "unit_id"]
SESSION_KEYS = ["group_id", "mouse", "line", "cohort", "depth_class",
                "population", "state"]


def treves_rolls(values):
    """Lifetime sparseness of nonnegative magnitudes; NaN if all are zero."""
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return np.nan
    square = np.mean(x*x)
    if not square > 0:
        return np.nan
    return float((1 - np.mean(x)**2/square)/(1 - 1/n))


def participation_ratio(values, *, normalize=True):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    denominator = np.sum(x*x)
    if not denominator > 0 or x.size == 0:
        return np.nan
    result = float(np.sum(x)**2/denominator)
    return result/x.size if normalize else result


def add_blank_contrast(features, *, column="raw_mean_sustained"):
    """Attach each unit's own blank response, within session and state."""
    blank = features[features.is_blank].set_index(UNIT_KEYS)
    joined = features.join(blank[column].rename("blank_response"), on=UNIT_KEYS)
    joined = joined.join(blank["excited"].rename("blank_excited"), on=UNIT_KEYS)
    joined["odor_minus_blank"] = joined[column] - joined["blank_response"]
    return joined


def unit_summary(features):
    """One row per unit, state, population and session."""
    features = add_blank_contrast(features)
    real = features[~features.is_blank]

    def reduce(frame):
        excitation = frame.excitation_area.to_numpy(float)
        suppression = frame.suppression_area.to_numpy(float)
        contrast = frame.odor_minus_blank.to_numpy(float)
        total = np.nansum(excitation) + np.nansum(suppression)
        return pd.Series({
            "n_odor": len(frame),
            # Responsivity against the pre-odor baseline.  Includes whatever
            # the delivery event itself evokes.
            "excitation_breadth": float(np.nanmean(frame.excited)),
            "suppression_breadth": float(np.nanmean(frame.suppressed)),
            "biphasic_breadth": float(np.nanmean(frame.biphasic)),
            "excitation_mass": float(np.nanmean(excitation)),
            "suppression_mass": float(np.nanmean(suppression)),
            "es_balance": float((np.nansum(excitation)-np.nansum(suppression))
                                / total) if total > 0 else np.nan,
            "excitation_sparseness": treves_rolls(excitation),
            "suppression_sparseness": treves_rolls(suppression),
            # Blank-independent: a constant added to every odor cancels.
            "across_odor_sd": float(np.nanstd(frame.raw_mean_sustained, ddof=1))
            if len(frame) > 1 else np.nan,
            "across_odor_range": float(np.nanmax(frame.raw_mean_sustained)
                                       - np.nanmin(frame.raw_mean_sustained)),
            "contrast_sparseness": treves_rolls(np.maximum(contrast, 0.)),
            "mean_odor_minus_blank": float(np.nanmean(contrast)),
            "max_odor_minus_blank": float(np.nanmax(contrast)),
            "blank_response": float(frame.blank_response.iloc[0]),
            "blank_excited": bool(frame.blank_excited.iloc[0]),
        })

    return real.groupby(UNIT_KEYS, observed=True).apply(
        reduce, include_groups=False).reset_index()


def odor_summary(features):
    """Population recruitment per odor, including the blank."""
    def reduce(frame):
        excitation = frame.excitation_area.to_numpy(float)
        suppression = frame.suppression_area.to_numpy(float)
        total = np.nansum(excitation) + np.nansum(suppression)
        return pd.Series({
            "n_unit": len(frame),
            "trials_per_odor": float(frame.trials_per_odor.iloc[0]),
            "excited_fraction": float(np.nanmean(frame.excited)),
            "suppressed_fraction": float(np.nanmean(frame.suppressed)),
            "biphasic_fraction": float(np.nanmean(frame.biphasic)),
            "excitation_participation": participation_ratio(excitation),
            "suppression_participation": participation_ratio(suppression),
            "es_balance": float((np.nansum(excitation)-np.nansum(suppression))
                                / total) if total > 0 else np.nan,
            "median_raw_sustained": float(np.nanmedian(frame.raw_mean_sustained)),
        })

    keys = SESSION_KEYS + ["odor_id", "odor_group", "is_blank"]
    return features.groupby(keys, observed=True).apply(
        reduce, include_groups=False).reset_index()


def session_summary(units, metrics):
    """Median across units: the session is the sampling unit, not the cell."""
    return units.groupby(SESSION_KEYS, observed=True)[metrics].median().reset_index()


def cohort_summary(sessions, metrics):
    """Sessions averaged within mouse, then described across mice."""
    mouse = (sessions.groupby(["cohort", "line", "depth_class", "population",
                               "state", "mouse"], observed=True)[metrics]
             .mean().reset_index())
    keys = ["cohort", "line", "depth_class", "population", "state"]
    output = mouse.groupby(keys, observed=True)[metrics].agg(
        ["median", "min", "max", "count"])
    return mouse, output
