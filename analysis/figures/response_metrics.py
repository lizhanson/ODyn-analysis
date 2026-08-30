"""Signed odor-response metrics for Figures 2 and 3.

Thresholds are estimated independently on the negative and positive tails of
mineral-oil responses within a session/state/population.  The analogue metrics
remain available so every thresholded conclusion can be checked continuously.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SignedThresholds:
    negative_z: float
    positive_z: float
    lower_quantile: float
    upper_quantile: float


def empirical_blank_thresholds(blank_values, *, tail_probability=.01):
    """Asymmetric empirical cutoffs from mineral-oil observations."""
    values = np.asarray(blank_values, float)
    values = values[np.isfinite(values)]
    if not 0 < float(tail_probability) < .5:
        raise ValueError("tail_probability must lie between 0 and 0.5")
    if values.size < 20:
        raise ValueError("at least 20 finite mineral-oil observations are required")
    q = float(tail_probability)
    return SignedThresholds(
        negative_z=float(np.quantile(values, q)),
        positive_z=float(np.quantile(values, 1-q)),
        lower_quantile=q, upper_quantile=1-q,
    )


def signed_response_components(values, thresholds: SignedThresholds):
    """Continuous positive/negative excess and independent responder calls."""
    values = np.asarray(values, float)
    excitation = np.maximum(values - thresholds.positive_z, 0.)
    suppression = np.maximum(thresholds.negative_z - values, 0.)
    return {
        "excitation_excess_z": excitation,
        "suppression_excess_z": suppression,
        "excited": values > thresholds.positive_z,
        "suppressed": values < thresholds.negative_z,
    }


def lifetime_sparseness(nonnegative, axis=-1):
    """Treves-Rolls sparseness; inputs are analogue nonnegative magnitudes."""
    x = np.asarray(nonnegative, float)
    n = x.shape[axis]
    if n < 2:
        raise ValueError("lifetime sparseness requires at least two odors")
    mean = np.nanmean(x, axis=axis)
    square = np.nanmean(x*x, axis=axis)
    ratio = np.divide(mean*mean, square, out=np.full_like(mean, np.nan),
                      where=square > 0)
    return (1-ratio)/(1-1/n)


def participation_ratio(nonnegative, axis=0, *, normalize=True):
    """Effective participating units from analogue magnitudes; no calls needed."""
    x = np.asarray(nonnegative, float)
    numerator = np.nansum(x, axis=axis)**2
    denominator = np.nansum(x*x, axis=axis)
    result = np.divide(numerator, denominator,
                       out=np.full_like(numerator, np.nan, dtype=float),
                       where=denominator > 0)
    if normalize:
        result = result / x.shape[axis]
    return result


def es_balance(excitation, suppression, axis=None):
    """Signed balance in [-1, 1], preserving both absolute components."""
    e = np.nansum(np.asarray(excitation, float), axis=axis)
    s = np.nansum(np.asarray(suppression, float), axis=axis)
    return np.divide(e-s, e+s, out=np.full_like(e+s, np.nan, dtype=float),
                     where=(e+s) > 0)


def aggregate_odor_responses(trial_values, odor_ids, states, *, reducer="median"):
    """Return unit x odor matrices separately by state using all available trials."""
    values = np.asarray(trial_values, float)
    odors = np.asarray(odor_ids)
    states = np.asarray(states)
    if values.ndim != 2 or values.shape[1] != len(odors) or odors.shape != states.shape:
        raise ValueError("trial response arrays do not align")
    function = {"median": np.nanmedian, "mean": np.nanmean}.get(reducer)
    if function is None:
        raise ValueError("reducer must be 'median' or 'mean'")
    output = {}
    for state in np.unique(states):
        selected = states == state
        levels = np.unique(odors[selected])
        matrix = np.column_stack([
            function(values[:, selected & (odors == odor)], axis=1)
            for odor in levels
        ])
        output[int(state)] = (levels, matrix)
    return output


def signed_population_metrics(odor_matrix, thresholds: SignedThresholds):
    """Unit lifetime and odor population summaries from unit x odor responses."""
    components = signed_response_components(odor_matrix, thresholds)
    e = components["excitation_excess_z"]
    s = components["suppression_excess_z"]
    excited = components["excited"]
    suppressed = components["suppressed"]
    # These zero-rectified versions are genuinely threshold-free analogue
    # measurements.  The excess versions below answer the complementary
    # question after applying the empirical blank cutoffs.
    analogue_e = np.maximum(np.asarray(odor_matrix, float), 0.)
    analogue_s = np.maximum(-np.asarray(odor_matrix, float), 0.)
    return {
        "unit_excitation_breadth": np.mean(excited, axis=1),
        "unit_suppression_breadth": np.mean(suppressed, axis=1),
        "unit_excitation_sparseness": lifetime_sparseness(e, axis=1),
        "unit_suppression_sparseness": lifetime_sparseness(s, axis=1),
        "odor_excited_fraction": np.mean(excited, axis=0),
        "odor_suppressed_fraction": np.mean(suppressed, axis=0),
        "odor_excitation_participation": participation_ratio(analogue_e, axis=0),
        "odor_suppression_participation": participation_ratio(analogue_s, axis=0),
        "odor_excitation_participation_thresholded": participation_ratio(e, axis=0),
        "odor_suppression_participation_thresholded": participation_ratio(s, axis=0),
        "odor_es_balance": es_balance(analogue_e, analogue_s, axis=0),
        "odor_es_balance_thresholded": es_balance(e, s, axis=0),
    }
