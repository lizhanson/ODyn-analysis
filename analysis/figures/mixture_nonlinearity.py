"""Descriptive mixture geometry relative to empirically measured components.

The key distinction is between two questions that need not have the same
answer: (1) how far reciprocal mixtures are from one another, and (2) how far
each mixture lies from the response axis defined by its two component odors.
The latter is a simple descriptive nonlinearity measure and is intentionally
kept separate from crossvalidated discriminability.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .population_metrics import _common, _window


FAMILIES = {
    "alpha 17/18": {"components": (4, 10), "mixtures": (17, 18)},
    "epsilon 31/32": {"components": (22, 30), "mixtures": (31, 32)},
    "lambda 39/40": {"components": (3, 12), "mixtures": (39, 40)},
}


def _rms(vector):
    return float(np.sqrt(np.nanmean(np.asarray(vector, float)**2)))


def _cosine_distance(a, b):
    denominator = np.linalg.norm(a)*np.linalg.norm(b)
    return float(1-np.dot(a, b)/denominator) if denominator > 0 else np.nan


def _family_metrics(centroids, component_ids, mixture_ids):
    a, b = (centroids[value] for value in component_ids)
    m1, m2 = (centroids[value] for value in mixture_ids)
    axis = b-a
    denominator = float(np.dot(axis, axis))
    if not np.isfinite(denominator) or denominator <= 0:
        return None
    output = {
        "component_distance_rms_z": _rms(b-a),
        "mixture_distance_rms_z": _rms(m2-m1),
        "mixture_cosine_distance": _cosine_distance(m1, m2),
    }
    residuals, positions = [], []
    for mixture in (m1, m2):
        position = float(np.dot(mixture-a, axis)/denominator)
        clipped = float(np.clip(position, 0, 1))
        prediction = a + clipped*axis
        residuals.append(mixture-prediction)
        positions.append(position)
    output.update(
        mean_off_axis_residual_rms_z=float(np.mean([_rms(x) for x in residuals])),
        max_off_axis_residual_rms_z=float(np.max([_rms(x) for x in residuals])),
        mixture_1_component_axis_position=positions[0],
        mixture_2_component_axis_position=positions[1],
    )
    # The odor-code direction of the 60:40/40:60 assignment is deliberately
    # not assumed. Report the better of the two reciprocal assignments.
    predictions_a = (.6*a+.4*b, .4*a+.6*b)
    predictions_b = predictions_a[::-1]
    error_a = np.mean([_rms(m1-predictions_a[0]), _rms(m2-predictions_a[1])])
    error_b = np.mean([_rms(m1-predictions_b[0]), _rms(m2-predictions_b[1])])
    output["best_60_40_residual_rms_z"] = float(min(error_a, error_b))
    output["best_60_40_assignment"] = "forward" if error_a <= error_b else "reversed"
    return output


def session_mixture_nonlinearity(data, row, population, *, bins=((0., 4.),),
                                 minimum_trials=2):
    """Return family-level component-axis and reciprocal-mixture metrics."""
    common = _common(row, population)
    output = []
    for state_code, state_name in enumerate(data["state_levels"]):
        in_state = data["state"] == state_code
        for start, stop in bins:
            selected_time = _window(data["time_s"], (start, stop))
            if not np.any(selected_time):
                continue
            response = np.nanmean(data["z"][:, :, selected_time], axis=2)
            for family, specification in FAMILIES.items():
                odors = (*specification["components"], *specification["mixtures"])
                counts = {odor: int(np.sum(in_state & (data["odor_id"] == odor)))
                          for odor in odors}
                if min(counts.values()) < int(minimum_trials):
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    centroids = {
                        odor: np.nanmedian(
                            response[:, in_state & (data["odor_id"] == odor)], axis=1)
                        for odor in odors
                    }
                metrics = _family_metrics(
                    centroids, specification["components"], specification["mixtures"])
                if metrics is None:
                    continue
                output.append(common | {
                    "state": state_name, "family": family,
                    "component_a": specification["components"][0],
                    "component_b": specification["components"][1],
                    "mixture_a": specification["mixtures"][0],
                    "mixture_b": specification["mixtures"][1],
                    "window_start_s": float(start), "window_stop_s": float(stop),
                    "minimum_trials": min(counts.values()),
                    **{f"n_odor_{odor}": count for odor, count in counts.items()},
                    **metrics,
                })
    return pd.DataFrame(output)
