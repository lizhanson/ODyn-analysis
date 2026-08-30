"""Within-odor arousal covariates for Figure 3.

The primary analysis asks whether trial-to-trial neural deviations track
trial-to-trial pupil/running deviations after removing odor identity.  It does
not treat pupil as a proxy for a specific neuromodulator.
"""

from __future__ import annotations

import numpy as np


def within_group_center(values, *groups):
    """Subtract the median within each combination of categorical groups."""
    x = np.asarray(values, float)
    keys = np.column_stack([np.asarray(group) for group in groups])
    if len(x) != len(keys):
        raise ValueError("values and grouping vectors do not align")
    result = np.full_like(x, np.nan)
    for key in np.unique(keys, axis=0):
        selected = np.all(keys == key, axis=1)
        result[selected] = x[selected] - np.nanmedian(x[selected])
    return result


def window_mean(values, time_s, start, stop):
    values, time_s = np.asarray(values, float), np.asarray(time_s, float)
    selected = (time_s >= start) & (time_s < stop)
    if values.shape[-1] != len(time_s) or not np.any(selected):
        raise ValueError("time window does not align with values")
    return np.nanmean(values[..., selected], axis=-1)


def trial_arousal_features(pupil, speed, time_s, *, odor_window=(0., 4.)):
    """Continuous odor-window pupil and running summaries per trial."""
    pupil = np.asarray(pupil, float)
    speed = np.asarray(speed, float)
    baseline = window_mean(pupil, time_s, -5., 0.)
    odor_pupil = window_mean(pupil, time_s, *odor_window)
    return {
        "pupil_delta": odor_pupil - baseline,
        "running_speed": window_mean(np.abs(speed), time_s, *odor_window),
    }
