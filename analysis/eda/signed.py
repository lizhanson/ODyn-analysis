"""Trace-based signed excitation/suppression features.

Every unit-odor pair gets two independent components, so a response that is
excited and then suppressed is not averaged away.

Detection uses a box-smoothed median trace, which enforces a minimum duration
without a separate persistence rule and lets the pre-odor segment of that same
trace serve as an exactly matched null: same unit, same trials, same smoothing,
same window length.  Because trials per odor range from 1 to 11 within a single
session, the null is stratified by trial count -- otherwise a single cutoff
would impose a different false-positive rate on every odor, and breadth would
track how many times an odor happened to be presented.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BASELINE_WINDOW = (-5.0, -1.0)
ODOR_WINDOW = (0.0, 4.0)
SUBWINDOWS = {"onset": (0.0, 1.0), "sustained": (1.0, 4.0), "offset": (4.0, 8.0)}
RAW_WINDOWS = {"baseline": (-4.0, -1.0), "onset": (0.0, 1.0),
               "sustained": (1.0, 4.0), "offset": (4.0, 8.0),
               "late": (8.0, 14.0)}
SMOOTH_S = 0.5
MIN_NULL = 40


def window_mask(time_s, start, stop):
    return (np.asarray(time_s) >= start) & (np.asarray(time_s) < stop)


def box_smooth(traces, width):
    """Running mean along the last axis, ignoring NaN, no wraparound."""
    width = max(1, int(width))
    x = np.asarray(traces, float)
    if width == 1:
        return x
    kernel = np.ones(width)
    filled = np.nan_to_num(x, nan=0.0)
    valid = np.isfinite(x).astype(float)
    flat_total = np.apply_along_axis(lambda v: np.convolve(v, kernel, "same"),
                                     -1, filled)
    flat_count = np.apply_along_axis(lambda v: np.convolve(v, kernel, "same"),
                                     -1, valid)
    return np.divide(flat_total, flat_count,
                     out=np.full_like(flat_total, np.nan), where=flat_count > 0)


def median_traces(z, odor_ids, states, state_code, odor_levels):
    """unit x odor x frame median across that odor's trials within one state."""
    selected = np.asarray(states) == state_code
    stack, counts = [], []
    for odor in odor_levels:
        index = selected & (np.asarray(odor_ids) == odor)
        counts.append(int(index.sum()))
        stack.append(np.nanmedian(z[:, index, :], axis=1) if index.any()
                     else np.full((z.shape[0], z.shape[2]), np.nan))
    return np.stack(stack, axis=1), np.asarray(counts)


@dataclass(frozen=True)
class ExcursionThresholds:
    """Per-trial-count cutoffs, plus the pooled fallback."""

    positive: dict
    negative: dict
    pooled_positive: float
    pooled_negative: float
    tail_probability: float
    n_null: dict

    def for_count(self, count):
        count = int(count)
        return (self.positive.get(count, self.pooled_positive),
                self.negative.get(count, self.pooled_negative))


def excursion_thresholds(smoothed, time_s, counts, *, tail_probability=.05):
    """Cutoffs from maximal pre-odor excursions of the same median traces.

    The null is the distribution of extreme values over a matched-length
    window, so the cutoff controls the false-positive rate per unit-odor pair
    rather than per time point.  Strata with too few observations fall back to
    the pooled cutoff.
    """
    q = float(tail_probability)
    if not 0 < q < .5:
        raise ValueError("tail_probability must lie between 0 and 0.5")
    baseline = smoothed[..., window_mask(time_s, *BASELINE_WINDOW)]
    if baseline.shape[-1] < 3:
        raise ValueError("baseline window is too short for an excursion null")
    high = np.nanmax(baseline, axis=-1)          # unit x odor
    low = np.nanmin(baseline, axis=-1)

    finite_high = high[np.isfinite(high)]
    finite_low = low[np.isfinite(low)]
    if finite_high.size < MIN_NULL or finite_low.size < MIN_NULL:
        raise ValueError("not enough finite baseline excursions")
    pooled_positive = float(np.quantile(finite_high, 1-q))
    pooled_negative = float(np.quantile(finite_low, q))

    positive, negative, sizes = {}, {}, {}
    counts = np.asarray(counts)
    for count in np.unique(counts[counts > 0]):
        column = counts == count
        h = high[:, column].ravel()
        l = low[:, column].ravel()
        h, l = h[np.isfinite(h)], l[np.isfinite(l)]
        sizes[int(count)] = int(h.size)
        if h.size >= MIN_NULL and l.size >= MIN_NULL:
            positive[int(count)] = float(np.quantile(h, 1-q))
            negative[int(count)] = float(np.quantile(l, q))
    return ExcursionThresholds(positive, negative, pooled_positive,
                               pooled_negative, q, sizes)


def _threshold_arrays(thresholds, counts):
    """Broadcast per-odor cutoffs to a 1 x odor row for vectorised comparison."""
    pairs = [thresholds.for_count(count) for count in counts]
    positive = np.asarray([p for p, _ in pairs], float)[None, :]
    negative = np.asarray([n for _, n in pairs], float)[None, :]
    return positive, negative


def suprathreshold_area(traces, time_s, positive, negative, window):
    """Positive and negative suprathreshold area in z*s over one window."""
    mask = window_mask(time_s, *window)
    x = np.asarray(traces, float)[..., mask]
    dt = float(np.median(np.diff(np.asarray(time_s, float))))
    excitation = np.nansum(np.maximum(x - positive[..., None], 0.), axis=-1)*dt
    suppression = np.nansum(np.maximum(negative[..., None] - x, 0.), axis=-1)*dt
    return excitation, suppression


def signed_features(z, time_s, odor_ids, states, state_code, odor_levels, *,
                    tail_probability=.05, smooth_s=SMOOTH_S):
    """Every signed quantity for one session, population, and state."""
    traces, counts = median_traces(z, odor_ids, states, state_code, odor_levels)
    frame_s = float(np.median(np.diff(np.asarray(time_s, float))))
    smoothed = box_smooth(traces, round(smooth_s/frame_s))
    thresholds = excursion_thresholds(smoothed, time_s, counts,
                                      tail_probability=tail_probability)
    positive, negative = _threshold_arrays(thresholds, counts)

    odor_mask = window_mask(time_s, *ODOR_WINDOW)
    peak_high = np.nanmax(smoothed[..., odor_mask], axis=-1)
    peak_low = np.nanmin(smoothed[..., odor_mask], axis=-1)
    excitation, suppression = suprathreshold_area(
        traces, time_s, positive, negative, ODOR_WINDOW)
    output = {
        "trials_per_odor": counts,
        "thresholds": thresholds,
        "threshold_positive": np.broadcast_to(positive, peak_high.shape).copy(),
        "threshold_negative": np.broadcast_to(negative, peak_high.shape).copy(),
        "mean_z": np.nanmean(traces[..., odor_mask], axis=-1),
        "peak_positive_z": peak_high,
        "peak_negative_z": peak_low,
        "excitation_area": excitation,
        "suppression_area": suppression,
        "excited": peak_high > positive,
        "suppressed": peak_low < negative,
    }
    output["biphasic"] = output["excited"] & output["suppressed"]
    # Threshold-free window means.  Mineral oil is not a null stimulus in this
    # prep, so the raw delivery-locked time course has to stay visible next to
    # every thresholded call.
    for name, window in RAW_WINDOWS.items():
        output[f"raw_mean_{name}"] = np.nanmean(
            traces[..., window_mask(time_s, *window)], axis=-1)
    for name, window in SUBWINDOWS.items():
        e, s = suprathreshold_area(traces, time_s, positive, negative, window)
        sub = smoothed[..., window_mask(time_s, *window)]
        output[f"excitation_area_{name}"] = e
        output[f"suppression_area_{name}"] = s
        output[f"excited_{name}"] = np.nanmax(sub, axis=-1) > positive
        output[f"suppressed_{name}"] = np.nanmin(sub, axis=-1) < negative
    return output
