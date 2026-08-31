"""Small-sample crossvalidated odor geometry used by Figures 2 and 3."""

from __future__ import annotations

import numpy as np


def cosine_distance_matrix(centroids):
    x = np.asarray(centroids, float)
    norms = np.linalg.norm(x, axis=1)
    safe = np.where(norms > 0, norms, np.nan)
    similarity = (x @ x.T) / (safe[:, None] * safe[None, :])
    return 1 - similarity


def diagonal_crossnobis(trials, labels, *, repeats=200, seed=0):
    """Repeated split-half, noise-standardized distances.

    `trials` is trial x feature.  A feature is normally one ROI's integrated
    odor response, but may also be an ROI-time bin for spatiotemporal geometry.
    The diagonal noise model is stable when features greatly outnumber trials;
    a shrinkage full-covariance sensitivity analysis can follow later.
    """
    x = np.asarray(trials, float)
    labels = np.asarray(labels)
    if x.ndim != 2 or len(labels) != len(x):
        raise ValueError("trials and labels do not align")
    levels = np.unique(labels)
    indices = {level: np.flatnonzero(labels == level) for level in levels}
    if any(len(value) < 2 for value in indices.values()):
        raise ValueError("every odor needs at least two trials")

    # Pooled within-odor residual variance: features with unstable trial
    # responses receive less weight.  These features are ROIs (or ROI-time bins),
    # not trials.
    residual = np.concatenate([
        x[index] - np.nanmean(x[index], axis=0, keepdims=True)
        for index in indices.values()
    ])
    variance = np.nanvar(residual, axis=0, ddof=1)
    floor = np.nanmedian(variance[np.isfinite(variance) & (variance > 0)]) * 1e-3
    variance = np.where(np.isfinite(variance) & (variance > floor), variance, floor)

    rng = np.random.default_rng(seed)
    accumulated = np.zeros((len(levels), len(levels)), float)
    used = np.zeros_like(accumulated, int)
    for _ in range(int(repeats)):
        halves = {}
        for level, index in indices.items():
            shuffled = rng.permutation(index)
            cut = len(shuffled)//2
            halves[level] = (shuffled[:cut], shuffled[cut:])
        for i in range(len(levels)):
            for j in range(i+1, len(levels)):
                ia, ib = halves[levels[i]]
                ja, jb = halves[levels[j]]
                da = np.nanmean(x[ia], axis=0) - np.nanmean(x[ja], axis=0)
                db = np.nanmean(x[ib], axis=0) - np.nanmean(x[jb], axis=0)
                value = np.nanmean(da*db/variance)
                if np.isfinite(value):
                    accumulated[i, j] += value
                    accumulated[j, i] += value
                    used[i, j] += 1
                    used[j, i] += 1
    distance = np.divide(accumulated, used, out=np.full_like(accumulated, np.nan),
                         where=used > 0)
    np.fill_diagonal(distance, 0.)
    return levels, distance


def time_resolved_crossnobis(traces, labels, *, bin_frames=1, repeats=200, seed=0):
    """Crossnobis RDM in successive time bins from trial x feature x time data."""
    x = np.asarray(traces, float)
    if x.ndim != 3:
        raise ValueError("traces must be trial x feature x time")
    if int(bin_frames) < 1:
        raise ValueError("bin_frames must be positive")
    rdms = []
    levels = None
    for start in range(0, x.shape[2], int(bin_frames)):
        binned = np.nanmean(x[:, :, start:start + int(bin_frames)], axis=2)
        levels, rdm = diagonal_crossnobis(
            binned, labels, repeats=repeats, seed=seed + start)
        rdms.append(rdm)
    return levels, np.stack(rdms), np.arange(0, x.shape[2], int(bin_frames))


def spatiotemporal_crossnobis(traces, labels, *, repeats=200, seed=0):
    """Preserve temporal pattern by treating feature-time bins as features."""
    x = np.asarray(traces, float)
    if x.ndim != 3:
        raise ValueError("traces must be trial x feature x time")
    return diagonal_crossnobis(
        x.reshape(x.shape[0], -1), labels, repeats=repeats, seed=seed)


def pair_distance(trials, labels, *, seed=0, repeats=100):
    """Crossnobis distance between exactly two conditions."""
    levels, matrix = diagonal_crossnobis(trials, labels, repeats=repeats,
                                         seed=seed)
    if len(levels) != 2:
        raise ValueError("pair_distance needs exactly two conditions")
    return float(matrix[0, 1])


def cosine_centroid_distance(trials, labels):
    """Gain-insensitive distance between two condition centroids."""
    levels = np.unique(labels)
    x = np.asarray(trials, float)
    a = np.nanmean(x[labels == levels[0]], axis=0)
    b = np.nanmean(x[labels == levels[1]], axis=0)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(1 - np.dot(a, b)/denominator) if denominator > 0 else np.nan


def permutation_null(trials, labels, observed, *, permutations=200, seed=0,
                     repeats=40):
    """Empirical one-sided p value, reported with the resolution it allows.

    A permutation null that has run out of resolution is flagged underpowered
    rather than summarised by a z score, which would invent precision the null
    does not contain.
    """
    rng = np.random.default_rng(seed)
    null = np.asarray([
        pair_distance(trials, rng.permutation(labels), seed=seed+index+1,
                      repeats=repeats)
        for index in range(int(permutations))
    ], float)
    null = null[np.isfinite(null)]
    if null.size == 0 or not np.isfinite(observed):
        return {"p_value": np.nan, "p_resolution": np.nan,
                "null_median": np.nan, "underpowered": True}
    resolution = 1.0/(null.size + 1)
    p_value = float((np.sum(null >= observed) + 1)/(null.size + 1))
    return {"p_value": p_value, "p_resolution": float(resolution),
            "null_median": float(np.median(null)),
            "underpowered": bool(p_value <= resolution + 1e-12)}
