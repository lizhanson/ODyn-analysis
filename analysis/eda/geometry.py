"""Crossvalidated population geometry for the exploratory pass.

Crossnobis is crossvalidated, so its expectation is zero when two conditions
are not reliably separated and it is not inflated by comparing noisy centroids.
Cosine distance is carried alongside as the gain-insensitive secondary measure.

Nulls are reported as empirical permutation p values together with the
resolution the permutation count allows.  A permutation null that has run out
of resolution is labelled underpowered rather than summarised by a Gaussian
z score, which would invent precision the null does not contain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def pooled_variance(x, labels):
    """Within-condition residual variance per feature, with a positive floor."""
    x = np.asarray(x, float)
    residual = np.concatenate([
        x[labels == level] - np.nanmean(x[labels == level], axis=0, keepdims=True)
        for level in np.unique(labels)
    ])
    variance = np.nanvar(residual, axis=0, ddof=1)
    finite = variance[np.isfinite(variance) & (variance > 0)]
    floor = np.nanmedian(finite)*1e-3 if finite.size else 1e-12
    return np.where(np.isfinite(variance) & (variance > floor), variance, floor)


def crossnobis_pair(x, labels, *, repeats=200, seed=0):
    """Repeated split-half noise-normalised distance between two conditions.

    The two independent halves make the estimate unbiased: uncorrelated noise
    contributes zero on average rather than a positive offset.
    """
    x = np.asarray(x, float)
    labels = np.asarray(labels)
    levels = np.unique(labels)
    if levels.size != 2:
        raise ValueError("crossnobis_pair needs exactly two conditions")
    index = {level: np.flatnonzero(labels == level) for level in levels}
    if any(len(value) < 2 for value in index.values()):
        return np.nan
    variance = pooled_variance(x, labels)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(int(repeats)):
        halves = {}
        for level, rows in index.items():
            shuffled = rng.permutation(rows)
            cut = max(1, len(shuffled)//2)
            halves[level] = (shuffled[:cut], shuffled[cut:])
        (ia, ib), (ja, jb) = halves[levels[0]], halves[levels[1]]
        if min(len(ib), len(jb)) == 0:
            continue
        da = np.nanmean(x[ia], axis=0) - np.nanmean(x[ja], axis=0)
        db = np.nanmean(x[ib], axis=0) - np.nanmean(x[jb], axis=0)
        value = np.nanmean(da*db/variance)
        if np.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else np.nan


def cosine_pair(x, labels):
    """Gain-insensitive distance between the two condition centroids."""
    levels = np.unique(labels)
    a = np.nanmean(np.asarray(x, float)[labels == levels[0]], axis=0)
    b = np.nanmean(np.asarray(x, float)[labels == levels[1]], axis=0)
    denominator = np.linalg.norm(a)*np.linalg.norm(b)
    return float(1 - np.dot(a, b)/denominator) if denominator > 0 else np.nan


@dataclass(frozen=True)
class PermutationResult:
    observed: float
    p_value: float
    resolution: float
    null_median: float
    null_upper95: float
    n_permutation: int
    underpowered: bool


def permutation_test(x, labels, *, repeats=60, permutations=200, seed=0):
    """Empirical one-sided p value for a positive crossnobis distance.

    The smallest attainable p value is 1/(permutations+1); when the observed
    value sits at that floor the result is flagged underpowered instead of
    being converted into a z score.
    """
    observed = crossnobis_pair(x, labels, repeats=repeats, seed=seed)
    if not np.isfinite(observed):
        return PermutationResult(np.nan, np.nan, np.nan, np.nan, np.nan,
                                 0, True)
    rng = np.random.default_rng(seed + 1)
    null = np.asarray([
        crossnobis_pair(x, rng.permutation(labels), repeats=max(10, repeats//4),
                        seed=seed + index + 2)
        for index in range(int(permutations))
    ], float)
    null = null[np.isfinite(null)]
    if null.size == 0:
        return PermutationResult(observed, np.nan, np.nan, np.nan, np.nan,
                                 0, True)
    resolution = 1.0/(null.size + 1)
    p_value = float((np.sum(null >= observed) + 1)/(null.size + 1))
    return PermutationResult(
        observed=float(observed), p_value=p_value, resolution=float(resolution),
        null_median=float(np.median(null)),
        null_upper95=float(np.quantile(null, .95)),
        n_permutation=int(null.size),
        underpowered=bool(p_value <= resolution + 1e-12),
    )


def rdm(x, labels, *, repeats=200, seed=0):
    """Full crossnobis representational dissimilarity matrix."""
    levels = np.unique(labels)
    matrix = np.full((levels.size, levels.size), np.nan)
    np.fill_diagonal(matrix, 0.)
    for i in range(levels.size):
        for j in range(i+1, levels.size):
            mask = np.isin(labels, (levels[i], levels[j]))
            value = crossnobis_pair(x[mask], labels[mask], repeats=repeats,
                                    seed=seed + i*levels.size + j)
            matrix[i, j] = matrix[j, i] = value
    return levels, matrix
