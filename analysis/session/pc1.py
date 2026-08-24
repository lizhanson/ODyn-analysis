"""One odor-protected PC1 scalar per trial, retained but never subtracted."""

from __future__ import annotations

import numpy as np


def trial_pc1(response_z, odor_ids):
    """PC1 of unit-by-trial responses after removing each unit/odor mean."""
    values = np.asarray(response_z, np.float64)
    odors = np.asarray(odor_ids)
    if values.ndim != 2 or values.shape[1] != len(odors):
        raise ValueError("response_z must be unit x trial and align with odor_ids")
    residual = values.copy()
    for odor in np.unique(odors):
        selected = odors == odor
        residual[:, selected] -= np.nanmean(residual[:, selected], axis=1, keepdims=True)
    matrix = np.nan_to_num(residual - np.nanmean(residual, axis=1, keepdims=True))
    u, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    score = vt[0] * singular[0]
    population = np.nanmean(matrix, axis=0)
    if np.std(score) and np.std(population) and np.corrcoef(score, population)[0, 1] < 0:
        score = -score
        u[:, 0] = -u[:, 0]
    total = float(np.sum(singular ** 2))
    return {
        "trial_score": score.astype(np.float32),
        "loadings": u[:, 0].astype(np.float32),
        "explained_variance_fraction": float(singular[0] ** 2 / total) if total else np.nan,
        "method": "PC1 of odor-mean-protected unit-by-trial odor responses",
    }
