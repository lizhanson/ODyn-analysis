"""A fixed spatial PC1 projected into a within-trial imaging time series."""

from __future__ import annotations

import numpy as np


def detrended_dff(traces, odor_on_frames, frame_rate, *, baseline_s=4.0):
    """Trialwise dF/F from already-detrended fluorescence."""
    traces = np.asarray(traces, dtype=np.float32)
    out = np.full_like(traces, np.nan)
    n_base = max(1, int(round(float(baseline_s) * float(frame_rate))))
    for trial, onset in enumerate(np.asarray(odor_on_frames, int)):
        baseline = traces[:, trial, max(0, onset - n_base):onset]
        f0 = np.nanmean(baseline, axis=1)
        out[:, trial] = ((traces[:, trial] - f0[:, None]) /
                         np.maximum(np.abs(f0[:, None]), 1e-6))
    return out


def fixed_pc1(dff, *, max_iterations=30, tolerance=1e-7, progress=None):
    """Fit one loading vector globally and project it within every trial."""
    dff = np.asarray(dff, dtype=np.float32)
    if dff.ndim != 3 or dff.shape[0] < 1:
        raise ValueError("dff must be analysis-unit x trial x frame")
    n_unit, n_trial, n_frame = dff.shape
    matrix = dff.transpose(1, 2, 0).reshape(-1, n_unit).copy()
    means = np.nanmean(matrix, axis=0)
    matrix -= means
    matrix[~np.isfinite(matrix)] = 0.0

    loading = np.full(n_unit, 1.0 / np.sqrt(n_unit), dtype=np.float64)
    converged = False
    iteration = 0
    for iteration in range(1, int(max_iterations) + 1):
        score = matrix @ loading
        updated = matrix.T @ score
        norm = np.linalg.norm(updated)
        if norm <= 1e-12:
            loading = np.zeros(n_unit)
            break
        updated /= norm
        change = min(np.linalg.norm(updated - loading), np.linalg.norm(updated + loading))
        loading = updated
        if progress is not None and (iteration == 1 or iteration % 5 == 0):
            progress(f"PC1 power iteration {iteration}/{max_iterations}")
        if change < tolerance:
            converged = True
            break

    score = matrix @ loading
    population = np.nanmean(matrix, axis=1)
    if np.std(score) and np.std(population) and np.corrcoef(score, population)[0, 1] < 0:
        loading = -loading
        score = -score
    total = float(np.sum(matrix.astype(np.float64) ** 2))
    explained = float(np.sum(score ** 2) / total) if total > 0 else np.nan
    return {
        "timecourse": score.reshape(n_trial, n_frame).astype(np.float32),
        "loadings": loading.astype(np.float32),
        "unit_mean_dff": means.astype(np.float32),
        "explained_variance_fraction": explained,
        "iterations": iteration,
        "converged": converged,
        "method": "fixed spatial PC1 by power iteration over concatenated detrended dF/F",
    }


def pc1_from_detrended(traces, odor_on_frames, frame_rate, *, baseline_s=4.0,
                       progress=None):
    return fixed_pc1(detrended_dff(
        traces, odor_on_frames, frame_rate, baseline_s=baseline_s,
    ), progress=progress)
