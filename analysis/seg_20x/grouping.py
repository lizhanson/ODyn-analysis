"""Group somas and processes by spatial proximity and temporal correlation."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

GROUPING_DEFAULTS = {
    # Gap is measured between nearest pixels, not centroids: a neurite's
    # centroid can sit far from the soma it plainly touches.
    "max_gap_um": 6.0,
    "min_correlation": 0.35,
    "max_lag_frames": 1,
    # A group of one is an ROI nothing joined. Reporting those as groups makes
    # every isolated soma and stray process a group, and the count meaningless.
    "drop_singletons": True,
    "profile_max_gap_um": 20.0,
    "profile_bin_um": 1.0,
}


def roi_index_image(soma_labels, process_labels):
    """One index image over both mask types, -1 where nothing is claimed."""
    soma_labels = np.asarray(soma_labels)
    process_labels = np.asarray(process_labels)
    if soma_labels.shape != process_labels.shape:
        raise ValueError("soma and process label images have differing shapes")

    index = np.full(soma_labels.shape, -1, np.int32)
    keys = []
    for kind, labels in (("soma", soma_labels), ("process", process_labels)):
        for ident in np.unique(labels[labels > 0]):
            index[labels == ident] = len(keys)
            keys.append((kind, int(ident)))
    return index, keys


def pairwise_gaps(index, n_roi, max_gap_px):
    """Nearest-pixel distance between every pair of ROIs closer than the cutoff."""
    from scipy.ndimage import distance_transform_edt, find_objects

    gaps: dict[tuple[int, int], float] = {}
    height, width = index.shape
    pad = int(np.ceil(max_gap_px)) + 1

    for i, window in enumerate(find_objects(index + 1, max_label=n_roi)):
        if window is None:
            continue
        rows = slice(max(0, window[0].start - pad), min(height, window[0].stop + pad))
        cols = slice(max(0, window[1].start - pad), min(width, window[1].stop + pad))
        patch = index[rows, cols]
        distance = distance_transform_edt(patch != i)
        near = (patch >= 0) & (patch != i) & (distance <= max_gap_px)
        for j in np.unique(patch[near]):
            j = int(j)
            gap = float(distance[near & (patch == j)].min())
            key = (i, j) if i < j else (j, i)
            if gap < gaps.get(key, np.inf):
                gaps[key] = gap
    return gaps


def _trace_matrix(traces, keys):
    missing = [key for key in keys if key not in traces]
    if missing:
        raise KeyError(
            f"No trace for {len(missing)} ROI(s), e.g. {missing[:3]}. "
            "Every curated ROI needs one; extract before grouping."
        )
    arrays = [np.asarray(traces[key], float) for key in keys]
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("Traces have differing shapes.")
    return np.stack(arrays)


def _cross_correlation(left, right):
    left = np.asarray(left, float).reshape(left.shape[0], -1)
    right = np.asarray(right, float).reshape(right.shape[0], -1)
    left = np.nan_to_num(left - np.nanmean(left, axis=1, keepdims=True), nan=0.0)
    right = np.nan_to_num(right - np.nanmean(right, axis=1, keepdims=True), nan=0.0)
    scale = np.linalg.norm(left, axis=1)[:, None] * np.linalg.norm(right, axis=1)[None, :]
    return np.divide(left @ right.T, scale, out=np.zeros_like(scale), where=scale > 0)


def _correlations(matrix, max_lag_frames=0):
    """Maximum Pearson correlation over the requested positive and negative lags."""
    matrix = np.asarray(matrix, float)
    if matrix.ndim == 2:
        return _cross_correlation(matrix, matrix)
    if matrix.ndim != 3:
        raise ValueError(f"Expected ROI x trial x frame traces, got {matrix.shape}.")

    best = _cross_correlation(matrix, matrix)
    for lag in range(1, int(max_lag_frames) + 1):
        if lag >= matrix.shape[2]:
            break
        shifted = _cross_correlation(matrix[:, :, lag:], matrix[:, :, :-lag])
        best = np.maximum(best, shifted)
        best = np.maximum(best, shifted.T)
    return np.clip(best, -1.0, 1.0)


def group_rois(soma_labels, process_labels, traces, *, um_per_px, params=None):
    """ROI groups from nearest-pixel gap and trace correlation, one soma each."""
    import pandas as pd

    columns = ["roi_a", "roi_b", "gap_um", "correlation", "linked", "status"]
    p = {**GROUPING_DEFAULTS, **(params or {})}
    um_per_px = float(um_per_px)
    index, keys = roi_index_image(soma_labels, process_labels)
    if not keys:
        return {}, pd.DataFrame(columns=columns)

    matrix = _trace_matrix(traces, keys)
    corr = _correlations(matrix, p["max_lag_frames"])
    gaps = pairwise_gaps(index, len(keys), float(p["max_gap_um"]) / um_per_px)

    parent = list(range(len(keys)))
    somas = [1 if kind == "soma" else 0 for kind, _ in keys]

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # Strongest link first, gap breaking ties, so the outcome does not depend
    # on the order the ROIs happened to be labelled in.
    candidates = sorted(gaps.items(), key=lambda item: (-corr[item[0]], item[1]))
    minimum = float(p["min_correlation"])
    status = {}

    for (i, j), _gap in candidates:
        if corr[i, j] < minimum:
            status[(i, j)] = "below_correlation"
            continue
        a, b = root(i), root(j)
        if a == b:
            status[(i, j)] = "already_grouped"
            continue
        if somas[a] and somas[b]:
            # The only merge that cannot be true: one ROI in two cells.
            status[(i, j)] = "two_somas"
            continue
        parent[b] = a
        somas[a] += somas[b]
        status[(i, j)] = "linked"

    members = defaultdict(list)
    for i in range(len(keys)):
        members[root(i)].append(i)
    if p["drop_singletons"]:
        members = {r: m for r, m in members.items() if len(m) > 1}

    # Number by first member so ids are stable against relabelling upstream.
    groups = {
        keys[i]: gid
        for gid, (_, component) in enumerate(
            sorted(members.items(), key=lambda item: min(item[1])), start=1
        )
        for i in component
    }

    diagnostics = pd.DataFrame(
        [
            {
                "roi_a": f"{keys[i][0][0]}{keys[i][1]}",
                "roi_b": f"{keys[j][0][0]}{keys[j][1]}",
                "gap_um": round(gap * um_per_px, 3),
                "correlation": round(float(corr[i, j]), 4),
                "linked": status[(i, j)] == "linked",
                "status": status[(i, j)],
            }
            for (i, j), gap in sorted(gaps.items())
        ],
        columns=columns,
    ).sort_values(["linked", "correlation"], ascending=False, ignore_index=True)

    return groups, diagnostics


def proximity_correlation_profile(
    soma_labels, process_labels, traces, *, um_per_px, params=None
):
    """Mean correlation against nearest-pixel gap, which is the control."""
    import pandas as pd

    p = {**GROUPING_DEFAULTS, **(params or {})}
    um_per_px = float(um_per_px)
    index, keys = roi_index_image(soma_labels, process_labels)
    matrix = _trace_matrix(traces, keys)
    corr = _correlations(matrix, p["max_lag_frames"])
    gaps = pairwise_gaps(index, len(keys), float(p["profile_max_gap_um"]) / um_per_px)

    rows = []
    for (i, j), gap in gaps.items():
        kinds = tuple(sorted((keys[i][0], keys[j][0])))
        rows.append({
            "gap_um": gap * um_per_px,
            "correlation": float(corr[i, j]),
            "pair_type": "-".join(kinds),
        })
    pairs = pd.DataFrame(rows, columns=["gap_um", "correlation", "pair_type"])
    if pairs.empty:
        return pairs.assign(n=[])

    edges = np.arange(0, float(p["profile_max_gap_um"]) + float(p["profile_bin_um"]),
                      float(p["profile_bin_um"]))
    pairs["gap_bin_um"] = edges[np.clip(np.digitize(pairs.gap_um, edges) - 1, 0, len(edges) - 1)]
    return (
        pairs.groupby(["pair_type", "gap_bin_um"])
        .agg(correlation=("correlation", "mean"), n=("correlation", "size"))
        .reset_index()
    )


def traces_from_round(
    round_path,
    roi_manifest,
    *,
    odor_s=4.0,
    post_s=4.0,
    smooth_sigma_frames=2.0,
):
    """Smoothed canonical z traces from odor onset through the post window."""
    from ..session.h5io import open_h5
    from scipy.ndimage import gaussian_filter1d

    with open_h5(round_path) as handle:
        if "traces/roi_z" not in handle:
            raise ValueError(
                f"{round_path} predates canonical trace z-scoring; re-extract it."
            )
        roi, source = handle["traces/roi_z"][:], "canonical_z"
        on_frames = handle["trials/odor_on_frame"][:].astype(int)
        frame_rate = float(handle.attrs["frame_rate"])

    roi = np.asarray(roi, float)
    n_window = max(2, int(round((float(odor_s) + float(post_s)) * frame_rate)))
    windows = np.full((roi.shape[0], roi.shape[1], n_window), np.nan, float)

    for trial, onset in enumerate(on_frames):
        stop = onset + n_window
        if onset < 1 or stop > roi.shape[2]:
            raise ValueError(
                f"Trial {trial} cannot provide {odor_s:g} s odor + {post_s:g} s post "
                f"from frame {onset}; trace length is {roi.shape[2]}."
            )
        windows[:, trial] = roi[:, trial, onset:stop]

    if smooth_sigma_frames > 0:
        windows = gaussian_filter1d(
            windows, sigma=float(smooth_sigma_frames), axis=2, mode="nearest"
        )

    traces = {
        (row["roi_type"], int(row["source_roi_id"])): windows[int(row["roi_id"]) - 1]
        for row in roi_manifest
    }
    return traces, source
