"""
Group somas and processes by spatial proximity and temporal correlation.

Manual grouping asks the curator to decide, for every process, which soma it
belongs to. The only evidence on screen is a static image, so the judgement is
made almost entirely on proximity -- which is exactly the cue that is also the
confound. A process ROI abutting a soma shares signal with it through the point
spread function whether or not it is that cell's neurite, so "they are close and
they covary" is partly guaranteed before any biology enters.

Two things follow, and this module is built around both.

Grouping is agglomerative, strongest link first, under one hard constraint: a
group holds at most one soma. Nothing else is required. A chain of processes
whose parent soma is out of plane -- which at 20x is most of them -- is a
perfectly good group on its own, so processes are never made to find a soma.
But a process is never placed in a group containing two somas either, because
that is the one merge that cannot be true: the ROI would be claiming to belong
to two cells at once. Where a process bridges two somas it goes to whichever it
correlates with more strongly, and the losing link is recorded as refused
rather than silently dropped.

That constraint is also what stops single linkage running away. A threshold
graph over touching neuropil will otherwise chain half the field into one
component; here the chain stops at the second soma it reaches.

And the confound stays visible. `proximity_correlation_profile` reports mean
correlation against gap for every pair in range. If correlation falls off with
distance and then flattens well above zero, the short-range links carry
something beyond adjacency. If it decays to nothing over roughly the width of
the PSF, the grouping is measuring the microscope, and the thresholds should be
read as such.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

GROUPING_DEFAULTS = {
    # Gap is measured between nearest pixels, not centroids: a neurite's
    # centroid can sit far from the soma it plainly touches.
    "max_gap_um": 6.0,
    "min_correlation": 0.35,
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
    """
    Nearest-pixel distance between every pair of ROIs closer than the cutoff.

    Each ROI is measured inside its own bounding box grown by the cutoff, so
    the cost scales with total ROI area rather than with frame area times ROI
    count. A pair within the cutoff always has pixels inside that box, so
    nothing in range is missed.
    """
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
    matrix = np.asarray([np.asarray(traces[key], float).ravel() for key in keys])
    lengths = {row.size for row in matrix} if matrix.dtype == object else {matrix.shape[1]}
    if len(lengths) != 1:
        raise ValueError("Traces have differing lengths.")
    return matrix


def _correlations(matrix):
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(matrix)
    # A flat trace -- an ROI that never left its baseline -- has no correlation
    # with anything. Zero is the right answer; NaN would poison every argmax.
    return np.nan_to_num(np.atleast_2d(corr), nan=0.0)


def group_rois(soma_labels, process_labels, traces, *, um_per_px, params=None):
    """
    ROI groups from nearest-pixel gap and trace correlation, one soma each.

    `traces` maps ("soma"|"process", label) to that ROI's time course. Returns
    ``(groups, diagnostics)``: the mapping the state stores, and a row per
    candidate pair recording gap, correlation, and what became of the link.
    """
    import pandas as pd

    columns = ["roi_a", "roi_b", "gap_um", "correlation", "linked", "status"]
    p = {**GROUPING_DEFAULTS, **(params or {})}
    um_per_px = float(um_per_px)
    index, keys = roi_index_image(soma_labels, process_labels)
    if not keys:
        return {}, pd.DataFrame(columns=columns)

    matrix = _trace_matrix(traces, keys)
    corr = _correlations(matrix)
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
    """
    Mean correlation against nearest-pixel gap, which is the control.

    Run this before trusting a grouping. Correlation that decays to zero within
    a micron or two of the PSF width says the links are adjacency; correlation
    that stays raised out to several microns says there is something to group.
    """
    import pandas as pd

    p = {**GROUPING_DEFAULTS, **(params or {})}
    um_per_px = float(um_per_px)
    index, keys = roi_index_image(soma_labels, process_labels)
    matrix = _trace_matrix(traces, keys)
    corr = _correlations(matrix)
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


def traces_from_round(round_path, roi_manifest, *, center_each_trial=True):
    """
    Per-ROI time courses from a finalized round, keyed for `group_rois`.

    Trials are concatenated rather than averaged, and each is centred and
    scaled on its own baseline first, for the same reason the correlation image
    centres each block: without it, an ROI that sits high in one trial and low
    in another contributes a large between-trial deviation to every pair, and
    ROIs correlate because they share the trial structure rather than because
    they covary in time. Detrended traces are used where the round has them.
    """
    from ..session.h5io import open_h5
    from ..session.responders import load_roi_traces

    with open_h5(round_path) as handle:
        roi, source = load_roi_traces(handle)

    roi = np.asarray(roi, float)
    if center_each_trial:
        mean = np.nanmean(roi, axis=2, keepdims=True)
        sd = np.nanstd(roi, axis=2, keepdims=True)
        roi = (roi - mean) / np.where(sd > 0, sd, np.inf)

    flat = np.nan_to_num(roi.reshape(roi.shape[0], -1), nan=0.0)
    traces = {
        (row["roi_type"], int(row["source_roi_id"])): flat[int(row["roi_id"]) - 1]
        for row in roi_manifest
    }
    return traces, source
