"""Merge ROI masks segmented independently per odor (and optionally per state)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MergedMasks:
    """Result of merging per-odor masks."""

    labels: np.ndarray
    n_rois: int
    # For each merged ROI (1-based), the (source_index, source_label) pairs
    # that produced it -- i.e. which odors detected it.
    provenance: dict[int, list[tuple[int, int]]] = field(default_factory=dict)

    def detections_per_roi(self) -> np.ndarray:
        """How many source maps found each ROI. 1 means a single odor only."""
        return np.array(
            [len(self.provenance[i]) for i in range(1, self.n_rois + 1)], dtype=int
        )


def pairwise_overlap(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    *,
    metric: str = "iou",
) -> np.ndarray:
    """Overlap between every ROI of `mask_a` and every ROI of `mask_b`."""

    if mask_a.shape != mask_b.shape:
        raise ValueError(f"Shape mismatch: {mask_a.shape} vs {mask_b.shape}.")

    n_a = int(mask_a.max())
    n_b = int(mask_b.max())

    if n_a == 0 or n_b == 0:
        return np.zeros((n_a, n_b), dtype=np.float32)

    # Joint label histogram: index = a * (n_b + 1) + b, background included.
    joint = np.bincount(
        (mask_a.ravel().astype(np.int64) * (n_b + 1) + mask_b.ravel()),
        minlength=(n_a + 1) * (n_b + 1),
    ).reshape(n_a + 1, n_b + 1)

    intersection = joint[1:, 1:].astype(np.float64)

    area_a = np.bincount(mask_a.ravel(), minlength=n_a + 1)[1:].astype(np.float64)
    area_b = np.bincount(mask_b.ravel(), minlength=n_b + 1)[1:].astype(np.float64)

    if metric == "iou":
        denom = area_a[:, None] + area_b[None, :] - intersection
    elif metric == "overlap":
        denom = np.minimum(area_a[:, None], area_b[None, :])
    else:
        raise ValueError(f"metric must be 'iou' or 'overlap', got {metric!r}.")

    return np.where(denom > 0, intersection / np.maximum(denom, 1e-12), 0).astype(
        np.float32
    )


def _largest_component(footprint: np.ndarray) -> np.ndarray:
    """The biggest 8-connected island in a boolean mask; ties go to the first."""

    from scipy.ndimage import label as connected_components

    pieces, n_pieces = connected_components(
        footprint, structure=np.ones((3, 3), dtype=int)
    )

    if n_pieces <= 1:
        return footprint

    sizes = np.bincount(pieces.ravel())[1:]

    return pieces == (int(sizes.argmax()) + 1)


def _cluster(overlap: np.ndarray, *, min_overlap: float, linkage: str) -> np.ndarray:
    """Group ROIs from an all-pairs overlap matrix."""

    n = overlap.shape[0]

    if n == 1:
        return np.zeros(1, dtype=int)

    from scipy.cluster.hierarchy import fcluster
    from scipy.cluster.hierarchy import linkage as scipy_linkage
    from scipy.spatial.distance import squareform

    distance = 1.0 - overlap
    np.fill_diagonal(distance, 0.0)

    # Enforce exact symmetry; float error upsets squareform.
    distance = np.minimum(distance, distance.T)

    condensed = squareform(distance, checks=False)
    tree = scipy_linkage(condensed, method=linkage)

    return fcluster(tree, t=1.0 - min_overlap, criterion="distance")


def merge_masks(
    masks: list[np.ndarray],
    *,
    min_overlap: float = 0.5,
    metric: str = "iou",
    consensus_fraction: float = 0.0,
    min_detections: int = 1,
    linkage: str = "complete",
    min_area_px: None | float = None,
) -> MergedMasks:
    """Merge per-odor label images into one consensus label image."""

    if not masks:
        raise ValueError("No masks given.")

    shape = masks[0].shape
    for i, mask in enumerate(masks):
        if mask.shape != shape:
            raise ValueError(f"Mask {i} has shape {mask.shape}, expected {shape}.")

    # Flat index for every (source, label) ROI in the input.
    index: dict[tuple[int, int], int] = {}
    for source, mask in enumerate(masks):
        for label in range(1, int(mask.max()) + 1):
            index[(source, label)] = len(index)

    if not index:
        return MergedMasks(labels=np.zeros(shape, dtype=np.int32), n_rois=0)

    # All-pairs overlap across every source, so the clustering sees the full
    # structure rather than only the pairs that happened to clear threshold.
    n_total = len(index)
    overlap_all = np.zeros((n_total, n_total), dtype=np.float32)

    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            block = pairwise_overlap(masks[i], masks[j], metric=metric)

            for a in range(block.shape[0]):
                for b in range(block.shape[1]):
                    if block[a, b] <= 0:
                        continue
                    fa, fb = index[(i, a + 1)], index[(j, b + 1)]
                    overlap_all[fa, fb] = block[a, b]
                    overlap_all[fb, fa] = block[a, b]

    np.fill_diagonal(overlap_all, 1.0)

    cluster_ids = _cluster(overlap_all, min_overlap=min_overlap, linkage=linkage)

    flat_to_key = {flat: key for key, flat in index.items()}
    groups: dict[int, list[tuple[int, int]]] = {}
    for flat, cid in enumerate(cluster_ids):
        groups.setdefault(int(cid), []).append(flat_to_key[flat])

    labels = np.zeros(shape, dtype=np.int32)
    provenance: dict[int, list[tuple[int, int]]] = {}
    next_label = 1

    for members in sorted(groups.values(), key=len, reverse=True):
        if len(members) < min_detections:
            continue

        votes = np.zeros(shape, dtype=np.int16)
        for source, label in members:
            votes += masks[source] == label

        needed = max(1, int(np.ceil(consensus_fraction * len(members))))
        footprint = votes >= needed

        if not footprint.any():
            continue

        # Only claim pixels not already taken, so ROIs stay disjoint and the
        # earlier (larger-consensus) group wins a contested pixel.
        footprint &= labels == 0
        if not footprint.any():
            continue

        footprint = _largest_component(footprint)

        # Re-check size AFTER competition and cleanup, not before: an ROI
        # eroded by a neighbour is no longer the object segmentation accepted.
        if min_area_px is not None and footprint.sum() < min_area_px:
            continue

        labels[footprint] = next_label
        provenance[next_label] = sorted(members)
        next_label += 1

    return MergedMasks(
        labels=labels,
        n_rois=next_label - 1,
        provenance=provenance,
    )
