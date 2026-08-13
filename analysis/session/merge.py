"""
Merge ROI masks segmented independently per odor (and optionally per state).

A combined activity image dilutes anything odor-specific: a glomerulus driven
by 1 of 16 odors contributes to one map and is averaged against fifteen where
it is silent. Segmenting each odor's map on its own lets that glomerulus be
found at full contrast, and the same glomerulus detected under several odors
then has to be recognised as one unit rather than several.

That recognition is by spatial overlap, resolved with complete-linkage
agglomerative clustering. Thresholding overlap and taking connected
components -- the obvious approach -- is single linkage, whose classic
pathology is chaining: if A-B and B-C clear threshold but A-C do not, all
three merge anyway, and on a dense glomerular field that walks across a row
of neighbours and returns one enormous ROI. Complete linkage requires every
pair in a group to clear threshold, so a chain cannot form by construction,
with no size cap needed. It also guarantees two ROIs from the same source map
never merge, since disjoint labels sit at maximum distance.

Two overlap metrics, because they fail differently:

    iou      intersection / union. Symmetric, punishes size mismatch. Use
             when the maps should agree closely.
    overlap  intersection / smaller area. Forgiving when one detection is a
             fragment of another, which happens when an odor drives only part
             of a glomerulus. Merges more readily; likelier to over-merge.
"""

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
    """
    Overlap between every ROI of `mask_a` and every ROI of `mask_b`.

    Returns an (n_a, n_b) matrix indexed from label 1. Computed with one
    joint histogram rather than per-pair intersection, so cost does not grow
    with the number of ROIs squared.
    """

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
    """
    Group ROIs from an all-pairs overlap matrix. Returns a cluster id per ROI.

    'single' links a cluster through any one qualifying pair, which is
    connected-components and chains: A-B and B-C merge A with C even when
    A and C do not overlap at all. On a dense field that walks across a row
    of neighbours.

    'complete' requires every pair in a cluster to clear the threshold, so a
    chain cannot form -- the guarantee comes from the linkage rule rather than
    from a size cap bolted on afterwards. Two ROIs from the same source map
    never merge under it either, since disjoint labels sit at distance 1.
    """

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
    """
    Merge per-odor label images into one consensus label image.

    `consensus_fraction` decides the merged footprint: 0.0 takes the union of
    every member's pixels, 1.0 the intersection, 0.5 the pixels that at least
    half the members agree on. Union grows ROIs with each extra detection;
    intersection shrinks them toward nothing. Half is a reasonable default
    only once several odors have detected the ROI.

    `min_detections` drops ROIs found by fewer than N maps, which is the knob
    for trading odor-specific sensitivity against false positives.

    `linkage` is 'complete' (every pair in a group must overlap; no chaining)
    or 'single' (connected components; chains). Complete is the default and
    should stay that way on a dense field.

    `min_area_px` drops merged ROIs smaller than this. It is needed because a
    merged footprint can end up below the size floor its own segmentation
    enforced: pixels already claimed by an earlier ROI are removed so the
    output stays disjoint, which can erode a later ROI to a sliver. Pass the
    segmentation's `min_area_px` to keep the two consistent.
    """

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

        # Keep one connected region, not a scatter of pixels that sum to one.
        #
        # Two steps here produce speckle. A `consensus_fraction` above 0 votes
        # pixel by pixel, and around the rim -- where the odors disagree about
        # the boundary -- the vote alternates, leaving a halo of isolated
        # pixels. Then the competition above removes whatever a neighbouring
        # ROI already claimed, which can cut the remainder into islands.
        #
        # A size floor on the total does not catch this: 700 scattered pixels
        # pass a 707-px floor as readily as one 700-px disc, and the resulting
        # "ROI" averages fluorescence from wherever its specks landed. Measured
        # on exp 132: 70 of 132 merged ROIs were non-contiguous, one of them in
        # 132 separate pieces. Taking the largest component makes the floor
        # mean what it says, and needs no extra parameter.
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
