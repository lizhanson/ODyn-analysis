"""
Segmentation state for the per-odor curation GUI, with no GUI in it.

Everything the interface does to the data lives here, so it can be tested
without opening a window: parameters per odor, the two-phase workflow, and
the manual edits.

The workflow is two phases on purpose.

    tune     per-odor sliders re-run segmentation live; no manual edits yet
    merge    per-odor masks are combined into one consensus mask; the merge
             parameters are live here and the segmentation ones are frozen
    curate   everything is frozen; ROIs are added, deleted, or masked out on
             the merged mask

Curation happens **after** merging, on the consensus, not per odor. Curating
each odor separately would mean repeating the same judgements up to sixteen
times and could produce per-odor masks that disagree, leaving the merge to
reconcile decisions a human already made. One pass on the merged result is
both less work and unambiguous.

The old GUI mixed tuning and curation, and its delete handler zeroed an ROI
without recording the deletion anywhere -- so the next slider move re-ran
segmentation and silently brought it back. Curation was lost without warning.
Sequential phases remove the failure mode rather than patching it: while
parameters can change there is nothing to lose, and once there is something to
lose the parameters cannot change.

Going back a phase is allowed but discards curation, and says so.
"""

from __future__ import annotations

import json

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..session.merge import merge_masks
from .watershed import GLOM_10X_DEFAULTS, segment_image

PHASE_TUNE = "tune"
PHASE_MERGE = "merge"
PHASE_CURATE = "curate"

PHASES = (PHASE_TUNE, PHASE_MERGE, PHASE_CURATE)

# Defaults for combining per-odor masks. `min_overlap` is the one worth
# sweeping: with few odors the merge is nearly a union, but across sixteen the
# detections-per-ROI curve is what tells you where to put it.
MERGE_DEFAULTS = {
    "min_overlap": 0.5,
    "metric": "iou",
    "linkage": "complete",
    "min_detections": 1,
    "consensus_fraction": 0.0,
}

MERGE_SLIDER_RANGES = {
    "min_overlap": (0.05, 0.95, 0.05),
    "min_detections": (1, 16, 1),
    "consensus_fraction": (0.0, 1.0, 0.1),
}

# Slider ranges. min/max diameter deliberately reach past GLOM_10X_DEFAULTS in
# both directions -- the old GUI's min_area slider stopped at 200 px when the
# settled minimum was 314, so the interface could not express its own defaults.
SLIDER_RANGES = {
    "threshold_pctl": (0.0, 99.5, 0.5),
    "adaptive_block_px": (0, 401, 10),
    "min_diameter_px": (2.0, 80.0, 1.0),
    "max_diameter_px": (10.0, 200.0, 1.0),
    "peak_distance_px": (2.0, 60.0, 1.0),
    "border_px": (0, 100, 1),
}

# How far a hand-placed seed is let off the segmentation's parameters. These
# scale `min_diameter_px`, `max_diameter_px` and `threshold_pctl` for manual
# additions only; the automatic pass is untouched. See `manual_bounds` for why
# the two should not share bounds at all.
#
# `peak_distance_px` has no manual counterpart and is not applied: it is the
# minimum separation between automatic watershed seeds, and a click may land
# as close to an existing ROI as the person clicking wants.
MANUAL_MIN_SCALE = 0.5
MANUAL_MAX_SCALE = 1.5
MANUAL_THRESHOLD_DROP = 25.0


@dataclass
class Curation:
    """Manual edits to the merged mask. Only meaningful in the curate phase."""

    deleted: set[int] = field(default_factory=set)
    # Watershed seed points, not fixed disks: a hand-placed ROI should follow
    # the image the way an automatic one does, so its footprint is comparable.
    added_seeds: list[tuple[int, int]] = field(default_factory=list)
    # Indices into `added_seeds` that were subsequently deleted. Seeds are
    # tombstoned rather than removed so the surviving ones keep their indices,
    # which is what `provenance` refers to.
    deleted_seeds: set[int] = field(default_factory=set)
    exclude_polygons: list[list[tuple[int, int]]] = field(default_factory=list)

    def live_seeds(self) -> list[tuple[int, int, int]]:
        """(index, y, x) for seeds that have not been deleted."""
        return [
            (i, y, x)
            for i, (y, x) in enumerate(self.added_seeds)
            if i not in self.deleted_seeds
        ]

    def is_empty(self) -> bool:
        return not (
            self.deleted
            or self.live_seeds()
            or self.exclude_polygons
        )


def border_mask(shape: tuple[int, int], border_px: int) -> np.ndarray:
    """
    True within `border_px` of any edge.

    Rigid motion correction with `border_nan="copy"` fills shifted edges by
    duplicating pixel content (see `odor_zscore_movies._crop_border`). Those
    duplicated columns correlate perfectly with their neighbours, so they
    produce high-correlation ROIs along the frame edge that are an artefact of
    registration rather than signal.
    """

    mask = np.zeros(shape, dtype=bool)

    if border_px <= 0:
        return mask

    b = int(border_px)
    mask[:b, :] = True
    mask[-b:, :] = True
    mask[:, :b] = True
    mask[:, -b:] = True

    return mask


def grow_seed(
    image: np.ndarray,
    seed: tuple[int, int],
    *,
    unclaimed: np.ndarray,
    min_diameter_px: float,
    max_diameter_px: float,
    threshold_pctl: float = 60.0,
) -> np.ndarray:
    """
    Grow one hand-placed seed into an ROI by watershed, as `segment_image` does.

    A fixed disk would give a hand-added ROI a footprint unlike every automatic
    one -- same area regardless of what is under it, ignoring the boundary the
    image plainly shows. Flooding the inverted image from the seed instead
    makes it follow the same landscape, so its trace means the same thing.

    The flood is confined to a window around the seed, to pixels no other ROI
    has claimed, and to pixels above a local threshold; a rim of background
    seeds around the window stops it leaking outward.

    The size bounds passed here should be the *manual* ones, not the
    segmentation's -- see `manual_bounds`. A click marks a glomerulus the
    automatic pass did not find, which usually means it is dimmer or smaller
    than the parameters were tuned for, so judging it by those parameters
    rejects precisely the ROIs the tool exists to add.
    """

    from skimage.segmentation import watershed

    h, w = image.shape
    y, x = int(seed[0]), int(seed[1])

    if not (0 <= y < h and 0 <= x < w):
        return np.zeros((h, w), dtype=bool)

    # Window generous enough to contain the largest allowed ROI.
    half = int(max_diameter_px)
    r0, r1 = max(y - half, 0), min(y + half + 1, h)
    c0, c1 = max(x - half, 0), min(x + half + 1, w)

    window = image[r0:r1, c0:c1]
    free = unclaimed[r0:r1, c0:c1]

    finite = window[np.isfinite(window)]
    if finite.size == 0:
        return np.zeros((h, w), dtype=bool)

    allowed = free & np.isfinite(window) & (window > np.percentile(finite, threshold_pctl))

    ly, lx = y - r0, x - c0

    def fallback_disk() -> np.ndarray:
        """A disk of the minimum size, over whatever pixels are free."""
        yy, xx = np.ogrid[: window.shape[0], : window.shape[1]]
        radius = min_diameter_px / 2
        disk = ((yy - ly) ** 2 + (xx - lx) ** 2 <= radius**2) & free
        out = np.zeros((h, w), dtype=bool)
        out[r0:r1, c0:c1] = disk
        return out

    if not allowed[ly, lx]:
        # Clicked below threshold or on a taken pixel: fall back to a disk so
        # the click still does something predictable.
        return fallback_disk()

    markers = np.zeros(window.shape, dtype=np.int32)

    # Background rim first, seed second, so the seed always wins. A seed near
    # the image edge lands on a clipped window's border, and writing the rim
    # afterwards would erase it -- the region then grows to nothing.
    #
    # The rim only goes on window edges that are not image edges: where the
    # window was clipped there is no "outside" to hold the flood back, and a
    # barrier there would just truncate a legitimate edge ROI.
    if r0 > 0:
        markers[0, :] = 2
    if r1 < h:
        markers[-1, :] = 2
    if c0 > 0:
        markers[:, 0] = 2
    if c1 < w:
        markers[:, -1] = 2

    markers[ly, lx] = 1

    flooded = watershed(
        -np.where(np.isfinite(window), window, -np.inf),
        markers=markers,
        mask=allowed,
    )

    region = flooded == 1

    # Respect the same ceiling the automatic path uses.
    max_area = np.pi * (max_diameter_px / 2) ** 2
    if region.sum() > max_area:
        distance = (np.indices(window.shape) - np.array([[[ly]], [[lx]]])) ** 2
        distance = distance.sum(axis=0)
        keep = np.argsort(distance[region])[: int(max_area)]
        trimmed = np.zeros_like(region)
        ys, xs = np.nonzero(region)
        trimmed[ys[keep], xs[keep]] = True
        region = trimmed

    # A flood that stalls -- the click landed on a narrow ridge, or the
    # neighbours have claimed most of the space -- gives back a handful of
    # pixels. Take the disk instead of returning something that will only be
    # thrown away: the click was deliberate, and an ROI of the minimum size is
    # a more honest reading of it than nothing.
    out = np.zeros((h, w), dtype=bool)
    out[r0:r1, c0:c1] = region

    min_area = np.pi * (min_diameter_px / 2) ** 2
    if out.sum() < min_area:
        disk = fallback_disk()
        if disk.sum() > out.sum():
            return disk

    return out


def manual_bounds(shared: dict) -> dict:
    """
    Size and threshold bounds for a hand-placed seed, relaxed from the shared
    segmentation parameters.

    The automatic parameters are tuned to be *selective*: they are what decides
    which of thousands of candidate maxima become ROIs, and they are set where
    they are because being wrong there is expensive. A manual seed is not a
    candidate -- someone looked at the image and asserted a glomerulus is at
    that pixel. There is no false-positive rate to control, so the same
    thresholds only serve to overrule the person using the tool.

    Concretely, and measured on exp 132, where both hand-added seeds were
    rejected: the diameter range widens to half the floor and 1.5x the
    ceiling, and the local intensity threshold drops by 25 percentiles, since a
    glomerulus the automatic pass missed is usually one that sat below it.

    The one constraint that does not relax is disjointness -- a manual ROI
    still grows only into unclaimed pixels -- because two ROIs sharing pixels
    would make two traces out of one piece of tissue.
    """

    return {
        "min_diameter_px": shared["min_diameter_px"] * MANUAL_MIN_SCALE,
        "max_diameter_px": shared["max_diameter_px"] * MANUAL_MAX_SCALE,
        "threshold_pctl": max(0.0, shared["threshold_pctl"] - MANUAL_THRESHOLD_DROP),
    }


class SegmentationState:
    """Parameters, phase, and edits for a set of per-odor images."""

    def __init__(self, images: dict, params: None | dict = None):
        if not images:
            raise ValueError("No images given.")

        shapes = {np.asarray(v).shape for v in images.values()}
        if len(shapes) != 1:
            raise ValueError(f"Images have differing shapes: {shapes}.")

        self.images = {k: np.asarray(v) for k, v in images.items()}
        self.keys = sorted(images, key=repr)
        self.shape = shapes.pop()

        self.shared = {**GLOM_10X_DEFAULTS, "border_px": 0}
        if params:
            self.shared.update(params)

        # Per-odor overrides, empty until a key is explicitly overridden.
        self.overrides: dict = {}
        self.merge_params = dict(MERGE_DEFAULTS)

        # One curation, applied to the merged mask -- not one per odor.
        self.curation = Curation()

        self.phase = PHASE_TUNE
        self.active = self.keys[0]

        self._masks: dict = {}
        self._merged = None

    # ------------------------------------------------------------------ #
    # Parameters
    # ------------------------------------------------------------------ #

    def params_for(self, key) -> dict:
        """Effective parameters for one odor: shared, plus any override."""
        return {**self.shared, **self.overrides.get(key, {})}

    def set_param(self, name: str, value, *, this_odor_only: bool) -> None:
        """Change a parameter. Refused once curation has begun."""

        if self.phase != PHASE_TUNE:
            raise RuntimeError(
                f"Segmentation parameters are frozen in the {self.phase} phase. "
                "Go back to tuning first."
            )

        if name not in self.shared:
            raise KeyError(f"Unknown parameter {name!r}.")

        if this_odor_only:
            self.overrides.setdefault(self.active, {})[name] = value
        else:
            self.shared[name] = value
            # A shared change is meant to be visible, so clear any override of
            # the same parameter that would mask it.
            for override in self.overrides.values():
                override.pop(name, None)

        self._masks.clear()
        self._merged = None

    def clear_override(self, key=None) -> None:
        self.overrides.pop(self.active if key is None else key, None)
        self._masks.clear()
        self._merged = None

    # ------------------------------------------------------------------ #
    # Phase
    # ------------------------------------------------------------------ #

    def set_merge_param(self, name: str, value) -> None:
        """Change a merge parameter. Only live in the merge phase."""

        if self.phase != PHASE_MERGE:
            raise RuntimeError(
                f"Merge parameters are only adjustable in the merge phase, "
                f"not {self.phase}."
            )

        if name not in self.merge_params:
            raise KeyError(f"Unknown merge parameter {name!r}.")

        self.merge_params[name] = value
        self._merged = None

    def begin_merge(self) -> None:
        """Freeze segmentation, segment every odor, and preview the merge."""
        self.segment_all()
        self.phase = PHASE_MERGE
        self._merged = None

    def begin_curation(self) -> None:
        """Freeze the merge and open editing on the merged mask."""
        if self.phase == PHASE_TUNE:
            self.begin_merge()
        self.merged()
        self.phase = PHASE_CURATE

    def has_curation(self) -> bool:
        return not self.curation.is_empty()

    def back(self, *, discard_curation: bool = False) -> str:
        """
        Step one phase back. Curation cannot survive a parameter change, so
        leaving the curate phase discards it and says how much.
        """

        if self.phase == PHASE_TUNE:
            return self.phase

        if self.phase == PHASE_CURATE:
            if self.has_curation() and not discard_curation:
                edits = self.curation
                raise RuntimeError(
                    f"Going back discards {len(edits.deleted)} deletion(s), "
                    f"{len(edits.live_seeds())} addition(s) and "
                    f"{len(edits.exclude_polygons)} exclusion(s). "
                    "Pass discard_curation=True to confirm."
                )
            self.curation = Curation()
            self.phase = PHASE_MERGE
        else:
            self.phase = PHASE_TUNE

        return self.phase

    # Kept so existing callers and tests keep working.
    def return_to_tuning(self, *, discard_curation: bool = False) -> None:
        while self.phase != PHASE_TUNE:
            self.back(discard_curation=discard_curation)

    # ------------------------------------------------------------------ #
    # Segmentation
    # ------------------------------------------------------------------ #

    def segment(self, key) -> np.ndarray:
        """Auto-segment one odor, cached until a parameter changes."""

        if key in self._masks:
            return self._masks[key]

        params = self.params_for(key)
        border = params.pop("border_px", 0)

        mask, record = segment_image(
            self.images[key],
            exclude_mask=border_mask(self.shape, border),
            **params,
        )

        record["border_px"] = int(border)
        record["group_key"] = repr(key)

        self._masks[key] = mask
        self._records = getattr(self, "_records", {})
        self._records[key] = record

        return mask

    def segment_all(self) -> dict:
        return {key: self.segment(key) for key in self.keys}

    def combined_image(self) -> np.ndarray:
        """
        Per-pixel maximum across odors -- what the merged mask sits on.

        The merged mask spans every odor, so curating it against a single
        odor's image would hide the structure that produced half the ROIs.
        Max rather than mean keeps a glomerulus driven by one odor as visible
        as one driven by all of them.
        """

        if getattr(self, "_combined", None) is None:
            self._combined = np.nanmax(
                np.stack([self.images[k] for k in self.keys]), axis=0
            )

        return self._combined

    def merged(self):
        """Per-odor masks combined into one consensus mask, cached."""

        if self._merged is None:
            self._merged = merge_masks(
                [self.segment(k) for k in self.keys],
                min_area_px=np.pi * (self.shared["min_diameter_px"] / 2) ** 2,
                **self.merge_params,
            )

        return self._merged

    def merged_provenance(self) -> dict[int, list]:
        """Which odors found each merged ROI, by key rather than index."""
        return {
            roi: [self.keys[source] for source, _ in members]
            for roi, members in self.merged().provenance.items()
        }

    def curated_mask(self, key=None) -> np.ndarray:
        """
        The merged mask with manual edits applied.

        `key` is accepted and ignored, so callers written against the earlier
        per-odor signature keep working; curation is on the merge now.
        """

        mask = self.merged().labels.copy()
        edits = self.curation
        min_area = np.pi * (self.shared["min_diameter_px"] / 2) ** 2

        for roi_id in edits.deleted:
            mask[mask == roi_id] = 0

        for polygon in edits.exclude_polygons:
            mask[_polygon_mask(self.shape, polygon)] = 0

        # An exclusion polygon is a region tool, not an ROI tool: it zeroes
        # every pixel it covers, including part of an ROI it merely clips at
        # the edge. What survives is a sliver -- one pixel, in the worst case
        # seen -- and a sliver still becomes a row in the trace table,
        # indistinguishable in form from a real glomerulus and made of
        # whatever the polygon happened not to cover.
        #
        # So a clipped ROI is re-judged by what remains, against the same
        # floor the segmentation and merge enforce. Disconnected leftovers go
        # too: a polygon cutting an ROI in two leaves pieces that would be
        # averaged into a single trace across a gap.
        if edits.exclude_polygons:
            mask, self._clipped = _drop_clipped(mask, min_area_px=min_area)
        else:
            self._clipped = {}

        # Renumber so labels stay contiguous after deletions.
        remaining = [i for i in np.unique(mask) if i > 0]
        out = np.zeros_like(mask)
        for new_id, old_id in enumerate(remaining, start=1):
            out[mask == old_id] = new_id

        # Where each output label came from, so a click can be traced back to
        # the thing that produced it: ("merged", id) or ("seed", index).
        provenance = {
            new_id: ("merged", int(old_id))
            for new_id, old_id in enumerate(remaining, start=1)
        }

        self._rejected_seeds: dict[int, int] = {}

        next_id = len(remaining) + 1
        image = self.combined_image()

        bounds = manual_bounds(self.shared)
        manual_min_area = np.pi * (bounds["min_diameter_px"] / 2) ** 2

        for index, y, x in edits.live_seeds():
            region = grow_seed(image, (y, x), unclaimed=out == 0, **bounds)

            # A floor still applies -- a click in a fully claimed corner can
            # leave a single pixel, and that ROI would yield a trace
            # indistinguishable in form from a real one -- but it is the
            # relaxed manual floor, not the segmentation's. See `manual_bounds`.
            if region.sum() >= manual_min_area:
                out[region] = next_id
                provenance[next_id] = ("seed", index)
                next_id += 1
            else:
                self._rejected_seeds[index] = int(region.sum())

        self._label_provenance = provenance

        return out

    # ------------------------------------------------------------------ #
    # Edits
    # ------------------------------------------------------------------ #

    def _require_curating(self) -> None:
        if self.phase != PHASE_CURATE:
            raise RuntimeError("Manual edits are only available while curating.")

    def delete_at(self, y: int, x: int) -> None | tuple[str, int]:
        """
        Delete whatever ROI is under (y, x), automatic or hand-added.

        Reading the id from the merged mask would miss hand-added ROIs
        entirely -- they do not exist there, so the lookup returns 0 and the
        click silently does nothing. Going through the curated mask and its
        provenance covers both, and records the deletion against whichever
        thing produced the ROI so it survives a redraw.
        """

        self._require_curating()

        labels = self.curated_mask()
        roi_id = int(labels[y, x])

        if roi_id <= 0:
            return None

        source, ident = self._label_provenance[roi_id]

        if source == "merged":
            self.curation.deleted.add(ident)
        else:
            self.curation.deleted_seeds.add(ident)

        return source, ident

    def add_at(self, y: int, x: int) -> int:
        """Place a watershed seed; the ROI grows to fit the image under it."""
        self._require_curating()
        self.curation.added_seeds.append((int(y), int(x)))
        return len(self.curation.added_seeds) - 1

    def exclude_polygon(self, vertices: list[tuple[int, int]]) -> None:
        self._require_curating()
        if len(vertices) >= 3:
            self.curation.exclude_polygons.append(list(vertices))

    def reset_curation(self, key=None) -> None:
        self._require_curating()
        self.curation = Curation()

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def summary(self) -> dict:
        per_odor = {}
        for key in self.keys:
            mask = self.segment(key)
            per_odor[repr(key)] = {
                "n_rois": int(mask.max()),
                "coverage_pct": round(100 * float((mask > 0).mean()), 2),
                "overridden_params": sorted(self.overrides.get(key, {})),
            }

        merged = self.curated_mask()
        areas = np.bincount(merged.ravel())[1:]
        diameters = 2 * np.sqrt(areas / np.pi) if len(areas) else np.array([])
        detections = (
            self.merged().detections_per_roi() if self.merged().n_rois else np.array([])
        )

        return {
            "per_odor": per_odor,
            "merged": {
                "n_rois": int(merged.max()),
                "median_diameter_px": (
                    round(float(np.median(diameters)), 1) if len(diameters) else None
                ),
                "coverage_pct": round(100 * float((merged > 0).mean()), 2),
                "found_by_multiple_odors": int((detections > 1).sum()),
                "deleted": len(self.curation.deleted),
                "added": len(self.curation.live_seeds()),
                "excluded_polygons": len(self.curation.exclude_polygons),
                "added_rejected_too_small": len(getattr(self, "_rejected_seeds", {})),
                "clipped_below_floor": len(getattr(self, "_clipped", {})),
                "min_area_px": round(
                    float(np.pi * (self.shared["min_diameter_px"] / 2) ** 2), 1
                ),
                "manual_min_area_px": round(
                    float(np.pi * (manual_bounds(self.shared)["min_diameter_px"] / 2) ** 2),
                    1,
                ),
            },
        }

    def save(self, path: str | Path) -> Path:
        """Write curated masks plus everything needed to reproduce them."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # The curated merged mask is the deliverable; the per-odor masks are
        # kept so the merge can be re-derived or re-tuned without re-reading
        # any movies.
        np.savez_compressed(
            path,
            labels=self.curated_mask(),
            **{f"mask_{repr(k)}": self.segment(k) for k in self.keys},
        )

        config = {
            "shared": self.shared,
            "overrides": {repr(k): v for k, v in self.overrides.items()},
            "merge_params": self.merge_params,
            "curation": {
                "deleted": sorted(self.curation.deleted),
                "added_seeds": [list(s) for s in self.curation.added_seeds],
                "deleted_seeds": sorted(self.curation.deleted_seeds),
                "exclude_polygons": self.curation.exclude_polygons,
            },
            "provenance": {
                str(roi): [repr(k) for k in keys]
                for roi, keys in self.merged_provenance().items()
            },
            "summary": self.summary(),
        }

        path.with_suffix(".json").write_text(json.dumps(config, indent=2))

        return path


def _drop_clipped(
    labels: np.ndarray, *, min_area_px: float
) -> tuple[np.ndarray, dict[int, int]]:
    """
    Re-judge every ROI after an exclusion polygon has cut into the mask.

    Each label keeps only its largest connected component, and is dropped
    outright if that component falls below `min_area_px`. Returns the cleaned
    mask and `{roi_id: surviving_area}` for everything removed, so the GUI can
    say what the polygon cost rather than have ROIs quietly disappear.
    """

    from scipy.ndimage import label as connected_components

    out = labels.copy()
    dropped: dict[int, int] = {}

    for roi_id in [int(i) for i in np.unique(labels) if i > 0]:
        region = labels == roi_id

        pieces, n_pieces = connected_components(region)
        sizes = np.bincount(pieces.ravel())[1:]
        largest = int(sizes.argmax()) + 1

        if n_pieces > 1:
            out[region & (pieces != largest)] = 0

        if sizes[largest - 1] < min_area_px:
            out[region] = 0
            dropped[roi_id] = int(sizes[largest - 1])

    return out, dropped


def _polygon_mask(shape: tuple[int, int], vertices: list[tuple[int, int]]) -> np.ndarray:
    from skimage.draw import polygon as sk_polygon

    ys = np.array([v[0] for v in vertices])
    xs = np.array([v[1] for v in vertices])

    mask = np.zeros(shape, dtype=bool)
    rr, cc = sk_polygon(ys, xs, shape=shape)
    mask[rr, cc] = True

    return mask
