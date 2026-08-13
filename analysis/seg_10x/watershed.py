"""
Headless watershed segmentation of glomeruli, per odor.

The segmentation core from `segmentation_gui.py`, lifted out of the GUI so it
can be called per odor and its parameters recorded. The GUI stays useful for
curating the result; this is what produces the result to curate.

Sizes are in PIXELS, deliberately.

The obvious move is to specify glomerulus size in micrometres and convert
using the session's micrometres-per-pixel. That conversion is not trustworthy
here: `experiments.width_um / width_px` is only accurate for 20x, so on a 10x
session it yields a scale that is simply wrong. Converting from it would bake
that error into the size filter while looking principled. Pixels are what the
image actually has, so pixels are what the parameters use -- and callers are
expected to set them per session against the image dimensions, which every
returned record includes.

The original computed `distance_transform_edt(binary)` and then never used it,
seeding the watershed from `peak_local_max` on the correlation image and
flooding `-corr`. That is a reasonable design -- basins follow the correlation
landscape rather than shape alone -- so it is kept, minus the dead transform.
"""

from __future__ import annotations

import numpy as np

from skimage.feature import peak_local_max
from skimage.measure import label as cc_label
from skimage.measure import regionprops
from skimage.segmentation import watershed

# Starting points only. Pixel scale varies several-fold between sessions, so
# these must be checked against each session's images rather than trusted.
DEFAULT_MIN_DIAMETER_PX = 20.0
DEFAULT_MAX_DIAMETER_PX = 55.0

# Settled on exp 202 (m462, TH-GCaMP, 10x, 526 x 550 px) against per-odor
# correlation maps, with glomerulus diameter judged by eye at 25-40 px.
# Gives 38 ROIs for epsilon and 36 for lambda, median diameter 28 px, 62% of
# ROIs inside the 25-40 px range, and 86% of the brightest pixels recovered.
#
# Each part earns its place:
#   adaptive + p60 floor  recovers dim glomeruli; the floor stops the local
#                         rule manufacturing ROIs from noise where there is
#                         no structure (without it, 126 ROIs became 257)
#   peak_distance 20      at 10 px the watershed shattered glomeruli into
#                         fragments that then failed the size floor
#   split_oversized       fused bright plateaus are re-seeded rather than
#                         discarded; recovered 5823 px in two regions
#
# Loosening peak_distance to 15 and the ceiling to 60 raises coverage of the
# upper bulb from 68% to 78%, but drops median diameter to 25 px and the
# fraction in range from 62% to 50%. Not used by default; the pieces it adds
# skew smaller than one glomerulus.
GLOM_10X_DEFAULTS = {
    "mode": "watershed",
    "threshold_pctl": 60.0,
    "adaptive_block_px": 101,
    "min_diameter_px": 20.0,
    "max_diameter_px": 55.0,
    "peak_distance_px": 20.0,
    "split_oversized": True,
}


# Size parameters in `GLOM_10X_DEFAULTS` were set on exp 202, whose reported
# scale is 2.0 um/px. Use `scale_params` to carry them to another session.
REFERENCE_UM_PER_PX = 2.0

# peak_distance is deliberately scaled less than the diameters. It sets seed
# spacing, which trades object size against noise; noise does not grow with
# magnification, so scaling it fully over-spaces the seeds. Measured on exp 132
# (1.0 um/px): a full 2x -> peak 40 gave 27 ROIs and 85% in the target band,
# while 1.5x -> peak 30 gave 35 ROIs and 94%.
PEAK_DISTANCE_SCALE_EXPONENT = 0.6


def scale_params(
    params: dict,
    *,
    from_um_per_px: float = REFERENCE_UM_PER_PX,
    to_um_per_px: float,
) -> dict:
    """
    Carry pixel size parameters from one session's scale to another's.

    `experiments.width_um / width_px` is not calibrated at 10x, so its absolute
    value is wrong there -- but the miscalibration is a fixed factor, so the
    *ratio* between two sessions is right and divides that factor out. Scaling
    by the ratio therefore works without ever trusting the absolute number.

    Only within one magnification. A 10x-to-20x ratio does not divide the
    factor out, since 20x carries none; it would be wrong by exactly the 10x
    calibration error. Keep a reference session per magnification.

    Verified on exp 202 (2.0 um/px) -> exp 132 (1.0 um/px): the predicted 2x
    matches the scaling the segmentation independently needed.
    """

    if from_um_per_px <= 0 or to_um_per_px <= 0:
        raise ValueError(
            f"Scales must be positive, got {from_um_per_px} and {to_um_per_px}."
        )

    # More um per pixel means fewer pixels per object, so pixel sizes scale
    # with the inverse ratio.
    factor = from_um_per_px / to_um_per_px

    out = dict(params)

    for name in ("min_diameter_px", "max_diameter_px"):
        if out.get(name) is not None:
            out[name] = round(float(out[name]) * factor, 1)

    if out.get("peak_distance_px") is not None:
        out["peak_distance_px"] = round(
            float(out["peak_distance_px"]) * factor**PEAK_DISTANCE_SCALE_EXPONENT, 1
        )

    # The adaptive window is a background estimator: it should stay a few
    # object-widths across, so it scales with the objects.
    if out.get("adaptive_block_px"):
        block = int(round(out["adaptive_block_px"] * factor))
        out["adaptive_block_px"] = block + 1 if block % 2 == 0 else block

    return out


def _area_bounds_px(min_diameter_px: float, max_diameter_px: float) -> tuple[float, float]:
    """Disk-equivalent area bounds for a diameter range, both in pixels."""

    if min_diameter_px <= 0:
        raise ValueError(f"min_diameter_px must be positive, got {min_diameter_px}.")

    if min_diameter_px >= max_diameter_px:
        raise ValueError(
            f"min_diameter_px ({min_diameter_px}) must be below "
            f"max_diameter_px ({max_diameter_px})."
        )

    return np.pi * (min_diameter_px / 2) ** 2, np.pi * (max_diameter_px / 2) ** 2


def _split_oversized(
    labels: np.ndarray,
    *,
    image: np.ndarray,
    binary: np.ndarray,
    max_area: float,
    peak_distance_px: float,
    threshold: float,
    max_rounds: int = 4,
) -> np.ndarray:
    """
    Re-seed and re-flood only the regions that came out larger than `max_area`.

    A single global seed spacing cannot serve a field where some glomeruli sit
    alone and others are fused into a bright plateau. Spacing tight enough to
    split the plateau shatters the isolated ones; spacing loose enough to keep
    those whole lets the plateau survive as one object, which the size ceiling
    then discards -- losing real signal rather than dividing it.

    So the spacing is global only for the first pass. Anything still oversized
    is re-seeded at successively tighter spacing, and only within its own
    bounding box, until it fits or the spacing bottoms out.
    """

    out = labels.copy()
    next_label = int(out.max()) + 1
    spacing = peak_distance_px

    for _ in range(max_rounds):
        oversized = [r for r in regionprops(out) if r.area > max_area]
        if not oversized:
            break

        spacing = max(2.0, spacing / 2)

        for region in oversized:
            r0, c0, r1, c1 = region.bbox
            sub_mask = out[r0:r1, c0:c1] == region.label
            sub_image = image[r0:r1, c0:c1]

            coords = peak_local_max(
                np.where(np.isfinite(sub_image) & sub_mask, sub_image, -np.inf),
                min_distance=max(1, int(round(spacing))),
                threshold_abs=threshold,
                exclude_border=False,
            )

            # One seed means nothing to split by; leave it for the size filter.
            if len(coords) < 2:
                continue

            seeds = np.zeros(sub_mask.shape, dtype=np.int32)
            for i, (y, x) in enumerate(coords, start=1):
                if sub_mask[y, x]:
                    seeds[y, x] = i

            if seeds.max() < 2:
                continue

            pieces = watershed(
                -np.where(np.isfinite(sub_image), sub_image, -np.inf),
                markers=seeds,
                mask=sub_mask,
            )

            out[r0:r1, c0:c1][sub_mask] = 0
            for piece in range(1, int(pieces.max()) + 1):
                sel = pieces == piece
                if sel.any():
                    out[r0:r1, c0:c1][sel] = next_label
                    next_label += 1

    return out


def segment_image(
    image: np.ndarray,
    *,
    threshold_pctl: float = 90.0,
    mode: str = "watershed",
    min_diameter_px: float = DEFAULT_MIN_DIAMETER_PX,
    max_diameter_px: float = DEFAULT_MAX_DIAMETER_PX,
    peak_distance_px: None | float = None,
    adaptive_block_px: None | int = None,
    adaptive_offset: float = 0.0,
    split_oversized: bool = True,
    exclude_mask: None | np.ndarray = None,
) -> tuple[np.ndarray, dict]:
    """
    Segment one image into a label mask.

    `threshold_pctl` is a percentile of the image's own values, so the same
    setting transfers between images on different scales -- a correlation map
    and a z-score map do not share units.

    `peak_distance_px` is the minimum separation between watershed seeds and
    defaults to the minimum diameter -- i.e. seeds no closer than the smallest
    object being looked for.

    Setting it much below that over-seeds badly. At half the minimum diameter
    on a real correlation map, every noise bump inside a glomerulus became its
    own seed: 316 seeds where there were perhaps 30 glomeruli, the watershed
    shattered each into fragments, and 90% of them then failed the size floor.
    Raising it from 5 to 10 px more than doubled the ROI count and the area
    covered, because the fragments became whole objects.

    Note that two blobs closer than about twice their own width have a single
    combined maximum and cannot be separated by any seeding rule, whatever
    this is set to.

    `adaptive_block_px` turns on local thresholding with that window size,
    which should be a few times the object diameter -- large enough to average
    over background, small enough to track the brightness gradient. It is
    combined with the global percentile by AND, so `threshold_pctl` becomes an
    absolute floor and should be lowered when adaptive is on.
    `adaptive_offset` raises the local bar; larger values are more selective.

    `split_oversized` re-seeds any region exceeding the size ceiling at tighter
    spacing until it fits, instead of discarding it. Without it, a fused bright
    plateau is simply dropped -- on one test image that silently threw away
    5823 px of real signal in two regions.

    Returns the label image and a JSON-serialisable record of what was used,
    including the image dimensions so the pixel settings can be judged.
    """

    if mode not in ("watershed", "threshold"):
        raise ValueError(f"mode must be 'watershed' or 'threshold', got {mode!r}.")

    image = np.asarray(image, dtype=np.float32)

    finite = image[np.isfinite(image)]
    if finite.size == 0:
        raise ValueError("Image has no finite values.")

    threshold = float(np.percentile(finite, threshold_pctl))
    binary = np.isfinite(image) & (image > threshold)

    if adaptive_block_px:
        # Local threshold: a pixel must stand above its own neighbourhood.
        # This is what recovers glomeruli in the dimmer parts of a field whose
        # brightness varies several-fold across the bulb.
        from skimage.filters import threshold_local

        block = int(adaptive_block_px)
        if block % 2 == 0:
            block += 1  # threshold_local requires an odd window

        filled = np.where(np.isfinite(image), image, threshold)
        local = threshold_local(
            filled, block_size=block, method="gaussian", offset=-adaptive_offset
        )

        # AND, not OR. A local criterion alone finds the brightest fraction of
        # every neighbourhood including empty ones, manufacturing ROIs out of
        # noise wherever there is no real structure. The global percentile acts
        # as an absolute floor that empty regions cannot clear, so `threshold_pctl`
        # should be set well below its non-adaptive value -- it is now a floor
        # rather than the operative threshold.
        binary &= filled > local

    if exclude_mask is not None:
        binary &= ~exclude_mask.astype(bool)

    min_area, max_area = _area_bounds_px(min_diameter_px, max_diameter_px)

    if mode == "threshold" or not binary.any():
        labels = cc_label(binary, connectivity=1)
        n_seeds = 0

    else:
        if peak_distance_px is None:
            peak_distance_px = min_diameter_px

        min_distance = max(1, int(round(peak_distance_px)))

        coords = peak_local_max(
            np.where(np.isfinite(image), image, -np.inf),
            min_distance=min_distance,
            threshold_abs=threshold,
            exclude_border=False,
        )
        n_seeds = len(coords)

        if n_seeds == 0:
            labels = np.zeros(image.shape, dtype=np.int32)
        else:
            seeds = np.zeros(image.shape, dtype=np.int32)
            for i, (y, x) in enumerate(coords, start=1):
                seeds[y, x] = i

            # Flood the inverted correlation landscape: basins grow outward
            # from each peak until they meet, so a boundary lands where the
            # image dips between two glomeruli.
            labels = watershed(
                -np.where(np.isfinite(image), image, -np.inf),
                markers=seeds,
                mask=binary,
            )

    if split_oversized:
        labels = _split_oversized(
            labels,
            image=image,
            binary=binary,
            max_area=max_area,
            peak_distance_px=peak_distance_px or min_diameter_px,
            threshold=threshold,
        )

    kept = [
        region.label
        for region in regionprops(labels)
        if min_area <= region.area <= max_area
    ]

    out = np.zeros(labels.shape, dtype=np.int32)
    for new_label, old_label in enumerate(sorted(kept), start=1):
        out[labels == old_label] = new_label

    params = {
        "image_shape_px": list(image.shape),
        "threshold_pctl": float(threshold_pctl),
        "threshold_value": threshold,
        "mode": mode,
        "min_diameter_px": float(min_diameter_px),
        "max_diameter_px": float(max_diameter_px),
        "min_area_px": round(min_area, 1),
        "max_area_px": round(max_area, 1),
        "peak_distance_px": None if peak_distance_px is None else float(peak_distance_px),
        "adaptive_block_px": None if not adaptive_block_px else int(adaptive_block_px),
        "adaptive_offset": float(adaptive_offset),
        "pixels_above_threshold_pct": round(100 * float(binary.mean()), 2),
        "n_seeds": int(n_seeds),
        "split_oversized": bool(split_oversized),
        "n_before_size_filter": int(labels.max()),
        "n_rois": int(out.max()),
    }

    return out, params


def segment_per_group(images: dict, **kwargs) -> tuple[list[np.ndarray], list, list[dict]]:
    """
    Segment one image per group key, ready for `merge.merge_masks`.

    Returns masks, the keys in the same order, and the per-image parameter
    records -- the keys are needed to say which odor found which ROI once the
    masks have been merged.
    """

    keys = sorted(images, key=repr)

    masks, params = [], []
    for key in keys:
        mask, record = segment_image(images[key], **kwargs)
        record["group_key"] = repr(key)
        masks.append(mask)
        params.append(record)

    return masks, keys, params
