"""Headless 20x reference-image, soma, and process segmentation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..seg_10x.state import grow_seed
SOMA_DEFAULTS = {
    "dog_threshold": 0.12,
    "dog_sigma_ratio": 1.4,
    "min_sigma_px": 2.5,
    "max_sigma_px": 7.0,
    "overlap": 0.5,
    "growth_threshold_pctl": 68.0,
    "min_diameter_px": 5.0,
    "max_diameter_px": 24.0,
    # The adaptive fill is still bounded: without this safeguard it expands
    # far beyond curated soma boundaries on real 20x structural images.
    "radius_scale": 2.3,
    "min_radius_px": 4.0,
    "max_radius_px": 16.0,
    "border_px": 10,
}

# The ridge image is the one expensive product that does not depend on every
# process parameter. Naming its inputs lets the cache survive a slider move
# that cannot change it, and lets it be dropped by one that can.
RIDGE_PARAMS = frozenset({"ridge_sigma_min_px", "ridge_sigma_max_px"})

PROCESS_DEFAULTS = {
    "global_ridge_pctl": 70.0,
    "adaptive_block_px": 11,
    "adaptive_offset": -0.002,
    "manual_ridge_corridor_px": 3,
    "manual_ridge_adaptive_block_px": 11,
    "manual_ridge_adaptive_offset": -0.006,
    "ridge_sigma_min_px": 1,
    "ridge_sigma_max_px": 4,
    "soma_guard_px": 2,
    "min_foreground_area_px": 20,
    "min_skeleton_length_px": 10,
    "min_filled_area_px": 20,
    "border_px": 10,
}


def build_reference_images(
    movie_paths,
    *,
    odor_on_frames,
    odor_off_frames,
    frames_per_trial: int = 3,
    structural_percentile: float = 75.0,
    structural_sigma_px: float = 0.8,
) -> tuple[dict[str, np.ndarray], dict]:
    """Robust structural composite from odor windows spanning the session.

    Each trial contributes one odor-window mean image. Taking an upper
    percentile across those trial means preserves processes visible in the
    brighter awake block instead of allowing dim anesthetized trials to wash
    them out, while still sampling the full session.
    """

    import tifffile
    from skimage.filters import gaussian

    paths = [Path(p) for p in movie_paths]
    on = np.asarray(odor_on_frames, int)
    off = np.asarray(odor_off_frames, int)
    if not (len(paths) == len(on) == len(off)) or not paths:
        raise ValueError("movie paths and odor windows must align and be nonempty")
    percentile = float(structural_percentile)
    if not 50 <= percentile <= 100:
        raise ValueError("structural_percentile must be between 50 and 100")
    per_trial = int(frames_per_trial)
    if per_trial < 1:
        raise ValueError("frames_per_trial must be at least 1")
    blocks, sampled = [], []
    for path, left, right in zip(paths, on, off):
        if right <= left:
            raise ValueError(f"invalid odor window [{left}, {right}) for {path.name}")
        indices = np.unique(np.linspace(left, right - 1, min(per_trial, right-left)).round().astype(int))
        with tifffile.TiffFile(path) as tif:
            block = np.stack([tif.pages[int(i)].asarray() for i in indices]).astype(np.float32)
        blocks.append(block)
        sampled.append({"movie_path": str(path), "frames": indices.tolist()})
    movie = np.concatenate(blocks, axis=0)

    trial_means = np.stack([block.mean(axis=0) for block in blocks])
    structural = gaussian(
        np.percentile(trial_means, percentile, axis=0),
        sigma=structural_sigma_px, preserve_range=True,
    ).astype(np.float32)
    return {"structural": structural}, {
        "sampling": "frames distributed across every session odor window",
        "structural_composite": "percentile across per-trial odor-window means",
        "structural_percentile": percentile,
        "sampled_trials": sampled,
        "n_frames": int(len(movie)),
        "frames_per_trial": per_trial,
        "structural_sigma_px": float(structural_sigma_px),
    }


def _border_mask(shape, border_px):
    mask = np.zeros(shape, bool)
    b = int(border_px)
    if b > 0:
        mask[:b] = mask[-b:] = True
        mask[:, :b] = mask[:, -b:] = True
    return mask


def detect_somas(image: np.ndarray, params: dict | None = None):
    """DoG soma-center candidates followed by adaptive watershed growth."""

    from scipy.ndimage import label as connected_components
    from skimage.feature import blob_dog

    p = {**SOMA_DEFAULTS, **(params or {})}
    image = np.asarray(image, dtype=np.float32)
    finite = image[np.isfinite(image)]
    lo, hi = np.percentile(finite, (1, 99.8))
    normal = np.clip((image - lo) / max(float(hi - lo), 1e-9), 0, 1)
    blobs = blob_dog(
        normal,
        min_sigma=float(p["min_sigma_px"]),
        max_sigma=float(p["max_sigma_px"]),
        sigma_ratio=float(p["dog_sigma_ratio"]),
        threshold=float(p["dog_threshold"]),
        overlap=float(p["overlap"]),
        exclude_border=int(p["border_px"]),
    )

    labels = np.zeros(image.shape, np.int32)
    occupied = _border_mask(image.shape, p["border_px"])
    yy, xx = np.ogrid[: image.shape[0], : image.shape[1]]
    order = np.argsort([image[int(y), int(x)] for y, x, _ in blobs])[::-1]
    seeds = []

    for candidate in order:
        y0, x0, sigma = blobs[candidate]
        y, x = int(round(y0)), int(round(x0))
        if occupied[y, x]:
            continue
        region = grow_seed(
            image,
            (y, x),
            unclaimed=~occupied,
            min_diameter_px=float(p["min_diameter_px"]),
            max_diameter_px=float(p["max_diameter_px"]),
            threshold_pctl=float(p["growth_threshold_pctl"]),
        )
        radius = np.clip(
            float(sigma) * float(p["radius_scale"]),
            float(p["min_radius_px"]),
            float(p["max_radius_px"]),
        )
        region &= (yy - y) ** 2 + (xx - x) ** 2 <= radius**2
        pieces, _ = connected_components(region)
        region = pieces == pieces[y, x] if pieces[y, x] > 0 else np.zeros_like(region)
        if region.sum() < np.pi * (float(p["min_diameter_px"]) / 2) ** 2:
            continue
        ident = int(labels.max()) + 1
        labels[region] = ident
        occupied |= region
        seeds.append((y, x, float(sigma)))

    return labels, {
        "params": p,
        "n_candidates": int(len(blobs)),
        "n_rois": int(labels.max()),
        "seeds": seeds,
    }


def ridge_image(structural: np.ndarray, params: dict | None = None):
    from skimage.filters import sato

    p = {**PROCESS_DEFAULTS, **(params or {})}
    finite = structural[np.isfinite(structural)]
    lo, hi = np.percentile(finite, (1, 99.8))
    normal = np.clip((structural - lo) / max(float(hi - lo), 1e-9), 0, 1)
    sigmas = range(int(p["ridge_sigma_min_px"]), int(p["ridge_sigma_max_px"]) + 1)
    return sato(normal, sigmas=sigmas, black_ridges=False).astype(np.float32)


def process_foreground(ridge, soma_labels, params: dict | None = None):
    from scipy.ndimage import binary_dilation
    from skimage.filters import threshold_local

    p = {**PROCESS_DEFAULTS, **(params or {})}
    exclude = _border_mask(ridge.shape, p["border_px"])
    floor = np.percentile(ridge[~exclude], float(p["global_ridge_pctl"]))
    block = max(3, int(p["adaptive_block_px"]))
    block += block % 2 == 0
    local = threshold_local(
        ridge, block, method="gaussian", offset=float(p["adaptive_offset"])
    )
    foreground = (ridge > floor) & (ridge > local) & ~exclude
    foreground &= ~binary_dilation(
        soma_labels > 0, iterations=int(p["soma_guard_px"])
    )
    # A direct component-size floor avoids skimage's min_size/max_size API
    # transition and keeps the exact inequality explicit.
    from scipy.ndimage import label as connected_components
    pieces, _ = connected_components(foreground, structure=np.ones((3, 3), int))
    sizes = np.bincount(pieces.ravel())
    foreground &= sizes[pieces] >= int(p["min_foreground_area_px"])
    return foreground, float(floor)


def detect_processes(
    structural: np.ndarray,
    soma_labels: np.ndarray,
    params: dict | None = None,
    *,
    ridge: np.ndarray | None = None,
):
    """Split ridge skeletons at junctions and watershed-fill each segment."""

    from scipy.ndimage import binary_dilation, convolve
    from skimage.measure import label, regionprops
    from skimage.morphology import skeletonize
    from skimage.segmentation import watershed

    p = {**PROCESS_DEFAULTS, **(params or {})}
    ridge = ridge_image(structural, p) if ridge is None else np.asarray(ridge)
    foreground, floor = process_foreground(ridge, soma_labels, p)
    skeleton = skeletonize(foreground)
    degree = convolve(
        skeleton.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant"
    ) - skeleton
    junctions = skeleton & (degree > 2)
    raw = label(skeleton & ~binary_dilation(junctions, iterations=1), connectivity=2)
    markers = np.zeros_like(raw)
    next_marker = 1
    for region in regionprops(raw):
        if region.area >= int(p["min_skeleton_length_px"]):
            markers[raw == region.label] = next_marker
            next_marker += 1

    flooded = watershed(-ridge, markers=markers, mask=foreground)
    labels = np.zeros_like(flooded)
    next_label = 1
    for region in regionprops(flooded):
        if region.area >= int(p["min_filled_area_px"]):
            labels[flooded == region.label] = next_label
            next_label += 1

    return labels, {
        "params": p,
        "global_ridge_value": floor,
        "n_skeleton_pixels": int(skeleton.sum()),
        "n_markers": int(markers.max()),
        "n_rois": int(labels.max()),
        "ridge": ridge,
        "foreground": foreground,
        "skeleton": skeleton,
        "markers": markers,
    }
