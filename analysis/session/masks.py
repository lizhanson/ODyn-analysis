"""Mask side-products: a MATLAB `.mat` and a picture of the ROIs on the image."""

from __future__ import annotations

import re
import warnings

from pathlib import Path

import numpy as np

from ..seg_10x.gui import _PALETTE

# Fraction of the ROI colour in the blend. 0.25 leaves the overlay 75%
# transparent: the ROI boundaries read clearly while the correlation map
# underneath -- the thing that justifies each ROI -- stays visible through them.
DEFAULT_OVERLAY_ALPHA = 0.25

# Same as the GUI's `_grey_image` and the QC stills, so the three agree.
GREY_PERCENTILES = (1.0, 99.5)


def _matlab_name(key) -> str:
    """A group key as a legal MATLAB struct field: `7` -> `odor7`."""

    if isinstance(key, tuple):
        text = "_".join(str(part) for part in key)
    else:
        text = str(key)

    text = re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9_]", "_", text)).strip("_")

    return text if text[:1].isalpha() else f"odor{text}"


def mask_overlay_rgb(
    image: np.ndarray,
    labels: np.ndarray,
    *,
    alpha: float = DEFAULT_OVERLAY_ALPHA,
    percentiles: tuple[float, float] = GREY_PERCENTILES,
) -> np.ndarray:
    """Blend a label image over a greyscale background, GUI-style."""

    image = np.asarray(image, dtype=np.float32)
    labels = np.asarray(labels)

    if image.shape != labels.shape:
        raise ValueError(
            f"image and labels must have the same shape, got "
            f"{image.shape} and {labels.shape}."
        )

    finite = image[np.isfinite(image)]
    low, high = np.percentile(finite, percentiles)

    scaled = np.clip((np.nan_to_num(image, nan=float(low)) - low)
                     / max(float(high - low), 1e-12), 0, 1)
    rgb = np.repeat((scaled * 255).astype(np.uint8)[:, :, None], 3, axis=2)

    roi = labels > 0
    if roi.any():
        colour = _PALETTE[(labels[roi] - 1) % len(_PALETTE)].astype(np.float32)
        rgb[roi] = (
            (1.0 - alpha) * rgb[roi].astype(np.float32) + alpha * colour
        ).round().astype(np.uint8)

    return rgb


def save_mask_overlay(
    path: str | Path,
    image: np.ndarray,
    labels: np.ndarray,
    *,
    alpha: float = DEFAULT_OVERLAY_ALPHA,
) -> Path:
    """Write the overlay as a PNG."""

    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    Image.fromarray(mask_overlay_rgb(image, labels, alpha=alpha)).save(path)

    return path


def background_image(images) -> np.ndarray:
    """One background for a merged mask, from the per-odor maps it came from."""

    if isinstance(images, dict):
        stack = np.stack([np.asarray(images[k], dtype=np.float32)
                          for k in sorted(images, key=repr)])

        # A pixel NaN in every map -- a corner the motion correction never
        # covered -- averages to NaN, which is the right answer and not worth
        # a warning. `mask_overlay_rgb` renders it at the low end.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(stack, axis=0)

    return np.asarray(images, dtype=np.float32)


def save_masks_mat(
    path: str | Path,
    labels: np.ndarray,
    *,
    per_group_masks: None | dict = None,
    exp_name: str = "",
    group_id: None | int = None,
    mask_hash: str = "",
    processed_on: str = "",
) -> Path:
    """Write the masks as a MATLAB v5 `.mat`, oriented as MATLAB expects."""

    from scipy.io import savemat

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    labels = np.asarray(labels).astype(np.int32)

    payload = {
        "labels": labels,
        "n_rois": int(labels.max()),
        "exp_name": exp_name,
        "group_id": -1 if group_id is None else int(group_id),
        "mask_hash": mask_hash,
        "processed_on": processed_on,
        "readme": (
            "ROI masks. labels is (rows, cols) as displayed: no permute "
            "needed, unlike h5read on the .h5 beside this file. Pixel value "
            "0 is background and N is ROI N, matching roi_id in /rois of the "
            ".h5, which is the file the traces live in. per_odor holds the "
            "single-odor masks from before the merge. mask_hash fingerprints "
            "labels; if it differs from the .h5's, they are not the same mask."
        ),
    }

    if per_group_masks:
        payload["per_odor"] = {
            _matlab_name(key): np.asarray(mask).astype(np.int32)
            for key, mask in per_group_masks.items()
        }

    savemat(str(path), payload, do_compression=True)

    return path

def find_saved_masks(output_dir: str | Path) -> list[Path]:
    """Every saved mask for a session, oldest first. Rounds before .mat files."""

    directory = Path(output_dir)

    if not directory.is_dir():
        return []

    rounds = sorted(directory.glob("group*_processed_*.h5"))
    mats = sorted(directory.glob("group*_masks_processed_*.mat"))

    return rounds + mats


def load_latest_mask(output_dir: str | Path) -> None | dict:
    """The most recent curated mask for a session, or None if there is none."""

    candidates = find_saved_masks(output_dir)

    if not candidates:
        return None

    latest = candidates[-1]

    if latest.suffix == ".h5":
        from .h5io import open_h5

        with open_h5(latest) as f:
            labels = f["masks/labels"][:]
            per_group = {k: f[f"masks/{k}"][:] for k in f["masks"] if k != "labels"}
            recorded = str(f.attrs.get("mask_hash", ""))
            has_traces = "traces" in f
    else:
        from scipy.io import loadmat

        data = loadmat(str(latest), squeeze_me=True, struct_as_record=False)
        labels = np.asarray(data["labels"])
        recorded = str(data.get("mask_hash", ""))
        has_traces = False
        per_group = {}

        struct = data.get("per_odor")
        if struct is not None and hasattr(struct, "_fieldnames"):
            per_group = {
                name: np.asarray(getattr(struct, name))
                for name in struct._fieldnames
            }

    return {
        "labels": labels,
        "per_group_masks": per_group,
        "mask_hash": recorded,
        "path": latest,
        "source": "round" if latest.suffix == ".h5" else "mat",
        "has_traces": has_traces,
        "n_rois": int(labels.max()),
        "n_available": len(candidates),
    }
