"""Mask loading and ROI overlay images."""

from __future__ import annotations

import warnings

from pathlib import Path
import json

import numpy as np

from ..seg_10x.gui import _PALETTE

# Fraction of the ROI colour in the blend. 0.25 leaves the overlay 75%
# transparent: the ROI boundaries read clearly while the correlation map
# underneath -- the thing that justifies each ROI -- stays visible through them.
DEFAULT_OVERLAY_ALPHA = 0.25

# Same as the GUI's `_grey_image` and the QC stills, so the three agree.
GREY_PERCENTILES = (1.0, 99.5)


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


def save_mask_bundle(
    path: str | Path,
    labels: np.ndarray,
    *,
    per_group_masks: None | dict = None,
    reference: None | np.ndarray = None,
    config: None | dict = None,
) -> Path:
    """Write a portable mask-only HDF5 for extraction on another computer."""

    import h5py

    path = Path(path).with_suffix(".h5")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")

    with h5py.File(partial, "w") as handle:
        handle.attrs["file_type"] = "odyn_10x_mask_bundle"
        handle.attrs["config_json"] = json.dumps(config or {}, default=str)
        masks = handle.create_group("masks")
        masks.create_dataset("labels", data=np.asarray(labels, np.int32), compression="gzip")
        groups = masks.create_group("per_group")
        for index, (key, mask) in enumerate((per_group_masks or {}).items()):
            dataset = groups.create_dataset(
                str(index), data=np.asarray(mask, np.int32), compression="gzip"
            )
            dataset.attrs["key"] = repr(key)
        if reference is not None:
            handle.create_dataset("reference", data=np.asarray(reference, np.float32),
                                  compression="gzip")

    partial.replace(path)
    return path


def load_mask_bundle(path: str | Path) -> dict:
    """Read a portable 10x mask bundle."""

    import h5py

    path = Path(path)
    with h5py.File(path, "r") as handle:
        if handle.attrs.get("file_type") != "odyn_10x_mask_bundle":
            raise ValueError(f"Not a 10x mask bundle: {path}")
        per_group = {
            handle[f"masks/per_group/{name}"].attrs["key"]:
            handle[f"masks/per_group/{name}"][:]
            for name in handle["masks/per_group"]
        }
        return {
            "path": path,
            "labels": handle["masks/labels"][:],
            "per_group": per_group,
            "reference": handle["reference"][:] if "reference" in handle else None,
            "config": json.loads(handle.attrs.get("config_json", "{}")),
        }


def load_10x_working_mask(path: str | Path) -> dict:
    """Load a GUI scratch checkpoint for recovery/publication, not analysis."""
    import h5py

    path = Path(path)
    with h5py.File(path, "r") as handle:
        if handle.attrs.get("file_type") != "odyn_10x_working_mask":
            raise ValueError(f"Not a 10x GUI working-mask checkpoint: {path}")
        config = json.loads(handle.attrs.get("config_json", "{}"))
        per_group = {
            name: handle[f"masks/{name}"][:]
            for name in handle["masks"] if name != "labels"
        }
        return {
            "path": path,
            "labels": handle["masks/labels"][:],
            "per_group": per_group,
            "reference": None,
            "config": {
                "segmentation": config.get("shared", {}),
                "merge": config.get("merge_params", {}),
                "curation": config.get("curation"),
                "working_checkpoint": config,
            },
            "source": "10x GUI working checkpoint",
        }


def find_saved_masks(output_dir: str | Path) -> list[Path]:
    """Every saved mask for a session, oldest first. Rounds before .mat files."""

    directory = Path(output_dir)

    if not directory.is_dir():
        return []

    rounds = sorted(directory.glob("group*_processed_*.h5"))
    bundles = sorted(directory.glob("group*_10x_masks_processed_*.h5"))
    mats = sorted(directory.glob("group*_masks_processed_*.mat"))

    return sorted([*rounds, *bundles, *mats], key=lambda path: path.stat().st_mtime)


def load_latest_mask(output_dir: str | Path) -> None | dict:
    """The most recent curated mask for a session, or None if there is none."""

    candidates = find_saved_masks(output_dir)

    if not candidates:
        return None

    latest = candidates[-1]

    if latest.name.find("_10x_masks_processed_") >= 0:
        bundle = load_mask_bundle(latest)
        return {
            **bundle,
            "source": "10x mask bundle",
            "has_traces": False,
            "n_rois": int(bundle["labels"].max()),
            "n_available": len(candidates),
        }

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
