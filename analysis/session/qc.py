"""
Quality-control stills for one session's per-group correlation maps.

The correlation map is what segmentation actually runs on, so it is the image
worth keeping: a session that segments badly usually shows it here first --
low contrast, scan-line striping, a bright plateau where the motion correction
drifted. Saving one per odor (and condition) makes that reviewable later
without re-streaming 76 GB.

Two files per group, because they answer different questions:

    <group>.tif   float32, the map itself. Quantitative -- re-segment or
                  re-measure from it without recomputing.
    <group>.png   8-bit render on a shared scale. For flicking through.

The PNG scale is shared across groups deliberately. Per-image autoscaling
makes a weak odor look as good as a strong one, which is exactly the
comparison a QC sweep is trying to make.
"""

from __future__ import annotations

import json

from pathlib import Path

import numpy as np


def _slug(key) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9_.-]", "_", repr(key))


def save_correlation_stills(
    images: dict,
    output_dir: str | Path,
    *,
    subfolder: str = "correlation_maps",
    percentiles: tuple[float, float] = (1.0, 99.5),
    context: None | dict = None,
) -> dict:
    """
    Write a float32 TIFF and a shared-scale PNG per group, plus a summary.

    Returns the summary dict, which is also written as `summary.json`:
    contrast and coverage per group, so a bad session is visible from the
    numbers alone.
    """

    import tifffile
    from matplotlib import cm
    from PIL import Image

    directory = Path(output_dir) / subfolder
    directory.mkdir(parents=True, exist_ok=True)

    # Existing stills are replaced rather than accumulated, matching the
    # z-movie cache: one current set per session.
    for stale in list(directory.glob("*.tif")) + list(directory.glob("*.png")):
        stale.unlink()

    keys = sorted(images, key=repr)
    stacked = np.stack([np.asarray(images[k]) for k in keys])
    finite = stacked[np.isfinite(stacked)]
    low, high = np.percentile(finite, percentiles)

    summary = {"scale": [float(low), float(high)], "groups": {}}

    for key in keys:
        image = np.asarray(images[key], dtype=np.float32)
        name = _slug(key)

        tifffile.imwrite(directory / f"{name}.tif", image)

        scaled = np.clip((image - low) / max(high - low, 1e-12), 0, 1)
        rgb = (cm.magma(np.nan_to_num(scaled)) * 255).astype(np.uint8)
        Image.fromarray(rgb).save(directory / f"{name}.png")

        median = float(np.median(image))
        summary["groups"][repr(key)] = {
            "median": round(median, 5),
            "contrast": round(float(np.percentile(image, 99.5) - median), 5),
            "max": round(float(np.nanmax(image)), 5),
        }

    if context:
        summary["context"] = context

    (directory / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    return summary
