"""
Finalise a session: freeze the mask and extract its traces in one step.

Extraction takes no decisions of its own. Given a mask it reads the same pixels
from every acquisition and writes the result; there is no parameter to weigh,
nothing to look at and judge. Everything discretionary happened upstream, in
segmentation and curation.

So it is not a separate stage. Leaving it separate invites the failure it
cannot detect: a mask edited after extraction, and traces on disk that no
longer correspond to the ROIs anyone is looking at. Nothing in the files would
say so -- both would load, both would have plausible shapes, and the mismatch
would only surface as noise in an analysis much later.

`finalize_session` couples them and records the coupling. The mask's content
hash goes into the trace metadata, so `verify` can say whether a set of traces
belongs to the mask beside it.

It writes one self-describing HDF5 per round into `<exp>/processed/python/`:

    group132_20260317_m316_e1_processed_20260812.h5

The date makes rounds additive: re-curate and re-extract, and the previous
round is still there to compare against rather than overwritten. See
`store.py` for the layout.

Two side-products go beside it, sharing its stem and date:

    group132_20260317_m316_e1_masks_processed_20260812.mat
    group132_20260317_m316_e1_masks_processed_20260812.png

The `.mat` is the mask for MATLAB, without the transpose `h5read` imposes; the
PNG is the ROIs drawn over the image as the GUI drew them. Neither is a source
of truth -- both carry the same `mask_hash` as the round. See `masks.py`.
"""

from __future__ import annotations

import hashlib
import json

from datetime import date
from pathlib import Path

import numpy as np


def mask_hash(labels: np.ndarray) -> str:
    """Content hash of a label image, ignoring dtype."""
    return hashlib.sha256(
        np.ascontiguousarray(labels.astype(np.int32)).tobytes()
    ).hexdigest()[:16]


def finalize_session(
    session,
    labels: np.ndarray,
    *,
    per_group_masks: None | dict = None,
    images: None | dict | np.ndarray = None,
    segmentation_params: None | dict = None,
    merge_params: None | dict = None,
    curation: None | dict = None,
    pre_s: float = 2.0,
    post_s: float = 2.0,
    neuropil: bool = True,
    extract: bool = True,
    overlay_alpha: None | float = None,
    processed_on: None | str = None,
    scratch_dir: None | str | Path = None,
) -> dict:
    """
    Write the mask and its traces together, linked by the mask's hash.

    `extract=False` writes only the mask, for the case where the traces are
    being produced elsewhere -- on a cluster, say. The mask hash is still
    recorded, so whatever extracts later can be checked against it.

    `images` is the per-odor correlation maps (or a single background image).
    Given them, the round also gets a PNG of the ROIs drawn over the image the
    way the GUI drew them; without them the PNG is skipped, since a mask
    picture with nothing underneath says nothing about whether the ROIs are in
    the right places.
    """

    from .extract import extract_traces
    from .masks import (
        DEFAULT_OVERLAY_ALPHA, background_image, save_mask_overlay,
        save_masks_mat,
    )
    from .store import session_filename, write_session

    output = session.output_dir
    digest = mask_hash(labels)
    alpha = DEFAULT_OVERLAY_ALPHA if overlay_alpha is None else overlay_alpha

    parameters = {
        "segmentation": segmentation_params,
        "merge": merge_params,
        "curation": curation,
        "window": {"pre_s": pre_s, "post_s": post_s, "neuropil": neuropil},
        "session": session.summary(),
    }

    path = output / session_filename(
        group_id=session.group_id, exp_name=session.exp_name,
        processed_on=processed_on,
    )

    scratch = None if scratch_dir is None else Path(scratch_dir)
    checkpoint = None if scratch is None else scratch / f"extract_{session.group_id}"

    traces = None
    if extract:
        traces = extract_traces(
            session.paths, labels,
            odor_on_frames=session.odor_on_frames,
            odor_off_frames=session.odor_off_frames,
            trial_ids=[int(v) for v in session.table.trial_id],
            odor_ids=session.odor_ids,
            states=session.states,
            frame_rate=session.frame_rate,
            pre_s=pre_s, post_s=post_s, neuropil=neuropil,
            checkpoint_dir=checkpoint, mask_hash=digest,
        )

    # Write locally, then copy.
    #
    # Writing HDF5 straight to the share means a dropped mount leaves a
    # truncated file where a valid round should be -- and it would still open,
    # since HDF5 does not require a footer. A local write followed by a copy
    # makes the failure mode "no file" instead of "a plausible wrong file", and
    # the copy is one sequential 36 MB transfer rather than the scattered small
    # writes HDF5 does as it builds the file.
    _write_round(
        path, scratch,
        labels=labels, traces=traces, per_group_masks=per_group_masks,
        exp_name=session.exp_name, group_id=session.group_id,
        mask_hash=digest, parameters=parameters,
    )

    if checkpoint is not None and checkpoint.exists():
        # Only now is the result safe somewhere permanent.
        from .checkpoint import ExtractionCheckpoint

        for stale in checkpoint.glob("*"):
            stale.unlink(missing_ok=True)
        try:
            checkpoint.rmdir()
        except OSError:
            pass

    # Side-products: same stem, same date, so they stay grouped with the round
    # they describe rather than with each other.
    def named(kind: str, suffix: str) -> Path:
        return output / session_filename(
            group_id=session.group_id, exp_name=session.exp_name,
            kind=kind, suffix=suffix, processed_on=processed_on,
        )

    mat_path = save_masks_mat(
        named("masks", ".mat"), labels,
        per_group_masks=per_group_masks,
        exp_name=session.exp_name, group_id=session.group_id,
        mask_hash=digest,
        processed_on=processed_on or date.today().strftime("%Y%m%d"),
    )

    png_path = None
    if images is not None:
        png_path = save_mask_overlay(
            named("masks", ".png"), background_image(images), labels, alpha=alpha,
        )

    return {
        "path": str(path),
        "mask_mat": str(mat_path),
        "mask_png": None if png_path is None else str(png_path),
        "overlay_alpha": alpha,
        "mask_hash": digest,
        "n_rois": int(labels.max()),
        "traces": traces,
        "summary": None if traces is None else traces.summary(),
    }



def _write_round(destination: Path, scratch: None | Path, **kwargs) -> Path:
    """Write the round to local scratch and copy it, or write in place."""

    import shutil

    from .store import write_session

    if scratch is None:
        return write_session(destination, **kwargs)

    scratch.mkdir(parents=True, exist_ok=True)
    staged = scratch / destination.name
    write_session(staged, **kwargs)

    destination.parent.mkdir(parents=True, exist_ok=True)

    # Copy to a temporary name and rename, so an interrupted copy cannot be
    # mistaken for a finished round by `find_rounds`.
    partial = destination.with_suffix(destination.suffix + ".partial")
    shutil.copyfile(staged, partial)
    partial.replace(destination)
    staged.unlink(missing_ok=True)

    return destination


def verify(output_dir: str | Path) -> dict:
    """
    Do the traces in this folder belong to the mask beside them?

    The check that matters after a re-curation: masks and traces both load
    fine when they disagree, and nothing else would notice.
    """

    from .h5io import open_h5
    from .store import find_rounds

    rounds = find_rounds(output_dir)
    if not rounds:
        return {"status": "no rounds found", "rounds": []}

    latest = rounds[-1]

    with open_h5(latest) as f:
        labels = f["masks/labels"][:]
        recorded = f.attrs.get("mask_hash")
        has_traces = "traces" in f

    current = mask_hash(labels)

    report = {
        "file": latest.name,
        "rounds": [p.name for p in rounds],
        "has_traces": bool(has_traces),
        "mask_hash": current,
        "traces_built_from": recorded,
        "match": current == recorded,
        "n_rois": int(labels.max()),
    }

    if not has_traces:
        report["status"] = "mask only, no traces"
    elif report["match"]:
        report["status"] = "ok"
    else:
        report["status"] = "STALE: traces predate the mask"

    return report
