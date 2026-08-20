"""Finalise a session: freeze the mask and extract its traces in one step."""

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
    detrend: bool = True,
) -> dict:
    """Write the mask and its traces together, linked by the mask's hash."""

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
            manipulation=session.manipulation,
            mcor_paths=session.paths,
            acq_ids=session.acq_ids,
            frame_rate=session.frame_rate,
            pre_s=pre_s, post_s=post_s, neuropil=neuropil,
            checkpoint_dir=checkpoint, mask_hash=digest,
        )

    detrend_result = None
    if extract and traces is not None and detrend:
        from .detrend import detrend_traces

        corrected, info = detrend_traces(
            traces.roi,
            odor_on_frames=traces.odor_on_frames,
            odor_off_frames=traces.odor_off_frames,
            frame_rate=traces.frame_rate,
        )
        if info.get("ok"):
            info["traces"] = corrected
            detrend_result = info

    _write_round(
        path, scratch,
        labels=labels, traces=traces, per_group_masks=per_group_masks,
        exp_name=session.exp_name, group_id=session.group_id,
        mask_hash=digest, parameters=parameters, detrend=detrend_result,
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
        "detrend": None if detrend_result is None else {
            k: v for k, v in detrend_result.items()
            if k not in ("traces", "a_fast", "a_slow", "fit")
        },
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
    """Do the traces in this folder belong to the mask beside them?"""

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
