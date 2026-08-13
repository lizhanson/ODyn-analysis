"""
Fixed-footprint trace extraction: the actual output of Stage 1.

One mask, applied unchanged to every acquisition. Detection happened once on
the pooled data, so the same pixels are read in both blocks -- which is what
makes a within-unit pre/post comparison mean anything. Re-detecting per block
would give two ROI sets that have to be matched afterwards, and any matching
error would land squarely on the effect being measured.

Output is `(n_rois, n_trials, n_frames)` covering the **whole acquisition** --
the entire pre-odor period, the odor, and everything after it (about 5 s, 4 s
and 11 s on a typical session). Nothing is cropped.

Windowing here would be a decision made too early. A window has to be chosen
before anyone has seen a trace, and it silently forecloses questions later:
slow post-odor dynamics, drift across the baseline, whether the response has
returned by the end of the trial. The whole acquisition is cheap -- 137 ROIs x
180 trials x 301 frames is 30 MB -- so there is nothing to buy by cutting it.

Odor onset varies by a frame or two between trials, so `odor_on_frame` is
recorded per trial and alignment is a downstream choice rather than something
baked into the array. `time_s` is provided against the median onset for
convenience; exact per-trial alignment uses the recorded onsets.

Raw fluorescence, deliberately. dF/F and baseline-z are Stage 3 decisions --
this stage should not bake in a normalisation that a later step then has to
undo or work around.

Neuropil is measured but not subtracted. The annulus around each ROI -- a 3 px
gap then a 12 px ring, with every other ROI's pixels removed -- is returned
alongside the ROI trace so a correction can be applied later with a coefficient
chosen against the data, rather than a factor fixed here that would be
invisible downstream.

Treat it as diagnostic at 10x. The correction assumes the surround is
out-of-focus neuropil, which holds for 20x somata; on a dense glomerular field
the rings instead fill the inter-glomerular space (median 671 px of ring per
771 px of ROI on exp 132), which is a different thing. It is measured because
measuring is cheap and discarding it later is easy; applying it here would not
be.

Output is written as a file plus two tables, which is the shape the database
will eventually take: `rois` and `trials` are rows, and the trace array is too
large for SQLite so it is referenced by path, exactly as `outputs` already
does for other artefacts.
"""

from __future__ import annotations

import json

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile


@dataclass
class Traces:
    """Extracted traces plus everything needed to interpret them."""

    # (n_rois, n_trials, n_frames), raw fluorescence
    roi: np.ndarray
    # (n_rois, n_trials, n_frames), surrounding annulus; None when disabled
    neuropil: None | np.ndarray

    roi_ids: np.ndarray          # label value in the mask, per row of `roi`
    roi_areas_px: np.ndarray
    neuropil_areas_px: None | np.ndarray

    trial_ids: np.ndarray        # aligns to the trial table
    odor_ids: np.ndarray
    states: np.ndarray

    frame_rate: float
    n_pre: int                   # median frames before odor onset
    n_odor: int
    n_post: int

    # Per-trial onset in frames, since it varies by a frame or two. Exact
    # alignment uses these; `time_s` uses their median.
    odor_on_frames: np.ndarray
    odor_off_frames: np.ndarray

    skipped: list[dict]

    @property
    def n_rois(self) -> int:
        return self.roi.shape[0]

    @property
    def time_s(self) -> np.ndarray:
        """Seconds relative to odor onset, one per frame of the window."""
        return (np.arange(self.roi.shape[2]) - self.n_pre) / self.frame_rate

    def summary(self) -> dict:
        return {
            "n_rois": int(self.n_rois),
            "n_trials": int(self.roi.shape[1]),
            "n_frames": int(self.roi.shape[2]),
            "frame_rate": round(float(self.frame_rate), 3),
            "window_s": [round(float(self.time_s[0]), 3), round(float(self.time_s[-1]), 3)],
            "n_pre": int(self.n_pre), "n_odor": int(self.n_odor), "n_post": int(self.n_post),
            "full_acquisition": bool(self.roi.shape[2] > self.n_pre + self.n_odor + self.n_post - 2),
            "roi_area_px": [int(self.roi_areas_px.min()), int(self.roi_areas_px.max())],
            "has_neuropil": self.neuropil is not None,
            "n_skipped": len(self.skipped),
        }

    def roi_table(self, labels: None | np.ndarray = None):
        """
        One row per ROI. Rows, not arrays -- this is what goes in a database.

        `labels` adds centroids, which need the mask and are worth having for
        matching ROIs between sessions later.
        """

        import pandas as pd

        rows = {
            "roi_id": self.roi_ids,
            "area_px": self.roi_areas_px,
            "diameter_px": np.round(2 * np.sqrt(self.roi_areas_px / np.pi), 1),
        }

        if self.neuropil_areas_px is not None:
            rows["neuropil_area_px"] = self.neuropil_areas_px

        if labels is not None:
            centroids = []
            for roi_id in self.roi_ids:
                ys, xs = np.nonzero(labels == roi_id)
                centroids.append((ys.mean(), xs.mean()) if ys.size else (np.nan, np.nan))
            rows["centroid_y"] = np.round([c[0] for c in centroids], 1)
            rows["centroid_x"] = np.round([c[1] for c in centroids], 1)

        # Per-ROI summary over the whole session, so a bad ROI is visible in
        # the table without loading the array.
        #
        # `n_trials_finite` used to live here and was dropped: trials are
        # skipped whole, so it was identical for every ROI and duplicated the
        # trials table's `extracted` column. These three genuinely vary per ROI.
        pre = self.roi[:, :, : self.n_pre]
        baseline = np.nanmean(pre, axis=(1, 2))
        rows["baseline_fluorescence"] = np.round(baseline, 2)
        rows["baseline_sd"] = np.round(np.nanstd(pre, axis=(1, 2)), 3)
        rows["baseline_snr"] = np.round(
            baseline / np.maximum(np.nanstd(pre, axis=(1, 2)), 1e-9), 2
        )

        return pd.DataFrame(rows)

    def trial_table(self):
        """One row per trial, aligned to the second axis of `roi`."""

        import pandas as pd

        return pd.DataFrame({
            "trial_index": np.arange(len(self.trial_ids)),
            "trial_id": self.trial_ids,
            "odor_id": self.odor_ids,
            "state": self.states,
            "odor_on_frame": self.odor_on_frames,
            "odor_off_frame": self.odor_off_frames,
            "extracted": np.isfinite(self.roi[0, :, 0]),
        })

    def save(self, path: str | Path, *, labels: None | np.ndarray = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        arrays = {
            "roi": self.roi,
            "roi_ids": self.roi_ids,
            "roi_areas_px": self.roi_areas_px,
            "trial_ids": self.trial_ids,
            "odor_ids": self.odor_ids,
            "states": self.states.astype("U32"),
            "time_s": self.time_s,
        }
        if self.neuropil is not None:
            arrays["neuropil"] = self.neuropil
            arrays["neuropil_areas_px"] = self.neuropil_areas_px

        np.savez_compressed(path, **arrays)
        path.with_suffix(".json").write_text(
            json.dumps(self.summary() | {"skipped": self.skipped}, indent=2, default=str)
        )

        # Tables beside the array, in the shape the database will want.
        self.roi_table(labels).to_csv(path.with_name(path.stem + "_rois.csv"), index=False)
        self.trial_table().to_csv(path.with_name(path.stem + "_trials.csv"), index=False)

        return path


def neuropil_rings(
    labels: np.ndarray, *, inner_px: int = 3, outer_px: int = 12
) -> np.ndarray:
    """
    An annulus around each ROI, excluding every ROI's pixels.

    `inner_px` leaves a gap so the ring does not sample the ROI's own blurred
    edge; `outer_px` sets its width. Pixels belonging to any ROI are removed
    from every ring, so a neighbouring glomerulus never contributes to another's
    background estimate.
    """

    from scipy.ndimage import binary_dilation

    rings = np.zeros_like(labels)
    occupied = labels > 0

    for roi_id in range(1, int(labels.max()) + 1):
        this = labels == roi_id
        if not this.any():
            continue

        inner = binary_dilation(this, iterations=inner_px)
        outer = binary_dilation(inner, iterations=outer_px - inner_px)

        ring = outer & ~inner & ~occupied & (rings == 0)
        rings[ring] = roi_id

    return rings


def _weight_matrix(labels: np.ndarray, roi_ids: np.ndarray):
    """
    Sparse (n_rois, n_pixels) matrix whose rows average an ROI's pixels.

    A matrix product over the flattened movie extracts every ROI for every
    frame in one operation, rather than looping ROIs per frame -- which at 64
    ROIs x 175 frames x 180 trials is the difference between seconds and
    minutes.
    """

    from scipy.sparse import csr_matrix

    flat = labels.ravel()
    order = {roi_id: row for row, roi_id in enumerate(roi_ids)}

    pixels = np.flatnonzero(flat > 0)
    rows = np.array([order[flat[p]] for p in pixels], dtype=np.int64)

    areas = np.bincount(rows, minlength=len(roi_ids)).astype(np.float64)
    weights = 1.0 / np.maximum(areas[rows], 1)

    matrix = csr_matrix(
        (weights, (rows, pixels)), shape=(len(roi_ids), flat.size), dtype=np.float32
    )

    return matrix, areas


def extract_traces(
    movie_paths: list[str | Path],
    labels: np.ndarray,
    *,
    odor_on_frames: list[int],
    odor_off_frames: list[int],
    trial_ids: list[int],
    odor_ids: list[int],
    states: list[str],
    frame_rate: float,
    full_acquisition: bool = True,
    pre_s: float = 2.0,
    post_s: float = 2.0,
    neuropil: bool = True,
    neuropil_inner_px: int = 3,
    neuropil_outer_px: int = 12,
    checkpoint_dir: None | str | Path = None,
    mask_hash: str = "",
    progress: bool = True,
) -> Traces:
    """
    Stream the session once, applying one mask to every acquisition.

    `full_acquisition=True` (the default) keeps every frame. Set it False and
    `pre_s`/`post_s` cut a window anchored on each trial's own onset -- only
    worth doing when memory genuinely forces it, since cropping here discards
    data that cannot be recovered without re-streaming the session.

    `checkpoint_dir` makes the run resumable: each trial is written to a local
    memory-mapped file as it completes, and a re-run skips what is already
    there. On a share that drops -- which this one does, mid-migration -- that
    is the difference between losing an hour and losing a minute. Pass
    `mask_hash` with it so a changed mask invalidates the checkpoint instead of
    stitching together trials measured from different ROIs.
    """

    if labels.max() < 1:
        raise ValueError("The mask contains no ROIs.")

    lengths = {len(movie_paths), len(odor_on_frames), len(odor_off_frames),
               len(trial_ids), len(odor_ids), len(states)}
    if len(lengths) != 1:
        raise ValueError(f"Per-trial inputs have differing lengths: {lengths}.")

    roi_ids = np.array(
        [i for i in range(1, int(labels.max()) + 1) if (labels == i).any()], dtype=np.int64
    )

    roi_matrix, roi_areas = _weight_matrix(labels, roi_ids)

    if neuropil:
        rings = neuropil_rings(
            labels, inner_px=neuropil_inner_px, outer_px=neuropil_outer_px
        )
        ring_matrix, ring_areas = _weight_matrix(rings, roi_ids)
    else:
        ring_matrix = ring_areas = None

    n_odor = int(round(np.median([b - a for a, b in zip(odor_on_frames, odor_off_frames)])))

    if full_acquisition:
        # Every frame of every acquisition. They share a frame count, so the
        # result is rectangular without padding.
        with tifffile.TiffFile(str(movie_paths[0])) as tif:
            n_span = int(tif.series[0].shape[0])
        starts = [0] * len(movie_paths)
        n_pre = int(round(np.median(odor_on_frames)))
        n_post = n_span - n_pre - n_odor
    else:
        n_pre = int(round(pre_s * frame_rate))
        n_post = int(round(post_s * frame_rate))
        n_span = n_pre + n_odor + n_post
        starts = [on - n_pre for on in odor_on_frames]

    shape = (len(roi_ids), len(movie_paths), n_span)
    skipped: list[dict] = []

    from .checkpoint import ExtractionCheckpoint, checkpoint_key
    from .zscore import _tracked

    store = None
    if checkpoint_dir is not None:
        store = ExtractionCheckpoint(
            checkpoint_dir,
            digest=checkpoint_key(
                movie_paths, mask_hash=mask_hash, shape=shape,
                starts=starts, neuropil=neuropil,
            ),
            shape=shape, neuropil=neuropil,
        )
        todo = list(store.pending())
        if store.resumed:
            print(f"resuming extraction: {store.n_done}/{shape[1]} trials already done")
    else:
        roi_out = np.full(shape, np.nan, dtype=np.float32)
        ring_out = np.full_like(roi_out, np.nan) if neuropil else None
        todo = list(range(len(movie_paths)))

    for index in _tracked(
        todo, total=len(todo), description="extracting traces", enabled=progress
    ):
        path, start = movie_paths[index], starts[index]
        stop = start + n_span

        with tifffile.TiffFile(str(path)) as tif:
            total = tif.series[0].shape[0]

            if start < 0 or stop > total:
                skipped.append({
                    "trial": int(trial_ids[index]), "file": Path(path).name,
                    "start": int(start), "stop": int(stop), "total": int(total),
                })
                if store is not None:
                    # Complete, not pending: a window that does not fit will
                    # never fit, and leaving it pending would make every resume
                    # retry it forever.
                    store.mark_skipped(index)
                    store.flush()
                continue

            stack = tif.series[0].asarray(key=slice(start, stop)).astype(np.float32)

        flat = stack.reshape(stack.shape[0], -1)

        # (n_rois, n_pixels) @ (n_pixels, n_frames) -> (n_rois, n_frames)
        roi_values = roi_matrix @ flat.T
        ring_values = (ring_matrix @ flat.T) if neuropil else None

        if store is not None:
            store.store(index, roi_values, ring_values)
            store.flush()
        else:
            roi_out[:, index, :] = roi_values
            if neuropil:
                ring_out[:, index, :] = ring_values

        del stack, flat

    if store is not None:
        roi_out, ring_out = store.arrays()

    return Traces(
        roi=roi_out,
        neuropil=ring_out,
        roi_ids=roi_ids,
        roi_areas_px=roi_areas.astype(np.int64),
        neuropil_areas_px=None if ring_areas is None else ring_areas.astype(np.int64),
        trial_ids=np.asarray(trial_ids),
        odor_ids=np.asarray(odor_ids),
        states=np.asarray(states, dtype=object),
        frame_rate=float(frame_rate),
        n_pre=n_pre, n_odor=n_odor, n_post=n_post,
        odor_on_frames=np.asarray(odor_on_frames),
        odor_off_frames=np.asarray(odor_off_frames),
        skipped=skipped,
    )
