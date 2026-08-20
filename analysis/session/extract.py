"""Fixed-footprint trace extraction: the actual output of Stage 1."""

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

    manipulations: np.ndarray | None = None
    # Motion-corrected file each trial came from. Carried so QC can name the
    # acquisition to re-examine rather than printing a trial index that then
    # has to be traced back through the database by hand.
    mcor_paths: np.ndarray | None = None
    acq_ids: np.ndarray | None = None

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
        """One row per ROI."""

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
            **({} if self.manipulations is None
               else {"manipulation": self.manipulations}),
            **({} if self.mcor_paths is None
               else {"mcor_path": self.mcor_paths}),
            **({} if self.acq_ids is None
               else {"acq_id": self.acq_ids}),
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
            **({} if self.manipulations is None
               else {"manipulations": self.manipulations.astype("U32")}),
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
    """An annulus around each ROI, excluding every ROI's pixels."""

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
    """Sparse (n_rois, n_pixels) matrix whose rows average an ROI's pixels."""

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
    manipulation: str | list[str] | None = None,
    mcor_paths: list | None = None,
    acq_ids: list | None = None,
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
    """Stream the session once, applying one mask to every acquisition."""

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
        manipulations=None if manipulation is None else np.asarray(
            manipulation if isinstance(manipulation, (list, tuple, np.ndarray))
            else [manipulation] * len(states), dtype=object,
        ),
        frame_rate=float(frame_rate),
        n_pre=n_pre, n_odor=n_odor, n_post=n_post,
        odor_on_frames=np.asarray(odor_on_frames),
        odor_off_frames=np.asarray(odor_off_frames),
        skipped=skipped,
    )
