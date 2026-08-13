"""
Streaming summary images for segmentation.

A pre/post-ket session is 160-224 acquisitions of ~0.6-3.3 GB each, held on an
SMB share that reads at ~5 MB/s. Reading one is unavoidable; reading it twice
is not affordable. So every summary a segmenter might want is accumulated in
a single pass, and the results -- a handful of 2-D images -- are cached.

The images that come out:

    mean             average fluorescence; anatomy, including silent structure
    max              max projection; catches transients the mean washes out
    correlation      local (8-neighbour) correlation; the classic cell-finding
                     image, high where neighbouring pixels co-vary
    activity         |strongest odor response| per pixel -- the detection map
    activity_signed  the same, keeping sign: positive excited, negative
                     suppressed
    response_by_odor per-odor signed response maps, one per odor_id

`activity` is deliberately two-sided. The z-score is computed per trial
against that trial's own pre-odor baseline, averaged *within odor* across
trials so noise cancels, and only then reduced across odors by taking each
pixel's largest absolute response. A unit that is only ever suppressed
produces a strong negative response and is exactly as findable as one that is
only ever excited -- which an earlier `max()`-based version was not, since a
purely inhibited unit peaked near zero.

Detection is therefore biased toward *responsive* units, excited or
suppressed, and away from units that never modulate. That is intended. What
it must not do is bias toward excitation specifically, because the
excited/suppressed balance analysis would then be measuring its own detection
step. `mean` and `correlation` remain available for detecting structurally
present but unmodulated units if that is ever wanted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile


@dataclass
class SummaryImages:
    """Per-pixel summaries accumulated over a session."""

    mean: np.ndarray
    max: np.ndarray
    correlation: np.ndarray
    activity: np.ndarray
    activity_signed: np.ndarray
    response_by_odor: dict[int, np.ndarray]
    n_acquisitions: int
    n_frames: int

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            path,
            mean=self.mean,
            max=self.max,
            correlation=self.correlation,
            activity=self.activity,
            activity_signed=self.activity_signed,
            n_acquisitions=self.n_acquisitions,
            n_frames=self.n_frames,
            odor_ids=np.array(sorted(self.response_by_odor)),
            response_stack=np.stack(
                [self.response_by_odor[k] for k in sorted(self.response_by_odor)]
            ),
        )

        return path

    @classmethod
    def load(cls, path: str | Path) -> SummaryImages:
        with np.load(path) as data:
            return cls(
                mean=data["mean"],
                max=data["max"],
                correlation=data["correlation"],
                activity=data["activity"],
                activity_signed=data["activity_signed"],
                response_by_odor={
                    int(k): v
                    for k, v in zip(data["odor_ids"], data["response_stack"])
                },
                n_acquisitions=int(data["n_acquisitions"]),
                n_frames=int(data["n_frames"]),
            )


def local_correlation(movie: np.ndarray) -> np.ndarray:
    """
    Correlation of each pixel's time course with its 8 neighbours.

    Equivalent in spirit to CaImAn's `local_correlations`, written out here so
    the summary pass does not pull in the full CaImAn import for one image.
    Pixels with no variance (dead or saturated) return 0 rather than NaN.
    """

    movie = movie.astype(np.float32, copy=False)

    centered = movie - movie.mean(axis=0)
    sigma = np.sqrt((centered**2).mean(axis=0))

    # Guard flat pixels: leave them at 0 correlation instead of dividing by 0.
    safe = np.where(sigma > 0, sigma, np.inf)
    normed = centered / safe

    total = np.zeros(movie.shape[1:], dtype=np.float32)
    count = np.zeros(movie.shape[1:], dtype=np.float32)

    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        shifted = np.roll(np.roll(normed, dy, axis=1), dx, axis=2)
        product = (normed * shifted).mean(axis=0)

        # np.roll wraps; blank the wrapped edge so borders are not fabricated.
        valid = np.ones(movie.shape[1:], dtype=bool)
        if dy:
            valid[:dy, :] = False
        if dx > 0:
            valid[:, :dx] = False
        elif dx < 0:
            valid[:, dx:] = False

        total += np.where(valid, product, 0)
        count += valid

        # Each neighbour pair is counted once from each side.
        back = np.roll(np.roll(product, -dy, axis=0), -dx, axis=1)
        back_valid = np.roll(np.roll(valid, -dy, axis=0), -dx, axis=1)
        total += np.where(back_valid, back, 0)
        count += back_valid

    return np.where(count > 0, total / np.maximum(count, 1), 0).astype(np.float32)


def trial_zscore_response(
    movie: np.ndarray,
    *,
    odor_on_frame: int,
    odor_off_frame: int,
    baseline_end_frame: None | int = None,
) -> np.ndarray:
    """
    Mean odor-window z-score per pixel for one trial, sign preserved.

    Baseline is the pre-odor portion of the same trial, so slow drift and
    bleaching cancel.

    The mean over the window is used rather than the peak, for two reasons.
    A peak taken as `max` is blind to suppression -- an inhibited unit has a
    negative deflection and would score near zero. And `max(|z|)` on a single
    trial is positively biased even for pure noise, because it selects the
    most extreme of ~100 noisy frames; averaging that across trials reduces
    its variance but not its bias. A signed mean has neither problem: noise
    averages toward zero, and excitation and suppression come out with
    opposite sign and comparable sensitivity.
    """

    if baseline_end_frame is None:
        baseline_end_frame = odor_on_frame

    if baseline_end_frame < 2:
        raise ValueError(
            f"Need at least 2 baseline frames, got {baseline_end_frame}."
        )

    baseline = movie[:baseline_end_frame]
    mu = baseline.mean(axis=0)
    sd = baseline.std(axis=0)

    safe = np.where(sd > 0, sd, np.inf)
    window = movie[odor_on_frame:odor_off_frame]

    if window.size == 0:
        return np.zeros(movie.shape[1:], dtype=np.float32)

    return (((window - mu) / safe).mean(axis=0)).astype(np.float32)


def build_summary_images(
    movie_paths: list[str | Path],
    *,
    odor_on_frames: list[int],
    odor_off_frames: list[int],
    odor_ids: None | list[int] = None,
) -> SummaryImages:
    """
    One streaming pass over a session's motion-corrected acquisitions.

    `odor_on_frames`/`odor_off_frames` give the odor window per acquisition, in
    frames relative to that acquisition's first frame. They are computed from
    the sync frame clock rather than assumed, because baseline length varies
    between sessions and frame rate varies between 12.9 and 47.7 Hz.

    Memory stays at roughly one acquisition; nothing is concatenated.
    """

    if not movie_paths:
        raise ValueError("No movie paths given.")

    if not (len(movie_paths) == len(odor_on_frames) == len(odor_off_frames)):
        raise ValueError(
            f"Length mismatch: {len(movie_paths)} movies, "
            f"{len(odor_on_frames)} onsets, {len(odor_off_frames)} offsets."
        )

    if odor_ids is None:
        odor_ids = [0] * len(movie_paths)

    total = None
    running_max = None
    corr_sum = None
    n_frames = 0

    # Per-odor accumulators, because responses are odor-specific: a glomerulus
    # driven by 1 of 16 odors would be diluted 16-fold by pooling. Each is a
    # single 2-D map, so this costs n_odors * H * W * 4 bytes (~18 MB for a
    # 16-odor session at 526x550), not a movie per odor.
    response_sum: dict[int, np.ndarray] = {}
    response_n: dict[int, int] = {}

    for path, on_frame, off_frame, odor_id in zip(
        movie_paths, odor_on_frames, odor_off_frames, odor_ids
    ):
        movie = tifffile.imread(str(path)).astype(np.float32, copy=False)

        if total is None:
            shape = movie.shape[1:]
            total = np.zeros(shape, dtype=np.float64)
            running_max = np.full(shape, -np.inf, dtype=np.float32)
            corr_sum = np.zeros(shape, dtype=np.float32)

        elif movie.shape[1:] != total.shape:
            raise ValueError(
                f"Frame shape {movie.shape[1:]} in '{Path(path).name}' does not "
                f"match {total.shape} from the first acquisition."
            )

        total += movie.sum(axis=0, dtype=np.float64)
        np.maximum(running_max, movie.max(axis=0), out=running_max)
        corr_sum += local_correlation(movie)

        response = trial_zscore_response(
            movie, odor_on_frame=on_frame, odor_off_frame=off_frame
        )

        if odor_id not in response_sum:
            response_sum[odor_id] = np.zeros(movie.shape[1:], dtype=np.float32)
            response_n[odor_id] = 0

        response_sum[odor_id] += response
        response_n[odor_id] += 1

        n_frames += movie.shape[0]

        del movie

    n = len(movie_paths)

    # Trial-average within odor first (noise cancels), then take each pixel's
    # strongest response across odors, keeping its sign. Detection uses the
    # magnitude, so a unit that is only ever suppressed is as findable as one
    # that is only ever excited.
    signed = {
        odor_id: response_sum[odor_id] / response_n[odor_id] for odor_id in response_sum
    }

    stacked = np.stack([signed[k] for k in sorted(signed)], axis=0)
    best = np.argmax(np.abs(stacked), axis=0)

    activity_signed = np.take_along_axis(stacked, best[None], axis=0)[0]

    return SummaryImages(
        mean=(total / n_frames).astype(np.float32),
        max=running_max,
        correlation=(corr_sum / n).astype(np.float32),
        activity=np.abs(activity_signed).astype(np.float32),
        activity_signed=activity_signed.astype(np.float32),
        response_by_odor={int(k): v.astype(np.float32) for k, v in signed.items()},
        n_acquisitions=n,
        n_frames=n_frames,
    )
