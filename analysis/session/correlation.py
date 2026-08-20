"""Local correlation over a concatenated set of trial-averaged z-score movies."""

from __future__ import annotations

from typing import Iterable

import numpy as np

# Offsets to the 4 unique neighbour pairs; each is counted from both sides,
# which covers all 8 neighbours without computing any pair twice.
NEIGHBOUR_OFFSETS = ((0, 1), (1, 0), (1, 1), (1, -1))


def _valid_mask(shape: tuple[int, int], dy: int, dx: int) -> np.ndarray:
    """False where a shifted array wrapped around the edge."""
    valid = np.ones(shape, dtype=bool)
    if dy:
        valid[:dy, :] = False
    if dx > 0:
        valid[:, :dx] = False
    elif dx < 0:
        valid[:, dx:] = False
    return valid


def concatenated_local_correlation(
    blocks: Iterable[np.ndarray],
    *,
    center_each_block: bool = True,
) -> tuple[np.ndarray, dict]:
    """Local 8-neighbour correlation over blocks concatenated along time."""

    sum_x = None
    sum_xx = None
    sum_xy: dict[tuple[int, int], np.ndarray] = {}
    n_frames = 0
    n_blocks = 0

    for block in blocks:
        block = np.asarray(block, dtype=np.float32)

        if block.ndim != 3:
            raise ValueError(f"Expected (T, H, W) blocks, got shape {block.shape}.")

        if sum_x is None:
            shape = block.shape[1:]
            sum_x = np.zeros(shape, dtype=np.float64)
            sum_xx = np.zeros(shape, dtype=np.float64)
            sum_xy = {
                offset: np.zeros(shape, dtype=np.float64)
                for offset in NEIGHBOUR_OFFSETS
            }

        elif block.shape[1:] != sum_x.shape:
            raise ValueError(
                f"Block shape {block.shape[1:]} does not match {sum_x.shape}."
            )

        if center_each_block:
            block = block - block.mean(axis=0)

        sum_x += block.sum(axis=0, dtype=np.float64)
        sum_xx += (block.astype(np.float64) ** 2).sum(axis=0)

        for dy, dx in NEIGHBOUR_OFFSETS:
            shifted = np.roll(np.roll(block, dy, axis=1), dx, axis=2)
            sum_xy[(dy, dx)] += (block.astype(np.float64) * shifted).sum(axis=0)

        n_frames += block.shape[0]
        n_blocks += 1

        del block

    if sum_x is None:
        raise ValueError("No blocks given.")

    mean = sum_x / n_frames
    var = sum_xx / n_frames - mean**2

    # A pixel with no variance (dead, saturated, or masked out) would divide by
    # zero; leave it at 0 correlation rather than NaN.
    sd = np.sqrt(np.maximum(var, 0))
    safe = np.where(sd > 0, sd, np.inf)

    total = np.zeros(sum_x.shape, dtype=np.float64)
    count = np.zeros(sum_x.shape, dtype=np.float64)

    for (dy, dx), cross in sum_xy.items():
        shifted_mean = np.roll(np.roll(mean, dy, axis=0), dx, axis=1)
        shifted_safe = np.roll(np.roll(safe, dy, axis=0), dx, axis=1)

        corr = (cross / n_frames - mean * shifted_mean) / (safe * shifted_safe)

        valid = _valid_mask(sum_x.shape, dy, dx)
        total += np.where(valid, corr, 0)
        count += valid

        # The same pair seen from the neighbour's side.
        total += np.where(
            np.roll(np.roll(valid, -dy, axis=0), -dx, axis=1),
            np.roll(np.roll(corr, -dy, axis=0), -dx, axis=1),
            0,
        )
        count += np.roll(np.roll(valid, -dy, axis=0), -dx, axis=1)

    correlation = np.where(count > 0, total / np.maximum(count, 1), 0)

    meta = {
        "n_blocks": n_blocks,
        "n_frames": n_frames,
        "centered_each_block": center_each_block,
    }

    return correlation.astype(np.float32), meta


def group_zscore_blocks(
    group,
    *,
    photobleach_window_s: float = 1.0,
    by: tuple[str, ...] = ("program_id", "odor_id"),
    keys=None,
    progress: bool = True,
):
    """One averaged z-score movie per condition, streamed from an odyn group."""

    trials = group.trials[
        group.trials["acq_id"].isin(group.approved_mcor_files.index)
    ]

    missing = [c for c in by if c not in trials.columns]
    if missing:
        raise KeyError(
            f"{group!r} trials table has no column(s) {missing}. "
            f"Available: {sorted(trials.columns)}"
        )

    # dropna=False so a null key cannot silently discard whole conditions --
    # the columns are NOT NULL in the schema, but the frame may be a join.
    grouped = list(trials.groupby(list(by), dropna=False))

    if keys is not None:
        wanted = {tuple(k) for k in keys}
        grouped = [(k, rows) for k, rows in grouped if tuple(k) in wanted]

    if not grouped:
        raise RuntimeError(
            f"{group!r} yielded no conditions with approved mcor files. "
            f"Run approve_mcor_files(), or check `keys`."
        )

    iterator = _tracked(grouped, total=len(grouped),
                        description="z-scoring conditions", enabled=progress)

    for key, rows in iterator:
        acq_ids = rows["acq_id"].astype(int).unique()
        total = None

        for acq_id in acq_ids:
            z = group.z_score_acquisition(
                acq_id=int(acq_id), photobleach_window_s=photobleach_window_s
            )
            total = z if total is None else total + z

        yield total / np.sqrt(len(acq_ids))


def _tracked(iterable, *, total: int, description: str, enabled: bool = True):
    """Progress bar if tqdm is available, the bare iterable otherwise."""

    if not enabled:
        return iterable

    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable

    return tqdm(iterable, total=total, desc=description, unit="cond", leave=False)
