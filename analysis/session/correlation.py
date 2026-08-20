"""
Local correlation over a concatenated set of trial-averaged z-score movies.

The segmentation image is built from one movie per odor *and condition* --
32 blocks for a 16-odor pre/post-ket session -- concatenated along time, with
a single local-correlation map computed over the whole thing.

Two reasons this beats computing a map per block and combining them:

  - Splitting awake from anesthetized before averaging keeps responses that
    differ between states from cancelling each other. A glomerulus excited
    awake and suppressed under ket/xyl averages toward nothing if the two are
    pooled.
  - Concatenating rather than combining removes the choice of combination
    rule. Taking a max across per-block maps inflates values purely by
    selecting the largest of N noisy estimates, which made an earlier
    raw-vs-z comparison not directly comparable. One map over one long series
    has no such knob.

The concatenation is never materialised. A Pearson correlation needs only
per-pixel sums, sums of squares, and sums of neighbour products, all of which
accumulate additively across blocks -- so memory is a handful of images
regardless of how many blocks there are or how long they run.

Each block is centred on its own mean before accumulating. Without that, a
pixel that sits high in one block and low in another contributes a large
between-block deviation to every pairwise product, and neighbouring pixels
would correlate because they share the block structure rather than because
they co-vary in time.
"""

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
    """
    Local 8-neighbour correlation over blocks concatenated along time.

    `blocks` yields (T, H, W) arrays, which may be memmaps. They are consumed
    one at a time and never held together.
    """

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
    """
    One averaged z-score movie per condition, streamed from an odyn group.

    Bridges `odyn.groups.Group` (the `avg-last-z-scores` branch) to
    `concatenated_local_correlation`, so the segmentation image can be built
    straight off `db.groups[...]` rather than off a resolved session.

    Yields rather than returning a dict. `Group.z_score_average_movies` builds
    every condition before returning any, which the method's own docstring
    flags as too RAM-intensive for large experiments -- a 16-odor pre/post
    session at 600x500x500 frames is ~19 GB held at once. Streaming one
    condition at a time costs nothing here, because
    `concatenated_local_correlation` consumes blocks one at a time and
    accumulates only per-pixel sums, so peak memory is one movie regardless of
    how many conditions there are.

    Averaging is `sum / sqrt(n)`, matching `z_score_average_movies` rather
    than the plain mean in `zscore.py`. That normalisation matters here and
    not elsewhere: the correlation is computed over every condition
    concatenated, so a block's weight follows its variance. Under a plain mean
    a condition with more trials has *less* noise and therefore *less* weight,
    which down-weights the best-measured conditions. Under sum/sqrt(n) the
    noise sits near 1 for every condition and signal grows with sqrt(n), so a
    condition contributes in proportion to how well it was measured.

    Blocks are one per `(program_id, odor_id)`. `program_id` is what separates
    pre from post anaesthesia -- both programs carry the same `program_name`
    ("16odors passive 4s scope"), so the id is the discriminator and the name
    is not. That gives the same 2 x n_odor blocks `zscore.py` builds.

    `z_score_average_movies` also keys on `outcome`, which is a behavioural
    field: it reads 'na' throughout these passive sessions, so grouping on it
    adds a constant and changes nothing. It is left out of the default rather
    than trusted to stay constant. Pass `by=(..., "outcome")` on a session
    where it means something.

    `keys` restricts which conditions are used, as tuples matching `by`.
    Leave it None for all of them.
    """

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
