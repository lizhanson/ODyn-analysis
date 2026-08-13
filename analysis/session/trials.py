"""
Build the per-session trial table and label its two blocks.

A pre/post session is one `loop` experiment of 160-224 acquisitions, one
acquisition per trial (~20 s of imaging every ~30 s). The olfactometer runs it
as *two programs*: a first block, then a manipulation during a gap with the
scope idle, then a second block.

Block therefore comes from `program_id`, not from splitting trials at the
median trial number. The distinction matters because the two blocks are not
guaranteed to be equal in length, and because there is no imaging during the
gap at all -- so there is no "injection frame", only a block boundary.

What the manipulation was is *not* in the database; it lives in the session
manifest. It is passed to `trial_table` and recorded per row, defaulting to
ketamine/xylazine.

Some sessions were split into two `exp_id`s at ingest rather than two
programs (see m357 20260623, "PA split into e1 and e2 for db integration"),
so state assignment falls back to experiment order when a session has only
one program.
"""

from __future__ import annotations

import pandas as pd

# Block 0 is always before the manipulation, block 1 after. What the
# manipulation *was* is not in the database -- it lives in the session
# manifest -- so it is passed in rather than assumed.
#
# An earlier version hardcoded block 0 = "awake" and block 1 = "anesthetized".
# That is simply wrong for a saline control (exp 132 is pre/post saline: both
# blocks are awake), and nothing would have flagged it -- every downstream
# analysis reading `state` would have mislabelled the controls.
BEFORE = "pre"
AFTER = "post"

DEFAULT_MANIPULATION = "ketamine/xylazine"

# Whether the second block is anaesthetised is a fact about the experiment,
# not something to infer from a free-text label. The notebook passes it in
# explicitly; `manipulation` is whatever the person running the session wants
# to write down.
#
# The inference this replaced read `manipulation` against a list of accepted
# spellings, so 'ketxyl' scored as not-anaesthetic and silently reclassified an
# anaesthetised session as a control.


def trial_table(
    group,
    *,
    exp_id: int,
    manipulation: str = DEFAULT_MANIPULATION,
) -> pd.DataFrame:
    """
    One row per trial, ordered in time, with state and timing columns.

    `manipulation` names what happened between the two blocks -- 'saline',
    'ketamine', 'no injection', anything. It is recorded per row rather than
    inferred, because the database does not hold it.

    Columns added on top of the database's own:
        block               0 before the manipulation, 1 after
        state               'pre' | 'post'
        manipulation        the label passed in
        trial_in_block      1-based index within the block
        baseline_s          pre-odor imaging in that trial, in seconds
        odor_duration_s     seconds the valve was commanded open
        odor_minus_acq_start_s  diagnostic; mixes two clocks, see below
    """

    trials = group.trials
    trials = trials[trials.exp_id == exp_id].copy()

    if trials.empty:
        raise ValueError(f"No trials for exp_id={exp_id}.")

    acqs = group.acquisitions
    acqs = acqs[acqs.exp_id == exp_id][["acq_id", "acq_start"]]

    table = trials.merge(acqs, on="acq_id", how="left", suffixes=("", "_acq"))

    for column in ("trial_start", "odor_start", "odor_end", "acq_start"):
        table[column] = pd.to_datetime(table[column])

    table = table.sort_values("trial_start").reset_index(drop=True)

    # Odor duration and baseline come from the recorded timestamps rather than
    # from the protocol name: baseline length varies between sessions and must
    # never be assumed.
    table["odor_duration_s"] = (
        table.odor_end - table.odor_start
    ).dt.total_seconds()

    # `baseline_s` is measured within the olfactometer's own clock.
    #
    # The obvious formula, `odor_start - acq_start`, is wrong: `odor_start`
    # comes from the olfactometer CSV and `acq_start` from the ScanImage TIFF,
    # and those two computers' clocks differ by ~1.24 s -- a difference odyn
    # already measures per trial as `h5_to_trial_ms` but never applies. On
    # exp 202 the mixed formula gives 6.25 s where the true pre-odor period is
    # 5.01 s, a 35-frame error at 28 Hz.
    #
    # `trial_start` is on the same clock as `odor_start`, and the acquisition
    # is triggered by the olfactometer's trial-start pulse, so their difference
    # is the pre-odor imaging period to within a few ms (5.0105 s here against
    # 5.0077 s measured from the frame clock).
    table["baseline_s"] = (
        table.odor_start - table.trial_start
    ).dt.total_seconds()

    # Kept for diagnostics, named for what it is rather than what it looks
    # like: a cross-clock difference, not a duration. It should sit about
    # `h5_to_trial_ms` above `baseline_s`, and a session where it does not is
    # worth investigating before trusting its timing.
    table["odor_minus_acq_start_s"] = (
        table.odor_start - table.acq_start
    ).dt.total_seconds()

    table = _assign_state(table)
    table["manipulation"] = manipulation

    return table


def _assign_state(table: pd.DataFrame) -> pd.DataFrame:
    """Label each trial pre/post from its program's position in time."""

    order = (
        table.groupby("program_id").trial_start.min().sort_values().index.tolist()
    )

    if len(order) == 1:
        # Single-program session: the awake/ket split lives across two exp_ids
        # rather than two programs, so this table is one state only. Caller
        # decides which by pairing the two experiments.
        table["block"] = 0
        table["state"] = pd.NA

    elif len(order) == 2:
        block = {program_id: i for i, program_id in enumerate(order)}
        table["block"] = table.program_id.map(block)
        table["state"] = table.block.map({0: BEFORE, 1: AFTER})

    else:
        raise ValueError(
            f"Expected 1 or 2 programs, found {len(order)}: {order}. "
            "Block assignment assumes one block before the manipulation "
            "and one after."
        )

    table["trial_in_block"] = table.groupby("block").cumcount() + 1

    return table


def block_gap_s(table: pd.DataFrame) -> None | float:
    """
    Seconds between the last trial of block 0 and the first of block 1.

    The manipulation window, and the only handle on when it took effect. It
    varies widely -- 9.0 min on exp 132, 34.8 min on exp 202 -- because the
    second block starts when the animal is ready, not after a fixed wait.
    """

    if table.block.nunique() < 2:
        return None

    last_before = table.loc[table.block == 0, "trial_start"].max()
    first_after = table.loc[table.block == 1, "trial_start"].min()

    return (first_after - last_before).total_seconds()


def summarize(table: pd.DataFrame) -> dict:
    """JSON-serializable summary, suitable for `set_output` once recorded."""

    gap = block_gap_s(table)

    # The gap between these two is the ScanImage-to-olfactometer clock offset.
    # Surfacing it makes a session whose clocks were in step -- or wildly out
    # of it -- visible without going back to the sync file.
    clock_offset = float(
        (table.odor_minus_acq_start_s - table.baseline_s).median()
    )

    return {
        "n_trials": int(len(table)),
        "n_odors": int(table.odor_id.nunique()),
        "n_blocks": int(table.block.nunique()),
        "trials_per_block": {
            str(block): int(n) for block, n in table.block.value_counts().items()
        },
        "manipulation": (
            table.manipulation.iloc[0] if "manipulation" in table else None
        ),
        "block_gap_s": None if gap is None else round(float(gap), 1),
        "odor_duration_s": round(float(table.odor_duration_s.median()), 3),
        "baseline_s": round(float(table.baseline_s.median()), 3),
        "clock_offset_s": round(clock_offset, 3),
        "trials_missing_acq": int(table.acq_id.isna().sum()),
    }
