"""Build the per-session trial table and label its two blocks."""

from __future__ import annotations

import pandas as pd

BEFORE = "pre"
AFTER = "post"

DEFAULT_MANIPULATION = "ketamine/xylazine"



def trial_table(
    group,
    *,
    exp_id: int,
    manipulation: str = DEFAULT_MANIPULATION,
) -> pd.DataFrame:
    """One row per trial, ordered in time, with state and timing columns."""

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

    table["baseline_s"] = (
        table.odor_start - table.trial_start
    ).dt.total_seconds()

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
    """Seconds between the last trial of block 0 and the first of block 1."""

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
