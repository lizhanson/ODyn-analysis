"""Build the per-session trial table and label its two blocks."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
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


def trial_table_from_events(
    group,
    *,
    exp_id: int,
    exp_dir: str | Path,
    manipulation: str = DEFAULT_MANIPULATION,
) -> pd.DataFrame:
    """Recover missing DB trials from two olfactometer event files.

    This deliberately supports only the unambiguous two-block case.  It keeps
    the database acquisition IDs (and therefore existing mcor approvals), but
    refuses to proceed unless event counts, odor IDs, block timestamps, and
    acquisition timing all agree exactly enough to establish the pairing.
    """

    exp_dir = Path(exp_dir)
    event_paths = sorted(exp_dir.glob("**/*-Events.csv"))
    if len(event_paths) != 2:
        raise ValueError(
            f"No trials for exp_id={exp_id}; event recovery requires exactly "
            f"two *-Events.csv files under {exp_dir}, found {len(event_paths)}."
        )

    odor_blocks: list[list[int]] = []
    program_starts: list[pd.Timestamp] = []
    timestamp_pattern = re.compile(
        r"(\d{4}_\d{2}_\d{2})-(\d{2}_\d{2}_\d{2})-Events\.csv$"
    )

    for path in event_paths:
        match = timestamp_pattern.search(path.name)
        if match is None:
            raise ValueError(
                f"Cannot recover trials: event filename has no program timestamp: {path}"
            )
        program_starts.append(
            pd.to_datetime(f"{match.group(1)} {match.group(2)}", format="%Y_%m_%d %H_%M_%S")
        )

        odors: list[int] = []
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.reader(stream):
                if len(row) >= 2:
                    odor = re.search(r"\bOdor I\s+(\d+)\s+-", row[1])
                    if odor:
                        odors.append(int(odor.group(1)))
        if not odors:
            raise ValueError(f"No odor events found in {path}.")
        odor_blocks.append(odors)

    order = np.argsort(np.asarray(program_starts, dtype="datetime64[ns]"))
    event_paths = [event_paths[int(i)] for i in order]
    odor_blocks = [odor_blocks[int(i)] for i in order]
    program_starts = [program_starts[int(i)] for i in order]

    acqs = group.acquisitions.query("exp_id == @exp_id").copy()
    required = {"acq_id", "acq_start", "odor_start", "odor_end"}
    missing_columns = required - set(acqs.columns)
    if missing_columns:
        raise ValueError(
            f"Cannot recover exp_id={exp_id} trials; acquisitions lack columns "
            f"{sorted(missing_columns)}."
        )
    for name in ("acq_start", "odor_start", "odor_end"):
        acqs[name] = pd.to_datetime(acqs[name])
    acqs = acqs.sort_values("acq_start").reset_index(drop=True)
    if acqs[list(required - {"acq_id"})].isna().any().any():
        raise ValueError(
            f"Cannot recover exp_id={exp_id} trials; acquisition/odor timestamps "
            "contain missing values."
        )

    odor_ids = [odor for values in odor_blocks for odor in values]
    if len(odor_ids) != len(acqs):
        raise ValueError(
            f"Events contain {len(odor_ids)} odors but exp_id={exp_id} has "
            f"{len(acqs)} acquisitions."
        )

    known_odors = set(map(int, group.odors.odor_id))
    unknown = sorted(set(odor_ids) - known_odors)
    if unknown:
        raise ValueError(f"Events reference unknown odor IDs: {unknown}.")

    # Each event program begins with its first trial.  Verify that its absolute
    # filename timestamp identifies the corresponding acquisition block rather
    # than relying on file ordering alone.
    offsets = np.cumsum([0] + [len(values) for values in odor_blocks[:-1]])
    for path, start, offset in zip(event_paths, program_starts, offsets):
        delta_s = abs((acqs.iloc[int(offset)].acq_start - start).total_seconds())
        if delta_s > 2.0:
            raise ValueError(
                f"Event block {path.name} starts {delta_s:.3f}s from acquisition "
                f"{int(acqs.iloc[int(offset)].acq_id)}; refusing positional pairing."
            )

    block = np.concatenate([
        np.full(len(values), index, dtype=np.int8)
        for index, values in enumerate(odor_blocks)
    ])
    table = acqs[["acq_id", "acq_start", "odor_start", "odor_end"]].copy()
    table["trial_id"] = np.arange(1, len(table) + 1, dtype=np.int32)
    table["trial_start"] = table["acq_start"]
    table["odor_id"] = odor_ids
    table["program_id"] = block
    table["block"] = block
    table["state"] = np.where(block == 0, BEFORE, AFTER)
    table["trial_in_block"] = table.groupby("block").cumcount() + 1
    table["odor_duration_s"] = (
        table["odor_end"] - table["odor_start"]
    ).dt.total_seconds()
    table["baseline_s"] = (
        table["odor_start"] - table["trial_start"]
    ).dt.total_seconds()
    table["odor_minus_acq_start_s"] = table["baseline_s"]
    table["manipulation"] = manipulation
    table["trial_source"] = "olfactometer_events+database_acquisitions"
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
