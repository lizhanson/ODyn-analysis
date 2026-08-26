"""Build the per-session trial table and label its two blocks."""

from __future__ import annotations

import csv
import itertools
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
    timestamp_pattern = re.compile(
        r"(\d{4}_\d{2}_\d{2})-(\d{2}_\d{2}_\d{2})-Events\.csv$"
    )

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

    # Older exports were placed beside the mouse folder instead of beneath
    # the experiment. Include that date folder, then use absolute program
    # timestamps and event counts to select this experiment's two files.
    roots = [exp_dir]
    date_root = next(
        (parent for parent in (exp_dir, *exp_dir.parents)
         if re.fullmatch(r"\d{8}", parent.name)),
        None,
    )
    if date_root is not None and date_root != exp_dir:
        roots.append(date_root)
    event_paths = sorted({path for root in roots for path in root.glob("**/*-Events.csv")})

    programs: list[tuple[Path, pd.Timestamp, list[int]]] = []
    for path in event_paths:
        match = timestamp_pattern.search(path.name)
        if match is None:
            continue
        program_start = pd.to_datetime(
            f"{match.group(1)} {match.group(2)}", format="%Y_%m_%d %H_%M_%S"
        )

        odors: list[int] = []
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.reader(stream):
                if len(row) >= 2:
                    odor = re.search(r"\bOdor I\s+(\d+)\s+-", row[1])
                    if odor:
                        odors.append(int(odor.group(1)))
        if not odors:
            continue
        programs.append((path, program_start, odors))

    # A recovered file may subsequently be copied into the experiment folder,
    # leaving the original at the date level. Treat byte-equivalent program
    # identities as one candidate and prefer the experiment-local copy.
    unique_programs: dict[tuple[pd.Timestamp, tuple[int, ...]], tuple[Path, pd.Timestamp, list[int]]] = {}
    for program in programs:
        key = (program[1], tuple(program[2]))
        current = unique_programs.get(key)
        if current is None or (
            exp_dir in program[0].parents and exp_dir not in current[0].parents
        ):
            unique_programs[key] = program
    programs = list(unique_programs.values())

    if len(programs) == 2:
        selected = [tuple(sorted(programs, key=lambda item: item[1]))]
    else:
        selected = []
        for pair in itertools.combinations(programs, 2):
            pair = tuple(sorted(pair, key=lambda item: item[1]))
            counts = [len(item[2]) for item in pair]
            if sum(counts) != len(acqs):
                continue
            offsets = (0, counts[0])
            if all(abs((acqs.iloc[offset].acq_start - item[1]).total_seconds()) <= 2.0
                   for item, offset in zip(pair, offsets)):
                selected.append(pair)
    if len(selected) != 1:
        raise ValueError(
            f"No trials for exp_id={exp_id}; event recovery found {len(programs)} "
            f"usable *-Events.csv files in {[str(root) for root in roots]}, but "
            f"{len(selected)} unique two-block pairs matched {len(acqs)} acquisitions "
            "by count and timestamp."
        )

    event_paths = [item[0] for item in selected[0]]
    program_starts = [item[1] for item in selected[0]]
    odor_blocks = [item[2] for item in selected[0]]

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
