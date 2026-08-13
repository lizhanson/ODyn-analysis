"""
Resolve one session's inputs from the database, wherever the data lives.

The development scripts globbed a local folder for `*_mcor.tif` and named the
sync file by hand. That works on a staged copy and nowhere else: it cannot tell
which acquisition a file belongs to except by parsing its name, silently picks
up files that are still being written, and has no idea which motion correction
was approved.

Where the database knows, it is used: `mcor_files` gives the correct path per
acquisition with `source` and `approved`, so a session with two
motion-correction runs resolves to the right one.

It usually does not know. Only 15 of 215 experiments have any `mcor_files`
rows -- the ones motion-corrected through `Group.run_motion_correction`.
Everything else was corrected outside odyn (exp 132 has a `processed/matlab`
folder beside `mcor`) and has files on disk with no database record at all. So
the directory is the fallback, and `mcor_source` in the result records which
route was taken.

Falling back means matching files to acquisitions by the number in the
filename, which is exactly the fragile assumption the database would remove.
It is checked: the count must match, and a mismatch raises rather than
silently pairing the wrong movie with the wrong trial.

Timing comes from whichever sync source the session has:

    sync/<mouse>_<exp>_<date>.h5   2pFrameSync + odorPulse, per-frame clock.
                                   Later sessions; the precise route.
    <exp>_00001.h5                 ImagingWindow (one trigger per acquisition)
                                   + OdorDelivery. Earlier sessions; enough to
                                   time odor onset against each acquisition's
                                   trigger, assuming the nominal frame rate
                                   holds within an acquisition.
    database only                  `baseline_s` from the trial table, good to
                                   ~10 ms on both sessions checked.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TIMING_FRAME_CLOCK = "frame_clock"
TIMING_TRIGGER = "acquisition_trigger"
TIMING_DATABASE = "database"


@dataclass
class SessionInputs:
    """Everything one session needs, resolved and checked."""

    exp_id: int
    exp_name: str
    group_id: int
    frame_rate: float
    shape: tuple[int, int]
    n_frames: int
    um_per_px: float

    paths: list[Path]
    odor_on_frames: list[int]
    odor_off_frames: list[int]
    odor_ids: list[int]
    states: list[str]

    timing_source: str
    path_source: str
    approved_only: bool
    exp_dir: Path
    table: pd.DataFrame
    missing: list[dict]

    @property
    def output_dir(self) -> Path:
        """
        Where this session's derived files belong: `processed/python/`.

        Beside `processed/mcor` and `processed/matlab`, so outputs live with
        the session they came from rather than in a scratch directory that
        disappears with the kernel, the session, or the machine.
        """

        directory = self.exp_dir / "processed" / "python"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    @property
    def cache_dir(self) -> Path:
        """Persistent z-movie cache for this session."""
        directory = self.output_dir / "zscore_cache"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def __getattr__(self, name: str):
        """
        Explain the stale-instance case instead of a bare AttributeError.

        `%autoreload` rebinds methods on existing objects but cannot add a
        field to one already constructed, so an instance made before a field
        was introduced runs the new code against the old data and fails
        somewhere unrelated to the cause. Anything reaching here is either a
        genuine typo or exactly that.
        """

        if name.startswith("_"):
            raise AttributeError(name)

        raise AttributeError(
            f"SessionInputs has no attribute {name!r}. If this object was "
            "created before the field was added, re-run the setup cell to "
            "rebuild it -- autoreload updates methods on existing objects but "
            "cannot add fields to them."
        )

    def summary(self) -> dict:
        return {
            "exp_id": int(self.exp_id),
            "exp_name": self.exp_name,
            "group_id": int(self.group_id),
            "n_acquisitions": len(self.paths),
            "frame_rate": round(float(self.frame_rate), 3),
            "shape_px": list(self.shape),
            "um_per_px_reported": round(float(self.um_per_px), 3),
            "timing_source": self.timing_source,
            "path_source": self.path_source,
            "output_dir": str(self.exp_dir / "processed" / "python"),
            "odor_on_frame_range": [min(self.odor_on_frames), max(self.odor_on_frames)],
            "odor_frames_range": [
                min(b - a for a, b in zip(self.odor_on_frames, self.odor_off_frames)),
                max(b - a for a, b in zip(self.odor_on_frames, self.odor_off_frames)),
            ],
            "n_missing": len(self.missing),
            "approved_only": self.approved_only,
        }


def _rising(signal: np.ndarray, threshold: float = 2.5) -> np.ndarray:
    high = signal > threshold
    return np.flatnonzero(~high[:-1] & high[1:]) + 1


def _falling(signal: np.ndarray, threshold: float = 2.5) -> np.ndarray:
    high = signal > threshold
    return np.flatnonzero(high[:-1] & ~high[1:]) + 1


def find_sync_file(exp_dir: Path) -> tuple[None | Path, str]:
    """Locate the best available timing source for a session."""

    sync_dir = exp_dir / "sync"
    if sync_dir.is_dir():
        candidates = sorted(sync_dir.glob("*.h5"))
        if candidates:
            return candidates[0], TIMING_FRAME_CLOCK

    # Older sessions keep the olfactometer h5 beside the folder.
    candidates = sorted(exp_dir.glob("*_00001.h5"))
    if candidates:
        return candidates[0], TIMING_TRIGGER

    return None, TIMING_DATABASE


def _windows_from_frame_clock(path: Path) -> list[None | tuple[int, int]]:
    from .sync import (
        acquisition_odor_windows, frame_onset_samples,
        group_frames_into_acquisitions, open_sync, pulse_intervals, read_channel,
    )

    sync = open_sync(path)
    blocks = group_frames_into_acquisitions(
        frame_onset_samples(sync), rate_hz=sync.rate_hz
    )
    pulses = pulse_intervals(read_channel(sync, "odorPulse"), rate_hz=sync.rate_hz)

    return acquisition_odor_windows(blocks, pulses, rate_hz=sync.rate_hz)


def _windows_from_trigger(
    path: Path, *, frame_rate: float, loop_interval_s: float
) -> list[None | tuple[int, int]]:
    """
    Odor onset relative to each acquisition's own trigger.

    `ImagingWindow` is one short pulse per acquisition, not a frame clock, so
    frame indices assume the nominal rate holds within an acquisition. Over a
    12 s acquisition a 0.04% rate error is ~5 ms, well under one frame.
    """

    from .h5io import open_h5

    with open_h5(path) as f:
        rate = float(f.attrs["samplerate"])
        triggers = _rising(f["ImagingWindow"][:])
        odor = f["OdorDelivery"][:]

    on_samples = _rising(odor)
    off_samples = _falling(odor)

    out: list[None | tuple[int, int]] = []

    for trigger in triggers:
        window = on_samples[
            (on_samples >= trigger) & (on_samples < trigger + loop_interval_s * rate)
        ]
        if window.size == 0:
            out.append(None)
            continue

        on_s = (window[0] - trigger) / rate
        after = off_samples[off_samples > window[0]]
        off_s = (after[0] - trigger) / rate if after.size else on_s

        out.append((int(round(on_s * frame_rate)), int(round(off_s * frame_rate))))

    return out


def _no_mcor_message(group, exp_id: int, exp_dir) -> str:
    """
    Explain a missing-mcor failure in terms of what to do about it.

    Two things make the bare version of this error misleading. First, it cannot
    distinguish "motion correction has not been run" from "it ran and the
    output is missing", which need opposite responses. Second, group ids and
    experiment ids are both small integers in overlapping ranges, so a number
    is usually valid as either -- group 217 is exp 213, while exp 217 is a
    different session entirely. Passing one where the other was meant lands on
    a real experiment that simply is not ready, and nothing in the message
    hints at the mix-up.
    """

    mcor_dir = exp_dir / "processed" / "mcor"
    raw = sorted((exp_dir / "raw").glob("*.tif")) if (exp_dir / "raw").is_dir() else []

    lines = [f"exp {exp_id}: no mcor_files rows and no *_mcor.tif under {mcor_dir}."]

    if not mcor_dir.exists() and raw:
        lines.append(
            f"  There are {len(raw)} raw acquisitions but no processed/ folder, "
            f"so motion correction has not been run on this experiment yet."
        )
    elif not raw:
        lines.append("  No raw tifs either -- check that this is the right experiment.")

    # Is this number also a group id? If so it almost certainly is the mix-up.
    try:
        mapping = group.group_experiments
        row = mapping[mapping.group_id == exp_id]

        if len(row):
            other = int(row.iloc[0].exp_id)
            if other != exp_id:
                names = group.experiments
                key = "name" if "name" in names.columns else names.columns[1]
                label = names[names.exp_id == other]
                label = str(label.iloc[0][key]) if len(label) else "?"
                lines.append(
                    f"  Note: {exp_id} is also a GROUP id, and group {exp_id} is "
                    f"exp {other} ({label}). resolve_session takes exp_id -- if "
                    f"you meant that session, pass exp_id={other}."
                )
    except Exception:
        # Diagnostics must never replace the original failure.
        pass

    return "\n".join(lines)


def experiments_in_group(group, group_id: int) -> list[int]:
    """
    The experiment ids a group contains, validated as one session.

    A group is usually one experiment, but a session can also be split into
    two: same field of view, same animal, `_e1` before the manipulation and
    `_e2` after. Those have to be analysed together -- the whole point is a
    within-glomerulus comparison across the two -- so the group, not the
    experiment, is the right unit to ask for.

    Order comes from `exp_start`, not from the name. `exp_name` cannot be
    trusted for this: in two of the four multi-experiment groups the second
    experiment is recorded as `_e1` as well, even though its acquisitions sit
    under `.../e2/raw/`. Sorting by name would put those in an arbitrary order
    and silently label the post block as pre -- the one error that would invert
    every result while looking entirely normal. Start times are unambiguous
    (10:13 vs 11:43 for group 193) and match the folder layout.

    Same animal and same frame size are still required, because one group in
    the database spans two mice on two different days at two different frame
    sizes. Concatenating that would produce a session that looks fine and mixes
    animals, so it is refused with the reason rather than guessed at.
    """

    mapping = group.group_experiments
    rows = mapping[mapping.group_id == group_id]

    if rows.empty:
        raise ValueError(
            f"No group {group_id} in the database. If {group_id} is an "
            f"experiment id, pass exp_id={group_id} instead."
        )

    exp_ids = sorted(int(v) for v in rows.exp_id)

    if len(exp_ids) == 1:
        return exp_ids

    table = group.experiments
    members = table[table.exp_id.isin(exp_ids)]

    mice = set(members.mouse_id)
    if len(mice) > 1:
        raise ValueError(
            f"Group {group_id} spans {len(mice)} mice ({sorted(mice)}), so its "
            f"experiments are not one session. Pass exp_id= for the one you "
            f"want: {dict(zip(members.exp_id, members.exp_name))}."
        )

    fovs = set(zip(members.height_px, members.width_px))
    if len(fovs) > 1:
        raise ValueError(
            f"Group {group_id}'s experiments have different frame sizes "
            f"({sorted(fovs)}), so they are not the same field of view and one "
            f"mask cannot serve both. Pass exp_id= to analyse them separately."
        )

    starts = pd.to_datetime(members.exp_start, errors="coerce")

    if starts.isna().any() or starts.nunique() != len(starts):
        raise ValueError(
            f"Group {group_id}'s experiments do not have distinct start times "
            f"({list(members.exp_start)}), so which came first cannot be told "
            f"from the database. Pass exp_id= explicitly: "
            f"{dict(zip(members.exp_id, members.exp_name))}."
        )

    ordered = members.assign(_start=starts).sort_values("_start")

    return [int(v) for v in ordered.exp_id]


def resolve_group(
    group,
    *,
    group_id: int,
    manipulation: None | str = None,
    **kwargs,
) -> SessionInputs:
    """
    Resolve a whole group, whether it is one experiment or a split session.

    For a single-experiment group this is `resolve_session` by another name.
    For an e1/e2 pair the two are resolved separately and concatenated, with
    `state` taken from which experiment each acquisition came from rather than
    from the olfactometer program -- in a split session the program numbering
    restarts, so it carries no before/after information.
    """

    exp_ids = experiments_in_group(group, group_id)

    if len(exp_ids) == 1:
        return resolve_session(
            group, exp_id=exp_ids[0], manipulation=manipulation, **kwargs
        )

    from .trials import AFTER, BEFORE

    parts = [
        resolve_session(group, exp_id=e, manipulation=manipulation, **kwargs)
        for e in exp_ids
    ]
    labels = [BEFORE, AFTER]

    return _combine(parts, labels, group_id=group_id)


def _combine(parts: list, labels: list[str], *, group_id: int) -> SessionInputs:
    """Concatenate resolved experiments into one session."""

    # Frame rates must agree, but only to the precision that matters.
    #
    # The reported rate is measured, not commanded, so the two halves of one
    # session differ in the last digit -- 13.998 against 13.999 on group 193.
    # Demanding exact equality rejects a perfectly good pairing. What would
    # matter is a rate difference large enough to shift odor onset within an
    # acquisition: at 0.1% over a 12 s acquisition that is ~0.17 frames, still
    # under the one-frame resolution of the alignment itself.
    rates = [float(p.frame_rate) for p in parts]
    spread = (max(rates) - min(rates)) / max(min(rates), 1e-9)

    if spread > 0.001:
        raise ValueError(
            f"Group {group_id}: experiments were acquired at frame rates "
            f"differing by {spread:.2%} ({[round(r, 3) for r in rates]}); a "
            f"shared time base is not meaningful."
        )

    shapes = {p.shape for p in parts}
    if len(shapes) > 1:
        raise ValueError(
            f"Group {group_id}: experiments have different frame shapes "
            f"({sorted(shapes)}); one mask cannot serve both."
        )

    # Both halves must be timed the same way.
    #
    # This is the one inconsistency that would corrupt the result rather than
    # merely degrade it. The whole purpose of a split session is comparing the
    # same glomeruli before and after, and the sources differ in where they put
    # odor onset -- the frame clock resolves it to the frame, the database to
    # ~10 ms of a nominal rate. Timing one block one way and the other block
    # the other puts a systematic offset exactly along the axis being measured,
    # and nothing downstream could distinguish it from a real effect.
    sources = {p.timing_source for p in parts}
    if len(sources) > 1:
        raise ValueError(
            f"Group {group_id}: the two halves resolved to different timing "
            f"sources ({sorted(sources)}). A systematic onset offset between "
            f"pre and post would be indistinguishable from a real state "
            f"effect. Fix the sync file for the affected experiment, or force "
            f"both onto the database with the same route, before combining."
        )

    first = parts[0]
    tables = []

    for part, label in zip(parts, labels):
        block = part.table.copy()
        block["state"] = label
        block["exp_id"] = part.exp_id
        tables.append(block)

    return SessionInputs(
        # The first experiment names the session; `group_id` is what identifies
        # it, and what the output filenames are built from.
        exp_id=first.exp_id,
        exp_name=first.exp_name,
        group_id=int(group_id),
        frame_rate=first.frame_rate,
        shape=first.shape,
        n_frames=first.n_frames,
        um_per_px=first.um_per_px,
        paths=[p for part in parts for p in part.paths],
        odor_on_frames=[v for part in parts for v in part.odor_on_frames],
        odor_off_frames=[v for part in parts for v in part.odor_off_frames],
        odor_ids=[v for part in parts for v in part.odor_ids],
        states=[label for part, label in zip(parts, labels) for _ in part.states],
        timing_source=first.timing_source,
        path_source=first.path_source,
        approved_only=first.approved_only,
        # Outputs go with the first experiment, so a split session writes one
        # set of files rather than two half-sets.
        exp_dir=first.exp_dir,
        table=pd.concat(tables, ignore_index=True),
        missing=[m for part in parts for m in part.missing],
    )


def resolve_session(
    group,
    *,
    exp_id: int,
    manipulation: None | str = None,
    mcor_source: None | str = None,
    approved_only: bool = False,
) -> SessionInputs:
    """
    Gather one session's motion-corrected paths, odor windows, and labels.

    Paths come from `mcor_files` rather than a directory listing, so each is
    tied to its acquisition and the right motion-correction run is chosen.
    `approved_only` is off by default because no rows are approved yet -- when
    the approval workflow is used, turn it on.

    `manipulation` is passed through to `trial_table`; leaving it None takes
    that function's default of ketamine/xylazine.
    """

    from .trials import DEFAULT_MANIPULATION, trial_table

    experiment = group.experiments.query("exp_id == @exp_id")
    if experiment.empty:
        raise ValueError(f"No experiment {exp_id} in this group.")
    experiment = experiment.iloc[0]

    frame_rate = float(experiment.frame_rate)
    main_folder = Path(group.main_folder)

    table = trial_table(
        group,
        exp_id=exp_id,
        manipulation=manipulation or DEFAULT_MANIPULATION,
    )

    # mcor path per acquisition, straight from the database.
    mcor = group.mcor_files
    acquisitions = group.acquisitions.query("exp_id == @exp_id")[["acq_id"]]
    mcor = mcor.merge(acquisitions, on="acq_id", how="inner")

    if mcor_source is not None:
        mcor = mcor[mcor.source == mcor_source]

    if approved_only:
        # Trial-level QC, read from the database rather than re-derived here.
        #
        # `approved` is where a judgement about an individual acquisition gets
        # recorded -- a trial where the PMT was off, one the motion correction
        # mangled, one with a z-jump. Session 213 has exactly such a trial:
        # acquisition 81, PMT off after the injection, mean intensity -4
        # against a session median of 477. It z-scores to unit variance like
        # any other trial, so nothing downstream flags it; it simply adds a
        # trial of noise to whatever average it lands in.
        #
        # Excluding it is a decision about data, not about code, which is why
        # it belongs in the database and not in a hardcoded list here.
        before = len(mcor)
        mcor = mcor[mcor.approved.astype(bool)]

        if mcor.empty:
            raise ValueError(
                f"exp {exp_id}: approved_only=True but none of the {before} "
                f"mcor rows are approved. Approve them in the database, or "
                f"pass approved_only=False."
            )

    # acq_id arrives as float64 whenever any trial has a NULL acquisition, so
    # an int-keyed dict would miss every lookup. Key on int explicitly.
    by_acq = {int(a): p for a, p in zip(mcor.acq_id, mcor.mcor_path)}
    path_source = "database"

    exp_dir = _resolve_path(main_folder, table.iloc[0].raw_path).parent.parent \
        if "raw_path" in table else None
    if exp_dir is None or not exp_dir.is_dir():
        acq_path = group.acquisitions.query("exp_id == @exp_id").iloc[0].raw_path
        exp_dir = _resolve_path(main_folder, acq_path).parent.parent

    if not by_acq:
        # Not registered in odyn. Fall back to the directory, matching on the
        # acquisition number in the filename.
        found = sorted((exp_dir / "processed" / "mcor").glob("*_mcor.tif"))
        acq_ids = sorted(
            int(a) for a in group.acquisitions.query("exp_id == @exp_id").acq_id
        )

        if not found:
            raise ValueError(_no_mcor_message(group, exp_id, exp_dir))

        if len(found) != len(acq_ids):
            raise ValueError(
                f"exp {exp_id}: {len(found)} mcor files on disk but "
                f"{len(acq_ids)} acquisitions in the database. Refusing to "
                "guess the pairing -- register them with run_motion_correction, "
                "or pass the paths explicitly."
            )

        by_acq = {acq: path for acq, path in zip(acq_ids, found)}
        path_source = "directory"

    sync_path, timing_source = find_sync_file(exp_dir)

    # Sync file first, database second.
    #
    # The frame clock is the better source when it is readable: it gives odor
    # onset to the frame, from the same 5 kHz recording as the valve, with no
    # assumption about frame rate or clock agreement. But rig code has changed
    # over the years and old sessions carry older layouts, and a session that
    # cannot be read here is not a session with unknown timing -- membership in
    # a group means it already passed odyn's sync checks and its trial timing
    # is recorded. Falling back to that is strictly better than refusing to
    # analyse the experiment, and the source is reported either way so nobody
    # has to guess which was used.
    windows = None

    if timing_source == TIMING_FRAME_CLOCK:
        try:
            windows = _windows_from_frame_clock(sync_path)
        except (KeyError, OSError, ValueError, RuntimeError) as error:
            # RuntimeError included deliberately: HDF5 raises it for a failed
            # close, which on SMB happens transiently and is not a statement
            # about the data. Letting it propagate would abort a session over
            # a filesystem hiccup when the database has the timing anyway.
            windows = None
            timing_source = f"{TIMING_TRIGGER} (sync unreadable: {type(error).__name__})"

    if windows is None and sync_path is not None:
        # The old olfactometer h5 sits beside the session even where a newer
        # sync/ folder exists, so it is worth trying before the database.
        legacy = sorted(exp_dir.glob("*_00001.h5"))
        candidate = sync_path if timing_source == TIMING_TRIGGER else (
            legacy[0] if legacy else None
        )

        if candidate is not None:
            try:
                windows = _windows_from_trigger(
                    candidate,
                    frame_rate=frame_rate,
                    loop_interval_s=float(experiment.loop_acq_interval_s) + 20.0,
                )
                timing_source = TIMING_TRIGGER
            except (KeyError, OSError, ValueError, RuntimeError):
                windows = None

    if windows is None:
        # Neither file route worked. The per-trial fallback below uses the
        # database's own timing, which is what group membership certifies.
        timing_source = TIMING_DATABASE

    paths, on_frames, off_frames, odors, states, missing = [], [], [], [], [], []

    for position, row in enumerate(table.itertuples()):
        acq_id = row.acq_id
        stored = None if pd.isna(acq_id) else by_acq.get(int(acq_id))

        if stored is None:
            missing.append({"trial": int(row.trial_id), "reason": "no motion-corrected file"})
            continue

        path = stored if isinstance(stored, Path) else _resolve_path(main_folder, stored)

        if windows is not None:
            window = windows[position] if position < len(windows) else None
            if window is None:
                missing.append({"trial": int(row.trial_id), "reason": "no odor pulse in sync"})
                continue
            on_frame, off_frame = window
        else:
            # Database fallback: `baseline_s` is measured within one clock and
            # agreed with the sync file to 3.6-7.4 ms on both sessions checked.
            on_frame = int(round(row.baseline_s * frame_rate))
            off_frame = int(round((row.baseline_s + row.odor_duration_s) * frame_rate))

        paths.append(path)
        on_frames.append(on_frame)
        off_frames.append(off_frame)
        odors.append(int(row.odor_id))
        states.append(row.state)

    if not paths:
        raise ValueError(
            f"No usable acquisitions for exp {exp_id}: {len(missing)} skipped."
        )

    # `pd` is imported at module level -- a local import here would shadow it
    # and make every earlier use in this function an UnboundLocalError.
    try:
        rows = pd.read_sql_query(
            "SELECT group_id FROM group_experiments WHERE exp_id = ?;",
            group.con, params=[int(exp_id)],
        )
        group_id = int(rows.group_id.iloc[0]) if len(rows) else int(exp_id)
    except Exception:
        # No group membership recorded: fall back to exp_id so the filename is
        # still unique and traceable.
        group_id = int(exp_id)

    return SessionInputs(
        exp_id=int(exp_id),
        exp_name=str(experiment.exp_name),
        group_id=group_id,
        frame_rate=frame_rate,
        shape=(int(experiment.height_px), int(experiment.width_px)),
        n_frames=int(experiment.frame_count),
        um_per_px=float(experiment.width_um) / float(experiment.width_px),
        paths=paths,
        odor_on_frames=on_frames,
        odor_off_frames=off_frames,
        odor_ids=odors,
        states=states,
        timing_source=timing_source,
        approved_only=bool(approved_only),
        path_source=path_source,
        exp_dir=exp_dir,
        table=table,
        missing=missing,
    )


def _resolve_path(main_folder: Path, stored: str) -> Path:
    """Stored paths are POSIX and relative to main_folder."""
    return Path(main_folder) / str(stored).replace("\\", "/")
