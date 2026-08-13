"""
Take a bounded snapshot of the shared database, and read that instead.

Why this exists, concretely. On 2026-08-12 a motion-correction run failed hours
in with `database is locked`, having already done all its compute. The cause
was a read: SQLite on SMB cannot use WAL, so it runs in rollback-journal mode,
where a reader holds SHARED and a writer needs EXCLUSIVE. A `SELECT COUNT(*)`
sweep across every table -- 245,185 rows in `events` alone -- held SHARED for
over two minutes. odyn's writer waits `DB_TIMEOUT_S = 30` and then raises.

Read-only is not the protection it sounds like. `mode=ro` guarantees the reader
cannot *corrupt* anything; it says nothing about how long the reader makes
everyone else wait. The damage here was done by a connection that could not
write a byte.

The fix is not to be careful with queries. Care does not survive contact with
an interactive session, where the whole point is to ask questions nobody
planned. The fix is to make the live database unreachable for analysis: copy it
once, under a bounded lock, then answer every question from the copy. A scan of
the copy can take an hour and cost nothing.

**Bounded** is the operative word, since a consistent snapshot must hold a read
lock for *some* window:

  * `Connection.backup(pages=N, sleep=S)` copies N pages per step and releases
    the source lock between steps, so a writer waiting on EXCLUSIVE gets it
    within roughly one step rather than one whole copy.
  * SQLite restarts a backup if the source is written mid-copy. That is correct
    -- it is what makes the result consistent -- but it can livelock against a
    busy writer, so attempts are capped and the previous snapshot is kept on
    failure. A snapshot from ten minutes ago beats a failed session, and beats
    blocking the writer to insist on a fresher one.

The copy is refreshed when the live file's mtime moves past it, so it cannot go
stale the way a hand-made copy does -- the one it replaced was 23 hours old and
missing 1,253 of 3,876 `mcor_files` rows, which silently changed which code
path `resolve_session` took.
"""

from __future__ import annotations

import shutil
import sqlite3
import time

from pathlib import Path

# Pages copied per backup step, at SQLite's 4 KB default page size.
#
# The instinct is to make this small so the lock is released often. That is
# backwards, and cost 222 s on the first real run against a busy share.
#
# SQLite restarts a backup from the beginning whenever the source is written
# during it. Small steps do release the lock more often, but they stretch the
# copy's wall-clock time, which widens the window for a write to land inside
# it, which causes a restart, which lengthens it further. With motion
# correction committing a row per saved file and the share reading at 4 MB/s,
# a 35 MB copy in 512-page steps restarted ~26 times and re-read 900 MB.
#
# Bigger steps are the fix: the copy finishes before a write is likely to
# interrupt it. 2048 pages is ~8 MB, so ~5 steps for a 35 MB database, each
# holding the lock ~2 s at 4 MB/s -- an order of magnitude inside odyn's 30 s
# writer patience, while being brief enough overall to usually dodge restarts.
BACKUP_PAGES = 2048

# No artificial sleep. It only added wall-clock time, which is the thing that
# invites restarts; SQLite still yields between steps regardless.
BACKUP_SLEEP = 0.0

# A backup that keeps restarting is consuming share bandwidth that the writer
# needs for its own files. Past this, give up and keep the previous snapshot:
# a copy a few minutes old beats starving the run that is producing the data.
MAX_ATTEMPTS = 2
ATTEMPT_BUDGET_S = 45.0

# The longest this may hold the source lock, in seconds.
#
# This is the number that keeps the promise. odyn's writers wait
# `DB_TIMEOUT_S = 30` before raising `database is locked`, so a snapshot that
# holds the lock longer than that fails someone else's run. 20 s leaves a third
# of the budget as margin, and at the 4 MB/s this share degrades to under load
# it is still twice what a 35 MB copy needs.
#
# It is enforced by aborting the copy, not by hoping: a progress handler checks
# the clock and cancels. Giving up and reusing a slightly older snapshot is
# always the right trade against breaking a run that has been going for hours.
LOCK_BUDGET_S = 20.0

# Rows of SQLite VM instructions between progress-handler calls. Small enough
# that the deadline is checked promptly, large enough not to matter.
PROGRESS_INTERVAL = 10_000


class SnapshotError(RuntimeError):
    """A snapshot could not be taken and no usable previous one exists."""


def _vacuum_into(
    source: Path, destination: Path, *, budget_s: float
) -> None | float:
    """
    Copy with `VACUUM INTO`, abandoning it if it runs past `budget_s`.

    Returns the elapsed seconds, or None if it could not be done -- too slow,
    unsupported, or the source busy -- leaving the caller to fall back.

    The deadline is enforced with a progress handler that returns non-zero,
    which makes SQLite abort the statement and release the lock. That is what
    turns "hold the lock for however long the copy takes" into a bound: on a
    share degraded enough that the copy would outlast a writer's patience, this
    gives up instead of taking the run down with it.
    """

    started = time.time()

    try:
        live = sqlite3.connect(f"file:{source}?mode=ro", uri=True, timeout=budget_s)
    except sqlite3.Error:
        return None

    try:
        live.set_progress_handler(
            lambda: 1 if time.time() - started > budget_s else 0, PROGRESS_INTERVAL
        )
        live.execute("VACUUM INTO ?;", (str(destination),))

    except sqlite3.Error:
        destination.unlink(missing_ok=True)
        return None

    finally:
        live.set_progress_handler(None, 0)
        live.close()

    return time.time() - started


def snapshot_database(
    source: str | Path,
    destination: str | Path,
    *,
    pages: int = BACKUP_PAGES,
    sleep: float = BACKUP_SLEEP,
    max_attempts: int = MAX_ATTEMPTS,
    budget_s: float = ATTEMPT_BUDGET_S,
) -> dict:
    """
    Copy `source` to `destination` with SQLite's backup API.

    Unlike `cp`, this cannot catch the database mid-transaction: the backup API
    is transaction-aware and restarts if the source changes underneath it. A
    plain file copy of a live SQLite database can produce a file that opens
    fine and is subtly wrong, which is the worst possible failure here.

    Returns timing and outcome. Raises `SnapshotError` only when there is also
    no previous snapshot to fall back on.
    """

    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    had_previous = destination.exists()
    partial = destination.with_suffix(destination.suffix + ".partial")

    last_error = None

    for attempt in range(1, max_attempts + 1):
        partial.unlink(missing_ok=True)
        started = time.time()

        # `VACUUM INTO` first: it runs in one read transaction, so unlike the
        # backup API it cannot be restarted by a write landing mid-copy. That
        # is the whole problem on a share where motion correction commits every
        # few seconds -- a copy that takes 8 s at 4 MB/s will meet a write, and
        # the backup API responds by starting over. The first real run took
        # 222 s and 26 restarts to copy 35 MB.
        #
        # The cost is that the lock is held continuously instead of in slices,
        # which is fine precisely because it is bounded: `LOCK_BUDGET_S` aborts
        # below the writer's patience.
        result = _vacuum_into(source, partial, budget_s=min(budget_s, LOCK_BUDGET_S))

        if result is not None:
            partial.replace(destination)
            size_mb = destination.stat().st_size / 1e6
            return {
                "ok": True,
                "method": "vacuum_into",
                "attempts": attempt,
                "seconds": round(result, 3),
                "size_mb": round(size_mb, 1),
                "restarts": 0,
                "steps": 1,
                "mb_per_s": round(size_mb / result, 1) if result > 0 else None,
                "source_mtime": source.stat().st_mtime,
                "reused_previous": False,
            }

        # Restarts are invisible in the return value, so they are counted from
        # the progress callback: `remaining` jumping back up means SQLite threw
        # the partial copy away and started over. Without this the only symptom
        # is a copy that inexplicably takes minutes.
        state = {"restarts": 0, "previous": None, "steps": 0}

        def progress(status, remaining, total):
            state["steps"] += 1
            if state["previous"] is not None and remaining > state["previous"]:
                state["restarts"] += 1
            state["previous"] = remaining

            if time.time() - started > budget_s:
                raise TimeoutError(
                    f"snapshot exceeded {budget_s}s after "
                    f"{state['restarts']} restart(s)"
                )

        try:
            # Read-only so this can never write to the shared file, and a short
            # timeout so a busy source is reported rather than waited out.
            live = sqlite3.connect(
                f"file:{source}?mode=ro", uri=True, timeout=budget_s
            )
            try:
                copy = sqlite3.connect(partial)
                try:
                    live.backup(copy, pages=pages, sleep=sleep, progress=progress)
                finally:
                    copy.close()
            finally:
                live.close()

        except (sqlite3.Error, TimeoutError) as error:
            last_error = error
            partial.unlink(missing_ok=True)
            if time.time() - started > budget_s:
                break
            continue

        # Replace only once the copy is complete, so an interrupted snapshot
        # cannot leave a truncated database where a good one used to be.
        partial.replace(destination)

        elapsed = time.time() - started
        size_mb = destination.stat().st_size / 1e6

        return {
            "ok": True,
            "attempts": attempt,
            "seconds": round(elapsed, 3),
            "size_mb": round(size_mb, 1),
            "restarts": state["restarts"],
            "steps": state["steps"],
            # Effective rate including any restarts, so a slow copy can be
            # attributed: a low rate with no restarts is a congested share, a
            # low rate with restarts is a busy writer.
            "mb_per_s": round(size_mb / elapsed, 1) if elapsed > 0 else None,
            "source_mtime": source.stat().st_mtime,
            "reused_previous": False,
        }

    if had_previous:
        return {
            "ok": False,
            "attempts": max_attempts,
            "reused_previous": True,
            "error": str(last_error),
            "age_s": round(time.time() - destination.stat().st_mtime, 1),
        }

    raise SnapshotError(
        f"Could not snapshot {source} and there is no previous copy at "
        f"{destination}. Last error: {last_error}"
    )


def snapshot_is_current(source: str | Path, destination: str | Path) -> bool:
    """True when the copy is at least as new as the live file."""

    source, destination = Path(source), Path(destination)

    if not destination.exists():
        return False

    return destination.stat().st_mtime >= source.stat().st_mtime


def ensure_snapshot(
    source: str | Path,
    destination: str | Path,
    *,
    refresh: bool = False,
    **kwargs,
) -> dict:
    """Snapshot only when the live file has moved on, or when forced."""

    if not refresh and snapshot_is_current(source, destination):
        return {
            "ok": True,
            "attempts": 0,
            "seconds": 0.0,
            "reused_previous": True,
            "current": True,
            "size_mb": round(Path(destination).stat().st_size / 1e6, 1),
        }

    return snapshot_database(source, destination, **kwargs)
