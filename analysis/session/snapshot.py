"""Take a lock-free, best-effort copy of the shared SQLite database."""

from __future__ import annotations

import os
import shutil
import sqlite3
import time

from pathlib import Path


MAX_ATTEMPTS = 2


class SnapshotError(RuntimeError):
    """A snapshot could not be taken and no usable previous one exists."""


ANALYSIS_TABLES = (
    "group_experiments",
    "experiments",
    "acquisitions",
    "mcor_files",
    "trials",
    "programs",
    "odors",
)


def _scan_analysis_tables(con: sqlite3.Connection) -> None:
    """Traverse every row that LocalGroup may consume, raising on read errors."""

    available = {
        row[0]
        for row in con.execute("SELECT name FROM sqlite_master WHERE type='table';")
    }
    missing = set(ANALYSIS_TABLES) - available
    if missing:
        raise sqlite3.DatabaseError(
            f"copy is missing analysis table(s): {', '.join(sorted(missing))}"
        )

    for table in ANALYSIS_TABLES:
        cursor = con.execute(f'SELECT * FROM "{table}";')
        while cursor.fetchmany(10_000):
            pass


def _validate_copy(path: Path) -> tuple[str, str | None]:
    """Validate locally, allowing readable tables in a damaged source database."""

    # Immutable mode prevents journal recovery or sidecar creation. ``path``
    # is the local partial copy; the shared source is never opened here.
    con = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
    try:
        rows = [row[0] for row in con.execute("PRAGMA quick_check;")]
        if rows == ["ok"]:
            return "quick_check", None

        _scan_analysis_tables(con)
    finally:
        con.close()

    first = str(rows[0]).splitlines()[0] if rows else "unknown integrity error"
    return "analysis_table_scan", f"quick_check reported: {first}"


def snapshot_database(
    source: str | Path,
    destination: str | Path,
    *,
    max_attempts: int = MAX_ATTEMPTS,
    # Kept so existing callers of the former SQLite-backup implementation do
    # not break. Raw copying has no pages, sleeps, or lock deadline.
    pages: int | None = None,
    sleep: float | None = None,
    budget_s: float | None = None,
) -> dict:
    """Copy ``source`` without opening it or taking a SQLite lock."""

    del pages, sleep, budget_s

    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    had_previous = destination.exists()
    partial = destination.with_suffix(destination.suffix + ".partial")
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        partial.unlink(missing_ok=True)
        started = time.time()

        try:
            before = source.stat()
            shutil.copyfile(source, partial)

            # Avoid giving an overlapping copy a misleading "now" mtime.
            os.utime(partial, ns=(before.st_atime_ns, before.st_mtime_ns))
            after = source.stat()
            validation, validation_warning = _validate_copy(partial)

            source_changed = (
                before.st_mtime_ns != after.st_mtime_ns
                or before.st_size != after.st_size
            )
            partial.replace(destination)

            elapsed = time.time() - started
            size_mb = destination.stat().st_size / 1e6
            return {
                "ok": True,
                "method": "raw_copy",
                "reason": "; ".join(
                    reason
                    for reason in (
                        validation_warning,
                        (
                            "source changed during copy; validated best-effort snapshot"
                            if source_changed
                            else None
                        ),
                    )
                    if reason
                ) or None,
                "attempts": attempt,
                "seconds": round(elapsed, 3),
                "size_mb": round(size_mb, 1),
                "restarts": 0,
                "steps": 1,
                "mb_per_s": round(size_mb / elapsed, 1) if elapsed > 0 else None,
                "source_mtime": before.st_mtime,
                "source_changed": source_changed,
                "validated": validation,
                "reused_previous": False,
            }

        except (OSError, sqlite3.Error) as error:
            last_error = error
            partial.unlink(missing_ok=True)

    if had_previous:
        return {
            "ok": False,
            "method": "reused_previous",
            "reason": "raw copy did not pass local validation",
            "attempts": max_attempts,
            "reused_previous": True,
            "error": str(last_error),
            "age_s": round(time.time() - destination.stat().st_mtime, 1),
        }

    raise SnapshotError(
        f"Could not copy and validate {source}, and there is no previous copy "
        f"at {destination}.\n  {last_error}"
    )


def snapshot_is_current(source: str | Path, destination: str | Path) -> bool:
    """True when the copy is at least as new as the live file."""

    source, destination = Path(source), Path(destination)
    return destination.exists() and destination.stat().st_mtime >= source.stat().st_mtime


def ensure_snapshot(
    source: str | Path,
    destination: str | Path,
    *,
    refresh: bool = False,
    max_age_s: float = 0.0,
    **kwargs,
) -> dict:
    """Copy when the source is newer, unless a recent copy is acceptable."""

    destination = Path(destination)

    if not refresh and max_age_s > 0 and destination.exists():
        age = time.time() - destination.stat().st_mtime
        if age < max_age_s:
            return {
                "ok": True,
                "method": "reused_recent",
                "reason": None,
                "attempts": 0,
                "seconds": 0.0,
                "reused_previous": True,
                "current": snapshot_is_current(source, destination),
                "age_s": round(age, 1),
                "size_mb": round(destination.stat().st_size / 1e6, 1),
            }

    if not refresh and snapshot_is_current(source, destination):
        return {
            "ok": True,
            "method": "reused_current",
            "reason": None,
            "attempts": 0,
            "seconds": 0.0,
            "reused_previous": True,
            "current": True,
            "size_mb": round(destination.stat().st_size / 1e6, 1),
        }

    return snapshot_database(source, destination, **kwargs)
