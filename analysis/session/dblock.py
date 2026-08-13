"""
Diagnose "database is locked" on the odyn database.

The database lives on an SMB share so several machines can reach it, and that
is the source of the trouble. SQLite coordinates writers with whole-file locks
taken through POSIX advisory locking, which network filesystems implement
inconsistently. WAL mode would let readers run during a write, but WAL needs
shared memory and does not work on a network filesystem at all -- so the
database is in rollback-journal mode, where a writer excludes everyone.

Two different failures produce the same message:

  contention    another machine is mid-write. Nothing is wrong; the loser
                should wait. `DB_TIMEOUT_S` is 30 s in odyn/utils.py, which a
                slow write over SMB can exhaust.

  stale lock    a process died mid-transaction and left `odyn.db-journal`
                behind. SQLite normally rolls this back automatically when the
                next connection opens, but that recovery itself needs an
                exclusive lock, which can fail over SMB.

`inspect` tells them apart. It does not delete anything.

DELETING A JOURNAL FILE CAN DESTROY THE DATABASE. A journal is not debris: it
holds the original pages of an interrupted write, and it is what SQLite uses
to put the database back into a consistent state. Deleting one that belongs to
a live transaction, or to a crash that has not yet been rolled back, leaves
the database corrupt in a way no error message announces. If a journal is
present, the safe sequence is: confirm no process anywhere is using the
database, take a copy of both files, and only then let SQLite recover by
opening it normally.
"""

from __future__ import annotations

import sqlite3
import time

from pathlib import Path


def inspect(db_path: str | Path, *, probe_timeout_s: float = 5.0) -> dict:
    """
    Report what state the database is in, without changing anything.

    `writable` is the useful field: it opens a transaction and rolls it back,
    which is what a writer would do, so it distinguishes "locked right now"
    from "fine".
    """

    db_path = Path(db_path)

    report = {
        "database": str(db_path),
        "exists": db_path.is_file(),
        "size_mb": round(db_path.stat().st_size / 1e6, 1) if db_path.is_file() else None,
    }

    if not report["exists"]:
        return report

    journal = db_path.with_name(db_path.name + "-journal")
    wal = db_path.with_name(db_path.name + "-wal")

    for label, path in (("journal", journal), ("wal", wal)):
        if path.is_file():
            stat = path.stat()
            report[label] = {
                "present": True,
                "size_bytes": stat.st_size,
                "age_s": round(time.time() - stat.st_mtime, 1),
            }
        else:
            report[label] = {"present": False}

    # Can we read?
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=probe_timeout_s)
        report["n_experiments"] = con.execute(
            "SELECT COUNT(*) FROM experiments;"
        ).fetchone()[0]
        report["journal_mode"] = con.execute("PRAGMA journal_mode;").fetchone()[0]
        report["readable"] = True
        con.close()
    except Exception as error:
        report["readable"] = False
        report["read_error"] = str(error)

    # Could a writer get in right now? Rolled back immediately -- nothing is
    # committed, but taking the lock is exactly what a real writer does.
    try:
        con = sqlite3.connect(db_path, timeout=probe_timeout_s)
        con.execute("BEGIN IMMEDIATE;")
        con.execute("ROLLBACK;")
        con.close()
        report["writable"] = True
    except sqlite3.OperationalError as error:
        report["writable"] = False
        report["write_error"] = str(error)

    report["diagnosis"] = _diagnose(report)

    return report


def _diagnose(report: dict) -> str:
    if not report.get("readable"):
        return "Cannot read the database at all -- check the share is mounted."

    if report.get("writable"):
        if report["journal"]["present"]:
            return (
                "Writable, but a journal file is present. Most likely a write "
                "is in progress right now. Nothing to do."
            )
        return "Healthy: readable, writable, no journal."

    if report["journal"]["present"]:
        age = report["journal"]["age_s"]
        if age > 600:
            return (
                f"Not writable and a journal has been sitting for {age / 60:.0f} "
                "min. Consistent with a stale lock from a crashed process -- "
                "but also with a very slow write. Confirm nothing is running "
                "on ANY machine before treating it as stale, and read this "
                "module's warning before touching the journal."
            )
        return (
            f"Not writable, journal {age:.0f} s old. Almost certainly a write "
            "in progress. Wait and re-check."
        )

    return (
        "Not writable with no journal present. Suggests a lock held by another "
        "connection, or an SMB lock that outlived its process. Re-check in a "
        "minute; if it persists with nothing running, the share may need "
        "remounting."
    )


def wait_until_writable(
    db_path: str | Path, *, timeout_s: float = 300.0, interval_s: float = 5.0
) -> bool:
    """Poll until a writer could get in, or the deadline passes."""

    deadline = time.time() + timeout_s

    while time.time() < deadline:
        if inspect(db_path, probe_timeout_s=2.0).get("writable"):
            return True
        time.sleep(interval_s)

    return False
