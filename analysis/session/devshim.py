"""
A read-only stand-in for `odyn.Group`, for developing against a local copy.

The real `Group` reaches the database on the MossLab SMB share, which reads at
~5 MB/s and whose `Database` writes a `method_calls` row on every decorated
call. Neither is wanted while iterating. This exposes the same attributes the
analysis code actually uses -- `.trials`, `.acquisitions`, `.experiments`,
`.mcor_files`, `.main_folder` -- backed by a local snapshot.

Because the attribute surface matches, the analysis functions are written
against `group` from the start and need no change when they are later wrapped
in `@record_call` and handed a real `Group`.
"""

from __future__ import annotations

import sqlite3

from functools import cached_property
from pathlib import Path

import pandas as pd


class LocalGroup:
    """Group-shaped read-only view over a local database copy."""

    def __init__(
        self,
        db_path: str | Path,
        main_folder: str | Path,
        *,
        snapshot_to: None | str | Path = None,
        refresh: bool = False,
        allow_live: bool = False,
        busy_timeout_s: float = 120.0,
    ):
        self.db_path = Path(db_path)
        self.main_folder = Path(main_folder)
        self.group_id = 0
        self.snapshot = None

        # Snapshot first, then read the copy.
        #
        # Every accessor below is `SELECT * FROM <table>` into pandas -- whole
        # tables, 23,394 rows for `acquisitions`. Against the live database on
        # the share that is a multi-second to multi-minute SHARED lock, and a
        # writer needs EXCLUSIVE. On 2026-08-12 exactly that pattern, run by
        # hand, took odyn's writer past its 30 s timeout and killed a motion
        # correction run hours into its compute.
        #
        # Read-only does not help: `mode=ro` stops this connection corrupting
        # anything, and does nothing about how long it makes others wait.
        # Neither does query discipline, because the value of an interactive
        # session is asking questions nobody planned for. So the live file is
        # copied once, under a bounded lock, and never read again.
        if snapshot_to is not None:
            from .snapshot import ensure_snapshot

            self.snapshot = ensure_snapshot(
                self.db_path, snapshot_to, refresh=refresh
            )
            self.source_path, self.db_path = self.db_path, Path(snapshot_to)

        elif not allow_live and self._looks_remote(self.db_path):
            raise ValueError(
                f"Refusing to read {self.db_path} directly: it is on a network "
                f"share, and the whole-table reads this class does would hold a "
                f"SQLite lock long enough to fail another machine's writes.\n\n"
                f"Pass snapshot_to=<local path> to copy it first (bounded, a "
                f"few seconds, refreshed when the live file changes), or "
                f"allow_live=True if you have a specific reason and know the "
                f"queries are small."
            )

        # Read-only, always.
        #
        # Belt and braces on top of the snapshot: even against a local copy,
        # opening read-write leaves the door open to taking a write lock or
        # leaving a rollback journal behind.
        self.con = sqlite3.connect(
            f"file:{self.db_path}?mode=ro", uri=True, timeout=busy_timeout_s
        )
        self.con.row_factory = sqlite3.Row

        # Declared, so callers can assert it without attempting a write. A
        # probe write is a bad test: it fails for reasons other than
        # permissions -- "table already exists" being the obvious one -- and a
        # naive try/except reads that as success. Worse, if the module in
        # memory is stale the probe genuinely writes.
        #
        # Reading this attribute is also the stale-module check: an older
        # devshim does not define it, so `group.read_only` raises
        # AttributeError rather than quietly reporting the wrong thing.
        self.read_only = True

        # Wait rather than raise if someone else holds the lock.
        self.con.execute(f"PRAGMA busy_timeout = {int(busy_timeout_s * 1000)};")

    @staticmethod
    def _looks_remote(path: Path) -> bool:
        """
        Is this path on a network mount?

        Deliberately crude -- `/Volumes/<share>` on macOS, a UNC path on
        Windows. A false positive costs one keyword argument; a false negative
        costs someone else's overnight run.
        """

        text = str(path)

        return text.startswith("/Volumes/") or text.startswith("\\\\")

    def _table(self, name: str) -> pd.DataFrame:
        return pd.read_sql_query(f"SELECT * FROM {name};", self.con)

    @cached_property
    def trials(self) -> pd.DataFrame:
        return self._table("trials")

    @cached_property
    def acquisitions(self) -> pd.DataFrame:
        return self._table("acquisitions")

    @cached_property
    def experiments(self) -> pd.DataFrame:
        return self._table("experiments")

    @cached_property
    def programs(self) -> pd.DataFrame:
        return self._table("programs")

    @cached_property
    def mcor_files(self) -> pd.DataFrame:
        return self._table("mcor_files")

    @property
    def group_experiments(self) -> pd.DataFrame:
        """
        Which experiments belong to which group.

        A plain `property`, not `cached_property`, unlike its neighbours.
        `cached_property` needs `__set_name__`, which Python calls only when a
        class body executes -- so one added to a class that `%autoreload` then
        patches in place has no name and raises

            TypeError: Cannot use cached_property instance without calling
            __set_name__ on it

        in a live kernel, forcing a restart for what should be a hot reload.
        The table is 231 rows from a local snapshot, so caching it buys
        nothing worth that.
        """

        return self._table("group_experiments")

    @cached_property
    def odors(self) -> pd.DataFrame:
        return self._table("odors")

    def add_output_file(self, path: str | Path) -> None:
        """No-op: nothing is recorded until the function is `@record_call`ed."""
