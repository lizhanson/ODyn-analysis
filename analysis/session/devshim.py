"""A read-only stand-in for `odyn.Group`, for developing against a local copy."""

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
        max_age_s: float = 0.0,
        allow_live: bool = False,
        busy_timeout_s: float = 120.0,
    ):
        self.db_path = Path(db_path)
        self.main_folder = Path(main_folder)
        self.group_id = 0
        self.snapshot = None

        if snapshot_to is not None:
            from .snapshot import ensure_snapshot

            self.snapshot = ensure_snapshot(
                self.db_path, snapshot_to, refresh=refresh, max_age_s=max_age_s
            )
            self.source_path, self.db_path = self.db_path, Path(snapshot_to)

        elif not allow_live and self._looks_remote(self.db_path):
            raise ValueError(
                f"Refusing to read {self.db_path} directly: it is on a network "
                f"share, and the whole-table reads this class does would hold a "
                f"SQLite lock long enough to fail another machine's writes.\n\n"
                f"Pass snapshot_to=<local path> to copy it first (lock-free, "
                f"validated locally, and refreshed when the live file changes), or "
                f"allow_live=True if you have a specific reason and know the "
                f"queries are small."
            )

        self.con = sqlite3.connect(
            f"file:{self.db_path}?mode=ro", uri=True, timeout=busy_timeout_s
        )
        self.con.row_factory = sqlite3.Row

        self.read_only = True

        # Wait rather than raise if someone else holds the lock.
        self.con.execute(f"PRAGMA busy_timeout = {int(busy_timeout_s * 1000)};")

    @staticmethod
    def _looks_remote(path: Path) -> bool:
        """Is this path on a network mount?"""

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
        """Which experiments belong to which group."""

        return self._table("group_experiments")

    @cached_property
    def odors(self) -> pd.DataFrame:
        return self._table("odors")

    def add_output_file(self, path: str | Path) -> None:
        """No-op: nothing is recorded until the function is `@record_call`ed."""
