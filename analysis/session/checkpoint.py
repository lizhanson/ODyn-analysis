"""Resumable per-trial accumulation on local disk."""

from __future__ import annotations

import hashlib
import json

from pathlib import Path

import numpy as np

MANIFEST = "manifest.json"


def checkpoint_key(
    movie_paths: list[str | Path],
    *,
    mask_hash: str,
    shape: tuple[int, int, int],
    starts: list[int],
) -> str:
    """Digest of everything that would invalidate a partial extraction."""

    payload = {
        "files": [
            [Path(p).name, Path(p).stat().st_size, Path(p).stat().st_mtime_ns]
            for p in movie_paths
        ],
        "mask": mask_hash,
        "shape": list(shape),
        "starts": [int(s) for s in starts],
        "version": 2,
    }

    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()[:16]


class ExtractionCheckpoint:
    """Memory-mapped partial extraction plus the record of what is done."""

    def __init__(
        self,
        directory: str | Path,
        *,
        digest: str,
        shape: tuple[int, int, int],
    ):
        self.directory = Path(directory)
        self.digest = digest
        self.shape = tuple(int(v) for v in shape)

        self.directory.mkdir(parents=True, exist_ok=True)

        self.roi_path = self.directory / "roi.dat"
        self.done_path = self.directory / "done.npy"

        self.resumed = self._matches_existing()

        if not self.resumed:
            self._clear()

        mode = "r+" if self.resumed else "w+"

        self.roi = np.memmap(self.roi_path, dtype=np.float32, mode=mode, shape=self.shape)
        if self.resumed:
            self.done = np.load(self.done_path)
        else:
            self.done = np.zeros(self.shape[1], dtype=bool)
            self.roi[:] = np.nan
            self._write_manifest()

    # ------------------------------------------------------------------ #

    def _matches_existing(self) -> bool:
        manifest = self.directory / MANIFEST

        if not (manifest.is_file() and self.roi_path.is_file()
                and self.done_path.is_file()):
            return False

        try:
            recorded = json.loads(manifest.read_text())
        except json.JSONDecodeError:
            return False

        if recorded.get("digest") != self.digest:
            return False
        if tuple(recorded.get("shape", ())) != self.shape:
            return False
        # A truncated array is worse than none: it would read as zeros.
        expected = int(np.prod(self.shape)) * 4
        if self.roi_path.stat().st_size != expected:
            return False
        return True

    def _clear(self) -> None:
        for path in (self.roi_path, self.done_path,
                     self.directory / MANIFEST):
            path.unlink(missing_ok=True)

    def _write_manifest(self) -> None:
        (self.directory / MANIFEST).write_text(json.dumps({
            "digest": self.digest,
            "shape": list(self.shape),
        }, indent=2))

    # ------------------------------------------------------------------ #

    def pending(self) -> np.ndarray:
        """Indices of trials still to be read."""
        return np.flatnonzero(~self.done)

    @property
    def n_done(self) -> int:
        return int(self.done.sum())

    def store(self, index: int, roi_values) -> None:
        """Record one trial and mark it complete."""

        self.roi[:, index, :] = roi_values

        self.done[index] = True

    def flush(self) -> None:
        """Push the arrays and the done-vector to disk."""

        self.roi.flush()
        partial = self.done_path.with_name(self.done_path.name + ".partial")

        with open(partial, "wb") as handle:
            np.save(handle, self.done)

        partial.replace(self.done_path)

    def mark_skipped(self, index: int) -> None:
        """A trial that cannot be read is complete, not pending."""
        self.done[index] = True

    def array(self) -> np.ndarray:
        """The accumulated data as an ordinary in-memory array."""
        return np.array(self.roi)

    def discard(self) -> None:
        """Delete the checkpoint; call once the result is written for real."""

        self.roi = None
        self._clear()

        try:
            self.directory.rmdir()
        except OSError:
            pass
