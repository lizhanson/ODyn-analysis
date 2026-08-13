"""
Resumable per-trial accumulation on local disk.

Extraction streams a whole session -- 160 acquisitions, an hour on a busy share
-- and accumulates into one array in RAM. Every trial is independent of every
other, so there is no reason a failure at trial 140 should discard the first
139. It did, twice, on 2026-08-12: once when the SMB mount dropped mid-read
(`OSError: [Errno 6] Device not configured`) and once when a work directory
went away underneath a run.

So the accumulator lives in a local memory-mapped file with a boolean vector
recording which trials are finished. A resumed run reads that vector and skips
what it already has. The cost is small: for 96 ROIs x 160 trials x 301 frames
the checkpoint is 18 MB, and flushing after each trial is nothing against the
seconds each one takes to read off the share.

**Local on purpose.** The point is to survive the network going away, so the
checkpoint cannot live on the network. It also means the per-trial flush is a
local write rather than 160 small writes over SMB.

**Validity is checked, not assumed.** The digest covers the movie files (name,
size, mtime), the mask, and the window geometry. Re-run motion correction, edit
the mask, or change the window and the checkpoint is discarded rather than
resumed -- resuming across a changed mask would produce a trace array whose
early trials came from different ROIs than its late ones, which would load
cleanly and be silently wrong.
"""

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
    neuropil: bool,
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
        "neuropil": bool(neuropil),
        "version": 1,
    }

    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()[:16]


class ExtractionCheckpoint:
    """
    Memory-mapped partial extraction plus the record of what is done.

    Open it, ask `pending()` which trials still need reading, write each result
    with `store()`, and `arrays()` at the end. `discard()` removes it once the
    result is safely written somewhere permanent.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        digest: str,
        shape: tuple[int, int, int],
        neuropil: bool,
    ):
        self.directory = Path(directory)
        self.digest = digest
        self.shape = tuple(int(v) for v in shape)
        self.neuropil = bool(neuropil)

        self.directory.mkdir(parents=True, exist_ok=True)

        self.roi_path = self.directory / "roi.dat"
        self.ring_path = self.directory / "neuropil.dat"
        self.done_path = self.directory / "done.npy"

        self.resumed = self._matches_existing()

        if not self.resumed:
            self._clear()

        mode = "r+" if self.resumed else "w+"

        self.roi = np.memmap(self.roi_path, dtype=np.float32, mode=mode, shape=self.shape)
        self.ring = (
            np.memmap(self.ring_path, dtype=np.float32, mode=mode, shape=self.shape)
            if self.neuropil else None
        )

        if self.resumed:
            self.done = np.load(self.done_path)
        else:
            self.done = np.zeros(self.shape[1], dtype=bool)
            self.roi[:] = np.nan
            if self.ring is not None:
                self.ring[:] = np.nan
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
        if bool(recorded.get("neuropil")) != self.neuropil:
            return False

        # A truncated array is worse than none: it would read as zeros.
        expected = int(np.prod(self.shape)) * 4
        if self.roi_path.stat().st_size != expected:
            return False
        if self.neuropil and (
            not self.ring_path.is_file() or self.ring_path.stat().st_size != expected
        ):
            return False

        return True

    def _clear(self) -> None:
        for path in (self.roi_path, self.ring_path, self.done_path,
                     self.directory / MANIFEST):
            path.unlink(missing_ok=True)

    def _write_manifest(self) -> None:
        (self.directory / MANIFEST).write_text(json.dumps({
            "digest": self.digest,
            "shape": list(self.shape),
            "neuropil": self.neuropil,
        }, indent=2))

    # ------------------------------------------------------------------ #

    def pending(self) -> np.ndarray:
        """Indices of trials still to be read."""
        return np.flatnonzero(~self.done)

    @property
    def n_done(self) -> int:
        return int(self.done.sum())

    def store(self, index: int, roi_values, ring_values=None) -> None:
        """Record one trial and mark it complete."""

        self.roi[:, index, :] = roi_values

        if self.ring is not None and ring_values is not None:
            self.ring[:, index, :] = ring_values

        self.done[index] = True

    def flush(self) -> None:
        """Push the arrays and the done-vector to disk."""

        self.roi.flush()
        if self.ring is not None:
            self.ring.flush()

        # Written last: a done-vector claiming a trial whose data never landed
        # would be the one failure this class exists to prevent.
        #
        # Via an open handle, not a path: `np.save` appends `.npy` to any name
        # that lacks it, so saving to `done.npy.partial` silently produces
        # `done.npy.partial.npy` and the rename below then fails on a file that
        # was never created.
        partial = self.done_path.with_name(self.done_path.name + ".partial")

        with open(partial, "wb") as handle:
            np.save(handle, self.done)

        partial.replace(self.done_path)

    def mark_skipped(self, index: int) -> None:
        """A trial that cannot be read is complete, not pending."""
        self.done[index] = True

    def arrays(self) -> tuple[np.ndarray, None | np.ndarray]:
        """The accumulated data as ordinary in-memory arrays."""

        roi = np.array(self.roi)
        ring = None if self.ring is None else np.array(self.ring)

        return roi, ring

    def discard(self) -> None:
        """Delete the checkpoint; call once the result is written for real."""

        self.roi = None
        self.ring = None
        self._clear()

        try:
            self.directory.rmdir()
        except OSError:
            pass
