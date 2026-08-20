"""Opening HDF5 files that live on the SMB share."""

from __future__ import annotations

import logging
import os

from pathlib import Path


class _H5Handle:
    """Context manager for one open HDF5 file."""

    def __init__(self, path: Path, mode: str, kwargs: dict):
        self.path = path
        self.mode = mode
        self.kwargs = kwargs
        self.handle = None

    def __enter__(self):
        import h5py

        try:
            self.handle = h5py.File(self.path, self.mode, locking=False, **self.kwargs)
        except TypeError:
            # h5py too old for the `locking` argument.
            os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
            self.handle = h5py.File(self.path, self.mode, **self.kwargs)

        return self.handle

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.handle.close()
        except (RuntimeError, OSError):
            if self.mode != "r":
                raise
            logging.getLogger(__name__).debug(
                "ignored close failure on read-only %s", self.path.name
            )

        return False


def open_h5(path: str | Path, mode: str = "r", **kwargs):
    """`h5py.File` with locking disabled, tolerating a failed close on reads."""

    return _H5Handle(Path(path), mode, kwargs)
