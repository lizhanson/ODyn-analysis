"""
Opening HDF5 files that live on the SMB share.

HDF5 1.10+ takes an advisory lock on every file it opens, including read-only.
On a network filesystem that lock is unreliable in both directions: it can fail
outright while nothing else holds the file, and it cannot be trusted when
something does. The failure surfaces as

    BlockingIOError: [Errno 35] unable to lock file, errno = 35,
    'Resource temporarily unavailable'

which is not a permissions problem, not a missing file, and not fixed by
retrying -- and it can appear on a file that opened perfectly a minute earlier.

`locking=False` turns it off. For readers that costs nothing: no reader has
ever been protected by it in a way that mattered, and the data is not changing
underneath. For writers it means concurrent writers to the *same file* are no
longer coordinated -- but two processes writing one session's output is already
a mistake, and the dated filenames make it unlikely by construction.

Everything here should be opened through this module rather than `h5py.File`
directly, so a session on the share behaves the same as one staged locally.
"""

from __future__ import annotations

import logging
import os

from pathlib import Path


class _H5Handle:
    """
    Context manager for one open HDF5 file. See `open_h5`.

    Written out rather than using `@contextlib.contextmanager` because this
    module is hot-reloaded constantly. The decorator returns a closure that
    resolves `_GeneratorContextManager` from `contextlib`'s globals when the
    context is entered, and under `%autoreload` that lookup can fail:

        NameError: name '_GeneratorContextManager' is not defined

    raised from inside contextlib itself, on a line of code nobody touched.
    The protocol is four lines to implement directly and depends on nothing
    that reloading can disturb.
    """

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
    """
    `h5py.File` with locking disabled, tolerating a failed close on reads.

    `locking=` arrived in h5py 3.5. On anything older the equivalent is the
    HDF5_USE_FILE_LOCKING environment variable, which HDF5 reads once at
    library load -- so setting it here only helps if nothing has opened a file
    yet. Both paths are covered; the fallback is best-effort.

    **Close failures on this share.** Reading the June sync files off SMB
    raises on *close*, not on open or read:

        RuntimeError: Can't decrement id ref count
        (unable to close file, errno = 89, 'Operation canceled')

    It happens with locking on, locking off, and with the environment variable
    set, on files that read perfectly a minute earlier, while plain reads of
    the same bytes run at 8 MB/s -- so it is the SMB client's handling of
    HDF5's close sequence, not the data and not the mount.

    For a reader that is noise: every byte wanted has already been returned,
    and the only cost of a failed close is a descriptor held until the process
    exits. Letting it propagate aborted whole sessions over a filesystem quirk.

    For a writer it is not noise -- a close is where buffered data is flushed,
    so a failure there may mean the file on disk is incomplete. Write modes
    therefore still raise.

    Use it as a context manager: `with open_h5(path) as f: ...`
    """

    return _H5Handle(Path(path), mode, kwargs)
