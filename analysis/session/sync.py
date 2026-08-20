"""Read the behavior sync H5 and recover true imaging frame times."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from .h5io import open_h5

# A channel sits at 0 V or ~5 V. Half-way is a safe edge threshold and matches
# ENC_THRESH in record_behavior.py.
LOGIC_THRESHOLD_V = 2.5

# Frames arrive at 12-48 Hz, so a real inter-frame gap is >= ~20 ms. Anything
# far below that is contact bounce on the clock line, not a frame.
MIN_FRAME_INTERVAL_S = 0.005


@dataclass(frozen=True)
class SyncFile:
    """Channels and timing from one behavior sync H5."""

    path: Path
    rate_hz: int
    start_time: datetime
    n_samples: int

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.rate_hz

    def sample_times_s(self) -> np.ndarray:
        """Seconds from `start_time`, one per sample."""
        return np.arange(self.n_samples) / self.rate_hz


def open_sync(path: str | Path) -> SyncFile:
    """Read header attributes without loading any channel data."""

    path = Path(path)

    with open_h5(path) as f:
        attrs = dict(f.attrs)

        if "n_samples" in attrs:
            n_samples = int(attrs["n_samples"])
        else:
            channels = [name for name in f if f[name].ndim >= 1]
            if not channels:
                raise KeyError(
                    f"{path.name}: no n_samples attribute and no channels to "
                    f"take a length from."
                )
            # Prefer the frame clock; any channel would do, they share a length.
            preferred = "2pFrameSync" if "2pFrameSync" in channels else channels[0]
            n_samples = int(f[preferred].shape[0])

    return SyncFile(
        path=path,
        rate_hz=int(attrs["rate_hz"]),
        start_time=datetime.fromisoformat(str(attrs["start_time"])),
        n_samples=n_samples,
    )


def read_channel(sync: SyncFile, name: str, column: None | int = None) -> np.ndarray:
    """Load one channel. `column` picks a column from a 2-D dataset."""

    with open_h5(sync.path) as f:
        data = f[name][:]

    if column is not None:
        data = data[:, column]

    return np.asarray(data)


def rising_edges(signal: np.ndarray, *, threshold_v: float = LOGIC_THRESHOLD_V) -> np.ndarray:
    """Sample indices where `signal` crosses `threshold_v` going up."""

    high = signal > threshold_v
    return np.flatnonzero(~high[:-1] & high[1:]) + 1


def frame_onset_samples(
    sync: SyncFile,
    *,
    channel: str = "2pFrameSync",
    min_interval_s: float = MIN_FRAME_INTERVAL_S,
) -> np.ndarray:
    """Sample index of every imaging frame onset in the session."""

    edges = rising_edges(read_channel(sync, channel))

    if edges.size == 0:
        return edges

    min_gap = int(round(min_interval_s * sync.rate_hz))
    keep = [edges[0]]

    for edge in edges[1:]:
        if edge - keep[-1] >= min_gap:
            keep.append(edge)

    return np.asarray(keep)


def group_frames_into_acquisitions(
    frame_samples: np.ndarray,
    *,
    rate_hz: int,
    gap_s: float = 2.0,
) -> list[np.ndarray]:
    """Split a session's frame onsets into per-acquisition blocks."""

    if frame_samples.size == 0:
        return []

    gaps = np.diff(frame_samples) / rate_hz
    breaks = np.flatnonzero(gaps > gap_s) + 1

    return np.split(frame_samples, breaks)


@dataclass(frozen=True)
class OdorAlignment:
    """Where one acquisition's odor pulse lands in that acquisition's frames."""

    on_frame: int
    off_frame: int
    # Seconds between the valve edge and the onset of the frame it was
    # assigned to. Positive means the valve opened after that frame began.
    on_residual_s: float
    off_residual_s: float
    frame_period_s: float
    n_frames: int

    @property
    def duration_frames(self) -> int:
        return self.off_frame - self.on_frame


def odor_frame_alignment(
    frame_blocks: list[np.ndarray],
    odor_pulses: np.ndarray,
    *,
    rate_hz: int,
    align: str = "nearest",
) -> list[None | OdorAlignment]:
    """Anchor each acquisition's odor onset to a frame -- odor time zero."""

    if align not in ("contains", "next", "nearest"):
        raise ValueError(
            f"align must be 'contains', 'next', or 'nearest', got {align!r}."
        )

    onsets = odor_pulses[:, 0] * rate_hz
    offsets = odor_pulses[:, 1] * rate_hz

    out: list[None | OdorAlignment] = []

    for block in frame_blocks:
        inside = np.flatnonzero((onsets >= block[0]) & (onsets <= block[-1]))

        if inside.size == 0:
            out.append(None)
            continue

        i = inside[0]
        period = float(np.median(np.diff(block))) / rate_hz

        def assign(sample: float) -> tuple[int, float]:
            # Index of the last frame that began at or before `sample`.
            below = int(np.searchsorted(block, sample, side="right") - 1)
            below = min(max(below, 0), block.size - 1)

            if align == "contains":
                frame = below
            elif align == "next":
                frame = min(below + 1, block.size - 1) if block[below] < sample else below
            else:
                above = min(below + 1, block.size - 1)
                frame = below if (sample - block[below]) <= (block[above] - sample) else above

            return frame, float(sample - block[frame]) / rate_hz

        on_frame, on_residual = assign(onsets[i])
        off_frame, off_residual = assign(offsets[i])

        out.append(
            OdorAlignment(
                on_frame=on_frame,
                off_frame=off_frame,
                on_residual_s=on_residual,
                off_residual_s=off_residual,
                frame_period_s=period,
                n_frames=int(block.size),
            )
        )

    return out


def acquisition_odor_windows(
    frame_blocks: list[np.ndarray],
    odor_pulses: np.ndarray,
    *,
    rate_hz: int,
    align: str = "nearest",
) -> list[None | tuple[int, int]]:
    """`(on_frame, off_frame)` per acquisition; see `odor_frame_alignment`."""

    return [
        None if a is None else (a.on_frame, a.off_frame)
        for a in odor_frame_alignment(
            frame_blocks, odor_pulses, rate_hz=rate_hz, align=align
        )
    ]


def pulse_intervals(
    signal: np.ndarray,
    *,
    rate_hz: int,
    threshold_v: float = LOGIC_THRESHOLD_V,
) -> np.ndarray:
    """Return (onset_s, offset_s) for every high pulse in `signal`."""

    high = (signal > threshold_v).astype(np.int8)
    change = np.diff(high)

    onsets = np.flatnonzero(change == 1) + 1
    offsets = np.flatnonzero(change == -1) + 1

    # A pulse already open at the start, or still open at the end, has no
    # matching edge; drop it rather than pairing it with the wrong one.
    if offsets.size and onsets.size and offsets[0] < onsets[0]:
        offsets = offsets[1:]

    n = min(onsets.size, offsets.size)

    return np.column_stack([onsets[:n], offsets[:n]]) / rate_hz
