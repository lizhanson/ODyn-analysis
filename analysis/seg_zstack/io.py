"""Read repeated-frame ScanImage z-stacks without loading raw stacks twice."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _progress(iterable, *, total, desc, enabled):
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit="plane")


def _frame_value(frame_data, name, default=None):
    value = frame_data.get(name, default)
    if isinstance(value, np.ndarray) and value.size == 1:
        return value.item()
    return value


def load_scanimage_zstack(path, *, frames_per_plane=None, channel=0, progress=True):
    """Average repeated frames at each plane and return ``(Z,Y,X)`` float32.

    ScanImage metadata is authoritative when available. ``frames_per_plane`` is
    an explicit fallback/override for older TIFFs. Multi-channel input must have
    a tifffile series axis named ``C``; ``channel`` is zero based.
    """
    import tifffile

    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    with tifffile.TiffFile(path) as tif:
        metadata = tif.scanimage_metadata or {}
        frame_data = metadata.get("FrameData", {})
        recorded = _frame_value(frame_data, "SI.hStackManager.framesPerSlice")
        repeats = int(frames_per_plane if frames_per_plane is not None else recorded or 0)
        if repeats < 1:
            raise ValueError("frames_per_plane is absent from metadata; pass it explicitly")
        z_step = _frame_value(frame_data, "SI.hStackManager.actualStackZStepSize")
        if z_step in (None, 0):
            z_step = _frame_value(frame_data, "SI.hStackManager.stackZStepSize")

        series = tif.series[0]
        axes, shape = series.axes, series.shape
        if "C" in axes:
            raw = series.asarray()
            raw = np.take(raw, int(channel), axis=axes.index("C"))
            axes = axes.replace("C", "")
        elif int(channel) != 0:
            raise ValueError(f"TIFF has no channel axis ({axes}); channel must be 0")
        elif axes.endswith("ZTYX"):
            # Reading a plane at a time gives useful progress (and avoids a
            # second full-size raw stack) for repeated-frame z-stacks.
            n_planes = int(shape[axes.index("Z")])
            if int(shape[axes.index("T")]) != repeats:
                raise ValueError(
                    f"metadata says {repeats} frames/plane but series shape is {shape} ({axes})"
                )
            planes = []
            for z in _progress(range(n_planes), total=n_planes,
                               desc="Loading z-stack", enabled=progress):
                block = series.asarray(key=slice(z * repeats, (z + 1) * repeats))
                planes.append(block.mean(axis=0, dtype=np.float32))
            averaged = np.stack(planes)
            raw = None
        else:
            raw = series.asarray()

    if raw is None:
        pass
    elif axes.endswith("ZTYX"):
        z_axis, t_axis = axes.index("Z"), axes.index("T")
        raw = np.moveaxis(raw, (z_axis, t_axis), (0, 1))
        if raw.shape[1] != repeats:
            raise ValueError(
                f"metadata says {repeats} frames/plane but series shape is {shape} ({axes})"
            )
        averaged = raw.mean(axis=1, dtype=np.float32)
    elif axes.endswith("TYX") or axes.endswith("IYX") or raw.ndim == 3:
        n_frames = raw.shape[0]
        if n_frames % repeats:
            raise ValueError(f"{n_frames} frames is not divisible by {repeats} frames/plane")
        averaged = raw.reshape(n_frames // repeats, repeats, *raw.shape[-2:]).mean(
            axis=1, dtype=np.float32
        )
    else:
        raise ValueError(f"unsupported TIFF series shape {shape} with axes {axes!r}")

    meta = {
        "source_path": str(path.resolve()),
        "source_axes": series.axes,
        "source_shape": tuple(int(v) for v in series.shape),
        "n_planes": int(averaged.shape[0]),
        "frames_per_plane": repeats,
        "z_step_um": None if z_step is None else abs(float(z_step)),
        "channel": int(channel),
        "averaging": "arithmetic mean of repeated frames within each ScanImage slice",
    }
    return np.asarray(averaged, np.float32), meta
