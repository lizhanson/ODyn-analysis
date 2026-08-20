"""Trial-averaged pixelwise z-score movies, one per odor."""

from __future__ import annotations

import ast
import hashlib
import json
import re

from pathlib import Path

import numpy as np
import tifffile

from scipy.ndimage import gaussian_filter

# Leave room for the frames being read, the filtered copy, and the OS.
MEMORY_HEADROOM = 0.5


def _available_bytes() -> None | int:
    """Free RAM, or None when it cannot be determined."""
    try:
        import psutil
    except ImportError:
        return None

    return int(psutil.virtual_memory().available)


def _tracked(iterable, *, total: int, description: str, enabled: bool = True):
    """Wrap an iterable in a progress bar, or return it unchanged."""

    if not enabled:
        return iterable

    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable

    return tqdm(iterable, total=total, desc=description, unit="acq", leave=False)


def _use_memory(required_bytes: int, backend: str) -> bool:
    """Decide whether accumulators go in RAM. See `memory_backend`."""

    if backend == "memory":
        return True

    if backend == "disk":
        return False

    if backend != "auto":
        raise ValueError(
            f"memory_backend must be 'auto', 'memory', or 'disk', got {backend!r}."
        )

    available = _available_bytes()

    # Without psutil, take the cautious branch: disk always works.
    if available is None:
        return False

    return required_bytes < available * MEMORY_HEADROOM


def cache_key(
    movie_paths: list[str | Path],
    *,
    odor_on_frames: list[int],
    odor_off_frames: list[int],
    group_keys: list,
    frame_rate: float,
    pre_s: float,
    post_s: float,
    spatial_sigma_px: None | float,
    min_baseline_std: float,
) -> str:
    """Digest of everything that changes the output."""

    payload = {
        "files": [
            [Path(p).name, Path(p).stat().st_size, Path(p).stat().st_mtime_ns]
            for p in movie_paths
        ],
        "on": list(map(int, odor_on_frames)),
        "off": list(map(int, odor_off_frames)),
        "groups": [repr(k) for k in group_keys],
        "frame_rate": round(float(frame_rate), 6),
        "pre_s": float(pre_s),
        "post_s": float(post_s),
        "sigma": None if spatial_sigma_px is None else float(spatial_sigma_px),
        "min_baseline_std": float(min_baseline_std),
        "version": 1,
    }

    blob = json.dumps(payload, sort_keys=True).encode()

    return hashlib.sha256(blob).hexdigest()[:16]


def _load_cached(directory: Path, digest: str) -> None | tuple[dict, dict]:
    """Return cached movies and metadata, or None when the entry does not match."""

    manifest_path = directory / "manifest.json"

    if not manifest_path.is_file():
        return None

    try:
        manifest = json.loads(manifest_path.read_text())

        if manifest.get("digest") != digest:
            return None

        shape = tuple(manifest["shape"])

        movies = {}
        for entry in manifest["groups"]:
            path = directory / entry["file"]
            if not path.is_file() or path.stat().st_size != np.prod(shape) * 4:
                return None
            # literal_eval, not eval: the manifest is a file on disk and
            # must not be able to execute anything.
            movies[ast.literal_eval(entry["key"])] = np.memmap(
                path, dtype=np.float32, mode="r", shape=shape
            )

    except Exception:
        # A half-written or unreadable entry is a miss, not an error: the
        # caller recomputes and overwrites it.
        return None

    meta = dict(manifest["meta"])
    meta["cache"] = "hit"

    return movies, meta


def _slug(key) -> str:
    """Filesystem-safe name for a group key, which may be a tuple."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", repr(key))


def make_group_keys(
    odor_ids: list[int],
    states: list,
    *,
    separate_by_condition: bool = False,
) -> list:
    """Group keys for `build_group_zscore_movies`, with or without state split."""

    if len(odor_ids) != len(states):
        raise ValueError(
            f"odor_ids and states must be the same length, "
            f"got {len(odor_ids)} and {len(states)}."
        )

    if not separate_by_condition:
        return list(odor_ids)

    return [(odor, state) for odor, state in zip(odor_ids, states)]


def _window_bounds(on_frame: int, *, n_pre: int, n_span: int) -> tuple[int, int]:
    """Fixed-length window anchored on odor onset."""
    return on_frame - n_pre, on_frame - n_pre + n_span


def build_group_zscore_movies(
    movie_paths: list[str | Path],
    *,
    odor_on_frames: list[int],
    odor_off_frames: list[int],
    group_keys: list,
    frame_rate: float,
    pre_s: float = 2.0,
    post_s: float = 2.0,
    spatial_sigma_px: None | float = None,
    min_baseline_std: float = 1e-6,
    memory_backend: str = "auto",
    work_dir: str | Path,
    cache_dir: None | str | Path = None,
    refresh_cache: bool = False,
    progress: bool = True,
) -> tuple[dict[int, np.ndarray], dict]:
    """Stream the session once and return a trial-averaged z-score movie per group."""

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    entry = None
    if cache_dir is not None:
        digest = cache_key(
            movie_paths,
            odor_on_frames=odor_on_frames,
            odor_off_frames=odor_off_frames,
            group_keys=group_keys,
            frame_rate=frame_rate,
            pre_s=pre_s,
            post_s=post_s,
            spatial_sigma_px=spatial_sigma_px,
            min_baseline_std=min_baseline_std,
        )
        entry = Path(cache_dir)

        if not refresh_cache:
            cached = _load_cached(entry, digest)
            if cached is not None:
                return cached

    n_pre = int(round(pre_s * frame_rate))
    n_post = int(round(post_s * frame_rate))

    odor_lengths = [off - on for on, off in zip(odor_on_frames, odor_off_frames)]
    n_odor = int(round(np.median(odor_lengths)))
    n_span = n_pre + n_odor + n_post

    with tifffile.TiffFile(str(movie_paths[0])) as tf:
        height, width = tf.series[0].shape[1:]

    unique = sorted(set(group_keys), key=repr)

    shape = (n_span, height, width)
    required = int(np.prod(shape)) * 4 * len(unique)
    in_memory = _use_memory(required, memory_backend)

    if in_memory:
        sums = {key: np.zeros(shape, dtype=np.float32) for key in unique}
    else:
        sums = {
            key: np.memmap(
                work_dir / f"zsum_{_slug(key)}.dat",
                dtype=np.float32,
                mode="w+",
                shape=shape,
            )
            for key in unique
        }

    counts = {key: 0 for key in unique}
    skipped = []

    for path, on_frame, off_frame, key in _tracked(
        zip(movie_paths, odor_on_frames, odor_off_frames, group_keys),
        total=len(movie_paths),
        description="streaming acquisitions",
        enabled=progress,
    ):
        start, stop = _window_bounds(on_frame, n_pre=n_pre, n_span=n_span)

        with tifffile.TiffFile(str(path)) as tf:
            total = tf.series[0].shape[0]

            if start < 0 or stop > total:
                skipped.append(
                    {"file": Path(path).name, "start": start, "stop": stop, "total": total}
                )
                continue

            stack = tf.series[0].asarray(key=slice(start, stop)).astype(np.float32)

        if spatial_sigma_px:
            # sigma 0 on the time axis: smooth within frames only.
            stack = gaussian_filter(stack, sigma=(0, spatial_sigma_px, spatial_sigma_px))

        baseline = stack[:n_pre]
        mu = baseline.mean(axis=0)
        sd = baseline.std(axis=0, ddof=1)

        safe = np.where(sd >= min_baseline_std, sd, np.inf)

        sums[key] += (stack - mu) / safe
        counts[key] += 1

        del stack

    movies = {}
    for key in unique:
        if counts[key]:
            sums[key] /= counts[key]

        if in_memory:
            movies[key] = sums[key]
        else:
            sums[key].flush()

            view = sums[key][:]
            view.flags.writeable = False        # read-only, as before
            movies[key] = view

    meta = {
        "n_pre": n_pre,
        "n_odor": n_odor,
        "n_post": n_post,
        "n_span": n_span,
        "frame_rate": float(frame_rate),
        "spatial_sigma_px": spatial_sigma_px,
        "trials_per_group": {repr(k): int(v) for k, v in counts.items()},
        "skipped": skipped,
        "memory_backend": "memory" if in_memory else "disk",
        "accumulator_gb": round(required / 1e9, 2),
        "cache": "miss" if entry is not None else "off",
    }

    if entry is not None:
        movies = _write_cache(entry, movies, meta=meta, shape=shape, digest=digest)

        _discard_accumulators(sums, work_dir, in_memory=in_memory)

    return movies, meta


def _discard_accumulators(sums: dict, work_dir: Path, *, in_memory: bool) -> None:
    """Close the disk-backed accumulators and delete their files."""

    if in_memory:
        return

    for key in list(sums):
        accumulator = sums.pop(key)
        if hasattr(accumulator, "_mmap") and accumulator._mmap is not None:
            accumulator._mmap.close()

    for stale in work_dir.glob("zsum_*.dat"):
        stale.unlink(missing_ok=True)

    # Only if it is now empty; the caller may have put other things here.
    try:
        work_dir.rmdir()
    except OSError:
        pass


def _write_cache(
    directory: Path, movies: dict, *, meta: dict, shape: tuple, digest: str
) -> dict:
    """Persist the movies, then the manifest, replacing whatever was there."""

    directory.mkdir(parents=True, exist_ok=True)

    stale_manifest = directory / "manifest.json"
    if stale_manifest.is_file():
        stale_manifest.unlink()

    # Group names change with the grouping, so clear every array rather than
    # only the ones about to be rewritten -- otherwise switching from
    # (odor, state) back to odor leaves the old per-state files behind.
    for old in directory.glob("z_*.dat"):
        old.unlink()

    groups = []
    out = {}

    for key, movie in movies.items():
        name = f"z_{_slug(key)}.dat"
        target = directory / name

        buffer = np.memmap(target, dtype=np.float32, mode="w+", shape=shape)
        buffer[:] = np.asarray(movie)
        buffer.flush()
        del buffer

        out[key] = np.memmap(target, dtype=np.float32, mode="r", shape=shape)
        groups.append({"key": repr(key), "file": name})

    manifest = {
        "digest": digest,
        "shape": list(shape),
        "groups": groups,
        "meta": {k: v for k, v in meta.items() if k != "cache"},
    }

    temporary = directory / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, default=str))
    temporary.replace(directory / "manifest.json")

    return out
