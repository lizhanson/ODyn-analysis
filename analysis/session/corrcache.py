"""
Cache the correlation maps, not the movies they came from.

The z-score movies are an intermediate. Nothing downstream reads them: they
exist to be collapsed, once, into one local-correlation map per group, and the
segmentation, the GUI and the merge all work from those maps alone.

Caching the intermediate instead of the product is expensive in a way that only
shows up on a busy share. Exp 213, 16 odors with conditions separated:

    z-score movies    32 x 284 MB  =  9.09 GB   ~56 min to write at 2.7 MB/s
    correlation maps  32 x 1.26 MB =  40.4 MB   ~15 s

A 225x ratio, and on that run the write phase was more than half the total
time -- spent on data nothing reads. The accumulators still have to be built,
so the streaming cost is unchanged; what disappears is pushing 9 GB over SMB
and back again.

The trade is that the movies are gone once the maps are computed. That is
deliberate: they can be rebuilt by re-streaming, and the alternative -- keeping
them "just in case" -- is what cost the 56 minutes. Anything that genuinely
needs per-pixel time courses should build them explicitly with
`build_group_zscore_movies`.

The digest is the movie cache's, plus the correlation parameters, so changing
either the z-scoring or the correlation invalidates the maps. Re-running motion
correction invalidates them too, since file size and mtime are in the digest.
"""

from __future__ import annotations

import hashlib
import json

from pathlib import Path

import numpy as np

CACHE_NAME = "correlation_cache"
MANIFEST = "manifest.json"
ARRAYS = "correlation_maps.npz"

# Bumped when the correlation itself changes, so old maps are not reused
# against new code that computes something different.
CORRELATION_VERSION = 1


def correlation_key(movie_digest: str) -> str:
    """The movie digest, extended by what turns a movie into a map."""

    blob = json.dumps(
        {"movies": movie_digest, "correlation_version": CORRELATION_VERSION},
        sort_keys=True,
    ).encode()

    return hashlib.sha256(blob).hexdigest()[:16]


def load_cached_maps(directory: str | Path, digest: str) -> None | tuple[dict, dict]:
    """Maps and metadata for this digest, or None."""

    directory = Path(directory)
    manifest = directory / MANIFEST
    arrays = directory / ARRAYS

    if not (manifest.exists() and arrays.exists()):
        return None

    try:
        recorded = json.loads(manifest.read_text())
    except json.JSONDecodeError:
        return None

    if recorded.get("digest") != digest:
        return None

    with np.load(arrays, allow_pickle=False) as data:
        # Keys are stored as their repr, which is what `_slug` and the rest of
        # the pipeline already use to name groups on disk.
        maps = {
            _unrepr(name): data[name] for name in data.files
        }

    return maps, recorded.get("meta", {})


def write_cached_maps(
    directory: str | Path, maps: dict, *, meta: dict, digest: str
) -> Path:
    """
    Replace whatever is cached here with these maps.

    Manifest last and atomic, matching the movie cache: an interrupted write
    leaves arrays with no manifest, which reads as a miss, rather than a
    manifest pointing at data that does not match it.
    """

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    manifest = directory / MANIFEST
    manifest.unlink(missing_ok=True)

    np.savez_compressed(
        directory / ARRAYS, **{repr(k): np.asarray(v) for k, v in maps.items()}
    )

    partial = manifest.with_suffix(".json.partial")
    partial.write_text(json.dumps(
        {"digest": digest, "meta": meta,
         "groups": [repr(k) for k in sorted(maps, key=repr)]},
        indent=2, default=str,
    ))
    partial.replace(manifest)

    return directory / ARRAYS


def _unrepr(name: str):
    """Turn a stored key name back into the key, falling back to the string."""

    import ast

    try:
        return ast.literal_eval(name)
    except (ValueError, SyntaxError):
        return name


def build_group_correlation_maps(
    movie_paths: list[str | Path],
    *,
    odor_on_frames: list[int],
    odor_off_frames: list[int],
    group_keys: list,
    frame_rate: float,
    work_dir: str | Path,
    cache_dir: None | str | Path = None,
    pre_s: float = 2.0,
    post_s: float = 2.0,
    spatial_sigma_px: None | float = None,
    min_baseline_std: float = 1e-6,
    memory_backend: str = "auto",
    refresh_cache: bool = False,
    keep_movies: bool = False,
    progress: bool = True,
) -> tuple[dict, dict]:
    """
    One local-correlation map per group, cached.

    On a hit this reads ~40 MB and returns in seconds. On a miss it streams the
    session once, accumulates the z-score movies into `work_dir` on local disk,
    collapses them to maps, caches the maps, and deletes the accumulators.

    `keep_movies=True` returns the movies alongside the maps for the rare case
    that wants them; they still are not cached.
    """

    from .summary import local_correlation
    from .zscore import build_group_zscore_movies, cache_key

    movie_digest = cache_key(
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
    digest = correlation_key(movie_digest)

    entry = None if cache_dir is None else Path(cache_dir)

    if entry is not None and not refresh_cache:
        cached = load_cached_maps(entry, digest)
        if cached is not None:
            maps, meta = cached
            meta = {**meta, "cache": "hit"}
            return maps, meta

    # Build the movies with no movie cache of their own: the accumulators are
    # local, temporary, and deleted below.
    movies, meta = build_group_zscore_movies(
        movie_paths,
        odor_on_frames=odor_on_frames,
        odor_off_frames=odor_off_frames,
        group_keys=group_keys,
        frame_rate=frame_rate,
        pre_s=pre_s,
        post_s=post_s,
        spatial_sigma_px=spatial_sigma_px,
        min_baseline_std=min_baseline_std,
        memory_backend=memory_backend,
        work_dir=work_dir,
        cache_dir=None,
        progress=progress,
    )

    from .zscore import _tracked

    ordered = sorted(movies, key=repr)
    maps = {
        key: local_correlation(np.asarray(movies[key]))
        for key in _tracked(
            ordered, total=len(ordered),
            description="correlation maps", enabled=progress,
        )
    }

    meta = {**meta, "cache": "miss" if entry is not None else "off",
            "correlation_version": CORRELATION_VERSION}

    if entry is not None:
        write_cached_maps(entry, maps, meta=meta, digest=digest)

    if not keep_movies:
        # Drop the memmaps before deleting their files, so nothing is left
        # holding a descriptor on a path that no longer exists.
        movies.clear()
        _remove_accumulators(Path(work_dir))
        return maps, meta

    return maps, {**meta, "movies": movies}


def convert_movie_cache(
    movie_cache_dir: str | Path,
    correlation_cache_dir: None | str | Path = None,
    *,
    delete_movies: bool = False,
) -> dict:
    """
    Turn an existing movie cache into a correlation cache, without re-streaming.

    For sessions cached before this module existed. Reading 9 GB back off the
    share is slow, but it is a one-off, and it is the difference between
    keeping that 9 GB forever and keeping 40 MB.

    The movie cache's own digest is reused, so the maps validate against a
    later call with the same parameters -- the conversion produces exactly what
    a fresh run would have.

    `delete_movies` removes the source arrays afterwards. Off by default: the
    conversion should be verified before anything is thrown away.
    """

    from .summary import local_correlation

    movie_cache_dir = Path(movie_cache_dir)
    manifest_path = movie_cache_dir / "manifest.json"

    if not manifest_path.is_file():
        return {"ok": False, "reason": f"no manifest in {movie_cache_dir}"}

    manifest = json.loads(manifest_path.read_text())
    movie_digest = manifest.get("digest")

    from .zscore import _load_cached

    loaded = _load_cached(movie_cache_dir, movie_digest)
    if loaded is None:
        return {"ok": False, "reason": "movie cache is incomplete or unreadable"}

    movies, meta = loaded
    target = Path(correlation_cache_dir or movie_cache_dir.parent / CACHE_NAME)

    from .zscore import _tracked

    ordered = sorted(movies, key=repr)
    maps = {
        key: local_correlation(np.asarray(movies[key]))
        for key in _tracked(
            ordered, total=len(ordered),
            description="converting cached movies", enabled=True,
        )
    }

    digest = correlation_key(movie_digest)
    write_cached_maps(target, maps, meta={**meta, "converted_from": str(movie_cache_dir)},
                      digest=digest)

    before = sum(p.stat().st_size for p in movie_cache_dir.glob("z_*.dat"))
    after = sum(p.stat().st_size for p in target.iterdir())

    removed = 0
    if delete_movies:
        movies.clear()
        for stale in movie_cache_dir.glob("z_*.dat"):
            stale.unlink(missing_ok=True)
            removed += 1
        manifest_path.unlink(missing_ok=True)
        try:
            movie_cache_dir.rmdir()
        except OSError:
            pass

    return {
        "ok": True,
        "groups": len(maps),
        "correlation_cache": str(target),
        "movies_gb": round(before / 1e9, 2),
        "maps_mb": round(after / 1e6, 2),
        "ratio": round(before / max(after, 1)),
        "movies_deleted": removed,
    }


def _remove_accumulators(work_dir: Path) -> None:
    """Delete the local z-score accumulators and the directory holding them."""

    for stale in work_dir.glob("zsum_*.dat"):
        stale.unlink(missing_ok=True)

    try:
        work_dir.rmdir()
    except OSError:
        pass
