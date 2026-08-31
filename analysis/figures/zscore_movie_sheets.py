"""Contact sheets built from the pre-rendered per-odor z-score movies.

These movies live under `<imaging root>/movies/{10x,20x}zscores` as one MP4 per
odor per block. They are a *rendering*, not data: the z scores have already
been passed through a diverging colormap with limits baked in, then through a
lossy H.264 encode, and a red stimulus marker is burned into the top-left
corner. Nothing measured should be read off them.

What they are very good for is deciding which panels are worth the expensive
treatment. One odor-block movie is ~34 MB against roughly 1.5 GB of
motion-corrected TIFF for the same panel computed properly, so a whole session
can be browsed for about the cost of one honest panel. Pick from the sheets,
then compute those few with `example_images`.

The odor period is read from the burned-in marker rather than from the trial
table, because the marker is in the movie's own frame coordinates and the
movies are trimmed relative to the acquisition.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from analysis.figures.paths import imaging_root, output_root, repo_path

# Each window is (start anchor, start offset seconds, stop anchor, stop offset).
# Anchors are the first and last frame the stimulus marker is lit.
MOVIE_WINDOWS = {
    "odor": ("onset", 0.0, "offset", 0.0),
    "odor_first_1s": ("onset", 0.0, "onset", 1.0),
    "odor_last_1s": ("offset", -1.0, "offset", 0.0),
    "post_offset_0.25_2.25s": ("offset", 0.25, "offset", 2.25),
}

WINDOW_TITLES = {
    "odor": "whole odor period",
    "odor_first_1s": "first 1 s of odor",
    "odor_last_1s": "last 1 s of odor",
    "post_offset_0.25_2.25s": "0.25-2.25 s after odor offset",
}

SESSION_PATTERN = re.compile(r"^(\d+)_")
ODOR_PATTERN = re.compile(r"_odor_(\d+)_")


def movie_root(root, objective="10x") -> Path:
    return Path(root) / "movies" / f"{objective.lower()}zscores"


def session_directories(root, objective="10x") -> dict[int, Path]:
    """Map group id to its movie directory."""
    base = movie_root(root, objective)
    if not base.is_dir():
        raise FileNotFoundError(f"no z-score movie directory at {base}")
    found = {}
    for entry in sorted(base.iterdir()):
        match = SESSION_PATTERN.match(entry.name)
        if entry.is_dir() and match:
            found[int(match.group(1))] = entry
    return found


def block_movies(session_dir) -> dict[str, dict[int, Path]]:
    """Map block name to {odor id: movie path}.

    Block directories are named `<block>_<program id>`; the program id varies
    between sessions and carries no meaning here.
    """
    output = {}
    for entry in sorted(Path(session_dir).iterdir()):
        if not entry.is_dir():
            continue
        block = entry.name.split("_")[0]
        odors = {}
        for movie in sorted(entry.glob("*.mp4")):
            match = ODOR_PATTERN.search(movie.name)
            if match:
                odors[int(match.group(1))] = movie
        if odors:
            output[block] = odors
    return output


def read_movie(path):
    """Decode one movie to (frames as uint8 RGB, frames per second)."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"could not open {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise ValueError(f"{Path(path).name} decoded to no frames")
    return np.stack(frames), fps


def stimulus_span(frames, *, corner=44):
    """First and last frame the burned-in marker is lit, and its pixel mask.

    The marker is a saturated red dot in the top-left corner, which no part of
    the diverging colormap reaches, so a plain colour test separates it from
    the strongest excitatory response.
    """
    block = np.asarray(frames)[:, :corner, :corner, :].astype(np.int16)
    lit = ((block[..., 0] > 190) & (block[..., 1] < 100) & (block[..., 2] < 100))
    per_frame = lit.reshape(len(block), -1).sum(1)
    on = np.flatnonzero(per_frame > max(4, int(0.05 * corner * corner)))
    if not on.size:
        raise ValueError("no stimulus marker found; cannot locate the odor period")
    mask = np.zeros(np.asarray(frames).shape[1:3], bool)
    mask[:corner, :corner] = lit[on].any(0)
    return int(on.min()), int(on.max()), mask


def window_frames(name, onset, offset, fps, n_frame):
    """Frame span for a named window, clipped to the movie."""
    if name not in MOVIE_WINDOWS:
        raise ValueError(f"window must be one of {sorted(MOVIE_WINDOWS)}, got {name!r}")
    start_anchor, start_offset, stop_anchor, stop_offset = MOVIE_WINDOWS[name]
    anchors = {"onset": onset, "offset": offset}
    start = anchors[start_anchor] + int(round(start_offset * fps))
    stop = anchors[stop_anchor] + int(round(stop_offset * fps))
    start, stop = max(int(start), 0), min(int(stop) + 1, int(n_frame))
    if stop - start < 1:
        raise ValueError(f"window {name!r} is empty in a {n_frame}-frame movie")
    return start, stop


def mask_marker(image, mask):
    """Replace the burned-in marker with the colour around it."""
    image = np.array(image, np.uint8, copy=True)
    if not mask.any():
        return image
    rows, cols = np.where(mask)
    pad = 6
    r0, r1 = max(rows.min() - pad, 0), min(rows.max() + pad + 1, image.shape[0])
    c0, c1 = max(cols.min() - pad, 0), min(cols.max() + pad + 1, image.shape[1])
    patch = image[r0:r1, c0:c1]
    ring = ~mask[r0:r1, c0:c1]
    if ring.any():
        image[mask] = np.median(patch[ring], axis=0).astype(np.uint8)
    return image


def window_means(path, windows=tuple(MOVIE_WINDOWS)):
    """Mean rendered frame over each named window, with the marker removed.

    Averaging colormapped RGB is not the colormap of the averaged z, but on a
    diverging map centred near neutral it is visually close and it removes most
    of the per-frame compression speckle, which is what makes these browsable.
    """
    frames, fps = read_movie(path)
    onset, offset, mask = stimulus_span(frames)
    output = {}
    for name in windows:
        start, stop = window_frames(name, onset, offset, fps, len(frames))
        mean = frames[start:stop].astype(np.float32).mean(0)
        output[name] = mask_marker(np.round(mean).astype(np.uint8), mask)
    return output, {"fps": fps, "n_frame": len(frames), "onset": onset,
                    "offset": offset,
                    "odor_s": round((offset - onset + 1) / fps, 2)}


ODOR_ORDER = (0, 1, 2, 3, 4, 10, 12, 17, 18, 21, 22, 30, 31, 32, 39, 40)


def odor_names(manifest_dir=None) -> dict[int, str]:
    import csv

    path = (Path(manifest_dir) if manifest_dir else
            repo_path("analysis", "stage0")) / "odor_dictionary.csv"
    # Labels are cosmetic; a sheet titled by odor id alone is still usable, so
    # an unreadable dictionary must not cost a whole run.
    try:
        with path.open(newline="") as stream:
            return {int(r["odor_id"]): r["odor_name"] for r in csv.DictReader(stream)}
    except (OSError, KeyError, ValueError):
        return {}


def contact_sheet(path, panels, *, title, names=None, columns=4):
    """One page of odors for a single group, block, and window."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = names or {}
    order = [o for o in ODOR_ORDER if o in panels]
    order += [o for o in sorted(panels) if o not in order]
    rows = int(np.ceil(len(order) / columns)) or 1
    fig, axes = plt.subplots(rows, columns, figsize=(3.0 * columns, 3.3 * rows),
                             constrained_layout=True, squeeze=False)
    for ax, odor in zip(axes.ravel(), order):
        ax.imshow(panels[odor], interpolation="nearest")
        label = names.get(odor, "")
        ax.set(xticks=[], yticks=[],
               title=f"{odor}  {label}"[:34] if label else f"odor {odor}")
        ax.title.set_fontsize(9)
    for ax in axes.ravel()[len(order):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--groups", nargs="+", type=int, required=True)
    parser.add_argument("--objective", default="10x")
    parser.add_argument("--blocks", nargs="+", default=["all"])
    parser.add_argument("--windows", nargs="+", default=list(MOVIE_WINDOWS))
    parser.add_argument("--imaging-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path,
                        default=output_root() / "contact_sheets")
    args = parser.parse_args(argv)

    unknown = [w for w in args.windows if w not in MOVIE_WINDOWS]
    if unknown:
        raise ValueError(f"unknown windows {unknown}; choose from {sorted(MOVIE_WINDOWS)}")
    root = imaging_root(args.imaging_root)
    sessions = session_directories(root, args.objective)
    missing = sorted(set(args.groups) - set(sessions))
    if missing:
        raise FileNotFoundError(
            f"no z-score movies for groups {missing}; have {sorted(sessions)}")
    names = odor_names()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for group_id in args.groups:
        session = sessions[group_id]
        blocks = block_movies(session)
        wanted = list(blocks) if args.blocks == ["all"] else args.blocks
        for block in wanted:
            if block not in blocks:
                print(f"group {group_id}: no {block!r} block, skipping", flush=True)
                continue
            movies = blocks[block]
            collected = {name: {} for name in args.windows}
            info = None
            for odor, movie in sorted(movies.items()):
                try:
                    means, info = window_means(movie, args.windows)
                except (ValueError, FileNotFoundError) as error:
                    print(f"  group {group_id} {block} odor {odor}: "
                          f"{type(error).__name__}: {error}", flush=True)
                    continue
                for name, image in means.items():
                    collected[name][odor] = image
            print(f"group {group_id} {block}: {len(movies)} odors, "
                  f"{info['n_frame'] if info else '?'} frames, "
                  f"odor {info['odor_s'] if info else '?'} s", flush=True)
            for name in args.windows:
                if not collected[name]:
                    continue
                out = (args.output_dir /
                       f"group{group_id}_{session.name}_{block}_{name}.png")
                contact_sheet(
                    out, collected[name], names=names,
                    title=(f"group {group_id}  {session.name}  {block}  —  "
                           f"{WINDOW_TITLES[name]}\n"
                           f"rendered z-score movies, mean over window; "
                           f"colour scale is baked in, not quantitative"))
                written.append(str(out))
    print(f"\nWrote {len(written)} contact sheets to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
