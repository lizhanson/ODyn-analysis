"""Build pixelwise 10x odor maps and reciprocal-mixture difference maps.

Each trial is normalized independently at every pixel:

    z_map = (mean odor fluorescence - mean pre-odor fluorescence)
            / pre-odor fluorescence SD

Trial maps are then reduced within state and odor (median by default).  This is
a true motion-corrected pixel map, not an ROI mask filled with trace values.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np

from analysis.figures.paths import imaging_root, repo_path
from analysis.figures.population_metrics import _source_path
from analysis.figures.session_data import available_sessions


ODOR_ORDER = (0, 1, 2, 3, 4, 10, 12, 17, 18, 21, 22, 30, 31, 32, 39, 40)
MIXTURE_PAIRS = ((17, 18), (31, 32), (39, 40))


def _decode(values):
    return [value.decode() if isinstance(value, bytes) else str(value)
            for value in values]


def _trial_paths(root, trial_ids):
    database = Path(root) / ".odyn" / "odyn.db"
    uri = f"file:{database}?immutable=1"
    placeholders = ",".join("?" for _ in trial_ids)
    query = f"""
        SELECT t.trial_id, m.mcor_path
        FROM trials AS t JOIN mcor_files AS m ON m.acq_id = t.acq_id
        WHERE m.approved = 1 AND t.trial_id IN ({placeholders})
    """
    with sqlite3.connect(uri, uri=True) as connection:
        found = dict(connection.execute(query, [int(value) for value in trial_ids]))
    missing = sorted(set(map(int, trial_ids)) - set(found))
    if missing:
        raise FileNotFoundError(f"approved motion-corrected files missing for {missing}")
    return [Path(root) / found[int(value)] for value in trial_ids]


def _trial_pixel_z(path, odor_on, odor_off):
    """Read only baseline and odor frames and return the odor-period z map."""
    import tifffile

    # Memory mapping avoids loading the unused post-odor frames. Some mounted
    # filesystems disallow mmap; page reads are a portable fallback.
    try:
        movie = tifffile.memmap(path)
        baseline = np.asarray(movie[:odor_on], dtype=np.float32)
        odor = np.asarray(movie[odor_on:odor_off], dtype=np.float32)
    except (OSError, ValueError):
        with tifffile.TiffFile(path) as handle:
            baseline = np.stack(
                [page.asarray() for page in handle.pages[:odor_on]]).astype(
                    np.float32, copy=False)
            odor = np.stack(
                [page.asarray() for page in handle.pages[odor_on:odor_off]]).astype(
                    np.float32, copy=False)
    baseline_mean = np.mean(baseline, axis=0)
    baseline_sd = np.std(baseline, axis=0, ddof=1)
    numerator = np.mean(odor, axis=0) - baseline_mean
    floor = np.nanpercentile(baseline_sd[baseline_sd > 0], 1)
    return np.divide(numerator, baseline_sd,
                     out=np.full_like(numerator, np.nan),
                     where=baseline_sd >= floor).astype(np.float32)


def session_pixel_maps(row, root, *, reducer="median"):
    import h5py
    from tqdm.auto import tqdm

    grouped_path = Path(row["grouped_path"])
    with h5py.File(grouped_path) as grouped:
        trial_ids = grouped["trial_id"][:]
        odor_ids = grouped["odor_id"][:]
        states = grouped["state"][:]
        state_levels = _decode(grouped["state_levels"][:])
        source_path = _source_path(grouped_path, grouped)
    with h5py.File(source_path) as source:
        odor_on = source["trials/odor_on_frame"][:]
        odor_off = source["trials/odor_off_frame"][:]
        source_ids = source["trials/trial_id"][:]
    if not np.array_equal(trial_ids, source_ids):
        raise ValueError("grouped and source trial ids are not aligned")
    paths = _trial_paths(root, trial_ids)
    collected = {}
    iterator = zip(paths, odor_on, odor_off, states, odor_ids)
    total = len(paths)
    for path, on, off, state, odor in tqdm(
            iterator, total=total, desc=f"group {row['group_id']} pixel maps",
            unit="trial"):
        key = (state_levels[int(state)], int(odor))
        collected.setdefault(key, []).append(
            _trial_pixel_z(path, int(on), int(off)))
    function = {"median": np.nanmedian, "mean": np.nanmean}[reducer]
    maps = {key: function(np.stack(values), axis=0).astype(np.float32)
            for key, values in collected.items()}
    counts = {key: len(values) for key, values in collected.items()}
    return maps, counts


def session_roi_maps(row, *, reducer="median"):
    """Project final analysis-unit responses back into curated mask pixels."""
    import h5py

    grouped_path = Path(row["grouped_path"])
    with h5py.File(grouped_path) as grouped:
        odor_ids = grouped["odor_id"][:]
        states = grouped["state"][:]
        state_levels = _decode(grouped["state_levels"][:])
        z = grouped["units/z"][:]
        members = [np.asarray(value, int)
                   for value in grouped["units/member_roi_ids"][:]]
        source_path = _source_path(grouped_path, grouped)
    with h5py.File(source_path) as source:
        time_s = source["traces/time_s"][:]
        labels = source["masks/labels"][:]
    odor_window = (time_s >= 0) & (time_s < 4)
    trial_response = np.nanmean(z[:, :, odor_window], axis=2)
    function = {"median": np.nanmedian, "mean": np.nanmean}[reducer]
    maps, counts = {}, {}
    for state_code, state in enumerate(state_levels):
        for odor in np.unique(odor_ids[states == state_code]):
            selected = (states == state_code) & (odor_ids == odor)
            unit_values = function(trial_response[:, selected], axis=1)
            image = np.full(labels.shape, np.nan, np.float32)
            for value, roi_ids in zip(unit_values, members):
                image[np.isin(labels, roi_ids)] = value
            maps[(state, int(odor))] = image
            counts[(state, int(odor))] = int(np.sum(selected))
    return maps, counts


def plot_odor_page(path, maps, counts, row, *, state="pre", limits=(-1.5, 3.0),
                   map_source="pixel"):
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    fig, axes = plt.subplots(4, 4, figsize=(12, 13), constrained_layout=True)
    norm = TwoSlopeNorm(vmin=limits[0], vcenter=0, vmax=limits[1])
    image = None
    for ax, odor in zip(axes.ravel(), ODOR_ORDER):
        values = maps.get((state, odor))
        if values is None:
            ax.axis("off"); continue
        image = ax.imshow(values, cmap="RdBu_r", norm=norm,
                          interpolation="nearest")
        ax.set(title=f"odor {odor}  n={counts[(state, odor)]}", xticks=[], yticks=[])
    label = "awake" if state == "pre" else "ket/xyl"
    description = ("pixelwise odor-period z" if map_source == "pixel" else
                   "final glomerular ROIs filled by odor-period z")
    fig.suptitle(f"{row['population']} 10x - {row['mouse']} - group {row['group_id']} - {label}\n"
                 f"{description}, median across trials")
    fig.colorbar(image, ax=axes, label="odor-period z", shrink=.72)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_mixture_differences(path, maps, counts, row, *, limits=(-1.5, 1.5),
                             map_source="pixel"):
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    fig, axes = plt.subplots(3, 2, figsize=(8, 11), constrained_layout=True)
    norm = TwoSlopeNorm(vmin=limits[0], vcenter=0, vmax=limits[1])
    image = None
    for pair_index, (odor_a, odor_b) in enumerate(MIXTURE_PAIRS):
        for state_index, state in enumerate(("pre", "post")):
            ax = axes[pair_index, state_index]
            a, b = maps.get((state, odor_a)), maps.get((state, odor_b))
            if a is None or b is None:
                ax.axis("off"); continue
            image = ax.imshow(a-b, cmap="RdBu_r", norm=norm,
                              interpolation="nearest")
            label = "awake" if state == "pre" else "ket/xyl"
            ax.set(title=(f"{odor_a} - {odor_b}, {label}\n"
                          f"n={counts[(state, odor_a)]}/{counts[(state, odor_b)]}"),
                   xticks=[], yticks=[])
    description = "pixelwise" if map_source == "pixel" else "ROI-projected"
    fig.suptitle(f"{row['population']} 10x - {row['mouse']} - {description} reciprocal-mixture differences")
    fig.colorbar(image, ax=axes, label="difference in odor-period z", shrink=.72)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", nargs="+", type=int, default=[214, 215, 219])
    parser.add_argument("--manifest", type=Path,
                        default=repo_path("analysis", "stage0", "ketxyl_16odor_session_manifest.csv"))
    parser.add_argument("--imaging-root", type=Path, default=None,
                        help="ImagingData root; defaults to ODYN_IMAGING_ROOT")
    parser.add_argument("--output-dir", type=Path,
                        default=repo_path("analysis", "figures", "spatial_maps_10x"))
    parser.add_argument("--reducer", choices=("median", "mean"), default="median")
    parser.add_argument("--map-source", choices=("roi", "pixel"), default="roi",
                        help="ROI projection is fast; true pixels reread all motion-corrected TIFFs")
    args = parser.parse_args(argv)
    root = imaging_root(args.imaging_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inventory = available_sessions(args.manifest, root, objective="10x")
    selected = {int(row["group_id"]): row for row in inventory
                if int(row["group_id"]) in args.groups}
    missing = sorted(set(args.groups) - set(selected))
    if missing:
        raise ValueError(f"groups absent from 10x manifest: {missing}")
    for group_id in args.groups:
        row = selected[group_id]
        if not row["available"]:
            raise FileNotFoundError(f"group {group_id} has no grouped product")
        if args.map_source == "pixel":
            maps, counts = session_pixel_maps(row, root, reducer=args.reducer)
        else:
            maps, counts = session_roi_maps(row, reducer=args.reducer)
        stem = f"group{group_id}_{row['mouse']}_{row['population'].split('-')[0]}"
        np.savez_compressed(
            args.output_dir / f"{stem}_{args.map_source}_maps.npz",
            **{f"{state}_odor{odor}": value for (state, odor), value in maps.items()})
        plot_odor_page(args.output_dir / f"{stem}_awake_16odor_{args.map_source}_maps.png",
                       maps, counts, row, state="pre", map_source=args.map_source)
        plot_mixture_differences(
            args.output_dir / f"{stem}_{args.map_source}_reciprocal_mixture_differences.png",
            maps, counts, row, map_source=args.map_source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
