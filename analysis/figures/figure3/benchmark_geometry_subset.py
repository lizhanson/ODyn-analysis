"""Exploratory 20x mixture geometry across representative line/depth sessions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analysis.figures.geometry import (cosine_centroid_distance,
                                       diagonal_crossnobis)
from ..paths import imaging_root, repo_path


MIXTURE_PAIRS = ((17, 18), (31, 32), (39, 40))


def distance(x, labels, seed=0):
    _, rdm = diagonal_crossnobis(x, labels, repeats=60, seed=seed)
    return float(rdm[0, 1])


def null_z(x, labels, observed, *, permutations=30, seed=0):
    rng = np.random.default_rng(seed)
    null = np.asarray([distance(x, rng.permutation(labels), seed + i + 1)
                       for i in range(permutations)])
    sd = np.nanstd(null, ddof=1)
    return float((observed - np.nanmean(null)) / sd) if sd > 0 else np.nan


def _source_path(grouped_path, handle):
    source = handle.attrs.get("source_round")
    if isinstance(source, bytes): source = source.decode()
    source = Path(str(source))
    if source.exists(): return source
    stem = grouped_path.name.split("_pertrial_median_20x_grouped.h5")[0]
    return grouped_path.parent / f"{stem}.h5"


def analyze_session(row, grouped_path, *, permutations=30):
    import h5py

    with h5py.File(grouped_path) as h:
        source = _source_path(grouped_path, h)
        odor = h["odor_id"][:]; state = h["state"][:]
        levels = [x.decode() if isinstance(x, bytes) else str(x)
                  for x in h["state_levels"][:]]
        populations = {key: h[f"{key}/z"][:] for key in ("somas", "processes")}
    with h5py.File(source) as h:
        time_s = h["traces/time_s"][:]
    edges = np.arange(0., 4., 1.)
    odor_window = (time_s >= 0) & (time_s < 4)
    output = []
    for compartment, z in populations.items():
        for state_name in ("pre", "post"):
            code = levels.index(state_name)
            for pair_index, pair in enumerate(MIXTURE_PAIRS):
                selected = (state == code) & np.isin(odor, pair)
                labels = odor[selected]
                counts = [np.sum(labels == item) for item in pair]
                if min(counts) < 2: continue
                traces = np.transpose(z[:, selected, :], (1, 0, 2))
                integrated_x = np.nanmean(traces[:, :, odor_window], axis=2)
                integrated = distance(integrated_x, labels, 1000 + pair_index)
                one_second = np.stack([
                    np.nanmean(traces[:, :, (time_s >= edge) & (time_s < edge + 1)], axis=2)
                    for edge in edges], axis=2)
                spatial_time = [distance(one_second[:, :, i], labels, 2000 + i)
                                for i in range(len(edges))]
                cumulative = [
                    distance(np.nanmean(
                        traces[:, :, (time_s >= 0) & (time_s < edge + 1)], axis=2),
                             labels, 3000 + i)
                    for i, edge in enumerate(edges)
                ]
                temporal_x = one_second.reshape(len(labels), -1)
                temporal = distance(temporal_x, labels, 4000 + pair_index)
                common = {
                    "group_id": int(row["group_id"]), "mouse": row["mouse"],
                    "line": row["population"].split("-")[0],
                    "depth_class": row["depth_class"], "compartment": compartment,
                    "state": state_name, "pair": f"{pair[0]}-{pair[1]}",
                    "n_a": int(counts[0]), "n_b": int(counts[1]),
                    "integrated_crossnobis": integrated,
                    "integrated_null_z": null_z(
                        integrated_x, labels, integrated,
                        permutations=permutations, seed=5000 + pair_index),
                    "integrated_cosine": cosine_centroid_distance(integrated_x, labels),
                    "spatiotemporal_crossnobis": temporal,
                    "spatiotemporal_null_z": null_z(
                        temporal_x, labels, temporal,
                        permutations=permutations, seed=6000 + pair_index),
                }
                for index, edge in enumerate(edges):
                    output.append(common | {
                        # A shared elapsed-time endpoint avoids tiny frame-rate
                        # differences masquerading as separate x coordinates
                        # when sessions are summarized together.
                        "time_s": float(edge + 1),
                        "one_second_crossnobis": spatial_time[index],
                        "cumulative_crossnobis": cumulative[index],
                    })
    return output


def main(argv=None):
    import pandas as pd
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=repo_path("analysis", "stage0", "ketxyl_16odor_session_manifest.csv"))
    parser.add_argument("--imaging-root", type=Path, default=None,
                        help="ImagingData root; defaults to ODYN_IMAGING_ROOT")
    parser.add_argument("--groups", nargs="+", type=int, default=[199, 201, 202, 203, 225, 227])
    parser.add_argument("--permutations", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv); args.imaging_root = imaging_root(args.imaging_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.manifest, newline="") as stream: manifest = list(csv.DictReader(stream))
    rows, failures = [], []
    for gid in args.groups:
        row = next(item for item in manifest if int(item["group_id"]) == gid)
        directory = args.imaging_root / row["date"] / row["mouse"] / row["exp"] / "processed/python"
        files = list(directory.glob("*_pertrial_median_20x_grouped.h5"))
        try:
            if not files: raise FileNotFoundError("no current grouped product")
            rows.extend(analyze_session(row, max(files), permutations=args.permutations))
        except Exception as error:
            failures.append({"group_id": gid, "error": f"{type(error).__name__}: {error}"})
    data = pd.DataFrame(rows)
    data.to_csv(args.output_dir / "mixture_geometry_subset.csv", index=False)
    summary_columns = ["group_id", "mouse", "line", "depth_class", "compartment",
                       "state", "pair", "integrated_crossnobis", "integrated_null_z",
                       "integrated_cosine", "spatiotemporal_crossnobis",
                       "spatiotemporal_null_z"]
    data[summary_columns].drop_duplicates().to_csv(
        args.output_dir / "mixture_geometry_session_summary.csv", index=False)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, constrained_layout=True)
    for column, pair in enumerate(("17-18", "31-32", "39-40")):
        for row_index, state in enumerate(("pre", "post")):
            ax = axes[row_index, column]
            selected = data[(data.pair == pair) & (data.state == state)]
            for (_, compartment), group in selected.groupby(["group_id", "compartment"]):
                ax.plot(group.time_s, group.cumulative_crossnobis,
                        alpha=.45, lw=1, ls="-" if compartment == "somas" else "--")
            median = selected.groupby("time_s").cumulative_crossnobis.median()
            ax.plot(median.index, median, color="black", lw=2, label="median session/population")
            ax.axhline(0, color=".6", lw=.7); ax.set(title=f"{pair}, {state}", ylabel="cumulative distance")
    for ax in axes[-1]: ax.set_xlabel("seconds accumulated")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.savefig(args.output_dir / "cumulative_crossnobis_subset.png", dpi=180)
    plt.close(fig)
    (args.output_dir / "report.json").write_text(json.dumps(
        {"groups": args.groups, "n_rows": len(data), "failures": failures}, indent=2) + "\n")
    print(data[summary_columns].drop_duplicates().groupby(
        ["line", "depth_class", "state", "pair"])[
        ["integrated_crossnobis", "integrated_cosine", "spatiotemporal_crossnobis"]
    ].median().to_string())
    print("failures", failures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
