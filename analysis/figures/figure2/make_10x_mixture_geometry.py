"""Generate paired 10x mixture-geometry panels for TH, DAT, and Thy1."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from analysis.figures.geometry import cosine_centroid_distance, pair_distance

# Reciprocal mixture pairs: the same two components at reciprocal ratios.
MIXTURE_PAIRS = ((17, 18), (31, 32), (39, 40))


def distance(x, labels, seed=0):
    return pair_distance(x, labels, seed=seed, repeats=60)
from analysis.figures.paths import imaging_root, repo_path


COLORS = {"TH": "#2b8cbe", "DAT": "#6a51a3", "Thy1": "#e34a33"}


def _source_path(grouped_path, handle):
    source = handle.attrs.get("source_component_round")
    if isinstance(source, bytes): source = source.decode()
    source = Path(str(source))
    if source.exists(): return source
    stem = grouped_path.name.split("_pertrial_median_10x_grouped.h5")[0]
    return grouped_path.parent / f"{stem}.h5"


def analyze_session(row, grouped_path):
    import h5py

    with h5py.File(grouped_path) as h:
        source = _source_path(grouped_path, h)
        odor = h["odor_id"][:]; state = h["state"][:]
        levels = [x.decode() if isinstance(x, bytes) else str(x)
                  for x in h["state_levels"][:]]
        z = h["units/z"][:]
    with h5py.File(source) as h:
        time_s = h["traces/time_s"][:]
    odor_window = (time_s >= 0) & (time_s < 4)
    edges = np.arange(0., 4., 1.)
    output = []
    for state_name in ("pre", "post"):
        code = levels.index(state_name)
        for pair_index, pair in enumerate(MIXTURE_PAIRS):
            selected = (state == code) & np.isin(odor, pair)
            labels = odor[selected]
            counts = [np.sum(labels == item) for item in pair]
            if min(counts) < 2: continue
            traces = np.transpose(z[:, selected, :], (1, 0, 2))
            integrated_x = np.nanmean(traces[:, :, odor_window], axis=2)
            one_second = np.stack([
                np.nanmean(traces[:, :, (time_s >= edge) & (time_s < edge + 1)], axis=2)
                for edge in edges], axis=2)
            temporal_x = one_second.reshape(len(labels), -1)
            common = {
                "group_id": int(row["group_id"]), "mouse": row["mouse"],
                "line": row["population"].split("-")[0], "state": state_name,
                "pair": f"{pair[0]}-{pair[1]}", "n_a": int(counts[0]),
                "n_b": int(counts[1]),
                "integrated_crossnobis": distance(integrated_x, labels, 100 + pair_index),
                "spatiotemporal_crossnobis": distance(temporal_x, labels, 200 + pair_index),
                "integrated_cosine": cosine_centroid_distance(integrated_x, labels),
            }
            for index, edge in enumerate(edges):
                cumulative_x = np.nanmean(
                    traces[:, :, (time_s >= 0) & (time_s < edge + 1)], axis=2)
                output.append(common | {
                    "seconds_accumulated": int(edge + 1),
                    "cumulative_crossnobis": distance(
                        cumulative_x, labels, 300 + index),
                })
    return output


def plot_state_comparison(path, summary):
    import matplotlib.pyplot as plt

    metrics = (
        ("integrated_crossnobis", "Integrated crossnobis"),
        ("spatiotemporal_crossnobis", "Spatiotemporal crossnobis"),
        ("integrated_cosine", "Integrated cosine distance"),
    )
    pairs = ("17-18", "31-32", "39-40")
    fig, axes = plt.subplots(3, 3, figsize=(11.5, 10), constrained_layout=True)
    for row_index, (metric, ylabel) in enumerate(metrics):
        for column, pair in enumerate(pairs):
            ax = axes[row_index, column]
            selected = summary[summary.pair == pair]
            for line in ("TH", "DAT", "Thy1"):
                population = selected[selected.line == line]
                for (_, _), session in population.groupby(["mouse", "group_id"]):
                    values = session.set_index("state")[metric]
                    if {"pre", "post"}.issubset(values.index):
                        ax.plot([0, 1], [values.pre, values.post],
                                color=COLORS[line], alpha=.20, lw=.8)
                # Sessions are first averaged within mouse; mice get equal weight.
                mouse = population.groupby(["mouse", "state"])[metric].mean().unstack()
                if {"pre", "post"}.issubset(mouse.columns):
                    for _, values in mouse.iterrows():
                        ax.scatter([0, 1], [values.pre, values.post], s=22,
                                   color=COLORS[line], alpha=.75, zorder=3)
                    center = mouse[["pre", "post"]].median(axis=0)
                    offset = {"TH": -.045, "DAT": 0., "Thy1": .045}[line]
                    ax.plot(np.array([0, 1]) + offset, center, color=COLORS[line],
                            marker="o", lw=2.5, label=line, zorder=4)
            ax.axhline(0, color=".75", lw=.7)
            ax.set(xticks=[0, 1], xticklabels=["awake", "ket/xyl"],
                   title=f"mixture {pair}")
            if column == 0: ax.set_ylabel(ylabel)
            if row_index == 0 and column == 0: ax.legend(frameon=False)
    fig.suptitle("10× dorsal-bulb mixture separation: sessions nested within mice")
    fig.savefig(path, dpi=200); plt.close(fig)


def plot_cumulative(path, data):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=False,
                             constrained_layout=True)
    for ax, pair in zip(axes, ("17-18", "31-32", "39-40")):
        selected = data[data.pair == pair]
        for line in ("TH", "DAT", "Thy1"):
            for state, style in (("pre", "-"), ("post", "--")):
                population = selected[(selected.line == line) & (selected.state == state)]
                mouse = population.groupby(
                    ["mouse", "seconds_accumulated"]
                ).cumulative_crossnobis.mean().unstack()
                center = mouse.median(axis=0)
                ax.plot(center.index, center, color=COLORS[line], ls=style, lw=2,
                        label=f"{line} {state}")
        ax.axhline(0, color=".7", lw=.7)
        ax.set(title=f"mixture {pair}", xlabel="seconds accumulated",
               xticks=[1, 2, 3, 4])
    axes[0].set_ylabel("cumulative crossnobis")
    axes[-1].legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle("When does 10× mixture separation emerge?")
    fig.savefig(path, dpi=200); plt.close(fig)


def main(argv=None):
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=repo_path("analysis", "stage0", "ketxyl_16odor_session_manifest.csv"))
    parser.add_argument("--imaging-root", type=Path, default=None,
                        help="ImagingData root; defaults to ODYN_IMAGING_ROOT")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv); args.imaging_root = imaging_root(args.imaging_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.manifest, newline="") as stream: manifest = list(csv.DictReader(stream))
    rows, inventory = [], []
    for row in manifest:
        if row["objective"].lower() != "10x": continue
        directory = args.imaging_root / row["date"] / row["mouse"] / row["exp"] / "processed/python"
        files = list(directory.glob("*_pertrial_median_10x_grouped.h5"))
        item = {"group_id": int(row["group_id"]), "mouse": row["mouse"],
                "line": row["population"].split("-")[0], "available": bool(files)}
        try:
            if not files: raise FileNotFoundError("no current grouped product")
            rows.extend(analyze_session(row, max(files)))
            item["status"] = "included"
        except Exception as error:
            item["status"] = f"{type(error).__name__}: {error}"
        inventory.append(item)
    data = pd.DataFrame(rows)
    data.to_csv(args.output_dir / "mixture_geometry_10x_long.csv", index=False)
    pd.DataFrame(inventory).to_csv(args.output_dir / "session_inventory.csv", index=False)
    summary_columns = ["group_id", "mouse", "line", "state", "pair", "n_a", "n_b",
                       "integrated_crossnobis", "spatiotemporal_crossnobis",
                       "integrated_cosine"]
    summary = data[summary_columns].drop_duplicates()
    summary.to_csv(args.output_dir / "mixture_geometry_10x_session_summary.csv", index=False)
    plot_state_comparison(args.output_dir / "10x_mixture_geometry_state_comparison.png", summary)
    plot_cumulative(args.output_dir / "10x_mixture_geometry_cumulative.png", data)
    print(pd.DataFrame(inventory).groupby(["line", "status"]).size().to_string())
    print(summary.groupby(["line", "state", "pair"])[
        ["integrated_crossnobis", "spatiotemporal_crossnobis", "integrated_cosine"]
    ].median().round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
