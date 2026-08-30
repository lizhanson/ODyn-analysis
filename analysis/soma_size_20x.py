"""Compare curated 20x soma-mask size across line and depth cohorts.

The per-soma table is descriptive.  Session and mouse summaries are written
separately so downstream inference does not mistake cells for replicates.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from .batch_20x import experiment_dir, manifest_rows, published_bundles


COHORT_ORDER = ("DAT superficial", "TH superficial", "TH deep")


def cohort_name(row: dict) -> str:
    line = row["population"].split("-")[0]
    return f"{line} {row['depth_class']}"


def soma_measurements(labels, *, um_per_px: float) -> list[dict]:
    """Return area and equivalent-circle diameter for every positive label."""
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError("soma labels must be a 2D image")
    if not float(um_per_px) > 0:
        raise ValueError("um_per_px must be positive")
    counts = np.bincount(labels.ravel().astype(np.int64))
    scale = float(um_per_px)
    rows = []
    for label, area_px in enumerate(counts[1:], start=1):
        if area_px == 0:
            continue
        area_um2 = float(area_px) * scale**2
        rows.append({
            "soma_label": label,
            "area_px": int(area_px),
            "area_um2": area_um2,
            "equivalent_diameter_um": 2.0 * np.sqrt(area_um2 / np.pi),
        })
    return rows


def round_um_per_px(round_path: Path) -> float:
    import h5py

    with h5py.File(round_path, "r") as handle:
        parameters = json.loads(handle.attrs["parameters_json"])
    value = parameters.get("session", {}).get("um_per_px_reported")
    if value is None:
        raise ValueError(f"No reported pixel scale in {round_path.name}")
    return float(value)


def session_rows(row: dict, imaging_root: Path, *, um_per_px=None) -> list[dict]:
    import h5py

    from .session.store import find_rounds

    bundles = published_bundles(row, imaging_root)
    rounds = find_rounds(experiment_dir(row, imaging_root) / "processed" / "python")
    if not bundles:
        raise FileNotFoundError("no published 20x mask bundle")
    if not rounds and um_per_px is None:
        raise FileNotFoundError("no extracted round from which to read pixel scale")
    scale = float(um_per_px) if um_per_px is not None else round_um_per_px(rounds[-1])
    with h5py.File(bundles[-1], "r") as handle:
        labels = handle["masks/soma"][:]
    measured = soma_measurements(labels, um_per_px=scale)
    common = {
        "group_id": int(row["group_id"]),
        "mouse": row["mouse"],
        "date": row["date"],
        "population": row["population"],
        "depth_class": row["depth_class"],
        "depth_um": row.get("depth_um", ""),
        "cohort": cohort_name(row),
        "um_per_px": scale,
        "mask_bundle": str(bundles[-1]),
    }
    return [{**common, **item} for item in measured]


def summarize_session(cells: list[dict]) -> list[dict]:
    groups: dict[int, list[dict]] = {}
    for row in cells:
        groups.setdefault(int(row["group_id"]), []).append(row)
    output = []
    for group_id, rows in groups.items():
        diameter = np.asarray([row["equivalent_diameter_um"] for row in rows])
        area = np.asarray([row["area_um2"] for row in rows])
        first = rows[0]
        output.append({
            key: first[key] for key in (
                "group_id", "mouse", "date", "population", "depth_class",
                "depth_um", "cohort", "um_per_px", "mask_bundle",
            )
        } | {
            "n_somas": len(rows),
            "median_area_um2": float(np.median(area)),
            "median_equivalent_diameter_um": float(np.median(diameter)),
            "diameter_q25_um": float(np.percentile(diameter, 25)),
            "diameter_q75_um": float(np.percentile(diameter, 75)),
        })
    return sorted(output, key=lambda row: int(row["group_id"]))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(path: Path, cells: list[dict], sessions: list[dict]) -> None:
    import matplotlib.pyplot as plt

    colors = {"DAT superficial": "#6a51a3", "TH superficial": "#2b8cbe",
              "TH deep": "#e34a33"}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for cohort in COHORT_ORDER:
        selected = [row for row in sessions if row["cohort"] == cohort]
        for session in selected:
            values = np.sort([
                row["area_um2"] for row in cells
                if int(row["group_id"]) == int(session["group_id"])
            ])
            y = (np.arange(len(values)) + .5) / len(values)
            axes[0].plot(values, y, color=colors[cohort], alpha=.28, lw=1)
        # One bold, cell-balanced curve is intentionally omitted: the session
        # curves show heterogeneity without giving cell-rich sessions more vote.
    axes[0].set(xlabel="curated soma-mask area (µm²)", ylabel="within-session ECDF",
                title="Curated soma-size distributions")

    mouse_colors = {}
    palette = plt.get_cmap("tab10")
    for index, mouse in enumerate(sorted({row["mouse"] for row in sessions})):
        mouse_colors[mouse] = palette(index)
    rng = np.random.default_rng(0)
    for x, cohort in enumerate(COHORT_ORDER):
        selected = [row for row in sessions if row["cohort"] == cohort]
        for row in selected:
            axes[1].scatter(
                x + rng.uniform(-.08, .08), row["median_area_um2"],
                s=38, color=mouse_colors[row["mouse"]], edgecolor="white", lw=.5,
                label=row["mouse"], zorder=3,
            )
    # Connect mouse-averaged superficial/deep TH estimates when both exist.
    for mouse in sorted(mouse_colors):
        values = {}
        for cohort in ("TH superficial", "TH deep"):
            found = [row["median_area_um2"] for row in sessions
                     if row["mouse"] == mouse and row["cohort"] == cohort]
            if found:
                values[cohort] = float(np.mean(found))
        if len(values) == 2:
            axes[1].plot([1, 2], [values["TH superficial"], values["TH deep"]],
                         color=mouse_colors[mouse], alpha=.55, lw=1)
    handles, labels = axes[1].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axes[1].legend(unique.values(), unique.keys(), title="mouse", frameon=False,
                   fontsize=8)
    axes[1].set(xticks=range(3), xticklabels=COHORT_ORDER,
                ylabel="session median soma-mask area (µm²)",
                title="Sessions are the plotted sampling units")
    axes[1].tick_params(axis="x", rotation=20)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=(
        "analysis/stage0/ketxyl_16odor_session_manifest.csv"))
    parser.add_argument("--imaging-root", default=os.environ.get(
        "ODYN_IMAGING_ROOT", "/Volumes/MossLab/ImagingData"))
    parser.add_argument("--groups", nargs="*", type=int, default=[])
    parser.add_argument("--um-per-px", type=float,
                        help="Override reported scale for every selected session.")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("analysis/results/soma_size_20x"))
    args = parser.parse_args(argv)

    cells = []
    failures = []
    selected_rows = manifest_rows(args.manifest, args.groups)
    try:
        from tqdm.auto import tqdm
        selected_rows = tqdm(selected_rows, desc="20x soma masks", unit="session")
    except ImportError:
        pass
    for row in selected_rows:
        try:
            cells.extend(session_rows(
                row, Path(args.imaging_root), um_per_px=args.um_per_px))
        except Exception as error:
            failures.append({"group_id": int(row["group_id"]),
                             "error": f"{type(error).__name__}: {error}"})
    sessions = summarize_session(cells)
    write_csv(args.output_dir / "soma_measurements.csv", cells)
    write_csv(args.output_dir / "session_summary.csv", sessions)
    plot_summary(args.output_dir / "soma_size_comparison.png", cells, sessions)
    report = {"n_cells": len(cells), "n_sessions": len(sessions),
              "failures": failures}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
