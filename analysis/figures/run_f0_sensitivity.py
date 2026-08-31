"""Run the F0/SNR sensitivity check for 10x and 20x DA populations."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .f0_sensitivity import (f0_change_associations, f0_sensitivity_table,
                             load_raw_population, session_sensitivity_summary)
from .paths import imaging_root, repo_path
from .session_data import available_sessions


def _plot(path, summary):
    import matplotlib.pyplot as plt

    comparisons = [
        ("DAT", "units", "DAT 10x"), ("TH", "units", "TH 10x"),
        ("DAT superficial", "somas", "DAT superficial somas"),
        ("TH superficial", "somas", "TH superficial somas"),
        ("TH deep", "somas", "TH deep somas"),
        ("DAT superficial", "processes", "DAT superficial processes"),
        ("TH superficial", "processes", "TH superficial processes"),
        ("TH deep", "processes", "TH deep processes"),
    ]
    filters = ["all units", "exclude lowest post-F0 quartile",
               "exclude lowest post-SNR quartile"]
    metrics = [("negative_auc_z_s_q75", "negative AUC q75 (z·s)"),
               ("negative_auc_dff_s_q75", "negative AUC q75 (ΔF/F0·s)")]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5), constrained_layout=True)
    for ax, (cohort, compartment, title) in zip(axes.ravel(), comparisons):
        subset = summary[(summary.cohort == cohort) &
                         (summary.compartment == compartment)]
        for metric_index, (metric, ylabel) in enumerate(metrics):
            offset = (metric_index-.5)*.12
            for filter_index, label in enumerate(filters):
                values = subset[subset.sensitivity_set == label]
                pivot = values.pivot_table(
                    index=["mouse", "group_id"], columns="state", values=metric).dropna()
                if pivot.empty:
                    continue
                x = filter_index + offset
                session_change = pivot.post-pivot.pre
                ax.scatter(np.full(len(session_change), x), session_change,
                           s=9, alpha=.22,
                           color=("#3b6fb6" if metric_index == 0 else "#d0772d"))
                mouse = pivot.groupby(level="mouse").median()
                change = mouse.post-mouse.pre
                ax.scatter(np.full(len(change), x), change, s=22,
                           color=("#3b6fb6" if metric_index == 0 else "#d0772d"),
                           label=ylabel if filter_index == 0 else None)
        ax.axhline(0, color=".65", lw=.8)
        ax.set(title=title, xticks=range(3),
               xticklabels=["all", "post-F0\nq25 removed", "post-SNR\nq25 removed"],
               ylabel="ket/xyl − awake, suppression-rich q75")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Loss of the suppressive tail: F0/SNR sensitivity\n"
                 "points are mouse medians; blue=z, orange=ΔF/F0")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=repo_path(
        "analysis", "stage0", "ketxyl_16odor_session_manifest.csv"))
    parser.add_argument("--imaging-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=repo_path(
        "analysis", "figures", "f0_sensitivity_outputs"))
    parser.add_argument("--groups", nargs="*", type=int, default=[])
    args = parser.parse_args(argv)
    root = imaging_root(args.imaging_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inventory = []
    for objective in ("10x", "20x"):
        inventory.extend(available_sessions(args.manifest, root, objective=objective))
    inventory = pd.DataFrame(inventory)
    inventory = inventory[inventory.available &
                          inventory.population.str.startswith(("TH-", "DAT-"))]
    if args.groups:
        inventory = inventory[inventory.group_id.astype(int).isin(args.groups)]
    parts, failures = [], []
    for row in tqdm(inventory.to_dict("records"), desc="F0 sensitivity", unit="session"):
        populations = ["units"] if row["objective"].lower() == "10x" else ["somas", "processes"]
        for population in populations:
            try:
                data = load_raw_population(row["grouped_path"], population)
                parts.append(f0_sensitivity_table(data, row, population))
            except Exception as error:
                failures.append({"group_id": int(row["group_id"]),
                                 "population": population,
                                 "error": f"{type(error).__name__}: {error}"})
    table = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    summary = session_sensitivity_summary(table)
    associations = f0_change_associations(table)
    table.to_csv(args.output_dir / "unit_odor_f0_sensitivity.csv", index=False)
    summary.to_csv(args.output_dir / "session_f0_sensitivity.csv", index=False)
    associations.to_csv(args.output_dir / "f0_suppression_change_associations.csv", index=False)
    pd.DataFrame(failures).to_csv(args.output_dir / "failures.csv", index=False)
    _plot(args.output_dir / "f0_sensitivity_suppressive_tail.png", summary)
    print({"unit_odor_rows": len(table), "sessions": summary.group_id.nunique(),
           "failures": len(failures)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
