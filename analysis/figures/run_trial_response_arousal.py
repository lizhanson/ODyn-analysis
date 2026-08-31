"""Build trial-level pupil/response-width scatters and alpha-family tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .paths import imaging_root, repo_path
from .population_metrics import load_population
from .session_data import available_sessions
from .state_arousal import find_auxiliary
from .trial_response_arousal import (trial_response_arousal_table,
                                     within_odor_family_correlations)


COLORS = {"other": ".72", "components 4/10": "#178f8f",
          "mixtures 17/18": "#c43c75"}


def plot_pupil_scatter(path, table, compartment):
    import matplotlib.pyplot as plt

    data = table[table.compartment == compartment]
    cohorts = list(dict.fromkeys(data.cohort.astype(str)))
    fig, axes = plt.subplots(len(cohorts), 2, figsize=(10, 3.4*len(cohorts)),
                             squeeze=False, constrained_layout=True)
    for row, cohort in enumerate(cohorts):
        subset = data[data.cohort.astype(str) == cohort]
        for column, (metric, label) in enumerate((
                ("pupil_odor_z", "mean pupil during odor (baseline SD)"),
                ("pupil_change_z", "pupil change during odor (baseline SD)"))):
            ax = axes[row, column]
            for family in ("other", "components 4/10", "mixtures 17/18"):
                values = subset[subset.alpha_class == family]
                ax.scatter(values[metric], values.response_width_z, s=12,
                           alpha=.34 if family == "other" else .65,
                           color=COLORS[family], label=family)
            ax.set(xlabel=label, ylabel="q90-q10 ROI response width (z)",
                   title=cohort)
            ax.axvline(0, color=".8", lw=.8)
        axes[row, -1].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Every awake trial: pupil and ROI response-distribution width — {compartment}")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_alpha_suppression(path, correlations):
    import matplotlib.pyplot as plt

    selected = correlations[(correlations.neural_metric == "q10_z") &
                            correlations.cohort.isin(("TH", "TH deep"))]
    families = ("components 4/10", "mixtures 17/18", "other")
    metrics = ("pupil_odor_z", "pupil_change_z", "running_speed")
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True,
                             constrained_layout=True)
    for row, cohort in enumerate(("TH", "TH deep")):
        cohort_data = selected[selected.cohort == cohort]
        for column, metric in enumerate(metrics):
            ax = axes[row, column]
            values = cohort_data[cohort_data.arousal_metric == metric]
            for x, family in enumerate(families):
                family_data = values[values.odor_family == family]
                ax.scatter([x]*len(family_data), family_data.rho, s=28, alpha=.65)
                mouse = family_data.groupby("mouse", as_index=False).rho.median()
                ax.scatter([x]*len(mouse), mouse.rho, s=65, facecolors="white",
                           edgecolors="black")
            ax.axhline(0, color=".7", lw=.8)
            ax.set(title=metric.replace("_", " "), xticks=range(3),
                   xticklabels=["4/10", "17/18", "other"], ylim=(-1, 1))
            if column == 0:
                ax.set_ylabel(f"{cohort}\nSpearman rho: q10 vs arousal")
    fig.suptitle("Does arousal selectively deepen the suppressive tail for 17/18?\n"
                 "odor medians removed; small=session, large=mouse median")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=repo_path("analysis", "stage0", "ketxyl_16odor_session_manifest.csv"))
    parser.add_argument("--imaging-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path,
                        default=repo_path("analysis", "figures", "trial_arousal_outputs"))
    parser.add_argument("--objective", choices=("10x", "20x", "all"), default="all")
    args = parser.parse_args(argv)
    root = imaging_root(args.imaging_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inventory = pd.DataFrame(available_sessions(args.manifest, root))
    inventory = inventory[inventory.available & inventory.population.str.startswith(("TH-", "DAT-"))]
    if args.objective != "all":
        inventory = inventory[inventory.objective.str.lower() == args.objective]
    parts, failures = [], []
    for row in tqdm(inventory.to_dict("records"), desc="trial response/arousal", unit="session"):
        auxiliary = find_auxiliary(row, root)
        if auxiliary is None:
            failures.append({"group_id": int(row["group_id"]), "error": "auxiliary file missing"})
            continue
        populations = ["units"] if row["objective"].lower() == "10x" else ["somas", "groups"]
        for population in populations:
            try:
                data = load_population(row["grouped_path"], population)
                parts.append(trial_response_arousal_table(
                    data, auxiliary, row, population))
            except Exception as error:
                failures.append({"group_id": int(row["group_id"]),
                                 "population": population,
                                 "error": f"{type(error).__name__}: {error}"})
    table = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    table.to_csv(args.output_dir / "trial_response_arousal.csv", index=False)
    correlations = within_odor_family_correlations(table)
    correlations.to_csv(args.output_dir / "within_odor_family_correlations.csv", index=False)
    pd.DataFrame(failures).to_csv(args.output_dir / "failures.csv", index=False)
    for compartment in table.compartment.unique():
        plot_pupil_scatter(args.output_dir / f"pupil_response_width_{compartment}.png",
                           table, compartment)
    plot_alpha_suppression(args.output_dir / "alpha_family_arousal_suppression.png",
                           correlations)
    print({"trial_rows": len(table), "correlation_rows": len(correlations),
           "failures": len(failures)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
