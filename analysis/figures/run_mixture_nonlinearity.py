"""Batch component-axis nonlinearity for 10x and 20x populations."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from .mixture_nonlinearity import session_mixture_nonlinearity
from .paths import imaging_root, repo_path
from .population_metrics import load_population
from .session_data import available_sessions


def plot_overview(path, table):
    import matplotlib.pyplot as plt

    integrated = table[(table.window_start_s == 0) & (table.window_stop_s == 4)]
    families = ["alpha 17/18", "epsilon 31/32", "lambda 39/40"]
    metrics = [("mean_off_axis_residual_rms_z", "off-component-axis residual (RMS z)"),
               ("mixture_distance_rms_z", "reciprocal-mixture distance (RMS z)"),
               ("best_60_40_residual_rms_z", "best 60:40 prediction residual (RMS z)")]
    cohorts = list(dict.fromkeys(integrated.cohort.astype(str)))
    fig, axes = plt.subplots(len(cohorts), 3, figsize=(12, 3.2*len(cohorts)),
                             squeeze=False, constrained_layout=True)
    colors = {"pre": "#2b6cb0", "post": "#c44e52"}
    for row_index, cohort in enumerate(cohorts):
        cohort_data = integrated[integrated.cohort.astype(str) == cohort]
        for ax, (metric, label) in zip(axes[row_index], metrics):
            for state, offset in (("pre", -.08), ("post", .08)):
                selected = cohort_data[cohort_data.state == state]
                for x, family in enumerate(families):
                    values = selected[selected.family == family]
                    mouse = values.groupby("mouse", as_index=False)[metric].median()
                    ax.scatter([x+offset]*len(mouse), mouse[metric], s=28,
                               color=colors[state], alpha=.75,
                               label=state if x == 0 else None)
                    if len(mouse):
                        ax.plot(x+offset, mouse[metric].median(), marker="_",
                                markersize=18, markeredgewidth=2.5,
                                color=colors[state])
            ax.set(xticks=range(3), xticklabels=["17/18", "31/32", "39/40"])
            if row_index == 0:
                ax.set_title(label)
            if ax is axes[row_index, 0]:
                ax.set_ylabel(cohort)
            if row_index == 0 and ax is axes[0, 0]:
                ax.legend(frameon=False)
    fig.suptitle("Mixture nonlinearity and reciprocal separation are distinct axes")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=repo_path("analysis", "stage0", "ketxyl_16odor_session_manifest.csv"))
    parser.add_argument("--imaging-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path,
                        default=repo_path("analysis", "figures", "mixture_nonlinearity_outputs"))
    parser.add_argument("--objective", choices=("all", "10x", "20x"), default="all")
    parser.add_argument("--minimum-trials", type=int, default=2)
    args = parser.parse_args(argv)
    root = imaging_root(args.imaging_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inventory = pd.DataFrame(available_sessions(args.manifest, root))
    inventory = inventory[inventory.available]
    if args.objective != "all":
        inventory = inventory[inventory.objective.str.lower() == args.objective]
    parts, failures = [], []
    bins = ((0., 1.), (1., 2.), (2., 3.), (3., 4.), (0., 4.))
    for row in tqdm(inventory.to_dict("records"), desc="mixture nonlinearity",
                    unit="session"):
        populations = ["units"] if row["objective"].lower() == "10x" else ["somas", "groups"]
        for population in populations:
            try:
                data = load_population(row["grouped_path"], population)
                parts.append(session_mixture_nonlinearity(
                    data, row, population, bins=bins,
                    minimum_trials=args.minimum_trials))
            except Exception as error:
                failures.append({"group_id": int(row["group_id"]),
                                 "population": population,
                                 "error": f"{type(error).__name__}: {error}"})
    table = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    table.to_csv(args.output_dir / "mixture_nonlinearity.csv", index=False)
    pd.DataFrame(failures).to_csv(args.output_dir / "failures.csv", index=False)
    if len(table):
        plot_overview(args.output_dir / "mixture_nonlinearity_overview.png", table)
    print({"rows": len(table), "failures": len(failures)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
