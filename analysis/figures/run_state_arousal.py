"""Run awake baseline and response-distribution arousal analyses.

Examples
--------
ODYN_IMAGING_ROOT=/path/to/ImagingData \
python -m analysis.figures.run_state_arousal
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .paths import imaging_root, repo_path
from .population_metrics import load_population
from .session_data import available_sessions
from .state_arousal import (baseline_covariation_tables, find_auxiliary,
                            response_distribution_table)


def _write(path, parts):
    table = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    table.to_csv(path, index=False)
    return table


def _cohort_label(row):
    line = row["population"].split("-")[0]
    depth = str(row.get("depth_class", "") or "")
    return f"{line} {depth}" if depth not in ("", "na") else f"{line} 10x"


def _plot_baseline(path, session):
    import matplotlib.pyplot as plt

    metrics = ["population_f0_vs_pupil_rho", "population_f0_vs_speed_rho",
               "population_f0_vs_respiration_rho"]
    labels = ["pupil", "running", "respiration frequency"]
    cohorts = list(dict.fromkeys(session.cohort.astype(str)))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True,
                             constrained_layout=True)
    rng = np.random.default_rng(2)
    for ax, metric, label in zip(axes, metrics, labels):
        for x, cohort in enumerate(cohorts):
            subset = session[session.cohort.astype(str) == cohort]
            for compartment, marker, color in (("units", "o", "#5b8db8"),
                                                ("somas", "o", "#5b8db8"),
                                                ("groups", "s", "#d88c4a")):
                values = subset[subset.compartment == compartment]
                if values.empty:
                    continue
                jitter = rng.uniform(-.10, .10, len(values))
                ax.scatter(x+jitter, values[metric], s=32, alpha=.7,
                           marker=marker, color=color,
                           label=compartment if x == 0 else None)
            mouse = subset.groupby("mouse", as_index=False)[metric].median()
            ax.scatter(np.full(len(mouse), x), mouse[metric], s=70,
                       facecolors="white", edgecolors="black", linewidths=1.2)
        ax.axhline(0, color=".6", lw=1)
        ax.set(title=f"detrended baseline F0 vs {label}",
               xticks=range(len(cohorts)), xticklabels=cohorts, ylim=(-1, 1))
        ax.tick_params(axis="x", rotation=35)
    axes[0].set_ylabel("within-session Spearman rho\nsmall=session; large=mouse median")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("Awake pre-odor tonic fluorescence covariation")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_baseline_partial(path, session):
    """Show whether running or respiration retains the independent F0 link."""
    import matplotlib.pyplot as plt

    metrics = ["population_f0_vs_speed_given_respiration_rho",
               "population_f0_vs_respiration_given_speed_rho"]
    labels = ["running | respiration", "respiration | running"]
    cohorts = list(dict.fromkeys(session.cohort.astype(str)))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.3), sharey=True,
                             constrained_layout=True)
    rng = np.random.default_rng(8)
    for ax, metric, label in zip(axes, metrics, labels):
        for x, cohort in enumerate(cohorts):
            subset = session[session.cohort.astype(str) == cohort]
            for compartment, marker, color in (("somas", "o", "#5b8db8"),
                                                ("groups", "s", "#d88c4a"),
                                                ("units", "o", "#5b8db8")):
                values = subset[subset.compartment == compartment]
                ax.scatter(x+rng.uniform(-.08, .08, len(values)), values[metric],
                           marker=marker, color=color, alpha=.65, s=30,
                           label=compartment if x == 0 else None)
            mouse = subset.groupby("mouse", as_index=False)[metric].median()
            ax.scatter([x]*len(mouse), mouse[metric], s=65, facecolors="white",
                       edgecolors="black", linewidths=1.1)
        ax.axhline(0, color=".6", lw=1)
        ax.set(title=label, xticks=range(len(cohorts)), xticklabels=cohorts,
               ylim=(-1, 1))
        ax.tick_params(axis="x", rotation=35)
    axes[0].set_ylabel("partial Spearman rho with detrended F0")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("DAT baseline F0 tracks respiration more consistently than running")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _tail_effects(distribution):
    odor = distribution[(distribution.time_s >= 0) & (distribution.time_s < 4)]
    keys = ["group_id", "mouse", "line", "depth_class", "cohort", "compartment",
            "arousal_measure", "arousal_class"]
    reduced = odor.groupby(keys, dropna=False).agg(
        q10_z=("q10_z", "mean"), q90_z=("q90_z", "mean"),
        central80_width_z=("central80_width_z", "mean"),
    ).reset_index()
    index = [name for name in keys if name != "arousal_class"]
    wide = reduced.pivot(index=index, columns="arousal_class",
                         values=["q10_z", "q90_z", "central80_width_z"])
    wide.columns = [f"{metric}_{level}" for metric, level in wide.columns]
    wide = wide.reset_index()
    wide["negative_tail_high_minus_low"] = wide.q10_z_high-wide.q10_z_low
    wide["positive_tail_high_minus_low"] = wide.q90_z_high-wide.q90_z_low
    wide["width_high_minus_low"] = (wide.central80_width_z_high-
                                     wide.central80_width_z_low)
    return wide


def _plot_tail_summary(path, effects):
    import matplotlib.pyplot as plt

    measures = ["pupil_dilation", "running"]
    metrics = ["negative_tail_high_minus_low", "positive_tail_high_minus_low",
               "width_high_minus_low"]
    labels = ["negative tail", "positive tail", "distribution width"]
    cohorts = list(dict.fromkeys(effects.cohort.astype(str)))
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True,
                             constrained_layout=True)
    rng = np.random.default_rng(4)
    for row_index, measure in enumerate(measures):
        subset_measure = effects[effects.arousal_measure == measure]
        for ax, metric, label in zip(axes[row_index], metrics, labels):
            for x, cohort in enumerate(cohorts):
                subset = subset_measure[subset_measure.cohort.astype(str) == cohort]
                for compartment, marker, color in (("units", "o", "#5b8db8"),
                                                    ("somas", "o", "#5b8db8"),
                                                    ("groups", "s", "#d88c4a")):
                    values = subset[subset.compartment == compartment]
                    jitter = rng.uniform(-.10, .10, len(values))
                    ax.scatter(x+jitter, values[metric], s=30, alpha=.65,
                               marker=marker, color=color,
                               label=compartment if x == 0 else None)
                mouse = subset.groupby("mouse", as_index=False)[metric].median()
                ax.scatter(np.full(len(mouse), x), mouse[metric], s=65,
                           facecolors="white", edgecolors="black", linewidths=1.1)
            ax.axhline(0, color=".6", lw=1)
            ax.set(title=f"{measure.replace('_', ' ')}: {label}",
                   xticks=range(len(cohorts)), xticklabels=cohorts)
            ax.tick_params(axis="x", rotation=35)
    axes[0, 0].set_ylabel("high - low arousal (z)")
    axes[1, 0].set_ylabel("high - low arousal (z)")
    axes[0, -1].legend(frameon=False, fontsize=8)
    fig.suptitle("Awake odor-period response-distribution changes\n"
                 "odor median time course removed; small=session, large=mouse median")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_example(path, distribution, group_id, population="somas"):
    import matplotlib.pyplot as plt

    subset = distribution[(distribution.group_id == int(group_id)) &
                          (distribution.compartment == population)]
    if subset.empty:
        return False
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True,
                             constrained_layout=True)
    for column, measure in enumerate(("pupil_dilation", "running")):
        data = subset[subset.arousal_measure == measure]
        for level, color in (("low", "#4979b8"), ("high", "#d24b40")):
            values = data[data.arousal_class == level].sort_values("time_s")
            axes[0, column].plot(values.time_s, values.q10_z, color=color,
                                 label=f"{level} q10")
            axes[0, column].plot(values.time_s, values.q90_z, color=color,
                                 linestyle="--", label=f"{level} q90")
            axes[1, column].plot(values.time_s, values.driver_value,
                                 color=color, label=level)
        for ax in axes[:, column]:
            ax.axvspan(0, 4, color=".9", zorder=-5)
            ax.axvline(0, color=".5", lw=1)
        axes[0, column].set_title(measure.replace("_", " "))
        axes[0, column].set_ylabel("neural distribution tail (z)")
        axes[1, column].set_ylabel(data.driver_unit.iloc[0])
        axes[1, column].set_xlabel("time from odor onset (s)")
        axes[0, column].legend(frameon=False, fontsize=8, ncol=2)
    first = subset.iloc[0]
    fig.suptitle(f"Group {group_id}: {first.cohort}, {population}\n"
                 "odor-centered neural response distributions")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=repo_path("analysis", "stage0", "ketxyl_16odor_session_manifest.csv"))
    parser.add_argument("--imaging-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path,
                        default=repo_path("analysis", "figures", "arousal_outputs"))
    parser.add_argument("--objective", choices=("all", "10x", "20x"), default="all")
    parser.add_argument("--groups", nargs="*", type=int, default=[])
    parser.add_argument("--example-group", type=int, default=202)
    args = parser.parse_args(argv)
    root = imaging_root(args.imaging_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inventory = pd.DataFrame(available_sessions(args.manifest, root))
    inventory = inventory[inventory.population.str.startswith(("TH-", "DAT-"))]
    inventory = inventory[inventory.available]
    if args.objective != "all":
        inventory = inventory[inventory.objective.str.lower() == args.objective]
    if args.groups:
        inventory = inventory[inventory.group_id.astype(int).isin(args.groups)]
    inventory.to_csv(args.output_dir / "session_inventory.csv", index=False)

    units, sessions, distributions, failures = [], [], [], []
    for row in tqdm(inventory.to_dict("records"), desc="awake arousal analyses",
                    unit="session"):
        populations = ["units"] if row["objective"].lower() == "10x" else ["somas", "groups"]
        row = dict(row); row["cohort"] = _cohort_label(row)
        aux = find_auxiliary(row, root)
        if aux is None:
            failures.append({"group_id": int(row["group_id"]), "error": "auxiliary file missing"})
            continue
        for population in populations:
            try:
                data = load_population(row["grouped_path"], population)
                unit, session = baseline_covariation_tables(
                    data, aux, row, population)
                units.append(unit); sessions.append(session)
                distributions.append(response_distribution_table(
                    data, aux, row, population, odor_center=True))
            except Exception as error:
                failures.append({"group_id": int(row["group_id"]),
                                 "population": population,
                                 "error": f"{type(error).__name__}: {error}"})
    unit = _write(args.output_dir / "baseline_f0_arousal_unit.csv", units)
    session = _write(args.output_dir / "baseline_f0_arousal_session.csv", sessions)
    distribution = _write(args.output_dir / "response_distribution_timecourse.csv",
                          distributions)
    effects = _tail_effects(distribution) if len(distribution) else pd.DataFrame()
    effects.to_csv(args.output_dir / "response_distribution_tail_effects.csv", index=False)
    pd.DataFrame(failures).to_csv(args.output_dir / "failures.csv", index=False)
    if len(session):
        _plot_baseline(args.output_dir / "baseline_f0_arousal_overview.png", session)
    if len(effects):
        _plot_tail_summary(args.output_dir / "response_distribution_tail_overview.png",
                           effects)
        _plot_example(args.output_dir / f"group{args.example_group}_distribution_example.png",
                      distribution, args.example_group)
    print({"sessions": len(session), "unit_rows": len(unit),
           "distribution_rows": len(distribution), "failures": len(failures)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
