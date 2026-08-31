"""Compute and plot 20x soma signed odor-response classes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .paths import imaging_root, repo_path
from .population_metrics import TemporalWindows, load_population, temporal_feature_table, tonic_table
from .session_data import available_sessions
from .signed_cell_classes import (awake_class_f0_change, cell_signed_classes,
                                  class_composition, class_state_transitions)


COLORS = {"neither": ".72", "excitation only": "#d45b35",
          "suppression only": "#3979b9", "both": "#704c9b"}


def _plot(path, composition, transitions, f0, *, unit_label="soma"):
    import matplotlib.pyplot as plt

    cohorts = ("DAT superficial", "TH superficial", "TH deep")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for column, cohort in enumerate(cohorts):
        ax = axes[0, column]
        data = composition[(composition.cohort == cohort) &
                           (composition.measure == "signed class")]
        for state, x in (("pre", 0), ("post", 1)):
            session = data[data.state == state]
            medians = session.groupby("value_label").fraction.median()
            bottom = 0.
            for label in COLORS:
                value = medians.get(label, 0.)
                ax.bar(x, value, bottom=bottom, color=COLORS[label], width=.62,
                       label=label if column == 0 and state == "pre" else None)
                bottom += value
        ax.set(title=cohort, xlim=(-.6, 1.6), ylim=(0, 1), xticks=(0, 1),
               xticklabels=("awake", "ket/xyl"), ylabel="median session fraction")
        if column == 0:
            ax.legend(frameon=False, fontsize=8, loc="upper left")
        ax = axes[1, column]
        data = f0[f0.cohort == cohort]
        order = ("neither", "excitation only", "suppression only", "both")
        for x, label in enumerate(order):
            session = data[data.signed_class == label].groupby(["mouse", "group_id"])["f0_log2_post_pre"].median()
            ax.scatter(np.full(len(session), x), session, s=14, color=COLORS[label], alpha=.35)
            mouse = session.groupby(level="mouse").median()
            ax.scatter(np.full(len(mouse), x), mouse, s=38, color=COLORS[label], edgecolor="black", lw=.4)
        ax.axhline(0, color=".65", lw=.8)
        ax.set(xticks=range(4), xticklabels=("neither", "E only", "S only", "both"),
               ylabel="log2 F0 ket/xyl / awake", title="Awake-defined cell class")
    fig.suptitle(f"20x {unit_label} signed response classes and tonic state change\n"
                 "calls: ≥4 trials per odor/state; points=session and mouse median")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_suppression_only_fate(path, transitions, *, unit_label="soma"):
    """Focused fate of awake suppression-only somas after ket/xyl."""
    import matplotlib.pyplot as plt

    cohorts = ("DAT superficial", "TH superficial", "TH deep")
    order = ("suppression only", "excitation only", "both", "neither")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True,
                             constrained_layout=True)
    data = transitions[transitions.awake_class == "suppression only"]
    for ax, cohort in zip(axes, cohorts):
        subset = data[data.cohort == cohort]
        session = subset.pivot_table(index=["mouse", "group_id"],
                                     columns="post_class", values="fraction")
        mouse = session.groupby(level="mouse").median()
        values = [mouse.get(label, pd.Series(dtype=float)).dropna().to_numpy()
                  for label in order]
        for x, value in enumerate(values):
            ax.scatter(np.full(len(value), x), value, color=COLORS[order[x]],
                       alpha=.4, s=20)
            if len(value):
                ax.scatter(x, np.median(value), color=COLORS[order[x]],
                           edgecolor="black", lw=.45, s=58, zorder=3)
        ax.set(title=cohort, xticks=range(4),
               xticklabels=("remain\nS only", "become\nE only", "become\nboth", "become\nneither"),
               ylim=(-.04, 1.04), ylabel="fraction of awake S-only cells")
    fig.suptitle(f"Fate of awake suppression-only {unit_label} ROIs under ket/xyl\n"
                 "small=session; large=mouse median")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=repo_path(
        "analysis", "stage0", "ketxyl_16odor_session_manifest.csv"))
    parser.add_argument("--imaging-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=repo_path(
        "analysis", "figures", "signed_cell_class_outputs"))
    parser.add_argument("--minimum-trials", type=int, default=4)
    parser.add_argument("--tail-probability", type=float, default=.01,
                        help="per odor/sign baseline-excursion false-positive rate")
    parser.add_argument("--population", choices=("somas", "processes"),
                        default="somas")
    args = parser.parse_args(argv)
    root = imaging_root(args.imaging_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = pd.DataFrame(available_sessions(args.manifest, root, objective="20x"))
    rows = rows[rows.available & rows.population.str.startswith(("TH-", "DAT-"))]
    temporal_parts, tonic_parts, failures = [], [], []
    for row in tqdm(rows.to_dict("records"), desc="signed soma classes", unit="session"):
        try:
            data = load_population(row["grouped_path"], args.population)
            temporal_parts.append(temporal_feature_table(
                data, row, args.population, windows=TemporalWindows(),
                tail_probability=args.tail_probability))
            tonic_parts.append(tonic_table(data, row, args.population))
        except Exception as error:
            failures.append({"group_id": int(row["group_id"]),
                             "error": f"{type(error).__name__}: {error}"})
    temporal = pd.concat(temporal_parts, ignore_index=True) if temporal_parts else pd.DataFrame()
    tonic = pd.concat(tonic_parts, ignore_index=True) if tonic_parts else pd.DataFrame()
    classes = cell_signed_classes(temporal, minimum_trials=args.minimum_trials)
    composition = class_composition(classes)
    transitions = class_state_transitions(classes)
    f0 = awake_class_f0_change(classes, tonic)
    temporal.to_csv(args.output_dir / "temporal_unit_odor_features.csv", index=False)
    label = "soma" if args.population == "somas" else "process"
    classes.to_csv(args.output_dir / f"{label}_signed_classes.csv", index=False)
    composition.to_csv(args.output_dir / f"{label}_signed_class_composition.csv", index=False)
    transitions.to_csv(args.output_dir / f"{label}_signed_class_transitions.csv", index=False)
    f0.to_csv(args.output_dir / "awake_class_f0_change.csv", index=False)
    pd.DataFrame(failures).to_csv(args.output_dir / "failures.csv", index=False)
    _plot(args.output_dir / f"{label}_signed_classes_and_f0.png",
          composition, transitions, f0, unit_label=label)
    _plot_suppression_only_fate(
        args.output_dir / f"{label}_suppression_only_fate.png", transitions,
        unit_label=label)
    print({"cells": len(classes), "sessions": classes.group_id.nunique(),
           "tail_probability": args.tail_probability,
           "failures": len(failures)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
