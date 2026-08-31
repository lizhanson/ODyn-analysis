"""Compare anesthesia effects across awake-defined TH soma phenotypes.

The mutually exclusive awake phenotypes are:
  * suppression only: at least one suppressed odor and no excited odors;
  * biphasic: at least one individual odor is both excited and suppressed;
  * cross-odor mixed: excitation and suppression occur for different odors,
    with no biphasic odor.

Calls use the already-generated q=.01 baseline-excursion tables and require at
least four trials in both states for an odor to enter paired evoked metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import repo_path


ID_KEYS = ["group_id", "mouse", "line", "depth_class", "cohort",
           "compartment", "unit_id"]
PHENOTYPES = ("suppression only", "biphasic", "cross-odor mixed")
COLORS = {"suppression only": "#3979b9", "biphasic": "#704c9b",
          "cross-odor mixed": "#d0772d"}


def awake_phenotypes(temporal, *, minimum_trials=4):
    awake = temporal[(temporal.state == "pre") & (~temporal.is_blank) &
                     (temporal.n_trials >= minimum_trials)]
    rows = []
    for key, group in awake.groupby(ID_KEYS, dropna=False):
        excited = set(group.loc[group.excited, "odor_id"].astype(int))
        suppressed = set(group.loc[group.suppressed, "odor_id"].astype(int))
        biphasic = set(group.loc[group.excited & group.suppressed,
                                  "odor_id"].astype(int))
        if suppressed and not excited:
            label = "suppression only"
        elif biphasic:
            label = "biphasic"
        elif excited and suppressed:
            label = "cross-odor mixed"
        elif excited:
            label = "excitation only"
        else:
            label = "neither"
        rows.append(dict(zip(ID_KEYS, key)) | {"awake_phenotype": label})
    return pd.DataFrame(rows)


def paired_state_effects(temporal, f0, *, minimum_trials=4):
    phenotype = awake_phenotypes(temporal, minimum_trials=minimum_trials)
    eligible = temporal[(~temporal.is_blank) &
                        (temporal.n_trials >= minimum_trials)].copy()
    odor_keys = ID_KEYS + ["odor_id"]
    paired = eligible.groupby(odor_keys, dropna=False).state.nunique()
    paired = paired[paired == 2].index
    eligible = eligible.set_index(odor_keys).loc[paired].reset_index()
    state = eligible.groupby(ID_KEYS + ["state"], dropna=False).agg(
        suppression_breadth=("suppressed", "mean"),
        excitation_breadth=("excited", "mean"),
        negative_auc=("negative_auc_z_s", "median"),
        positive_auc=("positive_auc_z_s", "median"),
    ).reset_index()
    wide = state.pivot(index=ID_KEYS, columns="state",
                       values=["suppression_breadth", "excitation_breadth",
                               "negative_auc", "positive_auc"]).reset_index()
    wide.columns = ["_".join(map(str, value)).rstrip("_")
                    for value in wide.columns]
    for metric in ("suppression_breadth", "excitation_breadth",
                   "negative_auc", "positive_auc"):
        wide[metric + "_change"] = wide[metric + "_post"] - wide[metric + "_pre"]
    wide["negative_auc_retained"] = np.divide(
        wide.negative_auc_post, wide.negative_auc_pre,
        out=np.full(len(wide), np.nan), where=wide.negative_auc_pre > 0)
    wide = wide.merge(phenotype, on=ID_KEYS, validate="one_to_one")
    tonic = f0[ID_KEYS + ["f0_log2_post_pre"]].merge(
        phenotype, on=ID_KEYS, validate="one_to_one")
    return wide, tonic


def mouse_summary(evoked, tonic):
    keys = ["mouse", "group_id", "cohort", "awake_phenotype"]
    metrics = ["suppression_breadth_change", "negative_auc_change",
               "negative_auc_retained", "excitation_breadth_change",
               "positive_auc_change"]
    session = evoked.groupby(keys, dropna=False)[metrics].median().reset_index()
    f0 = tonic.groupby(keys, dropna=False).f0_log2_post_pre.median().reset_index()
    session = session.merge(f0, on=keys, how="outer", validate="one_to_one")
    mouse_keys = ["mouse", "cohort", "awake_phenotype"]
    return session.groupby(mouse_keys, dropna=False).median(numeric_only=True).reset_index()


def plot(path, mouse):
    import matplotlib.pyplot as plt

    cohorts = ("TH superficial", "TH deep")
    metrics = (("f0_log2_post_pre", "tonic F0\nlog2 ket/xyl / awake", 0.),
               ("suppression_breadth_change", "change in suppression breadth", 0.),
               ("negative_auc_retained", "negative AUC retained\nket/xyl / awake", 1.),
               ("excitation_breadth_change", "change in excitation breadth", 0.))
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
    for row, cohort in enumerate(cohorts):
        data = mouse[mouse.cohort == cohort]
        for column, (metric, title, reference) in enumerate(metrics):
            ax = axes[row, column]
            for x, phenotype in enumerate(PHENOTYPES):
                values = data.loc[data.awake_phenotype == phenotype, metric].dropna()
                ax.scatter(np.full(len(values), x), values, s=34,
                           color=COLORS[phenotype], alpha=.72)
                if len(values):
                    ax.scatter(x, values.median(), s=76, facecolor="white",
                               edgecolor=COLORS[phenotype], lw=2, zorder=3)
            ax.axhline(reference, color=".65", lw=.8)
            ax.set(title=title, xticks=range(3),
                   xticklabels=("S only", "biphasic", "cross-odor\nmixed"))
            if column == 0:
                ax.set_ylabel(cohort)
    fig.suptitle("Anesthesia reorganizes awake-defined TH soma phenotypes\n"
                 "each point is one mouse median; q=.01 signed calls")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=repo_path(
        "analysis", "figures", "signed_cell_class_outputs_q01"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--minimum-trials", type=int, default=4)
    args = parser.parse_args(argv)
    output = args.output_dir or args.input_dir
    output.mkdir(parents=True, exist_ok=True)
    temporal = pd.read_csv(args.input_dir / "temporal_unit_odor_features.csv")
    f0 = pd.read_csv(args.input_dir / "awake_class_f0_change.csv")
    evoked, tonic = paired_state_effects(
        temporal, f0, minimum_trials=args.minimum_trials)
    mouse = mouse_summary(evoked, tonic)
    evoked.to_csv(output / "awake_phenotype_cell_state_effects.csv", index=False)
    mouse.to_csv(output / "awake_phenotype_mouse_state_effects.csv", index=False)
    plot(output / "awake_phenotype_anesthesia_effects.png", mouse)
    print({"cells": len(evoked), "mice": mouse.mouse.nunique()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
