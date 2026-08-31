"""Matched-session comparison of TH soma and process signed phenotypes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

def repo_path(*parts):
    return Path(__file__).resolve().parents[2].joinpath(*parts)


PHENOTYPES = ("suppression only", "biphasic", "cross-odor mixed",
              "excitation only", "neither")
POST_CLASSES = ("suppression only", "excitation only", "both", "neither")
COLORS = {"suppression only": "#3979b9", "biphasic": "#704c9b",
          "cross-odor mixed": "#d0772d", "excitation only": "#d45b35",
          "both": "#704c9b", "neither": ".72"}

ID_KEYS = ["group_id", "mouse", "line", "depth_class", "cohort",
           "compartment", "unit_id"]


def awake_phenotypes(temporal, *, minimum_trials=4):
    awake = temporal[(temporal.state == "pre") & (~temporal.is_blank) &
                     (temporal.n_trials >= minimum_trials)]
    rows = []
    for key, group in awake.groupby(ID_KEYS, dropna=False):
        excited = set(group.loc[group.excited, "odor_id"].astype(int))
        suppressed = set(group.loc[group.suppressed, "odor_id"].astype(int))
        biphasic = set(group.loc[group.excited & group.suppressed,
                                  "odor_id"].astype(int))
        label = ("suppression only" if suppressed and not excited else
                 "biphasic" if biphasic else
                 "cross-odor mixed" if excited and suppressed else
                 "excitation only" if excited else "neither")
        rows.append(dict(zip(ID_KEYS, key)) | {"awake_phenotype": label})
    return pd.DataFrame(rows)


def paired_state_effects(temporal, f0, *, minimum_trials=4):
    phenotype = awake_phenotypes(temporal, minimum_trials=minimum_trials)
    data = temporal[(~temporal.is_blank) &
                    (temporal.n_trials >= minimum_trials)].copy()
    odor_keys = ID_KEYS + ["odor_id"]
    paired = data.groupby(odor_keys, dropna=False).state.nunique()
    data = data.set_index(odor_keys).loc[paired[paired == 2].index].reset_index()
    state = data.groupby(ID_KEYS + ["state"], dropna=False).agg(
        suppression_breadth=("suppressed", "mean"),
        excitation_breadth=("excited", "mean"),
        negative_auc=("negative_auc_z_s", "median"),
        positive_auc=("positive_auc_z_s", "median")).reset_index()
    wide = state.pivot(index=ID_KEYS, columns="state",
                       values=["suppression_breadth", "excitation_breadth",
                               "negative_auc", "positive_auc"]).reset_index()
    wide.columns = ["_".join(map(str, value)).rstrip("_") for value in wide.columns]
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


def _load(root):
    temporal = pd.read_csv(root / "temporal_unit_odor_features.csv")
    f0 = pd.read_csv(root / "awake_class_f0_change.csv")
    phenotype = awake_phenotypes(temporal)
    evoked, tonic = paired_state_effects(temporal, f0)
    return phenotype, evoked, tonic


def _mouse_composition(phenotype, compartment):
    session = phenotype.groupby(
        ["mouse", "group_id", "cohort"]).awake_phenotype.value_counts(
            normalize=True).rename("fraction").reset_index()
    session["compartment"] = compartment
    return session.groupby(
        ["mouse", "cohort", "compartment", "awake_phenotype"],
        as_index=False).fraction.median()


def _mouse_fates(path, groups, compartment):
    data = pd.read_csv(path)
    data = data[(data.group_id.isin(groups)) &
                (data.awake_class == "suppression only")]
    data["compartment"] = compartment
    return data.groupby(
        ["mouse", "cohort", "compartment", "post_class"],
        as_index=False).fraction.median()


def plot_composition_fate(path, composition, fate):
    import matplotlib.pyplot as plt

    cohorts = ("TH superficial", "TH deep")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for row, cohort in enumerate(cohorts):
        ax = axes[row, 0]
        for x, compartment in enumerate(("soma", "process")):
            data = composition[(composition.cohort == cohort) &
                               (composition.compartment == compartment)]
            med = data.groupby("awake_phenotype").fraction.median()
            bottom = 0.
            for label in PHENOTYPES:
                value = med.get(label, 0.)
                ax.bar(x, value, bottom=bottom, color=COLORS[label], width=.62,
                       label=label if row == 0 and x == 0 else None)
                bottom += value
        ax.set(title=f"{cohort}: awake phenotype", xticks=(0, 1),
               xticklabels=("somas", "processes"), ylim=(0, 1),
               ylabel="mouse-median fraction")
        ax = axes[row, 1]
        for x, compartment in enumerate(("soma", "process")):
            data = fate[(fate.cohort == cohort) &
                        (fate.compartment == compartment)]
            med = data.groupby("post_class").fraction.median()
            bottom = 0.
            for label in POST_CLASSES:
                value = med.get(label, 0.)
                ax.bar(x, value, bottom=bottom, color=COLORS[label], width=.62)
                bottom += value
        ax.set(title="Fate of awake suppression-only ROIs", xticks=(0, 1),
               xticklabels=("somas", "processes"), ylim=(0, 1),
               ylabel="mouse-median fraction")
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=2, loc="upper center")
    fig.suptitle("TH signed odor-response phenotypes differ by compartment\n"
                 "matched sessions; q=.01 signed calls")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_state_effects(path, evoked, tonic):
    import matplotlib.pyplot as plt

    cohorts = ("TH superficial", "TH deep")
    metrics = (("f0_log2_post_pre", "tonic F0\nlog2 ket/xyl / awake", 0.),
               ("suppression_breadth_change", "change in suppression breadth", 0.),
               ("negative_auc_retained", "negative AUC retained", 1.),
               ("excitation_breadth_change", "change in excitation breadth", 0.))
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
    for row, cohort in enumerate(cohorts):
        for column, (metric, title, reference) in enumerate(metrics):
            ax = axes[row, column]
            source = tonic if metric == "f0_log2_post_pre" else evoked
            data = source[(source.cohort == cohort) &
                          (source.awake_phenotype == "suppression only")]
            for x, compartment in enumerate(("soma", "process")):
                values = data.loc[data.display_compartment == compartment, metric].dropna()
                ax.scatter(np.full(len(values), x), values, s=34, alpha=.65,
                           color=COLORS["suppression only"])
                if len(values):
                    ax.scatter(x, values.median(), s=76, facecolor="white",
                               edgecolor=COLORS["suppression only"], lw=2)
            ax.axhline(reference, color=".65", lw=.8)
            ax.set(title=title, xticks=(0, 1), xticklabels=("somas", "processes"))
            if column == 0:
                ax.set_ylabel(cohort)
    fig.suptitle("State effects on awake suppression-only TH ROIs\n"
                 "each point is one mouse median from matched sessions")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _mouse_effects(table, compartment):
    keys = ["mouse", "group_id", "cohort", "awake_phenotype"]
    metrics = [name for name in ("suppression_breadth_change",
                                 "negative_auc_retained",
                                 "excitation_breadth_change",
                                 "f0_log2_post_pre") if name in table]
    session = table.groupby(keys, dropna=False)[metrics].median().reset_index()
    mouse = session.groupby(
        ["mouse", "cohort", "awake_phenotype"], as_index=False).median(
            numeric_only=True)
    mouse["display_compartment"] = compartment
    return mouse


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--soma-dir", type=Path, default=repo_path(
        "analysis", "figures", "signed_cell_class_outputs_q01"))
    parser.add_argument("--process-dir", type=Path, default=repo_path(
        "analysis", "figures", "signed_process_class_outputs_q01"))
    parser.add_argument("--output-dir", type=Path, default=repo_path(
        "analysis", "figures", "signed_compartment_outputs_q01"))
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    soma_ph, soma_evoked, soma_tonic = _load(args.soma_dir)
    process_ph, process_evoked, process_tonic = _load(args.process_dir)
    groups = set(process_ph.group_id.unique())
    soma_ph = soma_ph[soma_ph.group_id.isin(groups)]
    soma_evoked = soma_evoked[soma_evoked.group_id.isin(groups)]
    soma_tonic = soma_tonic[soma_tonic.group_id.isin(groups)]
    composition = pd.concat([_mouse_composition(soma_ph, "soma"),
                             _mouse_composition(process_ph, "process")],
                            ignore_index=True)
    fate = pd.concat([
        _mouse_fates(args.soma_dir / "soma_signed_class_transitions.csv",
                     groups, "soma"),
        _mouse_fates(args.process_dir / "process_signed_class_transitions.csv",
                     groups, "process")], ignore_index=True)
    evoked = pd.concat([_mouse_effects(soma_evoked, "soma"),
                        _mouse_effects(process_evoked, "process")],
                       ignore_index=True)
    tonic = pd.concat([_mouse_effects(soma_tonic, "soma"),
                       _mouse_effects(process_tonic, "process")],
                      ignore_index=True)
    composition.to_csv(args.output_dir / "mouse_phenotype_composition.csv", index=False)
    fate.to_csv(args.output_dir / "mouse_suppression_only_fate.csv", index=False)
    evoked.to_csv(args.output_dir / "mouse_suppression_only_evoked_effects.csv", index=False)
    tonic.to_csv(args.output_dir / "mouse_suppression_only_f0_effects.csv", index=False)
    plot_composition_fate(args.output_dir / "soma_process_phenotypes_and_fate.png",
                          composition, fate)
    plot_state_effects(args.output_dir / "soma_process_suppression_only_state_effects.png",
                       evoked, tonic)
    print({"matched_sessions": len(groups), "mice": composition.mouse.nunique()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
