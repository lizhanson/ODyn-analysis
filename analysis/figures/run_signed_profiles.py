"""Answer the three signed-profile questions on the 20x cellular populations.

Question 1  How often is one cell excited by one odor and suppressed by another?
Question 2  Is there a suppression-only population distinct from a continuum?
Question 3  Are those cells preferentially silenced by anesthesia?

Question 3 uses a split-half design. A cell classified as suppression-only on
the same trials the state change is measured on is partly selected for noise,
and that noise is gone by the second measurement whatever anesthesia does. The
awake trials of each odor are therefore split: one half defines the class, the
other half provides the awake side of the comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.figures.paths import imaging_root, output_root, repo_path
from analysis.figures.population_metrics import (TemporalWindows,
                                                 load_population,
                                                 temporal_feature_table,
                                                 tonic_table)
from analysis.figures.session_data import available_sessions
from analysis.figures.signed_profiles import (KEYS, breadth_matched_change,
                                              class_fractions,
                                              mouse_then_cohort,
                                              paired_state_change,
                                              sign_permutation_null,
                                              split_awake_trials,
                                              subset_trials,
                                              unit_sign_profiles)

WINDOWS = TemporalWindows()
BLANK_ODOR = 0
TAIL_PROBABILITY = 0.05


def _features(data, row, population, tail_probability=TAIL_PROBABILITY):
    return temporal_feature_table(
        data, row, population, windows=WINDOWS, blank_odor=BLANK_ODOR,
        tail_probability=tail_probability)


def chance_bidirectional(n_odor, tail_probability):
    """Fraction of pure-noise cells that would be called bidirectional.

    Each unit-odor pair is called independently at `tail_probability` per sign,
    so over many odors a cell that responds to nothing still collects calls of
    both signs. Reported beside every observed fraction because the
    classification needs only one odor of each sign to fire.
    """
    either = 1.0 - (1.0 - float(tail_probability)) ** int(n_odor)
    return either ** 2


def session_tables(row, population, *, seed=0, tail_probability=TAIL_PROBABILITY):
    """Full-trial profiles, plus the split-half pair for the state question."""
    data = load_population(row["grouped_path"], population)
    full = _features(data, row, population, tail_probability)
    if full.empty:
        return None
    profiles = unit_sign_profiles(full)
    tonic = tonic_table(data, row, population)

    classify_mask, measure_mask = split_awake_trials(data, seed=seed)
    paired = pd.DataFrame()
    if classify_mask.any():
        classify = unit_sign_profiles(_features(
            subset_trials(data, classify_mask), row, population, tail_probability))
        measure = unit_sign_profiles(_features(
            subset_trials(data, measure_mask), row, population, tail_probability))
        if not classify.empty and not measure.empty:
            keys = [k for k in KEYS if k != "state"]
            labels = (classify[[*keys, "unit_id", "response_class",
                                "n_suppressed", "n_excited"]]
                      .rename(columns={"response_class": "awake_class",
                                       "n_suppressed": "awake_half_n_suppressed",
                                       "n_excited": "awake_half_n_excited"}))
            try:
                paired = paired_state_change(measure).merge(
                    labels, on=[*keys, "unit_id"], how="inner")
            except ValueError:
                paired = pd.DataFrame()      # session lacks one of the states
    return {"temporal": full, "profiles": profiles, "tonic": tonic,
            "paired": paired}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--population", default="somas",
                        help="20x compartment: somas, groups, or processes")
    parser.add_argument("--manifest", type=Path, default=repo_path(
        "analysis", "stage0", "ketxyl_16odor_session_manifest.csv"))
    parser.add_argument("--imaging-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path,
                        default=output_root() / "signed_profiles")
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--tail-probability", type=float, default=TAIL_PROBABILITY,
                        help="per unit-odor-sign false-positive rate of the "
                             "pre-odor excursion null; 0.01 is the strict "
                             "sensitivity setting")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    root = imaging_root(args.imaging_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inventory = [r for r in available_sessions(args.manifest, root, objective="20x")
                 if r["available"]]
    print(f"{len(inventory)} available 20x sessions", flush=True)

    parts = {"temporal": [], "profiles": [], "tonic": [], "paired": []}
    for index, row in enumerate(inventory, start=1):
        label = f"group {row['group_id']} {row['mouse']} {row['population']}"
        try:
            tables = session_tables(row, args.population, seed=args.seed,
                                    tail_probability=args.tail_probability)
        except Exception as error:
            print(f"  [{index}/{len(inventory)}] {label}: FAILED "
                  f"{type(error).__name__}: {error}", flush=True)
            continue
        if tables is None:
            print(f"  [{index}/{len(inventory)}] {label}: no features", flush=True)
            continue
        for name, table in tables.items():
            if table is not None and len(table):
                parts[name].append(table)
        print(f"  [{index}/{len(inventory)}] {label}: "
              f"{tables['profiles'].unit_id.nunique()} units, "
              f"{len(tables['paired'])} paired", flush=True)

    if not parts["profiles"]:
        raise RuntimeError("no sessions produced profiles")
    temporal = pd.concat(parts["temporal"], ignore_index=True)
    profiles = pd.concat(parts["profiles"], ignore_index=True)
    tonic = pd.concat(parts["tonic"], ignore_index=True) if parts["tonic"] else pd.DataFrame()
    paired = pd.concat(parts["paired"], ignore_index=True) if parts["paired"] else pd.DataFrame()

    fractions = class_fractions(profiles)
    metrics = ["silent", "excited_only", "suppressed_only", "bidirectional",
               "cross_odor_bidirectional", "within_odor_biphasic"]
    per_mouse, cohort = mouse_then_cohort(fractions, metrics)
    null = sign_permutation_null(temporal, repeats=args.repeats, seed=args.seed)

    stem = f"{args.population}_q{args.tail_probability:g}"
    profiles.to_csv(args.output_dir / f"{stem}_unit_profiles.csv", index=False)
    fractions.to_csv(args.output_dir / f"{stem}_session_class_fractions.csv", index=False)
    per_mouse.to_csv(args.output_dir / f"{stem}_mouse_class_fractions.csv", index=False)
    cohort.to_csv(args.output_dir / f"{stem}_cohort_class_fractions.csv", index=False)
    null.to_csv(args.output_dir / f"{stem}_sign_permutation_null.csv", index=False)
    if len(tonic):
        tonic.to_csv(args.output_dir / f"{stem}_tonic_f0.csv", index=False)
    if len(paired):
        if len(tonic):
            f0 = (tonic[tonic.state == "pre"]
                  [["group_id", "unit_id", "f0_log2_post_pre"]].drop_duplicates())
            paired = paired.merge(f0, on=["group_id", "unit_id"], how="left")
        paired.to_csv(args.output_dir / f"{stem}_paired_state_change.csv", index=False)

    print("\n=== Q1/Q2: awake class fractions, mice within cohort ===", flush=True)
    awake = cohort[cohort.state == "pre"] if "state" in cohort else cohort
    print(awake.to_string(index=False))
    print("\n=== Q2: suppression-only versus the within-odor sign null ===")
    if len(null):
        columns = ["cohort", "state", "n_unit", "suppressed_only_observed",
                   "suppressed_only_null_mean", "suppressed_only_excess",
                   "suppressed_only_p_greater"]
        print(null[null.state == "pre"][columns].to_string(index=False))
    print("\n=== Q3: state change by awake class (split-half) ===")
    matched = pd.DataFrame()
    if len(paired):
        summary = paired.groupby(["cohort", "awake_class"], dropna=False).agg(
            n=("unit_id", "size"),
            suppression_breadth_change=("suppression_breadth_change", "median"),
            excitation_breadth_change=("excitation_breadth_change", "median"),
            negative_auc_log2=("negative_auc_log2_change", "median"),
            f0_log2=("f0_log2_post_pre", "median")).reset_index()
        summary.to_csv(args.output_dir / f"{stem}_state_change_by_class.csv", index=False)
        print(summary.to_string(index=False))
        frames = [breadth_matched_change(paired, metric=metric)
                  for metric in ("suppression_breadth_change",
                                 "negative_auc_log2_change")]
        matched = pd.concat([f for f in frames if len(f)], ignore_index=True) \
            if any(len(f) for f in frames) else pd.DataFrame()
        if len(matched):
            matched.to_csv(args.output_dir / f"{stem}_breadth_matched.csv",
                           index=False)
            print("\n=== Q3 control: matched on awake suppression breadth ===")
            print("(if difference_matched collapses toward 0, starting level "
                  "was the story, not cell class)")
            print(matched[["cohort", "metric", "n_target", "n_reference",
                           "target_awake_breadth", "reference_awake_breadth",
                           "difference_raw", "difference_matched",
                           "target_coverage"]].to_string(index=False))
    chance = chance_bidirectional(int(profiles.n_odor.median()),
                                  args.tail_probability)
    print(f"\nChance bidirectional fraction for a pure-noise cell at "
          f"q={args.tail_probability:g} over {int(profiles.n_odor.median())} "
          f"odors: {chance:.3f}")
    print(f"Observed awake median n_excited="
          f"{profiles[profiles.state=='pre'].n_excited.median():.0f}, "
          f"n_suppressed={profiles[profiles.state=='pre'].n_suppressed.median():.0f}")
    print(f"\nWrote tables to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
