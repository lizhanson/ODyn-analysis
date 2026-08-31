"""Stage 2: unit, odor, session and cohort summaries plus the key diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .summarise import (SESSION_KEYS, cohort_summary, odor_summary,
                       session_summary, unit_summary)

UNIT_METRICS = [
    "excitation_breadth", "suppression_breadth", "biphasic_breadth",
    "excitation_mass", "suppression_mass", "es_balance",
    "excitation_sparseness", "suppression_sparseness",
    "across_odor_sd", "across_odor_range", "contrast_sparseness",
    "mean_odor_minus_blank", "max_odor_minus_blank", "blank_response",
]


def load_features(directory):
    files = sorted(Path(directory).glob("*_features.csv.gz"))
    if not files:
        raise FileNotFoundError(f"no feature tables in {directory}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def cancellation_diagnostics(features):
    """How often does a signed 4 s mean hide a real bidirectional response?"""
    real = features[~features.is_blank].copy()
    real["mean_visible"] = (
        (real.mean_z > real.threshold_positive)
        | (real.mean_z < real.threshold_negative))
    biphasic = real[real.biphasic]
    rows = []
    for keys, frame in real.groupby(["cohort", "population", "state"],
                                    observed=True):
        sub = biphasic[(biphasic.cohort == keys[0])
                       & (biphasic.population == keys[1])
                       & (biphasic.state == keys[2])]
        rows.append({
            "cohort": keys[0], "population": keys[1], "state": keys[2],
            "n_pair": len(frame),
            "excited_rate": float(frame.excited.mean()),
            "suppressed_rate": float(frame.suppressed.mean()),
            "biphasic_rate": float(frame.biphasic.mean()),
            # Of the bidirectional pairs, how many would a signed mean miss?
            "biphasic_hidden_by_mean": float((~sub.mean_visible).mean())
            if len(sub) else np.nan,
            "n_biphasic": len(sub),
            "any_response_rate": float(
                (frame.excited | frame.suppressed).mean()),
            "mean_visible_rate": float(frame.mean_visible.mean()),
        })
    return pd.DataFrame(rows)


def blank_false_positive(features):
    """Realised rate on mineral oil, against the pre-odor excursion null."""
    blank = features[features.is_blank]
    return (blank.groupby(["cohort", "population", "state"], observed=True)
            .agg(n=("excited", "size"), blank_excited=("excited", "mean"),
                 blank_suppressed=("suppressed", "mean"),
                 blank_raw_sustained=("raw_mean_sustained", "median"))
            .reset_index())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    out = args.output_dir
    features = load_features(out / "features")
    print(f"loaded {len(features):,} unit-odor rows from "
          f"{features.group_id.nunique()} sessions")

    diagnostics = cancellation_diagnostics(features)
    diagnostics.to_csv(out / "stage2_cancellation.csv", index=False)
    blank = blank_false_positive(features)
    blank.to_csv(out / "stage2_blank_rate.csv", index=False)

    units = unit_summary(features)
    units.to_csv(out / "stage2_unit_summary.csv.gz", index=False,
                 compression="gzip")
    odors = odor_summary(features)
    odors.to_csv(out / "stage2_odor_summary.csv.gz", index=False,
                 compression="gzip")
    sessions = session_summary(units, UNIT_METRICS)
    sessions.to_csv(out / "stage2_session_summary.csv", index=False)
    mouse, cohort = cohort_summary(sessions, UNIT_METRICS)
    mouse.to_csv(out / "stage2_mouse_summary.csv", index=False)
    cohort.to_csv(out / "stage2_cohort_summary.csv")

    print("\n=== cancellation: is the trace-based table justified? ===")
    print(diagnostics.round(3).to_string(index=False))
    print("\n=== mineral-oil rate against the pre-odor null ===")
    print(blank.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
