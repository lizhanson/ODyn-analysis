"""Batch the Figure 3 calculations outside Jupyter.

The notebook performs the same calculations section by section for review.
This CLI is for reproducible full-batch table generation and checkpointing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from analysis.figures.arousal_20x import (arousal_association_table,
                                          find_auxiliary)
from analysis.figures.cellular_20x import (
    TemporalWindows, load_population, reliability_tables, specificity_table,
    temporal_feature_table, tonic_table,
)
from analysis.figures.session_data import available_sessions
from analysis.figures.session_data import load_grouped
from analysis.figures.summaries import signed_session_tables


def _write(path, parts):
    table = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    table.to_csv(path, index=False)
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=Path("analysis/stage0/ketxyl_16odor_session_manifest.csv"))
    parser.add_argument("--imaging-root", type=Path,
                        default=Path("/Volumes/MossLab/ImagingData"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("analysis/figures/figure3/outputs/cellular_20x"))
    parser.add_argument("--groups", nargs="*", type=int, default=[])
    parser.add_argument("--reliability-repeats", type=int, default=50)
    parser.add_argument("--skip-arousal", action="store_true")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inventory = pd.DataFrame(available_sessions(
        args.manifest, args.imaging_root, objective="20x"))
    inventory = inventory[inventory.population.str.startswith(("TH-", "DAT-"))]
    if args.groups:
        inventory = inventory[inventory.group_id.astype(int).isin(args.groups)]
    inventory.to_csv(args.output_dir / "session_inventory.csv", index=False)
    inputs = inventory.loc[inventory.available].to_dict("records")
    windows = TemporalWindows()
    tonic_parts, temporal_parts, breadth_parts = [], [], []
    reliability_unit_parts, reliability_odor_parts, arousal_parts = [], [], []
    failures = []
    for row in tqdm(inputs, desc="20x cellular analyses", unit="session"):
        for population in ("somas", "processes"):
            try:
                data = load_population(row["grouped_path"], population)
                tonic_parts.append(tonic_table(data, row, population))
                temporal_parts.append(temporal_feature_table(
                    data, row, population, windows=windows))
                session = load_grouped(row, row["grouped_path"], population=population)
                breadth_rows, _ = signed_session_tables(
                    session, blank_odor=0, tail_probability=.01, reducer="median")
                breadth = pd.DataFrame(breadth_rows)
                breadth["cohort"] = f"{row['population'].split('-')[0]} {row['depth_class']}"
                breadth["compartment"] = population
                breadth_parts.append(breadth)
                unit, odor = reliability_tables(
                    data, row, population, repeats=args.reliability_repeats,
                    seed=int(row["group_id"]), windows=windows)
                reliability_unit_parts.append(unit); reliability_odor_parts.append(odor)
                if not args.skip_arousal:
                    aux = find_auxiliary(row, args.imaging_root)
                    if aux is not None:
                        arousal_parts.append(arousal_association_table(
                            row["grouped_path"], aux, row, population, windows=windows))
            except Exception as error:
                failures.append({"group_id": int(row["group_id"]),
                                 "population": population,
                                 "error": f"{type(error).__name__}: {error}"})
    tonic = _write(args.output_dir / "tonic_f0_units.csv", tonic_parts)
    temporal = _write(args.output_dir / "temporal_unit_odor_features.csv", temporal_parts)
    _write(args.output_dir / "signed_auc_specificity.csv",
           [specificity_table(temporal)] if len(temporal) else [])
    _write(args.output_dir / "asymmetric_blank_responder_breadth.csv", breadth_parts)
    _write(args.output_dir / "unit_tuning_reliability.csv", reliability_unit_parts)
    _write(args.output_dir / "odor_population_reliability.csv", reliability_odor_parts)
    _write(args.output_dir / "within_odor_arousal_associations.csv", arousal_parts)
    pd.DataFrame(failures).to_csv(args.output_dir / "failures.csv", index=False)
    print({"sessions": len(inputs), "tonic_rows": len(tonic),
           "temporal_rows": len(temporal), "failures": len(failures)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
