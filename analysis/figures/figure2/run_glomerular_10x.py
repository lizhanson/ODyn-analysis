"""Batch the Figure 2 calculations outside Jupyter.

The 10x glomerular path uses the same measurements as the 20x cellular path,
so the two scales can be compared directly rather than through differently
defined quantities.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from analysis.figures.paths import imaging_root, repo_path
from analysis.figures.population_metrics import (
    TemporalWindows, breadth_table, load_population, reliability_tables,
    specificity_table, temporal_feature_table, tonic_table,
)
from analysis.figures.panel_geometry import panel_tables
from analysis.figures.session_data import available_sessions

POPULATION = "units"


def _write(path, parts):
    table = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    table.to_csv(path, index=False)
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=repo_path(
        "analysis", "stage0", "ketxyl_16odor_session_manifest.csv"))
    parser.add_argument("--imaging-root", type=Path, default=None,
                        help="ImagingData root; defaults to ODYN_IMAGING_ROOT")
    parser.add_argument("--output-dir", type=Path, default=repo_path(
        "analysis", "figures", "figure2", "outputs", "glomerular_10x"))
    parser.add_argument("--groups", nargs="*", type=int, default=[])
    parser.add_argument("--lines", nargs="*", default=["TH", "DAT", "Thy1"])
    parser.add_argument("--reliability-repeats", type=int, default=50)
    parser.add_argument("--tail-probability", type=float, default=.05,
                        help="pre-odor excursion false-positive rate per unit-odor pair")
    args = parser.parse_args(argv)
    args.imaging_root = imaging_root(args.imaging_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    inventory = pd.DataFrame(available_sessions(
        args.manifest, args.imaging_root, objective="10x"))
    inventory["line"] = inventory.population.str.split("-").str[0]
    inventory = inventory[inventory.line.isin(args.lines)]
    if args.groups:
        inventory = inventory[inventory.group_id.astype(int).isin(args.groups)]
    inventory.to_csv(args.output_dir / "session_inventory.csv", index=False)
    inputs = inventory.loc[inventory.available].to_dict("records")

    windows = TemporalWindows()
    tonic_parts, temporal_parts, breadth_parts = [], [], []
    reliability_unit_parts, reliability_odor_parts, failures = [], [], []
    panel_accuracy_parts, panel_distance_parts = [], []
    for row in tqdm(inputs, desc="10x glomerular analyses", unit="session"):
        try:
            data = load_population(row["grouped_path"], POPULATION)
            tonic_parts.append(tonic_table(data, row, POPULATION))
            temporal = temporal_feature_table(
                data, row, POPULATION, windows=windows,
                tail_probability=args.tail_probability)
            temporal_parts.append(temporal)
            breadth_parts.append(breadth_table(temporal))
            unit, odor = reliability_tables(
                data, row, POPULATION, repeats=args.reliability_repeats,
                seed=int(row["group_id"]), windows=windows)
            reliability_unit_parts.append(unit)
            reliability_odor_parts.append(odor)
            accuracy, distances = panel_tables(data, row, POPULATION)
            panel_accuracy_parts.append(accuracy)
            panel_distance_parts.append(distances)
        except Exception as error:
            failures.append({"group_id": int(row["group_id"]),
                             "error": f"{type(error).__name__}: {error}"})
    _write(args.output_dir / "tonic_f0_units.csv", tonic_parts)
    temporal = _write(args.output_dir / "temporal_unit_odor_features.csv",
                      temporal_parts)
    _write(args.output_dir / "signed_auc_specificity.csv",
           [specificity_table(temporal)] if len(temporal) else [])
    _write(args.output_dir / "responder_breadth.csv", breadth_parts)
    _write(args.output_dir / "unit_tuning_reliability.csv", reliability_unit_parts)
    _write(args.output_dir / "odor_population_reliability.csv", reliability_odor_parts)
    _write(args.output_dir / "panel_classification_accuracy.csv", panel_accuracy_parts)
    _write(args.output_dir / "panel_odor_distances.csv", panel_distance_parts)
    pd.DataFrame(failures).to_csv(args.output_dir / "failures.csv", index=False)
    print({"sessions": len(inputs), "temporal_rows": len(temporal),
           "failures": len(failures)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
