"""Fast baseline-only arousal analysis using small H5 baseline arrays."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from .paths import imaging_root, repo_path
from .session_data import available_sessions
from .state_arousal import (baseline_covariation_tables, find_auxiliary,
                            load_baseline_population)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=repo_path("analysis", "stage0", "ketxyl_16odor_session_manifest.csv"))
    parser.add_argument("--imaging-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path,
                        default=repo_path("analysis", "figures", "arousal_outputs"))
    parser.add_argument("--objective", choices=("all", "10x", "20x"), default="20x")
    args = parser.parse_args(argv)
    root = imaging_root(args.imaging_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inventory = pd.DataFrame(available_sessions(args.manifest, root))
    inventory = inventory[inventory.available & inventory.population.str.startswith(("TH-", "DAT-"))]
    if args.objective != "all":
        inventory = inventory[inventory.objective.str.lower() == args.objective]
    unit_parts, session_parts, failures = [], [], []
    for row in tqdm(inventory.to_dict("records"), desc="baseline arousal", unit="session"):
        auxiliary = find_auxiliary(row, root)
        if auxiliary is None:
            failures.append({"group_id": int(row["group_id"]), "error": "auxiliary file missing"})
            continue
        populations = ["units"] if row["objective"].lower() == "10x" else ["somas", "groups"]
        for population in populations:
            try:
                data = load_baseline_population(row["grouped_path"], population)
                units, session = baseline_covariation_tables(
                    data, auxiliary, row, population)
                unit_parts.append(units); session_parts.append(session)
            except Exception as error:
                failures.append({"group_id": int(row["group_id"]),
                                 "population": population,
                                 "error": f"{type(error).__name__}: {error}"})
    units = pd.concat(unit_parts, ignore_index=True) if unit_parts else pd.DataFrame()
    sessions = pd.concat(session_parts, ignore_index=True) if session_parts else pd.DataFrame()
    units.to_csv(args.output_dir / "baseline_f0_arousal_unit.csv", index=False)
    sessions.to_csv(args.output_dir / "baseline_f0_arousal_session.csv", index=False)
    pd.DataFrame(failures).to_csv(args.output_dir / "baseline_failures.csv", index=False)
    print({"sessions": len(sessions), "units": len(units), "failures": len(failures)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
