"""Stage 3: population geometry, separated into four distinct questions.

The reciprocal mixture pairs differ only in the ratio of the same two
components, so distances between them measure fine ratio discrimination, not
odor identity discrimination.  Those are reported separately, alongside the
question that matters most once mineral oil turns out to drive the population:
is an odor's response distinguishable from the delivery response at all?
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .common import (BLANK_ODOR, MIXTURE_PAIRS, manifest_rows,
                    mixture_components, odor_groups, panel_odors)
from .geometry import cosine_pair, crossnobis_pair, permutation_test

FEATURE_SETS = {
    "onset": (0.0, 1.0),
    "early": (0.0, 2.0),        # first half of the odor
    "late": (2.0, 4.0),         # second half of the odor
    "sustained": (1.0, 4.0),
    "odor": (0.0, 4.0),
    "offset": (4.0, 8.0),       # after the odor is off
}
# Permutation nulls are expensive, so they run on the windows the first-half
# versus second-half question actually turns on.
PERMUTED_SETS = ("early", "late", "odor")
# Successive one-second windows: when, if ever, does separation appear?
SWEEP_EDGES = tuple(float(edge) for edge in range(0, 8))
MIN_TRIALS = 2


def load_cache(path):
    data = np.load(path, allow_pickle=True)
    populations = sorted({key.split("/")[0] for key in data.files if "/" in key})
    return data, populations


def feature_matrix(binned, centres, window, *, spatiotemporal=False):
    """trial x feature.  Features are units, or unit-time bins."""
    mask = (centres >= window[0]) & (centres < window[1])
    block = binned[:, :, mask]
    if spatiotemporal:
        return np.transpose(block, (1, 0, 2)).reshape(block.shape[1], -1)
    return np.transpose(np.nanmean(block, axis=2), (1, 0))


def _pair(x, odor, a, b, *, seed, permutations=0):
    mask = np.isin(odor, (a, b))
    labels = odor[mask]
    counts = [int(np.sum(labels == a)), int(np.sum(labels == b))]
    if min(counts) < MIN_TRIALS:
        return None
    record = {"odor_a": int(a), "odor_b": int(b),
              "n_a": counts[0], "n_b": counts[1],
              "crossnobis": crossnobis_pair(x[mask], labels, seed=seed),
              "cosine": cosine_pair(x[mask], labels)}
    if permutations:
        result = permutation_test(x[mask], labels, permutations=permutations,
                                  seed=seed)
        record |= {"p_value": result.p_value, "p_resolution": result.resolution,
                   "null_median": result.null_median,
                   "underpowered": result.underpowered}
    return record


def session_geometry(row, cache_path, *, permutations=200):
    data, populations = load_cache(cache_path)
    odor_all = data["odor_id"]
    state_all = data["state"]
    levels = [str(x) for x in data["state_levels"]]
    groups = odor_groups()
    components = mixture_components()
    panel = [o for o in panel_odors() if o != BLANK_ODOR]
    singles = [o for o in panel if groups[o] != "mixture"]
    mixtures = [o for o in panel if groups[o] == "mixture"]

    output = []
    for population in populations:
        binned = data[f"{population}/binned"]
        centres = data[f"{population}/bin_centre_s"]
        for code, state in enumerate(levels):
            rows = state_all == code
            if not rows.any():
                continue
            odor = odor_all[rows]
            for feature_name, window in FEATURE_SETS.items():
                x = feature_matrix(binned[:, rows, :], centres, window)
                common = {
                    "group_id": int(row["group_id"]), "mouse": row["mouse"],
                    "line": row["line"], "cohort": row["cohort"],
                    "population": population, "state": state,
                    "feature_set": feature_name,
                }
                # 1. Is the odor distinguishable from the delivery response?
                for odor_id in panel:
                    item = _pair(x, odor, odor_id, BLANK_ODOR, seed=11)
                    if item:
                        output.append(common | item | {
                            "comparison": "odor_vs_blank",
                            "odor_group": groups[odor_id]})
                # 2. Identity discrimination between distinct singles.
                for i, a in enumerate(singles):
                    for b in singles[i+1:]:
                        item = _pair(x, odor, a, b, seed=21)
                        if item:
                            output.append(common | item | {
                                "comparison": "identity", "odor_group": "single"})
                # 3. Ratio discrimination within reciprocal mixture pairs.
                for index, (a, b) in enumerate(MIXTURE_PAIRS):
                    item = _pair(x, odor, a, b, seed=31+index,
                                 permutations=permutations
                                 if feature_name in PERMUTED_SETS else 0)
                    if item:
                        output.append(common | item | {
                            "comparison": "ratio", "odor_group": "mixture"})
                # 4. Mixture against each of its own components.
                for mixture in mixtures:
                    for component in components.get(mixture, ()):
                        item = _pair(x, odor, mixture, component, seed=41)
                        if item:
                            output.append(common | item | {
                                "comparison": "mixture_vs_component",
                                "odor_group": "mixture"})
            # Time-resolved sweep for the ratio pairs: a separation that
            # emerges only late is invisible in any single wide window.
            for start in SWEEP_EDGES:
                x = feature_matrix(binned[:, rows, :], centres,
                                   (start, start+1.0))
                for index, (a, b) in enumerate(MIXTURE_PAIRS):
                    item = _pair(x, odor, a, b, seed=61+index)
                    if item:
                        output.append({
                            "group_id": int(row["group_id"]),
                            "mouse": row["mouse"], "line": row["line"],
                            "cohort": row["cohort"], "population": population,
                            "state": state, "feature_set": "sweep_1s",
                            "window_start_s": start,
                            "comparison": "ratio",
                            "odor_group": "mixture"} | item)
            # Spatiotemporal geometry for the ratio pairs only: the temporal
            # pattern matters most where the spatial pattern nearly matches.
            x = feature_matrix(binned[:, rows, :], centres, (0., 4.),
                               spatiotemporal=True)
            for index, (a, b) in enumerate(MIXTURE_PAIRS):
                item = _pair(x, odor, a, b, seed=51+index)
                if item:
                    output.append({
                        "group_id": int(row["group_id"]), "mouse": row["mouse"],
                        "line": row["line"], "cohort": row["cohort"],
                        "population": population, "state": state,
                        "feature_set": "spatiotemporal",
                        "comparison": "ratio", "odor_group": "mixture"} | item)
    return output


def main(argv=None):
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=200)
    args = parser.parse_args(argv)
    trials = args.output_dir / "trials"
    rows, failures = [], []
    for row in manifest_rows():
        cache = trials / f"group{int(row['group_id'])}_trials.npz"
        if not cache.exists():
            continue
        try:
            rows.extend(session_geometry(row, cache,
                                         permutations=args.permutations))
            print(f"  {row['group_id']:>4} ok ({len(rows):,} rows)", flush=True)
        except Exception as error:
            failures.append({"group_id": int(row["group_id"]),
                             "error": f"{type(error).__name__}: {error}"})
            print(f"  {row['group_id']:>4} FAILED {failures[-1]['error']}",
                  flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(args.output_dir / "stage3_geometry.csv.gz", index=False,
                 compression="gzip")
    print(f"\nwrote {len(table):,} rows, {len(failures)} failures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
