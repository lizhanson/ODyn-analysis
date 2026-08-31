"""Does subtracting the block-mean mineral-oil response help or hurt?

Subtracting a per-ROI, per-state blank waveform removes the delivery-locked
component shared by every odor.  It keeps every ROI and treats every odor
identically, so unlike ROI exclusion it cannot bias which cells survive.  The
cost is noise: the blank mean comes from 4-8 trials, and that noise lands on
every odor response.

The test is a fair split-half.  Odor tuning is estimated from each half
independently, and the blank subtracted from a half is estimated from that same
half, so the subtraction never sees the data it is scored against.  If
subtraction helps, tuning reliability rises; if the added noise dominates, it
falls.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .common import BLANK_ODOR, manifest_rows

WINDOW = (1.0, 4.0)
MIN_TRIALS = 4
REPEATS = 40


def _tuning(response, odor, levels):
    """unit x odor mean response over the supplied trials."""
    return np.stack([np.nanmean(response[:, odor == level], axis=1)
                     for level in levels], axis=1)


def _row_correlation(a, b):
    ac = a - np.nanmean(a, axis=1, keepdims=True)
    bc = b - np.nanmean(b, axis=1, keepdims=True)
    denominator = np.sqrt(np.nansum(ac*ac, axis=1)*np.nansum(bc*bc, axis=1))
    return np.divide(np.nansum(ac*bc, axis=1), denominator,
                     out=np.full(a.shape[0], np.nan), where=denominator > 0)


def session_rows(row, cache, *, repeats=REPEATS, seed=0):
    data = np.load(cache, allow_pickle=True)
    levels_state = [str(x) for x in data["state_levels"]]
    population = "units" if any(k.startswith("units/") for k in data.files) else "somas"
    populations = sorted({k.split("/")[0] for k in data.files if "/" in k})
    centres = data[f"{populations[0]}/bin_centre_s"]
    mask = (centres >= WINDOW[0]) & (centres < WINDOW[1])
    output = []
    for population in populations:
        binned = data[f"{population}/binned"]
        response_all = np.nanmean(binned[:, :, mask], axis=2)     # unit x trial
        for code, state in enumerate(levels_state):
            in_state = data["state"] == code
            if not in_state.any():
                continue
            odor = data["odor_id"][in_state]
            response = response_all[:, in_state]
            blank_index = np.flatnonzero(odor == BLANK_ODOR)
            real = np.array([int(o) for o in np.unique(odor) if o != BLANK_ODOR])
            usable = [o for o in real if np.sum(odor == o) >= MIN_TRIALS]
            if len(usable) < 3 or len(blank_index) < MIN_TRIALS:
                continue
            rng = np.random.default_rng(seed + code)
            raw, subtracted = [], []
            for _ in range(int(repeats)):
                halves = {}
                for level in usable:
                    index = rng.permutation(np.flatnonzero(odor == level))
                    cut = len(index)//2
                    halves[level] = (index[:cut], index[cut:])
                blank = rng.permutation(blank_index)
                cut = len(blank)//2
                blank_halves = (blank[:cut], blank[cut:])
                tuning = []
                tuning_sub = []
                for side in (0, 1):
                    columns = np.stack([
                        np.nanmean(response[:, halves[level][side]], axis=1)
                        for level in usable], axis=1)
                    tuning.append(columns)
                    # the blank subtracted from a half comes from that same
                    # half, so the correction never sees the other side
                    offset = np.nanmean(response[:, blank_halves[side]], axis=1)
                    tuning_sub.append(columns - offset[:, None])
                raw.append(_row_correlation(tuning[0], tuning[1]))
                subtracted.append(_row_correlation(tuning_sub[0], tuning_sub[1]))
            common = {
                "group_id": int(row["group_id"]), "mouse": row["mouse"],
                "cohort": row["cohort"], "population": population,
                "state": state, "n_odor": len(usable),
                "n_blank_trial": len(blank_index),
            }
            with np.errstate(invalid="ignore"):
                output.append(common | {
                    "reliability_raw": float(np.nanmedian(np.nanmedian(
                        np.stack(raw), axis=0))),
                    "reliability_subtracted": float(np.nanmedian(np.nanmedian(
                        np.stack(subtracted), axis=0))),
                })
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = []
    for row in manifest_rows():
        cache = args.output_dir/"trials"/f"group{int(row['group_id'])}_trials.npz"
        if not cache.exists():
            continue
        try:
            rows.extend(session_rows(row, cache, seed=int(row["group_id"])))
        except Exception as error:
            print(f"  {row['group_id']} FAILED {type(error).__name__}: {error}")
    table = pd.DataFrame(rows)
    table["delta"] = table.reliability_subtracted - table.reliability_raw
    table.to_csv(args.output_dir/"stage9_blank_subtraction.csv", index=False)
    print(f"{len(table)} session-population-state rows\n")
    print("=== odor tuning reliability, split-half (higher is better) ===")
    print(table.groupby(["population", "state"])[
        ["reliability_raw", "reliability_subtracted", "delta"]].median()
        .round(3).to_string())
    print("\n=== how often does subtraction help? ===")
    print(table.groupby(["population", "state"]).delta.apply(
        lambda s: f"{(s > 0).mean()*100:.0f}% of sessions").to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
