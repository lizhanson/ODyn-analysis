"""Full-panel odor geometry: rigorous RDMs beside simpler, cleaner views.

Three views of the same 16-odor panel, deliberately at different points on the
rigour/legibility trade-off:

  crossnobis RDM   crossvalidated, unbiased, zero means no reliable separation.
                   Noisy on this many trials, and hard to read at a glance.
  correlation RDM  1 - correlation between odor mean patterns.  Not
                   crossvalidated and positively biased, so it can never say a
                   difference is real -- but it is far cleaner and is the right
                   tool for spotting structure worth testing.
  confusion        leave-one-trial-out nearest centroid.  Intuitive units
                   (what gets mistaken for what) and shows the structure the
                   RDMs imply.

Odors are ordered so each mixture sits beside the two components it is made
from, which is what makes mixture-versus-component structure visible.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .common import manifest_rows
from .geometry import crossnobis_pair

# blank | lambda family | alpha family | epsilon family | remaining singles
ODOR_ORDER = [0, 3, 12, 39, 40, 4, 10, 17, 18, 22, 30, 31, 32, 1, 2, 21]
ODOR_LABEL = {
    0: "min oil", 1: "eugenol", 2: "me salicyl", 3: "acetophen", 4: "1-butanol",
    10: "α-pinene", 12: "limonene", 17: "α (4+10)", 18: "α' (4+10)",
    21: "et butyrate", 22: "cyclopent", 30: "2Me-2-pent",
    31: "ε (22+30)", 32: "ε' (22+30)", 39: "λ (3+12)", 40: "λ' (3+12)",
}
FAMILY_BREAKS = [1, 5, 9, 13]      # where to draw separators
WINDOW = (0.0, 4.0)
MIN_TRIALS = 3


def session_matrix(cache, state_name, window=WINDOW):
    """trial x unit responses plus odor labels for one state."""
    data = np.load(cache, allow_pickle=True)
    levels = [str(x) for x in data["state_levels"]]
    if state_name not in levels:
        return None
    rows = data["state"] == levels.index(state_name)
    if rows.sum() < 10:
        return None
    population = "units" if any(k.startswith("units/") for k in data.files) else "somas"
    binned = data[f"{population}/binned"][:, rows, :]
    centres = data[f"{population}/bin_centre_s"]
    mask = (centres >= window[0]) & (centres < window[1])
    x = np.transpose(np.nanmean(binned[:, :, mask], axis=2), (1, 0))
    return x, data["odor_id"][rows], population


def crossnobis_rdm(x, odor, order):
    n = len(order)
    matrix = np.full((n, n), np.nan)
    np.fill_diagonal(matrix, 0.)
    for i in range(n):
        for j in range(i+1, n):
            mask = np.isin(odor, (order[i], order[j]))
            if (np.sum(odor == order[i]) < MIN_TRIALS
                    or np.sum(odor == order[j]) < MIN_TRIALS):
                continue
            value = crossnobis_pair(x[mask], odor[mask], repeats=60,
                                    seed=i*n+j)
            matrix[i, j] = matrix[j, i] = value
    return matrix


def correlation_rdm(x, odor, order):
    """1 - correlation between odor mean patterns. Simple, biased, legible."""
    centroids = []
    for level in order:
        rows = odor == level
        centroids.append(np.nanmean(x[rows], axis=0) if rows.sum() >= 1
                         else np.full(x.shape[1], np.nan))
    c = np.vstack(centroids)
    good = np.isfinite(c).all(axis=0)
    c = c[:, good]
    if c.shape[1] < 3:
        return np.full((len(order), len(order)), np.nan)
    return 1 - np.corrcoef(c)


def confusion(x, odor, order):
    """Leave-one-trial-out nearest centroid, correlation distance."""
    n = len(order)
    matrix = np.zeros((n, n))
    counts = np.zeros(n)
    index = {level: i for i, level in enumerate(order)}
    good = np.isfinite(x).all(axis=0)
    x = x[:, good]
    if x.shape[1] < 3:
        return np.full((n, n), np.nan)
    for trial in range(x.shape[0]):
        true = index.get(int(odor[trial]))
        if true is None:
            continue
        best, best_r = None, -np.inf
        for level in order:
            rows = (odor == level).copy()
            rows[trial] = False
            if rows.sum() < 1:
                continue
            centroid = np.nanmean(x[rows], axis=0)
            if np.std(centroid) == 0 or np.std(x[trial]) == 0:
                continue
            r = np.corrcoef(centroid, x[trial])[0, 1]
            if np.isfinite(r) and r > best_r:
                best_r, best = r, index[level]
        if best is not None:
            matrix[true, best] += 1
            counts[true] += 1
    with np.errstate(invalid="ignore"):
        return matrix/counts[:, None]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    trials = args.output_dir / "trials"
    records = {}
    for row in manifest_rows():
        cache = trials / f"group{int(row['group_id'])}_trials.npz"
        if not cache.exists():
            continue
        for state in ("pre", "post"):
            got = session_matrix(cache, state)
            if got is None:
                continue
            x, odor, population = got
            key = (row["cohort"], population, state)
            records.setdefault(key, []).append({
                "group_id": int(row["group_id"]),
                "crossnobis": crossnobis_rdm(x, odor, ODOR_ORDER),
                "correlation": correlation_rdm(x, odor, ODOR_ORDER),
                "confusion": confusion(x, odor, ODOR_ORDER),
            })
        print(f"  {row['group_id']:>4} done", flush=True)

    payload = {}
    rows = []
    for (cohort, population, state), items in records.items():
        for name in ("crossnobis", "correlation", "confusion"):
            stack = np.stack([item[name] for item in items])
            median = np.nanmedian(stack, axis=0)
            payload[f"{cohort}|{population}|{state}|{name}"] = median
        rows.append({"cohort": cohort, "population": population, "state": state,
                     "n_session": len(items),
                     "mean_accuracy": float(np.nanmean(np.diag(
                         np.nanmedian(np.stack([i["confusion"] for i in items]),
                                      axis=0))))})
    np.savez_compressed(args.output_dir / "stage8_rdm.npz",
                        odor_order=np.asarray(ODOR_ORDER), **payload)
    table = pd.DataFrame(rows)
    table.to_csv(args.output_dir / "stage8_rdm_summary.csv", index=False)
    print()
    print(table.sort_values("mean_accuracy", ascending=False).round(3).to_string(index=False))
    print(f"\nchance accuracy = {1/len(ODOR_ORDER):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
