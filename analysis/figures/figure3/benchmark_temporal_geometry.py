"""Benchmark integrated and temporal mixture geometry in one 20x session."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def diagonal_crossnobis(trials, labels, *, repeats=200, seed=0):
    """Local two-condition-safe copy so this benchmark remains standalone."""
    x, labels = np.asarray(trials, float), np.asarray(labels)
    levels = np.unique(labels)
    indices = {level: np.flatnonzero(labels == level) for level in levels}
    residual = np.concatenate([
        x[index] - np.nanmean(x[index], axis=0, keepdims=True)
        for index in indices.values()
    ])
    variance = np.nanvar(residual, axis=0, ddof=1)
    positive = variance[np.isfinite(variance) & (variance > 0)]
    floor = np.nanmedian(positive) * 1e-3 if positive.size else 1e-12
    variance = np.where(np.isfinite(variance) & (variance > floor), variance, floor)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        halves = {}
        for level, index in indices.items():
            shuffled = rng.permutation(index); cut = len(shuffled) // 2
            halves[level] = shuffled[:cut], shuffled[cut:]
        ia, ib = halves[levels[0]]; ja, jb = halves[levels[1]]
        da = np.nanmean(x[ia], axis=0) - np.nanmean(x[ja], axis=0)
        db = np.nanmean(x[ib], axis=0) - np.nanmean(x[jb], axis=0)
        values.append(np.nanmean(da * db / variance))
    matrix = np.asarray([[0., np.nanmean(values)], [np.nanmean(values), 0.]])
    return levels, matrix


def _distance(x, labels, *, seed=0, repeats=100):
    _, matrix = diagonal_crossnobis(x, labels, seed=seed, repeats=repeats)
    return float(matrix[0, 1])


def _permutation_summary(x, labels, observed, *, permutations=100, seed=0):
    rng = np.random.default_rng(seed)
    null = np.asarray([
        _distance(x, rng.permutation(labels), seed=seed + index + 1, repeats=40)
        for index in range(permutations)
    ])
    sd = np.nanstd(null, ddof=1)
    return float(np.nanpercentile(null, 95)), float(
        (observed - np.nanmean(null)) / sd if sd > 0 else np.nan)


def benchmark(grouped_path, source_path, output_dir, *, odors=(17, 18),
              bin_seconds=.5, permutations=100):
    import h5py
    import pandas as pd
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(source_path) as handle:
        time_s = handle["traces/time_s"][:]
    with h5py.File(grouped_path) as handle:
        odor_id = handle["odor_id"][:]
        state = handle["state"][:]
        levels = [x.decode() if isinstance(x, bytes) else str(x)
                  for x in handle["state_levels"][:]]
        traces = {key: handle[f"{key}/z"][:] for key in ("somas", "processes")}

    width = max(1, int(round(bin_seconds / np.nanmedian(np.diff(time_s)))))
    starts = np.arange(np.searchsorted(time_s, -1.), np.searchsorted(time_s, 8.), width)
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, constrained_layout=True)
    for row_index, (compartment, values) in enumerate(traces.items()):
        for column, state_name in enumerate(("pre", "post")):
            state_code = levels.index(state_name)
            selected = (state == state_code) & np.isin(odor_id, odors)
            labels = odor_id[selected]
            trial_trace = np.transpose(values[:, selected, :], (1, 0, 2))
            odor_window = (time_s >= 0) & (time_s < 4)
            integrated_x = np.nanmean(trial_trace[:, :, odor_window], axis=2)
            integrated = _distance(integrated_x, labels, seed=11)
            integrated_null95, integrated_z = _permutation_summary(
                integrated_x, labels, integrated, permutations=permutations, seed=21)

            binned = np.stack([
                np.nanmean(trial_trace[:, :, start:start + width], axis=2)
                for start in starts
            ], axis=2)
            distances, null95 = [], []
            for index in range(binned.shape[2]):
                value = _distance(binned[:, :, index], labels, seed=100 + index)
                cutoff, _ = _permutation_summary(
                    binned[:, :, index], labels, value,
                    permutations=permutations, seed=200 + index)
                distances.append(value); null95.append(cutoff)
            # Temporal pattern at a modest resolution, not every imaging frame.
            spatiotemporal_x = binned[:, :, (time_s[starts] >= 0) & (time_s[starts] < 4)]
            spatiotemporal_x = spatiotemporal_x.reshape(len(labels), -1)
            spatiotemporal = _distance(spatiotemporal_x, labels, seed=31)
            temporal_null95, temporal_z = _permutation_summary(
                spatiotemporal_x, labels, spatiotemporal,
                permutations=permutations, seed=41)
            rows.append({
                "compartment": compartment, "state": state_name,
                "n_odor17": int(np.sum(labels == odors[0])),
                "n_odor18": int(np.sum(labels == odors[1])),
                "integrated_crossnobis": integrated,
                "integrated_null95": integrated_null95,
                "integrated_null_z": integrated_z,
                "spatiotemporal_crossnobis": spatiotemporal,
                "spatiotemporal_null95": temporal_null95,
                "spatiotemporal_null_z": temporal_z,
                "peak_time_resolved_crossnobis": float(np.nanmax(distances)),
                "peak_time_s": float(time_s[starts][int(np.nanargmax(distances))]),
            })
            ax = axes[row_index, column]
            ax.plot(time_s[starts], distances, color="#2b8cbe", label="crossnobis")
            ax.plot(time_s[starts], null95, color=".45", ls="--",
                    label="pointwise shuffled 95%")
            ax.axvspan(0, 4, color="#fee391", alpha=.25)
            ax.axhline(integrated, color="#d7301f", lw=1.2,
                       label="0–4 s integrated")
            ax.set(title=f"{compartment}, {state_name}", ylabel="distance")
    axes[-1, 0].set_xlabel("time from odor onset (s)")
    axes[-1, 1].set_xlabel("time from odor onset (s)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure = output_dir / "group202_odor17_18_temporal_geometry.png"
    fig.savefig(figure, dpi=180); plt.close(fig)
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "group202_odor17_18_temporal_geometry.csv", index=False)
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grouped_path", type=Path)
    parser.add_argument("source_path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=100)
    args = parser.parse_args(argv)
    print(benchmark(args.grouped_path, args.source_path, args.output_dir,
                    permutations=args.permutations).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
