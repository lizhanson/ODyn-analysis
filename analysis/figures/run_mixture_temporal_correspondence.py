"""Batch 10x temporal mixture/DA correspondence exploration."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from .mixture_temporal_correspondence import (family_time_correspondence,
                                              session_temporal_mixture_table)
from .paths import imaging_root, repo_path
from .population_metrics import load_population
from .session_data import available_sessions


COLORS = {"TH": "#2b8cbe", "DAT": "#6a51a3", "Thy1": "#e34a33"}


def plot_timecourses(path, population):
    import matplotlib.pyplot as plt

    pairs = ("17-18", "31-32", "39-40")
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 10), sharex=True,
                             constrained_layout=True)
    for row, pair in enumerate(pairs):
        selected = population[population.pair == pair]
        for line in ("Thy1", "TH", "DAT"):
            values = selected[selected.line == line].sort_values("bin_start_s")
            axes[row, 0].plot(values.bin_start_s+.5,
                              values.signed_cv_pattern_rms_z,
                              marker="o", color=COLORS[line], label=line)
        for line in ("TH", "DAT"):
            values = selected[selected.line == line].sort_values("bin_start_s")
            axes[row, 1].plot(values.bin_start_s+.5, values.q90_z,
                              color=COLORS[line], marker="o", label=f"{line} q90")
            axes[row, 1].plot(values.bin_start_s+.5, values.q10_z,
                              color=COLORS[line], marker="o", linestyle="--",
                              label=f"{line} q10")
        axes[row, 0].axhline(0, color=".7", lw=.8)
        axes[row, 0].set_ylabel(f"{pair}\nsigned CV pattern RMS (z)")
        axes[row, 1].set_ylabel("DA population tail (z)")
    axes[-1, 0].set_xlabel("time from odor onset (s)")
    axes[-1, 1].set_xlabel("time from odor onset (s)")
    axes[0, 0].set_title("reciprocal-mixture pattern separation")
    axes[0, 1].set_title("signed DA recruitment")
    axes[0, 0].legend(frameon=False, ncol=3, fontsize=8)
    axes[0, 1].legend(frameon=False, ncol=2, fontsize=8)
    fig.suptitle("Awake mixture separation and signed DA response structure\n"
                 "session medians within mouse; mouse medians shown")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=repo_path("analysis", "stage0", "ketxyl_16odor_session_manifest.csv"))
    parser.add_argument("--imaging-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path,
                        default=repo_path("analysis", "figures", "mixture_temporal_outputs"))
    parser.add_argument("--minimum-trials", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args(argv)
    root = imaging_root(args.imaging_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inventory = pd.DataFrame(available_sessions(args.manifest, root, objective="10x"))
    inventory = inventory[inventory.available]
    parts, failures = [], []
    for row in tqdm(inventory.to_dict("records"), desc="10x temporal mixtures",
                    unit="session"):
        try:
            data = load_population(row["grouped_path"], "units")
            parts.append(session_temporal_mixture_table(
                data, row, minimum_trials=args.minimum_trials,
                repeats=args.repeats))
        except Exception as error:
            failures.append({"group_id": int(row["group_id"]),
                             "error": f"{type(error).__name__}: {error}"})
    table = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    table.to_csv(args.output_dir / "temporal_mixture_session.csv", index=False)
    pd.DataFrame(failures).to_csv(args.output_dir / "failures.csv", index=False)
    if len(table):
        correlation, population = family_time_correspondence(table, state="pre")
        correlation.to_csv(args.output_dir / "thy1_da_family_time_correspondence.csv",
                           index=False)
        population.to_csv(args.output_dir / "mouse_aggregated_timecourses.csv", index=False)
        plot_timecourses(args.output_dir / "temporal_mixture_correspondence.png",
                         population)
    print({"rows": len(table), "failures": len(failures)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
