"""Four panels answering the signed-profile questions from the saved tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.figures.paths import output_root, repo_path

COHORTS = ("TH superficial", "TH deep", "DAT superficial")
COLORS = {"TH superficial": "#2b8cbe", "TH deep": "#08519c",
          "DAT superficial": "#6a51a3"}
CLASS_COLORS = {"bidirectional": "#7f7f7f", "excited_only": "#d62728",
                "suppressed_only": "#1f77b4", "silent": "#e0e0e0"}


def figure(output_dir, stem, *, path=None):
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    mouse = pd.read_csv(output_dir / f"{stem}_mouse_class_fractions.csv")
    null = pd.read_csv(output_dir / f"{stem}_sign_permutation_null.csv")
    matched = pd.read_csv(output_dir / f"{stem}_breadth_matched.csv")
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2), constrained_layout=True)
    order = [c for c in COHORTS if c in set(mouse.cohort)]
    x = np.arange(len(order))

    # A: awake class composition, one point per mouse.
    ax = axes[0]
    pre = mouse[mouse.state == "pre"]
    for offset, name in zip((-.22, 0, .22),
                            ("cross_odor_bidirectional", "excited_only",
                             "suppressed_only")):
        values = [pre.loc[pre.cohort == c, name].to_numpy() for c in order]
        ax.bar(x + offset, [np.median(v) if len(v) else np.nan for v in values],
               width=.2, color=CLASS_COLORS.get(
                   "bidirectional" if "bidir" in name else name), alpha=.75,
               label=name.replace("cross_odor_", "").replace("_", " "))
        for position, v in zip(x + offset, values):
            ax.plot(np.full(len(v), position), v, "k.", ms=4, alpha=.6)
    ax.axhline(pre.attrs.get("chance", np.nan), color="k", ls=":", lw=1)
    ax.set(xticks=x, xticklabels=[c.replace(" ", "\n") for c in order],
           ylabel="fraction of somas", title="A  awake profile, dots = mice",
           ylim=(0, 1))
    ax.legend(frameon=False, fontsize=8)

    # B: suppression-only against the within-odor sign null.
    ax = axes[1]
    pre_null = null[null.state == "pre"]
    for index, cohort in enumerate(order):
        here = pre_null[pre_null.cohort == cohort]
        observed = here.suppressed_only_observed.sum() / here.n_unit.sum()
        expected = here.suppressed_only_null_mean.sum() / here.n_unit.sum()
        ax.bar(index - .18, observed, width=.34, color=COLORS[cohort],
               label="observed" if index == 0 else None)
        ax.bar(index + .18, expected, width=.34, color="none",
               edgecolor=COLORS[cohort], hatch="///",
               label="sign-permutation null" if index == 0 else None)
    ax.set(xticks=x, xticklabels=[c.replace(" ", "\n") for c in order],
           ylabel="fraction suppression-only",
           title="B  suppression-only versus null")
    ax.legend(frameon=False, fontsize=8)

    # C: what anesthesia does to the composition.
    ax = axes[2]
    for index, cohort in enumerate(order):
        for offset, state, alpha in ((-.18, "pre", .95), (.18, "post", .45)):
            here = mouse[(mouse.cohort == cohort) & (mouse.state == state)]
            bottom = 0.
            for name in ("suppressed_only", "cross_odor_bidirectional",
                         "excited_only"):
                value = float(here[name].median()) if len(here) else 0.
                key = "bidirectional" if "bidir" in name else name
                ax.bar(index + offset, value, bottom=bottom, width=.34,
                       color=CLASS_COLORS[key], alpha=alpha,
                       edgecolor="white", lw=.5)
                bottom += value
    ax.set(xticks=x, xticklabels=[c.replace(" ", "\n") for c in order],
           ylabel="fraction of somas",
           title="C  awake (left) vs ket/xyl (right)")

    # D: the control that decides question 3.
    ax = axes[3]
    breadth = matched[matched.metric == "suppression_breadth_change"]
    for index, cohort in enumerate(order):
        here = breadth[breadth.cohort == cohort]
        if not len(here):
            continue
        ax.bar(index - .18, float(here.difference_raw.iloc[0]), width=.34,
               color=COLORS[cohort], label="raw" if index == 0 else None)
        ax.bar(index + .18, float(here.difference_matched.iloc[0]), width=.34,
               color="none", edgecolor=COLORS[cohort], hatch="///",
               label="breadth-matched" if index == 0 else None)
    ax.axhline(0, color="k", lw=.8)
    ax.set(xticks=x, xticklabels=[c.replace(" ", "\n") for c in order],
           ylabel="extra suppression lost\n(suppression-only − bidirectional)",
           title="D  the effect is starting level")
    ax.legend(frameon=False, fontsize=8)

    path = Path(path or output_dir / f"{stem}_summary.png")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=output_root() / "signed_profiles")
    parser.add_argument("--stem", default="somas_q0.05")
    args = parser.parse_args(argv)
    print("Wrote", figure(args.output_dir, args.stem))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
