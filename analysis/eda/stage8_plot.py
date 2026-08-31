"""Figures for the full-panel odor geometry."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from .stage8_rdm import ODOR_ORDER, ODOR_LABEL, FAMILY_BREAKS, session_matrix

mpl.rcParams.update({
    "font.size": 7.5, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "figure.dpi": 200, "savefig.dpi": 200, "axes.linewidth": .6,
    "xtick.major.width": .6, "ytick.major.width": .6,
})
LABELS = [ODOR_LABEL[o] for o in ODOR_ORDER]
FAMILIES = [(1, 5, "λ family"), (5, 9, "α family"), (9, 13, "ε family")]


def family_lines(ax, n):
    for b in FAMILY_BREAKS:
        ax.axhline(b-.5, color="k", lw=.7, alpha=.55)
        ax.axvline(b-.5, color="k", lw=.7, alpha=.55)


def ticks(ax, rotate=90):
    ax.set_xticks(range(len(LABELS))); ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=rotate, ha="center", fontsize=5.6)
    ax.set_yticklabels(LABELS, fontsize=5.6)


def plot_confusion(data, keys, path, state="pre"):
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 9.2), constrained_layout=True)
    for index, (ax, (cohort, population, title)) in enumerate(
            zip(axes.ravel(), keys)):
        key = f"{cohort}|{population}|{state}|confusion"
        if key not in data:
            ax.axis("off"); continue
        m = data[key]
        im = ax.imshow(m, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
        family_lines(ax, len(LABELS)); ticks(ax)
        # Only the bottom row carries x labels, so upper x labels cannot
        # collide with the titles beneath them.
        if index < 3:
            ax.set_xticklabels([])
        ax.set_title(f"{title}\n{np.nanmean(np.diag(m))*100:.0f}% correct "
                     f"(chance {100/len(LABELS):.0f}%)", pad=6)
        if index == 2:
            fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.55,
                         label="fraction classified as")
    for ax in axes[-1]: ax.set_xlabel("classified as")
    for ax in axes[:, 0]: ax.set_ylabel("true odor")
    fig.suptitle("Leave-one-trial-out nearest-centroid confusion, awake, all 16 odors\n"
                 "odors grouped so each mixture sits beside its two components",
                 fontsize=10)
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def plot_rdm_contrast(data, keys, path, state="pre"):
    fig, axes = plt.subplots(2, len(keys), figsize=(3.6*len(keys), 7.4),
                             constrained_layout=True)
    for column, (cohort, population, title) in enumerate(keys):
        simple = data.get(f"{cohort}|{population}|{state}|correlation")
        strict = data.get(f"{cohort}|{population}|{state}|crossnobis")
        ax = axes[0, column]
        im = ax.imshow(simple, cmap="viridis", vmin=0, vmax=2, interpolation="nearest")
        family_lines(ax, len(LABELS)); ticks(ax); ax.set_title(f"{title}\ncorrelation distance")
        if column == len(keys)-1:
            fig.colorbar(im, ax=axes[0].tolist(), shrink=.7, label="1 − r")
        ax = axes[1, column]
        finite = strict[np.isfinite(strict)]
        top = float(np.nanpercentile(finite, 95)) if finite.size else 1.
        im = ax.imshow(strict, cmap="viridis", vmin=0, vmax=max(top, 1e-6),
                       interpolation="nearest")
        family_lines(ax, len(LABELS)); ticks(ax); ax.set_title("crossnobis (crossvalidated)")
        if column == len(keys)-1:
            fig.colorbar(im, ax=axes[1].tolist(), shrink=.7, label="distance")
    fig.suptitle("Same data, two views: the simple correlation RDM is biased but legible;\n"
                 "the crossvalidated RDM is unbiased but noisy at these trial counts",
                 fontsize=10)
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def plot_trial_matrix(cache, path, title, state="pre"):
    got = session_matrix(cache, state)
    if got is None: return False
    x, odor, _ = got
    keep = np.isin(odor, ODOR_ORDER)
    x, odor = x[keep], odor[keep]
    rank = {o: i for i, o in enumerate(ODOR_ORDER)}
    order = np.argsort([rank[int(o)] for o in odor], kind="stable")
    x, odor = x[order], odor[order]
    good = np.isfinite(x).all(axis=0)
    c = np.corrcoef(x[:, good])
    fig, ax = plt.subplots(figsize=(7.4, 6.4), constrained_layout=True)
    im = ax.imshow(c, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    edges, pos = [], []
    for o in ODOR_ORDER:
        n = int(np.sum(odor == o))
        if n: edges.append(n)
    bounds = np.cumsum(edges)
    for b in bounds[:-1]:
        ax.axhline(b-.5, color="k", lw=.35, alpha=.5)
        ax.axvline(b-.5, color="k", lw=.35, alpha=.5)
    centres = np.concatenate([[0], bounds[:-1]]) + np.array(edges)/2
    present = [ODOR_LABEL[o] for o in ODOR_ORDER if np.sum(odor == o)]
    ax.set_xticks(centres); ax.set_yticks(centres)
    ax.set_xticklabels(present, rotation=90, fontsize=5.6)
    ax.set_yticklabels(present, fontsize=5.6)
    fig.colorbar(im, ax=ax, shrink=.75, label="trial-to-trial pattern correlation")
    ax.set_title(f"{title}\nevery trial against every trial, sorted by odor", fontsize=9)
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    out = args.output_dir / "figures"; out.mkdir(parents=True, exist_ok=True)
    data = dict(np.load(args.output_dir / "stage8_rdm.npz", allow_pickle=True))

    confusion_keys = [
        ("Thy1 10x", "units", "Thy1 10× (glomerular)"),
        ("DAT 10x", "units", "DAT 10×"),
        ("TH 10x", "units", "TH 10×"),
        ("DAT 20x superficial", "somas", "DAT 20× superficial somas"),
        ("TH 20x superficial", "somas", "TH 20× superficial somas"),
        ("TH 20x deep", "somas", "TH 20× deep somas"),
    ]
    plot_confusion(data, confusion_keys, out / "panel_confusion_awake.png")
    plot_confusion(data, confusion_keys, out / "panel_confusion_anesth.png",
                   state="post")
    plot_rdm_contrast(data, [
        ("Thy1 10x", "units", "Thy1 10×"),
        ("DAT 10x", "units", "DAT 10×"),
        ("TH 20x deep", "somas", "TH 20× deep somas"),
    ], out / "panel_rdm_simple_vs_crossnobis.png")
    plot_trial_matrix(args.output_dir / "trials/group213_trials.npz",
                      out / "panel_trial_correlation_thy1.png",
                      "Thy1 10×, awake (group 213)")
    print("wrote:", *(p.name for p in sorted(out.glob("*.png"))), sep="\n  ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
