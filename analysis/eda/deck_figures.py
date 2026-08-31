"""Slide-ready figures: opaque background, larger type, 16:9 friendly aspects."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from .common import manifest_rows
from .stage8_rdm import FAMILY_BREAKS, ODOR_LABEL, ODOR_ORDER, session_matrix

INK = "#14202A"; MUTED = "#4C5A63"; RULE = "#C6CFD1"; PAPER = "#FFFFFF"
EXCITE = "#A8632A"; SUPPRESS = "#215761"
SERIES = ["#8A4033", "#A8632A", "#215761", "#2F5578", "#5C4570", "#39603B"]
COHORT_COLOR = {
    "TH 20x deep": SERIES[0], "TH 20x superficial": SERIES[1],
    "DAT 20x superficial": SERIES[2], "DAT 10x": SERIES[3],
    "TH 10x": SERIES[4], "Thy1 10x": SERIES[5],
}


def style():
    mpl.rcParams.update({
        "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11.5,
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": RULE, "axes.linewidth": 1.0,
        "figure.facecolor": PAPER, "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER, "legend.frameon": False,
        "xtick.major.width": 1.0, "ytick.major.width": 1.0,
    })


def save(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    print("  ", path.name, flush=True)


def slope(path, rows, *, title, ylabel, ylim, zero=False):
    """Series are named in a legend; labels pinned to the line ends collide
    wherever values converge, which for these metrics is most of them."""
    fig, ax = plt.subplots(figsize=(8.6, 4.6), constrained_layout=True)
    for name, colour, a, b in rows:
        ax.plot([0, 1], [a, b], color=colour, lw=2.6, marker="o", ms=8,
                alpha=.92, label=f"{name}   {a:+.2f} to {b:+.2f}")
    if zero:
        ax.axhline(0, color=MUTED, lw=1.2, ls="--")
    ax.set(xlim=(-.08, 1.08), ylim=ylim, xticks=[0, 1],
           xticklabels=["awake", "ket/xyl"], ylabel=ylabel, title=title)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9.5, loc="center left", bbox_to_anchor=(1.02, .5))
    save(fig, path)


def bars(path, rows, *, title, xlabel, chance=None):
    fig, ax = plt.subplots(figsize=(7.6, 4.6), constrained_layout=True)
    y = np.arange(len(rows))[::-1]
    for index, (name, colour, a, b) in enumerate(rows):
        ax.barh(y[index]+.19, a, height=.36, color=colour, alpha=.95)
        ax.barh(y[index]-.19, b, height=.36, color=colour, alpha=.42)
        ax.text(a+1.5, y[index]+.19, f"{a:.0f}", va="center", fontsize=10, color=MUTED)
        ax.text(b+1.5, y[index]-.19, f"{b:.0f}", va="center", fontsize=10, color=MUTED)
    if chance is not None:
        ax.axvline(chance, color=MUTED, ls="--", lw=1.4)
        ax.text(chance+1, len(rows)-.35, "chance", fontsize=9.5, color=MUTED)
    ax.set(yticks=y, yticklabels=[r[0] for r in rows], xlim=(0, 108),
           xlabel=xlabel, title=title)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, path)


def lines(path, series, x, *, title, ylabel, xlabel, shade=None, ylim=None):
    fig, ax = plt.subplots(figsize=(7.6, 4.6), constrained_layout=True)
    if shade:
        ax.axvspan(*shade, color=RULE, alpha=.45, lw=0)
    for name, colour, values in series:
        ax.plot(x, values, color=colour, lw=2.4, label=name)
    ax.axhline(0, color=MUTED, lw=1.1)
    ax.set(title=title, ylabel=ylabel, xlabel=xlabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9.5, loc="upper right", ncol=1)
    save(fig, path)


def blank_trace(out, path):
    cohorts = {}
    for row in manifest_rows():
        cache = out/"trials"/f"group{int(row['group_id'])}_trials.npz"
        if not cache.exists():
            continue
        data = np.load(cache, allow_pickle=True)
        population = "units" if any(k.startswith("units/") for k in data.files) else "somas"
        binned = data[f"{population}/binned"]; centres = data[f"{population}/bin_centre_s"]
        levels = [str(x) for x in data["state_levels"]]
        for state in ("pre", "post"):
            if state not in levels:
                continue
            rows_in = data["state"] == levels.index(state)
            odor = data["odor_id"][rows_in]; block = binned[:, rows_in, :]
            for label, mask in (("blank", odor == 0), ("odor", odor != 0)):
                if mask.any():
                    cohorts.setdefault((row["cohort"], state, label), []).append(
                        np.nanmedian(np.nanmedian(block[:, mask, :], axis=1), axis=0))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), sharey=True,
                             constrained_layout=True)
    for ax, state in zip(axes, ("pre", "post")):
        for cohort, title, lw in (("TH 10x", "TH 10x glomeruli", 2.8),
                                  ("DAT 20x superficial", "DAT 20x somas", 1.9)):
            for label, colour, dash in (("odor", EXCITE, "-"), ("blank", SUPPRESS, "--")):
                stack = cohorts.get((cohort, state, label))
                if stack:
                    ax.plot(centres, np.nanmedian(np.stack(stack), axis=0), color=colour,
                            ls=dash, lw=lw, label=f"{title} · {label}" if state == "pre" else None)
        ax.axvspan(0, 4, color=RULE, alpha=.45, lw=0)
        ax.axhline(0, color=MUTED, lw=1)
        ax.set(xlabel="time from odor onset (s)", xlim=(-5, 10),
               title="awake" if state == "pre" else "ket/xyl")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("population median (z)")
    axes[0].legend(fontsize=9, loc="upper left")
    save(fig, path)


def biphasic(out, path):
    import pandas as pd
    from .stage2_summarise import load_features
    features = load_features(out/"features")
    pick = features[(features.biphasic) & (~features.is_blank)
                    & (features.state == "pre") & (features.mean_z.abs() < .15)
                    & (features.excitation_area > 1.5) & (features.suppression_area > 1.5)]
    chosen = pick.sort_values("suppression_area", ascending=False).iloc[0]
    data = np.load(out/"trials"/f"group{int(chosen.group_id)}_trials.npz", allow_pickle=True)
    population = chosen.population
    index = [str(u) for u in data[f"{population}/unit_id"]].index(str(chosen.unit_id))
    levels = [str(x) for x in data["state_levels"]]
    rows_in = (data["state"] == levels.index("pre")) & (data["odor_id"] == chosen.odor_id)
    centres = data[f"{population}/bin_centre_s"]
    trace = np.nanmedian(data[f"{population}/binned"][index, rows_in, :], axis=0)
    fig, ax = plt.subplots(figsize=(8.2, 4.4), constrained_layout=True)
    ax.axvspan(0, 4, color=RULE, alpha=.45, lw=0)
    window = (centres >= 0) & (centres < 4)
    ax.fill_between(centres, 0, trace, where=(trace > 0) & window, color=EXCITE,
                    alpha=.6, lw=0, label="excitation area")
    ax.fill_between(centres, 0, trace, where=(trace < 0) & window, color=SUPPRESS,
                    alpha=.6, lw=0, label="suppression area")
    ax.plot(centres, trace, color=INK, lw=2)
    ax.axhline(0, color=MUTED, lw=1)
    ax.plot([0, 4], [chosen.mean_z]*2, color=INK, ls=":", lw=2.6,
            label=f"4 s signed mean = {chosen.mean_z:+.2f} z")
    ax.set(xlabel="time from odor onset (s)", ylabel="response (z)", xlim=(-5, 10))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=10, loc="lower left")
    save(fig, path)


def confusion(out, path):
    store = dict(np.load(out/"stage8_rdm.npz", allow_pickle=True))
    keys = [("Thy1 10x", "units", "Thy1 10x — 93% correct"),
            ("DAT 10x", "units", "DAT 10x — 89% correct"),
            ("TH 10x", "units", "TH 10x — 78% correct"),
            ("TH 20x deep", "somas", "TH 20x deep somas — 27% correct")]
    labels = [ODOR_LABEL[o] for o in ODOR_ORDER]
    fig, axes = plt.subplots(1, 4, figsize=(15.2, 4.5), constrained_layout=True)
    for index, (ax, (cohort, population, title)) in enumerate(zip(axes, keys)):
        matrix = store[f"{cohort}|{population}|pre|confusion"]
        image = ax.imshow(matrix, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
        for edge in FAMILY_BREAKS:
            ax.axhline(edge-.5, color="w", lw=.8, alpha=.6)
            ax.axvline(edge-.5, color="w", lw=.8, alpha=.6)
        ax.set_xticks(range(16)); ax.set_yticks(range(16))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticklabels(labels if index == 0 else [], fontsize=6)
        ax.set_title(title, fontsize=11)
    fig.colorbar(image, ax=axes.tolist(), shrink=.7, label="fraction classified as")
    save(fig, path)


def trial_matrix(out, path, group_id=213):
    x, odor, _ = session_matrix(out/"trials"/f"group{group_id}_trials.npz", "pre")
    keep = np.isin(odor, ODOR_ORDER); x, odor = x[keep], odor[keep]
    rank = {o: i for i, o in enumerate(ODOR_ORDER)}
    order = np.argsort([rank[int(o)] for o in odor], kind="stable")
    x, odor = x[order], odor[order]
    good = np.isfinite(x).all(axis=0)
    fig, ax = plt.subplots(figsize=(7.2, 6.0), constrained_layout=True)
    image = ax.imshow(np.corrcoef(x[:, good]), cmap="RdBu_r", vmin=-1, vmax=1,
                      interpolation="nearest")
    counts = [int(np.sum(odor == o)) for o in ODOR_ORDER if np.sum(odor == o)]
    bounds = np.cumsum(counts)
    for b in bounds[:-1]:
        ax.axhline(b-.5, color=INK, lw=.4, alpha=.5)
        ax.axvline(b-.5, color=INK, lw=.4, alpha=.5)
    centres = np.concatenate([[0], bounds[:-1]]) + np.array(counts)/2
    present = [ODOR_LABEL[o] for o in ODOR_ORDER if np.sum(odor == o)]
    ax.set_xticks(centres); ax.set_yticks(centres)
    ax.set_xticklabels(present, rotation=90, fontsize=7.5)
    ax.set_yticklabels(present, fontsize=7.5)
    fig.colorbar(image, ax=ax, shrink=.8, label="trial-to-trial pattern correlation")
    save(fig, path)


TIME = list(range(-5, 9))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eda-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    out, dst = args.eda_dir, args.output_dir
    dst.mkdir(parents=True, exist_ok=True)
    style()
    print("rendering deck figures:")
    blank_trace(out, dst/"blank.png")
    biphasic(out, dst/"biphasic.png")
    confusion(out, dst/"confusion.png")
    trial_matrix(out, dst/"trials.png")
    slope(dst/"suppression.png", [
        ("TH 20x deep somas", SERIES[0], .483, .025),
        ("TH 20x sup somas", SERIES[1], .467, .067),
        ("DAT 20x sup somas", SERIES[2], .365, .067),
        ("DAT 10x units", SERIES[3], .274, .031),
        ("TH 10x units", SERIES[4], .183, .000),
        ("Thy1 10x units", SERIES[5], .084, .000)],
        title="Suppression breadth collapses in every population",
        ylabel="fraction of odors suppressing the unit", ylim=(-.02, .55))
    slope(dst/"balance.png", [
        ("DAT 20x superficial", SERIES[2], .673, .579),
        ("TH 20x superficial", SERIES[1], .339, .470),
        ("TH 20x deep", SERIES[0], -.410, .703)],
        title="E/S balance: a depth gradient awake, erased by anesthesia",
        ylabel="+1 all excitation   ·   -1 all suppression", ylim=(-.6, .9), zero=True)
    bars(dst/"accuracy.png", [
        ("Thy1 10x", SERIES[5], 93.2, 82.8), ("DAT 10x", SERIES[3], 89.4, 70.9),
        ("TH 10x", SERIES[4], 78.1, 73.0), ("DAT 20x sup somas", SERIES[2], 58.8, 33.5),
        ("TH 20x sup somas", SERIES[1], 32.1, 31.1), ("TH 20x deep somas", SERIES[0], 27.3, 24.1)],
        title="Sixteen-way odor classification accuracy",
        xlabel="% correct   ·   solid awake, faded ket/xyl", chance=6.25)
    lines(dst/"ratio.png", [
        ("Thy1 10x", SERIES[5], [1.84, 1.45, 2.73, 3.16, 1.97, 1.02, .81, .49]),
        ("DAT 10x", SERIES[3], [.58, .75, 1.11, 1.66, .81, 1.09, .35, .29]),
        ("TH 20x deep somas", SERIES[0], [.03, .49, .84, .87, 1.64, .36, .19, .24]),
        ("TH 10x", SERIES[4], [.12, .12, .29, .64, .01, -.05, -.04, -.06]),
        ("TH 20x sup somas", SERIES[1], [.79, .27, .25, .22, .18, .06, .16, .24])],
        list(range(8)), title="Ratio separation for pair 39/40 builds late, awake",
        ylabel="crossvalidated distance",
        xlabel="window start, seconds from odor onset", shade=(0, 4))
    lines(dst/"running.png", [
        ("TH 20x sup somas", SERIES[1], [.01, 0, .03, -.01, -.03, -.02, .21, .31, .27, .27, .26, .23, .18, .13]),
        ("Thy1 10x", SERIES[5], [0, .04, .03, 0, -.01, .12, .18, .24, .23, .11, .07, .08, .07, .07]),
        ("TH 20x deep somas", SERIES[0], [.03, -.02, .03, -.02, 0, .07, .17, .23, .17, .12, .21, .13, .14, .08]),
        ("TH 10x", SERIES[4], [.02, 0, -.02, -.03, .04, .07, .20, .22, .17, .16, .03, .08, .09, .11]),
        ("DAT 10x", SERIES[3], [.02, -.04, .02, -.04, .02, 0, .13, .22, .21, .16, .07, .12, .09, .07]),
        ("DAT 20x sup somas", SERIES[2], [-.01, -.01, -.01, 0, .03, -.02, .09, .17, .13, .09, .01, .05, .08, .06])],
        TIME, title="Running coupling is flat before the odor and peaks at 2 s",
        ylabel="median r", xlabel="time from odor onset (s)", shade=(0, 4))
    lines(dst/"pupil.png", [
        ("TH 20x sup somas", SERIES[1], [.01, .02, -.01, -.03, 0, .11, .21, .23, .24, .17, .15, .05, .02, .05]),
        ("DAT 10x", SERIES[3], [0, -.01, -.02, .01, .04, 0, .08, .15, .12, .08, .07, .01, 0, -.03]),
        ("TH 10x", SERIES[4], [.04, .06, -.03, -.12, .04, .10, .15, .11, .01, .05, .11, .04, .02, 0]),
        ("Thy1 10x", SERIES[5], [.03, -.01, -.01, -.02, .02, 0, .04, .07, .03, .02, .02, 0, -.01, -.03])],
        TIME, title="Pupil dilation: same shape, about three times weaker",
        ylabel="median r", xlabel="time from odor onset (s)", shade=(0, 4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
