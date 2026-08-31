"""Figures for the exploratory-pass report, rendered for light and dark themes."""
from __future__ import annotations
import argparse, base64, io, json
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from .common import manifest_rows
from .stage8_rdm import (FAMILY_BREAKS, ODOR_LABEL, ODOR_ORDER,
                         session_matrix)

THEMES = {
    "light": {"fg": "#15202A", "muted": "#55646F", "bg": "none", "grid": "#DCE3E4"},
    "dark":  {"fg": "#E4EBEB", "muted": "#9DACB3", "bg": "none", "grid": "#28343A"},
}
EXCITE = {"light": "#A8632A", "dark": "#D69355"}
SUPPRESS = {"light": "#2C6C77", "dark": "#5FA9B4"}


def style(theme):
    c = THEMES[theme]
    mpl.rcParams.update({
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8.5,
        "text.color": c["fg"], "axes.labelcolor": c["fg"],
        "xtick.color": c["muted"], "ytick.color": c["muted"],
        "axes.edgecolor": c["grid"], "axes.linewidth": .8,
        "figure.facecolor": "none", "axes.facecolor": "none",
        "savefig.facecolor": "none", "savefig.transparent": True,
        "xtick.major.width": .8, "ytick.major.width": .8,
        "legend.frameon": False,
    })
    return c


def encode(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=190, bbox_inches="tight",
                transparent=True)
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode()


# ---------------------------------------------------------------- blank trace
def blank_trace_figure(out, theme):
    c = style(theme)
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
            rows = data["state"] == levels.index(state)
            odor = data["odor_id"][rows]
            block = binned[:, rows, :]
            for label, mask in (("blank", odor == 0), ("odor", odor != 0)):
                if not mask.any():
                    continue
                trace = np.nanmedian(np.nanmedian(block[:, mask, :], axis=1), axis=0)
                cohorts.setdefault((row["cohort"], state, label), []).append(trace)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9), sharey=True,
                             constrained_layout=True)
    show = [("TH 10x", "TH 10x units"), ("DAT 20x superficial", "DAT 20x somas")]
    for ax, state in zip(axes, ("pre", "post")):
        for (cohort, title), width in zip(show, (2.0, 1.4)):
            for label, colour, dash in (("odor", EXCITE[theme], "-"),
                                        ("blank", SUPPRESS[theme], "--")):
                stack = cohorts.get((cohort, state, label))
                if not stack:
                    continue
                ax.plot(centres, np.nanmedian(np.stack(stack), axis=0),
                        color=colour, ls=dash, lw=width, alpha=.95,
                        label=f"{title} · {label}" if state == "pre" else None)
        ax.axvspan(0, 4, color=c["grid"], alpha=.45, lw=0)
        ax.axhline(0, color=c["muted"], lw=.6)
        ax.set(xlabel="time from odor onset (s)", xlim=(-5, 10),
               title="awake" if state == "pre" else "ket/xyl")
    axes[0].set_ylabel("population median (z)")
    axes[0].legend(fontsize=6.5, loc="upper left")
    return encode(fig)


# ------------------------------------------------------------ biphasic example
def biphasic_figure(out, theme):
    c = style(theme)
    import pandas as pd
    from .stage2_summarise import load_features
    features = load_features(out/"features")
    pick = features[(features.biphasic) & (~features.is_blank)
                    & (features.state == "pre")
                    & (features.mean_z.abs() < .15)
                    & (features.excitation_area > 1.5)
                    & (features.suppression_area > 1.5)]
    if pick.empty:
        return None
    chosen = pick.sort_values("suppression_area", ascending=False).iloc[0]
    cache = out/"trials"/f"group{int(chosen.group_id)}_trials.npz"
    data = np.load(cache, allow_pickle=True)
    population = chosen.population
    units = [str(u) for u in data[f"{population}/unit_id"]]
    index = units.index(str(chosen.unit_id))
    levels = [str(x) for x in data["state_levels"]]
    rows = (data["state"] == levels.index("pre")) & (data["odor_id"] == chosen.odor_id)
    centres = data[f"{population}/bin_centre_s"]
    trace = np.nanmedian(data[f"{population}/binned"][index, rows, :], axis=0)
    fig, ax = plt.subplots(figsize=(5.4, 2.9), constrained_layout=True)
    ax.axvspan(0, 4, color=c["grid"], alpha=.45, lw=0)
    window = (centres >= 0) & (centres < 4)
    ax.fill_between(centres, 0, trace, where=(trace > 0) & window,
                    color=EXCITE[theme], alpha=.55, lw=0, label="excitation area")
    ax.fill_between(centres, 0, trace, where=(trace < 0) & window,
                    color=SUPPRESS[theme], alpha=.55, lw=0, label="suppression area")
    ax.plot(centres, trace, color=c["fg"], lw=1.4)
    ax.axhline(0, color=c["muted"], lw=.6)
    ax.plot([0, 4], [chosen.mean_z]*2, color=c["fg"], ls=":", lw=1.8,
            label=f"4 s signed mean = {chosen.mean_z:+.2f} z")
    ax.set(xlabel="time from odor onset (s)", ylabel="response (z)", xlim=(-5, 10),
           title=f"one unit-odor pair the signed mean cannot see\n"
                 f"{chosen.cohort} · {population} · unit {chosen.unit_id} · odor {int(chosen.odor_id)}")
    ax.legend(fontsize=6.5, loc="lower left", framealpha=0)
    return encode(fig)


# ------------------------------------------------------------------- confusion
def confusion_figure(out, theme, state="pre"):
    c = style(theme)
    keys = [("Thy1 10x", "Thy1 10x"), ("DAT 10x", "DAT 10x"), ("TH 10x", "TH 10x"),
            ("DAT 20x superficial", "DAT 20x sup. somas"),
            ("TH 20x superficial", "TH 20x sup. somas"),
            ("TH 20x deep", "TH 20x deep somas")]
    store = dict(np.load(out/"stage8_rdm.npz", allow_pickle=True))
    labels = [ODOR_LABEL[o] for o in ODOR_ORDER]
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 8.4), constrained_layout=True)
    for index, (ax, (cohort, title)) in enumerate(zip(axes.ravel(), keys)):
        population = "units" if "10x" in cohort else "somas"
        matrix = store.get(f"{cohort}|{population}|{state}|confusion")
        if matrix is None:
            ax.axis("off"); continue
        image = ax.imshow(matrix, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
        for edge in FAMILY_BREAKS:
            ax.axhline(edge-.5, color=c["fg"], lw=.6, alpha=.5)
            ax.axvline(edge-.5, color=c["fg"], lw=.6, alpha=.5)
        ax.set_xticks(range(16)); ax.set_yticks(range(16))
        ax.set_yticklabels(labels, fontsize=5.4)
        if index < 3:
            ax.set_xticklabels([])
        else:
            ax.set_xticklabels(labels, rotation=90, fontsize=5.4)
            ax.set_xlabel("classified as")
        if index % 3 == 0:
            ax.set_ylabel("true odor")
        ax.set_title(f"{title} — {np.nanmean(np.diag(matrix))*100:.0f}% correct", pad=5)
    bar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=.5,
                       label="fraction classified as")
    bar.outline.set_edgecolor(c["grid"])
    return encode(fig)


def trial_matrix_figure(out, theme, group_id=213):
    c = style(theme)
    got = session_matrix(out/"trials"/f"group{group_id}_trials.npz", "pre")
    x, odor, _ = got
    keep = np.isin(odor, ODOR_ORDER); x, odor = x[keep], odor[keep]
    rank = {o: i for i, o in enumerate(ODOR_ORDER)}
    order = np.argsort([rank[int(o)] for o in odor], kind="stable")
    x, odor = x[order], odor[order]
    good = np.isfinite(x).all(axis=0)
    fig, ax = plt.subplots(figsize=(6.6, 5.8), constrained_layout=True)
    image = ax.imshow(np.corrcoef(x[:, good]), cmap="RdBu_r", vmin=-1, vmax=1,
                      interpolation="nearest")
    counts = [int(np.sum(odor == o)) for o in ODOR_ORDER if np.sum(odor == o)]
    bounds = np.cumsum(counts)
    for b in bounds[:-1]:
        ax.axhline(b-.5, color=c["fg"], lw=.35, alpha=.45)
        ax.axvline(b-.5, color=c["fg"], lw=.35, alpha=.45)
    centres = np.concatenate([[0], bounds[:-1]]) + np.array(counts)/2
    present = [ODOR_LABEL[o] for o in ODOR_ORDER if np.sum(odor == o)]
    ax.set_xticks(centres); ax.set_yticks(centres)
    ax.set_xticklabels(present, rotation=90, fontsize=5.6)
    ax.set_yticklabels(present, fontsize=5.6)
    bar = fig.colorbar(image, ax=ax, shrink=.8, label="trial-to-trial pattern correlation")
    bar.outline.set_edgecolor(c["grid"])
    ax.set_title("Thy1 10x awake — every trial against every trial, sorted by odor")
    return encode(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    out = args.output_dir
    payload = {}
    for theme in ("light", "dark"):
        payload[f"blank_{theme}"] = blank_trace_figure(out, theme)
        payload[f"biphasic_{theme}"] = biphasic_figure(out, theme)
        payload[f"confusion_{theme}"] = confusion_figure(out, theme)
        payload[f"trials_{theme}"] = trial_matrix_figure(out, theme)
        print("rendered", theme, flush=True)
    (out/"report_figures.json").write_text(json.dumps(payload))
    total = sum(len(v) for v in payload.values() if v)
    print(f"wrote {len(payload)} images, {total/1e6:.2f} MB base64")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
