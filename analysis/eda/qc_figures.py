"""Acquisition-QC figures: reagent batch, blank behaviour, pupil, protocol."""
from __future__ import annotations
import argparse, base64, io, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from .report_figures import THEMES, EXCITE, SUPPRESS, style, encode, blank_trace_figure

SERIES = {"light": {"DAT": "#2F5578", "TH": "#5C4570", "Thy1": "#39603B"},
          "dark":  {"DAT": "#7BA7D4", "TH": "#B295C4", "Thy1": "#8CBA8E"}}
MAKES = pd.to_datetime(["2026-07-06", "2026-07-22", "2026-08-06"])
ORDER = ["before 7/6 (older stock)", "07/06", "07/22", "08/06"]


def batch_figure(table, theme):
    c = style(theme); colour = SERIES[theme]
    b = table[table.scale == "10x"].copy()
    b["bi"] = b.batch.map({k: i for i, k in enumerate(ORDER)})
    rows = []
    for (line, mouse), g in b.groupby(["line", "mouse"]):
        m = (g.groupby(["bi", "batch"]).mo_z.median().reset_index()
             .sort_values("bi"))
        for i in range(len(m)-1):
            rows.append({"line": line, "mouse": mouse,
                         "from": m.iloc[i].batch, "to": m.iloc[i+1].batch,
                         "a": m.iloc[i].mo_z, "b": m.iloc[i+1].mo_z})
    transitions = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3), constrained_layout=True)
    ax = axes[0]
    for _, r in transitions.iterrows():
        ax.plot([0, 1], [r.a, r.b], "o-", color=colour[r.line], lw=2.4, ms=9,
                alpha=.92)
        ax.annotate(f"{r.mouse}  {r['from']} to {r.to}", (0, r.a),
                    xytext=(-9, 7), textcoords="offset points", fontsize=8.2,
                    ha="right", color=colour[r.line])
    ax.axhline(0, color=c["muted"], lw=1)
    ax.set(xticks=[0, 1], xticklabels=["older batch", "newer batch"],
           xlim=(-.95, 1.2), ylabel="mineral-oil response (z, 1–4 s)",
           title="Every within-mouse batch transition declines")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    s = b[b.days_since_make.notna()]
    for line, g in s.groupby("line"):
        ax.plot(g.days_since_make, g.mo_z, "o", ms=9, color=colour[line],
                label=line, alpha=.85)
    th = s[s.line == "Thy1"]
    fit = np.polyfit(th.days_since_make, th.mo_z, 1)
    xs = np.linspace(0, 15, 10)
    ax.plot(xs, np.polyval(fit, xs), "-", color=colour["Thy1"], lw=2, alpha=.65)
    ax.axhline(0, color=c["muted"], lw=1)
    ax.set(xlabel="days since the odors were made",
           ylabel="mineral-oil response (z, 1–4 s)",
           title="Thy1, same two mice throughout:  rho = +0.73, p = 0.039")
    ax.legend(frameon=False, fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)
    return encode(fig)


def timeline_figure(table, inventory, theme):
    c = style(theme); colour = SERIES[theme]
    aw = table[(table.scale == "10x")].copy()
    fig, axes = plt.subplots(2, 1, figsize=(11.4, 6.0), constrained_layout=True,
                             sharex=True)
    ax = axes[0]
    for line, g in aw.groupby("line"):
        g = g.sort_values("d")
        ax.plot(pd.to_datetime(g.d), g.mo_z, "o-", ms=8, lw=1.8,
                color=colour[line], label=line, alpha=.9)
    for make in MAKES:
        ax.axvline(make, color=EXCITE[theme], lw=1.6, ls="--", alpha=.8)
        ax.annotate("odors made", (make, ax.get_ylim()[1]), xytext=(3, -10),
                    textcoords="offset points", fontsize=8, color=EXCITE[theme],
                    rotation=90, va="top")
    ax.axhline(0, color=c["muted"], lw=1)
    ax.set(ylabel="mineral oil (z)", title="Mineral-oil response, 10x awake")
    ax.legend(frameon=False, fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    # Acquired trials, from the auxiliary product: what was actually run,
    # before any processing exclusions.
    inv = inventory.dropna(subset=["aux_n_trial"]).copy()
    inv["d"] = pd.to_datetime(inv.date, format="%Y%m%d")
    ax.plot(inv.d, inv.aux_n_trial, "o", ms=7, color=SUPPRESS[theme], alpha=.85)
    for make in MAKES:
        ax.axvline(make, color=EXCITE[theme], lw=1.6, ls="--", alpha=.8)
    ax.annotate("session length steps 224 to 160\nat the 7/22 make",
                (pd.Timestamp("2026-07-23"), 205), xytext=(8, 0),
                textcoords="offset points", fontsize=8.5, color=c["fg"])
    ax.set(ylabel="trials acquired per session", ylim=(140, 245),
           title="Session length changed once, at the 7/22 odor make")
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    for label in ax.get_xticklabels():
        label.set_rotation(45); label.set_ha("right")
    return encode(fig)


def pupil_figure(quality, theme):
    c = style(theme)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), constrained_layout=True)
    for ax, (metric, title) in zip(axes, [
            ("cov_masked", "Usable pupil samples after quality masking"),
            ("clipped", "Fraction of samples clipped")]):
        for index, state in enumerate(["pre", "post"]):
            s = quality[quality.state == state]
            x = np.full(len(s), index) + np.random.default_rng(0).uniform(
                -.13, .13, len(s))
            ax.plot(x, s[metric], "o", ms=8, alpha=.75,
                    color=(EXCITE[theme] if state == "pre" else SUPPRESS[theme]))
        ax.set(xticks=[0, 1], xticklabels=["awake", "ket/xyl"], xlim=(-.5, 1.5),
               ylim=(-.05, 1.05), ylabel="fraction of samples", title=title)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].axhline(.7, color=c["muted"], ls="--", lw=1.3)
    axes[0].annotate("coverage cutoff", (1.45, .72), fontsize=8.5,
                     color=c["muted"], ha="right")
    return encode(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eda-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    table = pd.read_csv(args.eda_dir/"mo_by_age.csv")
    quality = pd.read_csv(args.eda_dir/"pupil_quality.csv")
    inventory = pd.read_csv(args.eda_dir/"stage0_inventory.csv")
    payload = {}
    for theme in ("light", "dark"):
        payload[f"batch_{theme}"] = batch_figure(table, theme)
        payload[f"timeline_{theme}"] = timeline_figure(table, inventory, theme)
        payload[f"pupil_{theme}"] = pupil_figure(quality, theme)
        payload[f"trace_{theme}"] = blank_trace_figure(args.eda_dir, theme)
        print("rendered", theme, flush=True)
    args.output.write_text(json.dumps(payload))
    print(f"wrote {len(payload)} images, "
          f"{sum(len(v) for v in payload.values())/1e6:.2f} MB base64")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
