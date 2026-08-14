"""
A picture of what the baseline correction did, per ROI and across the session.

The numbers in `detrend_traces` say the correction worked -- pre-odor drift
removed, responses preserved at r = 0.99. They cannot say whether it did
something silly to a particular glomerulus, and a correction applied to every
trace deserves to be looked at rather than trusted.

The figure is laid out to make the two failure modes visible:

  **Rows are ROIs**, a high responder and a median one, plus mineral oil on the
  same high responder as a control. If the correction were absorbing signal it
  would show as the corrected response being smaller than the raw one -- and
  mineral oil, which should have no response, is where an invented one would
  be most obvious.

  **Columns are early and late in the session.** The transient's amplitude
  drifts -- it roughly doubled across group 217 -- so a correction that looks
  right at trial 10 can be wrong at trial 150. Early and late side by side is
  the check that the per-trial amplitudes are tracking it.

Raw, the fitted curve, and the corrected trace are drawn together, so the thing
being subtracted is visible rather than implied.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _response_rank(roi, on_frames, off_frames, trials, n_base=25):
    """Median dF/F per ROI over the given trials, for picking examples."""

    idx = np.flatnonzero(trials)
    base = np.stack([roi[:, k, max(on_frames[k] - n_base, 0):on_frames[k]].mean(1)
                     for k in idx], 1)
    resp = np.stack([roi[:, k, on_frames[k]:off_frames[k]].mean(1) for k in idx], 1)

    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nanmedian((resp - base) / np.abs(base), axis=1)


def detrend_qc_figure(
    path: str | Path,
    *,
    odors: None | list[int] = None,
    n_bin: int = 5,
    output: None | str | Path = None,
):
    """
    Draw the before/after comparison for one processing round.

    Reads `/traces/roi`, `/traces/roi_detrended` and `/detrend` from a round
    written by `finalize_session`, so it needs no re-extraction and no refit.
    """

    import matplotlib
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    from .h5io import open_h5

    path = Path(path)

    with open_h5(path) as f:
        if "traces/roi_detrended" not in f:
            raise ValueError(
                f"{path.name} has no /traces/roi_detrended. Re-run "
                f"finalize_session with detrend=True."
            )
        raw = f["traces/roi"][:]
        corrected = f["traces/roi_detrended"][:]
        on = f["trials/odor_on_frame"][:]
        off = f["trials/odor_off_frame"][:]
        odor_ids = f["trials/odor_id"][:]
        a_fast = f["detrend/a_fast"][:]
        a_slow = f["detrend/a_slow"][:]
        tau_fast = float(f["detrend"].attrs["tau_fast_s"])
        tau_slow = float(f["detrend"].attrs["tau_slow_s"])
        frame_rate = float(f.attrs["frame_rate"])
        exp_name = str(f.attrs["exp_name"])

    n_trial = raw.shape[1]
    time = np.arange(raw.shape[2]) / frame_rate

    def early_late(odor):
        """First and last occurrences *of this odor*, not of any trial.

        Taking the first 20 trials of the session gives one or two trials of a
        given odor when sixteen are interleaved, which is too few to average
        and leaves panels empty. Selecting per odor keeps the counts equal and
        comparable between the two columns.
        """
        occurrences = np.flatnonzero(odor_ids == odor)
        take = max(1, min(n_bin, len(occurrences) // 2))
        first = np.zeros(n_trial, bool); first[occurrences[:take]] = True
        last = np.zeros(n_trial, bool); last[occurrences[-take:]] = True
        return first, last

    available = sorted(int(o) for o in np.unique(odor_ids))
    if odors is None:
        # The two strongest odors, plus mineral oil (0) as the control.
        strength = {o: np.nanmax(_response_rank(raw, on, off, odor_ids == o))
                    for o in available if o != 0}
        odors = sorted(strength, key=strength.get, reverse=True)[:2]
        if 0 in available:
            odors = odors + [0]

    rank = _response_rank(raw, on, off, np.isin(odor_ids, [o for o in odors if o != 0]))
    high = int(np.nanargmax(rank))
    median = int(np.argsort(rank)[len(rank) // 2])

    panels = [(high, o, f"high responder (ROI {high})") for o in odors[:2]]
    panels.append((median, odors[0], f"median ROI ({median})"))
    if 0 in odors:
        panels.append((high, 0, f"ROI {high}, mineral oil"))

    fig, axes = plt.subplots(len(panels), 2, figsize=(14, 3.1 * len(panels)),
                             sharex=True, squeeze=False)

    for row, (roi_index, odor, label) in enumerate(panels):
        first, last = early_late(odor)
        for col, (window, when) in enumerate(((first, "early"), (last, "late"))):
            select = window & (odor_ids == odor)
            ax = axes[row][col]

            if not select.any():
                ax.set_axis_off()
                ax.set_title(f"{label} · odor {odor} · {when}: no trials", fontsize=9)
                continue

            r = np.nanmean(raw[roi_index, select, :], axis=0)
            c = np.nanmean(corrected[roi_index, select, :], axis=0)
            fitted = r - c                        # exactly what was subtracted

            ax.plot(time, r, color="0.35", lw=1.3, label="raw")
            ax.plot(time, c, color="steelblue", lw=1.4, label="detrended")
            ax.plot(time, fitted + np.nanmean(c[:on.min()]), color="crimson",
                    lw=1.1, ls="--", label="subtracted (offset for display)")
            ax.axvspan(on.min() / frame_rate, off.max() / frame_rate,
                       color="0.92", zorder=0)
            ax.set_title(
                f"{label} · odor {odor} · {when} ({select.sum()} trials, "
                f"a_fast {np.nanmedian(a_fast[roi_index, select]):.0f})",
                fontsize=9,
            )
            if row == 0 and col == 0:
                ax.legend(fontsize=7)
            if col == 0:
                ax.set_ylabel("F (a.u.)")

    for ax in axes[-1]:
        ax.set_xlabel("time from acquisition start (s)")

    fig.suptitle(
        f"detrend QC — {exp_name} — tau {tau_fast:.2f} / {tau_slow:.2f} s"
        f"   (early = first {n_bin} trials, late = last {n_bin})",
        y=1.0, fontsize=12,
    )
    fig.tight_layout()

    output = Path(output) if output is not None else path.with_name(
        path.stem + "_detrendqc.png"
    )
    fig.savefig(output, dpi=110, bbox_inches="tight")
    plt.close(fig)

    return {
        "figure": str(output),
        "odors": [int(o) for o in odors],
        "high_roi": high,
        "median_roi": median,
        "tau_fast_s": tau_fast,
        "tau_slow_s": tau_slow,
        "a_fast_early": float(np.nanmedian(a_fast[:, :n_bin])),
        "a_fast_late": float(np.nanmedian(a_fast[:, -n_bin:])),
        "a_slow_early": float(np.nanmedian(a_slow[:, :n_bin])),
        "a_slow_late": float(np.nanmedian(a_slow[:, -n_bin:])),
        # Per-trial amplitudes far from the session median mark trials whose
        # baseline the model struggled with -- worth seeing next to the figure.
        "a_fast_outlier_trials": [
            int(i) for i in np.flatnonzero(
                np.abs(np.nanmedian(a_fast, axis=0)
                       - np.nanmedian(a_fast))
                > 3 * np.nanstd(np.nanmedian(a_fast, axis=0))
            )
        ],
    }
