"""
QC on the extracted traces: does this session look like odor coding?

`qc.py` checks the *images* -- contrast, striping, drift in the correlation
maps. That catches a session that segmented badly, but it says nothing about
whether the ROIs it produced respond to anything, and a mask can look
immaculate over a session where the traces are noise.

The check that answers that is sparsity, in its two directions. They are not
interchangeable, and a session can fail either one:

    population sparsity   for one odor, what fraction of glomeruli respond.
                          Low means the odor drove the whole field -- which in
                          the bulb is usually motion, a z-shift, or a baseline
                          taken over a bleaching window, not a real response.
    lifetime sparsity     for one glomerulus, how selectively it responds
                          across odors. Zero everywhere means the ROIs are not
                          resolving odor identity: often ROIs so large they
                          average several glomeruli together, or a mask that
                          drifted off the tissue.

Both are reported two ways, because they disagree in informative places:

    Treves-Rolls    continuous, uses the whole response distribution. 0 is a
                    flat response to everything, 1 is all the response in one
                    odor (or one glomerulus).
    count           fraction of |z| over threshold. Blunter, but it is the
                    number people actually report, and it does not move when
                    a single large response rescales the distribution.

Responses are rectified before either, so a suppressed glomerulus counts as
responsive -- these recordings have real suppression, and a measure built on
positive deflections alone would score a suppressed unit as silent.

Two bleaching diagnostics come along with it, because they corrupt everything
above before they are obvious anywhere else. Measured on exp 132: -10% across
the 5 s pre-odor window within a trial, and -40% in baseline F from the first
acquisition to the last. The first is why the baseline window here defaults to
the last second before onset rather than the whole pre-period -- averaging
F0 over 5 s of decay places it above the trace by the time the odor arrives,
and every dF/F comes out negative.

Runs off the written round, so it costs seconds and needs no re-streaming.
"""

from __future__ import annotations

import json

from datetime import date
from pathlib import Path

import numpy as np

# Seconds before odor onset used for F0. Short on purpose -- see the module
# docstring on within-trial bleaching.
BASELINE_S = 1.0

# Fallback thresholds, used only when the null cannot be built (a pre-odor
# period too short to hold a sham window). Asymmetric, because suppression in
# these recordings is real but reliably smaller in z than excitation -- a
# glomerulus can only go down as far as its baseline, while there is no
# ceiling going up. A symmetric threshold conservative enough for excitation
# scores nearly every suppressed unit as silent: the first pass over exp 132
# returned 0 suppressed ROIs against 27 excited.
Z_EXCITED = 1.5
Z_SUPPRESSED = -0.5

# Share of ROI-odor pairs the null is allowed to put past each threshold. Per
# tail, so the two sides are calibrated independently and come out asymmetric
# on their own rather than by assumption.
TARGET_FPR = 0.01

# Spacing of sham windows through the pre-odor period, as a fraction of the
# response window. Below 1 the windows overlap, which is deliberate: the pre
# period only holds two non-overlapping windows, far too few to place a 1%
# percentile. Overlapping windows are correlated and do not each count as an
# independent sample, which is why the threshold is validated on offsets held
# out from the ones it was fitted on rather than trusted from the sample size.
SHAM_STEP_FRACTION = 0.2


def treves_rolls(responses: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Sparseness of a rectified response vector, 0 (flat) to 1 (one-of-N).

        S = [1 - (mean r)^2 / mean(r^2)] / (1 - 1/N)

    The normalisation by `1 - 1/N` is what makes values comparable between
    sessions with different numbers of odors or ROIs; without it the maximum
    attainable sparseness depends on N, and a 7-odor panel could not be put
    beside a 16-odor one.
    """

    r = np.abs(np.asarray(responses, dtype=np.float64))
    n = r.shape[axis]

    if n < 2:
        return np.full(np.delete(r.shape, axis), np.nan)

    mean = np.nanmean(r, axis=axis)
    mean_sq = np.nanmean(r**2, axis=axis)

    with np.errstate(invalid="ignore", divide="ignore"):
        raw = 1.0 - (mean**2) / mean_sq

    return np.where(mean_sq > 0, raw / (1.0 - 1.0 / n), np.nan)


def trial_responses(
    roi: np.ndarray,
    *,
    odor_on_frames: np.ndarray,
    odor_off_frames: np.ndarray,
    frame_rate: float,
    baseline_s: float = BASELINE_S,
) -> dict:
    """
    Per-ROI, per-trial response amplitude, aligned on each trial's own onset.

    Returns dF/F and a z-score against the baseline's own variability. Both
    are kept: dF/F is the interpretable magnitude, z is what decides
    responsive-or-not, and they rank ROIs differently -- a dim ROI can carry a
    large dF/F on very little signal.
    """

    n_roi, n_trial, n_frame = roi.shape
    n_base = max(1, int(round(baseline_s * frame_rate)))

    dff = np.full((n_roi, n_trial), np.nan)
    z = np.full((n_roi, n_trial), np.nan)
    f0 = np.full((n_roi, n_trial), np.nan)

    for trial in range(n_trial):
        on = int(odor_on_frames[trial])
        off = int(odor_off_frames[trial])

        if on <= 0 or off <= on or off > n_frame:
            continue

        base = roi[:, trial, max(on - n_base, 0):on]
        resp = roi[:, trial, on:off]

        if base.shape[1] == 0 or resp.shape[1] == 0:
            continue

        base_mean = np.nanmean(base, axis=1)
        base_sd = np.nanstd(base, axis=1)
        resp_mean = np.nanmean(resp, axis=1)

        f0[:, trial] = base_mean

        with np.errstate(invalid="ignore", divide="ignore"):
            dff[:, trial] = (resp_mean - base_mean) / np.where(
                np.abs(base_mean) > 1e-9, base_mean, np.nan
            )
            z[:, trial] = (resp_mean - base_mean) / np.where(
                base_sd > 1e-9, base_sd, np.nan
            )

    return {"dff": dff, "z": z, "f0": f0, "n_baseline_frames": n_base}


def sham_null(
    roi: np.ndarray,
    *,
    odor_on_frames: np.ndarray,
    odor_off_frames: np.ndarray,
    odor_ids: np.ndarray,
    frame_rate: float,
    baseline_s: float,
    target_fpr: float = TARGET_FPR,
    step_fraction: float = SHAM_STEP_FRACTION,
) -> dict:
    """
    Build this session's null for the response statistic, and read the
    thresholds off it.

    The thresholds should be outputs, not inputs. A number like 1.5 carries an
    implied false-positive rate that depends on trial count, baseline length,
    bleaching, and how noisy these particular ROIs are -- none of which a
    z table knows, and all of which differ between sessions. Reading the
    threshold off the session's own null makes the *rate* the thing that is
    fixed and comparable, which is what anyone actually wants held constant
    when comparing two experiments.

    Two ingredients:

    **A sham window.** The whole measurement is repeated with the odor window
    slid back into the pre-odor period, so baseline and response both sit
    before the valve opens. Several offsets are used, spaced so each has its
    own baseline. The sham inherits everything real about the session --
    the same trials, same decay, same noise -- and differs only in that no
    odor was delivered.

    **The real odor grouping.** The statistic being thresholded is not a
    single trial but a mean over the ~25 trials of one odor, whose null is far
    narrower than a per-trial one. So each sham window is collapsed by odor
    exactly as the real one is.

    It is tempting to shuffle which trials form each group, since that gives
    unlimited null samples. It is also wrong, and measurably so: shuffling
    assumes trials are exchangeable, and here they are not. Odors are
    delivered in blocks and baseline F falls ~40% across a session, so trials
    of one odor share a position in time and their mean is genuinely more
    variable than a mean over trials drawn from everywhere. On exp 132 the
    shuffled null put `z_excited` at +0.22, which then fired on 6.9% of sham
    ROI-odor pairs against a 1% target. Keeping the real grouping costs
    samples and gets the width right.

    **Held-out validation.** Because the windows overlap, the sample count
    overstates how much independent information there is, so `achieved_fpr` is
    measured on offsets the threshold was not fitted to. That number, not the
    sample count, is what says whether the calibration holds.

    The two tails are calibrated separately, so asymmetry emerges from the
    data rather than being assumed. Both a pooled threshold (one pair of
    numbers for the session) and per-ROI thresholds are returned; per-ROI is
    stricter on noisy ROIs and more sensitive on quiet ones, at the cost of a
    threshold that no longer means one thing across the figure.
    """

    on = np.asarray(odor_on_frames)
    off = np.asarray(odor_off_frames)
    odor_ids = np.asarray(odor_ids)

    span = int(np.median(off - on))
    n_base = max(1, int(round(baseline_s * frame_rate)))

    # A window fits when its response ends at or before onset and its baseline
    # still starts inside the recording.
    earliest, latest = span, int(on.min()) - n_base
    step = max(1, int(round(step_fraction * span)))

    offsets = list(range(earliest, latest + 1, step))

    if len(offsets) < 2:
        return {"available": False, "reason": "pre-odor period too short"}

    keys = np.unique(odor_ids)

    # One (n_roi, n_odor) matrix per offset, built exactly as the real one is.
    matrices = []
    for shift in offsets:
        sham = trial_responses(
            roi, odor_on_frames=on - shift, odor_off_frames=on - shift + span,
            frame_rate=frame_rate, baseline_s=baseline_s,
        )
        matrices.append(_by_odor(sham["z"], odor_ids, keys))

    # Alternate offsets, so the fitted and held-out sets are interleaved
    # through the pre period rather than split early-versus-late -- the decay
    # makes those two halves genuinely different.
    fit = np.concatenate(matrices[0::2], axis=1)
    held_out = np.concatenate(matrices[1::2], axis=1)

    low = 100.0 * target_fpr
    high = 100.0 * (1.0 - target_fpr)

    fit_values = fit[np.isfinite(fit)]
    z_up = float(np.percentile(fit_values, high))
    z_down = float(np.percentile(fit_values, low))

    per_roi_excited = np.nanpercentile(fit, high, axis=1)
    per_roi_suppressed = np.nanpercentile(fit, low, axis=1)

    return {
        "available": True,
        "target_fpr": target_fpr,
        "sham_offsets_frames": [int(s) for s in offsets],
        "n_offsets": {"fitted": len(matrices[0::2]), "held_out": len(matrices[1::2])},
        "window_frames": {"baseline": n_base, "response": span},
        "n_null_samples": int(fit_values.size),
        "z_excited": round(z_up, 4),
        "z_suppressed": round(z_down, 4),
        "per_roi": {
            "z_excited": per_roi_excited,
            "z_suppressed": per_roi_suppressed,
            "samples_per_roi": int(fit.shape[1]),
            # A percentile needs enough samples that the tail is populated
            # rather than extrapolated. Below ~5 expected samples beyond the
            # threshold, the "1st percentile" of a row is essentially its
            # minimum, which is a different and much more permissive quantity.
            "reliable": bool(fit.shape[1] * target_fpr >= 5),
            "excited_range": [_round(np.nanmin(per_roi_excited)),
                              _round(np.nanmax(per_roi_excited))],
            "suppressed_range": [_round(np.nanmin(per_roi_suppressed)),
                                 _round(np.nanmax(per_roi_suppressed))],
        },
        # On offsets the thresholds were never fitted to. This is the number
        # that says whether the calibration holds.
        "achieved_fpr": {
            "excited": _round(np.nanmean(held_out >= z_up)),
            "suppressed": _round(np.nanmean(held_out <= z_down)),
        },
    }


def _by_odor(values: np.ndarray, odor_ids: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """Trial-average an (n_roi, n_trial) array into (n_roi, n_odor)."""

    out = np.full((values.shape[0], len(keys)), np.nan)

    for i, key in enumerate(keys):
        trials = odor_ids == key
        if trials.any():
            out[:, i] = np.nanmean(values[:, trials], axis=1)

    return out


def bleaching(roi: np.ndarray, odor_on_frames: np.ndarray, *, n_pre: int) -> dict:
    """
    Two decay measures, within a trial and across the session.

    Both matter and they have different causes: within-trial decay is
    photobleaching under the beam during one acquisition, while across-session
    decline mixes cumulative bleaching with z-drift and can mean the mask has
    slid off the tissue it was drawn on.
    """

    within = np.nanmean(roi[:, :, :n_pre], axis=(0, 1))
    across = np.nanmean(roi[:, :, :n_pre], axis=(0, 2))

    def pct_change(series: np.ndarray) -> float:
        head, tail = float(series[0]), float(series[-1])
        return float("nan") if abs(head) < 1e-9 else 100.0 * (tail - head) / head

    return {
        "within_trial_pre_pct": round(pct_change(within), 2),
        "across_session_pct": round(pct_change(across), 2),
        "baseline_f_first_trial": round(float(across[0]), 1),
        "baseline_f_last_trial": round(float(across[-1]), 1),
    }


def response_qc(
    path: str | Path,
    *,
    baseline_s: float = BASELINE_S,
    thresholds: str = "calibrated",
    target_fpr: float = TARGET_FPR,
    per_roi_thresholds: bool = False,
    z_excited: float = Z_EXCITED,
    z_suppressed: float = Z_SUPPRESSED,
    by_state: bool = True,
    save: bool = True,
) -> dict:
    """
    Sparsity and health metrics for one processed round.

    `thresholds="calibrated"` (the default) reads the two z thresholds off
    this session's own null at `target_fpr` per tail -- see `sham_null`.
    `"fixed"` uses `z_excited`/`z_suppressed` as given, which is what to pass
    when several sessions must share one threshold rather than one error rate.

    `per_roi_thresholds` calibrates each ROI separately instead of pooling.
    More accurate per ROI, but the reported threshold stops being a single
    number, so it is off by default.

    Writes `<stem>_responseqc.png` and `.json` beside the round when `save`,
    and returns the metrics. `by_state` also reports each block separately,
    which is the comparison the manipulation is about.
    """

    if thresholds not in ("calibrated", "fixed"):
        raise ValueError(
            f"thresholds must be 'calibrated' or 'fixed', got {thresholds!r}."
        )

    from .h5io import open_h5

    path = Path(path)

    with open_h5(path) as f:
        roi = f["traces/roi"][:]
        odor_ids = f["trials/odor_id"][:]
        states = f["trials/state"][:]
        state_levels = [
            s.decode() if isinstance(s, bytes) else str(s)
            for s in f["trials/state_levels"][:]
        ]
        on_frames = f["trials/odor_on_frame"][:]
        off_frames = f["trials/odor_off_frame"][:]
        areas = f["rois/area_px"][:]
        n_pre = int(f.attrs["n_pre"])
        frame_rate = float(f.attrs["frame_rate"])
        exp_name = str(f.attrs["exp_name"])
        mask_digest = str(f.attrs["mask_hash"])

    resp = trial_responses(
        roi, odor_on_frames=on_frames, odor_off_frames=off_frames,
        frame_rate=frame_rate, baseline_s=baseline_s,
    )

    keys = np.unique(odor_ids)

    null = sham_null(
        roi, odor_on_frames=on_frames, odor_off_frames=off_frames,
        odor_ids=odor_ids, frame_rate=frame_rate, baseline_s=baseline_s,
        target_fpr=target_fpr,
    )

    # Thresholds come off the null unless asked for fixed ones, or unless the
    # null could not be built -- in which case the constants are used and the
    # report says so rather than silently reporting a calibrated rate.
    threshold_source = thresholds
    per_roi_used = False

    if thresholds == "calibrated" and null["available"]:
        if per_roi_thresholds and null["per_roi"]["reliable"]:
            up = null["per_roi"]["z_excited"][:, None]
            down = null["per_roi"]["z_suppressed"][:, None]
            per_roi_used = True
            threshold_source = "calibrated per ROI"
        else:
            up, down = null["z_excited"], null["z_suppressed"]
            if per_roi_thresholds:
                threshold_source = "calibrated pooled (per-ROI unreliable)"
    else:
        if thresholds == "calibrated":
            threshold_source = "fixed (null unavailable)"
        up, down = z_excited, z_suppressed

    def block(selection: np.ndarray, name: str) -> dict:
        dff = _by_odor(resp["dff"][:, selection], odor_ids[selection], keys)
        z = _by_odor(resp["z"][:, selection], odor_ids[selection], keys)

        excited = z >= up
        suppressed = z <= down
        responsive = excited | suppressed

        # Sparseness on dF/F, not z: z is scaled by each ROI's own baseline
        # noise, so a quiet ROI's small response and a noisy ROI's large one
        # can land at the same z, and sparseness would be reading the noise.
        lifetime = treves_rolls(dff, axis=1)      # per ROI, across odors
        population = treves_rolls(dff, axis=0)    # per odor, across ROIs

        return {
            "name": name,
            "n_trials": int(selection.sum()),
            "lifetime_sparseness": {
                "median": _round(np.nanmedian(lifetime)),
                "iqr": [_round(np.nanpercentile(lifetime, 25)),
                        _round(np.nanpercentile(lifetime, 75))],
                "per_roi": [_round(v) for v in lifetime],
            },
            "population_sparseness": {
                "median": _round(np.nanmedian(population)),
                "per_odor": {int(k): _round(v) for k, v in zip(keys, population)},
            },
            "responsive": {
                "z_excited": _threshold_value(up),
                "z_suppressed": _threshold_value(down),
                "threshold_source": threshold_source,
                # Count-based population sparsity: of all ROIs, how many did
                # this odor drive.
                "fraction_of_rois_per_odor": {
                    int(k): _round(responsive[:, i].mean())
                    for i, k in enumerate(keys)
                },
                "fraction_excited_per_odor": {
                    int(k): _round(excited[:, i].mean()) for i, k in enumerate(keys)
                },
                "fraction_suppressed_per_odor": {
                    int(k): _round(suppressed[:, i].mean()) for i, k in enumerate(keys)
                },
                # Count-based lifetime sparsity: how many odors drove each ROI.
                "odors_per_roi_median": _round(np.median(responsive.sum(axis=1))),
                "rois_responsive_to_any": int((responsive.any(axis=1)).sum()),
                "rois_responsive_to_all": int((responsive.all(axis=1)).sum()),
                "n_rois": int(roi.shape[0]),
                "excited": int(excited.any(axis=1).sum()),
                "suppressed": int(suppressed.any(axis=1).sum()),
                # A unit pushed below the threshold by every odor alike is
                # almost certainly reading the within-trial bleaching rather
                # than any odor. Kept separate rather than subtracted, since
                # a genuinely broadly-suppressed glomerulus looks the same.
                "suppressed_to_all_odors": int(suppressed.all(axis=1).sum()),
            },
            "dff": {
                "median": _round(np.nanmedian(dff)),
                "p95": _round(np.nanpercentile(dff, 95)),
                "p05": _round(np.nanpercentile(dff, 5)),
            },
            "_matrices": {"dff": dff, "z": z, "responsive": responsive,
                          "excited": excited, "suppressed": suppressed,
                          "lifetime": lifetime, "population": population},
        }

    blocks = [block(np.ones(len(odor_ids), dtype=bool), "pooled")]

    if by_state:
        for code, name in enumerate(state_levels):
            selection = states == code
            if selection.sum() >= len(keys):
                blocks.append(block(selection, name))

    report = {
        "file": path.name,
        "exp_name": exp_name,
        "mask_hash": mask_digest,
        "n_rois": int(roi.shape[0]),
        "n_odors": int(len(keys)),
        "odor_ids": [int(k) for k in keys],
        "baseline_s": baseline_s,
        "baseline_frames": resp["n_baseline_frames"],
        "roi_area_px": [int(np.min(areas)), int(np.max(areas))],
        "rois_under_10px": int((areas < 10).sum()),
        "bleaching": bleaching(roi, on_frames, n_pre=n_pre),
        # Per-ROI arrays are dropped from the report: they are n_roi long and
        # would swamp the JSON, and the range already says what they spread.
        "sham": {k: v for k, v in null.items() if k != "per_roi"},
        "sham_per_roi_range": null.get("per_roi", {}).get("excited_range"),
        "sham_samples_per_roi": null.get("per_roi", {}).get("samples_per_roi"),
        "thresholds": {
            "source": threshold_source,
            "per_roi": per_roi_used,
            "per_roi_requested": bool(per_roi_thresholds),
            "z_excited": _threshold_value(up),
            "z_suppressed": _threshold_value(down),
        },
        "blocks": {b["name"]: {k: v for k, v in b.items() if k != "_matrices"}
                   for b in blocks},
    }

    report["flags"] = _flags(report, blocks[0])

    if save:
        stem = path.with_suffix("")
        (Path(f"{stem}_responseqc.json")).write_text(
            json.dumps(report, indent=2, default=str)
        )
        report["figure"] = str(
            _figure(
                f"{stem}_responseqc.png", blocks, keys, report,
                trials={"dff": resp["dff"], "odor_ids": odor_ids,
                        "states": states, "state_levels": state_levels},
            )
        )
        report["json"] = f"{stem}_responseqc.json"

    return report


def _null_line(report: dict) -> str:
    """One line saying where the thresholds came from and whether they hold."""

    sham = report["sham"]

    if not sham.get("available"):
        return f"  (not calibrated: {sham.get('reason', 'unknown')})"

    achieved = sham["achieved_fpr"]

    return (
        f"  from {sham['n_null_samples']:,} null samples at "
        f"{sham['target_fpr']:.0%}/tail; achieved "
        f"{_pct(achieved['excited']).strip()} / "
        f"{_pct(achieved['suppressed']).strip()}"
    )


def _threshold_value(threshold):
    """A scalar threshold as itself; a per-ROI one as its median."""
    array = np.asarray(threshold)
    return float(array) if array.ndim == 0 else float(np.nanmedian(array))


def _pct(value) -> str:
    return "  n/a" if value is None else f"{value:5.1%}"


def _round(value, places: int = 4):
    value = float(value)
    return None if not np.isfinite(value) else round(value, places)


def _flags(report: dict, pooled: dict) -> list[str]:
    """Plain statements of what looks wrong, or an empty list."""

    out = []
    responsive = pooled["responsive"]
    fractions = list(responsive["fraction_of_rois_per_odor"].values())
    fractions = [v for v in fractions if v is not None]

    if fractions and max(fractions) > 0.8:
        out.append(
            f"Population sparsity is very low: one odor drove "
            f"{max(fractions):.0%} of ROIs. In the bulb that usually means "
            f"motion, a z-shift, or a baseline taken over a bleaching window "
            f"rather than a real response."
        )

    if responsive["rois_responsive_to_any"] < 0.1 * responsive["n_rois"]:
        out.append(
            f"Only {responsive['rois_responsive_to_any']} of "
            f"{responsive['n_rois']} ROIs respond to any odor "
            f"(z >= {responsive['z_excited']:g} or "
            f"z <= {responsive['z_suppressed']:g}). Check alignment and the "
            f"mask before reading anything into the sparsity numbers."
        )

    if responsive["suppressed"] and (
        responsive["suppressed_to_all_odors"] > 0.5 * responsive["suppressed"]
    ):
        out.append(
            f"{responsive['suppressed_to_all_odors']} of "
            f"{responsive['suppressed']} suppressed ROIs are suppressed by "
            f"every odor alike, which is the signature of the within-trial "
            f"decay rather than odor-driven suppression. Treat the suppressed "
            f"count as an upper bound."
        )

    lifetime = pooled["lifetime_sparseness"]["median"]
    if lifetime is not None and lifetime < 0.1:
        out.append(
            f"Median lifetime sparseness is {lifetime:.2f}: ROIs respond about "
            f"equally to every odor. Suspect ROIs large enough to average "
            f"several glomeruli, or a mask that has drifted."
        )

    tiny = report.get("rois_under_10px", 0)
    if tiny:
        out.append(
            f"{tiny} ROI(s) are under 10 px. A footprint that small is not a "
            f"glomerulus; its trace is a single pixel's noise wearing the same "
            f"shape as a real one. Re-curate and re-extract."
        )

    sham = report.get("sham", {})
    knobs = report.get("thresholds", {})

    if knobs.get("per_roi_requested") and not knobs.get("per_roi"):
        out.append(
            f"Per-ROI thresholds were requested but the null has only "
            f"{report.get('sham_samples_per_roi', 'too few')} samples per ROI "
            f"to place a {sham.get('target_fpr', 0):.0%} percentile, so the "
            f"pooled threshold was used instead. Per-ROI needs a longer "
            f"pre-odor period or more odors."
        )

    if not sham.get("available"):
        out.append(
            f"Thresholds could not be calibrated ({sham.get('reason', 'unknown')}), "
            f"so the fixed fallbacks were used. Their false-positive rate on "
            f"this session is unknown."
        )
    else:
        # Calibrated on one set of sham offsets, checked on another. A gap
        # means the null does not generalise across the pre-odor period --
        # usually because it is not flat, so no single threshold describes it.
        for sign in ("excited", "suppressed"):
            achieved = sham["achieved_fpr"][sign]
            target = sham["target_fpr"]

            if achieved is not None and achieved > 3 * target:
                out.append(
                    f"The {sign} threshold was calibrated for a {target:.0%} "
                    f"false-positive rate but lands at {achieved:.1%} on "
                    f"held-out sham windows. The pre-odor period is not "
                    f"stationary, so treat the {sign} counts as an upper bound."
                )

        observed = [
            v for v in pooled["responsive"]["fraction_of_rois_per_odor"].values()
            if v is not None
        ]
        if observed and max(observed) < 4 * sham["target_fpr"]:
            out.append(
                f"No odor drives more than {max(observed):.1%} of ROIs, barely "
                f"above the {sham['target_fpr']:.0%} the thresholds allow by "
                f"construction. This session has close to no detectable "
                f"response."
            )

    bleach = report["bleaching"]
    if bleach["within_trial_pre_pct"] < -5:
        out.append(
            f"Fluorescence falls {abs(bleach['within_trial_pre_pct']):.0f}% "
            f"across the pre-odor window within a trial. F0 is taken from the "
            f"last {report['baseline_s']} s to avoid biasing dF/F negative; "
            f"anything computed off the whole pre-period will be."
        )

    if bleach["across_session_pct"] < -25:
        out.append(
            f"Baseline F falls {abs(bleach['across_session_pct']):.0f}% from "
            f"the first acquisition to the last. Compare pre and post blocks "
            f"with care -- some of any difference between them is this."
        )

    return out


def _figure(
    path: str | Path,
    blocks: list[dict],
    keys: np.ndarray,
    report: dict,
    *,
    trials: None | dict = None,
):
    """
    Seven panels, led by the single-trial response matrix.

    The trial-resolved heatmap goes first and gets the full width because it
    is the only panel where trial-to-trial reliability is visible. An odor
    average of eight trials looks identical whether the response happened
    every time or once at four times the amplitude, and those are different
    findings; the average is kept below it for reading tuning structure, which
    is easier without 180 columns.
    """

    import matplotlib
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    pooled = blocks[0]
    dff = pooled["_matrices"]["dff"]
    lifetime = pooled["_matrices"]["lifetime"]
    population = pooled["_matrices"]["population"]
    responsive = pooled["_matrices"]["responsive"]
    z_up = pooled["responsive"]["z_excited"]
    z_down = pooled["responsive"]["z_suppressed"]

    # ROI order, shared by both heatmaps so rows mean the same thing in each:
    # by preferred odor, then by response amplitude within it.
    best = np.nanargmax(np.abs(np.nan_to_num(dff)), axis=1)
    peak = np.nanmax(np.abs(np.nan_to_num(dff)), axis=1)
    order = np.lexsort((-peak, best))

    limit = float(np.nanpercentile(np.abs(dff), 98)) or 1.0

    fig = plt.figure(figsize=(17, 14))
    grid = fig.add_gridspec(3, 3, height_ratios=[1.35, 1, 1], hspace=0.35, wspace=0.28)

    # 1. Every trial, grouped by odor and then by block. Reliability, drift and
    # single-trial outliers all live here and nowhere else in this figure.
    ax_trials = fig.add_subplot(grid[0, :])

    if trials is not None:
        _trial_heatmap(
            fig, ax_trials, trials, order=order, keys=keys, limit=limit
        )
    else:
        ax_trials.axis("off")

    # 2. The odor average of the panel above.
    ax = fig.add_subplot(grid[1, 0])
    image = ax.imshow(
        dff[order], aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit,
        interpolation="nearest",
    )
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([str(int(k)) for k in keys])
    ax.set_xlabel("odor id"); ax.set_ylabel("ROI (same order as above)")
    ax.set_title("trial-averaged dF/F")
    fig.colorbar(image, ax=ax, label="dF/F")

    # 3. Lifetime sparseness across glomeruli.
    ax = fig.add_subplot(grid[1, 1])
    ax.hist(lifetime[np.isfinite(lifetime)], bins=20, range=(0, 1), color="0.35")
    median = np.nanmedian(lifetime)
    ax.axvline(median, color="crimson", ls="--", label=f"median {median:.2f}")
    ax.set_xlabel("lifetime sparseness (0 flat, 1 one odor)")
    ax.set_ylabel("glomeruli")
    ax.set_title("how selective each glomerulus is")
    ax.legend(fontsize=8)

    # 4. Population sparseness per odor, with the count version beside it,
    # split by sign because the two thresholds differ.
    ax = fig.add_subplot(grid[1, 2])
    excited = pooled["_matrices"]["excited"].mean(axis=0)
    suppressed = pooled["_matrices"]["suppressed"].mean(axis=0)
    x = np.arange(len(keys))
    ax.bar(x - 0.22, population, 0.3, color="0.35", label="Treves-Rolls")
    ax.bar(x + 0.16, excited, 0.3, color="indianred", label=f"excited z>={z_up:g}")
    ax.bar(x + 0.16, suppressed, 0.3, bottom=excited, color="steelblue",
           label=f"suppressed z<={z_down:g}")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(k)) for k in keys])
    ax.set_ylim(0, 1)
    ax.set_xlabel("odor id"); ax.set_ylabel("sparseness / fraction of ROIs")
    ax.set_title("how selective each odor is")
    ax.legend(fontsize=8)

    # 5. Tuning breadth: the count form of lifetime sparsity.
    ax = fig.add_subplot(grid[2, 0])
    counts = responsive.sum(axis=1)
    ax.hist(counts, bins=np.arange(len(keys) + 2) - 0.5, color="0.35")
    ax.set_xlabel(f"odors driving an ROI (z>={z_up:g} or z<={z_down:g})")
    ax.set_ylabel("glomeruli")
    ax.set_title(f"{int((counts == 0).sum())} of {len(counts)} respond to nothing")

    # 6. Pre vs post, if both blocks are present.
    ax = fig.add_subplot(grid[2, 1])
    if len(blocks) > 1:
        for b, colour in zip(blocks[1:], ("steelblue", "indianred")):
            values = b["_matrices"]["lifetime"]
            ax.hist(values[np.isfinite(values)], bins=20, range=(0, 1),
                    histtype="step", lw=2, color=colour, label=b["name"])
        ax.set_xlabel("lifetime sparseness")
        ax.set_ylabel("glomeruli")
        ax.set_title("selectivity by block")
        ax.legend(fontsize=8)
    else:
        ax.axis("off")

    # 7. Bleaching and flags -- what quietly corrupts all of the above.
    bleach = report["bleaching"]
    counts_pooled = pooled["responsive"]
    ax = fig.add_subplot(grid[2, 2])
    ax.axis("off")
    ax.text(
        0.0, 1.0,
        "\n".join([
            f"{report['exp_name']}   mask {report['mask_hash']}",
            f"{report['n_rois']} ROIs x {report['n_odors']} odors",
            f"F0 window: last {report['baseline_s']} s "
            f"({report['baseline_frames']} frames)",
            "",
            f"thresholds ({report['thresholds']['source']}):",
            f"  excited    z >= {z_up:+.3f}",
            f"  suppressed z <= {z_down:+.3f}",
            _null_line(report),
            "",
            f"excited    {counts_pooled['excited']:3d}",
            f"suppressed {counts_pooled['suppressed']:3d}"
            f"   ({counts_pooled['suppressed_to_all_odors']} to every odor)",
            "",
            f"within-trial pre decay  {bleach['within_trial_pre_pct']:+.1f}%",
            f"across-session baseline {bleach['across_session_pct']:+.1f}%",
            f"  {bleach['baseline_f_first_trial']:.0f} -> "
            f"{bleach['baseline_f_last_trial']:.0f} a.u.",
            "",
            "FLAGS" if report["flags"] else "no flags",
        ] + [f"- {_wrap(flag)}" for flag in report["flags"]]),
        va="top", ha="left", family="monospace", fontsize=8.5,
        transform=ax.transAxes,
    )

    fig.suptitle(f"response QC - {report['file']}", y=0.995, fontsize=13)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)

    return Path(path)


def _trial_heatmap(fig, ax, trials: dict, *, order, keys, limit) -> None:
    """
    Single-trial dF/F, columns grouped by odor and split by block.

    Trials are sorted rather than left in acquisition order: chronological
    columns interleave the odors and nothing is legible. Sorting by odor, then
    block, then time keeps each odor's trials adjacent while preserving the
    within-block time axis, so a response that fades over a block still reads
    as a gradient rather than as noise.
    """

    dff = trials["dff"]
    odor_ids = np.asarray(trials["odor_ids"])
    states = np.asarray(trials["states"])
    levels = trials["state_levels"]

    columns = np.lexsort((np.arange(len(odor_ids)), states, odor_ids))

    image = ax.imshow(
        dff[np.ix_(order, columns)], aspect="auto", cmap="RdBu_r",
        vmin=-limit, vmax=limit, interpolation="nearest",
    )
    fig.colorbar(image, ax=ax, label="dF/F", pad=0.01)

    sorted_odors = odor_ids[columns]
    sorted_states = states[columns]

    # Solid rules between odors, dotted between the two blocks within an odor.
    boundaries = np.flatnonzero(np.diff(sorted_odors)) + 1
    for edge in boundaries:
        ax.axvline(edge - 0.5, color="black", lw=1.2)

    state_edges = np.flatnonzero(
        (np.diff(sorted_states) != 0) & (np.diff(sorted_odors) == 0)
    ) + 1
    for edge in state_edges:
        ax.axvline(edge - 0.5, color="0.4", lw=0.8, ls=":")

    centres = [
        float(np.mean(np.flatnonzero(sorted_odors == k))) for k in keys
    ]
    ax.set_xticks(centres)
    ax.set_xticklabels([str(int(k)) for k in keys])
    ax.set_xlabel(
        f"trial, grouped by odor id  (solid = odor, dotted = "
        f"{' | '.join(levels)} within an odor)"
    )
    ax.set_ylabel("ROI (sorted by preferred odor)")
    ax.set_title("single-trial dF/F")


def _wrap(text: str, width: int = 52) -> str:
    import textwrap

    return "\n  ".join(textwrap.wrap(text, width))
