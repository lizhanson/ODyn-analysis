"""QC on the extracted traces: does this session look like odor coding?"""

from __future__ import annotations

import json

from datetime import date
from pathlib import Path

import numpy as np

# Seconds before odor onset used for F0. Short on purpose -- see the module
# docstring on within-trial bleaching.
BASELINE_S = 1.0

SHAM_RESPONSE_S = 2.0
SHAM_BASELINE_S = 2.0

Z_EXCITED = 1.5
Z_SUPPRESSED = -0.5

# Share of ROI-odor pairs the null is allowed to put past each threshold. Per
# tail, so the two sides are calibrated independently and come out asymmetric
# on their own rather than by assumption.
TARGET_FPR = 0.01

HEATMAP_Z_MIN = -2.0
HEATMAP_Z_MAX = 10.0

SHAM_STEP_FRACTION = 0.2


def treves_rolls(responses: np.ndarray, axis: int = -1) -> np.ndarray:
    """Sparseness of a rectified response vector, 0 (flat) to 1 (one-of-N)."""

    r = np.abs(np.asarray(responses, dtype=np.float64))
    n = r.shape[axis]

    if n < 2:
        return np.full(np.delete(r.shape, axis), np.nan)

    mean = np.nanmean(r, axis=axis)
    mean_sq = np.nanmean(r**2, axis=axis)

    with np.errstate(invalid="ignore", divide="ignore"):
        raw = 1.0 - (mean**2) / mean_sq

    return np.where(mean_sq > 0, raw / (1.0 - 1.0 / n), np.nan)


SD_BASELINE_S = None

POOL_SD = True


def trial_responses(
    roi: np.ndarray,
    *,
    odor_on_frames: np.ndarray,
    odor_off_frames: np.ndarray,
    frame_rate: float,
    baseline_s: float = BASELINE_S,
    sd_baseline_s: float | None = SD_BASELINE_S,
    pool_sd: bool = POOL_SD,
) -> dict:
    """Per-ROI, per-trial response amplitude, aligned on each trial's own onset."""

    n_roi, n_trial, n_frame = roi.shape
    n_base = max(1, int(round(baseline_s * frame_rate)))
    n_sd = (None if sd_baseline_s is None
            else max(1, int(round(sd_baseline_s * frame_rate))))

    dff = np.full((n_roi, n_trial), np.nan)
    z = np.full((n_roi, n_trial), np.nan)
    f0 = np.full((n_roi, n_trial), np.nan)

    numerator = np.full((n_roi, n_trial), np.nan)
    per_trial_sd = np.full((n_roi, n_trial), np.nan)
    window_frames = np.zeros(n_trial, dtype=int)

    # Residuals about each trial's own mean, so the pooled estimate below is a
    # within-trial spread and not that plus the drift in level between trials.
    residuals = []

    for trial in range(n_trial):
        on = int(odor_on_frames[trial])
        off = int(odor_off_frames[trial])

        if on <= 0 or off <= on or off > n_frame:
            continue

        base = roi[:, trial, max(on - n_base, 0):on]
        resp = roi[:, trial, on:off]

        if base.shape[1] == 0 or resp.shape[1] == 0:
            continue

        window = (roi[:, trial, :on] if n_sd is None
                  else roi[:, trial, max(on - n_sd, 0):on])

        if window.shape[1] >= 2:
            per_trial_sd[:, trial] = np.nanstd(window, axis=1)
            window_frames[trial] = window.shape[1]
            residuals.append(window - np.nanmean(window, axis=1, keepdims=True))

        base_mean = np.nanmean(base, axis=1)
        base_sd = per_trial_sd[:, trial]
        resp_mean = np.nanmean(resp, axis=1)
        numerator[:, trial] = resp_mean - base_mean

        f0[:, trial] = base_mean

        with np.errstate(invalid="ignore", divide="ignore"):
            dff[:, trial] = (resp_mean - base_mean) / np.where(
                np.abs(base_mean) > 1e-9, np.abs(base_mean), np.nan
            )

    if pool_sd and residuals:
        # One SD per ROI over every baseline frame in the session. ddof is not
        # worth chasing: the sample is thousands of frames per ROI.
        pooled = np.sqrt(np.nanmean(
            np.concatenate(residuals, axis=1) ** 2, axis=1
        ))
        sd = np.repeat(pooled[:, None], n_trial, axis=1)
    else:
        sd = per_trial_sd

    with np.errstate(invalid="ignore", divide="ignore"):
        z = numerator / np.where(sd > 1e-9, sd, np.nan)

    n_window = (int(np.median(window_frames[window_frames > 0]))
                if (window_frames > 0).any() else 0)
    pooled_used = bool(pool_sd and residuals)

    return {
        "dff": dff,
        "z": z,
        "f0": f0,
        "n_baseline_frames": n_base,
        "n_sd_frames_per_trial": n_window,
        "n_sd_frames_total": n_window * len(residuals) if pooled_used else n_window,
        "sd_pooled": pooled_used,
        "baseline_sd": sd[:, 0] if pooled_used else per_trial_sd,
    }


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
    """Build this session's null for the response statistic, and read the thresholds off it."""

    on = np.asarray(odor_on_frames)
    off = np.asarray(odor_off_frames)
    odor_ids = np.asarray(odor_ids)

    # Sham geometry is its own, not the real window's -- see SHAM_RESPONSE_S.
    span = max(1, int(round(SHAM_RESPONSE_S * frame_rate)))
    n_base = max(1, int(round(SHAM_BASELINE_S * frame_rate)))

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
        "window_s": {"baseline": SHAM_BASELINE_S, "response": SHAM_RESPONSE_S},
        "real_response_frames": int(np.median(off - on)),
        "n_null_samples": int(fit_values.size),
        "z_excited": round(z_up, 4),
        "z_suppressed": round(z_down, 4),
        "per_roi": {
            "z_excited": per_roi_excited,
            "z_suppressed": per_roi_suppressed,
            "samples_per_roi": int(fit.shape[1]),
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



BASELINE_OUTLIER_SD = 5.0


def baseline_outliers(
    roi: np.ndarray,
    *,
    odor_on_frames: np.ndarray,
    n_base: int,
    threshold_sd: float = BASELINE_OUTLIER_SD,
) -> dict:
    """Find trials whose baseline fluorescence is implausible for this session."""

    n_trial = roi.shape[1]
    f0 = np.array([
        float(np.nanmean(roi[:, k, max(int(odor_on_frames[k]) - n_base, 0):
                              int(odor_on_frames[k])]))
        for k in range(n_trial)
    ])

    keep = np.isfinite(f0)
    for _ in range(n_trial):
        mean, sd = np.nanmean(f0[keep]), np.nanstd(f0[keep])
        if not np.isfinite(sd) or sd <= 0:
            break
        deviation = np.abs(f0 - mean) / sd
        updated = keep & (deviation <= threshold_sd)
        if updated.sum() == keep.sum():
            break
        keep = updated

    mean, sd = np.nanmean(f0[keep]), np.nanstd(f0[keep])
    dropped = np.flatnonzero(~keep)

    return {
        "keep": keep,
        "f0": f0,
        "session_mean": float(mean),
        "session_sd": float(sd),
        "threshold_sd": float(threshold_sd),
        "n_dropped": int(len(dropped)),
        "dropped": [
            {"trial_index": int(i), "f0": round(float(f0[i]), 2),
             "sd_from_mean": round(float(abs(f0[i] - mean) / sd), 1)
             if sd > 0 else None}
            for i in dropped
        ],
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
    """Two decay measures, within a trial and across the session."""

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
    baseline_outlier_sd: float = BASELINE_OUTLIER_SD,
    per_trial: bool = True,
    deglobal: str | None = "pc1",
    thresholds: str = "calibrated",
    target_fpr: float = TARGET_FPR,
    per_roi_thresholds: bool = False,
    z_excited: float = Z_EXCITED,
    z_suppressed: float = Z_SUPPRESSED,
    by_state: bool = True,
    save: bool = True,
) -> dict:
    """Sparsity and health metrics for one processed round."""

    if per_trial:
        # The per-trial path supersedes the sham null for deciding responsive.
        # See `responders` for why: the sham null's resolution is capped by the
        # offset count, and this one is not capped at all.
        return _per_trial_report(
            path, deglobal=deglobal, save=save,
            baseline_outlier_sd=baseline_outlier_sd,
        )

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

    # Drop trials whose baseline is implausible before anything is summarised.
    # One dead acquisition otherwise sets the colour scale, dominates every
    # odor average it lands in, and makes all ROIs look responsive.
    guard = baseline_outliers(
        roi, odor_on_frames=on_frames, n_base=resp["n_baseline_frames"],
        threshold_sd=baseline_outlier_sd,
    )
    usable = guard["keep"]

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

    blocks = [block(usable.copy(), "pooled")]

    if by_state:
        for code, name in enumerate(state_levels):
            selection = (states == code) & usable
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
        "excluded_trials": {
            "n": guard["n_dropped"],
            "threshold_sd": guard["threshold_sd"],
            "session_mean_f0": round(guard["session_mean"], 2),
            "session_sd_f0": round(guard["session_sd"], 2),
            "trials": guard["dropped"],
            "applied": True,
        },
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
                trials={"dff": resp["z"], "odor_ids": odor_ids,
                        "states": states, "state_levels": state_levels,
                        "usable": usable},
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
            f"last {report['baseline_s']} s before each trial's own odor "
            f"onset to avoid biasing dF/F negative; "
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
    """Seven panels, led by the single-trial response matrix."""

    import matplotlib
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    pooled = blocks[0]
    # Heatmaps show baseline-z, not dF/F: z is already normalised per ROI by
    # its own baseline noise, so a fixed +/-5 window means the same thing for
    # a dim ROI and a bright one.
    dff = pooled["_matrices"]["z"]
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

    norm = TwoSlopeNorm(vmin=HEATMAP_Z_MIN, vcenter=0.0, vmax=HEATMAP_Z_MAX)

    fig = plt.figure(figsize=(17, 14))
    grid = fig.add_gridspec(3, 3, height_ratios=[1.35, 1, 1], hspace=0.35, wspace=0.28)

    # 1. Every trial, grouped by odor and then by block. Reliability, drift and
    # single-trial outliers all live here and nowhere else in this figure.
    ax_trials = fig.add_subplot(grid[0, :])

    if trials is not None:
        _trial_heatmap(
            fig, ax_trials, trials, order=order, keys=keys, norm=norm
        )
    else:
        ax_trials.axis("off")

    # 2. The odor average of the panel above.
    ax = fig.add_subplot(grid[1, 0])
    image = ax.imshow(
        dff[order], aspect="auto", cmap="RdBu_r", norm=norm,
        interpolation="nearest",
    )
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([str(int(k)) for k in keys])
    ax.set_xlabel("odor id"); ax.set_ylabel("ROI (same order as above)")
    ax.set_title("trial-averaged z")
    fig.colorbar(image, ax=ax, label="baseline z")

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
            (f"excluded {report['excluded_trials']['n']} trial(s) "
             f"at >{report['excluded_trials']['threshold_sd']:g} SD from mean F0"
             if report["excluded_trials"]["n"] else "no trials excluded"),
            *[f"   trial {d['trial_index'] + 1}: F0 {d['f0']:.0f} "
              f"({d['sd_from_mean']:.0f} SD)"
              for d in report["excluded_trials"]["trials"][:4]],
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


def _trial_heatmap(fig, ax, trials: dict, *, order, keys, norm) -> None:
    """Single-trial dF/F, columns grouped by odor and split by block."""

    dff = trials["dff"]
    odor_ids = np.asarray(trials["odor_ids"])
    states = np.asarray(trials["states"])
    levels = trials["state_levels"]

    usable = trials.get("usable")
    order_all = np.lexsort((np.arange(len(odor_ids)), states, odor_ids))
    columns = (order_all if usable is None
               else np.array([c for c in order_all if usable[c]]))

    image = ax.imshow(
        dff[np.ix_(order, columns)], aspect="auto", cmap="RdBu_r",
        norm=norm, interpolation="nearest",
    )
    fig.colorbar(image, ax=ax, label="baseline z", pad=0.01)

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
    ax.set_title("single-trial z")


def _wrap(text: str, width: int = 52) -> str:
    import textwrap

    return "\n  ".join(textwrap.wrap(text, width))


def _per_trial_report(
    path,
    *,
    deglobal: str | None = "pc1",
    baseline_outlier_sd: float = BASELINE_OUTLIER_SD,
    save: bool = True,
) -> dict:
    """The QC report with responsiveness decided per trial rather than per pair."""

    from .h5io import open_h5
    from .responders import response_figure

    path = Path(path)

    with open_h5(path) as f:
        roi = f["traces/roi"][:]
        on_frames = f["trials/odor_on_frame"][:]
        areas = f["rois/area_px"][:]
        n_pre = int(f.attrs["n_pre"])
        frame_rate = float(f.attrs["frame_rate"])
        exp_name = str(f.attrs["exp_name"])
        mask_digest = str(f.attrs["mask_hash"])
        trial_ids = (f["trials/trial_id"][:] if "trials/trial_id" in f
                     else np.arange(roi.shape[1]))
        acq_ids = f["trials/acq_id"][:] if "trials/acq_id" in f else None
        mcor = _trial_mcor_paths(f)

    stem = path.with_suffix("")
    n_base = max(1, int(round(BASELINE_S * frame_rate)))

    guard = baseline_outliers(
        roi, odor_on_frames=on_frames, n_base=n_base,
        threshold_sd=baseline_outlier_sd,
    )

    # Actually exclude them. The counts were reported here from the start but
    # the mask was never passed on, so the JSON claimed exclusions that had
    # not happened -- worse than no guard, since it reads as protection.
    figure = response_figure(
        path, deglobal=deglobal, save=save, usable=guard["keep"],
        out_path=f"{stem}_responseqc.png",
    )

    calls, sparsity = figure["calls"], figure["sparsity"]
    check, levels = figure["split_half"], figure["levels"]

    def block(name: str) -> dict:
        s = sparsity[name]
        return {
            "excited_per_odor": {int(k): _round(v) for k, v in
                                 zip(s["odor_ids"], s["excited"]["mean"])},
            "excited_sem_per_odor": {int(k): _round(v) for k, v in
                                     zip(s["odor_ids"], s["excited"]["sem"])},
            "suppressed_per_odor": {int(k): _round(v) for k, v in
                                    zip(s["odor_ids"], s["suppressed"]["mean"])},
            "suppressed_sem_per_odor": {int(k): _round(v) for k, v in
                                        zip(s["odor_ids"], s["suppressed"]["sem"])},
            "excited_median": _round(np.nanmedian(s["excited"]["mean"])),
            "suppressed_median": _round(np.nanmedian(s["suppressed"]["mean"])),
            "lifetime_sparseness_median": _round(np.nanmedian(figure["lifetime"][name])),
            "n_trials": int(s["excited"]["n_trials"].sum()),
        }

    report = {
        "file": path.name,
        "exp_name": exp_name,
        "mask_hash": mask_digest,
        "n_rois": int(roi.shape[0]),
        "n_odors": len(sparsity["pooled"]["odor_ids"]),
        "odor_ids": sparsity["pooled"]["odor_ids"],
        "method": "per-trial",
        "trace_source": figure["trace_source"],
        # None on rounds written before the field existed. Grouping sessions
        # by this is what separates the manipulation from time in session.
        "manipulation": figure["manipulation"],
        "deglobal": deglobal,
        "window_frames": calls["window_frames"],
        "roi_area_px": [int(np.min(areas)), int(np.max(areas))],
        "rois_under_10px": int((areas < 10).sum()),
        # Counts only. Which trials, and which files to go and look at, is a
        # thing to act on rather than a table to carry: it goes in `flags`.
        "excluded_trials": {
            "n": guard["n_dropped"],
            "threshold_sd": guard["threshold_sd"],
            "n_trials_used": figure["n_trials_used"],
        },
        "bleaching": bleaching(roi, on_frames, n_pre=n_pre),
        "thresholds": {
            "target_fdr": calls["target_fdr"],
            "z_excited": _round(calls["z_excited_threshold"]),
            "z_suppressed": _round(calls["z_suppressed_threshold"]),
            "estimated_fdr_excited": calls["thresholds"]["excited"]["estimated_fdr"],
            "estimated_fdr_suppressed": calls["thresholds"]["suppressed"]["estimated_fdr"],
        },
        "chance": sparsity["pooled"]["chance"],
        # Rerun of the identical test where no odor was delivered. Not a
        # correction -- if this is far from zero the counts are not usable.
        "control": {
            "false_positive_rate": check["false_positive_rate"],
            "kurtosis": check["kurtosis"],
            "n_tests": check["n_tests"],
        },
        "blocks": {name: block(name) for name in ["pooled", *levels]},
        "block_order": levels,
    }

    report["flags"] = _per_trial_flags(
        report, dropped=guard["dropped"], trial_ids=trial_ids,
        acq_ids=acq_ids, mcor=mcor,
    )

    if save:
        Path(f"{stem}_responseqc.json").write_text(
            json.dumps(report, indent=2, default=str)
        )
        report["json"] = f"{stem}_responseqc.json"
        report["figure"] = figure["figure"]

    return report


def _trial_mcor_paths(f):
    """Per-trial motion-corrected filenames, or None on rounds without them."""

    if "trials/mcor_path" not in f:
        return None

    codes = f["trials/mcor_path"][:]
    levels = [s.decode() if isinstance(s, bytes) else str(s)
              for s in f["trials/mcor_path_levels"][:]]

    return [levels[int(c)] for c in codes]


def _per_trial_flags(
    report: dict, *, dropped=(), trial_ids=None, acq_ids=None, mcor=None,
) -> list[str]:
    """Plain statements of what looks wrong, or an empty list."""

    out = []
    pooled = report["blocks"]["pooled"]
    thresholds = report["thresholds"]

    # Which acquisitions, and where to find them. The counts live in
    # `excluded_trials`; naming the files belongs here, because it is
    # something to go and act on rather than a table to carry around.
    if len(dropped):
        lines = []

        for d in list(dropped)[:6]:
            i = int(d["trial_index"])

            if acq_ids is not None and i < len(acq_ids):
                where = f"acq_id {int(acq_ids[i])}"
                if mcor is not None and i < len(mcor):
                    where += f" ({Path(mcor[i]).name})"
            elif mcor is not None and i < len(mcor):
                where = Path(mcor[i]).name
            elif trial_ids is not None and i < len(trial_ids):
                where = f"trial_id {int(trial_ids[i])}"
            else:
                where = f"trial index {i}"

            lines.append(f"{where} (F0 {d['f0']:.0f}, {d['sd_from_mean']:.0f} SD)")

        more = "" if len(dropped) <= 6 else f", and {len(dropped) - 6} more"
        out.append(
            f"{len(dropped)} approved acquisition(s) have a baseline over "
            f"{report['excluded_trials']['threshold_sd']:.0f} SD from the "
            f"session mean and were excluded. They passed manual mcor "
            f"approval, so they are worth re-examining: "
            + "; ".join(lines) + more + "."
        )

    excited = [v for v in pooled["excited_per_odor"].values() if v is not None]
    if excited and max(excited) > 0.8:
        out.append(
            f"One odor drove {max(excited):.0%} of glomeruli. Dense activation "
            f"is real at high concentration, but check it against motion and "
            f"a z-shift before reading it as odor coding."
        )

    if thresholds["z_suppressed"] is None or not np.isfinite(
        thresholds["z_suppressed"] or np.nan
    ):
        out.append(
            "No suppression threshold reaches the target FDR: the suppressed "
            "tail is not separable from the pre-odor control on this session. "
            "The suppressed counts are 'undetermined', not zero."
        )

    if report.get("manipulation") is None:
        out.append(
            "No manipulation label in this round, so pre/post cannot be told "
            "apart from a saline control downstream. Re-run the round to "
            "record it."
        )

    control = report["control"]["false_positive_rate"]
    if control is not None and control > 0.02:
        out.append(
            f"The split-half control calls {control:.1%} of a set whose truth "
            f"is zero. The pre-odor window is not behaving like a null, so "
            f"treat both counts as upper bounds."
        )

    tiny = report.get("rois_under_10px", 0)
    if tiny:
        out.append(
            f"{tiny} ROI(s) are under 10 px. A footprint that small is not a "
            f"glomerulus; its trace is a single pixel's noise wearing the same "
            f"shape as a real one."
        )

    bleach = report["bleaching"]
    if bleach["within_trial_pre_pct"] < -5:
        out.append(
            f"Fluorescence falls {abs(bleach['within_trial_pre_pct']):.0f}% "
            f"across the pre-odor window within a trial, on the raw traces. "
            f"The per-trial calls run on the detrended ones where this is "
            f"corrected, so this is a note on the round, not on the counts."
        )

    if abs(bleach["across_session_pct"]) > 25:
        out.append(
            f"Baseline F moves {bleach['across_session_pct']:+.0f}% from the "
            f"first acquisition to the last. Each trial is referenced to its "
            f"own pre-odor window so the counts absorb this, but any pre/post "
            f"comparison is also a comparison across that drift."
        )

    return out
