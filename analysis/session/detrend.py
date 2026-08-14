"""
Remove the instrumental decay at the start of each acquisition.

Every acquisition opens with a fast exponential fall in fluorescence -- 665 to
604 a.u. over the first four seconds on group 217, ~10% -- which settles just
as the odor arrives. It is instrumental, not physiological: the laser is
unblanked at acquisition start and the detector settles.

It matters because F0 is taken from the last second before odor onset, which
is exactly the floor of that decay. Everything after is then compared against a
trough, so the whole post-odor period reads 5-8% elevated and never appears to
return to baseline. On group 217 that made all 112 ROIs "responsive".

Moving the F0 window does not help: every earlier window sits higher on the
curve and biases dF/F the other way. There is no unbiased window on a decaying
baseline, so the decay itself has to go.

**Two components, not one.** Fitting a single exponential to the pre-odor
period alone made things worse: it lands on an asymptote of 592 while the trace
actually settles near 648, so subtracting it lowered F0 and *widened* the
apparent post-odor elevation, from +7.4% to +9.3%. A fit that only ever sees
the first five seconds cannot know where the trace ends up.

So the model is

    F(t) = A_fast * exp(-t / tau_fast) + A_slow * exp(-t / tau_slow) + C

fitted across the *whole* acquisition with the odor window and its aftermath
masked out, so both components and the true asymptote are constrained. `t` is
in frames from acquisition start, since the transient is locked to the
acquisition rather than to the odor. Both exponential terms are subtracted;
`C` is kept, because subtracting the entire fit would leave a zero-mean
residual and dF/F would divide by nothing.

**Why this is signal-safe here.** The offset it removes is field-wide, not
glomerular: the neuropil rings show it as strongly as the ROIs (+0.077 against
+0.072), it is far more uniform across ROIs than the odor response (CV 0.13
against a four-fold spread), it does not scale with each ROI's own response
(r = -0.13), it is the same before and after anaesthesia, and mineral oil shows
it at full strength. None of that is true of an odor-evoked tail.

The neuropil is deliberately *not* fitted -- it is held back as the check. If
the correction is right, the neuropil's late elevation collapses with the
ROIs'.

**Fitting.** Both time constants are instrumental, so they are estimated once
from the population mean rather than per ROI, where the fit would chase noise.
With them fixed the model is linear in the two amplitudes and the offset, so
those come from a closed-form least squares per ROI per trial -- fast, and it
lets each ROI carry its own amplitude, which is what a field-wide transient
scaled by brightness looks like.

**Raw F is not modified.** The correction applies to traces read out of the
round, never to what is written into it, so the stored `/traces/roi` stays raw
and any correction can be revisited without re-extracting.
"""

from __future__ import annotations

import numpy as np

# The slow constant is parameterised as a *ratio* to the fast one rather than
# in seconds of its own.
#
# Two exponentials free to take any constants are not identifiable when those
# constants are close: the fit expresses the fall-and-recover shape as the
# difference of two nearly identical curves, with amplitudes ballooning to
# +869/-811 to cancel, and the pair swinging 2.5-fold between halves of one
# session. Fixing that with a hard floor on the slow constant just pins it
# there -- which is what happened, at 5.0 s in every subset.
#
# A minimum ratio keeps them apart in a way the optimiser cannot defeat, and
# scales with whatever the fast constant turns out to be, so it needs no
# retuning per rig.
TAU_FAST_BOUNDS_S = (0.1, 10.0)
TAU_RATIO_BOUNDS = (2.0, 40.0)

# Frames after odor offset excluded from the fit, in seconds. The response and
# its decay must not constrain the baseline model, or the fit bends to absorb
# real signal. Everything before onset and after this point is used.
RESPONSE_GUARD_S = 6.0


def fit_baseline(
    roi: np.ndarray,
    *,
    odor_on_frames: np.ndarray,
    odor_off_frames: np.ndarray,
    frame_rate: float,
    trials: None | np.ndarray = None,
    guard_s: float = RESPONSE_GUARD_S,
    components: int = 2,
    fit_window: str = "outside",
) -> dict:
    """
    Estimate the baseline's time constants from the population-mean trace.

    `components=1` fits a single exponential; the constant then has to describe
    both the fast settling and whatever slower drift follows, so it lands
    somewhere between. Worth comparing against, because two components are only
    justified if they earn it downstream -- the second one costs identifiability
    and neither version was stable across blocks.

    The odor window and `guard_s` after it are masked out, so the response
    cannot pull the baseline model. What remains -- the pre-odor period and the
    late tail -- is what constrains the two components and the asymptote.
    """

    from scipy.optimize import curve_fit

    if trials is None:
        trials = np.ones(roi.shape[1], dtype=bool)

    n_frame = roi.shape[2]
    on = int(np.min(odor_on_frames))
    off = int(np.max(odor_off_frames))
    guard = int(round(guard_s * frame_rate))

    mask = np.ones(n_frame, dtype=bool)

    if fit_window == "pre":
        # Only the pre-odor period. This is the honest model when the
        # post-odor elevation is real rather than instrumental: it removes the
        # acquisition-start transient and asserts nothing about what follows.
        # It cannot flatten the late trace, and should not be expected to.
        mask[on:] = False
    elif fit_window == "outside":
        mask[on:min(off + guard, n_frame)] = False
    else:
        raise ValueError(f"fit_window must be 'pre' or 'outside', got {fit_window!r}.")

    if mask.sum() < 20:
        return {"ok": False, "reason": "not enough frames outside the odor window"}

    mean = np.nanmean(roi[:, trials, :], axis=(0, 1))
    t = np.arange(n_frame, dtype=np.float64)

    # tau_slow is expressed as a multiple of tau_fast, not in seconds of its
    # own, so the optimiser cannot collapse the two onto each other.
    def model(t, a_fast, tau_fast, a_slow, ratio, offset):
        return (a_fast * np.exp(-t / tau_fast)
                + a_slow * np.exp(-t / (tau_fast * ratio)) + offset)

    span = float(mean[mask][0] - mean[mask][-1])

    if components == 1:
        def model(t, amplitude, tau, offset):          # noqa: F811
            return amplitude * np.exp(-t / tau) + offset

        # One component must cover both timescales, so it gets a wider range.
        guess1 = (span, 2.0 * frame_rate, float(mean[mask][-1]))
        bounds1 = (
            [-np.inf, TAU_FAST_BOUNDS_S[0] * frame_rate, -np.inf],
            [np.inf, 60.0 * frame_rate, np.inf],
        )
        try:
            params, _ = curve_fit(model, t[mask], mean[mask], p0=guess1,
                                  bounds=bounds1, maxfev=40000)
        except (RuntimeError, ValueError) as error:
            return {"ok": False, "reason": str(error)}

        amplitude, tau, offset = params
        fitted = model(t, *params)
        residual = mean[mask] - fitted[mask]
        total = mean[mask] - mean[mask].mean()
        return {
            "ok": True, "components": 1,
            "tau_fast_s": float(tau / frame_rate),
            "tau_slow_s": float("nan"),
            "tau_fast_frames": float(tau),
            "tau_slow_frames": float("inf"),
            "a_fast": float(amplitude), "a_slow": 0.0,
            "tau_ratio": float("nan"), "offset": float(offset),
            "r_squared": float(1.0 - residual.var() / max(total.var(), 1e-12)),
            "n_frames_fitted": int(mask.sum()),
            "masked": [on, min(off + guard, n_frame)],
            "fit_window": fit_window,
        }

    guess = (span, 1.0 * frame_rate, -span, 4.0, float(mean[mask][-1]))
    bounds = (
        [-np.inf, TAU_FAST_BOUNDS_S[0] * frame_rate,
         -np.inf, TAU_RATIO_BOUNDS[0], -np.inf],
        [np.inf, TAU_FAST_BOUNDS_S[1] * frame_rate,
         np.inf, TAU_RATIO_BOUNDS[1], np.inf],
    )

    try:
        params, _ = curve_fit(model, t[mask], mean[mask], p0=guess,
                              bounds=bounds, maxfev=40000)
    except (RuntimeError, ValueError) as error:
        return {"ok": False, "reason": str(error)}

    a_fast, tau_fast, a_slow, ratio, offset = params
    tau_slow = tau_fast * ratio
    fitted = model(t, *params)
    residual = mean[mask] - fitted[mask]
    total = mean[mask] - mean[mask].mean()

    return {
        "ok": True,
        "components": 2,
        "tau_fast_s": float(tau_fast / frame_rate),
        "tau_slow_s": float(tau_slow / frame_rate),
        "tau_fast_frames": float(tau_fast),
        "tau_slow_frames": float(tau_slow),
        "a_fast": float(a_fast),
        "a_slow": float(a_slow),
        "offset": float(offset),
        "r_squared": float(1.0 - residual.var() / max(total.var(), 1e-12)),
        "tau_ratio": float(ratio),
        "n_frames_fitted": int(mask.sum()),
        "masked": [on, min(off + guard, n_frame)],
    }


def detrend_traces(
    roi: np.ndarray,
    *,
    odor_on_frames: np.ndarray,
    odor_off_frames: np.ndarray,
    frame_rate: float,
    fit: None | dict = None,
    trials: None | np.ndarray = None,
    guard_s: float = RESPONSE_GUARD_S,
) -> tuple[np.ndarray, dict]:
    """
    Subtract both fitted components from every trace, keeping the offset.

    `fit` reuses a result from `fit_baseline` -- pass the ROI fit when
    correcting neuropil, so both are corrected by the same model and the
    neuropil stays an independent check rather than being fitted to itself.
    """

    n_roi, n_trial, n_frame = roi.shape

    if fit is None:
        fit = fit_baseline(
            roi, odor_on_frames=odor_on_frames, odor_off_frames=odor_off_frames,
            frame_rate=frame_rate, trials=trials, guard_s=guard_s,
        )
    if not fit.get("ok"):
        return np.array(roi, copy=True), {"ok": False, "reason": fit.get("reason"),
                                          "fit": fit}

    t = np.arange(n_frame, dtype=np.float64)
    fast = np.exp(-t / fit["tau_fast_frames"])
    slow = np.exp(-t / fit["tau_slow_frames"])

    on = int(np.min(odor_on_frames))
    off = int(np.max(odor_off_frames))
    guard = int(round(guard_s * frame_rate))
    mask = np.ones(n_frame, dtype=bool)
    mask[on:min(off + guard, n_frame)] = False

    out = np.array(roi, dtype=np.float32, copy=True)
    amps = np.full((n_roi, n_trial, 2), np.nan)

    design_full = np.column_stack([fast, slow, np.ones(n_frame)])
    design = design_full[mask]

    for trial in range(n_trial):
        values = roi[:, trial, :][:, mask].T
        finite = np.isfinite(values).all(axis=1)
        if finite.sum() < 8:
            continue

        coeffs, *_ = np.linalg.lstsq(design[finite], values[finite], rcond=None)
        a_fast, a_slow = coeffs[0], coeffs[1]
        amps[:, trial, 0] = a_fast
        amps[:, trial, 1] = a_slow

        out[:, trial, :] -= (a_fast[:, None] * fast[None, :]
                             + a_slow[:, None] * slow[None, :])

    return out, {
        "ok": True,
        "tau_fast_s": fit["tau_fast_s"],
        "tau_slow_s": fit["tau_slow_s"],
        "tau_fast_frames": fit["tau_fast_frames"],
        "tau_slow_frames": fit["tau_slow_frames"],
        "r_squared": fit["r_squared"],
        "guard_s": float(guard_s),
        "median_a_fast": float(np.nanmedian(amps[:, :, 0])),
        "median_a_slow": float(np.nanmedian(amps[:, :, 1])),
        # Full per-ROI per-trial coefficients, so the correction is
        # reproducible from the round without refitting -- and auditable, since
        # an amplitude far from its neighbours marks a trial whose baseline did
        # something the model did not expect.
        "a_fast": amps[:, :, 0],
        "a_slow": amps[:, :, 1],
        "fit": fit,
    }


# --------------------------------------------------------------------------- #
# Non-parametric alternative
# --------------------------------------------------------------------------- #


def fit_shape(
    roi: np.ndarray,
    *,
    odor_on_frames: np.ndarray,
    odor_off_frames: np.ndarray,
    frame_rate: float,
    trials: None | np.ndarray = None,
    guard_s: float = RESPONSE_GUARD_S,
) -> dict:
    """
    Measure the settling profile instead of assuming its functional form.

    Every ROI-trial trace is divided by its own mean over the unmasked frames,
    which removes brightness, and the median across all of them at each frame
    is the common shape. With ~17,800 traces per frame it needs no smoothing.

    The parametric fit this replaces was degenerate: `tau_slow` pinned to its
    bound in every subset while the two amplitudes ballooned to +869/-811 to
    cancel, and the resulting correction differed by 5.4% of mean F between
    halves of one session -- the same size as the effect being removed. A
    measured shape has no constants to rail against.

    The shape is centred on its own mean, so subtracting `scale * shape` leaves
    the trace's overall level alone; only the time-varying part goes.
    """

    if trials is None:
        trials = np.ones(roi.shape[1], dtype=bool)

    n_frame = roi.shape[2]
    on = int(np.min(odor_on_frames))
    off = int(np.max(odor_off_frames))
    guard = int(round(guard_s * frame_rate))
    stop = min(off + guard, n_frame)

    mask = np.ones(n_frame, dtype=bool)
    mask[on:stop] = False

    block = roi[:, trials, :]
    level = np.nanmean(block[:, :, mask], axis=2, keepdims=True)
    normed = block / np.where(np.abs(level) > 1e-9, level, np.nan)

    with np.errstate(invalid="ignore"):
        shape = np.nanmedian(normed.reshape(-1, n_frame), axis=0)

    # The response lives inside the mask, so the shape is interpolated across
    # it rather than measured there -- the settling is smooth, the response is
    # not, and taking the median through it would let signal into the baseline.
    index = np.arange(n_frame)
    shape[~mask] = np.interp(index[~mask], index[mask], shape[mask])

    centred = shape - shape[mask].mean()

    return {
        "ok": True,
        "shape": centred,
        "raw_shape": shape,
        "mask": mask,
        "masked": [on, stop],
        "range": float(centred[mask].max() - centred[mask].min()),
    }


def detrend_by_shape(
    roi: np.ndarray,
    *,
    shape: dict,
    trials: None | np.ndarray = None,
) -> tuple[np.ndarray, dict]:
    """
    Regress a measured shape out of every trace, one scale per ROI per trial.

    Pass the ROI shape when correcting neuropil, so both are corrected by the
    same profile and the neuropil stays an independent check.
    """

    centred = shape["shape"]
    mask = shape["mask"]
    design = np.column_stack([centred[mask], np.ones(mask.sum())])

    out = np.array(roi, dtype=np.float32, copy=True)
    scales = np.full((roi.shape[0], roi.shape[1]), np.nan)

    for trial in range(roi.shape[1]):
        values = roi[:, trial, :][:, mask].T
        finite = np.isfinite(values).all(axis=1)
        if finite.sum() < 8:
            continue
        coeffs, *_ = np.linalg.lstsq(design[finite], values[finite], rcond=None)
        scales[:, trial] = coeffs[0]
        out[:, trial, :] -= coeffs[0][:, None] * centred[None, :]

    return out, {
        "ok": True,
        "method": "measured_shape",
        "median_scale": float(np.nanmedian(scales)),
        "scale_iqr": [float(np.nanpercentile(scales, 25)),
                      float(np.nanpercentile(scales, 75))],
        "shape_range": shape["range"],
    }


def fit_per_trial(
    roi: np.ndarray,
    *,
    odor_on_frames: np.ndarray,
    odor_off_frames: np.ndarray,
    frame_rate: float,
    trials: None | np.ndarray = None,
    guard_s: float = RESPONSE_GUARD_S,
    components: int = 2,
) -> dict:
    """
    One set of time constants per trial, from that trial's mean across ROIs.

    Amplitudes are per ROI per trial either way -- that is what the least
    squares in `detrend_traces` already does. What this adds is letting the
    *shape* drift across the session, which it should if the transient damps as
    the tissue is repeatedly exposed.

    Averaging across ROIs within the trial before fitting is deliberate. Going
    all the way to per ROI per trial means fitting five nonlinear parameters to
    ~200 noisy frames, seventeen thousand times; the spread that produces is
    estimation error, not biology, and the amplitudes already carry the per-ROI
    part.

    Returns per-trial fits with NaN where a trial did not converge, so the
    caller can see the coverage rather than silently getting a fallback.
    """

    if trials is None:
        trials = np.ones(roi.shape[1], dtype=bool)

    out = {"tau_fast_s": [], "tau_slow_s": [], "a_fast": [],
           "r_squared": [], "ok": [], "trial": []}

    for trial in range(roi.shape[1]):
        if not trials[trial]:
            continue

        single = roi[:, trial:trial + 1, :]
        fit = fit_baseline(
            single,
            odor_on_frames=odor_on_frames[trial:trial + 1],
            odor_off_frames=odor_off_frames[trial:trial + 1],
            frame_rate=frame_rate, guard_s=guard_s, components=components,
        )
        out["trial"].append(trial)
        out["ok"].append(bool(fit.get("ok")))
        for key in ("tau_fast_s", "tau_slow_s", "a_fast", "r_squared"):
            out[key].append(float(fit[key]) if fit.get("ok") else float("nan"))

    return {k: np.asarray(v) for k, v in out.items()}
