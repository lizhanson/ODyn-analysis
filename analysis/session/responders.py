"""
Which ROI-odor pairs are excited or suppressed, by a paired test per pair.

`response_qc.py` answers this with a sham null: slide the measurement window
back into the pre-odor period, collect the values it produces, and read a
threshold off that distribution. The null is what limits it. An empirical
p-value cannot go below `1/(n_null+1)`, and because the null is built as one
(n_roi x n_odor) matrix per sham offset, the number of tests grows exactly as
fast as the number of null samples -- so the ROI and odor counts cancel and
only the offset count matters. With the two fitted offsets a 5 s pre-odor
period allows, no pair can score below p = 2.8e-4, and BH only returns
discoveries when enough pairs pile up at that floor together. Excitation
clears it collectively and the threshold collapses toward noise; suppression,
being smaller, never reaches the critical mass and gets nothing. On group223
that came out as 110/110 ROIs excited at z >= 0.286 and 0 suppressed on every
odor.

The fix here is to stop estimating a null and test each pair on its own trials.
Every trial of one odor gives one number -- the odor window minus that trial's
own pre-odor window -- and a one-sample t-test asks whether those numbers
differ from zero. The p-value is analytic, so it has no resolution floor and
does not depend on how many other pairs are significant: a pair is judged on
its own evidence. Excitation and suppression are the same test read in two
directions, which is what makes the two tails comparable.

Two things this leans on:

**Detrended traces.** The correction in `detrend.py` removes the instrumental
settling transient, which is what made a pre-odor window unusable as F0 --
`finalize.py` notes the last second sits at the floor of a ~10% transient, so
anything earlier reads elevated and biases dF/F positive. `/traces/roi` is
stored raw and `response_qc` reads it; this module prefers
`/traces/roi_detrended` and records which it used.

**The whole pre-odor window as F0.** `BASELINE_S = 1.0` in `response_qc` is a
workaround for that same transient. Once it is corrected the constraint is
gone, and a baseline over ~5 s instead of ~1 s is the largest free gain
available here: the variance of (response - baseline) falls with the frame
count in the baseline, and at this rig's geometry that is a ~2.8x reduction,
or ~1.7x on t, before any change of statistic. Fluorescence noise is
autocorrelated, so treat that as an upper bound -- the realised gain is
smaller, and `control_test` is what measures whether it was worth it.

Nothing here calibrates against the pre-odor period, but it is still used, as
a check rather than a threshold: `control_test` reruns the identical test on a
split of the pre-odor window, where there is no odor and the answer should be
"nothing". Rejections there are residual drift the detrend did not remove, and
they invalidate the real counts rather than adjusting them.

What this trades away: a t-test over trials rewards consistency, not
amplitude. A pair that responds enormously on 2 of 10 trials and not at all on
the other 8 will not pass, and under the sham-null threshold it would have.
That is the right trade for suppression, which is small and (by hypothesis)
reliable, but it is a real change in what "responsive" means and it is worth
stating in a methods section rather than discovering later.

Power is set by the trial count, and 10 trials per odor is not many. At this
rig's geometry a suppression of 0.5 per-trial SD is detected ~14% of the time
and 1.0 SD ~56%; 40 trials per odor takes those to 68% and >99%. No choice of
test fixes that -- see the note on `MIN_TRIALS`.
"""

from __future__ import annotations

import numpy as np

# Target false discovery rate across all ROI-odor pairs, both tails together.
#
# One family, not one per tail: excitation and suppression are the same test
# read in two directions, and splitting them into two families of the same m
# would charge the multiple-comparisons cost twice while controlling neither
# jointly. The sign is read off the t statistic after the fact.
TARGET_FDR = 0.05

# Fewer trials than this and the pair is not tested at all.
#
# A t-test on 3 trials is not wrong so much as useless: it has almost no power
# against anything, and its p-values still enter the BH denominator, so every
# under-powered pair makes the threshold stricter for the pairs that could
# have passed. Dropping them outright is better than letting them dilute.
MIN_TRIALS = 5


def read_manipulation(f) -> str | None:
    """
    The manipulation label from a round, or None if it predates the field.

    `state` says which side of the manipulation a trial is on; this says what
    it was. Both are needed to read a pre/post design -- a saline control and
    a ketamine session have identical `state` columns, and without this the
    time-in-session confound cannot be separated from the manipulation.

    Stored per trial and constant in practice, so the single label is returned
    when there is one and a joined string when a round somehow carries more,
    rather than silently reporting the first.
    """

    if "trials/manipulation" not in f:
        return None

    codes = f["trials/manipulation"][:]
    levels = [s.decode() if isinstance(s, bytes) else str(s)
              for s in f["trials/manipulation_levels"][:]]

    present = sorted({levels[int(c)] for c in codes})

    return present[0] if len(present) == 1 else " + ".join(present)


def load_roi_traces(f, *, prefer_detrended: bool = True) -> tuple[np.ndarray, str]:
    """
    The detrended traces if the round carries them, the raw ones otherwise.

    Returns which it used rather than silently picking, because every number
    downstream depends on it and the two are not interchangeable -- see the
    module docstring on why a raw trace cannot use a long baseline.
    """

    if prefer_detrended and "traces/roi_detrended" in f:
        return f["traces/roi_detrended"][:], "detrended"

    return f["traces/roi"][:], "raw"


def trial_deltas(
    roi: np.ndarray,
    *,
    odor_on_frames: np.ndarray,
    odor_off_frames: np.ndarray,
    baseline_s: float | None = None,
    frame_rate: float,
) -> dict:
    """
    Per-ROI, per-trial (odor window mean - pre-odor window mean).

    `baseline_s=None` uses the whole pre-odor period of each trial, which is
    the point of running on detrended traces. Passing a number instead takes
    only that many seconds immediately before onset, which is what to do if
    the detrend is unavailable and the settling transient is still in there.

    Returns the difference in the traces' own units and the baseline level
    beside it, so a caller can express the effect as dF/F without recomputing
    F0. The test itself uses the difference: a t statistic is invariant to
    scaling within a pair, so normalising first would change the reported
    effect size and not the p-value.
    """

    n_roi, n_trial, n_frame = roi.shape

    delta = np.full((n_roi, n_trial), np.nan)
    f0 = np.full((n_roi, n_trial), np.nan)
    baseline_frames = np.zeros(n_trial, dtype=int)
    response_frames = np.zeros(n_trial, dtype=int)

    n_base = (None if baseline_s is None
              else max(1, int(round(baseline_s * frame_rate))))

    for trial in range(n_trial):
        on = int(odor_on_frames[trial])
        off = int(odor_off_frames[trial])

        if on <= 0 or off <= on or off > n_frame:
            continue

        start = 0 if n_base is None else max(on - n_base, 0)

        base = roi[:, trial, start:on]
        resp = roi[:, trial, on:off]

        if base.shape[1] == 0 or resp.shape[1] == 0:
            continue

        base_mean = np.nanmean(base, axis=1)

        delta[:, trial] = np.nanmean(resp, axis=1) - base_mean
        f0[:, trial] = base_mean
        baseline_frames[trial] = base.shape[1]
        response_frames[trial] = resp.shape[1]

    return {
        "delta": delta,
        "f0": f0,
        "baseline_frames": baseline_frames,
        "response_frames": response_frames,
    }


def bh_reject(p: np.ndarray, q: float) -> tuple[np.ndarray, float | None]:
    """
    Benjamini-Hochberg step-up over every finite p-value in `p`.

    Sort ascending, find the largest rank k with p_(k) <= (k/m)*q, reject
    everything at or below that p. Returns the mask in `p`'s own shape and the
    cutoff, or `(all False, None)` when nothing survives.

    Note this is a step-UP: the bar rises as more tests pass, so a single test
    does not need to clear q/m -- it needs to clear (k/m)*q for whatever k the
    family reaches. That is why the sham-null version could return 1388
    discoveries while its own `underpowered` check said none were possible.
    """

    values = np.asarray(p, dtype=np.float64)
    flat = values.ravel()
    finite = np.isfinite(flat)

    kept = flat[finite]
    m = kept.size

    if m == 0:
        return np.zeros(values.shape, dtype=bool), None

    ordered = np.sort(kept)
    passing = ordered <= (np.arange(1, m + 1) / m) * q

    if not passing.any():
        return np.zeros(values.shape, dtype=bool), None

    cutoff = float(ordered[int(np.flatnonzero(passing).max())])

    out = np.zeros(flat.shape, dtype=bool)
    out[finite] = kept <= cutoff

    return out.reshape(values.shape), cutoff


def responder_test(
    delta: np.ndarray,
    odor_ids: np.ndarray,
    *,
    f0: np.ndarray | None = None,
    target_fdr: float = TARGET_FDR,
    min_trials: int = MIN_TRIALS,
    trials: np.ndarray | None = None,
) -> dict:
    """
    One-sample t-test per ROI-odor pair against zero, then BH over all pairs.

    `delta` is (n_roi, n_trial) from `trial_deltas`; `trials` optionally
    restricts which trials count, for testing a block (pre, post) on its own.

    Excited and suppressed come from one BH pass split by the sign of t, not
    from two passes. Two one-tailed families would each carry the full m and
    control neither jointly; one two-sided family costs a factor of two on
    each p-value and controls the thing actually being claimed -- the share of
    *responsive calls* that are wrong, regardless of direction.
    """

    from scipy import stats

    odor_ids = np.asarray(odor_ids)
    keys = np.unique(odor_ids)

    n_roi = delta.shape[0]
    shape = (n_roi, len(keys))

    t = np.full(shape, np.nan)
    p = np.full(shape, np.nan)
    effect = np.full(shape, np.nan)
    effect_dff = np.full(shape, np.nan)
    n_trials = np.zeros(len(keys), dtype=int)

    for j, key in enumerate(keys):
        selected = odor_ids == key
        if trials is not None:
            selected = selected & trials

        x = delta[:, selected]
        n_trials[j] = int(selected.sum())

        if x.shape[1] < min_trials:
            continue

        # nan_policy="omit" so one bad trial costs that pair a degree of
        # freedom rather than the whole pair. Pairs left under min_trials
        # worth of finite values come back as nan and drop out of BH below.
        with np.errstate(invalid="ignore", divide="ignore"):
            result = stats.ttest_1samp(x, 0.0, axis=1, nan_policy="omit")

        t[:, j] = np.asarray(result.statistic, dtype=np.float64)
        p[:, j] = np.asarray(result.pvalue, dtype=np.float64)
        effect[:, j] = np.nanmean(x, axis=1)

        if f0 is not None:
            level = np.nanmean(f0[:, selected], axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                effect_dff[:, j] = effect[:, j] / np.where(
                    np.abs(level) > 1e-9, np.abs(level), np.nan
                )

    # A pair with too few finite trials has p = nan and is excluded from m
    # rather than counted as a non-rejection, which would inflate m and make
    # the threshold stricter for everything else.
    reject, cutoff = bh_reject(p, target_fdr)

    excited = reject & (t > 0)
    suppressed = reject & (t < 0)

    return {
        "odor_ids": keys,
        "t": t,
        "p": p,
        "effect": effect,
        "effect_dff": effect_dff,
        "excited": excited,
        "suppressed": suppressed,
        "responsive": excited | suppressed,
        "p_cutoff": cutoff,
        "target_fdr": target_fdr,
        "n_tests": int(np.isfinite(p).sum()),
        "n_excited": int(excited.sum()),
        "n_suppressed": int(suppressed.sum()),
        "trials_per_odor": {int(k): int(n) for k, n in zip(keys, n_trials)},
        "rois_excited": int(excited.any(axis=1).sum()),
        "rois_suppressed": int(suppressed.any(axis=1).sum()),
        "n_roi": int(n_roi),
    }


def trial_calls(
    roi: np.ndarray,
    *,
    odor_on_frames: np.ndarray,
    odor_ids: np.ndarray | None = None,
    deglobal: str | None = None,
    window_frames: int | None = None,
    target_fdr: float = TARGET_FDR,
    robust: bool = True,
) -> dict:
    """
    Per ROI, per *trial*: is the odor-locked response above or below chance?

    `responder_test` asks whether a pair responds on average over its trials,
    so a single trial has no answer of its own. This asks the per-trial
    question directly, which is the one that survives if suppression is not
    locked to odor identity -- a trial-resolved call can be regressed against
    anything measured per trial, and an odor-collapsed one cannot.

    A single trial cannot supply its own null: the pre-odor period holds only
    a couple of response-length windows and they overlap almost completely, so
    the null has to be pooled somewhere. It is pooled across trials, per ROI,
    which keeps the ROI-to-ROI differences in noise that a session-wide pool
    would flatten.

    The geometry splits the pre-odor period in half and uses the whole thing:

        pre1 = [on-2W, on-W)    pre2 = [on-W, on)    response = [on, on+W)

        d = mean(response) - mean(pre2)      the measurement
        c = mean(pre2)     - mean(pre1)      the same thing, no odor

    `c` is length-matched to `d` by construction, which is the point -- an
    unmatched control window averages a different number of frames, and its
    spread is then the wrong yardstick for `d`. `W` defaults to half the
    shortest pre-odor period, so the two control windows tile it exactly.

    Each ROI's `c` across trials gives a centre and a scale. Centring is what
    removes residual within-trial drift: after detrending, group223's pre-odor
    window still falls 0.5%, which is small but lands entirely in the
    suppressed tail if it is not subtracted. `robust` uses median and MAD
    rather than mean and SD, so a dead trial widens nothing.

        z = (d - centre) / scale

    p-values are two-sided normal. That is a distributional assumption, unlike
    the empirical null it replaces, but a deliberate one: `d` is a difference
    of two means of W frames each, so the CLT is doing real work, and the
    assumption buys unbounded p-resolution -- which is exactly what the sham
    null could not provide. `split_half_control` measures whether it holds
    instead of asserting it.

    BH runs over every (ROI, trial) pair at once and the sign of z splits the
    survivors, same as `responder_test`.
    """

    from scipy import stats

    on = np.asarray(odor_on_frames).astype(int)
    n_roi, n_trial, n_frame = roi.shape

    if window_frames is None:
        # Half the shortest pre-odor period, capped so the response window
        # cannot run past the end of the recording on any trial.
        window_frames = int(min(on.min() // 2, (n_frame - on.max())))

    w = int(window_frames)
    if w < 2:
        raise ValueError(
            f"window_frames came out {w}: the pre-odor period "
            f"({int(on.min())} frames) cannot hold two control windows."
        )

    d = np.full((n_roi, n_trial), np.nan)
    c = np.full((n_roi, n_trial), np.nan)

    for trial in range(n_trial):
        start = int(on[trial])

        if start - 2 * w < 0 or start + w > n_frame:
            continue

        pre1 = np.nanmean(roi[:, trial, start - 2 * w:start - w], axis=1)
        pre2 = np.nanmean(roi[:, trial, start - w:start], axis=1)
        resp = np.nanmean(roi[:, trial, start:start + w], axis=1)

        d[:, trial] = resp - pre2
        c[:, trial] = pre2 - pre1

    centre, scale = _centre_scale(c, robust=robust)
    d_control = c

    with np.errstate(invalid="ignore", divide="ignore"):
        z = (d - centre) / np.where(scale > 1e-12, scale, np.nan)

    # The shared trial-to-trial component is removed from the real and the
    # control z alike. Correcting only the real one would leave the chance
    # rate measured against a wider distribution than the calls are, which
    # flatters every ratio on the figure.
    component = None
    if deglobal is not None:
        if odor_ids is None:
            raise ValueError("deglobal needs odor_ids to protect odor means.")

        control_z = (d_control - centre) / np.where(scale > 1e-12, scale, np.nan)
        z, component = _deglobal(z, odor_ids, method=deglobal)
        control_z, _ = _deglobal(control_z, odor_ids, method=deglobal)
    else:
        control_z = (d_control - centre) / np.where(scale > 1e-12, scale, np.nan)

    p = 2.0 * stats.norm.sf(np.abs(z))

    result = {
        "z": z,
        "control_z": control_z,
        "deglobal": deglobal,
        "global_component": component,
        "p": p,
        "d": d,
        "control": c,
        "centre": centre.ravel(),
        "scale": scale.ravel(),
        "window_frames": w,
        "target_fdr": target_fdr,
    }

    # Each tail gets its own threshold against its own control tail. A single
    # BH pass over both signs controls only the false share of the two
    # together, and the tails here are nowhere near the same size -- far more
    # glomeruli are excited than suppressed, and that is expected rather than
    # a defect.
    #
    # FDR is not inherited by a subset, and the suppressed calls are always
    # the subset that matters. On group223 the joint pass let the dense
    # excited tail lift the bar and the sparse suppressed tail ride on it:
    # 918 suppressed calls at a 60% false share against a 5% target, where
    # thresholding the tails separately gives 254 at 4.7%. The excited tail is
    # fine either way (4.8% joint), which is exactly why the joint number
    # looks reassuring and is not.
    thresholds = empirical_trial_threshold(result, target_fdr=target_fdr)

    excited = z >= thresholds["excited"]["z_threshold"]
    suppressed = z <= thresholds["suppressed"]["z_threshold"]

    # The joint-BH result is kept for comparison only -- `n_excited` and
    # `n_suppressed` below come from the per-tail thresholds.
    reject, cutoff = bh_reject(p, target_fdr)

    result |= {
        "excited": excited,
        "suppressed": suppressed,
        "responsive": excited | suppressed,
        "thresholds": thresholds,
        "z_excited_threshold": thresholds["excited"]["z_threshold"],
        "z_suppressed_threshold": thresholds["suppressed"]["z_threshold"],
        "joint_bh": {
            "excited": int((reject & (z > 0)).sum()),
            "suppressed": int((reject & (z < 0)).sum()),
            "p_cutoff": cutoff,
        },
        "n_tests": int(np.isfinite(p).sum()),
        "n_excited": int(excited.sum()),
        "n_suppressed": int(suppressed.sum()),
        # Per trial, how much of the field went each way. This is the series
        # to correlate against anything else measured per trial.
        "excited_per_trial": excited.sum(axis=0),
        "suppressed_per_trial": suppressed.sum(axis=0),
        "n_roi": int(n_roi),
        "n_trial": int(n_trial),
    }

    return result


def empirical_trial_threshold(
    calls: dict, *, target_fdr: float | None = None,
) -> dict:
    """
    Per-trial z thresholds read off the control distribution, per tail.

    `trial_calls` gets its p-values from a normal, and on real traces that is
    optimistic: group223's control z has kurtosis 2.0, and the split-half
    check rejects 1.4% of a set where the truth is zero. This drops the
    distributional assumption entirely.

    Both `d` and `c` are standardised by the same per-ROI centre and scale, so
    the control z is what the real z would look like with no odor. For a
    candidate threshold t, the share of control values beyond t estimates the
    share of real values beyond t that are false:

        FDR(t)  ~  P(control > t) / P(real > t)

    The smallest t meeting the target is chosen per tail, so the two come out
    asymmetric on their own. That is the whole point: excitation and
    suppression differ in both signal size and null width, and one number
    imposed on both is what produced 0 suppressed pairs.

    This is the same ratio-of-tails logic as the old sham null, but it is not
    subject to what broke that one. The threshold is a ratio of two rates, not
    a quantile of the null, so it needs the null to be *shaped* right rather
    than to resolve a 1e-5 tail, and the count of control samples here is
    n_roi x n_trial rather than n_roi x n_odor per offset.

    Returns `inf`/`-inf` for a tail where no threshold reaches the target,
    which is an honest "nothing here" rather than a threshold that misses it.
    """

    q = calls["target_fdr"] if target_fdr is None else target_fdr

    z = calls["z"]

    # Use the standardised control the caller already built when it exists:
    # with `deglobal` set it has had the same component removed as `z`, and
    # rebuilding it from `c` here would compare a corrected z against an
    # uncorrected null and understate the chance rate.
    control = calls.get("control_z")
    if control is None:
        control = (calls["control"] - calls["centre"][:, None]) / np.where(
            calls["scale"][:, None] > 1e-12, calls["scale"][:, None], np.nan
        )

    real = z[np.isfinite(z)]
    null = control[np.isfinite(control)]

    out = {"target_fdr": q, "n_real": int(real.size), "n_control": int(null.size)}

    for tail, sign in (("excited", 1.0), ("suppressed", -1.0)):
        r = np.sort(sign * real)[::-1]
        n = np.sort(sign * null)[::-1]

        # Candidate thresholds are the observed values themselves, so every
        # achievable operating point is considered and no grid is imposed.
        threshold = np.inf
        n_called = 0
        achieved = None

        for i, t in enumerate(r):
            called = i + 1
            false_rate = np.searchsorted(-n, -t, side="left") / max(null.size, 1)
            fdr = (false_rate * real.size) / called

            if fdr <= q:
                threshold, n_called, achieved = float(t), int(called), float(fdr)

        out[tail] = {
            "z_threshold": sign * threshold if np.isfinite(threshold) else sign * np.inf,
            "n_called": n_called,
            "fraction_called": n_called / max(real.size, 1),
            "estimated_fdr": achieved,
        }

    return out



def _deglobal(
    z: np.ndarray, odor_ids: np.ndarray, *, method: str = "pc1",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove the trial-to-trial component shared across ROIs, keeping odor tuning.

    Glomeruli do not fluctuate independently: on these sessions one component
    carries 42-57% of the trial-to-trial variance once each odor's mean
    response is taken out, and pairwise correlations run 0.24-0.61. That
    shared term widens the control distribution as much as the real one, so it
    costs sensitivity in both tails without adding signal to either.

    The order of operations is what keeps this from eating the result:

    1. Take out each odor's mean response per ROI. What is left is variation
       *within* an odor -- trial-to-trial, by construction not odor tuning.
    2. Find the component. `"pc1"` takes the leading right singular vector of
       that residual, so ROIs contribute in proportion to how much they follow
       it. `"median"` uses the population median per trial, which is blunter
       but needs no explaining in a methods section.
    3. Re-centre the component within each odor, so it sums to zero over the
       trials of any one odor. This is the guard that matters: a component
       with zero mean inside every odor cannot shift any odor's mean response,
       so no amount of regressing it out can remove tuning.
    4. Regress it out of the *uncorrected* z per ROI, each with its own slope.
       Per-ROI slopes matter -- ROIs differ in how strongly they follow the
       shared term, and a single global subtraction would over-correct the
       ones that ignore it into spurious suppression.

    Removing the odor mean first is why this is defensible where the neuropil
    subtraction was not. Neuropil carried real odor-specific response (r=0.64
    with each ROI's own tuning), so subtracting it removed signal. This
    component is orthogonal to odor identity before it is ever used.

    Returns the corrected z and the per-trial component that was removed --
    the latter is worth keeping, since it is a per-trial measure of how much
    the whole field moved together and can be correlated against anything else
    recorded per trial.
    """

    odor_ids = np.asarray(odor_ids)
    z = np.asarray(z, dtype=np.float64)

    # 1. residual within odor
    residual = z.copy()
    for key in np.unique(odor_ids):
        mask = odor_ids == key
        residual[:, mask] -= np.nanmean(residual[:, mask], axis=1, keepdims=True)

    residual = np.nan_to_num(residual)

    # 2. the shared component, as a per-trial series
    if method == "pc1":
        # Centre per ROI so the SVD describes covariation, not level.
        centred = residual - residual.mean(axis=1, keepdims=True)
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        component = vt[0]
        # Sign is arbitrary out of an SVD; fix it so the component is positive
        # where most ROIs are, or the returned series flips between sessions.
        if np.corrcoef(component, np.nanmean(residual, axis=0))[0, 1] < 0:
            component = -component
    elif method == "median":
        component = np.nanmedian(residual, axis=0)
    else:
        raise ValueError(f"method must be 'pc1' or 'median', got {method!r}.")

    # 3. zero-mean within every odor
    for key in np.unique(odor_ids):
        mask = odor_ids == key
        component[mask] -= np.nanmean(component[mask])

    # 4. per-ROI regression
    denominator = float(np.sum(component ** 2))
    if denominator <= 1e-12:
        return z, component

    slopes = np.nan_to_num(residual) @ component / denominator

    return z - slopes[:, None] * component[None, :], component


def population_sparsity(
    calls: dict,
    odor_ids: np.ndarray,
    *,
    trials: np.ndarray | None = None,
) -> dict:
    """
    Per odor, the share of glomeruli excited and suppressed, with a spread.

    The unit is one trial: each trial of an odor gives one percentage, and the
    mean and SEM are taken over those trials. That is the quantity a reader
    wants -- "on a given presentation, what fraction of the field responds" --
    and it comes with an honest error bar, which a measure computed on the
    trial-averaged matrix cannot have.

    `chance` is the same percentage computed on the pre-odor control windows,
    and belongs on the plot beside the bars: a 6% excited share means
    something different against a 1% chance rate than against a 5% one.
    """

    odor_ids = np.asarray(odor_ids)
    keys = np.unique(odor_ids)

    excited, suppressed = calls["excited"], calls["suppressed"]
    out = {"odor_ids": [int(k) for k in keys]}

    for name, mask in (("excited", excited), ("suppressed", suppressed)):
        mean, sem, n = [], [], []

        for key in keys:
            selected = odor_ids == key
            if trials is not None:
                selected = selected & trials

            if not selected.any():
                mean.append(np.nan); sem.append(np.nan); n.append(0)
                continue

            # One value per trial: the fraction of ROIs called on that trial.
            per_trial = np.nanmean(mask[:, selected], axis=0)
            mean.append(float(np.nanmean(per_trial)))
            sem.append(float(np.nanstd(per_trial, ddof=1) / np.sqrt(len(per_trial)))
                       if len(per_trial) > 1 else 0.0)
            n.append(int(selected.sum()))

        out[name] = {"mean": np.array(mean), "sem": np.array(sem),
                     "n_trials": np.array(n)}

    control = calls.get("control_z")
    if control is None:
        control = ((calls["control"] - calls["centre"][:, None])
                   / calls["scale"][:, None])

    finite = np.isfinite(control)
    out["chance"] = {
        "excited": float(np.mean(control[finite] >= calls["z_excited_threshold"])),
        "suppressed": float(np.mean(control[finite] <= calls["z_suppressed_threshold"])),
    }

    return out


def _centre_scale(c: np.ndarray, *, robust: bool) -> tuple[np.ndarray, np.ndarray]:
    """Per-ROI centre and spread of the control statistic, as (n_roi, 1)."""

    if robust:
        centre = np.nanmedian(c, axis=1, keepdims=True)
        mad = np.nanmedian(np.abs(c - centre), axis=1, keepdims=True)
        return centre, 1.4826 * mad

    return (np.nanmean(c, axis=1, keepdims=True),
            np.nanstd(c, axis=1, keepdims=True, ddof=1))


def split_half_control(
    calls: dict, *, target_fdr: float | None = None, robust: bool = True,
) -> dict:
    """
    Does the normal approximation in `trial_calls` hold on this session?

    Standardising the control values with a centre and scale computed from
    those same values is circular -- it cannot help but look calibrated. So
    the trials are split: odd trials estimate the centre and scale, even
    trials are standardised with them and run through the identical BH pass.
    No odor is involved in either half, so every rejection is false, and BH
    under a complete null should return approximately none.

    A rejection count well above zero means the tails of `d` are heavier than
    normal, and the per-trial calls are over-confident by however much.
    """

    from scipy import stats

    c = calls["control"]
    q = calls["target_fdr"] if target_fdr is None else target_fdr

    fit, test = c[:, 0::2], c[:, 1::2]
    centre, scale = _centre_scale(fit, robust=robust)

    with np.errstate(invalid="ignore", divide="ignore"):
        z = (test - centre) / np.where(scale > 1e-12, scale, np.nan)

    p = 2.0 * stats.norm.sf(np.abs(z))
    reject, cutoff = bh_reject(p, q)

    n = int(np.isfinite(p).sum())

    return {
        "n_tests": n,
        "n_excited": int((reject & (z > 0)).sum()),
        "n_suppressed": int((reject & (z < 0)).sum()),
        "false_positive_rate": None if not n else int(reject.sum()) / n,
        "p_cutoff": cutoff,
        # Heavier-than-normal tails show up here before they show up in a
        # rejection count, and are informative even when nothing is rejected.
        "kurtosis": float(stats.kurtosis(z[np.isfinite(z)])),
    }


def control_test(
    roi: np.ndarray,
    *,
    odor_on_frames: np.ndarray,
    odor_ids: np.ndarray,
    frame_rate: float,
    target_fdr: float = TARGET_FDR,
    min_trials: int = MIN_TRIALS,
    trials: np.ndarray | None = None,
) -> dict:
    """
    The same test where the answer is known to be nothing.

    Each trial's pre-odor period is split in half and the second half is
    treated as the "response", the first as its baseline. No odor was
    delivered in either, so every rejection is a false one, and BH should
    return approximately zero -- not `target_fdr` of the pairs. BH controls the
    false share *among rejections*, so under a complete null it rejects nothing
    at all with probability >= 1 - q.

    This does not calibrate anything. It measures whether the detrend left the
    pre-odor period flat enough for a long baseline to be honest. Rejections
    here mean residual drift, and they invalidate the real counts rather than
    correcting them -- fall back to `baseline_s=1.0` and rerun.

    The halves are shorter than the real windows, so this is if anything a
    generous check: less averaging means noisier values, which makes a
    spurious rejection harder, not easier. It catches bias, not variance.
    """

    on = np.asarray(odor_on_frames)

    pseudo_on = (on // 2).astype(int)
    pseudo_off = on.astype(int)

    control = trial_deltas(
        roi,
        odor_on_frames=pseudo_on,
        odor_off_frames=pseudo_off,
        baseline_s=None,
        frame_rate=frame_rate,
    )

    out = responder_test(
        control["delta"], odor_ids,
        target_fdr=target_fdr, min_trials=min_trials, trials=trials,
    )

    out["false_positive_rate"] = (
        None if not out["n_tests"]
        else (out["n_excited"] + out["n_suppressed"]) / out["n_tests"]
    )

    return out


def responders(
    path,
    *,
    baseline_s: float | None = None,
    target_fdr: float = TARGET_FDR,
    min_trials: int = MIN_TRIALS,
    prefer_detrended: bool = True,
    by_state: bool = True,
    drop_baseline_outliers: bool = True,
) -> dict:
    """
    Excited and suppressed ROI-odor pairs for one processed round.

    Reads the round, prefers the detrended traces, takes each trial's odor
    window against its own whole pre-odor window, and tests each ROI-odor pair
    over its trials -- see the module docstring for why this replaces the sham
    null rather than tuning it.

    Runs the pooled block and, with `by_state`, each state on its own. The
    per-state blocks halve the trials per odor and lose power accordingly;
    they are reported because the manipulation is the question, not because
    they are as trustworthy as the pooled numbers.

    `control` carries the same test run inside the pre-odor period, where
    nothing should be found. Read it before the counts.
    """

    from pathlib import Path

    from .h5io import open_h5

    path = Path(path)

    with open_h5(path) as f:
        roi, trace_source = load_roi_traces(f, prefer_detrended=prefer_detrended)
        odor_ids = f["trials/odor_id"][:]
        states = f["trials/state"][:]
        state_levels = [
            s.decode() if isinstance(s, bytes) else str(s)
            for s in f["trials/state_levels"][:]
        ]
        on_frames = f["trials/odor_on_frame"][:]
        off_frames = f["trials/odor_off_frame"][:]
        frame_rate = float(f.attrs["frame_rate"])
        exp_name = str(f.attrs["exp_name"])
        mask_digest = str(f.attrs["mask_hash"])

    deltas = trial_deltas(
        roi,
        odor_on_frames=on_frames, odor_off_frames=off_frames,
        baseline_s=baseline_s, frame_rate=frame_rate,
    )

    usable = np.ones(roi.shape[1], dtype=bool)
    dropped = 0

    if drop_baseline_outliers:
        # Reuse response_qc's guard rather than restating it: one dead
        # acquisition otherwise dominates every odor mean it lands in, and a
        # t-test over 10 trials is not robust to it.
        from .response_qc import baseline_outliers

        n_base = int(np.median(deltas["baseline_frames"]))
        guard = baseline_outliers(
            roi, odor_on_frames=on_frames, n_base=max(n_base, 1),
        )
        usable = guard["keep"]
        dropped = guard["n_dropped"]

    blocks = {
        "pooled": responder_test(
            deltas["delta"], odor_ids, f0=deltas["f0"],
            target_fdr=target_fdr, min_trials=min_trials, trials=usable,
        )
    }

    if by_state:
        for code, name in enumerate(state_levels):
            selection = (states == code) & usable
            blocks[name] = responder_test(
                deltas["delta"], odor_ids, f0=deltas["f0"],
                target_fdr=target_fdr, min_trials=min_trials,
                trials=selection,
            )

    control = control_test(
        roi, odor_on_frames=on_frames, odor_ids=odor_ids,
        frame_rate=frame_rate, target_fdr=target_fdr,
        min_trials=min_trials, trials=usable,
    )

    return {
        "file": path.name,
        "exp_name": exp_name,
        "mask_hash": mask_digest,
        "trace_source": trace_source,
        "baseline_s": baseline_s,
        "baseline_frames_median": int(np.median(deltas["baseline_frames"])),
        "response_frames_median": int(np.median(deltas["response_frames"])),
        "target_fdr": target_fdr,
        "trials_excluded": int(dropped),
        "blocks": blocks,
        "control": control,
        "flags": _flags(blocks["pooled"], control, trace_source),
    }


def _flags(pooled: dict, control: dict, trace_source: str) -> list[str]:
    """Plain statements of what would make the counts above unreadable."""

    out = []

    if trace_source != "detrended":
        out.append(
            "Ran on raw traces: the round carries no /traces/roi_detrended. "
            "A baseline over the whole pre-odor window is not safe on raw "
            "fluorescence -- the settling transient biases dF/F positive, "
            "which inflates excitation and hides suppression. Either "
            "re-finalize with detrend=True or pass baseline_s=1.0."
        )

    rate = control.get("false_positive_rate")
    if rate:
        out.append(
            f"The control test found {control['n_excited']} excited and "
            f"{control['n_suppressed']} suppressed pairs inside the pre-odor "
            f"period ({rate:.1%} of tests), where there is no odor and the "
            f"answer should be zero. The pre-odor window is not flat, so the "
            f"long baseline is carrying drift into the real test. Re-run with "
            f"baseline_s=1.0 and treat the counts below as unusable."
        )

    thin = [k for k, n in pooled["trials_per_odor"].items() if n < 10]
    if thin:
        out.append(
            f"{len(thin)} odor(s) have under 10 usable trials. A t-test over "
            f"that many detects a 1.0 per-trial-SD effect about half the time "
            f"and a 0.5 SD effect about one time in seven, so a zero here is "
            f"weak evidence of absence -- particularly for suppression, which "
            f"is the smaller effect."
        )

    if pooled["n_tests"] and not (pooled["n_excited"] + pooled["n_suppressed"]):
        out.append(
            f"No ROI-odor pair survived BH at {pooled['target_fdr']:.0%} over "
            f"{pooled['n_tests']} tests. Unlike the sham-null version this is "
            f"not a resolution limit -- the p-values are analytic -- so it "
            f"means the responses are not consistent across trials."
        )

    excited_share = (pooled["rois_excited"] / pooled["n_roi"]
                     if pooled["n_roi"] else 0.0)
    if excited_share > 0.9:
        out.append(
            f"{pooled['rois_excited']} of {pooled['n_roi']} ROIs are excited "
            f"by something. A field that responds this uniformly is usually "
            f"motion, a z-shift, or a baseline problem rather than odor."
        )

    return out


def trial_counts(
    roi: np.ndarray,
    *,
    odor_on_frames: np.ndarray,
    z_excited: float = 2.0,
    z_suppressed: float = -2.0,
    window_frames: int | None = None,
) -> dict:
    """
    Per trial: how many glomeruli are excited, how many suppressed, and what
    each of those counts would be by chance.

    The plain version of `trial_calls` -- same z, no FDR, no per-tail
    threshold search. You pick the thresholds; this says what they mean.

    z is the odor window mean minus the preceding equal-length window mean,
    divided by the spread of that same quantity measured across trials in the
    pre-odor period. The denominator is the SD of a *window mean*, not of
    single frames. That distinction is the whole ballgame for reading a
    threshold: `response_qc`'s z divides by the SD of individual baseline
    frames, so it is inflated by roughly sqrt(window length) -- about 10x at
    this rig's geometry -- and a "z > 2" there is nowhere near two standard
    deviations of anything. Thresholds are not transferable between the two.

    `chance` is the identical computation on windows inside the pre-odor
    period, where no odor was delivered. It is the empirical answer to "what
    would this count be anyway", and it is measured rather than assumed
    because fluorescence is autocorrelated: the nominal 2.3% per tail beyond
    |z| > 2 for a normal distribution is not what these traces do.

    Read the two together. A per-trial excited count is interesting to the
    extent it exceeds the chance count on the same threshold, and the two
    tails have different chance rates, which is why they get separate
    thresholds.
    """

    calls = trial_calls(
        roi, odor_on_frames=odor_on_frames, window_frames=window_frames,
    )

    z = calls["z"]

    # Use the standardised control the caller already built when it exists:
    # with `deglobal` set it has had the same component removed as `z`, and
    # rebuilding it from `c` here would compare a corrected z against an
    # uncorrected null and understate the chance rate.
    control = calls.get("control_z")
    if control is None:
        control = (calls["control"] - calls["centre"][:, None]) / np.where(
            calls["scale"][:, None] > 1e-12, calls["scale"][:, None], np.nan
        )

    excited = z >= z_excited
    suppressed = z <= z_suppressed

    finite = np.isfinite(control)
    n_roi = z.shape[0]

    chance_excited = float(np.mean(control[finite] >= z_excited))
    chance_suppressed = float(np.mean(control[finite] <= z_suppressed))

    return {
        "z": z,
        "control_z": control,
        "window_frames": calls["window_frames"],
        "z_excited": z_excited,
        "z_suppressed": z_suppressed,
        # The two series to plot or regress against anything per-trial.
        "excited_per_trial": excited.sum(axis=0),
        "suppressed_per_trial": suppressed.sum(axis=0),
        "n_roi": n_roi,
        "chance": {
            "excited_fraction": chance_excited,
            "suppressed_fraction": chance_suppressed,
            # The same numbers as a count of glomeruli, which is the unit the
            # per-trial series is in.
            "excited_per_trial": chance_excited * n_roi,
            "suppressed_per_trial": chance_suppressed * n_roi,
        },
        "observed": {
            "excited_fraction": float(np.nanmean(excited)),
            "suppressed_fraction": float(np.nanmean(suppressed)),
            "excited_per_trial_mean": float(excited.sum(axis=0).mean()),
            "suppressed_per_trial_mean": float(suppressed.sum(axis=0).mean()),
        },
    }


def chance_table(
    counts: dict, thresholds=(1.5, 2.0, 2.5, 3.0, 3.5, 4.0),
) -> str:
    """
    Observed against chance across candidate thresholds, as a printable table.

    For picking a threshold with the consequences visible, rather than
    inheriting 2.0 from a paper whose baseline geometry was different.
    """

    z = counts["z"]
    control = counts["control_z"]
    n_roi = counts["n_roi"]

    real = z[np.isfinite(z)]
    null = control[np.isfinite(control)]

    rows = [
        f"{'|z|':>5} {'excited obs':>12} {'chance':>8} {'ratio':>7}   "
        f"{'suppressed obs':>15} {'chance':>8} {'ratio':>7}",
        "-" * 74,
    ]

    for t in thresholds:
        oe, ce = np.mean(real >= t), np.mean(null >= t)
        os_, cs = np.mean(real <= -t), np.mean(null <= -t)
        rows.append(
            f"{t:>5.1f} {oe * n_roi:>7.1f} gloms {ce * n_roi:>7.1f} "
            f"{oe / ce if ce else np.inf:>6.1f}x   "
            f"{os_ * n_roi:>10.1f} gloms {cs * n_roi:>7.1f} "
            f"{os_ / cs if cs else np.inf:>6.1f}x"
        )

    return "\n".join(rows)


# Colour limits for the response heatmaps, in per-trial z.
#
# Not the same numbers as response_qc's HEATMAP_Z_MIN/MAX, and importing those
# here would be a bug: that z divides a window mean by a single-frame SD and
# runs about 5x smaller, so its +10 ceiling saturates almost the whole field
# on this scale (p95 is +27 on group223) and the heatmap goes flat red.
#
# Fixed rather than per-session, so two sessions can be put side by side. The
# range is asymmetric because the tails are: suppression is bounded by the
# baseline and reaches about -5, while excitation has no ceiling. Values past
# +20 do saturate, which is deliberate -- they are all "very responsive" and
# spending range on separating them costs the contrast where the thresholds
# actually sit (around +1.7 / -3.4).
HEATMAP_Z_MIN = -6.0
HEATMAP_Z_MAX = 20.0


THRESHOLD_NOTE = """thresholds: how they are set
  1. z per ROI per trial = (odor window mean - preceding
     equal-length window mean), over the SD of that same
     quantity measured across trials in the pre-odor period.
     Null SD is 1 by construction, so z is in its own units.
  2. the pre-odor period also yields a control value per
     trial (its two halves differenced), which is what the
     same measurement gives when no odor was delivered.
  3. each tail's threshold is the least extreme z at which
     P(control beyond z) / P(observed beyond z) <= target FDR.
     Searched over the observed values, so every achievable
     operating point is considered.
  Tails are solved separately. They are not mirror images:
  far more glomeruli are excited than suppressed, and one
  threshold imposed on both charges the sparse tail the
  dense tail's error budget."""


def response_figure(
    path,
    *,
    deglobal: str | None = "pc1",
    target_fdr: float = TARGET_FDR,
    prefer_detrended: bool = True,
    save: bool = True,
    out_path=None,
):
    """
    QC display driven by the per-trial z, not the sham-null one.

    Six panels. The trial-resolved heatmap leads and takes the full width
    because it is the only one where trial-to-trial reliability is visible --
    an odor average looks identical whether a response happened every time or
    once at ten times the amplitude.

    Population sparsity is reported as excited and suppressed percentages with
    an error bar over trials, rather than as Treves-Rolls. The unit is one
    trial, so the spread is real and measured rather than a property of a
    trial-averaged matrix, and the chance rate from the pre-odor control is
    drawn beside it -- a 3% suppressed share means one thing against a 0.1%
    chance rate and nothing at all against a 3% one. Treves-Rolls is kept for
    lifetime sparseness, where a continuous measure over odors is what is
    wanted and there is no per-trial unit to take a spread over.
    """

    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    from .h5io import open_h5
    from .response_qc import treves_rolls

    path = Path(path)

    with open_h5(path) as f:
        roi, trace_source = load_roi_traces(f, prefer_detrended=prefer_detrended)
        odor_ids = f["trials/odor_id"][:]
        states = f["trials/state"][:]
        state_levels = [s.decode() if isinstance(s, bytes) else str(s)
                        for s in f["trials/state_levels"][:]]
        on_frames = f["trials/odor_on_frame"][:]
        exp_name = str(f.attrs["exp_name"])
        mask_digest = str(f.attrs["mask_hash"])
        manipulation = read_manipulation(f)

    calls = trial_calls(
        roi, odor_on_frames=on_frames, odor_ids=odor_ids,
        deglobal=deglobal, target_fdr=target_fdr,
    )

    z = calls["z"]
    keys = np.unique(odor_ids)

    levels, rank = _display_order(state_levels)

    blocks = {"pooled": np.ones(len(odor_ids), dtype=bool)}
    for name in levels:
        blocks[name] = states == state_levels.index(name)

    sparsity = {n: population_sparsity(calls, odor_ids, trials=m)
                for n, m in blocks.items()}

    def averaged_by(mask):
        return np.stack(
            [np.nanmean(z[:, (odor_ids == k) & mask], axis=1) for k in keys],
            axis=1,
        )

    averages = {n: averaged_by(m) for n, m in blocks.items()}
    lifetime = {n: treves_rolls(a, axis=1) for n, a in averages.items()}

    # ROI order, shared by every heatmap so a row means the same thing in each.
    order = np.lexsort((
        -np.nanmax(np.abs(np.nan_to_num(z)), axis=1),
        np.nanargmax(np.abs(np.nan_to_num(averages["pooled"])), axis=1),
    ))

    norm = TwoSlopeNorm(vmin=HEATMAP_Z_MIN, vcenter=0.0, vmax=HEATMAP_Z_MAX)

    fig = plt.figure(figsize=(18, 17.5))
    grid = fig.add_gridspec(4, 3, height_ratios=[1.25, 1, 1, 0.8],
                            hspace=0.45, wspace=0.26)

    # 1. every trial, columns grouped by odor then by block in display order
    ax = fig.add_subplot(grid[0, :])
    columns = np.lexsort((np.arange(len(odor_ids)), rank[states], odor_ids))
    image = ax.imshow(z[np.ix_(order, columns)], aspect="auto", cmap="RdBu_r",
                      norm=norm, interpolation="nearest")
    fig.colorbar(image, ax=ax, label="per-trial z", pad=0.01)

    sorted_odor, sorted_rank = odor_ids[columns], rank[states][columns]
    for edge in np.flatnonzero(np.diff(sorted_odor)) + 1:
        ax.axvline(edge - 0.5, color="black", lw=1.2)
    for edge in np.flatnonzero((np.diff(sorted_rank) != 0)
                               & (np.diff(sorted_odor) == 0)) + 1:
        ax.axvline(edge - 0.5, color="0.4", lw=0.8, ls=":")

    ax.set_xticks([float(np.mean(np.flatnonzero(sorted_odor == k))) for k in keys])
    ax.set_xticklabels([str(int(k)) for k in keys])
    ax.set_xlabel(f"trial, grouped by odor  (dotted = {' | '.join(levels)})")
    ax.set_ylabel("ROI")
    ax.set_title("single-trial z")

    # 2. odor averages: pooled first, then each block in display order. Same
    # ROI order and colour scale throughout, so the blocks can be read against
    # each other and against the pooled panel by eye.
    for column, name in enumerate((["pooled"] + list(levels))[:3]):
        ax = fig.add_subplot(grid[1, column])
        image = ax.imshow(averages[name][order], aspect="auto", cmap="RdBu_r",
                          norm=norm, interpolation="nearest")
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([str(int(k)) for k in keys], fontsize=7)
        ax.set_xlabel("odor id")
        ax.set_ylabel("ROI (same order)" if column == 0 else "")
        ax.set_title(f"trial-averaged z ({name})")
        fig.colorbar(image, ax=ax, label="z" if column == 2 else "")

    # 3. population sparsity, pooled
    _sparsity_panel(fig.add_subplot(grid[2, 0]), sparsity["pooled"], keys,
                    title="population sparsity (pooled)")

    # 4. population sparsity, by block
    _sparsity_panel(fig.add_subplot(grid[2, 1]),
                    {n: sparsity[n] for n in levels}, keys,
                    title="population sparsity by block", grouped=True)

    # 5. lifetime sparseness by block
    ax = fig.add_subplot(grid[2, 2])
    values = lifetime["pooled"]
    ax.hist(values[np.isfinite(values)], bins=20, range=(0, 1),
            color="0.85", zorder=0, label="pooled")
    for name, colour in zip(levels, ("steelblue", "indianred", "seagreen")):
        values = lifetime[name]
        ax.hist(values[np.isfinite(values)], bins=20, range=(0, 1),
                histtype="step", lw=2, color=colour, label=name)
    ax.set_xlabel("lifetime sparseness (Treves-Rolls)")
    ax.set_ylabel("glomeruli")
    ax.set_title("selectivity by block")
    ax.legend(fontsize=8)

    # 6. how the thresholds were set
    ax = fig.add_subplot(grid[3, 0]); ax.axis("off")
    ax.text(0.0, 1.0, THRESHOLD_NOTE, va="top", ha="left",
            family="monospace", fontsize=7.4, transform=ax.transAxes)

    # 7. numbers
    ax = fig.add_subplot(grid[3, 1]); ax.axis("off")
    check = split_half_control(calls)
    thresholds = calls["thresholds"]
    pooled = sparsity["pooled"]
    ax.text(
        0.0, 1.0, "\n".join([
            f"{exp_name}   mask {mask_digest}",
            f"manipulation: {manipulation or 'not recorded'}",
            f"{z.shape[0]} ROIs x {len(keys)} odors x {z.shape[1]} trials",
            f"traces: {trace_source}",
            f"window: {calls['window_frames']} frames each side",
            f"global component removed: {deglobal or 'none'}",
            "",
            f"target FDR {target_fdr:.0%} per tail",
            f"  excited    z >= {thresholds['excited']['z_threshold']:+.2f}",
            f"  suppressed z <= {thresholds['suppressed']['z_threshold']:+.2f}",
            "",
            "median across odors:",
            f"  excited    {np.nanmedian(pooled['excited']['mean']):.1%}"
            f"   chance {pooled['chance']['excited']:.2%}",
            f"  suppressed {np.nanmedian(pooled['suppressed']['mean']):.1%}"
            f"   chance {pooled['chance']['suppressed']:.2%}",
            "",
            "split-half control (truth is zero):",
            f"  {check['false_positive_rate']:.2%} called, "
            f"kurtosis {check['kurtosis']:.2f}",
        ]), va="top", ha="left", family="monospace", fontsize=8,
        transform=ax.transAxes,
    )

    # 8. what was removed. Worth showing rather than only naming: this is the
    # per-trial strength of the field-wide component, and a block difference
    # in it is a difference in state, not in odor coding.
    ax = fig.add_subplot(grid[3, 2])
    component = calls.get("global_component")
    if component is None:
        ax.axis("off")
    else:
        for name, colour in zip(levels, ("steelblue", "indianred", "seagreen")):
            mask = blocks[name]
            ax.plot(np.flatnonzero(mask), component[mask], ".", ms=3,
                    color=colour, label=name)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel("trial")
        ax.set_ylabel("component (a.u.)")
        ax.set_title(f"removed global component ({deglobal})")
        ax.legend(fontsize=7)

    fig.suptitle(f"response QC - {path.name}", y=0.995, fontsize=13)

    out = Path(out_path) if out_path else path.with_suffix("").with_name(
        path.with_suffix("").name + "_responsefig.png"
    )
    if save:
        fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)

    return {"figure": str(out), "calls": calls, "sparsity": sparsity,
            "lifetime": lifetime, "averages": averages, "levels": levels,
            "split_half": check, "trace_source": trace_source,
            "manipulation": manipulation}


# Blocks are displayed in this order when present, whatever order the round
# stores them in. These sessions encode state_levels as [post, pre], so every
# panel came out with the manipulation ahead of its own baseline.
PREFERRED_STATE_ORDER = ("pre", "post")


def _display_order(state_levels: list) -> tuple[list, np.ndarray]:
    """
    Block names in reading order, and a display rank per state code.

    Fixed once here rather than at each plotting call, so the heatmap columns,
    the grouped bars, the histograms and the legends cannot disagree about
    which block comes first. Levels not in PREFERRED_STATE_ORDER keep their
    stored order and follow the known ones.
    """

    known = [n for n in PREFERRED_STATE_ORDER if n in state_levels]
    rest = [n for n in state_levels if n not in PREFERRED_STATE_ORDER]
    levels = known + rest

    return levels, np.array([levels.index(n) for n in state_levels])


def _sparsity_panel(ax, sparsity, keys, *, title, grouped=False):
    """
    Excited above the axis, suppressed below, error bars over trials.

    Mirroring about zero rather than stacking keeps both tails readable when
    one is twenty times the other, which is the usual case. Chance is drawn as
    a dashed line on each side, because a percentage with no chance rate
    beside it cannot be interpreted.
    """

    x = np.arange(len(keys))

    if not grouped:
        blocks = {"": sparsity}
        width = 0.6
    else:
        blocks = sparsity
        width = 0.8 / len(blocks)

    colours = (("indianred", "steelblue"), ("darkred", "navy"))

    for i, (name, block) in enumerate(blocks.items()):
        offset = 0.0 if not grouped else (i - (len(blocks) - 1) / 2) * width
        up, down = colours[i % len(colours)]

        ax.bar(x + offset, block["excited"]["mean"], width * 0.92,
               yerr=block["excited"]["sem"], color=up, capsize=2,
               error_kw={"lw": 0.8}, label=f"{name} excited".strip())
        ax.bar(x + offset, -block["suppressed"]["mean"], width * 0.92,
               yerr=block["suppressed"]["sem"], color=down, capsize=2,
               error_kw={"lw": 0.8}, label=f"{name} suppressed".strip())

    chance = list(blocks.values())[0]["chance"]
    ax.axhline(chance["excited"], color="0.3", ls="--", lw=0.9)
    ax.axhline(-chance["suppressed"], color="0.3", ls="--", lw=0.9)
    ax.axhline(0, color="black", lw=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(k)) for k in keys], fontsize=7)
    ax.set_xlabel("odor id")
    ax.set_ylabel("suppressed  <-  fraction of glomeruli  ->  excited")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2, loc="upper left",
              framealpha=0.85, borderpad=0.3)
    ax.margins(y=0.18)

    labels = ax.get_yticks()
    ax.set_yticklabels([f"{abs(v):.0%}" for v in labels])
