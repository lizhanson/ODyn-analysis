"""
Sniff frequency from the thermocouple, as a trace rather than a per-trial number.

The rig records respiration at 5 kHz for the whole session, in two columns:
`[:, 0]` is the raw thermocouple and `[:, 1]` is the same signal bandpassed
with the cutoffs listed in `/filter_changes` (0.5-20 Hz on m466). The filtered
column is what this uses; the raw one is kept for quality scoring, since a
bandpass hides exactly the failure it needs to detect.

Frequency is estimated from **inhalation onsets**, not from a Hilbert phase.
Sniffing is not sinusoidal -- inhalation is fast and exhalation slow -- and an
analytic phase on a sawtooth-like waveform reports a frequency that swings
within each cycle. Onsets are also the physiological event: odor reaches the
epithelium on inhalation, so the interval between onsets is the quantity the
olfactory literature reports, and 1/interval is directly interpretable.

Output is a continuous trace at the imaging frame times, not an average over
a fixed window. Odor presentation drives a dynamic change in sniffing in awake
mice -- a burst at onset, often a second shift at offset -- so any fixed
pre/post window bakes in an assumption about when that happens. A trace lets
the window be chosen afterwards, and lets the time course itself be the
result.

**Quality.** A thermocouple that has drifted out of the nostril still produces
a signal; it is just no longer respiration. Scoring is per second, from the
raw column, as the share of power inside the sniff band. A displaced probe
loses band power to broadband noise and drift while its total power may not
change at all, which is why the score is a ratio rather than an amplitude.

Two traces come back, deliberately:

    frequency         every estimate, including across bad stretches
    frequency_masked  the same, NaN wherever quality falls below threshold

Keeping both means a downstream average can be recomputed at a different
threshold, and that a gap is visibly a gap rather than silently absent.
"""

from __future__ import annotations

import numpy as np

# Sniff band. Mice breathe ~2-4 Hz at rest and sniff to ~12 Hz; 15 Hz leaves
# headroom without admitting the 60 Hz line or the fast edge of the filter.
SNIFF_BAND_HZ = (0.5, 15.0)

# Cutoffs for the bandpass this module applies to the RAW column, rather than
# using the rig's pre-filtered `respiration[:, 1]`.
#
# Three reasons not to use the stored column. Its cutoffs are not constant
# across sessions -- `/filter_changes` reads 0.5-20 Hz on m466 and 0.5-15 Hz
# on m472 -- so traces filtered differently are not comparable, and nothing
# downstream would notice. It carries a +12 ms lag against a zero-phase
# filter, which shifts every onset by the same amount and so biases any
# alignment to odor onset. And a signal riding a slow upward drift may never
# cross zero at all, losing the breath entirely; a slightly higher high-pass
# corner recovers those (5.09 -> 5.64 crossings/s on m472).
#
# The high-pass sits at 1.0 Hz rather than 1.5: anaesthetised mice breathe
# down to 1-2 Hz, and a corner at 1.5 would attenuate the post block of every
# ket/xyl session while looking fine on the awake one.
# The low-pass sits at 50 Hz, far above the sniff band, because it is not
# there to define the band -- it is there to keep the inhalation edge sharp.
# A fast bout is 8-12 Hz with a sawtooth-like waveform, so a 20 Hz corner
# passes only the fundamental and first harmonic and rounds the onset off
# until it stops crossing zero.
#
# Measured on m472 across cutoffs, comparing a quiet stretch (60-66 s) with a
# fast bout (68-71 s):
#
#     band     quiet    bout
#     1-15      3.33    5.67
#     1-20      3.33    6.00
#     1-40      3.33    6.67
#     1-50      3.33    7.00
#     1-60      3.33    7.00
#     1-80      3.83    7.00   <- quiet stretch rises: admitting noise
#
# The quiet rate is flat to 60 Hz while the bout climbs, so the extra onsets
# are recovered sharpness rather than noise. 50 rather than 60 because mains
# sits at 60 Hz at 5.8x the local median here, and a Butterworth corner is
# only -3 dB at the corner; 50 gives the same counts without it.
FILTER_BAND_HZ = (1.0, 50.0)
FILTER_ORDER = 2

# Physiological bounds on a single inter-breath interval. Anything outside is
# a missed or spurious onset rather than a real breath, and is dropped before
# interpolation instead of being allowed to pull the trace.
MIN_FREQ_HZ, MAX_FREQ_HZ = 0.5, 15.0

# Comparison band for the quality score: what the signal looks like where
# there is no respiration. Above the 20 Hz filter cutoff and below the
# digitisation noise floor, which on this rig holds 56% of raw power above
# 100 Hz. The score is in-band power over this, so it is an SNR and collapses
# toward 1 when the probe stops seeing breath -- unlike a share-of-total,
# which cannot exceed ~0.02 here however good the signal is, because 41% of
# raw power sits below 0.2 Hz as slow drift.
NOISE_BAND_HZ = (20.0, 100.0)

# Hysteresis: how far below zero the signal must go between accepted onsets,
# as a fraction of a typical trough. Without it every noise wiggle across zero
# counts -- on m466 the bare crossing rate is 9.35 Hz against an FFT peak at
# 3.22 Hz. This rejects crossings that are not part of a full breath while
# leaving the timing of real onsets untouched.
#
# The trough is the 10th percentile of the signal, NOT a MAD. `1.4826 * MAD`
# is the Gaussian-calibrated estimate of sigma and does not transfer to a
# periodic waveform: for a pure sine it evaluates to 1.047 x the amplitude, so
# a floor placed there is never reached and every onset after the first is
# rejected. A percentile is shape-agnostic and scales with whatever the
# waveform actually does.
# 1.1 is set from m466's spectrum, not from theory: 50% of in-band power sits
# at 1.5-4 Hz and 29% at 4-8 Hz (real sniff bouts), so a median instantaneous
# rate near 3-4.5 Hz is what the signal supports. 0.5 gave 8.7 Hz -- counting
# noise wiggles -- and 1.4 gave 2.7 Hz, missing bout breaths. Verify against
# the waveform on a new rig with `onset_figure` rather than inheriting it.
# The floor is applied to an AMPLITUDE-NORMALISED signal, not the raw one.
# Breath amplitude is not stationary: on m466 the local 90th percentile of
# |signal| swings 4x within a session (CV 79%, p10/p90 = 0.24), against 19%
# on the clean m472. One global floor therefore cannot suit both the quiet
# and the loud stretches of the same recording -- set deep enough to reject
# noise in a loud stretch, it rejects real breaths in a quiet one, which is
# undercounting on a clean session and overcounting on a noisy one at the
# same parameter value. Dividing by a running envelope first puts the
# threshold in units of local amplitude, where it means the same thing
# everywhere.
# Width of the running median subtracted before crossing detection. The 1 Hz
# high-pass does not remove everything: a residual baseline offset of up to
# 23% of local amplitude survives it on m472, and where the baseline wanders
# upward a real breath may never cross zero at all, so the onset is lost
# outright rather than mistimed.
#
# 0.30 s, measured on m472 across a quiet stretch, a fast bout, and a wobbly
# tail (onsets/s):
#
#     window     quiet   bout    end
#     none        3.33   7.00   4.00
#     0.20s       3.83   7.00   7.00   <- quiet rises: tracking noise
#     0.25s       3.33   6.67   7.00
#     0.30s       3.33   7.00   7.00
#     0.40s       3.33   6.67   6.00
#     0.50s       3.33   6.00   4.00
#
# Shorter and the filter starts following the noise; longer and it spans
# several breaths of a fast bout and begins subtracting the signal itself.
# 0.30 s is the one width that leaves the quiet and bout rates untouched
# while recovering the tail.
BASELINE_MEDIAN_S = 0.30

# Breaths in the running median used to smooth the rate trace. Smoothing is
# in the breath domain, not the time domain: a boxcar of fixed seconds spans
# a different number of breaths depending on how fast the animal is going, so
# it smooths a bout harder than a quiet stretch. A median rather than a mean
# because odor onset drives a step change in rate, and a mean bleeds that
# step across the breaths either side of it -- which is the feature the trace
# exists to show. 3 is enough to remove single-breath jitter while leaving a
# step at full height after one breath.
SMOOTH_BREATHS = 3

ENVELOPE_WINDOW_S = 2.0
ENVELOPE_PERCENTILE = 90.0

# 0.7 in normalised units. Checked against each session's own spectral peak:
# m472 3.66 vs 3.60 Hz, m465 3.27 vs 3.40, m462 4.06 vs 3.00 (bouts lift the
# mean above the mode, as expected). It does NOT rescue m466, which reads
# 6.08 against 2.20 -- that session is too noisy to detect breaths in at any
# threshold, and the quality mask rather than this parameter is what handles
# it.
TROUGH_PERCENTILE = 10.0
MIN_EXCURSION_FRAC = 0.7

# Resolution of the quality score. One second is long enough to hold at least
# one breath at the slowest plausible rate and short enough to catch a probe
# that shifts mid-session.
QUALITY_WINDOW_S = 1.0

# Share of in-band power below which a window is called bad. Set from the
# data rather than from theory -- see `quality_report` for what a session's
# distribution looks like before choosing.
# In-band / out-of-band power. Set from where the sessions actually separate,
# not from theory. Per-second windows retained, over 180 s of each session:
#
#                        >1.5   >3.0   >5.0   >8.0
#     m472 07-17  good    99%    99%    97%    91%
#     m462 07-21  good   100%   100%    98%    96%
#     m465 07-23  good    99%    98%    97%    89%
#     m466 07-27  mid     92%    87%    79%    55%
#     m466 07-16  bad     83%    42%    32%    29%
#     m462 07-17  bad     88%    34%     5%     1%
#
# 1.5 was far too lenient: it kept 88% of m462 07-17, whose spectral peak sits
# at 0.80 Hz -- not a breathing mouse but the low edge of the 0.5 Hz filter.
# 5.0 keeps 97-98% of the good sessions and 5% of that one. 8.0 separates no
# better and starts costing real data.
#
# This is a per-window cut, so it masks bad stretches inside an otherwise fine
# session. It is not a substitute for looking at the session-level retention:
# m466 07-16 keeping only 32% is a signal that the recording as a whole is
# not worth using, which no per-window threshold can express.
QUALITY_THRESHOLD = 5.0

# Session-level retention below which the whole recording is suspect, not just
# the masked stretches. A per-window cut can say "these seconds are bad"; it
# cannot say "this session is not worth using", and the two need separating.
# m462 07-17 retains 5% and is unambiguous, but m466 07-16 retains 32% and
# still yields a masked median of 2.36 Hz that looks perfectly reasonable --
# built on a third of the data. That is the case this flag exists to catch.
MIN_SESSION_RETENTION = 0.70


def bandpass(x: np.ndarray, *, rate_hz: float, band=FILTER_BAND_HZ,
             order: int = FILTER_ORDER) -> np.ndarray:
    """
    Zero-phase Butterworth bandpass. See FILTER_BAND_HZ for why this exists
    rather than using the rig's stored filtered column.

    `filtfilt`, not `lfilter`: a causal filter delays the signal, and the
    delay is frequency-dependent, so onsets would move by an amount that
    depends on how fast the animal is breathing.
    """

    from scipy.signal import butter, filtfilt

    nyquist = rate_hz / 2.0
    b, a = butter(order, [band[0] / nyquist, band[1] / nyquist], btype="band")

    return filtfilt(b, a, np.asarray(x, dtype=np.float64))


def inhalation_onsets(
    signal: np.ndarray, *, rate_hz: float, band=SNIFF_BAND_HZ,
    min_excursion_frac: float = MIN_EXCURSION_FRAC,
) -> np.ndarray:
    """
    Sample indices of inhalation onsets: upward zero crossings of the
    bandpassed signal.

    A zero crossing rather than a peak. Peak height varies with how well the
    probe sits and with flow rate, so a peak finder needs a prominence
    threshold that then has to track those changes; the crossing is where the
    signal changes sign and needs no amplitude parameter at all. On a
    bandpassed trace the crossing sits at the steepest part of the cycle,
    which is also where it is least sensitive to noise.
    """

    x = np.asarray(signal, dtype=np.float64)
    x = x - np.nanmean(x)

    finite = np.isfinite(x)
    if not finite.all():
        x = np.interp(np.arange(x.size), np.flatnonzero(finite), x[finite])

    # Local baseline out first: see BASELINE_MEDIAN_S. Applied before the
    # envelope so both the crossing test and the amplitude test see the same
    # baseline-corrected signal -- correcting only the crossings admits noise
    # in quiet stretches, measurably (3.33 -> 3.50 onsets/s on m472).
    x = x - _running_median(x, rate_hz)
    x = x / _envelope(x, rate_hz)

    crossings = np.flatnonzero((x[:-1] <= 0) & (x[1:] > 0))

    if crossings.size < 2:
        return crossings

    # An onset counts only if the signal reached the trough of a real breath
    # since the last one. See TROUGH_PERCENTILE / MIN_EXCURSION_FRAC.
    trough = np.percentile(x, TROUGH_PERCENTILE)
    floor = min_excursion_frac * min(trough, 0.0)

    # `last_low[i]` = most recent sample at or below the floor, at or before i
    last_low = np.maximum.accumulate(np.where(x <= floor, np.arange(x.size), -1))

    keep = []
    min_gap = rate_hz / MAX_FREQ_HZ

    for sample in crossings:
        if keep and sample - keep[-1] < min_gap:
            continue
        if keep and last_low[sample] <= keep[-1]:
            continue          # never went low enough since the last onset
        keep.append(sample)

    return np.array(keep, dtype=int)


def _running_median(x: np.ndarray, rate_hz: float,
                    window_s: float = BASELINE_MEDIAN_S) -> np.ndarray:
    """Local baseline: a median over `window_s`, tracking wander not breaths."""

    from scipy.ndimage import median_filter

    size = max(int(window_s * rate_hz) | 1, 3)

    return median_filter(x, size=size, mode="nearest")


def _envelope(x: np.ndarray, rate_hz: float,
              window_s: float = ENVELOPE_WINDOW_S) -> np.ndarray:
    """
    Running amplitude of `x`, one value per sample by interpolation.

    A high percentile of |x| rather than an RMS: RMS is pulled down by the
    long quiet exhalation between breaths, so it tracks duty cycle as much as
    amplitude. The floor at a fifth of the median stops a silent stretch --
    a probe out of the nostril -- from dividing by near-zero and amplifying
    its own noise into apparent breaths.
    """

    step = max(int(window_s * rate_hz), 8)
    n_win = int(np.ceil(x.size / step))

    padded = np.pad(x, (0, n_win * step - x.size), constant_values=np.nan)
    amplitude = np.nanpercentile(
        np.abs(padded.reshape(n_win, step)), ENVELOPE_PERCENTILE, axis=1
    )

    floor = np.nanmedian(amplitude) * 0.2
    amplitude = np.maximum(amplitude, floor if np.isfinite(floor) else 1.0)

    centres = (np.arange(n_win) + 0.5) * step

    return np.interp(np.arange(x.size), centres, amplitude)


def band_power_quality(
    raw: np.ndarray, *, rate_hz: float, window_s: float = QUALITY_WINDOW_S,
    band=SNIFF_BAND_HZ, noise_band=NOISE_BAND_HZ,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-window share of power inside the sniff band, from the RAW column.

    Returns `(centres_s, score)`. Scored on the raw signal on purpose: the
    filtered column has already had everything outside 0.5-20 Hz removed, so
    a probe sitting in free air would score perfectly there while carrying no
    respiration at all. The ratio on the raw signal is what distinguishes a
    breathing trace from drift and broadband noise.
    """

    x = np.asarray(raw, dtype=np.float64)
    step = int(round(window_s * rate_hz))

    if step < 8:
        raise ValueError(f"window_s={window_s} is under 8 samples at {rate_hz} Hz.")

    n_win = x.size // step
    x = x[:n_win * step].reshape(n_win, step)
    x = x - x.mean(axis=1, keepdims=True)

    freqs = np.fft.rfftfreq(step, 1.0 / rate_hz)
    power = np.abs(np.fft.rfft(x * np.hanning(step), axis=1)) ** 2

    in_band = (freqs >= band[0]) & (freqs <= band[1])
    off_band = (freqs >= noise_band[0]) & (freqs <= noise_band[1])

    signal = power[:, in_band].mean(axis=1)
    noise = power[:, off_band].mean(axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        score = np.where(noise > 0, signal / noise, 0.0)

    centres = (np.arange(n_win) + 0.5) * window_s

    return centres, score


def instantaneous_frequency(
    raw: np.ndarray,
    *,
    rate_hz: float,
    band=FILTER_BAND_HZ,
    at_s: np.ndarray,
    quality_threshold: float = QUALITY_THRESHOLD,
    window_s: float = QUALITY_WINDOW_S,
    smooth_breaths: int = SMOOTH_BREATHS,
) -> dict:
    """
    Sniff frequency sampled at the times in `at_s`, with and without masking.

    `at_s` is normally the imaging frame times, so the result drops straight
    beside `/traces/roi` and can be indexed by the same trial and frame.

    Frequency is held constant across each breath -- the value at a time is
    1/(interval containing it) -- rather than smoothly interpolated between
    onsets. Interpolating invents a rate the animal never breathed at, and on
    a sharp change at odor onset it smears the transition across the breath
    before it, which is precisely the moment being measured.
    """

    # Filter here rather than taking a pre-filtered column, so every session
    # is treated identically regardless of what the rig happened to store.
    filtered = bandpass(raw, rate_hz=rate_hz, band=band)

    onsets = inhalation_onsets(filtered, rate_hz=rate_hz)

    result = {
        "onsets_s": onsets / rate_hz,
        "n_breaths": int(max(onsets.size - 1, 0)),
        "quality_threshold": quality_threshold,
    }

    at_s = np.asarray(at_s, dtype=np.float64)

    if onsets.size < 2:
        # Same keys as the normal return, so a caller never has to branch on
        # whether a session happened to yield breaths.
        nan = np.full(at_s.shape, np.nan)
        return {**result, "frequency": nan, "frequency_masked": nan.copy(),
                "frequency_smooth": nan.copy(),
                "frequency_smooth_masked": nan.copy(),
                "smooth_breaths": int(smooth_breaths),
                "quality_at_s": np.full(at_s.shape, np.nan),
                "quality_centres_s": np.zeros(0), "quality_score": np.zeros(0),
                "fraction_good": 0.0, "median_hz": float("nan"),
                "flags": ["No inhalation onsets detected at all -- the probe "
                          "was not recording respiration."]}

    starts = onsets[:-1] / rate_hz
    intervals = np.diff(onsets) / rate_hz

    with np.errstate(divide="ignore"):
        rates = 1.0 / intervals

    # Drop implausible intervals rather than clipping them: a clipped value is
    # a number the animal did not produce, and it would be indistinguishable
    # from a real breath at the bound.
    ok = (rates >= MIN_FREQ_HZ) & (rates <= MAX_FREQ_HZ)
    rates = np.where(ok, rates, np.nan)

    # Step-hold: each sample takes the rate of the breath it falls inside.
    index = np.searchsorted(starts, at_s, side="right") - 1
    inside = (index >= 0) & (index < rates.size)

    frequency = np.full(at_s.shape, np.nan)
    frequency[inside] = rates[index[inside]]

    smoothed_rates = _median_over_breaths(rates, smooth_breaths)

    frequency_smooth = np.full(at_s.shape, np.nan)
    frequency_smooth[inside] = smoothed_rates[index[inside]]

    centres, score = band_power_quality(
        raw, rate_hz=rate_hz, window_s=window_s
    )
    quality_at = np.interp(at_s, centres, score, left=np.nan, right=np.nan)

    masked = np.where(quality_at >= quality_threshold, frequency, np.nan)
    masked_smooth = np.where(
        quality_at >= quality_threshold, frequency_smooth, np.nan
    )

    fraction_good = float(np.mean(score >= quality_threshold))

    flags = []
    if fraction_good < MIN_SESSION_RETENTION:
        flags.append(
            f"Only {fraction_good:.0%} of 1 s windows reach SNR "
            f"{quality_threshold:g} (floor {MIN_SESSION_RETENTION:.0%}). The "
            f"masked trace will still return a plausible-looking number, but "
            f"it rests on {fraction_good:.0%} of the session -- treat this "
            f"recording's sniff frequency as unusable rather than masked."
        )

    return {
        **result,
        "flags": flags,
        "frequency": frequency,
        "frequency_masked": masked,
        "frequency_smooth": frequency_smooth,
        "frequency_smooth_masked": masked_smooth,
        "smooth_breaths": int(smooth_breaths),
        "quality_at_s": quality_at,
        "quality_centres_s": centres,
        "quality_score": score,
        "fraction_good": fraction_good,
        "median_hz": float(np.nanmedian(frequency)),
    }


def _median_over_breaths(rates: np.ndarray, k: int) -> np.ndarray:
    """
    Running median over `k` consecutive breath rates, NaN-aware.

    Centred, and shorter at the ends rather than padded: padding would invent
    breaths, and the first and last breath of a recording are exactly where an
    invented neighbour would be least defensible.
    """

    if k <= 1 or rates.size == 0:
        return rates.astype(np.float64)

    half = k // 2
    out = np.full(rates.shape, np.nan)

    for i in range(rates.size):
        window = rates[max(i - half, 0):i + half + 1]
        finite = window[np.isfinite(window)]

        if finite.size:
            out[i] = np.median(finite)

    return out


def quality_report(score: np.ndarray, thresholds=(0.1, 0.2, 0.25, 0.3, 0.4, 0.5)) -> str:
    """
    What each candidate threshold would keep, as a printable table.

    For choosing `QUALITY_THRESHOLD` against a session rather than inheriting
    it: the right cut depends on how the probe sat that day.
    """

    rows = [f"{'SNR threshold':>14}{'% windows kept':>17}", "-" * 31]

    for t in thresholds:
        rows.append(f"{t:>14.2f}{100 * np.mean(score >= t):>16.1f}%")

    rows.append("")
    rows.append(f"score percentiles: " + "  ".join(
        f"p{q}={np.percentile(score, q):.2f}" for q in (1, 5, 25, 50, 75, 95)
    ))

    return "\n".join(rows)


def onset_figure(raw, *, rate_hz, out_path, seconds=(0.0, 12.0),
                 min_excursion_frac=MIN_EXCURSION_FRAC,
                 odor_windows=(), smooth_breaths=SMOOTH_BREATHS):
    """
    Detected onsets on the waveform, with the rate trace and odor windows.

    `MIN_EXCURSION_FRAC` cannot be set from theory -- it depends on how the
    probe sat -- so it has to be checked against the trace it is applied to.
    This is that check.

    `odor_windows` is an iterable of `(on_s, off_s)`. Marking them is not
    decoration: a rate change at odor onset is the measurement, and an
    unmarked bout is indistinguishable from a spontaneous one by eye.
    """

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a, b = int(seconds[0] * rate_hz), int(seconds[1] * rate_hz)
    x = bandpass(np.asarray(raw, dtype=np.float64), rate_hz=rate_hz)[a:b]
    x = x - np.nanmean(x)
    t = np.arange(x.size) / rate_hz + seconds[0]

    onsets = inhalation_onsets(x, rate_hz=rate_hz,
                               min_excursion_frac=min_excursion_frac)

    corrected = x - _running_median(x, rate_hz)
    corrected = corrected / _envelope(corrected, rate_hz)
    floor = min_excursion_frac * min(np.percentile(corrected, TROUGH_PERCENTILE), 0.0)

    fig, ax = plt.subplots(2, 1, figsize=(15, 6.5), sharex=True,
                           gridspec_kw={"height_ratios": [2, 1.1]})

    for on_s, off_s in odor_windows:
        for panel in ax:
            panel.axvspan(on_s, off_s, color="goldenrod", alpha=0.18, zorder=0)
        ax[0].axvline(on_s, color="darkgoldenrod", lw=1.6)
        ax[1].axvline(on_s, color="darkgoldenrod", lw=1.6)

    ax[0].plot(t, corrected, lw=0.8, color="0.3", label="baseline-corrected")
    ax[0].axhline(0, color="black", lw=0.6)
    ax[0].axhline(floor, color="crimson", ls="--", lw=0.9,
                  label=f"rejection floor (frac={min_excursion_frac})")
    for o in onsets:
        ax[0].axvline(t[o], color="steelblue", lw=0.7, alpha=0.85)
    ax[0].set_ylabel("respiration (normalised)")
    ax[0].set_title(
        f"{len(onsets)} onsets in {seconds[1] - seconds[0]:.0f} s"
        + ("   |   shaded = odor" if len(odor_windows) else "")
    )
    ax[0].legend(fontsize=8, loc="upper right")

    if onsets.size > 1:
        rate = rate_hz / np.diff(onsets)
        starts = t[onsets[:-1]]
        ax[1].step(starts, rate, where="post", color="0.6", lw=1.0,
                   label="per breath")
        ax[1].step(starts, _median_over_breaths(rate, smooth_breaths),
                   where="post", color="indianred", lw=2.0,
                   label=f"median of {smooth_breaths}")
        ax[1].legend(fontsize=8, loc="upper left")

    ax[1].set_ylabel("sniff rate (Hz)")
    ax[1].set_xlabel("time (s)")
    ax[1].set_ylim(0, 16)
    ax[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    return out_path


# Share of a trial that may be quality-masked before the trial is flagged.
# A trial with a fifth of its frames missing can still be averaged, but its
# mean is over a different window than its neighbours', and that difference
# is invisible once it is in a group average.
MAX_MASKED_FRACTION = 0.20

# Y limits for the QC figure, in Hz. Fixed rather than autoscaled: a handful
# of trials carrying a 100 Hz outlier otherwise sets the range and flattens
# every real trace to a line near zero. 15 Hz is the detector's own ceiling
# region (MAX_FREQ_HZ is 15), so nothing meaningful is cropped, and a fixed
# range also lets two sessions be put side by side.
FIGURE_Y_HZ = (0.0, 15.0)


def respiration_from_round(
    round_path,
    *,
    sync_path=None,
    out_dir=None,
    smooth_breaths: int = SMOOTH_BREATHS,
    quality_threshold: float = QUALITY_THRESHOLD,
    max_masked: float = MAX_MASKED_FRACTION,
    save: bool = True,
) -> dict:
    """
    Sniff rate per trial, aligned to the imaging frames, written beside the round.

    Runs off the sync file and a written round: the round supplies the trial
    labels, the sync file supplies respiration and the 2p frame clock, and the
    two are joined on the frame clock rather than on nominal timing. On m466
    that gives 224 acquisitions of exactly 545 frames each, matching the
    round's own trial axis one to one.

    Output is `(n_trial, n_frame)` on the round's frame grid, so it indexes
    exactly like `/traces/roi` -- same trial, same frame, no resampling at the
    point of use. What is stored is the smoothed, quality-masked rate, with
    the unmasked version beside it so a downstream analysis can see what was
    dropped rather than inferring it from a gap.

    Trials with more than `max_masked` of their frames masked are flagged
    rather than removed. Removing them silently would make a group average
    quietly rest on fewer trials for some odors than others; flagging leaves
    that decision where it can be seen.
    """

    from pathlib import Path

    from .h5io import open_h5
    from .sync import (frame_onset_samples, group_frames_into_acquisitions,
                       open_sync, read_channel, rising_edges)

    round_path = Path(round_path)

    with open_h5(round_path) as f:
        odor_ids = f["trials/odor_id"][:]
        states = f["trials/state"][:]
        state_levels = [s.decode() if isinstance(s, bytes) else str(s)
                        for s in f["trials/state_levels"][:]]
        trial_ids = f["trials/trial_id"][:]
        on_frames = f["trials/odor_on_frame"][:]
        off_frames = f["trials/odor_off_frame"][:]
        n_pre = int(f.attrs["n_pre"])
        frame_rate = float(f.attrs["frame_rate"])
        exp_name = str(f.attrs["exp_name"])
        acq_ids = f["trials/acq_id"][:] if "trials/acq_id" in f else None
        manipulation = _decode_label(f, "manipulation")

    if sync_path is None:
        sync_path = find_behavior_sync(round_path.parents[2])

    sync = open_sync(sync_path)

    raw = read_channel(sync, "respiration", column=0)
    frames = frame_onset_samples(sync)
    groups = group_frames_into_acquisitions(frames, rate_hz=sync.rate_hz)

    n_trial = len(odor_ids)

    if len(groups) != n_trial:
        raise ValueError(
            f"{len(groups)} acquisitions in {Path(sync_path).name} but "
            f"{n_trial} trials in {round_path.name}. These must correspond "
            f"one to one; check that the sync file matches this session."
        )

    # One frequency estimate for the whole session, sampled at every 2p frame.
    # Estimating per trial instead would restart the breath history at each
    # acquisition boundary and lose the breath spanning it.
    at_s = frames / sync.rate_hz
    result = instantaneous_frequency(
        raw, rate_hz=sync.rate_hz, at_s=at_s,
        quality_threshold=quality_threshold, smooth_breaths=smooth_breaths,
    )

    widths = {len(g) for g in groups}
    if len(widths) != 1:
        raise ValueError(f"Acquisitions differ in length: {sorted(widths)}.")

    n_frame = widths.pop()
    index = np.concatenate([np.searchsorted(frames, g) for g in groups])

    def reshape(values):
        return values[index].reshape(n_trial, n_frame)

    rate = reshape(result["frequency_smooth_masked"])
    rate_unmasked = reshape(result["frequency_smooth"])
    quality = reshape(result["quality_at_s"])

    masked_fraction = np.mean(~np.isfinite(rate), axis=1)
    flagged = masked_fraction > max_masked

    odor_pulse = read_channel(sync, "odorPulse")
    pulse_on = rising_edges(odor_pulse) / sync.rate_hz

    report = {
        "round": round_path.name,
        "sync": Path(sync_path).name,
        "exp_name": exp_name,
        "manipulation": manipulation,
        "n_trial": int(n_trial),
        "n_frame": int(n_frame),
        "frame_rate": frame_rate,
        "n_pre": n_pre,
        "smooth_breaths": int(smooth_breaths),
        "quality_threshold": float(quality_threshold),
        "max_masked_fraction": float(max_masked),
        "n_breaths": result["n_breaths"],
        "median_hz": result["median_hz"],
        "session_fraction_good": result["fraction_good"],
        "n_flagged": int(flagged.sum()),
        "flagged_trials": [int(i) for i in np.flatnonzero(flagged)],
        "n_odor_pulses": int(pulse_on.size),
        "flags": list(result["flags"]),
        "rate": rate,
        "rate_unmasked": rate_unmasked,
        "quality": quality,
        "masked_fraction": masked_fraction,
        "flagged": flagged,
        "odor_ids": odor_ids,
        "states": states,
        "state_levels": state_levels,
        "trial_ids": trial_ids,
        "acq_ids": acq_ids,
        "on_frames": on_frames,
        "off_frames": off_frames,
    }

    if flagged.any():
        report["flags"].append(
            f"{int(flagged.sum())} of {n_trial} trials have more than "
            f"{max_masked:.0%} of their frames quality-masked. Their means "
            f"rest on a different window than the rest, which is invisible "
            f"once they are inside a group average: trials "
            f"{[int(i) for i in np.flatnonzero(flagged)][:8]}"
            + ("..." if flagged.sum() > 8 else "") + "."
        )

    if save:
        out_dir = Path(out_dir) if out_dir else round_path.parent / "aux"
        out_dir.mkdir(parents=True, exist_ok=True)

        stem = round_path.with_suffix("").name
        report["h5"] = str(_write_respiration(out_dir / f"{stem}_respiration.h5", report))
        report["figure"] = str(respiration_figure(
            report, out_dir / f"{stem}_respiration.png"
        ))

    return report


# Channels that identify a behaviour sync file. The legacy per-experiment h5
# is also 5 kHz and also sits near the session, but holds only ImagingWindow
# and OdorDelivery -- it is what the database ingest reads, and matching it
# here would fail later with a bare KeyError on 'respiration'.
SYNC_CHANNELS = ("2pFrameSync", "respiration", "odorPulse")


def find_behavior_sync(session_dir):
    """
    The six-channel behaviour sync file for a session, wherever it lives.

    Two locations are in use: `<session>/sync/` and the mouse directory above
    it. The mouse directory is the older convention, chosen so the file does
    not collide with the legacy per-experiment h5 that the database ingest
    reads -- and that legacy file is the reason this checks channels rather
    than trusting the name. It is 5 kHz, sits in the session directory, and
    carries ImagingWindow and OdorDelivery, so a glob for "*.h5" finds it and
    everything downstream then fails on a missing 'respiration'.
    """

    from pathlib import Path

    import h5py

    session_dir = Path(session_dir)

    searched = [session_dir / "sync", session_dir, session_dir.parent]
    rejected = []

    for folder in searched:
        for candidate in sorted(folder.glob("*.h5")):
            try:
                with h5py.File(candidate, "r") as f:
                    if all(c in f for c in SYNC_CHANNELS):
                        return candidate
                    rejected.append((candidate.name, sorted(f.keys())[:3]))
            except OSError:
                continue

    detail = "".join(f"\n    {n}: {k}" for n, k in rejected[:4])

    raise FileNotFoundError(
        f"No behaviour sync file for {session_dir}. Looked in "
        f"{[str(s) for s in searched]}; needs channels {list(SYNC_CHANNELS)}."
        + (f" Rejected:{detail}" if rejected else "")
    )


def _decode_label(f, column: str) -> str | None:
    """A session-constant coded label out of a round, or None if absent."""

    if f"trials/{column}" not in f:
        return None

    levels = [s.decode() if isinstance(s, bytes) else str(s)
              for s in f[f"trials/{column}_levels"][:]]
    present = sorted({levels[int(c)] for c in f[f"trials/{column}"][:]})

    return present[0] if len(present) == 1 else " + ".join(present)


def _write_respiration(path, report: dict):
    """One self-describing HDF5, laid out like the round it sits beside."""

    import h5py

    from .store import h5_string_dtype

    with h5py.File(path, "w") as f:
        f.attrs["description"] = (
            "Sniff rate per trial on the round's own frame grid. "
            "/respiration/rate is smoothed and quality-masked; "
            "/respiration/rate_unmasked is the same without the mask, so what "
            "was dropped is visible rather than inferred from a gap."
        )
        for key in ("round", "sync", "exp_name", "manipulation", "n_trial",
                    "n_frame", "frame_rate", "n_pre", "smooth_breaths",
                    "quality_threshold", "max_masked_fraction", "n_breaths",
                    "median_hz", "session_fraction_good", "n_flagged"):
            value = report[key]
            f.attrs[key] = "" if value is None else value

        r = f.create_group("respiration")
        r.create_dataset("rate", data=report["rate"], compression="gzip")
        r["rate"].attrs["units"] = "Hz"
        r["rate"].attrs["description"] = (
            "(trial, frame), smoothed over "
            f"{report['smooth_breaths']} breaths and NaN where quality < "
            f"{report['quality_threshold']:g}."
        )
        r.create_dataset("rate_unmasked", data=report["rate_unmasked"],
                         compression="gzip")
        r.create_dataset("quality", data=report["quality"], compression="gzip")
        r["quality"].attrs["description"] = (
            "In-band / out-of-band power of the raw thermocouple, per frame."
        )

        t = f.create_group("trials")
        t.create_dataset("trial_index", data=np.arange(report["n_trial"]))
        t.create_dataset("trial_id", data=report["trial_ids"])
        t.create_dataset("odor_id", data=report["odor_ids"])
        t.create_dataset("state", data=report["states"])
        t.create_dataset("state_levels",
                         data=np.array(report["state_levels"],
                                       dtype=h5_string_dtype()))
        t.create_dataset("odor_on_frame", data=report["on_frames"])
        t.create_dataset("odor_off_frame", data=report["off_frames"])
        t.create_dataset("masked_fraction", data=report["masked_fraction"])
        t.create_dataset("flagged", data=report["flagged"].astype(np.int8))
        t["flagged"].attrs["description"] = (
            f"1 where more than {report['max_masked_fraction']:.0%} of the "
            f"trial is quality-masked."
        )

        if report["acq_ids"] is not None:
            t.create_dataset("acq_id", data=report["acq_ids"])

        f.create_dataset(
            "time_s",
            data=(np.arange(report["n_frame"]) - report["n_pre"])
            / report["frame_rate"],
        )
        f["time_s"].attrs["description"] = "Seconds relative to odor onset."

    return path


def respiration_figure(report: dict, out_path):
    """
    Odor-averaged sniff rate with confidence intervals, pre against post.

    One panel per odor, both blocks overlaid, because the comparison the
    design is built on is within-odor across blocks -- putting them in
    separate figures makes the eye do the subtraction.

    The interval is a 95% CI of the mean (1.96 * SEM), computed per frame over
    the trials that are finite at that frame. The n therefore varies along the
    trace where quality masking bit, which is why it is drawn: a band that
    widens mid-trial is the mask, not the animal.
    """

    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rate = report["rate"]
    odor_ids = np.asarray(report["odor_ids"])
    states = np.asarray(report["states"])
    levels, rank = _display_order(report["state_levels"])

    keys = np.unique(odor_ids)
    time_s = (np.arange(report["n_frame"]) - report["n_pre"]) / report["frame_rate"]
    odor_end = float(np.median(report["off_frames"] - report["on_frames"])) / report["frame_rate"]

    n_col = 4
    n_row = int(np.ceil(len(keys) / n_col))
    # sharey=False deliberately: the bottom row holds a percentage axis, and
    # under sharing its 0-105 range silently overrides the Hz clip on every
    # odor panel -- which is how a 0-15 Hz limit came out drawn as 0-100.
    fig, axes = plt.subplots(n_row + 1, n_col, figsize=(4.2 * n_col, 2.7 * (n_row + 1)),
                             sharex=True, sharey=False)
    axes = np.atleast_2d(axes)

    colours = {levels[0]: "steelblue"}
    if len(levels) > 1:
        colours[levels[1]] = "indianred"

    def band(ax, values, colour, label):
        finite = np.isfinite(values)
        n = finite.sum(axis=0)
        mean = np.where(n > 0, np.nansum(np.nan_to_num(values), axis=0)
                        / np.maximum(n, 1), np.nan)
        sd = np.array([np.nanstd(values[:, i]) if n[i] > 1 else np.nan
                       for i in range(values.shape[1])])
        sem = sd / np.sqrt(np.maximum(n, 1))
        ax.plot(time_s, mean, color=colour, lw=1.6, label=f"{label} (n={int(np.median(n))})")
        ax.fill_between(time_s, mean - 1.96 * sem, mean + 1.96 * sem,
                        color=colour, alpha=0.22, lw=0)

    for i, key in enumerate(keys):
        ax = axes[i // n_col, i % n_col]
        ax.axvspan(0, odor_end, color="goldenrod", alpha=0.15, lw=0)
        ax.axvline(0, color="darkgoldenrod", lw=1.0)

        for name in levels:
            mask = (odor_ids == key) & (states == report["state_levels"].index(name))
            if mask.any():
                band(ax, rate[mask], colours[name], name)

        ax.set_title(f"odor {int(key)}", fontsize=9)
        ax.set_ylim(*FIGURE_Y_HZ)
        if i % n_col:
            ax.set_yticklabels([])
        ax.grid(alpha=0.25)
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(len(keys), n_row * n_col):
        axes[j // n_col, j % n_col].axis("off")

    # Bottom row: all odors pooled, and the masking that shaped it.
    pooled = axes[n_row, 0]
    pooled.axvspan(0, odor_end, color="goldenrod", alpha=0.15, lw=0)
    pooled.axvline(0, color="darkgoldenrod", lw=1.0)
    for name in levels:
        mask = states == report["state_levels"].index(name)
        if mask.any():
            band(pooled, rate[mask], colours[name], name)
    pooled.set_title("all odors pooled", fontsize=9)
    pooled.set_ylim(*FIGURE_Y_HZ)
    pooled.legend(fontsize=7); pooled.grid(alpha=0.25)

    cover = axes[n_row, 1]
    cover.plot(time_s, 100 * np.mean(np.isfinite(rate), axis=0), color="0.3")
    cover.axhline(100 * (1 - report["max_masked_fraction"]), color="crimson",
                  ls="--", lw=0.9)
    cover.set_title("% trials unmasked, per frame", fontsize=9)
    cover.set_ylim(0, 105); cover.grid(alpha=0.25); cover.set_ylabel("%")

    text = axes[n_row, 2]; text.axis("off")
    text.text(0.0, 1.0, "\n".join([
        f"{report['exp_name']}",
        f"manipulation: {report['manipulation'] or 'not recorded'}",
        f"{report['n_trial']} trials x {report['n_frame']} frames",
        f"sync: {report['sync']}",
        "",
        f"breaths detected     {report['n_breaths']}",
        f"session median       {report['median_hz']:.2f} Hz",
        f"quality kept         {report['session_fraction_good']:.0%}",
        f"smoothing            {report['smooth_breaths']} breaths",
        "",
        f"flagged trials       {report['n_flagged']} "
        f"(>{report['max_masked_fraction']:.0%} masked)",
    ]), va="top", ha="left", family="monospace", fontsize=8,
        transform=text.transAxes)

    flags = axes[n_row, 3]; flags.axis("off")
    if report["flags"]:
        import textwrap
        body = "\n\n".join("- " + "\n  ".join(textwrap.wrap(f, 46))
                           for f in report["flags"])
    else:
        body = "no flags"
    flags.text(0.0, 1.0, body, va="top", ha="left", family="monospace",
               fontsize=7.2, transform=flags.transAxes)

    for ax in axes[n_row - 1, :]:
        ax.set_xlabel("time from odor onset (s)")
    for r in range(n_row):
        axes[r, 0].set_ylabel("sniff rate (Hz)")

    fig.suptitle(f"respiration - {report['round']}", y=0.997, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)

    return Path(out_path)


def _display_order(state_levels: list) -> tuple[list, np.ndarray]:
    """Blocks in reading order: pre before post, whatever the round stores."""

    known = [n for n in ("pre", "post") if n in state_levels]
    rest = [n for n in state_levels if n not in ("pre", "post")]
    levels = known + rest

    return levels, np.array([levels.index(n) for n in state_levels])


def extract_respiration(
    sync_path,
    *,
    acq_ids,
    odor_ids,
    states,
    trial_ids=None,
    state_levels=("pre", "post"),
    exp_name: str = "",
    manipulation: str | None = None,
    out_dir=None,
    smooth_breaths: int = SMOOTH_BREATHS,
    quality_threshold: float = QUALITY_THRESHOLD,
    max_masked: float = MAX_MASKED_FRACTION,
    save: bool = True,
) -> dict:
    """
    Sniff rate per acquisition, from the sync file alone.

    This is the default path, and it deliberately does not need a round. A
    session that has only been motion-corrected is exactly when someone wants
    to know whether the respiration is usable -- before spending an hour on
    segmentation, not after. Everything the round would have supplied is
    recoverable without it: frame times and the valve pulse are both in the
    sync file at 5 kHz, so `sync.py` anchors odor onset to a frame directly,
    which is ground truth and needs no assumption about frame rate or
    baseline length. Only the labels come from elsewhere, from the database.

    **`acq_ids` is the join key, and it is required.** Aligning on position
    would be wrong: this runs over every acquisition in the sync file, while
    the round is later extracted with `approved_only=True` and may contain
    fewer. Row *i* here and row *i* there are then different acquisitions, and
    nothing about the shapes would reveal it. `acq_id` is what the database,
    the rig and the mcor filenames all agree on, so it survives that. See
    `align_to_round`.
    """

    from pathlib import Path

    from .sync import (acquisition_odor_windows, frame_onset_samples,
                       group_frames_into_acquisitions, open_sync,
                       pulse_intervals, read_channel)

    sync = open_sync(sync_path)

    raw = read_channel(sync, "respiration", column=0)
    frames = frame_onset_samples(sync)
    groups = group_frames_into_acquisitions(frames, rate_hz=sync.rate_hz)

    odor_ids = np.asarray(odor_ids)
    states = np.asarray(states)
    acq_ids = np.asarray(acq_ids)
    n_trial = odor_ids.size

    if acq_ids.size != n_trial:
        raise ValueError(
            f"{acq_ids.size} acq_ids for {n_trial} trials -- these must "
            f"correspond, since acq_id is the key everything downstream "
            f"joins on."
        )

    if len(groups) != n_trial:
        raise ValueError(
            f"{len(groups)} acquisitions in {Path(sync_path).name} but "
            f"{n_trial} trials supplied."
        )

    # (onset_s, offset_s) pairs, which is what `odor_frame_alignment` wants --
    # bare rising edges are the wrong shape and the wrong units.
    pulses = pulse_intervals(read_channel(sync, "odorPulse"), rate_hz=sync.rate_hz)
    windows = acquisition_odor_windows(groups, pulses, rate_hz=sync.rate_hz)

    if any(w is None for w in windows):
        missing = [i for i, w in enumerate(windows) if w is None]
        raise ValueError(
            f"No valve pulse found inside acquisition(s) {missing[:6]}"
            + ("..." if len(missing) > 6 else "")
            + ". These cannot be aligned to odor onset."
        )

    on_frames = np.array([w[0] for w in windows])
    off_frames = np.array([w[1] for w in windows])

    # Frame rate from the clock itself rather than the nominal value.
    frame_rate = float(sync.rate_hz / np.median(np.diff(frames)))

    at_s = frames / sync.rate_hz
    result = instantaneous_frequency(
        raw, rate_hz=sync.rate_hz, at_s=at_s,
        quality_threshold=quality_threshold, smooth_breaths=smooth_breaths,
    )

    widths = {len(g) for g in groups}
    if len(widths) != 1:
        raise ValueError(f"Acquisitions differ in length: {sorted(widths)}.")

    n_frame = widths.pop()
    index = np.concatenate([np.searchsorted(frames, g) for g in groups])

    def reshape(values):
        return values[index].reshape(n_trial, n_frame)

    rate = reshape(result["frequency_smooth_masked"])
    masked_fraction = np.mean(~np.isfinite(rate), axis=1)
    flagged = masked_fraction > max_masked

    report = {
        "round": f"{exp_name or Path(sync_path).stem} (no round)",
        "sync": Path(sync_path).name,
        "exp_name": exp_name or Path(sync_path).stem,
        "manipulation": manipulation,
        "n_trial": int(n_trial),
        "n_frame": int(n_frame),
        "frame_rate": frame_rate,
        # Onsets vary by a frame between acquisitions; the figure needs one
        # grid, so it uses the median and the spread is reported rather than
        # silently absorbed.
        "n_pre": int(np.median(on_frames)),
        "onset_frame_spread": int(on_frames.max() - on_frames.min()),
        "smooth_breaths": int(smooth_breaths),
        "quality_threshold": float(quality_threshold),
        "max_masked_fraction": float(max_masked),
        "n_breaths": result["n_breaths"],
        "median_hz": result["median_hz"],
        "session_fraction_good": result["fraction_good"],
        "n_flagged": int(flagged.sum()),
        "flagged_trials": [int(i) for i in np.flatnonzero(flagged)],
        "n_odor_pulses": int(len(pulses)),
        "flags": list(result["flags"]),
        "rate": rate,
        "rate_unmasked": reshape(result["frequency_smooth"]),
        "quality": reshape(result["quality_at_s"]),
        "masked_fraction": masked_fraction,
        "flagged": flagged,
        "odor_ids": odor_ids,
        "states": states,
        "state_levels": list(state_levels),
        "trial_ids": (np.arange(n_trial) if trial_ids is None
                      else np.asarray(trial_ids)),
        "acq_ids": np.asarray(acq_ids),
        "on_frames": on_frames,
        "off_frames": off_frames,
    }

    if flagged.any():
        report["flags"].append(
            f"{int(flagged.sum())} of {n_trial} trials have more than "
            f"{max_masked:.0%} of their frames quality-masked."
        )

    if save and out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = report["exp_name"]
        report["h5"] = str(_write_respiration(out_dir / f"{stem}_respiration.h5", report))
        report["figure"] = str(respiration_figure(
            report, out_dir / f"{stem}_respiration.png"
        ))

    return report


def respiration_for_experiment(
    exp_name: str,
    *,
    connection,
    sync_path=None,
    out_dir=None,
    **kwargs,
) -> dict:
    """
    `extract_respiration` with the labels fetched from odyn for one experiment.

    Trials are ordered by `acq_id`, which is the order the acquisitions appear
    in the sync file. `state` is derived from `program_id`: a session runs as
    two programs, the first before the manipulation and the second after, and
    the two carry the same `program_name` -- so the id is the discriminator
    and the name is not.
    """

    import pandas as pd

    trials = pd.read_sql_query(
        "SELECT t.trial_id, t.acq_id, t.odor_id, t.program_id "
        "FROM trials t JOIN experiments e ON e.exp_id = t.exp_id "
        "WHERE e.exp_name = ? AND t.acq_id IS NOT NULL "
        "ORDER BY t.acq_id;",
        connection, params=[exp_name],
    )

    if trials.empty:
        raise ValueError(
            f"No trials with an acq_id for {exp_name!r}. Without acq_id there "
            f"is no key to align the result to a round later."
        )

    programs = sorted(trials.program_id.unique())

    if len(programs) != 2:
        raise ValueError(
            f"{exp_name}: expected 2 programs (pre and post), found "
            f"{len(programs)}: {programs}."
        )

    if sync_path is None:
        raise ValueError("sync_path is required.")

    return extract_respiration(
        sync_path,
        acq_ids=trials.acq_id.to_numpy(),
        odor_ids=trials.odor_id.to_numpy(),
        states=(trials.program_id == programs[1]).astype(int).to_numpy(),
        trial_ids=trials.trial_id.to_numpy(),
        exp_name=exp_name,
        out_dir=out_dir,
        **kwargs,
    )


def align_to_round(aux_path, round_path):
    """
    Row indices joining an aux respiration file to a round, on `acq_id`.

    Returns `(aux_rows, round_rows)`: `rate[aux_rows]` and `roi[:, round_rows]`
    then describe the same acquisitions in the same order.

    Position is not a valid join. The aux file covers every acquisition in the
    sync file; the round covers those that survived `approved_only=True` and
    the per-trial guards. Row *i* in one is not row *i* in the other, and
    nothing about the two shapes would reveal the mismatch -- the arrays would
    broadcast happily and every result would be wrong by however many
    acquisitions were dropped before the first one they disagree on.

    Raises rather than returning a partial join when the round carries no
    `acq_id`: a round written before that column existed cannot be aligned
    this way, and silently falling back to position is the failure this
    function exists to prevent.
    """

    import h5py

    with h5py.File(aux_path, "r") as f:
        if "trials/acq_id" not in f:
            raise ValueError(f"{aux_path} has no /trials/acq_id.")
        aux = f["trials/acq_id"][:]

    with h5py.File(round_path, "r") as f:
        if "trials/acq_id" not in f:
            raise ValueError(
                f"{round_path} has no /trials/acq_id, so it cannot be joined "
                f"on acquisition. Re-extract it; aligning on position instead "
                f"would silently mismatch wherever approval dropped a trial."
            )
        rnd = f["trials/acq_id"][:]

    lookup = {int(a): i for i, a in enumerate(aux)}
    pairs = [(lookup[int(a)], j) for j, a in enumerate(rnd) if int(a) in lookup]

    if not pairs:
        raise ValueError("No acq_id in common between the aux file and round.")

    aux_rows = np.array([p[0] for p in pairs])
    round_rows = np.array([p[1] for p in pairs])

    return aux_rows, round_rows
