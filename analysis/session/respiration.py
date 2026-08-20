"""Sniff frequency from the thermocouple, as a trace rather than a per-trial number."""

from __future__ import annotations

import numpy as np

# Sniff band. Mice breathe ~2-4 Hz at rest and sniff to ~12 Hz; 15 Hz leaves
# headroom without admitting the 60 Hz line or the fast edge of the filter.
SNIFF_BAND_HZ = (0.5, 15.0)

FILTER_BAND_HZ = (1.0, 50.0)
FILTER_ORDER = 2

# Physiological bounds on a single inter-breath interval. Anything outside is
# a missed or spurious onset rather than a real breath, and is dropped before
# interpolation instead of being allowed to pull the trace.
MIN_FREQ_HZ, MAX_FREQ_HZ = 0.5, 15.0

NOISE_BAND_HZ = (20.0, 100.0)

BASELINE_MEDIAN_S = 0.30

SMOOTH_BREATHS = 3

ENVELOPE_WINDOW_S = 2.0
ENVELOPE_PERCENTILE = 90.0

TROUGH_PERCENTILE = 10.0
MIN_EXCURSION_FRAC = 0.7

# Resolution of the quality score. One second is long enough to hold at least
# one breath at the slowest plausible rate and short enough to catch a probe
# that shifts mid-session.
QUALITY_WINDOW_S = 1.0

QUALITY_THRESHOLD = 5.0

MIN_SESSION_RETENTION = 0.70


def bandpass(x: np.ndarray, *, rate_hz: float, band=FILTER_BAND_HZ,
             order: int = FILTER_ORDER) -> np.ndarray:
    """Zero-phase Butterworth bandpass."""

    from scipy.signal import butter, filtfilt

    nyquist = rate_hz / 2.0
    b, a = butter(order, [band[0] / nyquist, band[1] / nyquist], btype="band")

    return filtfilt(b, a, np.asarray(x, dtype=np.float64))


def inhalation_onsets(
    signal: np.ndarray, *, rate_hz: float, band=SNIFF_BAND_HZ,
    min_excursion_frac: float = MIN_EXCURSION_FRAC,
) -> np.ndarray:
    """Sample indices of inhalation onsets: upward zero crossings of the bandpassed signal."""

    x = np.asarray(signal, dtype=np.float64)
    x = x - np.nanmean(x)

    finite = np.isfinite(x)
    if not finite.all():
        x = np.interp(np.arange(x.size), np.flatnonzero(finite), x[finite])

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
    """Running amplitude of `x`, one value per sample by interpolation."""

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
    """Per-window share of power inside the sniff band, from the RAW column."""

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
    """Sniff frequency sampled at the times in `at_s`, with and without masking."""

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
    """Running median over `k` consecutive breath rates, NaN-aware."""

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
    """What each candidate threshold would keep, as a printable table."""

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
    """Detected onsets on the waveform, with the rate trace and odor windows."""

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


MAX_MASKED_FRACTION = 0.20

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
    """Sniff rate per trial, aligned to the imaging frames, written beside the round."""

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


SYNC_CHANNELS = ("2pFrameSync", "respiration", "odorPulse")


def find_behavior_sync(session_dir):
    """The six-channel behaviour sync file for a session, wherever it lives."""

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
    """Odor-averaged sniff rate with confidence intervals, pre against post."""

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
    """Sniff rate per acquisition, from the sync file alone."""

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
    """`extract_respiration` with the labels fetched from odyn for one experiment."""

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
    """Row indices joining an aux respiration file to a round, on `acq_id`."""

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
