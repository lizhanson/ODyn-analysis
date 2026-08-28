"""Within-window dynamics: how large, how bidirectional, and how fast.

Four scalars per unit and trial -- excursion range, bidirectionality, an AR(1)
timescale, and a zero-crossing rate -- computed identically on the pre-odor
baseline and on the odor window. Every odor value therefore carries a matched
noise reference from the same unit, the same trial, and the same state.

The baseline window is a legitimate control for these statistics in a way it is
not for mean responses. Subtracting the per-trial baseline mean leaves range,
peak ratio, autocorrelation, and crossing rate unchanged, so the comparison is
not contaminated by the centring that produced the z scores in the first place.

Bidirectionality must never be read without its baseline value. A pure-noise
window has peak_positive ~ -peak_negative and so scores near one: the statistic
is *maximised* by noise. Only the odor-minus-baseline difference is meaningful.

Both windows are forced to the same frame count. Range and crossing rate both
grow with window length, so unequal windows would not be comparable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BASELINE_S = 4.0

# Below this lag-1 autocorrelation the AR(1) timescale is shorter than one
# sampling interval and is reported as zero rather than as a negative or
# imaginary number.
_MIN_LAG1 = 1e-6


@dataclass
class WindowDynamics:
    """Per unit and trial. Every array is unit x trial."""

    range_z: np.ndarray
    bidirectionality: np.ndarray
    tau_s: np.ndarray
    lag1: np.ndarray
    zero_crossing_hz: np.ndarray
    n_frames: int
    frame_rate: float


@dataclass
class DynamicsComparison:
    """Matched odor and baseline windows of identical length."""

    odor: WindowDynamics
    baseline: WindowDynamics
    n_frames: int
    frame_rate: float

    def excess(self) -> dict[str, np.ndarray]:
        """Odor minus baseline, the only form in which these are interpretable."""
        return {
            name: getattr(self.odor, name) - getattr(self.baseline, name)
            for name in ("range_z", "bidirectionality", "tau_s", "zero_crossing_hz")
        }


def window_dynamics(z, *, onsets, n_frames, frame_rate) -> WindowDynamics:
    """Fixed-length window anchored on each trial's onset frame."""
    z = np.asarray(z, np.float32)
    onsets = np.asarray(onsets, int)
    if z.ndim != 3:
        raise ValueError("z must be unit x trial x frame")
    if onsets.shape != (z.shape[1],):
        raise ValueError("onsets must align with the trial axis")

    n_frames = int(n_frames)
    if n_frames < 3:
        raise ValueError("a dynamics window needs at least three frames")

    frame_rate = float(frame_rate)
    if not frame_rate > 0:
        raise ValueError("frame_rate must be positive")

    if np.any(onsets < 0) or np.any(onsets + n_frames > z.shape[2]):
        bad = np.flatnonzero((onsets < 0) | (onsets + n_frames > z.shape[2]))
        raise ValueError(
            f"{len(bad)} trial(s) cannot provide a {n_frames}-frame window "
            f"within {z.shape[2]} frames; first trial index {int(bad[0])}."
        )

    n_unit, n_trial = z.shape[0], z.shape[1]
    shape = (n_unit, n_trial)
    span = np.empty(shape, np.float32)
    bidirectional = np.empty(shape, np.float32)
    tau = np.empty(shape, np.float32)
    lag1 = np.empty(shape, np.float32)
    crossing = np.empty(shape, np.float32)

    dt = 1.0 / frame_rate
    duration = (n_frames - 1) * dt

    for trial, start in enumerate(onsets):
        window = z[:, trial, start:start + n_frames]

        # Amplitude and sign structure use the z scores as they stand: they are
        # already centred on this trial's pre-odor baseline, so "above" and
        # "below" mean above and below that baseline.
        high = np.nanmax(window, axis=1)
        low = np.nanmin(window, axis=1)
        span[:, trial] = high - low

        up = np.maximum(high, 0.0)
        down = np.maximum(-low, 0.0)
        larger = np.maximum(up, down)
        bidirectional[:, trial] = np.where(
            larger > 0, np.minimum(up, down) / np.where(larger > 0, larger, 1.0), 0.0
        )

        # Timescale and crossing rate describe the wiggle, not the offset, so
        # they take the window's own mean out first.
        # Double precision here: the sums below run over every frame in the
        # window, and float32 accumulation costs real accuracy in r1.
        wide = np.asarray(window, np.float64)
        centred = wide - np.nanmean(wide, axis=1, keepdims=True)
        centred = np.nan_to_num(centred, nan=0.0)

        power = np.sum(centred * centred, axis=1)
        product = np.sum(centred[:, :-1] * centred[:, 1:], axis=1)
        r1 = np.where(power > 0, product / np.where(power > 0, power, 1.0), np.nan)
        lag1[:, trial] = r1

        usable = np.isfinite(r1) & (r1 > _MIN_LAG1)
        clipped = np.clip(r1, _MIN_LAG1, 1.0 - _MIN_LAG1)
        tau[:, trial] = np.where(
            usable, -dt / np.log(np.where(usable, clipped, 0.5)), 0.0
        )

        flips = np.count_nonzero(np.diff(np.signbit(centred), axis=1), axis=1)
        crossing[:, trial] = flips / duration

    return WindowDynamics(
        range_z=span,
        bidirectionality=bidirectional,
        tau_s=tau,
        lag1=lag1,
        zero_crossing_hz=crossing,
        n_frames=n_frames,
        frame_rate=frame_rate,
    )


def odor_versus_baseline(
    z, *, odor_on_frames, odor_off_frames, frame_rate, baseline_s=BASELINE_S,
) -> DynamicsComparison:
    """The odor window and its matched pre-odor control, same length in frames."""
    on = np.asarray(odor_on_frames, int)
    off = np.asarray(odor_off_frames, int)
    if on.shape != off.shape:
        raise ValueError("odor_on_frames and odor_off_frames must align")
    if np.any(off <= on):
        raise ValueError("every trial needs a positive odor duration")

    n_baseline = int(round(float(baseline_s) * float(frame_rate)))
    # The shortest odor caps the window so that one length serves every trial.
    n_frames = int(min(n_baseline, int(np.min(off - on))))
    if n_frames < 3:
        raise ValueError(
            f"matched window is only {n_frames} frame(s); the shortest odor is "
            f"{int(np.min(off - on))} frames and the baseline is {n_baseline}."
        )

    return DynamicsComparison(
        odor=window_dynamics(z, onsets=on, n_frames=n_frames, frame_rate=frame_rate),
        baseline=window_dynamics(
            z, onsets=on - n_frames, n_frames=n_frames, frame_rate=frame_rate,
        ),
        n_frames=n_frames,
        frame_rate=float(frame_rate),
    )
