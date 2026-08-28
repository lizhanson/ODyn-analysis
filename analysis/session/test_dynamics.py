import numpy as np

from .dynamics import odor_versus_baseline, window_dynamics


def _sine(frequency_hz, *, frame_rate=30.0, n_frames=120, amplitude=1.0):
    t = np.arange(n_frames) / frame_rate
    return (amplitude * np.sin(2 * np.pi * frequency_hz * t)).astype(np.float32)


def test_sine_recovers_its_own_frequency_and_ar1_timescale():
    frame_rate, frequency = 30.0, 5.0
    z = _sine(frequency)[None, None, :]
    result = window_dynamics(
        z, onsets=[0], n_frames=120, frame_rate=frame_rate,
    )
    # A sinusoid crosses zero twice per cycle.
    assert abs(float(result.zero_crossing_hz[0, 0]) - 2 * frequency) < .5
    # And its lag-1 autocorrelation is cos(omega * dt) exactly.
    assert abs(float(result.lag1[0, 0]) - np.cos(2 * np.pi * frequency / frame_rate)) < .05


def test_noise_maximises_bidirectionality_and_leaves_no_timescale():
    rng = np.random.default_rng(0)
    z = rng.standard_normal((8, 4, 120)).astype(np.float32)
    result = window_dynamics(z, onsets=[0] * 4, n_frames=120, frame_rate=30.0)
    # This is the reason bidirectionality is never read without its baseline.
    assert float(np.mean(result.bidirectionality)) > .6
    assert float(np.mean(result.tau_s)) < .02


def test_a_purely_positive_deflection_is_not_bidirectional():
    z = np.zeros((1, 1, 120), np.float32)
    z[0, 0, 30:90] = np.hanning(60) * 5
    result = window_dynamics(z, onsets=[0], n_frames=120, frame_rate=30.0)
    assert float(result.bidirectionality[0, 0]) == 0
    assert abs(float(result.range_z[0, 0]) - 5) < .1


def test_shape_statistics_ignore_scale_and_range_tracks_it():
    rng = np.random.default_rng(1)
    z = rng.standard_normal((4, 3, 90)).astype(np.float32)
    plain = window_dynamics(z, onsets=[0] * 3, n_frames=90, frame_rate=25.0)
    scaled = window_dynamics(z * 7, onsets=[0] * 3, n_frames=90, frame_rate=25.0)
    for name in ("bidirectionality", "tau_s", "lag1", "zero_crossing_hz"):
        assert np.allclose(getattr(plain, name), getattr(scaled, name), atol=1e-5)
    assert np.allclose(plain.range_z * 7, scaled.range_z, rtol=1e-4)


def test_offsetting_a_window_leaves_timescale_and_crossings_alone():
    z = _sine(4.0)[None, None, :]
    result = window_dynamics(z, onsets=[0], n_frames=120, frame_rate=30.0)
    shifted = window_dynamics(z + 5, onsets=[0], n_frames=120, frame_rate=30.0)
    assert np.allclose(result.tau_s, shifted.tau_s, atol=1e-5)
    # One crossing of slack: the offset lands in the float32 input, so a sample
    # sitting almost exactly on zero may fall either side of it.
    one_crossing = 1.0 / ((120 - 1) / 30.0)
    assert np.allclose(
        result.zero_crossing_hz, shifted.zero_crossing_hz, atol=one_crossing,
    )


def test_matched_windows_take_the_shorter_of_baseline_and_odor():
    z = np.zeros((2, 3, 300), np.float32)
    on = np.array([120, 120, 120])
    comparison = odor_versus_baseline(
        z, odor_on_frames=on, odor_off_frames=on + 60,
        frame_rate=30.0, baseline_s=4.0,
    )
    # Baseline would allow 120 frames; the 2 s odor caps both windows at 60.
    assert comparison.n_frames == 60
    assert comparison.odor.n_frames == comparison.baseline.n_frames


def test_baseline_window_sits_immediately_before_odor_onset():
    z = np.zeros((1, 1, 300), np.float32)
    z[0, 0, 80:100] = 9          # a deflection inside the baseline window only
    z[0, 0, 120:180] = 0         # the odor window is flat
    comparison = odor_versus_baseline(
        z, odor_on_frames=[120], odor_off_frames=[180],
        frame_rate=30.0, baseline_s=2.0,
    )
    assert comparison.n_frames == 60
    assert float(comparison.baseline.range_z[0, 0]) == 9
    assert float(comparison.odor.range_z[0, 0]) == 0
    assert float(comparison.excess()["range_z"][0, 0]) == -9


def test_windows_that_do_not_fit_are_refused():
    z = np.zeros((1, 2, 50), np.float32)
    for kwargs, fragment in (
        (dict(onsets=[0, 0], n_frames=2, frame_rate=30.0), "at least three frames"),
        (dict(onsets=[0, 40], n_frames=30, frame_rate=30.0), "cannot provide"),
        (dict(onsets=[0, 0], n_frames=10, frame_rate=0), "frame_rate must be positive"),
        (dict(onsets=[0], n_frames=10, frame_rate=30.0), "align with the trial axis"),
    ):
        try:
            window_dynamics(z, **kwargs)
        except ValueError as error:
            assert fragment in str(error), (fragment, str(error))
        else:
            raise AssertionError(f"expected ValueError mentioning {fragment!r}")
