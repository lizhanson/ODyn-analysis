import numpy as np
from .signed import (box_smooth, excursion_thresholds, median_traces,
                    signed_features, suprathreshold_area)

TIME = np.arange(-5., 10., .05)


def _session(n_unit=30, n_trial=60, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1, (n_unit, n_trial, TIME.size))
    odor = np.repeat(np.arange(6), n_trial//6)
    state = np.zeros(n_trial, int)
    return z, odor, state


def test_box_smooth_reduces_noise_and_keeps_shape():
    x = np.random.default_rng(0).normal(size=(4, 200))
    smoothed = box_smooth(x, 10)
    assert smoothed.shape == x.shape
    assert np.nanstd(smoothed) < np.nanstd(x)


def test_box_smooth_ignores_nan():
    x = np.array([[1., np.nan, 1., 1.]])
    assert np.allclose(box_smooth(x, 3), 1.)


def test_biphasic_pair_is_detected_though_its_mean_is_zero():
    """The case the 4 s mean cannot represent."""
    z, odor, state = _session()
    excited = (TIME >= 0) & (TIME < 2)
    suppressed = (TIME >= 2) & (TIME < 4)
    z[0, odor == 1] += 6*excited - 6*suppressed
    result = signed_features(z, TIME, odor, state, 0, np.arange(6))
    assert result["excited"][0, 1] and result["suppressed"][0, 1]
    assert result["biphasic"][0, 1]
    assert abs(result["mean_z"][0, 1]) < .5      # invisible to a signed mean
    assert result["excitation_area"][0, 1] > 0
    assert result["suppression_area"][0, 1] > 0


def test_onset_and_offset_windows_separate():
    z, odor, state = _session()
    z[0, odor == 2] += 6*((TIME >= 4) & (TIME < 8))
    result = signed_features(z, TIME, odor, state, 0, np.arange(6))
    assert result["excited_offset"][0, 2]
    assert not result["excited_onset"][0, 2]


def test_blank_false_positive_rate_tracks_the_tail_probability():
    """Pure noise: the realised rate should sit near the nominal one."""
    z, odor, state = _session(n_unit=200, n_trial=120, seed=3)
    result = signed_features(z, TIME, odor, state, 0, np.arange(6),
                             tail_probability=.05)
    rate = result["excited"].mean()
    assert .01 < rate < .12


def test_thresholds_are_stratified_by_trial_count():
    """More trials per odor means a quieter median trace and a tighter cutoff."""
    rng = np.random.default_rng(1)
    n_unit = 80
    odor = np.array([0]*2 + [1]*16)
    z = rng.normal(0, 1, (n_unit, odor.size, TIME.size))
    state = np.zeros(odor.size, int)
    traces, counts = median_traces(z, odor, state, 0, np.array([0, 1]))
    thresholds = excursion_thresholds(box_smooth(traces, 10), TIME, counts)
    assert thresholds.positive[2] > thresholds.positive[16]


def test_suprathreshold_area_is_zero_below_threshold():
    traces = np.zeros((2, 3, TIME.size))
    positive = np.full((1, 3), 1.); negative = np.full((1, 3), -1.)
    e, s = suprathreshold_area(traces, TIME, positive, negative, (0., 4.))
    assert np.allclose(e, 0.) and np.allclose(s, 0.)
