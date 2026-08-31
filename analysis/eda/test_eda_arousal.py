import numpy as np
from .arousal import (detect_onsets, odor_free_mask, within_odor_deviation)


def test_odor_free_mask_excludes_odor_and_post_odor():
    time = np.array([[-4., -0.5, 0., 2., 6., 9., 13.]])
    mask = odor_free_mask(time)[0]
    assert mask.tolist() == [True, False, False, False, False, True, True]


def test_within_odor_deviation_removes_the_odor_median():
    values = np.array([1., 3., 10., 14.])
    odor = np.array([7, 7, 8, 8])
    state = np.array(["pre", "pre", "pre", "pre"])
    np.testing.assert_allclose(within_odor_deviation(values, odor, state),
                               [-1., 1., -2., 2.])


def test_within_odor_deviation_separates_states():
    values = np.array([1., 3., 21., 23.])
    odor = np.array([7, 7, 7, 7])
    state = np.array(["pre", "pre", "post", "post"])
    np.testing.assert_allclose(within_odor_deviation(values, odor, state),
                               [-1., 1., -1., 1.])


def test_detect_onsets_finds_a_quiet_to_active_transition():
    signal = np.zeros((1, 40)); signal[0, 20:] = 5.
    mask = np.ones((1, 40), bool)
    onsets = detect_onsets(signal, mask, threshold=1.,
                           min_quiet_samples=10, min_run_samples=5)
    assert onsets.tolist() == [[0, 20]]


def test_detect_onsets_ignores_events_outside_the_odor_free_mask():
    signal = np.zeros((1, 40)); signal[0, 20:] = 5.
    mask = np.ones((1, 40), bool); mask[0, 15:30] = False
    assert len(detect_onsets(signal, mask, threshold=1.,
                             min_quiet_samples=10, min_run_samples=5)) == 0


def test_detect_onsets_requires_a_sustained_crossing():
    signal = np.zeros((1, 40)); signal[0, 20] = 5.     # single-sample blip
    mask = np.ones((1, 40), bool)
    assert len(detect_onsets(signal, mask, threshold=1.,
                             min_quiet_samples=10, min_run_samples=5)) == 0


def test_window_mean_selects_the_right_frames():
    from .stage4_regression import window_mean
    values = np.array([[1., 2., 3., 4.]])
    time = np.array([[-1., 0., 1., 5.]])
    np.testing.assert_allclose(window_mean(values, time, (0., 4.)), [2.5])


def test_standardized_slope_needs_variation():
    from .stage4_regression import standardized_slope
    assert np.isnan(standardized_slope(np.ones(20), np.arange(20.)))
    assert np.isclose(standardized_slope(np.arange(20.), np.arange(20.)), 1.)
