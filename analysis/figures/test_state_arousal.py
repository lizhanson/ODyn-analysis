import numpy as np

from analysis.figures.state_arousal import (_linear_detrend_rows, _odor_center,
                                            _partial_rank_correlation,
                                            _rank_correlation, _zscore)


def test_linear_detrend_removes_trial_order_slope():
    order = np.arange(12, dtype=float)
    values = np.vstack([3 + 2*order, -5 + .25*order])
    residual = _linear_detrend_rows(values, order)
    assert np.nanmax(np.abs(residual)) < 1e-10


def test_odor_center_removes_each_odor_median_timecourse():
    odors = np.array([1, 1, 2, 2])
    values = np.array([[[1., 3.], [3., 5.], [10., 20.], [14., 24.]]])
    centered = _odor_center(values, odors)
    np.testing.assert_allclose(np.median(centered[:, :2], axis=1), 0)
    np.testing.assert_allclose(np.median(centered[:, 2:], axis=1), 0)


def test_rank_correlation_and_robust_centered_zscore():
    x = np.arange(10, dtype=float)
    assert np.isclose(_rank_correlation(x, x), 1)
    z = _zscore(np.array([1., 2., 3., 100.]))
    assert np.median(z) == 0


def test_partial_rank_removes_shared_monotonic_driver():
    rng = np.random.default_rng(3)
    nuisance = np.arange(200, dtype=float)
    x = nuisance + rng.normal(0, 20, len(nuisance))
    y = nuisance + rng.normal(0, 20, len(nuisance))
    assert abs(_partial_rank_correlation(x, y, [nuisance])) < .2
