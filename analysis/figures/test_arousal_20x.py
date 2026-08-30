import numpy as np

from .arousal_20x import _within_odor


def test_within_odor_residuals_have_zero_median():
    values = np.array([[1., 3., 10., 14.], [2., 4., 20., 24.]])
    odors = np.array([1, 1, 2, 2])
    residual = _within_odor(values, odors)
    np.testing.assert_allclose(np.median(residual[:, :2], axis=1), 0)
    np.testing.assert_allclose(np.median(residual[:, 2:], axis=1), 0)
