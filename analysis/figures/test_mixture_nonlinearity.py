import numpy as np

from analysis.figures.mixture_nonlinearity import _family_metrics
from analysis.figures.mixture_temporal_correspondence import crossvalidated_pattern_energy


def test_on_axis_reciprocal_mixtures_have_zero_residual():
    centroids = {
        1: np.array([0., 0.]), 2: np.array([10., 0.]),
        3: np.array([4., 0.]), 4: np.array([6., 0.]),
    }
    result = _family_metrics(centroids, (1, 2), (3, 4))
    assert np.isclose(result["mean_off_axis_residual_rms_z"], 0)
    assert np.isclose(result["best_60_40_residual_rms_z"], 0)


def test_off_axis_mixture_has_nonzero_residual():
    centroids = {
        1: np.array([0., 0.]), 2: np.array([10., 0.]),
        3: np.array([4., 3.]), 4: np.array([6., -3.]),
    }
    result = _family_metrics(centroids, (1, 2), (3, 4))
    assert result["mean_off_axis_residual_rms_z"] > 0


def test_crossvalidated_pattern_energy_recovers_stable_difference():
    rng = np.random.default_rng(2)
    a = rng.normal(0, .1, (8, 3))
    b = rng.normal(0, .1, (8, 3)) + np.array([1., -1., .5])
    value = crossvalidated_pattern_energy(
        np.vstack([a, b]), np.repeat([1, 2], 8), repeats=30, seed=4)
    assert value > .4
