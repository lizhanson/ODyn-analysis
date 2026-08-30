import numpy as np

from .geometry import (cosine_distance_matrix, diagonal_crossnobis,
                       spatiotemporal_crossnobis, time_resolved_crossnobis)


def test_cosine_ignores_common_gain():
    distance = cosine_distance_matrix([[1., 0.], [3., 0.], [0., 1.]])
    assert np.isclose(distance[0, 1], 0.)
    assert np.isclose(distance[0, 2], 1.)


def test_crossnobis_finds_reproducible_feature_difference():
    rng = np.random.default_rng(2)
    a = rng.normal(0, .2, (10, 3))
    b = rng.normal(0, .2, (10, 3)) + np.array([2., 0., 0.])
    levels, distance = diagonal_crossnobis(
        np.vstack([a, b]), np.r_[np.zeros(10), np.ones(10)], repeats=40)
    assert levels.tolist() == [0., 1.]
    assert distance[0, 1] > 10


def test_temporal_geometry_shapes():
    rng = np.random.default_rng(2)
    labels = np.repeat([1, 2], 4)
    traces = rng.normal(size=(8, 3, 6))
    levels, rdms, starts = time_resolved_crossnobis(
        traces, labels, bin_frames=2, repeats=5)
    assert levels.tolist() == [1, 2]
    assert rdms.shape == (3, 2, 2)
    assert starts.tolist() == [0, 2, 4]
    _, full = spatiotemporal_crossnobis(traces, labels, repeats=5)
    assert full.shape == (2, 2)
