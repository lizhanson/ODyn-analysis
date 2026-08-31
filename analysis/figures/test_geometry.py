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


def test_pair_distance_matches_the_matrix_entry():
    from .geometry import diagonal_crossnobis, pair_distance
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(0, .3, (8, 6)), rng.normal(1, .3, (8, 6))])
    labels = np.repeat([0, 1], 8)
    _, matrix = diagonal_crossnobis(x, labels, repeats=40, seed=3)
    assert np.isclose(pair_distance(x, labels, seed=3, repeats=40), matrix[0, 1])


def test_cosine_centroid_distance_ignores_gain():
    from .geometry import cosine_centroid_distance
    x = np.array([[1., 0.], [1., 0.], [4., 0.], [4., 0.]])
    assert np.isclose(cosine_centroid_distance(x, np.repeat([0, 1], 2)), 0.,
                      atol=1e-9)


def test_permutation_null_flags_underpowered_rather_than_extrapolating():
    from .geometry import pair_distance, permutation_null
    rng = np.random.default_rng(2)
    x = np.vstack([rng.normal(0, .3, (8, 20)), rng.normal(3, .3, (8, 20))])
    labels = np.repeat([0, 1], 8)
    observed = pair_distance(x, labels, seed=1, repeats=40)
    result = permutation_null(x, labels, observed, permutations=20, seed=1)
    assert result["p_value"] == result["p_resolution"]
    assert result["underpowered"]


def test_permutation_null_false_positive_rate_is_near_nominal():
    """One noise draw can land anywhere; the rate across draws should not."""
    from .geometry import pair_distance, permutation_null
    labels = np.repeat([0, 1], 8)
    significant = 0
    for seed in range(20):
        x = np.random.default_rng(seed).normal(0, 1, (16, 20))
        observed = pair_distance(x, labels, seed=1, repeats=40)
        result = permutation_null(x, labels, observed, permutations=100, seed=1)
        significant += result["p_value"] <= .05
    assert significant <= 3          # nominal is 1 of 20
