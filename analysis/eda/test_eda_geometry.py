import numpy as np
from .geometry import (cosine_pair, crossnobis_pair, permutation_test,
                      pooled_variance, rdm)


def two_conditions(separation=0., n=8, features=40, seed=0, scale=1.):
    rng = np.random.default_rng(seed)
    a = rng.normal(0, scale, (n, features))
    b = rng.normal(0, scale, (n, features))
    b[:, :5] += separation
    return np.vstack([a, b]), np.repeat([0, 1], n)


def test_crossnobis_is_near_zero_without_a_real_difference():
    """The property an uncrossvalidated distance does not have."""
    values = [crossnobis_pair(*two_conditions(0., seed=s), seed=s)
              for s in range(12)]
    assert abs(np.mean(values)) < 0.25


def test_crossnobis_grows_with_real_separation():
    small = crossnobis_pair(*two_conditions(.5, seed=1), seed=1)
    large = crossnobis_pair(*two_conditions(2., seed=1), seed=1)
    assert large > small > 0


def test_crossnobis_downweights_noisy_features():
    """Separation carried by a noisy feature counts for less."""
    rng = np.random.default_rng(0)
    quiet = np.vstack([rng.normal(0, .1, (8, 1)), rng.normal(1, .1, (8, 1))])
    noisy = np.vstack([rng.normal(0, 3., (8, 1)), rng.normal(1, 3., (8, 1))])
    labels = np.repeat([0, 1], 8)
    assert (crossnobis_pair(quiet, labels, seed=0)
            > crossnobis_pair(noisy, labels, seed=0))


def test_returns_nan_with_a_single_trial_condition():
    x = np.random.default_rng(0).normal(size=(3, 5))
    assert np.isnan(crossnobis_pair(x, np.array([0, 1, 1]), seed=0))


def test_cosine_ignores_gain():
    x = np.array([[1., 0.], [1., 0.], [3., 0.], [3., 0.]])
    assert np.isclose(cosine_pair(x, np.repeat([0, 1], 2)), 0., atol=1e-9)


def test_permutation_flags_underpowered_instead_of_extrapolating():
    x, labels = two_conditions(4., seed=2)
    result = permutation_test(x, labels, permutations=20, seed=2)
    assert result.p_value == result.resolution
    assert result.underpowered           # p is at the floor, not a z score


def test_permutation_p_is_large_for_a_null_difference():
    x, labels = two_conditions(0., seed=5)
    result = permutation_test(x, labels, permutations=100, seed=5)
    assert result.p_value > .05
    assert not result.underpowered


def test_rdm_is_symmetric_with_zero_diagonal():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(18, 10))
    labels = np.repeat([1, 2, 3], 6)
    levels, matrix = rdm(x, labels, repeats=20)
    assert levels.tolist() == [1, 2, 3]
    assert np.allclose(np.diag(matrix), 0.)
    assert np.allclose(matrix, matrix.T, equal_nan=True)


def test_pooled_variance_has_a_positive_floor():
    x = np.zeros((6, 4)); labels = np.repeat([0, 1], 3)
    assert np.all(pooled_variance(x, labels) > 0)
