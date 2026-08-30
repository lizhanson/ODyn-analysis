import numpy as np

from .response_metrics import (
    empirical_blank_thresholds, es_balance, lifetime_sparseness,
    participation_ratio, signed_response_components,
)


def test_empirical_thresholds_are_independent_and_asymmetric():
    blank = np.r_[np.linspace(-2, 0, 50), np.linspace(0, 5, 50)]
    threshold = empirical_blank_thresholds(blank, tail_probability=.1)
    assert threshold.negative_z < 0 < threshold.positive_z
    assert abs(threshold.negative_z) != threshold.positive_z


def test_sign_components_can_coexist_across_an_odor_profile():
    from .response_metrics import SignedThresholds
    t = SignedThresholds(-.5, 1., .01, .99)
    result = signed_response_components(np.array([[-1., 0., 2.]]), t)
    assert result["suppressed"].tolist() == [[True, False, False]]
    assert result["excited"].tolist() == [[False, False, True]]


def test_participation_ratio_is_analogue_and_normalized():
    assert np.isclose(participation_ratio([1., 1., 0.]), 2/3)
    assert np.isclose(participation_ratio([1., 0., 0.]), 1/3)
    assert lifetime_sparseness([[1., 0., 0.]])[0] == 1
    assert es_balance([2., 0.], [0., 1.]) == 1/3
