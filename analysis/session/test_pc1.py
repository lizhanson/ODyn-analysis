import numpy as np

from .pc1 import fixed_pc1


def test_fixed_pc1_recovers_one_loading_and_within_trial_timecourse():
    loading = np.array([1.0, 2.0, -1.0])
    loading /= np.linalg.norm(loading)
    score = np.linspace(-2, 2, 40).reshape(4, 10)
    dff = loading[:, None, None] * score[None, :, :]
    result = fixed_pc1(dff)

    assert result["timecourse"].shape == (4, 10)
    assert result["loadings"].shape == (3,)
    assert abs(np.dot(result["loadings"], loading)) > 0.999
    assert result["explained_variance_fraction"] > 0.999


def test_fixed_pc1_sign_tracks_population_mean():
    dff = np.zeros((2, 2, 5), float)
    dff[0] = np.arange(10).reshape(2, 5)
    dff[1] = 2 * dff[0]
    result = fixed_pc1(dff)
    population = dff.mean(axis=0).ravel()
    assert np.corrcoef(result["timecourse"].ravel(), population)[0, 1] > 0.99
