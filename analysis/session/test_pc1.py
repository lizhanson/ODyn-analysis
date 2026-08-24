import numpy as np

from .pc1 import trial_pc1


def test_trial_pc1_recovers_shared_trial_component_after_odor_protection():
    loading = np.array([1.0, 2.0, -1.0])
    loading /= np.linalg.norm(loading)
    score = np.array([-2., 1., 2., -1., -1., 2., 1., -2.])
    odors = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    tuning = np.array([[3.], [2.], [1.]]) * odors[None, :]
    response = loading[:, None] * score[None, :] + tuning
    result = trial_pc1(response, odors)
    assert result["trial_score"].shape == (8,)
    assert result["loadings"].shape == (3,)
    assert abs(np.corrcoef(result["trial_score"], score)[0, 1]) > .999
    assert result["explained_variance_fraction"] > .999
