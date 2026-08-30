import numpy as np

from .arousal import trial_arousal_features, within_group_center


def test_within_group_center_removes_odor_medians():
    result = within_group_center([1, 3, 10, 14], [7, 7, 8, 8])
    np.testing.assert_allclose(result, [-1, 1, -2, 2])


def test_trial_arousal_features():
    time = np.arange(-5, 5)
    pupil = np.tile(time, (2, 1))
    speed = np.ones_like(pupil)
    result = trial_arousal_features(pupil, speed, time)
    np.testing.assert_allclose(result["running_speed"], 1)
    assert np.all(result["pupil_delta"] > 0)
