import inspect

import numpy as np

from .responders import response_figure, trial_calls
from .response_qc import response_qc


def test_pc1_is_recorded_but_not_removed_by_default():
    rng = np.random.default_rng(7)
    n_roi, n_trial, n_frame = 8, 12, 30
    on = np.full(n_trial, 15)
    odors = np.tile([1, 2, 3], 4)
    roi = rng.normal(0, 0.1, (n_roi, n_trial, n_frame))
    block = np.repeat([-1.0, 1.0], n_trial // 2)
    roi[:, :, 15:22] += block[None, :, None]

    kept = trial_calls(roi, odor_on_frames=on, odor_ids=odors)
    removed = trial_calls(roi, odor_on_frames=on, odor_ids=odors, deglobal="pc1")

    assert kept["deglobal"] is None
    assert kept["global_component"] is None
    assert kept["pc1_component"].shape == (n_trial,)
    np.testing.assert_allclose(kept["pc1_component"], removed["pc1_component"])
    assert not np.allclose(kept["z"], removed["z"], equal_nan=True)


def test_qc_entry_points_default_to_no_pc1_subtraction():
    assert inspect.signature(response_figure).parameters["deglobal"].default is None
    assert inspect.signature(response_qc).parameters["deglobal"].default is None
