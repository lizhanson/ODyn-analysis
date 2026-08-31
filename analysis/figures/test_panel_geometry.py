import numpy as np

from .panel_geometry import (PANEL_ORDER, confusion_matrix, correlation_rdm,
                             crossnobis_rdm, panel_tables, trial_matrix)

TIME = np.arange(-5., 8., .25)
ROW = {"group_id": 1, "mouse": "m1", "population": "Thy1-GCaMP",
       "depth_class": "na"}


def _data(per_odor=6, n_unit=24, separation=6., seed=0):
    """Each odor drives its own distinct subset of units."""
    rng = np.random.default_rng(seed)
    odor_ids = np.repeat(PANEL_ORDER, per_odor)
    z = rng.normal(0, 1, (n_unit, odor_ids.size, TIME.size))
    during = (TIME >= 0) & (TIME < 4)
    for index, level in enumerate(PANEL_ORDER):
        rows = odor_ids == level
        z[index % n_unit, rows] += separation*during
    return {"unit_id": np.array([f"u{i}" for i in range(n_unit)]), "z": z,
            "baseline_mean": np.ones((n_unit, odor_ids.size)),
            "odor_id": odor_ids, "state": np.zeros(odor_ids.size, int),
            "state_levels": np.array(["pre"]),
            "trial_id": np.arange(odor_ids.size), "time_s": TIME}


def test_confusion_recovers_separable_odors():
    x, odor = trial_matrix(_data(), "pre")
    matrix = confusion_matrix(x, odor)
    assert np.nanmean(np.diag(matrix)) > .8
    np.testing.assert_allclose(np.nansum(matrix, axis=1), 1., atol=1e-9)


def test_confusion_is_near_chance_on_noise():
    x, odor = trial_matrix(_data(separation=0., seed=5), "pre")
    accuracy = np.nanmean(np.diag(confusion_matrix(x, odor)))
    assert accuracy < .25          # chance is 1/16


def test_correlation_rdm_is_symmetric_with_zero_diagonal():
    x, odor = trial_matrix(_data(), "pre")
    matrix = correlation_rdm(x, odor)
    assert matrix.shape == (16, 16)
    assert np.allclose(np.diag(matrix), 0., atol=1e-9)
    assert np.allclose(matrix, matrix.T, equal_nan=True)


def test_crossnobis_rdm_skips_odors_with_too_few_repeats():
    data = _data(per_odor=2)
    x, odor = trial_matrix(data, "pre")
    matrix = crossnobis_rdm(x, odor, minimum_trials=3)
    assert np.isnan(matrix[0, 1])


def test_panel_tables_shape_and_chance_column():
    accuracy, distances = panel_tables(_data(), ROW, "units")
    assert len(accuracy) == 16
    assert len(distances) == 16*15//2
    assert np.allclose(accuracy.chance, 1/16)
    assert set(accuracy.cohort) == {"Thy1"}
