import numpy as np
import pandas as pd

from .population_metrics import (TemporalWindows, box_smooth, breadth_table,
                                 excursion_thresholds, temporal_feature_table)

TIME = np.arange(-5., 8., .05)


def _session(n_unit=40, per_odor=6, odors=(0, 1, 2, 3), seed=0):
    rng = np.random.default_rng(seed)
    odor_ids = np.repeat(odors, per_odor)
    z = rng.normal(0, 1, (n_unit, odor_ids.size, TIME.size))
    return {"unit_id": np.array([f"g{i}" for i in range(n_unit)]), "z": z,
            "baseline_mean": np.ones((n_unit, odor_ids.size)),
            "odor_id": odor_ids, "state": np.zeros(odor_ids.size, int),
            "state_levels": np.array(["pre"]),
            "trial_id": np.arange(odor_ids.size), "time_s": TIME}


ROW_20X = {"group_id": 1, "mouse": "m1", "population": "TH-GCaMP",
           "depth_class": "deep"}
ROW_10X = {"group_id": 2, "mouse": "m2", "population": "Thy1-GCaMP",
           "depth_class": "na"}


def test_cohort_drops_the_na_depth_class_at_10x():
    table = temporal_feature_table(_session(), ROW_10X, "units")
    assert set(table.cohort) == {"Thy1"}
    table = temporal_feature_table(_session(), ROW_20X, "somas")
    assert set(table.cohort) == {"TH deep"}


def test_biphasic_response_is_detected_though_its_mean_is_zero():
    """The case a signed four-second mean cannot represent."""
    data = _session()
    up = (TIME >= 0) & (TIME < 2)
    down = (TIME >= 2) & (TIME < 4)
    data["z"][0, data["odor_id"] == 1] += 8*up - 8*down
    table = temporal_feature_table(data, ROW_20X, "somas")
    row = table[(table.unit_id == "g0") & (table.odor_id == 1)].iloc[0]
    assert row.excited and row.suppressed and row.biphasic
    assert abs(row.mean_response_z) < .5          # invisible to a signed mean
    assert row.positive_auc_z_s > 0 and row.negative_auc_z_s > 0


def test_blank_is_excluded_from_breadth_but_can_be_reported():
    data = _session()
    table = temporal_feature_table(data, ROW_20X, "somas")
    assert 0 not in set(table.odor_id)
    with_blank = temporal_feature_table(data, ROW_20X, "somas",
                                        include_blank=True)
    assert 0 in set(with_blank.odor_id)
    assert breadth_table(with_blank).n_odor.max() == 3


def test_false_positive_rate_tracks_the_nominal_tail():
    """Pure noise: the realised responder rate sits near the nominal one."""
    table = temporal_feature_table(_session(n_unit=120, seed=3), ROW_20X,
                                   "somas", tail_probability=.05)
    assert .01 < table.excited.mean() < .12


def test_threshold_is_stratified_by_trial_count():
    """A median over more trials is quieter, so its cutoff is tighter."""
    rng = np.random.default_rng(1)
    odor_ids = np.array([1]*2 + [2]*16)
    z = rng.normal(0, 1, (80, odor_ids.size, TIME.size))
    stack = np.stack([np.nanmedian(z[:, odor_ids == o, :], axis=1)
                      for o in (1, 2)], axis=1)
    thresholds = excursion_thresholds(box_smooth(stack, 10), TIME,
                                      np.array([2, 16]))
    assert thresholds["positive"][2] > thresholds["positive"][16]


def test_breadth_and_balance_are_computed_per_unit():
    data = _session()
    data["z"][1, data["odor_id"] == 2] += 8*((TIME >= 0) & (TIME < 4))
    table = temporal_feature_table(data, ROW_20X, "somas")
    breadth = breadth_table(table)
    assert len(breadth) == 40
    row = breadth[breadth.unit_id == "g1"].iloc[0]
    assert row.excitation_breadth > 0
    assert -1 <= row.es_balance <= 1


def test_box_smooth_is_nan_aware_and_shape_preserving():
    x = np.array([[1., np.nan, 1., 1.]])
    assert np.allclose(box_smooth(x, 3), 1.)
    noisy = np.random.default_rng(0).normal(size=(3, 200))
    assert np.nanstd(box_smooth(noisy, 10)) < np.nanstd(noisy)
