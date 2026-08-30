import numpy as np
import pandas as pd

from .cellular_20x import (TemporalWindows, reliability_tables,
                           specificity_table, temporal_feature_table,
                           tonic_table)


def _data():
    time = np.arange(-1, 5, .5)
    odors = np.tile([1, 1, 2, 2], 2)
    states = np.repeat([0, 1], 4)
    z = np.zeros((2, 8, len(time)))
    for trial, odor in enumerate(odors):
        z[0, trial, time >= 0] = 2. if odor == 1 else .2
        z[1, trial, time >= 0] = -1. if odor == 2 else -.1
    return {"unit_id": np.array(["g1", "g2"]), "z": z,
            "baseline_mean": np.array([[10]*4+[5]*4, [20]*4+[10]*4]),
            "odor_id": odors, "state": states,
            "state_levels": np.array(["pre", "post"]),
            "trial_id": np.arange(8), "time_s": time}


ROW = {"group_id": 1, "mouse": "m1", "population": "TH-GCaMP",
       "depth_class": "deep"}


def test_tonic_and_temporal_features():
    data = _data()
    tonic = tonic_table(data, ROW, "somas")
    assert np.allclose(tonic.f0_log2_post_pre, -1)
    temporal = temporal_feature_table(data, ROW, "somas",
                                      windows=TemporalWindows())
    assert temporal.positive_auc_z_s.max() > 0
    assert temporal.negative_auc_z_s.max() > 0
    specificity = specificity_table(temporal)
    assert len(specificity) == 4


def test_reliability_shapes():
    data = _data()
    # Add a third odor because tuning reliability requires >=3 odors.
    data["odor_id"] = np.tile([1, 1, 2, 2], 2)
    units, odors = reliability_tables(data, ROW, "somas", repeats=3)
    assert units.empty and odors.empty
