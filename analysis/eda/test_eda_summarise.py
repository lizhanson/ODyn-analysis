import numpy as np
import pandas as pd
from .summarise import (add_blank_contrast, participation_ratio, treves_rolls,
                       unit_summary)


def frame(n_unit=3, n_odor=4):
    rows = []
    for unit in range(n_unit):
        for odor in range(n_odor):
            rows.append({
                "group_id": 1, "mouse": "m1", "line": "TH", "cohort": "TH 20x",
                "depth_class": "deep", "population": "somas", "state": "pre",
                "unit_id": f"g{unit}", "odor_id": odor, "odor_group": "single",
                "is_blank": odor == 0, "trials_per_odor": 5,
                "excitation_area": float(odor), "suppression_area": 0.,
                "excited": odor > 0, "suppressed": False, "biphasic": False,
                "raw_mean_sustained": 1.0 + odor,
            })
    return pd.DataFrame(rows)


def test_treves_rolls_bounds():
    assert treves_rolls([1., 0., 0., 0.]) == 1.
    assert np.isclose(treves_rolls([1., 1., 1., 1.]), 0.)
    assert np.isnan(treves_rolls([0., 0.]))


def test_participation_ratio_normalised():
    assert np.isclose(participation_ratio([1., 1., 0., 0.]), .5)
    assert np.isclose(participation_ratio([1., 0., 0., 0.]), .25)


def test_blank_contrast_subtracts_the_units_own_blank():
    joined = add_blank_contrast(frame())
    real = joined[~joined.is_blank]
    # raw_mean_sustained is 1+odor and the blank is odor 0, so the contrast
    # is exactly the odor index.
    np.testing.assert_allclose(real.odor_minus_blank, real.odor_id)


def test_across_odor_sd_is_blank_invariant():
    """A constant delivery response added to every odor must not change it."""
    base = frame()
    shifted = base.copy()
    shifted["raw_mean_sustained"] = shifted.raw_mean_sustained + 7.0
    a = unit_summary(base).across_odor_sd.to_numpy()
    b = unit_summary(shifted).across_odor_sd.to_numpy()
    np.testing.assert_allclose(a, b)


def test_unit_summary_shape_and_breadth():
    units = unit_summary(frame(n_unit=3, n_odor=4))
    assert len(units) == 3
    assert np.allclose(units.n_odor, 3)
    assert np.allclose(units.excitation_breadth, 1.0)
