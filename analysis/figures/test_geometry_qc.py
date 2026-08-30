import pandas as pd

from .geometry_qc import audit_geometry


def test_geometry_audit_separates_eligibility_from_outliers():
    table = pd.DataFrame({
        "line": ["TH"] * 5, "state": ["pre"] * 5, "pair": ["17-18"] * 5,
        "n_a": [2, 4, 4, 4, 4], "n_b": [3, 4, 4, 4, 4],
        "integrated_crossnobis": [0, 0, .1, -.1, 10],
        "spatiotemporal_crossnobis": [0, 0, .1, -.1, 10],
        "integrated_cosine": [0, 0, .01, -.01, 1],
    })
    result = audit_geometry(table, minimum_trials=3)
    assert not result.loc[0, "eligible_primary"]
    assert result.loc[4, "integrated_crossnobis_outlier_flag"]
    assert result.loc[4, "eligible_primary"]
