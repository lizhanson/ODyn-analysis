import numpy as np

from .soma_size_20x import soma_measurements, summarize_session


def test_soma_measurements_uses_curated_label_area_and_scale():
    labels = np.array([[0, 1, 1], [2, 2, 2]], dtype=np.int32)
    rows = soma_measurements(labels, um_per_px=0.5)
    assert [row["area_px"] for row in rows] == [2, 3]
    assert [row["area_um2"] for row in rows] == [0.5, 0.75]
    assert np.isclose(rows[0]["equivalent_diameter_um"], 2 * np.sqrt(0.5 / np.pi))


def test_session_summary_keeps_session_as_sampling_level():
    common = {"mouse": "m1", "date": "20260101", "population": "TH-GCaMP",
              "depth_class": "deep", "depth_um": "150", "cohort": "TH deep",
              "um_per_px": 1.0, "mask_bundle": "mask.h5"}
    cells = [
        {**common, "group_id": 1, "area_um2": 4., "equivalent_diameter_um": 2.},
        {**common, "group_id": 1, "area_um2": 16., "equivalent_diameter_um": 4.},
        {**common, "group_id": 2, "area_um2": 9., "equivalent_diameter_um": 3.},
    ]
    rows = summarize_session(cells)
    assert len(rows) == 2
    assert rows[0]["n_somas"] == 2
    assert rows[0]["median_equivalent_diameter_um"] == 3.
