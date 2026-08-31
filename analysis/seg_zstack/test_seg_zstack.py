import numpy as np
import pytest

from .io import load_scanimage_zstack
from .state import StructuralZStackState, resolve_depth_reference


def test_loader_averages_repeats(tmp_path):
    import tifffile

    raw = np.arange(6*4*5, dtype=np.int16).reshape(6,4,5)
    path = tmp_path / "stack.tif"
    tifffile.imwrite(path, raw)
    averaged, meta = load_scanimage_zstack(path, frames_per_plane=2, progress=False)
    np.testing.assert_allclose(averaged, raw.reshape(3,2,4,5).mean(axis=1))
    assert averaged.shape == (3,4,5)
    assert meta["frames_per_plane"] == 2


def test_physical_depth_and_profile_density(monkeypatch):
    soma = np.zeros((12,20), np.int32); soma[2:4,3:5] = 1
    process = np.zeros_like(soma)
    monkeypatch.setattr("analysis.seg_20x.state.detect_somas", lambda image, params: (soma.copy(), {}))
    monkeypatch.setattr("analysis.seg_20x.state.ridge_image", lambda image, params: np.zeros_like(image))
    monkeypatch.setattr("analysis.seg_20x.state.detect_processes", lambda image, somas, params, ridge=None: (process.copy(), {}))
    state = StructuralZStackState(np.zeros((2,12,20), np.float32), metadata={"z_step_um":10})
    for _ in range(4): state.advance(progress=False)
    table = state.roi_table(um_per_px=2, depth_zero_plane=1, depth_direction=1)
    assert table.depth_um.tolist() == [-10, 0]
    assert table.area_um2.tolist() == [16, 16]
    filtered = state.roi_table(um_per_px=2, min_soma_diameter_um=5)
    assert filtered[filtered.roi_type == "soma"].empty
    centered = state.roi_table(center_depth_um=-150)
    assert centered.depth_um.tolist() == [-155, -145]
    summary = state.depth_summary(um_per_px=2)
    assert summary.n_soma_profiles.tolist() == [1,1]
    assert summary.soma_profile_density_per_mm2.tolist() == [1/(12*20*4/1e6)]*2


def test_bundle_roundtrip_preserves_curated_masks(monkeypatch, tmp_path):
    soma = np.zeros((10,10), np.int32); soma[2:5,2:5] = 1
    process = np.zeros_like(soma); process[7,2:8] = 1
    monkeypatch.setattr("analysis.seg_20x.state.detect_somas", lambda image, params: (soma.copy(), {}))
    monkeypatch.setattr("analysis.seg_20x.state.ridge_image", lambda image, params: np.zeros_like(image))
    monkeypatch.setattr("analysis.seg_20x.state.detect_processes", lambda image, somas, params, ridge=None: (process.copy(), {}))
    state = StructuralZStackState(np.zeros((2,10,10), np.float32), metadata={"z_step_um":5})
    for _ in range(4): state.advance(progress=False)
    path = state.save(tmp_path/"round", um_per_px=1.5)
    resumed = StructuralZStackState.load(path)
    np.testing.assert_array_equal(resumed.masks()[0], state.masks()[0])
    np.testing.assert_array_equal(resumed.masks()[1], state.masks()[1])
    assert path.with_name("round_rois.csv").exists()
    assert path.with_name("round_depth_summary.csv").exists()


def test_centered_depth_coordinates():
    assert resolve_depth_reference(0, -150, 31) == (15.0, -150.0)
    assert resolve_depth_reference(2, None, 5) == (2.0, 0.0)

    with pytest.raises(ValueError, match="outside"):
        resolve_depth_reference(8, None, 4)
