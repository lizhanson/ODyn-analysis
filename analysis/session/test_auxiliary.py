import numpy as np

from .auxiliary import (
    _representative_trial,
    _trial_block_indices,
    discover_pupil_videos,
    extract_treadmill,
    saved_pupil_config,
    write_auxiliary,
)


def _sync_file(path):
    import h5py

    rate = 1000
    samples = np.zeros(5000, np.float32)
    frame_samples = np.array([100, 200, 300, 400, 500, 3100, 3200, 3300, 3400, 3500])
    samples[frame_samples] = 5
    encoder = np.zeros((len(samples), 3), np.float32)
    encoder[:, 2] = np.linspace(-2, 3, len(samples))
    with h5py.File(path, "w") as handle:
        handle.attrs.update(rate_hz=rate, n_samples=len(samples),
                            start_time="2026-01-01T00:00:00")
        handle.create_dataset("2pFrameSync", data=samples)
        data = handle.create_dataset("encoder", data=encoder)
        data.attrs["wheel_diam_cm"] = 15.24
        data.attrs["vel_smooth_s"] = .1
    return frame_samples


def test_treadmill_is_acquisition_aligned_and_keeps_signed_velocity(tmp_path):
    path = tmp_path / "sync.h5"
    frame_samples = _sync_file(path)
    result = extract_treadmill(path, odor_on_frames=[2, 2])
    assert result["velocity"].shape == (2, 5)
    np.testing.assert_allclose(
        result["velocity"].ravel(),
        np.linspace(-2, 3, 5000)[frame_samples], rtol=1e-6,
    )
    np.testing.assert_allclose(result["time_from_odor_s"][:, 2], 0)
    assert result["encoder_attrs"]["wheel_diam_cm"] == 15.24


def test_treadmill_uses_selected_original_sync_block(tmp_path):
    path = tmp_path / "sync.h5"
    frame_samples = _sync_file(path)
    result = extract_treadmill(
        path, odor_on_frames=[2], acquisition_indices=[1]
    )
    np.testing.assert_allclose(
        result["velocity"].ravel(),
        np.linspace(-2, 3, 5000)[frame_samples[5:]], rtol=1e-6,
    )


def test_discovers_videos_in_named_sibling_directory(tmp_path):
    exp_dir = tmp_path / "20260807" / "m496" / "e1"
    exp_dir.mkdir(parents=True)
    video_root = exp_dir.parent / "20260807_m496_e1_video"
    expected = []
    for index in (1, 2):
        movie = video_root / f"pupil_video_{index}" / "converted" / f"pupil_video_{index}.mp4"
        movie.parent.mkdir(parents=True)
        movie.touch()
        expected.append(movie)
    assert discover_pupil_videos(exp_dir) == expected


def test_saved_pupil_config_falls_back_across_date_spelling(tmp_path):
    import json

    aux = tmp_path / "processed" / "python" / "aux"
    aux.mkdir(parents=True)
    tuning = aux / "group206_20260713_m462_e1_pupil_tuning.json"
    tuning.write_text(json.dumps({
        "roi": [1, 20, 2, 30], "threshold_offset": 7,
        "bright_percentile": 97,
    }))
    config, source = saved_pupil_config(
        tmp_path, group_id=206, exp_name="2026_07_13_m462_e1"
    )
    assert config.roi == (1, 20, 2, 30)
    assert config.threshold_offset == 7
    assert source == tuning


def test_representative_trial_uses_state_median_quality():
    quality = np.array([
        [0.5, 0.5, 0.5],
        [1.5, 1.5, 1.5],
        [9.0, 9.0, 9.0],
        [20.0, 20.0, 20.0],
    ])
    assert _representative_trial(quality, [True, True, True, False]) == 1
    assert _representative_trial(quality, [False, False, False, True]) == 3


def test_trial_blocks_exclude_only_short_pulse_free_fragments():
    blocks = [np.arange(10), np.arange(100), np.arange(100), np.arange(8)]
    windows = [None, (20, 40), (20, 40), None]
    assert _trial_block_indices(blocks, windows, 2) == [1, 2]


def test_trial_blocks_refuse_ambiguous_long_pulse_free_acquisition():
    import pytest

    blocks = [np.arange(60), np.arange(100), np.arange(100)]
    windows = [None, (20, 40), (20, 40)]
    with pytest.raises(ValueError, match="half a real acquisition"):
        _trial_block_indices(blocks, windows, 2)


def test_default_respiration_snr_threshold_is_three():
    from .respiration import QUALITY_THRESHOLD

    assert QUALITY_THRESHOLD == 3.0


def test_respiration_onsets_ignore_fast_structure_outside_sniff_band():
    from .respiration import instantaneous_frequency

    rate_hz = 500.0
    time = np.arange(20 * int(rate_hz)) / rate_hz
    raw = np.sin(2 * np.pi * 3 * time) + 0.8 * np.sin(2 * np.pi * 30 * time)
    result = instantaneous_frequency(
        raw, rate_hz=rate_hz, at_s=np.arange(0.5, 19.5, 0.04),
        quality_threshold=0,
    )
    assert 2.8 < np.nanmedian(result["frequency"]) < 3.2


def test_combined_h5_joins_by_acq_id_and_marks_missing_modalities(tmp_path):
    sync = tmp_path / "sync.h5"
    _sync_file(sync)
    treadmill = extract_treadmill(sync, odor_on_frames=[2, 2])
    metadata = {
        "exp_name": "20260101_m1_e1", "group_id": 1, "mouse": "m1",
        "date": "20260101", "manipulation": "ketamine/xylazine",
        "frame_rate": 10, "acq_ids": [20, 10], "trial_ids": [2, 1],
        "odor_ids": [7, 8], "states": ["pre", "post"],
        "odor_on_frames": [2, 2], "odor_off_frames": [4, 4],
    }
    out = write_auxiliary(
        tmp_path / "aux.h5", metadata=metadata, respiration=None, pupil=None,
        treadmill=treadmill, sources={"sync": str(sync)},
    )
    import h5py
    with h5py.File(out) as handle:
        assert handle.attrs["pupil_available"] == 0
        assert handle.attrs["respiration_available"] == 0
        assert handle["trials/acq_id"][:].tolist() == [20, 10]
        assert handle["treadmill/velocity"].shape == (2, 5)
        assert handle["acquisition/time_from_odor_s"].shape == (2, 5)
