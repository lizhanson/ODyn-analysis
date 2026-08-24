import numpy as np

from .auxiliary import discover_pupil_videos, extract_treadmill, write_auxiliary


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
