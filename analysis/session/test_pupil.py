import numpy as np
import pytest

from . import pupil


def test_alignment_validation_refuses_mismatch_before_segmentation(tmp_path, monkeypatch):
    csv_path = tmp_path / "frametimes.csv"
    csv_path.write_text("time\n0.0\n0.1\n")
    monkeypatch.setattr(pupil, "count_video_frames", lambda path: 3)
    with pytest.raises(ValueError, match="decoded video=3.*pulses=2.*rows=2"):
        pupil.validate_alignment_counts("movie.mp4", [1, 2], frametimes_path=csv_path)


def test_alignment_validation_needs_no_frametimes_csv(monkeypatch):
    monkeypatch.setattr(pupil, "count_video_frames", lambda path: 2)
    assert pupil.validate_alignment_counts("movie.mp4", [10, 20]) == 2


def test_count_mismatch_uses_absolute_frametimes_with_endpoint_qc(tmp_path):
    from datetime import datetime
    from types import SimpleNamespace

    movie = tmp_path / "movie.mp4"
    csv_path = tmp_path / "movie_frametimes.csv"
    csv_path.write_text(
        "frame_index,elapsed_ms\n0,1000\n1,1100\n2,1300\n"
    )
    sync = SimpleNamespace(
        rate_hz=10,
        start_time=datetime.fromisoformat("2026-01-01T12:00:00-08:00"),
    )
    stamp = datetime.fromisoformat("2026-01-01T12:00:00-08:00")
    times, qc = pupil.camera_frame_times(
        [movie], [(stamp, "test")], [np.array([10, 11, 12, 13])],
        sync, [3], frametimes_paths=[csv_path],
    )
    np.testing.assert_allclose(times, [1.0, 1.1, 1.3])
    assert qc["methods"] == ["micromanager_frametimes"]
    assert qc["pulse_count_deltas"] == [1]
    assert qc["start_error_s"] == [0.0]
    assert qc["end_error_s"] == [0.0]


def test_two_videos_are_ordered_by_timestamp_in_path(tmp_path):
    post = tmp_path / "pupil_20260819_140000" / "converted" / "movie.mp4"
    pre = tmp_path / "pupil_20260819_120000" / "converted" / "movie.mp4"
    post.parent.mkdir(parents=True)
    pre.parent.mkdir(parents=True)
    post.touch()
    pre.touch()
    ordered, stamps = pupil.order_pupil_videos([post, pre])
    assert ordered == [pre, post]
    assert [source for _, source in stamps] == ["path timestamp", "path timestamp"]


def test_converter_metadata_timestamp_takes_priority(tmp_path):
    import json

    converted = tmp_path / "pupil_video_1" / "converted"
    converted.mkdir(parents=True)
    movie = converted / "pupil_video_1.mp4"
    movie.touch()
    metadata = {"mm_summary": {"StartTime": "2026-08-05 12:39:41.048 -0700"}}
    (converted / "pupil_video_1_metadata.json").write_text(json.dumps(metadata))

    stamp, source = pupil.video_recording_timestamp(movie)
    assert stamp.isoformat() == "2026-08-05T12:39:41.048000-07:00"
    assert "mm_summary.StartTime" in source


def test_local_staging_copies_movie_metadata_and_reuses_cache(tmp_path):
    source_dir = tmp_path / "server"
    source_dir.mkdir()
    movie = source_dir / "pupil_video_1.mp4"
    metadata = source_dir / "pupil_video_1_metadata.json"
    movie.write_bytes(b"movie bytes")
    metadata.write_text('{"mm_summary": {"StartTime": "2026-08-05 12:00:00.000 -0700"}}')
    frametimes = source_dir / "pupil_video_1_frametimes.csv"
    frametimes.write_text("frame_index,elapsed_ms\n0,0\n")

    staged = pupil.stage_pupil_videos([movie], tmp_path / "scratch")
    assert staged[0].with_name(frametimes.name).read_text() == frametimes.read_text()
    assert staged[0].read_bytes() == b"movie bytes"
    assert staged[0].with_name(metadata.name).read_text() == metadata.read_text()
    first_mtime = staged[0].stat().st_mtime_ns
    assert pupil.stage_pupil_videos([movie], tmp_path / "scratch")[0].stat().st_mtime_ns == first_mtime


def test_sync_staging_reuses_verified_local_copy(tmp_path):
    source = tmp_path / "server" / "behavior.h5"
    source.parent.mkdir()
    source.write_bytes(b"sync bytes")
    staged = pupil.stage_sync_file(source, tmp_path / "scratch")
    assert staged.read_bytes() == b"sync bytes"
    first_mtime = staged.stat().st_mtime_ns
    assert pupil.stage_sync_file(source, tmp_path / "scratch").stat().st_mtime_ns == first_mtime


def test_standalone_aux_joins_later_round_by_acq_id(tmp_path):
    import h5py

    aux = tmp_path / "pupil.h5"
    rnd = tmp_path / "round.h5"
    with h5py.File(aux, "w") as f:
        f.create_dataset("trials/acq_id", data=[30, 10, 20])
    with h5py.File(rnd, "w") as f:
        f.create_dataset("trials/acq_id", data=[20, 30])

    aux_rows, round_rows = pupil.align_to_round(aux, rnd)
    assert aux_rows.tolist() == [2, 0]
    assert round_rows.tolist() == [0, 1]


def test_bright_tail_floor_separates_pupil_from_eye_tissue():
    frame = np.zeros((100, 100), np.uint8)
    frame[20:80, 20:80] = 50
    frame[40:60, 40:60] = 220
    mask, threshold = pupil.segment_bright_pupil(frame, bright_percentile=97)
    assert threshold >= 50
    assert mask[50, 50]
    assert not mask[30, 30]


def test_ransac_removes_straight_lid_chord_and_recovers_ellipse():
    yy, xx = np.ogrid[:120, :140]
    full = ((xx - 70) / 35) ** 2 + ((yy - 60) / 24) ** 2 <= 1
    clipped = full & (yy >= 54)
    fit = pupil.fit_ellipse_ransac(clipped, random_seed=4, max_trials=500)
    assert fit is not None
    assert fit["chord_fraction"] >= 0.16
    assert abs(fit["x"] - 70) < 3
    assert abs(fit["y"] - 60) < 4
    assert abs(fit["major"] - 35) < 5
    assert abs(fit["minor"] - 24) < 5


def test_pupil_tuner_settings_round_trip(tmp_path):
    frame = np.zeros((20, 30), np.uint8)
    gui = pupil.PupilTuningGUI(
        frame, frame, save_path=tmp_path / "tuning.json",
        roi=(2, 18, 3, 25), threshold_offset=-7,
    )
    gui.save_path.write_text(__import__("json").dumps(gui.settings()))
    loaded = pupil.load_pupil_tuning(gui.save_path)
    assert loaded["roi"] == (2, 18, 3, 25)
    assert loaded["threshold_offset"] == -7


def test_pupil_qc_status_uses_frame_indices_not_pixel_mask():
    assert pupil._pupil_qc_status(40_224, [40_224, 42_917, 40_000]) == "EXCLUDED: fit"
    assert pupil._pupil_qc_status(6_948, [40_224, 42_917, 40_000]) == "accepted"


def test_consensus_filter_removes_tail_but_rejects_tiny_corner_change():
    yy, xx = np.ogrid[:120, :150]
    pupil_mask = ((xx - 65) / 35) ** 2 + ((yy - 60) / 25) ** 2 <= 1
    fit = {"x": 65.0, "y": 60.0, "major": 35.0, "minor": 25.0,
           "theta": 0.0}

    tailed = pupil_mask.copy()
    tailed[55:66, 98:125] = True
    cleaned, removed = pupil.filter_concave_protrusions(tailed, fit)
    assert removed >= 0.02
    assert not np.any(cleaned[:, 110:])
    assert np.all(cleaned[pupil_mask])

    tiny = pupil_mask.copy()
    tiny[59:62, 99:103] = True
    unchanged, removed = pupil.filter_concave_protrusions(tiny, fit)
    assert removed == 0.0
    np.testing.assert_array_equal(unchanged, tiny)


def test_numeric_checkpoint_round_trip_is_atomic(tmp_path):
    values = {"diameter": np.array([1., np.nan, 3.]),
              "threshold": np.array([10., np.nan, 30.])}
    completed = np.array([True, False, True])
    path = tmp_path / "checkpoint.npz"
    pupil._save_pupil_checkpoint(path, values, completed, "signature")

    restored = {name: np.full(3, np.nan) for name in values}
    restored_completed = pupil._load_pupil_checkpoint(
        path, restored, "signature"
    )
    assert restored_completed.tolist() == completed.tolist()
    np.testing.assert_equal(restored["diameter"], values["diameter"])
    assert not path.with_suffix(".npz.part").exists()


def test_atomic_publish_replaces_only_after_complete_copy(tmp_path):
    source = tmp_path / "local" / "result.h5"
    destination = tmp_path / "share" / "result.h5"
    source.parent.mkdir()
    source.write_bytes(b"verified local output")
    assert pupil._atomic_publish(source, destination) == destination
    assert destination.read_bytes() == source.read_bytes()
    assert not destination.with_suffix(".h5.part").exists()


def test_frame_worker_is_spawn_process_safe():
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    yy, xx = np.ogrid[:60, :70]
    frame = np.zeros((60, 70), np.uint8)
    frame[((xx - 35) / 14) ** 2 + ((yy - 30) / 10) ** 2 <= 1] = 220
    task = (0, frame, pupil.PupilConfig(roi=(5, 55, 5, 65), ransac_trials=20))
    try:
        executor = ProcessPoolExecutor(max_workers=2,
                                       mp_context=mp.get_context("spawn"))
    except (PermissionError, NotImplementedError) as error:
        pytest.skip(f"sandbox disallows multiprocessing semaphores: {error}")
    with executor:
        result = list(executor.map(pupil._analyze_pupil_frame, [task, task]))
    assert [item[0] for item in result] == [0, 0]
    assert all(item[2] is not None for item in result)


def test_frame_worker_rejects_photometrically_dark_frame():
    frame = np.full((40, 50), 42, np.uint8)
    task = (7, frame, pupil.PupilConfig(roi=(2, 38, 2, 48)))
    index, threshold, metrics = pupil._analyze_pupil_frame(task)
    assert index == 7
    assert np.isnan(threshold)
    assert metrics["illumination_active"] == 0
    assert metrics["illumination_peak"] == 42


def test_csv_count_accepts_header_or_numeric_first_row(tmp_path):
    header = tmp_path / "header.csv"
    numeric = tmp_path / "numeric.csv"
    header.write_text("frame,time\n0,0.0\n1,0.1\n")
    numeric.write_text("0,0.0\n1,0.1\n")
    assert pupil.count_csv_frames(header) == 2
    assert pupil.count_csv_frames(numeric) == 2


def test_blink_needs_two_signals_and_bad_fit_is_clipped():
    metrics = {
        "area": np.array([100., 100., 10.]),
        "axis_ratio": np.array([.8, .8, .1]),
        "residual": np.array([1., 5., 5.]),
        "inlier_fraction": np.array([.9, .3, .2]),
        "diameter": np.array([10., 10., 10.]),
    }
    blink, clipped, _ = pupil.classify_frames(metrics, np.arange(3.))
    assert blink.tolist() == [False, False, True]
    assert clipped.tolist() == [False, True, False]


def test_inactive_dark_frames_are_neither_blinks_nor_clipping():
    metrics = {
        "area": np.array([100., np.nan]),
        "axis_ratio": np.array([.8, np.nan]),
        "residual": np.array([1., np.nan]),
        "inlier_fraction": np.array([.9, np.nan]),
        "diameter": np.array([10., np.nan]),
    }
    blink, clipped, _ = pupil.classify_frames(
        metrics, np.arange(2.), active=[True, False]
    )
    assert not blink[1]
    assert not clipped[1]


def test_qc_examples_are_three_bad_fits_and_diameter_diverse_accepted():
    values = {
        "diameter": np.arange(1, 21, dtype=float),
        "inlier_fraction": np.full(20, .9),
        "residual": np.ones(20),
    }
    bad = np.asarray([2, 8, 15, 18])
    values["inlier_fraction"][bad] = .1
    values["residual"][bad] = [4., 5., 6., 7.]
    excluded, accepted = pupil.pupil_qc_frame_indices(
        values, np.ones(20, bool), counts=[10, 10]
    )
    assert excluded.tolist() == [18, 15, 8]
    assert len(accepted) == 9
    assert 0 in accepted and 9 in accepted
    assert 10 in accepted and 19 in accepted
    assert not set(excluded) & set(accepted)
