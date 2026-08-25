import json

import h5py
import numpy as np

from . import auxiliary_revision


def _auxiliary_file(path, sync_path):
    shape = (2, 4)
    with h5py.File(path, "w") as handle:
        handle.attrs.update(
            exp_name="20260101_m1_e1", group_id=7, frame_rate_hz=10.0,
            manipulation="ketamine/xylazine",
            sources_json=json.dumps({"sync": str(sync_path)}),
        )
        trials = handle.create_group("trials")
        trials.create_dataset("acq_id", data=[10, 11])
        trials.create_dataset("trial_id", data=[20, 21])
        trials.create_dataset("odor_id", data=[1, 2])
        trials.create_dataset("state", data=[0, 1])
        trials.create_dataset("state_levels", data=np.asarray([b"pre", b"post"]))
        trials.create_dataset("odor_on_frame", data=[1, 1])
        trials.create_dataset("odor_off_frame", data=[3, 3])
        acquisition = handle.create_group("acquisition")
        acquisition.create_dataset(
            "time_from_odor_s", data=np.tile(np.arange(4), (2, 1))
        )
        treadmill = handle.create_group("treadmill")
        treadmill.create_dataset("velocity", data=np.zeros(shape))
        respiration = handle.create_group("respiration")
        for name in (
            "filtered_v", "sniff_frequency_hz",
            "sniff_frequency_unmasked_hz", "sniff_frequency_instantaneous_hz",
            "sniff_frequency_instantaneous_unmasked_hz", "snr",
        ):
            respiration.create_dataset(name, data=np.zeros(shape))
        pupil = handle.create_group("pupil")
        pupil.attrs["config_json"] = json.dumps({
            "min_inlier_fraction": 0.55, "max_residual_px": 3.0,
        })
        pupil.create_dataset(
            "diameter_unmasked_px", data=np.asarray([[10., 11., 12., 13.]] * 2)
        )
        pupil.create_dataset(
            "major_radius_px", data=np.asarray([[5., 6., 7., 8.]] * 2)
        )
        pupil.create_dataset("diameter_px", data=np.ones(shape))
        pupil.create_dataset(
            "fit_inlier_fraction",
            data=np.asarray([[.9, .2, .9, .9], [.9, .9, .9, .9]]),
        )
        pupil.create_dataset("fit_residual_px", data=np.ones(shape))
        pupil.create_dataset(
            "blink", data=np.asarray([[0, 0, 1, 0], [0, 0, 0, 0]], np.int8)
        )
        pupil.create_dataset("clipped", data=np.zeros(shape, np.int8))
    return shape


def test_revision_remasks_pupil_and_atomically_updates_outputs(tmp_path, monkeypatch):
    sync = tmp_path / "sync.h5"
    sync.touch()
    path = tmp_path / "group7_20260101_m1_e1_auxiliary.h5"
    shape = _auxiliary_file(path, sync)
    rate = np.full(shape, 3.0)
    rate[0, 0] = np.nan
    respiration = {
        "round": "test", "sync": "sync.h5", "exp_name": "20260101_m1_e1",
        "manipulation": "ketamine/xylazine", "n_trial": 2, "n_frame": 4,
        "frame_rate": 10.0, "n_pre": 1, "smooth_breaths": 3,
        "quality_threshold": 1.5, "max_masked_fraction": .2,
        "n_breaths": 42, "median_hz": 3.0, "session_fraction_good": .875,
        "n_flagged": 1, "flags": [], "rate": rate,
        "rate_unmasked": np.full(shape, 3.0),
        "rate_instantaneous": rate.copy(),
        "rate_instantaneous_unmasked": np.full(shape, 3.0),
        "filtered": np.ones(shape), "quality": np.full(shape, 2.0),
        "masked_fraction": np.mean(~np.isfinite(rate), axis=1),
        "odor_ids": np.asarray([1, 2]), "states": np.asarray([0, 1]),
        "state_levels": ["pre", "post"], "on_frames": np.asarray([1, 1]),
        "off_frames": np.asarray([3, 3]),
    }
    monkeypatch.setattr(auxiliary_revision, "extract_respiration",
                        lambda *args, **kwargs: respiration)
    monkeypatch.setattr(
        auxiliary_revision, "combined_qc_figure",
        lambda output, **kwargs: output.write_bytes(b"combined"),
    )
    monkeypatch.setattr(
        auxiliary_revision, "respiration_figure",
        lambda report, output: output.write_bytes(b"respiration"),
    )
    monkeypatch.setattr(
        auxiliary_revision, "treadmill_figure",
        lambda output, **kwargs: output.write_bytes(b"treadmill"),
    )

    result = auxiliary_revision.revise_auxiliary(path, snr_threshold=1.5)

    with h5py.File(path) as handle:
        masked = handle["pupil/diameter_px"][:]
        assert np.isnan(masked[0, 1])  # low-inlier fit
        assert np.isnan(masked[0, 2])  # blink
        assert handle["respiration"].attrs["n_breaths"] == 42
        assert handle["respiration"].attrs["snr_threshold"] == 1.5
        assert "revision_json" in handle.attrs
    assert path.with_name(f"{path.stem}_qc.png").read_bytes() == b"combined"
    assert path.with_name(
        f"{path.stem}_respiration_odors.png"
    ).read_bytes() == b"respiration"
    assert path.with_name(
        f"{path.stem}_treadmill_odors.png"
    ).read_bytes() == b"treadmill"
    assert result["pupil_figure"].startswith("preserved")
