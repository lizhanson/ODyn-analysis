import json

import numpy as np

from .batch_pupil_tuning import TuningQueueApp
from .batch_auxiliary import cleanup_staged_inputs, completed_outputs, inventory


def test_queue_loads_pending_cache_and_existing_complete_is_optional(tmp_path):
    from bokeh.document import Document

    cache = tmp_path / "frames.npz"
    np.savez(cache, dim=np.zeros((12, 16), np.uint8),
             bright=np.full((12, 16), 200, np.uint8))
    complete = tmp_path / "complete.json"
    complete.write_text(json.dumps({
        "roi": [1, 10, 2, 14], "threshold_offset": 4,
        "bright_percentile": 97,
    }))
    items = []
    for group_id, output in ((1, tmp_path / "pending.json"), (2, complete)):
        items.append({
            "group_id": group_id, "status": "prepared", "cache": str(cache),
            "output": str(output), "date": "20260101", "mouse": "m1",
            "objective": "10x", "exp_name": "20260101_m1_e1",
        })
    queue = tmp_path / "queue.json"
    queue.write_text(json.dumps({"version": 1, "items": items}))

    pending = TuningQueueApp(Document(), queue)
    assert [item["group_id"] for item in pending.items] == [1]
    review = TuningQueueApp(Document(), queue, include_complete=True)
    assert [item["group_id"] for item in review.items] == [1, 2]


def test_completed_auxiliary_requires_valid_h5_and_all_figures(tmp_path):
    import h5py

    row = {"group_id": "7", "date": "20260101", "mouse": "m1", "exp": "e1"}
    aux = tmp_path / "processed" / "python" / "aux"
    aux.mkdir(parents=True)
    stem = "group7_20260101_m1_e1_auxiliary"
    with h5py.File(aux / f"{stem}.h5", "w") as handle:
        handle.attrs.update(group_id=7, n_trial=2, n_frame=3)
        handle.create_dataset("trials/acq_id", data=[1, 2])
        handle.create_dataset("treadmill/velocity", data=np.zeros((2, 3)))
        handle.create_dataset("respiration/filtered_v", data=np.zeros((2, 3)))
        handle.create_dataset("pupil/diameter_px", data=np.zeros((2, 3)))
    for suffix in ("_qc.png", "_respiration_odors.png", "_treadmill_odors.png",
                   "_pupil_qc.png"):
        (aux / f"{stem}{suffix}").write_bytes(b"png")
    assert completed_outputs(tmp_path, row, expect_pupil=True)["h5"].endswith(".h5")
    (aux / f"{stem}_pupil_qc.png").unlink()
    assert completed_outputs(tmp_path, row, expect_pupil=True) is None


def test_inventory_selects_valid_behavior_sync_not_first_h5(tmp_path):
    import h5py

    row = {"group_id": "7", "date": "20260101", "mouse": "m1", "exp": "e1"}
    sync = tmp_path / "20260101" / "m1" / "e1" / "sync"
    sync.mkdir(parents=True)
    with h5py.File(sync / "a_pre_artifact.h5", "w") as handle:
        handle.attrs["samplerate"] = 5000
        handle.create_dataset("ImagingWindow", data=[0])
    with h5py.File(sync / "z_behavior.h5", "w") as handle:
        for name in ("2pFrameSync", "respiration", "odorPulse"):
            handle.create_dataset(name, data=np.zeros(10))
    result = inventory(row, tmp_path)
    assert result["sync"].endswith("z_behavior.h5")


def test_cleanup_removes_only_staged_inputs_and_preserves_checkpoint(tmp_path):
    checkpoint = tmp_path / "session_pupil_checkpoint.npz"
    checkpoint.write_bytes(b"checkpoint")
    for name in ("staged_videos", "staged_sync"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "cached-input").write_bytes(b"data")
    removed = cleanup_staged_inputs(tmp_path)
    assert set(map(lambda value: value.rsplit("/", 1)[-1], removed)) == {
        "staged_videos", "staged_sync"
    }
    assert checkpoint.read_bytes() == b"checkpoint"
