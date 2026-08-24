import json

import numpy as np

from .batch_pupil_tuning import TuningQueueApp


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
