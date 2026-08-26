import csv
import json

import h5py
import numpy as np

from .batch_20x import load_mask_bundle, manifest_rows


def test_manifest_rows_selects_only_20x_and_requested_groups(tmp_path):
    path = tmp_path / "manifest.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("group_id", "objective"))
        writer.writeheader()
        writer.writerows((
            {"group_id": 1, "objective": "20x"},
            {"group_id": 2, "objective": "10x"},
            {"group_id": 3, "objective": "20X"},
        ))
    assert [int(row["group_id"]) for row in manifest_rows(path)] == [1, 3]
    assert [int(row["group_id"]) for row in manifest_rows(path, [3])] == [3]


def test_load_mask_bundle_builds_contiguous_typed_masks_and_manifest(tmp_path):
    path = tmp_path / "bundle.h5"
    soma = np.array([[0, 4, 4], [0, 0, 0]], np.int32)
    process = np.array([[0, 0, 0], [9, 9, 0]], np.int32)
    config = {"groups": {"soma:4": 7, "process:9": 7},
              "summary": {"phase": "group"}}
    with h5py.File(path, "w") as handle:
        masks = handle.create_group("masks")
        masks.create_dataset("soma", data=soma)
        masks.create_dataset("process", data=process)
        handle.create_group("images").create_dataset(
            "structural", data=np.ones_like(soma, np.float32))
        handle.attrs["config_json"] = json.dumps(config)

    loaded = load_mask_bundle(path)
    np.testing.assert_array_equal(
        loaded["labels"], np.array([[0, 1, 1], [2, 2, 0]], np.int32))
    assert loaded["roi_manifest"] == [
        {"roi_id": 1, "roi_type": "soma", "source_roi_id": 4,
         "roi_group_id": 7},
        {"roi_id": 2, "roi_type": "process", "source_roi_id": 9,
         "roi_group_id": 7},
    ]
