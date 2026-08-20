import h5py
import numpy as np

from .finalize import mask_hash, verify
from .store import find_rounds


def test_portable_mask_bundle_is_not_a_finalized_round(tmp_path):
    portable = tmp_path / "group212_example_20x_masks_processed_20260819.h5"
    with h5py.File(portable, "w") as handle:
        masks = handle.create_group("masks")
        masks.create_dataset("soma", data=np.zeros((5, 5), np.int32))
        masks.create_dataset("process", data=np.zeros((5, 5), np.int32))

    assert find_rounds(tmp_path) == []
    assert verify(tmp_path) == {"status": "no rounds found", "rounds": []}


def test_verify_ignores_bundle_and_reads_standard_round(tmp_path):
    labels = np.zeros((5, 5), np.int32)
    labels[1:3, 1:3] = 1
    portable = tmp_path / "group212_example_20x_masks_processed_20260819.h5"
    round_path = tmp_path / "group212_example_processed_20260819.h5"

    with h5py.File(portable, "w") as handle:
        handle.create_group("masks").create_dataset("soma", data=labels)
    with h5py.File(round_path, "w") as handle:
        handle.create_group("masks").create_dataset("labels", data=labels)
        handle.attrs["mask_hash"] = mask_hash(labels)

    assert find_rounds(tmp_path) == [round_path]
    report = verify(tmp_path)
    assert report["file"] == round_path.name
    assert report["status"] == "mask only, no traces"
