import csv

from .batch_10x import manifest_rows


def test_manifest_rows_selects_only_10x(tmp_path):
    path = tmp_path / "manifest.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("group_id", "objective"))
        writer.writeheader()
        writer.writerows((
            {"group_id": 1, "objective": "10x"},
            {"group_id": 2, "objective": "20x"},
            {"group_id": 3, "objective": "10X"},
        ))
    assert [int(row["group_id"]) for row in manifest_rows(path)] == [1, 3]
    assert [int(row["group_id"]) for row in manifest_rows(path, [3])] == [3]
