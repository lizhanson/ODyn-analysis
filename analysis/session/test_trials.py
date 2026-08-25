from types import SimpleNamespace

import pandas as pd
import pytest

from .trials import trial_table_from_events


def _write_events(root, timestamp, odors):
    path = root / f"program-{timestamp}-Events.csv"
    rows = ["Mode: Custom", "TimeStamp,Events"]
    rows.extend(
        f'{index * 1000:.3f},"Odor I {odor:02d} - test,Output 4"'
        for index, odor in enumerate(odors)
    )
    path.write_text("\n".join(rows), encoding="utf-8")


def _group():
    starts = pd.to_datetime([
        "2026-07-30 12:54:05.087",
        "2026-07-30 12:54:35.087",
        "2026-07-30 13:42:26.730",
        "2026-07-30 13:42:56.730",
    ])
    return SimpleNamespace(
        acquisitions=pd.DataFrame({
            "exp_id": [234] * 4,
            "acq_id": [10, 11, 12, 13],
            "acq_start": starts,
            "odor_start": starts + pd.to_timedelta(5, unit="s"),
            "odor_end": starts + pd.to_timedelta(9, unit="s"),
        }),
        odors=pd.DataFrame({"odor_id": [0, 1, 2]}),
    )


def test_event_recovery_preserves_acquisition_ids_and_assigns_blocks(tmp_path):
    _write_events(tmp_path, "2026_07_30-12_54_05", [1, 2])
    _write_events(tmp_path, "2026_07_30-13_42_27", [2, 0])

    table = trial_table_from_events(
        _group(), exp_id=234, exp_dir=tmp_path, manipulation="ketxyl"
    )

    assert table.acq_id.tolist() == [10, 11, 12, 13]
    assert table.odor_id.tolist() == [1, 2, 2, 0]
    assert table.state.tolist() == ["pre", "pre", "post", "post"]
    assert table.trial_source.unique().tolist() == [
        "olfactometer_events+database_acquisitions"
    ]


def test_event_recovery_refuses_count_mismatch(tmp_path):
    _write_events(tmp_path, "2026_07_30-12_54_05", [1])
    _write_events(tmp_path, "2026_07_30-13_42_27", [2, 0])

    with pytest.raises(ValueError, match="3 odors.*4 acquisitions"):
        trial_table_from_events(_group(), exp_id=234, exp_dir=tmp_path)


def test_event_recovery_refuses_unaligned_block_timestamp(tmp_path):
    _write_events(tmp_path, "2026_07_30-12_54_05", [1, 2])
    _write_events(tmp_path, "2026_07_30-14_42_27", [2, 0])

    with pytest.raises(ValueError, match="refusing positional pairing"):
        trial_table_from_events(_group(), exp_id=234, exp_dir=tmp_path)
