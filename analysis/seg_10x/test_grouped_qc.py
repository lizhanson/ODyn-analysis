import json

import numpy as np

from .grouped_qc import aggregate_joined_raw, cleanup_10x_caches
from .grouping import JoiningState, load_groups


def test_joined_raw_is_pixel_weighted_and_components_are_absent():
    raw = np.array([[[10., 20.]], [[30., 50.]], [[70., 90.]]])
    population = aggregate_joined_raw(raw, np.array([1., 3., 2.]), {1: 8, 2: 8})
    assert population.unit_ids == ["j8", "r3"]
    assert population.members == [[1, 2], [3]]
    np.testing.assert_allclose(population.raw[0, 0], [25., 42.5])


def test_one_member_assignment_remains_a_singleton():
    raw = np.arange(6, dtype=float).reshape(2, 1, 3)
    population = aggregate_joined_raw(raw, np.ones(2), {1: 9})
    assert population.unit_ids == ["r1", "r2"]
    assert population.members == [[1], [2]]


def test_joining_state_round_trips_with_mask_guard(tmp_path):
    path = tmp_path / "groups.json"
    state = JoiningState(
        np.zeros((2, 2), int), np.zeros((2, 2)), tmp_path / "round.h5",
        "abc", candidates=None, params={"max_gap_px": 4}, groups={1: 3, 2: 3})
    state.save(path)
    groups, metadata = load_groups(path, expected_mask_hash="abc")
    assert groups == {1: 3, 2: 3}
    assert metadata["source_round"].endswith("round.h5")


def test_cleanup_requires_outputs_then_removes_only_known_caches(tmp_path):
    output = tmp_path / "output"; scratch = tmp_path / "scratch"
    paths = {}
    for key in ("grouped_h5", "continuous_qc", "baseline_qc", "spatial_qc", "json"):
        path = tmp_path / key
        path.write_text("ok")
        paths[key] = str(path)
    targets = [scratch / "correlation_cache" / "group7",
               output / "correlation_cache", output / "zscore_cache"]
    for target in targets:
        target.mkdir(parents=True); (target / "cache").write_text("x")
    unrelated = output / "keep"; unrelated.mkdir(parents=True)
    removed = cleanup_10x_caches(output, scratch, 7, qc_outputs=paths)
    assert set(removed) == {str(path) for path in targets}
    assert unrelated.exists()
