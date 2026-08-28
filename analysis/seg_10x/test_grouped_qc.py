import json

import numpy as np

from .grouped_qc import aggregate_joined_raw, cleanup_10x_caches
from .grouping import JoiningState, fully_connected_suggestions, load_groups


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


def test_multi_roi_suggestions_require_every_pair_to_pass():
    import pandas as pd

    complete = pd.DataFrame([
        {"roi_a": 1, "roi_b": 2, "gap_px": 1., "correlation": .91,
         "spatial_neighbor": True, "correlation_pass": True, "suggested": True},
        {"roi_a": 2, "roi_b": 3, "gap_px": 2., "correlation": .89,
         "spatial_neighbor": True, "correlation_pass": True, "suggested": True},
        # ROI 1 and 3 are not spatial neighbors, but their traces pass. The
        # spatial path 1-2-3 therefore permits the fully correlated triple.
        {"roi_a": 1, "roi_b": 3, "gap_px": np.nan, "correlation": .87,
         "spatial_neighbor": False, "correlation_pass": True, "suggested": False},
    ])
    suggestions = fully_connected_suggestions(complete)
    assert suggestions.members.tolist() == [(1, 2, 3)]
    assert suggestions.iloc[0].min_correlation == .87

    # A-B and B-C do not chain when the non-neighbor A-C correlation fails.
    chain = complete.copy()
    chain.loc[2, "correlation_pass"] = False
    suggestions = fully_connected_suggestions(chain)
    assert set(suggestions.members) == {(1, 2), (2, 3)}
    assert not any(len(members) == 3 for members in suggestions.members)


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
