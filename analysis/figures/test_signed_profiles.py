import numpy as np
import pandas as pd
import pytest

from .signed_profiles import (CLASSES, breadth_matched_change,
                              class_fractions, mouse_then_cohort,
                              paired_state_change, sign_permutation_null,
                              split_awake_trials, subset_trials,
                              unit_sign_profiles)

COMMON = {"group_id": 1, "mouse": "m1", "line": "TH", "depth_class": "superficial",
          "cohort": "TH superficial", "compartment": "somas"}


def _temporal(calls, state="pre", **overrides):
    """calls maps unit_id -> {odor_id: (excited, suppressed)}."""
    rows = []
    for unit_id, odors in calls.items():
        for odor_id, (excited, suppressed) in odors.items():
            rows.append(COMMON | overrides | {
                "unit_id": unit_id, "state": state, "odor_id": odor_id,
                "is_blank": odor_id == 0, "excited": bool(excited),
                "suppressed": bool(suppressed),
                "biphasic": bool(excited and suppressed),
                "positive_auc_z_s": 2.0 if excited else 0.1,
                "negative_auc_z_s": 2.0 if suppressed else 0.1})
    return pd.DataFrame(rows)


# --- classification --------------------------------------------------------

def test_profiles_assign_the_four_classes():
    table = _temporal({
        "silent": {1: (0, 0), 2: (0, 0)},
        "exc": {1: (1, 0), 2: (0, 0)},
        "sup": {1: (0, 1), 2: (0, 1)},
        "both": {1: (1, 0), 2: (0, 1)}})
    profiles = unit_sign_profiles(table).set_index("unit_id")
    assert profiles.loc["silent", "response_class"] == "silent"
    assert profiles.loc["exc", "response_class"] == "excited_only"
    assert profiles.loc["sup", "response_class"] == "suppressed_only"
    assert profiles.loc["both", "response_class"] == "bidirectional"
    assert set(profiles.response_class) <= set(CLASSES)


def test_a_single_biphasic_odor_is_not_cross_odor_bidirectional():
    """Excited then suppressed by one odor is biphasic, not bidirectional."""
    table = _temporal({"u": {1: (1, 1), 2: (0, 0)}})
    row = unit_sign_profiles(table).iloc[0]
    assert row.response_class == "bidirectional"      # both signs present
    assert not row.cross_odor_bidirectional           # but from one odor
    assert row.within_odor_biphasic


def test_biphasic_plus_a_second_odor_is_cross_odor_bidirectional():
    table = _temporal({"u": {1: (1, 1), 2: (1, 0)}})
    row = unit_sign_profiles(table).iloc[0]
    assert row.cross_odor_bidirectional and row.within_odor_biphasic


def test_blank_odor_is_excluded_from_the_profile():
    table = _temporal({"u": {0: (1, 0), 1: (0, 1)}})
    row = unit_sign_profiles(table).iloc[0]
    assert row.n_odor == 1 and row.n_excited == 0
    assert row.response_class == "suppressed_only"


def test_breadths_are_fractions_of_the_odors_tested():
    table = _temporal({"u": {1: (1, 0), 2: (1, 0), 3: (0, 1), 4: (0, 0)}})
    row = unit_sign_profiles(table).iloc[0]
    assert row.excitation_breadth == pytest.approx(.5)
    assert row.suppression_breadth == pytest.approx(.25)


def test_class_fractions_sum_to_one():
    table = _temporal({
        "a": {1: (1, 0)}, "b": {1: (0, 1)}, "c": {1: (0, 0)}, "d": {1: (1, 1)}})
    wide = class_fractions(unit_sign_profiles(table))
    assert len(wide) == 1
    assert sum(float(wide.iloc[0][name]) for name in CLASSES) == pytest.approx(1.)
    assert wide.iloc[0].n_unit == 4


def test_mouse_then_cohort_weights_animals_not_sessions():
    """One animal with many sessions must not outvote the others."""
    table = pd.DataFrame([
        {"cohort": "TH", "state": "pre", "mouse": "m1", "v": 1.},
        {"cohort": "TH", "state": "pre", "mouse": "m1", "v": 1.},
        {"cohort": "TH", "state": "pre", "mouse": "m1", "v": 1.},
        {"cohort": "TH", "state": "pre", "mouse": "m2", "v": 0.}])
    per_mouse, summary = mouse_then_cohort(table, ["v"])
    assert len(per_mouse) == 2
    assert float(summary.v_median.iloc[0]) == pytest.approx(.5)
    assert int(summary.v_size.iloc[0]) == 2


# --- the null --------------------------------------------------------------

def test_null_finds_no_excess_when_sign_is_not_a_cell_property():
    """Signs scattered at random across cells should match the null."""
    rng = np.random.default_rng(0)
    calls = {}
    for unit in range(60):
        odors = {}
        for odor in range(1, 9):
            if rng.random() < .4:
                excited = rng.random() < .5
                odors[odor] = (excited, not excited)
            else:
                odors[odor] = (0, 0)
        calls[f"u{unit}"] = odors
    result = sign_permutation_null(_temporal(calls), repeats=200, seed=1)
    assert float(result.suppressed_only_p_greater.iloc[0]) > .05


def test_null_detects_a_genuinely_suppression_only_population():
    """Twenty cells that only ever suppress, among cells that only excite."""
    rng = np.random.default_rng(2)
    calls = {}
    for unit in range(40):
        calls[f"e{unit}"] = {o: (rng.random() < .4, 0) for o in range(1, 9)}
    for unit in range(20):
        calls[f"s{unit}"] = {o: (0, rng.random() < .4) for o in range(1, 9)}
    result = sign_permutation_null(_temporal(calls), repeats=200, seed=3)
    assert float(result.suppressed_only_p_greater.iloc[0]) < .01
    assert float(result.suppressed_only_excess.iloc[0]) > 5


def test_null_preserves_each_units_responsiveness():
    """A rarely-responding cell cannot become single-sign by accident."""
    calls = {f"u{i}": {o: (i % 2 == 0, i % 2 == 1) for o in range(1, 5)}
             for i in range(10)}
    result = sign_permutation_null(_temporal(calls), repeats=50, seed=4)
    row = result.iloc[0]
    # Every cell responds to every odor, so nothing can ever be silent.
    assert row.silent_observed == 0 and row.silent_null_mean == 0


def test_null_p_values_are_bounded_away_from_zero():
    """An empirical null has finite resolution and must not report p = 0."""
    calls = {f"s{i}": {o: (0, 1) for o in range(1, 5)} for i in range(10)}
    result = sign_permutation_null(_temporal(calls), repeats=20, seed=5)
    assert float(result.suppressed_only_p_greater.iloc[0]) >= 1 / 21


# --- state pairing ---------------------------------------------------------

def _population(n_trial_per_odor=6, n_unit=4, odors=(1, 2, 3)):
    time_s = np.arange(-5., 8., .5)
    odor_ids, states = [], []
    for state in (0, 1):
        for odor in odors:
            odor_ids += [odor] * n_trial_per_odor
            states += [state] * n_trial_per_odor
    n = len(odor_ids)
    return {"unit_id": np.array([f"u{i}" for i in range(n_unit)]),
            "z": np.zeros((n_unit, n, time_s.size), np.float32),
            "baseline_mean": np.ones((n_unit, n), np.float32),
            "odor_id": np.array(odor_ids), "state": np.array(states),
            "state_levels": np.array(["pre", "post"]),
            "trial_id": np.arange(n), "time_s": time_s}


def test_subset_trials_slices_every_trial_indexed_array():
    data = _population()
    mask = data["state"] == 0
    out = subset_trials(data, mask)
    assert out["z"].shape[1] == mask.sum()
    assert out["baseline_mean"].shape[1] == mask.sum()
    assert set(out["state"]) == {0}
    assert out["z"].shape[0] == data["z"].shape[0]      # units untouched


def test_split_awake_trials_partitions_only_the_awake_block():
    data = _population()
    classify, measure = split_awake_trials(data, seed=0)
    awake = data["state"] == 0
    assert not np.any(classify & ~awake)               # never touches post
    assert np.all(measure[~awake])                     # post kept whole
    assert not np.any(classify & measure)              # disjoint
    assert np.array_equal(classify | measure, np.ones(len(awake), bool))


def test_split_awake_trials_halves_each_odor_separately():
    data = _population(n_trial_per_odor=6)
    classify, _ = split_awake_trials(data, seed=1)
    awake = data["state"] == 0
    for odor in (1, 2, 3):
        here = awake & (data["odor_id"] == odor)
        assert classify[here].sum() == 3


def test_split_awake_trials_drops_odors_with_a_single_repeat():
    data = _population(n_trial_per_odor=1)
    classify, measure = split_awake_trials(data, seed=0)
    assert classify.sum() == 0


def test_split_awake_trials_requires_an_awake_block():
    data = _population()
    data["state_levels"] = np.array(["post"])
    with pytest.raises(ValueError, match="no 'pre' state"):
        split_awake_trials(data)


def test_paired_state_change_reports_signed_differences():
    pre = _temporal({"u": {1: (0, 1), 2: (0, 1)}}, state="pre")
    post = _temporal({"u": {1: (0, 0), 2: (0, 0)}}, state="post")
    profiles = unit_sign_profiles(pd.concat([pre, post], ignore_index=True))
    paired = paired_state_change(profiles)
    assert len(paired) == 1
    assert paired.suppression_breadth_change.iloc[0] == pytest.approx(-1.)
    assert paired.negative_auc_log2_change.iloc[0] < 0


def test_paired_state_change_needs_both_states():
    profiles = unit_sign_profiles(_temporal({"u": {1: (0, 1)}}, state="pre"))
    with pytest.raises(ValueError, match="no 'post' state"):
        paired_state_change(profiles)


# --- matched comparison ----------------------------------------------------

def _paired(n=600, seed=0, floor_only=True, separated=False):
    """Cells whose change is a pure function of starting level, not class.

    Every cell falls to the same floor, so any apparent class difference comes
    only from where each class started. The classes overlap in breadth, as they
    do in real fields; `separated=True` builds the degenerate case where they
    do not and matching is therefore impossible.
    """
    rng = np.random.default_rng(seed)
    breadth = rng.uniform(0, 1, n)
    if separated:
        awake_class = np.where(breadth > .7, "suppressed_only", "bidirectional")
    else:
        # Suppression-only cells are commoner at high breadth but both classes
        # span the range, so shared strata exist.
        chance = .15 + .5 * breadth
        awake_class = np.where(rng.random(n) < chance,
                               "suppressed_only", "bidirectional")
    change = -breadth.copy()
    if not floor_only:
        change = change - .3 * (awake_class == "suppressed_only")
    return pd.DataFrame({
        "cohort": "TH deep", "unit_id": [f"u{i}" for i in range(n)],
        "awake_class": awake_class, "suppression_breadth__pre": breadth,
        "suppression_breadth_change": change})


def test_matched_comparison_collapses_a_pure_floor_effect():
    """If change depends only on starting level, matching must remove it."""
    result = breadth_matched_change(_paired()).iloc[0]
    assert result.difference_raw < -.1           # apparent before matching
    assert abs(result.difference_matched) < .05  # gone after
    assert result.target_coverage > .9           # and the matching was real


def test_matched_comparison_keeps_a_real_class_effect():
    result = breadth_matched_change(_paired(floor_only=False)).iloc[0]
    assert result.difference_matched == pytest.approx(-.3, abs=.08)


def test_matched_comparison_reveals_when_classes_cannot_be_matched():
    """Perfectly separated classes have no shared strata; coverage says so."""
    result = breadth_matched_change(_paired(separated=True)).iloc[0]
    assert result.target_coverage < .5
    assert result.matched_strata <= 2


def test_matched_comparison_skips_a_cohort_with_too_few_cells():
    small = _paired().head(4).assign(awake_class="suppressed_only")
    assert breadth_matched_change(small).empty


def test_matched_comparison_rejects_a_missing_column():
    with pytest.raises(KeyError, match="awake_class"):
        breadth_matched_change(_paired().drop(columns="awake_class"))
