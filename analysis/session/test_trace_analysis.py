import numpy as np

from .trace_qc import _preferred_odor_order
from .trace_analysis import (
    aggregate_epoch_table, epoch_scores, standardize_traces, trial_epoch_table,
)


def test_standardize_uses_each_trials_own_baseline_mean_and_sd():
    traces = np.zeros((1, 2, 12), np.float32)
    traces[0, 0, :4] = [8, 9, 11, 12]
    traces[0, 1, :4] = [96, 98, 102, 104]
    traces[0, 0, 4:] = 12
    traces[0, 1, 4:] = 104
    result = standardize_traces(
        traces, odor_on_frames=[4, 4], states=[0, 1], n_state_levels=2,
        frame_rate=1, baseline_s=4, baseline_sd_mode="per_trial",
    )
    expected = np.sqrt(np.mean([
        np.var([8, 9, 11, 12], ddof=1), np.var([96, 98, 102, 104], ddof=1)
    ]))
    np.testing.assert_allclose(result.baseline_sd_session, [expected])
    np.testing.assert_allclose(result.z[0, :, :4].mean(axis=1), [0, 0], atol=1e-6)
    np.testing.assert_allclose(result.z[0, :, :4].std(axis=1, ddof=1), [1, 1])
    np.testing.assert_allclose(
        result.baseline_sd_trial,
        [[np.std([8, 9, 11, 12], ddof=1), np.std([96, 98, 102, 104], ddof=1)]],
    )
    assert result.baseline_sd_block[0, 1] > result.baseline_sd_block[0, 0]


def test_standardize_can_apply_pre_block_pooled_sd_to_every_trial():
    traces = np.zeros((1, 3, 8), np.float32)
    traces[0, 0, :4] = [8, 9, 11, 12]
    traces[0, 1, :4] = [18, 19, 21, 22]
    traces[0, 2, :4] = [80, 90, 110, 120]
    traces[:, :, 4:] = traces[:, :, :4].mean(axis=2, keepdims=True) + 10
    result = standardize_traces(
        traces, odor_on_frames=[4, 4, 4], states=[1, 1, 0],
        n_state_levels=2, frame_rate=1, baseline_s=4,
        baseline_sd_mode="pre_block_pooled", pre_state_code=1,
    )

    pre_sd = np.sqrt(np.mean([
        np.var([8, 9, 11, 12], ddof=1),
        np.var([18, 19, 21, 22], ddof=1),
    ]))
    np.testing.assert_allclose(result.normalization_sd, pre_sd)
    np.testing.assert_allclose(result.z[0, :, 4:].mean(axis=1), 10 / pre_sd)
    assert result.baseline_sd_mode == "pre_block_pooled"


def test_epoch_scores_requires_exact_odor_and_four_second_post_windows():
    z = np.zeros((1, 2, 14), np.float32)
    z[0, 0, 4:8] = 2
    z[0, 0, 8:12] = -3
    z[0, 1, 4:8] = 5
    scores = epoch_scores(
        z, odor_on_frames=[4, 4], odor_off_frames=[8, 8],
        frame_rate=1, post_odor_s=4,
    )
    assert scores.mean["odor"][0, 0] == 2
    assert scores.mean["post_odor"][0, 0] == -3
    # The second trial cannot provide all four post-odor seconds.
    with np.testing.assert_raises(ValueError):
        epoch_scores(z, odor_on_frames=[4, 4], odor_off_frames=[8, 11],
                     frame_rate=1, post_odor_s=4)


def test_tables_are_trial_level_then_unit_odor_block_epoch_level():
    z = np.zeros((1, 4, 14), np.float32)
    z[0, :, 4:8] = np.array([1, 3, 2, 4])[:, None]
    scores = epoch_scores(z, odor_on_frames=[4] * 4, odor_off_frames=[8] * 4,
                          frame_rate=1)
    trials = trial_epoch_table(
        scores, unit_ids=["g1"], odor_ids=[7] * 4, states=[0, 0, 1, 1],
        state_levels=["pre", "post"], trial_ids=[10, 11, 12, 13],
    )
    summary = aggregate_epoch_table(trials)
    odor = summary[summary.epoch == "odor"].set_index("block")
    assert odor.loc["pre", "n_trials"] == 2
    assert odor.loc["pre", "mean_response_z"] == 2
    assert odor.loc["post", "mean_response_z"] == 3
    assert set(summary.columns).isdisjoint({"p_value", "q_value", "call"})


def test_preferred_odor_order_uses_only_reference_trials():
    odor_ids = np.array([1, 2, 1, 2])
    response = np.array([
        [4, 1, 9, 9],
        [2, 3, 9, 9],
        [5, 1, 9, 9],
        [np.nan, np.nan, 9, 9],
    ])
    order, preference, keys = _preferred_odor_order(
        response, odor_ids,
        reference_trials=np.array([True, True, False, False]),
    )

    np.testing.assert_array_equal(keys, [1, 2])
    np.testing.assert_array_equal(order, [2, 0, 1, 3])
    np.testing.assert_array_equal(preference, [0, 0, 1, 2])
