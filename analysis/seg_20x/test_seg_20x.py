from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from .session import resolve_approved_group
from .grouping import group_rois, pairwise_gaps, proximity_correlation_profile
from .state import (
    PHASE_GROUP, PHASE_PROCESS_CURATE, PHASE_PROCESS_TUNE, PHASE_SOMA_CURATE,
    BUNDLE_TYPE, SCHEMA_VERSION, Segmentation20xState, save_portable_state,
)
from .gui import Segmentation20xGUI
from .qc import aggregate_raw_units


def test_group_qc_aggregates_raw_f_by_pixels_and_keeps_singletons():
    raw = np.array([
        [[[10., 20.]]],   # soma, 4 px
        [[[30., 50.]]],   # process, 2 px; grouped with soma
        [[[70., 90.]]],   # ungrouped process, 3 px
    ]).reshape(3, 1, 2)
    areas = np.array([4, 2, 3])
    manifest = [
        {"roi_id": 1, "roi_type": "soma", "source_roi_id": 1, "roi_group_id": 7},
        {"roi_id": 2, "roi_type": "process", "source_roi_id": 1, "roi_group_id": 7},
        {"roi_id": 3, "roi_type": "process", "source_roi_id": 2, "roi_group_id": None},
    ]

    populations = aggregate_raw_units(raw, areas, manifest)

    np.testing.assert_allclose(populations["groups"].raw[0, 0], [50 / 3, 30])
    np.testing.assert_allclose(populations["somas"].raw[0, 0], [10, 20])
    np.testing.assert_allclose(populations["processes"].raw[0, 0], [30, 50])
    np.testing.assert_allclose(populations["processes"].raw[1, 0], [70, 90])
    assert populations["groups"].members == [[1, 2], [3]]
    assert populations["groups"].area_px.tolist() == [6, 3]


def test_post_extraction_groups_override_saved_manifest_groups():
    raw = np.arange(12, dtype=float).reshape(3, 1, 4)
    manifest = [
        {"roi_id": 1, "roi_type": "soma", "source_roi_id": 1, "roi_group_id": -1},
        {"roi_id": 2, "roi_type": "process", "source_roi_id": 1, "roi_group_id": -1},
        {"roi_id": 3, "roi_type": "process", "source_roi_id": 2, "roi_group_id": 9},
    ]
    populations = aggregate_raw_units(
        raw, np.ones(3), manifest, {("soma", 1): 4, ("process", 1): 4}
    )
    assert populations["groups"].members == [[1, 2], [3]]
    assert populations["groups"].unit_ids == ["g4", "p3"]


def test_group_qc_can_build_one_float32_population_for_bounded_memory():
    raw = np.arange(24, dtype=np.float32).reshape(3, 2, 4)
    manifest = [
        {"roi_id": 1, "roi_type": "soma", "source_roi_id": 1, "roi_group_id": 1},
        {"roi_id": 2, "roi_type": "process", "source_roi_id": 1, "roi_group_id": 1},
        {"roi_id": 3, "roi_type": "process", "source_roi_id": 2, "roi_group_id": None},
    ]
    populations = aggregate_raw_units(
        raw, np.ones(3), manifest, only=("processes",),
    )
    assert set(populations) == {"processes"}
    assert populations["processes"].raw.dtype == np.float32


def test_gui_migrates_live_legacy_log_state_to_dog(tmp_path):
    state = Segmentation20xState(np.zeros((20, 20), np.float32))
    state.soma_params.pop("dog_threshold", None)
    state.soma_params.pop("dog_sigma_ratio", None)
    state.soma_params["log_threshold"] = 0.16
    Segmentation20xGUI(state, tmp_path / "masks.h5")
    assert state.soma_params["dog_threshold"] == 0.12
    assert state.soma_params["dog_sigma_ratio"] == 1.4


def test_resolver_uses_only_approved_mcor(tmp_path):
    group = SimpleNamespace(
        main_folder=tmp_path,
        group_experiments=pd.DataFrame({"group_id":[7],"exp_id":[3]}),
        experiments=pd.DataFrame({
            "exp_id":[3],"exp_name":["example"],"exp_start":["2026-01-01"],
            "height_px":[20],"width_px":[30],"height_um":[20.],"width_um":[30.],
            "frame_count":[10],"frame_rate":[5.],
        }),
        acquisitions=pd.DataFrame({"acq_id":[1,2,3],"exp_id":[3,3,3]}),
        mcor_files=pd.DataFrame({
            "acq_id":[1,2,3],"mcor_path":["a.tif","b.tif","c.tif"],
            "source":["caiman"]*3,"approved":[0,1,1],
        }),
    )
    resolved = resolve_approved_group(group, 7)
    assert resolved.acq_ids == (2,3)
    assert [p.name for p in resolved.paths] == ["b.tif","c.tif"]


def test_ordered_phases_and_mixed_roi_groups(monkeypatch, tmp_path):
    soma = np.zeros((30,30),np.int32); soma[5:9,5:9]=1
    process = np.zeros_like(soma); process[15,10:20]=1

    def fake_somas(image, params):
        return soma.copy(), {"n_rois":1}

    def fake_processes(structural, soma_labels, params, ridge=None):
        return process.copy(), {"n_rois":1,"ridge":np.zeros_like(structural),
                                "foreground":process>0,"skeleton":process>0,"markers":process}

    monkeypatch.setattr("analysis.seg_20x.state.detect_somas",fake_somas)
    monkeypatch.setattr("analysis.seg_20x.state.detect_processes",fake_processes)
    monkeypatch.setattr("analysis.seg_20x.state.ridge_image",lambda image,params:np.zeros_like(image))

    state=Segmentation20xState(np.zeros((30,30),np.float32))
    for _ in range(4): state.advance()
    assert state.phase == PHASE_GROUP
    state.toggle_selection(6,6); state.toggle_selection(15,12)
    assert state.assign_group(4)==2
    table = state.roi_table()
    assert set(table.roi_group_id.astype(int)) == {4}
    assert set(table.roi_type) == {"soma","process"}
    path = state.save(tmp_path/"round")
    assert path.name == "round.h5"
    resumed = Segmentation20xState.load(path)
    assert resumed.phase == PHASE_GROUP
    assert resumed.groups == {("soma",1):4, ("process",1):4}
    np.testing.assert_array_equal(resumed.curated_somas(), soma)
    np.testing.assert_array_equal(resumed.curated_processes(), process)


def test_standalone_saver_recovers_live_state_without_bound_save(tmp_path):
    import h5py

    state = Segmentation20xState(np.zeros((12, 12), np.float32))
    state._automatic_somas = np.zeros(state.shape, np.int32)
    state._automatic_processes = np.zeros(state.shape, np.int32)
    path = save_portable_state(state, tmp_path / "recovered.npz")

    assert path.suffix == ".h5"
    with h5py.File(path, "r") as handle:
        assert handle.attrs["file_type"] == BUNDLE_TYPE
        assert handle.attrs["schema_version"] == SCHEMA_VERSION


def test_manual_ridge_fill_is_narrow_and_locally_thresholded():
    shape = (31, 31)
    state = Segmentation20xState(np.zeros(shape, np.float32))
    state._automatic_somas = np.zeros(shape, np.int32)
    state._automatic_processes = np.zeros(shape, np.int32)
    state._ridge = np.full(shape, 0.5, np.float32)
    state.phase = PHASE_PROCESS_CURATE
    state.add_skeleton([(15, 10), (15, 20)])

    labels = state.curated_processes()

    # A flat local background fails the deliberately tight local threshold,
    # so the user's 11-pixel ridge itself remains instead of a broad corridor.
    assert np.count_nonzero(labels) == 11


def test_trace_notebook_uses_standard_h5_finalizer():
    notebook = (__file__.replace("test_seg_20x.py", "../Stage1_20x_segmentation.ipynb"))
    text = open(notebook).read()
    assert "finalize_session" in text
    assert "traces_20x.npz" not in text


def test_assign_button_advances_group_number():
    gui = Segmentation20xGUI.__new__(Segmentation20xGUI)
    gui.state = SimpleNamespace(assign_group=lambda value: 2, next_group_id=lambda: 8)
    gui.group_id = SimpleNamespace(value="7")
    gui._refresh = lambda message="": None
    gui._say = lambda message: None

    gui._assign()

    assert gui.group_id.value == "8"


def test_gui_upgrades_live_state_created_before_new_parameters(tmp_path):
    state = Segmentation20xState(np.zeros((20, 20), np.float32))
    del state.process_params["manual_ridge_corridor_px"]
    del state.process_params["manual_ridge_adaptive_block_px"]
    del state.process_params["manual_ridge_adaptive_offset"]

    Segmentation20xGUI(state, tmp_path / "round.npz")

    assert state.process_params["manual_ridge_corridor_px"] == 3
    assert state.process_params["manual_ridge_adaptive_block_px"] == 11
    assert state.process_params["manual_ridge_adaptive_offset"] == -0.006


def test_ridge_cache_is_dropped_when_a_ridge_parameter_changes():
    state = Segmentation20xState(np.random.default_rng(0).random((40, 40)).astype(np.float32))
    state.phase = PHASE_PROCESS_TUNE

    before = state.ridge().copy()
    state.set_process_param("ridge_sigma_max_px", 12)

    assert not np.array_equal(before, state.ridge())


def test_a_non_ridge_parameter_keeps_the_ridge_cache():
    state = Segmentation20xState(np.random.default_rng(0).random((40, 40)).astype(np.float32))
    state.phase = PHASE_PROCESS_TUNE
    before = state.ridge()

    state.set_process_param("global_ridge_pctl", 55.0)

    assert state.ridge() is before


def test_curated_masks_are_cached_but_follow_edits(monkeypatch):
    calls = []

    def counted_somas(image, params):
        calls.append(1)
        labels = np.zeros(image.shape, np.int32)
        labels[5:9, 5:9] = 1
        return labels, {}

    monkeypatch.setattr("analysis.seg_20x.state.detect_somas", counted_somas)
    state = Segmentation20xState(np.zeros((30, 30), np.float32))
    state.phase = PHASE_SOMA_CURATE

    first = state.curated_somas()
    assert state.curated_somas() is first
    assert len(calls) == 1

    state.delete_soma_at(6, 6)

    assert state.curated_somas() is not first
    assert not state.curated_somas().any()


def test_resumed_bundle_uses_stored_labels_instead_of_detecting_again(monkeypatch, tmp_path):
    soma = np.zeros((30, 30), np.int32); soma[5:9, 5:9] = 1
    process = np.zeros_like(soma); process[15, 10:20] = 1
    monkeypatch.setattr(
        "analysis.seg_20x.state.detect_somas", lambda image, params: (soma.copy(), {})
    )
    monkeypatch.setattr(
        "analysis.seg_20x.state.detect_processes",
        lambda structural, soma_labels, params, ridge=None: (process.copy(), {}),
    )
    monkeypatch.setattr(
        "analysis.seg_20x.state.ridge_image", lambda image, params: np.zeros_like(image)
    )
    state = Segmentation20xState(np.zeros((30, 30), np.float32))
    for _ in range(4):
        state.advance()
    path = state.save(tmp_path / "round")

    def refuse(*args, **kwargs):
        raise AssertionError("a resumed round must not run the detectors again")

    monkeypatch.setattr("analysis.seg_20x.state.detect_somas", refuse)
    monkeypatch.setattr("analysis.seg_20x.state.detect_processes", refuse)

    resumed = Segmentation20xState.load(path)

    np.testing.assert_array_equal(resumed.curated_somas(), soma)
    np.testing.assert_array_equal(resumed.curated_processes(), process)


def test_bundle_whose_curated_masks_disagree_with_its_edits_is_refused(monkeypatch, tmp_path):
    soma = np.zeros((30, 30), np.int32); soma[5:9, 5:9] = 1
    monkeypatch.setattr(
        "analysis.seg_20x.state.detect_somas", lambda image, params: (soma.copy(), {})
    )
    monkeypatch.setattr(
        "analysis.seg_20x.state.detect_processes",
        lambda structural, soma_labels, params, ridge=None: (np.zeros_like(soma), {}),
    )
    monkeypatch.setattr(
        "analysis.seg_20x.state.ridge_image", lambda image, params: np.zeros_like(image)
    )
    state = Segmentation20xState(np.zeros((30, 30), np.float32))
    for _ in range(4):
        state.advance()
    path = state.save(tmp_path / "round")

    import h5py
    with h5py.File(path, "r+") as handle:
        handle["masks/soma"][...] = np.zeros_like(soma)

    with pytest.raises(ValueError, match="does not reproduce"):
        Segmentation20xState.load(path)


def test_gaps_are_measured_between_nearest_pixels_not_centroids():
    index = np.full((20, 40), -1, np.int32)
    index[10, 0:20] = 0          # a long horizontal process
    index[10, 23:26] = 1         # nearest pixels are columns 19 and 23
    index[0, 0] = 2              # far away

    gaps = pairwise_gaps(index, 3, max_gap_px=5.0)

    # Centre to centre, as the distance transform measures it, and the far ROI
    # is not reported at all rather than reported as out of range.
    assert gaps == {(0, 1): 4.0}


def test_grouping_needs_both_proximity_and_correlation():
    somas = np.zeros((20, 60), np.int32)
    somas[9:12, 2:5] = 1                       # soma 1, left
    somas[9:12, 32:35] = 2                     # soma 2, right
    processes = np.zeros_like(somas)
    processes[10, 6:20] = 1                    # touches soma 1, correlated
    processes[10, 36:50] = 2                   # touches soma 2, correlated
    processes[10, 21:28] = 3                   # near soma 1 but uncorrelated
    processes[2, 50:58] = 4                    # correlated but nowhere near

    rng = np.random.default_rng(0)
    a, b = rng.normal(size=400), rng.normal(size=400)
    traces = {
        ("soma", 1): a, ("process", 1): a + 0.1 * rng.normal(size=400),
        ("soma", 2): b, ("process", 2): b + 0.1 * rng.normal(size=400),
        ("process", 3): rng.normal(size=400),
        ("process", 4): a + 0.1 * rng.normal(size=400),
    }

    groups, diagnostics = group_rois(
        somas, processes, traces, um_per_px=0.5,
        params={"max_gap_um": 1.5, "min_correlation": 0.5},
    )

    assert groups[("soma", 1)] == groups[("process", 1)]
    assert groups[("soma", 2)] == groups[("process", 2)]
    assert groups[("soma", 1)] != groups[("soma", 2)]
    # Adjacent but uncorrelated, and correlated but not adjacent: neither joins.
    assert ("process", 3) not in groups
    assert ("process", 4) not in groups
    assert diagnostics.linked.sum() == 2


def test_two_somas_never_chain_into_one_group():
    somas = np.zeros((20, 40), np.int32)
    somas[9:12, 2:5] = 1
    somas[9:12, 20:23] = 2
    processes = np.zeros_like(somas)
    processes[10, 6:19] = 1     # bridges the two somas, correlated with both

    shared = np.random.default_rng(1).normal(size=300)
    traces = {key: shared for key in
              (("soma", 1), ("soma", 2), ("process", 1))}

    groups, _ = group_rois(
        somas, processes, traces, um_per_px=0.5,
        params={"max_gap_um": 2.0, "min_correlation": 0.5, "drop_singletons": False},
    )

    # The bridge joins one soma; the other keeps its own group rather than
    # being fused through it, which is what single linkage would have done.
    assert groups[("soma", 1)] != groups[("soma", 2)]
    assert len(set(groups.values())) == 2
    assert groups[("process", 1)] in (groups[("soma", 1)], groups[("soma", 2)])


def test_proximity_profile_reports_correlation_against_gap():
    somas = np.zeros((20, 40), np.int32)
    somas[9:12, 2:5] = 1
    processes = np.zeros_like(somas)
    processes[10, 6:12] = 1
    processes[10, 20:26] = 2

    shared = np.random.default_rng(2).normal(size=200)
    traces = {("soma", 1): shared, ("process", 1): shared,
              ("process", 2): np.random.default_rng(3).normal(size=200)}

    profile = proximity_correlation_profile(
        somas, processes, traces, um_per_px=0.5,
        params={"profile_max_gap_um": 12.0, "profile_bin_um": 2.0},
    )

    assert set(profile.columns) == {"pair_type", "gap_bin_um", "correlation", "n"}
    assert profile.n.sum() == 3
    nearest = profile.sort_values("gap_bin_um").iloc[0]
    assert nearest.correlation > 0.9


def test_traces_from_round_are_smoothed_dff_event_windows(tmp_path):
    import h5py
    from .grouping import traces_from_round

    shape = np.array([0., 1., 2., 1., 0., -.5, 0., 0.])
    roi = np.zeros((2, 2, 12), float)
    for trial, scale in enumerate((1.0, 1.5)):
        roi[0, trial, :4] = 10 * scale
        roi[0, trial, 4:] = 10 * scale * (1 + shape)
        roi[1, trial, :4] = 100 * scale
        roi[1, trial, 4:] = 100 * scale * (1 + shape)
    path = tmp_path / "round.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("traces/roi", data=roi)
        handle.create_dataset("trials/odor_on_frame", data=[4, 4])
        handle.attrs["frame_rate"] = 1.0

    manifest = [
        {"roi_id": 1, "roi_type": "soma", "source_roi_id": 1},
        {"roi_id": 2, "roi_type": "process", "source_roi_id": 1},
    ]
    traces, source = traces_from_round(
        path, manifest, baseline_s=4, odor_s=4, post_s=4,
        smooth_sigma_frames=1,
    )

    assert source == "raw"
    assert set(traces) == {("soma", 1), ("process", 1)}
    assert traces[("soma", 1)].shape == (2, 8)
    np.testing.assert_allclose(traces[("soma", 1)], traces[("process", 1)], atol=1e-6)


def test_grouping_accepts_one_frame_lag():
    from .grouping import _correlations

    base = np.sin(np.linspace(0, 3 * np.pi, 40))
    traces = np.stack([base, np.r_[base[1:], base[-1]]])[:, None, :]
    assert _correlations(traces, max_lag_frames=0)[0, 1] < 0.99
    assert _correlations(traces, max_lag_frames=1)[0, 1] > 0.999


def test_processes_group_without_any_soma():
    somas = np.zeros((20, 40), np.int32)
    somas[2:4, 36:38] = 1            # a soma far from everything below
    processes = np.zeros_like(somas)
    processes[10, 2:12] = 1
    processes[10, 13:23] = 2         # abuts process 1, same signal

    shared = np.random.default_rng(4).normal(size=300)
    traces = {
        ("soma", 1): np.random.default_rng(5).normal(size=300),
        ("process", 1): shared,
        ("process", 2): shared + 0.05 * np.random.default_rng(6).normal(size=300),
    }

    groups, _ = group_rois(somas, processes, traces, um_per_px=0.5,
                           params={"max_gap_um": 1.5, "min_correlation": 0.5})

    # A neurite chain whose parent soma is out of plane is still a group.
    assert groups[("process", 1)] == groups[("process", 2)]
    assert ("soma", 1) not in groups


def test_a_process_bridging_two_somas_joins_only_the_stronger_one():
    somas = np.zeros((20, 40), np.int32)
    somas[9:12, 2:5] = 1
    somas[9:12, 20:23] = 2
    processes = np.zeros_like(somas)
    processes[10, 6:19] = 1          # touches both

    rng = np.random.default_rng(7)
    shared = rng.normal(size=400)
    traces = {
        ("soma", 1): shared + 0.05 * rng.normal(size=400),   # the better match
        ("soma", 2): shared + 0.60 * rng.normal(size=400),
        ("process", 1): shared,
    }

    groups, diagnostics = group_rois(
        somas, processes, traces, um_per_px=0.5,
        params={"max_gap_um": 2.0, "min_correlation": 0.3, "drop_singletons": False},
    )

    assert groups[("process", 1)] == groups[("soma", 1)]
    assert groups[("soma", 2)] != groups[("soma", 1)]
    # The losing link is reported as refused, not quietly missing.
    refused = diagnostics[diagnostics.status == "two_somas"]
    assert len(refused) == 1
    assert set(refused.iloc[0][["roi_a", "roi_b"]]) == {"p1", "s2"}


def test_no_group_ever_holds_two_somas():
    somas = np.zeros((20, 60), np.int32)
    for n, x in enumerate((2, 20, 38), start=1):
        somas[9:12, x:x+3] = n
    processes = np.zeros_like(somas)
    processes[10, 6:19] = 1          # bridges somas 1 and 2
    processes[10, 24:37] = 2         # bridges somas 2 and 3

    shared = np.random.default_rng(8).normal(size=400)
    traces = {key: shared for key in
              [("soma", 1), ("soma", 2), ("soma", 3), ("process", 1), ("process", 2)]}

    groups, _ = group_rois(somas, processes, traces, um_per_px=0.5,
                           params={"max_gap_um": 2.0, "min_correlation": 0.3})

    per_group = {}
    for (kind, _), gid in groups.items():
        per_group[gid] = per_group.get(gid, 0) + (kind == "soma")
    assert per_group and max(per_group.values()) == 1
