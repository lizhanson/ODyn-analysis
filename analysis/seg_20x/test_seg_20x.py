from types import SimpleNamespace

import numpy as np
import pandas as pd

from .session import resolve_approved_group
from .state import PHASE_GROUP, PHASE_PROCESS_CURATE, Segmentation20xState
from .gui import Segmentation20xGUI


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
    path=tmp_path/"round.npz"; state.save(path)
    table=pd.read_csv(path.with_suffix(".csv"))
    assert set(table.roi_group_id.dropna().astype(int)) == {4}
    assert set(table.roi_type) == {"soma","process"}
    resumed = Segmentation20xState.load(path)
    assert resumed.phase == PHASE_GROUP
    assert resumed.groups == {("soma",1):4, ("process",1):4}
    portable = Segmentation20xState.load_portable(path.with_suffix(".h5"))
    assert portable.phase == PHASE_GROUP
    assert portable.groups == resumed.groups
    np.testing.assert_array_equal(portable.curated_somas(), resumed.curated_somas())
    np.testing.assert_array_equal(portable.curated_processes(), resumed.curated_processes())


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
