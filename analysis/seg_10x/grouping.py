"""Post-extraction joining of neighboring, correlated 10x ROI fragments."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

GROUPING_DEFAULTS = {
    "max_gap_px": 8.0,
    "min_correlation": 0.70,
    "max_lag_frames": 1,
    "odor_s": 4.0,
    "post_s": 4.0,
}


def trace_windows(round_path, *, odor_s=4.0, post_s=4.0):
    """Canonical component traces over odor plus post-odor time."""
    from scipy.ndimage import gaussian_filter1d
    from ..session.h5io import open_h5

    with open_h5(round_path) as handle:
        z = handle["traces/roi_z"][:]
        on = handle["trials/odor_on_frame"][:].astype(int)
        rate = float(handle.attrs["frame_rate"])
    length = int(round((float(odor_s) + float(post_s)) * rate))
    windows = np.stack([z[:, trial, start:start + length]
                        for trial, start in enumerate(on)], axis=1)
    return gaussian_filter1d(windows, sigma=2.0, axis=2, mode="nearest")


def candidate_pairs(labels, round_path, *, params=None):
    """All nearby ROI pairs with trace correlation, strongest first."""
    import pandas as pd
    from ..seg_20x.grouping import _correlations, pairwise_gaps

    p = {**GROUPING_DEFAULTS, **(params or {})}
    labels = np.asarray(labels)
    roi_ids = [int(v) for v in np.unique(labels[labels > 0])]
    index = np.full(labels.shape, -1, np.int32)
    for position, roi_id in enumerate(roi_ids):
        index[labels == roi_id] = position
    traces = trace_windows(round_path, odor_s=p["odor_s"], post_s=p["post_s"])
    if traces.shape[0] != len(roi_ids):
        raise ValueError("Extracted trace rows do not align with mask ROI ids")
    corr = _correlations(traces, p["max_lag_frames"])
    gaps = pairwise_gaps(index, len(roi_ids), p["max_gap_px"])
    rows = [{
        "roi_a": roi_ids[i], "roi_b": roi_ids[j],
        "gap_px": round(float(gap), 3),
        "correlation": round(float(corr[i, j]), 4),
        "suggested": bool(corr[i, j] >= p["min_correlation"]),
    } for (i, j), gap in gaps.items()]
    return pd.DataFrame(rows, columns=(
        "roi_a", "roi_b", "gap_px", "correlation", "suggested"
    )).sort_values(["suggested", "correlation"], ascending=False,
                   ignore_index=True)


@dataclass
class JoiningState:
    labels: np.ndarray
    reference: np.ndarray
    round_path: Path
    mask_hash: str
    candidates: object
    params: dict
    groups: dict[int, int] = field(default_factory=dict)
    selected: set[int] = field(default_factory=set)

    def assign(self, group_id):
        gid = int(group_id)
        for roi_id in self.selected:
            self.groups[int(roi_id)] = gid
        count = len(self.selected)
        self.selected.clear()
        return count

    def clear_selected_groups(self):
        for roi_id in self.selected:
            self.groups.pop(int(roi_id), None)
        self.selected.clear()

    def next_group_id(self):
        return max(self.groups.values(), default=0) + 1

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "file_type": "odyn_10x_roi_groups",
            "source_round": str(self.round_path),
            "mask_hash": self.mask_hash,
            "params": self.params,
            "groups": {str(k): int(v) for k, v in sorted(self.groups.items())},
        }
        partial = path.with_suffix(path.suffix + ".partial")
        partial.write_text(json.dumps(payload, indent=2) + "\n")
        partial.replace(path)
        return path


def load_groups(path, *, expected_mask_hash=None):
    payload = json.loads(Path(path).read_text())
    if payload.get("file_type") != "odyn_10x_roi_groups":
        raise ValueError(f"Not a 10x ROI-group file: {path}")
    if expected_mask_hash and payload.get("mask_hash") != expected_mask_hash:
        raise ValueError("ROI groups were made from a different segmentation mask")
    return {int(k): int(v) for k, v in payload.get("groups", {}).items()}, payload


class JoiningGUI:
    """Click ROIs to assign reviewed fragment joins."""

    def __init__(self, state, save_path):
        self.state = state
        self.save_path = Path(save_path)
        suggested = state.candidates[state.candidates.suggested].copy()
        self.suggestions = (suggested if len(suggested)
                            else state.candidates.copy()).head(100).reset_index(drop=True)
        self.suggestion_index = 0

    def modify_doc(self, doc):
        from bokeh.events import Tap
        from bokeh.layouts import column, row
        from bokeh.models import (
            Button, ColumnDataSource, DataTable, Div, LinearColorMapper,
            NumberFormatter, TableColumn, TextInput,
        )
        from bokeh.palettes import Greys256
        from bokeh.plotting import figure
        from .gui import _grey_image

        h, w = self.state.labels.shape
        image, lo, hi = _grey_image(self.state.reference)
        plot_width = 620
        plot_height = min(570, max(380, int(plot_width * h / w) + 25))
        self.fig = figure(width=plot_width, height=plot_height,
                          x_range=(0, w), y_range=(0, h),
                          tools="pan,wheel_zoom,reset", active_scroll="wheel_zoom")
        self.fig.axis.visible = False; self.fig.grid.visible = False
        self.fig.image(image=[image], x=0, y=0, dw=w, dh=h,
                       color_mapper=LinearColorMapper(palette=Greys256, low=lo, high=hi))
        self.overlay = ColumnDataSource(dict(image=[self._overlay()]))
        self.fig.image_rgba(image="image", x=0, y=0, dw=w, dh=h,
                            source=self.overlay)
        self.fig.on_event(Tap, self._tap)
        instructions = Div(width=360, text=(
            "<b>How to join fragments</b><br>"
            "<span style='color:#e67e00'>Orange</span> = current correlated-neighbor "
            "suggestion; <span style='color:#b59b00'>yellow</span> = selected; "
            "colored = saved join; gray = singleton.<br>"
            "Click any ROI to toggle it, or click a candidate row and use "
            "<b>Select orange pair</b>. Then assign. Only explicitly assigned "
            "ROIs are joined."
        ))
        self.status = Div(width=980)
        self.group_id = TextInput(value=str(self.state.next_group_id()),
                                  title="join group id", width=150)
        assign = Button(label="Assign selected", button_type="primary", width=175)
        assign.on_click(self._assign)
        clear = Button(label="Make selected singleton", width=175)
        clear.on_click(self._clear)
        save = Button(label="Save reviewed groups", button_type="success", width=350)
        save.on_click(self._save)
        previous = Button(label="← Previous", width=105)
        previous.on_click(lambda: self._move_suggestion(-1))
        select_pair = Button(label="Select orange pair", width=140,
                             button_type="warning")
        select_pair.on_click(self._select_suggestion)
        following = Button(label="Next →", width=105)
        following.on_click(lambda: self._move_suggestion(1))

        table_data = {name: self.suggestions[name].tolist()
                      for name in ("roi_a", "roi_b", "gap_px", "correlation")}
        self.candidate_source = ColumnDataSource(table_data)
        self.candidate_source.selected.on_change("indices", self._table_selection)
        candidate_table = DataTable(
            source=self.candidate_source, width=360, height=250,
            index_position=None, selectable=True,
            columns=[
                TableColumn(field="roi_a", title="ROI A", width=65),
                TableColumn(field="roi_b", title="ROI B", width=65),
                TableColumn(field="gap_px", title="gap px", width=75,
                            formatter=NumberFormatter(format="0.0")),
                TableColumn(field="correlation", title="r", width=75,
                            formatter=NumberFormatter(format="0.000")),
            ],
        )
        controls = column(
            instructions,
            Div(text=(f"<b>Ranked suggestions</b>: r ≥ "
                      f"{self.state.params['min_correlation']:.2f}, gap ≤ "
                      f"{self.state.params['max_gap_px']:g} px"), width=360),
            candidate_table, row(previous, select_pair, following),
            row(self.group_id, assign), clear, save, width=360,
        )
        doc.add_root(column(self.status, row(self.fig, controls)))
        self._refresh()

    def _overlay(self):
        from .gui import _PALETTE
        h, w = self.state.labels.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        for roi_id in np.unique(self.state.labels[self.state.labels > 0]):
            gid = self.state.groups.get(int(roi_id))
            colour = ((145, 145, 145) if gid is None
                      else _PALETTE[(int(gid) - 1) % len(_PALETTE)])
            rgba[self.state.labels == roi_id] = (*colour, 105 if gid is None else 145)
        pair = self._suggestion_pair()
        for roi_id in pair:
            rgba[self.state.labels == roi_id] = (255, 125, 0, 190)
        for roi_id in self.state.selected:
            rgba[self.state.labels == roi_id] = (255, 230, 0, 235)
        return rgba.view(np.uint32).reshape(h, w)

    def _suggestion_pair(self):
        if not len(self.suggestions):
            return ()
        item = self.suggestions.iloc[self.suggestion_index]
        return int(item.roi_a), int(item.roi_b)

    def _move_suggestion(self, step):
        if len(self.suggestions):
            self.suggestion_index = ((self.suggestion_index + int(step))
                                     % len(self.suggestions))
            self.candidate_source.selected.indices = [self.suggestion_index]
            self._refresh()

    def _select_suggestion(self):
        self.state.selected = set(self._suggestion_pair())
        self._refresh("Selected suggested pair")

    def _table_selection(self, attr, old, new):
        if new:
            self.suggestion_index = int(new[-1])
            self._refresh()

    def _tap(self, event):
        y, x = int(round(event.y)), int(round(event.x))
        if 0 <= y < self.state.labels.shape[0] and 0 <= x < self.state.labels.shape[1]:
            roi_id = int(self.state.labels[y, x])
            if roi_id:
                if roi_id in self.state.selected:
                    self.state.selected.remove(roi_id)
                else:
                    self.state.selected.add(roi_id)
                self._refresh()

    def _assign(self):
        self.state.assign(self.group_id.value)
        self.group_id.value = str(self.state.next_group_id())
        self._refresh("Assigned group")

    def _clear(self):
        self.state.clear_selected_groups(); self._refresh("Restored singleton(s)")

    def _save(self):
        self.state.save(self.save_path); self._refresh(f"Saved {self.save_path.name}")

    def _refresh(self, message=""):
        self.overlay.data = dict(image=[self._overlay()])
        chosen = sorted(self.state.selected)
        pair = self.state.candidates
        detail = ""
        if len(chosen) == 2:
            match = pair[((pair.roi_a == chosen[0]) & (pair.roi_b == chosen[1])) |
                         ((pair.roi_a == chosen[1]) & (pair.roi_b == chosen[0]))]
            if len(match):
                row = match.iloc[0]
                detail = f"; gap {row.gap_px:g}px, r={row.correlation:.3f}"
        suggestion = self._suggestion_pair()
        suggestion_text = "none"
        if suggestion:
            item = self.suggestions.iloc[self.suggestion_index]
            suggestion_text = (f"{self.suggestion_index + 1}/{len(self.suggestions)}: "
                               f"ROI {suggestion[0]} + {suggestion[1]}, "
                               f"gap {item.gap_px:g}px, r={item.correlation:.3f}")
        self.status.text = (
            f"<b>{message}</b> Current suggestion {suggestion_text}. "
            f"Selected {chosen}{detail}; {len(set(self.state.groups.values()))} joins."
        )


def prepare_joining(round_path, reference, *, params=None, groups_path=None):
    from ..session.finalize import mask_hash
    from ..session.h5io import open_h5
    p = {**GROUPING_DEFAULTS, **(params or {})}
    with open_h5(round_path) as handle:
        labels = handle["masks/labels"][:]
    digest = mask_hash(labels)
    groups = {}
    if groups_path is not None and Path(groups_path).exists():
        groups, _ = load_groups(groups_path, expected_mask_hash=digest)
    return JoiningState(labels, np.asarray(reference), Path(round_path), digest,
                        candidate_pairs(labels, round_path, params=p), p, groups)


def launch_joining(state, save_path):
    import bokeh.plotting as bpl
    from ..session.bokeh import ensure_notebook_output

    ensure_notebook_output()
    gui = JoiningGUI(state, save_path)
    bpl.show(gui.modify_doc, session_token_expiration=24 * 60 * 60)
    return gui
