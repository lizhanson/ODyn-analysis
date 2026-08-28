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
    """All ROI correlations, annotated with the local spatial-neighbor graph."""
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
    rows = []
    for i in range(len(roi_ids)):
        for j in range(i + 1, len(roi_ids)):
            gap = gaps.get((i, j))
            spatial = gap is not None
            correlation_pass = bool(corr[i, j] >= p["min_correlation"])
            rows.append({
                "roi_a": roi_ids[i], "roi_b": roi_ids[j],
                "gap_px": np.nan if gap is None else round(float(gap), 3),
                "correlation": round(float(corr[i, j]), 4),
                "spatial_neighbor": spatial,
                "correlation_pass": correlation_pass,
                "suggested": spatial and correlation_pass,
            })
    return pd.DataFrame(rows, columns=(
        "roi_a", "roi_b", "gap_px", "correlation", "spatial_neighbor",
        "correlation_pass", "suggested",
    )).sort_values(["suggested", "correlation"], ascending=False,
                   ignore_index=True)


def fully_connected_suggestions(pairs):
    """Maximal sets with connected spatial graphs and complete correlation graphs.

    Every member must spatially neighbor at least one other member (equivalently,
    the induced spatial graph is connected), but spatial adjacency need not be
    pairwise. Correlation *is* pairwise: every possible member pair must pass.
    """
    import pandas as pd

    spatial_neighbors = {}
    correlation_pass = {}
    edge_values = {}
    for row in pairs.itertuples(index=False):
        a, b = int(row.roi_a), int(row.roi_b)
        spatial = bool(getattr(row, "spatial_neighbor", np.isfinite(row.gap_px)))
        passes = bool(getattr(row, "correlation_pass", row.suggested))
        correlation_pass[(min(a, b), max(a, b))] = passes
        edge_values[(min(a, b), max(a, b))] = (
            float(row.gap_px), float(row.correlation), spatial)
        spatial_neighbors.setdefault(a, set())
        spatial_neighbors.setdefault(b, set())
        if spatial:
            spatial_neighbors[a].add(b)
            spatial_neighbors[b].add(a)

    valid_sets = set()
    visited = set()

    def correlations_pass(node, chosen):
        return all(correlation_pass.get((min(node, other), max(node, other)), False)
                   for other in chosen)

    def expand(chosen):
        frozen = frozenset(chosen)
        if frozen in visited:
            return
        visited.add(frozen)
        valid_sets.add(frozen)
        frontier = set().union(*(spatial_neighbors[node] for node in chosen)) - chosen
        for node in sorted(frontier):
            if correlations_pass(node, chosen):
                expand(set(chosen) | {node})

    # Seed only with spatial edges whose two traces correlate. Expansion may
    # then chain spatially, but checks a new ROI against every existing trace.
    for (a, b), (_gap, _corr, spatial) in edge_values.items():
        if spatial and correlation_pass[(a, b)]:
            expand({a, b})

    maximal = [members for members in valid_sets
               if not any(members < other for other in valid_sets)]
    rows = []
    for frozen in sorted(maximal, key=lambda value: tuple(sorted(value))):
        members = tuple(sorted(frozen))
        values = [edge_values[(members[i], members[j])]
                  for i in range(len(members)) for j in range(i + 1, len(members))]
        spatial_gaps = [value[0] for value in values if value[2]]
        rows.append({
            "members": members,
            "n_rois": len(members),
            "max_gap_px": max(spatial_gaps),
            "min_correlation": min(value[1] for value in values),
            "mean_correlation": float(np.mean([value[1] for value in values])),
        })
    return pd.DataFrame(rows, columns=(
        "members", "n_rois", "max_gap_px", "min_correlation", "mean_correlation"
    )).sort_values(
        ["n_rois", "min_correlation", "max_gap_px"],
        ascending=[False, False, True], ignore_index=True,
    )


@dataclass
class JoiningState:
    labels: np.ndarray
    reference: np.ndarray
    round_path: Path
    mask_hash: str
    candidates: object
    params: dict
    suggestions: object = None
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
        suggestions = getattr(state, "suggestions", None)
        required = {"members", "n_rois", "max_gap_px", "min_correlation"}
        if suggestions is None or not required.issubset(
            getattr(suggestions, "columns", ())
        ):
            # Autoreload can update this GUI class while an existing state still
            # carries the previous pair-level suggestion table. Upgrade it at
            # the launch boundary rather than failing inside Tornado later.
            suggestions = fully_connected_suggestions(state.candidates)
            state.suggestions = suggestions
        self.suggestions = suggestions.head(100).reset_index(drop=True)
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
            "<span style='color:#e67e00'>Orange</span> = current fully connected "
            "correlated-neighbor set; "
            "<span style='color:#b59b00'>yellow</span> = selected; "
            "colored = saved join; gray = singleton.<br>"
            "A set may chain spatially, but its displayed minimum r is computed "
            "across <i>every</i> member pair.<br>"
            "Click any ROI to toggle it, or click a candidate row and use "
            "<b>Select orange set</b>. Then assign. Only explicitly assigned "
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
        select_pair = Button(label="Select orange set", width=140,
                             button_type="warning")
        select_pair.on_click(self._select_suggestion)
        following = Button(label="Next →", width=105)
        following.on_click(lambda: self._move_suggestion(1))

        table_data = {
            "members": [", ".join(map(str, value)) for value in self.suggestions.members],
            "n_rois": self.suggestions.n_rois.tolist(),
            "max_gap_px": self.suggestions.max_gap_px.tolist(),
            "min_correlation": self.suggestions.min_correlation.tolist(),
        }
        self.candidate_source = ColumnDataSource(table_data)
        self.candidate_source.selected.on_change("indices", self._table_selection)
        candidate_table = DataTable(
            source=self.candidate_source, width=360, height=250,
            index_position=None, selectable=True,
            columns=[
                TableColumn(field="members", title="ROI members", width=130),
                TableColumn(field="n_rois", title="n", width=35),
                TableColumn(field="max_gap_px", title="max gap", width=70,
                            formatter=NumberFormatter(format="0.0")),
                TableColumn(field="min_correlation", title="min r", width=65,
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
        suggestion = self._suggestion_rois()
        for roi_id in suggestion:
            rgba[self.state.labels == roi_id] = (255, 125, 0, 190)
        for roi_id in self.state.selected:
            rgba[self.state.labels == roi_id] = (255, 230, 0, 235)
        return rgba.view(np.uint32).reshape(h, w)

    def _suggestion_rois(self):
        if not len(self.suggestions):
            return ()
        item = self.suggestions.iloc[self.suggestion_index]
        return tuple(int(value) for value in item.members)

    def _move_suggestion(self, step):
        if len(self.suggestions):
            self.suggestion_index = ((self.suggestion_index + int(step))
                                     % len(self.suggestions))
            self.candidate_source.selected.indices = [self.suggestion_index]
            self._refresh()

    def _select_suggestion(self):
        self.state.selected = set(self._suggestion_rois())
        self._refresh("Selected suggested set")

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
        suggestion = self._suggestion_rois()
        suggestion_text = "none"
        if suggestion:
            item = self.suggestions.iloc[self.suggestion_index]
            suggestion_text = (f"{self.suggestion_index + 1}/{len(self.suggestions)}: "
                               f"ROIs {', '.join(map(str, suggestion))}, "
                               f"max gap {item.max_gap_px:g}px, "
                               f"min r={item.min_correlation:.3f}")
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
    pairs = candidate_pairs(labels, round_path, params=p)
    suggestions = fully_connected_suggestions(pairs)
    return JoiningState(labels, np.asarray(reference), Path(round_path), digest,
                        pairs, p, suggestions, groups)


def launch_joining(state, save_path):
    import bokeh.plotting as bpl
    from ..session.bokeh import ensure_notebook_output

    ensure_notebook_output()
    gui = JoiningGUI(state, save_path)
    bpl.show(gui.modify_doc, session_token_expiration=24 * 60 * 60)
    return gui
