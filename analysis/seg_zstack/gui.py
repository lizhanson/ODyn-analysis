"""Bokeh plane browser and curator for structural z-stacks."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..seg_20x.gui import _overlay
from ..seg_20x.segmentation import PROCESS_DEFAULTS, SOMA_DEFAULTS
from ..seg_20x.state import (
    PHASE_GROUP, PHASE_PROCESS_CURATE, PHASE_PROCESS_TUNE,
    PHASE_SOMA_CURATE, PHASE_SOMA_TUNE,
)
from .state import resolve_depth_reference


class StructuralZStackGUI:
    def __init__(self, state, save_path, *, um_per_px=None, min_soma_diameter_um=None,
                 depth_zero_plane=0, center_depth_um=None, depth_direction=1):
        self.state = state
        self.save_path = Path(save_path)
        self.um_per_px = um_per_px
        self.min_soma_diameter_um = min_soma_diameter_um
        self.depth_reference_plane, self.depth_reference_um = resolve_depth_reference(
            depth_zero_plane, center_depth_um, len(state.planes)
        )
        self.center_depth_um = center_depth_um
        self.depth_direction = int(depth_direction)
        self.vertices = []
        self._confirm_back = False

    def modify_doc(self, doc):
        from bokeh.events import Tap
        from bokeh.layouts import column, row
        from bokeh.models import Button, ColumnDataSource, Div, LinearColorMapper, RadioButtonGroup, Slider
        from bokeh.palettes import Greys256
        from bokeh.plotting import figure

        z, h, w = self.state.structural.shape
        finite = self.state.structural[np.isfinite(self.state.structural)]
        lo, hi = np.percentile(finite, (1, 99.7))
        # Keep the complete viewer within a typical notebook output width.
        plot_width = 620
        control_width = 290
        widget_width = 270
        button_width = 130
        total_width = plot_width + control_width + 20
        self.fig = figure(width=plot_width, height=int(plot_width*h/w)+35,
                          x_range=(0,w), y_range=(0,h),
                          tools="pan,wheel_zoom,reset", active_scroll="wheel_zoom")
        self.fig.axis.visible = False; self.fig.grid.visible = False
        self.bg = ColumnDataSource(dict(image=[self.state.current.structural]))
        self.fig.image("image", x=0, y=0, dw=w, dh=h, source=self.bg,
                       color_mapper=LinearColorMapper(palette=Greys256, low=lo, high=hi))
        self.mask_source = ColumnDataSource(dict(image=[_overlay(np.zeros((h,w), int))]))
        self.fig.image_rgba("image", x=0, y=0, dw=w, dh=h, source=self.mask_source)
        self.line_source = ColumnDataSource(dict(x=[], y=[]))
        self.fig.line("x", "y", source=self.line_source, color="yellow", line_width=3)
        self.fig.scatter("x", "y", source=self.line_source, color="yellow", size=6)
        self.fig.on_event(Tap, self._tap)

        self.status = Div(width=total_width)
        self.detail = Div(width=total_width, styles={"font-size":"88%", "color":"#555"})
        self.plane = Slider(start=0, end=z-1, step=1, value=0, title="plane (0 based)", width=widget_width)
        self.plane.on_change("value", self._plane)
        self.action = RadioButtonGroup(labels=["delete", "add / draw"], active=0, width=widget_width)
        self.opacity = Slider(start=25,end=240,step=5,value=145,title="mask opacity",width=widget_width)
        self.opacity.on_change("value", lambda a,o,n:self._refresh())

        self.soma_sliders = {}
        for name,title,start,end,step in (
            ("dog_threshold","soma: DoG threshold",.01,.40,.01),
            ("growth_threshold_pctl","soma: growth percentile",30,95,1),
            ("min_diameter_px","soma: min diameter px",2,20,1),
            ("max_diameter_px","soma: max diameter px",10,50,1),
        ):
            widget = Slider(start=start,end=end,step=step,value=self.state.current.soma_params[name],title=title,width=widget_width)
            widget.on_change("value_throttled", self._parameter("soma", name))
            self.soma_sliders[name] = widget
        self.process_sliders = {}
        for name,title,start,end,step in (
            ("global_ridge_pctl","process: global ridge percentile",0,99,1),
            ("adaptive_block_px","process: adaptive block px",3,101,2),
            ("min_skeleton_length_px","process: min segment length px",2,50,1),
            ("soma_guard_px","process: soma guard px",0,10,1),
        ):
            widget = Slider(start=start,end=end,step=step,value=self.state.current.process_params[name],title=title,width=widget_width)
            widget.on_change("value_throttled", self._parameter("process", name))
            self.process_sliders[name] = widget
        self.advance = Button(label="Lock soma parameters", button_type="primary", width=button_width)
        self.advance.on_click(self._advance)
        self.back = Button(label="Back", width=button_width); self.back.on_click(self._back)
        self.fill = Button(label="Fill ridge", width=button_width); self.fill.on_click(self._fill)
        self.cancel = Button(label="Cancel line", width=button_width); self.cancel.on_click(self._cancel)
        self.save = Button(label="Save stack + tables", button_type="success", width=widget_width); self.save.on_click(self._save)
        controls = column(self.plane, self.action, self.opacity, Div(text="<b>soma tuning (all planes)</b>"),
                          *self.soma_sliders.values(), Div(text="<b>process tuning (all planes)</b>"),
                          *self.process_sliders.values(), row(self.fill,self.cancel),
                          row(self.advance,self.back), self.save, width=control_width)
        doc.add_root(column(self.status, row(self.fig,controls), self.detail))
        self._refresh("Ready")

    def _plane(self, attr, old, new):
        self.state.set_plane(int(new)); self.vertices=[]; self._draw_line(); self._refresh()

    def _parameter(self, kind, name):
        integer = name in {"adaptive_block_px","min_skeleton_length_px","soma_guard_px"}
        def callback(attr, old, new):
            try:
                value = int(new) if integer else float(new)
                (self.state.set_soma_param if kind == "soma" else self.state.set_process_param)(name, value)
                self._refresh()
            except Exception as error: self._say(str(error))
        return callback

    def _tap(self, event):
        y = min(max(int(round(event.y)),0), self.state.current.shape[0]-1)
        x = min(max(int(round(event.x)),0), self.state.current.shape[1]-1)
        phase = self.state.phase
        try:
            if phase == PHASE_SOMA_CURATE:
                self.state.current.delete_soma_at(y,x) if self.action.active == 0 else self.state.current.add_soma(y,x)
            elif phase == PHASE_PROCESS_CURATE:
                if self.action.active == 0: self.state.current.delete_process_at(y,x)
                else:
                    self.vertices.append((y,x)); self._draw_line(); return
            else:
                self._say("Lock tuning before curation; choose delete or add for the current phase."); return
            self._refresh()
        except Exception as error: self._say(str(error))

    def _advance(self):
        try:
            self._say("Segmenting every plane; this can take a minute …")
            self.state.advance(); self.vertices=[]; self._draw_line(); self._confirm_back=False; self._refresh()
        except Exception as error: self._say(str(error))

    def _back(self):
        try: self.state.back()
        except RuntimeError as error:
            if self._confirm_back: self.state.back(discard_downstream=True); self._confirm_back=False
            else: self._confirm_back=True; self._say(f"{error} Press Back again to confirm."); return
        self._refresh()

    def _fill(self):
        if self.state.phase != PHASE_PROCESS_CURATE: self._say("Ridge drawing is available during process curation."); return
        if len(self.vertices) < 2: self._say("Draw at least two ridge vertices."); return
        self.state.current.add_skeleton(self.vertices); self._cancel(); self._refresh("Filled manual ridge")

    def _cancel(self): self.vertices=[]; self._draw_line()
    def _draw_line(self): self.line_source.data=dict(x=[v[1] for v in self.vertices],y=[v[0] for v in self.vertices])
    def _say(self, message): self.detail.text=message
    def _save(self):
        path = self.state.save(self.save_path, um_per_px=self.um_per_px,
                               min_soma_diameter_um=self.min_soma_diameter_um,
                               depth_zero_plane=self.depth_reference_plane,
                               center_depth_um=self.center_depth_um,
                               depth_direction=self.depth_direction)
        self._refresh(f"Saved {path.name} and CSV tables")

    def _refresh(self, message=""):
        state, phase = self.state.current, self.state.phase
        self.bg.data = dict(image=[state.structural])
        if phase == PHASE_SOMA_TUNE: somas, processes = state.automatic_somas(), None
        elif phase == PHASE_SOMA_CURATE: somas, processes = state.curated_somas(), None
        elif phase == PHASE_PROCESS_TUNE:
            somas = state.curated_somas(); processes = state.process_preview()[0].astype(np.int32)
        else: somas, processes = state.curated_somas(), state.curated_processes()
        self.mask_source.data = dict(image=[_overlay(somas, processes, alpha=int(self.opacity.value))])
        n_soma = int(somas.max()); n_proc = 0 if processes is None else int(processes.max())
        step = self.state.metadata.get("z_step_um")
        depth = "uncalibrated z" if step is None else f"depth {self.depth_reference_um + (self.state.plane-self.depth_reference_plane)*step*self.depth_direction:g} µm"
        self.status.text = f"<b>[{phase.upper()}]</b> plane {self.state.plane+1}/{len(self.state.planes)} · {depth} · {n_soma} somas · {n_proc} process ROIs"
        self.advance.label = {PHASE_SOMA_TUNE:"Auto-segment all → curate somas", PHASE_SOMA_CURATE:"Freeze somas → tune processes",
                              PHASE_PROCESS_TUNE:"Auto-segment all → curate processes", PHASE_PROCESS_CURATE:"Finish curation",
                              PHASE_GROUP:"Complete"}[phase]
        self.advance.disabled = phase == PHASE_GROUP; self.back.disabled = phase == PHASE_SOMA_TUNE
        guidance = {PHASE_SOMA_TUNE:"Tune on representative planes; parameters apply to all planes.",
                    PHASE_SOMA_CURATE:"Visit planes and delete/add soma profiles.",
                    PHASE_PROCESS_TUNE:"Tune process foreground on representative planes.",
                    PHASE_PROCESS_CURATE:"Visit planes and delete/add process ridges.",
                    PHASE_GROUP:"Structural curation is complete; save the bundle and CSV summaries."}[phase]
        self._say((message+" · " if message else "")+guidance)


def launch(state, *, save_path, um_per_px=None, min_soma_diameter_um=None,
           depth_zero_plane=0, center_depth_um=None, depth_direction=1):
    import os, sys
    import bokeh.plotting as bpl
    from bokeh.io import output_notebook
    from ..session.bokeh import stop_notebook_servers
    if "ipykernel" in sys.modules:
        os.environ["BOKEH_ALLOW_WS_ORIGIN"] = "*"; output_notebook(hide_banner=True)
    gui = StructuralZStackGUI(state, save_path, um_per_px=um_per_px,
                              min_soma_diameter_um=min_soma_diameter_um,
                              depth_zero_plane=depth_zero_plane,
                              center_depth_um=center_depth_um,
                              depth_direction=depth_direction)
    stop_notebook_servers()
    bpl.show(gui.modify_doc, port=5007, session_token_expiration=24*60*60)
    return gui
