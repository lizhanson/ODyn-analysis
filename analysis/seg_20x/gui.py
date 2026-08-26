"""Streamlined Bokeh GUI for the complete 20x segmentation workflow."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .segmentation import PROCESS_DEFAULTS, SOMA_DEFAULTS
from .state import (
    PHASE_GROUP,
    PHASE_PROCESS_CURATE,
    PHASE_PROCESS_TUNE,
    PHASE_SOMA_CURATE,
    PHASE_SOMA_TUNE,
    Segmentation20xState,
)


def _overlay(somas, processes=None, selected=(), alpha=145):
    out = np.zeros((*somas.shape, 4), np.uint8)
    out[somas > 0] = (0, 220, 255, alpha)
    if processes is not None:
        out[processes > 0] = (255, 0, 210, alpha)
    for kind, ident in selected:
        labels = somas if kind == "soma" else processes
        if labels is not None:
            out[labels == ident] = (255, 220, 0, 230)
    return out.view(np.uint32).reshape(somas.shape)


class Segmentation20xGUI:
    def __init__(self, state: Segmentation20xState, save_path: str | Path):
        self.state = state
        # Notebook autoreload can leave a live state created by an older code version.
        self.state.soma_params = {**SOMA_DEFAULTS, **self.state.soma_params}
        # A live notebook can hold the pre-DoG segmentation module while
        # autoreload has already replaced this GUI class. Do not rely on the
        # imported defaults alone: migrate that mixed-version state explicitly.
        self.state.soma_params.setdefault("dog_threshold", 0.12)
        self.state.soma_params.setdefault("dog_sigma_ratio", 1.4)
        self.state.process_params = {**PROCESS_DEFAULTS, **self.state.process_params}
        self.save_path = Path(save_path)
        self.vertices = []
        self._confirm_back = False

    def modify_doc(self, doc):
        from bokeh.events import Tap
        from bokeh.layouts import column, row
        from bokeh.models import (
            Button, ColumnDataSource, Div, LinearColorMapper,
            RadioButtonGroup, Slider, TextInput,
        )
        from bokeh.palettes import Greys256
        from bokeh.plotting import figure

        h,w = self.state.shape
        finite = self.state.structural[np.isfinite(self.state.structural)]
        lo,hi = np.percentile(finite,(1,99.5))
        self.fig = figure(width=900,height=int(900*h/w)+35,x_range=(0,w),y_range=(0,h),
                          tools="pan,wheel_zoom,reset",active_scroll="wheel_zoom")
        self.fig.axis.visible=False; self.fig.grid.visible=False
        self.bg = ColumnDataSource(dict(image=[self.state.structural]))
        self.fig.image("image",x=0,y=0,dw=w,dh=h,source=self.bg,
                       color_mapper=LinearColorMapper(palette=Greys256,low=lo,high=hi))
        self.mask_source = ColumnDataSource(dict(image=[_overlay(np.zeros(self.state.shape,int))]))
        self.fig.image_rgba("image",x=0,y=0,dw=w,dh=h,source=self.mask_source)
        self.line_source=ColumnDataSource(dict(x=[],y=[]))
        self.fig.line("x","y",source=self.line_source,color="yellow",line_width=3)
        self.fig.scatter("x","y",source=self.line_source,color="yellow",size=6)
        self.fig.on_event(Tap,self._tap)

        self.status=Div(width=900); self.detail=Div(width=900,styles={"font-size":"88%","color":"#555"})
        self.action=RadioButtonGroup(labels=["delete","add soma / draw ridge","select for group"],active=0,width=330)
        self.opacity=Slider(start=25,end=240,step=5,value=145,title="mask opacity",width=330)
        self.opacity.on_change("value",lambda a,o,n:self._refresh())

        self.soma_sliders={}
        for name,title,start,end,step in (
            ("dog_threshold","soma: DoG threshold",.01,.40,.01),
            ("growth_threshold_pctl","soma: growth percentile",30,95,1),
            ("min_diameter_px","soma: min diameter px",2,20,1),
            ("max_diameter_px","soma: max diameter px",10,50,1),
        ):
            widget=Slider(start=start,end=end,step=step,value=self.state.soma_params[name],title=title,width=330)
            widget.on_change("value_throttled",self._soma_callback(name)); self.soma_sliders[name]=widget

        self.process_sliders={}
        for name,title,start,end,step in (
            ("global_ridge_pctl","process: global ridge percentile",0,99,1),
            ("adaptive_block_px","process: adaptive block px",3,101,2),
            ("manual_ridge_corridor_px","manual fill: corridor px",1,10,1),
            ("manual_ridge_adaptive_block_px","manual fill: local block px",3,51,2),
            ("manual_ridge_adaptive_offset","manual fill: local offset",-.03,.03,.001),
            ("min_skeleton_length_px","process: min segment length px",2,50,1),
            ("soma_guard_px","process: soma guard px",0,10,1),
        ):
            widget=Slider(start=start,end=end,step=step,value=self.state.process_params[name],title=title,width=330)
            widget.on_change("value_throttled",self._process_callback(name)); self.process_sliders[name]=widget

        self.group_id=TextInput(value="1",title="roi_group_id",width=155)
        self.advance=Button(label="Lock soma parameters",button_type="primary",width=160); self.advance.on_click(self._advance)
        self.back=Button(label="Back",width=160); self.back.on_click(self._back)
        self.fill=Button(label="Fill drawn ridge",width=160); self.fill.on_click(self._fill)
        self.cancel=Button(label="Cancel line",width=160); self.cancel.on_click(self._cancel)
        self.assign=Button(label="Assign selected",width=160); self.assign.on_click(self._assign)
        self.next_group=Button(label="Next group",width=160); self.next_group.on_click(self._next_group)
        self.clear_selection=Button(label="Clear selection",width=330); self.clear_selection.on_click(self._clear_selection)
        self.save=Button(label="Save masks + groups",button_type="success",width=330); self.save.on_click(self._save)

        controls=column(self.action,self.opacity,Div(text="<b>soma tuning</b>"),*self.soma_sliders.values(),
                        Div(text="<b>process tuning</b>"),*self.process_sliders.values(),
                        row(self.fill,self.cancel),row(self.group_id,self.next_group),self.assign,
                        self.clear_selection,row(self.advance,self.back),self.save,width=350)
        doc.add_root(column(self.status,row(self.fig,controls),self.detail))
        self._refresh("Ready")

    def _soma_callback(self,name):
        def cb(attr,old,new):
            try:self.state.set_soma_param(name,float(new));self._refresh()
            except Exception as error:self._say(str(error))
        return cb

    def _process_callback(self,name):
        def cb(attr,old,new):
            value=int(new) if name in {"adaptive_block_px","manual_ridge_corridor_px",
                                       "manual_ridge_adaptive_block_px","min_skeleton_length_px",
                                       "soma_guard_px"} else float(new)
            try:self.state.set_process_param(name,value);self._refresh()
            except Exception as error:self._say(str(error))
        return cb

    def _coords(self,event):
        h,w=self.state.shape
        return min(max(int(round(event.y)),0),h-1),min(max(int(round(event.x)),0),w-1)

    def _tap(self,event):
        y,x=self._coords(event); phase=self.state.phase
        try:
            if phase==PHASE_SOMA_CURATE:
                if self.action.active==0:self.state.delete_soma_at(y,x)
                elif self.action.active==1:self.state.add_soma(y,x)
            elif phase==PHASE_PROCESS_CURATE:
                if self.action.active==0:self.state.delete_process_at(y,x)
                elif self.action.active==1:
                    self.vertices.append((y,x));self._draw_line();self._refresh();return
            elif phase==PHASE_GROUP and self.action.active==2:
                self.state.toggle_selection(y,x)
            else:
                self._say("Lock the current tuning phase or choose the phase-appropriate action.");return
            self._refresh()
        except Exception as error:self._say(str(error))

    def _advance(self):
        self.state.advance();self.vertices=[];self._draw_line();self._confirm_back=False;self._refresh()

    def _back(self):
        try:self.state.back()
        except RuntimeError as error:
            if self._confirm_back:self.state.back(discard_downstream=True);self._confirm_back=False
            else:self._confirm_back=True;self._say(f"{error} Press Back again to confirm.");return
        self._refresh()

    def _fill(self):
        if self.state.phase!=PHASE_PROCESS_CURATE:self._say("Drawn ridges are available during process curation.");return
        if len(self.vertices)<2:self._say("Draw at least two ridge vertices.");return
        self.state.add_skeleton(self.vertices);self._cancel();self._refresh("Filled manual ridge")

    def _cancel(self):self.vertices=[];self._draw_line()
    def _draw_line(self):self.line_source.data=dict(x=[v[1] for v in self.vertices],y=[v[0] for v in self.vertices])

    def _assign(self):
        try:
            n=self.state.assign_group(self.group_id.value)
            if n:
                self.group_id.value=str(self.state.next_group_id())
            self._refresh(f"Assigned {n} ROI(s)")
        except Exception as error:self._say(str(error))
    def _next_group(self):self.group_id.value=str(self.state.next_group_id())
    def _clear_selection(self):self.state.selected.clear();self._refresh("Selection cleared")
    def _save(self):
        path=self.state.save(self.save_path)
        self._refresh(f"Saved portable bundle {path.name}")
    def _say(self,message):self.detail.text=message

    def _refresh(self,message=""):
        phase=self.state.phase; zero=np.zeros(self.state.shape,np.int32)
        if phase==PHASE_SOMA_TUNE:
            somas=self.state.automatic_somas();processes=None
        elif phase==PHASE_SOMA_CURATE:
            somas=self.state.curated_somas();processes=None
        elif phase==PHASE_PROCESS_TUNE:
            somas=self.state.curated_somas();foreground,_=self.state.process_preview();processes=foreground.astype(np.int32)
        else:
            somas=self.state.curated_somas();processes=self.state.curated_processes()
        selected=self.state.selected if phase==PHASE_GROUP else ()
        self.mask_source.data=dict(image=[_overlay(somas,processes,selected,int(self.opacity.value))])
        n_soma=len(np.unique(somas))-1;n_proc=0 if processes is None else len(np.unique(processes))-1
        self.status.text=(f"<b>[{phase.upper()}]</b> &nbsp; {n_soma} somas · {n_proc} process "
                          f"{'regions' if phase==PHASE_PROCESS_TUNE else 'ROIs'} · "
                          f"{len(self.state.selected)} selected · {len(set(self.state.groups.values()))} groups")
        self.advance.label={PHASE_SOMA_TUNE:"Lock → curate somas",PHASE_SOMA_CURATE:"Freeze somas → tune processes",
                            PHASE_PROCESS_TUNE:"Lock → curate processes",PHASE_PROCESS_CURATE:"Freeze → group ROIs",
                            PHASE_GROUP:"Final phase"}[phase]
        self.advance.disabled=phase==PHASE_GROUP;self.back.disabled=phase==PHASE_SOMA_TUNE
        guidance={PHASE_SOMA_TUNE:"Tune automatic soma candidates, then lock.",
                  PHASE_SOMA_CURATE:"Delete false somas or click to add watershed seeds. Finish before process detection.",
                  PHASE_PROCESS_TUNE:"Tune ridge foreground; magenta is the process foreground preview.",
                  PHASE_PROCESS_CURATE:"Delete process ROIs or click a polyline and fill the missing ridge.",
                  PHASE_GROUP:"Select any soma/process mixture, enter roi_group_id, and assign."}[phase]
        self._say((message+" · " if message else "")+guidance)


def launch(
    structural=None, *, save_path,
    soma_params=None, process_params=None, state=None,
):
    """Open the complete 20x workflow inside a Jupyter/VS Code notebook."""
    import os
    import sys

    import bokeh.plotting as bpl

    from bokeh.io import output_notebook

    if "ipykernel" in sys.modules:
        os.environ["BOKEH_ALLOW_WS_ORIGIN"] = "*"
        # VS Code can discard the frontend output while Bokeh's Python state
        # still says notebook mode is active. Reinitialize it for every launch.
        output_notebook(hide_banner=True)
    if state is None:
        if structural is None:
            raise ValueError("Pass structural= for a new round or state= to resume one.")
        state = Segmentation20xState(
            structural,
            soma_params=soma_params, process_params=process_params,
        )
    gui = Segmentation20xGUI(state, save_path)
    # VS Code Remote/SSH may take longer than Bokeh's five-minute default to
    # establish the proxied websocket, especially after an interactive kernel
    # has been idle.  Keep the one-time session-creation token valid for a day;
    # this does not change the lifetime of an already connected session.
    # Let the OS choose a free ephemeral port. A fixed notebook port remains
    # occupied by an earlier Bokeh server after its output is closed or the
    # launch cell is rerun, producing EADDRINUSE even though no GUI is visible.
    bpl.show(
        gui.modify_doc, port=0,
        session_token_expiration=24 * 60 * 60,
    )
    return gui
