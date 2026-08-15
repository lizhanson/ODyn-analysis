"""
One HDF5 file per processing round: masks, traces, and their tables together.

Why one file, and why HDF5.

`.npz` is a zip of `.npy` members with **no native MATLAB reader**, and this lab
already analyses these sessions in MATLAB -- every experiment has a
`processed/matlab` folder beside `processed/mcor`. Writing the primary product
in a Python-only container would have made it unreadable to half the people who
need it. HDF5 reads from both (`h5read` in MATLAB, `h5py` here), and from R,
Julia, and ImageJ besides.

Splitting masks, traces, and tables across `.npz` and `.csv` also made the set
separable: three files that can be copied, moved, or overwritten independently,
with nothing tying them together. One file per round cannot come apart.

Naming carries the provenance, so a file identifies itself when it turns up in
someone else's folder:

    group132_20260317_m316_e1_processed_20260812.h5
    ^        ^                  ^          ^
    group    experiment         what       processing date

The date makes rounds additive rather than destructive: re-curate and re-extract
and yesterday's file is still there to compare against.

Layout (MATLAB reads every dataset with `h5read(file, '/path')`):

    /masks/labels              int32  (H, W)          the curated mask
    /masks/<group>             int32  (H, W)          per-odor masks
    /traces/roi                single (n_roi, n_trial, n_frame)  raw F
    /traces/neuropil           single (same)           surrounding annulus
    /traces/time_s             double (n_frame,)       seconds from odor onset
    /rois/<column>             one dataset per column
    /trials/<column>           one dataset per column
    /trials/state              int8 + /trials/state_levels for the labels
    /trials/manipulation       int8 + /trials/manipulation_levels

    attributes on '/': exp_name, group_id, mask_hash, frame_rate, n_pre,
                       n_odor, n_post, processed_on, and the parameters as JSON

**MATLAB transposes.** HDF5 is row-major and MATLAB is column-major, so
`h5read` returns dimensions reversed: `/traces/roi` arrives as
(n_frame, n_trial, n_roi). Use `permute(x, [3 2 1])` to match the Python view.
"""

from __future__ import annotations

import json

from datetime import date
from pathlib import Path

import numpy as np

STAGE = "processed"


def session_filename(
    *,
    group_id: int,
    exp_name: str,
    kind: None | str = None,
    stage: str = STAGE,
    processed_on: None | str = None,
    suffix: str = ".h5",
) -> str:
    """
    `group132_20260317_m316_e1_processed_20260812.h5`

    `kind` names a side-product that is not the round itself, and sits before
    the stage: `..._masks_processed_20260812.mat`. The round keeps the bare
    name, so `find_rounds` picks it out without having to exclude anything.
    """

    stamp = processed_on or date.today().strftime("%Y%m%d")
    what = f"_{kind}" if kind else ""

    return f"group{group_id}_{exp_name}{what}_{stage}_{stamp}{suffix}"


# Every dataset carries a plain-language `description`, and a unit where it has
# one, so `h5disp` or HDFView explains the file without anyone needing this
# module. A file that needs external documentation is not really portable.
COLUMN_DOCS = {
    # /rois
    "roi_id": ("Label value in /masks/labels for this ROI. Row order here "
               "matches the first axis of /traces/roi.", "index"),
    "area_px": ("Number of pixels in the ROI.", "pixels"),
    "diameter_px": ("Diameter of a circle with the same area. Convenience for "
                    "comparing against expected glomerulus size.", "pixels"),
    "neuropil_area_px": ("Pixels in this ROI's surrounding annulus (3 px gap, "
                         "12 px wide, other ROIs removed).", "pixels"),
    "centroid_y": ("Row of the ROI centre of mass in /masks/labels.", "pixels"),
    "centroid_x": ("Column of the ROI centre of mass in /masks/labels.", "pixels"),
    "baseline_fluorescence": ("Mean raw F over all pre-odor frames of all "
                              "trials. Not background-subtracted.", "a.u."),
    "baseline_sd": ("SD of raw F over the same pre-odor frames. Near zero "
                    "means a dead or saturated ROI.", "a.u."),
    "baseline_snr": ("baseline_fluorescence / baseline_sd. Low values flag "
                     "ROIs whose trace is mostly noise.", "ratio"),
    # /trials
    "trial_index": ("Position along the second axis of /traces/roi. Use this "
                    "to index the traces.", "index"),
    "trial_id": ("Primary key in the odyn database `trials` table.", "index"),
    "odor_id": ("Foreign key into the odyn `odors` table.", "index"),
    "state": ("Block label: pre or post the manipulation. Stored as an integer "
              "code; see state_levels for the strings.", ""),
    "manipulation": ("What happened between the pre and post blocks -- "
                     "'saline', 'ketamine/xylazine', 'no injection'. Stored as "
                     "an integer code; see manipulation_levels for the "
                     "strings. Needed to tell a control session from a treated "
                     "one: `state` alone says which side of the manipulation a "
                     "trial is on, not what it was.", ""),
    "odor_on_frame": ("Frame within the acquisition at which the valve opened. "
                      "Varies by a frame or two between trials, so use this for "
                      "exact alignment rather than /traces/time_s.", "frames"),
    "odor_off_frame": ("Frame at which the valve closed.", "frames"),
    "extracted": ("1 if this trial produced data, 0 if it was skipped because "
                  "its window fell outside the acquisition.", "boolean"),
    "detrend_a_fast": ("Median across ROIs of the fast component's amplitude "
                       "for this trial. Tracks how large the acquisition-onset "
                       "transient was; a value far from its neighbours marks a "
                       "trial whose baseline misbehaved.", "a.u."),
    "detrend_a_slow": ("Median across ROIs of the slow component's amplitude "
                       "for this trial.", "a.u."),
}


def _readme() -> str:
    return """Processed output: ROI masks and the raw fluorescence traces extracted from them.

WHAT IS HERE
  /masks/labels     the final ROI mask. 0 = background, N = ROI number N.
  /masks/<odor>     the mask found from that odor alone, before merging.
  /traces/roi       raw fluorescence, axes (roi, trial, frame).
  /traces/neuropil  the surrounding ring, same axes. Measured, not subtracted.
  /traces/time_s    seconds from the MEDIAN odor onset, one per frame.
                    Onset varies by a frame or two; /trials/odor_on_frame has
                    the exact frame for each trial.
  /rois/*           one row per ROI. Row order matches axis 0 of /traces/roi.
  /trials/*         one row per trial. Row order matches axis 1 of /traces/roi.

HOW TO LINE THINGS UP
  Trace of ROI in row i, trial in row j:  /traces/roi[i, j, :]
  That ROI's label in the mask:           /rois/roi_id[i]
  That trial's odor:                      /trials/odor_id[j]
  Time axis for any trace:                /traces/time_s

TEXT COLUMNS
  Stored as integer codes with a companion <column>_levels listing the strings,
  because MATLAB handles variable-length strings poorly. Code k means
  levels[k].

COVERAGE
  Every frame of every acquisition -- the whole pre-odor period, the odor, and
  everything after it. Nothing is cropped, so post-odor dynamics and baseline
  drift are all still here.

WHAT HAS NOT BEEN DONE
  These are raw fluorescence values. No dF/F, no baseline subtraction, no
  neuropil correction, no filtering. Those are later decisions, kept out of
  here deliberately so this file does not commit you to any of them.

MATLAB
  h5disp(file) lists everything with these descriptions.
  x = h5read(file, '/traces/roi') arrives TRANSPOSED as (frame, trial, roi);
  use permute(x, [3 2 1]) to match the layout described above.

PROVENANCE
  Root attributes record exp_name, group_id, processed_on, the frame rate, the
  window, and every parameter as JSON. mask_hash identifies the exact mask
  these traces came from -- see its own attribute for what that is for."""


def _describe(dataset, name: str) -> None:
    doc = COLUMN_DOCS.get(name)
    if doc:
        dataset.attrs["description"], unit = doc
        if unit:
            dataset.attrs["units"] = unit


def _write_table(parent, table) -> None:
    """
    One dataset per column, not a compound type.

    MATLAB reads a compound dataset as a struct whose fields need unpacking,
    and writes back inconsistently. Plain columns are `h5read(f, '/rois/area_px')`
    in MATLAB and `f['rois/area_px'][:]` here.
    """

    for column in table.columns:
        values = table[column].to_numpy()

        if values.dtype == object or values.dtype.kind in "US":
            # Strings become an integer code plus a legend: MATLAB handles
            # variable-length strings badly, and codes sort and filter cleanly.
            levels = sorted({str(v) for v in values})
            lookup = {level: i for i, level in enumerate(levels)}
            coded = parent.create_dataset(
                column, data=np.array([lookup[str(v)] for v in values], dtype=np.int8)
            )
            _describe(coded, column)
            legend = parent.create_dataset(
                f"{column}_levels",
                data=np.array(levels, dtype=h5_string_dtype()),
            )
            legend.attrs["description"] = (
                f"Strings for the integer codes in {column}: code i is entry i here."
            )
        elif values.dtype == bool:
            _describe(parent.create_dataset(column, data=values.astype(np.int8)), column)
        else:
            _describe(parent.create_dataset(column, data=values), column)


def h5_string_dtype():
    import h5py

    return h5py.string_dtype(encoding="ascii")


def write_session(
    path: str | Path,
    *,
    labels: np.ndarray,
    traces=None,
    per_group_masks: None | dict = None,
    exp_name: str,
    group_id: int,
    mask_hash: str,
    parameters: None | dict = None,
    detrend: None | dict = None,
) -> Path:
    """Write one processing round. Overwrites only a file of the same name."""

    from .h5io import open_h5

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open_h5(path, "w") as f:
        f.attrs["exp_name"] = exp_name
        f.attrs["group_id"] = int(group_id)
        f.attrs["mask_hash"] = mask_hash
        f.attrs["mask_hash_description"] = (
            "A fingerprint of /masks/labels: the first 16 hex characters of a "
            "SHA-256 over the label image's contents. Change one pixel of the "
            "mask and it changes completely. Its purpose is to tie these "
            "traces to the exact mask they came from -- re-curate the mask "
            "afterwards and the recorded hash no longer matches, which is the "
            "only way to notice that traces on disk describe ROIs nobody is "
            "looking at any more. Both files would otherwise load fine."
        )
        f.attrs["processed_on"] = date.today().isoformat()
        f.attrs["stage"] = STAGE
        f.attrs["layout_note"] = (
            "MATLAB h5read returns dimensions reversed; "
            "permute(x, [3 2 1]) for /traces/roi"
        )
        if parameters:
            f.attrs["parameters_json"] = json.dumps(parameters, default=str)

        f.create_dataset("README", data=np.array(_readme(), dtype=h5_string_dtype()))
        f["README"].attrs["description"] = "Plain-language guide to this file."

        masks = f.create_group("masks")
        masks.attrs["description"] = (
            "Label images. 0 is background; every positive integer is one ROI, "
            "and that integer is the roi_id in /rois."
        )
        primary = masks.create_dataset(
            "labels", data=labels.astype(np.int32), compression="gzip"
        )
        primary.attrs["description"] = (
            "The final mask, after merging across odors and any manual "
            "curation. This is what /traces was extracted from."
        )
        for key, mask in (per_group_masks or {}).items():
            per_odor = masks.create_dataset(
                str(key), data=np.asarray(mask).astype(np.int32), compression="gzip"
            )
            per_odor.attrs["description"] = (
                f"Mask segmented from odor/group {key} alone, before merging. "
                "Kept so the merge can be redone without re-reading movies."
            )

        if traces is None:
            return path

        group = f.create_group("traces")
        group.attrs["description"] = (
            "Raw fluorescence per ROI per trial. Axes are "
            "(roi, trial, frame): roi follows /rois row order, trial follows "
            "/trials/trial_index, frame is /traces/time_s. No dF/F, no "
            "baseline subtraction, no neuropil correction."
        )

        roi_data = group.create_dataset("roi", data=traces.roi, compression="gzip")
        roi_data.attrs["description"] = (
            "Mean raw fluorescence of the ROI's pixels, per frame. NaN for a "
            "trial that was skipped; see /trials/extracted."
        )
        roi_data.attrs["units"] = "a.u."
        roi_data.attrs["dimensions"] = "roi, trial, frame"

        if traces.neuropil is not None:
            ring = group.create_dataset(
                "neuropil", data=traces.neuropil, compression="gzip"
            )
            ring.attrs["description"] = (
                "Mean raw fluorescence of the surrounding annulus, same axes. "
                "Measured, NOT subtracted -- apply a correction downstream with "
                "a coefficient chosen against the data. Treat as diagnostic at "
                "10x, where the annulus falls in inter-glomerular space rather "
                "than out-of-focus neuropil."
            )
            ring.attrs["units"] = "a.u."
            ring.attrs["dimensions"] = "roi, trial, frame"

        if detrend is not None and detrend.get("ok"):
            # Detrended traces sit BESIDE the raw, never replacing them. The
            # correction is a model with choices in it -- guard window, number
            # of components -- and anyone should be able to redo it, or refuse
            # it, without re-streaming the session.
            corrected = group.create_dataset(
                "roi_detrended", data=detrend["traces"], compression="gzip"
            )
            corrected.attrs["description"] = (
                "Raw F with the fitted acquisition-onset transient subtracted: "
                "F - a_fast*exp(-t/tau_fast) - a_slow*exp(-t/tau_slow). The "
                "constant term is kept, so this is still fluorescence in the "
                "same units, not a residual. See /detrend for the coefficients "
                "and the group attributes for the time constants."
            )
            corrected.attrs["units"] = "a.u."
            corrected.attrs["dimensions"] = "roi, trial, frame"

            fit = f.create_group("detrend")
            fit.attrs["description"] = (
                "Per-ROI per-trial coefficients of the baseline model. Time "
                "constants are estimated once from the population mean, where "
                "they are well determined; amplitudes are fitted per ROI per "
                "trial, which is what lets the correction track a transient "
                "whose size drifts across a session."
            )
            for key in ("tau_fast_s", "tau_slow_s", "r_squared", "guard_s"):
                if detrend.get(key) is not None:
                    fit.attrs[key] = float(detrend[key])
            fit.attrs["model"] = (
                "F(t) = a_fast*exp(-t/tau_fast) + a_slow*exp(-t/tau_slow) + C"
            )
            for name, doc in (
                ("a_fast", "Amplitude of the fast component, per ROI per trial."),
                ("a_slow", "Amplitude of the slow component, per ROI per trial."),
            ):
                d = fit.create_dataset(name, data=np.asarray(detrend[name],
                                                             dtype=np.float32))
                d.attrs["description"] = doc
                d.attrs["units"] = "a.u."
                d.attrs["dimensions"] = "roi, trial"

        times = group.create_dataset("time_s", data=traces.time_s)
        times.attrs["description"] = (
            "Seconds relative to odor onset, one per frame. Negative is "
            "pre-odor; 0 is the frame nearest the valve opening."
        )
        times.attrs["units"] = "seconds"

        f.attrs["frame_rate"] = float(traces.frame_rate)
        f.attrs["n_pre"] = int(traces.n_pre)
        f.attrs["n_odor"] = int(traces.n_odor)
        f.attrs["n_post"] = int(traces.n_post)

        rois = f.create_group("rois")
        rois.attrs["description"] = (
            "One row per ROI; every dataset here has length n_roi and shares "
            "row order with the first axis of /traces/roi."
        )
        _write_table(rois, traces.roi_table(labels))

        trials = f.create_group("trials")
        trial_table = traces.trial_table()

        if detrend is not None and detrend.get("ok"):
            # One number per trial alongside the trial table, so a bad trial is
            # visible in the same place as odor id and state rather than only
            # inside a 2-D array someone has to go looking for.
            trial_table = trial_table.copy()
            trial_table["detrend_a_fast"] = np.nanmedian(detrend["a_fast"], axis=0)
            trial_table["detrend_a_slow"] = np.nanmedian(detrend["a_slow"], axis=0)

        trials.attrs["description"] = (
            "One row per trial; every dataset here has length n_trial and "
            "shares row order with the second axis of /traces/roi."
        )
        _write_table(trials, trial_table)

    return path


def read_session(path: str | Path) -> dict:
    """Read a round back into plain arrays and DataFrames."""

    import pandas as pd

    from .h5io import open_h5

    path = Path(path)
    out: dict = {}

    with open_h5(path) as f:
        out["attrs"] = {
            k: (json.loads(v) if k.endswith("_json") else v)
            for k, v in f.attrs.items()
        }
        out["masks"] = {k: f[f"masks/{k}"][:] for k in f["masks"]}

        if "traces" in f:
            out["traces"] = {k: f[f"traces/{k}"][:] for k in f["traces"]}

        for table in ("rois", "trials"):
            if table not in f:
                continue
            columns = {}
            for name in f[table]:
                if name.endswith("_levels"):
                    continue
                values = f[f"{table}/{name}"][:]
                legend = f.get(f"{table}/{name}_levels")
                if legend is not None:
                    labels_ = [
                        s.decode() if isinstance(s, bytes) else str(s) for s in legend[:]
                    ]
                    values = np.array([labels_[i] for i in values], dtype=object)
                columns[name] = values
            out[table] = pd.DataFrame(columns)

    return out


def find_rounds(output_dir: str | Path, *, stage: str = STAGE) -> list[Path]:
    """Every processing round in a session folder, newest last."""
    return sorted(Path(output_dir).glob(f"group*_{stage}_*.h5"))
