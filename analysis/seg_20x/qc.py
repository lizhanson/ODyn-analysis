"""QC for 20x soma/process rounds, with grouped ROIs as analysis units."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class UnitPopulation:
    """One representation of the same biological units."""

    name: str
    raw: np.ndarray                 # unit x trial x frame, raw pixel-weighted F
    area_px: np.ndarray
    unit_ids: list[str]
    members: list[list[int]]        # final roi_ids contributing to each row
    member_types: list[tuple[str, ...]]


def _manifest_groups(parameters: dict) -> dict[tuple[str, int], object]:
    rows = parameters.get("segmentation", {}).get("roi_manifest", [])
    return {
        (str(row["roi_type"]), int(row["source_roi_id"])): row.get("roi_group_id")
        for row in rows
        if row.get("roi_group_id") not in (None, -1, "-1")
    }


def aggregate_raw_units(raw, areas, manifest, groups=None) -> dict[str, UnitPopulation]:
    """Pixel-weight ROI means into whole, soma-only, and process-only units.

    Grouped ROIs share a unit. Every ungrouped ROI becomes a singleton. Weighting
    the raw ROI means by their pixel counts is exactly the mean over the union of
    their (disjoint) mask pixels. No detrending or normalization happens here.
    """
    raw = np.asarray(raw, float)
    areas = np.asarray(areas, float)
    if raw.ndim != 3 or raw.shape[0] != len(areas):
        raise ValueError("raw must be ROI x trial x frame and align with areas")

    rows = sorted(manifest, key=lambda row: int(row["roi_id"]))
    if len(rows) != raw.shape[0]:
        raise ValueError("roi_manifest does not align with the raw trace rows")
    supplied = None if groups is None else dict(groups)

    buckets: dict[object, list[dict]] = {}
    for row in rows:
        roi_id = int(row["roi_id"])
        key = (str(row["roi_type"]), int(row["source_roi_id"]))
        gid = row.get("roi_group_id") if supplied is None else supplied.get(key)
        grouped = gid not in (None, -1, "-1")
        unit = ("group", gid) if grouped else ("roi", roi_id)
        buckets.setdefault(unit, []).append(row)

    def build(name, kind=None):
        traces, pixel_counts, ids, members, member_types = [], [], [], [], []
        for unit, unit_rows in buckets.items():
            chosen = unit_rows if kind is None else [r for r in unit_rows if r["roi_type"] == kind]
            if not chosen:
                continue
            indices = np.array([int(r["roi_id"]) - 1 for r in chosen])
            weights = areas[indices]
            traces.append(np.average(raw[indices], axis=0, weights=weights))
            pixel_counts.append(weights.sum())
            ids.append(f"g{unit[1]}" if unit[0] == "group" else f"{chosen[0]['roi_type'][0]}{unit[1]}")
            members.append([int(r["roi_id"]) for r in chosen])
            member_types.append(tuple(sorted({str(r["roi_type"]) for r in chosen})))
        shape = (0, *raw.shape[1:])
        return UnitPopulation(name, np.stack(traces) if traces else np.empty(shape),
                              np.asarray(pixel_counts), ids, members, member_types)

    return {
        "groups": build("whole groups + singleton ROIs"),
        "somas": build("soma component", "soma"),
        "processes": build("group minus soma", "process"),
    }


def _dff(raw, on_frames, frame_rate, baseline_s=4.0):
    out = np.full_like(raw, np.nan, dtype=float)
    n_base = max(1, int(round(baseline_s * frame_rate)))
    for trial, onset in enumerate(np.asarray(on_frames, int)):
        f0 = np.nanmean(raw[:, trial, max(0, onset - n_base):onset], axis=1)
        out[:, trial] = (raw[:, trial] - f0[:, None]) / np.maximum(np.abs(f0[:, None]), 1e-6)
    return out


def _snr(raw, on_frames, frame_rate, baseline_s=4.0):
    n_base = max(1, int(round(baseline_s * frame_rate)))
    pieces = [raw[:, i, max(0, int(on) - n_base):int(on)] for i, on in enumerate(on_frames)]
    baseline = np.concatenate(pieces, axis=1)
    return np.nanmean(baseline, axis=1) / np.maximum(np.nanstd(baseline, axis=1), 1e-9)


def _analyse(populations, *, on, off, odor_ids, states, state_levels, frame_rate,
             usable=None, deglobal=None, target_fdr=0.05):
    from ..session.detrend import detrend_traces
    from ..session.pc1 import fixed_pc1
    from ..session.responders import population_sparsity, trial_calls

    analysed = {}
    for key, pop in populations.items():
        corrected, fit = detrend_traces(pop.raw, odor_on_frames=on,
                                        odor_off_frames=off, frame_rate=frame_rate)
        dff = _dff(corrected, on, frame_rate)
        calls = trial_calls(dff, odor_on_frames=on, odor_ids=odor_ids,
                            usable=usable, deglobal=deglobal, target_fdr=target_fdr)
        masks = {"pooled": np.ones(len(odor_ids), bool)}
        masks.update({name: states == i for i, name in enumerate(state_levels)})
        analysed[key] = {
            "population": pop, "corrected": corrected,
            "dff": dff, "fit": fit, "calls": calls,
            "time_pc1": fixed_pc1(dff),
            "sparsity": {name: population_sparsity(calls, odor_ids, trials=mask)
                         for name, mask in masks.items()},
            "snr": _snr(corrected, on, frame_rate),
        }
    return analysed


def _spatial_figure(path, structural, masks, manifest, groups, analysed):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    labels = np.asarray(masks["labels"])
    finite = np.concatenate([analysed[k]["snr"] for k in analysed])
    finite = finite[np.isfinite(finite) & (finite > 0)]
    lo, hi = (np.percentile(finite, [5, 95]) if len(finite) else (1, 10))
    lo, hi = max(float(lo), 1e-3), max(float(hi), float(lo) * 1.01)
    base_lo, base_hi = np.nanpercentile(structural, [1, 99.5])
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), constrained_layout=True)
    for ax, key in zip(axes[:3], ("groups", "somas", "processes")):
        values = np.full(labels.shape, np.nan)
        pop = analysed[key]["population"]
        for value, member_ids in zip(analysed[key]["snr"], pop.members):
            values[np.isin(labels, member_ids)] = value
        ax.imshow(structural, cmap="gray", vmin=base_lo, vmax=base_hi)
        image = ax.imshow(np.ma.masked_invalid(values), cmap="viridis",
                          norm=LogNorm(lo, hi), alpha=.85)
        ax.set_title(pop.name + ": baseline SNR")
        ax.set(xticks=[], yticks=[])
    fig.colorbar(image, ax=axes[:3], label="baseline F / SD", shrink=.75)
    for key, colour in zip(("groups", "somas", "processes"),
                           ("black", "deepskyblue", "magenta")):
        axes[3].scatter(analysed[key]["population"].area_px,
                        analysed[key]["snr"], s=14, alpha=.45, color=colour,
                        label=key)
    axes[3].set(xlabel="component area (px)", ylabel="baseline F / SD",
                title="size versus SNR")
    axes[3].legend(fontsize=8)
    fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)


def _atlas(stem, key, item, on, off, frame_rate, rows_per_page=24):
    import matplotlib.pyplot as plt

    pop, dff = item["population"], item["dff"]
    paths = []
    t = (np.arange(dff.shape[2]) - int(np.median(on))) / frame_rate
    odor_end = np.median(np.asarray(off) - np.asarray(on)) / frame_rate
    for page, start in enumerate(range(0, len(pop.unit_ids), rows_per_page), 1):
        stop = min(start + rows_per_page, len(pop.unit_ids))
        fig, axes = plt.subplots(stop - start, 1, figsize=(13, max(3, .8 * (stop-start))),
                                 sharex=True, squeeze=False)
        for ax, i in zip(axes[:, 0], range(start, stop)):
            ax.plot(t, dff[i].T, color="0.72", lw=.35, alpha=.45)
            ax.plot(t, np.nanmean(dff[i], axis=0), color="black", lw=1)
            ax.axvspan(0, odor_end, color="gold", alpha=.16)
            ax.text(.002, .82, f"{pop.unit_ids[i]}  {pop.area_px[i]:.0f}px  SNR {item['snr'][i]:.1f}",
                    transform=ax.transAxes, fontsize=7)
            ax.axhline(0, color="0.5", lw=.4)
        axes[-1, 0].set_xlabel("seconds from odor onset")
        fig.suptitle(pop.name + " — detrend then dF/F; trials gray, mean black")
        out = Path(f"{stem}_traceatlas_{key}_p{page:02d}.png")
        fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig); paths.append(str(out))
    return paths


def _response_figure(path, analysed, odor_ids, states, state_levels):
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from ..session.responders import _display_order, _sparsity_panel

    keys = np.unique(odor_ids); levels, rank = _display_order(list(state_levels))
    columns = np.lexsort((np.arange(len(odor_ids)), rank[states], odor_ids))
    fig, axes = plt.subplots(3, 3, figsize=(19, 14), constrained_layout=True)
    norm = TwoSlopeNorm(vmin=-5, vcenter=0, vmax=5)
    for row, key in enumerate(("groups", "somas", "processes")):
        item = analysed[key]; z = item["calls"]["z"]
        # Whole units containing a soma first, then process-only units. The
        # component displays naturally contain only one type.
        type_rank = np.array([0 if "soma" in kinds else 1
                              for kinds in item["population"].member_types])
        order = np.lexsort((-np.nanmax(np.abs(np.nan_to_num(z)), axis=1), type_rank))
        axes[row, 0].imshow(z[np.ix_(order, columns)], aspect="auto", cmap="RdBu_r",
                            norm=norm, interpolation="nearest")
        axes[row, 0].set(title=f"{item['population'].name}: single-trial z",
                         ylabel="analysis unit")
        sorted_odor, sorted_rank = odor_ids[columns], rank[states][columns]
        for edge in np.flatnonzero(np.diff(sorted_odor)) + 1:
            axes[row, 0].axvline(edge-.5, color="black", lw=1)
        for edge in np.flatnonzero((np.diff(sorted_rank) != 0) & (np.diff(sorted_odor) == 0)) + 1:
            axes[row, 0].axvline(edge-.5, color="0.4", ls=":", lw=.7)
        axes[row, 0].set_xticks([np.mean(np.flatnonzero(sorted_odor == k)) for k in keys],
                                [str(int(k)) for k in keys])
        _sparsity_panel(axes[row, 1], item["sparsity"]["pooled"], keys,
                        title="excited / suppressed (pooled)")
        by_state = {name: item["sparsity"][name] for name in levels}
        _sparsity_panel(axes[row, 2], by_state, keys,
                        title="excited / suppressed by state", grouped=True)
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)


def _snr_figure(path, analysed):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    for key, colour in zip(("groups", "somas", "processes"), ("black", "deepskyblue", "magenta")):
        values = analysed[key]["snr"]; values = values[np.isfinite(values)]
        ax.hist(values, bins=35, histtype="step", lw=2, color=colour,
                label=f"{analysed[key]['population'].name} (n={len(values)})")
    ax.set(xlabel="baseline SNR (mean F / SD)", ylabel="analysis units", title="20x baseline SNR")
    ax.legend(fontsize=8); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def qc_20x(round_path, *, groups=None, structural=None, usable=None,
           deglobal=None, target_fdr=0.05, save=True) -> dict:
    """Generate grouped 20x QC; all transformations follow raw-F aggregation."""
    from ..session.h5io import open_h5
    round_path = Path(round_path)
    with open_h5(round_path) as f:
        raw = f["traces/roi"][:]; areas = f["rois/area_px"][:]
        on = f["trials/odor_on_frame"][:]; off = f["trials/odor_off_frame"][:]
        odor_ids = f["trials/odor_id"][:]; states = f["trials/state"][:]
        state_levels = [x.decode() if isinstance(x, bytes) else str(x)
                        for x in f["trials/state_levels"][:]]
        frame_rate = float(f.attrs["frame_rate"])
        parameters = json.loads(f.attrs["parameters_json"])
        masks = {name: f[f"masks/{name}"][:] for name in ("labels", "soma", "process")}
    manifest = parameters["segmentation"]["roi_manifest"]
    effective_groups = _manifest_groups(parameters) if groups is None else groups
    populations = aggregate_raw_units(raw, areas, manifest, effective_groups)
    analysed = _analyse(populations, on=on, off=off, odor_ids=odor_ids, states=states,
                         state_levels=state_levels, frame_rate=frame_rate, usable=usable,
                         deglobal=deglobal, target_fdr=target_fdr)
    stem = round_path.with_suffix("")
    outputs = {"trace_atlas": {}}
    if save:
        spatial = Path(f"{stem}_20x_spatialqc.png")
        if structural is None:
            structural = np.where(masks["labels"] > 0, masks["labels"], 0)
        _spatial_figure(spatial, np.asarray(structural), masks, manifest, effective_groups, analysed)
        response = Path(f"{stem}_20x_responseqc.png"); _response_figure(response, analysed, odor_ids, states, state_levels)
        snr = Path(f"{stem}_20x_snrqc.png"); _snr_figure(snr, analysed)
        outputs.update(spatial=str(spatial), response=str(response), snr=str(snr))
        for key, item in analysed.items():
            outputs["trace_atlas"][key] = _atlas(stem, key, item, on, off, frame_rate)
        pc1_path = Path(f"{stem}_20x_pc1.json")
        pc1_path.write_text(json.dumps({
            "description": "Odor-protected PC1 trial score; recorded, not subtracted"
                           if deglobal is None else f"Odor-protected PC1; deglobal={deglobal}",
            "trial_index": list(range(len(odor_ids))),
            "odor_id": [int(v) for v in odor_ids],
            "state": [state_levels[int(v)] for v in states],
            "populations": {
                key: [None if not np.isfinite(v) else round(float(v), 8)
                      for v in item["calls"]["pc1_component"]]
                for key, item in analysed.items()
            },
        }, indent=2))
        outputs["pc1"] = str(pc1_path)
        import h5py
        time_pc1_path = Path(f"{stem}_20x_pc1_timeseries.h5")
        with h5py.File(time_pc1_path, "w") as handle:
            handle.attrs["description"] = (
                "Fixed spatial PC1 timecourses from grouped detrended dF/F; "
                "recorded without subtraction from traces"
            )
            handle.create_dataset("time_s", data=(
                np.arange(raw.shape[2]) - int(np.median(on))
            ) / frame_rate)
            handle.create_dataset("trial_index", data=np.arange(len(odor_ids)))
            handle.create_dataset("odor_id", data=odor_ids)
            handle.create_dataset("state", data=states)
            for key, item in analysed.items():
                group = handle.create_group(key)
                pc = item["time_pc1"]
                group.create_dataset("timecourse", data=pc["timecourse"], compression="gzip")
                group.create_dataset("loadings", data=pc["loadings"])
                group.create_dataset("area_px", data=item["population"].area_px)
                group.create_dataset("unit_id", data=np.asarray(
                    item["population"].unit_ids, dtype=h5py.string_dtype("utf-8")
                ))
                group.attrs["explained_variance_fraction"] = pc["explained_variance_fraction"]
                group.attrs["method"] = pc["method"]
        outputs["pc1_timecourse"] = str(time_pc1_path)
    outputs["n_units"] = {key: len(pop.unit_ids) for key, pop in populations.items()}
    outputs["analysis_order"] = "raw pixel-weighted aggregation -> detrend -> dF/F and trial z/calls"
    return outputs
