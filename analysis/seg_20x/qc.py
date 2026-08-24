"""QC for 20x soma/process rounds, with grouped ROIs as analysis units."""

from __future__ import annotations

import json
import time
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


def aggregate_raw_units(raw, areas, manifest, groups=None, *, only=None) -> dict[str, UnitPopulation]:
    """Pixel-weight ROI means into whole, soma-only, and process-only units.

    Grouped ROIs share a unit. Every ungrouped ROI becomes a singleton. Weighting
    the raw ROI means by their pixel counts is exactly the mean over the union of
    their (disjoint) mask pixels. No detrending or normalization happens here.
    """
    raw = np.asarray(raw, np.float32)
    areas = np.asarray(areas, np.float32)
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
        selections = []
        for unit, unit_rows in buckets.items():
            chosen = unit_rows if kind is None else [r for r in unit_rows if r["roi_type"] == kind]
            if not chosen:
                continue
            selections.append((unit, chosen))
        traces = np.empty((len(selections), *raw.shape[1:]), np.float32)
        pixel_counts, ids, members, member_types = [], [], [], []
        for position, (unit, chosen) in enumerate(selections):
            indices = np.array([int(r["roi_id"]) - 1 for r in chosen])
            weights = areas[indices]
            traces[position] = np.asarray(
                np.average(raw[indices], axis=0, weights=weights), np.float32
            )
            pixel_counts.append(weights.sum())
            ids.append(f"g{unit[1]}" if unit[0] == "group" else f"{chosen[0]['roi_type'][0]}{unit[1]}")
            members.append([int(r["roi_id"]) for r in chosen])
            member_types.append(tuple(sorted({str(r["roi_type"]) for r in chosen})))
        return UnitPopulation(name, traces,
                              np.asarray(pixel_counts), ids, members, member_types)

    builders = {
        "groups": ("whole groups + singleton ROIs", None),
        "somas": ("soma component", "soma"),
        "processes": ("group minus soma", "process"),
    }
    selected = builders if only is None else {key: builders[key] for key in only}
    return {key: build(*spec) for key, spec in selected.items()}


def _analyse(populations, *, on, off, odor_ids, states, state_levels, frame_rate,
             report=lambda message: None):
    from ..session.detrend import detrend_traces
    from ..session.pc1 import trial_pc1
    from ..session.trace_analysis import epoch_scores, standardize_traces

    analysed = {}
    for key, pop in populations.items():
        report(f"{key}: detrending {len(pop.unit_ids)} analysis units")
        corrected, fit = detrend_traces(pop.raw, odor_on_frames=on,
                                        odor_off_frames=off, frame_rate=frame_rate)
        if not fit.get("ok"):
            raise RuntimeError(
                f"{key} detrending failed; refusing to continue with a "
                f"normalization fallback. Details: {fit}"
            )
        report(f"{key}: per-trial baseline centering and SD z-scoring")
        standardized = standardize_traces(
            corrected, odor_on_frames=on, states=states,
            n_state_levels=len(state_levels), frame_rate=frame_rate,
        )
        standardized.state_levels = list(state_levels)
        snr = np.nanmedian(
            standardized.baseline_mean /
            np.maximum(standardized.baseline_sd_trial, 1e-9), axis=1,
        )
        del corrected
        scores = epoch_scores(
            standardized.z, odor_on_frames=on, odor_off_frames=off,
            frame_rate=frame_rate,
        )
        report(f"{key}: fitting odor-protected trial PC1")
        pc1 = trial_pc1(scores.mean["odor"], odor_ids)
        analysed[key] = {
            "population": pop,
            "z": standardized.z, "standardized": standardized,
            "scores": scores, "fit": fit,
            "pc1": pc1,
            "snr": snr,
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
        ax.set_title(pop.name + ": median trial baseline SNR")
        ax.set(xticks=[], yticks=[])
    fig.colorbar(image, ax=axes[:3], label="median trial baseline F / SD", shrink=.75)
    for key, colour in zip(("groups", "somas", "processes"),
                           ("black", "deepskyblue", "magenta")):
        axes[3].scatter(analysed[key]["population"].area_px,
                        analysed[key]["snr"], s=14, alpha=.45, color=colour,
                        label=key)
    axes[3].set(xlabel="component area (px)", ylabel="median trial baseline F / SD",
                title="size versus SNR")
    axes[3].legend(fontsize=8)
    fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)


def _response_figure(path, analysed, odor_ids, states, state_levels,
                     normalization_label="per-trial baseline SD"):
    from ..session.trace_qc import continuous_response_figure
    path = Path(path)
    outputs = {}
    for key, item in analysed.items():
        out = path.with_name(path.stem + f"_{key}" + path.suffix)
        outputs[key] = continuous_response_figure(
            out, scores=item["scores"], odor_ids=odor_ids, states=states,
            state_levels=state_levels, unit_label=item["population"].name,
            normalization_label=normalization_label,
            pc1_scores=item["pc1"]["trial_score"],
            pc1_variance=item["pc1"]["explained_variance_fraction"],
        )
    return outputs


def _baseline_figures(path, analysed, states, state_levels):
    from ..session.trace_qc import baseline_qc_figure
    path = Path(path)
    outputs = {}
    for key, item in analysed.items():
        out = path.with_name(path.stem + f"_{key}" + path.suffix)
        standardized = item["standardized"]
        outputs[key] = baseline_qc_figure(
            out, baseline_mean=standardized.baseline_mean,
            baseline_sd=standardized.baseline_sd_trial,
            states=states, state_levels=state_levels,
            unit_label=item["population"].name,
        )
    return outputs


def _snr_figure(path, analysed):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    for key, colour in zip(
        ("groups", "somas", "processes"),
        ("black", "deepskyblue", "magenta"),
    ):
        values = np.asarray(analysed[key]["snr"], float)
        values = values[np.isfinite(values)]
        ax.hist(
            values, bins=35, histtype="step", lw=2, color=colour,
            label=f"{analysed[key]['population'].name} (n={len(values)})",
        )
    ax.set(xlabel="median trial baseline SNR (mean F / SD)",
           ylabel="analysis units", title="20x baseline SNR")
    ax.legend(fontsize=8); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def qc_20x(round_path, *, groups=None, structural=None,
           save=True, progress=True) -> dict:
    """Grouped 20x continuous QC using the sole canonical z-score path."""
    from ..session.h5io import open_h5
    round_path = Path(round_path)
    started = time.perf_counter()

    def report(message):
        if progress:
            elapsed = time.perf_counter() - started
            print(f"[20x QC {elapsed:7.1f}s] {message}", flush=True)

    report(f"loading extracted round {round_path.name}")
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
    stem = round_path.with_suffix("")
    outputs = {}
    analysed = {}
    grouped_h5 = Path(f"{stem}_20x_grouped.h5")
    if save:
        import h5py
        with h5py.File(grouped_h5, "w") as grouped_file:
            grouped_file.attrs["source_round"] = str(round_path)
            grouped_file.attrs["description"] = (
                "Grouped 20x raw and canonical traces, continuous responses, "
                "baseline diagnostics, membership, and one PC1 scalar per trial"
            )
            grouped_file.create_dataset("trial_id", data=np.arange(len(odor_ids)))
            grouped_file.create_dataset("odor_id", data=odor_ids)
            grouped_file.create_dataset("state", data=states)
            grouped_file.create_dataset(
                "state_levels", data=np.asarray(state_levels, dtype=h5py.string_dtype("utf-8"))
            )

    for key in ("groups", "somas", "processes"):
            report(f"{key}: pixel-weighting raw fluorescence")
            populations = aggregate_raw_units(
                raw, areas, manifest, effective_groups, only=(key,),
            )
            item = _analyse(
                populations, on=on, off=off, odor_ids=odor_ids, states=states,
                state_levels=state_levels, frame_rate=frame_rate,
                report=report,
            )[key]

            if save:
                from ..session.trace_analysis import (
                    aggregate_epoch_table, trial_epoch_table,
                )
                pop = item["population"]
                trial_table = trial_epoch_table(
                    item["scores"], unit_ids=pop.unit_ids, odor_ids=odor_ids,
                    states=states, state_levels=state_levels,
                    unit_types=["+".join(v) for v in pop.member_types],
                    group_ids=[u if u.startswith("g") else None for u in pop.unit_ids],
                )
                summary_table = aggregate_epoch_table(trial_table)
                import h5py
                from ..session.store import _write_table
                with h5py.File(grouped_h5, "a") as grouped_file:
                    destination = grouped_file.create_group(key)
                    destination.create_dataset("unit_id", data=np.asarray(
                        pop.unit_ids, dtype=h5py.string_dtype("utf-8")
                    ))
                    members = destination.create_dataset(
                        "member_roi_ids", (len(pop.members),),
                        dtype=h5py.vlen_dtype(np.dtype("int64")),
                    )
                    for index, member_ids in enumerate(pop.members):
                        members[index] = np.asarray(member_ids, np.int64)
                    destination.create_dataset("area_px", data=pop.area_px)
                    destination.create_dataset("raw", data=pop.raw, compression="gzip")
                    destination.create_dataset("z", data=item["z"], compression="gzip")
                    standard = item["standardized"]
                    destination.create_dataset("baseline_mean", data=standard.baseline_mean)
                    destination.create_dataset("baseline_sd_trial", data=standard.baseline_sd_trial)
                    destination.create_dataset("baseline_sd_block", data=standard.baseline_sd_block)
                    responses = destination.create_group("responses")
                    for name in ("odor", "post_odor"):
                        epoch = responses.create_group(name)
                        epoch.create_dataset("mean_z", data=item["scores"].mean[name])
                        epoch.create_dataset("peak_positive_z", data=item["scores"].peak_positive[name])
                        epoch.create_dataset("peak_negative_z", data=item["scores"].peak_negative[name])
                    pc = destination.create_group("pc1")
                    pc.create_dataset("trial_score", data=item["pc1"]["trial_score"])
                    pc.create_dataset("loadings", data=item["pc1"]["loadings"])
                    pc.attrs["explained_variance_fraction"] = item["pc1"]["explained_variance_fraction"]
                    pc.attrs["method"] = item["pc1"]["method"]
                    _write_table(destination.create_group("response_summary"), summary_table)
                outputs.setdefault("response_tables", {})[key] = (
                    f"{grouped_h5}:/{key}/response_summary"
                )

            # Retain only compact summaries needed by the combined figures.
            del item["z"]
            item["standardized"].z = np.empty((0, 0, 0), np.float32)
            item["population"].raw = np.empty((0, 0, 0), np.float32)
            analysed[key] = item
            del populations

    if save:
        report("rendering spatial quality map")
        spatial = Path(f"{stem}_20x_spatialqc.png")
        if structural is None:
            structural = np.where(masks["labels"] > 0, masks["labels"], 0)
        _spatial_figure(spatial, np.asarray(structural), masks, manifest, effective_groups, analysed)
        report("rendering continuous odor and post-odor response distributions")
        response = _response_figure(
            Path(f"{stem}_20x_continuousqc.png"), analysed,
            odor_ids, states, state_levels,
        )
        report("rendering trial F0 and baseline SD QC")
        baseline = _baseline_figures(
            Path(f"{stem}_20x_baselineqc.png"), analysed, states, state_levels,
        )
        report("rendering SNR distributions")
        snr = Path(f"{stem}_20x_snrqc.png"); _snr_figure(snr, analysed)
        outputs.update(spatial=str(spatial), response=response,
                       baseline=baseline, snr=str(snr))
        outputs["pc1_trial_variance"] = {
            key: float(item["pc1"]["explained_variance_fraction"])
            for key, item in analysed.items()
        }
        outputs["grouped_h5"] = str(grouped_h5)
    outputs["n_units"] = {
        key: len(item["population"].unit_ids) for key, item in analysed.items()
    }
    outputs["analysis_order"] = (
        "raw pixel-weighted aggregation -> detrend -> canonical z -> "
        "continuous odor/post-odor summaries"
    )
    outputs["elapsed_s"] = round(time.perf_counter() - started, 3)
    if save:
        report_path = Path(f"{stem}_20x_qc.json")
        report_path.write_text(json.dumps(outputs, indent=2))
        outputs["json"] = str(report_path)
    report(f"complete in {outputs['elapsed_s']:.1f} s")
    return outputs
