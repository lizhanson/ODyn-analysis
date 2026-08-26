"""Final 10x traces and QC after reviewed fragment joining."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from ..seg_20x.qc import UnitPopulation, _analyse


def aggregate_joined_raw(raw, areas, groups):
    """Return joined glomeruli plus unassigned singletons; never components."""
    raw = np.asarray(raw, np.float32)
    areas = np.asarray(areas, np.float32)
    if raw.ndim != 3 or raw.shape[0] != len(areas):
        raise ValueError("raw component traces and ROI areas do not align")
    groups = {int(k): int(v) for k, v in groups.items()}
    valid_ids = set(range(1, raw.shape[0] + 1))
    unknown = set(groups) - valid_ids
    if unknown:
        raise ValueError(f"Group assignments name unknown ROI ids: {sorted(unknown)}")
    by_group = {}
    for roi_id, group_id in groups.items():
        by_group.setdefault(group_id, []).append(roi_id)
    # A one-member join is not a join. Preserve it as an ordinary singleton.
    joined_ids = {roi_id for members in by_group.values() if len(members) > 1
                  for roi_id in members}
    units = [("join", gid, sorted(members)) for gid, members in sorted(by_group.items())
             if len(members) > 1]
    units.extend(("roi", roi_id, [roi_id]) for roi_id in sorted(valid_ids - joined_ids))
    units.sort(key=lambda item: min(item[2]))

    traces = np.empty((len(units), *raw.shape[1:]), np.float32)
    unit_areas, unit_ids, members = [], [], []
    for position, (kind, ident, roi_ids) in enumerate(units):
        indices = np.asarray(roi_ids, int) - 1
        weights = areas[indices]
        traces[position] = np.average(raw[indices], axis=0, weights=weights)
        unit_areas.append(float(weights.sum()))
        unit_ids.append(f"j{ident}" if kind == "join" else f"r{ident}")
        members.append(roi_ids)
    return UnitPopulation(
        "joined glomeruli + singleton ROIs", traces, np.asarray(unit_areas),
        unit_ids, members, [("glomerulus",)] * len(units))


def _spatial_figure(path, reference, labels, population, snr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    values = np.full(labels.shape, np.nan)
    for value, members in zip(snr, population.members):
        values[np.isin(labels, members)] = value
    finite = np.asarray(snr)
    finite = finite[np.isfinite(finite) & (finite > 0)]
    lo, hi = np.percentile(finite, [5, 95]) if len(finite) else (1, 10)
    lo, hi = max(float(lo), 1e-3), max(float(hi), float(lo) * 1.01)
    base_lo, base_hi = np.nanpercentile(reference, [1, 99.5])
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    axes[0].imshow(reference, cmap="gray", vmin=base_lo, vmax=base_hi)
    image = axes[0].imshow(np.ma.masked_invalid(values), cmap="viridis",
                           norm=LogNorm(lo, hi), alpha=.85)
    axes[0].set(title="final analysis-unit baseline SNR", xticks=[], yticks=[])
    fig.colorbar(image, ax=axes[0], label="median trial baseline F / SD")
    axes[1].scatter(population.area_px, snr, s=16, alpha=.5)
    axes[1].set(xlabel="analysis-unit area (px)",
                ylabel="median trial baseline F / SD",
                title="final units only: joins + singletons")
    fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)
    return str(path)


def finalize_grouped_10x(round_path, groups_path, *, reference,
                         baseline_sd_mode="pre_block_pooled", save=True):
    """Create the sole final 10x unit product and its grouped-only QC."""
    import h5py
    from ..session.finalize import mask_hash
    from ..session.h5io import open_h5
    from ..session.store import _write_table
    from ..session.trace_analysis import aggregate_epoch_table, trial_epoch_table
    from ..session.trace_qc import baseline_qc_figure, continuous_response_figure
    from .grouping import load_groups

    round_path = Path(round_path)
    with open_h5(round_path) as handle:
        raw = handle["traces/roi"][:]
        areas = handle["rois/area_px"][:]
        labels = handle["masks/labels"][:]
        on = handle["trials/odor_on_frame"][:]
        off = handle["trials/odor_off_frame"][:]
        odor_ids = handle["trials/odor_id"][:]
        states = handle["trials/state"][:]
        trial_ids = handle["trials/trial_id"][:]
        state_levels = [v.decode() if isinstance(v, bytes) else str(v)
                        for v in handle["trials/state_levels"][:]]
        frame_rate = float(handle.attrs["frame_rate"])
    digest = mask_hash(labels)
    groups, grouping_metadata = load_groups(
        groups_path, expected_mask_hash=digest)
    population = aggregate_joined_raw(raw, areas, groups)
    analysed = _analyse(
        {"final": population}, on=on, off=off, odor_ids=odor_ids,
        states=states, state_levels=state_levels, frame_rate=frame_rate,
        baseline_sd_mode=baseline_sd_mode)["final"]
    scores, standard, pc1 = (analysed["scores"], analysed["standardized"],
                             analysed["pc1"])
    trial_table = trial_epoch_table(
        scores, unit_ids=population.unit_ids, odor_ids=odor_ids, states=states,
        state_levels=state_levels, trial_ids=trial_ids,
        unit_types=["joined" if len(v) > 1 else "singleton"
                    for v in population.members],
        group_ids=[u if u.startswith("j") else None for u in population.unit_ids])
    summary = aggregate_epoch_table(trial_table)
    stem = round_path.with_suffix("")
    grouped_h5 = Path(f"{stem}_10x_grouped.h5")
    outputs = {"grouped_h5": str(grouped_h5),
               "n_units": len(population.unit_ids),
               "n_joined_units": sum(len(v) > 1 for v in population.members),
               "n_singletons": sum(len(v) == 1 for v in population.members)}
    if not save:
        return outputs

    partial = grouped_h5.with_suffix(grouped_h5.suffix + ".partial")
    with h5py.File(partial, "w") as handle:
        handle.attrs["file_type"] = "odyn_10x_grouped_traces"
        handle.attrs["source_component_round"] = str(round_path)
        handle.attrs["mask_hash"] = digest
        handle.attrs["grouping_json"] = json.dumps(grouping_metadata)
        handle.attrs["baseline_sd_mode"] = baseline_sd_mode
        handle.create_dataset("trial_id", data=trial_ids)
        handle.create_dataset("odor_id", data=odor_ids)
        handle.create_dataset("state", data=states)
        handle.create_dataset("state_levels", data=np.asarray(
            state_levels, dtype=h5py.string_dtype("utf-8")))
        units = handle.create_group("units")
        units.create_dataset("unit_id", data=np.asarray(
            population.unit_ids, dtype=h5py.string_dtype("utf-8")))
        member_data = units.create_dataset(
            "member_roi_ids", (len(population.members),),
            dtype=h5py.vlen_dtype(np.dtype("int64")))
        for index, member_ids in enumerate(population.members):
            member_data[index] = np.asarray(member_ids, np.int64)
        units.create_dataset("area_px", data=population.area_px)
        units.create_dataset("raw", data=population.raw, compression="gzip")
        units.create_dataset("z", data=analysed["z"], compression="gzip")
        units.create_dataset("baseline_mean", data=standard.baseline_mean)
        units.create_dataset("baseline_sd_trial", data=standard.baseline_sd_trial)
        units.create_dataset("normalization_sd", data=standard.normalization_sd)
        responses = units.create_group("responses")
        for epoch in ("odor", "post_odor"):
            destination = responses.create_group(epoch)
            destination.create_dataset("mean_z", data=scores.mean[epoch])
            destination.create_dataset("peak_positive_z", data=scores.peak_positive[epoch])
            destination.create_dataset("peak_negative_z", data=scores.peak_negative[epoch])
        pc = units.create_group("pc1")
        pc.create_dataset("trial_score", data=pc1["trial_score"])
        pc.create_dataset("loadings", data=pc1["loadings"])
        pc.attrs["explained_variance_fraction"] = pc1["explained_variance_fraction"]
        _write_table(units.create_group("response_summary"), summary)
    partial.replace(grouped_h5)

    continuous = Path(f"{stem}_10x_continuousqc.png")
    baseline = Path(f"{stem}_10x_baselineqc.png")
    spatial = Path(f"{stem}_10x_spatialqc.png")
    continuous_response_figure(
        continuous, scores=scores, odor_ids=odor_ids, states=states,
        state_levels=state_levels, unit_label="final glomerular unit",
        normalization_label=("pre-anesthesia pooled baseline SD"
                             if baseline_sd_mode == "pre_block_pooled"
                             else "per-trial baseline SD"),
        pc1_scores=pc1["trial_score"],
        pc1_variance=pc1["explained_variance_fraction"])
    baseline_qc_figure(
        baseline, baseline_mean=standard.baseline_mean,
        baseline_sd=standard.baseline_sd_trial, states=states,
        state_levels=state_levels, unit_label="final glomerular unit")
    _spatial_figure(spatial, np.asarray(reference), labels,
                    population, analysed["snr"])
    outputs.update(continuous_qc=str(continuous), baseline_qc=str(baseline),
                   spatial_qc=str(spatial))
    report = Path(f"{stem}_10x_qc.json")
    report.write_text(json.dumps(outputs, indent=2) + "\n")
    outputs["json"] = str(report)
    return outputs


def cleanup_10x_caches(output_dir, scratch_root, group_id, *, qc_outputs):
    """Delete only caches after grouped HDF5 and every QC artifact exist."""
    required = [Path(qc_outputs[key]) for key in
                ("grouped_h5", "continuous_qc", "baseline_qc", "spatial_qc", "json")]
    missing = [str(path) for path in required
               if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"Refusing cache cleanup; missing final outputs: {missing}")
    targets = [Path(scratch_root) / "correlation_cache" / f"group{int(group_id)}",
               Path(output_dir) / "correlation_cache",
               Path(output_dir) / "zscore_cache"]
    removed = []
    for path in targets:
        if path.is_dir():
            shutil.rmtree(path); removed.append(str(path))
    return removed
