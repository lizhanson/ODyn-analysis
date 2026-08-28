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


def _group_membership_figure(path, reference, labels, population):
    """Joined units share a color; unjoined singleton components remain gray."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .gui import _PALETTE

    reference = np.asarray(reference, float)
    labels = np.asarray(labels)
    lo, hi = np.nanpercentile(reference, [1, 99.5])
    rgba = np.zeros((*labels.shape, 4), float)
    n_joined = 0
    n_singletons = 0
    for unit_index, members in enumerate(population.members):
        selected = np.isin(labels, members)
        if len(members) > 1:
            colour = _PALETTE[n_joined % len(_PALETTE)] / 255.0
            rgba[selected, :3] = colour
            rgba[selected, 3] = .72
            n_joined += 1
        else:
            rgba[selected, :3] = .65
            rgba[selected, 3] = .42
            n_singletons += 1

    fig, ax = plt.subplots(figsize=(11, 8), constrained_layout=True)
    ax.imshow(reference, cmap="gray", vmin=lo, vmax=hi)
    ax.imshow(rgba)
    ax.set(
        title=(f"reviewed 10x joins — {n_joined} joined units in distinct colors; "
               f"{n_singletons} singleton ROIs in gray"),
        xticks=[], yticks=[],
    )
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def temporal_unit_odor_heatmaps(z, *, odor_ids, states, state_levels,
                                odor_on_frames, odor_off_frames, frame_rate,
                                pre_s=4.0, post_s=4.0):
    """Block heatmaps with shared pre-block latency-to-peak row ordering."""
    z = np.asarray(z, float)
    odor_ids = np.asarray(odor_ids)
    states = np.asarray(states, int)
    on = np.asarray(odor_on_frames, int)
    off = np.asarray(odor_off_frames, int)
    pre_frames = int(round(float(pre_s) * frame_rate))
    odor_frames = int(np.max(off - on))
    post_frames = int(round(float(post_s) * frame_rate))
    length = pre_frames + odor_frames + post_frames
    aligned = np.full((z.shape[0], z.shape[1], length), np.nan, float)
    for trial, start in enumerate(on):
        left = start - pre_frames
        right = min(z.shape[2], left + length)
        source_left = max(0, left)
        destination_left = source_left - left
        aligned[:, trial, destination_left:destination_left + right - source_left] = (
            z[:, trial, source_left:right]
        )

    odors = np.unique(odor_ids)
    levels = list(state_levels)
    reference_code = levels.index("pre") if "pre" in levels else 0

    def block_means(code):
        return {
            int(odor): np.nanmean(
                aligned[:, (states == code) & (odor_ids == odor)], axis=1
            )
            for odor in odors
        }

    means = {code: block_means(code) for code in range(len(levels))}
    odor_window = slice(pre_frames, pre_frames + odor_frames)
    orders = {}
    for odor in odors:
        response = means[reference_code][int(odor)][:, odor_window]
        safe = np.where(np.isfinite(response), response, -np.inf)
        peak_latency = np.argmax(safe, axis=1)
        no_data = ~np.isfinite(response).any(axis=1)
        peak_latency[no_data] = odor_frames + 1
        peak_height = np.max(safe, axis=1)
        orders[int(odor)] = np.lexsort((-peak_height, peak_latency))

    heatmaps = {}
    for code, level in enumerate(levels):
        heatmaps[level] = np.concatenate([
            means[code][int(odor)][orders[int(odor)]] for odor in odors
        ], axis=0)
    centers = (np.arange(len(odors)) * z.shape[0] + (z.shape[0] - 1) / 2)
    boundaries = np.arange(1, len(odors)) * z.shape[0] - .5
    time_s = (np.arange(length) - pre_frames) / float(frame_rate)
    return {
        "heatmaps": heatmaps, "time_s": time_s, "odors": odors,
        "odor_centers": centers, "odor_boundaries": boundaries,
        "odor_offset_s": float(np.nanmedian((off - on) / frame_rate)),
        "orders": orders,
    }


def _joined_continuous_figure(path, *, analysed, odor_ids, states, state_levels,
                              odor_on_frames, odor_off_frames, frame_rate,
                              baseline_sd_mode):
    """Temporal atlases plus the original trial-scalar QC summaries."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    temporal = temporal_unit_odor_heatmaps(
        analysed["z"], odor_ids=odor_ids, states=states,
        state_levels=state_levels, odor_on_frames=odor_on_frames,
        odor_off_frames=odor_off_frames, frame_rate=frame_rate,
    )
    levels = [level for level in ("pre", "post") if level in state_levels]
    levels += [level for level in state_levels if level not in levels]
    levels = levels[:2]
    fig = plt.figure(figsize=(20, 24), constrained_layout=True)
    grid = fig.add_gridspec(
        5, 2, height_ratios=(1.35, 1., .65, .8, .32),
    )
    heat_axes = [fig.add_subplot(grid[0, column]) for column in range(2)]
    scalar_axes = [fig.add_subplot(grid[1, column]) for column in range(2)]
    histogram_axes = [fig.add_subplot(grid[2, column]) for column in range(2)]
    scatter_axes = [fig.add_subplot(grid[3, column]) for column in range(2)]
    pc_axis = fig.add_subplot(grid[4, :])
    norm = TwoSlopeNorm(vmin=-3, vcenter=0, vmax=6)
    image = None
    for column, level in enumerate(levels):
        ax = heat_axes[column]
        image = ax.imshow(
            temporal["heatmaps"][level], aspect="auto", cmap="RdBu_r", norm=norm,
            extent=(temporal["time_s"][0], temporal["time_s"][-1],
                    len(temporal["heatmaps"][level]) - .5, -.5),
            interpolation="nearest",
        )
        ax.axvline(0, color="black", lw=1)
        ax.axvline(temporal["odor_offset_s"], color="black", lw=1, ls="--")
        top = -.5
        ax.text(temporal["time_s"][0] / 2, top, "baseline", ha="center",
                va="bottom", fontsize=8, clip_on=False)
        ax.text(temporal["odor_offset_s"] / 2, top, "odor", ha="center",
                va="bottom", fontsize=8, clip_on=False)
        ax.text((temporal["odor_offset_s"] + temporal["time_s"][-1]) / 2,
                top, "post odor", ha="center", va="bottom", fontsize=8,
                clip_on=False)
        for boundary in temporal["odor_boundaries"]:
            ax.axhline(boundary, color="black", lw=.6)
        ax.set(
            title=(f"{level}: continuous baseline → odor → post-odor z; "
                   "shared pre latency-to-peak order"),
            xlabel="time from odor onset (s)", ylabel="odor id",
            yticks=temporal["odor_centers"],
            yticklabels=[str(int(value)) for value in temporal["odors"]],
        )
    for column in range(len(levels), 2):
        heat_axes[column].axis("off"); scatter_axes[column].axis("off")
    if image is not None:
        fig.colorbar(image, ax=heat_axes[:len(levels)], label="mean temporal z", shrink=.8)

    scores = analysed["scores"]
    odors = np.unique(odor_ids)
    cmap = plt.get_cmap("tab20", len(odors))
    preferred_levels = [value for value in ("pre", "post")
                        if value in state_levels]
    preferred_levels += [value for value in state_levels
                         if value not in preferred_levels]
    state_rank = np.array([
        preferred_levels.index(state_levels[int(code)]) for code in states
    ])
    trial_order = np.lexsort((np.arange(len(odor_ids)), state_rank, odor_ids))
    reference_trials = np.ones(len(states), dtype=bool)
    preference_source = "all trials"
    if "pre" in state_levels:
        reference_trials = states == state_levels.index("pre")
        preference_source = "pre-anesthesia trials"
    from ..session.trace_qc import _preferred_odor_order
    unit_order, _, _ = _preferred_odor_order(
        scores.mean["odor"], odor_ids, reference_trials,
    )
    sorted_odors = odor_ids[trial_order]
    sorted_ranks = state_rank[trial_order]
    for axis, epoch in zip(scalar_axes, ("odor", "post_odor")):
        values = np.asarray(scores.mean[epoch], float)
        scalar_image = axis.imshow(
            values[np.ix_(unit_order, trial_order)], aspect="auto",
            cmap="RdBu_r", norm=norm, interpolation="nearest",
        )
        for edge in np.flatnonzero(np.diff(sorted_odors)) + 1:
            axis.axvline(edge - .5, color="black", lw=1)
        for edge in np.flatnonzero(
            (np.diff(sorted_ranks) != 0) & (np.diff(sorted_odors) == 0)
        ) + 1:
            axis.axvline(edge - .5, color=".35", lw=.7, ls=":")
        axis.set_xticks(
            [np.mean(np.flatnonzero(sorted_odors == odor)) for odor in odors],
            [str(int(odor)) for odor in odors],
        )
        axis.set(
            xlabel="odor id (individual trials; pre then post)",
            ylabel="final glomerular unit",
            title=(f"{epoch}: trial mean z; rows by odor preference "
                   f"({preference_source})"),
        )
        fig.colorbar(scalar_image, ax=axis, label="mean z", shrink=.8)

    if "pre" in state_levels and "post" in state_levels:
        pre = states == state_levels.index("pre")
        post = states == state_levels.index("post")
        for axis, epoch in zip(histogram_axes, ("odor", "post_odor")):
            values = np.asarray(scores.mean[epoch], float)
            before = np.concatenate([
                np.nanmean(values[:, pre & (odor_ids == odor)], axis=1)
                for odor in odors
            ])
            after = np.concatenate([
                np.nanmean(values[:, post & (odor_ids == odor)], axis=1)
                for odor in odors
            ])
            before = before[np.isfinite(before)]
            after = after[np.isfinite(after)]
            combined = np.concatenate((before, after))
            if len(combined):
                lo, hi = np.min(combined), np.max(combined)
                bins = np.linspace(lo, hi, 31) if hi > lo else 30
                axis.hist(before, bins=bins, density=True, histtype="step",
                          lw=2, color="steelblue", label="pre anesthesia")
                axis.hist(after, bins=bins, density=True, histtype="step",
                          lw=2, color="indianred", label="post anesthesia")
            axis.axvline(0, color="black", lw=.7)
            axis.set(
                xlabel="final-unit × odor mean z", ylabel="density",
                title=f"{epoch}: pre/post response distributions",
            )
            axis.legend(fontsize=8)
    else:
        for axis in histogram_axes:
            axis.axis("off")

    all_values = []
    for column, level in enumerate(levels):
        code = state_levels.index(level)
        ax = scatter_axes[column]
        for odor_index, odor in enumerate(odors):
            selected = (states == code) & (odor_ids == odor)
            x = np.nanmean(scores.mean["odor"][:, selected], axis=1)
            y = np.nanmean(scores.mean["post_odor"][:, selected], axis=1)
            finite = np.isfinite(x) & np.isfinite(y)
            ax.scatter(x[finite], y[finite], s=15, alpha=.55,
                       color=cmap(odor_index), edgecolors="none",
                       label=str(int(odor)))
            all_values.extend((x[finite], y[finite]))
        ax.axhline(0, color="black", lw=.7); ax.axvline(0, color="black", lw=.7)
        ax.set(xlabel="odor-epoch mean z", ylabel="post-odor mean z",
               title=f"{level}: every final unit × odor")
        ax.legend(title="odor", fontsize=6, title_fontsize=7, ncol=2,
                  loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
    finite_values = np.concatenate(all_values) if all_values else np.array([])
    finite_values = finite_values[np.isfinite(finite_values)]
    if len(finite_values):
        lo, hi = np.percentile(finite_values, [.5, 99.5])
        span = max(float(hi - lo), 1.)
        limits = (float(lo - .05 * span), float(hi + .05 * span))
        for ax in scatter_axes[:len(levels)]:
            ax.plot(limits, limits, color=".35", lw=.8, ls=":")
            ax.set(xlim=limits, ylim=limits, aspect="equal", adjustable="box")

    pc1 = analysed["pc1"]
    for code, level in enumerate(state_levels):
        selected = states == code
        pc_axis.scatter(np.flatnonzero(selected), pc1["trial_score"][selected],
                        s=14, alpha=.75, label=level)
    pc_axis.axhline(0, color="black", lw=.7)
    pc_axis.set(
        xlabel="trial index", ylabel="PC1 scalar",
        title=("odor-protected trial PC1; variance "
               f"{pc1['explained_variance_fraction']:.1%}"),
    )
    pc_axis.legend(fontsize=8)
    normalization = ("pre-anesthesia pooled baseline SD"
                     if baseline_sd_mode == "pre_block_pooled"
                     else "per-trial baseline SD")
    fig.suptitle("final 10x continuous QC — " + normalization)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def finalize_grouped_10x(round_path, groups_path, *, reference,
                         baseline_sd_mode="pre_block_pooled", save=True):
    """Create the sole final 10x unit product and its grouped-only QC."""
    import h5py
    from ..session.finalize import mask_hash
    from ..session.h5io import open_h5
    from ..session.store import _write_table
    from ..session.trace_analysis import aggregate_epoch_table, trial_epoch_table
    from ..session.trace_qc import baseline_qc_figure
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
    groups_figure = Path(f"{stem}_10x_groups.png")
    _joined_continuous_figure(
        continuous, analysed=analysed, odor_ids=odor_ids, states=states,
        state_levels=state_levels, odor_on_frames=on, odor_off_frames=off,
        frame_rate=frame_rate, baseline_sd_mode=baseline_sd_mode)
    baseline_qc_figure(
        baseline, baseline_mean=standard.baseline_mean,
        baseline_sd=standard.baseline_sd_trial, states=states,
        state_levels=state_levels, unit_label="final glomerular unit")
    _spatial_figure(spatial, np.asarray(reference), labels,
                    population, analysed["snr"])
    _group_membership_figure(groups_figure, np.asarray(reference), labels, population)
    outputs.update(continuous_qc=str(continuous), baseline_qc=str(baseline),
                   spatial_qc=str(spatial), groups_figure=str(groups_figure))
    report = Path(f"{stem}_10x_qc.json")
    report.write_text(json.dumps(outputs, indent=2) + "\n")
    outputs["json"] = str(report)
    return outputs


def cleanup_10x_caches(output_dir, scratch_root, group_id, *, qc_outputs):
    """Delete only caches after grouped HDF5 and every QC artifact exist."""
    required = [Path(qc_outputs[key]) for key in (
        "grouped_h5", "continuous_qc", "baseline_qc", "spatial_qc",
        "groups_figure", "json",
    )]
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
