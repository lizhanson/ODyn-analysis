"""Continuous-response QC for canonical trace z scores."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .trace_analysis import aggregate_epoch_table, trial_epoch_table


def _decode(values):
    return [v.decode() if isinstance(v, bytes) else str(v) for v in values]


def _preferred_odor_order(odor_response, odor_ids, reference_trials=None):
    """Group units by preferred odor, strongest preference first per group."""
    values = np.asarray(odor_response, float)
    odor_ids = np.asarray(odor_ids)
    if reference_trials is None:
        reference_trials = np.ones(len(odor_ids), dtype=bool)
    reference_trials = np.asarray(reference_trials, bool)
    keys = np.unique(odor_ids)
    tuning_columns = []
    for odor in keys:
        samples = values[:, reference_trials & (odor_ids == odor)]
        count = np.sum(np.isfinite(samples), axis=1)
        total = np.nansum(samples, axis=1)
        tuning_columns.append(
            np.divide(total, count, out=np.full(len(values), np.nan), where=count > 0)
        )
    tuning = np.column_stack(tuning_columns)
    finite = np.isfinite(tuning)
    safe = np.where(finite, tuning, -np.inf)
    preference = np.argmax(safe, axis=1)
    strength = safe[np.arange(len(values)), preference]
    missing = ~finite.any(axis=1)
    preference[missing] = len(keys)
    strength[missing] = -np.inf
    order = np.lexsort((-strength, preference))
    return order, preference[order], keys


def continuous_response_figure(path, *, scores, odor_ids, states, state_levels,
                               unit_label="ROI",
                               normalization_label="per-trial baseline SD",
                               pc1_scores=None, pc1_variance=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    odor_ids = np.asarray(odor_ids)
    states = np.asarray(states, int)
    keys = np.unique(odor_ids)
    preferred = [x for x in ("pre", "post") if x in state_levels]
    levels = preferred + [x for x in state_levels if x not in preferred]
    rank = np.array([levels.index(state_levels[int(code)]) for code in states])
    columns = np.lexsort((np.arange(len(odor_ids)), rank, odor_ids))
    reference = np.ones(len(states), dtype=bool)
    preference_source = "all trials"
    if "pre" in state_levels:
        reference = states == state_levels.index("pre")
        preference_source = "pre-anesthesia trials"
    order, _, _ = _preferred_odor_order(
        scores.mean["odor"], odor_ids, reference,
    )
    fig, axes = plt.subplots(
        3, 4, figsize=(24, 13), constrained_layout=True,
        gridspec_kw={"height_ratios": [1, 1, .38]},
    )
    norm = TwoSlopeNorm(vmin=-3, vcenter=0, vmax=6)

    for row, epoch in enumerate(("odor", "post_odor")):
        values = np.asarray(scores.mean[epoch], float)
        image = axes[row, 0].imshow(
            values[np.ix_(order, columns)], aspect="auto", cmap="RdBu_r",
            norm=norm, interpolation="nearest",
        )
        axes[row, 0].set(
            title=(f"{epoch}: trial mean z; rows by odor preference "
                   f"({preference_source})"),
            ylabel=unit_label,
        )
        sorted_odor, sorted_rank = odor_ids[columns], rank[columns]
        for edge in np.flatnonzero(np.diff(sorted_odor)) + 1:
            axes[row, 0].axvline(edge - .5, color="black", lw=1)
        for edge in np.flatnonzero(
            (np.diff(sorted_rank) != 0) & (np.diff(sorted_odor) == 0)
        ) + 1:
            axes[row, 0].axvline(edge - .5, color=".35", lw=.7, ls=":")
        axes[row, 0].set_xticks(
            [np.mean(np.flatnonzero(sorted_odor == key)) for key in keys],
            [str(int(key)) for key in keys],
        )
        fig.colorbar(image, ax=axes[row, 0], label="mean z")

        x = np.arange(len(keys))
        if "pre" in state_levels and "post" in state_levels:
            pre = states == state_levels.index("pre")
            post = states == state_levels.index("post")
            before_by_odor, after_by_odor = [], []
            changes = []
            for odor in keys:
                before = np.nanmean(values[:, pre & (odor_ids == odor)], axis=1)
                after = np.nanmean(values[:, post & (odor_ids == odor)], axis=1)
                before_by_odor.append(before)
                after_by_odor.append(after)
                changes.append(after - before)

            cmap = plt.get_cmap("tab20", len(keys))
            for i, (odor, before, after) in enumerate(
                zip(keys, before_by_odor, after_by_odor)
            ):
                finite = np.isfinite(before) & np.isfinite(after)
                axes[row, 1].scatter(
                    before[finite], after[finite], s=13, alpha=.55,
                    color=cmap(i), edgecolors="none", label=str(int(odor)),
                )
            paired = np.concatenate([
                np.concatenate(before_by_odor), np.concatenate(after_by_odor)
            ])
            paired = paired[np.isfinite(paired)]
            if len(paired):
                lo, hi = np.min(paired), np.max(paired)
                span = max(float(hi - lo), 1.)
                lo, hi = float(lo - .05 * span), float(hi + .05 * span)
                axes[row, 1].plot([lo, hi], [lo, hi], color="black", lw=.8, ls="--")
                axes[row, 1].set(xlim=(lo, hi), ylim=(lo, hi))
            axes[row, 1].set_aspect("equal", adjustable="box")
            axes[row, 1].set(
                xlabel="pre-anesthesia mean z", ylabel="post-anesthesia mean z",
                title=f"{epoch}: paired unit–odor means",
            )
            axes[row, 1].legend(
                title="odor", fontsize=6, title_fontsize=7, ncol=2,
                loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0,
            )

            axes[row, 2].boxplot(changes, positions=x, showfliers=False)
            axes[row, 2].axhline(0, color="black", lw=.7)
            axes[row, 2].set(xticks=x, xticklabels=[str(int(k)) for k in keys],
                             xlabel="odor id", ylabel="post − pre mean z",
                             title=f"{epoch}: paired unit changes")

            before_all = np.concatenate(before_by_odor)
            after_all = np.concatenate(after_by_odor)
            before_all = before_all[np.isfinite(before_all)]
            after_all = after_all[np.isfinite(after_all)]
            combined = np.concatenate([before_all, after_all])
            if len(combined):
                hist_lo, hist_hi = np.min(combined), np.max(combined)
                bins = np.linspace(hist_lo, hist_hi, 31)
                axes[row, 3].hist(before_all, bins=bins, density=True,
                                  histtype="step", lw=2, color="steelblue",
                                  label="pre anesthesia")
                axes[row, 3].hist(after_all, bins=bins, density=True,
                                  histtype="step", lw=2, color="indianred",
                                  label="post anesthesia")
            axes[row, 3].axvline(0, color="black", lw=.7)
            axes[row, 3].set(
                xlabel="unit–odor mean z", ylabel="density",
                title=f"{epoch}: response distributions",
            )
            axes[row, 3].legend(fontsize=8)
        else:
            for column in (1, 2, 3):
                axes[row, column].axis("off")

    for ax in axes[2]:
        ax.remove()
    pcax = fig.add_subplot(axes[2, 0].get_gridspec()[2, :])
    if pc1_scores is None:
        pcax.axis("off")
    else:
        for index, level in enumerate(state_levels):
            selected = states == index
            pcax.scatter(np.flatnonzero(selected), np.asarray(pc1_scores)[selected],
                         s=14, alpha=.75, label=level)
        pcax.axhline(0, color="black", lw=.7)
        variance = "" if pc1_variance is None else f"; variance {pc1_variance:.1%}"
        pcax.set(xlabel="trial index", ylabel="PC1 scalar",
                 title="odor-protected trial PC1 (recorded, not subtracted)" + variance)
        pcax.legend(fontsize=8)

    fig.suptitle(
        "continuous response QC — " + normalization_label
        + "; no responder thresholds or PC1 subtraction"
    )
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def baseline_qc_figure(path, *, baseline_mean, baseline_sd, states, state_levels,
                       unit_label="ROI"):
    """Show every unit-trial F0 and SD plus their trial population medians."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f0 = np.asarray(baseline_mean, float)
    sd = np.asarray(baseline_sd, float)
    states = np.asarray(states, int)
    fig, axes = plt.subplots(2, 2, figsize=(18, 9), constrained_layout=True)
    for row, (values, label) in enumerate(((f0, "baseline mean / F0"),
                                            (sd, "baseline SD"))):
        lo, hi = np.nanpercentile(values, [1, 99])
        image = axes[row, 0].imshow(values, aspect="auto", cmap="viridis",
                                    vmin=lo, vmax=hi, interpolation="nearest")
        axes[row, 0].set(xlabel="trial index", ylabel=unit_label,
                         title=f"every unit × trial {label}")
        fig.colorbar(image, ax=axes[row, 0], label=label)
        trial_median = np.nanmedian(values, axis=0)
        for code, level in enumerate(state_levels):
            selected = states == code
            axes[row, 1].scatter(np.flatnonzero(selected), trial_median[selected],
                                 s=18, label=level)
        axes[row, 1].set(xlabel="trial index", ylabel=f"median {label}",
                         title=f"population median {label} by trial")
        axes[row, 1].legend(fontsize=8)
    fig.suptitle("baseline QC — each trial uses its own pre-odor mean and SD")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def trace_qc(round_path, *, save=True):
    """QC and continuous trial/aggregate tables from one canonical round."""
    from .h5io import open_h5
    from .trace_analysis import EpochScores

    round_path = Path(round_path)
    with open_h5(round_path) as handle:
        required = ("traces/roi_z", "responses/mean_z", "responses/epoch_levels")
        missing = [name for name in required if name not in handle]
        if missing:
            raise ValueError(
                f"{round_path.name} predates canonical trace analysis; missing {missing}. "
                "Re-extract it."
            )
        epoch_levels = _decode(handle["responses/epoch_levels"][:])
        means = handle["responses/mean_z"][:]
        high = handle["responses/peak_positive_z"][:]
        low = handle["responses/peak_negative_z"][:]
        scores = EpochScores(
            mean={name: means[:, :, i] for i, name in enumerate(epoch_levels)},
            peak_positive={name: high[:, :, i] for i, name in enumerate(epoch_levels)},
            peak_negative={name: low[:, :, i] for i, name in enumerate(epoch_levels)},
            post_odor_frames=int(handle["responses"].attrs["post_odor_frames"]),
        )
        odor_ids = handle["trials/odor_id"][:]
        states = handle["trials/state"][:]
        state_levels = _decode(handle["trials/state_levels"][:])
        trial_ids = handle["trials/trial_id"][:]
        unit_ids = handle["rois/roi_id"][:]
        session_sd = handle["traces/baseline_sd_session"][:]
        block_sd = handle["traces/baseline_sd_block"][:]
        trial_sd = handle["traces/baseline_sd_trial"][:]
        baseline_mean = handle["traces/baseline_mean"][:]
        pc1_scores = handle["responses/pc1_trial_score"][:]
        pc1_variance = float(handle["responses/pc1_trial_score"].attrs[
            "explained_variance_fraction"
        ])

    trials = trial_epoch_table(
        scores, unit_ids=unit_ids, odor_ids=odor_ids, states=states,
        state_levels=state_levels, trial_ids=trial_ids,
    )
    aggregate = aggregate_epoch_table(trials)
    stem = round_path.with_suffix("")
    report = {
        "file": round_path.name,
        "method": "detrend -> per-trial baseline mean and SD z-score",
        "n_units": len(unit_ids), "n_trials": len(trial_ids),
        "epochs": epoch_levels,
        "median_baseline_sd_session": float(np.nanmedian(session_sd)),
        "median_baseline_sd_trial": float(np.nanmedian(trial_sd)),
        "median_baseline_sd_by_block": {
            level: float(np.nanmedian(block_sd[:, i]))
            for i, level in enumerate(state_levels)
        },
    }
    if save:
        figure = Path(f"{stem}_continuousqc.png")
        continuous_response_figure(
            figure, scores=scores, odor_ids=odor_ids, states=states,
            state_levels=state_levels, pc1_scores=pc1_scores,
            pc1_variance=pc1_variance,
        )
        baseline_figure = Path(f"{stem}_baselineqc.png")
        baseline_qc_figure(
            baseline_figure, baseline_mean=baseline_mean, baseline_sd=trial_sd,
            states=states, state_levels=state_levels,
        )
        from .store import _write_table
        with open_h5(round_path, "r+") as handle:
            responses = handle["responses"]
            if "summary" in responses:
                del responses["summary"]
            _write_table(responses.create_group("summary"), aggregate)
        report.update(figure=str(figure), baseline_figure=str(baseline_figure),
                      response_data=f"{round_path}:/responses",
                      summary_table=f"{round_path}:/responses/summary")
        report_path = Path(f"{stem}_continuousqc.json")
        report_path.write_text(json.dumps(report, indent=2))
        report["json"] = str(report_path)
    return report
