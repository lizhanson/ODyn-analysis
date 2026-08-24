"""The one canonical transformation from detrended F to response summaries.

There are deliberately no alternate normalizations or responder calls here.
Traces are centred and scaled independently using their own immediate pre-odor
window. Session- and block-pooled baseline SDs are retained only as diagnostics.
Odor and post-odor epochs are then summarized as continuous z scores.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BASELINE_S = 4.0
POST_ODOR_S = 4.0


@dataclass
class StandardizedTraces:
    z: np.ndarray                         # unit x trial x frame
    baseline_mean: np.ndarray             # unit x trial
    baseline_sd_session: np.ndarray       # unit
    baseline_sd_trial: np.ndarray         # unit x trial
    baseline_sd_block: np.ndarray         # unit x state level
    baseline_frames: int
    state_levels: list[str] | None = None


@dataclass
class EpochScores:
    mean: dict[str, np.ndarray]            # each unit x trial
    peak_positive: dict[str, np.ndarray]
    peak_negative: dict[str, np.ndarray]
    post_odor_frames: int


def standardize_traces(
    detrended,
    *,
    odor_on_frames,
    states,
    n_state_levels,
    frame_rate,
    baseline_s=BASELINE_S,
) -> StandardizedTraces:
    """Centre and scale detrended F using each trial's pre-odor baseline."""
    traces = np.asarray(detrended, np.float32)
    on = np.asarray(odor_on_frames, int)
    states = np.asarray(states, int)
    if traces.ndim != 3:
        raise ValueError("detrended traces must be unit x trial x frame")
    if on.shape != (traces.shape[1],) or states.shape != (traces.shape[1],):
        raise ValueError("odor_on_frames and states must align with the trial axis")
    n_base = max(2, int(round(float(baseline_s) * float(frame_rate))))
    if np.any(on - n_base < 0):
        bad = np.flatnonzero(on - n_base < 0)
        raise ValueError(
            f"{len(bad)} trial(s) cannot provide the required {baseline_s:g} s "
            f"baseline; first trial index {int(bad[0])}."
        )

    n_unit, n_trial, _ = traces.shape
    baseline_mean = np.full((n_unit, n_trial), np.nan, np.float32)
    baseline_var = np.full_like(baseline_mean, np.nan)
    z = np.empty_like(traces)
    for trial, onset in enumerate(on):
        baseline = traces[:, trial, onset - n_base:onset]
        mu = np.nanmean(baseline, axis=1).astype(np.float32)
        baseline_mean[:, trial] = mu
        baseline_var[:, trial] = np.nanvar(
            baseline - mu[:, None], axis=1, ddof=1
        ).astype(np.float32)
        z[:, trial] = traces[:, trial] - mu[:, None]

    # Pooled estimates remain useful QC diagnostics, but are not the z-score
    # denominator. Equal trial weighting prevents imbalanced blocks from
    # dominating those diagnostic estimates.
    session_sd = np.sqrt(np.nanmean(baseline_var, axis=1)).astype(np.float32)
    trial_sd = np.sqrt(baseline_var).astype(np.float32)
    safe = np.where(trial_sd > 1e-9, trial_sd, np.nan)
    z /= safe[:, :, None]

    block_sd = np.full((n_unit, int(n_state_levels)), np.nan, np.float32)
    for code in range(int(n_state_levels)):
        selected = states == code
        if selected.any():
            block_sd[:, code] = np.sqrt(
                np.nanmean(baseline_var[:, selected], axis=1)
            ).astype(np.float32)

    return StandardizedTraces(
        z=z,
        baseline_mean=baseline_mean,
        baseline_sd_session=session_sd,
        baseline_sd_trial=trial_sd,
        baseline_sd_block=block_sd,
        baseline_frames=n_base,
    )


def epoch_scores(
    z,
    *,
    odor_on_frames,
    odor_off_frames,
    frame_rate,
    post_odor_s=POST_ODOR_S,
) -> EpochScores:
    """Continuous summaries for odor and the fixed four-second offset epoch."""
    z = np.asarray(z, np.float32)
    on = np.asarray(odor_on_frames, int)
    off = np.asarray(odor_off_frames, int)
    if z.ndim != 3 or on.shape != (z.shape[1],) or off.shape != (z.shape[1],):
        raise ValueError("epoch inputs do not align")
    n_post = max(1, int(round(float(post_odor_s) * float(frame_rate))))
    shape = z.shape[:2]
    means = {name: np.full(shape, np.nan, np.float32)
             for name in ("odor", "post_odor")}
    high = {name: np.full(shape, np.nan, np.float32) for name in means}
    low = {name: np.full(shape, np.nan, np.float32) for name in means}
    for trial, (start, stop) in enumerate(zip(on, off)):
        windows = {
            "odor": (int(start), int(stop)),
            "post_odor": (int(stop), int(stop + n_post)),
        }
        for name, (left, right) in windows.items():
            if left < 0 or right > z.shape[2] or right <= left:
                raise ValueError(
                    f"trial {trial} cannot provide the complete {name} window "
                    f"[{left}, {right}) within {z.shape[2]} frames"
                )
            window = z[:, trial, left:right]
            means[name][:, trial] = np.nanmean(window, axis=1)
            high[name][:, trial] = np.nanmax(window, axis=1)
            low[name][:, trial] = np.nanmin(window, axis=1)

    return EpochScores(means, high, low, n_post)


def trial_epoch_table(
    scores: EpochScores,
    *,
    unit_ids,
    odor_ids,
    states,
    state_levels,
    trial_ids=None,
    unit_types=None,
    group_ids=None,
):
    """One row per analysis unit, trial, and epoch; no p or q values."""
    import pandas as pd

    unit_ids = list(unit_ids)
    odor_ids = np.asarray(odor_ids)
    states = np.asarray(states, int)
    trial_ids = np.arange(len(odor_ids)) if trial_ids is None else np.asarray(trial_ids)
    unit_types = (["roi"] * len(unit_ids) if unit_types is None else list(unit_types))
    group_ids = ([None] * len(unit_ids) if group_ids is None else list(group_ids))
    rows = []
    for epoch in ("odor", "post_odor"):
        for unit, unit_id in enumerate(unit_ids):
            for trial in range(len(odor_ids)):
                rows.append({
                    "unit_id": unit_id,
                    "unit_type": unit_types[unit],
                    "group_id": group_ids[unit],
                    "trial_id": int(trial_ids[trial]),
                    "trial_index": trial,
                    "odor_id": int(odor_ids[trial]),
                    "block": state_levels[int(states[trial])],
                    "epoch": epoch,
                    "response_mean_z": float(scores.mean[epoch][unit, trial]),
                    "response_peak_positive_z": float(scores.peak_positive[epoch][unit, trial]),
                    "response_peak_negative_z": float(scores.peak_negative[epoch][unit, trial]),
                })
    return pd.DataFrame(rows)


def aggregate_epoch_table(table):
    """Unit-odor-block-epoch summaries over presentations of that odor."""
    import pandas as pd
    from scipy.stats import t

    keys = ["unit_id", "unit_type", "group_id", "odor_id", "block", "epoch"]

    def summarize(frame):
        values = frame["response_mean_z"].dropna().to_numpy(float)
        n = len(values)
        mean = float(np.mean(values)) if n else np.nan
        sem = float(np.std(values, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
        half = float(t.ppf(.975, n - 1) * sem) if n > 1 else np.nan
        return pd.Series({
            "n_trials": n,
            "mean_response_z": mean,
            "median_response_z": float(np.median(values)) if n else np.nan,
            "response_q25": float(np.percentile(values, 25)) if n else np.nan,
            "response_q75": float(np.percentile(values, 75)) if n else np.nan,
            "response_ci95_low": mean - half if n > 1 else np.nan,
            "response_ci95_high": mean + half if n > 1 else np.nan,
        })

    # Build rows explicitly rather than relying on pandas' changing groupby.apply
    # treatment of grouping columns. This works on both workstation and cluster
    # pandas versions and makes the exported schema deterministic.
    rows = []
    for values, frame in table.groupby(keys, dropna=False, sort=False):
        if not isinstance(values, tuple):
            values = (values,)
        row = dict(zip(keys, values))
        row.update(summarize(frame).to_dict())
        rows.append(row)
    return pd.DataFrame(rows)
