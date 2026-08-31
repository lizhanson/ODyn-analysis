"""Pupil and running covariates, aligned to the imaging trials.

The grouped and auxiliary products cannot be joined by position or by state
code: 20x grouped files store trial_id as a positional arange, grouped
products drop imaging trials the auxiliary product still holds, and the two
files order their state levels oppositely ('post','pre') versus ('pre','post').
Alignment therefore goes through the (odor, state-name) sequence, and every
result carries the alignment mode that produced it.
"""

from __future__ import annotations

import numpy as np

from .common import decode, find_auxiliary, find_grouped

TRIAL_COVARIATES = (
    "treadmill_mean_speed_cm_s", "treadmill_running_fraction",
    "treadmill_mean_velocity_cm_s", "pupil_median_diameter_px",
    "pupil_blink_fraction", "pupil_coverage_fraction",
    "respiration_mean_sniff_hz", "respiration_masked_fraction",
)
ODOR_FREE_WINDOWS = ((-5.0, -1.0), (8.0, 14.0))


def _keys(odor, state, levels):
    return np.array([f"{int(o)}|{levels[int(s)]}" for o, s in zip(odor, state)])


def align_to_auxiliary(row):
    """Indices into the auxiliary trials for each grouped trial.

    Returns None when either product is missing or no consistent ordering
    exists.  A contiguous block covers most sessions; the rest need an
    in-order subsequence match because dropped trials are scattered.
    """
    import h5py

    grouped, auxiliary = find_grouped(row), find_auxiliary(row)
    if grouped is None or auxiliary is None:
        return None
    with h5py.File(grouped, "r") as handle:
        g_key = _keys(handle["odor_id"][:], handle["state"][:],
                      decode(handle["state_levels"][:]))
    with h5py.File(auxiliary, "r") as handle:
        a_key = _keys(handle["trials/odor_id"][:], handle["trials/state"][:],
                      decode(handle["trials/state_levels"][:]))

    n, m = len(g_key), len(a_key)
    for offset in range(m - n + 1):
        if np.array_equal(a_key[offset:offset+n], g_key):
            return {"index": np.arange(offset, offset+n), "mode": "contiguous",
                    "n_dropped": m-n, "auxiliary": auxiliary}
    index, cursor = [], 0
    for key in g_key:
        while cursor < m and a_key[cursor] != key:
            cursor += 1
        if cursor == m:
            return None
        index.append(cursor)
        cursor += 1
    return {"index": np.asarray(index), "mode": "scattered", "n_dropped": m-n,
            "auxiliary": auxiliary}


def trial_covariates(row, alignment=None):
    """Per-trial arousal summaries in grouped-trial order."""
    import h5py
    import pandas as pd

    alignment = alignment or align_to_auxiliary(row)
    if alignment is None:
        return None
    index = alignment["index"]
    with h5py.File(alignment["auxiliary"], "r") as handle:
        data = {name: handle[f"trials/{name}"][:][index]
                for name in TRIAL_COVARIATES if f"trials/{name}" in handle}
        levels = decode(handle["trials/state_levels"][:])
        data["state"] = [levels[int(s)] for s in handle["trials/state"][:][index]]
        data["odor_id"] = handle["trials/odor_id"][:][index]
    frame = pd.DataFrame(data)
    frame["alignment_mode"] = alignment["mode"]
    return frame


def continuous_channels(row, alignment=None, *, names=("treadmill/speed",
                                                       "pupil/diameter_px")):
    """Trial x frame channels in grouped-trial order, plus the shared clock."""
    import h5py

    alignment = alignment or align_to_auxiliary(row)
    if alignment is None:
        return None
    index = alignment["index"]
    output = {}
    with h5py.File(alignment["auxiliary"], "r") as handle:
        for name in names:
            if name in handle:
                output[name.split("/")[-1]] = handle[name][:][index]
        output["time_from_odor_s"] = handle["acquisition/time_from_odor_s"][:][index]
    return output


def odor_free_mask(time_from_odor, windows=ODOR_FREE_WINDOWS):
    """Samples far enough from odor delivery to count as spontaneous."""
    time_from_odor = np.asarray(time_from_odor, float)
    mask = np.zeros(time_from_odor.shape, bool)
    for start, stop in windows:
        mask |= (time_from_odor >= start) & (time_from_odor < stop)
    return mask


def detect_onsets(signal, mask, *, threshold, min_quiet_samples=10,
                  min_run_samples=5):
    """Onsets where a channel crosses threshold after a quiet period.

    Both the quiet period and the crossing must lie inside the odor-free mask,
    so nothing anchored to odor delivery is counted as spontaneous.
    """
    signal = np.asarray(signal, float)
    mask = np.asarray(mask, bool)
    onsets = []
    for trial in range(signal.shape[0]):
        active = (signal[trial] > threshold) & mask[trial]
        quiet = (signal[trial] <= threshold) & mask[trial]
        for frame in range(min_quiet_samples,
                           signal.shape[1] - min_run_samples):
            if (active[frame]
                    and quiet[frame-min_quiet_samples:frame].all()
                    and active[frame:frame+min_run_samples].all()):
                onsets.append((trial, frame))
    return np.asarray(onsets, int).reshape(-1, 2)


def within_odor_deviation(values, odor_ids, states):
    """Residual after removing each odor's median within each state."""
    values = np.asarray(values, float)
    output = np.full(values.shape, np.nan)
    odor_ids, states = np.asarray(odor_ids), np.asarray(states)
    for state in np.unique(states):
        for odor in np.unique(odor_ids[states == state]):
            selected = (states == state) & (odor_ids == odor)
            if selected.sum() > 1:
                output[selected] = (values[selected]
                                    - np.nanmedian(values[selected]))
    return output
