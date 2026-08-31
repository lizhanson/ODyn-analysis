"""Scale-agnostic population metrics for the 10x and 20x figure paths.

Both scales are measured the same way so they can be compared directly. Two
choices differ from the earlier mean-based path, and both follow from the
exploratory pass:

Excitation and suppression are integrated separately over the response
waveform rather than read from a signed four-second mean. A unit that is
excited and then suppressed averages to nothing over four seconds, and that
describes most bidirectional responses in this dataset.

Responder calls are referenced to each unit's own pre-odor excursions rather
than to mineral oil. Mineral oil is not a null stimulus here: it evokes a real
delivery response whose size varies with state and scale, so a blank-referenced
cutoff moves between conditions for reasons unrelated to odor coding. The null
is stratified by trial count because a median over more trials is quieter, and
repeats per odor range from one to eleven within a single session.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .response_metrics import lifetime_sparseness

SMOOTH_S = 0.5
MIN_NULL = 40


@dataclass(frozen=True)
class TemporalWindows:
    odor: tuple[float, float] = (0., 4.)
    early: tuple[float, float] = (0., 2.)
    late: tuple[float, float] = (2., 4.)
    baseline: tuple[float, float] = (-5., -1.)


def _decode(values):
    return np.asarray([v.decode() if isinstance(v, bytes) else str(v)
                       for v in values])


def _window(time_s, limits):
    time_s = np.asarray(time_s, float)
    return (time_s >= limits[0]) & (time_s < limits[1])


def _source_path(grouped_path: Path, handle) -> Path:
    """Locate the extraction round behind a 10x or 20x grouped product."""
    for key in ("source_round", "source_component_round"):
        value = handle.attrs.get(key)
        if value is None:
            continue
        if isinstance(value, bytes):
            value = value.decode()
        source = Path(str(value))
        if source.exists():
            return source
    for objective in ("20x", "10x"):
        marker = f"_pertrial_median_{objective}_grouped.h5"
        if marker in grouped_path.name:
            stem = grouped_path.name.split(marker)[0]
            fallback = grouped_path.parent / f"{stem}.h5"
            if fallback.exists():
                return fallback
    raise FileNotFoundError(f"source round not found for {grouped_path.name}")


def load_population(grouped_path, population):
    """Load one population from a grouped product, with database trial ids."""
    import h5py

    grouped_path = Path(grouped_path)
    with h5py.File(grouped_path) as handle:
        source = _source_path(grouped_path, handle)
        root = handle[population]
        data = {
            "unit_id": _decode(root["unit_id"][:]),
            "z": root["z"][:],
            "baseline_mean": root["baseline_mean"][:],
            "odor_id": handle["odor_id"][:],
            "state": handle["state"][:],
            "state_levels": _decode(handle["state_levels"][:]),
            "trial_id": handle["trial_id"][:],
        }
    with h5py.File(source) as handle:
        data["time_s"] = handle["traces/time_s"][:]
        # The grouped 20x product stores positional indices in /trial_id; the
        # 10x product stores database ids. The source round always carries the
        # database ids, and its trial order matches the grouped product.
        source_trial_ids = handle["trials/trial_id"][:]
        if len(source_trial_ids) != data["z"].shape[1]:
            raise ValueError("source/grouped trial counts do not align")
        data["trial_id"] = source_trial_ids
    return data


def _common(row, population):
    line = row["population"].split("-")[0]
    depth = str(row.get("depth_class", "") or "")
    # 10x rows carry depth_class 'na'; the cohort is then just the line.
    cohort = f"{line} {depth}".strip() if depth and depth != "na" else line
    return {
        "group_id": int(row["group_id"]), "mouse": row["mouse"],
        "line": line, "depth_class": depth, "cohort": cohort,
        "compartment": population,
    }


def box_smooth(traces, width):
    """Running mean along the last axis, NaN-aware, no wraparound."""
    width = max(1, int(width))
    x = np.asarray(traces, float)
    if width == 1:
        return x
    kernel = np.ones(width)
    total = np.apply_along_axis(
        lambda v: np.convolve(v, kernel, "same"), -1, np.nan_to_num(x, nan=0.))
    count = np.apply_along_axis(
        lambda v: np.convolve(v, kernel, "same"), -1,
        np.isfinite(x).astype(float))
    return np.divide(total, count, out=np.full_like(total, np.nan),
                     where=count > 0)


def excursion_thresholds(smoothed, time_s, counts, *, tail_probability=.05,
                         baseline=(-5., -1.)):
    """Cutoffs from maximal pre-odor excursions of the same median waveforms.

    Smoothing enforces a minimum duration without a separate persistence rule,
    and the pre-odor segment of the very same waveform is an exactly matched
    null: same unit, same trials, same smoothing, same window length. Returns
    positive and negative cutoffs keyed by trial count, plus pooled fallbacks.
    """
    q = float(tail_probability)
    if not 0 < q < .5:
        raise ValueError("tail_probability must lie between 0 and 0.5")
    segment = smoothed[..., _window(time_s, baseline)]
    if segment.shape[-1] < 3:
        raise ValueError("baseline window is too short for an excursion null")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        high = np.nanmax(segment, axis=-1)
        low = np.nanmin(segment, axis=-1)
    finite_high, finite_low = high[np.isfinite(high)], low[np.isfinite(low)]
    if finite_high.size < MIN_NULL or finite_low.size < MIN_NULL:
        raise ValueError("not enough finite baseline excursions")
    pooled = (float(np.quantile(finite_high, 1-q)),
              float(np.quantile(finite_low, q)))
    positive, negative = {}, {}
    counts = np.asarray(counts)
    for count in np.unique(counts[counts > 0]):
        column = counts == count
        h, l = high[:, column].ravel(), low[:, column].ravel()
        h, l = h[np.isfinite(h)], l[np.isfinite(l)]
        if h.size >= MIN_NULL and l.size >= MIN_NULL:
            positive[int(count)] = float(np.quantile(h, 1-q))
            negative[int(count)] = float(np.quantile(l, q))
    return {"positive": positive, "negative": negative, "pooled": pooled,
            "tail_probability": q}


def _cutoffs(thresholds, count):
    count = int(count)
    return (thresholds["positive"].get(count, thresholds["pooled"][0]),
            thresholds["negative"].get(count, thresholds["pooled"][1]))


def tonic_table(data, row, population):
    common = _common(row, population)
    output = []
    for code, state_name in enumerate(data["state_levels"]):
        selected = data["state"] == code
        if not np.any(selected):
            continue
        values = np.nanmedian(data["baseline_mean"][:, selected], axis=1)
        for unit_id, value in zip(data["unit_id"], values):
            output.append(common | {"unit_id": unit_id, "state": state_name,
                                    "baseline_f": float(value)})
    table = pd.DataFrame(output)
    if table.empty:
        return table
    pivot = table.pivot(index=[*common, "unit_id"], columns="state",
                        values="baseline_f").reset_index()
    if {"pre", "post"}.issubset(pivot.columns):
        positive = table.loc[table.baseline_f > 0, "baseline_f"]
        floor = float(positive.median()) * 1e-9 if len(positive) else 1e-12
        pivot["f0_log2_post_pre"] = np.log2(
            np.maximum(pivot["post"], floor) / np.maximum(pivot["pre"], floor))
        table = table.merge(
            pivot[[*common, "unit_id", "f0_log2_post_pre"]],
            on=[*common, "unit_id"], how="left")
    return table


def temporal_feature_table(data, row, population, *, windows=TemporalWindows(),
                           reducer="median", blank_odor=0,
                           tail_probability=.05, smooth_s=SMOOTH_S,
                           include_blank=False):
    """Per unit and odor waveform features plus pre-odor responder calls."""
    common = _common(row, population)
    function = {"median": np.nanmedian, "mean": np.nanmean}[reducer]
    time_s = np.asarray(data["time_s"], float)
    odor_mask = _window(time_s, windows.odor)
    odor_time = time_s[odor_mask]
    frame_s = float(np.median(np.diff(time_s)))
    rows = []
    for code, state_name in enumerate(data["state_levels"]):
        in_state = data["state"] == code
        if not np.any(in_state):
            continue
        levels = [int(o) for o in np.unique(data["odor_id"][in_state])]
        reported = [o for o in levels if include_blank or o != int(blank_odor)]
        if not reported:
            continue
        # One pass builds every odor's waveform, so the pre-odor null is drawn
        # from exactly the traces the calls are made on.
        waveforms, counts = [], []
        for odor in levels:
            selected = in_state & (data["odor_id"] == odor)
            counts.append(int(np.sum(selected)))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                waveforms.append(function(data["z"][:, selected, :], axis=1))
        stack = np.stack(waveforms, axis=1)            # unit x odor x frame
        counts = np.asarray(counts)
        try:
            thresholds = excursion_thresholds(
                box_smooth(stack, round(smooth_s/frame_s)), time_s, counts,
                tail_probability=tail_probability, baseline=windows.baseline)
        except ValueError:
            thresholds = None
        smoothed = box_smooth(stack, round(smooth_s/frame_s))
        for position, odor in enumerate(levels):
            if odor not in reported:
                continue
            waveform = stack[:, position, :]
            odor_wave = waveform[:, odor_mask]
            smooth_wave = smoothed[:, position, odor_mask]
            positive_auc = np.trapezoid(np.maximum(odor_wave, 0), odor_time, axis=1)
            negative_auc = np.trapezoid(np.maximum(-odor_wave, 0), odor_time, axis=1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                peak_pos_index = np.nanargmax(odor_wave, axis=1)
                peak_neg_index = np.nanargmin(odor_wave, axis=1)
                early = np.nanmean(waveform[:, _window(time_s, windows.early)], axis=1)
                late = np.nanmean(waveform[:, _window(time_s, windows.late)], axis=1)
                smooth_high = np.nanmax(smooth_wave, axis=1)
                smooth_low = np.nanmin(smooth_wave, axis=1)
            denominator = np.abs(early) + np.abs(late)
            early_late = np.divide(late - early, denominator,
                                   out=np.full_like(early, np.nan),
                                   where=denominator > 0)
            if thresholds is None:
                cut_high = cut_low = np.nan
            else:
                cut_high, cut_low = _cutoffs(thresholds, counts[position])
            for index, unit_id in enumerate(data["unit_id"]):
                rows.append(common | {
                    "unit_id": unit_id, "state": state_name, "odor_id": int(odor),
                    "is_blank": int(odor) == int(blank_odor),
                    "n_trials": int(counts[position]),
                    "mean_response_z": float(np.nanmean(odor_wave[index])),
                    "positive_auc_z_s": float(positive_auc[index]),
                    "negative_auc_z_s": float(negative_auc[index]),
                    "peak_positive_z": float(odor_wave[index, peak_pos_index[index]]),
                    "peak_negative_z": float(odor_wave[index, peak_neg_index[index]]),
                    "peak_positive_latency_s": float(odor_time[peak_pos_index[index]]),
                    "peak_negative_latency_s": float(odor_time[peak_neg_index[index]]),
                    "early_mean_z": float(early[index]), "late_mean_z": float(late[index]),
                    "early_late_index": float(early_late[index]),
                    "threshold_positive_z": float(cut_high),
                    "threshold_negative_z": float(cut_low),
                    "excited": bool(smooth_high[index] > cut_high)
                    if np.isfinite(cut_high) else False,
                    "suppressed": bool(smooth_low[index] < cut_low)
                    if np.isfinite(cut_low) else False,
                })
    table = pd.DataFrame(rows)
    if not table.empty:
        table["biphasic"] = table.excited & table.suppressed
    return table


def breadth_table(temporal):
    """Responder breadth per unit, from pre-odor-referenced calls."""
    if temporal.empty:
        return temporal
    real = temporal[~temporal.is_blank] if "is_blank" in temporal else temporal
    keys = ["group_id", "mouse", "line", "depth_class", "cohort",
            "compartment", "state", "unit_id"]
    grouped = real.groupby(keys, dropna=False).agg(
        n_odor=("odor_id", "size"),
        excitation_breadth=("excited", "mean"),
        suppression_breadth=("suppressed", "mean"),
        biphasic_breadth=("biphasic", "mean"),
        median_positive_auc_z_s=("positive_auc_z_s", "median"),
        median_negative_auc_z_s=("negative_auc_z_s", "median"),
    ).reset_index()
    total = (grouped.median_positive_auc_z_s + grouped.median_negative_auc_z_s)
    grouped["es_balance"] = np.divide(
        grouped.median_positive_auc_z_s - grouped.median_negative_auc_z_s, total,
        out=np.full(len(grouped), np.nan), where=total > 0)
    return grouped


def specificity_table(temporal):
    """Threshold-free signed lifetime sparseness and breadth from AUC."""
    rows = []
    keys = ["group_id", "mouse", "line", "depth_class", "cohort",
            "compartment", "state", "unit_id"]
    real = temporal[~temporal.is_blank] if "is_blank" in temporal else temporal
    for key, group in real.groupby(keys, dropna=False):
        positive = group.sort_values("odor_id").positive_auc_z_s.to_numpy()
        negative = group.sort_values("odor_id").negative_auc_z_s.to_numpy()
        row = dict(zip(keys, key))
        row.update(
            excitation_auc_lifetime_sparseness=float(lifetime_sparseness(positive)),
            suppression_auc_lifetime_sparseness=float(lifetime_sparseness(negative)),
            median_positive_auc_z_s=float(np.nanmedian(positive)),
            median_negative_auc_z_s=float(np.nanmedian(negative)),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _row_correlation(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ac = a - np.nanmean(a, axis=1, keepdims=True)
    bc = b - np.nanmean(b, axis=1, keepdims=True)
    denominator = np.sqrt(np.nansum(ac*ac, axis=1) * np.nansum(bc*bc, axis=1))
    return np.divide(np.nansum(ac*bc, axis=1), denominator,
                     out=np.full(a.shape[0], np.nan), where=denominator > 0)


def reliability_tables(data, row, population, *, repeats=100, seed=0,
                       blank_odor=0, windows=TemporalWindows()):
    """Repeated split-half unit tuning and population-pattern reliability."""
    common = _common(row, population)
    response = np.nanmean(data["z"][:, :, _window(data["time_s"], windows.odor)], axis=2)
    rng = np.random.default_rng(seed)
    unit_rows, odor_rows = [], []
    for code, state_name in enumerate(data["state_levels"]):
        odors = [int(o) for o in np.unique(data["odor_id"][data["state"] == code])
                 if int(o) != int(blank_odor)]
        indices = {odor: np.flatnonzero((data["state"] == code) &
                                        (data["odor_id"] == odor)) for odor in odors}
        usable = [odor for odor in odors if len(indices[odor]) >= 2]
        if len(usable) < 3:
            continue
        unit_values = {"signed": [], "excitation": [], "suppression": []}
        odor_values = {odor: [] for odor in usable}
        for _ in range(int(repeats)):
            halves_a, halves_b = [], []
            for odor in usable:
                shuffled = rng.permutation(indices[odor]); cut = len(shuffled)//2
                halves_a.append(np.nanmean(response[:, shuffled[:cut]], axis=1))
                halves_b.append(np.nanmean(response[:, shuffled[cut:]], axis=1))
            a, b = np.column_stack(halves_a), np.column_stack(halves_b)
            unit_values["signed"].append(_row_correlation(a, b))
            unit_values["excitation"].append(_row_correlation(np.maximum(a, 0),
                                                               np.maximum(b, 0)))
            unit_values["suppression"].append(_row_correlation(np.maximum(-a, 0),
                                                                np.maximum(-b, 0)))
            for odor_index, odor in enumerate(usable):
                av, bv = a[:, odor_index], b[:, odor_index]
                denom = np.linalg.norm(av) * np.linalg.norm(bv)
                odor_values[odor].append(np.dot(av, bv)/denom if denom > 0 else np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            medians = {name: np.nanmedian(np.stack(values), axis=0)
                       for name, values in unit_values.items()}
        for index, unit_id in enumerate(data["unit_id"]):
            unit_rows.append(common | {
                "unit_id": unit_id, "state": state_name, "n_odors": len(usable),
                "tuning_reliability_signed": float(medians["signed"][index]),
                "tuning_reliability_excitation": float(medians["excitation"][index]),
                "tuning_reliability_suppression": float(medians["suppression"][index]),
            })
        for odor, values in odor_values.items():
            odor_rows.append(common | {
                "state": state_name, "odor_id": odor,
                "population_pattern_reliability": float(np.nanmedian(values)),
                "n_trials": len(indices[odor]),
            })
    return pd.DataFrame(unit_rows), pd.DataFrame(odor_rows)


def matched_compartment_table(tables, *, value_columns):
    """Join soma/process unit tables using shared curated group IDs."""
    soma, process = tables["somas"].copy(), tables["processes"].copy()
    keys = [name for name in ("group_id", "mouse", "line", "depth_class",
                              "cohort", "state", "odor_id", "unit_id")
            if name in soma and name in process]
    soma = soma[keys + list(value_columns)]
    process = process[keys + list(value_columns)]
    return soma.merge(process, on=keys, suffixes=("_soma", "_process"),
                      validate="one_to_one")
