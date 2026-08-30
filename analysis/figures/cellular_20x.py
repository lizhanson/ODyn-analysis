"""Reusable 20x cellular metrics for the Figure 3 notebook.

Every returned row retains group, mouse, cohort, compartment, state, and unit
identity so ROIs can be summarized within session and sessions within mouse.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from .response_metrics import lifetime_sparseness


@dataclass(frozen=True)
class TemporalWindows:
    odor: tuple[float, float] = (0., 4.)
    early: tuple[float, float] = (0., 2.)
    late: tuple[float, float] = (2., 4.)


def _decode(values):
    return np.asarray([v.decode() if isinstance(v, bytes) else str(v)
                       for v in values])


def _source_path(grouped_path: Path, handle) -> Path:
    value = handle.attrs.get("source_round")
    if isinstance(value, bytes):
        value = value.decode()
    source = Path(str(value))
    if source.exists():
        return source
    stem = grouped_path.name.split("_pertrial_median_20x_grouped.h5")[0]
    fallback = grouped_path.parent / f"{stem}.h5"
    if not fallback.exists():
        raise FileNotFoundError(f"source round not found: {source}")
    return fallback


def load_population(grouped_path, population):
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
        # The grouped 20x product historically stored positional indices in
        # /trial_id.  Use the source round's database trial IDs for auxiliary
        # joins; trial order is otherwise identical.
        source_trial_ids = handle["trials/trial_id"][:]
        if len(source_trial_ids) != data["z"].shape[1]:
            raise ValueError("source/grouped trial counts do not align")
        data["trial_id"] = source_trial_ids
    return data


def _common(row, population):
    line = row["population"].split("-")[0]
    return {
        "group_id": int(row["group_id"]), "mouse": row["mouse"],
        "line": line, "depth_class": row.get("depth_class", ""),
        "cohort": f"{line} {row.get('depth_class', '')}".strip(),
        "compartment": population,
    }


def tonic_table(data, row, population):
    common = _common(row, population)
    output = []
    for code, state_name in enumerate(data["state_levels"]):
        selected = data["state"] == code
        values = np.nanmedian(data["baseline_mean"][:, selected], axis=1)
        for unit_id, value in zip(data["unit_id"], values):
            output.append(common | {"unit_id": unit_id, "state": state_name,
                                    "baseline_f": float(value)})
    table = pd.DataFrame(output)
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


def _window(time_s, limits):
    return (time_s >= limits[0]) & (time_s < limits[1])


def temporal_feature_table(data, row, population, *, windows=TemporalWindows(),
                           reducer="median", blank_odor=0):
    common = _common(row, population)
    function = {"median": np.nanmedian, "mean": np.nanmean}[reducer]
    time_s = np.asarray(data["time_s"], float)
    odor_mask = _window(time_s, windows.odor)
    early_mask = _window(time_s, windows.early)
    late_mask = _window(time_s, windows.late)
    odor_time = time_s[odor_mask]
    rows = []
    for code, state_name in enumerate(data["state_levels"]):
        for odor in np.unique(data["odor_id"][data["state"] == code]):
            if int(odor) == int(blank_odor):
                continue
            selected = (data["state"] == code) & (data["odor_id"] == odor)
            waveform = function(data["z"][:, selected, :], axis=1)
            odor_wave = waveform[:, odor_mask]
            positive_auc = np.trapezoid(np.maximum(odor_wave, 0), odor_time, axis=1)
            negative_auc = np.trapezoid(np.maximum(-odor_wave, 0), odor_time, axis=1)
            peak_pos_index = np.nanargmax(odor_wave, axis=1)
            peak_neg_index = np.nanargmin(odor_wave, axis=1)
            early = np.nanmean(waveform[:, early_mask], axis=1)
            late = np.nanmean(waveform[:, late_mask], axis=1)
            denominator = np.abs(early) + np.abs(late)
            early_late = np.divide(late - early, denominator,
                                   out=np.full_like(early, np.nan),
                                   where=denominator > 0)
            for index, unit_id in enumerate(data["unit_id"]):
                rows.append(common | {
                    "unit_id": unit_id, "state": state_name, "odor_id": int(odor),
                    "n_trials": int(np.sum(selected)),
                    "mean_response_z": float(np.nanmean(odor_wave[index])),
                    "positive_auc_z_s": float(positive_auc[index]),
                    "negative_auc_z_s": float(negative_auc[index]),
                    "peak_positive_z": float(odor_wave[index, peak_pos_index[index]]),
                    "peak_negative_z": float(odor_wave[index, peak_neg_index[index]]),
                    "peak_positive_latency_s": float(odor_time[peak_pos_index[index]]),
                    "peak_negative_latency_s": float(odor_time[peak_neg_index[index]]),
                    "early_mean_z": float(early[index]), "late_mean_z": float(late[index]),
                    "early_late_index": float(early_late[index]),
                })
    return pd.DataFrame(rows)


def specificity_table(temporal):
    """Threshold-free signed lifetime sparseness and breadth from AUC."""
    rows = []
    keys = ["group_id", "mouse", "line", "depth_class", "cohort",
            "compartment", "state", "unit_id"]
    for key, group in temporal.groupby(keys, dropna=False):
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
