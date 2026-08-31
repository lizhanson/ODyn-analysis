"""Pupil coupling, matched to the running analysis.

Pupil needs three things running did not.

First, the awake trace is censored where it matters: coverage is lost mainly to
clipping (up to 41% of awake samples), not blinks, and clipping happens when
the pupil is large.  Masking those samples would discard real dilations, so the
unmasked fit is primary with blinks removed, and the masked trace is carried as
a sensitivity check.

Second, pupil is measured in pixels that differ between sessions, so every
covariate is expressed as a fraction of that session's own baseline.

Third, awake pupil and running covary, so a raw pupil coupling could simply be
the running coupling.  Partial correlations separate them, and the anesthetized
block gives a locomotion-free control: running is identically zero there while
pupil still varies.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .arousal import align_to_auxiliary, continuous_channels, trial_covariates
from .common import decode, manifest_rows, odor_groups
from .stage4_regression import window_mean
from .stage6_arousal_time import columnwise_correlation

ODOR_WINDOW = (0.0, 4.0)
BASELINE_WINDOW = (-5.0, -1.0)
MIN_TRIALS = 20
MIN_COVERAGE = 0.5
MIN_GROUP_TRIALS = 8


def centre_by_odor(response, odor):
    """Odor-centred deviation for unit x trial x bin, vectorised over units."""
    output = np.full(response.shape, np.nan)
    for level in np.unique(odor):
        rows = odor == level
        if rows.sum() > 1:
            median = np.nanmedian(response[:, rows, :], axis=1, keepdims=True)
            output[:, rows, :] = response[:, rows, :] - median
    return output


def centre_vector(values, odor):
    output = np.full(values.shape, np.nan)
    for level in np.unique(odor):
        rows = odor == level
        if rows.sum() > 1:
            output[rows] = values[rows] - np.nanmedian(values[rows])
    return output


def partial_correlation(a, b, control):
    """Correlation of a and b with the linear effect of `control` removed."""
    good = np.isfinite(a) & np.isfinite(b) & np.isfinite(control)
    if good.sum() < 10:
        return np.nan
    a, b, c = a[good], b[good], control[good]
    if min(np.std(a), np.std(b), np.std(c)) <= 0:
        return np.nan
    rab, rac, rbc = (np.corrcoef(a, b)[0, 1], np.corrcoef(a, c)[0, 1],
                     np.corrcoef(b, c)[0, 1])
    denominator = np.sqrt((1-rac**2)*(1-rbc**2))
    return float((rab - rac*rbc)/denominator) if denominator > 0 else np.nan


def pupil_channels(auxiliary, index, *, masked=False):
    """Blink-free pupil trace in grouped-trial order."""
    import h5py

    with h5py.File(auxiliary, "r") as handle:
        if "pupil" not in handle:
            return None
        name = "diameter_px" if masked else "diameter_unmasked_px"
        diameter = handle[f"pupil/{name}"][:][index].astype(float)
        blink = handle["pupil/blink"][:][index].astype(bool)
        clipped = handle["pupil/clipped"][:][index].astype(bool)
    diameter[blink] = np.nan          # blinks are artifact; clipping is not
    return diameter, clipped


def session_rows(row, cache_path, features, *, masked=False):
    alignment = align_to_auxiliary(row)
    if alignment is None:
        return []
    covariates = trial_covariates(row, alignment)
    channels = continuous_channels(row, alignment)
    pupil = pupil_channels(alignment["auxiliary"], alignment["index"],
                           masked=masked)
    if covariates is None or pupil is None:
        return []
    diameter, clipped = pupil
    data = np.load(cache_path, allow_pickle=True)
    levels = [str(x) for x in data["state_levels"]]
    if len(covariates) != len(data["odor_id"]):
        return []

    time = np.asarray(channels["time_from_odor_s"], float)
    speed = (np.abs(np.asarray(channels["speed"], float))
             if "speed" in channels else np.full(diameter.shape, np.nan))
    groups = odor_groups()
    output = []

    for code, state in enumerate(levels):
        rows = data["state"] == code
        if rows.sum() < MIN_TRIALS:
            continue
        coverage = float(np.isfinite(diameter[rows]).mean())
        if coverage < MIN_COVERAGE:
            continue
        base = window_mean(diameter, time, BASELINE_WINDOW)[rows]
        during = window_mean(diameter, time, ODOR_WINDOW)[rows]
        scale = np.nanmedian(base)
        if not np.isfinite(scale) or scale <= 0:
            continue
        # Fractions of this session's own baseline pupil, so sessions with
        # different optics and eye sizes are comparable.
        tonic = base/scale
        phasic = (during - base)/scale
        running = window_mean(speed, time, ODOR_WINDOW)[rows]

        odor = data["odor_id"][rows]
        tonic_d = centre_vector(tonic, odor)
        phasic_d = centre_vector(phasic, odor)
        running_d = centre_vector(running, odor)

        for population in sorted({k.split("/")[0] for k in data.files if "/" in k}):
            binned = data[f"{population}/binned"][:, rows, :]
            centres = data[f"{population}/bin_centre_s"]
            unit_ids = np.asarray([str(u) for u in data[f"{population}/unit_id"]])
            deviation = centre_by_odor(binned, odor)
            common = {"group_id": int(row["group_id"]), "mouse": row["mouse"],
                      "line": row["line"], "cohort": row["cohort"],
                      "population": population, "state": state,
                      "coverage": coverage, "masked": masked,
                      "clipped_fraction": float(np.mean(clipped[rows])),
                      "n_trial": int(rows.sum())}

            for label, covariate in (("pupil_phasic", phasic_d),
                                     ("pupil_tonic", tonic_d),
                                     ("running", running_d)):
                coupling = columnwise_correlation(deviation, covariate)
                for index, centre in enumerate(centres):
                    output.append(common | {
                        "analysis": "time_course", "covariate": label,
                        "subset": "all", "bin_centre_s": float(centre),
                        "median_r": float(np.nanmedian(coupling[:, index])),
                        "n_unit": int(np.isfinite(coupling[:, index]).sum())})

            window = (centres >= ODOR_WINDOW[0]) & (centres < ODOR_WINDOW[1])
            response = np.nanmean(deviation[:, :, window], axis=2)

            # Pupil with running removed, and running with pupil removed.
            if np.isfinite(running_d).sum() >= 10:
                for label, a, control in (("pupil_phasic|running", phasic_d, running_d),
                                          ("running|pupil_phasic", running_d, phasic_d)):
                    values = [partial_correlation(response[i], a, control)
                              for i in range(response.shape[0])]
                    output.append(common | {
                        "analysis": "partial", "covariate": label,
                        "subset": "all", "bin_centre_s": np.nan,
                        "median_r": float(np.nanmedian(values)),
                        "n_unit": int(np.isfinite(values).sum())})

            # Dilation trials against constriction trials.
            for label, mask in (("dilation", phasic_d > 0),
                                ("constriction", phasic_d < 0)):
                if mask.sum() < MIN_GROUP_TRIALS:
                    continue
                sub = centre_by_odor(binned[:, mask, :], odor[mask])
                sub_response = np.nanmean(sub[:, :, window], axis=2)
                coupling = columnwise_correlation(
                    sub_response[:, :, None], centre_vector(phasic[mask], odor[mask]))
                output.append(common | {
                    "analysis": "dilation_split", "covariate": "pupil_phasic",
                    "subset": label, "bin_centre_s": np.nan,
                    "median_r": float(np.nanmedian(coupling[:, 0])),
                    "n_unit": int(np.isfinite(coupling[:, 0]).sum()),
                    "n_trial_subset": int(mask.sum())})

            # Odor groups, with the reciprocal pair 17/18 kept separate.
            label_by_trial = np.where(
                np.isin(odor, (17, 18)), "pair 17/18",
                np.array([groups.get(int(o), "other") for o in odor]))
            for name in np.unique(label_by_trial):
                rows_in = label_by_trial == name
                if rows_in.sum() < MIN_GROUP_TRIALS:
                    continue
                sub = centre_by_odor(binned[:, rows_in, :], odor[rows_in])
                sub_response = np.nanmean(sub[:, :, window], axis=2)
                coupling = columnwise_correlation(
                    sub_response[:, :, None],
                    centre_vector(phasic[rows_in], odor[rows_in]))
                output.append(common | {
                    "analysis": "odor_group", "covariate": "pupil_phasic",
                    "subset": str(name), "bin_centre_s": np.nan,
                    "median_r": float(np.nanmedian(coupling[:, 0])),
                    "n_unit": int(np.isfinite(coupling[:, 0]).sum()),
                    "n_trial_subset": int(rows_in.sum())})

            # Split by the sign of the unit's own response to that odor.
            table = features[(features.group_id == int(row["group_id"]))
                             & (features.population == population)
                             & (features.state == state)]
            if len(table):
                unit_index = {unit: i for i, unit in enumerate(unit_ids)}
                for kind in ("excited", "suppressed"):
                    values = []
                    selected = table[table[kind]]
                    for unit, odor_id in zip(selected.unit_id, selected.odor_id):
                        i = unit_index.get(str(unit))
                        if i is None:
                            continue
                        trials = odor == odor_id
                        if trials.sum() < 4:
                            continue
                        a, b = response[i][trials], phasic_d[trials]
                        good = np.isfinite(a) & np.isfinite(b)
                        if good.sum() < 4 or np.std(a[good]) == 0 or np.std(b[good]) == 0:
                            continue
                        values.append(np.corrcoef(a[good], b[good])[0, 1])
                    if values:
                        output.append(common | {
                            "analysis": "response_sign", "covariate": "pupil_phasic",
                            "subset": kind, "bin_centre_s": np.nan,
                            "median_r": float(np.nanmedian(values)),
                            "n_unit": len(values)})
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--masked", action="store_true",
                        help="sensitivity run on the quality-masked trace")
    args = parser.parse_args(argv)
    from .stage2_summarise import load_features
    features = load_features(args.output_dir / "features")
    features = features[~features.is_blank]
    rows, failures = [], []
    for row in manifest_rows():
        cache = args.output_dir / "trials" / f"group{int(row['group_id'])}_trials.npz"
        if not cache.exists():
            continue
        try:
            new = session_rows(row, cache, features, masked=args.masked)
            rows.extend(new)
            print(f"  {row['group_id']:>4} {len(new):>5} rows", flush=True)
        except Exception as error:
            failures.append({"group_id": int(row["group_id"]),
                             "error": f"{type(error).__name__}: {error}"})
            print(f"  {row['group_id']:>4} FAILED {failures[-1]['error']}",
                  flush=True)
    name = "stage7_pupil_masked.csv" if args.masked else "stage7_pupil.csv"
    pd.DataFrame(rows).to_csv(args.output_dir / name, index=False)
    print(f"\nwrote {len(rows):,} rows, {len(failures)} failures -> {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
