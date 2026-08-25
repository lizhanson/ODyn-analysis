"""Revise derived auxiliary traces without reopening pupil videos."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from .auxiliary import combined_qc_figure, treadmill_figure
from .respiration import QUALITY_THRESHOLD, extract_respiration, respiration_figure


def _decoded(values):
    return [value.decode() if isinstance(value, bytes) else str(value)
            for value in values]


def _replace_dataset(group, name, values):
    values = np.asarray(values)
    if name in group and group[name].shape == values.shape:
        group[name][...] = values
        return
    if name in group:
        del group[name]
    group.create_dataset(name, data=values, compression="gzip")


def _existing_context(path):
    """Arrays and labels needed to revise one consolidated auxiliary file."""
    from .h5io import open_h5

    with open_h5(path) as handle:
        attrs = dict(handle.attrs)
        levels = _decoded(handle["trials/state_levels"][:])
        state_codes = handle["trials/state"][:]
        states = np.asarray([levels[int(code)] for code in state_codes])
        metadata = {
            "exp_name": str(attrs["exp_name"]),
            "group_id": int(attrs["group_id"]),
            "frame_rate": float(attrs["frame_rate_hz"]),
            "manipulation": str(attrs.get("manipulation", "")),
            "acq_ids": handle["trials/acq_id"][:],
            "trial_ids": handle["trials/trial_id"][:],
            "odor_ids": handle["trials/odor_id"][:],
            "states": states,
            "state_codes": state_codes,
            "state_levels": levels,
            "odor_on_frames": handle["trials/odor_on_frame"][:],
            "odor_off_frames": handle["trials/odor_off_frame"][:],
        }
        treadmill = {
            "velocity": handle["treadmill/velocity"][:],
            "time_from_odor_s": handle["acquisition/time_from_odor_s"][:],
        }
        pupil = None
        if "pupil" in handle:
            eye = handle["pupil"]
            config = json.loads(eye.attrs["config_json"])
            equivalent = (
                eye["equivalent_diameter_unmasked_px"][:]
                if "equivalent_diameter_unmasked_px" in eye
                else eye["diameter_unmasked_px"][:]
            )
            unmasked = 2.0 * eye["major_radius_px"][:]
            blink = eye["blink"][:].astype(bool)
            fit_bad = (
                (eye["fit_inlier_fraction"][:] < config["min_inlier_fraction"])
                | (eye["fit_residual_px"][:] > config["max_residual_px"])
                | ~np.isfinite(unmasked)
            )
            clipped = fit_bad & ~blink
            masked = unmasked.copy()
            masked[blink | fit_bad] = np.nan
            pupil = {
                "diameter_masked": masked,
                "diameter_unmasked": unmasked,
                "equivalent_diameter_masked": np.where(
                    blink | fit_bad, np.nan, equivalent
                ),
                "equivalent_diameter_unmasked": equivalent,
                "blink": blink,
                "clipped": clipped,
            }
        sources = json.loads(str(attrs.get("sources_json", "{}")))
    return metadata, treadmill, pupil, sources


def _update_h5(path, *, respiration, pupil, snr_threshold):
    """Update a same-directory temporary copy and return its path."""
    from .h5io import open_h5

    path = Path(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}_revision_", suffix=".h5", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(path, temporary)
        with open_h5(temporary, "r+") as handle:
            resp = handle["respiration"]
            mapping = {
                "filtered_v": "filtered",
                "sniff_frequency_hz": "rate",
                "sniff_frequency_unmasked_hz": "rate_unmasked",
                "sniff_frequency_instantaneous_hz": "rate_instantaneous",
                "sniff_frequency_instantaneous_unmasked_hz":
                    "rate_instantaneous_unmasked",
                "snr": "quality",
            }
            for name, key in mapping.items():
                _replace_dataset(resp, name, respiration[key])
            resp.attrs["snr_threshold"] = float(snr_threshold)
            resp.attrs["sniff_smoothing_breaths"] = respiration["smooth_breaths"]
            resp.attrs["session_fraction_good"] = respiration["session_fraction_good"]
            resp.attrs["n_breaths"] = respiration["n_breaths"]

            trials = handle["trials"]
            _replace_dataset(
                trials, "respiration_masked_fraction",
                respiration["masked_fraction"],
            )
            _replace_dataset(
                trials, "respiration_mean_sniff_hz",
                np.nanmean(respiration["rate"], axis=1),
            )
            if pupil is not None:
                eye = handle["pupil"]
                _replace_dataset(eye, "diameter_px", pupil["diameter_masked"])
                _replace_dataset(
                    eye, "diameter_unmasked_px", pupil["diameter_unmasked"]
                )
                _replace_dataset(
                    eye, "equivalent_diameter_px",
                    pupil["equivalent_diameter_masked"],
                )
                _replace_dataset(
                    eye, "equivalent_diameter_unmasked_px",
                    pupil["equivalent_diameter_unmasked"],
                )
                _replace_dataset(eye, "clipped", pupil["clipped"].astype(np.int8))
                eye["diameter_px"].attrs["description"] = (
                    "Full fitted major-axis length, with blink and bad-fit "
                    "frames set to NaN."
                )
                eye["diameter_unmasked_px"].attrs["description"] = (
                    "Full fitted major-axis length before quality masking."
                )
                eye["equivalent_diameter_px"].attrs["description"] = (
                    "2 * sqrt(major * minor), with quality masking."
                )
                _replace_dataset(
                    trials, "pupil_masked_fraction",
                    np.mean(~np.isfinite(pupil["diameter_masked"]), axis=1),
                )
                _replace_dataset(
                    trials, "pupil_blink_fraction",
                    np.mean(pupil["blink"], axis=1),
                )
                _replace_dataset(
                    trials, "pupil_clipped_fraction",
                    np.mean(pupil["clipped"], axis=1),
                )
            handle.attrs["revision_json"] = json.dumps({
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "respiration_detection_band_hz": [0.5, 15.0],
                "respiration_snr_threshold": float(snr_threshold),
                "pupil_diameter": "full fitted major-axis length",
                "pupil_mask": "blink or bad ellipse fit",
                "pupil_videos_reopened": False,
            })
        return temporary
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _temporary_figure(destination):
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.stem}_revision_", suffix=destination.suffix,
        dir=destination.parent,
    )
    os.close(descriptor)
    return Path(name)


def revise_auxiliary(path, *, sync_path=None,
                     snr_threshold=QUALITY_THRESHOLD):
    """Revise one existing auxiliary HDF5 and its derivable QC atomically."""
    path = Path(path)
    metadata, treadmill, pupil, sources = _existing_context(path)
    sync_path = Path(sync_path or sources.get("sync", ""))
    if not sync_path.is_file():
        raise FileNotFoundError(f"Respiration sync file is unavailable: {sync_path}")

    respiration = extract_respiration(
        sync_path,
        acq_ids=metadata["acq_ids"], odor_ids=metadata["odor_ids"],
        states=metadata["state_codes"], trial_ids=metadata["trial_ids"],
        state_levels=metadata["state_levels"], exp_name=metadata["exp_name"],
        manipulation=metadata["manipulation"],
        quality_threshold=float(snr_threshold), save=False,
    )
    if respiration["rate"].shape != treadmill["velocity"].shape:
        raise ValueError(
            f"Recomputed respiration shape {respiration['rate'].shape} does not "
            f"match existing auxiliary shape {treadmill['velocity'].shape}."
        )

    combined_path = path.with_name(f"{path.stem}_qc.png")
    respiration_path = path.with_name(f"{path.stem}_respiration_odors.png")
    treadmill_path = path.with_name(f"{path.stem}_treadmill_odors.png")
    temporary_h5 = _update_h5(
        path, respiration=respiration, pupil=pupil,
        snr_threshold=snr_threshold,
    )
    temporary_combined = _temporary_figure(combined_path)
    temporary_respiration = _temporary_figure(respiration_path)
    temporary_treadmill = _temporary_figure(treadmill_path)
    try:
        combined_qc_figure(
            temporary_combined, metadata=metadata, respiration=respiration,
            pupil=pupil, treadmill=treadmill,
        )
        respiration_figure(respiration, temporary_respiration)
        treadmill_figure(
            temporary_treadmill, metadata=metadata, treadmill=treadmill
        )
        os.replace(temporary_h5, path)
        os.replace(temporary_combined, combined_path)
        os.replace(temporary_respiration, respiration_path)
        os.replace(temporary_treadmill, treadmill_path)
    except Exception:
        for temporary in (
            temporary_h5, temporary_combined, temporary_respiration,
            temporary_treadmill,
        ):
            temporary.unlink(missing_ok=True)
        raise

    return {
        "h5": str(path),
        "combined_figure": str(combined_path),
        "respiration_figure": str(respiration_path),
        "treadmill_figure": str(treadmill_path),
        "pupil_figure": "preserved (image examples require pupil videos)",
        "n_breaths": int(respiration["n_breaths"]),
        "respiration_fraction_good": float(respiration["session_fraction_good"]),
        "pupil_masked_fraction": (
            None if pupil is None else
            float(np.mean(~np.isfinite(pupil["diameter_masked"])))
        ),
    }
