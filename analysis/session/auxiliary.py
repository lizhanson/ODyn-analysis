"""Segmentation-independent pupil, respiration, and treadmill processing.

The acquisition clock is the common grid. Every modality is written as
``trial x imaging_frame`` and joined to later imaging rounds by ``acq_id``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

RUNNING_THRESHOLD_CM_S = 1.0


@dataclass
class AuxiliarySession:
    group_id: int
    exp_id: int
    exp_name: str
    exp_dir: Path
    frame_rate: float
    manipulation: str
    acq_ids: list[int]
    odor_ids: list[int]
    states: list[str]
    odor_on_frames: list[int]
    odor_off_frames: list[int]
    table: object
    trial_source: str = "database"
    sync_path: Path | None = None

    @property
    def output_dir(self):
        return self.exp_dir / "processed" / "python"


def resolve_auxiliary(group, *, group_id, manipulation="ketamine/xylazine",
                      sync_path=None):
    """Resolve trials and clocks without requiring images or motion correction."""
    from .resolve import _resolve_path, experiments_in_group
    from .respiration import find_behavior_sync
    from .sync import (acquisition_odor_windows, frame_onset_samples,
                       group_frames_into_acquisitions, open_sync,
                       pulse_intervals, read_channel)
    from .trials import trial_table, trial_table_from_events

    exp_ids = experiments_in_group(group, int(group_id))
    if len(exp_ids) != 1:
        raise ValueError(
            f"Auxiliary group {group_id} spans {len(exp_ids)} experiments; "
            "split-session auxiliary resolution is not implemented."
        )
    exp_id = exp_ids[0]
    experiment = group.experiments.query("exp_id == @exp_id").iloc[0]
    acquisition = group.acquisitions.query("exp_id == @exp_id").iloc[0]
    exp_dir = _resolve_path(Path(group.main_folder), acquisition.raw_path).parent.parent
    try:
        table = trial_table(group, exp_id=exp_id, manipulation=manipulation)
        trial_source = "database"
    except ValueError as error:
        if f"No trials for exp_id={exp_id}" not in str(error):
            raise
        table = trial_table_from_events(
            group, exp_id=exp_id, exp_dir=exp_dir, manipulation=manipulation
        )
        trial_source = "olfactometer_events+database_acquisitions"
    sync_path = Path(sync_path) if sync_path is not None else find_behavior_sync(exp_dir)
    sync = open_sync(sync_path)
    blocks = group_frames_into_acquisitions(
        frame_onset_samples(sync), rate_hz=sync.rate_hz
    )
    pulses = pulse_intervals(read_channel(sync, "odorPulse"), rate_hz=sync.rate_hz)
    windows = acquisition_odor_windows(blocks, pulses, rate_hz=sync.rate_hz)
    if len(windows) != len(table) or any(window is None for window in windows):
        raise ValueError(
            f"{experiment.exp_name}: {len(windows)} clock acquisitions and "
            f"{len(table)} database trials do not form a complete one-to-one alignment."
        )
    if table.acq_id.isna().any():
        raise ValueError(f"{experiment.exp_name}: trial table contains missing acq_id values.")
    return AuxiliarySession(
        group_id=int(group_id), exp_id=int(exp_id),
        exp_name=str(experiment.exp_name), exp_dir=exp_dir,
        frame_rate=float(experiment.frame_rate), manipulation=manipulation,
        acq_ids=table.acq_id.astype(int).tolist(),
        odor_ids=table.odor_id.astype(int).tolist(),
        states=table.state.astype(str).tolist(),
        odor_on_frames=[int(window[0]) for window in windows],
        odor_off_frames=[int(window[1]) for window in windows],
        table=table, trial_source=trial_source, sync_path=sync_path,
    )


def _state_codes(states):
    values = np.asarray(states).astype(str)
    preferred = [name for name in ("pre", "post") if name in values]
    levels = preferred + [name for name in np.unique(values) if name not in preferred]
    lookup = {name: index for index, name in enumerate(levels)}
    return np.asarray([lookup[value] for value in values], dtype=np.int16), levels


def _frame_layout(sync_path, n_trial):
    from .sync import frame_onset_samples, group_frames_into_acquisitions, open_sync

    sync = open_sync(sync_path)
    frames = frame_onset_samples(sync)
    blocks = group_frames_into_acquisitions(frames, rate_hz=sync.rate_hz)
    if len(blocks) != int(n_trial):
        raise ValueError(
            f"{len(blocks)} imaging acquisitions in {Path(sync_path).name}, "
            f"but {n_trial} trials were supplied."
        )
    widths = {len(block) for block in blocks}
    if len(widths) != 1:
        raise ValueError(f"Imaging acquisitions differ in length: {sorted(widths)}.")
    return sync, frames, blocks, widths.pop()


def extract_treadmill(
    sync_path,
    *,
    odor_on_frames,
    running_threshold_cm_s=RUNNING_THRESHOLD_CM_S,
):
    """Encoder velocity and derived movement features on the imaging frame grid."""
    from .h5io import open_h5
    from .sync import read_channel

    on = np.asarray(odor_on_frames, dtype=int)
    sync, frames, blocks, n_frame = _frame_layout(sync_path, len(on))
    with open_h5(sync.path) as handle:
        if "encoder" not in handle:
            raise KeyError(f"{sync.path.name} has no encoder channel.")
        encoder_attrs = {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in handle["encoder"].attrs.items()
        }
    velocity_sync = read_channel(sync, "encoder", column=2).astype(np.float64)
    block_samples = np.concatenate(blocks)
    velocity = velocity_sync[block_samples].reshape(len(blocks), n_frame)
    speed = np.abs(velocity)
    time_sync_s = block_samples.reshape(len(blocks), n_frame) / float(sync.rate_hz)
    time_from_odor_s = (
        time_sync_s - time_sync_s[np.arange(len(blocks)), on][:, None]
    )
    acceleration = np.gradient(velocity, axis=1) / np.gradient(time_sync_s, axis=1)
    dt = np.diff(time_sync_s, axis=1, prepend=time_sync_s[:, :1])
    distance = np.cumsum(velocity * dt, axis=1)
    running = speed >= float(running_threshold_cm_s)
    return {
        "velocity": velocity.astype(np.float32),
        "speed": speed.astype(np.float32),
        "acceleration": acceleration.astype(np.float32),
        "distance": distance.astype(np.float32),
        "running": running,
        "running_threshold_cm_s": float(running_threshold_cm_s),
        "time_sync_s": time_sync_s,
        "time_from_odor_s": time_from_odor_s,
        "encoder_attrs": encoder_attrs,
    }


def discover_pupil_videos(exp_dir):
    """Find converted pupil MP4s using the established experiment layout."""
    from .pupil import order_pupil_videos

    exp_dir = Path(exp_dir)
    roots = [exp_dir / name for name in ("video", "videos") if (exp_dir / name).is_dir()]
    roots += sorted(path for path in exp_dir.glob("*_video") if path.is_dir())
    # Some acquisitions keep imaging in ``mouse/e1`` but videos in the sibling
    # ``mouse/YYYYMMDD_mouse_e1_video`` directory.
    roots += sorted(
        path for path in exp_dir.parent.glob(f"*_{exp_dir.name}_video")
        if path.is_dir()
    )
    roots = list(dict.fromkeys(roots))
    converted = sorted({path for root in roots for path in root.rglob("converted")})
    movies = sorted({movie for folder in converted for movie in folder.glob("*.mp4")})
    if not movies:
        raise FileNotFoundError(f"No converted pupil MP4s below {exp_dir}.")
    return order_pupil_videos(movies)[0] if len(movies) == 2 else movies


def saved_pupil_config(exp_dir, *, group_id, exp_name):
    """Load the session-specific tuning required for unattended extraction."""
    from dataclasses import replace

    from .pupil import PupilConfig, load_pupil_tuning

    aux = Path(exp_dir) / "processed" / "python" / "aux"
    candidates = sorted(aux.glob(f"group{group_id}_{exp_name}*pupil_tuning.json"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one saved pupil tuning file for group {group_id} in {aux}; "
            f"found {len(candidates)}. Tune this session before batch extraction."
        )
    return replace(PupilConfig(), **load_pupil_tuning(candidates[0])), candidates[0]


def _trial_features(respiration, pupil, treadmill):
    n_trial = treadmill["velocity"].shape[0]
    rows = {
        "treadmill_running_fraction": treadmill["running"].mean(axis=1),
        "treadmill_mean_velocity_cm_s": np.nanmean(treadmill["velocity"], axis=1),
        "treadmill_mean_speed_cm_s": np.nanmean(treadmill["speed"], axis=1),
    }
    if respiration is not None:
        rows["respiration_masked_fraction"] = np.asarray(respiration["masked_fraction"])
        rate = np.asarray(respiration["rate"])
        count = np.sum(np.isfinite(rate), axis=1)
        rows["respiration_mean_sniff_hz"] = np.divide(
            np.nansum(rate, axis=1), count,
            out=np.full(n_trial, np.nan), where=count > 0,
        )
    else:
        rows["respiration_masked_fraction"] = np.full(n_trial, np.nan)
        rows["respiration_mean_sniff_hz"] = np.full(n_trial, np.nan)
    if pupil is not None:
        rows["pupil_masked_fraction"] = np.asarray(pupil["masked_fraction"])
        rows["pupil_blink_fraction"] = np.asarray(pupil["blink_fraction"])
        rows["pupil_clipped_fraction"] = np.asarray(pupil["clipped_fraction"])
        rows["pupil_coverage_fraction"] = np.asarray(pupil["coverage_fraction"])
        rows["pupil_median_diameter_px"] = np.nanmedian(pupil["diameter_masked"], axis=1)
    else:
        for name in ("pupil_masked_fraction", "pupil_blink_fraction",
                     "pupil_clipped_fraction", "pupil_coverage_fraction",
                     "pupil_median_diameter_px"):
            rows[name] = np.full(n_trial, np.nan)
    return rows


def write_auxiliary(path, *, metadata, respiration, pupil, treadmill, sources):
    """Write one self-describing acquisition-aligned auxiliary HDF5."""
    import json

    from .h5io import open_h5
    from .store import h5_string_dtype

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state_codes, state_levels = _state_codes(metadata["states"])
    n_trial, n_frame = treadmill["velocity"].shape
    features = _trial_features(respiration, pupil, treadmill)
    with open_h5(path, "w") as handle:
        handle.attrs["description"] = (
            "Segmentation-independent behavioral and physiological signals on "
            "the imaging acquisition grid; join later rounds by /trials/acq_id."
        )
        handle.attrs["schema_version"] = "1.0"
        for name in ("exp_name", "group_id", "mouse", "date", "manipulation",
                     "trial_source"):
            if metadata.get(name) is not None:
                handle.attrs[name] = metadata[name]
        handle.attrs["n_trial"] = n_trial
        handle.attrs["n_frame"] = n_frame
        handle.attrs["frame_rate_hz"] = float(metadata["frame_rate"])
        handle.attrs["respiration_available"] = int(respiration is not None)
        handle.attrs["pupil_available"] = int(pupil is not None)
        handle.attrs["treadmill_available"] = 1
        handle.attrs["sources_json"] = json.dumps(sources, default=str)

        acquisition = handle.create_group("acquisition")
        acquisition.create_dataset("time_sync_s", data=treadmill["time_sync_s"],
                                   compression="gzip")
        acquisition.create_dataset("time_from_odor_s",
                                   data=treadmill["time_from_odor_s"], compression="gzip")
        acquisition["time_from_odor_s"].attrs["units"] = "s"

        trials = handle.create_group("trials")
        trials.create_dataset("trial_index", data=np.arange(n_trial))
        for name in ("acq_ids", "trial_ids", "odor_ids", "odor_on_frames",
                     "odor_off_frames"):
            trials.create_dataset(name.removesuffix("s"), data=np.asarray(metadata[name]))
        trials.create_dataset("state", data=state_codes)
        trials.create_dataset("state_levels",
                              data=np.asarray(state_levels, dtype=h5_string_dtype()))
        for name, values in features.items():
            trials.create_dataset(name, data=values)

        tread = handle.create_group("treadmill")
        for name in ("velocity", "speed", "acceleration", "distance", "running"):
            tread.create_dataset(name, data=treadmill[name], compression="gzip")
        tread["velocity"].attrs["units"] = "cm/s"
        tread["speed"].attrs["units"] = "cm/s"
        tread["acceleration"].attrs["units"] = "cm/s^2"
        tread["distance"].attrs["units"] = "cm"
        tread.attrs["running_threshold_cm_s"] = treadmill["running_threshold_cm_s"]
        tread.attrs["source_encoder_attrs_json"] = json.dumps(
            treadmill["encoder_attrs"], default=str
        )

        if respiration is not None:
            resp = handle.create_group("respiration")
            for name, key in (
                ("filtered_v", "filtered"),
                ("sniff_frequency_hz", "rate"),
                ("sniff_frequency_unmasked_hz", "rate_unmasked"),
                ("sniff_frequency_instantaneous_hz", "rate_instantaneous"),
                ("sniff_frequency_instantaneous_unmasked_hz",
                 "rate_instantaneous_unmasked"),
                ("snr", "quality"),
            ):
                resp.create_dataset(name, data=respiration[key], compression="gzip")
            resp.attrs["filter_band_hz"] = np.asarray((1.0, 50.0))
            resp.attrs["filter_order"] = 2
            resp.attrs["snr_threshold"] = respiration["quality_threshold"]
            resp.attrs["sniff_smoothing_breaths"] = respiration["smooth_breaths"]
            resp.attrs["session_fraction_good"] = respiration["session_fraction_good"]
            resp.attrs["n_breaths"] = respiration["n_breaths"]

        if pupil is not None:
            eye = handle.create_group("pupil")
            mapping = {
                "diameter_px": "diameter_masked",
                "diameter_unmasked_px": "diameter",
                "equivalent_diameter_px": "equivalent_diameter_masked",
                "equivalent_diameter_unmasked_px": "equivalent_diameter",
                "area_px2": "area",
                "axis_ratio": "axis_ratio",
                "fit_inlier_fraction": "inlier_fraction",
                "fit_residual_px": "residual",
                "blink": "blink",
                "clipped": "clipped",
                "illumination_active": "illumination_active",
                "imaging_active": "imaging_active",
                "alignment_valid": "alignment_valid",
                "nearest_frame_error_s": "nearest_frame_error_s",
                "center_x_px": "x",
                "center_y_px": "y",
                "major_radius_px": "major",
                "minor_radius_px": "minor",
                "theta_rad": "theta",
            }
            for name, key in mapping.items():
                eye.create_dataset(name, data=pupil[key], compression="gzip")
            eye["diameter_px"].attrs["description"] = (
                "Full fitted major-axis length, with quality masking."
            )
            eye["diameter_unmasked_px"].attrs["description"] = (
                "Full fitted major-axis length before quality masking."
            )
            eye["equivalent_diameter_px"].attrs["description"] = (
                "2 * sqrt(major * minor), with quality masking."
            )
            for name in (
                "diameter_px", "diameter_unmasked_px",
                "equivalent_diameter_px", "equivalent_diameter_unmasked_px",
            ):
                eye[name].attrs["units"] = "pixels"
            eye.attrs["config_json"] = json.dumps(asdict(pupil["config"]), default=str)
            eye.attrs["camera_imaging_fraction"] = pupil["camera_imaging_fraction"]
            eye.attrs["camera_alignment_json"] = json.dumps(
                pupil["camera_alignment"], default=str
            )
    return path


def _mean_band(ax, time, values, mask, color, label):
    selected = np.asarray(values)[mask]
    if not len(selected):
        return
    finite = np.isfinite(selected)
    n = np.sum(finite, axis=0)
    mean = np.divide(
        np.nansum(selected, axis=0), n,
        out=np.full(selected.shape[1:], np.nan, dtype=float), where=n > 0,
    )
    squared = np.nansum((selected - mean) ** 2, axis=0)
    variance = np.divide(
        squared, n, out=np.full_like(mean, np.nan), where=n > 0,
    )
    sem = np.sqrt(variance) / np.sqrt(np.maximum(n, 1))
    ax.plot(time, mean, color=color, lw=1.5, label=label)
    ax.fill_between(time, mean - 1.96 * sem, mean + 1.96 * sem,
                    color=color, alpha=.2, lw=0)


def _median_iqr_traces(ax, time, values, mask, color, label):
    """Fine individual traces beneath a median and interquartile band."""
    selected = np.asarray(values)[mask]
    if not len(selected):
        return
    for trace in selected:
        ax.plot(time, trace, color=color, lw=.45, alpha=.16, zorder=1)
    median = np.nanmedian(selected, axis=0)
    q25, q75 = np.nanpercentile(selected, (25, 75), axis=0)
    ax.fill_between(time, q25, q75, color=color, alpha=.20, lw=0, zorder=2)
    ax.plot(time, median, color=color, lw=1.7,
            label=f"{label} median", zorder=3)


def _representative_trial(values, mask):
    """Trial whose median finite value is closest to its state's median."""
    indices = np.flatnonzero(mask)
    if not len(indices):
        return None
    scores = np.nanmedian(np.asarray(values)[indices], axis=1)
    finite = np.isfinite(scores)
    if not finite.any():
        return int(indices[0])
    target = np.nanmedian(scores)
    local = np.flatnonzero(finite)[np.argmin(np.abs(scores[finite] - target))]
    return int(indices[local])


def combined_qc_figure(path, *, metadata, respiration, pupil, treadmill):
    """One-page overview of all available auxiliary modalities."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time = np.nanmedian(treadmill["time_from_odor_s"], axis=0)
    state_codes, levels = _state_codes(metadata["states"])
    colors = ("steelblue", "indianred", "darkgreen", "purple")
    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True,
                             constrained_layout=True)
    datasets = [
        (respiration["rate"] if respiration else None, "sniff frequency (Hz)"),
        (pupil["diameter_masked"] if pupil else None, "pupil diameter (px)"),
        (treadmill["velocity"], "treadmill velocity (cm/s)"),
    ]
    top = axes[0]
    if respiration is None:
        top.text(.5, .5, "not available", transform=top.transAxes,
                 ha="center", va="center")
    else:
        snr_axis = top.twinx()
        for code, level in enumerate(levels):
            index = _representative_trial(
                respiration["quality"], state_codes == code
            )
            if index is None:
                continue
            color = colors[code % len(colors)]
            acq_id = np.asarray(metadata["acq_ids"])[index]
            median_snr = np.nanmedian(respiration["quality"][index])
            top.plot(time, respiration["filtered"][index], color=color, lw=1.1,
                     label=(f"{level} example: acq {acq_id}, "
                            f"median SNR {median_snr:.2f}"))
            snr_axis.plot(time, respiration["quality"][index], color=color,
                          lw=.9, ls="--", alpha=.75)
        threshold = float(respiration["quality_threshold"])
        snr_axis.axhline(threshold, color="0.25", lw=.8, ls=":",
                         label=f"SNR cutoff {threshold:g}")
        snr_axis.set_ylabel("SNR (dashed)")
        snr_axis.set_yscale("log")
        handles, labels = top.get_legend_handles_labels()
        handles2, labels2 = snr_axis.get_legend_handles_labels()
        top.legend(handles + handles2, labels + labels2, frameon=False, ncol=3)
    top.set_ylabel("filtered respiration (V)")
    top.grid(alpha=.25)

    for ax, (values, ylabel) in zip(axes[1:], datasets):
        if values is None:
            ax.text(.5, .5, "not available", transform=ax.transAxes,
                    ha="center", va="center")
        else:
            for code, level in enumerate(levels):
                plot = (_median_iqr_traces
                        if ylabel.startswith("treadmill") else _mean_band)
                plot(ax, time, values, state_codes == code,
                     colors[code % len(colors)], level)
        ax.set_ylabel(ylabel); ax.grid(alpha=.25)
    odor_end = np.nanmedian(
        (np.asarray(metadata["odor_off_frames"]) -
         np.asarray(metadata["odor_on_frames"])) / metadata["frame_rate"]
    )
    for ax in axes:
        ax.axvspan(0, odor_end, color="goldenrod", alpha=.12)
    axes[1].legend(frameon=False, ncol=len(levels))
    axes[-1].set_xlabel("seconds from odor onset")
    fig.suptitle(f"auxiliary QC — {metadata['exp_name']}")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return Path(path)


def treadmill_figure(path, *, metadata, treadmill):
    """Individual treadmill traces with median and IQR, pre versus post."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    odors = np.asarray(metadata["odor_ids"])
    states, levels = _state_codes(metadata["states"])
    time = np.nanmedian(treadmill["time_from_odor_s"], axis=0)
    keys = np.unique(odors)
    n_col = 4
    n_row = int(np.ceil(len(keys) / n_col))
    fig, axes = plt.subplots(n_row, n_col, figsize=(16, 2.8*n_row),
                             sharex=True, sharey=True, squeeze=False)
    for index, odor in enumerate(keys):
        ax = axes.flat[index]
        for code, level in enumerate(levels):
            _median_iqr_traces(
                ax, time, treadmill["velocity"],
                (odors == odor) & (states == code),
                ("steelblue", "indianred")[code % 2], level,
            )
        ax.axhline(0, color="0.4", lw=.7)
        ax.set_title(f"odor {int(odor)}"); ax.grid(alpha=.25)
    for ax in axes.flat[len(keys):]:
        ax.axis("off")
    axes.flat[0].legend(frameon=False)
    for ax in axes[-1]: ax.set_xlabel("seconds from odor onset")
    for ax in axes[:, 0]: ax.set_ylabel("velocity (cm/s)")
    fig.suptitle(f"odor-averaged treadmill — {metadata['exp_name']}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return Path(path)


def process_auxiliary(
    session,
    *,
    video_paths=None,
    source_video_paths=None,
    source_sync_path=None,
    pupil_config=None,
    out_dir=None,
    workers=None,
    checkpoint_dir=None,
    pupil_resume=True,
    running_threshold_cm_s=RUNNING_THRESHOLD_CM_S,
):
    """Run all auxiliary modalities for one resolved session and publish outputs."""
    from .pupil import PupilConfig, extract_pupil, pupil_figure
    from .respiration import extract_respiration, find_behavior_sync, respiration_figure

    sync_path = (Path(session.sync_path) if session.sync_path is not None
                 else find_behavior_sync(session.exp_dir))
    source_sync_path = sync_path if source_sync_path is None else Path(source_sync_path)
    out_dir = Path(out_dir) if out_dir else session.output_dir / "aux"
    out_dir.mkdir(parents=True, exist_ok=True)
    states, state_levels = _state_codes(session.states)
    trial_ids = session.table.trial_id.to_numpy()
    metadata = {
        "exp_name": session.exp_name, "group_id": session.group_id,
        "mouse": session.exp_name.split("_")[1],
        "date": session.exp_name.split("_")[0],
        "trial_source": session.trial_source,
        "manipulation": session.manipulation,
        "frame_rate": session.frame_rate,
        "acq_ids": np.asarray(session.acq_ids), "trial_ids": trial_ids,
        "odor_ids": np.asarray(session.odor_ids), "states": np.asarray(session.states),
        "odor_on_frames": np.asarray(session.odor_on_frames),
        "odor_off_frames": np.asarray(session.odor_off_frames),
    }
    try:
        from tqdm.auto import tqdm
        stage_message = tqdm.write
    except ImportError:
        stage_message = print
    stage_message(f"group {session.group_id}: respiration")
    respiration = extract_respiration(
        sync_path, acq_ids=session.acq_ids, odor_ids=session.odor_ids,
        states=states, state_levels=state_levels, trial_ids=trial_ids,
        exp_name=session.exp_name, manipulation=session.manipulation, save=False,
    )
    stage_message(f"group {session.group_id}: treadmill")
    treadmill = extract_treadmill(
        sync_path, odor_on_frames=session.odor_on_frames,
        running_threshold_cm_s=running_threshold_cm_s,
    )
    if video_paths is None:
        try:
            video_paths = discover_pupil_videos(session.exp_dir)
        except FileNotFoundError:
            video_paths = None
    pupil = None
    if video_paths:
        stage_message(
            f"group {session.group_id}: pupil preflight, alignment, and fitting"
        )
        if pupil_config is None:
            pupil_config, tuning_path = saved_pupil_config(
                session.exp_dir, group_id=session.group_id, exp_name=session.exp_name
            )
        else:
            tuning_path = None
        pupil = extract_pupil(
            video_paths, sync_path, acq_ids=session.acq_ids,
            odor_ids=session.odor_ids, states=session.states,
            odor_on_frames=session.odor_on_frames,
            odor_off_frames=session.odor_off_frames,
            frame_rate=session.frame_rate, exp_name=session.exp_name,
            trial_ids=trial_ids, out_dir=out_dir,
            config=pupil_config, save=False,
            workers=workers, checkpoint_dir=checkpoint_dir,
            resume=pupil_resume,
        )
    stage_message(f"group {session.group_id}: writing auxiliary outputs and QC")
    source_video_paths = video_paths if source_video_paths is None else source_video_paths
    stem = f"group{session.group_id}_{session.exp_name}_auxiliary"
    outputs = {
        "h5": write_auxiliary(
            out_dir / f"{stem}.h5", metadata=metadata, respiration=respiration,
            pupil=pupil, treadmill=treadmill,
            sources={"sync": str(source_sync_path),
                     "pupil_videos": [] if source_video_paths is None
                                     else list(map(str, source_video_paths)),
                     "pupil_tuning": None if pupil is None or tuning_path is None
                                     else str(tuning_path)},
        ),
        "combined_figure": combined_qc_figure(
            out_dir / f"{stem}_qc.png", metadata=metadata,
            respiration=respiration, pupil=pupil, treadmill=treadmill,
        ),
        "respiration_figure": respiration_figure(
            respiration, out_dir / f"{stem}_respiration_odors.png"
        ),
        "treadmill_figure": treadmill_figure(
            out_dir / f"{stem}_treadmill_odors.png", metadata=metadata,
            treadmill=treadmill,
        ),
    }
    if pupil is not None:
        outputs["pupil_figure"] = pupil_figure(
            pupil, out_dir / f"{stem}_pupil_qc.png"
        )
    return {"outputs": outputs, "metadata": metadata,
            "respiration": respiration, "pupil": pupil, "treadmill": treadmill}
