"""Inventory or run consolidated auxiliary processing from the session manifest."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from analysis.session.auxiliary import (
    discover_pupil_videos,
    process_auxiliary,
    resolve_auxiliary,
)
from analysis.session.devshim import LocalGroup
from analysis.session.pupil import stage_pupil_videos, stage_sync_file


def manifest_rows(path, groups=()):
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected = {int(value) for value in groups}
    return [row for row in rows if not selected or int(row["group_id"]) in selected]


def inventory(row, imaging_root):
    from analysis.session.respiration import find_behavior_sync

    exp_dir = (Path(imaging_root) / row["date"] / row["mouse"] / row["exp"])
    try:
        sync_path = find_behavior_sync(exp_dir)
    except FileNotFoundError:
        sync_path = None
    try:
        videos = discover_pupil_videos(exp_dir)
    except FileNotFoundError:
        videos = []
    tuning = sorted((exp_dir / "processed" / "python" / "aux").glob(
        f"group{row['group_id']}_*pupil_tuning.json"
    ))
    completion = completed_outputs(exp_dir, row, expect_pupil=bool(videos))
    return {
        "group_id": int(row["group_id"]), "exp_dir": str(exp_dir),
        "sync": None if sync_path is None else str(sync_path),
        "pupil_videos": len(videos),
        "pupil_tuning_files": len(tuning),
        "complete": completion is not None,
        "outputs": completion,
        "ready": sync_path is not None and (
            len(videos) == 0 or (len(videos) == 2 and len(tuning) == 1)
        ),
    }


def completed_outputs(exp_dir, row, *, expect_pupil):
    """Return validated published outputs, or None for an incomplete group."""
    import h5py

    exp_name = f"{row['date']}_{row['mouse']}_{row['exp']}"
    stem = f"group{row['group_id']}_{exp_name}_auxiliary"
    aux = Path(exp_dir) / "processed" / "python" / "aux"
    paths = {
        "h5": aux / f"{stem}.h5",
        "combined_figure": aux / f"{stem}_qc.png",
        "respiration_figure": aux / f"{stem}_respiration_odors.png",
        "treadmill_figure": aux / f"{stem}_treadmill_odors.png",
    }
    if expect_pupil:
        paths["pupil_figure"] = aux / f"{stem}_pupil_qc.png"
    if any(not path.is_file() or path.stat().st_size == 0 for path in paths.values()):
        return None
    try:
        with h5py.File(paths["h5"], "r") as handle:
            if int(handle.attrs.get("group_id", -1)) != int(row["group_id"]):
                return None
            n_trial = int(handle.attrs["n_trial"])
            n_frame = int(handle.attrs["n_frame"])
            required = ("trials/acq_id", "treadmill/velocity",
                        "respiration/filtered_v")
            if any(name not in handle for name in required):
                return None
            if handle["treadmill/velocity"].shape != (n_trial, n_frame):
                return None
            if expect_pupil and (
                "pupil/diameter_px" not in handle or
                handle["pupil/diameter_px"].shape != (n_trial, n_frame)
            ):
                return None
    except (OSError, KeyError, TypeError, ValueError):
        return None
    return {name: str(path) for name, path in paths.items()}


def cleanup_staged_inputs(checkpoint_dir):
    """Remove only locally staged source copies, preserving checkpoints."""
    import shutil

    checkpoint_dir = Path(checkpoint_dir)
    removed = []
    for name in ("staged_videos", "staged_sync"):
        path = checkpoint_dir / name
        if path.is_dir():
            shutil.rmtree(path)
            removed.append(str(path))
    return removed


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", default="analysis/stage0/ketxyl_16odor_session_manifest.csv"
    )
    parser.add_argument("--imaging-root", default=os.environ.get(
        "ODYN_IMAGING_ROOT", "/Volumes/MossLab/ImagingData"))
    parser.add_argument("--scratch-root", default=os.environ.get(
        "ODYN_SCRATCH_ROOT", str(Path.home() / "odyn_scratch")))
    parser.add_argument("--groups", nargs="*", type=int, default=[])
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--execute", action="store_true",
                        help="Process ready groups; otherwise only print inventory.")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate groups with validated completed outputs.")
    parser.add_argument("--no-stage-videos", action="store_true",
                        help="Read videos from shared storage instead of local scratch.")
    parser.add_argument("--keep-staged-inputs", action="store_true",
                        help="Retain local sync/video copies after successful publication.")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)

    rows = manifest_rows(args.manifest, args.groups)
    status = [inventory(row, args.imaging_root) for row in rows]
    if args.execute:
        try:
            from tqdm.auto import tqdm
            sessions = tqdm(status, desc="Auxiliary sessions", unit="session",
                            dynamic_ncols=True)
        except ImportError:
            sessions = status
        scratch = Path(args.scratch_root)
        group = LocalGroup(
            Path(args.imaging_root) / ".odyn" / "odyn.db", args.imaging_root,
            snapshot_to=scratch / "odyn_snapshot.db", max_age_s=1800,
        )
        for item in sessions:
            checkpoint_dir = scratch / "auxiliary" / f"group{item['group_id']}"
            if hasattr(sessions, "set_postfix_str"):
                sessions.set_postfix_str(f"group {item['group_id']}")
            if not item["ready"]:
                item["status"] = "not_ready"
                continue
            if item["complete"] and not args.force:
                item["status"] = "already_complete"
                if not args.keep_staged_inputs:
                    item["removed_staged_inputs"] = cleanup_staged_inputs(
                        checkpoint_dir
                    )
                continue
            try:
                source_sync = Path(item["sync"])
                staged_sync = stage_sync_file(source_sync, checkpoint_dir / "staged_sync")
                session = resolve_auxiliary(
                    group, group_id=item["group_id"], sync_path=staged_sync
                )
                try:
                    source_videos = discover_pupil_videos(session.exp_dir)
                except FileNotFoundError:
                    source_videos = []
                video_paths = source_videos
                staged_dir = checkpoint_dir / "staged_videos"
                if source_videos and not args.no_stage_videos:
                    video_paths = stage_pupil_videos(
                        source_videos, staged_dir
                    )
                report = process_auxiliary(
                    session, video_paths=video_paths,
                    source_video_paths=source_videos, source_sync_path=source_sync,
                    workers=args.workers,
                    checkpoint_dir=checkpoint_dir,
                    pupil_resume=not args.force,
                )
                item["status"] = "complete"
                item["outputs"] = {key: str(value)
                                   for key, value in report["outputs"].items()}
                if not args.keep_staged_inputs:
                    try:
                        item["removed_staged_inputs"] = cleanup_staged_inputs(
                            checkpoint_dir
                        )
                    except OSError as cleanup_error:
                        item["cleanup_error"] = (
                            f"{type(cleanup_error).__name__}: {cleanup_error}"
                        )
            except Exception as error:
                item["status"] = "failed"
                item["error"] = f"{type(error).__name__}: {error}"
    print(json.dumps(status, indent=2))
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(status, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
