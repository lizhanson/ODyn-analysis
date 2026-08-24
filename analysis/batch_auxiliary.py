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


def manifest_rows(path, groups=()):
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected = {int(value) for value in groups}
    return [row for row in rows if not selected or int(row["group_id"]) in selected]


def inventory(row, imaging_root):
    exp_dir = (Path(imaging_root) / row["date"] / row["mouse"] / row["exp"])
    candidates = sorted((exp_dir / "sync").glob("*.h5"))
    candidates += sorted(exp_dir.parent.glob(f"*_{row['exp']}_*.h5"))
    sync_path = candidates[0] if candidates else None
    try:
        videos = discover_pupil_videos(exp_dir)
    except FileNotFoundError:
        videos = []
    tuning = sorted((exp_dir / "processed" / "python" / "aux").glob(
        f"group{row['group_id']}_*pupil_tuning.json"
    ))
    return {
        "group_id": int(row["group_id"]), "exp_dir": str(exp_dir),
        "sync": None if sync_path is None else str(sync_path),
        "pupil_videos": len(videos),
        "pupil_tuning_files": len(tuning),
        "ready": sync_path is not None and (
            len(videos) == 0 or (len(videos) == 2 and len(tuning) == 1)
        ),
    }


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
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)

    rows = manifest_rows(args.manifest, args.groups)
    status = [inventory(row, args.imaging_root) for row in rows]
    if args.execute:
        scratch = Path(args.scratch_root)
        group = LocalGroup(
            Path(args.imaging_root) / ".odyn" / "odyn.db", args.imaging_root,
            snapshot_to=scratch / "odyn_snapshot.db", max_age_s=1800,
        )
        for item in status:
            if not item["ready"]:
                item["status"] = "not_ready"
                continue
            try:
                session = resolve_auxiliary(group, group_id=item["group_id"])
                report = process_auxiliary(
                    session, workers=args.workers,
                    checkpoint_dir=scratch / "auxiliary" / f"group{item['group_id']}",
                )
                item["status"] = "complete"
                item["outputs"] = {key: str(value)
                                   for key, value in report["outputs"].items()}
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
