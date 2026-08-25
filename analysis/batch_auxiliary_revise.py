"""Revise published auxiliary traces and QC without reopening pupil videos."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from analysis.batch_auxiliary import manifest_rows
from analysis.session.auxiliary_revision import revise_auxiliary
from analysis.session.respiration import QUALITY_THRESHOLD


def revision_item(row, imaging_root):
    import h5py

    exp_name = f"{row['date']}_{row['mouse']}_{row['exp']}"
    exp_dir = Path(imaging_root) / row["date"] / row["mouse"] / row["exp"]
    stem = f"group{row['group_id']}_{exp_name}_auxiliary"
    path = exp_dir / "processed" / "python" / "aux" / f"{stem}.h5"
    sync_path = None
    if path.is_file():
        try:
            with h5py.File(path, "r") as handle:
                sources = json.loads(str(handle.attrs.get("sources_json", "{}")))
            recorded = Path(sources.get("sync", ""))
            if recorded.is_file():
                sync_path = recorded
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
    if sync_path is None:
        sync_candidates = sorted((exp_dir / "sync").glob("*.h5"))
        sync_path = sync_candidates[0] if len(sync_candidates) == 1 else None
    return {
        "group_id": int(row["group_id"]),
        "exp_name": exp_name,
        "h5": str(path),
        "sync": None if sync_path is None else str(sync_path),
        "ready": path.is_file() and sync_path is not None,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=("Revise respiration detection, pupil fit masking, and "
                     "derivable QC from published auxiliary outputs.")
    )
    parser.add_argument(
        "--manifest", default="analysis/stage0/ketxyl_16odor_session_manifest.csv"
    )
    parser.add_argument("--imaging-root", default=os.environ.get(
        "ODYN_IMAGING_ROOT", "/Volumes/MossLab/ImagingData"
    ))
    parser.add_argument("--groups", nargs="*", type=int, default=[])
    parser.add_argument("--snr-threshold", type=float, default=QUALITY_THRESHOLD)
    parser.add_argument("--execute", action="store_true",
                        help="Publish revisions; otherwise print inventory only.")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)

    items = [revision_item(row, args.imaging_root)
             for row in manifest_rows(args.manifest, args.groups)]
    if args.execute:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(items, desc="Revise auxiliary", unit="session",
                            dynamic_ncols=True)
        except ImportError:
            iterator = items
        for item in iterator:
            if not item["ready"]:
                item["status"] = "not_ready"
                continue
            try:
                item["outputs"] = revise_auxiliary(
                    item["h5"], sync_path=item["sync"],
                    snr_threshold=args.snr_threshold,
                )
                item["status"] = "revised"
            except Exception as error:
                item["status"] = "failed"
                item["error"] = f"{type(error).__name__}: {error}"
    print(json.dumps(items, indent=2))
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(items, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
