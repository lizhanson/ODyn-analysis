"""Inventory or batch-extract traces from published 10x mask bundles."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


def manifest_rows(path, groups=()):
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected = {int(value) for value in groups}
    return [row for row in rows
            if row.get("objective", "").strip().lower() == "10x"
            and (not selected or int(row["group_id"]) in selected)]


def experiment_dir(row, imaging_root):
    return Path(imaging_root) / row["date"] / row["mouse"] / row["exp"]


def published_bundles(row, imaging_root, *, output_dir=None):
    output = (experiment_dir(row, imaging_root) / "processed" / "python"
              if output_dir is None else Path(output_dir))
    return sorted(output.glob(
        f"group{int(row['group_id'])}_*_10x_masks_processed_*.h5"))


def inventory(row, imaging_root, *, output_dir=None,
              baseline_sd_mode="pre_block_pooled"):
    from analysis.session.finalize import mask_hash, verify
    from analysis.session.masks import load_mask_bundle

    manifest_exp_dir = experiment_dir(row, imaging_root)
    output = (manifest_exp_dir / "processed" / "python"
              if output_dir is None else Path(output_dir))
    bundles = published_bundles(row, imaging_root, output_dir=output)
    item = {
        "group_id": int(row["group_id"]),
        "manifest_exp_dir": str(manifest_exp_dir),
        "output_dir": str(output),
        "bundle_count": len(bundles),
        "bundle": str(bundles[-1]) if bundles else None,
        "ready": False, "current": False,
    }
    if not bundles:
        item["status"] = "no_published_bundle"
        item["output_dir_exists"] = output.is_dir()
        item["nearby_h5"] = (
            [path.name for path in sorted(output.glob(f"group{item['group_id']}_*.h5"))]
            if output.is_dir() else []
        )
        return item
    try:
        bundle = load_mask_bundle(bundles[-1])
        if bundle["reference"] is None:
            raise ValueError("Portable bundle has no reference image")
        item["n_rois"] = int(bundle["labels"].max())
        item["mask_hash"] = mask_hash(bundle["labels"])
        existing = verify(output)
        item["existing"] = existing
        item["current"] = (
            existing.get("status") == "ok"
            and existing.get("mask_hash") == item["mask_hash"]
            and existing.get("baseline_sd_mode") == baseline_sd_mode)
        item["ready"] = True
        item["status"] = "already_complete" if item["current"] else "ready"
    except Exception as error:
        item["status"] = "invalid_bundle"
        item["error"] = f"{type(error).__name__}: {error}"
    return item


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", default="analysis/stage0/ketxyl_16odor_session_manifest.csv")
    parser.add_argument("--imaging-root", default=os.environ.get(
        "ODYN_IMAGING_ROOT", "/Volumes/MossLab/ImagingData"))
    parser.add_argument("--scratch-root", default=os.environ.get(
        "ODYN_SCRATCH_ROOT", str(Path.home() / "odyn_scratch")))
    parser.add_argument("--groups", nargs="*", type=int, default=[])
    parser.add_argument("--manipulation", default="")
    parser.add_argument("--baseline-sd-mode",
                        choices=("pre_block_pooled", "per_trial"),
                        default="pre_block_pooled")
    parser.add_argument("--checkpoint-every", type=int, default=16)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    from analysis.session.devshim import LocalGroup
    from analysis.session.resolve import resolve_group

    rows = manifest_rows(args.manifest, args.groups)
    scratch = Path(args.scratch_root)
    group = LocalGroup(
        Path(args.imaging_root) / ".odyn" / "odyn.db", args.imaging_root,
        snapshot_to=scratch / "odyn_snapshot.db", max_age_s=1800)
    status = []
    sessions = {}
    for row in rows:
        group_id = int(row["group_id"])
        try:
            session = resolve_group(
                group, group_id=group_id, manipulation=args.manipulation,
                approved_only=True)
            sessions[group_id] = session
            status.append(inventory(
                row, args.imaging_root, output_dir=session.output_dir,
                baseline_sd_mode=args.baseline_sd_mode))
        except Exception as error:
            status.append({
                "group_id": group_id, "ready": False, "current": False,
                "status": "session_resolution_failed",
                "error": f"{type(error).__name__}: {error}",
            })

    if args.execute:
        from analysis.session.finalize import finalize_session
        from analysis.session.masks import load_mask_bundle

        try:
            from tqdm.auto import tqdm
            items = tqdm(status, desc="10x trace extraction", unit="session",
                         dynamic_ncols=True)
        except ImportError:
            items = status
        for item in items:
            if hasattr(items, "set_postfix_str"):
                items.set_postfix_str(f"group {item['group_id']}")
            if not item["ready"] or (item["current"] and not args.force):
                continue
            try:
                bundle = load_mask_bundle(item["bundle"])
                config = bundle["config"]
                session = sessions[item["group_id"]]
                result = finalize_session(
                    session, bundle["labels"],
                    per_group_masks=bundle["per_group"],
                    images=bundle["reference"],
                    segmentation_params=config.get("segmentation", {}),
                    merge_params=config.get("merge", {}),
                    curation=config.get("curation"),
                    full_acquisition=True,
                    checkpoint_every=args.checkpoint_every,
                    scratch_dir=scratch, detrend=True,
                    baseline_sd_mode=args.baseline_sd_mode)
                item.update(status="complete", output=result["path"],
                            mask_png=result["mask_png"])
            except Exception as error:
                item["status"] = "failed"
                item["error"] = f"{type(error).__name__}: {error}"

    print(json.dumps(status, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(status, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
