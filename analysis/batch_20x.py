"""Inventory or batch-finalize published 20x mask bundles into trace rounds."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np


def manifest_rows(path, groups=()):
    """Return selected 20x rows from the session manifest."""
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected = {int(value) for value in groups}
    return [row for row in rows
            if row.get("objective", "").strip().lower() == "20x"
            and (not selected or int(row["group_id"]) in selected)]


def experiment_dir(row, imaging_root):
    return Path(imaging_root) / row["date"] / row["mouse"] / row["exp"]


def published_bundles(row, imaging_root):
    output = experiment_dir(row, imaging_root) / "processed" / "python"
    return sorted(output.glob(
        f"group{int(row['group_id'])}_*_20x_masks_processed_*.h5"))


def load_mask_bundle(path):
    """Load and validate the portable inputs needed by finalization."""
    import h5py

    path = Path(path)
    with h5py.File(path, "r") as saved:
        required = ("masks/soma", "masks/process", "images/structural")
        missing = [name for name in required if name not in saved]
        if "config_json" not in saved.attrs:
            missing.append("config_json")
        if missing:
            raise ValueError(f"Portable mask bundle is missing: {', '.join(missing)}")
        soma = saved["masks/soma"][:]
        process = saved["masks/process"][:]
        structural = saved["images/structural"][:]
        config = json.loads(saved.attrs["config_json"])
    if soma.shape != process.shape or soma.shape != structural.shape:
        raise ValueError(
            f"Bundle arrays differ in shape: soma={soma.shape}, "
            f"process={process.shape}, structural={structural.shape}")

    combined = np.zeros_like(soma, dtype=np.int32)
    soma_mask = np.zeros_like(combined)
    process_mask = np.zeros_like(combined)
    manifest = []
    next_roi = 1
    configured_groups = config.get("groups", {})
    for roi_type, source_labels, typed_mask in (
        ("soma", soma, soma_mask), ("process", process, process_mask),
    ):
        for source_id in np.unique(source_labels[source_labels > 0]):
            pixels = source_labels == source_id
            combined[pixels] = next_roi
            typed_mask[pixels] = next_roi
            manifest.append({
                "roi_id": next_roi, "roi_type": roi_type,
                "source_roi_id": int(source_id),
                "roi_group_id": configured_groups.get(
                    f"{roi_type}:{int(source_id)}"),
            })
            next_roi += 1
    return {
        "labels": combined,
        "soma_labels": soma,
        "process_labels": process,
        "per_group_masks": {"soma": soma_mask, "process": process_mask},
        "structural": structural, "config": config,
        "roi_manifest": manifest,
    }


def inventory(row, imaging_root, *, baseline_sd_mode="pre_block_pooled"):
    from analysis.session.finalize import mask_hash, verify

    bundles = published_bundles(row, imaging_root)
    item = {
        "group_id": int(row["group_id"]),
        "exp_dir": str(experiment_dir(row, imaging_root)),
        "bundle_count": len(bundles),
        "bundle": str(bundles[-1]) if bundles else None,
        "ready": False, "current": False,
    }
    if not bundles:
        item["status"] = "no_published_bundle"
        return item
    try:
        bundle = load_mask_bundle(bundles[-1])
        item["n_rois"] = int(bundle["labels"].max())
        item["mask_hash"] = mask_hash(bundle["labels"])
        item["phase"] = bundle["config"].get("summary", {}).get("phase")
        output = experiment_dir(row, imaging_root) / "processed" / "python"
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
    parser.add_argument("--manipulation", default="ketxyl")
    parser.add_argument(
        "--baseline-sd-mode", choices=("pre_block_pooled", "per_trial"),
        default="pre_block_pooled")
    parser.add_argument("--checkpoint-every", type=int, default=16)
    parser.add_argument("--execute", action="store_true",
                        help="Extract ready groups; otherwise print inventory only.")
    parser.add_argument("--force", action="store_true",
                        help="Re-extract even when the current mask is complete.")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)

    rows = manifest_rows(args.manifest, args.groups)
    status = [inventory(row, args.imaging_root,
                        baseline_sd_mode=args.baseline_sd_mode) for row in rows]
    if args.execute:
        from analysis.session.devshim import LocalGroup
        from analysis.session.finalize import finalize_session
        from analysis.session.resolve import resolve_group

        try:
            from tqdm.auto import tqdm
            items = tqdm(status, desc="20x trace extraction", unit="session",
                         dynamic_ncols=True)
        except ImportError:
            items = status
        scratch = Path(args.scratch_root)
        group = LocalGroup(
            Path(args.imaging_root) / ".odyn" / "odyn.db", args.imaging_root,
            snapshot_to=scratch / "odyn_snapshot.db", max_age_s=1800)
        for item in items:
            if hasattr(items, "set_postfix_str"):
                items.set_postfix_str(f"group {item['group_id']}")
            if not item["ready"]:
                continue
            if item["current"] and not args.force:
                item["status"] = "already_complete"
                continue
            try:
                bundle = load_mask_bundle(item["bundle"])
                session = resolve_group(
                    group, group_id=item["group_id"],
                    manipulation=args.manipulation, approved_only=True)
                config = bundle["config"]
                result = finalize_session(
                    session, bundle["labels"],
                    per_group_masks=bundle["per_group_masks"],
                    images=bundle["structural"],
                    segmentation_params={
                        "workflow": "20x_soma_process",
                        "soma": config.get("soma_params", {}),
                        "process": config.get("process_params", {}),
                        "roi_manifest": bundle["roi_manifest"],
                    },
                    curation=config.get("curation", {}),
                    full_acquisition=True,
                    checkpoint_every=args.checkpoint_every,
                    scratch_dir=scratch, detrend=True,
                    baseline_sd_mode=args.baseline_sd_mode)
                item["status"] = "complete"
                item["output"] = result["path"]
                item["mask_png"] = result["mask_png"]
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
