"""Rebuild final joined 10x QC without repeating component extraction."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .batch_10x import experiment_dir, manifest_rows, published_bundles


def inventory(row, imaging_root):
    from .session.store import find_rounds

    output = experiment_dir(row, imaging_root) / "processed" / "python"
    rounds = find_rounds(output)
    bundles = published_bundles(row, imaging_root)
    groups = output / f"group{int(row['group_id'])}_10x_roi_groups.json"
    ready = bool(rounds and bundles and groups.exists())
    missing = []
    if not rounds:
        missing.append("extracted round")
    if not bundles:
        missing.append("portable mask bundle")
    if not groups.exists():
        missing.append("reviewed joins JSON")
    return {
        "group_id": int(row["group_id"]), "output_dir": str(output),
        "round": str(rounds[-1]) if rounds else None,
        "bundle": str(bundles[-1]) if bundles else None,
        "groups": str(groups), "ready": ready,
        "status": "ready" if ready else "missing: " + ", ".join(missing),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", default="analysis/stage0/ketxyl_16odor_session_manifest.csv")
    parser.add_argument("--imaging-root", default=os.environ.get(
        "ODYN_IMAGING_ROOT", "/Volumes/MossLab/ImagingData"))
    parser.add_argument("--groups", nargs="*", type=int, default=[])
    parser.add_argument("--baseline-sd-mode",
                        choices=("pre_block_pooled", "per_trial"),
                        default="per_trial")
    parser.add_argument("--temporal-reducer", choices=("mean", "median"),
                        default="median")
    parser.add_argument("--temporal-sort", choices=("latency", "mean_response"),
                        default="mean_response")
    parser.add_argument("--z-limits", nargs=2, type=float, default=(-5, 15))
    parser.add_argument("--output-suffix", default="_pertrial_median")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    items = [inventory(row, args.imaging_root)
             for row in manifest_rows(args.manifest, args.groups)]
    if args.execute:
        from .seg_10x.grouped_qc import finalize_grouped_10x
        from .session.masks import load_mask_bundle
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(items, desc="10x QC rebuild", unit="session",
                            dynamic_ncols=True)
        except ImportError:
            iterator = items
        for item in iterator:
            if not item["ready"]:
                continue
            try:
                bundle = load_mask_bundle(item["bundle"])
                item["outputs"] = finalize_grouped_10x(
                    item["round"], item["groups"], reference=bundle["reference"],
                    baseline_sd_mode=args.baseline_sd_mode,
                    temporal_reducer=args.temporal_reducer,
                    temporal_sort=args.temporal_sort,
                    z_limits=tuple(args.z_limits),
                    output_suffix=args.output_suffix,
                )
                item["status"] = "complete"
            except Exception as error:
                item["status"] = "failed"
                item["error"] = f"{type(error).__name__}: {error}"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(items, indent=2) + "\n")
    print(json.dumps(items, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
