"""Rebuild 20x QC from extracted rounds without re-extracting movie traces."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .batch_20x import experiment_dir, manifest_rows


def inventory(row, imaging_root):
    from .session.store import find_rounds

    output = experiment_dir(row, imaging_root) / "processed" / "python"
    rounds = find_rounds(output)
    return {
        "group_id": int(row["group_id"]),
        "exp_dir": str(experiment_dir(row, imaging_root)),
        "round": str(rounds[-1]) if rounds else None,
        "ready": bool(rounds),
        "status": "ready" if rounds else "no_extracted_round",
    }


def recover_groups(round_path):
    """Recover reviewed assignments from the canonical prior grouped product."""
    import h5py

    round_path = Path(round_path)
    grouped = round_path.with_name(round_path.stem + "_20x_grouped.h5")
    if not grouped.exists():
        raise FileNotFoundError(
            f"No prior reviewed grouping product beside {round_path.name}: "
            f"expected {grouped.name}"
        )
    with h5py.File(round_path, "r") as source:
        parameters = json.loads(source.attrs["parameters_json"])
        manifest = parameters["segmentation"]["roi_manifest"]
    lookup = {
        int(row["roi_id"]): (str(row["roi_type"]), int(row["source_roi_id"]))
        for row in manifest
    }
    groups = {}
    with h5py.File(grouped, "r") as saved:
        source_round = Path(str(saved.attrs.get("source_round", ""))).name
        if source_round and source_round != round_path.name:
            raise ValueError(
                f"{grouped.name} belongs to {source_round}, not {round_path.name}"
            )
        members = saved["groups/member_roi_ids"]
        for group_id, member_ids in enumerate(members, start=1):
            for roi_id in member_ids:
                roi_id = int(roi_id)
                if roi_id not in lookup:
                    raise ValueError(
                        f"ROI {roi_id} in {grouped.name} is absent from the round"
                    )
                groups[lookup[roi_id]] = group_id
    if set(groups) != set(lookup.values()):
        raise ValueError(f"{grouped.name} does not cover every ROI in the round")
    return groups, grouped


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
    parser.add_argument("--output-suffix", default="_pertrial_median")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    items = [inventory(row, args.imaging_root)
             for row in manifest_rows(args.manifest, args.groups)]
    if args.execute:
        from .seg_20x.qc import qc_20x
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(items, desc="20x QC rebuild", unit="session",
                            dynamic_ncols=True)
        except ImportError:
            iterator = items
        for item in iterator:
            if not item["ready"]:
                continue
            try:
                item["outputs"] = qc_20x(
                    item["round"], groups=recover_groups(item["round"])[0],
                    baseline_sd_mode=args.baseline_sd_mode,
                    temporal_reducer=args.temporal_reducer,
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
