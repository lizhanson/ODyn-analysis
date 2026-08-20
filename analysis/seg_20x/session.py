"""Resolve approved 20x motion-corrected acquisitions without requiring trials."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ApprovedGroupInputs:
    group_id: int
    exp_ids: tuple[int, ...]
    exp_names: tuple[str, ...]
    paths: tuple[Path, ...]
    acq_ids: tuple[int, ...]
    frame_rate: float
    shape: tuple[int, int]
    n_frames: int
    um_per_px: float
    main_folder: Path

    def summary(self) -> dict:
        return {
            "group_id": self.group_id,
            "exp_ids": list(self.exp_ids),
            "exp_names": list(self.exp_names),
            "n_approved_acquisitions": len(self.paths),
            "frame_rate": round(self.frame_rate, 4),
            "shape_px": list(self.shape),
            "n_frames": self.n_frames,
            "um_per_px": round(self.um_per_px, 4),
            "approved_only": True,
        }


def resolve_approved_group(group, group_id: int, *, mcor_source: str | None = None):
    """
    Resolve only database-registered, approved mcor files for a group.

    Unlike ``session.resolve_group``, this does not require odor/trial rows.
    That is necessary for spontaneous or otherwise acquisition-only 20x
    recordings such as group 198. There is deliberately no directory fallback:
    if approval is absent from ``mcor_files``, the acquisition is not eligible.
    """

    mapping = group.group_experiments
    members = mapping[mapping.group_id == int(group_id)]
    if members.empty:
        raise ValueError(f"No group {group_id} in the database snapshot.")

    exp_ids = tuple(int(v) for v in members.exp_id)
    experiments = group.experiments[group.experiments.exp_id.isin(exp_ids)].copy()
    experiments = experiments.sort_values("exp_start", na_position="last")
    if experiments.empty:
        raise ValueError(f"Group {group_id} has no experiment rows.")

    shapes = set(zip(experiments.height_px.astype(int), experiments.width_px.astype(int)))
    if len(shapes) != 1:
        raise ValueError(f"Group {group_id} spans differing frame shapes: {sorted(shapes)}")

    acquisitions = group.acquisitions[group.acquisitions.exp_id.isin(exp_ids)][
        ["acq_id", "exp_id"]
    ]
    mcor = group.mcor_files.merge(acquisitions, on="acq_id", how="inner")
    n_registered = len(mcor)
    if mcor_source is not None:
        mcor = mcor[mcor.source == mcor_source]
    n_matching_source = len(mcor)
    approved = mcor.approved.fillna(0).astype(int).eq(1)
    mcor = mcor[approved].sort_values("acq_id")
    if mcor.empty:
        snapshot = getattr(group, "snapshot", None) or {}
        snapshot_detail = ""
        if snapshot:
            snapshot_detail = (
                f" Snapshot method={snapshot.get('method', '?')}, "
                f"age_s={snapshot.get('age_s', '?')}, "
                f"source_changed={snapshot.get('source_changed', False)}."
            )
        raise ValueError(
            f"Group {group_id} has no approved mcor files in this database "
            f"snapshot ({n_registered} registered for the group, "
            f"{n_matching_source} matching source, 0 approved)."
            f"{snapshot_detail} Refresh the snapshot after approving "
            "acquisitions in odyn; unapproved files are never used."
        )

    main = Path(group.main_folder)
    paths = tuple(main / str(p).replace("\\", "/") for p in mcor.mcor_path)
    first = experiments.iloc[0]
    rates = experiments.frame_rate.astype(float)
    if rates.max() - rates.min() > 1e-6:
        raise ValueError(f"Group {group_id} spans differing frame rates.")

    return ApprovedGroupInputs(
        group_id=int(group_id),
        exp_ids=tuple(int(v) for v in experiments.exp_id),
        exp_names=tuple(str(v) for v in experiments.exp_name),
        paths=paths,
        acq_ids=tuple(int(v) for v in mcor.acq_id),
        frame_rate=float(first.frame_rate),
        shape=(int(first.height_px), int(first.width_px)),
        n_frames=int(first.frame_count),
        um_per_px=float(first.width_um) / float(first.width_px),
        main_folder=main,
    )
