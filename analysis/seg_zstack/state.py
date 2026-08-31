"""Multi-plane wrapper around the tested 20x segmentation state machine."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..seg_20x.state import PHASES, Segmentation20xState

BUNDLE_TYPE = "odyn_structural_zstack_bundle"
SCHEMA_VERSION = 1


def resolve_depth_reference(depth_zero_plane, center_depth_um, n_planes):
    """Return ``(reference_plane, reference_depth_um)`` for depth assignment."""
    if center_depth_um is not None:
        return (int(n_planes) - 1) / 2, float(center_depth_um)
    zero = float(depth_zero_plane)
    if not 0 <= zero <= int(n_planes) - 1:
        raise ValueError(f"depth_zero_plane {zero:g} is outside this {n_planes}-plane stack")
    return zero, 0.0


class StructuralZStackState:
    def __init__(self, structural, *, metadata=None, soma_params=None, process_params=None):
        structural = np.asarray(structural, np.float32)
        if structural.ndim != 3:
            raise ValueError(f"structural stack must be Z,Y,X; got {structural.shape}")
        self.structural = structural
        self.metadata = dict(metadata or {})
        self.planes = [
            Segmentation20xState(im, soma_params=soma_params, process_params=process_params)
            for im in structural
        ]
        self.plane = 0

    @property
    def current(self):
        return self.planes[self.plane]

    @property
    def phase(self):
        phases = {p.phase for p in self.planes}
        if len(phases) != 1:
            raise RuntimeError(f"plane phases diverged: {sorted(phases)}")
        return next(iter(phases))

    def set_plane(self, plane):
        plane = int(plane)
        if not 0 <= plane < len(self.planes):
            raise IndexError(plane)
        self.plane = plane

    def set_soma_param(self, name, value):
        for state in self.planes:
            state.set_soma_param(name, value)

    def set_process_param(self, name, value):
        for state in self.planes:
            state.set_process_param(name, value)

    def advance(self, *, progress=True):
        # Materialize every plane before changing phase, so all planes receive
        # automatic segmentation even if the user has not visited them.
        planes = self.planes
        if progress:
            try:
                from tqdm.auto import tqdm
                planes = tqdm(planes, total=len(planes), desc="Segmenting z-stack", unit="plane")
            except ImportError:
                pass
        for state in planes:
            state.advance()
        return self.phase

    def back(self, *, discard_downstream=False):
        target = PHASES.index(self.phase) - 1
        if target < 0:
            return self.phase
        if not discard_downstream and any(p._has_downstream_edits(target) for p in self.planes):
            raise RuntimeError("Going back discards downstream curation; confirm first.")
        for state in self.planes:
            state.back(discard_downstream=True)
        return self.phase

    def masks(self):
        return (
            np.stack([p.curated_somas() for p in self.planes]),
            np.stack([p.curated_processes() for p in self.planes]),
        )

    def roi_table(self, *, um_per_px=None, min_soma_diameter_um=None,
                  depth_zero_plane=0, center_depth_um=None, depth_direction=1):
        import pandas as pd

        z_step = self.metadata.get("z_step_um")
        reference_plane, reference_depth_um = resolve_depth_reference(
            depth_zero_plane, center_depth_um, len(self.planes)
        )
        rows = []
        for z, state in enumerate(self.planes):
            table = state.roi_table()
            table.insert(0, "plane", z)
            table.insert(
                1, "depth_um",
                np.nan if z_step is None else reference_depth_um +
                (z - reference_plane) * float(z_step) * int(depth_direction),
            )
            if um_per_px is not None:
                table["area_um2"] = table.area_px * float(um_per_px) ** 2
                table["equivalent_diameter_um"] = 2 * np.sqrt(table.area_um2 / np.pi)
            if min_soma_diameter_um is not None:
                if um_per_px is None:
                    raise ValueError("um_per_px is required for a micron-based soma diameter cutoff")
                table = table[
                    (table.roi_type != "soma") |
                    (table.equivalent_diameter_um >= float(min_soma_diameter_um))
                ].copy()
            rows.append(table)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    def depth_summary(self, *, um_per_px=None, min_soma_diameter_um=None,
                      depth_zero_plane=0, center_depth_um=None, depth_direction=1):
        import pandas as pd

        reference_plane, reference_depth_um = resolve_depth_reference(
            depth_zero_plane, center_depth_um, len(self.planes)
        )
        table = self.roi_table(
            um_per_px=um_per_px, min_soma_diameter_um=min_soma_diameter_um,
            depth_zero_plane=depth_zero_plane,
            center_depth_um=center_depth_um,
            depth_direction=depth_direction,
        )
        rows = []
        image_area_px = int(np.prod(self.structural.shape[-2:]))
        for z in range(len(self.planes)):
            plane = table[table.plane == z]
            somas = plane[plane.roi_type == "soma"]
            processes = plane[plane.roi_type == "process"]
            row = {
                "plane": z,
                "depth_um": plane.depth_um.iloc[0] if len(plane) else (
                    np.nan if self.metadata.get("z_step_um") is None else
                    reference_depth_um + (z-reference_plane)*self.metadata["z_step_um"]*depth_direction
                ),
                "n_soma_profiles": len(somas),
                "n_process_rois": len(processes),
                "mean_soma_area_px": somas.area_px.mean(),
                "median_soma_area_px": somas.area_px.median(),
                "soma_profile_density_per_1e6_px2": len(somas) / image_area_px * 1e6,
            }
            if um_per_px is not None:
                area_mm2 = image_area_px * float(um_per_px) ** 2 / 1e6
                row.update({
                    "fov_area_mm2": area_mm2,
                    "soma_profile_density_per_mm2": len(somas) / area_mm2,
                    "mean_soma_area_um2": somas.area_um2.mean(),
                    "median_soma_area_um2": somas.area_um2.median(),
                })
            rows.append(row)
        return pd.DataFrame(rows)

    def save(self, path, *, um_per_px=None, min_soma_diameter_um=None,
             depth_zero_plane=0, center_depth_um=None, depth_direction=1):
        import h5py

        path = Path(path).with_suffix(".h5")
        path.parent.mkdir(parents=True, exist_ok=True)
        reference_plane, reference_depth_um = resolve_depth_reference(
            depth_zero_plane, center_depth_um, len(self.planes)
        )
        soma, process = self.masks()
        configs = [p.configuration() for p in self.planes]
        attrs = {
            "file_type": BUNDLE_TYPE,
            "schema_version": SCHEMA_VERSION,
            "metadata_json": json.dumps(self.metadata),
            "plane_configs_json": json.dumps(configs),
            "um_per_px": np.nan if um_per_px is None else float(um_per_px),
            "min_soma_diameter_um": (
                np.nan if min_soma_diameter_um is None else float(min_soma_diameter_um)
            ),
            "depth_reference_plane": float(reference_plane),
            "depth_reference_um": float(reference_depth_um),
            "depth_assignment": "center_depth" if center_depth_um is not None else "zero_plane",
            "depth_direction": int(depth_direction),
        }
        with h5py.File(path, "w") as f:
            for key, value in attrs.items():
                f.attrs[key] = value
            f.create_dataset("images/structural", data=self.structural, compression="gzip")
            f.create_dataset("masks/soma", data=soma, compression="gzip")
            f.create_dataset("masks/process", data=process, compression="gzip")
            f.create_dataset(
                "masks/soma_automatic",
                data=np.stack([p.automatic_somas() for p in self.planes]), compression="gzip",
            )
            f.create_dataset(
                "masks/process_automatic",
                data=np.stack([p.automatic_processes() for p in self.planes]), compression="gzip",
            )
        self.roi_table(um_per_px=um_per_px,
                       min_soma_diameter_um=min_soma_diameter_um,
                       depth_zero_plane=depth_zero_plane,
                       center_depth_um=center_depth_um,
                       depth_direction=depth_direction).to_csv(path.with_name(path.stem+"_rois.csv"), index=False)
        self.depth_summary(um_per_px=um_per_px,
                           min_soma_diameter_um=min_soma_diameter_um,
                           depth_zero_plane=depth_zero_plane,
                           center_depth_um=center_depth_um,
                           depth_direction=depth_direction).to_csv(path.with_name(path.stem+"_depth_summary.csv"), index=False)
        return path

    @classmethod
    def load(cls, path):
        import h5py

        path = Path(path)
        with h5py.File(path, "r") as f:
            if f.attrs.get("file_type") != BUNDLE_TYPE:
                raise ValueError(f"not a structural z-stack bundle: {path}")
            structural = f["images/structural"][:]
            metadata = json.loads(f.attrs["metadata_json"])
            configs = json.loads(f.attrs["plane_configs_json"])
            automatic_soma = f["masks/soma_automatic"][:]
            automatic_process = f["masks/process_automatic"][:]
        out = cls(structural, metadata=metadata)
        out.planes = []
        for z, config in enumerate(configs):
            state = Segmentation20xState._from_config(structural[z], config)
            state._automatic_somas = automatic_soma[z]
            state._automatic_processes = automatic_process[z]
            out.planes.append(state)
        return out
