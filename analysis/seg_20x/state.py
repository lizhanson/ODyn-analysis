"""Headless ordered state machine for 20x segmentation and curation."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

from ..seg_10x.state import grow_seed
from .segmentation import (
    PROCESS_DEFAULTS,
    RIDGE_PARAMS,
    SOMA_DEFAULTS,
    detect_processes,
    detect_somas,
    process_foreground,
    ridge_image,
)

PHASE_SOMA_TUNE = "soma_tune"
PHASE_SOMA_CURATE = "soma_curate"
PHASE_PROCESS_TUNE = "process_tune"
PHASE_PROCESS_CURATE = "process_curate"
PHASE_GROUP = "group"
PHASES = (
    PHASE_SOMA_TUNE,
    PHASE_SOMA_CURATE,
    PHASE_PROCESS_TUNE,
    PHASE_PROCESS_CURATE,
    PHASE_GROUP,
)

BUNDLE_TYPE = "odyn_20x_mask_bundle"

# 1: curated masks only, so resuming had to re-run the detectors.
SCHEMA_VERSION = 2


def save_portable_state(state, path, *, config=None):
    """Save any live 20x state without relying on its bound method globals."""

    import json
    from pathlib import Path

    import h5py

    # Literals here are intentional. A recovery function must not depend on
    # globals that may themselves be absent from an autoreload-created globals
    # dictionary. Tests assert these match the public module constants.
    bundle_type = "odyn_20x_mask_bundle"
    schema_version = 2

    path = Path(path).with_suffix(".h5")
    path.parent.mkdir(parents=True, exist_ok=True)
    config = state.configuration() if config is None else config
    table = state.roi_table()
    arrays = {
        "masks/soma": state.curated_somas(),
        "masks/process": state.curated_processes(),
        "masks/soma_automatic": state.automatic_somas(),
        "masks/process_automatic": state.automatic_processes(),
        "images/structural": state.structural,
    }
    if state.correlation is not None:
        arrays["images/correlation"] = state.correlation

    with h5py.File(path, "w") as handle:
        handle.attrs["file_type"] = bundle_type
        handle.attrs["schema_version"] = schema_version
        handle.attrs["config_json"] = json.dumps(config)
        for name, data in arrays.items():
            handle.create_dataset(name, data=data, compression="gzip")
        rois = handle.create_group("rois")
        string_type = h5py.string_dtype("utf-8")
        for column in table:
            values = table[column]
            rois.create_dataset(
                column,
                data=values.to_numpy(),
                dtype=string_type if values.dtype == object else None,
            )
    return path


class Segmentation20xState:
    """All segmentation, curation, grouping, and persistence without GUI code."""

    def __init__(self, structural, correlation=None, *, soma_params=None, process_params=None):
        self.structural = np.asarray(structural, dtype=np.float32)
        self.correlation = None if correlation is None else np.asarray(correlation, dtype=np.float32)
        if self.structural.ndim != 2:
            raise ValueError(f"structural image must be 2D, got {self.structural.shape}")
        if self.correlation is not None and self.correlation.shape != self.structural.shape:
            raise ValueError("structural and correlation images have differing shapes")
        self.shape = self.structural.shape
        self.soma_params = {**SOMA_DEFAULTS, **(soma_params or {})}
        self.process_params = {**PROCESS_DEFAULTS, **(process_params or {})}
        self.phase = PHASE_SOMA_TUNE

        self.soma_deleted = set()
        self.soma_added_seeds = []
        self.soma_deleted_seeds = set()
        self.process_deleted = set()
        self.manual_skeletons = []
        self.deleted_manual_skeletons = set()
        self.selected = set()
        self.groups = {}

        self._automatic_somas = None
        self._soma_record = None
        self._ridge = None
        self._automatic_processes = None
        self._process_record = None

        # Replaying every hand-placed seed costs a watershed each, and the
        # group phase asks for the curated labels on every single click. Cache
        # the replay and drop it when an edit actually changes it.
        self._curated_somas = None
        self._curated_processes = None
        self._manual_soma_ids = {}
        self._manual_process_ids = {}

    # ---- persistence -----------------------------------------------------------

    @classmethod
    def load(cls, path):
        """Resume a saved portable HDF5 round."""
        path = Path(path)
        if path.suffix != ".h5":
            raise ValueError("20x segmentation state must be a portable .h5 bundle")
        return cls.load_portable(path)

    @classmethod
    def load_portable(cls, path):
        """Resume from a single portable 20x mask bundle."""
        # Resolve these inside the call.
        import json
        import h5py
        from pathlib import Path

        from .state import BUNDLE_TYPE as bundle_type

        path = Path(path)
        with h5py.File(path, "r") as handle:
            if handle.attrs.get("file_type", "") != bundle_type:
                raise ValueError(f"Not an ODyn 20x mask bundle: {path}")
            config = json.loads(handle.attrs["config_json"])
            images = {name: handle["images"][name][:] for name in handle["images"]}
            masks = {name: handle["masks"][name][:] for name in handle["masks"]}

        state = cls._from_config(images.get("structural"), images.get("correlation"), config)
        state._restore_masks(masks, config, source=path)
        return state

    @classmethod
    def _from_config(cls, structural, correlation, config):
        """A state carrying the saved parameters, edits, groups, and phase."""
        state = cls(
            structural, correlation,
            soma_params=config.get("soma_params"),
            process_params=config.get("process_params"),
        )
        edits = config.get("curation", {})
        state.soma_deleted = set(edits.get("soma_deleted", []))
        state.soma_added_seeds = [tuple(v) for v in edits.get("soma_added_seeds", [])]
        state.soma_deleted_seeds = set(edits.get("soma_deleted_seeds", []))
        state.process_deleted = set(edits.get("process_deleted", []))
        state.manual_skeletons = [
            [tuple(vertex) for vertex in vertices]
            for vertices in edits.get("manual_skeletons", [])
        ]
        state.deleted_manual_skeletons = set(edits.get("deleted_manual_skeletons", []))
        state.groups = {
            (kind, int(ident)): int(group_id)
            for key, group_id in config.get("groups", {}).items()
            for kind, ident in [key.split(":", 1)]
        }
        phase = config.get("summary", {}).get("phase", PHASE_GROUP)
        if phase not in PHASES:
            raise ValueError(f"Saved round has unknown phase {phase!r}.")
        state.phase = phase
        return state

    def _restore_masks(self, arrays, config, *, source):
        """Take the saved labels rather than detecting them again."""
        automatic = {
            "soma": arrays.get("soma_automatic"),
            "process": arrays.get("process_automatic"),
        }
        if automatic["soma"] is None or automatic["process"] is None:
            warnings.warn(
                f"{source} predates schema {SCHEMA_VERSION} and stores only curated "
                "masks, so its labels have to be detected again and may differ from "
                "the ones that were curated. Re-save it to pin them.",
                stacklevel=3,
            )
            return

        self._automatic_somas = np.asarray(automatic["soma"], np.int32)
        self._automatic_processes = np.asarray(automatic["process"], np.int32)

        for kind, expected in (("soma", arrays.get("soma")), ("process", arrays.get("process"))):
            if expected is None:
                continue
            replayed = self.curated_somas() if kind == "soma" else self.curated_processes()
            if not np.array_equal(replayed, np.asarray(expected, np.int32)):
                raise ValueError(
                    f"Replaying the saved edits on {source} does not reproduce its stored "
                    f"{kind} labels. The masks in the file are authoritative; this state "
                    "cannot be edited further without diverging from them."
                )

    def save(self, path):
        """Write the one bundle that holds masks, images, edits, and groups."""
        return self.export_portable(Path(path).with_suffix(".h5"))

    def export_portable(self, path, *, config=None):
        """Write one self-contained, cross-computer mask/curation bundle."""
        return save_portable_state(self, path, config=config)

    def roi_table(self):
        """One row per curated ROI, with its group where it has one."""
        import pandas as pd
        from skimage.measure import regionprops

        rows = []
        for kind, labels in (("soma", self.curated_somas()), ("process", self.curated_processes())):
            for region in regionprops(labels):
                ident = int(region.label)
                rows.append({
                    "roi_type": kind,
                    "source_roi_id": ident,
                    "roi_group_id": self.groups.get((kind, ident), -1),
                    "area_px": int(region.area),
                    "centroid_y": round(float(region.centroid[0]), 1),
                    "centroid_x": round(float(region.centroid[1]), 1),
                })
        return pd.DataFrame(rows, columns=[
            "roi_type", "source_roi_id", "roi_group_id",
            "area_px", "centroid_y", "centroid_x",
        ])

    # ---- parameters and phases -------------------------------------------------

    def set_soma_param(self, name, value):
        if self.phase != PHASE_SOMA_TUNE:
            raise RuntimeError("Soma parameters are frozen after soma tuning.")
        if name not in self.soma_params:
            raise KeyError(name)
        self.soma_params[name] = value
        self._automatic_somas = self._soma_record = None
        self._invalidate_somas()

    def set_process_param(self, name, value):
        if self.phase != PHASE_PROCESS_TUNE:
            raise RuntimeError("Process parameters are only live during process tuning.")
        if name not in self.process_params:
            raise KeyError(name)
        self.process_params[name] = value
        if name in RIDGE_PARAMS:
            self._ridge = None
        self._automatic_processes = self._process_record = None
        self._invalidate_processes()

    def _invalidate_somas(self):
        # Processes are detected against the curated somas and flooded around
        # them, so a soma edit invalidates both.
        self._curated_somas = None
        self._invalidate_processes()

    def _invalidate_processes(self):
        self._curated_processes = None

    def advance(self):
        index = PHASES.index(self.phase)
        if index == len(PHASES) - 1:
            return self.phase
        if self.phase == PHASE_SOMA_TUNE:
            self.automatic_somas()
        elif self.phase == PHASE_SOMA_CURATE:
            self.curated_somas()
            self._automatic_processes = self._process_record = None
            self._invalidate_processes()
        elif self.phase == PHASE_PROCESS_TUNE:
            self.automatic_processes()
        elif self.phase == PHASE_PROCESS_CURATE:
            self.curated_processes()
        self.phase = PHASES[index + 1]
        return self.phase

    def back(self, *, discard_downstream=False):
        index = PHASES.index(self.phase)
        if index == 0:
            return self.phase
        if not discard_downstream and self._has_downstream_edits(index - 1):
            raise RuntimeError("Going back discards downstream curation/grouping; confirm first.")
        target = index - 1
        self._discard_after(target)
        self.phase = PHASES[target]
        return self.phase

    def _has_downstream_edits(self, target):
        if target < 1 and (self.soma_deleted or self.soma_added_seeds):
            return True
        if target < 3 and (self.process_deleted or self.manual_skeletons):
            return True
        return bool(self.groups)

    def _discard_after(self, target):
        if target < 1:
            self.soma_deleted.clear(); self.soma_added_seeds.clear(); self.soma_deleted_seeds.clear()
        if target < 2:
            self._automatic_processes = self._process_record = None
        if target < 3:
            self.process_deleted.clear(); self.manual_skeletons.clear(); self.deleted_manual_skeletons.clear()
        self.selected.clear(); self.groups.clear()
        self._invalidate_somas()

    # ---- soma ------------------------------------------------------------------

    def automatic_somas(self):
        if self._automatic_somas is None:
            self._automatic_somas, self._soma_record = detect_somas(
                self.structural, self.soma_params
            )
        return self._automatic_somas

    def curated_somas(self):
        if self._curated_somas is None:
            self._curated_somas = self._replay_somas()
            self._curated_somas.flags.writeable = False
        return self._curated_somas

    def _replay_somas(self):
        labels = self.automatic_somas().copy()
        for ident in self.soma_deleted:
            labels[labels == ident] = 0
        occupied = labels > 0
        next_id = int(self.automatic_somas().max()) + 1
        self._manual_soma_ids = {}
        yy, xx = np.ogrid[: self.shape[0], : self.shape[1]]
        for index, (y, x) in enumerate(self.soma_added_seeds):
            if index in self.soma_deleted_seeds:
                continue
            region = grow_seed(
                self.structural, (y, x), unclaimed=~occupied,
                min_diameter_px=float(self.soma_params["min_diameter_px"]) / 2,
                max_diameter_px=float(self.soma_params["max_diameter_px"]),
                threshold_pctl=float(self.soma_params["growth_threshold_pctl"]),
            )
            region &= (yy-y)**2 + (xx-x)**2 <= float(self.soma_params["max_radius_px"])**2
            if region.sum() < 5:
                continue
            labels[region] = next_id
            occupied |= region
            self._manual_soma_ids[next_id] = index
            next_id += 1
        return labels

    def add_soma(self, y, x):
        if self.phase != PHASE_SOMA_CURATE:
            raise RuntimeError("Somas can only be added during soma curation.")
        self.soma_added_seeds.append((int(y), int(x)))
        self._invalidate_somas()

    def delete_soma_at(self, y, x):
        if self.phase != PHASE_SOMA_CURATE:
            raise RuntimeError("Somas can only be deleted during soma curation.")
        labels = self.curated_somas()
        ident = int(labels[y, x])
        if ident <= 0:
            return None
        if ident in self._manual_soma_ids:
            self.soma_deleted_seeds.add(self._manual_soma_ids[ident])
        else:
            self.soma_deleted.add(ident)
        self._invalidate_somas()
        return ident

    # ---- process ---------------------------------------------------------------

    def ridge(self):
        if self._ridge is None:
            self._ridge = ridge_image(self.structural, self.process_params)
        return self._ridge

    def automatic_processes(self):
        if self._automatic_processes is None:
            self._automatic_processes, self._process_record = detect_processes(
                self.structural, self.curated_somas(), self.process_params, ridge=self.ridge()
            )
        return self._automatic_processes

    def process_preview(self):
        foreground, floor = process_foreground(
            self.ridge(), self.curated_somas(), self.process_params
        )
        return foreground, floor

    def curated_processes(self):
        if self._curated_processes is None:
            self._curated_processes = self._replay_processes()
            self._curated_processes.flags.writeable = False
        return self._curated_processes

    def _replay_processes(self):
        from scipy.ndimage import binary_dilation
        from skimage.draw import line
        from skimage.filters import threshold_local
        from skimage.segmentation import watershed

        labels = self.automatic_processes().copy()
        for ident in self.process_deleted:
            labels[labels == ident] = 0
        occupied = (self.curated_somas() > 0) | (labels > 0)
        next_id = int(self.automatic_processes().max()) + 1
        self._manual_process_ids = {}
        for index, vertices in enumerate(self.manual_skeletons):
            if index in self.deleted_manual_skeletons or len(vertices) < 2:
                continue
            marker = np.zeros(self.shape, bool)
            for (y0,x0),(y1,x1) in zip(vertices[:-1], vertices[1:]):
                rr,cc = line(y0,x0,y1,x1); marker[rr,cc] = True
            marker &= ~occupied
            if not marker.any():
                continue
            # A drawn line is a local watershed seed, not permission to consume
            # the whole surrounding ridge field. Restrict it to a narrow
            # corridor and require pixels to pass a local adaptive threshold.
            corridor = binary_dilation(
                marker, iterations=int(self.process_params["manual_ridge_corridor_px"])
            )
            block = max(3, int(self.process_params["manual_ridge_adaptive_block_px"]))
            block += block % 2 == 0
            local = threshold_local(
                self.ridge(), block, method="gaussian",
                offset=float(self.process_params["manual_ridge_adaptive_offset"]),
            )
            allowed = corridor & np.isfinite(self.ridge()) & (self.ridge() > local) & ~occupied
            allowed |= marker
            region = watershed(-self.ridge(), markers=marker.astype(np.int32), mask=allowed) == 1
            if region.sum() < 5:
                region = marker
            labels[region] = next_id
            occupied |= region
            self._manual_process_ids[next_id] = index
            next_id += 1
        return labels

    def delete_process_at(self, y, x):
        if self.phase != PHASE_PROCESS_CURATE:
            raise RuntimeError("Processes can only be deleted during process curation.")
        labels = self.curated_processes()
        ident = int(labels[y,x])
        if ident <= 0:
            return None
        if ident in self._manual_process_ids:
            self.deleted_manual_skeletons.add(self._manual_process_ids[ident])
        else:
            self.process_deleted.add(ident)
        self._invalidate_processes()
        return ident

    def add_skeleton(self, vertices):
        if self.phase != PHASE_PROCESS_CURATE:
            raise RuntimeError("Skeletons can only be added during process curation.")
        if len(vertices) >= 2:
            self.manual_skeletons.append([(int(y),int(x)) for y,x in vertices])
            self._invalidate_processes()

    # ---- grouping --------------------------------------------------------------

    def roi_at(self, y, x):
        somas = self.curated_somas(); processes = self.curated_processes()
        if somas[y,x] > 0:
            return "soma", int(somas[y,x])
        if processes[y,x] > 0:
            return "process", int(processes[y,x])
        return None

    def toggle_selection(self, y, x):
        if self.phase != PHASE_GROUP:
            raise RuntimeError("ROI grouping is only available in the group phase.")
        roi = self.roi_at(y,x)
        if roi is None:
            return None
        self.selected.symmetric_difference_update({roi})
        return roi

    def assign_group(self, group_id):
        if self.phase != PHASE_GROUP:
            raise RuntimeError("ROI grouping is only available in the group phase.")
        gid = int(group_id)
        for roi in self.selected:
            self.groups[roi] = gid
        count = len(self.selected); self.selected.clear()
        return count

    def next_group_id(self):
        return max(self.groups.values(), default=0) + 1

    def autogroup(self, traces, *, um_per_px, params=None, keep_manual=False):
        """Replace the grouping with one derived from proximity and correlation."""
        from .grouping import group_rois

        if self.phase != PHASE_GROUP:
            raise RuntimeError("ROI grouping is only available in the group phase.")
        groups, diagnostics = group_rois(
            self.curated_somas(), self.curated_processes(), traces,
            um_per_px=um_per_px, params=params,
        )
        manual = dict(self.groups) if keep_manual else {}
        self.groups = {**groups, **manual}
        self.selected.clear()
        return diagnostics

    # ---- output ----------------------------------------------------------------

    def summary(self):
        somas, processes = self.curated_somas(), self.curated_processes()
        return {
            "phase": self.phase,
            "n_somas": len(np.unique(somas)) - 1,
            "n_processes": len(np.unique(processes)) - 1,
            "soma_deleted": len(self.soma_deleted),
            "soma_added": len(self.soma_added_seeds) - len(self.soma_deleted_seeds),
            "process_deleted": len(self.process_deleted),
            "process_added": len(self.manual_skeletons) - len(self.deleted_manual_skeletons),
            "n_groups": len(set(self.groups.values())),
            "n_grouped_rois": len(self.groups),
        }

    def configuration(self):
        """Serializable parameters, edits, grouping, and workflow position."""
        return {
            "soma_params": self.soma_params,
            "process_params": self.process_params,
            "curation": {
                "soma_deleted": sorted(self.soma_deleted),
                "soma_added_seeds": self.soma_added_seeds,
                "soma_deleted_seeds": sorted(self.soma_deleted_seeds),
                "process_deleted": sorted(self.process_deleted),
                "manual_skeletons": self.manual_skeletons,
                "deleted_manual_skeletons": sorted(self.deleted_manual_skeletons),
            },
            "groups": {
                f"{kind}:{ident}": gid
                for (kind, ident), gid in self.groups.items()
            },
            "summary": self.summary(),
        }
