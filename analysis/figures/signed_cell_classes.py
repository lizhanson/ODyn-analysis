"""Cell-level signed odor-response classes for 20x soma data.

Classes use independent positive and negative baseline-excursion calls across
the odor panel.  ``both_different_odors`` is deliberately stricter than a
biphasic response to a single odor: it requires at least one excited odor and
at least one *different* suppressed odor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


KEYS = ["group_id", "mouse", "line", "depth_class", "cohort",
        "compartment", "state", "unit_id"]


def cell_signed_classes(temporal, *, minimum_trials=4):
    """Classify each cell within each state from odor-specific signed calls."""
    real = temporal[(~temporal.is_blank) &
                    (temporal.n_trials >= int(minimum_trials))].copy()
    rows = []
    for key, group in real.groupby(KEYS, dropna=False):
        excited = set(group.loc[group.excited, "odor_id"].astype(int))
        suppressed = set(group.loc[group.suppressed, "odor_id"].astype(int))
        excitation_any = bool(excited)
        suppression_any = bool(suppressed)
        both_any = excitation_any and suppression_any
        both_different = any(a != b for a in excited for b in suppressed)
        if both_any:
            signed_class = "both"
        elif excitation_any:
            signed_class = "excitation only"
        elif suppression_any:
            signed_class = "suppression only"
        else:
            signed_class = "neither"
        rows.append(dict(zip(KEYS, key)) | {
            "n_eligible_odors": int(group.odor_id.nunique()),
            "n_excited_odors": len(excited), "n_suppressed_odors": len(suppressed),
            "excitation_any": excitation_any, "suppression_any": suppression_any,
            "both_any": both_any, "both_different_odors": both_different,
            "signed_class": signed_class,
        })
    return pd.DataFrame(rows)


def class_composition(classes):
    """One fraction per session/state for hierarchical summaries."""
    keys = ["group_id", "mouse", "line", "depth_class", "cohort",
            "compartment", "state"]
    rows = []
    for key, group in classes.groupby(keys, dropna=False):
        for label in ("neither", "excitation only", "suppression only", "both"):
            rows.append(dict(zip(keys, key)) | {
                "measure": "signed class", "value_label": label,
                "fraction": float(np.mean(group.signed_class == label)),
                "n_cells": len(group),
            })
        for label in ("both_any", "both_different_odors"):
            rows.append(dict(zip(keys, key)) | {
                "measure": "cross-odor sign", "value_label": label,
                "fraction": float(group[label].mean()), "n_cells": len(group),
            })
    return pd.DataFrame(rows)


def class_state_transitions(classes):
    """Within-cell awake-to-post transition fractions, summarized by session."""
    id_keys = ["group_id", "mouse", "line", "depth_class", "cohort",
               "compartment", "unit_id"]
    wide = classes.pivot(index=id_keys, columns="state", values="signed_class").dropna()
    rows = []
    session_keys = id_keys[:-1]
    labels = ("neither", "excitation only", "suppression only", "both")
    for key, group in wide.groupby(level=session_keys, dropna=False):
        for awake in labels:
            denominator = int(np.sum(group.pre == awake))
            for post in labels:
                rows.append(dict(zip(session_keys, key)) | {
                    "awake_class": awake, "post_class": post,
                    "n_awake_cells": denominator,
                    "fraction": (float(np.mean(group.loc[group.pre == awake, "post"] == post))
                                 if denominator else np.nan),
                })
    return pd.DataFrame(rows)


def awake_class_f0_change(classes, tonic):
    """Does an awake response class predict tonic F0 decline under ket/xyl?"""
    id_keys = ["group_id", "mouse", "line", "depth_class", "cohort",
               "compartment", "unit_id"]
    awake = classes[classes.state == "pre"][id_keys + ["signed_class"]]
    f0 = tonic.pivot(index=id_keys, columns="state", values="baseline_f")
    f0 = f0.dropna().reset_index()
    f0["f0_log2_post_pre"] = np.log2(f0.post / f0.pre)
    return awake.merge(f0[id_keys + ["f0_log2_post_pre"]], on=id_keys,
                       how="inner", validate="one_to_one")
