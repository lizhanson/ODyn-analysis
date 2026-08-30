"""Trial-count eligibility and descriptive outlier flags for geometry tables."""

from __future__ import annotations

import numpy as np
import pandas as pd


def audit_geometry(table, *, minimum_trials=3, robust_z_cutoff=3.5):
    """Add auditable eligibility and within-stratum robust-outlier columns.

    Outlier flags are prompts for source/QC inspection, never automatic
    exclusions. Eligibility is determined only by the predeclared repeat floor.
    """
    output = table.copy()
    output["min_trials"] = output[["n_a", "n_b"]].min(axis=1)
    output["trial_imbalance"] = (output["n_a"] - output["n_b"]).abs()
    output["eligible_primary"] = output["min_trials"] >= int(minimum_trials)
    strata = [name for name in ("line", "depth_class", "compartment", "state", "pair")
              if name in output]
    for metric in ("integrated_crossnobis", "spatiotemporal_crossnobis",
                   "integrated_cosine"):
        robust = pd.Series(np.nan, index=output.index, dtype=float)
        for _, group in output.groupby(strata, dropna=False):
            median = group[metric].median()
            mad = (group[metric] - median).abs().median()
            if np.isfinite(mad) and mad > 0:
                robust.loc[group.index] = .6745 * (group[metric] - median) / mad
        output[f"{metric}_robust_z"] = robust
        output[f"{metric}_outlier_flag"] = robust.abs() > robust_z_cutoff
    return output
