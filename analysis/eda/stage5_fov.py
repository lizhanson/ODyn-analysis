"""Does ratio discriminability follow what the field of view actually samples?

Every mixture component is also a single in the 16-odor panel, so each
reciprocal pair has a matched control inside the same session: how far apart
the two components themselves are in that field of view.  If a field does not
contain the glomeruli or cells that separate acetophenone from limonene, it
cannot separate their two ratios either, and pair specificity plus
session-to-session variability would both follow from sampling rather than
from anything about mixture coding.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .common import mixture_components

# Reciprocal mixture pair -> the two component odors they are both built from.
PAIR_COMPONENTS = {"17/18": (4, 10), "31/32": (22, 30), "39/40": (3, 12)}
KEYS = ["group_id", "mouse", "line", "cohort", "population", "state"]


def component_discriminability(geometry):
    """Crossnobis between the two components, from the identity comparisons."""
    identity = geometry[geometry.comparison == "identity"].copy()
    identity["cpair"] = identity.apply(
        lambda r: tuple(sorted((int(r.odor_a), int(r.odor_b)))), axis=1)
    lookup = {pair: tuple(sorted(components))
              for pair, components in PAIR_COMPONENTS.items()}
    rows = []
    for pair, components in lookup.items():
        selected = identity[identity.cpair == components]
        for _, row in selected.iterrows():
            rows.append({k: row[k] for k in KEYS}
                        | {"feature_set": row.feature_set, "pair": pair,
                           "component_crossnobis": row.crossnobis,
                           "component_cosine": row.cosine})
    return pd.DataFrame(rows)


def component_drive(features):
    """How hard the components drive this field: responsive fraction and mass."""
    rows = []
    for pair, components in PAIR_COMPONENTS.items():
        selected = features[features.odor_id.isin(components)]
        grouped = selected.groupby(KEYS, observed=True).agg(
            component_excited=("excited", "mean"),
            component_suppressed=("suppressed", "mean"),
            component_any=("excited", lambda s: np.nan),
            component_mass=("excitation_area", "mean"),
            component_abs_response=("raw_mean_sustained",
                                    lambda s: float(np.nanmean(np.abs(s)))),
            n_unit_odor=("excited", "size"),
        ).reset_index()
        any_response = selected.assign(
            any=selected.excited | selected.suppressed).groupby(
                KEYS, observed=True)["any"].mean().reset_index(name="any_rate")
        grouped = grouped.drop(columns=["component_any"]).merge(
            any_response, on=KEYS)
        grouped["pair"] = pair
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-set", default="odor")
    args = parser.parse_args(argv)
    out = args.output_dir

    geometry = pd.read_csv(out / "stage3_geometry.csv.gz")
    geometry["pair"] = geometry.odor_a.astype(str)+"/"+geometry.odor_b.astype(str)
    ratio = geometry[(geometry.comparison == "ratio")
                     & (geometry.feature_set == args.feature_set)][
        KEYS + ["pair", "crossnobis", "cosine", "n_a", "n_b"]].rename(
            columns={"crossnobis": "ratio_crossnobis",
                     "cosine": "ratio_cosine"})

    components = component_discriminability(
        geometry[geometry.feature_set == args.feature_set])

    from .stage2_summarise import load_features
    features = load_features(out / "features")
    drive = component_drive(features)

    joined = (ratio.merge(components.drop(columns=["feature_set"]),
                          on=KEYS + ["pair"], how="left")
                   .merge(drive, on=KEYS + ["pair"], how="left"))
    joined.to_csv(out / "stage5_fov.csv", index=False)
    print(f"joined {len(joined)} session x pair rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
