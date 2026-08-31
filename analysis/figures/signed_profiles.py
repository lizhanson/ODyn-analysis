"""Per-cell signed response profiles across odors, and their state dependence.

Three questions, kept separate because they have different failure modes:

1. How often is one cell excited by one odor and suppressed by another?
2. Is there a population of cells that only ever suppress, distinct from the
   tail of a continuum?
3. Are those cells preferentially silenced by anesthesia?

Responder calls are not re-invented here. They come from
`population_metrics.temporal_feature_table`, which references every call to
that unit's own pre-odor excursions, so this analysis inherits exactly the
calls the breadth panels in Figures 2 and 3 are built from.

Two design points carry most of the weight.

**Cross-odor, not within-odor.** A cell that is excited and then suppressed by
a single odor is biphasic, not bidirectional across the panel. The primary
classification therefore requires the excitation and the suppression to come
from different odors; the within-odor count is reported alongside so the two
are never conflated.

**The null for question 2 must not be a straw man.** Cells differ in how many
odors they respond to at all, and odors differ in how excitatory they are.
Either alone manufactures apparent single-sign cells. The null here permutes
the sign of responsive cell-odor events *within each odor*, which holds both of
those fixed and destroys only the cell-level preference being tested.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

KEYS = ["group_id", "mouse", "line", "depth_class", "cohort", "compartment",
        "state"]
CLASSES = ("silent", "excited_only", "suppressed_only", "bidirectional")


def _classify(n_excited, n_suppressed):
    excited, suppressed = n_excited > 0, n_suppressed > 0
    return np.select(
        [~excited & ~suppressed, excited & ~suppressed, ~excited & suppressed],
        ["silent", "excited_only", "suppressed_only"], default="bidirectional")


def unit_sign_profiles(temporal) -> pd.DataFrame:
    """One row per unit and state: how many odors drove it each way.

    `cross_odor_bidirectional` demands that the excitation and the suppression
    come from different odors. The union of the two odor sets must therefore
    hold at least two odors, which rules out a unit whose only response is a
    single biphasic one.
    """
    if temporal.empty:
        return temporal
    real = temporal[~temporal.is_blank] if "is_blank" in temporal else temporal
    grouped = real.groupby([*KEYS, "unit_id"], dropna=False).agg(
        n_odor=("odor_id", "size"),
        n_excited=("excited", "sum"),
        n_suppressed=("suppressed", "sum"),
        n_biphasic=("biphasic", "sum"),
        median_positive_auc_z_s=("positive_auc_z_s", "median"),
        median_negative_auc_z_s=("negative_auc_z_s", "median"),
        max_positive_auc_z_s=("positive_auc_z_s", "max"),
        max_negative_auc_z_s=("negative_auc_z_s", "max"),
    ).reset_index()
    for column in ("n_excited", "n_suppressed", "n_biphasic"):
        grouped[column] = grouped[column].astype(int)
    grouped["response_class"] = _classify(grouped.n_excited, grouped.n_suppressed)
    distinct_odors = (grouped.n_excited + grouped.n_suppressed
                      - grouped.n_biphasic)
    grouped["cross_odor_bidirectional"] = (
        (grouped.n_excited > 0) & (grouped.n_suppressed > 0)
        & (distinct_odors >= 2))
    grouped["within_odor_biphasic"] = grouped.n_biphasic > 0
    grouped["excitation_breadth"] = grouped.n_excited / grouped.n_odor
    grouped["suppression_breadth"] = grouped.n_suppressed / grouped.n_odor
    total = (grouped.median_positive_auc_z_s + grouped.median_negative_auc_z_s)
    grouped["auc_sign_index"] = np.divide(
        grouped.median_positive_auc_z_s - grouped.median_negative_auc_z_s,
        total, out=np.full(len(grouped), np.nan), where=total > 0)
    return grouped


def class_fractions(profiles) -> pd.DataFrame:
    """Fraction of units in each class, per session and state."""
    if profiles.empty:
        return profiles
    counts = (profiles.groupby([*KEYS, "response_class"], dropna=False)
              .size().rename("n").reset_index())
    totals = counts.groupby(KEYS, dropna=False)["n"].sum().rename("n_unit")
    table = counts.merge(totals, on=KEYS)
    table["fraction"] = table.n / table.n_unit
    wide = table.pivot_table(index=KEYS, columns="response_class",
                             values="fraction", fill_value=0.).reset_index()
    for name in CLASSES:
        if name not in wide:
            wide[name] = 0.
    extra = profiles.groupby(KEYS, dropna=False).agg(
        n_unit=("unit_id", "size"),
        cross_odor_bidirectional=("cross_odor_bidirectional", "mean"),
        within_odor_biphasic=("within_odor_biphasic", "mean"),
        n_odor=("n_odor", "median")).reset_index()
    return wide.merge(extra, on=KEYS)


def mouse_then_cohort(table, columns, *, keys=("cohort", "state")):
    """Average sessions within mouse, then summarize mice within cohort.

    Sessions from one animal are not independent observations of a cohort, and
    the mice are few enough that one animal with many sessions would otherwise
    carry a panel on its own.
    """
    columns = [c for c in columns if c in table]
    keys = list(keys)
    per_mouse = table.groupby([*keys, "mouse"], dropna=False)[columns].mean().reset_index()
    summary = per_mouse.groupby(keys, dropna=False)[columns].agg(
        ["median", "mean", "std", "size"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    return per_mouse, summary.reset_index()


# --------------------------------------------------------------------------
# Question 2: is a single-sign cell more than a continuum tail?
# --------------------------------------------------------------------------

def sign_permutation_null(temporal, *, repeats=1000, seed=0):
    """Compare observed single-sign cell counts against a within-odor null.

    For each odor, the signs of that odor's responsive cell-odor events are
    permuted across the cells that responded to it. That holds two nuisance
    structures fixed:

    * how many odors each cell responds to at all, so a cell that simply
      responds rarely cannot become a "suppression-only" cell by accident;
    * each odor's own excitation-to-suppression composition, so a cell tuned to
      odors that happen to be broadly suppressive is not counted as evidence of
      a suppression-specialized cell type.

    What it destroys is exactly the claim under test: that the *sign* of a
    response is a property of the cell. An excess of suppression-only cells
    over this null is evidence for cell-level sign preference; agreement with
    the null means the observed counts follow from breadth and odor identity.
    """
    if temporal.empty:
        return pd.DataFrame()
    real = temporal[~temporal.is_blank] if "is_blank" in temporal else temporal
    rng = np.random.default_rng(seed)
    rows = []
    for key, session in real.groupby(KEYS, dropna=False):
        units = session.unit_id.to_numpy()
        order = {unit: index for index, unit in enumerate(pd.unique(units))}
        n_unit = len(order)
        index = np.array([order[unit] for unit in units])
        odor = session.odor_id.to_numpy()
        excited = session.excited.to_numpy(bool)
        suppressed = session.suppressed.to_numpy(bool)
        responsive = excited | suppressed
        if not responsive.any():
            continue
        observed = _single_sign_counts(index, excited, suppressed, n_unit)
        null = {name: np.empty(int(repeats), int) for name in observed}
        for repeat in range(int(repeats)):
            shuffled_e = excited.copy()
            shuffled_s = suppressed.copy()
            for level in np.unique(odor):
                rows_here = np.flatnonzero((odor == level) & responsive)
                if rows_here.size < 2:
                    continue
                permuted = rng.permutation(rows_here)
                shuffled_e[rows_here] = excited[permuted]
                shuffled_s[rows_here] = suppressed[permuted]
            counts = _single_sign_counts(index, shuffled_e, shuffled_s, n_unit)
            for name, value in counts.items():
                null[name][repeat] = value
        record = dict(zip(KEYS, key)) | {"n_unit": n_unit, "repeats": int(repeats)}
        for name, value in observed.items():
            draws = null[name]
            record[f"{name}_observed"] = int(value)
            record[f"{name}_null_mean"] = float(draws.mean())
            record[f"{name}_null_sd"] = float(draws.std(ddof=1))
            record[f"{name}_p_greater"] = float((np.sum(draws >= value) + 1)
                                                / (len(draws) + 1))
            record[f"{name}_excess"] = float(value - draws.mean())
        rows.append(record)
    return pd.DataFrame(rows)


def _single_sign_counts(index, excited, suppressed, n_unit):
    """How many units fall in each class, given per-event sign calls."""
    per_unit_e = np.zeros(n_unit, int)
    per_unit_s = np.zeros(n_unit, int)
    np.add.at(per_unit_e, index, excited.astype(int))
    np.add.at(per_unit_s, index, suppressed.astype(int))
    classes = _classify(per_unit_e, per_unit_s)
    return {name: int(np.sum(classes == name)) for name in CLASSES}


# --------------------------------------------------------------------------
# Question 3: what anesthesia does to each class
# --------------------------------------------------------------------------

def subset_trials(data, mask) -> dict:
    """A population dict restricted to a subset of trials."""
    mask = np.asarray(mask, bool)
    out = dict(data)
    out["z"] = data["z"][:, mask, :]
    for name in ("odor_id", "state", "trial_id"):
        out[name] = np.asarray(data[name])[mask]
    if "baseline_mean" in data:
        out["baseline_mean"] = np.asarray(data["baseline_mean"])[:, mask]
    return out


def split_awake_trials(data, *, seed=0, awake="pre"):
    """Split the awake trials of each odor in half, leaving other states whole.

    Classifying a cell and then measuring how much it changes on the very same
    trials guarantees regression to the mean: the cells selected as the most
    suppressive awake are partly selected for noise, and that noise is gone by
    the second measurement whatever anesthesia does. The halves let the class
    be defined on trials the state comparison never sees.
    """
    levels = list(data["state_levels"])
    if awake not in levels:
        raise ValueError(f"no {awake!r} state; found {levels}")
    code = levels.index(awake)
    rng = np.random.default_rng(seed)
    states = np.asarray(data["state"])
    odors = np.asarray(data["odor_id"])
    select = np.zeros(len(states), bool)
    for odor in np.unique(odors[states == code]):
        here = np.flatnonzero((states == code) & (odors == odor))
        if len(here) < 2:
            continue                      # too few repeats to split; drop both
        chosen = rng.permutation(here)[: len(here) // 2]
        select[chosen] = True
    classify = select
    measure = (~select) & (states == code)
    other = states != code
    return classify, measure | other


def paired_state_change(profiles, *, awake="pre", anesthetized="post"):
    """Join each unit's awake class to its response on both sides of the state.

    The class comes from the caller's classification pass; the measurements
    come from whatever pass built `profiles`. Keeping them separate is the
    point, so this function never classifies anything itself.
    """
    keys = [k for k in KEYS if k != "state"]
    wide = profiles.pivot_table(
        index=[*keys, "unit_id"], columns="state",
        values=["n_excited", "n_suppressed", "excitation_breadth",
                "suppression_breadth", "median_negative_auc_z_s",
                "median_positive_auc_z_s", "max_negative_auc_z_s",
                "max_positive_auc_z_s"])
    wide.columns = [f"{a}__{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    for state in (awake, anesthetized):
        if f"suppression_breadth__{state}" not in wide:
            raise ValueError(f"profiles have no {state!r} state to pair")
    wide["suppression_breadth_change"] = (
        wide[f"suppression_breadth__{anesthetized}"]
        - wide[f"suppression_breadth__{awake}"])
    wide["excitation_breadth_change"] = (
        wide[f"excitation_breadth__{anesthetized}"]
        - wide[f"excitation_breadth__{awake}"])
    wide["negative_auc_log2_change"] = _log2_ratio(
        wide[f"median_negative_auc_z_s__{anesthetized}"],
        wide[f"median_negative_auc_z_s__{awake}"])
    wide["positive_auc_log2_change"] = _log2_ratio(
        wide[f"median_positive_auc_z_s__{anesthetized}"],
        wide[f"median_positive_auc_z_s__{awake}"])
    return wide


def _log2_ratio(after, before):
    after = np.asarray(after, float)
    before = np.asarray(before, float)
    positive = before[np.isfinite(before) & (before > 0)]
    floor = float(np.median(positive)) * 1e-3 if positive.size else 1e-9
    return np.log2(np.maximum(after, floor) / np.maximum(before, floor))


def breadth_matched_change(paired, *, target="suppressed_only",
                           reference="bidirectional", awake="pre",
                           metric="suppression_breadth_change", bins=8,
                           minimum_per_cell=3):
    """Compare two awake classes at matched awake suppression breadth.

    A suppression-only cell is by construction among the most suppressed cells
    in its field, so it has the most suppression available to lose. Comparing
    its state change against the bidirectional population as a whole therefore
    confounds cell class with starting level, and a floor effect alone would
    reproduce the result. Stratifying on the awake breadth and reweighting the
    reference cells onto the target's own distribution removes that.

    Returns the raw difference and the matched difference side by side; if the
    matched difference collapses toward zero, starting level was the story.
    """
    column = f"suppression_breadth__{awake}"
    for name in (column, metric, "awake_class", "cohort"):
        if name not in paired:
            raise KeyError(f"paired table has no {name!r} column")
    rows = []
    for cohort, group in paired.groupby("cohort", dropna=False):
        selected = group[group.awake_class.isin([target, reference])]
        here = selected[np.isfinite(selected[column])
                        & np.isfinite(selected[metric])]
        target_cells = here[here.awake_class == target]
        reference_cells = here[here.awake_class == reference]
        if len(target_cells) < minimum_per_cell or len(reference_cells) < minimum_per_cell:
            continue
        edges = np.unique(np.quantile(here[column], np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:
            continue
        strata = np.digitize(here[column], edges[1:-1])
        here = here.assign(_stratum=strata)
        target_cells = here[here.awake_class == target]
        reference_cells = here[here.awake_class == reference]
        weights = target_cells._stratum.value_counts(normalize=True)
        reference_medians = reference_cells.groupby("_stratum")[metric].median()
        shared = weights.index.intersection(reference_medians.index)
        if not len(shared):
            continue
        # Reweight the reference onto the target's own breadth distribution;
        # strata with no reference cell cannot contribute and are renormalized.
        share = weights.loc[shared] / weights.loc[shared].sum()
        matched_reference = float((reference_medians.loc[shared] * share).sum())
        rows.append({
            "cohort": cohort, "metric": metric,
            "n_target": int(len(target_cells)),
            "n_reference": int(len(reference_cells)),
            "target_awake_breadth": float(target_cells[column].median()),
            "reference_awake_breadth": float(reference_cells[column].median()),
            "target_change": float(target_cells[metric].median()),
            "reference_change_raw": float(reference_cells[metric].median()),
            "reference_change_matched": matched_reference,
            "difference_raw": float(target_cells[metric].median()
                                    - reference_cells[metric].median()),
            "difference_matched": float(target_cells[metric].median()
                                        - matched_reference),
            "matched_strata": int(len(shared)),
            "target_coverage": float(weights.loc[shared].sum()),
        })
    return pd.DataFrame(rows)
