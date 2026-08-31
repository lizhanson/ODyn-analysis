"""Superseded session-level outputs, retained to reproduce earlier tables.

These metrics read a signed four-second mean and reference responder calls to
mineral oil. The exploratory pass showed both are unsound for this dataset: a
unit excited then suppressed averages to nothing over four seconds, and mineral
oil evokes a real delivery response whose size varies with state and scale, so
a blank-referenced cutoff shifts between conditions for reasons unrelated to
odor coding. New work should use `population_metrics.temporal_feature_table`
and `breadth_table`, which integrate excitation and suppression separately and
reference each unit's own pre-odor excursions.
"""

from __future__ import annotations

import numpy as np

from .response_metrics import (aggregate_odor_responses,
                               empirical_blank_thresholds,
                               signed_population_metrics)


def signed_session_tables(session, *, blank_odor=0, tail_probability=.01,
                          reducer="median"):
    """Return unit- and odor-level dictionaries, nested within one session."""
    matrices = aggregate_odor_responses(
        session.mean_z, session.odor_ids, session.states, reducer=reducer)
    units, odors = [], []
    session_fields = {
        "group_id": session.group_id, "mouse": session.mouse,
        "line": session.line, "objective": session.objective,
        "depth_class": session.depth_class, "population": session.population,
    }
    for state, (odor_ids, matrix) in matrices.items():
        blank = session.mean_z[:,
            (session.odor_ids == blank_odor) & (session.states == state)]
        thresholds = empirical_blank_thresholds(
            blank.ravel(), tail_probability=tail_probability)
        common = session_fields | {
            "negative_threshold_z": thresholds.negative_z,
            "positive_threshold_z": thresholds.positive_z,
        }
        keep = odor_ids != blank_odor
        odor_ids, matrix = odor_ids[keep], matrix[:, keep]
        metrics = signed_population_metrics(matrix, thresholds)
        for index, unit_id in enumerate(session.unit_ids):
            units.append(common | {
                "state": session.state_levels[int(state)],
                "unit_id": unit_id.decode() if isinstance(unit_id, bytes) else str(unit_id),
                "excitation_breadth": metrics["unit_excitation_breadth"][index],
                "suppression_breadth": metrics["unit_suppression_breadth"][index],
                "excitation_lifetime_sparseness": metrics[
                    "unit_excitation_sparseness"][index],
                "suppression_lifetime_sparseness": metrics[
                    "unit_suppression_sparseness"][index],
            })
        for index, odor_id in enumerate(odor_ids):
            odors.append(common | {
                "state": session.state_levels[int(state)], "odor_id": int(odor_id),
                **{name: np.asarray(value)[index] for name, value in metrics.items()
                   if name.startswith("odor_")},
            })
    return units, odors
