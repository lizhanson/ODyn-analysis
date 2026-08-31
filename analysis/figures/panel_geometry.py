"""Whole-panel odor geometry: three views at different points on one trade-off.

Crossvalidated distances are unbiased but noisy at these trial counts, and hard
to read across sixteen odors. Two cheaper views are kept alongside them because
they are far more legible, and legibility is what generates hypotheses:

  confusion        leave-one-trial-out nearest centroid. Intuitive units --
                   what gets mistaken for what -- and directly comparable
                   across populations against a fixed chance level.
  correlation RDM  1 - correlation between odor mean patterns. Not
                   crossvalidated and positively biased, so it can never
                   establish that a difference is real.
  crossnobis RDM   crossvalidated; the one to quote when claiming a
                   difference is reproducible.

Odors are ordered so each mixture sits beside the two components it is built
from, which is what makes mixture-versus-component structure legible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .geometry import diagonal_crossnobis
from .population_metrics import TemporalWindows, _common, _window

# blank | lambda family | alpha family | epsilon family | remaining singles
PANEL_ORDER = (0, 3, 12, 39, 40, 4, 10, 17, 18, 22, 30, 31, 32, 1, 2, 21)
PANEL_LABEL = {
    0: "min oil", 1: "eugenol", 2: "me salicyl", 3: "acetophen", 4: "1-butanol",
    10: "a-pinene", 12: "limonene", 17: "alpha (4+10)", 18: "alpha' (4+10)",
    21: "et butyrate", 22: "cyclopent", 30: "2Me-2-pent",
    31: "eps (22+30)", 32: "eps' (22+30)", 39: "lam (3+12)", 40: "lam' (3+12)",
}
FAMILY_BREAKS = (1, 5, 9, 13)
MIN_TRIALS = 3


def trial_matrix(data, state_name, *, windows=TemporalWindows()):
    """trial x unit responses and odor labels for one state."""
    levels = list(data["state_levels"])
    if state_name not in levels:
        return None
    rows = data["state"] == levels.index(state_name)
    if not np.any(rows):
        return None
    mask = _window(data["time_s"], windows.odor)
    block = data["z"][:, rows, :][:, :, mask]
    return np.transpose(np.nanmean(block, axis=2), (1, 0)), data["odor_id"][rows]


def _present(odor, order):
    return [level for level in order if np.sum(odor == level) >= 1]


def correlation_rdm(x, odor, order=PANEL_ORDER):
    """1 - correlation between odor mean patterns. Biased, but legible."""
    centroids = []
    for level in order:
        rows = odor == level
        centroids.append(np.nanmean(x[rows], axis=0) if np.any(rows)
                         else np.full(x.shape[1], np.nan))
    c = np.vstack(centroids)
    usable = np.isfinite(c).all(axis=0)
    if usable.sum() < 3:
        return np.full((len(order), len(order)), np.nan)
    return 1 - np.corrcoef(c[:, usable])


def crossnobis_rdm(x, odor, order=PANEL_ORDER, *, repeats=60,
                   minimum_trials=MIN_TRIALS):
    """Crossvalidated RDM; entries stay NaN where repeats are too few."""
    n = len(order)
    matrix = np.full((n, n), np.nan)
    np.fill_diagonal(matrix, 0.)
    for i in range(n):
        for j in range(i+1, n):
            if (np.sum(odor == order[i]) < minimum_trials
                    or np.sum(odor == order[j]) < minimum_trials):
                continue
            rows = np.isin(odor, (order[i], order[j]))
            _, pair = diagonal_crossnobis(x[rows], odor[rows], repeats=repeats,
                                          seed=i*n + j)
            matrix[i, j] = matrix[j, i] = pair[0, 1]
    return matrix


def confusion_matrix(x, odor, order=PANEL_ORDER):
    """Leave-one-trial-out nearest centroid under correlation distance."""
    n = len(order)
    matrix = np.zeros((n, n))
    counts = np.zeros(n)
    position = {level: i for i, level in enumerate(order)}
    usable = np.isfinite(x).all(axis=0)
    if usable.sum() < 3:
        return np.full((n, n), np.nan)
    x = x[:, usable]
    for trial in range(x.shape[0]):
        true = position.get(int(odor[trial]))
        if true is None or np.std(x[trial]) == 0:
            continue
        best, best_r = None, -np.inf
        for level in order:
            rows = odor == level
            rows[trial] = False
            if not np.any(rows):
                continue
            centroid = np.nanmean(x[rows], axis=0)
            if np.std(centroid) == 0:
                continue
            r = np.corrcoef(centroid, x[trial])[0, 1]
            if np.isfinite(r) and r > best_r:
                best_r, best = r, position[level]
        if best is not None:
            matrix[true, best] += 1
            counts[true] += 1
    with np.errstate(invalid="ignore"):
        return matrix/counts[:, None]


def panel_tables(data, row, population, *, order=PANEL_ORDER):
    """Per-state confusion accuracy plus the two RDMs, as tidy rows."""
    common = _common(row, population)
    accuracy, distances = [], []
    for state_name in data["state_levels"]:
        got = trial_matrix(data, state_name)
        if got is None:
            continue
        x, odor = got
        confusion = confusion_matrix(x, odor, order)
        correlation = correlation_rdm(x, odor, order)
        crossnobis = crossnobis_rdm(x, odor, order)
        for index, level in enumerate(order):
            if not np.isfinite(confusion[index, index]):
                continue
            accuracy.append(common | {
                "state": state_name, "odor_id": int(level),
                "odor": PANEL_LABEL.get(int(level), str(level)),
                "n_trials": int(np.sum(odor == level)),
                "correct_fraction": float(confusion[index, index]),
                "chance": 1/len(order),
            })
        for i, a in enumerate(order):
            for j, b in enumerate(order):
                if j <= i:
                    continue
                distances.append(common | {
                    "state": state_name, "odor_a": int(a), "odor_b": int(b),
                    "confusion_symmetric": float(
                        (confusion[i, j] + confusion[j, i])/2),
                    "correlation_distance": float(correlation[i, j]),
                    "crossnobis": float(crossnobis[i, j]),
                })
    return pd.DataFrame(accuracy), pd.DataFrame(distances)


def plot_confusion(matrices, path, *, order=PANEL_ORDER, suptitle=None):
    """Grid of confusion matrices; `matrices` maps a title to a matrix."""
    import matplotlib.pyplot as plt

    labels = [PANEL_LABEL.get(level, str(level)) for level in order]
    items = list(matrices.items())
    columns = min(3, len(items))
    rows = int(np.ceil(len(items)/columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4.2*columns, 4.6*rows),
                             squeeze=False, constrained_layout=True)
    flat = axes.ravel()
    for index, (title, matrix) in enumerate(items):
        ax = flat[index]
        image = ax.imshow(matrix, cmap="magma", vmin=0, vmax=1,
                          interpolation="nearest")
        for edge in FAMILY_BREAKS:
            ax.axhline(edge-.5, color="k", lw=.7, alpha=.55)
            ax.axvline(edge-.5, color="k", lw=.7, alpha=.55)
        ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=5.6)
        # Only the bottom row carries x labels, so they cannot collide with
        # the titles of the row beneath.
        if index < len(items) - columns:
            ax.set_xticklabels([])
        else:
            ax.set_xticklabels(labels, rotation=90, fontsize=5.6)
            ax.set_xlabel("classified as")
        if index % columns == 0:
            ax.set_ylabel("true odor")
        ax.set_title(f"{title}\n{np.nanmean(np.diag(matrix))*100:.0f}% correct "
                     f"(chance {100/len(order):.0f}%)", fontsize=9, pad=6)
    for ax in flat[len(items):]:
        ax.axis("off")
    fig.colorbar(image, ax=flat.tolist(), shrink=.55,
                 label="fraction classified as")
    if suptitle:
        fig.suptitle(suptitle, fontsize=10)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path
