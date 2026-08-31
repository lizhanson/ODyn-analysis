"""Example field images at four levels of processing, by odor and block.

The four levels are the same field rendered with progressively more analysis
applied, so a talk can show what each step adds without changing the field:

1. ``fluorescence``  mean raw fluorescence over the pre-odor frames. No
   normalization of any kind; this is what the microscope recorded.
2. ``roi_outline``   the same image with the curated ROI boundaries drawn on
   top, so the segmentation can be judged against the anatomy it came from.
3. ``pixel_z``       per-pixel odor-period z, computed independently at every
   pixel and reduced across the selected trials. A true motion-corrected pixel
   map, not a mask filled with trace values.
4. ``roi_z``         the final analysis units painted with their own
   odor-period z. This is the quantity every downstream figure is built from.

Levels 3 and 4 answer different questions and are deliberately produced from
the same trials with the same window, so the difference between them is
attributable to ROI definition rather than to normalization or trial choice.

Selection is by group, block, and odor. "Block" is the manifest's state level:
``pre`` is awake and ``post`` is after the ket/xyl injection.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from analysis.figures.paths import imaging_root, output_root, repo_path
from analysis.figures.population_metrics import _source_path
from analysis.figures.session_data import available_sessions
from analysis.session.h5io import open_h5

LEVELS = ("fluorescence", "roi_outline", "pixel_z", "roi_z")

# Matches `analysis.session.masks.GREY_PERCENTILES` so these stills, the
# curation GUI, and the QC pages stretch the same field identically.
GREY_PERCENTILES = (1.0, 99.5)

BLOCK_LABELS = {"pre": "awake", "post": "ket/xyl"}

# Response windows in seconds from odor onset. Applied to both the pixel and
# the ROI level so the two are directly comparable; `window_s=None` instead
# uses each trial's exact recorded valve frames.
#
# The names match the epochs the rest of the pipeline already measures, so a
# panel and a summary statistic mean the same thing: `odor`, `early`, and
# `late` follow `population_metrics.TemporalWindows`, and `post_odor` is the
# fixed four-second offset epoch from `session.trace_analysis`.
WINDOWS = {
    "odor": (0.0, 4.0),
    "early": (0.0, 2.0),
    "late": (2.0, 4.0),
    "post_odor": (4.0, 8.0),
}

DEFAULT_WINDOW_S = WINDOWS["odor"]

# The baseline is always the pre-odor frames, whatever the response window is.
# A post-odor panel is therefore still referenced to the same quiet period as
# the odor panel, and the two are directly comparable.
DEFAULT_BASELINE_S = None       # None uses every frame before odor onset

DEFAULT_POPULATION = {"10x": "units", "20x": "groups"}


def _decode(values):
    return [value.decode() if isinstance(value, bytes) else str(value)
            for value in values]


def resolve_window(window):
    """Accept a named window, an explicit (start, stop) in seconds, or None."""
    if window is None:
        return None
    if isinstance(window, str):
        if window not in WINDOWS:
            raise ValueError(f"window must be one of {sorted(WINDOWS)} or a "
                             f"(start, stop) pair in seconds, got {window!r}")
        return WINDOWS[window]
    start, stop = (float(value) for value in window)
    if stop <= start:
        raise ValueError(f"window stop must exceed start, got {(start, stop)}")
    return (start, stop)


def window_label(window) -> str:
    """Short caption for a resolved window."""
    if window is None:
        return "valve window"
    named = [name for name, span in WINDOWS.items() if span == tuple(window)]
    span = f"{window[0]:g}-{window[1]:g} s"
    return f"{named[0]} ({span})" if named else span


def smooth_frames(stack, sigma_px):
    """Gaussian-blur every frame of a stack, in pixels, without spreading NaN.

    Smoothing is applied to the frames before any statistic is computed, so the
    baseline mean, the baseline SD, and the response mean all describe the same
    smoothed image and the resulting z stays a real z. Smoothing a finished z
    map instead blurs a ratio, which is not the same quantity.

    Pixels the motion correction never covered stay missing: they are filled by
    a normalized convolution rather than by zero, so the field edge is not
    pulled toward the background, and they are restored to NaN afterwards.

    What smoothing does to the resulting z is worth being explicit about. It
    does not lower the background noise of the z map: the pixel noise it
    removes from the numerator is removed from the baseline SD in the
    denominator too, and the two cancel. What it does is raise the z of
    spatially coherent signal, because that signal survives the blur while the
    per-pixel SD dividing it shrinks. Contrast against the background therefore
    improves, but the numbers move: **z values are not comparable across
    different sigma**, so a figure must hold sigma fixed across any panels that
    share a colour scale. `sigma_px` is recorded in the caption, the filename,
    and the export manifest for that reason.
    """
    sigma = float(sigma_px or 0.0)
    stack = np.asarray(stack, np.float32)
    if sigma <= 0:
        return stack
    from scipy.ndimage import gaussian_filter

    spread = (0.0, sigma, sigma)
    valid = np.isfinite(stack)
    if valid.all():
        return gaussian_filter(stack, spread, mode="nearest")
    weight = gaussian_filter(valid.astype(np.float32), spread, mode="nearest")
    total = gaussian_filter(np.where(valid, stack, 0.0).astype(np.float32),
                            spread, mode="nearest")
    smoothed = np.divide(total, weight, out=np.full_like(total, np.nan),
                         where=weight > 1e-6)
    smoothed[~valid] = np.nan
    return smoothed


# --------------------------------------------------------------------------
# Session inputs
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SessionContext:
    """Everything needed to build images for one session, read once.

    Notebooks loop over many odors within a session, and the imaging root is
    usually a network mount, so the per-session reads are kept out of the loop.
    """

    row: dict
    root: Path
    objective: str
    population: str
    grouped_path: Path
    source_path: Path
    trial_ids: np.ndarray
    odor_ids: np.ndarray
    states: np.ndarray
    state_levels: tuple[str, ...]
    odor_on: np.ndarray
    odor_off: np.ndarray
    labels: np.ndarray
    members: list
    unit_ids: np.ndarray
    z: np.ndarray
    time_s: np.ndarray
    frame_rate: float
    um_per_px: float | None

    @property
    def unit_labels(self) -> np.ndarray:
        """Label image of final analysis units: joined ROIs share one label."""
        return paint_units(self, np.arange(1, len(self.members) + 1),
                           background=0).astype(np.int32)

    def describe(self) -> str:
        line = str(self.row.get("population", "")).split("-")[0]
        return (f"group {self.row['group_id']} - {self.row['mouse']} - {line} "
                f"{self.objective}")


def _pixel_scale(handle) -> float | None:
    """Reported micron scale from the round's parameters, when it recorded one."""
    raw = handle.attrs.get("parameters_json")
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode()
    try:
        value = json.loads(raw).get("session", {}).get("um_per_px_reported")
    except (ValueError, AttributeError):
        return None
    return None if value is None else float(value)


def load_context(row, root, *, population=None) -> SessionContext:
    """Open one session's grouped product and the extraction round behind it."""
    grouped_path = Path(row["grouped_path"])
    objective = str(row["objective"]).lower()
    population = population or DEFAULT_POPULATION[objective]
    with open_h5(grouped_path) as grouped:
        source_path = _source_path(grouped_path, grouped)
        if population not in grouped:
            raise KeyError(f"{grouped_path.name} has no '{population}' population; "
                           f"found {sorted(k for k in grouped if hasattr(grouped[k], 'keys'))}")
        unit = grouped[population]
        odor_ids = grouped["odor_id"][:]
        states = grouped["state"][:]
        state_levels = tuple(_decode(grouped["state_levels"][:]))
        members = [np.asarray(value, np.int64) for value in unit["member_roi_ids"][:]]
        unit_ids = np.asarray(_decode(unit["unit_id"][:]))
        z = unit["z"][:]
    with open_h5(source_path) as source:
        # The 20x grouped product stores positional indices in /trial_id; the
        # source round always carries the database ids in the same trial order.
        trial_ids = source["trials/trial_id"][:]
        odor_on = source["trials/odor_on_frame"][:]
        odor_off = source["trials/odor_off_frame"][:]
        labels = source["masks/labels"][:]
        time_s = source["traces/time_s"][:]
        um_per_px = _pixel_scale(source)
    if len(trial_ids) != z.shape[1]:
        raise ValueError(f"{grouped_path.name}: grouped trials ({z.shape[1]}) do not "
                         f"align with the source round ({len(trial_ids)})")
    steps = np.diff(np.asarray(time_s, float))
    frame_rate = float(1.0 / np.median(steps)) if steps.size else float("nan")
    return SessionContext(
        row=dict(row), root=Path(root), objective=objective, population=population,
        grouped_path=grouped_path, source_path=Path(source_path),
        trial_ids=np.asarray(trial_ids), odor_ids=np.asarray(odor_ids),
        states=np.asarray(states), state_levels=state_levels,
        odor_on=np.asarray(odor_on), odor_off=np.asarray(odor_off),
        labels=np.asarray(labels), members=members, unit_ids=unit_ids, z=z,
        time_s=np.asarray(time_s, float), frame_rate=frame_rate,
        um_per_px=um_per_px)


def available_blocks(context) -> list[str]:
    return [level for index, level in enumerate(context.state_levels)
            if np.any(context.states == index)]


def available_odors(context, block) -> list[int]:
    selected = context.states == _block_code(context, block)
    return sorted(int(value) for value in np.unique(context.odor_ids[selected]))


def _block_code(context, block) -> int:
    if block not in context.state_levels:
        raise ValueError(f"block {block!r} is not one of {context.state_levels}")
    return context.state_levels.index(block)


def select_trials(context, *, block, odor_id) -> np.ndarray:
    """Positional indices of the trials for one block and odor."""
    selected = ((context.states == _block_code(context, block))
                & (context.odor_ids == int(odor_id)))
    indices = np.flatnonzero(selected)
    if indices.size == 0:
        raise ValueError(f"no {block} trials of odor {odor_id} in "
                         f"group {context.row['group_id']}")
    return indices


# --------------------------------------------------------------------------
# Movie-derived levels: raw fluorescence and pixelwise z
# --------------------------------------------------------------------------

def _movie_paths(context, indices) -> list[Path]:
    """Approved motion-corrected files for the selected trials.

    The database is authoritative and stores paths relative to the imaging
    root, so it travels between workstations. The extraction round records the
    absolute path it actually read, which is used only as a fallback.
    """
    wanted = [int(context.trial_ids[index]) for index in indices]
    database = Path(context.root) / ".odyn" / "odyn.db"
    found = {}
    if database.exists():
        placeholders = ",".join("?" for _ in wanted)
        query = f"""
            SELECT t.trial_id, m.mcor_path
            FROM trials AS t JOIN mcor_files AS m ON m.acq_id = t.acq_id
            WHERE m.approved = 1 AND t.trial_id IN ({placeholders})
        """
        with sqlite3.connect(f"file:{database}?immutable=1", uri=True) as connection:
            found = dict(connection.execute(query, wanted))
    paths = []
    for position, trial in zip(indices, wanted):
        if trial in found:
            paths.append(Path(context.root) / str(found[trial]).replace("\\", "/"))
            continue
        with open_h5(context.source_path) as source:
            if "trials/mcor_path" not in source:
                raise FileNotFoundError(
                    f"no approved motion-corrected file for trial {trial}")
            levels = _decode(source["trials/mcor_path_levels"][:])
            recorded = Path(levels[int(source["trials/mcor_path"][position])])
        if not recorded.exists():
            raise FileNotFoundError(
                f"trial {trial}: database has no approved file and the recorded "
                f"path does not exist on this computer ({recorded})")
        paths.append(recorded)
    return paths


class WindowOutsideAcquisition(ValueError):
    """The requested window does not fit inside the recording."""


def _check_spans(path, n_frame, response_span, baseline_span):
    for name, (first, last) in (("response", response_span),
                                ("baseline", baseline_span)):
        if first < 0 or last > n_frame:
            raise WindowOutsideAcquisition(
                f"{Path(path).name}: the {name} window needs frames "
                f"{first}-{last} but the acquisition has {n_frame}. Choose a "
                f"window inside the recording or pass window_s=None to use the "
                f"recorded valve frames.")


def _read_trial_windows(path, response_span, baseline_span):
    """Read only the baseline and response frames of one motion-corrected file.

    The requested spans are checked against the acquisition first. A window
    that runs off the end of the recording would otherwise be silently
    truncated by the slice, and a post-odor window on a short acquisition would
    quietly become a shorter window than the one in the caption. The frame
    count comes from the handle already being opened, because on a network
    share a second open to ask how long the file is costs as much as the read.
    """
    import tifffile

    # Memory mapping avoids loading the frames outside both windows. Some
    # mounted filesystems disallow mmap; page reads are a portable fallback.
    try:
        movie = tifffile.memmap(path)
        _check_spans(path, movie.shape[0], response_span, baseline_span)
        baseline = np.asarray(movie[baseline_span[0]:baseline_span[1]], np.float32)
        response = np.asarray(movie[response_span[0]:response_span[1]], np.float32)
    except WindowOutsideAcquisition:
        raise
    except (OSError, ValueError):
        with tifffile.TiffFile(path) as handle:
            pages = handle.pages
            _check_spans(path, len(pages), response_span, baseline_span)
            baseline = np.stack([page.asarray()
                                 for page in pages[baseline_span[0]:baseline_span[1]]])
            response = np.stack([page.asarray()
                                 for page in pages[response_span[0]:response_span[1]]])
        baseline = baseline.astype(np.float32, copy=False)
        response = response.astype(np.float32, copy=False)
    if baseline.shape[0] < 2 or response.shape[0] < 1:
        raise ValueError(f"{Path(path).name}: empty baseline or response window")
    return baseline, response


def trial_pixel_images(baseline, response, *, sigma_px=0.0):
    """Baseline mean, response mean, and the per-pixel response z of one trial.

    Every pixel is normalized by its own pre-odor SD. Pixels whose baseline SD
    falls below the first percentile of the positive SDs are returned as NaN
    rather than as a very large z: those are dead or saturated pixels, and
    dividing by their noise manufactures structure that is not there.

    `sigma_px` smooths the frames before the statistics are taken. The two raw
    images returned are always unsmoothed, so the fluorescence level stays an
    honest picture of what the microscope recorded whatever the z level does.
    """
    baseline_mean = np.mean(baseline, axis=0)
    response_mean = np.mean(response, axis=0)
    if sigma_px:
        baseline = smooth_frames(baseline, sigma_px)
        response = smooth_frames(response, sigma_px)
    baseline_sd = np.nanstd(baseline, axis=0, ddof=1)
    smoothed_baseline_mean = np.nanmean(baseline, axis=0)
    smoothed_response_mean = np.nanmean(response, axis=0)
    positive = baseline_sd[np.isfinite(baseline_sd) & (baseline_sd > 0)]
    floor = np.nanpercentile(positive, 1) if positive.size else np.inf
    numerator = smoothed_response_mean - smoothed_baseline_mean
    # A zero-SD pixel is excluded whatever the floor works out to: dividing by
    # it returns infinity, which then propagates through every later reduction.
    usable = np.isfinite(baseline_sd) & (baseline_sd > 0) & (baseline_sd >= floor)
    z = np.divide(numerator, baseline_sd,
                  out=np.full_like(numerator, np.nan), where=usable)
    return (baseline_mean.astype(np.float32), response_mean.astype(np.float32),
            z.astype(np.float32))


def _seconds_to_frame(context, index, seconds):
    rate = context.frame_rate
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("frame rate is unknown; pass window_s=None to use the "
                         "recorded valve frames")
    return int(context.odor_on[index]) + int(round(float(seconds) * rate))


def _response_frames(context, index, window_s):
    """Frame span of the response window for one trial, in movie coordinates."""
    if window_s is None:
        return int(context.odor_on[index]), int(context.odor_off[index])
    first = _seconds_to_frame(context, index, window_s[0])
    last = _seconds_to_frame(context, index, window_s[1])
    return first, max(last, first + 1)


def _baseline_frames(context, index, baseline_s):
    """Frame span of the pre-odor baseline, which never moves with the window."""
    on = int(context.odor_on[index])
    if baseline_s is None:
        return 0, on
    if baseline_s[1] > 0:
        raise ValueError(f"the baseline must end at or before odor onset, got "
                         f"{tuple(baseline_s)}")
    first = _seconds_to_frame(context, index, baseline_s[0])
    last = _seconds_to_frame(context, index, baseline_s[1])
    return max(first, 0), min(max(last, first + 2), on)


def pixel_level(context, indices, *, window_s=DEFAULT_WINDOW_S,
                baseline_s=DEFAULT_BASELINE_S, sigma_px=0.0, reducer="median",
                progress=True):
    """Reduce per-trial raw and pixelwise-z images across the selected trials.

    `window_s` chooses which frames the response is measured over — any span in
    seconds from odor onset, or one of `WINDOWS` by name, so a post-odor panel
    is `"post_odor"`. `baseline_s` is independent of it and defaults to every
    pre-odor frame, so panels at different windows share one reference period.
    """
    window_s = resolve_window(window_s)
    baseline_s = resolve_window(baseline_s)
    paths = _movie_paths(context, indices)
    iterator = list(zip(indices, paths))
    if progress:
        from tqdm.auto import tqdm
        iterator = tqdm(iterator, desc="trial movies", unit="trial", leave=False)
    baselines, responses, maps = [], [], []
    for index, path in iterator:
        baseline, response = _read_trial_windows(
            path, _response_frames(context, index, window_s),
            _baseline_frames(context, index, baseline_s))
        raw_baseline, raw_response, z = trial_pixel_images(
            baseline, response, sigma_px=sigma_px)
        baselines.append(raw_baseline)
        responses.append(raw_response)
        maps.append(z)
    reduce = {"median": np.nanmedian, "mean": np.nanmean}[reducer]
    # A pixel excluded on every trial -- an uncovered corner, or one the SD
    # floor rejects throughout -- reduces to NaN, which is the right answer and
    # not worth a warning. The renderers draw those pixels as missing.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return (reduce(np.stack(baselines), axis=0).astype(np.float32),
                reduce(np.stack(responses), axis=0).astype(np.float32),
                reduce(np.stack(maps), axis=0).astype(np.float32))


def reference_image(context):
    """The background image the QC pages already draw ROIs on, if there is one.

    A 10x session publishes a portable mask bundle beside its round, and that
    bundle's `/reference` is the exact image the spatial QC page and the
    curation GUI use. Reusing it keeps a slide consistent with the QC everyone
    has already looked at, and costs no movie reads. Returns None when the
    session has no bundle, which is the usual case at 20x.
    """
    pattern = f"group{int(context.row['group_id'])}_*_masks_processed_*.h5"
    for bundle in sorted(context.source_path.parent.glob(pattern), reverse=True):
        with open_h5(bundle) as handle:
            if "reference" in handle:
                return np.asarray(handle["reference"][:], np.float32)
    return None


# --------------------------------------------------------------------------
# ROI level
# --------------------------------------------------------------------------

def unit_response(context, indices, *, window_s=DEFAULT_WINDOW_S, reducer="median"):
    """Odor-period z per analysis unit, reduced across the selected trials."""
    window = resolve_window(window_s) or DEFAULT_WINDOW_S
    frames = (context.time_s >= window[0]) & (context.time_s < window[1])
    if not frames.any():
        raise ValueError(f"window {window} contains no frames of the trace time axis")
    reduce = {"median": np.nanmedian, "mean": np.nanmean}[reducer]
    per_trial = np.nanmean(context.z[:, indices][:, :, frames], axis=2)
    return reduce(per_trial, axis=1).astype(np.float32)


def unit_lookup(context) -> np.ndarray:
    """Map each ROI label to its analysis unit, as 1-based indices with 0 spare.

    One pass over the label image beats one `np.isin` per unit: a 10x field has
    a few hundred units, and the odor sweeps paint two images per example.
    """
    labels = context.labels
    lookup = np.zeros(int(labels.max(initial=0)) + 1, np.int64)
    for index, roi_ids in enumerate(context.members, start=1):
        ids = np.asarray(roi_ids, np.int64)
        lookup[ids[(ids > 0) & (ids < lookup.size)]] = index
    return lookup


def paint_units(context, values, *, background=np.nan) -> np.ndarray:
    """Write one value per analysis unit into every pixel of its member ROIs."""
    values = np.asarray(values, np.float32)
    if len(values) != len(context.members):
        raise ValueError(f"expected {len(context.members)} unit values, got {len(values)}")
    index = unit_lookup(context)[context.labels]
    image = np.full(context.labels.shape, background, np.float32)
    painted = index > 0
    image[painted] = values[index[painted] - 1]
    return image


# --------------------------------------------------------------------------
# Assembled example
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ExampleImages:
    """The same field at four levels of processing, for one block and odor."""

    context: SessionContext
    block: str
    odor_id: int
    trial_ids: np.ndarray
    window_s: tuple[float, float] | None
    baseline_s: tuple[float, float] | None
    sigma_px: float
    fluorescence_source: str
    fluorescence: np.ndarray
    response_fluorescence: np.ndarray
    pixel_z: np.ndarray
    unit_values: np.ndarray
    roi_z: np.ndarray
    labels: np.ndarray
    unit_labels: np.ndarray

    @property
    def n_trials(self) -> int:
        return len(self.trial_ids)

    @property
    def block_label(self) -> str:
        return BLOCK_LABELS.get(self.block, self.block)

    @property
    def window_label(self) -> str:
        return window_label(self.window_s)

    @property
    def has_pixel_level(self) -> bool:
        """False when the movies were not read, so `pixel_z` is entirely NaN."""
        return bool(np.any(np.isfinite(self.pixel_z)))

    @property
    def available_levels(self) -> tuple[str, ...]:
        """The levels this example actually has data for.

        A `pixel=False` example has no pixel z. Rendering it anyway would
        composite an all-NaN map over the anatomy and produce a panel that
        looks like a real z map reading zero everywhere, which is exactly the
        sort of thing that ends up on a slide.
        """
        if self.has_pixel_level:
            return LEVELS
        return tuple(level for level in LEVELS if level != "pixel_z")

    def caption(self) -> str:
        """Everything a reader needs to know the panel was made honestly."""
        parts = [f"n={self.n_trials} trials", "median", self.window_label]
        if self.sigma_px:
            parts.append(f"sigma {self.sigma_px:g} px")
        if self.fluorescence_source != "trials":
            parts.append(self.fluorescence_source)
        return ", ".join(parts)

    def level(self, name) -> np.ndarray:
        if name not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}, got {name!r}")
        return {"fluorescence": self.fluorescence, "roi_outline": self.fluorescence,
                "pixel_z": self.pixel_z, "roi_z": self.roi_z}[name]

    def stem(self) -> str:
        line = str(self.context.row.get("population", "")).split("-")[0]
        window = ("valve" if self.window_s is None
                  else f"{self.window_s[0]:g}to{self.window_s[1]:g}s")
        smoothing = f"_sigma{self.sigma_px:g}" if self.sigma_px else ""
        return (f"group{self.context.row['group_id']}_{self.context.row['mouse']}"
                f"_{line}_{self.context.objective}_{self.block}_odor{self.odor_id}"
                f"_{window}{smoothing}")


def build_example(context, *, block, odor_id, window_s=DEFAULT_WINDOW_S,
                  baseline_s=DEFAULT_BASELINE_S, sigma_px=0.0, reducer="median",
                  pixel=True, reference=None, progress=True) -> ExampleImages:
    """Build all four levels for one block and odor.

    `window_s` selects the response window, by name from `WINDOWS` or as a
    (start, stop) pair in seconds from odor onset; `baseline_s` stays on the
    pre-odor frames whatever it is set to. `sigma_px` smooths the frames before
    the pixel z is computed and leaves the fluorescence level unsmoothed.

    `pixel=False` skips rereading the motion-corrected movies, which is the
    expensive step on a network mount. The two ROI levels are still built, and
    the fluorescence level falls back to `reference` — the published mask
    bundle's image, the same one the QC pages use — so the ROI outline level is
    still real and layouts can be checked before committing to the full pass.
    """
    indices = select_trials(context, block=block, odor_id=odor_id)
    window_s = resolve_window(window_s)
    baseline_s = resolve_window(baseline_s)
    source = "trials"
    if pixel:
        fluorescence, response, pixel_z = pixel_level(
            context, indices, window_s=window_s, baseline_s=baseline_s,
            sigma_px=sigma_px, reducer=reducer, progress=progress)
    else:
        empty = np.full(context.labels.shape, np.nan, np.float32)
        response, pixel_z = empty, empty
        background = reference if reference is not None else reference_image(context)
        if background is None:
            fluorescence, source = empty, "no background image"
        elif np.shape(background) != context.labels.shape:
            raise ValueError(f"reference image {np.shape(background)} does not "
                             f"match the mask {context.labels.shape}")
        else:
            fluorescence = np.asarray(background, np.float32)
            source = "published reference image"
    if pixel and reference is not None:
        fluorescence = np.asarray(reference, np.float32)
        source = "published reference image"
    values = unit_response(context, indices, window_s=window_s, reducer=reducer)
    return ExampleImages(
        context=context, block=block, odor_id=int(odor_id),
        trial_ids=context.trial_ids[indices], window_s=window_s,
        baseline_s=baseline_s, sigma_px=float(sigma_px or 0.0),
        fluorescence_source=source,
        fluorescence=fluorescence, response_fluorescence=response,
        pixel_z=pixel_z, unit_values=values, roi_z=paint_units(context, values),
        labels=context.labels, unit_labels=context.unit_labels)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def crop_image(image, crop):
    """Crop as (y0, y1, x0, x1); None returns the image unchanged."""
    if crop is None:
        return image
    y0, y1, x0, x1 = (int(v) for v in crop)
    return np.asarray(image)[y0:y1, x0:x1]


def stretch(image, *, percentiles=GREY_PERCENTILES, limits=None):
    """Scale to 0..1 by percentiles of the finite pixels, or by fixed limits."""
    image = np.asarray(image, np.float32)
    if limits is not None:
        low, high = (float(v) for v in limits)
    else:
        finite = image[np.isfinite(image)]
        if finite.size == 0:
            return np.zeros(image.shape, np.float32), 0.0, 1.0
        low, high = (float(v) for v in np.percentile(finite, percentiles))
    scaled = np.clip((np.nan_to_num(image, nan=low) - low)
                     / max(high - low, 1e-12), 0, 1)
    return scaled.astype(np.float32), low, high


def grayscale_rgb(image, *, percentiles=GREY_PERCENTILES, limits=None,
                  gamma=1.0) -> np.ndarray:
    """Level 1: raw fluorescence as an 8-bit greyscale RGB image.

    `gamma` below 1 lifts the dim structure that a linear stretch buries in a
    field with a few very bright ROIs. It is a display choice only and is
    never applied to anything that is measured.
    """
    scaled, _, _ = stretch(image, percentiles=percentiles, limits=limits)
    if gamma != 1.0:
        scaled = scaled ** float(gamma)
    return np.repeat((scaled * 255).round().astype(np.uint8)[:, :, None], 3, axis=2)


def boundaries(labels) -> np.ndarray:
    """Inner boundary pixels of every labelled region, 4-connected."""
    labels = np.asarray(labels)
    edge = np.zeros(labels.shape, bool)
    vertical = labels[:-1, :] != labels[1:, :]
    horizontal = labels[:, :-1] != labels[:, 1:]
    edge[:-1, :] |= vertical
    edge[1:, :] |= vertical
    edge[:, :-1] |= horizontal
    edge[:, 1:] |= horizontal
    return edge & (labels > 0)


def _palette_colors(labels) -> np.ndarray:
    from analysis.seg_10x.gui import _PALETTE
    return _PALETTE[(np.asarray(labels) - 1) % len(_PALETTE)]


def outline_rgb(image, labels, *, color=(255, 255, 0), palette=False, width=1,
                fill_alpha=0.0, **grey) -> np.ndarray:
    """Level 2: ROI boundaries drawn over the greyscale fluorescence image.

    Outlines rather than the GUI's translucent fill: a filled overlay hides the
    anatomy it is meant to be judged against, which is the whole point of
    showing this level next to level 1.
    """
    rgb = grayscale_rgb(image, **grey)
    labels = np.asarray(labels)
    if rgb.shape[:2] != labels.shape:
        raise ValueError(f"image {rgb.shape[:2]} and labels {labels.shape} differ")
    if fill_alpha > 0:
        inside = labels > 0
        if inside.any():
            colors = (_palette_colors(labels[inside]) if palette
                      else np.asarray(color, np.float32))
            rgb[inside] = ((1 - fill_alpha) * rgb[inside].astype(np.float32)
                           + fill_alpha * colors).round().astype(np.uint8)
    edge = boundaries(labels)
    for _ in range(max(int(width), 1) - 1):
        thicker = np.zeros_like(edge)
        thicker[:-1, :] |= edge[1:, :]
        thicker[1:, :] |= edge[:-1, :]
        thicker[:, :-1] |= edge[:, 1:]
        thicker[:, 1:] |= edge[:, :-1]
        edge |= thicker & (labels > 0)
    if edge.any():
        rgb[edge] = (_palette_colors(labels[edge]) if palette
                     else np.asarray(color, np.uint8))
    return rgb


def signed_rgb(values, *, limits=(-2.0, 4.0), cmap="RdBu_r", background=None,
               threshold=None, nan_color=(20, 20, 20), **grey) -> np.ndarray:
    """Levels 3 and 4: signed z on a diverging map, optionally over the anatomy.

    With a `background`, each pixel's opacity ramps from transparent at z=0 to
    opaque at `threshold`, so weak pixels show the field underneath instead of
    a wash of pale colour. Excitation and suppression keep the same centre, so
    the visual zero is the real zero.
    """
    import matplotlib as mpl
    from matplotlib.colors import TwoSlopeNorm

    values = np.asarray(values, np.float32)
    norm = TwoSlopeNorm(vmin=limits[0], vcenter=0.0, vmax=limits[1])
    colored = mpl.colormaps[cmap](norm(np.nan_to_num(values, nan=0.0)))
    rgb = (colored[:, :, :3] * 255).round().astype(np.uint8)
    missing = ~np.isfinite(values)
    if background is None:
        rgb[missing] = np.asarray(nan_color, np.uint8)
        return rgb
    grey_rgb = grayscale_rgb(background, **grey)
    if grey_rgb.shape[:2] != values.shape:
        raise ValueError(f"background {grey_rgb.shape[:2]} and values "
                         f"{values.shape} differ")
    cutoff = float(threshold if threshold is not None
                   else max(abs(limits[0]), abs(limits[1])) / 2)
    alpha = np.clip(np.abs(np.nan_to_num(values, nan=0.0)) / max(cutoff, 1e-12), 0, 1)
    alpha[missing] = 0.0
    alpha = alpha[:, :, None]
    return (alpha * rgb + (1 - alpha) * grey_rgb).round().astype(np.uint8)


def render_level(example, level, *, crop=None, background=True, limits=(-2.0, 4.0),
                 threshold=None, palette=False, outline_color=(255, 255, 0),
                 outline_width=1, unit_outlines=True, gamma=1.0,
                 percentiles=GREY_PERCENTILES) -> np.ndarray:
    """Render one of `LEVELS` from a built example as an 8-bit RGB image."""
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
    if level == "pixel_z" and not example.has_pixel_level:
        raise ValueError(
            "this example has no pixel z: it was built with pixel=False, so the "
            "motion-corrected movies were never read. Rebuild with pixel=True, "
            "or render only example.available_levels.")
    grey = {"percentiles": percentiles, "gamma": gamma}
    fluorescence = crop_image(example.fluorescence, crop)
    labels = crop_image(example.unit_labels if unit_outlines else example.labels, crop)
    if level == "fluorescence":
        return grayscale_rgb(fluorescence, **grey)
    if level == "roi_outline":
        return outline_rgb(fluorescence, labels, color=outline_color,
                           palette=palette, width=outline_width, **grey)
    values = crop_image(example.pixel_z if level == "pixel_z" else example.roi_z, crop)
    rgb = signed_rgb(values, limits=limits, threshold=threshold,
                     background=fluorescence if background else None, **grey)
    if level == "roi_z" and outline_width > 0:
        edge = boundaries(labels)
        rgb[edge] = np.asarray(outline_color, np.uint8)
    return rgb


def save_png(path, rgb) -> Path:
    """Write a bare image: no axes, no margins, one file pixel per image pixel."""
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(rgb, np.uint8)).save(path)
    return path


# --------------------------------------------------------------------------
# Slide figures
# --------------------------------------------------------------------------

LEVEL_TITLES = {
    "fluorescence": "raw fluorescence",
    "roi_outline": "curated ROIs",
    "pixel_z": "pixelwise z",
    "roi_z": "ROI z",
}


def _scale_bar(ax, um_per_px, *, length_um=100, color="white"):
    """Draw a scale bar in the lower-right corner, in data coordinates."""
    if not um_per_px:
        return None
    x0, x1 = ax.get_xlim()
    y1, y0 = ax.get_ylim()          # imshow inverts the y axis
    width = float(length_um) / float(um_per_px)
    span = abs(x1 - x0)
    if width > 0.6 * span:          # too long to read; fall back to a round tenth
        length_um = round(0.2 * span * float(um_per_px), -1) or 1
        width = float(length_um) / float(um_per_px)
    pad = 0.04 * span
    right = max(x0, x1) - pad
    bottom = max(y0, y1) - pad
    ax.plot([right - width, right], [bottom, bottom], color=color, lw=3,
            solid_capstyle="butt")
    ax.text(right - width / 2, bottom - 0.02 * span, f"{int(length_um)} µm",
            color=color, ha="center", va="bottom", fontsize=8)
    return length_um


def ladder_figure(path, example, *, levels=None, crop=None, limits=(-2.0, 4.0),
                  background=True, scale_bar_um=100, dpi=300, **render):
    """One row showing the same field at each requested level of processing.

    Defaults to the levels the example actually has, so a `pixel=False` run
    produces a three-panel ladder rather than a blank panel labelled as data.
    """
    levels = example.available_levels if levels is None else levels
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.cm import ScalarMappable

    levels = tuple(levels)
    height = 3.4
    fig, axes = plt.subplots(1, len(levels), figsize=(3.2 * len(levels), height),
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, level in zip(axes, levels):
        ax.imshow(render_level(example, level, crop=crop, limits=limits,
                               background=background, **render),
                  interpolation="nearest")
        ax.set(xticks=[], yticks=[], title=LEVEL_TITLES[level])
        for spine in ax.spines.values():
            spine.set_visible(False)
    if scale_bar_um:
        _scale_bar(axes[0], example.context.um_per_px, length_um=scale_bar_um)
    if any(level in ("pixel_z", "roi_z") for level in levels):
        mappable = ScalarMappable(
            norm=TwoSlopeNorm(vmin=limits[0], vcenter=0.0, vmax=limits[1]),
            cmap="RdBu_r")
        fig.colorbar(mappable, ax=axes, label="odor-period z", shrink=0.8,
                     fraction=0.035)
    fig.suptitle(f"{example.context.describe()} - {example.block_label} - "
                 f"odor {example.odor_id}\n{example.caption()}", fontsize=10)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def export_example(output_dir, example, *, levels=None, crop=None,
                   limits=(-2.0, 4.0), background=True, ladder=True,
                   save_arrays=True, dpi=300, **render) -> dict:
    """Write one bare PNG per level, the labelled ladder, and the raw arrays.

    The bare PNGs carry no axes or titles so they can be placed directly on a
    slide; the arrays are kept so the same example can be re-rendered with
    different limits without rereading any movie.
    """
    levels = example.available_levels if levels is None else levels
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = example.stem()
    written = {}
    for position, level in enumerate(levels, start=1):
        rgb = render_level(example, level, crop=crop, limits=limits,
                           background=background, **render)
        written[level] = str(save_png(output_dir / f"{stem}_{position}_{level}.png", rgb))
    if ladder:
        written["ladder"] = str(ladder_figure(
            output_dir / f"{stem}_ladder.png", example, levels=levels, crop=crop,
            limits=limits, background=background, dpi=dpi, **render))
    if save_arrays:
        arrays = output_dir / f"{stem}_arrays.npz"
        np.savez_compressed(
            arrays, fluorescence=example.fluorescence,
            response_fluorescence=example.response_fluorescence,
            pixel_z=example.pixel_z, roi_z=example.roi_z,
            unit_values=example.unit_values, labels=example.labels,
            unit_labels=example.unit_labels, trial_ids=example.trial_ids,
            unit_ids=example.context.unit_ids)
        written["arrays"] = str(arrays)
    written.update(group_id=int(example.context.row["group_id"]),
                   mouse=example.context.row["mouse"], block=example.block,
                   odor_id=example.odor_id, n_trials=example.n_trials,
                   window=example.window_label, sigma_px=example.sigma_px,
                   fluorescence_source=example.fluorescence_source,
                   um_per_px=example.context.um_per_px)
    return written


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--groups", nargs="+", type=int, required=True)
    parser.add_argument("--blocks", nargs="+", default=["pre"],
                        help="state levels; pre is awake, post is ket/xyl. "
                             "'all' uses every block the session contains")
    parser.add_argument("--odors", nargs="+", required=True,
                        help="odor_id values, or 'all' for every odor present "
                             "in that session and block")
    parser.add_argument("--population", default=None,
                        help="10x: units. 20x: groups, somas, or processes")
    parser.add_argument("--manifest", type=Path, default=repo_path(
        "analysis", "stage0", "ketxyl_16odor_session_manifest.csv"))
    parser.add_argument("--imaging-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path,
                        default=output_root() / "example_images")
    parser.add_argument("--reducer", choices=("median", "mean"), default="median")
    parser.add_argument("--limits", nargs=2, type=float, default=(-2.0, 4.0))
    parser.add_argument("--window", default="odor",
                        help=f"response window: a name from {sorted(WINDOWS)}, "
                             f"a 'start,stop' pair in seconds from odor onset, "
                             f"or 'valve' for the recorded valve frames")
    parser.add_argument("--baseline", default=None,
                        help="pre-odor baseline as 'start,stop' in seconds; "
                             "defaults to every frame before odor onset and "
                             "never moves with --window")
    parser.add_argument("--sigma-px", type=float, default=0.0,
                        help="Gaussian spatial smoothing applied to the frames "
                             "before the pixel z; 0 disables it")
    parser.add_argument("--no-pixel", action="store_true",
                        help="skip rereading movies; ROI levels only")
    parser.add_argument("--no-background", action="store_true",
                        help="render z levels on a flat map, not over the anatomy")
    args = parser.parse_args(argv)

    def parse_span(value):
        if value is None:
            return None
        if value == "valve":
            return None
        if "," in str(value):
            start, stop = str(value).split(",")
            return (float(start), float(stop))
        return str(value)

    window = parse_span(args.window)
    baseline = parse_span(args.baseline)
    root = imaging_root(args.imaging_root)
    inventory = {int(row["group_id"]): row
                 for row in available_sessions(args.manifest, root)}
    missing = sorted(set(args.groups) - set(inventory))
    if missing:
        raise ValueError(f"groups absent from the manifest: {missing}")
    report = []
    for group_id in args.groups:
        row = inventory[group_id]
        if not row["available"]:
            raise FileNotFoundError(f"group {group_id} has no grouped product")
        context = load_context(row, root, population=args.population)
        blocks = (available_blocks(context) if args.blocks == ["all"]
                  else args.blocks)
        for block in blocks:
            odors = (available_odors(context, block) if args.odors == ["all"]
                     else [int(value) for value in args.odors])
            for odor_id in odors:
                example = build_example(
                    context, block=block, odor_id=odor_id, window_s=window,
                    baseline_s=baseline, sigma_px=args.sigma_px,
                    reducer=args.reducer, pixel=not args.no_pixel)
                report.append(export_example(
                    args.output_dir, example, limits=tuple(args.limits),
                    background=not args.no_background))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
