"""Bright-pupil extraction aligned to the behavioural and two-photon clocks."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re

import numpy as np

PUPIL_FITTER_VERSION = 7  # invalidate checkpoints made by older fit algorithms


@dataclass(frozen=True)
class PupilConfig:
    """Parameters whose defaults are deliberately conservative diagnostics."""

    roi: tuple[int, int, int, int] | None = None  # y0, y1, x0, x1
    bright_percentile: float = 97.0
    threshold_offset: float = 0.0
    ransac_residual_px: float = 2.0
    ransac_trials: int = 200
    consensus_margin_px: float = 6.0
    consensus_margin_fraction: float = 0.06
    min_concavity_fraction: float = 0.02
    max_concavity_fraction: float = 0.20
    min_illumination_peak: float = 60.0
    min_inlier_fraction: float = 0.55
    max_residual_px: float = 3.0
    min_axis_ratio: float = 0.25
    max_diameter_rate_px_s: float = 150.0
    max_bad_fraction: float = 0.20
    random_seed: int = 0


def count_csv_frames(path) -> int:
    """Count data rows in a frametimes CSV, tolerating a single header row."""

    rows = []
    with Path(path).open(newline="") as handle:
        for row in csv.reader(handle):
            if row and any(cell.strip() for cell in row):
                rows.append(row)
    if not rows:
        return 0
    try:
        [float(cell) for cell in rows[0] if cell.strip()]
        header = 0
    except ValueError:
        header = 1
    return len(rows) - header


def iter_gray_frames(video_path):
    """Yield grayscale uint8 frames through PyAV without retaining the movie."""

    try:
        import av
    except ImportError as exc:
        raise ImportError("Pupil extraction requires PyAV (`pip install av`).") from exc

    with av.open(str(video_path)) as container:
        for frame in container.decode(video=0):
            yield frame.to_ndarray(format="gray")


def count_video_frames(video_path, *, progress_desc=None) -> int:
    """Exact decoded frame count from a bounded-memory streaming pass."""

    frames = iter_gray_frames(video_path)
    progress = None
    if progress_desc is not None:
        total = None
        metadata_path = Path(video_path).with_name(
            f"{Path(video_path).stem}_metadata.json"
        )
        try:
            import json
            total = int(json.loads(metadata_path.read_text()).get("n_frames"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
        try:
            from tqdm.auto import tqdm
            progress = tqdm(
                frames, total=total, desc=progress_desc, unit="frame",
                leave=False, dynamic_ncols=True,
            )
            frames = progress
        except ImportError:
            pass
    try:
        return sum(1 for _ in frames)
    finally:
        if progress is not None:
            progress.close()


def video_recording_timestamp(video_path) -> tuple[datetime, str]:
    """Recording timestamp, preferring MP4 metadata over path or mtime."""

    path = Path(video_path)
    # The converter preserves Micro-Manager's acquisition timestamp here.
    # This is preferable to MP4 creation_time, which may be conversion time.
    metadata_candidates = [
        path.with_name(f"{path.stem}_metadata.json"),
        *sorted(path.parent.glob("*_metadata.json")),
    ]
    for metadata_path in dict.fromkeys(metadata_candidates):
        if not metadata_path.exists():
            continue
        try:
            import json
            metadata = json.loads(metadata_path.read_text())
            value = metadata.get("mm_summary", {}).get("StartTime")
            if value:
                parsed = datetime.strptime(value, "%Y-%m-%d %H:%M:%S.%f %z")
                return parsed, f"{metadata_path.name}: mm_summary.StartTime"
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue

    try:
        import av
        with av.open(str(path)) as container:
            metadata = dict(container.metadata)
            if container.streams.video:
                metadata.update(container.streams.video[0].metadata)
        for key in ("creation_time", "date", "timestamp"):
            value = metadata.get(key)
            if value:
                parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed, f"MP4 {key}"
    except (ImportError, OSError, ValueError, TypeError):
        pass

    # Accept compact and ISO-like timestamp forms.
    text = str(path)
    patterns = (
        (r"(?<!\d)(\d{8})[_-]?(\d{6})(?!\d)", "%Y%m%d%H%M%S"),
        (r"(?<!\d)(\d{4})-(\d{2})-(\d{2})[T_ -](\d{2})[-:]?(\d{2})[-:]?(\d{2})(?!\d)",
         "%Y%m%d%H%M%S"),
    )
    for pattern, fmt in patterns:
        match = re.search(pattern, text)
        if match:
            parsed = datetime.strptime("".join(match.groups()), fmt)
            return parsed.replace(tzinfo=timezone.utc), "path timestamp"

    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc), "file mtime fallback"


def order_pupil_videos(video_paths):
    """Exactly two recordings ordered chronologically as pre then post."""

    paths = [Path(path) for path in video_paths]
    if len(paths) != 2:
        raise ValueError(f"Expected exactly two pupil MP4s, found {len(paths)}.")
    stamped = [(video_recording_timestamp(path), path) for path in paths]
    stamped.sort(key=lambda item: item[0][0])
    if stamped[0][0][0] == stamped[1][0][0]:
        raise ValueError("The two pupil videos have identical timestamps; pre/post is ambiguous.")
    return [item[1] for item in stamped], [item[0] for item in stamped]


def stage_pupil_videos(video_paths, stage_dir):
    """Copy MP4s and converter metadata once to a verified local cache."""

    import json
    import os
    import shutil

    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    staged = []
    for index, source in enumerate(map(Path, video_paths)):
        source_stat = source.stat()
        destination_dir = stage_dir / f"recording_{index + 1}"
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / source.name
        manifest_path = destination.with_suffix(destination.suffix + ".source.json")
        expected = {
            "source": str(source), "size": source_stat.st_size,
            "mtime_ns": source_stat.st_mtime_ns,
        }
        hit = False
        if destination.exists() and manifest_path.exists():
            try:
                recorded = json.loads(manifest_path.read_text())
                hit = recorded == expected and destination.stat().st_size == source_stat.st_size
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        if not hit:
            temporary = destination.with_suffix(destination.suffix + ".part")
            shutil.copyfile(source, temporary)
            if temporary.stat().st_size != source_stat.st_size:
                raise IOError(
                    f"Incomplete pupil-video copy: {temporary.stat().st_size} of "
                    f"{source_stat.st_size} bytes for {source}."
                )
            os.replace(temporary, destination)
            manifest_path.write_text(json.dumps(expected, indent=2))

        # Preserve converter metadata beside the staged MP4, since its
        # Micro-Manager StartTime is the acquisition-time authority.
        metadata = source.with_name(f"{source.stem}_metadata.json")
        if metadata.exists():
            shutil.copyfile(metadata, destination_dir / metadata.name)
        frametimes = find_frametimes_csv(source)
        if frametimes is not None:
            shutil.copyfile(frametimes, destination_dir / frametimes.name)
        staged.append(destination)
    return staged


def stage_sync_file(source, stage_dir):
    """Copy a behavior sync HDF5 to local scratch with cache validation."""

    import json
    import os
    import shutil

    source = Path(source)
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    destination = stage_dir / source.name
    manifest_path = destination.with_suffix(destination.suffix + ".source.json")
    stat = source.stat()
    expected = {"source": str(source), "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns}
    hit = False
    if destination.exists() and manifest_path.exists():
        try:
            hit = (json.loads(manifest_path.read_text()) == expected and
                   destination.stat().st_size == stat.st_size)
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    if not hit:
        temporary = destination.with_suffix(destination.suffix + ".part")
        shutil.copyfile(source, temporary)
        if temporary.stat().st_size != stat.st_size:
            raise IOError(
                f"Incomplete sync copy: {temporary.stat().st_size} of "
                f"{stat.st_size} bytes for {source}."
            )
        os.replace(temporary, destination)
        manifest_path.write_text(json.dumps(expected, indent=2))
    return destination


def validate_alignment_counts(video_path, camera_samples, frametimes_path=None) -> int:
    """Hard preflight against the camera clock before image processing."""

    n_video = count_video_frames(video_path)
    n_pulse = int(np.asarray(camera_samples).size)
    n_csv = None if frametimes_path is None else count_csv_frames(frametimes_path)
    counts_match = n_video == n_pulse and (n_csv is None or n_csv == n_video)
    if not counts_match:
        csv_detail = "" if n_csv is None else f", frametimes.csv rows={n_csv}"
        raise ValueError(
            "Camera alignment failed before segmentation: "
            f"decoded video={n_video}, cameraFrameSync pulses={n_pulse}"
            f"{csv_detail}. A mismatch indicates dropped or "
            "extra frames; no pupil output was written."
        )
    return n_video


def find_frametimes_csv(video_path):
    """Return converter-exported Micro-Manager frame times beside an MP4."""

    video_path = Path(video_path)
    candidates = [
        video_path.with_name(f"{video_path.stem}_frametimes.csv"),
        *sorted(video_path.parent.glob("*frametimes*.csv")),
    ]
    return next((path for path in dict.fromkeys(candidates) if path.exists()), None)


def camera_frame_times(video_paths, video_stamps, camera_blocks, sync, counts,
                       *, frametimes_paths=None, max_endpoint_error_s=0.25):
    """Resolve one timestamp per decoded frame, with a validated CSV fallback.

    Exact frame/pulse matches retain the acquisition-clock pulses. A count
    mismatch uses Micro-Manager elapsed times anchored by its absolute
    ``StartTime``; endpoint agreement with the acquisition clock is required.
    """

    if frametimes_paths is None:
        frametimes_paths = [find_frametimes_csv(path) for path in video_paths]
    if len(counts) != len(video_paths) or len(camera_blocks) != len(video_paths):
        raise ValueError("Video, count, and camera-block lengths differ.")
    times, methods, start_errors, end_errors, csv_paths = [], [], [], [], []
    for path, stamp, block, count, csv_path in zip(
            video_paths, video_stamps, camera_blocks, counts, frametimes_paths):
        pulse_time = np.asarray(block, dtype=float) / sync.rate_hz
        delta = int(len(block) - count)
        if delta == 0:
            frame_time = pulse_time
            method = "camera_pulses"
            start_error = end_error = 0.0
        else:
            if csv_path is None:
                raise ValueError(
                    f"{Path(path).name}: {count:,} frames and {len(block):,} pulses; "
                    "no frametimes.csv is available for recovery."
                )
            elapsed_ms = np.loadtxt(
                csv_path, delimiter=",", skiprows=1, usecols=1, ndmin=1
            )
            if len(elapsed_ms) != count:
                raise ValueError(
                    f"{Path(csv_path).name}: {len(elapsed_ms):,} rows but "
                    f"{count:,} decoded frames."
                )
            stamp_dt = stamp[0]
            sync_start = sync.start_time
            if sync_start.tzinfo is None and stamp_dt.tzinfo is not None:
                sync_start = sync_start.replace(tzinfo=stamp_dt.tzinfo)
            origin_s = stamp_dt.timestamp() - sync_start.timestamp()
            frame_time = origin_s + np.asarray(elapsed_ms, float) / 1000.0
            start_error = float(frame_time[0] - pulse_time[0])
            end_error = float(frame_time[-1] - pulse_time[-1])
            if max(abs(start_error), abs(end_error)) > max_endpoint_error_s:
                raise ValueError(
                    f"{Path(path).name}: frametimes disagree with camera clock "
                    f"at endpoints by {start_error:+.3f}/{end_error:+.3f} s."
                )
            # Millisecond timestamps can tie during a camera disturbance.
            # Preserve their ordering with a negligible monotonic epsilon.
            frame_time = np.maximum.accumulate(
                frame_time + np.arange(count, dtype=float) * 1e-9
            )
            method = "micromanager_frametimes"
        times.append(frame_time)
        methods.append(method)
        start_errors.append(start_error)
        end_errors.append(end_error)
        csv_paths.append(None if csv_path is None else str(csv_path))
    return np.concatenate(times), {
        "methods": methods,
        "pulse_count_deltas": [int(len(block) - count)
                               for block, count in zip(camera_blocks, counts)],
        "start_error_s": start_errors,
        "end_error_s": end_errors,
        "frametimes_paths": csv_paths,
    }


def _otsu(image: np.ndarray) -> float:
    """Otsu threshold for an 8-bit ROI (implemented here to keep I/O lean)."""

    values = np.asarray(image, dtype=np.uint8)
    hist = np.bincount(values.ravel(), minlength=256).astype(float)
    if hist.sum() == 0:
        return np.nan
    p = hist / hist.sum()
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    total = mu[-1]
    denom = omega * (1.0 - omega)
    score = np.zeros(256)
    good = denom > 0
    score[good] = (total * omega[good] - mu[good]) ** 2 / denom[good]
    return float(np.argmax(score))


def segment_bright_pupil(frame: np.ndarray, *, roi=None, bright_percentile=97.0,
                         threshold_offset=0.0):
    """Bright-tail adaptive threshold, hole-fill, and largest component."""

    from scipy import ndimage

    image = np.asarray(frame)
    y0, y1, x0, x1 = roi or (0, image.shape[0], 0, image.shape[1])
    crop = image[y0:y1, x0:x1]
    threshold = float(np.clip(_otsu(crop) + threshold_offset, 0, 255))
    seed_threshold = max(
        threshold, float(np.percentile(crop, bright_percentile))
    )
    # Blank/partial endpoint frames can contain a single encoded stripe. They
    # do not have enough bright-tail separation to support segmentation.
    if np.percentile(crop, 99) - np.percentile(crop, 10) < 5:
        return np.zeros(image.shape, dtype=bool), threshold
    # Hysteresis segmentation: the upper threshold identifies a confidently
    # bright pupil core; Otsu supplies the dimmer outer boundary. This avoids
    # forcing every pupil, large or small, to occupy the same top 3% of pixels.
    low = crop > threshold
    labels, count = ndimage.label(low)
    mask = np.zeros(image.shape, dtype=bool)
    if count:
        seeds = crop >= seed_threshold
        seed_counts = np.bincount(labels[seeds].ravel(), minlength=count + 1)
        seed_counts[0] = 0
        if not np.any(seed_counts):
            return mask, threshold
        component = labels == int(np.argmax(seed_counts))
        component = ndimage.binary_fill_holes(component)
        mask[y0:y1, x0:x1] = component
    return mask, threshold


def fit_ellipse_ransac(mask, *, residual_px=2.0, max_trials=200,
                       random_seed=0):
    """Robust fast ellipse fit, with full RANSAC only as fallback."""

    from scipy import ndimage
    from skimage.measure import EllipseModel, LineModelND, ransac

    # scikit-image renamed ``random_state`` to ``rng``. Select by signature;
    # retrying after an arbitrary internal TypeError can mask the real fitting
    # failure with an unsupported-keyword error from the other API version.
    import inspect
    seed_parameter = getattr(fit_ellipse_ransac, "_seed_parameter", None)
    if seed_parameter is None:
        seed_parameter = ("rng" if "rng" in inspect.signature(ransac).parameters
                          else "random_state")
        fit_ellipse_ransac._seed_parameter = seed_parameter
    def seeded(seed):
        if seed_parameter == "rng":
            return {"rng": np.random.default_rng(seed)}
        return {"random_state": seed}

    boundary = mask & ~ndimage.binary_erosion(mask)
    yy, xx = np.nonzero(boundary)
    if xx.size < 12:
        return None
    points_all = np.column_stack((xx, yy))
    points = points_all
    chord_fraction = 0.0

    line_trials = min(int(max_trials), 80)
    # The anatomical lid edge is usually only approximately straight.  This
    # corridor admits a few pixels of shallow curvature; continuity and
    # interior-position gates below still distinguish it from pupil arcs.
    chord_detect_px = max(3.5, 1.75 * float(residual_px))
    try:
        line, line_inliers = ransac(
            points_all, LineModelND, min_samples=2,
            residual_threshold=chord_detect_px,
            max_trials=line_trials, **seeded(random_seed + 104729),
        )
    except (TypeError, ValueError, ArithmeticError):
        line = line_inliers = None
    if line is not None and line_inliers is not None:
        line_points = points_all[line_inliers]
        direction = np.asarray(
            line.direction if hasattr(line, "direction") else line.params[1],
            float,
        )
        projection = line_points @ direction
        span = float(np.ptp(projection)) if projection.size else 0.0
        bbox_diag = float(np.hypot(np.ptp(xx), np.ptp(yy)))
        fraction = float(np.mean(line_inliers))
        ordered_projection = np.sort(projection)
        largest_gap = (float(np.max(np.diff(ordered_projection)))
                       if ordered_projection.size > 1 else np.inf)
        gap_fraction = largest_gap / max(span, np.finfo(float).eps)
        normal = np.array([-direction[1], direction[0]])
        contour_center = np.mean(points_all, axis=0)
        line_center = np.mean(line_points, axis=0)
        normal_extent = float(np.max(np.abs(
            (points_all - contour_center) @ normal
        )))
        normalized_offset = abs(float(
            (line_center - contour_center) @ normal
        )) / max(normal_extent, np.finfo(float).eps)
        if (line_points.shape[0] >= 12 and fraction >= 0.10 and
                span >= 0.25 * bbox_diag and gap_fraction <= 0.30 and
                normalized_offset <= 0.90):
            # Remove a wider curved-band approximation so pixelated, thick, or
            # gently bowed lid edges do not leak into the ellipse fit.
            expanded_residuals = line.residuals(points_all)
            chord_points = np.isfinite(expanded_residuals) & (
                expanded_residuals <= 1.5 * chord_detect_px
            )
            points = points_all[~chord_points]
            chord_fraction = float(np.mean(chord_points))
    if points.shape[0] < 12:
        return None

    def unpack(model):
        if all(hasattr(model, name) for name in ("center", "axis_lengths", "theta")):
            xc, yc = map(float, model.center)
            a, b = map(float, model.axis_lengths)
            theta = float(model.theta)
        else:
            xc, yc, a, b, theta = map(float, model.params)
        return xc, yc, a, b, theta

    def geometry_ok(parameters):
        xc, yc, a, b, theta = parameters
        return (
            all(np.isfinite(parameters)) and
            0 <= xc < mask.shape[1] and 0 <= yc < mask.shape[0] and
            min(a, b) >= 2 and max(a, b) <= 0.5 * max(mask.shape)
        )

    def direct_estimate(data):
        if hasattr(EllipseModel, "from_estimate"):
            model = EllipseModel.from_estimate(data)
            return model if model else None
        model = EllipseModel()
        return model if model.estimate(data) else None

    # Fast path: direct conic estimate followed by three rounds of geometric
    # residual trimming. The floor at residual_px avoids trimming legitimate
    # pixelation from a clean boundary merely because its MAD is tiny.
    direct = direct_estimate(points)
    direct_points = points
    direct_ok = False
    if direct is not None:
        for _ in range(3):
            residuals = direct.residuals(points)
            finite = np.isfinite(residuals)
            if np.sum(finite) < 12:
                break
            median = float(np.median(residuals[finite]))
            mad = float(np.median(np.abs(residuals[finite] - median)))
            cutoff = max(float(residual_px), median + 3.0 * 1.4826 * mad)
            keep = finite & (residuals <= cutoff)
            if np.sum(keep) < 12:
                break
            candidate = direct_estimate(points[keep])
            if candidate is None:
                break
            direct, direct_points = candidate, points[keep]
        parameters = unpack(direct)
        direct_residuals = direct.residuals(points)
        direct_inliers = np.isfinite(direct_residuals) & (
            direct_residuals <= residual_px
        )
        direct_fraction = float(np.sum(direct_inliers) / len(points_all))
        direct_median = (float(np.median(direct_residuals[direct_inliers]))
                         if np.any(direct_inliers) else np.inf)
        direct_ok = (
            geometry_ok(parameters) and direct_fraction >= 0.45 and
            direct_median <= residual_px
        )
    if direct_ok:
        xc, yc, a, b, theta = parameters
        if a >= b:
            major, minor = a, b
        else:
            major, minor = b, a
            theta += np.pi / 2.0
        return {
            "x": xc, "y": yc, "major": major, "minor": minor,
            "theta": theta, "diameter": 2.0 * major,
            "equivalent_diameter": 2.0 * np.sqrt(major * minor),
            "area": np.pi * major * minor,
            "axis_ratio": minor / major if major > 0 else np.nan,
            "inlier_fraction": direct_fraction,
            "chord_fraction": chord_fraction,
            "residual": direct_median, "used_ransac": 0.0,
        }

    # Fast robust fallback. OpenCV fits each candidate in compiled code, while
    # vectorized first-order geometric distances score the complete contour.
    try:
        import cv2
    except ImportError:
        cv2 = None

    def cv_parameters(data):
        if cv2 is None or len(data) < 5:
            return None
        try:
            (xc, yc), (width, height), angle = cv2.fitEllipse(
                np.asarray(data, np.float32).reshape(-1, 1, 2)
            )
        except (cv2.error, ValueError):
            return None
        return (float(xc), float(yc), 0.5 * float(width),
                0.5 * float(height), np.deg2rad(float(angle)))

    def fast_residuals(data, p):
        xc, yc, a, b, theta = p
        ct, st = np.cos(theta), np.sin(theta)
        dx, dy = data[:, 0] - xc, data[:, 1] - yc
        xr, yr = ct * dx + st * dy, -st * dx + ct * dy
        q = np.sqrt((xr / a) ** 2 + (yr / b) ** 2)
        safe_q = np.maximum(q, np.finfo(float).eps)
        gradient = np.sqrt((xr / (a * a * safe_q)) ** 2 +
                           (yr / (b * b * safe_q)) ** 2)
        return np.abs(q - 1.0) / np.maximum(gradient, np.finfo(float).eps)

    parameters = None
    inliers = None
    if cv2 is not None:
        rng = np.random.default_rng(random_seed)
        best_score = (-1, -np.inf)
        for _ in range(int(max_trials)):
            candidate = cv_parameters(points[rng.choice(len(points), 5, replace=False)])
            if candidate is None or not geometry_ok(candidate):
                continue
            candidate_residuals = fast_residuals(points, candidate)
            candidate_inliers = np.isfinite(candidate_residuals) & (
                candidate_residuals <= residual_px
            )
            count = int(np.sum(candidate_inliers))
            median = (float(np.median(candidate_residuals[candidate_inliers]))
                      if count else np.inf)
            score = (count, -median)
            if score > best_score:
                best_score, parameters, inliers = score, candidate, candidate_inliers
        if inliers is not None and np.sum(inliers) >= 5:
            for _ in range(2):
                refined = cv_parameters(points[inliers])
                if refined is None or not geometry_ok(refined):
                    break
                parameters = refined
                refined_residuals = fast_residuals(points, refined)
                inliers = np.isfinite(refined_residuals) & (
                    refined_residuals <= residual_px
                )

    # Keep compatibility with environments without OpenCV.
    if parameters is None or inliers is None or np.sum(inliers) < 5:
        try:
            model, inliers = ransac(
                points, EllipseModel, min_samples=5, residual_threshold=residual_px,
                max_trials=max_trials, **seeded(random_seed),
            )
        except (TypeError, ValueError, ArithmeticError):
            return None
        if model is None or inliers is None or not np.any(inliers):
            return None
        parameters = unpack(model)
        residual = float(np.median(model.residuals(points[inliers])))
    else:
        residual = float(np.median(fast_residuals(points[inliers], parameters)))
    xc, yc, a, b, theta = parameters
    if not geometry_ok((xc, yc, a, b, theta)):
        return None
    if a >= b:
        major, minor = a, b
    else:
        major, minor = b, a
        theta += np.pi / 2.0
    return {
        "x": xc, "y": yc, "major": major, "minor": minor,
        "theta": theta, "diameter": 2.0 * major,
        "equivalent_diameter": 2.0 * np.sqrt(major * minor),
        "area": np.pi * major * minor,
        "axis_ratio": minor / major if major > 0 else np.nan,
        # Fraction is against the complete boundary, not only the points left
        # after chord rejection, so clipped fits do not look artificially good.
        "inlier_fraction": float(np.sum(inliers) / len(points_all)),
        "chord_fraction": chord_fraction, "residual": residual,
        "used_ransac": 1.0,
    }


def filter_concave_protrusions(mask, fit, *, margin_px=6.0,
                               margin_fraction=0.06,
                               min_removed_fraction=0.02,
                               max_removed_fraction=0.20):
    """Remove material exterior tails while preserving ordinary pupil corners."""

    from scipy import ndimage

    mask = np.asarray(mask, bool)
    if fit is None or not np.any(mask):
        return mask, 0.0
    margin = max(float(margin_px), float(margin_fraction) * fit["major"])
    yy, xx = np.indices(mask.shape)
    ct, st = np.cos(fit["theta"]), np.sin(fit["theta"])
    dx, dy = xx - fit["x"], yy - fit["y"]
    xr, yr = ct * dx + st * dy, -st * dx + ct * dy
    a, b = float(fit["major"]), float(fit["minor"])
    q = np.sqrt((xr / a) ** 2 + (yr / b) ** 2)
    safe_q = np.maximum(q, np.finfo(float).eps)
    gradient = np.sqrt((xr / (a * a * safe_q)) ** 2 +
                       (yr / (b * b * safe_q)) ** 2)
    outside_distance = np.maximum(q - 1.0, 0.0) / np.maximum(
        gradient, np.finfo(float).eps
    )
    cleaned = mask & (outside_distance <= margin)
    labels, count = ndimage.label(cleaned)
    if count:
        sizes = np.bincount(labels.ravel())
        cleaned = labels == (np.argmax(sizes[1:]) + 1)
    removed = 1.0 - float(np.sum(cleaned)) / max(int(np.sum(mask)), 1)
    if (removed < min_removed_fraction or removed > max_removed_fraction or
            np.sum(cleaned) < 20):
        return mask, 0.0
    return ndimage.binary_fill_holes(cleaned), removed


def fit_pupil_mask(mask, config=PupilConfig(), *, random_seed=0):
    """Fit once, conservatively remove exterior protrusions, then refit."""

    first = fit_ellipse_ransac(
        mask, residual_px=config.ransac_residual_px,
        max_trials=config.ransac_trials, random_seed=random_seed,
    )
    cleaned, removed = filter_concave_protrusions(
        mask, first, margin_px=config.consensus_margin_px,
        margin_fraction=config.consensus_margin_fraction,
        min_removed_fraction=config.min_concavity_fraction,
        max_removed_fraction=config.max_concavity_fraction,
    )
    fit = first if removed == 0.0 else fit_ellipse_ransac(
        cleaned, residual_px=config.ransac_residual_px,
        max_trials=config.ransac_trials, random_seed=random_seed,
    )
    if fit is not None:
        fit["concavity_fraction"] = removed
    return cleaned, fit


def classify_frames(metrics: dict, camera_time_s, config=PupilConfig(), active=None):
    """Classify blinks and fits that must be excluded from the pupil trace."""

    area = metrics["area"]
    ratio = metrics["axis_ratio"]
    resid = metrics["residual"]
    inliers = metrics["inlier_fraction"]
    diameter = metrics["diameter"]
    finite_area = area[np.isfinite(area)]
    typical_area = np.nanmedian(finite_area) if finite_area.size else np.nan
    collapse = (area < 0.25 * typical_area) | (ratio < config.min_axis_ratio)
    fit_bad = ((inliers < config.min_inlier_fraction) |
               (resid > config.max_residual_px) | ~np.isfinite(diameter))
    dt = np.diff(camera_time_s, prepend=np.nan)
    rate = np.abs(np.diff(diameter, prepend=np.nan)) / dt
    jump = rate > config.max_diameter_rate_px_s
    # Two independent temporal/shape symptoms constitute a blink. Poor ellipse
    # fits are invalid independently: keeping them creates plausible-looking
    # diameter values from a failed fit.
    active = np.ones(area.shape, dtype=bool) if active is None else np.asarray(active, bool)
    blink = active & ((collapse.astype(int) + fit_bad.astype(int) + jump.astype(int)) >= 2)
    clipped = active & fit_bad & ~blink
    return blink, clipped, rate


def pupil_qc_frame_indices(values, active, counts, config=PupilConfig(),
                           n_excluded=3, n_accepted=9):
    """Three worst fit exclusions plus diameter-diverse accepted frames."""
    active = np.asarray(active, bool)
    fit_bad = active & (
        (values["inlier_fraction"] < config.min_inlier_fraction)
        | (values["residual"] > config.max_residual_px)
        | ~np.isfinite(values["diameter"])
    )
    score = np.nan_to_num(values["residual"], nan=np.inf) + 5 * (
        1 - np.nan_to_num(values["inlier_fraction"])
    )
    bad = np.flatnonzero(fit_bad)
    excluded = bad[np.argsort(score[bad])[-int(n_excluded):]][::-1]

    accepted_mask = active & ~fit_bad & np.isfinite(values["diameter"])
    accepted = []
    offsets = np.r_[0, np.cumsum(np.asarray(counts, dtype=int))]
    n_video = max(len(counts), 1)
    allocations = [int(n_accepted) // n_video] * n_video
    for index in range(int(n_accepted) % n_video):
        allocations[index] += 1
    for start, stop, number in zip(offsets[:-1], offsets[1:], allocations):
        candidates = np.flatnonzero(accepted_mask[start:stop]) + start
        if not len(candidates) or number <= 0:
            continue
        ordered = candidates[np.argsort(values["diameter"][candidates])]
        positions = np.linspace(0, len(ordered) - 1, min(number, len(ordered)))
        accepted.extend(map(int, ordered[np.rint(positions).astype(int)]))
    return np.asarray(excluded, dtype=int), np.asarray(accepted, dtype=int)


def _nearest(source_t, target_t):
    right = np.searchsorted(source_t, target_t)
    right = np.clip(right, 0, len(source_t) - 1)
    left = np.maximum(right - 1, 0)
    choose_left = np.abs(target_t - source_t[left]) <= np.abs(source_t[right] - target_t)
    return np.where(choose_left, left, right)


def _analyze_pupil_frame(task):
    """Multiprocessing-safe exact per-frame segmentation and RANSAC worker."""

    index, frame, config = task
    y0, y1, x0, x1 = config.roi or (0, frame.shape[0], 0, frame.shape[1])
    illumination_peak = float(np.percentile(frame[y0:y1, x0:x1], 99.9))
    if illumination_peak < config.min_illumination_peak:
        return index, np.nan, {
            "illumination_peak": illumination_peak,
            "illumination_active": 0.0,
        }
    mask, threshold = segment_bright_pupil(
        frame, roi=config.roi, bright_percentile=config.bright_percentile,
        threshold_offset=config.threshold_offset,
    )
    _, fit = fit_pupil_mask(
        mask, config, random_seed=config.random_seed + index
    )
    if fit is None:
        fit = {}
    fit["illumination_peak"] = illumination_peak
    fit["illumination_active"] = 1.0
    return index, threshold, fit


def _checkpoint_signature(video_paths, counts, sync_path, config):
    """Identity of inputs and parameters that make a checkpoint reusable."""

    import json
    from dataclasses import asdict

    files = []
    for path in [*map(Path, video_paths), Path(sync_path)]:
        manifest_path = path.with_suffix(path.suffix + ".source.json")
        source = None
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
                source = (str(manifest["source"]), int(manifest["size"]),
                          int(manifest["mtime_ns"]))
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                pass
        if source is None:
            stat = path.stat()
            source = (str(path), int(stat.st_size), int(stat.st_mtime_ns))
        files.append(source)
    return json.dumps(
        {"fitter_version": PUPIL_FITTER_VERSION,
         "files": files, "counts": list(map(int, counts)),
         "config": asdict(config)}, sort_keys=True,
    )


def _save_pupil_checkpoint(path, values, completed, signature):
    """Atomically save numeric progress locally."""

    import os

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as handle:
        np.savez(
            handle, completed=np.asarray(completed, np.uint8),
            signature=np.asarray(signature),
            **{f"value_{name}": array for name, array in values.items()},
        )
    os.replace(temporary, path)


def _load_pupil_checkpoint(path, values, signature):
    """Restore matching numeric progress, returning the completed mask."""

    path = Path(path)
    if not path.exists():
        return np.zeros(next(iter(values.values())).shape, dtype=bool)
    with np.load(path, allow_pickle=False) as saved:
        if str(saved["signature"].item()) != signature:
            raise ValueError(
                f"Checkpoint {path} belongs to different videos, sync data, "
                "or pupil parameters. Move it aside or use resume=False."
            )
        completed = saved["completed"].astype(bool)
        for name in values:
            values[name][:] = saved[f"value_{name}"]
    return completed


def _atomic_publish(source, destination):
    """Copy a verified local artifact into place with an atomic final rename."""

    import os
    import shutil

    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    shutil.copyfile(source, temporary)
    if temporary.stat().st_size != source.stat().st_size:
        raise IOError(f"Incomplete publish of {source} to {destination}.")
    os.replace(temporary, destination)
    return destination


def _verify_local_pupil_output(path, report):
    """Refuse to publish a local HDF5 unless its essential arrays reopen."""

    from .h5io import open_h5

    expected = (report["n_trial"], report["n_frame"])
    with open_h5(path) as handle:
        for name in ("pupil/diameter", "pupil/diameter_masked",
                     "pupil/inlier_fraction", "trials/acq_id"):
            if name not in handle:
                raise IOError(f"Local pupil output is missing /{name}: {path}")
        if handle["pupil/diameter"].shape != expected:
            raise IOError(
                f"Local pupil output has diameter shape "
                f"{handle['pupil/diameter'].shape}, expected {expected}."
            )
        if handle["trials/acq_id"].shape != (report["n_trial"],):
            raise IOError(f"Local pupil output has a malformed acq_id axis: {path}")
    return Path(path)


def pupil_from_round(video_path, round_path, *, sync_path=None,
                     frametimes_path=None, out_dir=None, config=PupilConfig(),
                     save=True, validate_counts=True, validated_counts=None,
                     workers=None, batch_size=256, checkpoint_every=5000,
                     checkpoint_dir=None, resume=True, acquisition_indices=None,
                     _metadata=None):
    """Extract, align, save, and plot pupil diameter for one processing round."""

    from .h5io import open_h5
    from .respiration import find_behavior_sync
    from .sync import (frame_onset_samples, group_frames_into_acquisitions,
                       open_sync)

    if _metadata is None:
        round_path = Path(round_path)
        with open_h5(round_path) as f:
            required = ("trials/acq_id", "trials/trial_id", "trials/odor_id",
                        "trials/odor_on_frame", "trials/odor_off_frame")
            missing = [key for key in required if key not in f]
            if missing:
                raise ValueError(f"{round_path.name} lacks required datasets {missing}.")
            trial = {key.split("/")[-1]: f[key][:] for key in required}
            if "trials/state" in f:
                trial["state"] = f["trials/state"][:]
            n_trial = len(trial["acq_id"])
            frame_rate = float(f.attrs["frame_rate"])
            n_pre = int(f.attrs["n_pre"])
            exp_name = str(f.attrs.get("exp_name", round_path.stem))
        round_name = round_path.name
        sync_path = Path(sync_path) if sync_path else find_behavior_sync(round_path.parents[2])
    else:
        if sync_path is None:
            raise ValueError("sync_path is required for round-independent pupil extraction.")
        trial = _metadata["trial"]
        n_trial = len(trial["acq_id"])
        frame_rate = float(_metadata["frame_rate"])
        n_pre = int(_metadata["n_pre"])
        exp_name = str(_metadata["exp_name"])
        round_name = f"{exp_name} (no round)"
        round_path = None
        sync_path = Path(sync_path)
    sync = open_sync(sync_path)
    camera_samples_all = frame_onset_samples(sync, channel="cameraFrameSync")
    if isinstance(video_path, (str, Path)):
        video_paths = [Path(video_path)]
        video_stamps = [(video_recording_timestamp(video_paths[0]))]
        camera_blocks = [camera_samples_all]
    else:
        video_paths, video_stamps = order_pupil_videos(video_path)
        camera_blocks = group_frames_into_acquisitions(
            camera_samples_all, rate_hz=sync.rate_hz
        )
        if len(camera_blocks) != len(video_paths):
            raise ValueError(
                f"Found {len(video_paths)} pupil videos but cameraFrameSync has "
                f"{len(camera_blocks)} contiguous recording blocks."
            )
    if frametimes_path is not None and len(video_paths) != 1:
        raise ValueError("frametimes_path is only supported for a single legacy video.")
    frametimes_paths = ([Path(frametimes_path)] if frametimes_path is not None
                        else [find_frametimes_csv(path) for path in video_paths])
    if validate_counts:
        counts = []
        for video_index, (path, csv_path) in enumerate(
                zip(video_paths, frametimes_paths), start=1):
            count = count_video_frames(
                path,
                progress_desc=f"Pupil preflight {video_index}/{len(video_paths)}",
            )
            if csv_path is not None and count_csv_frames(csv_path) != count:
                raise ValueError(
                    f"{Path(path).name}: decoded frame count does not match "
                    f"{Path(csv_path).name}."
                )
            counts.append(count)
    else:
        if validated_counts is None:
            raise ValueError(
                "validated_counts is required when validate_counts=False; "
                "this prevents an unchecked extraction from masquerading as preflighted."
            )
        counts = [int(value) for value in validated_counts]
        if len(counts) != len(video_paths):
            raise ValueError(
                f"Previously validated {len(counts)} video counts for "
                f"{len(video_paths)} videos."
            )
    n_camera = int(sum(counts))
    camera_time_s, alignment_qc = camera_frame_times(
        video_paths, video_stamps, camera_blocks, sync, counts,
        frametimes_paths=frametimes_paths,
    )

    two_p_samples = frame_onset_samples(sync)
    blocks = group_frames_into_acquisitions(two_p_samples, rate_hz=sync.rate_hz)
    if acquisition_indices is not None:
        blocks = [blocks[int(index)] for index in acquisition_indices]
    if len(blocks) != n_trial:
        raise ValueError(f"2p clock has {len(blocks)} acquisitions but round has {n_trial} trials.")
    widths = {len(block) for block in blocks}
    if len(widths) != 1:
        raise ValueError(f"2p acquisitions differ in length: {sorted(widths)}.")
    n_frame = widths.pop()

    imaging_active = np.zeros(n_camera, dtype=bool)
    for block in blocks:
        lo = np.searchsorted(camera_time_s, block[0] / sync.rate_hz, side="left")
        hi = np.searchsorted(camera_time_s, block[-1] / sync.rate_hz, side="right")
        imaging_active[lo:hi] = True

    names = ("diameter", "equivalent_diameter", "area", "axis_ratio",
             "inlier_fraction", "chord_fraction",
             "concavity_fraction", "residual",
             "used_ransac",
             "x", "y", "major", "minor", "theta", "threshold",
             "illumination_peak", "illumination_active")
    values = {name: np.full(n_camera, np.nan) for name in names}
    import os
    import time
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp

    workers = (max(1, min(8, (os.cpu_count() or 2) - 2))
               if workers is None else max(1, int(workers)))
    checkpoint_path = None
    completed = np.zeros(n_camera, dtype=bool)
    if checkpoint_dir is not None:
        checkpoint_path = Path(checkpoint_dir) / f"{exp_name}_pupil_checkpoint.npz"
        signature = _checkpoint_signature(video_paths, counts, sync_path, config)
        if resume:
            completed = _load_pupil_checkpoint(checkpoint_path, values, signature)
        else:
            completed[:] = False
    active_total = int(imaging_active.sum())
    done = int(np.sum(completed & imaging_active))
    started = time.monotonic()
    last_checkpoint_done = done
    print(
        f"Pupil fitting: {active_total:,} illuminated frames, {workers} worker(s), "
        f"{done:,} resumed."
    )

    def accept(results):
        nonlocal done, last_checkpoint_done
        accepted = 0
        for i, threshold, fit in results:
            values["threshold"][i] = threshold
            if fit:
                for name, value in fit.items():
                    values[name][i] = value
            completed[i] = True
            done += 1
            accepted += 1
        if progress is not None:
            progress.update(accepted)
        elif accepted and (done == active_total or done % 1000 < accepted):
            elapsed = max(time.monotonic() - started, 1e-6)
            rate = max((done - resumed_done) / elapsed, 1e-9)
            eta_min = (active_total - done) / rate / 60
            print(
                f"  {done:,}/{active_total:,} ({done/active_total:.1%}) · "
                f"{rate:.1f} frames/s · ETA {eta_min:.1f} min", flush=True,
            )
        if (checkpoint_path is not None and
                done - last_checkpoint_done >= checkpoint_every):
            _save_pupil_checkpoint(checkpoint_path, values, completed, signature)
            last_checkpoint_done = done

    resumed_done = done
    try:
        from tqdm.auto import tqdm
        progress = tqdm(
            total=active_total, initial=done, desc=f"Pupil {exp_name}",
            unit="frame", leave=False, dynamic_ncols=True,
        )
    except ImportError:
        progress = None
    offset = 0
    executor = None
    try:
        if workers > 1:
            executor = ProcessPoolExecutor(
                max_workers=workers, mp_context=mp.get_context("spawn")
            )
        for path, expected in zip(video_paths, counts):
            decoded = 0
            batch = []
            for local_i, frame in enumerate(iter_gray_frames(path)):
                i = offset + local_i
                decoded += 1
                if not imaging_active[i] or completed[i]:
                    continue
                batch.append((i, frame, config))
                if len(batch) >= batch_size:
                    results = (map(_analyze_pupil_frame, batch) if executor is None
                               else executor.map(_analyze_pupil_frame, batch,
                                                 chunksize=8))
                    accept(results)
                    batch = []
            if batch:
                results = (map(_analyze_pupil_frame, batch) if executor is None
                           else executor.map(_analyze_pupil_frame, batch,
                                             chunksize=8))
                accept(results)
            if decoded != expected:
                raise RuntimeError(
                    f"{path.name} changed during extraction: expected "
                    f"{expected} frames, decoded {decoded}."
                )
            offset += decoded
    except BaseException:
        if checkpoint_path is not None:
            _save_pupil_checkpoint(checkpoint_path, values, completed, signature)
            print(f"Checkpoint saved after interruption: {checkpoint_path}")
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        if progress is not None:
            progress.close()
    if checkpoint_path is not None:
        _save_pupil_checkpoint(checkpoint_path, values, completed, signature)

    illumination_active = values["illumination_active"] == 1
    measurement_active = imaging_active & illumination_active
    blink, clipped, diameter_rate = classify_frames(
        values, camera_time_s, config, active=measurement_active
    )
    values["diameter_rate"] = diameter_rate
    values["blink"] = blink
    values["clipped"] = clipped
    values["imaging_active"] = imaging_active
    values["illumination_active"] = illumination_active
    values["diameter_masked"] = values["diameter"].copy()
    values["diameter_masked"][blink | clipped] = np.nan
    values["equivalent_diameter_masked"] = values["equivalent_diameter"].copy()
    values["equivalent_diameter_masked"][blink | clipped] = np.nan

    # Re-read once (fast relative to fitting) for exactly three excluded fits
    # and accepted examples spanning each video's fitted diameter range.
    excluded_indices, accepted_indices = pupil_qc_frame_indices(
        values, measurement_active, counts, config
    )
    target_indices = set(map(int, np.r_[excluded_indices, accepted_indices]))
    example_frames = {}
    offset = 0
    for path, expected in zip(video_paths, counts):
        local_targets = {i - offset for i in target_indices if offset <= i < offset + expected}
        if local_targets:
            for local_i, frame in enumerate(iter_gray_frames(path)):
                if local_i not in local_targets:
                    continue
                i = offset + local_i
                mask, _ = segment_bright_pupil(
                    frame, roi=config.roi,
                    bright_percentile=config.bright_percentile,
                    threshold_offset=config.threshold_offset,
                )
                cleaned, _ = fit_pupil_mask(
                    mask, config, random_seed=config.random_seed + i
                )
                example_frames[i] = (frame.copy(), mask, cleaned)
                if len(example_frames) == len(target_indices):
                    break
        offset += expected

    flat_2p = np.concatenate(blocks) / sync.rate_hz
    nearest = _nearest(camera_time_s, flat_2p)
    nearest_error = np.abs(camera_time_s[nearest] - flat_2p)
    camera_period = float(np.median(np.concatenate([
        np.diff(block / sync.rate_hz) for block in camera_blocks
        if len(block) > 1
    ])))
    alignment_valid = nearest_error <= 2 * camera_period
    def grid(array):
        return np.asarray(array)[nearest].reshape(n_trial, n_frame)
    aligned = {name: grid(value) for name, value in values.items()}
    valid_grid = alignment_valid.reshape(n_trial, n_frame)
    error_grid = nearest_error.reshape(n_trial, n_frame)
    for name, array in aligned.items():
        if array.dtype.kind == "b":
            array[~valid_grid] = False
        else:
            array[~valid_grid] = np.nan
    aligned["alignment_valid"] = valid_grid
    aligned["nearest_frame_error_s"] = error_grid
    coverage_fraction = np.mean(valid_grid, axis=1)
    masked_fraction = np.mean(~np.isfinite(aligned["diameter_masked"]), axis=1)
    blink_fraction = np.mean(aligned["blink"], axis=1)
    clipped_fraction = np.mean(aligned["clipped"], axis=1)
    flagged = masked_fraction > config.max_bad_fraction
    report = dict(round=round_name, sync=sync_path.name, exp_name=exp_name,
                  n_trial=n_trial, n_frame=n_frame, frame_rate=frame_rate,
                  n_pre=n_pre, n_camera=n_camera, config=config, **trial,
                  video_paths=video_paths, video_counts=np.asarray(counts),
                  video_timestamps=[stamp[0].isoformat() for stamp in video_stamps],
                  video_timestamp_sources=[stamp[1] for stamp in video_stamps],
                  camera_alignment=alignment_qc,
                  camera_imaging_fraction=float(np.mean(imaging_active)),
                  **aligned, masked_fraction=masked_fraction,
                  coverage_fraction=coverage_fraction,
                  blink_fraction=blink_fraction,
                  clipped_fraction=clipped_fraction, flagged=flagged,
                  camera_values=values, camera_time_s=camera_time_s,
                  worst_frame_indices=np.asarray(excluded_indices),
                  accepted_frame_indices=np.asarray(accepted_indices),
                  example_frames=example_frames)
    if save:
        if out_dir is None and round_path is None:
            raise ValueError("out_dir is required for round-independent extraction.")
        out_dir = Path(out_dir) if out_dir else round_path.parent / "aux"
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = round_path.stem if round_path is not None else exp_name
        if checkpoint_dir is not None:
            local_output = Path(checkpoint_dir) / "completed"
            local_output.mkdir(parents=True, exist_ok=True)
        else:
            local_output = out_dir
        local_h5 = _write_pupil(local_output / f"{stem}_pupil.h5", report)
        local_figure = pupil_figure(report, local_output / f"{stem}_pupil.png")
        _verify_local_pupil_output(local_h5, report)
        if not local_figure.exists() or local_figure.stat().st_size == 0:
            raise IOError(f"Local pupil QC figure was not written: {local_figure}")
        final_h5 = out_dir / local_h5.name
        final_figure = out_dir / local_figure.name
        if local_output != out_dir:
            _atomic_publish(local_h5, final_h5)
            _atomic_publish(local_figure, final_figure)
        report["h5"] = str(final_h5)
        report["figure"] = str(final_figure)
        # Keep the checkpoint until both verified local artifacts have been
        # published. A share outage during copy therefore never loses compute.
        if checkpoint_path is not None and checkpoint_path.exists():
            checkpoint_path.unlink()
    return report


def extract_pupil(
    video_paths, sync_path, *, acq_ids, odor_ids, states,
    odor_on_frames, odor_off_frames, frame_rate, exp_name,
    trial_ids=None, out_dir, config=PupilConfig(), save=True,
    validate_counts=True, validated_counts=None,
    workers=None, batch_size=256, checkpoint_every=5000,
    checkpoint_dir=None, resume=True,
    acquisition_indices=None,
):
    """Round-independent pupil extraction on the acquisition frame grid."""

    acq_ids = np.asarray(acq_ids)
    n_trial = len(acq_ids)
    trial = {
        "acq_id": acq_ids,
        "trial_id": (np.arange(n_trial) if trial_ids is None
                     else np.asarray(trial_ids)),
        "odor_id": np.asarray(odor_ids),
        "state": np.asarray(states),
        "odor_on_frame": np.asarray(odor_on_frames),
        "odor_off_frame": np.asarray(odor_off_frames),
    }
    wrong = {key: len(value) for key, value in trial.items() if len(value) != n_trial}
    if wrong:
        raise ValueError(f"Trial metadata lengths do not match {n_trial} acq_ids: {wrong}.")
    n_pre = int(np.median(trial["odor_on_frame"]))
    return pupil_from_round(
        video_paths, None, sync_path=sync_path, out_dir=out_dir,
        config=config, save=save, validate_counts=validate_counts,
        validated_counts=validated_counts,
        workers=workers, batch_size=batch_size,
        checkpoint_every=checkpoint_every,
        checkpoint_dir=checkpoint_dir, resume=resume,
        acquisition_indices=acquisition_indices,
        _metadata={"trial": trial, "frame_rate": frame_rate,
                   "n_pre": n_pre, "exp_name": exp_name},
    )


def _write_pupil(path, report):
    """Write the aux file with raw/masked traces and per-frame quality."""

    from .h5io import open_h5
    with open_h5(path, "w") as f:
        f.attrs["description"] = "Bright-pupil ellipse measurements on the round frame grid."
        for key in ("round", "sync", "exp_name", "n_trial", "n_frame", "frame_rate", "n_pre", "n_camera"):
            f.attrs[key] = report[key]
        f.attrs["roi_y0_y1_x0_x1"] = np.asarray(
            report["config"].roi or (-1, -1, -1, -1), dtype=np.int32
        )
        f.attrs["threshold_offset"] = report["config"].threshold_offset
        f.attrs["bright_percentile"] = report["config"].bright_percentile
        f.attrs["min_illumination_peak"] = report["config"].min_illumination_peak
        f.attrs["videos_pre_post"] = np.asarray([p.name for p in report["video_paths"]], dtype="S")
        f.attrs["video_timestamps_pre_post"] = np.asarray(report["video_timestamps"], dtype="S")
        alignment = report["camera_alignment"]
        f.attrs["camera_alignment_methods"] = np.asarray(alignment["methods"], dtype="S")
        f.attrs["camera_pulse_count_deltas"] = np.asarray(
            alignment["pulse_count_deltas"], dtype=np.int32
        )
        f.attrs["camera_alignment_start_error_s"] = np.asarray(alignment["start_error_s"])
        f.attrs["camera_alignment_end_error_s"] = np.asarray(alignment["end_error_s"])
        f.attrs["frametimes_paths"] = np.asarray(
            [value or "" for value in alignment["frametimes_paths"]], dtype="S"
        )
        p = f.create_group("pupil")
        for key in ("diameter", "diameter_masked", "equivalent_diameter",
                    "equivalent_diameter_masked", "area", "axis_ratio",
                    "inlier_fraction", "chord_fraction", "concavity_fraction",
                    "residual", "used_ransac", "threshold", "illumination_peak",
                    "illumination_active", "blink",
                    "clipped", "diameter_rate", "x", "y", "major", "minor",
                    "theta", "imaging_active", "alignment_valid",
                    "nearest_frame_error_s"):
            p.create_dataset(key, data=report[key], compression="gzip")
        p["diameter"].attrs["units"] = "pixels"
        p["diameter"].attrs["description"] = "Full fitted major-axis length."
        p["equivalent_diameter"].attrs["units"] = "pixels"
        p["equivalent_diameter"].attrs["description"] = (
            "Area-equivalent ellipse diameter: 2 * sqrt(major * minor)."
        )
        p["diameter_masked"].attrs["description"] = (
            "Diameter with blink and bad-fit frames set to NaN."
        )
        p["imaging_active"].attrs["description"] = (
            "1 where camera time falls inside a 2p acquisition; dark inter-acquisition frames are 0."
        )
        p["alignment_valid"].attrs["description"] = (
            "1 where the nearest recorded pupil frame is within two nominal camera periods."
        )
        p["nearest_frame_error_s"].attrs["units"] = "seconds"
        t = f.create_group("trials")
        for key in ("acq_id", "trial_id", "odor_id", "odor_on_frame", "odor_off_frame",
                    "masked_fraction", "blink_fraction", "clipped_fraction",
                    "coverage_fraction", "flagged"):
            t.create_dataset(key, data=report[key])
        if "state" in report:
            state = np.asarray(report["state"])
            if state.dtype.kind in "UO":
                state = state.astype("S")
            t.create_dataset("state", data=state)
        f.create_dataset("time_s", data=(np.arange(report["n_frame"]) - report["n_pre"]) / report["frame_rate"])
    return path


def pupil_figure(report, out_path):
    """QC: session-spanning and worst fits, plus trace with odor shading."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    camera = report["camera_values"]
    examples = report["example_frames"]
    excluded = list(map(int, report["worst_frame_indices"]))
    accepted = list(map(int, report.get("accepted_frame_indices", ())))
    if not accepted:  # Compatibility with reports produced before this layout.
        accepted = [int(index) for index in sorted(examples)
                    if int(index) not in excluded]
    chosen = list(dict.fromkeys(excluded[:3] + accepted[:9]))
    fig = plt.figure(figsize=(15, 9), constrained_layout=True)
    gs = fig.add_gridspec(4, 4)
    for slot, idx in enumerate(chosen):
        ax = fig.add_subplot(gs[slot // 4, slot % 4])
        frame, original_mask, mask = examples[idx]
        ax.imshow(frame, cmap="gray")
        excluded = original_mask & ~mask
        if np.any(excluded):
            yellow = np.zeros((*excluded.shape, 4), dtype=float)
            yellow[excluded] = (1.0, 0.85, 0.0, 0.75)
            ax.imshow(yellow)
        ax.contour(mask, levels=[0.5], colors="cyan", linewidths=.7)
        if np.isfinite(camera["major"][idx]):
            ax.add_patch(Ellipse((camera["x"][idx], camera["y"][idx]),
                                 2*camera["major"][idx], 2*camera["minor"][idx],
                                 angle=np.degrees(camera["theta"][idx]), fill=False,
                                 edgecolor="magenta", linewidth=1))
        status = "EXCLUDED: fit" if idx in excluded else "accepted"
        ax.set_title(
            f"{status} · frame {idx}\n"
            f"in={camera['inlier_fraction'][idx]:.2f}  "
            f"res={camera['residual'][idx]:.2f}px  "
            f"diam={camera['diameter'][idx]:.1f}px"
        )
        ax.axis("off")
    trace_grid = gs[3, :].subgridspec(1, 2, wspace=0.12)
    time = report["time_s"] if "time_s" in report else (np.arange(report["n_frame"])-report["n_pre"])/report["frame_rate"]
    on = np.nanmedian(report["odor_on_frame"] - report["n_pre"]) / report["frame_rate"]
    off = np.nanmedian(report["odor_off_frame"] - report["n_pre"]) / report["frame_rate"]
    states = np.asarray(report["state"]).astype(str)
    odors = np.asarray(report["odor_id"])
    unique_odors = np.unique(odors)
    cmap = plt.get_cmap("tab20", max(len(unique_odors), 1))
    odor_colors = {odor: cmap(i) for i, odor in enumerate(unique_odors)}
    for panel, state in enumerate(("pre", "post")):
        ax = fig.add_subplot(trace_grid[0, panel])
        selected = np.flatnonzero(states == state)
        for i in selected:
            ax.plot(time, report["diameter_masked"][i],
                    color=odor_colors[odors[i]], alpha=.28, linewidth=.65)
        ax.plot(time, np.nanmedian(report["diameter_masked"][selected], axis=0),
                color="black", linewidth=1.5, label="median")
        ax.axvspan(on, off, color="0.5", alpha=.12)
        ax.set(title=f"{state} ({len(selected)} trials)",
               xlabel="seconds from odor onset")
        if panel == 0:
            ax.set_ylabel("major-axis diameter (px)")
        else:
            ax.tick_params(labelleft=False)
        for odor in unique_odors:
            ax.plot([], [], color=odor_colors[odor], label=str(odor))
        if panel == 1:
            ax.legend(title="odor", ncol=4, fontsize=6, title_fontsize=7,
                      frameon=False, loc="upper right")
    fig.suptitle(f"pupil QC — {report['round']}")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return Path(out_path)


class PupilTuningGUI:
    """Two-frame Bokeh editor for a shared ROI and adaptive-threshold offset."""

    def __init__(self, dim_frame, bright_frame, *, save_path, roi=None,
                 threshold_offset=0.0, bright_percentile=97.0, on_save=None):
        self.frames = [np.asarray(dim_frame), np.asarray(bright_frame)]
        if self.frames[0].shape != self.frames[1].shape:
            raise ValueError("Dimmest and brightest tuning frames differ in shape.")
        self.save_path = Path(save_path)
        self.bright_percentile = float(bright_percentile)
        h, w = self.frames[0].shape
        self.roi = roi or (0, h, 0, w)
        self.threshold_offset = float(threshold_offset)
        self.on_save = on_save

    def settings(self):
        return {"roi": tuple(map(int, self.roi)),
                "threshold_offset": float(self.threshold_offset),
                "bright_percentile": self.bright_percentile}

    @staticmethod
    def _rgba(mask):
        rgba = np.zeros((*mask.shape, 4), np.uint8)
        rgba[mask] = (0, 235, 255, 115)
        return rgba.view(np.uint32).reshape(mask.shape)

    def modify_doc(self, doc):
        from bokeh.layouts import column, row
        from bokeh.models import (Button, ColumnDataSource, Div,
                                  LinearColorMapper, Slider)
        from bokeh.palettes import Greys256
        from bokeh.plotting import figure

        h, w = self.frames[0].shape
        y0, y1, x0, x1 = self.roi
        self.roi_source = ColumnDataSource(dict(
            x=[(x0 + x1) / 2], y=[(y0 + y1) / 2],
            width=[x1 - x0], height=[y1 - y0],
        ))
        self.mask_sources = []
        figures = []
        values = np.concatenate([frame.ravel() for frame in self.frames])
        mapper = LinearColorMapper(palette=Greys256,
                                   low=float(np.percentile(values, 1)),
                                   high=float(np.percentile(values, 99.5)))
        for label, frame in zip(("dimmest", "brightest"), self.frames):
            fig = figure(width=500, height=int(500 * h / w) + 35,
                         x_range=(0, w), y_range=(h, 0),
                         title=label, tools="pan,wheel_zoom,reset",
                         active_scroll="wheel_zoom")
            fig.axis.visible = False
            fig.grid.visible = False
            fig.image(image=[frame], x=0, y=0, dw=w, dh=h,
                      color_mapper=mapper)
            mask_source = ColumnDataSource(dict(
                image=[self._rgba(np.zeros(frame.shape, bool))]
            ))
            fig.image_rgba("image", x=0, y=0, dw=w, dh=h,
                           source=mask_source)
            fig.rect("x", "y", "width", "height",
                     source=self.roi_source, fill_alpha=0.03,
                     fill_color="yellow", line_color="yellow",
                     line_width=2)
            self.mask_sources.append(mask_source)
            figures.append(fig)

        self.slider = Slider(start=-80, end=80, step=1,
                             value=self.threshold_offset,
                             title="offset from per-frame Otsu threshold")
        self.roi_sliders = {
            "x0": Slider(start=0, end=w - 1, step=1, value=x0, title="ROI x0"),
            "x1": Slider(start=1, end=w, step=1, value=x1, title="ROI x1"),
            "y0": Slider(start=0, end=h - 1, step=1, value=y0, title="ROI y0"),
            "y1": Slider(start=1, end=h, step=1, value=y1, title="ROI y1"),
        }
        self.status = Div(width=1000)
        save = Button(label="Save ROI + threshold", button_type="success", width=250)
        self.slider.on_change("value_throttled", self._changed)
        for slider in self.roi_sliders.values():
            slider.on_change("value_throttled", self._changed)
        save.on_click(self._save)
        doc.add_root(column(
            row(*figures),
            row(self.roi_sliders["x0"], self.roi_sliders["x1"]),
            row(self.roi_sliders["y0"], self.roi_sliders["y1"]),
            self.slider, save, self.status,
        ))
        self._refresh()

    def _read_roi(self):
        if not hasattr(self, "roi_sliders"):
            return self.roi
        h, w = self.frames[0].shape
        x0, x1 = sorted((int(self.roi_sliders["x0"].value),
                         int(self.roi_sliders["x1"].value)))
        y0, y1 = sorted((int(self.roi_sliders["y0"].value),
                         int(self.roi_sliders["y1"].value)))
        if x0 == x1:
            x1 = min(w, x0 + 1)
        if y0 == y1:
            y1 = min(h, y0 + 1)
        return (max(0, y0), min(h, y1), max(0, x0), min(w, x1))

    def _changed(self, attr, old, new):
        self._refresh()

    def _refresh(self):
        self.roi = self._read_roi()
        self.threshold_offset = float(self.slider.value)
        y0, y1, x0, x1 = self.roi
        self.roi_source.data = dict(
            x=[(x0 + x1) / 2], y=[(y0 + y1) / 2],
            width=[x1 - x0], height=[y1 - y0],
        )
        thresholds = []
        areas = []
        for frame, source in zip(self.frames, self.mask_sources):
            mask, threshold = segment_bright_pupil(
                frame, roi=self.roi,
                bright_percentile=self.bright_percentile,
                threshold_offset=self.threshold_offset,
            )
            source.data = dict(image=[self._rgba(mask)])
            thresholds.append(threshold)
            areas.append(int(mask.sum()))
        self.status.text = (
            f"ROI (y0, y1, x0, x1) = {self.roi} &nbsp; · &nbsp; "
            f"thresholds dim/bright = {thresholds[0]:.0f}/{thresholds[1]:.0f} "
            f"&nbsp; · &nbsp; mask areas = {areas[0]}/{areas[1]} px"
        )

    def _save(self):
        import json
        self._refresh()
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.save_path.write_text(json.dumps(self.settings(), indent=2))
        self.status.text += f" &nbsp; · &nbsp; saved {self.save_path}"
        if self.on_save is not None:
            self.on_save(self.save_path)


def launch_pupil_tuner(dim_frame, bright_frame, *, save_path, roi=None,
                       threshold_offset=0.0, bright_percentile=97.0):
    """Launch the ROI/threshold tuner in a Jupyter or VS Code notebook."""

    import os
    import sys
    import bokeh.plotting as bpl
    from bokeh.io import output_notebook
    if "ipykernel" in sys.modules:
        os.environ["BOKEH_ALLOW_WS_ORIGIN"] = "*"
        output_notebook(hide_banner=True)
    gui = PupilTuningGUI(
        dim_frame, bright_frame, save_path=save_path, roi=roi,
        threshold_offset=threshold_offset,
        bright_percentile=bright_percentile,
    )
    bpl.show(gui.modify_doc)
    return gui


def load_pupil_tuning(path):
    """Read settings saved by :func:`launch_pupil_tuner`."""

    import json
    values = json.loads(Path(path).read_text())
    values["roi"] = tuple(map(int, values["roi"]))
    return values


def align_to_round(aux_path, round_path):
    """Row indices joining standalone pupil output to a trace round by acq_id."""

    from .h5io import open_h5

    with open_h5(aux_path) as f:
        if "trials/acq_id" not in f:
            raise ValueError(f"{aux_path} has no /trials/acq_id.")
        aux = f["trials/acq_id"][:]
    with open_h5(round_path) as f:
        if "trials/acq_id" not in f:
            raise ValueError(f"{round_path} has no /trials/acq_id.")
        rnd = f["trials/acq_id"][:]
    lookup = {int(acq_id): row for row, acq_id in enumerate(aux)}
    pairs = [(lookup[int(acq_id)], row) for row, acq_id in enumerate(rnd)
             if int(acq_id) in lookup]
    if not pairs:
        raise ValueError("No acq_id in common between pupil aux file and round.")
    return (np.asarray([pair[0] for pair in pairs]),
            np.asarray([pair[1] for pair in pairs]))
