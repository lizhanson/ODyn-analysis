from pathlib import Path

import numpy as np
import pytest

from .example_images import (LEVELS, WINDOWS, boundaries, build_example,
                             crop_image, grayscale_rgb, load_context,
                             outline_rgb, paint_units, pixel_level,
                             reference_image, render_level, resolve_window,
                             select_trials, signed_rgb, smooth_frames, stretch,
                             trial_pixel_images, unit_response, window_label)

SHAPE = (24, 20)
TIME = np.arange(-5., 8., .25)
ODOR_ON, ODOR_OFF = 20, 36          # 16 frames at 4 Hz is a 4 s valve window
N_FRAME = 52


def _labels():
    """Two ROIs that are joined into one unit, plus a third standalone ROI."""
    labels = np.zeros(SHAPE, np.int32)
    labels[4:8, 4:8] = 1
    labels[4:8, 10:14] = 2
    labels[14:18, 6:10] = 3
    return labels


def _write_session(tmp_path, *, responses=(6., 0.), n_trial=6):
    """A source round and its grouped product, plus one TIFF per trial."""
    import h5py
    import tifffile

    labels = _labels()
    odor_ids = np.array([1, 1, 1, 2, 2, 2])[:n_trial]
    states = np.array([0, 0, 0, 0, 0, 0])[:n_trial]
    trial_ids = np.arange(100, 100 + n_trial)
    rng = np.random.default_rng(0)

    movies = []
    for trial, odor in zip(trial_ids, odor_ids):
        movie = rng.normal(100, 4, (N_FRAME, *SHAPE)).astype(np.float32)
        gain = responses[0] if odor == 1 else responses[1]
        movie[ODOR_ON:ODOR_OFF][:, labels == 1] += gain
        movie[ODOR_ON:ODOR_OFF][:, labels == 3] -= gain
        path = tmp_path / f"trial{trial}_mcor.tif"
        tifffile.imwrite(path, movie)
        movies.append(path)

    source = tmp_path / "group7_e1_processed_20260101.h5"
    with h5py.File(source, "w") as handle:
        handle.create_dataset("masks/labels", data=labels)
        handle.create_dataset("traces/time_s", data=TIME)
        handle.create_dataset("trials/trial_id", data=trial_ids)
        handle.create_dataset("trials/odor_on_frame", data=np.full(n_trial, ODOR_ON))
        handle.create_dataset("trials/odor_off_frame", data=np.full(n_trial, ODOR_OFF))
        handle.create_dataset("trials/mcor_path",
                              data=np.arange(n_trial, dtype=np.uint8))
        handle.create_dataset("trials/mcor_path_levels",
                              data=np.array([str(p) for p in movies],
                                            dtype=h5py.string_dtype("utf-8")))
        handle.attrs["parameters_json"] = '{"session": {"um_per_px_reported": 2.0}}'

    # Unit 0 joins ROIs 1 and 2; unit 1 is ROI 3 alone.
    members = [np.array([1, 2], np.int64), np.array([3], np.int64)]
    during = (TIME >= 0) & (TIME < 4)
    z = np.zeros((2, n_trial, TIME.size), np.float32)
    for trial, odor in enumerate(odor_ids):
        gain = 3. if odor == 1 else 0.
        z[0, trial, during] = gain
        z[1, trial, during] = -gain

    grouped = tmp_path / "group7_e1_pertrial_median_10x_grouped.h5"
    with h5py.File(grouped, "w") as handle:
        handle.attrs["source_component_round"] = str(source)
        handle.create_dataset("trial_id", data=trial_ids)
        handle.create_dataset("odor_id", data=odor_ids)
        handle.create_dataset("state", data=states)
        handle.create_dataset("state_levels",
                              data=np.array(["pre"], dtype=h5py.string_dtype("utf-8")))
        units = handle.create_group("units")
        units.create_dataset("unit_id", data=np.array(
            ["join_1", "roi_3"], dtype=h5py.string_dtype("utf-8")))
        member_data = units.create_dataset(
            "member_roi_ids", (2,), dtype=h5py.vlen_dtype(np.dtype("int64")))
        for index, ids in enumerate(members):
            member_data[index] = ids
        units.create_dataset("z", data=z)

    row = {"group_id": 7, "mouse": "m1", "population": "TH-GCaMP",
           "objective": "10x", "depth_class": "na", "grouped_path": str(grouped)}
    return row, labels


@pytest.fixture
def session(tmp_path):
    row, labels = _write_session(tmp_path)
    # No .odyn database here, so path resolution falls through to the round.
    return load_context(row, tmp_path), labels


# --- array-level behaviour -------------------------------------------------

def test_trial_pixel_images_recovers_a_known_response():
    rng = np.random.default_rng(1)
    baseline = rng.normal(50, 2, (30, *SHAPE)).astype(np.float32)
    response = baseline[:10] + 8.
    _, _, z = trial_pixel_images(baseline, response)
    assert np.nanmedian(z) == pytest.approx(4., abs=.6)   # 8 over an SD near 2


def test_trial_pixel_images_returns_nan_for_dead_pixels():
    rng = np.random.default_rng(2)
    baseline = rng.normal(50, 2, (30, *SHAPE)).astype(np.float32)
    baseline[:, :, 0] = 5.                                 # zero-variance column
    response = baseline[:10] + 1.
    _, _, z = trial_pixel_images(baseline, response)
    assert np.all(np.isnan(z[:, 0]))
    # The floor is the 1st percentile of the live SDs, so it drops about one
    # live pixel in a hundred as well. Everything else must survive.
    assert np.isfinite(z[:, 1:]).mean() > .95


def test_trial_pixel_images_never_returns_infinity():
    """An all-constant field has no usable noise estimate anywhere."""
    baseline = np.tile(np.arange(SHAPE[1], dtype=np.float32), (30, SHAPE[0], 1))
    _, _, z = trial_pixel_images(baseline, baseline[:10] + 1.)
    assert np.all(np.isnan(z))


def test_boundaries_are_the_inner_edge_of_each_roi():
    labels = _labels()
    edge = boundaries(labels)
    assert not edge[labels == 0].any()                     # never in background
    assert edge[4, 4] and not edge[6, 6]                   # corner yes, centre no
    assert set(np.unique(labels[edge])) == {1, 2, 3}


def test_stretch_clips_to_the_percentile_range():
    image = np.arange(400, dtype=np.float32).reshape(20, 20)
    scaled, low, high = stretch(image, percentiles=(10, 90))
    assert scaled.min() == 0. and scaled.max() == 1.
    assert low < high


def test_stretch_tolerates_an_all_nan_image():
    scaled, _, _ = stretch(np.full(SHAPE, np.nan, np.float32))
    assert np.all(scaled == 0.)


def test_crop_image_selects_the_requested_window():
    assert crop_image(np.zeros(SHAPE), (4, 12, 2, 10)).shape == (8, 8)
    assert crop_image(np.zeros(SHAPE), None).shape == SHAPE


def test_signed_rgb_keeps_zero_at_the_colormap_centre():
    values = np.array([[-2., 0., 4.]], np.float32)
    rgb = signed_rgb(values, limits=(-2., 4.))
    centre = rgb[0, 1].astype(int)
    assert centre.min() > 240 and centre.max() - centre.min() <= 4   # near white
    assert rgb[0, 0, 2] > rgb[0, 0, 0]                     # suppression is blue
    assert rgb[0, 2, 0] > rgb[0, 2, 2]                     # excitation is red


def test_signed_rgb_composites_by_response_magnitude():
    """z=0 shows the anatomy untouched; a strong z shows the colormap alone."""
    background = np.linspace(0, 100, SHAPE[0] * SHAPE[1],
                             dtype=np.float32).reshape(SHAPE)
    values = np.zeros(SHAPE, np.float32)
    values[10, 10] = 4.                                    # at the opaque cutoff
    rgb = signed_rgb(values, limits=(-2., 4.), threshold=4., background=background)
    grey = grayscale_rgb(background)
    quiet = np.ones(SHAPE, bool)
    quiet[10, 10] = False
    np.testing.assert_array_equal(rgb[quiet], grey[quiet])
    np.testing.assert_array_equal(rgb[10, 10],
                                  signed_rgb(values, limits=(-2., 4.))[10, 10])


def test_outline_rgb_rejects_a_mismatched_mask():
    with pytest.raises(ValueError, match="differ"):
        outline_rgb(np.zeros(SHAPE, np.float32), np.zeros((5, 5), np.int32))


# --- session-level behaviour ----------------------------------------------

def test_load_context_reads_masks_scale_and_units(session):
    context, labels = session
    np.testing.assert_array_equal(context.labels, labels)
    assert context.um_per_px == 2.0
    assert context.frame_rate == pytest.approx(4.0)
    assert list(context.unit_ids) == ["join_1", "roi_3"]


def test_unit_labels_collapse_a_join_into_one_label(session):
    context, labels = session
    unit_labels = context.unit_labels
    assert unit_labels[labels == 1][0] == unit_labels[labels == 2][0]
    assert unit_labels[labels == 3][0] != unit_labels[labels == 1][0]
    assert not unit_labels[labels == 0].any()


def test_select_trials_filters_by_block_and_odor(session):
    context, _ = session
    np.testing.assert_array_equal(select_trials(context, block="pre", odor_id=1),
                                  [0, 1, 2])
    with pytest.raises(ValueError, match="no pre trials of odor 99"):
        select_trials(context, block="pre", odor_id=99)


def test_select_trials_rejects_an_unknown_block(session):
    context, _ = session
    with pytest.raises(ValueError, match="not one of"):
        select_trials(context, block="post", odor_id=1)


def test_unit_response_reduces_over_the_window(session):
    context, _ = session
    values = unit_response(context, select_trials(context, block="pre", odor_id=1))
    np.testing.assert_allclose(values, [3., -3.], atol=1e-5)


def test_paint_units_writes_every_member_roi(session):
    context, labels = session
    image = paint_units(context, [7., -7.])
    assert image[labels == 1][0] == 7. and image[labels == 2][0] == 7.
    assert image[labels == 3][0] == -7.
    assert np.all(np.isnan(image[labels == 0]))


def test_paint_units_rejects_the_wrong_number_of_values(session):
    context, _ = session
    with pytest.raises(ValueError, match="expected 2 unit values"):
        paint_units(context, [1., 2., 3.])


def test_pixel_level_recovers_signed_structure_from_the_movies(session):
    context, labels = session
    indices = select_trials(context, block="pre", odor_id=1)
    fluorescence, _, pixel_z = pixel_level(context, indices, progress=False)
    assert fluorescence.shape == SHAPE
    assert np.nanmedian(pixel_z[labels == 1]) > 1.
    assert np.nanmedian(pixel_z[labels == 3]) < -1.
    assert abs(np.nanmedian(pixel_z[labels == 0])) < .5


def test_pixel_level_window_matches_the_valve_window(session):
    context, _ = session
    indices = select_trials(context, block="pre", odor_id=1)
    windowed = pixel_level(context, indices, progress=False)[2]
    valve = pixel_level(context, indices, window_s=None, progress=False)[2]
    np.testing.assert_allclose(windowed, valve, atol=1e-6)


# --- assembled example -----------------------------------------------------

def test_build_example_agrees_between_the_pixel_and_roi_levels(session):
    context, labels = session
    example = build_example(context, block="pre", odor_id=1, progress=False)
    assert example.n_trials == 3
    np.testing.assert_array_equal(example.trial_ids, [100, 101, 102])
    # The two levels are built from the same trials and window, so an excited
    # unit must be excited in both and a suppressed unit suppressed in both.
    assert np.nanmedian(example.pixel_z[labels == 1]) > 0
    assert np.nanmedian(example.roi_z[labels == 1]) > 0
    assert np.nanmedian(example.pixel_z[labels == 3]) < 0
    assert np.nanmedian(example.roi_z[labels == 3]) < 0


def test_build_example_without_pixels_skips_the_movies(session):
    context, _ = session
    example = build_example(context, block="pre", odor_id=1, pixel=False,
                            progress=False)
    assert np.all(np.isnan(example.pixel_z))
    assert np.isfinite(example.unit_values).all()


def test_build_example_stem_names_the_block_and_odor(session):
    context, _ = session
    example = build_example(context, block="pre", odor_id=2, pixel=False,
                            progress=False)
    assert example.stem() == "group7_m1_TH_10x_pre_odor2_0to4s"
    assert example.block_label == "awake"


def test_render_level_returns_rgb_for_every_level(session):
    context, _ = session
    example = build_example(context, block="pre", odor_id=1, progress=False)
    for level in LEVELS:
        rgb = render_level(example, level)
        assert rgb.shape == (*SHAPE, 3) and rgb.dtype == np.uint8


def test_render_level_honours_a_crop(session):
    context, _ = session
    example = build_example(context, block="pre", odor_id=1, progress=False)
    assert render_level(example, "roi_z", crop=(4, 12, 2, 10)).shape == (8, 8, 3)


def test_render_level_rejects_an_unknown_level(session):
    context, _ = session
    example = build_example(context, block="pre", odor_id=1, pixel=False,
                            progress=False)
    with pytest.raises(ValueError, match="level must be one of"):
        render_level(example, "dF_over_F")


def test_export_example_writes_a_file_for_every_level(session, tmp_path):
    from .example_images import export_example

    context, _ = session
    example = build_example(context, block="pre", odor_id=1, progress=False)
    written = export_example(tmp_path / "out", example)
    for level in LEVELS:
        assert Path(written[level]).exists()
    assert Path(written["ladder"]).exists()
    assert np.load(written["arrays"])["pixel_z"].shape == SHAPE
    assert written["n_trials"] == 3 and written["block"] == "pre"


# --- windows ---------------------------------------------------------------

def test_resolve_window_accepts_names_pairs_and_none():
    assert resolve_window("post_odor") == (4., 8.)
    assert resolve_window((1., 3.)) == (1., 3.)
    assert resolve_window(None) is None


def test_resolve_window_rejects_a_bad_name_or_empty_span():
    with pytest.raises(ValueError, match="window must be one of"):
        resolve_window("during_odor")
    with pytest.raises(ValueError, match="stop must exceed start"):
        resolve_window((4., 4.))


def test_window_label_names_the_known_epochs():
    assert window_label(WINDOWS["post_odor"]).startswith("post_odor")
    assert window_label((1., 3.)) == "1-3 s"
    assert window_label(None) == "valve window"


def test_pixel_level_measures_a_different_window_against_the_same_baseline(session):
    """The late window sees the response; the post-odor window sees it end."""
    context, labels = session
    indices = select_trials(context, block="pre", odor_id=1)
    late = pixel_level(context, indices, window_s="late", progress=False)[2]
    after = pixel_level(context, indices, window_s="post_odor", progress=False)[2]
    assert np.nanmedian(late[labels == 1]) > 1.
    assert abs(np.nanmedian(after[labels == 1])) < .5


def test_pixel_level_baseline_does_not_move_with_the_window(session):
    """A post-odor window is still referenced to the pre-odor frames."""
    context, _ = session
    indices = select_trials(context, block="pre", odor_id=1)
    default = pixel_level(context, indices, window_s="post_odor", progress=False)[0]
    explicit = pixel_level(context, indices, window_s="post_odor",
                           baseline_s=(-5., 0.), progress=False)[0]
    np.testing.assert_allclose(default, explicit, atol=1e-6)


def test_pixel_level_rejects_a_window_past_the_end_of_the_acquisition(session):
    context, _ = session
    indices = select_trials(context, block="pre", odor_id=1)
    with pytest.raises(ValueError, match="but the acquisition has"):
        pixel_level(context, indices, window_s=(0., 30.), progress=False)


def test_baseline_may_not_run_into_the_odor(session):
    context, _ = session
    indices = select_trials(context, block="pre", odor_id=1)
    with pytest.raises(ValueError, match="end at or before odor onset"):
        pixel_level(context, indices, baseline_s=(-2., 2.), progress=False)


def test_unit_response_accepts_a_named_window(session):
    context, _ = session
    indices = select_trials(context, block="pre", odor_id=1)
    np.testing.assert_allclose(unit_response(context, indices, window_s="early"),
                               unit_response(context, indices, window_s=(0., 2.)))


# --- spatial smoothing -----------------------------------------------------

def test_smooth_frames_reduces_noise_without_moving_the_mean():
    rng = np.random.default_rng(7)
    stack = rng.normal(10., 3., (5, *SHAPE)).astype(np.float32)
    smoothed = smooth_frames(stack, 2.)
    assert smoothed.std() < stack.std() / 2
    assert smoothed.mean() == pytest.approx(stack.mean(), abs=.3)


def test_smooth_frames_is_a_no_op_at_zero_sigma():
    stack = np.ones((3, *SHAPE), np.float32)
    assert smooth_frames(stack, 0.) is not None
    np.testing.assert_array_equal(smooth_frames(stack, 0.), stack)


def test_smooth_frames_does_not_spread_missing_pixels():
    stack = np.ones((2, *SHAPE), np.float32)
    stack[:, 5, 5] = np.nan
    smoothed = smooth_frames(stack, 2.)
    assert np.isnan(smoothed[:, 5, 5]).all()
    assert np.isfinite(smoothed[:, 0, 0]).all()
    # Normalized convolution, so an uncovered pixel does not drag its
    # neighbours toward zero.
    assert smoothed[0, 5, 7] == pytest.approx(1., abs=1e-4)


def test_smoothing_raises_response_z_without_lowering_background_noise():
    """The pixel noise it removes from the numerator leaves the denominator too.

    So the background z SD is flat in sigma and sits at its analytic value,
    while a spatially coherent response grows. Contrast improves; the numbers
    are not comparable across sigma.
    """
    rng = np.random.default_rng(0)
    size, n_baseline, n_response = 96, 20, 16
    baseline = rng.normal(100., 4., (n_baseline, size, size)).astype(np.float32)
    response = rng.normal(100., 4., (n_response, size, size)).astype(np.float32)
    response[:, 32:64, 32:64] += 6.
    core = (slice(44, 52), slice(44, 52))          # inside the response square
    far = (slice(4, 24), slice(4, 24))             # background, clear of the blur

    background, signal = {}, {}
    for sigma in (0., 1., 3.):
        _, _, z = trial_pixel_images(baseline, response, sigma_px=sigma)
        background[sigma] = np.nanstd(z[far])
        signal[sigma] = np.nanmedian(z[core])
    analytic = np.sqrt(1 / n_response + 1 / n_baseline)
    for sigma, value in background.items():
        assert value == pytest.approx(analytic, abs=.1), sigma
    assert signal[0.] < signal[1.] < signal[3.]
    assert signal[3.] / background[3.] > 3 * signal[0.] / background[0.]


def test_smoothing_blurs_the_edge_of_an_roi_it_is_wide_relative_to():
    """A sigma near the ROI radius pulls the background in over its boundary.

    Not a defect, but the reason a slide should keep sigma well under the
    radius of the structure being shown.
    """
    rng = np.random.default_rng(11)
    labels = _labels()
    baseline = rng.normal(100., 4., (30, *SHAPE)).astype(np.float32)
    response = baseline[:16].copy()
    response[:, labels == 1] += 6.
    edge = np.zeros(SHAPE, bool)
    edge[4, 4:8] = True                            # the ROI's top row
    centre = np.zeros(SHAPE, bool)
    centre[5:7, 5:7] = True
    _, _, tight = trial_pixel_images(baseline, response, sigma_px=0.5)
    _, _, wide = trial_pixel_images(baseline, response, sigma_px=2.5)
    assert np.nanmean(wide[edge]) < np.nanmean(wide[centre])
    assert np.nanmean(tight[edge]) / np.nanmean(tight[centre]) > \
        np.nanmean(wide[edge]) / np.nanmean(wide[centre])


def test_smoothing_raises_the_response_of_a_real_session_map(session):
    context, labels = session
    indices = select_trials(context, block="pre", odor_id=1)
    rough = pixel_level(context, indices, progress=False)[2]
    smooth = pixel_level(context, indices, sigma_px=1.5, progress=False)[2]
    assert np.nanmedian(smooth[labels == 1]) > np.nanmedian(rough[labels == 1])
    assert np.nanmedian(smooth[labels == 3]) < np.nanmedian(rough[labels == 3])


def test_smoothing_leaves_the_fluorescence_level_untouched(session):
    context, _ = session
    indices = select_trials(context, block="pre", odor_id=1)
    plain = pixel_level(context, indices, progress=False)[0]
    smoothed = pixel_level(context, indices, sigma_px=2., progress=False)[0]
    np.testing.assert_allclose(plain, smoothed, atol=1e-6)


# --- published reference image ---------------------------------------------

def test_reference_image_is_none_without_a_published_bundle(session):
    context, _ = session
    assert reference_image(context) is None


def test_build_example_without_pixels_uses_the_published_reference(session, tmp_path):
    import h5py

    context, labels = session
    bundle = context.source_path.parent / "group7_e1_10x_masks_processed_20260101.h5"
    with h5py.File(bundle, "w") as handle:
        handle.create_dataset("reference", data=np.arange(
            labels.size, dtype=np.float32).reshape(labels.shape))
    assert reference_image(context).shape == labels.shape
    example = build_example(context, block="pre", odor_id=1, pixel=False,
                            progress=False)
    assert example.fluorescence_source == "published reference image"
    assert np.isfinite(example.fluorescence).all()
    # The ROI outline level is real, so it can be checked before the full pass.
    assert render_level(example, "roi_outline").shape == (*SHAPE, 3)


def test_build_example_rejects_a_reference_of_the_wrong_shape(session):
    context, _ = session
    with pytest.raises(ValueError, match="does not match the mask"):
        build_example(context, block="pre", odor_id=1, pixel=False,
                      reference=np.zeros((4, 4), np.float32), progress=False)


def test_example_records_its_own_provenance(session):
    context, _ = session
    example = build_example(context, block="pre", odor_id=1, window_s="late",
                            sigma_px=1.5, progress=False)
    assert example.window_s == (2., 4.) and example.sigma_px == 1.5
    assert "late" in example.caption() and "sigma 1.5 px" in example.caption()
    assert example.stem().endswith("_odor1_2to4s_sigma1.5")


def test_example_without_pixels_reports_only_the_levels_it_has(session):
    context, _ = session
    example = build_example(context, block="pre", odor_id=1, pixel=False,
                            progress=False)
    assert not example.has_pixel_level
    assert example.available_levels == ("fluorescence", "roi_outline", "roi_z")


def test_rendering_a_missing_pixel_level_is_refused_not_faked(session):
    """An all-NaN map over the anatomy would look like a real z map at zero."""
    context, _ = session
    example = build_example(context, block="pre", odor_id=1, pixel=False,
                            progress=False)
    with pytest.raises(ValueError, match="no pixel z"):
        render_level(example, "pixel_z")


def test_export_of_a_roi_only_example_omits_the_pixel_panel(session, tmp_path):
    from .example_images import export_example

    context, _ = session
    example = build_example(context, block="pre", odor_id=1, pixel=False,
                            progress=False)
    written = export_example(tmp_path / "roi_only", example)
    assert "pixel_z" not in written
    assert Path(written["roi_z"]).exists() and Path(written["ladder"]).exists()


def test_full_example_still_exports_all_four_levels(session, tmp_path):
    from .example_images import export_example

    context, _ = session
    example = build_example(context, block="pre", odor_id=1, progress=False)
    assert example.available_levels == LEVELS
    written = export_example(tmp_path / "full", example)
    assert all(Path(written[level]).exists() for level in LEVELS)
