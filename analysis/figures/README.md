# Figure analyses

All manifests, code, and outputs are resolved relative to the repository by
`analysis.figures.paths`. The external imaging-data mount is deliberately not
hard-coded because its location differs across workstations and servers.

Set it once in the environment before opening Jupyter or running a figure CLI:

```bash
export ODYN_IMAGING_ROOT=/path/to/ImagingData
```

Command-line analyses may instead receive `--imaging-root /path/to/ImagingData`.
Notebook output remains under the corresponding repository-relative
`analysis/figures/figure*/outputs/` directory.

Workshop-specific exploratory panels are reproducible from three small CLIs:

```bash
python -m analysis.figures.run_baseline_arousal
python -m analysis.figures.run_state_arousal --objective 20x
python -m analysis.figures.run_mixture_temporal_correspondence
python -m analysis.figures.run_mixture_nonlinearity --objective 10x
```

The first baseline command reads only the small F0 arrays. The full arousal
command additionally loads neural time courses and is therefore substantially
slower on a network-mounted imaging root.

## Example field images

`Example_images.ipynb` builds presentation stills of one field, by block and
odor, for whichever groups you name. The same field is rendered at four levels
of processing — raw fluorescence, ROI outlines over that fluorescence,
pixelwise odor-period z, and the final analysis units painted with their own z.
The two z levels use the same trials and the same window, so the difference
between them is attributable to ROI definition and nothing else.

Each example writes four bare PNGs with no axes or titles for dropping onto a
slide, a labelled ladder figure with a scale bar and colourbar, and an `.npz`
of the underlying arrays so colour limits, crop, and gamma can be changed later
without rereading a movie.

```bash
python -m analysis.figures.example_images --groups 214 215 219 --blocks pre --odors 1 17 18
```

`--no-pixel` skips rereading the motion-corrected TIFFs and builds the ROI
levels only, which is much faster on a network-mounted imaging root and is the
right setting while choosing groups, odors, and crops. `--population somas`
selects a 20x compartment; 10x has only `units`.

### Windows, baselines, and smoothing

The response window is chosen with `--window`: one of the named epochs
(`odor`, `early`, `late`, `post_odor`, matching what the rest of the pipeline
measures), an explicit `start,stop` in seconds from odor onset, or `valve` for
each trial's exact recorded valve frames. `--baseline` is independent of it and
defaults to every pre-odor frame, so a post-odor panel is still referenced to
the same quiet period as the odor panel. A window that would run past the end
of an acquisition is refused rather than silently truncated.

`--sigma-px` Gaussian-smooths each frame *before* the pixel z is computed, so
the result stays a real z rather than a blurred ratio, and pixels the motion
correction never covered are not smeared into their neighbours.

Smoothing does not lower the background noise of the z map: the pixel noise it
removes from the numerator is removed from the baseline SD in the denominator
too. What it raises is the z of spatially coherent signal, which survives the
blur while the per-pixel SD dividing it shrinks. Contrast improves, but **z
values are not comparable across different sigma**, so any panels sharing a
colour scale must share a sigma. The value is recorded in the caption, the
filename, and the export manifest.

### Relation to the spatial QC pages

The QC pages already show ROIs over a reference image — `*_10x_spatialqc.png`
and `*_10x_groups.png` at 10x, and the groups/somas/processes panels of
`*_20x_spatialqc.png` at 20x. The `roi_outline` level here differs in being a
bare slide-ready panel whose background is the mean fluorescence of the
selected trials, so an awake and a ket/xyl panel of one field genuinely differ.
`--no-pixel` instead uses the published mask bundle's reference image — the
same one the QC pages draw on — and costs no movie reads.
