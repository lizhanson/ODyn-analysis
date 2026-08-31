# Exploratory pass

The one-off analysis behind the figure decisions, kept so the numbers in the
write-up can be traced. This is not part of the figure pipeline: production
metrics live in `analysis/figures/population_metrics.py`, which this pass
motivated.

Run the stages as modules, in order. Stage 1 is the only one that reads the
imaging share; it caches a unit x trial x 0.5 s array per session locally, and
every later stage runs off that cache.

```
python -m analysis.eda.stage0_inventory   --output-dir <out>   # coverage, trials per odor
python -m analysis.eda.stage0b_align      --output-dir <out>   # grouped <-> auxiliary alignment
python -m analysis.eda.stage1_extract     --output-dir <out>   # signed features + trial cache
python -m analysis.eda.stage2_summarise   --output-dir <out>   # breadth, balance, diagnostics
python -m analysis.eda.stage3_geometry    --output-dir <out>   # crossnobis by comparison type
python -m analysis.eda.stage4_feasibility --output-dir <out>   # is arousal regression viable
python -m analysis.eda.stage4_regression  --output-dir <out>   # within-odor running coupling
python -m analysis.eda.stage5_fov         --output-dir <out>   # component vs ratio separability
python -m analysis.eda.stage6_arousal_time --output-dir <out>  # time-resolved, odor-specific
python -m analysis.eda.stage7_pupil       --output-dir <out>   # pupil, with a masked sensitivity run
python -m analysis.eda.stage8_rdm         --output-dir <out>   # whole-panel RDMs and confusion
python -m analysis.eda.stage8_plot        --output-dir <out>   # figures from stage 8
```

`ODYN_IMAGING_ROOT` must point at the ImagingData root, as for the figure code.

## What this pass changed

- Mineral oil is not a null stimulus: it crosses an excitation threshold in
  16-87% of unit-odor pairs, and its size varies with state and scale. Responder
  calls moved to a pre-odor excursion null, stratified by trial count.
- A signed four-second mean hides 69-88% of bidirectional responses, so
  excitation and suppression are now integrated separately over the waveform.
- The grouped and auxiliary products cannot be joined by position or by state
  code. The source round carries database trial ids and its trial order matches
  the grouped product; `population_metrics.load_population` uses them.
- Whole-panel confusion matrices read far more clearly than crossvalidated
  distances at these trial counts, so both are produced.

## The report

`report/odyn_exploratory_pass.html` is a complete, self-contained HTML
document: open it in any browser, print it to PDF, or email it. Figures are
embedded as data URIs; the only external request is the Google Fonts
stylesheet, which falls back to the declared stacks offline.

`report/_artifact_body.html` is the same body without the
`<!doctype>`/`<head>`/`<body>` wrapper, which is the form the Artifact
publisher expects.

To rebuild both from an assembled body:

```
python -m analysis.eda.build_report --body <body.html> \
    --standalone analysis/eda/report/odyn_exploratory_pass.html \
    --artifact  analysis/eda/report/_artifact_body.html
```

The embedded figures come from `report_figures.json`, regenerated with
`python -m analysis.eda.report_figures --output-dir <out>` once stage 1 has
written its trial caches.

## The slide deck

`report/odyn_exploratory_pass.pptx` — 14 slides, 16:9, native text and images
with speaker notes on every slide. Cambria and Calibri, both of which ship with
Office. `report/odyn_exploratory_pass_slides.pdf` is the same deck as PDF.

Both come from one specification in `build_deck.py`, so the PDF is a faithful
preview of the PowerPoint geometry rather than a separate design:

```
python -m analysis.eda.deck_figures --eda-dir <out> --output-dir report/deck_figures
python -m analysis.eda.build_deck   --figures report/deck_figures \
    --output-dir report --png-dir <review>   # --png-dir writes one PNG per slide
```
