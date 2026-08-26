# ODyn analysis

10x glomerular and 20x soma/process segmentation and trace extraction for Ca imaging built on top of [odyn](https://github.com/lizhanson/ODyn),
which does motion correction, alignment, and provenance via the database. The goal is for this 
repo to do downstream things important for analyzing data (session assembly, segmentation, trace
extraction, QC, etc.)

Work in progress!

## Layout

```
analysis/
  session/      session assembly: resolve, sync, z-scores, extraction, QC
  seg_10x/      10x glomerular segmentation: watershed, merge, curation GUI
  seg_20x/      20x soma/process segmentation, ordered curation, ROI grouping
  seg_20x/grouping.py   soma-anchored ROI groups from gap + trace correlation
  stage0/       odor and mixture tables (data-independent)
  Stage1_10x_segmentation.ipynb    10x segmentation workflow
  Stage1_20x_segmentation.ipynb    approved-mcor 20x workflow for VS Code
```

## Use

Open `analysis/Stage1_10x_segmentation.ipynb`
and set the top cell:

```python
GROUP_ID = 217             
MANIPULATION = ""
SEPARATE_BY_CONDITION = False #Average odors across blocks or separate by pre/post
APPROVED_ONLY = False #Refers to mcor approval by trial
```

`sys.path` is relative to the notebook, so it imports `analysis.*` without
installation.

## Notes

Session metadata is read from a local copy to avoid slow reads from the server.

Correlation maps are cached in local scratch during 10x processing and deleted after extraction and QC complete. Only masks and final outputs are written to the server.

Detection should usually be pooled across states even when responses might differ slightly because separating by block splits the trials across twice as many groups. Since the olfactometer randomizes with replacement some groups have just one or a few trials.

Extraction is resumable and staged locally. Each trial is written to a local
memmap as it completes, and the completed HDF5 is copied from scratch to the server.

## Output

Session outputs are written under the experiment directory:

```text
<experiment>/processed/python/
├── group<group_id>_<experiment>_10x_masks_processed_<YYYYMMDD>.h5
├── group<group_id>_<experiment>_processed_<YYYYMMDD>.h5
├── group<group_id>_<experiment>_masks_processed_<YYYYMMDD>.png
├── group<group_id>_<experiment>_processed_<YYYYMMDD>_continuousqc.json
├── group<group_id>_<experiment>_processed_<YYYYMMDD>_continuousqc.png
├── group<group_id>_<experiment>_processed_<YYYYMMDD>_baselineqc.png
├── group<group_id>_<experiment>_processed_<YYYYMMDD>_groups.png
├── group<group_id>_<experiment>_processed_<YYYYMMDD>_20x_grouped.h5
├── group<group_id>_<experiment>_processed_<YYYYMMDD>_20x_qc.json
└── aux/
    ├── group<group_id>_<experiment>_pupil.h5
    ├── group<group_id>_<experiment>_pupil.png
    ├── group<group_id>_<experiment>_pupil_tuning.json
    ├── group<group_id>_<experiment>_pupil_preflight_qc.png
    ├── group<group_id>_<experiment>_processed_<YYYYMMDD>_respiration.h5
    └── group<group_id>_<experiment>_processed_<YYYYMMDD>_respiration.png
```

The 10x mask bundle is the portable segmentation output. It contains the curated labels, per-group masks, reference image, and GUI configuration, allowing extraction on another computer without copying caches or rerunning segmentation.

The 10x workflow deliberately separates segmentation, component extraction,
reviewed fragment joining, final grouped-unit QC, and cache deletion. Inventory
or run extraction from all published 10x bundles with:

```bash
python -m analysis.batch_10x
python -m analysis.batch_10x --execute --report 10x_extraction_report.json
python -m analysis.batch_10x --groups 191 193 --execute
```

Return to the 10x notebook session by session after extraction. Its joining GUI
ranks spatial neighbors by odor/post-odor trace correlation but saves only the
joins explicitly reviewed by the user. The final `*_10x_grouped.h5` and 10x QC
figures contain joined glomerular units and unassigned singleton ROIs; component
traces remain only in the intermediate extracted round. Cache cleanup is a
separate final notebook cell and refuses to run until every final output exists.

The processed HDF5 is the final 10x or 20x round. It contains `/masks`,
`/traces`, `/responses`, `/rois`, and `/trials`; the PNG is a mask overlay. Raw
fluorescence is detrended, centred by each trial's immediate four-second
pre-odor mean, and divided by that unit-trial's SD from the same baseline
window. Equal-trial-weighted session and block baseline SDs are also recorded
as diagnostics, but are not used to scale responses. There is no
dF/F, PC1 subtraction, alternate normalization, or binary responder test.

Pupil and respiration outputs live in `processed/python/aux/`. Their `acq_id` arrays align trial rows to `/trials/acq_id` in the processed round. `respiration_from_round()` uses the processed-round filename stem.

The consolidated, segmentation-independent auxiliary pipeline is in
`analysis.session.auxiliary`. It writes one `*_auxiliary.h5` containing
acquisition-aligned treadmill velocity and derived movement features,
bandpassed respiration and masked/unmasked sniff frequency, and masked/unmasked
pupil diameter with fit-quality variables. Every array is trial x imaging frame
and joins future image-processing rounds by `trials/acq_id`. It also writes a
combined auxiliary QC page, the multipanel pupil QC page, and odor-averaged
respiration and treadmill figures.

Use `python -m analysis.batch_auxiliary` for a read-only manifest inventory.
Add `--execute` only after pupil tuning files have been reviewed; unattended
pupil extraction refuses to use a generic ROI when a session has videos but no
saved tuning.

Execution validates existing auxiliary HDF5 structure and required QC figures,
then reports and skips intact groups as `already_complete`. Use `--force` to
regenerate them. During execution the sync file and pupil videos are staged to
local scratch because processing reads them repeatedly. Those staged copies are
deleted immediately after their group publishes successfully, but retained on
failure so it can restart efficiently; `--keep-staged-inputs` opts out of
cleanup and `--no-stage-videos` avoids video staging. Pupil checkpoints remain
resumable and are rewritten every 5,000 fitted frames.

To revise already-published auxiliary products after changing respiration
detection or pupil fit-masking rules, without reopening pupil videos, first
inventory the targets and then execute:

```bash
python -m analysis.batch_auxiliary_revise --groups 220
python -m analysis.batch_auxiliary_revise --groups 220 --execute
```

The revision command rereads only the recorded sync file, recomputes
respiration with the current SNR threshold, remasks pupil diameter from the fit
metrics stored in the auxiliary HDF5, and atomically replaces the HDF5 plus the
combined, respiration, and treadmill QC figures. The image-based pupil-fit QC is preserved
because its example video frames are not stored in the consolidated HDF5.

Prepare the manifest-wide pupil-tuning queue with:

```bash
python -m analysis.batch_pupil_tuning --prepare
```

This performs the slow video/count preflight once and caches dim/bright
acquisition-active frames under `<ODYN_SCRATCH_ROOT>/pupil_tuning`. Then tune
the outstanding sessions consecutively in one browser window:

```bash
python -m analysis.batch_pupil_tuning --serve
```

The tuner reloads any existing session settings and provides Previous, Skip,
and Save-and-advance behavior. Use `--groups 222 217` with `--prepare` for a
targeted queue, `--refresh` to rebuild cached frames, or `--include-complete`
with `--serve` to review already-tuned sessions. Tuning JSON files are written
to each experiment's `processed/python/aux` directory, where batch auxiliary
processing discovers them automatically.

Pupil alignment normally uses one camera pulse per decoded frame. If those
counts differ, the pipeline uses the converter's Micro-Manager
`*_frametimes.csv`, but only when its absolute start and end agree with the
acquisition clock within 250 ms. Two-photon samples farther than two camera
periods from a recorded pupil frame are left missing rather than extrapolated.
The HDF5 records the alignment method, pulse-count delta, endpoint residuals,
nearest-frame timing error, frame-validity mask, and per-trial coverage.

The 20x GUI checkpoint is local:

```text
<ODYN_SCRATCH_ROOT>/seg_20x/group<group_id>/curated_20x_rois.h5
```

It contains the reference images, automatic and curated masks, parameters, edits, ROI table, and groups. The finalized server HDF5 contains the 20x masks and extracted traces.

Published 20x mask bundles can be inventoried and finalized in a batch. The
command is a dry run unless `--execute` is supplied; completed rounds whose
mask and baseline mode already match are skipped, and interrupted extraction
resumes from the scratch checkpoint:

```bash
python -m analysis.batch_20x
python -m analysis.batch_20x --execute --report 20x_extraction_report.json
python -m analysis.batch_20x --groups 175 176 177 --execute
```

After trace-based grouping, the 20x notebook also writes grouped QC sidecars:
`*_20x_spatialqc.png`, `*_20x_snrqc.png`, separate
`*_20x_continuousqc_{groups,somas,processes}.png` and
`*_20x_baselineqc_{groups,somas,processes}.png` figures. Group traces are formed by
pixel-weighting raw ROI fluorescence before the canonical detrend and z-score;
ROIs without a group remain singleton analysis units. Continuous mean and peak
z scores are stored for both odor onset-to-offset and offset-to-offset+4 s.
Finalized rounds store one odor-protected PC1 scalar per trial plus its unit
loadings under `/responses`. PC1 is calculated from the unit-by-trial odor
response matrix, displayed in continuous QC, and never subtracted. Grouped 20x
traces, responses, memberships, response summaries, baseline diagnostics, and
trial PC1 scalars live together in `*_20x_grouped.h5`; no CSV or PC1-timeseries
sidecars are written.
