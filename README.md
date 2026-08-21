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
├── group<group_id>_<experiment>_processed_<YYYYMMDD>_responseqc.json
├── group<group_id>_<experiment>_processed_<YYYYMMDD>_responseqc.png
├── group<group_id>_<experiment>_processed_<YYYYMMDD>_groups.png
└── aux/
    ├── group<group_id>_<experiment>_pupil.h5
    ├── group<group_id>_<experiment>_pupil.png
    ├── group<group_id>_<experiment>_pupil_tuning.json
    ├── group<group_id>_<experiment>_pupil_preflight_qc.png
    ├── group<group_id>_<experiment>_processed_<YYYYMMDD>_respiration.h5
    └── group<group_id>_<experiment>_processed_<YYYYMMDD>_respiration.png
```

The 10x mask bundle is the portable segmentation output. It contains the curated labels, per-group masks, reference image, and GUI configuration, allowing extraction on another computer without copying caches or rerunning segmentation.

The processed HDF5 is the final 10x or 20x round. It contains `/masks`, `/traces`, `/rois`, and `/trials`; the PNG is a mask overlay. No `.mat` or trace `.npz` is written.

Pupil and respiration outputs live in `processed/python/aux/`. Their `acq_id` arrays align trial rows to `/trials/acq_id` in the processed round. `respiration_from_round()` uses the processed-round filename stem.

The 20x GUI checkpoint is local:

```text
<ODYN_SCRATCH_ROOT>/seg_20x/group<group_id>/curated_20x_rois.h5
```

It contains the reference images, automatic and curated masks, parameters, edits, ROI table, and groups. The finalized server HDF5 contains the 20x masks and extracted traces.
