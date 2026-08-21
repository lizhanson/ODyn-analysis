# ODyn analysis

10x glomerular segmentation and trace extraction for Ca imaging built on top of [odyn](https://github.com/lizhanson/ODyn),
which does motion correction, alignment, and provenance via the database. The goal is for this 
repo to do everything downstream (session assembly, segmentation, trace
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

Correlation maps are cached after z-scoring to save time copying to the server.And because there are other better zscore movies that we should eventually use for this instead.

Detection should usually be pooled across states even when responses might differ slightly because aeparating by block splits the trials across twice as many groups. Since the olfactometer randomizes with replacement some groups have just one or a few trials. 

Extraction is resumable and staged locally. Each trial is written to a local
memmap as it completes, and the HDF5 is written to scratch and copied (nice for a long process that can be interrupted by dropping the network connection.

## Output

One HDF5 per processing round, goes to the experiment folder/processed/python. Ex:

```
<exp>/processed/python/group217_20260805_m426_e1_processed_20260813.h5
                       group217_..._masks_processed_20260813.mat   (MATLAB)
                       group217_..._masks_processed_20260813.png   (overlay)
```

The mask's content hash is recorded in the traces, so `verify()` can say whether
a set of traces still belongs to the mask beside it.

A 20x segmentation round is saved as one HDF5 bundle. `Segmentation20xState.save` writes the curated
masks, the detector output they were curated from, the reference images, the
parameters, the recorded edits, the ROI table, and the groups into a single
`.h5`. (Side note): There's a complicated reload/resumption process designed around the idea that ROIs can be manually assigned to groups, 
but it probably makes sense to drop that if all group assignments are made by correlation later. Manual assignment is probably a waste of time. 

New stuff not vetted: 20x ROI groups can maybe be automatically assigned by a combo of spatial proximity and temporal correlation. `seg_20x/grouping.py`
links ROIs by nearest-pixel gap and trace correlation, strongest link
first with, at most, one soma per group. 
