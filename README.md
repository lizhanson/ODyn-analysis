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

## Running it

Open `analysis/Stage1_10x_segmentation.ipynb`
and set the top cell:

```python
GROUP_ID = 217              # not exp_id -- the two id spaces overlap
MANIPULATION = ""
SEPARATE_BY_CONDITION = False
APPROVED_ONLY = False
```

`sys.path` is set relative to the notebook, so it imports `analysis.*` without
installation.

## Notes

**Use group ID.** removes the ambiguity, and handles
sessions that are split across `_e1`/`_e2`.

**Session metadata is read from a local copy.** This avoids repeated slow reads from the server.

**Correlation maps are cached after making z-score movies.** Saves time and data storage. See `session/corrcache.py`.

**Detection can be state-agnostic.** Separating splits the trials across twice as many groups, and since
the olfactometer randomises with replacement rather than counterbalancing, some
groups end up with one trial — whose "trial average" is a single trial of noise. Pooling avoids this and is still valid for segmentation.

**Extraction is resumable and staged locally.** Each trial is written to a local
memmap as it completes, and the HDF5 is written to scratch and copied (nice for a long process that can be interrupted by dropping the network connection.

## Output

One self-describing HDF5 per processing round, beside the session:

```
<exp>/processed/python/group217_20260805_m426_e1_processed_20260813.h5
                       group217_..._masks_processed_20260813.mat   (MATLAB)
                       group217_..._masks_processed_20260813.png   (overlay)
```

The mask's content hash is recorded in the traces, so `verify()` can say whether
a set of traces still belongs to the mask beside it.

**A 20x round is one HDF5 bundle.** `Segmentation20xState.save` writes the curated
masks, the detector output they were curated from, the reference images, the
parameters, the recorded edits, the ROI table, and the groups into a single
`.h5`. Resuming reads the stored labels rather than running the detectors
again: every group is keyed by label id, so a scikit-image that returns the
same ROIs in a different order would silently reassign which cell is in which
group. Replaying the edits on the stored labels must reproduce the stored
curated masks or the load is refused.

**20x ROI groups can be derived rather than drawn.** `seg_20x/grouping.py`
agglomerates ROIs by nearest-pixel gap and trace correlation, strongest link
first, under one constraint: at most one soma per group. Processes need not
reach a soma -- a neurite chain whose parent is out of plane is still a group --
but no process is placed in a group holding two somas, and the refused link is
reported. Run `proximity_correlation_profile` first: an ROI abutting a soma
shares signal with it through the PSF, so correlation that decays to nothing
within a micron or two means the links are adjacency.
