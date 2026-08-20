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
  stage0/       odor and mixture tables (data-independent)
  Stage1_10x_segmentation.ipynb    the working record/ trial and error
  Stage1_20x_segmentation.ipynb    approved-mcor 20x workflow for VS Code
```

## Running it

Open `analysis/Stage1_10x_segmentation.ipynb`
and set the top cell:

```python
GROUP_ID = 217              # not exp_id -- the two id spaces overlap
MANIPULATION = "ketamine/xylazine"
SEPARATE_BY_CONDITION = False
APPROVED_ONLY = False
```

`sys.path` is set relative to the notebook, so it imports `analysis.*` without
installation.

## Notes

**Use group ID.** removes the ambiguity, and handles
sessions that are split across `_e1`/`_e2`.

**The database is read through a snapshot.** `LocalGroup` copies the database first, under a
bounded lock, and refuses a network path unless forced.

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
