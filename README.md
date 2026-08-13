# ODyn analysis

Calcium-imaging analysis built on top of [odyn](https://github.com/lizhanson/ODyn),
which owns acquisition, motion correction, and the provenance database. This
repository owns everything downstream: session assembly, segmentation, trace
extraction, and QC.

Kept separate from odyn on purpose. odyn is shared infrastructure with a schema
and a `@record_call` contract; this is analysis in progress, which changes daily
and should not drag a shared package along with it. The split is expected to
narrow over time — settled analysis moves into odyn as recorded methods.

## Layout

```
analysis/
  session/      session assembly: resolve, sync, z-scores, extraction, QC
  seg_10x/      10x glomerular segmentation: watershed, merge, curation GUI
  stage0/       canonical odor and mixture tables (data-independent)
  Stage1_10x_segmentation.ipynb    the working record
```

## Running it

The notebook drives everything. Open `analysis/Stage1_10x_segmentation.ipynb`
and set the top cell:

```python
GROUP_ID = 217              # not exp_id -- the two id spaces overlap
MANIPULATION = "ketamine/xylazine"
SEPARATE_BY_CONDITION = False
APPROVED_ONLY = False
```

`sys.path` is set relative to the notebook, so it imports `analysis.*` without
installation.

## Things that are load-bearing

**Ask by group, not experiment.** Group and experiment ids are both small
integers in overlapping ranges, so most numbers are valid as either and land on
a real but wrong session — group 217 is exp 213, while exp 217 is a different
experiment entirely. `resolve_group` removes the ambiguity, and handles the
sessions that are split across `_e1`/`_e2`.

**The database is read through a snapshot, never directly.** Every accessor
reads whole tables, which on the SMB share holds a SQLite lock long enough to
fail another machine's writes — that is how a motion-correction run died after
hours of compute on 2026-08-12. `LocalGroup` copies the database first, under a
bounded lock, and refuses a network path unless forced.

**Correlation maps are cached, not the z-score movies.** The movies are an
intermediate nothing downstream reads; caching them cost 9 GB and ~56 minutes
of writing per session to produce 40 MB of maps. See `session/corrcache.py`.

**Detection is state-agnostic.** Pooling the pre/post blocks for segmentation is
deliberate. Separating splits the trials across twice as many groups, and since
the olfactometer randomises with replacement rather than counterbalancing, some
groups end up with one trial — whose "trial average" is a single trial of noise.

**Extraction is resumable and staged locally.** Each trial is written to a local
memmap as it completes, and the HDF5 is written to scratch and copied, so a
dropped mount costs the outstanding trials rather than the whole run.

## Output

One self-describing HDF5 per processing round, beside the session:

```
<exp>/processed/python/group217_20260805_m426_e1_processed_20260813.h5
                       group217_..._masks_processed_20260813.mat   (MATLAB)
                       group217_..._masks_processed_20260813.png   (overlay)
```

The mask's content hash is recorded in the traces, so `verify()` can say whether
a set of traces still belongs to the mask beside it.
