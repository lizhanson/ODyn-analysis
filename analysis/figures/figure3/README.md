# Figure 3 — 20x DA subtype/depth, compartment, and state

**Claim:** TH superficial, TH deep, and DAT superficial cells differ in signed
odor coding, and anesthesia reorganizes soma and grouped-process responses.

The process population is process-only, aggregated within curated biological
groups (not individual fragments). Pupil/running analyses are awake,
within-session and within-odor: odor identity and the mean odor-time course are
removed before testing whether neural deviations track arousal deviations.
This is an association analysis, not an NE or causal claim.

Thy1 is deliberately excluded from this cellular DA figure.

Run `Figure3_cellular_state_coding.ipynb`; outputs go in `outputs/`.

The notebook now follows this order, with a saved table and reviewable panel at
each step: tonic F0; raw positive/negative AUC; signed lifetime sparseness;
unit and population reliability; qualitative QC-style temporal heatmaps;
matched soma-process comparisons; within-odor pupil/running association; and a
reliability gate before geometry. The same tables can be generated headlessly
with `python -m analysis.figures.figure3.run_cellular_20x`.
