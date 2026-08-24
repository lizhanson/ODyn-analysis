# Stage 0 — odor metadata

Build stimulus metadata from the odor workbook (`Print v3`, `Print v1`, `Mono`, and `Mixes`):

```bash
python -m analysis.stage0.build_odor_tables path/to/Moss_Lab_Odors.xlsx
```

Outputs are keyed by `odor_id`:

| File | Contents |
|---|---|
| `odor_dictionary.csv` | odor identity, role, class, chemical group, and vapor pressure |
| `mixture_composition.csv` | mixture components, fractions, and vapor-pressure metadata |
| `odor_panels.csv` | panel vial positions and delivered concentrations |
| `mixture_feasibility.csv` | availability and dose match of mixture components presented alone |

## Session manifest

`session_manifest.csv` is the canonical candidate-session manifest for the
pre/post ketamine/xylazine dataset. It is keyed by `group_id` and records mouse,
date, experiment, reporter population, objective, depth class, protocol, and
treatment. The 20x rows also carry the numerical imaging depth transcribed from
the date/mouse in-vivo spreadsheet log and the log's path relative to the
`ImagingData` root.

Depth conventions:

- `depth_um` is a positive numerical distance below the surface for analysis.
- `depth_raw` preserves the spreadsheet notation, including sign and
  approximation marks.
- `depth_class` is the design-level `superficial`/`deep` label and is not
  recomputed from `depth_um`.
- `metadata_flag` records missing or conflicting source metadata; flagged rows
  are retained rather than silently repaired or excluded.

`inclusion_status`, `exclusion_reason`, and `postprocessing_qc` are deliberately
blank until cohort selection and postprocessing QC are complete. Analyses must
select sessions from this manifest rather than discovering candidate sessions
from whatever processed files happen to be present on disk. Saline controls
belong in a separate manifest.

The 7-odor panel is the first seven positions of `Print v1`; the 12-odor panel uses all of `Print v1`, and the 16-odor panel uses `Print v3`. Blank class fields remain missing values. Prefer `chemical_class` when the workbook's chemical labels disagree.
