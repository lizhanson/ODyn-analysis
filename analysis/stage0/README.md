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

The 7-odor panel is the first seven positions of `Print v1`; the 12-odor panel uses all of `Print v1`, and the 16-odor panel uses `Print v3`. Blank class fields remain missing values. Prefer `chemical_class` when the workbook's chemical labels disagree.
