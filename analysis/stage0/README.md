# Stage 0 — canonical odor metadata

Generated from the Moss Lab odor spreadsheet (`Print v3`, `Print v1`, `Mono`,
`Mixes` tabs). Data-independent: describes the stimulus set, not any recording.

```bash
python -m analysis.stage0.build_odor_tables path/to/Moss_Lab_Odors.xlsx
```

Everything is keyed on `odor_id`, which is the spreadsheet's "Moss Lab Odor #"
and the same key as `odors.odor_id` in the odyn database — so these tables join
straight onto `trials.odor_id`, and retrofitting them into SQL later is a
`CREATE TABLE` + `INSERT` with no re-keying.

## Tables

| File | Grain | Purpose |
|---|---|---|
| `odor_dictionary.csv` | one row per `odor_id` | role (single/mix/control), class I/II, chemical group, vapor pressure |
| `mixture_composition.csv` | one row per (mix, component) | component ids, nominal fraction, VP ratio, VP-mismatch flag |
| `odor_panels.csv` | one row per (panel, vial) | which odors are loaded on the 7-, 12-, and 16-odor panels, and at what delivered ppm |
| `mixture_feasibility.csv` | one row per (panel, mix, component) | derived — whether Stage 5 mixture integration is well-posed |

The 7-odor panel is the first 7 vial positions of the same rack as the 12-odor
panel (`Print v1`), not a separate sheet. The 16-odor panel is `Print v3`.

## Findings that affect the analysis plan

**1. Four panel odors are missing from the database.** The 16-odor panel uses
`odor_id` 31, 32 (epsilon, epsilon') and 39, 40 (lambda, lambda'). The `odors`
table stops at 30, and `trials.odor_id` has a foreign key onto it that
`add_experiment` enforces with `PRAGMA foreign_keys = ON`. Ingesting a 16-odor
session today fails with `FOREIGN KEY constraint failed`. Odors 33–44 (zeta,
kappa, omicron, rho and the singles 2-heptanone, methyl tiglate) are missing
too. Fixing this is a schema-data change, not a schema-structure change —
`INSERT` rows into `odors`.

**2. Mixture integration is only well-posed for lambda and epsilon.** The
prediction needs each component measured as a single at the concentration it
takes inside the mix:

- **epsilon / epsilon'** and **lambda / lambda'** — components are on the panel
  as singles at 1.0 ppm, and sit at 1.0 / 0.6 ppm inside the mixes. One
  component per mix is an exact match; the other needs a 0.6× dose
  interpolation.
- **alpha / alpha'** — components sit at 13–44 ppm inside the mix but are only
  presented as singles at 1.0 ppm. That is a 13–44× dose *extrapolation* far
  outside anything measured, so alpha cannot support the analysis. It appears to
  carry over the `Print v1` recipe (matched to the old liquid-volume mix)
  rather than the ppm-matched design in the `Mixes` tab.
- **7- and 12-odor panels** — mixture components are never presented as singles
  at all, confirming integration is a 16-odor-session-only analysis.

**3. Class I/II is partial.** Assigned for the carboxylic acids (I), aromatics
(II), and two "I and II" odors; blank for the alcohols, terpenes, and all mixes.
Blank means unassigned, not "neither" — treat it as missing data.

**4. `chemical_group` and `chemical_class` are inconsistent in the source.**
(-) limonene is group "Saturated hydrocarbon" / class "Terpene"; (+) alpha
pinene is "Terpenoid" / "Terpene"; (-) alpha pinene has a blank group. Prefer
`chemical_class` and clean up at the source if these become a figure axis.

## Retrofit path

When these stabilize, the DB shape is:

```sql
ALTER TABLE odors ADD COLUMN role TEXT;            -- single | mix | control
ALTER TABLE odors ADD COLUMN odor_class TEXT;      -- 'I' | 'II' | 'I and II' | NULL
ALTER TABLE odors ADD COLUMN vapor_pressure REAL;

CREATE TABLE odor_mixtures
    ( mix_odor_id       INTEGER NOT NULL
    , component_odor_id INTEGER NOT NULL
    , component_slot    TEXT NOT NULL CHECK(component_slot IN ('A','B'))
    , nominal_fraction  REAL NOT NULL
    , PRIMARY KEY (mix_odor_id, component_odor_id)
    , FOREIGN KEY (mix_odor_id)       REFERENCES odors(odor_id)
    , FOREIGN KEY (component_odor_id) REFERENCES odors(odor_id)
    ) STRICT;
```

Panel membership is per-session rather than per-odor, so `odor_panels` most
likely belongs on `experiments` as a `panel` column rather than as its own
table.
