"""
Stage 0: build the canonical odor tables from the Moss Lab odor spreadsheet.

These tables are data-independent (they describe the stimulus set, not any
recording) and everything downstream keys off them, so they are built first.

They deliberately live *outside* the odyn database for now, as flat CSVs keyed
on `odor_id`. That key is the same `odors.odor_id` the database already uses in
`trials.odor_id`, so retrofitting these into SQL later is a straight
`CREATE TABLE` + `INSERT` with no re-keying.

Run:
    python -m analysis.stage0.build_odor_tables path/to/Moss_Lab_Odors.xlsx

Outputs, written next to this file:
    odor_dictionary.csv      one row per odor_id (singles, mixes, control)
    mixture_composition.csv  one row per (mix, component) pair
    odor_panels.csv          one row per (panel, vial position)

Parsing uses only the standard library: an .xlsx is a zip of XML, and openpyxl
is not installed in the `caiman` environment this repo runs in.
"""

from __future__ import annotations

import argparse
import csv
import re
import zipfile
import xml.etree.ElementTree as ET

from pathlib import Path

NS = {
    "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pr": "http://schemas.openxmlformats.org/package/2006/relationships",
}

OUT_DIR = Path(__file__).parent

# Rows in `Mono` whose name is a placeholder rather than a real reagent.
MONO_PLACEHOLDERS = {"used for mixes", "not assigned"}

# Panels are defined by which sheet they print from and how many vial
# positions are loaded. The 7-odor panel is the first 7 positions of the
# same physical rack as the 12-odor panel, so it is a slice of "Print v1"
# rather than a sheet of its own.
PANELS = [
    ("panel_16", "Print v3", None),
    ("panel_12", "Print v1", None),
    ("panel_7", "Print v1", 7),
]


# --------------------------------------------------------------------------- #
# Minimal .xlsx reader
# --------------------------------------------------------------------------- #


def _col_to_idx(ref: str) -> int:
    """'AA3' -> 26 (zero-based column index)."""
    letters = re.match(r"([A-Z]+)", ref).group(1)
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def read_sheets(path: Path) -> dict[str, list[dict[int, str]]]:
    """Return {sheet_name: [row_dicts]}, each row {col_index: cell_text}."""

    with zipfile.ZipFile(path) as z:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root.findall("m:si", NS):
                shared.append("".join(t.text or "" for t in si.iter(f"{{{NS['m']}}}t")))

        wb = ET.fromstring(z.read("xl/workbook.xml"))
        rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        rel_map = {
            rel.get("Id"): rel.get("Target")
            for rel in rels.findall("pr:Relationship", NS)
        }

        sheets: dict[str, list[dict[int, str]]] = {}

        for sh in wb.find("m:sheets", NS).findall("m:sheet", NS):
            target = rel_map[sh.get(f"{{{NS['r']}}}id")].lstrip("/")
            if not target.startswith("xl/"):
                target = "xl/" + target

            root = ET.fromstring(z.read(target))
            data = root.find("m:sheetData", NS)
            rows: list[dict[int, str]] = []

            if data is not None:
                for row in data.findall("m:row", NS):
                    cells: dict[int, str] = {}
                    for c in row.findall("m:c", NS):
                        t = c.get("t")
                        v = c.find("m:v", NS)

                        if t == "s" and v is not None:
                            val = shared[int(v.text)]
                        elif t == "inlineStr":
                            is_el = c.find("m:is", NS)
                            val = (
                                "".join(
                                    x.text or "" for x in is_el.iter(f"{{{NS['m']}}}t")
                                )
                                if is_el is not None
                                else ""
                            )
                        elif v is not None:
                            val = v.text
                        else:
                            val = ""

                        val = (val or "").strip()
                        if val:
                            cells[_col_to_idx(c.get("r"))] = val

                    rows.append(cells)

            sheets[sh.get("name")] = rows

    return sheets


def _text(row: dict[int, str], col: int) -> str:
    return row.get(col, "").strip()


def _num(row: dict[int, str], col: int) -> None | float:
    raw = _text(row, col)
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _int(row: dict[int, str], col: int) -> None | int:
    val = _num(row, col)
    return None if val is None else int(round(val))


def _round(val: None | float, digits: int) -> None | float:
    """Round for output, dropping spreadsheet float noise (0.30000000000000004)."""
    return None if val is None else round(val, digits)


# --------------------------------------------------------------------------- #
# Table builders
# --------------------------------------------------------------------------- #


def build_mixtures(sheets) -> tuple[list[dict], dict[int, dict]]:
    """Parse `Mixes` into one row per (mix, component) pair.

    Column letters from the sheet header:
      A mix id | B mix name | C used for | D tested | E VP ratio | F flag
      G/H/I/J  component A name / id / VP / goal ppm
      T/U/V/W  component B name / id / VP / goal ppm
      AG/AH/AI mineral oil ml / odor sccm / total sccm
    """

    rows: list[dict] = []
    mixes: dict[int, dict] = {}

    for row in sheets["Mixes"][1:]:
        mix_id = _int(row, 0)
        mix_name = _text(row, 1)

        if mix_id is None or not mix_name:
            continue

        flag = _text(row, 5)
        vp_ratio = _num(row, 4)

        # The sheet flags mismatch in prose ("big VP mismatch", "HORRIBLE,
        # big VP mismatch") and leaves the cell blank when the components
        # were chosen to have comparable volatility.
        vp_mismatch = "mismatch" in flag.lower()

        mixes[mix_id] = {
            "mix_name": mix_name,
            "purpose": _text(row, 2),
            "tested": _text(row, 3).lower() == "y",
            "vp_ratio": vp_ratio,
            "vp_mismatch": vp_mismatch,
            "flag_note": flag,
        }

        for slot, (name_c, id_c, vp_c, ppm_c) in (
            ("A", (6, 7, 8, 9)),
            ("B", (19, 20, 21, 22)),
        ):
            comp_id = _int(row, id_c)
            if comp_id is None:
                continue

            rows.append(
                {
                    "mix_odor_id": mix_id,
                    "mix_name": mix_name,
                    "component_slot": slot,
                    "component_odor_id": comp_id,
                    "component_name": _text(row, name_c).lower(),
                    "nominal_fraction": _round(_num(row, ppm_c), 4),
                    "component_vp_mmhg": _num(row, vp_c),
                    "mix_vp_ratio": _round(vp_ratio, 4),
                    "vp_mismatch": int(vp_mismatch),
                    "vp_mismatch_note": flag,
                    "purpose": _text(row, 2),
                }
            )

    return rows, mixes


def build_dictionary(sheets, mixes: dict[int, dict]) -> list[dict]:
    """Parse `Mono` and merge in the mixes to give one row per odor_id.

    Column letters from the sheet header:
      A id | B name | C CAS | W notes | Y descriptors
      AA class I/II | AB chemical group | AC chemical class | AD VP
    """

    out: dict[int, dict] = {}

    for row in sheets["Mono"][1:]:
        odor_id = _int(row, 0)
        name = _text(row, 1)

        if odor_id is None or not name or name.lower() in MONO_PLACEHOLDERS:
            continue

        # `Class I or II (OR-based)` is only filled in where the lab has
        # assigned it; blank means unassigned, not "neither".
        odor_class = _text(row, 26)

        out[odor_id] = {
            "odor_id": odor_id,
            "odor_name": name.lower(),
            "role": "control" if odor_id == 0 else "single",
            "odor_class": odor_class,
            "chemical_group": _text(row, 27),
            "chemical_class": _text(row, 28),
            "vapor_pressure_mmhg": _num(row, 29),
            "cas": _text(row, 2),
            "descriptors": _text(row, 24),
            "notes": _text(row, 22),
            "mix_components": "",
            "mix_purpose": "",
            "vp_mismatch": "",
        }

    for mix_id, mix in mixes.items():
        out[mix_id] = {
            "odor_id": mix_id,
            "odor_name": mix["mix_name"],
            "role": "mix",
            "odor_class": "",
            "chemical_group": "",
            "chemical_class": "",
            "vapor_pressure_mmhg": None,
            "cas": "",
            "descriptors": "",
            "notes": mix["flag_note"],
            "mix_components": "",
            "mix_purpose": mix["purpose"],
            "vp_mismatch": int(mix["vp_mismatch"]),
        }

    return [out[k] for k in sorted(out)]


def build_panels(sheets) -> list[dict]:
    """Parse the print sheets into one row per (panel, vial position).

    `Print v3` and `Print v1` differ: v3 targets a delivered ppm per
    component directly, v1 mixes by liquid volume fraction. Both expose the
    resulting per-component ppm, which is the column analyses should use.
    """

    # (vial, odor id, mix name, A id, A ppm, B id, B ppm, odor sccm, total sccm)
    layouts = {
        "Print v3": dict(
            vial=0, odor=1, mix=2, a_id=4, a_ppm=5, b_id=8, b_ppm=9, sccm=11, total=14
        ),
        "Print v1": dict(
            vial=0, odor=1, mix=2, a_id=4, a_ppm=20, b_id=8, b_ppm=21, sccm=14, total=15
        ),
    }

    rows: list[dict] = []

    for panel_name, sheet_name, limit in PANELS:
        cols = layouts[sheet_name]
        n = 0

        for row in sheets[sheet_name][1:]:
            vial = _int(row, cols["vial"])
            odor_id = _int(row, cols["odor"])

            # Trailing rows carry a vial number but no loaded odor, and the
            # sheets end in unrelated notes ("Made by", "Goal:").
            if vial is None or odor_id is None:
                continue

            n += 1
            if limit is not None and n > limit:
                break

            a_ppm = _num(row, cols["a_ppm"])
            b_ppm = _num(row, cols["b_ppm"])

            rows.append(
                {
                    "panel": panel_name,
                    "vial_position": vial,
                    "odor_id": odor_id,
                    "mix_name": _text(row, cols["mix"]).replace("na", ""),
                    "component_a_odor_id": _int(row, cols["a_id"]),
                    "component_a_ppm": _round(a_ppm, 4),
                    "component_b_odor_id": _int(row, cols["b_id"]),
                    "component_b_ppm": _round(b_ppm, 4),
                    "odor_sccm": _num(row, cols["sccm"]),
                    "total_sccm": _num(row, cols["total"]),
                }
            )

    return rows


def build_mixture_feasibility(panel_rows: list[dict]) -> list[dict]:
    """Per (panel, mix, component): can the mixture prediction be built?

    Stage 5 predicts a mixture response from the *measured* responses to its
    component singles. That needs each component to appear on the same panel
    as a single, at the concentration it takes inside the mix. Where the two
    concentrations differ, the prediction relies on a dose extrapolation that
    was never measured, so the size of that gap is what decides whether a mix
    is usable.
    """

    rows: list[dict] = []

    for panel in {r["panel"] for r in panel_rows}:
        vials = [r for r in panel_rows if r["panel"] == panel]

        # ppm at which each odor is presented alone on this panel
        singles = {
            v["component_a_odor_id"]: v["component_a_ppm"]
            for v in vials
            if not v["mix_name"] and v["component_a_odor_id"] is not None
        }

        for vial in vials:
            if not vial["mix_name"]:
                continue

            for slot in ("a", "b"):
                comp_id = vial[f"component_{slot}_odor_id"]
                in_mix = vial[f"component_{slot}_ppm"]

                if comp_id is None:
                    continue

                single = singles.get(comp_id)
                ratio = (
                    in_mix / single
                    if single not in (None, 0) and in_mix is not None
                    else None
                )

                if single is None:
                    status = "component not on panel"
                elif ratio is None:
                    status = "unknown"
                elif abs(ratio - 1.0) < 1e-6:
                    status = "exact"
                elif 0.5 <= ratio <= 2.0:
                    status = "interpolation"
                else:
                    status = "extrapolation"

                rows.append(
                    {
                        "panel": panel,
                        "mix_odor_id": vial["odor_id"],
                        "mix_name": vial["mix_name"],
                        "component_slot": slot.upper(),
                        "component_odor_id": comp_id,
                        "ppm_in_mix": in_mix,
                        "ppm_as_single_on_panel": single,
                        "dose_ratio": _round(ratio, 3),
                        "status": status,
                    }
                )

    order = {"panel_7": 0, "panel_12": 1, "panel_16": 2}
    rows.sort(key=lambda r: (order.get(r["panel"], 9), r["mix_odor_id"], r["component_slot"]))
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path.name}")

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"  wrote {path.name:<26} {len(rows):>3} rows")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path, help="Moss Lab odor .xlsx")
    args = parser.parse_args()

    sheets = read_sheets(args.workbook)

    mixture_rows, mixes = build_mixtures(sheets)
    dictionary_rows = build_dictionary(sheets, mixes)
    panel_rows = build_panels(sheets)

    # Backfill the readable component list onto the mix rows of the dictionary.
    components: dict[int, list[str]] = {}
    for row in mixture_rows:
        components.setdefault(row["mix_odor_id"], []).append(row["component_name"])

    for row in dictionary_rows:
        if row["role"] == "mix":
            row["mix_components"] = " + ".join(components.get(row["odor_id"], []))

    print(f"Reading {args.workbook.name}")
    write_csv(OUT_DIR / "odor_dictionary.csv", dictionary_rows)
    write_csv(OUT_DIR / "mixture_composition.csv", mixture_rows)
    write_csv(OUT_DIR / "odor_panels.csv", panel_rows)
    write_csv(OUT_DIR / "mixture_feasibility.csv", build_mixture_feasibility(panel_rows))


if __name__ == "__main__":
    main()
