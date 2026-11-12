"""dLGN insertion protocol from the left-hemisphere centers of mass.

The probe enters at (AP = C_VS AP, ML = C_dLGN ML, DV = 0), is tilted by
ML_rot around the ML axis and inserted by Insert_tr to reach C_dLGN.

A protocol is a dict with ML_rot (rad), ML_rot_deg, ML_tr, AP_tr, Insert_tr
(mm). It can be exported to / imported from JSON, to apply the protocol of one
geometry to another one, and appended to a CSV table.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

#: CSV column -> protocol key (angle in degrees, translations in mm)
CSV_COLUMNS = {
    "insertion_angle": "ML_rot_deg",
    "MD_translation": "ML_tr",
    "AP_translation": "AP_tr",
    "insertion_depth": "Insert_tr",
}


def get_insertion_protocol(regions, verbose=False, log=print) -> dict:
    """Insertion from above the visual cortex to the dLGN, in the AP-DV plane."""
    C_dLGN = regions["dLGN"].lcom
    C_VS = regions["visual_cortex"].lcom

    di_AP = C_dLGN[0] - C_VS[0]
    di_DV = C_dLGN[2]
    prot = {
        "ML_tr": float(C_dLGN[1]),
        "AP_tr": float(C_VS[0]),
        "Insert_tr": float(np.hypot(di_AP, di_DV)),
        "ML_rot": float(np.arctan(di_AP / di_DV)),
    }
    prot["ML_rot_deg"] = float(np.degrees(prot["ML_rot"]))

    if verbose:
        log(format_protocol(prot))
    return prot


def make_protocol(ML_tr, AP_tr, Insert_tr, ML_rot=None, ML_rot_deg=None) -> dict:
    """Protocol dict from its values; the rotation is given in rad or in degrees."""
    if ML_rot is None:
        if ML_rot_deg is None:
            raise ValueError("the protocol needs ML_rot (rad) or ML_rot_deg")
        ML_rot = np.radians(ML_rot_deg)
    return {"ML_tr": float(ML_tr), "AP_tr": float(AP_tr), "Insert_tr": float(Insert_tr),
            "ML_rot": float(ML_rot), "ML_rot_deg": float(np.degrees(ML_rot))}


def protocol_lines(prot) -> list[str]:
    return [
        f"ML rotation: {prot['ML_rot_deg']:.2f}° ({prot['ML_rot']:.4f} rad)",
        f"ML translation: {prot['ML_tr']:.3f} mm",
        f"AP translation: {prot['AP_tr']:.3f} mm",
        f"Insertion depth: {prot['Insert_tr']:.3f} mm",
    ]


def format_protocol(prot) -> str:
    return "\n".join(protocol_lines(prot))


def entry_point(prot) -> np.ndarray:
    """(AP, ML, DV) of the insertion entry point."""
    return np.array([prot["AP_tr"], prot["ML_tr"], 0.0])


def target_point(prot) -> np.ndarray:
    """(AP, ML, DV) reached at the end of the insertion."""
    return entry_point(prot) + prot["Insert_tr"] * np.array(
        [np.sin(prot["ML_rot"]), 0.0, np.cos(prot["ML_rot"])])


def target_offset(prot, regions) -> np.ndarray:
    """(AP, ML, DV) from the dLGN left center of mass to the protocol target,
    e.g. to test the protocol of another geometry on this one."""
    return target_point(prot) - np.asarray(regions["dLGN"].lcom)


def in_craniotomy_area(coordinates, prot, thickness=1, radius=1) -> np.ndarray:
    """Mask of the points within radius (AP-ML plane) of the entry point and
    |DV| < thickness."""
    center = entry_point(prot)[:2]  # (AP, ML), same order as coordinates[:, :2]
    in_circle = np.linalg.norm(coordinates[:, :2] - center, axis=1) < radius
    in_section = np.abs(coordinates[:, 2]) < thickness
    return in_circle & in_section


def protocol_report(prot, regions=None, **info) -> dict:
    """JSON-ready summary: the protocol, extra info (name, origin, settings...)
    and, if given, the files and centers of mass of the geometry it comes from."""
    report = {**info}
    if regions:
        src_dirs = {str(r.path.parent) for r in regions.values()}
        report.update({
            "src_dir": src_dirs.pop() if len(src_dirs) == 1 else sorted(src_dirs),
            "files": {k: r.path.name for k, r in regions.items()},
            "com": {k: np.asarray(r.com).tolist() for k, r in regions.items()},
            "lcom": {k: np.asarray(r.lcom).tolist()
                     for k, r in regions.items() if r.is_double},
        })
    report["protocol"] = prot
    return report


def save_protocol(path, prot, regions=None, **info) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(protocol_report(prot, regions, **info), indent=2))
    return path


def load_protocol(path) -> dict:
    """Read a protocol JSON: a file written by save_protocol, or a bare protocol dict."""
    data = json.loads(Path(path).read_text())
    prot = data.get("protocol", data) if isinstance(data, dict) else None
    if not isinstance(prot, dict):
        raise ValueError(f"{path}: no protocol found")
    missing = [k for k in ("ML_tr", "AP_tr", "Insert_tr") if k not in prot]
    if missing:
        raise ValueError(f"{path}: protocol is missing {missing}")
    return make_protocol(prot["ML_tr"], prot["AP_tr"], prot["Insert_tr"],
                         prot.get("ML_rot"), prot.get("ML_rot_deg"))


def append_protocol_csv(path, prot, decimals=4) -> bool:
    """Append the protocol as one row of a CSV table (columns: CSV_COLUMNS).

    The file is created with its header row if it does not exist. Returns True
    if the file was created, False if the row was appended.
    """
    path = Path(path)
    header = list(CSV_COLUMNS)
    row = [round(float(prot[key]), decimals) for key in CSV_COLUMNS.values()]

    exists = path.is_file() and path.stat().st_size > 0
    if exists:
        with path.open(newline="") as f:
            first = next(csv.reader(f), [])
        if [c.strip() for c in first] != header:
            raise ValueError(f"{path} has columns {first}, expected {header}")
        # Make sure the new row starts on its own line
        with path.open("rb") as f:
            f.seek(-1, 2)
            needs_newline = f.read(1) not in (b"\n", b"\r")
        if needs_newline:
            with path.open("a", newline="") as f:
                f.write("\n")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(header)
        writer.writerow(row)
    return not exists


def count_csv_rows(path) -> int | None:
    """Number of data rows of a protocol CSV (None if the file does not exist)."""
    path = Path(path)
    if not path.is_file():
        return None
    with path.open(newline="") as f:
        return max(sum(1 for r in csv.reader(f) if r) - 1, 0)
