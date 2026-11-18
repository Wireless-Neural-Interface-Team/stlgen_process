"""dLGN insertion protocol from the left-hemisphere centers of mass.

The probe enters at (AP = C_VS AP, ML = C_dLGN ML, DV = 0), is tilted by
ML_rot around the ML axis and inserted by Insert_tr to reach C_dLGN.

A protocol is a dict with ML_rot (rad), ML_rot_deg, ML_tr, AP_tr, Insert_tr
(mm) and AP_tr_rel. It can be exported to / imported from JSON, to apply the
protocol of one geometry to another one, and appended to a CSV table.

AP_tr_rel is AP_tr divided by the bregma-lambda distance of the geometry the
protocol was made on (None if that distance is unknown). Applied to another
skull, the relative AP translation is AP_tr_rel times that skull's
bregma-lambda distance (see apply_ap_mode). The distance is read from a JSON
of the geometry directory (key BREGMA_LAMBDA_KEY, see find_bregma_lambda).
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
#: key of the bregma-lambda distance (mm) in the JSON of a geometry directory
BREGMA_LAMBDA_KEY = "bregma_lambda_distance_skull"
#: how the AP translation of a protocol is applied to a geometry
AP_MODES = {
    "absolute": "AP_tr as saved (mm)",
    "relative": "AP_tr_rel x bregma-lambda distance of the current skull",
}


def read_bregma_lambda(path, key=BREGMA_LAMBDA_KEY) -> float:
    """Bregma-lambda distance (mm) stored under key in a JSON file."""
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"{Path(path).name}: cannot read the JSON ({exc})") from exc
    value = data.get(key) if isinstance(data, dict) else None
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{Path(path).name}: no positive '{key}' value")
    return float(value)


def bregma_lambda_files(src_dir, key=BREGMA_LAMBDA_KEY) -> dict[Path, float]:
    """JSON file -> bregma-lambda distance, for the JSON files of src_dir that have key."""
    found = {}
    for path in sorted(Path(src_dir).glob("*.json")):
        try:
            found[path] = read_bregma_lambda(path, key)
        except ValueError:
            continue
    return found


def find_bregma_lambda(src_dir, key=BREGMA_LAMBDA_KEY):
    """(distance, json path) of the bregma-lambda distance found in the JSON
    files of src_dir, or (None, None) if no JSON has it.

    Raises ValueError if several JSON files give different distances.
    """
    found = bregma_lambda_files(src_dir, key)
    if not found:
        return None, None
    if len(set(found.values())) > 1:
        raise ValueError(f"different {key} in {src_dir}: "
                         + ", ".join(f"{p.name}={v}" for p, v in found.items()))
    path, value = next(iter(found.items()))
    return value, path


def _relative(AP_tr, bregma_lambda):
    return float(AP_tr) / bregma_lambda if bregma_lambda else None


def get_insertion_protocol(regions, bregma_lambda=None, verbose=False, log=print) -> dict:
    """Insertion from above the visual cortex to the dLGN, in the AP-DV plane.

    bregma_lambda: bregma-lambda distance (mm) of this geometry, to also give
    the relative AP translation AP_tr_rel (None if not given).
    """
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
    prot["AP_tr_rel"] = _relative(prot["AP_tr"], bregma_lambda)

    if verbose:
        log(format_protocol(prot))
    return prot


def make_protocol(ML_tr, AP_tr, Insert_tr, ML_rot=None, ML_rot_deg=None,
                  AP_tr_rel=None) -> dict:
    """Protocol dict from its values; the rotation is given in rad or in degrees."""
    if ML_rot is None:
        if ML_rot_deg is None:
            raise ValueError("the protocol needs ML_rot (rad) or ML_rot_deg")
        ML_rot = np.radians(ML_rot_deg)
    return {"ML_tr": float(ML_tr), "AP_tr": float(AP_tr), "Insert_tr": float(Insert_tr),
            "ML_rot": float(ML_rot), "ML_rot_deg": float(np.degrees(ML_rot)),
            "AP_tr_rel": None if AP_tr_rel is None else float(AP_tr_rel)}


def with_relative_ap(prot, bregma_lambda) -> dict:
    """Copy of the protocol with AP_tr_rel recomputed as AP_tr / bregma_lambda."""
    if not bregma_lambda:
        raise ValueError("the bregma-lambda distance of the geometry is unknown")
    return {**prot, "AP_tr_rel": _relative(prot["AP_tr"], bregma_lambda)}


def apply_ap_mode(prot, mode="absolute", bregma_lambda=None) -> dict:
    """The protocol as applied to a geometry of bregma-lambda distance bregma_lambda.

    "absolute" returns it unchanged; "relative" sets AP_tr = AP_tr_rel *
    bregma_lambda (AP_tr_rel is kept, so the result can be saved as is).
    """
    if mode == "absolute":
        return prot
    if mode != "relative":
        raise ValueError(f"unknown AP mode: {mode!r}, choose from {list(AP_MODES)}")
    if prot.get("AP_tr_rel") is None:
        raise ValueError("the protocol has no relative AP translation (AP_tr_rel)")
    if not bregma_lambda:
        raise ValueError("the bregma-lambda distance of the geometry is unknown")
    return {**prot, "AP_tr": float(prot["AP_tr_rel"]) * bregma_lambda}


def protocol_lines(prot) -> list[str]:
    return [
        f"ML rotation: {prot['ML_rot_deg']:.2f}° ({prot['ML_rot']:.4f} rad)",
        f"ML translation: {prot['ML_tr']:.3f} mm",
        f"AP translation: {prot['AP_tr']:.3f} mm"
        + ("" if prot.get("AP_tr_rel") is None
           else f" ({prot['AP_tr_rel']:.4f} x bregma-lambda)"),
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
                         prot.get("ML_rot"), prot.get("ML_rot_deg"), prot.get("AP_tr_rel"))


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
