"""Electrode section inside the target regions: the length of the probe
inside the visual cortex and the dLGN, computed exactly on the meshes.

The probe is the segment entry + t * u, t in [0, Insert_tr], with
entry = (AP_tr, ML_tr, 0) and u = (sin ML_rot, 0, cos ML_rot), as in
protocol.py. Its line is intersected with the triangles of each closed mesh
(Moller-Trumbore), the crossings are paired into inside intervals, and the
intervals are clipped to [0, Insert_tr].
"""

from __future__ import annotations

import numpy as np

from .mesh import load_stl

#: regions whose section is measured
TARGETS = ("visual_cortex", "dLGN")
#: short names of the targets, for labels and tables
SHORT = {"visual_cortex": "VS", "dLGN": "dLGN"}

_CACHE: dict = {}  # (path, mtime) of the targets -> meshes


class RegionMesh:
    """Triangles of one region, for line intersection."""

    def __init__(self, tris, com=None):
        tris = np.asarray(tris, dtype=np.float64)
        self.v0 = tris[:, 0]
        self.e1 = tris[:, 1] - self.v0
        self.e2 = tris[:, 2] - self.v0
        self.ymin = tris[:, :, 1].min(axis=1)
        self.ymax = tris[:, :, 1].max(axis=1)
        self.com = None if com is None else np.asarray(com, dtype=np.float64)

    @classmethod
    def from_stl(cls, path, com=None):
        return cls(load_stl(path), com)

    def crossings(self, origin, direction) -> np.ndarray:
        """Sorted t of the crossings of the line origin + t * direction.

        The probe stays in the plane y = origin.y, so only the triangles
        spanning that y are tested.
        """
        m = (self.ymin <= origin[1]) & (self.ymax >= origin[1])
        if not m.any():
            return np.empty(0)
        v0, e1, e2 = self.v0[m], self.e1[m], self.e2[m]
        p = np.cross(direction, e2)
        det = np.einsum("ij,ij->i", e1, p)
        ok = np.abs(det) > 1e-12
        inv = np.divide(1.0, det, out=np.zeros_like(det), where=ok)
        s = origin - v0
        u = np.einsum("ij,ij->i", s, p) * inv
        q = np.cross(s, e1)
        v = (q @ direction) * inv
        t = np.einsum("ij,ij->i", e2, q) * inv
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1)
        t = np.sort(t[hit])
        if t.size > 1:  # a crossing on a shared edge is found twice
            t = t[np.concatenate([[True], np.diff(t) > 1e-9])]
        return t

    def intervals(self, origin, direction) -> np.ndarray:
        """(n, 2) inside intervals [t_in, t_out] along the line."""
        t = self.crossings(origin, direction)
        if t.size % 2:  # grazing hit or open mesh: drop the unpaired crossing
            t = t[:-1]
        return t.reshape(-1, 2)


def _length(intervals, a, b) -> float:
    if intervals.size == 0:
        return 0.0
    return float(np.sum(np.clip(intervals[:, 1], a, b) - np.clip(intervals[:, 0], a, b)))


def line(AP, ML, rot_deg):
    """(entry point, unit direction) of a probe."""
    rot = np.radians(rot_deg)
    return np.array([AP, ML, 0.0]), np.array([np.sin(rot), 0.0, np.cos(rot)])


def protocol_lengths(prot, meshes) -> dict:
    """Electrode length (mm) inside each target region for a protocol dict."""
    origin, direction = line(prot["AP_tr"], prot["ML_tr"], prot["ML_rot_deg"])
    return {k: _length(meshes[k].intervals(origin, direction), 0.0, prot["Insert_tr"])
            for k in TARGETS}


def format_lengths(lengths) -> str:
    """Text like "dLGN 1.234 mm, VS 0.567 mm"."""
    return ", ".join(f"{SHORT[k]} {lengths[k]:.3f} mm" for k in ("dLGN", "visual_cortex"))


def load_meshes(regions, log=print) -> dict:
    """RegionMesh of the visual cortex and dLGN of loaded regions (cached while
    the STL files are unchanged)."""
    key = tuple((str(regions[k].path), regions[k].path.stat().st_mtime_ns) for k in TARGETS)
    if key not in _CACHE:
        meshes = {}
        for k in TARGETS:
            reg = regions[k]
            log(f"  reading triangles of {reg.path.name}...")
            meshes[k] = RegionMesh.from_stl(reg.path, reg.lcom if reg.is_double else reg.com)
        _CACHE.clear()
        _CACHE[key] = meshes
    return _CACHE[key]
