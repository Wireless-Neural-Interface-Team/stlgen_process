"""STL reading and triangle-array geometry shared by every tool of the package.

A mesh is handled as an (N, 3, 3) float64 array of triangle vertices, which is
what STL stores (no shared vertices).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def load_stl(path) -> np.ndarray:
    """Return an (N, 3, 3) float64 array of triangle vertices (binary or ASCII STL)."""
    data = Path(path).read_bytes()

    # Binary STL: 80-byte header + uint32 triangle count + 50 bytes per triangle
    if len(data) >= 84:
        n = int.from_bytes(data[80:84], "little")
        if len(data) == 84 + 50 * n:
            dtype = np.dtype([("normal", "<f4", (3,)),
                              ("verts", "<f4", (3, 3)),
                              ("attr", "<u2")])
            tris = np.frombuffer(data, dtype=dtype, count=n, offset=84)
            return tris["verts"].astype(np.float64)

    # Otherwise treat it as ASCII
    text = data.decode("ascii", errors="ignore")
    nums = re.findall(r"vertex\s+(\S+)\s+(\S+)\s+(\S+)", text)
    if not nums:
        raise ValueError(f"no triangles found in {path} (not a valid STL?)")
    return np.array(nums, dtype=np.float64).reshape(-1, 3, 3)


def is_closed(tris, decimals=6) -> bool:
    """A mesh is watertight if every edge is shared by exactly 2 triangles."""
    # STL has no shared vertices, so merge identical coordinates first
    _, idx = np.unique(tris.reshape(-1, 3).round(decimals), axis=0, return_inverse=True)
    faces = idx.reshape(-1, 3)
    edges = np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return bool(np.all(counts == 2))


def bbox(tris) -> tuple[np.ndarray, np.ndarray]:
    """(min, max) corners of the axis-aligned bounding box."""
    pts = tris.reshape(-1, 3)
    return pts.min(axis=0), pts.max(axis=0)


def mesh_com(tris) -> tuple[np.ndarray, bool]:
    """Solid center of mass of a closed mesh (uniform density), or the
    area-weighted surface centroid if it is open. Returns (com, is_solid)."""
    # Work relative to the bbox center to limit floating-point error
    bb_min, bb_max = bbox(tris)
    origin = (bb_min + bb_max) / 2
    v0, v1, v2 = (tris - origin).transpose(1, 0, 2)

    # Signed volume of the tetrahedron formed by each triangle and the origin
    vols = np.einsum("ij,ij->i", v0, np.cross(v1, v2)) / 6.0
    volume = vols.sum()

    if abs(volume) > 0 and is_closed(tris):
        # Volume-weighted centroid of the tetrahedra = solid center of mass
        com = (vols[:, None] * (v0 + v1 + v2) / 4).sum(axis=0) / volume
        return com + origin, True

    # Open mesh: area-weighted surface centroid instead
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    com = (areas[:, None] * (v0 + v1 + v2) / 3).sum(axis=0) / areas.sum()
    return com + origin, False


def mesh_stats(tris) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return (center_of_mass, bbox_size, is_solid)."""
    bb_min, bb_max = bbox(tris)
    com, solid = mesh_com(tris)
    return com, bb_max - bb_min, solid


def sample_surface(tris, n_points, seed=0) -> np.ndarray:
    """Uniformly sample n_points on the mesh surface (area-weighted), as float32."""
    if n_points >= len(tris) * 3:
        # Asked for more points than the mesh has vertices: keep the vertices
        return np.unique(tris.reshape(-1, 3), axis=0).astype(np.float32)

    rng = np.random.default_rng(seed)
    v0, v1, v2 = tris.transpose(1, 0, 2)
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    face = rng.choice(len(tris), size=n_points, p=areas / areas.sum())
    # Uniform barycentric coordinates
    r1, r2 = rng.random((2, n_points))
    s = np.sqrt(r1)
    a, b, c = 1 - s, s * (1 - r2), s * r2
    pts = a[:, None] * v0[face] + b[:, None] * v1[face] + c[:, None] * v2[face]
    return pts.astype(np.float32)


def find_stl_files(folder, recursive=False) -> list[Path]:
    """Sorted STL files of a folder."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"folder not found: {folder}")
    pattern = "**/*" if recursive else "*"
    return sorted(p for p in folder.glob(pattern)
                  if p.is_file() and p.suffix.lower() == ".stl")
