"""Brain regions of one geometry: STL -> cached point cloud + centers of mass.

All the meshes of one geometry live in a single directory. Each region file is
given explicitly or found with the default glob pattern of REGIONS.

Each STL is converted once into an under-sampled point cloud (area-weighted
surface sampling) saved as a .npz in cache_dir (relative to the source
directory), together with its centers of mass. Later loads read the .npz
instead of the STL, as long as the STL is unchanged.

Axis convention: x -> AP, y -> ML, z -> DV, in mm, origin at bregma.
The "left" hemisphere is y > 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .mesh import load_stl, mesh_com, sample_surface

# label -> default file pattern, plot color, and whether the region has two
# hemispheres (only the left one, y > 0, is used for the protocol)
REGIONS = {
    "skull":         {"pattern": None,               "color": "#A0A4A4", "is_double": False},
    "whole_brain":   {"pattern": "whole_brain*.stl", "color": "#89BAC3", "is_double": False},
    "dLGN":          {"pattern": "*LGd*.stl",        "color": "#095562", "is_double": True},
    "visual_cortex": {"pattern": "*VIS*.stl",        "color": "#096256", "is_double": True},
}
N_POINTS = 50_000          # points sampled on each STL surface
CACHE_DIR = "_points"      # relative to the source directory
COM_METHODS = {
    "volume": "solid CoM (surface centroid if the mesh is open)",
    "points": "mean of the sampled points (like G01_Compute_CoM.ipynb)",
}
COM_METHOD = "volume"


@dataclass
class Region:
    label: str
    path: Path
    color: str
    is_double: bool
    pts: np.ndarray = field(repr=False)  # (n, 3) sampled surface points
    com: np.ndarray                      # center of mass of the whole region
    lcom: np.ndarray | None = None       # left hemisphere (y > 0), double regions only


def _left(tris):
    """Triangles of the left hemisphere (centroid y > 0)."""
    return tris[tris[:, :, 1].mean(axis=1) > 0]


def _compute_region(path, is_double, n_points, com_method, log):
    tris = load_stl(path)
    pts = sample_surface(tris, n_points)

    if com_method == "points":
        com = pts.mean(axis=0)
        lcom = pts[pts[:, 1] > 0].mean(axis=0) if is_double else None
    elif com_method == "volume":
        com, solid = mesh_com(tris)
        if not solid:
            log(f"  warning: {path.name} is not closed, using surface centroid")
        lcom = mesh_com(_left(tris))[0] if is_double else None
    else:
        raise ValueError(f"unknown com_method: {com_method!r}, choose from {list(COM_METHODS)}")
    return pts, com, lcom


def cache_path(path, n_points=N_POINTS, com_method=COM_METHOD, cache_dir=CACHE_DIR):
    """Where the .npz of an STL is stored (None if caching is disabled)."""
    if cache_dir is None:
        return None
    path = Path(path)
    return path.parent / cache_dir / f"{path.stem}_{n_points}_{com_method}.npz"


def load_region(label, path, color, is_double, n_points=N_POINTS,
                com_method=COM_METHOD, cache_dir=CACHE_DIR, force=False, log=print):
    """Load one region, from its .npz cache if it is up to date, else from the STL."""
    path = Path(path)
    cache = cache_path(path, n_points, com_method, cache_dir)

    stat = path.stat()
    key = np.array([stat.st_size, stat.st_mtime_ns], dtype=np.int64)
    if cache is not None and cache.is_file() and not force:
        data = np.load(cache)
        if np.array_equal(data["key"], key) and (not is_double or "lcom" in data):
            log(f"  [{label}] {path.name} (cached)")
            return Region(label, path, color, is_double, data["pts"], data["com"],
                          data["lcom"] if is_double else None)

    log(f"  [{label}] reading {path.name}...")
    pts, com, lcom = _compute_region(path, is_double, n_points, com_method, log)

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        extra = {"lcom": lcom} if is_double else {}
        np.savez(cache, key=key, pts=pts, com=com, **extra)
    return Region(label, path, color, is_double, pts, com, lcom)


def find_file(src_dir, pattern) -> Path:
    """Return the single file in src_dir matching pattern (a file name or glob)."""
    matches = sorted(Path(src_dir).glob(pattern))
    if len(matches) != 1:
        found = [m.name for m in matches] or "nothing"
        raise FileNotFoundError(
            f"expected one file matching {pattern!r} in {src_dir}, found {found}")
    return matches[0]


def load_regions(src_dir, whole_brain=None, visual_cortex=None, dlgn=None, skull=None,
                 n_points=N_POINTS, com_method=COM_METHOD, cache_dir=CACHE_DIR,
                 force=False, log=print) -> dict[str, Region]:
    """Load the regions of one geometry directory.

    whole_brain, visual_cortex, dlgn, skull: file name (or glob) of each region
    inside src_dir. None uses the default pattern of REGIONS; the skull is only
    loaded when given.
    """
    src_dir = Path(src_dir)
    if not src_dir.is_dir():
        raise FileNotFoundError(f"directory not found: {src_dir}")

    given = {"skull": skull, "whole_brain": whole_brain,
             "dLGN": dlgn, "visual_cortex": visual_cortex}
    regions = {}
    for label, ppt in REGIONS.items():
        pattern = given[label] or ppt["pattern"]
        if pattern is None:
            continue
        regions[label] = load_region(label, find_file(src_dir, pattern), ppt["color"],
                                     ppt["is_double"], n_points, com_method,
                                     cache_dir, force, log)
    return regions
