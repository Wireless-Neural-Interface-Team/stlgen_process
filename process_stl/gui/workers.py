"""Background QThread workers wrapping the process_stl library.

Every slow operation (reading STLs, sampling points, computing centers of
mass) runs here, off the UI thread, and reports back through Qt signals.
"""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from ..regions import load_regions
from ..stats import load_or_compute
from ..viz import read_mesh


class RegionsWorker(QThread):
    """Load the regions of one directory and compute their centers of mass."""

    log = Signal(str)
    finished_ok = Signal(dict)  # label -> Region
    failed = Signal(str)

    def __init__(self, src_dir, files, n_points, com_method, cache_dir, force, parent=None):
        super().__init__(parent)
        self.src_dir = src_dir
        self.files = files  # load_regions keyword -> file name
        self.n_points = n_points
        self.com_method = com_method
        self.cache_dir = cache_dir
        self.force = force

    def run(self):
        try:
            regions = load_regions(self.src_dir, n_points=self.n_points,
                                   com_method=self.com_method, cache_dir=self.cache_dir,
                                   force=self.force, log=self.log.emit, **self.files)
            self.finished_ok.emit(regions)
        except Exception as exc:
            self.failed.emit(str(exc))


class MeshWorker(QThread):
    """Read STL meshes for the "surface" display style."""

    log = Signal(str)
    finished_ok = Signal(dict)  # label -> pv.PolyData
    failed = Signal(str)

    def __init__(self, paths, parent=None):
        super().__init__(parent)
        self.paths = paths  # label -> Path

    def run(self):
        try:
            meshes = {}
            for label, path in self.paths.items():
                self.log.emit(f"  reading mesh {path.name}...")
                meshes[label] = read_mesh(path)
            self.finished_ok.emit(meshes)
        except Exception as exc:
            self.failed.emit(str(exc))


class StatsWorker(QThread):
    """Collect (or reload) the bbox / center of mass table of several folders."""

    log = Signal(str)
    finished_ok = Signal(object)  # DataFrame
    failed = Signal(str)

    def __init__(self, folders, recursive, data_file, force, axes, parent=None):
        super().__init__(parent)
        self.folders = folders
        self.recursive = recursive
        self.data_file = data_file
        self.force = force
        self.axes = axes

    def run(self):
        try:
            df = load_or_compute(self.folders, self.recursive, self.data_file,
                                 self.force, self.axes, log=self.log.emit)
            self.finished_ok.emit(df)
        except Exception as exc:
            self.failed.emit(str(exc))
