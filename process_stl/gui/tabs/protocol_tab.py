"""Tab 1: load the regions of one geometry, compute their centers of mass, and
build / edit / compare insertion protocols on it.

Each protocol is either computed from the centers of mass, loaded from a JSON
file (e.g. the protocol of another skull), or typed by hand, and can then be
edited; every protocol shown is drawn on the current geometry.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ... import viz
from ...mesh import find_stl_files
from ...protocol import (
    CSV_COLUMNS,
    append_protocol_csv,
    count_csv_rows,
    entry_point,
    get_insertion_protocol,
    load_protocol,
    make_protocol,
    save_protocol,
    target_offset,
    target_point,
)
from ...regions import CACHE_DIR, COM_METHOD, COM_METHODS, N_POINTS, REGIONS
from ..widgets import CheckMenu, ColorButton, LogBox, Viewer3D, path_row
from ..workers import MeshWorker, RegionsWorker

NONE_ITEM = "(none)"
OTHER = "other"  # display settings shared by the extra parts
#: REGIONS label -> load_regions keyword
REGION_KWARGS = {"skull": "skull", "whole_brain": "whole_brain",
                 "dLGN": "dlgn", "visual_cortex": "visual_cortex"}
JSON_FILTER = "Protocol JSON (*.json)"
CSV_FILTER = "Protocol table, adds a row (*.csv)"
#: protocol key -> (label, unit, range, decimals, step) of its spin box
PROTOCOL_SPINS = {
    "ML_rot_deg": ("ML rotation:", "°", (-90.0, 90.0), 2, 0.5),
    "ML_tr": ("ML translation:", " mm", (-20.0, 20.0), 3, 0.05),
    "AP_tr": ("AP translation:", " mm", (-20.0, 20.0), 3, 0.05),
    "Insert_tr": ("Insertion depth:", " mm", (0.0, 30.0), 3, 0.05),
}


def _guess_file(label, names):
    """Default file of a region among the STL names of the directory."""
    pattern = REGIONS[label]["pattern"]
    if pattern is None:  # skull: no fixed name, pick a file mentioning it
        matches = [n for n in names if "skull" in n.lower()]
    else:
        matches = [n for n in names if fnmatch(n, pattern)]
    return matches[0] if len(matches) == 1 else None


def _fmt(v):
    return "(" + ", ".join(f"{x:.3f}" for x in v) + ")"


def _color_icon(color) -> QIcon:
    pixmap = QPixmap(12, 12)
    pixmap.fill(QColor(color))
    return QIcon(pixmap)


@dataclass
class ProtocolEntry:
    """One protocol of the tab."""

    name: str
    color: str
    prot: dict | None = None
    origin: str = ""             # "CoM of <geometry>", "<file>.json", "manual"
    modified: bool = False       # edited since computed / loaded
    regions: dict | None = None  # geometry it was computed from (JSON export)
    settings: dict = field(default_factory=dict)
    visible: bool = True

    def title(self) -> str:
        text = self.name
        if self.origin:
            text += f" — {self.origin}"
        return text + (" (edited)" if self.modified else "")


class ProtocolTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.regions: dict = {}
        self._regions_key = None       # settings the regions were computed with
        self.protocols: list[ProtocolEntry] = []
        self._pending_protocol: ProtocolEntry | None = None  # waiting for the CoM
        self._updating = False         # filling the protocol widgets
        self._meshes: dict = {}        # str(path) -> pv.PolyData (surfaces, parts)
        self.parts: list[Path] = []    # other STLs of the directory, shown only
        self._worker = None
        self._mesh_worker = None
        self._camera_set = False

        self.viewer = Viewer3D()

        controls = QScrollArea()
        controls.setWidget(self._build_controls())
        controls.setWidgetResizable(True)
        controls.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        controls.setMinimumWidth(430)
        controls.setMaximumWidth(560)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(controls)
        splitter.addWidget(self.viewer)
        splitter.setStretchFactor(1, 1)
        QVBoxLayout(self).addWidget(splitter)
        # the legend font follows the window size: redraw once resizing stops
        self._resize_timer = QTimer(self, singleShot=True, interval=250)
        self._resize_timer.timeout.connect(self._render)
        self.viewer.resized.connect(self._resize_timer.start)
        self._set_parts([])
        self._on_add_protocol()

    # ------------------------------------------------------------------ UI
    def _build_controls(self) -> QWidget:
        panel = QWidget()
        root = QVBoxLayout(panel)

        # --- source files
        src_box = QGroupBox("Source geometry")
        src_l = QFormLayout(src_box)
        row, self.dir_edit = path_row(self, "dir", placeholder="folder with the region STLs")
        self.dir_edit.editingFinished.connect(self._refresh_files)
        src_l.addRow("Directory:", row)
        self.file_combos = {}
        for label in REGIONS:
            combo = QComboBox()
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(26)
            combo.currentTextChanged.connect(self._refresh_buttons)
            self.file_combos[label] = combo
            src_l.addRow(f"{label}:", combo)
        root.addWidget(src_box)

        # --- centers of mass
        com_box = QGroupBox("Centers of mass")
        com_l = QFormLayout(com_box)
        self.n_points_spin = QSpinBox()
        self.n_points_spin.setRange(1_000, 5_000_000)
        self.n_points_spin.setSingleStep(10_000)
        self.n_points_spin.setValue(N_POINTS)
        self.n_points_spin.setToolTip("points sampled on each STL surface (display, "
                                      "and CoM with the 'points' method)")
        com_l.addRow("Points per STL:", self.n_points_spin)
        self.method_combo = QComboBox()
        for i, (name, desc) in enumerate(COM_METHODS.items()):
            self.method_combo.addItem(name)
            self.method_combo.setItemData(i, desc, Qt.ToolTipRole)
        self.method_combo.setCurrentText(COM_METHOD)
        com_l.addRow("CoM method:", self.method_combo)
        cache_row = QHBoxLayout()
        self.cache_check = QCheckBox("cache in")
        self.cache_check.setChecked(True)
        self.cache_edit = QLineEdit(CACHE_DIR)
        self.cache_edit.setToolTip("folder relative to the source directory "
                                   "(or absolute) where the sampled points are saved")
        self.cache_check.toggled.connect(self.cache_edit.setEnabled)
        cache_row.addWidget(self.cache_check)
        cache_row.addWidget(self.cache_edit)
        com_l.addRow("Points cache:", cache_row)
        self.force_check = QCheckBox("ignore the cache and recompute")
        com_l.addRow(self.force_check)
        self.compute_btn = QPushButton("Compute CoM")
        self.compute_btn.clicked.connect(lambda: self._start_com())
        com_l.addRow(self.compute_btn)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        com_l.addRow(self.progress)
        self.com_table = QTableWidget(0, 3)
        self.com_table.setHorizontalHeaderLabels(["Region", "CoM (AP, ML, DV)", "Left CoM"])
        self.com_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.com_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.com_table.setMaximumHeight(150)
        com_l.addRow(self.com_table)
        root.addWidget(com_box)

        # --- insertion protocols
        prot_box = QGroupBox("Insertion protocol")
        prot_l = QFormLayout(prot_box)
        select_row = QHBoxLayout()
        self.protocol_combo = QComboBox()
        self.protocol_combo.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.protocol_combo.setMinimumContentsLength(20)
        self.protocol_combo.currentIndexChanged.connect(self._on_protocol_selected)
        self.add_btn = QPushButton("Add protocol")
        self.add_btn.setToolTip("add a protocol to compare with the others")
        self.add_btn.clicked.connect(self._on_add_protocol)
        self.remove_btn = QPushButton("Remove")
        self.remove_btn.clicked.connect(self._on_remove_protocol)
        select_row.addWidget(self.protocol_combo, stretch=1)
        select_row.addWidget(self.add_btn)
        select_row.addWidget(self.remove_btn)
        prot_l.addRow(select_row)

        source_row = QHBoxLayout()
        self.from_com_btn = QPushButton("Compute from CoM")
        self.from_com_btn.setToolTip("protocol of the loaded geometry "
                                     "(computes the centers of mass first if needed)")
        self.from_com_btn.clicked.connect(self._on_protocol_from_com)
        self.load_btn = QPushButton("Load…")
        self.load_btn.setToolTip("protocol JSON file, e.g. computed on another geometry")
        self.load_btn.clicked.connect(self._on_load_protocol)
        source_row.addWidget(self.from_com_btn)
        source_row.addWidget(self.load_btn)
        prot_l.addRow(source_row)

        self.origin_label = QLabel("")
        self.origin_label.setWordWrap(True)
        prot_l.addRow("Source:", self.origin_label)

        self.protocol_spins = {}
        for key, (text, suffix, (lo, hi), decimals, step) in PROTOCOL_SPINS.items():
            spin = QDoubleSpinBox()
            spin.setRange(lo, hi)
            spin.setDecimals(decimals)
            spin.setSingleStep(step)
            spin.setSuffix(suffix)
            spin.setKeyboardTracking(False)  # typed values apply on Enter / focus out
            spin.valueChanged.connect(lambda v, k=key: self._on_value_changed(k, v))
            self.protocol_spins[key] = spin
            if key == "ML_rot_deg":
                rot_row = QHBoxLayout()
                self.rad_label = QLabel("")
                rot_row.addWidget(spin, stretch=1)
                rot_row.addWidget(self.rad_label)
                prot_l.addRow(text, rot_row)
            else:
                prot_l.addRow(text, spin)
        self.point_labels = {}
        for key, text in [("entry", "Entry (AP, ML, DV):"), ("target", "Target (AP, ML, DV):"),
                          ("offset", "Target − C_dLGN:")]:
            lab = QLabel("—")
            lab.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self.point_labels[key] = lab
            prot_l.addRow(text, lab)
        self.point_labels["offset"].setToolTip(
            "distance (AP, ML, DV) from the dLGN left center of mass of the loaded "
            "geometry to the protocol target")

        last_row = QHBoxLayout()
        self.view_check = QCheckBox("View")
        self.view_check.setToolTip("draw this protocol in the 3D view")
        self.view_check.toggled.connect(self._on_view_toggled)
        self.save_btn = QPushButton("Save…")
        self.save_btn.setToolTip("JSON file (protocol + geometry), or a row added to a CSV "
                                 "table (" + ", ".join(CSV_COLUMNS) + "; created if missing)")
        self.save_btn.clicked.connect(self._on_save_protocol)
        last_row.addWidget(self.view_check)
        last_row.addStretch(1)
        last_row.addWidget(self.save_btn)
        prot_l.addRow(last_row)
        root.addWidget(prot_box)

        # --- display
        disp_box = QGroupBox("Display")
        disp_l = QFormLayout(disp_box)
        style_row = QHBoxLayout()
        self.style_combo = QComboBox()
        self.style_combo.addItems(["points", "surface"])
        self.style_combo.setToolTip("surface reads the full STLs (slower for big meshes)")
        self.style_combo.currentTextChanged.connect(self._render)
        self.parts_btn = QToolButton()
        self.parts_btn.setPopupMode(QToolButton.InstantPopup)
        self.parts_btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.parts_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.parts_btn.setToolTip("other STL files of the directory, shown in the 3D "
                                  "view only (ticked = shown)")
        self.parts_menu = CheckMenu(self.parts_btn)
        self.parts_menu.aboutToShow.connect(self._fill_parts_menu)
        self.parts_btn.setMenu(self.parts_menu)
        style_row.addWidget(self.style_combo)
        style_row.addWidget(self.parts_btn, stretch=1)
        disp_l.addRow("Style:", style_row)

        # one line per component: color, opacity, and whether opacity is set at all
        self.color_btns, self.opacity_spins, self.opacity_checks = {}, {}, {}
        defaults = {label: (info["color"], viz.OPACITY.get(
            label, viz.DOUBLE_OPACITY if info["is_double"] else 0.05))
            for label, info in REGIONS.items()}
        defaults[OTHER] = (viz.PART_COLOR, 0.3)
        for label, (color, opacity) in defaults.items():
            row = QHBoxLayout()
            color_btn = ColorButton(color)
            color_btn.colorChanged.connect(self._render)
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 1.0)
            spin.setSingleStep(0.1)
            spin.setDecimals(3)
            spin.setValue(opacity)
            spin.setPrefix("opacity ")
            spin.setToolTip("0 hides the component")
            spin.valueChanged.connect(self._render)
            check = QCheckBox()
            check.setChecked(True)
            check.setToolTip("untick to draw without any opacity setting (opaque, "
                             "with full shading)")
            check.toggled.connect(spin.setEnabled)
            check.toggled.connect(self._render)
            row.addWidget(color_btn)
            row.addWidget(spin, stretch=1)
            row.addWidget(check)
            self.color_btns[label] = color_btn
            self.opacity_spins[label] = spin
            self.opacity_checks[label] = check
            disp_l.addRow(f"{label}:", row)

        show_row = QHBoxLayout()
        self.com_check = QCheckBox("left centers of mass")
        self.com_check.setChecked(True)
        self.legend_check = QCheckBox("legend")
        self.legend_check.setChecked(True)
        self.font_spin = QSpinBox()
        self.font_spin.setRange(6, 40)
        self.font_spin.setValue(viz.LEGEND_FONT_SIZE)
        self.font_spin.setPrefix("font ")
        self.font_spin.setSuffix(" px")
        self.legend_check.toggled.connect(self.font_spin.setEnabled)
        for widget in (self.com_check, self.legend_check):
            widget.toggled.connect(self._render)
        self.font_spin.valueChanged.connect(self._render)
        show_row.addWidget(self.com_check)
        show_row.addWidget(self.legend_check)
        show_row.addWidget(self.font_spin)
        show_row.addStretch(1)
        disp_l.addRow("Show:", show_row)

        cranio_row = QHBoxLayout()
        self.cranio_check = QCheckBox("show")
        self.cranio_check.setToolTip("around the entry point of the selected protocol")
        self.cranio_check.toggled.connect(self._render)
        self.radius_spin = QDoubleSpinBox()
        self.radius_spin.setRange(0.05, 10)
        self.radius_spin.setSingleStep(0.1)
        self.radius_spin.setValue(1.0)
        self.radius_spin.setPrefix("r ")
        self.radius_spin.setSuffix(" mm")
        self.thick_spin = QDoubleSpinBox()
        self.thick_spin.setRange(0.05, 10)
        self.thick_spin.setSingleStep(0.1)
        self.thick_spin.setValue(1.0)
        self.thick_spin.setPrefix("|DV| < ")
        self.thick_spin.setSuffix(" mm")
        for spin in (self.radius_spin, self.thick_spin):
            spin.valueChanged.connect(self._render)
        cranio_row.addWidget(self.cranio_check)
        cranio_row.addWidget(self.radius_spin)
        cranio_row.addWidget(self.thick_spin)
        disp_l.addRow("Craniotomy (skull):", cranio_row)

        view_row = QHBoxLayout()
        self.view_combo = QComboBox()
        self.view_combo.addItems(list(viz.VIEWS))
        self.view_combo.activated.connect(self._apply_view)
        reset_btn = QPushButton("Reset camera")
        reset_btn.clicked.connect(self._apply_view)
        self.save_png_btn = QPushButton("Screenshot…")
        self.save_png_btn.clicked.connect(self._on_save_screenshot)
        view_row.addWidget(self.view_combo)
        view_row.addWidget(reset_btn)
        view_row.addWidget(self.save_png_btn)
        disp_l.addRow("View:", view_row)
        root.addWidget(disp_box)

        self.log_box = LogBox()
        self.log_box.setMinimumHeight(120)
        root.addWidget(self.log_box, stretch=1)
        return panel

    # ------------------------------------------------------------ inputs
    def set_initial(self, src_dir=None, skull=None):
        if src_dir:
            self.dir_edit.setText(str(src_dir))
            self._refresh_files()
        if skull:
            self.file_combos["skull"].setCurrentText(Path(skull).name)

    def _refresh_files(self):
        src = self.dir_edit.text().strip()
        try:
            names = [p.name for p in find_stl_files(src)] if src else []
        except FileNotFoundError:
            names = []
            self.log_box.log(f"directory not found: {src}")
        for label, combo in self.file_combos.items():
            previous = combo.currentText()
            combo.clear()
            if label == "skull":
                combo.addItem(NONE_ITEM)
            combo.addItems(names)
            if previous in names:
                combo.setCurrentText(previous)
            elif (guess := _guess_file(label, names)) is not None:
                combo.setCurrentText(guess)
            else:
                combo.setCurrentIndex(0 if label == "skull" else -1)
        if src and names:
            self.log_box.log(f"{len(names)} STL file(s) in {src}")
        if any(p.parent != Path(src) for p in self.parts):
            self._set_parts([p for p in self.parts if p.parent == Path(src)])
        self._refresh_buttons()

    def _refresh_buttons(self, *_):
        if not hasattr(self, "save_btn"):  # still building the UI
            return
        busy = self._worker is not None and self._worker.isRunning()
        files_ok = all(self.file_combos[lab].currentText()
                       for lab in REGIONS if lab != "skull")
        entry = self._current()
        self.compute_btn.setEnabled(files_ok and not busy)
        self.from_com_btn.setEnabled(files_ok and not busy)
        self.remove_btn.setEnabled(len(self.protocols) > 1)
        self.save_btn.setEnabled(entry is not None and entry.prot is not None)
        self.save_png_btn.setEnabled(bool(self.regions) or self._shown_protocols() != [])

    def _cache_dir(self):
        return self.cache_edit.text().strip() or None if self.cache_check.isChecked() else None

    def _com_request(self):
        """(src_dir, files, n_points, method) of the current inputs."""
        files = {}
        for label, combo in self.file_combos.items():
            name = combo.currentText()
            if name and name != NONE_ITEM:
                files[REGION_KWARGS[label]] = name
        return (self.dir_edit.text().strip(), files, self.n_points_spin.value(),
                self.method_combo.currentText())

    def _geometry_name(self):
        return Path(self.dir_edit.text().strip()).name or "geometry"

    # --------------------------------------------------- centers of mass
    def _start_com(self, then_protocol: ProtocolEntry | None = None):
        src, files, n_points, method = self._com_request()
        self._pending_protocol = then_protocol
        self.log_box.log(f"Loading regions of {src}")
        self.progress.setVisible(True)
        self._worker = RegionsWorker(src, files, n_points, method, self._cache_dir(),
                                     self.force_check.isChecked())
        self._worker.log.connect(self.log_box.log)
        self._worker.finished_ok.connect(self._on_com_computed)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._refresh_buttons)
        self._worker.start()
        self._refresh_buttons()

    def _on_failed(self, message):
        self.progress.setVisible(False)
        self._pending_protocol = None
        self.log_box.log(f"error: {message}")
        QMessageBox.critical(self, "Failed", message)

    def _on_com_computed(self, regions):
        self.progress.setVisible(False)
        if not self.regions:
            self._camera_set = False  # frame the brain, not only the trajectories
        self.regions = regions
        self._regions_key = self._com_request()
        self._show_com_table()
        self.log_box.log("Centers of mass computed.")
        entry, self._pending_protocol = self._pending_protocol, None
        if entry is not None and entry in self.protocols:
            self._set_protocol_from_com(entry)
        self._show_protocol()  # offset to the new dLGN
        self._render()

    def _show_com_table(self):
        self.com_table.setRowCount(len(self.regions))
        for r, (label, reg) in enumerate(self.regions.items()):
            self.com_table.setItem(r, 0, QTableWidgetItem(f"{label} ({reg.path.name})"))
            self.com_table.setItem(r, 1, QTableWidgetItem(_fmt(reg.com)))
            self.com_table.setItem(r, 2, QTableWidgetItem(
                _fmt(reg.lcom) if reg.is_double else "—"))

    # ---------------------------------------------------------- protocols
    def _current(self) -> ProtocolEntry | None:
        i = self.protocol_combo.currentIndex() if hasattr(self, "protocol_combo") else -1
        return self.protocols[i] if 0 <= i < len(self.protocols) else None

    def _shown_protocols(self):
        return [e for e in self.protocols if e.prot is not None and e.visible]

    def _on_add_protocol(self):
        used = {e.color for e in self.protocols}
        colors = [c for c in viz.PROTOCOL_COLORS if c not in used] or viz.PROTOCOL_COLORS
        n = 1
        while any(e.name == f"Protocol {n}" for e in self.protocols):
            n += 1
        entry = ProtocolEntry(name=f"Protocol {n}", color=colors[0])
        self.protocols.append(entry)
        self.protocol_combo.addItem(_color_icon(entry.color), entry.title())
        self.protocol_combo.setCurrentIndex(len(self.protocols) - 1)
        self._refresh_buttons()

    def _on_remove_protocol(self):
        entry = self._current()
        if entry is None or len(self.protocols) < 2:
            return
        i = self.protocols.index(entry)
        self.protocols.pop(i)
        self.protocol_combo.removeItem(i)  # selects a neighbour
        self._render()
        self._refresh_buttons()

    def _update_combo_item(self, entry):
        i = self.protocols.index(entry)
        self.protocol_combo.setItemText(i, entry.title())

    def _on_protocol_from_com(self):
        entry = self._current()
        if entry is None:
            return
        if (self.regions and self._regions_key == self._com_request()
                and not self.force_check.isChecked()):
            self._set_protocol_from_com(entry)
            self._show_protocol()
            self._render()
        else:  # (re)compute the centers of mass first
            self._start_com(then_protocol=entry)

    def _set_protocol_from_com(self, entry):
        entry.prot = get_insertion_protocol(self.regions)
        entry.origin = f"CoM of {self._geometry_name()}"
        entry.modified = False
        entry.regions = dict(self.regions)
        entry.settings = {"n_points": self.n_points_spin.value(),
                          "com_method": self.method_combo.currentText()}
        entry.visible = True
        self._update_combo_item(entry)
        self.log_box.log(f"{entry.name}: computed from the centers of mass of "
                         f"{self._geometry_name()}")

    def _on_load_protocol(self):
        entry = self._current()
        if entry is None:
            return
        start = self.dir_edit.text().strip()
        path, _ = QFileDialog.getOpenFileName(self, "Load protocol", start,
                                              f"{JSON_FILTER};;All files (*)")
        if not path:
            return
        try:
            prot = load_protocol(path)
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            return
        entry.prot, entry.origin, entry.modified = prot, Path(path).name, False
        entry.regions, entry.settings, entry.visible = None, {"file": str(path)}, True
        self._update_combo_item(entry)
        self.log_box.log(f"{entry.name}: loaded from {path}")
        self._show_protocol()
        self._render()

    def _on_value_changed(self, key, value):
        """Manual edit of a protocol value: update the protocol and the 3D view."""
        if self._updating:
            return
        entry = self._current()
        if entry is None:
            return
        if entry.prot is None:  # typed from scratch
            entry.prot = make_protocol(0.0, 0.0, 0.0, ML_rot_deg=0.0)
            entry.origin = "manual"
            entry.visible = True
        else:
            entry.modified = True
        if key == "ML_rot_deg":
            p = entry.prot
            entry.prot = make_protocol(p["ML_tr"], p["AP_tr"], p["Insert_tr"],
                                       ML_rot_deg=value)
        else:
            entry.prot[key] = float(value)
        self._update_combo_item(entry)
        self._show_protocol()
        self._render()

    def _on_view_toggled(self, checked):
        entry = self._current()
        if entry is not None and not self._updating:
            entry.visible = checked
            self._render()

    def _offset(self, prot):
        if prot is None or "dLGN" not in self.regions:
            return None
        return np.round(target_offset(prot, self.regions), 3) + 0.0  # no "-0.000"

    def _offset_lines(self, prot) -> list[str]:
        d = self._offset(prot)
        if d is None:
            return []
        return [f"target - C_dLGN: {np.linalg.norm(d):.3f} mm",
                "    (AP, ML, DV): (" + ", ".join(f"{x:+.3f}" for x in d) + ")"]

    def _show_protocol(self, *_):
        """Fill the protocol widgets with the selected protocol."""
        entry = self._current()
        if entry is None:
            return
        p = entry.prot
        self._updating = True
        for key, spin in self.protocol_spins.items():
            spin.setValue(p[key] if p is not None else 0.0)
        self.view_check.setChecked(entry.visible)
        self.view_check.setEnabled(p is not None)
        self._updating = False

        if p is None:
            self.origin_label.setText("none yet: compute from CoM, load a file, "
                                      "or type the values")
            self.rad_label.setText("")
            for lab in self.point_labels.values():
                lab.setText("—")
        else:
            self.origin_label.setText(entry.origin + (" (edited)" if entry.modified else ""))
            self.rad_label.setText(f"{p['ML_rot']:.4f} rad")
            self.point_labels["entry"].setText(_fmt(entry_point(p)))
            self.point_labels["target"].setText(_fmt(target_point(p)))
            d = self._offset(p)
            self.point_labels["offset"].setText(
                "—" if d is None else
                f"{np.linalg.norm(d):.3f} mm  (" + ", ".join(f"{x:+.3f}" for x in d) + ")")
        self._refresh_buttons()

    def _on_protocol_selected(self, *_):
        self._show_protocol()
        self._render()  # craniotomy of the selected protocol

    # -------------------------------------------------------------- save
    def _on_save_protocol(self):
        entry = self._current()
        if entry is None or entry.prot is None:
            return
        stem = re.sub(r"[^\w.-]+", "_", f"{self._geometry_name()}_{entry.name}").strip("_")
        start = str(Path(self.dir_edit.text().strip() or ".") / f"{stem}.json")
        # Qt dialog: the native macOS one would ask to "replace" a CSV we append to
        path, chosen = QFileDialog.getSaveFileName(
            self, f"Save {entry.name}", start, f"{JSON_FILTER};;{CSV_FILTER}",
            options=QFileDialog.Option.DontConfirmOverwrite
            | QFileDialog.Option.DontUseNativeDialog)
        if not path:
            return
        path = Path(path)
        if path.suffix.lower() not in (".json", ".csv"):
            path = path.with_suffix(".csv" if chosen == CSV_FILTER else ".json")

        try:
            if path.suffix.lower() == ".csv":
                n = count_csv_rows(path)
                append_protocol_csv(path, entry.prot)
                self.log_box.log(f"{entry.name}: created {path}" if n is None else
                                 f"{entry.name}: added row {n + 1} to {path}")
            else:
                if path.is_file() and QMessageBox.question(
                        self, "Overwrite?", f"{path.name} already exists. Overwrite it?") \
                        != QMessageBox.StandardButton.Yes:
                    return
                save_protocol(path, entry.prot, entry.regions, name=entry.name,
                              origin=entry.origin, modified=entry.modified,
                              settings=entry.settings)
                self.log_box.log(f"{entry.name}: saved to {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))

    def _on_save_screenshot(self):
        src = Path(self.dir_edit.text().strip())
        path, _ = QFileDialog.getSaveFileName(
            self, "Save screenshot", str(src / "insertion_protocol.png"),
            "Images (*.png *.jpg *.tif)")
        if not path:
            return
        self.viewer.plotter.screenshot(path)
        self.log_box.log(f"Screenshot saved to {path}")

    # ----------------------------------------------------------- display
    def _opacity(self, label):
        """Opacity of a component, None if its opacity is not set."""
        if not self.opacity_checks[label].isChecked():
            return None
        return self.opacity_spins[label].value()

    def _part_names(self):
        """STL files of the directory not used as a region."""
        src = self.dir_edit.text().strip()
        try:
            paths = find_stl_files(src) if src else []
        except FileNotFoundError:
            paths = []
        used = {c.currentText() for c in self.file_combos.values()}
        return [p for p in paths if p.name not in used]

    def _fill_parts_menu(self):
        self.parts_menu.clear()
        paths = self._part_names()
        if not paths:
            self.parts_menu.addAction("no other STL file in the directory").setEnabled(False)
            return
        hide_all = self.parts_menu.addAction("Hide all")
        hide_all.setEnabled(bool(self.parts))
        hide_all.triggered.connect(lambda: self._set_parts([]))
        self.parts_menu.addSeparator()
        for path in paths:
            action = self.parts_menu.addAction(path.name)
            action.setCheckable(True)
            action.setChecked(path in self.parts)
            action.toggled.connect(lambda on, p=path: self._toggle_part(p, on))

    def _toggle_part(self, path, on):
        parts = [p for p in self.parts if p != path]
        self._set_parts(parts + [path] if on else parts)

    def _set_parts(self, parts):
        self.parts = list(parts)
        self.parts_btn.setText(f"Add / remove parts ({len(self.parts)} shown)")
        self.parts_btn.setToolTip(
            "other STL files of the directory, shown in the 3D view only\n"
            + ("shown: " + ", ".join(p.name for p in self.parts) if self.parts
               else "none shown"))
        self._render()

    def _missing_meshes(self, style):
        """Path key -> Path of the meshes to read before rendering."""
        paths = [p for p in self.parts if self._opacity(OTHER) != 0]
        if style == "surface":
            paths += [r.path for k, r in self.regions.items() if self._opacity(k) != 0]
        return {str(p): p for p in paths if str(p) not in self._meshes}

    def _render(self, *_):
        if not hasattr(self, "log_box"):  # still building the UI
            return
        shown = self._shown_protocols()
        if not self.regions and not shown and not self.parts:
            self.viewer.plotter.clear()
            self.viewer.plotter.render()
            return
        style = self.style_combo.currentText()
        missing = self._missing_meshes(style)
        if missing:
            self._load_meshes(missing)
            return  # re-rendered once the meshes are read

        plotter = self.viewer.plotter
        camera = plotter.camera_position if self._camera_set else None
        plotter.clear()
        legend = viz.add_regions(
            plotter, self.regions, {k: self._opacity(k) for k in self.regions}, style,
            {k: self._meshes.get(str(r.path)) for k, r in self.regions.items()},
            show_com=self.com_check.isChecked(),
            colors={k: self.color_btns[k].color() for k in self.regions})
        viz.add_parts(plotter, {p.name: self._meshes[str(p)] for p in self.parts
                                if str(p) in self._meshes},
                      self.color_btns[OTHER].color(), self._opacity(OTHER), style)
        for entry in shown:
            legend += viz.add_trajectory(plotter, entry.prot, color=entry.color,
                                         title=entry.title(),
                                         extra_lines=self._offset_lines(entry.prot))

        current = self._current()
        if (self.cranio_check.isChecked() and "skull" in self.regions
                and current is not None and current.prot is not None):
            viz.add_craniotomy(plotter, self.regions["skull"], current.prot,
                               self.thick_spin.value(), self.radius_spin.value(),
                               color=current.color)

        viz.decorate(plotter, self.view_combo.currentText(),
                     legend=legend if self.legend_check.isChecked() else None,
                     font_size=self.font_spin.value() * self.devicePixelRatioF())
        if camera is not None:
            plotter.camera_position = camera
        else:
            plotter.reset_camera()
            self._camera_set = True
        plotter.render()

    def _apply_view(self, *_):
        viz.set_view(self.viewer.plotter, self.view_combo.currentText())
        self.viewer.plotter.reset_camera()
        self.viewer.plotter.render()

    def _load_meshes(self, paths):
        if self._mesh_worker is not None and self._mesh_worker.isRunning():
            return  # _on_meshes re-renders, which asks for what is still missing
        self.progress.setVisible(True)
        self._mesh_worker = MeshWorker(paths)
        self._mesh_worker.log.connect(self.log_box.log)
        self._mesh_worker.failed.connect(self._on_failed)
        self._mesh_worker.finished_ok.connect(self._on_meshes)
        self._mesh_worker.start()

    def _on_meshes(self, meshes):
        self.progress.setVisible(False)
        self._meshes.update(meshes)
        self._render()

    def shutdown(self):
        self.viewer.close()
