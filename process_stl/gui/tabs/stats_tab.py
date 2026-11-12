"""Tab 2: compare the bbox size / center of mass of the STLs of several folders."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ...plotting import PLOT_KINDS, plot_boxplots
from ...stats import AXES, STAT_TEST, TEST_NAMES, compute_pvalues, summary
from ..widgets import DataFrameTable, LogBox, PlotCanvas, path_row
from ..workers import StatsWorker

#: axis convention presets (names of the STL x, y, z axes)
AXES_PRESETS = {
    "ML, DV, AP  (skull STLs)": AXES,
    "AP, ML, DV  (region STLs)": ("AP", "ML", "DV"),
}


class StatsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.df = None
        self.pvals = None
        self._axes = AXES
        self._order: list[str] = []
        self._worker = None

        self.results = QTabWidget()
        self.canvas = PlotCanvas()
        self.data_table = DataFrameTable()
        self.summary_table = DataFrameTable()
        self.pvals_table = DataFrameTable()
        self.results.addTab(self.canvas, "Box plots")
        self.results.addTab(self.data_table, "Data")
        self.results.addTab(self.summary_table, "Summary")
        self.results.addTab(self.pvals_table, "p-values")

        controls = QScrollArea()
        controls.setWidget(self._build_controls())
        controls.setWidgetResizable(True)
        controls.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        controls.setMinimumWidth(430)
        controls.setMaximumWidth(560)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(controls)
        splitter.addWidget(self.results)
        splitter.setStretchFactor(1, 1)
        QVBoxLayout(self).addWidget(splitter)
        self._refresh_buttons()

    # ------------------------------------------------------------------ UI
    def _build_controls(self) -> QWidget:
        panel = QWidget()
        root = QVBoxLayout(panel)

        # --- folders
        folders_box = QGroupBox("1. Folders to compare")
        folders_l = QVBoxLayout(folders_box)
        self.folders_table = QTableWidget(0, 2)
        self.folders_table.setHorizontalHeaderLabels(["Label", "Folder"])
        self.folders_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.folders_table.setToolTip("double-click a label to rename it")
        self.folders_table.itemChanged.connect(self._refresh_buttons)
        folders_l.addWidget(self.folders_table)
        btn_row = QHBoxLayout()
        add_btn = QPushButton("Add folder…")
        add_btn.clicked.connect(self._on_add_folder)
        remove_btn = QPushButton("Remove selected")
        remove_btn.clicked.connect(self._on_remove_folder)
        btn_row.addWidget(add_btn)
        btn_row.addWidget(remove_btn)
        folders_l.addLayout(btn_row)
        self.recursive_check = QCheckBox("also search subfolders")
        folders_l.addWidget(self.recursive_check)
        root.addWidget(folders_box)

        # --- computation
        comp_box = QGroupBox("2. Measure")
        comp_l = QFormLayout(comp_box)
        self.axes_combo = QComboBox()
        self.axes_combo.setEditable(True)
        self.axes_combo.addItems(list(AXES_PRESETS))
        self.axes_combo.setToolTip("names of the STL x, y, z axes, e.g. 'ML, DV, AP'")
        comp_l.addRow("Axes (x, y, z):", self.axes_combo)
        row, self.data_edit = path_row(self, "save", "CSV (*.csv)",
                                       placeholder="optional, e.g. stl_stats.csv")
        self.data_edit.setToolTip("table reloaded instead of recomputed while it "
                                  "still matches the folders")
        comp_l.addRow("Data file:", row)
        self.force_check = QCheckBox("ignore the data file and recompute")
        comp_l.addRow(self.force_check)
        self.compute_btn = QPushButton("Compute statistics")
        self.compute_btn.clicked.connect(self._on_compute)
        comp_l.addRow(self.compute_btn)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        comp_l.addRow(self.progress)
        root.addWidget(comp_box)

        # --- statistics / plot
        plot_box = QGroupBox("3. Statistics and plot")
        plot_l = QFormLayout(plot_box)
        self.test_combo = QComboBox()
        for name, desc in TEST_NAMES.items():
            self.test_combo.addItem(desc, name)
        self.test_combo.setCurrentIndex(list(TEST_NAMES).index(STAT_TEST))
        self.test_combo.currentIndexChanged.connect(self._update_results)
        plot_l.addRow("Test:", self.test_combo)
        self.kind_combo = QComboBox()
        for name, desc in PLOT_KINDS.items():
            self.kind_combo.addItem(desc, name)
        self.kind_combo.currentIndexChanged.connect(self._update_plot)
        plot_l.addRow("Metrics:", self.kind_combo)
        root.addWidget(plot_box)

        # --- save
        save_box = QGroupBox("4. Save")
        save_l = QHBoxLayout(save_box)
        self.save_plot_btn = QPushButton("Plot…")
        self.save_plot_btn.clicked.connect(self._on_save_plot)
        self.save_data_btn = QPushButton("Data CSV…")
        self.save_data_btn.clicked.connect(lambda: self._save_csv(self.df, "stl_stats.csv"))
        self.save_pvals_btn = QPushButton("p-values CSV…")
        self.save_pvals_btn.clicked.connect(
            lambda: self._save_csv(self.pvals, "stl_pvalues.csv", index=False))
        for b in (self.save_plot_btn, self.save_data_btn, self.save_pvals_btn):
            save_l.addWidget(b)
        root.addWidget(save_box)

        self.log_box = LogBox()
        self.log_box.setMinimumHeight(120)
        root.addWidget(self.log_box, stretch=1)
        return panel

    # ------------------------------------------------------------ inputs
    def add_folder(self, folder, label=None):
        folder = str(folder)
        label = label or Path(folder.rstrip("/")).name
        r = self.folders_table.rowCount()
        self.folders_table.insertRow(r)
        self.folders_table.setItem(r, 0, QTableWidgetItem(label))
        path_item = QTableWidgetItem(folder)
        path_item.setFlags(path_item.flags() & ~Qt.ItemIsEditable)
        self.folders_table.setItem(r, 1, path_item)
        self._refresh_buttons()

    def set_initial(self, folders=None, data_file=None):
        for item in folders or []:
            label, sep, folder = item.partition("=")
            self.add_folder(folder, label) if sep else self.add_folder(item)
        if data_file:
            self.data_edit.setText(str(data_file))

    def _on_add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select a folder of STL files")
        if folder:
            self.add_folder(folder)

    def _on_remove_folder(self):
        for r in sorted({i.row() for i in self.folders_table.selectedIndexes()}, reverse=True):
            self.folders_table.removeRow(r)
        self._refresh_buttons()

    def _folders(self) -> dict:
        folders = {}
        for r in range(self.folders_table.rowCount()):
            label_item, path_item = self.folders_table.item(r, 0), self.folders_table.item(r, 1)
            if label_item and path_item and label_item.text().strip():
                folders[label_item.text().strip()] = path_item.text()
        return folders

    def _parse_axes(self):
        text = self.axes_combo.currentText()
        if text in AXES_PRESETS:
            return AXES_PRESETS[text]
        axes = tuple(a.strip() for a in text.replace(";", ",").split(",") if a.strip())
        if len(axes) != 3:
            raise ValueError(f"expected 3 comma-separated axis names, got {text!r}")
        return axes

    def _refresh_buttons(self, *_):
        busy = self._worker is not None and self._worker.isRunning()
        has = self.df is not None and not self.df.empty
        self.compute_btn.setEnabled(bool(self._folders()) and not busy)
        for b in (self.save_plot_btn, self.save_data_btn, self.save_pvals_btn):
            b.setEnabled(has)

    # ----------------------------------------------------------- compute
    def _on_compute(self):
        folders = self._folders()
        if len(folders) != self.folders_table.rowCount():
            QMessageBox.warning(self, "Labels", "Every folder needs a unique, non-empty label.")
            return
        try:
            axes = self._parse_axes()
        except ValueError as exc:
            QMessageBox.warning(self, "Axes", str(exc))
            return
        self.progress.setVisible(True)
        self._worker = StatsWorker(folders, self.recursive_check.isChecked(),
                                   self.data_edit.text().strip() or None,
                                   self.force_check.isChecked(), axes)
        self._worker.log.connect(self.log_box.log)
        self._worker.finished_ok.connect(lambda df: self._on_computed(df, folders, axes))
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._refresh_buttons)
        self._worker.start()
        self._refresh_buttons()

    def _on_failed(self, message):
        self.progress.setVisible(False)
        self.log_box.log(f"error: {message}")
        QMessageBox.critical(self, "Failed", message)

    def _on_computed(self, df, folders, axes):
        self.progress.setVisible(False)
        if df.empty:
            self.log_box.log("No data collected, check the folders.")
            return
        self.df, self._axes, self._order = df, axes, list(folders)
        self.data_table.set_dataframe(df)
        self.summary_table.set_dataframe(summary(df, axes))
        self._update_results()
        self._refresh_buttons()

    def _update_results(self, *_):
        if self.df is None:
            return
        test = self.test_combo.currentData()
        self.pvals = compute_pvalues(self.df, self._order, test, self._axes)
        self.pvals_table.set_dataframe(self.pvals)
        self._update_plot()

    def _update_plot(self, *_):
        if self.df is None or self.pvals is None:
            return
        plot_boxplots(self.df, self._order, self.pvals, self.test_combo.currentData(),
                      kind=self.kind_combo.currentData(), axes=self._axes,
                      fig=self.canvas.figure)
        self.canvas.draw()

    # -------------------------------------------------------------- save
    def _on_save_plot(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save plot", "stl_boxplots.png", "Images (*.png *.pdf *.svg)")
        if path:
            self.canvas.figure.savefig(path, dpi=150)
            self.log_box.log(f"Plot saved to {path}")

    def _save_csv(self, df, default, index=False):
        path, _ = QFileDialog.getSaveFileName(self, "Save table", default, "CSV (*.csv)")
        if path:
            df.to_csv(path, index=index)
            self.log_box.log(f"Table saved to {path}")
