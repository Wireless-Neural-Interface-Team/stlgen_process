"""Dialog of the optimized insertion protocol (see process_stl/optimize.py).

Experimental and self-contained: remove this file, optimize_history.py,
process_stl/optimize.py and the lines marked "optimize" in tabs/protocol_tab.py
to drop the feature.

Each protocol parameter is either fixed to a value or optimized within bounds.
The default bounds are the bounding box of the skull for the translations,
[-90, 90] deg for the rotation and [0, bbox diagonal] for the depth.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..optimize import (
    METHOD,
    METHODS,
    OBJECTIVE,
    OBJECTIVES,
    RANDOM_BATCH,
    PARAMS,
    TARGETS,
    OptimizeSettings,
    default_bounds,
    optimize_protocol,
    protocol_lengths,
    save_history,
)
from ..sections import format_lengths, load_meshes
from .widgets import LogBox

#: parameter -> (label, unit, range, decimals, step)
PARAM_SPINS = {
    "AP_tr": ("AP translation", " mm", (-50.0, 50.0), 3, 0.1),
    "ML_tr": ("ML translation", " mm", (-50.0, 50.0), 3, 0.1),
    "ML_rot_deg": ("ML rotation", "°", (-90.0, 90.0), 2, 1.0),
    "Insert_tr": ("Insertion depth", " mm", (0.0, 50.0), 3, 0.1),
}

def _fmt_lengths(lengths):
    return format_lengths(lengths)


class OptimizeWorker(QThread):
    log = Signal(str)
    progress = Signal(int, dict)  # generation, best evaluation
    finished_ok = Signal(dict)
    failed = Signal(str)

    def __init__(self, regions, settings, bregma_lambda, parent=None):
        super().__init__(parent)
        self.regions = regions
        self.settings = settings
        self.bregma_lambda = bregma_lambda
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            meshes = load_meshes(self.regions, log=self.log.emit)
            result = optimize_protocol(
                meshes, self.settings, self.bregma_lambda,
                progress=lambda gen, best: self.progress.emit(gen, best),
                should_stop=lambda: self._stop, log=self.log.emit)
            result["stopped"] = self._stop
            self.finished_ok.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class OptimizeDialog(QDialog):
    """Set the fixed / optimized parameters, run the search, and return the
    best protocol in self.result (None until a search has finished)."""

    def __init__(self, regions, start, bregma_lambda=None, geometry="", parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Optimize the insertion protocol — {geometry}".rstrip(" —"))
        self.regions = regions
        self.start = start              # protocol giving the fixed values and the seed
        self.bregma_lambda = bregma_lambda
        self.result = None
        self._worker = None
        self._bounds = default_bounds(regions)

        layout = QVBoxLayout(self)
        ref = "skull" if "skull" in regions else "whole brain (no skull loaded)"
        params_box = QGroupBox(f"Parameters (default bounds: bounding box of the {ref})")
        grid = QGridLayout(params_box)
        for col, text in enumerate(["", "Optimize", "Fixed value", "Min", "Max"]):
            grid.addWidget(QLabel(f"<b>{text}</b>"), 0, col)
        self.opt_checks, self.value_spins, self.min_spins, self.max_spins = {}, {}, {}, {}
        for row, (key, (text, suffix, (lo, hi), dec, step)) in enumerate(PARAM_SPINS.items(), 1):
            check = QCheckBox()
            check.setChecked(True)
            spins = []
            for _ in range(3):
                spin = QDoubleSpinBox()
                spin.setRange(lo, hi)
                spin.setDecimals(dec)
                spin.setSingleStep(step)
                spin.setSuffix(suffix)
                spins.append(spin)
            value, vmin, vmax = spins
            value.setValue(start[key])
            check.toggled.connect(lambda on, k=key: self._on_toggled(k, on))
            grid.addWidget(QLabel(text), row, 0)
            grid.addWidget(check, row, 1, alignment=Qt.AlignCenter)
            grid.addWidget(value, row, 2)
            grid.addWidget(vmin, row, 3)
            grid.addWidget(vmax, row, 4)
            self.opt_checks[key] = check
            self.value_spins[key] = value
            self.min_spins[key] = vmin
            self.max_spins[key] = vmax
        self.opt_checks["Insert_tr"].setToolTip(
            "the depth is not searched: for each trajectory the best depth within the "
            "bounds is computed exactly (exit of the last region crossed)")
        reset_btn = QPushButton("Reset bounds")
        reset_btn.clicked.connect(self._reset_bounds)
        grid.addWidget(reset_btn, len(PARAM_SPINS) + 1, 4)
        layout.addWidget(params_box)
        self._reset_bounds()
        for key in PARAMS:
            self._on_toggled(key, self.opt_checks[key].isChecked())

        opts_box = QGroupBox("Search")
        form = QFormLayout(opts_box)
        self.objective_combo = QComboBox()
        for i, (name, desc) in enumerate(OBJECTIVES.items()):
            self.objective_combo.addItem(f"{name}: {desc}", name)
        self.objective_combo.setCurrentIndex(list(OBJECTIVES).index(OBJECTIVE))
        form.addRow("Objective:", self.objective_combo)
        method_row = QHBoxLayout()
        self.method_combo = QComboBox()
        for name, desc in METHODS.items():
            self.method_combo.addItem(desc, name)
        self.method_combo.setCurrentIndex(list(METHODS).index(METHOD))
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 1_000_000)
        method_row.addWidget(self.method_combo, stretch=1)
        method_row.addWidget(QLabel("random seed"))
        method_row.addWidget(self.seed_spin)
        form.addRow("Method:", method_row)
        self.maxiter_spin = QSpinBox()
        self.maxiter_spin.setRange(1, 10_000)
        self.maxiter_spin.setValue(200)
        self.popsize_spin = QSpinBox()
        self.popsize_spin.setRange(5, 500)
        self.popsize_spin.setValue(30)
        self.popsize_spin.setToolTip("population = this x number of optimized parameters")
        self.evo_widget = QWidget()
        evo_row = QHBoxLayout(self.evo_widget)
        evo_row.setContentsMargins(0, 0, 0, 0)
        for text, w in (("generations", self.maxiter_spin), ("population", self.popsize_spin)):
            evo_row.addWidget(QLabel(text))
            evo_row.addWidget(w)
        evo_row.addStretch(1)
        form.addRow("Differential evolution:", self.evo_widget)
        self.samples_spin = QSpinBox()
        self.samples_spin.setRange(1, 10_000_000)
        self.samples_spin.setSingleStep(10_000)
        self.samples_spin.setValue(20_000)
        self.samples_spin.setSuffix(" tries")
        self.samples_spin.setToolTip("trajectories drawn uniformly within the bounds")
        form.addRow("Random search:", self.samples_spin)
        self.seed_check = QCheckBox("start from the current protocol")
        self.seed_check.setChecked(True)
        self.refine_check = QCheckBox("refine locally (Nelder-Mead)")
        self.refine_check.setChecked(True)
        check_row = QHBoxLayout()
        check_row.addWidget(self.seed_check)
        check_row.addWidget(self.refine_check)
        self.track_check = QCheckBox("track cost")
        self.track_check.setToolTip("record every trajectory tried, with its lengths "
                                    "inside VS and dLGN and its score, to save as CSV "
                                    "or plot after the search")
        check_row.addWidget(self.track_check)
        form.addRow(check_row)
        layout.addWidget(opts_box)
        self._on_method_changed()

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.status_label)
        self.log_box = LogBox()
        self.log_box.setMinimumHeight(110)
        layout.addWidget(self.log_box, stretch=1)

        buttons = QHBoxLayout()
        self.run_btn = QPushButton("Run")
        self.run_btn.clicked.connect(self._on_run)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop)
        self.save_hist_btn = QPushButton("Save tries…")
        self.save_hist_btn.setToolTip("CSV of the trajectories tried (needs 'track cost')")
        self.save_hist_btn.clicked.connect(self._on_save_history)
        self.plot_hist_btn = QPushButton("Plot tries…")
        self.plot_hist_btn.setToolTip("3D view and bar plot of the trajectories tried "
                                      "(needs 'track cost')")
        self.plot_hist_btn.clicked.connect(self._on_plot_history)
        for b in (self.save_hist_btn, self.plot_hist_btn):
            b.setEnabled(False)
        self.apply_btn = QPushButton("Apply to protocol")
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(self.accept)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        for b in (self.run_btn, self.stop_btn):
            buttons.addWidget(b)
        buttons.addStretch(1)
        buttons.addWidget(self.save_hist_btn)
        buttons.addWidget(self.plot_hist_btn)
        buttons.addWidget(self.apply_btn)
        buttons.addWidget(close_btn)
        layout.addLayout(buttons)
        self.resize(720, 640)

    # ------------------------------------------------------------- inputs
    def _on_toggled(self, key, optimized):
        self.value_spins[key].setEnabled(not optimized)
        self.min_spins[key].setEnabled(optimized)
        self.max_spins[key].setEnabled(optimized)

    def _on_method_changed(self, *_):
        random = self.method_combo.currentData() == "random"
        self.evo_widget.setEnabled(not random)
        self.samples_spin.setEnabled(random)
        self.seed_check.setText("include the current protocol" if random
                                else "start from the current protocol")

    def _reset_bounds(self):
        for key, (lo, hi) in self._bounds.items():
            self.min_spins[key].setValue(lo)
            self.max_spins[key].setValue(hi)

    def _settings(self) -> OptimizeSettings:
        fixed, bounds = {}, {}
        for key in PARAMS:
            if self.opt_checks[key].isChecked():
                lo, hi = self.min_spins[key].value(), self.max_spins[key].value()
                if lo >= hi:
                    raise ValueError(f"{PARAM_SPINS[key][0]}: min must be below max")
                bounds[key] = (lo, hi)
            else:
                fixed[key] = self.value_spins[key].value()
        return OptimizeSettings(
            fixed=fixed, bounds=bounds, objective=self.objective_combo.currentData(),
            start=self.start if self.seed_check.isChecked() else None,
            maxiter=self.maxiter_spin.value(), popsize=self.popsize_spin.value(),
            seed=self.seed_spin.value(), refine=self.refine_check.isChecked(),
            track=self.track_check.isChecked(), method=self.method_combo.currentData(),
            n_samples=self.samples_spin.value())

    # ---------------------------------------------------------------- run
    def _on_run(self):
        try:
            settings = self._settings()
        except ValueError as exc:
            QMessageBox.critical(self, "Invalid settings", str(exc))
            return
        self.result = None
        self.apply_btn.setEnabled(False)
        self.save_hist_btn.setEnabled(False)
        self.plot_hist_btn.setEnabled(False)
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("running…")
        self.log_box.log("Optimizing, fixed: " + (", ".join(
            f"{k}={v:g}" for k, v in settings.fixed.items()) or "none"))
        self._worker = OptimizeWorker(self.regions, settings, self.bregma_lambda)
        self._worker.log.connect(self.log_box.log)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _on_stop(self):
        if self._worker is not None:
            self._worker.stop()
            self.status_label.setText("stopping after this generation…")

    def _on_progress(self, generation, best):
        step = (f"{min(generation * RANDOM_BATCH, self.samples_spin.value())} tries"
                if self.method_combo.currentData() == "random" else f"generation {generation}")
        self.status_label.setText(f"{step}: best score {best['score']:.4f} — "
                                  + _fmt_lengths(best["lengths"]))

    def _on_done(self, result):
        self.result = result
        p = result["protocol"]
        start = protocol_lengths(self.start, load_meshes(self.regions))
        text = (f"{'Stopped' if result['stopped'] else 'Done'}: {result['n_eval']} "
                f"evaluations in {result['time_s']:.1f} s, score {result['score']:.4f}\n"
                f"inside: {_fmt_lengths(result['lengths'])}\n"
                f"(current protocol: {_fmt_lengths(start)})\n"
                f"AP {p['AP_tr']:.3f} mm, ML {p['ML_tr']:.3f} mm, "
                f"rotation {p['ML_rot_deg']:.2f}°, depth {p['Insert_tr']:.3f} mm")
        self.status_label.setText(text)
        self.log_box.log(text)
        if min(result["lengths"].values()) == 0:
            self.log_box.log("warning: the best protocol misses a region; "
                             "widen the bounds or free more parameters")
        self.apply_btn.setEnabled(True)
        tracked = bool(result.get("history"))
        self.save_hist_btn.setEnabled(tracked)
        self.plot_hist_btn.setEnabled(tracked)
        if tracked:
            self.log_box.log(f"{len(result['history'])} tries recorded")

    def _on_save_history(self):
        if not self.result or not self.result.get("history"):
            return
        start = str(self.regions["dLGN"].path.parent / "optimization_tries.csv")
        path, _ = QFileDialog.getSaveFileName(self, "Save tries", start, "CSV (*.csv)")
        if not path:
            return
        try:
            save_history(path, self.result["history"])
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        self.log_box.log(f"tries saved to {path}")

    def _on_plot_history(self):
        if not self.result or not self.result.get("history"):
            return
        from .optimize_history import HistoryWindow

        window = HistoryWindow(self.regions, self.result, parent=self)
        window.show()

    def _on_failed(self, message):
        self.status_label.setText(f"error: {message}")
        self.log_box.log(f"error: {message}")

    def _on_finished(self):
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def done(self, code):
        if self._worker is not None and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait()
        super().done(code)

