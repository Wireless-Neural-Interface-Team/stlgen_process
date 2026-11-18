"""Window showing the trajectories tried by an optimization ("track cost").

Part of the experimental optimized protocol (see optimize_dialog.py).

Left: 3D view of the regions with the tried trajectories, colored by try order
or by score (the best one in red). Right: for each try, a stacked bar with the
length inside the dLGN (bottom) and inside VS (top). With many tries, both
views show an evenly spaced subset, always including the best try.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv
from matplotlib.colors import to_rgb
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .. import viz
from ..optimize import history_frame
from ..sections import line
from .widgets import PlotCanvas, Viewer3D

COLOR_BY = {"try order": "try", "score": "score"}
CMAP = "viridis"
BEST_COLOR = "red"
MAX_SHOWN = 200


def _lighter(color, amount=0.55):
    """Color blended with white, to tell apart two close region colors."""
    return tuple(c + (1 - c) * amount for c in to_rgb(color))


class HistoryWindow(QWidget):
    def __init__(self, regions, result, parent=None):
        super().__init__(parent, Qt.Window)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle("Optimization tries")
        self.regions = regions
        self.df = history_frame(result["history"])
        self.best = int(self.df["score"].idxmax())

        controls = QHBoxLayout()
        self.n_spin = QSpinBox()
        self.n_spin.setRange(1, len(self.df))
        self.n_spin.setValue(min(MAX_SHOWN, len(self.df)))
        self.n_spin.setToolTip(f"evenly spaced among the {len(self.df)} tries, "
                               "the best one always included")
        self.color_combo = QComboBox()
        self.color_combo.addItems(list(COLOR_BY))
        self.view_combo = QComboBox()
        self.view_combo.addItems(list(viz.VIEWS))
        update_btn = QPushButton("Update")
        self.n_spin.editingFinished.connect(self._draw)
        self.color_combo.currentTextChanged.connect(self._draw_3d)
        self.view_combo.activated.connect(self._draw_3d)
        update_btn.clicked.connect(self._draw)
        for text, w in (("tries shown", self.n_spin), ("color by", self.color_combo),
                        ("view", self.view_combo)):
            controls.addWidget(QLabel(text))
            controls.addWidget(w)
        controls.addWidget(update_btn)
        controls.addStretch(1)
        b = self.df.loc[self.best]
        controls.addWidget(QLabel(
            f"{len(self.df)} tries — best: try {int(b['try'])}, "
            f"dLGN {b['length_dLGN']:.3f} mm, VS {b['length_VS']:.3f} mm"))

        self.viewer = Viewer3D()
        self.canvas = PlotCanvas()
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.viewer)
        splitter.addWidget(self.canvas)
        splitter.setSizes([700, 700])

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(splitter, stretch=1)
        self.resize(1500, 800)
        self._draw()

    def _shown(self):
        """Rows of the tries shown: evenly spaced, plus the best."""
        n = len(self.df)
        idx = np.unique(np.linspace(0, n - 1, min(self.n_spin.value(), n)).round().astype(int))
        return self.df.iloc[np.union1d(idx, [self.best])]

    def _draw(self, *_):
        self._draw_3d()
        self._draw_bars()

    def _draw_3d(self, *_):
        rows = self._shown()
        plotter = self.viewer.plotter
        plotter.clear()
        legend = viz.add_regions(plotter, self.regions, style="points")

        points, cells = [], []
        for i, r in enumerate(rows.itertuples()):
            origin, direction = line(r.AP_tr, r.ML_tr, r.ML_rot_deg)
            points += [origin, origin + r.Insert_tr * direction]
            cells += [2, 2 * i, 2 * i + 1]
        poly = pv.PolyData(np.array(points), lines=np.array(cells))
        key = COLOR_BY[self.color_combo.currentText()]
        poly.cell_data[key] = rows[key].to_numpy(dtype=float)
        plotter.add_mesh(poly, scalars=key, cmap=CMAP, line_width=2,
                         scalar_bar_args={"title": self.color_combo.currentText()})

        b = self.df.loc[self.best]
        origin, direction = line(b["AP_tr"], b["ML_tr"], b["ML_rot_deg"])
        best = np.array([origin, origin + b["Insert_tr"] * direction])
        plotter.add_lines(best, color=BEST_COLOR, width=7)
        plotter.add_points(best, color=BEST_COLOR, point_size=14,
                           render_points_as_spheres=True)
        legend.append([f"best: try {int(b['try'])}", BEST_COLOR, "rectangle"])
        viz.decorate(plotter, self.view_combo.currentText(), legend=legend)
        plotter.reset_camera()
        plotter.render()

    def _draw_bars(self):
        rows = self._shown()
        fig = self.canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        x = np.arange(len(rows))
        d, v = rows["length_dLGN"].to_numpy(), rows["length_VS"].to_numpy()
        ax.bar(x, d, width=0.85, color=self.regions["dLGN"].color, label="dLGN")
        ax.bar(x, v, width=0.85, bottom=d,
               color=_lighter(self.regions["visual_cortex"].color), label="VS")
        best = int(np.flatnonzero(rows.index.to_numpy() == self.best)[0])
        ax.plot(best, d[best] + v[best], marker="*", color=BEST_COLOR, markersize=14,
                linestyle="none", label="best")
        # phase changes (evolution -> refine)
        phases = rows["phase"].to_numpy()
        for i in np.flatnonzero(phases[1:] != phases[:-1]) + 1:
            ax.axvline(i - 0.5, color="grey", linestyle="--", linewidth=1)
            ax.text(i - 0.5, 0.98, f" {phases[i]}", va="top", color="grey",
                    transform=ax.get_xaxis_transform())
        ticks = np.unique(np.linspace(0, len(rows) - 1, min(len(rows), 12)).round().astype(int))
        ax.set_xticks(ticks, rows["try"].to_numpy()[ticks])
        ax.set_xlim(-0.6, len(rows) - 0.4)
        ax.set_xlabel("try")
        ax.set_ylabel("length inside the region (mm)")
        ax.legend(loc="upper left", reverse=True)  # VS on top, as in the bars
        fig.tight_layout()
        self.canvas.draw()

    def closeEvent(self, event):
        self.viewer.close()
        super().closeEvent(event)
