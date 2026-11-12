"""Small reusable widgets: path rows, color button, check menu, log box,
DataFrame table, 3D viewer, plot canvas."""

from __future__ import annotations

import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QColorDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLineEdit,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

VIEWER_BACKGROUND = "white"


def path_row(parent, mode="dir", filter_="", placeholder="") -> tuple[QHBoxLayout, QLineEdit]:
    """A line edit + "Browse…" button.

    mode: "dir", "open", "save", or "any" (an existing or a new file, no
    overwrite confirmation, e.g. a file to import from or append to).
    """
    row = QHBoxLayout()
    edit = QLineEdit()
    edit.setPlaceholderText(placeholder)
    button = QPushButton("Browse…")

    def browse():
        start = edit.text().strip()
        if mode == "dir":
            path = QFileDialog.getExistingDirectory(parent, "Select folder", start)
        elif mode == "open":
            path, _ = QFileDialog.getOpenFileName(parent, "Select file", start, filter_)
        elif mode == "save":
            path, _ = QFileDialog.getSaveFileName(parent, "Save as", start, filter_)
        else:
            path, _ = QFileDialog.getSaveFileName(
                parent, "Select file", start, filter_,
                options=QFileDialog.Option.DontConfirmOverwrite)
        if path:
            edit.setText(path)
            edit.editingFinished.emit()

    button.clicked.connect(browse)
    row.addWidget(edit)
    row.addWidget(button)
    return row, edit


class ColorButton(QPushButton):
    """Button showing a color; a click opens a color picker."""

    colorChanged = Signal(str)  # "#rrggbb"

    def __init__(self, color="#808080", parent=None):
        super().__init__(parent)
        self.setFixedWidth(36)
        self.setToolTip("change the color")
        self.clicked.connect(self._pick)
        self.set_color(color)

    def color(self) -> str:
        return self._color

    def set_color(self, color):
        self._color = QColor(color).name()
        self.setStyleSheet(f"background-color: {self._color}; border: 1px solid #555;"
                           "border-radius: 3px; min-height: 18px;")

    def _pick(self):
        color = QColorDialog.getColor(QColor(self._color), self, "Color")
        if color.isValid() and color.name() != self._color:
            self.set_color(color.name())
            self.colorChanged.emit(self._color)


class CheckMenu(QMenu):
    """Menu of checkable actions that stays open while they are toggled."""

    def mouseReleaseEvent(self, event):
        action = self.activeAction()
        if action is not None and action.isEnabled() and action.isCheckable():
            action.trigger()
            event.accept()
        else:
            super().mouseReleaseEvent(event)


class LogBox(QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(2000)
        self.setPlaceholderText("log")

    def log(self, text: str):
        self.appendPlainText(text)


class DataFrameTable(QTableWidget):
    """Read-only table showing a DataFrame (index included if named / not a range)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEditTriggers(QTableWidget.NoEditTriggers)
        self.setSortingEnabled(False)

    def set_dataframe(self, df: pd.DataFrame, float_format="{:.4g}"):
        show_index = not isinstance(df.index, pd.RangeIndex)
        if show_index:
            df = df.reset_index()
        self.clear()
        self.setRowCount(len(df))
        self.setColumnCount(len(df.columns))
        self.setHorizontalHeaderLabels([
            " ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in df.columns])
        for r, row in enumerate(df.itertuples(index=False)):
            for c, value in enumerate(row):
                text = float_format.format(value) if isinstance(value, float) else str(value)
                self.setItem(r, c, QTableWidgetItem(text))
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)


class Viewer3D(QWidget):
    """pyvista QtInteractor embedded in the layout."""

    resized = Signal()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.resized.emit()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter.interactor)
        self.plotter.set_background(VIEWER_BACKGROUND)

    def close(self):
        self.plotter.close()
        return super().close()


class PlotCanvas(QWidget):
    """Matplotlib figure + navigation toolbar."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(11, 7))
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(NavigationToolbar2QT(self.canvas, self))
        layout.addWidget(self.canvas)

    def draw(self):
        self.canvas.draw_idle()
