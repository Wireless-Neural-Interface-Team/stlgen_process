"""Main window: one tab per tool."""

from __future__ import annotations

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMainWindow, QMessageBox, QTabWidget

from .tabs.protocol_tab import ProtocolTab
from .tabs.stats_tab import StatsTab


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Process STL — insertion protocol & geometry statistics")

        self.protocol_tab = ProtocolTab()
        self.stats_tab = StatsTab()

        self.tabs = QTabWidget()
        self.tabs.addTab(self.protocol_tab, "Insertion protocol")
        self.tabs.addTab(self.stats_tab, "STL statistics")
        self.setCentralWidget(self.tabs)

        help_menu = self.menuBar().addMenu("&Help")
        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
        self.statusBar().showMessage("Ready.")

    def _show_about(self):
        QMessageBox.about(
            self, "About",
            "Process STL GUI\n\n"
            "Insertion protocol: load the whole brain, visual cortex, dLGN (and "
            "optionally skull) STLs of one geometry, compute their centers of mass "
            "and the dLGN insertion protocol, and view them in 3D.\n\n"
            "STL statistics: compare the bounding box size and center of mass of "
            "the STL files of several folders, with p-values.",
        )

    def closeEvent(self, event):
        self.protocol_tab.shutdown()
        super().closeEvent(event)
