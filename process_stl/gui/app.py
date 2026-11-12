"""Entry point: ``process-stl-gui``

Optional flags pre-fill the tabs, e.g.:

    process-stl-gui --protocol-dir ../0_stl/2_full/MuMo_25um_paper \
        --skull Calgary_Adult_Skull_Atlas.stl \
        --stats-folder C57=../sources/C57pn56 --stats-folder CD1=../sources/CD1pn56
"""

from __future__ import annotations

import argparse
import sys

import matplotlib

matplotlib.use("QtAgg")

from PySide6.QtWidgets import QApplication  # noqa: E402

from .main_window import MainWindow  # noqa: E402


def _parse_args(argv):
    parser = argparse.ArgumentParser(prog="process-stl-gui",
                                     description="Process STL GUI (flags pre-fill the tabs)")
    parser.add_argument("--protocol-dir", help="region STL directory of the protocol tab")
    parser.add_argument("--skull", help="skull STL file name in --protocol-dir")
    parser.add_argument("--stats-folder", action="append", metavar="[LABEL=]FOLDER",
                        help="folder of the statistics tab (repeat for each folder)")
    parser.add_argument("--data-file", help="CSV data file of the statistics tab")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    app = QApplication(sys.argv[:1])  # keep our own flags away from Qt
    app.setApplicationName("Process STL")
    window = MainWindow()
    window.protocol_tab.set_initial(args.protocol_dir, args.skull)
    window.stats_tab.set_initial(args.stats_folder, args.data_file)
    window.resize(1500, 950)
    window.show()
    return app.exec()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
