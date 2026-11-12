"""``python -m process_stl`` launches the GUI (use ``process-stl`` for the command line)."""

import sys

from .gui.app import main

sys.exit(main())
