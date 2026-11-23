"""Thin wrapper kept for backwards compatibility; prefer the installed `cc-resize` command."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calligraphic_cut.cli import main  # noqa: E402

main()
