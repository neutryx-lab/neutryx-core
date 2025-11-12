"""Pytest configuration for ensuring source layout is importable."""

from __future__ import annotations

import pathlib
import sys

SRC_PATH = pathlib.Path(__file__).resolve().parent.parent / "src"
if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
