"""Ensure repository paths are importable during tests."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SRC = REPO_ROOT / "src"
DEV = REPO_ROOT / "dev"

for path in (SRC, DEV):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))
