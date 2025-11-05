"""Configure test environment for importing the project package."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(autouse=True)
def reset_observability():
    """Reset observability state after each test to prevent cross-test contamination."""
    yield
    # Reset after test completes
    try:
        from neutryx.infrastructure.observability import reset_metrics_recorder
        reset_metrics_recorder()
    except ImportError:
        pass  # Module not available in some test contexts
