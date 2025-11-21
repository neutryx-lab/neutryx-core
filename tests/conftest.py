"""Configure test environment for importing the project package."""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
DEV = REPO_ROOT / "dev"

for path in (SRC, DEV):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))


@pytest.fixture(scope="session")
def anyio_backend():
    """Override anyio_backend to only use asyncio, not trio."""
    return "asyncio", {"use_uvloop": False}


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
