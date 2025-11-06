"""Command-line helpers for repository maintenance tasks."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATHS: tuple[str, ...] = ("src", "tests", "examples", "tools")


def _run_command(command: Sequence[str]) -> None:
    """Execute a shell command relative to the repository root."""
    result = subprocess.run(command, cwd=REPO_ROOT)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def _normalize_paths(paths: Sequence[str] | None) -> list[str]:
    if paths:
        return list(paths)
    return list(DEFAULT_PATHS)


def format_code(*paths: str) -> None:
    """Format the codebase using Ruff (fix) and Black."""
    targets = _normalize_paths(paths)
    _run_command(["ruff", "check", "--fix", *targets])
    _run_command(["black", *targets])


def lint_code(*paths: str) -> None:
    """Run lint checks using Ruff and Black in check mode."""
    targets = _normalize_paths(paths)
    _run_command(["ruff", "check", *targets])
    _run_command(["black", "--check", *targets])


def validate_configs(*paths: str) -> None:
    """Validate configuration files via the Pydantic schemas."""
    from neutryx.infrastructure.config.schemas import collect_and_validate

    target_paths = [Path(path) for path in paths] if paths else [REPO_ROOT / "config"]
    configs = collect_and_validate(target_paths)
    message = f"Validated {len(configs)} configuration file(s)."
    print(message)


if __name__ == "__main__":  # pragma: no cover
    cmd, *cmd_paths = sys.argv[1:] or ["lint"]
    if cmd == "lint":
        lint_code(*cmd_paths)
    elif cmd == "format":
        format_code(*cmd_paths)
    elif cmd == "validate-configs":
        validate_configs(*cmd_paths)
    else:
        raise SystemExit(f"Unknown command: {cmd}")
