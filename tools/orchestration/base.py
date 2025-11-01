"""Shared data structures for job orchestration utilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Sequence


@dataclass(slots=True)
class JobSpec:
    """Specification of a job entrypoint.

    Parameters
    ----------
    name:
        Human friendly job name.
    entrypoint:
        The command or script invoked by the scheduler.
    args:
        Positional arguments forwarded to the entrypoint.
    env:
        Environment variables exported prior to launching the job.
    working_dir:
        Optional working directory for the process.
    output_path:
        Location where scheduler logs are written.
    """

    name: str
    entrypoint: str
    args: Sequence[str] = field(default_factory=tuple)
    env: MutableMapping[str, str] = field(default_factory=dict)
    working_dir: Path | None = None
    output_path: Path | None = None

    def merged_env(self, extra: Mapping[str, str] | None = None) -> Dict[str, str]:
        """Return a merged environment dictionary."""

        merged: Dict[str, str] = dict(self.env)
        if extra:
            merged.update(extra)
        return merged


@dataclass(slots=True)
class SubmissionResult:
    """Container with scheduler submission outcome."""

    command: Sequence[str]
    stdout: str = ""
    stderr: str = ""
    job_id: str | None = None
    script_path: Path | None = None
