"""Utilities for Ray Job submission."""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, MutableMapping, Sequence

from .base import JobSpec, SubmissionResult


@dataclass(slots=True)
class RayJobSpec:
    """Specification for submitting Ray Jobs."""

    job: JobSpec
    address: str = "auto"
    runtime_env: Mapping[str, object] = field(default_factory=dict)
    working_dir: Path | None = None
    submission_id: str | None = None


class RaySubmitter:
    """Create and submit Ray Jobs via the Ray Jobs CLI."""

    def __init__(self, *, ray_cmd: Sequence[str] | None = None) -> None:
        self.ray_cmd: Sequence[str] = tuple(ray_cmd or ("ray", "job", "submit"))

    def build_command(self, spec: RayJobSpec) -> Sequence[str]:
        """Return the Ray CLI command for the provided specification."""

        command = list(self.ray_cmd)
        command.extend(["--address", spec.address])
        if spec.working_dir:
            command.extend(["--working-dir", str(spec.working_dir)])
        if spec.runtime_env:
            command.extend(["--runtime-env", json.dumps(spec.runtime_env)])
        if spec.submission_id:
            command.extend(["--submission-id", spec.submission_id])
        command.append("--")
        command.append(spec.job.entrypoint)
        command.extend(spec.job.args)
        return command

    def submit(
        self,
        spec: RayJobSpec,
        *,
        dry_run: bool = False,
        env: MutableMapping[str, str] | None = None,
    ) -> SubmissionResult:
        """Submit the job to Ray and return the submission metadata."""

        command = list(self.build_command(spec))
        merged_env = os.environ.copy()
        if spec.job.env:
            merged_env.update(spec.job.env)
        if env:
            merged_env.update(env)

        if dry_run:
            return SubmissionResult(command=command)

        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            env=merged_env,
        )
        return SubmissionResult(
            command=command,
            stdout=completed.stdout.strip(),
            stderr=completed.stderr.strip(),
            job_id=spec.submission_id,
        )
