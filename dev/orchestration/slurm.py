"""Utilities to submit jobs to Slurm clusters."""
from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Mapping, MutableMapping, Sequence

from .base import JobSpec, SubmissionResult


_SLURM_ID_PATTERN = re.compile(r"Submitted batch job (?P<jobid>\d+)")


@dataclass(slots=True)
class SlurmJobOptions:
    """Slurm specific options for batch jobs."""

    partition: str | None = None
    time: str = "01:00:00"
    gpus: int = 0
    cpus_per_task: int = 4
    mem: str = "16G"
    qos: str | None = None
    comment: str | None = None
    mail_user: str | None = None
    mail_type: Sequence[str] = field(default_factory=tuple)
    additional_directives: Sequence[str] = field(default_factory=tuple)
    export: Mapping[str, str] = field(default_factory=dict)


class SlurmSubmitter:
    """Helper responsible for materialising and submitting Slurm jobs."""

    def __init__(self, *, sbatch_cmd: str = "sbatch") -> None:
        self.sbatch_cmd = sbatch_cmd

    def render_script(self, job: JobSpec, options: SlurmJobOptions) -> str:
        """Render a Slurm batch script for the provided job specification."""

        directives: List[str] = ["#!/bin/bash"]
        directives.append(f"#SBATCH --job-name={job.name}")
        if job.output_path:
            directives.append(f"#SBATCH --output={job.output_path}")
        directives.append(f"#SBATCH --time={options.time}")
        directives.append(f"#SBATCH --cpus-per-task={options.cpus_per_task}")
        directives.append(f"#SBATCH --mem={options.mem}")
        if options.partition:
            directives.append(f"#SBATCH --partition={options.partition}")
        if options.gpus:
            directives.append(f"#SBATCH --gres=gpu:{options.gpus}")
        if options.qos:
            directives.append(f"#SBATCH --qos={options.qos}")
        if options.comment:
            directives.append(f"#SBATCH --comment={options.comment}")
        if options.mail_user:
            directives.append(f"#SBATCH --mail-user={options.mail_user}")
        if options.mail_type:
            directives.append("#SBATCH --mail-type=" + ",".join(options.mail_type))
        directives.extend(options.additional_directives)

        body: List[str] = []
        exports = job.merged_env(options.export)
        if exports:
            body.append("export " + " ".join(f"{k}={v}" for k, v in exports.items()))
        if job.working_dir:
            body.append(f"cd {job.working_dir}")
        cmd_parts = [job.entrypoint, *job.args]
        body.append(" ".join(cmd_parts))
        return "\n".join(directives + ["", *body, ""]) + "\n"

    def submit(
        self,
        job: JobSpec,
        options: SlurmJobOptions,
        *,
        dry_run: bool = False,
        script_path: Path | None = None,
        env: MutableMapping[str, str] | None = None,
    ) -> SubmissionResult:
        """Submit the job to Slurm returning the scheduler output."""

        script_text = self.render_script(job, options)
        target_path = script_path or Path.cwd() / f"{job.name}.sbatch"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(script_text, encoding="utf-8")

        command = [self.sbatch_cmd, str(target_path)]
        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)

        if dry_run:
            return SubmissionResult(command=command, script_path=target_path)

        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            env=merged_env,
        )
        stdout = completed.stdout.strip()
        match = _SLURM_ID_PATTERN.search(stdout)
        job_id = match.group("jobid") if match else None
        return SubmissionResult(
            command=command,
            stdout=stdout,
            stderr=completed.stderr.strip(),
            job_id=job_id,
            script_path=target_path,
        )

    @staticmethod
    def resume_command(job_id: str) -> Sequence[str]:
        """Return a best-effort resume command (requeue) for the job."""

        return ["scontrol", "requeue", job_id]
