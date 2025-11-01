from pathlib import Path
import sys
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    path_str = str(candidate)
    if path_str not in sys.path:
        sys.path.append(path_str)

from tools.orchestration import (
    JobSpec,
    RayJobSpec,
    RaySubmitter,
    SlurmJobOptions,
    SlurmSubmitter,
)


def test_slurm_submitter_dry_run_generates_script(tmp_path: Path) -> None:
    script_path = tmp_path / "job.sbatch"
    job = JobSpec(
        name="test-job",
        entrypoint="python",
        args=("script.py", "--epochs", "10"),
        env={"FOO": "bar"},
        working_dir=tmp_path,
        output_path=tmp_path / "out.log",
    )
    options = SlurmJobOptions(time="00:30:00", cpus_per_task=2, mem="8G", export={"BAR": "baz"})
    submitter = SlurmSubmitter()

    result = submitter.submit(job, options, dry_run=True, script_path=script_path)

    assert script_path.exists()
    script = script_path.read_text()
    assert "#SBATCH --job-name=test-job" in script
    assert "#SBATCH --time=00:30:00" in script
    assert "export FOO=bar BAR=baz" in script
    assert f"cd {tmp_path}" in script
    assert "python script.py --epochs 10" in script
    assert result.command == ["sbatch", str(script_path)]


def test_slurm_submitter_parses_job_id(tmp_path: Path) -> None:
    script_path = tmp_path / "job.sbatch"
    job = JobSpec(name="run", entrypoint="python", args=("main.py",))
    options = SlurmJobOptions()
    submitter = SlurmSubmitter()

    completed = mock.Mock()
    completed.stdout = "Submitted batch job 12345\n"
    completed.stderr = ""
    with mock.patch("subprocess.run", return_value=completed) as run_mock:
        result = submitter.submit(job, options, script_path=script_path)

    run_mock.assert_called_once_with(
        ["sbatch", str(script_path)],
        check=True,
        capture_output=True,
        text=True,
        env=mock.ANY,
    )
    assert result.job_id == "12345"
    assert result.stdout == "Submitted batch job 12345"


def test_ray_submitter_builds_command(tmp_path: Path) -> None:
    job = JobSpec(name="ray-job", entrypoint="python", args=("train.py",))
    spec = RayJobSpec(
        job=job,
        address="ray://head:10001",
        runtime_env={"env_vars": {"FOO": "BAR"}},
        working_dir=tmp_path,
        submission_id="abc-123",
    )
    submitter = RaySubmitter()

    result = submitter.submit(spec, dry_run=True, env={"BAR": "baz"})

    assert result.command[:3] == ["ray", "job", "submit"]
    assert "--address" in result.command
    assert "python" in result.command


def test_ray_submitter_executes_subprocess(tmp_path: Path) -> None:
    job = JobSpec(name="ray-job", entrypoint="python", args=("main.py",))
    spec = RayJobSpec(job=job)
    submitter = RaySubmitter()

    completed = mock.Mock()
    completed.stdout = "submitted"
    completed.stderr = ""
    with mock.patch("subprocess.run", return_value=completed) as run_mock:
        result = submitter.submit(spec)

    run_mock.assert_called_once()
    assert result.stdout == "submitted"
