"""Workflow utilities with checkpoint and resume support."""
from __future__ import annotations

import copy
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, MutableMapping, Optional, Tuple

from neutryx.infrastructure.governance import DataFlowRecorder, get_dataflow_recorder

State = MutableMapping[str, Any]
StepFn = Callable[[int, State], State]


@dataclass(slots=True)
class CheckpointManager:
    """Persist workflow state to disk to enable resuming."""

    directory: Path
    prefix: str = "ckpt"

    def __post_init__(self) -> None:
        self.directory = Path(self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    @property
    def meta_path(self) -> Path:
        return self.directory / f"{self.prefix}_meta.json"

    def _state_path(self, step: int) -> Path:
        return self.directory / f"{self.prefix}_state_{step}.pkl"

    def chunk_path(self, name: str, step: int) -> Path:
        """Return the path for a chunk artifact stored alongside checkpoints."""

        return self.directory / f"{self.prefix}_{name}_{step}.pkl"

    def save(self, completed_steps: int, state: State) -> None:
        """Persist the provided state and completed step count."""

        recorder = get_dataflow_recorder()
        record = recorder.record_flow(
            job_id=f"checkpoint:{self.prefix}",
            source="neutryx.infrastructure.workflows.CheckpointManager.save",
            inputs={"step": completed_steps},
            outputs={"state_keys": sorted(state.keys())},
            context={"directory": str(self.directory)},
        )
        metadata = state.setdefault("_metadata", {})
        if isinstance(metadata, MutableMapping):
            DataFlowRecorder.inject_lineage(metadata, record.lineage_id)
        else:
            state["_metadata"] = {"lineage_id": record.lineage_id}

        payload = {"step": completed_steps, "state": copy.deepcopy(state)}
        path = self._state_path(completed_steps)
        tmp_path = path.with_suffix(".tmp")
        with tmp_path.open("wb") as fh:
            pickle.dump(payload, fh)
        tmp_path.replace(path)
        with self.meta_path.open("w", encoding="utf-8") as meta:
            json.dump({"latest_step": completed_steps}, meta)

    def load_latest(self) -> Tuple[int, State] | None:
        """Load the last persisted state if available."""

        if not self.meta_path.exists():
            return None
        with self.meta_path.open("r", encoding="utf-8") as meta:
            info = json.load(meta)
        step = int(info.get("latest_step", 0))
        path = self._state_path(step)
        if not path.exists():
            return None
        with path.open("rb") as fh:
            # nosec B301: Loading trusted workflow state files written by this application
            # Users should ensure the checkpoint directory is properly secured
            payload = pickle.load(fh)  # nosec B301
        return int(payload["step"]), copy.deepcopy(payload["state"])

    def mark_complete(self) -> None:
        """Mark the workflow as complete and remove metadata files."""

        if self.meta_path.exists():
            self.meta_path.unlink()
        for file in self.directory.glob(f"{self.prefix}_state_*.pkl"):
            if file.exists():
                file.unlink()

    def cleanup_chunks(self) -> None:
        """Remove persisted chunk artifacts."""

        for file in self.directory.glob(f"{self.prefix}_*_*.pkl"):
            if "_state_" in file.name:
                continue
            file.unlink()


@dataclass(slots=True)
class ModelWorkflow:
    """Execute a step-based workflow with checkpoint/resume semantics."""

    name: str
    total_steps: int
    checkpoint_manager: CheckpointManager | None = None
    resume: bool = True

    def run(self, step_fn: StepFn, *, initial_state: Optional[State] = None) -> State:
        """Execute the workflow, resuming from checkpoints if possible."""

        state: State = {} if initial_state is None else initial_state
        start_step = 0
        if self.resume and self.checkpoint_manager:
            loaded = self.checkpoint_manager.load_latest()
            if loaded:
                start_step, state = loaded
        completed_normally = True
        for step in range(start_step, self.total_steps):
            state = step_fn(step, state)
            interrupt = bool(state.pop("_interrupt", False))
            if self.checkpoint_manager:
                self.checkpoint_manager.save(step + 1, state)
            if interrupt:
                completed_normally = False
                break
        if self.checkpoint_manager and completed_normally:
            self.checkpoint_manager.mark_complete()
        return state
