"""Utilities for collecting JAX kernel level profiling information.

The implementation relies on ``jax.profiler``'s ability to emit trace
files (``*.xplane.pb``) that contain detailed timing information about
host and device execution.  This module wraps the trace lifecycle in a
Pythonic context manager so that instrumentation can be turned on for a
single block of code and automatically converted into structured data.

Example
-------
>>> from tools.profiling import KernelProfiler
>>> import jax.numpy as jnp
>>> with KernelProfiler("/tmp/profile") as profiler:
...     x = jnp.ones((512, 512))
...     y = x @ x
...     _ = y.block_until_ready()
>>> df = profiler.to_pandas()
>>> print(df.head())

The returned dataframe contains per-kernel timing metrics which can be
aggregated or visualised.
"""

from __future__ import annotations

import dataclasses
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence

import jax.profiler

try:  # Pandas is part of the project dependencies but import lazily.
    import pandas as pd
except Exception:  # pragma: no cover - exercised when pandas is missing.
    pd = None  # type: ignore


@dataclass(frozen=True)
class KernelEvent:
    """Representation of a single kernel (or host) profiling event."""

    name: str
    plane: str
    line: str
    start_ns: int
    duration_ns: int

    @property
    def duration_ms(self) -> float:
        """Duration expressed in milliseconds."""

        return self.duration_ns / 1_000_000.0


class KernelProfiler:
    """Context manager that captures kernel timing information.

    Parameters
    ----------
    logdir:
        Directory where trace artifacts will be written.  The directory
        will be created if it does not exist.
    include_python_events:
        Whether to keep Python stack samples.  By default only XLA
        related lines are kept which focuses the report on kernel level
        timings instead of every single Python call.
    wait_for_flush_s:
        Amount of time to wait after ``stop_trace`` is invoked so that
        the runtime has a chance to flush profiling data to disk.  A
        small delay greatly improves reliability for short traces.
    """

    def __init__(
        self,
        logdir: str | Path,
        *,
        include_python_events: bool = False,
        wait_for_flush_s: float = 0.05,
    ) -> None:
        self.logdir = Path(logdir)
        self.include_python_events = include_python_events
        self.wait_for_flush_s = max(0.0, wait_for_flush_s)
        self._active = False
        self._existing_runs: set[str] = set()
        self._events: Optional[List[KernelEvent]] = None
        self._trace_run_dirs: List[Path] = []

    @classmethod
    def from_existing(
        cls,
        logdir: str | Path,
        *,
        include_python_events: bool = False,
    ) -> "KernelProfiler":
        """Instantiate a profiler that loads data from an existing trace.

        This is primarily useful for offline analysis or dashboards where
        profiling has already been performed.
        """

        profiler = cls(logdir, include_python_events=include_python_events)
        profile_root = profiler._profile_root()
        if profile_root.exists():
            run_dirs = [p for p in profile_root.iterdir() if p.is_dir()]
            profiler._trace_run_dirs = sorted(run_dirs, key=lambda p: p.stat().st_mtime)
            profiler._events = profiler._load_events(profiler._trace_run_dirs)
        else:
            profiler._trace_run_dirs = []
            profiler._events = []
        return profiler

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------
    def __enter__(self) -> "KernelProfiler":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        # Always attempt to stop tracing even if the wrapped block raised.
        try:
            self.stop()
        except Exception as err:  # pragma: no cover - defensive programming.
            warnings.warn(f"Failed to stop kernel profiler cleanly: {err}", RuntimeWarning)
        # Returning ``False`` propagates any exception raised inside the context.
        return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def events(self) -> List[KernelEvent]:
        """Return the captured events.

        ``stop`` must have been called before accessing this property.
        """

        if self._events is None:
            raise RuntimeError("profiling data not available; did you call stop()?")
        return self._events

    def start(self) -> None:
        """Start tracing kernels."""

        if self._active:
            raise RuntimeError("KernelProfiler is already running")

        self.logdir.mkdir(parents=True, exist_ok=True)
        profile_root = self._profile_root()
        if profile_root.exists():
            self._existing_runs = {p.name for p in profile_root.iterdir() if p.is_dir()}
        else:
            self._existing_runs = set()

        jax.profiler.start_trace(str(self.logdir))
        self._active = True

    def stop(self) -> None:
        """Stop tracing kernels and collect events."""

        if not self._active:
            return

        try:
            jax.profiler.stop_trace()
        finally:
            self._active = False

        if self.wait_for_flush_s:
            time.sleep(self.wait_for_flush_s)

        self._trace_run_dirs = self._collect_new_run_dirs()
        self._events = self._load_events(self._trace_run_dirs)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def to_pandas(self) -> "pd.DataFrame":
        """Return the captured events as a :class:`pandas.DataFrame`.

        Raises
        ------
        RuntimeError
            If ``stop`` has not been called yet.
        ImportError
            If pandas is not available.
        """

        if pd is None:  # pragma: no cover - dependent on optional import.
            raise ImportError("pandas is required for to_pandas()")

        records = [dataclasses.asdict(evt) | {"duration_ms": evt.duration_ms} for evt in self.events]
        df = pd.DataFrame.from_records(records)
        if df.empty:
            return df

        # Provide a sensible default ordering focused on costly kernels.
        return df.sort_values("duration_ns", ascending=False).reset_index(drop=True)

    def summary(self, top_k: int = 20) -> "pd.DataFrame":
        """Aggregate events grouped by kernel name.

        Parameters
        ----------
        top_k:
            Limit the summary to the most expensive kernels by total
            time.  ``None`` keeps every row.
        """

        if pd is None:  # pragma: no cover - dependent on optional import.
            raise ImportError("pandas is required for summary()")

        df = self.to_pandas()
        if df.empty:
            return df

        grouped = (
            df.groupby(["name", "plane", "line"], as_index=False)["duration_ns"]
            .agg(["count", "sum", "max", "mean"])
            .rename(columns={
                "count": "calls",
                "sum": "total_duration_ns",
                "max": "max_duration_ns",
                "mean": "mean_duration_ns",
            })
        )
        grouped = grouped.sort_values("total_duration_ns", ascending=False)
        if top_k is not None:
            grouped = grouped.head(top_k)
        grouped["total_duration_ms"] = grouped["total_duration_ns"] / 1_000_000.0
        grouped["mean_duration_ms"] = grouped["mean_duration_ns"] / 1_000_000.0
        grouped["max_duration_ms"] = grouped["max_duration_ns"] / 1_000_000.0
        return grouped.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _profile_root(self) -> Path:
        return self.logdir / "plugins" / "profile"

    def _collect_new_run_dirs(self) -> List[Path]:
        profile_root = self._profile_root()
        if not profile_root.exists():
            warnings.warn(
                "No profiling artifacts produced; did any work run while tracing?",
                RuntimeWarning,
            )
            return []

        run_dirs = [p for p in profile_root.iterdir() if p.is_dir()]
        new_dirs = [p for p in run_dirs if p.name not in self._existing_runs]
        if not new_dirs:
            # Fallback to the most recent run to avoid silent failures.
            if run_dirs:
                newest = max(run_dirs, key=lambda p: p.stat().st_mtime)
                warnings.warn(
                    "Profiling did not produce a new run directory; using the most recent trace instead.",
                    RuntimeWarning,
                )
                return [newest]
            return []

        return sorted(new_dirs, key=lambda p: p.stat().st_mtime)

    def _load_events(self, run_dirs: Sequence[Path]) -> List[KernelEvent]:
        events: List[KernelEvent] = []
        for run_dir in run_dirs:
            for pb_file in sorted(run_dir.glob("*.xplane.pb")):
                events.extend(self._load_events_from_file(pb_file))
        return events

    def _load_events_from_file(self, pb_file: Path) -> List[KernelEvent]:
        events: List[KernelEvent] = []
        profile_data = jax.profiler.ProfileData.from_file(str(pb_file))
        for plane in profile_data.planes:
            plane_name = getattr(plane, "name", "")
            for line in plane.lines:
                line_name = getattr(line, "name", "")
                if not self.include_python_events and line_name.lower() == "python":
                    continue
                for event in line.events:
                    duration_ns = getattr(event, "duration_ns", None)
                    start_ns = getattr(event, "start_ns", None)
                    if duration_ns is None or start_ns is None:
                        continue
                    try:
                        duration_ns_int = int(duration_ns)
                        start_ns_int = int(start_ns)
                    except (TypeError, ValueError):
                        continue
                    if duration_ns_int <= 0:
                        continue
                    events.append(
                        KernelEvent(
                            name=getattr(event, "name", "<unnamed>"),
                            plane=plane_name,
                            line=line_name,
                            start_ns=start_ns_int,
                            duration_ns=duration_ns_int,
                        )
                    )
        return events


def iter_events(events: Iterable[KernelEvent]) -> Iterator[KernelEvent]:
    """Simple iterator helper exposed for convenience."""

    return iter(events)

