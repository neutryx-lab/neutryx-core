"""Request-level profiling middleware."""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Deque, Optional

import pstats

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from .config import ProfilingConfig

try:  # pragma: no cover - optional import path
    import cProfile as profile
except ImportError:  # pragma: no cover - fallback
    import profile  # type: ignore[no-redef]


@contextmanager
def profiling_session() -> profile.Profile:
    """Context manager that yields an active profiler."""

    profiler = profile.Profile()
    profiler.enable()
    try:
        yield profiler
    finally:
        profiler.disable()


@dataclass
class _ProfileArtifact:
    path: str
    timestamp: float


class ProfilingMiddleware(BaseHTTPMiddleware):
    """Middleware capturing cProfile dumps for slow requests."""

    def __init__(self, app, config: ProfilingConfig):
        super().__init__(app)
        self._config = config
        self._artifacts: Deque[_ProfileArtifact] = deque(maxlen=config.retain)
        self._lock = threading.Lock()
        os.makedirs(self._config.output_dir, exist_ok=True)

    async def dispatch(self, request: Request, call_next):
        if not self._config.enabled or self._should_skip(request.url.path):
            return await call_next(request)

        start = time.time()
        with self._profile_scope() as profiler:
            response: Response = await call_next(request)

        duration = time.time() - start
        if duration < self._config.min_duration_seconds:
            return response

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        safe_path = request.url.path.strip("/").replace("/", "_") or "root"
        base_name = f"{timestamp}_{request.method.lower()}_{safe_path}"
        binary_path = os.path.join(self._config.output_dir, f"{base_name}.prof")
        profiler.dump_stats(binary_path)

        if self._config.emit_text_reports:
            text_path = os.path.join(self._config.output_dir, f"{base_name}.txt")
            with open(text_path, "w", encoding="utf-8") as handle:
                stats = pstats.Stats(profiler, stream=handle)
                stats.sort_stats("cumulative")
                stats.print_stats(40)

        with self._lock:
            self._artifacts.append(_ProfileArtifact(path=binary_path, timestamp=time.time()))

        return response

    def _should_skip(self, path: str) -> bool:
        if self._config.include_paths:
            return all(not path.startswith(prefix) for prefix in self._config.include_paths)
        return any(path.startswith(prefix) for prefix in self._config.exclude_paths)

    @contextmanager
    def _profile_scope(self):
        with profiling_session() as profiler:
            yield profiler


__all__ = ["ProfilingMiddleware"]
