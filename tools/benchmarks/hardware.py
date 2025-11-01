"""Utilities for selecting and validating JAX hardware backends."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import jax


@dataclass(frozen=True)
class HardwareTarget:
    """Represents a desired hardware backend for benchmarking."""

    name: str
    backend: Optional[str] = None

    def resolve_backend(self) -> Optional[str]:
        """Return a backend string understood by ``jax.jit``."""

        candidate = self.backend or (None if self.name == "default" else self.name)
        if candidate is None:
            return None
        validate_backend(candidate)
        return candidate

    def describe(self) -> str:
        backend = self.resolve_backend()
        if backend is None:
            return f"{self.name} (default)"
        devices = jax.devices(backend)
        device_names = {device.device_kind for device in devices}
        return f"{self.name} ({backend}, {len(devices)} device(s): {', '.join(sorted(device_names))})"


def available_backends() -> List[str]:
    """Return a sorted list of available backend names."""

    platforms = {device.platform for device in jax.devices()}
    return sorted(platforms)


def validate_backend(backend: str) -> None:
    """Ensure the requested backend is available."""

    try:
        devices = jax.devices(backend)
    except RuntimeError as exc:  # pragma: no cover - depends on environment
        raise ValueError(f"Backend '{backend}' is not recognised by JAX") from exc
    if not devices:
        raise ValueError(f"Backend '{backend}' is not available on this system")


def parse_targets(targets: Iterable[str]) -> List[HardwareTarget]:
    """Parse CLI arguments into :class:`HardwareTarget` instances."""

    resolved = []
    for raw in targets:
        parts = raw.split(":", maxsplit=1)
        name = parts[0].strip().lower()
        backend = parts[1].strip().lower() if len(parts) == 2 else None
        resolved.append(HardwareTarget(name=name, backend=backend))
    if not resolved:
        resolved.append(HardwareTarget(name="default", backend=None))
    return resolved
