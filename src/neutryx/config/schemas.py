"""Pydantic-based configuration schemas and helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)


class EngineSettings(BaseModel):
    """Simulation engine configuration parsed from YAML."""

    model_config = ConfigDict(extra="forbid")

    steps: int = Field(gt=0, description="Number of timesteps per path")
    paths: int = Field(gt=0, description="Number of Monte Carlo paths")
    dtype: str = Field(default="float32", description="Floating point precision for simulations")
    antithetic: bool = Field(default=False, description="Use antithetic variance reduction")

    @field_validator("dtype")
    @classmethod
    def validate_dtype(cls, value: str) -> str:
        allowed = {"float32", "float64", "bfloat16"}
        canonical = value.lower()
        if canonical not in allowed:
            raise ValueError(f"dtype must be one of {sorted(allowed)}")
        return canonical

    @model_validator(mode="after")
    def validate_antithetic(self) -> "EngineSettings":
        if self.antithetic and self.paths % 2 != 0:
            raise ValueError("Antithetic sampling requires an even number of paths")
        return self


class MarketSettings(BaseModel):
    """Market data inputs used by pricing engines."""

    model_config = ConfigDict(extra="forbid")

    r: float = Field(description="Risk-free rate")
    q: float = Field(default=0.0, description="Dividend yield")


class AppConfig(BaseModel):
    """Top-level configuration container for experiments or runs."""

    model_config = ConfigDict(extra="forbid")

    seed: int = Field(ge=0, description="Seed for PRNG initialisation")
    engine: EngineSettings
    market: MarketSettings

    def to_mc_config(self):
        """Convert to the internal :class:`~neutryx.core.engine.MCConfig`."""
        from jax import numpy as jnp

        from neutryx.core.engine import MCConfig

        dtype_map = {
            "float32": jnp.float32,
            "float64": jnp.float64,
            "bfloat16": jnp.bfloat16,
        }
        dtype = dtype_map[self.engine.dtype]
        return MCConfig(
            steps=self.engine.steps,
            paths=self.engine.paths,
            dtype=dtype,
            antithetic=self.engine.antithetic,
        )


class ConfigValidationError(RuntimeError):
    """Raised when one or more configuration files fail validation."""

    def __init__(self, errors: list[tuple[Path, ValidationError]]):
        message_lines = ["Configuration validation failed:"]
        for path, error in errors:
            message_lines.append(f"- {path}: {error}")
        super().__init__("\n".join(message_lines))
        self.errors = errors


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration at {path} must contain a mapping")
    return data


def load_config(path: Path | str) -> AppConfig:
    """Load a configuration file into an :class:`AppConfig`."""
    target = Path(path)
    payload = _load_yaml(target)
    return AppConfig.model_validate(payload)


def discover_config_files(paths: Iterable[Path | str]) -> list[Path]:
    """Discover YAML configuration files from provided paths."""
    discovered: list[Path] = []
    seen = set()
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_file() and path.suffix in {".yml", ".yaml"}:
            resolved = path.resolve()
            if resolved not in seen:
                discovered.append(resolved)
                seen.add(resolved)
        elif path.is_dir():
            for pattern in ("*.yml", "*.yaml"):
                for candidate in sorted(path.rglob(pattern)):
                    resolved = candidate.resolve()
                    if resolved not in seen:
                        discovered.append(resolved)
                        seen.add(resolved)
    return discovered


def collect_and_validate(paths: Iterable[Path | str]) -> list[AppConfig]:
    """Validate all configuration files under the given paths."""
    files = discover_config_files(paths)
    errors: list[tuple[Path, ValidationError]] = []
    configs: list[AppConfig] = []
    for file in files:
        try:
            configs.append(load_config(file))
        except ValidationError as error:
            errors.append((file, error))
    if errors:
        raise ConfigValidationError(errors)
    return configs


__all__ = [
    "EngineSettings",
    "MarketSettings",
    "AppConfig",
    "load_config",
    "discover_config_files",
    "collect_and_validate",
    "ConfigValidationError",
]
