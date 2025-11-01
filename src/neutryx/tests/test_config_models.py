from __future__ import annotations

from pathlib import Path

import pytest

from neutryx.core.engine import MCConfig
from neutryx.core.config.schemas import (
    AppConfig,
    ConfigValidationError,
    EngineSettings,
    collect_and_validate,
    load_config,
)


def test_load_config_produces_valid_app_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
seed: 101
engine:
  dtype: float32
  paths: 64
  steps: 16
market:
  r: 0.01
  q: 0.0
""".strip()
    )

    config = load_config(config_path)
    assert isinstance(config, AppConfig)
    assert config.engine.paths == 64
    assert config.market.r == pytest.approx(0.01)


def test_to_mc_config_conversion() -> None:
    config = AppConfig(
        seed=1,
        engine=EngineSettings(dtype="float32", paths=32, steps=8),
        market={"r": 0.02, "q": 0.0},
    )

    mc_config = config.to_mc_config()
    assert isinstance(mc_config, MCConfig)
    assert mc_config.paths == 32
    assert mc_config.steps == 8


def test_collect_and_validate_detects_invalid(tmp_path: Path) -> None:
    good = tmp_path / "good.yaml"
    bad = tmp_path / "bad.yaml"
    good.write_text(
        """
seed: 1
engine:
  dtype: float32
  paths: 32
  steps: 8
market:
  r: 0.01
""".strip()
    )
    bad.write_text(
        """
seed: 1
engine:
  dtype: bad_precision
  paths: 10
  steps: 0
""".strip()
    )

    with pytest.raises(ConfigValidationError) as exc:
        collect_and_validate([tmp_path])

    assert "bad.yaml" in str(exc.value)


def test_engine_settings_requires_even_paths_for_antithetic() -> None:
    with pytest.raises(ValueError):
        EngineSettings(paths=5, steps=2, antithetic=True)

    settings = EngineSettings(paths=6, steps=2, antithetic=True)
    assert settings.paths == 6
