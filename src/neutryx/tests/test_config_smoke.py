from __future__ import annotations

import logging
import random

import jax
import numpy as np

from neutryx.config import get_config, init_environment
from neutryx.core.rng import KeySeq


def _draw_python_random() -> list[float]:
    return [random.random() for _ in range(5)]


def test_python_and_numpy_determinism() -> None:
    config = get_config({"seed": 123, "logging": {"level": "DEBUG"}})

    init_environment(config)
    expected_python = _draw_python_random()
    expected_numpy = np.random.rand(4)

    init_environment(config)
    actual_python = _draw_python_random()
    actual_numpy = np.random.rand(4)

    assert expected_python == actual_python
    np.testing.assert_allclose(expected_numpy, actual_numpy)


def test_key_sequence_determinism() -> None:
    config = get_config({"seed": 7})

    first_cfg = init_environment(config)
    key_seq_first = KeySeq.from_config(first_cfg)
    values_first = jax.random.normal(key_seq_first.next(), (3,))

    second_cfg = init_environment(config)
    key_seq_second = KeySeq.from_config(second_cfg)
    values_second = jax.random.normal(key_seq_second.next(), (3,))

    np.testing.assert_allclose(np.asarray(values_first), np.asarray(values_second))

    logging.getLogger(__name__).debug("Deterministic smoke test completed")
