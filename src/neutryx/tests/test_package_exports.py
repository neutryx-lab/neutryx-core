"""Tests for the public exports of the :mod:`neutryx` package."""

import neutryx


def test_public_attributes_resolvable() -> None:
    """Every name listed in ``neutryx.__all__`` should be importable."""

    for name in neutryx.__all__:
        assert getattr(neutryx, name) is not None
