"""Common product abstractions and utilities.

This module defines lightweight product base classes that expose a unified
interface to Monte Carlo (path based) and PDE (terminal state) engines.  Path
based engines can simply ``vmap`` the :meth:`payoff_path` method, while PDE
solvers can query :meth:`terminal_payoff` for payoffs that depend only on the
terminal asset level.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import jax
import jax.numpy as jnp

Array = jnp.ndarray


class SupportsPayoff(Protocol):
    """Protocol describing the minimal payoff interface."""

    T: float

    def payoff_path(self, path: Array) -> Array:
        """Return the payoff for a single simulated price path."""

    def payoff_terminal(self, spot: Array) -> Array:
        """Return the payoff for a terminal spot value."""


@dataclass
class Product:
    """Base class for products supported by Neutryx engines.

    Sub-classes are expected to implement :meth:`payoff_terminal`.  The default
    :meth:`payoff_path` simply evaluates the terminal payoff on the last entry
    of the path which works for path-independent claims such as vanilla
    Europeans.  Path-dependent claims can override :meth:`payoff_path` and set
    :pyattr:`requires_path` to ``True``.
    """

    T: float

    @property
    def requires_path(self) -> bool:
        """Whether the product requires the full simulation path."""

        return False

    @property
    def supports_pde(self) -> bool:
        """Return True when the product exposes a terminal payoff."""

        return not self.requires_path

    def payoff_terminal(self, spot: Array) -> Array:
        """Return the payoff for the provided terminal spot value."""

        raise NotImplementedError

    def payoff_path(self, path: Array) -> Array:
        """Return the payoff for a single simulated path."""

        path_arr = jnp.asarray(path)
        if path_arr.ndim == 0:
            # Scalar input - treat as terminal value directly
            terminal = path_arr
        else:
            # Array input - extract last element
            terminal = path_arr[-1]
        return self.payoff_terminal(terminal)

    def payoff(self, path: Array) -> Array:
        """Alias to :meth:`payoff_path` to match historical API."""

        return self.payoff_path(path)

    # ------------------------------------------------------------------
    # Vectorised helpers
    # ------------------------------------------------------------------
    def path_payoffs(self, paths: Array) -> Array:
        """Vectorise :meth:`payoff_path` across Monte Carlo paths."""

        arr = jnp.asarray(paths)
        if arr.ndim == 1:
            return self.payoff_path(arr)
        return jax.vmap(self.payoff_path)(arr)

    def terminal_payoffs(self, spots: Array) -> Array:
        """Vectorise :meth:`payoff_terminal` for PDE grids."""

        if self.requires_path:
            raise NotImplementedError(
                "Terminal payoff unavailable for path-dependent product."
            )
        arr = jnp.asarray(spots)
        if arr.ndim == 0:
            return self.payoff_terminal(arr)
        return jax.vmap(self.payoff_terminal)(arr)


@dataclass
class PathProduct(Product):
    """Base class for path-dependent claims."""

    @property
    def requires_path(self) -> bool:  # pragma: no cover - trivial property
        return True

    @property
    def supports_pde(self) -> bool:  # pragma: no cover - trivial property
        return False

    def payoff_terminal(self, spot: Array) -> Array:  # pragma: no cover
        raise NotImplementedError(
            "Path dependent product does not admit a terminal-only payoff."
        )
