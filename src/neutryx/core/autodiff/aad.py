"""Utilities for algorithmic adjoint differentiation (AAD).

This module contains helpers that wrap :mod:`jax` primitives to expose common
AAD workflows such as evaluating a function together with its gradient and
Hessian-vector products.  The utilities keep the public API independent of the
exact autodiff backend so that pricing models can request sensitivities without
manually wiring the JAX transforms themselves.
"""
from __future__ import annotations

from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp

ArrayLike = Any


def _wrap_target(
    func: Callable[..., ArrayLike], argnum: int, args: Tuple[Any, ...], kwargs: dict[str, Any]
) -> Callable[[ArrayLike], ArrayLike]:
    """Create a function where ``argnum`` becomes the sole argument.

    Args:
        func: Original callable that accepts ``*args``/``**kwargs``.
        argnum: Index of the argument with respect to which differentiation is
            performed.
        args: Positional arguments supplied to the wrapper.
        kwargs: Keyword arguments supplied to the wrapper.

    Returns:
        A function that only takes the differentiated argument.
    """

    def _partial(target: ArrayLike) -> ArrayLike:
        new_args = list(args)
        new_args[argnum] = target
        return func(*new_args, **kwargs)

    return _partial


def value_grad_hvp(
    func: Callable[..., ArrayLike], argnum: int = 0
) -> Callable[..., Tuple[ArrayLike, ArrayLike, Callable[[ArrayLike], ArrayLike]]]:
    """Build a callable returning value, gradient and Hessian-vector product.

    The returned function evaluates ``func`` with the provided arguments and
    differentiates with respect to ``argnum`` using reverse-mode AD.  The
    Hessian is never materialised explicitly; instead we expose a closure that
    computes ``H @ v`` for arbitrary vectors ``v``.  This keeps the complexity
    linear in the cost of one reverse sweep and matches the workflow expected by
    AAD libraries.

    Args:
        func: Callable to differentiate.  It must be compatible with JAX
            transformations.
        argnum: Positional argument index with respect to which derivatives are
            taken.

    Returns:
        A callable mirroring ``func``'s signature (aside from the differentiated
        argument) that yields a tuple ``(value, grad, apply_hvp)``.  The
        ``apply_hvp`` callable accepts a vector with the same pytree structure
        as the differentiated argument and returns the Hessian-vector product.
    """

    def _evaluate(*args: Any, **kwargs: Any):
        target = args[argnum]
        partial = _wrap_target(func, argnum, args, kwargs)
        value = partial(target)
        grad, linear = jax.linearize(jax.grad(partial), target)

        def _apply(vector: ArrayLike) -> ArrayLike:
            vector_array = jnp.asarray(vector)
            return linear(vector_array)

        return value, grad, _apply

    return _evaluate


def hessian_vector_product(
    func: Callable[..., ArrayLike], argnum: int = 0
) -> Callable[..., Callable[[ArrayLike], ArrayLike]]:
    """Convenience wrapper returning only the Hessian-vector product closure."""

    def _evaluate(*args: Any, **kwargs: Any):
        _, _, hvp = value_grad_hvp(func, argnum)(*args, **kwargs)
        return hvp

    return _evaluate
