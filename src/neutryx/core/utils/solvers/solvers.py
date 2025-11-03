"""
Numerical solvers for financial mathematics.

This module provides root-finding and optimization algorithms commonly used
in quantitative finance, including:
- Newton-Raphson method for root finding
- Bisection method for robust bracketing
- Brent's method combining bisection and secant
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import jax.numpy as jnp
from jax import grad


class ConvergenceError(Exception):
    """Raised when a solver fails to converge."""

    pass


def newton_raphson(
    f: Callable[[float], float],
    x0: float,
    fprime: Optional[Callable[[float], float]] = None,
    tol: float = 1e-8,
    max_iter: int = 100,
    bounds: Optional[Tuple[float, float]] = None,
) -> float:
    """
    Find root of function using Newton-Raphson method.

    The Newton-Raphson method uses the iteration:
        x_{n+1} = x_n - f(x_n) / f'(x_n)

    Args:
        f: Function for which to find root
        x0: Initial guess
        fprime: Derivative of f (if None, will use automatic differentiation)
        tol: Convergence tolerance (default 1e-8)
        max_iter: Maximum number of iterations (default 100)
        bounds: Optional (lower, upper) bounds to constrain solution

    Returns:
        Root of f (value x such that f(x) ≈ 0)

    Raises:
        ConvergenceError: If method fails to converge

    Example:
        >>> # Find root of f(x) = x^2 - 2 (answer should be sqrt(2) ≈ 1.414)
        >>> f = lambda x: x**2 - 2
        >>> root = newton_raphson(f, x0=1.0)
        >>> abs(root - 1.41421356) < 1e-6
        True
    """
    # Use automatic differentiation if derivative not provided
    if fprime is None:
        fprime = grad(f)

    x = x0

    for i in range(max_iter):
        fx = f(x)

        # Check convergence
        if abs(fx) < tol:
            return x

        # Compute derivative
        dfx = fprime(x)

        # Check for zero derivative
        if abs(dfx) < 1e-14:
            raise ConvergenceError(f"Zero derivative at iteration {i}, x={x}")

        # Newton-Raphson update
        x_new = x - fx / dfx

        # Apply bounds if specified
        if bounds is not None:
            lower, upper = bounds
            x_new = max(lower, min(upper, x_new))

        # Check for convergence in x
        if abs(x_new - x) < tol:
            return x_new

        x = x_new

    raise ConvergenceError(
        f"Failed to converge after {max_iter} iterations. Last value: {x}, f(x)={f(x)}"
    )


def bisection(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    """
    Find root of function using bisection method.

    The bisection method requires that f(a) and f(b) have opposite signs,
    and repeatedly halves the interval [a, b] until the root is found.

    More robust than Newton-Raphson but slower convergence.

    Args:
        f: Function for which to find root
        a: Lower bound
        b: Upper bound
        tol: Convergence tolerance (default 1e-8)
        max_iter: Maximum number of iterations (default 100)

    Returns:
        Root of f in interval [a, b]

    Raises:
        ValueError: If f(a) and f(b) have the same sign
        ConvergenceError: If method fails to converge

    Example:
        >>> # Find root of f(x) = x^2 - 2 in [0, 2]
        >>> f = lambda x: x**2 - 2
        >>> root = bisection(f, 0.0, 2.0)
        >>> abs(root - 1.41421356) < 1e-6
        True
    """
    fa = f(a)
    fb = f(b)

    if fa * fb > 0:
        raise ValueError(
            f"f(a) and f(b) must have opposite signs: f({a})={fa}, f({b})={fb}"
        )

    for i in range(max_iter):
        # Midpoint
        c = (a + b) / 2.0
        fc = f(c)

        # Check convergence
        if abs(fc) < tol or (b - a) / 2.0 < tol:
            return c

        # Update interval
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

    raise ConvergenceError(
        f"Failed to converge after {max_iter} iterations. Interval: [{a}, {b}]"
    )


def brent(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    """
    Find root using Brent's method (combination of bisection, secant, and inverse quadratic interpolation).

    Brent's method is generally the best all-around root-finding algorithm:
    - More robust than Newton-Raphson (doesn't require derivatives)
    - Faster than pure bisection
    - Guaranteed to converge if root is bracketed

    Args:
        f: Function for which to find root
        a: Lower bound
        b: Upper bound
        tol: Convergence tolerance (default 1e-8)
        max_iter: Maximum number of iterations (default 100)

    Returns:
        Root of f in interval [a, b]

    Raises:
        ValueError: If f(a) and f(b) have the same sign
        ConvergenceError: If method fails to converge

    Example:
        >>> # Find root of f(x) = x^3 - 2*x - 5
        >>> f = lambda x: x**3 - 2*x - 5
        >>> root = brent(f, 2.0, 3.0)
        >>> abs(f(root)) < 1e-8
        True
    """
    fa = f(a)
    fb = f(b)

    if fa * fb > 0:
        raise ValueError(
            f"f(a) and f(b) must have opposite signs: f({a})={fa}, f({b})={fb}"
        )

    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    c = a
    fc = fa
    mflag = True
    d = 0.0

    for i in range(max_iter):
        if abs(fb) < tol:
            return b

        if fa != fc and fb != fc:
            # Inverse quadratic interpolation
            s = (
                a * fb * fc / ((fa - fb) * (fa - fc))
                + b * fa * fc / ((fb - fa) * (fb - fc))
                + c * fa * fb / ((fc - fa) * (fc - fb))
            )
        else:
            # Secant method
            s = b - fb * (b - a) / (fb - fa)

        # Conditions for bisection
        tmp1 = (3 * a + b) / 4
        cond1 = not ((s > tmp1 and s < b) or (s < tmp1 and s > b))
        cond2 = mflag and abs(s - b) >= abs(b - c) / 2
        cond3 = not mflag and abs(s - b) >= abs(c - d) / 2
        cond4 = mflag and abs(b - c) < tol
        cond5 = not mflag and abs(c - d) < tol

        if cond1 or cond2 or cond3 or cond4 or cond5:
            # Use bisection
            s = (a + b) / 2
            mflag = True
        else:
            mflag = False

        fs = f(s)
        d = c
        c = b

        if fa * fs < 0:
            b = s
            fb = fs
        else:
            a = s
            fa = fs

        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa

    raise ConvergenceError(
        f"Failed to converge after {max_iter} iterations. Last value: {b}, f(b)={fb}"
    )


def implied_volatility_newton(
    option_price: float,
    pricing_func: Callable[[float], float],
    vega_func: Callable[[float], float],
    vol_guess: float = 0.2,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> float:
    """
    Calculate implied volatility using Newton-Raphson method.

    This is a specialized version of Newton-Raphson optimized for implied volatility
    calculation in options pricing.

    Args:
        option_price: Observed market price of option
        pricing_func: Function that takes volatility and returns option price
        vega_func: Function that takes volatility and returns vega (∂price/∂vol)
        vol_guess: Initial volatility guess (default 0.2 = 20%)
        tol: Convergence tolerance (default 1e-6)
        max_iter: Maximum iterations (default 50)

    Returns:
        Implied volatility

    Raises:
        ConvergenceError: If unable to find implied volatility

    Example:
        >>> # Calculate implied volatility from Black-Scholes price
        >>> from neutryx.pricing.black_scholes import black_scholes_call, vega
        >>> # Suppose we observe a call option trading at $10
        >>> target_price = 10.0
        >>> S, K, T, r = 100.0, 100.0, 1.0, 0.05
        >>> pricing = lambda vol: black_scholes_call(S, K, T, r, vol)
        >>> vega_calc = lambda vol: vega(S, K, T, r, vol)
        >>> # Find the implied vol
        >>> iv = implied_volatility_newton(target_price, pricing, vega_calc)
        >>> abs(pricing(iv) - target_price) < 1e-6  # Should price correctly
        True
    """

    def objective(vol):
        return pricing_func(vol) - option_price

    def objective_deriv(vol):
        return vega_func(vol)

    try:
        iv = newton_raphson(
            objective, vol_guess, fprime=objective_deriv, tol=tol, max_iter=max_iter, bounds=(0.001, 5.0)
        )
        return iv
    except ConvergenceError as e:
        raise ConvergenceError(f"Failed to find implied volatility: {e}")
