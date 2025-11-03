"""
Numerical solvers for root finding and optimization.

This module provides various root-finding algorithms commonly used in
financial mathematics for:
- Bootstrapping yield curves
- Implied volatility calculation
- Finding par rates
- Solving for discount factors
"""

from typing import Callable, Optional, Tuple

import jax.numpy as jnp
from jax import grad


def newton_raphson(
    f: Callable[[float], float],
    x0: float,
    fprime: Optional[Callable[[float], float]] = None,
    tol: float = 1e-8,
    max_iter: int = 100,
    bounds: Optional[Tuple[float, float]] = None,
) -> float:
    """
    Newton-Raphson method for finding roots of f(x) = 0.

    Uses automatic differentiation via JAX if derivative not provided.
    Newton-Raphson converges quadratically for smooth functions near the root.

    Args:
        f: Function for which to find root
        x0: Initial guess
        fprime: Derivative of f (optional, will use JAX grad if not provided)
        tol: Tolerance for convergence
        max_iter: Maximum number of iterations
        bounds: Optional (lower, upper) bounds to constrain search

    Returns:
        Root x such that f(x) ≈ 0

    Raises:
        ValueError: If convergence fails or bounds are violated

    Example:
        >>> # Find square root of 2 (solve x² - 2 = 0)
        >>> def f(x): return x**2 - 2.0
        >>> root = newton_raphson(f, x0=1.0)
        >>> print(f"{root:.10f}")  # Should be ≈ 1.4142135624
    """
    # Use automatic differentiation if derivative not provided
    if fprime is None:
        fprime = grad(f)

    x = x0
    for i in range(max_iter):
        fx = f(x)

        # Check convergence based on function value
        if abs(fx) < tol:
            return x

        # Compute derivative
        dfx = fprime(x)

        # Check for zero derivative
        if abs(dfx) < 1e-15:
            raise ValueError(
                f"Newton-Raphson encountered zero derivative at x={x}. "
                "Try a different initial guess or use a bracketing method."
            )

        # Newton-Raphson update
        x_new = x - fx / dfx

        # Apply bounds if provided
        if bounds is not None:
            x_new = jnp.clip(x_new, bounds[0], bounds[1])

        # Check convergence based on change in x
        if abs(x_new - x) < tol * max(1.0, abs(x)):
            return x_new

        x = x_new

    # Check if we're close enough even if not within strict tolerance
    if abs(f(x)) < tol * 100:
        return x

    raise ValueError(
        f"Newton-Raphson failed to converge after {max_iter} iterations. "
        f"Last value: x={x}, f(x)={f(x)}"
    )


def bisection(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    """
    Bisection method for finding roots of f(x) = 0.

    Robust bracketing method that's guaranteed to converge if f(a) and f(b)
    have opposite signs. Slower than Newton-Raphson but more reliable.

    Args:
        f: Function for which to find root
        a: Lower bound of bracket
        b: Upper bound of bracket
        tol: Tolerance for convergence
        max_iter: Maximum number of iterations

    Returns:
        Root x such that f(x) ≈ 0

    Raises:
        ValueError: If f(a) and f(b) have same sign or convergence fails

    Example:
        >>> def f(x): return x**2 - 4.0
        >>> root = bisection(f, a=0.0, b=3.0)
        >>> print(f"{root:.10f}")  # Should be ≈ 2.0
    """
    fa = f(a)
    fb = f(b)

    # Check that function has opposite signs at endpoints
    if fa * fb > 0:
        raise ValueError(
            f"Function must have opposite signs at endpoints. "
            f"f({a})={fa}, f({b})={fb}"
        )

    for i in range(max_iter):
        # Midpoint
        c = (a + b) / 2.0
        fc = f(c)

        # Check convergence
        if abs(fc) < tol or abs(b - a) < tol:
            return c

        # Update bracket
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

    raise ValueError(
        f"Bisection failed to converge after {max_iter} iterations. "
        f"Last interval: [{a}, {b}]"
    )


def brent(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    """
    Brent's method for finding roots of f(x) = 0.

    Combines bisection, secant method, and inverse quadratic interpolation
    for fast convergence with guaranteed bracketing. Generally the best
    all-around root-finding method.

    Args:
        f: Function for which to find root
        a: Lower bound of bracket
        b: Upper bound of bracket
        tol: Tolerance for convergence
        max_iter: Maximum number of iterations

    Returns:
        Root x such that f(x) ≈ 0

    Raises:
        ValueError: If f(a) and f(b) have same sign or convergence fails

    Example:
        >>> def f(x): return x**3 - 2*x - 5
        >>> root = brent(f, a=1.0, b=3.0)
        >>> print(f"{root:.10f}")  # Should be ≈ 2.0946
    """
    fa = f(a)
    fb = f(b)

    # Check that function has opposite signs at endpoints
    if fa * fb > 0:
        raise ValueError(
            f"Function must have opposite signs at endpoints. "
            f"f({a})={fa}, f({b})={fb}"
        )

    # Ensure |f(b)| <= |f(a)|
    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    c = a
    fc = fa
    mflag = True

    for i in range(max_iter):
        # Check convergence
        if abs(fb) < tol or abs(b - a) < tol:
            return b

        # Try inverse quadratic interpolation
        if fa != fc and fb != fc:
            s = (
                a * fb * fc / ((fa - fb) * (fa - fc))
                + b * fa * fc / ((fb - fa) * (fb - fc))
                + c * fa * fb / ((fc - fa) * (fc - fb))
            )
        else:
            # Use secant method
            s = b - fb * (b - a) / (fb - fa)

        # Check if we should use bisection instead
        tmp1 = (3 * a + b) / 4
        tmp2 = b
        if tmp1 > tmp2:
            tmp1, tmp2 = tmp2, tmp1

        cond1 = not (tmp1 < s < tmp2)
        cond2 = mflag and abs(s - b) >= abs(b - c) / 2
        cond3 = not mflag and abs(s - b) >= abs(c - a) / 2
        cond4 = mflag and abs(b - c) < tol
        cond5 = not mflag and abs(c - a) < tol

        if cond1 or cond2 or cond3 or cond4 or cond5:
            s = (a + b) / 2
            mflag = True
        else:
            mflag = False

        fs = f(s)
        a = c
        fa = fc
        c = b
        fc = fb

        if fa * fs < 0:
            b = s
            fb = fs
        else:
            a = s
            fa = fs

        # Ensure |f(b)| <= |f(a)|
        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa

    # Check if we're close enough even if not within strict tolerance
    if abs(fb) < tol * 1000 or abs(b - a) < tol * 1000:
        return b

    raise ValueError(
        f"Brent's method failed to converge after {max_iter} iterations. "
        f"Last value: x={b}, f(x)={fb}"
    )


def implied_volatility_newton(
    market_price: float,
    pricing_func: Callable[[float], float],
    vol0: float = 0.2,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> float:
    """
    Calculate implied volatility using Newton-Raphson with vega.

    Specialized solver for finding implied volatility from option prices.
    This is a common application in options markets.

    Args:
        market_price: Observed market price of the option
        pricing_func: Function that takes volatility and returns option price
        vol0: Initial volatility guess
        tol: Tolerance for convergence
        max_iter: Maximum number of iterations

    Returns:
        Implied volatility

    Raises:
        ValueError: If convergence fails

    Example:
        >>> # Black-Scholes call price
        >>> def bs_call(vol):
        ...     # Simplified - actual BS formula needed
        ...     return vol * 10.0  # Placeholder
        >>> iv = implied_volatility_newton(market_price=2.5, pricing_func=bs_call)
    """

    def price_error(vol):
        return pricing_func(vol) - market_price

    return newton_raphson(price_error, x0=vol0, tol=tol, max_iter=max_iter)
