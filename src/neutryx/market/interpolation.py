"""Interpolation methods for curves (yield curves, volatility surfaces, etc.).

Provides various interpolation techniques commonly used in quantitative finance:
- Linear interpolation
- Natural cubic spline
- Hermite cubic spline
- Monotone-preserving cubic spline (for forward rates)
"""
from __future__ import annotations

from typing import Callable

import jax.numpy as jnp
from jax import jit


@jit
def linear_interpolation(x: jnp.ndarray, y: jnp.ndarray, x_new: float) -> float:
    """Linear interpolation.

    Parameters
    ----------
    x : Array
        Known x-coordinates (must be sorted)
    y : Array
        Known y-coordinates
    x_new : float
        Point at which to interpolate

    Returns
    -------
    float
        Interpolated value at x_new

    Notes
    -----
    For points outside the range, uses flat extrapolation (returns boundary value).

    Examples
    --------
    >>> x = jnp.array([0.0, 1.0, 2.0])
    >>> y = jnp.array([0.0, 1.0, 4.0])
    >>> linear_interpolation(x, y, 1.5)
    2.5
    """
    # Handle extrapolation
    x_new = jnp.clip(x_new, x[0], x[-1])

    # Find the interval
    i = jnp.searchsorted(x, x_new) - 1
    i = jnp.clip(i, 0, len(x) - 2)

    # Linear interpolation formula
    x0, x1 = x[i], x[i + 1]
    y0, y1 = y[i], y[i + 1]

    # Handle division by zero
    dx = x1 - x0
    slope = jnp.where(dx > 1e-10, (y1 - y0) / dx, 0.0)

    return y0 + slope * (x_new - x0)


@jit
def natural_cubic_spline_coefficients(
    x: jnp.ndarray, y: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute natural cubic spline coefficients.

    A natural cubic spline has zero second derivative at the boundaries.

    Parameters
    ----------
    x : Array
        Known x-coordinates (must be sorted), shape (n,)
    y : Array
        Known y-coordinates, shape (n,)

    Returns
    -------
    a, b, c, d : Arrays
        Spline coefficients for each interval
        For interval [x[i], x[i+1]], the spline is:
        S_i(t) = a[i] + b[i]*(t-x[i]) + c[i]*(t-x[i])^2 + d[i]*(t-x[i])^3

    Notes
    -----
    Natural boundary conditions: S''(x[0]) = S''(x[-1]) = 0

    Algorithm:
    1. Set up tridiagonal system for second derivatives
    2. Solve for second derivatives (M)
    3. Compute coefficients a, b, c, d from M

    Examples
    --------
    >>> x = jnp.array([0.0, 1.0, 2.0, 3.0])
    >>> y = jnp.array([0.0, 1.0, 0.5, 2.0])
    >>> a, b, c, d = natural_cubic_spline_coefficients(x, y)
    """
    n = len(x) - 1

    # Step sizes
    h = jnp.diff(x)

    # Right-hand side for tridiagonal system
    alpha = jnp.zeros(n - 1)
    for i in range(1, n):
        alpha = alpha.at[i - 1].set(
            (3.0 / h[i]) * (y[i + 1] - y[i]) - (3.0 / h[i - 1]) * (y[i] - y[i - 1])
        )

    # Solve tridiagonal system for second derivatives
    # Natural boundary conditions: c[0] = c[n] = 0
    l = jnp.ones(n + 1)
    mu = jnp.zeros(n + 1)
    z = jnp.zeros(n + 1)

    for i in range(1, n):
        l = l.at[i].set(2.0 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1])
        mu = mu.at[i].set(h[i] / l[i])
        z = z.at[i].set((alpha[i - 1] - h[i - 1] * z[i - 1]) / l[i])

    # Back substitution
    c_vals = jnp.zeros(n + 1)
    b_vals = jnp.zeros(n)
    d_vals = jnp.zeros(n)

    for j in range(n - 1, -1, -1):
        c_vals = c_vals.at[j].set(z[j] - mu[j] * c_vals[j + 1])
        b_vals = b_vals.at[j].set((y[j + 1] - y[j]) / h[j] - h[j] * (c_vals[j + 1] + 2.0 * c_vals[j]) / 3.0)
        d_vals = d_vals.at[j].set((c_vals[j + 1] - c_vals[j]) / (3.0 * h[j]))

    a_vals = y[:-1]
    c_vals = c_vals[:-1]

    return a_vals, b_vals, c_vals, d_vals


@jit
def cubic_spline_eval(
    x: jnp.ndarray,
    a: jnp.ndarray,
    b: jnp.ndarray,
    c: jnp.ndarray,
    d: jnp.ndarray,
    x_new: float,
) -> float:
    """Evaluate cubic spline at a point.

    Parameters
    ----------
    x : Array
        Knot points (x-coordinates)
    a, b, c, d : Arrays
        Spline coefficients
    x_new : float
        Point at which to evaluate

    Returns
    -------
    float
        Spline value at x_new

    Notes
    -----
    For x_new in [x[i], x[i+1]], evaluates:
        S_i(x_new) = a[i] + b[i]*dx + c[i]*dx^2 + d[i]*dx^3
    where dx = x_new - x[i]
    """
    # Handle extrapolation with flat extrapolation
    x_new = jnp.clip(x_new, x[0], x[-1])

    # Find interval
    i = jnp.searchsorted(x, x_new) - 1
    i = jnp.clip(i, 0, len(x) - 2)

    # Evaluate spline
    dx = x_new - x[i]
    return a[i] + b[i] * dx + c[i] * dx * dx + d[i] * dx * dx * dx


class CubicSpline:
    """Natural cubic spline interpolation.

    A cubic spline interpolator with natural boundary conditions
    (zero second derivative at endpoints).

    Parameters
    ----------
    x : Array
        Known x-coordinates (must be sorted)
    y : Array
        Known y-coordinates

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> x = jnp.array([0.0, 1.0, 2.0, 3.0])
    >>> y = jnp.array([0.0, 1.0, 0.5, 2.0])
    >>> spline = CubicSpline(x, y)
    >>> spline(1.5)  # Interpolate at x=1.5
    """

    def __init__(self, x: jnp.ndarray, y: jnp.ndarray):
        """Initialize cubic spline with knot points."""
        if len(x) != len(y):
            raise ValueError("x and y must have the same length")
        if len(x) < 2:
            raise ValueError("Need at least 2 points for interpolation")

        self.x = jnp.asarray(x)
        self.y = jnp.asarray(y)

        # Compute spline coefficients
        if len(x) == 2:
            # Linear interpolation for 2 points
            self.a = y[:-1]
            self.b = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
            self.c = jnp.zeros(1)
            self.d = jnp.zeros(1)
        else:
            self.a, self.b, self.c, self.d = natural_cubic_spline_coefficients(x, y)

    def __call__(self, x_new: float | jnp.ndarray) -> float | jnp.ndarray:
        """Evaluate spline at one or more points.

        Parameters
        ----------
        x_new : float or Array
            Point(s) at which to evaluate

        Returns
        -------
        float or Array
            Interpolated value(s)
        """
        # Handle scalar input
        if isinstance(x_new, (int, float)):
            if len(self.x) == 2:
                return float(linear_interpolation(self.x, self.y, float(x_new)))
            return float(cubic_spline_eval(self.x, self.a, self.b, self.c, self.d, float(x_new)))

        # Handle array input
        x_new = jnp.asarray(x_new)
        if len(self.x) == 2:
            return jnp.array([float(linear_interpolation(self.x, self.y, float(xi))) for xi in x_new])
        return jnp.array([float(cubic_spline_eval(self.x, self.a, self.b, self.c, self.d, float(xi))) for xi in x_new])

    def derivative(self, x_new: float) -> float:
        """Compute first derivative of spline at a point.

        Parameters
        ----------
        x_new : float
            Point at which to compute derivative

        Returns
        -------
        float
            First derivative at x_new

        Notes
        -----
        S'(x) = b + 2*c*dx + 3*d*dx^2
        """
        if len(self.x) == 2:
            return float(self.b[0])

        x_new = jnp.clip(x_new, self.x[0], self.x[-1])
        i = jnp.searchsorted(self.x, x_new) - 1
        i = jnp.clip(i, 0, len(self.x) - 2)

        dx = x_new - self.x[i]
        return float(self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx * dx)


class MonotoneCubicSpline:
    """Monotone-preserving cubic spline (Fritsch-Carlson method).

    A cubic Hermite spline that preserves monotonicity of the data.
    Useful for forward rate curves where monotonicity must be maintained.

    Parameters
    ----------
    x : Array
        Known x-coordinates (must be sorted)
    y : Array
        Known y-coordinates

    Notes
    -----
    Uses the Fritsch-Carlson algorithm to ensure the interpolant is
    monotone if the data is monotone. This is critical for forward
    rate curves and other financial applications where monotonicity
    violations can lead to arbitrage.

    References
    ----------
    Fritsch, F. N., and Carlson, R. E. (1980).
    "Monotone Piecewise Cubic Interpolation."
    SIAM Journal on Numerical Analysis, 17(2), 238-246.

    Examples
    --------
    >>> x = jnp.array([1.0, 2.0, 3.0, 5.0])
    >>> y = jnp.array([0.03, 0.035, 0.04, 0.045])  # Monotone increasing
    >>> spline = MonotoneCubicSpline(x, y)
    >>> spline(2.5)  # Guaranteed to preserve monotonicity
    """

    def __init__(self, x: jnp.ndarray, y: jnp.ndarray):
        """Initialize monotone cubic spline."""
        if len(x) != len(y):
            raise ValueError("x and y must have the same length")
        if len(x) < 2:
            raise ValueError("Need at least 2 points for interpolation")

        self.x = jnp.asarray(x)
        self.y = jnp.asarray(y)

        # Compute slopes
        n = len(x)
        h = jnp.diff(x)
        delta = jnp.diff(y) / h

        # Initialize tangents
        m = jnp.zeros(n)

        # Interior points: weighted harmonic mean
        for i in range(1, n - 1):
            if delta[i - 1] * delta[i] > 0:
                w1 = 2.0 * h[i] + h[i - 1]
                w2 = h[i] + 2.0 * h[i - 1]
                m = m.at[i].set((w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i]))
            else:
                m = m.at[i].set(0.0)

        # Boundary conditions (one-sided derivatives)
        m = m.at[0].set(delta[0])
        m = m.at[-1].set(delta[-1])

        # Apply monotonicity constraints
        for i in range(n - 1):
            if jnp.abs(delta[i]) < 1e-10:
                m = m.at[i].set(0.0)
                m = m.at[i + 1].set(0.0)
            else:
                alpha = m[i] / delta[i]
                beta = m[i + 1] / delta[i]
                tau = 3.0 / jnp.sqrt(alpha * alpha + beta * beta)

                if alpha * alpha + beta * beta > 9.0:
                    m = m.at[i].set(tau * alpha * delta[i])
                    m = m.at[i + 1].set(tau * beta * delta[i])

        self.m = m

    def __call__(self, x_new: float | jnp.ndarray) -> float | jnp.ndarray:
        """Evaluate monotone spline at point(s).

        Uses Hermite cubic interpolation with monotonicity-preserving tangents.
        """
        if isinstance(x_new, (int, float)):
            return float(self._eval_scalar(float(x_new)))

        x_new = jnp.asarray(x_new)
        return jnp.array([float(self._eval_scalar(float(xi))) for xi in x_new])

    def _eval_scalar(self, x_new: float) -> float:
        """Evaluate at a single point."""
        x_new = jnp.clip(x_new, self.x[0], self.x[-1])
        i = jnp.searchsorted(self.x, x_new) - 1
        i = jnp.clip(i, 0, len(self.x) - 2)

        h = self.x[i + 1] - self.x[i]
        t = (x_new - self.x[i]) / h

        # Hermite basis functions
        h00 = 2.0 * t * t * t - 3.0 * t * t + 1.0
        h10 = t * t * t - 2.0 * t * t + t
        h01 = -2.0 * t * t * t + 3.0 * t * t
        h11 = t * t * t - t * t

        return (
            h00 * self.y[i]
            + h10 * h * self.m[i]
            + h01 * self.y[i + 1]
            + h11 * h * self.m[i + 1]
        )


__all__ = [
    "CubicSpline",
    "MonotoneCubicSpline",
    "cubic_spline_eval",
    "linear_interpolation",
    "natural_cubic_spline_coefficients",
]
