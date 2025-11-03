"""Tests for interpolation methods."""

import jax.numpy as jnp
import pytest

from neutryx.market.interpolation import (
    CubicSpline,
    MonotoneCubicSpline,
    linear_interpolation,
    natural_cubic_spline_coefficients,
)


def test_linear_interpolation_basic():
    """Test basic linear interpolation."""
    x = jnp.array([0.0, 1.0, 2.0])
    y = jnp.array([0.0, 1.0, 4.0])

    # Midpoint interpolation
    result = linear_interpolation(x, y, 0.5)
    assert jnp.isclose(result, 0.5)

    result = linear_interpolation(x, y, 1.5)
    assert jnp.isclose(result, 2.5)


def test_linear_interpolation_extrapolation():
    """Test that linear interpolation uses flat extrapolation."""
    x = jnp.array([0.0, 1.0, 2.0])
    y = jnp.array([0.0, 1.0, 4.0])

    # Before first point
    result = linear_interpolation(x, y, -1.0)
    assert jnp.isclose(result, y[0])

    # After last point
    result = linear_interpolation(x, y, 3.0)
    assert jnp.isclose(result, y[-1])


def test_cubic_spline_basic():
    """Test basic cubic spline interpolation."""
    x = jnp.array([0.0, 1.0, 2.0, 3.0])
    y = jnp.array([0.0, 1.0, 0.5, 2.0])

    spline = CubicSpline(x, y)

    # Should pass through knot points
    for xi, yi in zip(x, y):
        assert jnp.isclose(spline(float(xi)), yi, atol=1e-6)


def test_cubic_spline_smoothness():
    """Test that cubic spline is smooth (continuous derivatives)."""
    x = jnp.array([0.0, 1.0, 2.0, 3.0])
    y = jnp.array([0.0, 1.0, 0.5, 2.0])

    spline = CubicSpline(x, y)

    # Check that derivative is continuous at knot points
    for xi in x[1:-1]:
        # Evaluate derivative just before and after knot
        eps = 1e-6
        d_before = spline.derivative(float(xi) - eps)
        d_after = spline.derivative(float(xi) + eps)

        # Should be close (continuous first derivative)
        assert jnp.isclose(d_before, d_after, atol=1e-3)


def test_cubic_spline_linear_function():
    """Test cubic spline on linear data."""
    x = jnp.array([0.0, 1.0, 2.0, 3.0])
    y = 2.0 * x + 1.0  # Linear function

    spline = CubicSpline(x, y)

    # Should reproduce linear function exactly
    test_points = jnp.array([0.5, 1.5, 2.5])
    for xp in test_points:
        expected = 2.0 * xp + 1.0
        assert jnp.isclose(spline(float(xp)), expected, atol=1e-6)


def test_cubic_spline_quadratic():
    """Test cubic spline on quadratic data."""
    x = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = x * x  # Quadratic function

    spline = CubicSpline(x, y)

    # Should interpolate quadratic function accurately
    test_points = jnp.array([0.5, 1.5, 2.5, 3.5])
    for xp in test_points:
        expected = xp * xp
        # Natural spline won't be exact for quadratic, but should be close
        assert jnp.abs(spline(float(xp)) - expected) < 0.2


def test_cubic_spline_two_points():
    """Test cubic spline with only two points (should be linear)."""
    x = jnp.array([0.0, 1.0])
    y = jnp.array([0.0, 2.0])

    spline = CubicSpline(x, y)

    # Should be linear interpolation
    assert jnp.isclose(spline(0.5), 1.0)


def test_cubic_spline_array_input():
    """Test cubic spline with array input."""
    x = jnp.array([0.0, 1.0, 2.0, 3.0])
    y = jnp.array([0.0, 1.0, 0.5, 2.0])

    spline = CubicSpline(x, y)

    # Evaluate at multiple points
    x_new = jnp.array([0.5, 1.5, 2.5])
    results = spline(x_new)

    assert len(results) == len(x_new)
    assert all(jnp.isfinite(r) for r in results)


def test_monotone_cubic_spline_monotonicity():
    """Test that monotone spline preserves monotonicity."""
    # Monotone increasing data
    x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = jnp.array([0.03, 0.035, 0.04, 0.042, 0.045])

    spline = MonotoneCubicSpline(x, y)

    # Check that interpolated values maintain monotonicity
    test_points = jnp.linspace(1.0, 5.0, 20)
    values = spline(test_points)

    # All differences should be non-negative (monotone increasing)
    diffs = jnp.diff(values)
    assert jnp.all(diffs >= -1e-10), "Monotonicity violated"


def test_monotone_cubic_spline_passes_through_points():
    """Test that monotone spline passes through data points."""
    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    y = jnp.array([0.03, 0.035, 0.04, 0.045])

    spline = MonotoneCubicSpline(x, y)

    # Should pass through all knot points
    for xi, yi in zip(x, y):
        assert jnp.isclose(spline(float(xi)), yi, atol=1e-6)


def test_monotone_vs_natural_spline():
    """Compare monotone and natural splines on monotone data."""
    # Monotone data
    x = jnp.array([0.0, 1.0, 2.0, 3.0])
    y = jnp.array([1.0, 2.0, 3.0, 4.0])  # Linear, so both should be similar

    natural = CubicSpline(x, y)
    monotone = MonotoneCubicSpline(x, y)

    # For linear data, both should give similar results
    test_point = 1.5
    nat_val = natural(test_point)
    mon_val = monotone(test_point)

    assert jnp.isclose(nat_val, mon_val, rtol=0.1)


def test_monotone_spline_flat_section():
    """Test monotone spline with flat section."""
    x = jnp.array([0.0, 1.0, 2.0, 3.0])
    y = jnp.array([1.0, 2.0, 2.0, 3.0])  # Flat between 1 and 2

    spline = MonotoneCubicSpline(x, y)

    # Should handle flat section without oscillation
    test_points = jnp.linspace(1.0, 2.0, 10)
    values = spline(test_points)

    # Values in flat region should be close to 2.0
    assert jnp.all(values >= 1.9)
    assert jnp.all(values <= 2.1)


def test_spline_validation():
    """Test that splines validate input."""
    # Mismatched lengths
    with pytest.raises(ValueError, match="same length"):
        CubicSpline(jnp.array([0.0, 1.0]), jnp.array([0.0]))

    # Too few points
    with pytest.raises(ValueError, match="at least 2 points"):
        CubicSpline(jnp.array([0.0]), jnp.array([0.0]))


def test_yield_curve_application():
    """Test cubic spline for yield curve interpolation."""
    # Typical yield curve data (maturities in years, yields as decimals)
    maturities = jnp.array([0.5, 1.0, 2.0, 5.0, 10.0])
    yields = jnp.array([0.02, 0.025, 0.03, 0.035, 0.04])

    spline = CubicSpline(maturities, yields)

    # Interpolate yield for 3-year maturity
    yield_3y = spline(3.0)

    # Should be between 2-year and 5-year yields
    assert yields[2] < yield_3y < yields[3]

    # Should be smooth
    assert jnp.isfinite(yield_3y)


def test_forward_rate_monotonicity():
    """Test monotone spline for forward rate curves."""
    # Forward rates should be monotone (or at least not wildly oscillating)
    maturities = jnp.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    forward_rates = jnp.array([0.025, 0.028, 0.030, 0.032, 0.033, 0.035])

    # Use monotone spline to preserve monotonicity
    spline = MonotoneCubicSpline(maturities, forward_rates)

    # Check interpolated values
    test_maturities = jnp.array([1.5, 2.5, 4.0, 6.0, 8.0])
    interpolated = spline(test_maturities)

    # Should all be within reasonable range
    assert jnp.all(interpolated >= 0.024)
    assert jnp.all(interpolated <= 0.036)

    # Should preserve monotonicity
    assert jnp.all(jnp.diff(interpolated) >= -1e-10)


def test_cubic_spline_derivative():
    """Test cubic spline derivative calculation."""
    x = jnp.array([0.0, 1.0, 2.0, 3.0])
    y = jnp.array([0.0, 1.0, 4.0, 9.0])

    spline = CubicSpline(x, y)

    # Derivative should be positive for increasing function
    for xi in [0.5, 1.5, 2.5]:
        deriv = spline.derivative(xi)
        assert deriv > 0


def test_natural_spline_coefficients():
    """Test that coefficient computation produces valid splines."""
    x = jnp.array([0.0, 1.0, 2.0, 3.0])
    y = jnp.array([0.0, 1.0, 0.0, 1.0])

    a, b, c, d = natural_cubic_spline_coefficients(x, y)

    # Should have n-1 coefficient sets for n points
    assert len(a) == len(x) - 1
    assert len(b) == len(x) - 1
    assert len(c) == len(x) - 1
    assert len(d) == len(x) - 1

    # Coefficients should be finite
    assert jnp.all(jnp.isfinite(a))
    assert jnp.all(jnp.isfinite(b))
    assert jnp.all(jnp.isfinite(c))
    assert jnp.all(jnp.isfinite(d))
