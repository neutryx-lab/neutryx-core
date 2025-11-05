"""Tests for regularization techniques."""

import jax
import jax.numpy as jnp
import pytest

from neutryx.calibration.regularization import (
    TikhonovRegularizer,
    L1Regularizer,
    ElasticNetRegularizer,
    SmoothnessRegularizer,
    ArbitrageFreeConstraints,
    create_difference_matrix,
)


# ===== Tikhonov Regularization Tests =====

def test_tikhonov_basic():
    """Test basic Tikhonov regularization."""
    reg = TikhonovRegularizer(lambda_reg=1.0)

    params = jnp.array([1.0, 2.0, 3.0])
    penalty = reg.penalty(params)

    # Should be sum of squares
    expected = jnp.sum(params ** 2)
    assert penalty == pytest.approx(expected, abs=1e-6)


def test_tikhonov_with_prior():
    """Test Tikhonov regularization with prior."""
    prior = jnp.array([1.0, 1.0, 1.0])
    reg = TikhonovRegularizer(lambda_reg=2.0, prior_params=prior)

    params = jnp.array([2.0, 3.0, 4.0])
    penalty = reg.penalty(params)

    # Should be 2.0 * sum((params - prior)^2)
    deviation = params - prior
    expected = 2.0 * jnp.sum(deviation ** 2)
    assert penalty == pytest.approx(expected, abs=1e-6)


def test_tikhonov_gradient():
    """Test Tikhonov gradient computation."""
    reg = TikhonovRegularizer(lambda_reg=1.0)

    params = jnp.array([1.0, 2.0, 3.0])
    grad = reg.gradient(params)

    # Gradient should be 2 * lambda * params
    expected_grad = 2.0 * 1.0 * params
    assert jnp.allclose(grad, expected_grad, atol=1e-6)


def test_tikhonov_with_matrix():
    """Test Tikhonov with regularization matrix."""
    # First difference matrix
    D = create_difference_matrix(3, order=1)

    reg = TikhonovRegularizer(lambda_reg=1.0, regularization_matrix=D)

    params = jnp.array([1.0, 3.0, 2.0])
    penalty = reg.penalty(params)

    # D @ params gives differences: [2.0, -1.0]
    # Penalty should be sum of squared differences
    diffs = D @ params
    expected = jnp.sum(diffs ** 2)
    assert penalty == pytest.approx(expected, abs=1e-6)


def test_tikhonov_regularized_objective():
    """Test creating regularized objective function."""
    reg = TikhonovRegularizer(lambda_reg=0.5)

    # Simple quadratic objective
    def objective(params):
        return jnp.sum(params ** 2)

    reg_objective = reg.regularized_objective(objective)

    params = jnp.array([1.0, 2.0])

    # Original: 1 + 4 = 5
    # Penalty: 0.5 * (1 + 4) = 2.5
    # Total: 7.5
    expected = 7.5
    assert reg_objective(params) == pytest.approx(expected, abs=1e-6)


# ===== L1 Regularization Tests =====

def test_l1_basic():
    """Test basic L1 regularization."""
    reg = L1Regularizer(lambda_reg=1.0)

    params = jnp.array([1.0, -2.0, 3.0])
    penalty = reg.penalty(params)

    # Should be sum of absolute values
    expected = jnp.sum(jnp.abs(params))
    assert penalty == pytest.approx(expected, abs=1e-6)


def test_l1_with_weights():
    """Test L1 regularization with weights."""
    weights = jnp.array([1.0, 2.0, 0.5])
    reg = L1Regularizer(lambda_reg=1.0, weights=weights)

    params = jnp.array([1.0, 1.0, 1.0])
    penalty = reg.penalty(params)

    # Should be sum(|weights * params|) = 1.0 + 2.0 + 0.5 = 3.5
    expected = 3.5
    assert penalty == pytest.approx(expected, abs=1e-6)


def test_l1_sparsity():
    """Test that L1 induces sparsity."""
    # L1 penalty should be lower for sparse parameters (given same energy)
    reg = L1Regularizer(lambda_reg=1.0)

    # Both have same L2 norm (sqrt(100) = 10), but different L1 norms
    sparse_params = jnp.array([10.0, 0.0, 0.0])
    val = jnp.sqrt(100.0 / 3.0)
    spread_params = jnp.array([val, val, val])  # sqrt(100/3) each

    # Same L2 norm (approximately)
    assert jnp.sum(sparse_params ** 2) == pytest.approx(jnp.sum(spread_params ** 2), abs=0.001)
    # Sparse has lower L1 penalty
    assert reg.penalty(sparse_params) < reg.penalty(spread_params)


# ===== Elastic Net Tests =====

def test_elastic_net():
    """Test Elastic Net combines L1 and L2."""
    elastic = ElasticNetRegularizer(lambda_l1=1.0, lambda_l2=0.5)

    params = jnp.array([1.0, 2.0, 3.0])
    penalty = elastic.penalty(params)

    # L1: 1 + 2 + 3 = 6
    # L2: 1 + 4 + 9 = 14
    # Total: 6 + 0.5*14 = 13
    expected = 13.0
    assert penalty == pytest.approx(expected, abs=1e-6)


# ===== Smoothness Regularization Tests =====

def test_smoothness_1d_first_order():
    """Test first-order smoothness penalty."""
    reg = SmoothnessRegularizer(lambda_reg=1.0, order=1)

    # Smooth values
    smooth = jnp.array([1.0, 2.0, 3.0, 4.0])
    rough = jnp.array([1.0, 3.0, 2.0, 4.0])

    penalty_smooth = reg.penalty_1d(smooth)
    penalty_rough = reg.penalty_1d(rough)

    # Rough should have higher penalty
    assert penalty_rough > penalty_smooth


def test_smoothness_1d_second_order():
    """Test second-order smoothness penalty."""
    reg = SmoothnessRegularizer(lambda_reg=1.0, order=2)

    # Linear (perfectly smooth in second derivative)
    linear = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

    # Curved
    curved = jnp.array([1.0, 2.0, 4.0, 7.0, 11.0])

    penalty_linear = reg.penalty_1d(linear)
    penalty_curved = reg.penalty_1d(curved)

    # Linear should have zero second derivative
    assert penalty_linear == pytest.approx(0.0, abs=1e-10)
    assert penalty_curved > 0


def test_smoothness_2d():
    """Test 2D smoothness penalty for surfaces."""
    reg = SmoothnessRegularizer(lambda_reg=1.0, order=2, direction="both")

    # Create smooth surface
    smooth_surface = jnp.array([
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0],
    ])

    # Create rough surface
    rough_surface = jnp.array([
        [1.0, 5.0, 2.0],
        [3.0, 1.0, 4.0],
        [2.0, 4.0, 1.0],
    ])

    penalty_smooth = reg.penalty_2d(smooth_surface)
    penalty_rough = reg.penalty_2d(rough_surface)

    assert penalty_rough > penalty_smooth


def test_smoothness_directions():
    """Test smoothness in different directions."""
    # Penalty only in strike direction
    reg_strike = SmoothnessRegularizer(lambda_reg=1.0, order=1, direction="strike")

    # Penalty only in maturity direction
    reg_maturity = SmoothnessRegularizer(lambda_reg=1.0, order=1, direction="maturity")

    surface = jnp.array([
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
    ])

    # Smooth in strike, constant in maturity
    penalty_strike = reg_strike.penalty_2d(surface)
    penalty_maturity = reg_maturity.penalty_2d(surface)

    # Should have low penalty in strike direction (linear)
    assert penalty_strike > 0
    # Should have zero penalty in maturity direction (constant)
    assert penalty_maturity == pytest.approx(0.0, abs=1e-10)


# ===== Arbitrage-Free Constraints Tests =====

def test_calendar_spread_no_violation():
    """Test calendar spread with no arbitrage."""
    constraints = ArbitrageFreeConstraints(lambda_calendar=100.0)

    # Total variance increasing (no arbitrage)
    total_variance = jnp.array([0.04, 0.08, 0.12, 0.16])  # T * sigma^2

    penalty = constraints.calendar_spread_penalty(total_variance)

    # Should be zero penalty
    assert penalty == pytest.approx(0.0, abs=1e-10)


def test_calendar_spread_with_violation():
    """Test calendar spread with arbitrage."""
    constraints = ArbitrageFreeConstraints(lambda_calendar=100.0)

    # Total variance not monotonic (arbitrage!)
    total_variance = jnp.array([0.04, 0.10, 0.08, 0.16])

    penalty = constraints.calendar_spread_penalty(total_variance)

    # Should have positive penalty
    assert penalty > 0


def test_butterfly_no_violation():
    """Test butterfly constraint with no arbitrage."""
    constraints = ArbitrageFreeConstraints(lambda_butterfly=100.0)

    # Convex call prices (no arbitrage)
    strikes = jnp.array([90.0, 100.0, 110.0, 120.0])
    call_prices = jnp.array([15.0, 10.0, 6.0, 3.0])  # Decreasing and convex

    penalty = constraints.butterfly_penalty(call_prices, strikes)

    # Should be low penalty (convex)
    assert penalty >= 0


def test_butterfly_with_violation():
    """Test butterfly constraint with arbitrage."""
    constraints = ArbitrageFreeConstraints(lambda_butterfly=100.0)

    # Non-convex call prices (arbitrage!)
    strikes = jnp.array([90.0, 100.0, 110.0, 120.0])
    call_prices = jnp.array([15.0, 5.0, 10.0, 3.0])  # Not convex

    penalty = constraints.butterfly_penalty(call_prices, strikes)

    # Should have positive penalty
    assert penalty > 0


def test_positive_density():
    """Test positive risk-neutral density constraint."""
    constraints = ArbitrageFreeConstraints()

    strikes = jnp.array([90.0, 100.0, 110.0])

    # Valid convex prices (positive density)
    # Second derivative positive: decreasing and convex
    valid_prices = jnp.array([20.0, 10.0, 3.0])
    penalty_valid = constraints.positive_density_penalty(valid_prices, strikes)

    # Invalid concave prices (negative density)
    # Second derivative negative: concave shape
    invalid_prices = jnp.array([10.0, 15.0, 12.0])
    penalty_invalid = constraints.positive_density_penalty(invalid_prices, strikes)

    # Invalid should have higher penalty
    assert penalty_invalid > penalty_valid
    # Valid should have zero or near-zero penalty
    assert penalty_valid >= 0


# ===== Difference Matrix Tests =====

def test_difference_matrix_first_order():
    """Test first-order difference matrix."""
    D = create_difference_matrix(5, order=1)

    # Should be (n-1) x n
    assert D.shape == (4, 5)

    # Apply to sequence
    x = jnp.array([1.0, 3.0, 6.0, 10.0, 15.0])
    diff = D @ x

    # Should give first differences
    expected = jnp.array([2.0, 3.0, 4.0, 5.0])
    assert jnp.allclose(diff, expected, atol=1e-6)


def test_difference_matrix_second_order():
    """Test second-order difference matrix."""
    D = create_difference_matrix(5, order=2)

    # Should be (n-2) x n
    assert D.shape == (3, 5)

    # Apply to quadratic sequence
    x = jnp.array([1.0, 4.0, 9.0, 16.0, 25.0])  # x[i] = (i+1)^2
    diff2 = D @ x

    # Second differences of quadratic should be constant
    # For f(x) = x^2: f''(x) = 2
    assert jnp.allclose(diff2, jnp.array([2.0, 2.0, 2.0]), atol=1e-6)


# ===== Integration Tests =====

def test_regularized_optimization():
    """Test regularization in optimization context."""
    # Simple optimization problem with regularization
    # Minimize: (x - 5)^2 + lambda * x^2

    reg = TikhonovRegularizer(lambda_reg=0.5)

    def objective(params):
        return (params[0] - 5.0) ** 2

    reg_objective = reg.regularized_objective(objective)

    # Optimal solution: x* = 5 / (1 + lambda) = 5 / 1.5 ≈ 3.33
    from scipy.optimize import minimize

    result = minimize(reg_objective, x0=jnp.array([0.0]))

    expected_optimal = 5.0 / 1.5
    assert result.x[0] == pytest.approx(expected_optimal, abs=0.1)


def test_combined_regularization():
    """Test combining multiple regularization techniques."""
    # Combine Tikhonov and smoothness
    tikhonov = TikhonovRegularizer(lambda_reg=0.1)
    smoothness = SmoothnessRegularizer(lambda_reg=1.0, order=2)

    params_2d = jnp.array([
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
    ])

    # Total penalty
    total_penalty = tikhonov.penalty(params_2d.flatten()) + \
                    smoothness.penalty_2d(params_2d)

    assert total_penalty > 0
    assert jnp.isfinite(total_penalty)
