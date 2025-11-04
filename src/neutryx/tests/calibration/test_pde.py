"""Tests for PDE solvers."""

import jax.numpy as jnp

from neutryx.models.bs import price as bs_price
from neutryx.models.pde import (
    PDEGrid,
    price_american_put_pde,
    price_european_option_pde,
)


def test_pde_grid_properties():
    """Test PDEGrid properties are correctly computed."""
    grid = PDEGrid(S_min=0.0, S_max=200.0, T=1.0, N_S=201, N_T=100)

    assert grid.dS == 1.0, f"Expected dS=1.0, got {grid.dS}"
    assert grid.dt == 0.01, f"Expected dt=0.01, got {grid.dt}"
    assert len(grid.S_grid) == 201
    assert grid.S_grid[0] == 0.0
    assert grid.S_grid[-1] == 200.0


def test_european_call_pde_vs_bs():
    """Test PDE solver for European call matches Black-Scholes."""
    S0, K, T = 100.0, 100.0, 1.0
    r, q, sigma = 0.05, 0.02, 0.2

    # Price using PDE
    pde_price = price_european_option_pde(
        S0, K, T, r, q, sigma,
        option_type="call",
        N_S=300,  # Fine grid for accuracy
        N_T=200
    )

    # Price using Black-Scholes
    bs_call = bs_price(S0, K, T, r, q, sigma, kind="call")

    # Should match within 0.5% (PDE has discretization error)
    rel_error = abs(pde_price - bs_call) / bs_call
    assert rel_error < 0.005, f"PDE price {pde_price:.4f} differs from BS {bs_call:.4f} by {rel_error:.2%}"


def test_european_put_pde_vs_bs():
    """Test PDE solver for European put matches Black-Scholes."""
    S0, K, T = 100.0, 100.0, 1.0
    r, q, sigma = 0.05, 0.02, 0.2

    # Price using PDE
    pde_price = price_european_option_pde(
        S0, K, T, r, q, sigma,
        option_type="put",
        N_S=300,
        N_T=200
    )

    # Price using Black-Scholes
    bs_put = bs_price(S0, K, T, r, q, sigma, kind="put")

    # Should match within 0.5%
    rel_error = abs(pde_price - bs_put) / bs_put
    assert rel_error < 0.005, f"PDE price {pde_price:.4f} differs from BS {bs_put:.4f} by {rel_error:.2%}"


def test_european_put_call_parity_pde():
    """Test put-call parity for PDE-priced European options."""
    S0, K, T = 100.0, 100.0, 1.0
    r, q, sigma = 0.05, 0.02, 0.2

    call_pde = price_european_option_pde(S0, K, T, r, q, sigma, "call", N_S=300, N_T=200)
    put_pde = price_european_option_pde(S0, K, T, r, q, sigma, "put", N_S=300, N_T=200)

    # Put-call parity: C - P = S0*exp(-qT) - K*exp(-rT)
    parity_lhs = call_pde - put_pde
    parity_rhs = S0 * jnp.exp(-q * T) - K * jnp.exp(-r * T)

    assert abs(parity_lhs - parity_rhs) < 0.1, \
        f"Put-call parity violated: {parity_lhs:.4f} != {parity_rhs:.4f}"


def test_american_put_pde_vs_european():
    """Test American put is worth more than European put (when ITM)."""
    S0, K, T = 90.0, 100.0, 1.0  # ITM put
    r, q, sigma = 0.05, 0.02, 0.2

    # American put
    american_price = price_american_put_pde(S0, K, T, r, q, sigma, N_S=300, N_T=200)

    # European put
    european_price = price_european_option_pde(S0, K, T, r, q, sigma, "put", N_S=300, N_T=200)

    # American should be >= European
    assert american_price >= european_price - 0.01, \
        f"American put {american_price:.4f} should be >= European put {european_price:.4f}"

    # Should have meaningful early exercise premium for ITM option
    premium = american_price - european_price
    assert premium > 0.1, f"Early exercise premium too small: {premium:.4f}"


def test_american_put_vs_intrinsic():
    """Test American put is at least intrinsic value."""
    S0, K, T = 80.0, 100.0, 1.0  # Deep ITM
    r, q, sigma = 0.05, 0.02, 0.2

    intrinsic = max(K - S0, 0.0)
    american_price = price_american_put_pde(S0, K, T, r, q, sigma, N_S=200, N_T=100)

    assert american_price >= intrinsic - 0.01, \
        f"American put {american_price:.4f} should be >= intrinsic {intrinsic:.4f}"


def test_pde_convergence():
    """Test PDE solution converges to BS with grid refinement."""
    S0, K, T = 100.0, 100.0, 1.0
    r, q, sigma = 0.05, 0.02, 0.2

    bs_call = bs_price(S0, K, T, r, q, sigma, kind="call")

    # Coarse grid
    price_coarse = price_european_option_pde(S0, K, T, r, q, sigma, "call", N_S=100, N_T=50)
    error_coarse = abs(price_coarse - bs_call)

    # Fine grid
    price_fine = price_european_option_pde(S0, K, T, r, q, sigma, "call", N_S=300, N_T=200)
    error_fine = abs(price_fine - bs_call)

    # Fine grid should have smaller error
    assert error_fine < error_coarse, \
        f"Fine grid error {error_fine:.6f} should be < coarse grid error {error_coarse:.6f}"


def test_pde_otm_call():
    """Test PDE pricing for OTM call option."""
    S0, K, T = 100.0, 120.0, 1.0  # OTM
    r, q, sigma = 0.05, 0.02, 0.2

    pde_price = price_european_option_pde(S0, K, T, r, q, sigma, "call", N_S=200, N_T=100)
    bs_call = bs_price(S0, K, T, r, q, sigma, kind="call")

    # Should be close
    rel_error = abs(pde_price - bs_call) / bs_call
    assert rel_error < 0.01, f"OTM call PDE price off by {rel_error:.2%}"


def test_pde_itm_put():
    """Test PDE pricing for ITM put option."""
    S0, K, T = 80.0, 100.0, 1.0  # ITM
    r, q, sigma = 0.05, 0.02, 0.2

    pde_price = price_european_option_pde(S0, K, T, r, q, sigma, "put", N_S=200, N_T=100)
    bs_put = bs_price(S0, K, T, r, q, sigma, kind="put")

    # Should be close
    rel_error = abs(pde_price - bs_put) / bs_put
    assert rel_error < 0.01, f"ITM put PDE price off by {rel_error:.2%}"


if __name__ == "__main__":
    # Run all tests
    test_pde_grid_properties()
    test_european_call_pde_vs_bs()
    test_european_put_pde_vs_bs()
    test_european_put_call_parity_pde()
    test_american_put_pde_vs_european()
    test_american_put_vs_intrinsic()
    test_pde_convergence()
    test_pde_otm_call()
    test_pde_itm_put()
    print("All PDE tests passed!")
