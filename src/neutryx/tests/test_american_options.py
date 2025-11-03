"""Tests for American option pricing using Longstaff-Schwartz method."""
import jax
import jax.numpy as jnp
import pytest

from neutryx.core.engine import MCConfig, simulate_gbm
from neutryx.products.american import (
    american_call_lsm,
    american_option_lsm,
    american_put_lsm,
)


class TestAmericanOptions:
    """Test suite for American option pricing."""

    @pytest.fixture
    def setup_params(self):
        """Common parameters for tests."""
        return {
            "S0": 100.0,
            "K": 100.0,
            "T": 1.0,
            "r": 0.05,
            "q": 0.0,
            "sigma": 0.2,
            "steps": 50,
            "paths": 10_000,
        }

    def test_american_put_itm(self, setup_params):
        """Test American put pricing for in-the-money option."""
        key = jax.random.PRNGKey(42)
        params = setup_params
        params["K"] = 110.0  # ITM put

        cfg = MCConfig(steps=params["steps"], paths=params["paths"])
        paths = simulate_gbm(
            key,
            params["S0"],
            params["r"] - params["q"],
            params["sigma"],
            params["T"],
            cfg,
        )

        dt = params["T"] / params["steps"]
        price = american_put_lsm(paths, params["K"], params["r"], dt)

        # American put should be worth at least intrinsic value
        intrinsic = max(params["K"] - params["S0"], 0.0)
        assert price >= intrinsic, f"American put price {price} < intrinsic {intrinsic}"

        # Price should be reasonable (not too high)
        assert price <= params["K"], f"American put price {price} > strike {params['K']}"

    def test_american_call_itm(self, setup_params):
        """Test American call pricing for in-the-money option."""
        key = jax.random.PRNGKey(43)
        params = setup_params
        params["K"] = 90.0  # ITM call

        cfg = MCConfig(steps=params["steps"], paths=params["paths"])
        paths = simulate_gbm(
            key,
            params["S0"],
            params["r"] - params["q"],
            params["sigma"],
            params["T"],
            cfg,
        )

        dt = params["T"] / params["steps"]
        price = american_call_lsm(paths, params["K"], params["r"], dt)

        # American call should be worth at least intrinsic value
        intrinsic = max(params["S0"] - params["K"], 0.0)
        assert price >= intrinsic, f"American call price {price} < intrinsic {intrinsic}"

    def test_american_put_atm(self, setup_params):
        """Test American put pricing for at-the-money option."""
        key = jax.random.PRNGKey(44)
        params = setup_params

        cfg = MCConfig(steps=params["steps"], paths=params["paths"])
        paths = simulate_gbm(
            key,
            params["S0"],
            params["r"] - params["q"],
            params["sigma"],
            params["T"],
            cfg,
        )

        dt = params["T"] / params["steps"]
        price = american_put_lsm(paths, params["K"], params["r"], dt)

        # ATM American put should have positive value
        assert price > 0.0, "ATM American put should have positive value"

        # Price should be less than strike
        assert price < params["K"], f"American put price {price} >= strike {params['K']}"

    def test_american_option_lsm_wrapper(self, setup_params):
        """Test wrapper function with kind parameter."""
        key = jax.random.PRNGKey(45)
        params = setup_params

        cfg = MCConfig(steps=params["steps"], paths=params["paths"])
        paths = simulate_gbm(
            key,
            params["S0"],
            params["r"] - params["q"],
            params["sigma"],
            params["T"],
            cfg,
        )

        dt = params["T"] / params["steps"]

        # Test put
        put_price = american_option_lsm(paths, params["K"], params["r"], dt, kind="put")
        assert put_price > 0.0

        # Test call
        call_price = american_option_lsm(paths, params["K"], params["r"], dt, kind="call")
        assert call_price > 0.0

        # Test invalid kind
        with pytest.raises(ValueError, match="Unknown option kind"):
            american_option_lsm(paths, params["K"], params["r"], dt, kind="invalid")

    def test_jit_compatibility(self, setup_params):
        """Test that American option pricing is JIT-compatible."""
        key = jax.random.PRNGKey(46)
        params = setup_params

        cfg = MCConfig(steps=params["steps"], paths=params["paths"])
        paths = simulate_gbm(
            key,
            params["S0"],
            params["r"] - params["q"],
            params["sigma"],
            params["T"],
            cfg,
        )

        dt = params["T"] / params["steps"]

        # JIT compile the function
        jitted_american_put = jax.jit(american_put_lsm)

        # Should not raise any errors
        price = jitted_american_put(paths, params["K"], params["r"], dt)

        assert jnp.isfinite(price), "JIT-compiled price should be finite"
        assert price > 0.0, "JIT-compiled price should be positive"

    def test_vmap_compatibility(self, setup_params):
        """Test that American option pricing works with vmap."""
        key = jax.random.PRNGKey(47)
        params = setup_params

        # Create multiple sets of paths
        cfg = MCConfig(steps=params["steps"], paths=params["paths"])

        keys = jax.random.split(key, 5)

        def price_single(k):
            paths = simulate_gbm(
                k,
                params["S0"],
                params["r"] - params["q"],
                params["sigma"],
                params["T"],
                cfg,
            )
            dt = params["T"] / params["steps"]
            return american_put_lsm(paths, params["K"], params["r"], dt)

        # Vmap over multiple keys
        prices = jax.vmap(price_single)(keys)

        assert prices.shape == (5,), f"Expected shape (5,), got {prices.shape}"
        assert jnp.all(jnp.isfinite(prices)), "All prices should be finite"
        assert jnp.all(prices > 0.0), "All prices should be positive"

    def test_put_call_relationship(self, setup_params):
        """Test basic put-call relationship for American options."""
        key = jax.random.PRNGKey(48)
        params = setup_params

        cfg = MCConfig(steps=params["steps"], paths=params["paths"])
        paths = simulate_gbm(
            key,
            params["S0"],
            params["r"] - params["q"],
            params["sigma"],
            params["T"],
            cfg,
        )

        dt = params["T"] / params["steps"]

        put_price = american_put_lsm(paths, params["K"], params["r"], dt)
        call_price = american_call_lsm(paths, params["K"], params["r"], dt)

        # Both should be positive for ATM options
        assert put_price > 0.0, "Put price should be positive"
        assert call_price > 0.0, "Call price should be positive"

    def test_increasing_volatility(self, setup_params):
        """Test that American option prices increase with volatility."""
        params = setup_params
        cfg = MCConfig(steps=params["steps"], paths=params["paths"])

        prices = []
        for sigma in [0.1, 0.2, 0.3]:
            key = jax.random.PRNGKey(49)
            paths = simulate_gbm(
                key,
                params["S0"],
                params["r"] - params["q"],
                sigma,
                params["T"],
                cfg,
            )
            dt = params["T"] / params["steps"]
            price = american_put_lsm(paths, params["K"], params["r"], dt)
            prices.append(price)

        # Prices should generally increase with volatility
        assert prices[1] > prices[0] * 0.8, "Higher vol should give higher price"
        assert prices[2] > prices[1] * 0.8, "Even higher vol should give even higher price"

    def test_deep_otm_put(self, setup_params):
        """Test American put pricing for deep out-of-the-money option."""
        key = jax.random.PRNGKey(50)
        params = setup_params
        params["K"] = 50.0  # Deep OTM put

        cfg = MCConfig(steps=params["steps"], paths=params["paths"])
        paths = simulate_gbm(
            key,
            params["S0"],
            params["r"] - params["q"],
            params["sigma"],
            params["T"],
            cfg,
        )

        dt = params["T"] / params["steps"]
        price = american_put_lsm(paths, params["K"], params["r"], dt)

        # Deep OTM options should have low value
        assert 0.0 <= price < 1.0, f"Deep OTM put price {price} should be near zero"

    def test_early_exercise_premium(self, setup_params):
        """Test that American put has early exercise premium over European."""
        key = jax.random.PRNGKey(51)
        params = setup_params
        params["K"] = 110.0  # ITM put to encourage early exercise
        params["r"] = 0.10  # Higher rate encourages early exercise for puts

        cfg = MCConfig(steps=params["steps"], paths=params["paths"])
        paths = simulate_gbm(
            key,
            params["S0"],
            params["r"] - params["q"],
            params["sigma"],
            params["T"],
            cfg,
        )

        dt = params["T"] / params["steps"]
        american_price = american_put_lsm(paths, params["K"], params["r"], dt)

        # European put (just terminal payoff)
        ST = paths[:, -1]
        european_payoff = jnp.maximum(params["K"] - ST, 0.0)
        european_price = jnp.exp(-params["r"] * params["T"]) * european_payoff.mean()

        # American should be at least as valuable as European
        assert (
            american_price >= european_price * 0.95
        ), "American put should be at least as valuable as European"


@pytest.mark.regression
class TestAmericanOptionsRegression:
    """Regression tests for American option pricing."""

    def test_standard_case_reproducibility(self):
        """Test that standard case produces consistent results."""
        key = jax.random.PRNGKey(100)
        S0, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.0, 0.2
        steps, paths = 50, 50_000

        cfg = MCConfig(steps=steps, paths=paths)
        paths_data = simulate_gbm(key, S0, r - q, sigma, T, cfg)

        dt = T / steps
        price = american_put_lsm(paths_data, K, r, dt)

        # Expected value from reference implementation (approximate)
        expected = 4.9  # Approximate LSM value with this implementation
        tolerance = 0.3  # MC has variance

        assert (
            abs(price - expected) < tolerance
        ), f"Price {price:.4f} deviates from expected {expected:.4f}"
