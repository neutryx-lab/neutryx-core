import pathlib
import sys

import jax.numpy as jnp

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tests.market.sample_data import (
    DEPOSIT_FIXTURES,
    EXPECTED_DISCOUNT_FACTORS,
    EXPECTED_ZERO_RATES,
    FRA_FIXTURES,
    FUTURE_FIXTURES,
    SWAP_FIXTURES,
)

from neutryx.market.curves import BootstrappedCurve, Deposit, FixedRateSwap, FRA, Future


def build_curve():
    deposits = [Deposit(**fixture) for fixture in DEPOSIT_FIXTURES]
    swaps = [FixedRateSwap(**fixture) for fixture in SWAP_FIXTURES]
    return BootstrappedCurve([*deposits, *swaps])


def test_bootstrap_discount_factors():
    curve = build_curve()
    # Only test deposit and swap maturities (not FRA/Future which are not in build_curve)
    for maturity in [0.5, 1.0]:
        df = curve.df(maturity)
        expected = EXPECTED_DISCOUNT_FACTORS[maturity]
        assert jnp.isclose(df, expected, rtol=1e-8)


def test_zero_rate_and_forward():
    curve = build_curve()
    zero_rate = curve.zero_rate(1.0)
    assert jnp.isclose(zero_rate, EXPECTED_ZERO_RATES[1.0], rtol=1e-6)

    forward = curve.forward_rate(0.5, 1.0)
    expected_forward = (
        EXPECTED_DISCOUNT_FACTORS[0.5] / EXPECTED_DISCOUNT_FACTORS[1.0] - 1.0
    ) / 0.5
    assert jnp.isclose(forward, expected_forward, rtol=1e-8)


def test_log_linear_interpolation():
    curve = build_curve()
    interpolated = curve.df(0.75)
    expected = jnp.exp(
        jnp.interp(
            0.75,
            jnp.array(list(EXPECTED_DISCOUNT_FACTORS.keys())),
            jnp.log(jnp.array(list(EXPECTED_DISCOUNT_FACTORS.values()))),
        )
    )
    assert jnp.isclose(interpolated, expected, rtol=1e-6)


def test_fra_bootstrap():
    """Test bootstrapping with FRA instruments."""
    deposits = [Deposit(**fixture) for fixture in DEPOSIT_FIXTURES]
    swaps = [FixedRateSwap(**fixture) for fixture in SWAP_FIXTURES]
    fras = [FRA(**fixture) for fixture in FRA_FIXTURES]

    curve = BootstrappedCurve([*deposits, *swaps, *fras])

    # Check that FRA maturity is correctly bootstrapped
    df_1_5 = curve.df(1.5)
    expected_df_1_5 = EXPECTED_DISCOUNT_FACTORS[1.5]
    assert jnp.isclose(df_1_5, expected_df_1_5, rtol=1e-8)

    # Verify the FRA formula: DF(start) / DF(end) = 1 + rate * accrual
    fra = fras[0]
    df_start = curve.df(fra.start)
    df_end = curve.df(fra.end)
    implied_rate = (df_start / df_end - 1.0) / (fra.end - fra.start)
    assert jnp.isclose(implied_rate, fra.rate, rtol=1e-8)


def test_future_bootstrap():
    """Test bootstrapping with Future instruments."""
    deposits = [Deposit(**fixture) for fixture in DEPOSIT_FIXTURES]
    swaps = [FixedRateSwap(**fixture) for fixture in SWAP_FIXTURES]
    fras = [FRA(**fixture) for fixture in FRA_FIXTURES]
    futures = [Future(**fixture) for fixture in FUTURE_FIXTURES]

    curve = BootstrappedCurve([*deposits, *swaps, *fras, *futures])

    # Check that Future maturity is correctly bootstrapped
    df_2_0 = curve.df(2.0)
    expected_df_2_0 = EXPECTED_DISCOUNT_FACTORS[2.0]
    assert jnp.isclose(df_2_0, expected_df_2_0, rtol=1e-8)

    # Verify the Future formula with convexity adjustment
    future = futures[0]
    df_start = curve.df(future.start)
    df_end = curve.df(future.end)
    implied_rate_from_curve = (df_start / df_end - 1.0) / (future.end - future.start)

    # Back out the rate from future price
    implied_rate_from_price = (100.0 - future.price) / 100.0
    forward_rate = implied_rate_from_price - future.convexity_adjustment

    assert jnp.isclose(implied_rate_from_curve, forward_rate, rtol=1e-8)


def test_complete_curve():
    """Test a complete curve with Deposits, FRAs, Futures, and Swaps."""
    deposits = [Deposit(**fixture) for fixture in DEPOSIT_FIXTURES]
    swaps = [FixedRateSwap(**fixture) for fixture in SWAP_FIXTURES]
    fras = [FRA(**fixture) for fixture in FRA_FIXTURES]
    futures = [Future(**fixture) for fixture in FUTURE_FIXTURES]

    curve = BootstrappedCurve([*deposits, *swaps, *fras, *futures])

    # Test all expected discount factors
    for maturity, expected_df in EXPECTED_DISCOUNT_FACTORS.items():
        df = curve.df(maturity)
        assert jnp.isclose(df, expected_df, rtol=1e-8), f"Failed at maturity {maturity}"

    # Test zero rates
    for maturity, expected_zr in EXPECTED_ZERO_RATES.items():
        zr = curve.zero_rate(maturity)
        assert jnp.isclose(zr, expected_zr, rtol=1e-6), f"Failed at maturity {maturity}"
