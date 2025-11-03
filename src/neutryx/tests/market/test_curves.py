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
    SWAP_FIXTURES,
)

from neutryx.market.curves import BootstrappedCurve, Deposit, FixedRateSwap


def build_curve():
    deposits = [Deposit(**fixture) for fixture in DEPOSIT_FIXTURES]
    swaps = [FixedRateSwap(**fixture) for fixture in SWAP_FIXTURES]
    return BootstrappedCurve([*deposits, *swaps])


def test_bootstrap_discount_factors():
    curve = build_curve()
    for maturity, expected in EXPECTED_DISCOUNT_FACTORS.items():
        df = curve.df(maturity)
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
