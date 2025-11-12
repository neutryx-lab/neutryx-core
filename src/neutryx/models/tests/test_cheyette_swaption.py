import sys
from pathlib import Path

import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neutryx.models.cheyette import CheyetteParams, swaption_price_analytical


def test_multi_factor_monte_carlo_matches_single_factor_price():
    """Monte Carlo multi-factor price should align with single-factor analytic price."""

    forward_curve = lambda t: 0.02

    params_single = CheyetteParams(
        kappa=0.2,
        sigma_fn=lambda t: 0.0,
        forward_curve_fn=forward_curve,
        r0=0.02,
        n_factors=1,
    )

    def sigma_multi(_t):
        return jnp.array([0.0, 0.0])

    params_multi = CheyetteParams(
        kappa=jnp.array([0.2, 0.2]),
        sigma_fn=sigma_multi,
        forward_curve_fn=forward_curve,
        r0=0.02,
        n_factors=2,
        rho=jnp.eye(2),
    )

    strike = 0.02
    option_expiry = 1.0
    swap_tenor = 2.0
    payment_frequency = 0.5

    price_single = swaption_price_analytical(
        params_single,
        strike,
        option_expiry,
        swap_tenor,
        payment_frequency=payment_frequency,
    )

    price_multi, std_err = swaption_price_analytical(
        params_multi,
        strike,
        option_expiry,
        swap_tenor,
        payment_frequency=payment_frequency,
        mc_paths=1000,
        mc_steps=40,
    )

    assert std_err < 1e-8
    assert abs(price_multi - price_single) < 1e-6
