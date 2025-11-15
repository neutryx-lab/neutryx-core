import pathlib
import sys

import jax.numpy as jnp

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tests.market.sample_data import (
    IMPLIED_VOL_SURFACE,
    SABR_SURFACE_DATA,
    SABR_VALIDATION_POINT,
)

from neutryx.market.vol import (
    ImpliedVolSurface,
    SABRParameters,
    SABRSurface,
    sabr_implied_vol,
)


def test_sabr_implied_volatility_matches_reference():
    params = SABRParameters(
        alpha=SABR_SURFACE_DATA["alphas"][0],
        beta=SABR_SURFACE_DATA["betas"][0],
        rho=SABR_SURFACE_DATA["rhos"][0],
        nu=SABR_SURFACE_DATA["nus"][0],
    )
    vol = sabr_implied_vol(
        forward=SABR_SURFACE_DATA["forwards"][0],
        strike=SABR_VALIDATION_POINT["strike"],
        maturity=SABR_SURFACE_DATA["expiries"][0],
        params=params,
    )
    assert jnp.isclose(vol, 0.24161040376433723, rtol=1e-6)


def test_sabr_surface_interpolation():
    surface = SABRSurface(
        expiries=SABR_SURFACE_DATA["expiries"],
        forwards=SABR_SURFACE_DATA["forwards"],
        params=[
            SABRParameters(alpha=a, beta=b, rho=r, nu=n)
            for a, b, r, n in zip(
                SABR_SURFACE_DATA["alphas"],
                SABR_SURFACE_DATA["betas"],
                SABR_SURFACE_DATA["rhos"],
                SABR_SURFACE_DATA["nus"],
            )
        ],
    )
    vol = surface.implied_vol(
        SABR_VALIDATION_POINT["expiry"],
        SABR_VALIDATION_POINT["strike"],
    )
    assert jnp.isclose(vol, SABR_VALIDATION_POINT["expected_vol"], rtol=1e-6)


def test_implied_vol_surface_interpolator():
    surface = ImpliedVolSurface(
        expiries=IMPLIED_VOL_SURFACE["expiries"],
        strikes=IMPLIED_VOL_SURFACE["strikes"],
        vols=IMPLIED_VOL_SURFACE["vols"],
    )
    validation = IMPLIED_VOL_SURFACE["validation"]
    vol = surface.implied_vol(validation["expiry"], validation["strike"])
    assert jnp.isclose(vol, validation["expected"], rtol=1e-6)
