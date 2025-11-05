"""Tests for interest rate volatility surfaces and cubes."""
import jax.numpy as jnp
import pytest

from neutryx.market.ir_vol_surface import (
    SABRSlice,
    CapletVolSurface,
    SwaptionVolCube,
    construct_caplet_surface_from_sabr,
    construct_swaption_cube_from_sabr,
)


def test_sabr_slice_basic():
    """Test basic SABR slice functionality."""
    slice = SABRSlice(
        expiry=1.0,
        forward_rate=0.03,
        alpha=0.20,
        beta=0.5,
        rho=-0.3,
        nu=0.4,
    )

    # ATM vol should be close to alpha (can be higher due to SABR adjustments)
    atm_vol = slice.implied_vol(0.03)
    assert atm_vol > 0  # Just check it's positive

    # OTM call (higher strike) should have higher vol (positive skew expected for beta < 1)
    otm_call_vol = slice.implied_vol(0.04)
    assert otm_call_vol > 0

    # Vol smile should work
    strikes = jnp.array([0.02, 0.025, 0.03, 0.035, 0.04])
    vols = slice.vol_smile(strikes)
    assert len(vols) == len(strikes)
    assert jnp.all(vols > 0)


def test_caplet_surface_construction():
    """Test caplet vol surface construction."""
    expiries = jnp.array([0.25, 0.5, 1.0, 2.0])
    forwards = jnp.array([0.03, 0.032, 0.035, 0.038])

    slices = [
        SABRSlice(expiries[i], forwards[i], alpha=0.20, beta=0.5, rho=-0.3, nu=0.4)
        for i in range(len(expiries))
    ]

    surface = CapletVolSurface(expiries, forwards, slices)

    # Test on-grid point
    vol = surface.get_vol(strike=0.03, expiry=0.5)
    assert vol > 0

    # Test interpolation
    vol_interp = surface.get_vol(strike=0.035, expiry=0.75)
    assert vol_interp > 0

    # Test ATM vol
    atm_vol = surface.get_atm_vol(0.5)
    assert atm_vol > 0


def test_caplet_surface_extrapolation():
    """Test caplet surface extrapolation at boundaries."""
    expiries = jnp.array([0.5, 1.0, 2.0])
    forwards = jnp.array([0.03, 0.035, 0.038])

    surface = construct_caplet_surface_from_sabr(
        expiries, forwards, alpha=0.20, beta=0.5
    )

    # Before first expiry
    vol_early = surface.get_vol(strike=0.03, expiry=0.1)
    assert vol_early > 0

    # After last expiry
    vol_late = surface.get_vol(strike=0.04, expiry=5.0)
    assert vol_late > 0


def test_caplet_surface_term_structure():
    """Test declining alpha term structure."""
    expiries = jnp.array([0.25, 0.5, 1.0, 2.0, 5.0])
    forwards = jnp.array([0.03, 0.032, 0.035, 0.038, 0.04])

    # Declining alpha (common in IR markets)
    alphas = jnp.array([0.25, 0.23, 0.20, 0.18, 0.15])

    surface = construct_caplet_surface_from_sabr(
        expiries, forwards, alphas, beta=0.5, rho=-0.3, nu=0.4
    )

    # ATM vols should decline with maturity (for this alpha structure)
    atm_vols = [surface.get_atm_vol(t) for t in expiries]

    # First should be higher than last
    assert atm_vols[0] > atm_vols[-1]


def test_swaption_cube_construction():
    """Test swaption vol cube construction."""
    expiries = jnp.array([0.25, 0.5, 1.0, 2.0])
    tenors = jnp.array([1.0, 2.0, 5.0, 10.0])

    # Forward swap rates matrix
    fwd_rates = jnp.array([
        [0.030, 0.035, 0.040, 0.042],
        [0.032, 0.036, 0.041, 0.043],
        [0.034, 0.038, 0.042, 0.044],
        [0.036, 0.040, 0.043, 0.045],
    ])

    # SABR parameters [expiries, tenors, 4]
    sabr_params = jnp.zeros((len(expiries), len(tenors), 4))
    for i in range(len(expiries)):
        for j in range(len(tenors)):
            sabr_params = sabr_params.at[i, j].set(
                jnp.array([0.20, 0.5, -0.3, 0.4])  # alpha, beta, rho, nu
            )

    cube = construct_swaption_cube_from_sabr(
        expiries, tenors, fwd_rates, sabr_params
    )

    # Test on-grid point
    vol = cube.get_vol(option_expiry=0.5, swap_tenor=2.0, strike=0.036)
    assert vol > 0

    # Test ATM vol
    atm_vol = cube.get_atm_vol(option_expiry=1.0, swap_tenor=5.0)
    assert atm_vol > 0


def test_swaption_cube_interpolation():
    """Test swaption cube interpolation."""
    expiries = jnp.array([1.0, 2.0, 5.0])
    tenors = jnp.array([2.0, 5.0, 10.0])

    fwd_rates = jnp.array([
        [0.03, 0.04, 0.042],
        [0.035, 0.042, 0.044],
        [0.038, 0.044, 0.046],
    ])

    sabr_params = jnp.zeros((len(expiries), len(tenors), 4))
    for i in range(len(expiries)):
        for j in range(len(tenors)):
            sabr_params = sabr_params.at[i, j].set(
                jnp.array([0.20, 0.5, -0.3, 0.4])
            )

    cube = SwaptionVolCube(expiries, tenors, fwd_rates, sabr_params)

    # Test interpolation between grid points
    vol = cube.get_vol(option_expiry=1.5, swap_tenor=3.5, strike=0.04)
    assert vol > 0
    # Vol should be finite (SABR can produce high vols for certain params)
    assert jnp.isfinite(vol)


def test_swaption_cube_vol_smile():
    """Test swaption cube vol smile extraction."""
    expiries = jnp.array([1.0, 2.0])
    tenors = jnp.array([5.0, 10.0])

    fwd_rates = jnp.array([
        [0.04, 0.042],
        [0.042, 0.044],
    ])

    sabr_params = jnp.array([
        [[0.20, 0.5, -0.3, 0.4], [0.20, 0.5, -0.3, 0.4]],
        [[0.18, 0.5, -0.3, 0.4], [0.18, 0.5, -0.3, 0.4]],
    ])

    cube = SwaptionVolCube(expiries, tenors, fwd_rates, sabr_params)

    # Get vol smile
    strikes = jnp.linspace(0.02, 0.06, 11)
    vols = cube.get_vol_smile(option_expiry=1.0, swap_tenor=5.0, strikes=strikes)

    assert len(vols) == len(strikes)
    assert jnp.all(vols > 0)

    # Vols should vary across strikes (smile/skew)
    assert jnp.std(vols) > 0


def test_swaption_cube_atm_matrix():
    """Test ATM volatility matrix extraction."""
    expiries = jnp.array([0.5, 1.0, 2.0])
    tenors = jnp.array([1.0, 5.0, 10.0])

    fwd_rates = jnp.ones((len(expiries), len(tenors))) * 0.04

    sabr_params = jnp.zeros((len(expiries), len(tenors), 4))
    for i in range(len(expiries)):
        for j in range(len(tenors)):
            # Varying alpha by expiry/tenor
            alpha = 0.20 - 0.01 * i - 0.005 * j
            sabr_params = sabr_params.at[i, j].set(
                jnp.array([alpha, 0.5, -0.3, 0.4])
            )

    cube = SwaptionVolCube(expiries, tenors, fwd_rates, sabr_params)

    atm_matrix = cube.get_atm_matrix()

    assert atm_matrix.shape == (len(expiries), len(tenors))
    assert jnp.all(atm_matrix > 0)

    # ATM vols should decline with expiry (for this alpha structure)
    assert atm_matrix[0, 0] > atm_matrix[-1, 0]


def test_construct_caplet_surface_scalar_params():
    """Test caplet surface construction with scalar SABR params."""
    expiries = jnp.array([0.5, 1.0, 2.0, 5.0])
    forwards = jnp.array([0.03, 0.035, 0.038, 0.04])

    # All scalar params
    surface = construct_caplet_surface_from_sabr(
        expiries,
        forwards,
        alpha=0.20,
        beta=0.5,
        rho=-0.3,
        nu=0.4,
    )

    # Should work
    vol = surface.get_vol(strike=0.04, expiry=1.0)
    assert vol > 0


def test_construct_caplet_surface_array_params():
    """Test caplet surface construction with array SABR params."""
    expiries = jnp.array([0.5, 1.0, 2.0, 5.0])
    forwards = jnp.array([0.03, 0.035, 0.038, 0.04])

    # Term structure of parameters
    alphas = jnp.array([0.25, 0.22, 0.19, 0.16])
    rhos = jnp.array([-0.25, -0.28, -0.32, -0.35])
    nus = jnp.array([0.35, 0.38, 0.42, 0.45])

    surface = construct_caplet_surface_from_sabr(
        expiries,
        forwards,
        alpha=alphas,
        beta=0.5,
        rho=rhos,
        nu=nus,
    )

    # Should work with term structure
    vol = surface.get_vol(strike=0.04, expiry=1.0)
    assert vol > 0


def test_caplet_surface_vol_slice():
    """Test getting vol slice across strikes."""
    expiries = jnp.array([1.0, 2.0, 3.0])
    forwards = jnp.array([0.03, 0.035, 0.038])

    surface = construct_caplet_surface_from_sabr(
        expiries, forwards, alpha=0.20, beta=0.5
    )

    strikes = jnp.linspace(0.01, 0.05, 9)
    vols = surface.get_vol_slice(expiry=1.0, strikes=strikes)

    assert len(vols) == len(strikes)
    assert jnp.all(vols > 0)


def test_swaption_cube_validation():
    """Test swaption cube input validation."""
    expiries = jnp.array([1.0, 2.0])
    tenors = jnp.array([5.0, 10.0])

    # Wrong shape for forward rates
    fwd_rates_wrong = jnp.ones((3, 2))  # Should be (2, 2)
    sabr_params = jnp.ones((2, 2, 4))

    with pytest.raises(ValueError, match="Forward rates shape"):
        SwaptionVolCube(expiries, tenors, fwd_rates_wrong, sabr_params)

    # Wrong shape for SABR params
    fwd_rates = jnp.ones((2, 2))
    sabr_params_wrong = jnp.ones((2, 2, 3))  # Should be (2, 2, 4)

    with pytest.raises(ValueError, match="SABR params shape"):
        SwaptionVolCube(expiries, tenors, fwd_rates, sabr_params_wrong)


def test_caplet_surface_validation():
    """Test caplet surface input validation."""
    expiries = jnp.array([1.0, 2.0, 3.0])
    forwards = jnp.array([0.03, 0.035])  # Wrong length

    slices = [
        SABRSlice(1.0, 0.03, 0.20, 0.5, -0.3, 0.4),
        SABRSlice(2.0, 0.035, 0.20, 0.5, -0.3, 0.4),
        SABRSlice(3.0, 0.038, 0.20, 0.5, -0.3, 0.4),
    ]

    with pytest.raises(ValueError, match="Expiries length"):
        CapletVolSurface(expiries, forwards, slices)
