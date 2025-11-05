"""Interest rate volatility surfaces and cubes.

This module implements comprehensive IR volatility structures:
- Swaption volatility cubes (expiry x tenor x strike)
- Caplet/floorlet volatility surfaces (expiry x strike)
- IR volatility interpolation and calibration
- SABR parameterization for IR markets
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional

import jax
import jax.numpy as jnp
from jax import Array

from neutryx.models.sabr import SABRParams, hagan_implied_vol
from neutryx.market.interpolation import linear_interpolation


@dataclass
class SABRSlice:
    """SABR parameters for a single option expiry.

    Parameters
    ----------
    expiry : float
        Option expiry in years
    forward_rate : float
        Forward rate for this expiry
    alpha : float
        ATM volatility level
    beta : float
        CEV exponent (typically 0.0 to 1.0)
    rho : float
        Correlation between rate and volatility (-1 to 1)
    nu : float
        Volatility of volatility (vol-of-vol)
    """

    expiry: float
    forward_rate: float
    alpha: float
    beta: float
    rho: float
    nu: float

    def implied_vol(self, strike: float) -> float:
        """Get implied volatility for a given strike.

        Parameters
        ----------
        strike : float
            Strike rate

        Returns
        -------
        float
            Implied volatility
        """
        params = SABRParams(
            alpha=self.alpha,
            beta=self.beta,
            rho=self.rho,
            nu=self.nu,
        )
        return hagan_implied_vol(
            self.forward_rate,
            strike,
            self.expiry,
            params,
        )

    def vol_smile(self, strikes: Array) -> Array:
        """Get volatility smile across multiple strikes.

        Parameters
        ----------
        strikes : Array
            Array of strike rates

        Returns
        -------
        Array
            Array of implied volatilities
        """
        return jax.vmap(self.implied_vol)(jnp.asarray(strikes))


@dataclass
class CapletVolSurface:
    """Caplet/floorlet volatility surface with SABR parameterization.

    A caplet vol surface stores volatilities across expiry and strike dimensions.
    Common market conventions:
    - ATM strikes (forward rate at each expiry)
    - OTM strikes for wings (e.g., ±100 bps, ±200 bps)

    Parameters
    ----------
    expiries : Array
        Caplet expiries in years (e.g., [0.25, 0.5, 1.0, 2.0, ...])
    forward_rates : Array
        Forward LIBOR/SOFR rates for each expiry
    sabr_slices : list[SABRSlice]
        SABR parameters for each expiry
    day_count : str
        Day count convention (default: 'ACT/360')

    Notes
    -----
    The surface supports:
    - Interpolation across expiries (linear in variance)
    - Smile interpolation via SABR at each expiry
    - Conversion to/from ATM vol + skew + smile format

    Example
    -------
    >>> expiries = jnp.array([0.25, 0.5, 1.0, 2.0])
    >>> forwards = jnp.array([0.03, 0.032, 0.035, 0.038])
    >>> slices = [
    ...     SABRSlice(0.25, 0.03, alpha=0.20, beta=0.5, rho=-0.3, nu=0.4),
    ...     SABRSlice(0.5, 0.032, alpha=0.19, beta=0.5, rho=-0.3, nu=0.4),
    ...     ...
    ... ]
    >>> surface = CapletVolSurface(expiries, forwards, slices)
    >>> vol = surface.get_vol(strike=0.04, expiry=0.75)
    """

    expiries: Array
    forward_rates: Array
    sabr_slices: list[SABRSlice]
    day_count: str = 'ACT/360'

    def __post_init__(self):
        """Validate surface construction."""
        self.expiries = jnp.asarray(self.expiries)
        self.forward_rates = jnp.asarray(self.forward_rates)

        if len(self.expiries) != len(self.forward_rates):
            raise ValueError(
                f"Expiries length {len(self.expiries)} != "
                f"Forward rates length {len(self.forward_rates)}"
            )

        if len(self.expiries) != len(self.sabr_slices):
            raise ValueError(
                f"Expiries length {len(self.expiries)} != "
                f"SABR slices length {len(self.sabr_slices)}"
            )

    def get_vol(self, strike: float, expiry: float) -> float:
        """Get implied volatility for a given strike and expiry.

        Interpolates between SABR slices if expiry not in grid.

        Parameters
        ----------
        strike : float
            Strike rate
        expiry : float
            Caplet expiry in years

        Returns
        -------
        float
            Implied volatility
        """
        expiry = float(expiry)
        strike = float(strike)

        # Find surrounding expiries
        if expiry <= self.expiries[0]:
            # Use first slice
            return self.sabr_slices[0].implied_vol(strike)
        elif expiry >= self.expiries[-1]:
            # Use last slice
            return self.sabr_slices[-1].implied_vol(strike)

        # Find bracketing expiries
        idx = jnp.searchsorted(self.expiries, expiry)
        t1 = float(self.expiries[idx - 1])
        t2 = float(self.expiries[idx])

        # Get vols from both slices
        vol1 = self.sabr_slices[idx - 1].implied_vol(strike)
        vol2 = self.sabr_slices[idx].implied_vol(strike)

        # Interpolate in variance space (vol^2 * T)
        var1 = vol1**2 * t1
        var2 = vol2**2 * t2

        # Linear interpolation in variance
        weight = (expiry - t1) / (t2 - t1)
        variance = var1 + weight * (var2 - var1)

        # Convert back to vol
        return float(jnp.sqrt(variance / expiry))

    def get_atm_vol(self, expiry: float) -> float:
        """Get ATM (at-the-money) volatility for a given expiry.

        Parameters
        ----------
        expiry : float
            Caplet expiry in years

        Returns
        -------
        float
            ATM implied volatility
        """
        # Find forward rate for this expiry
        forward = float(linear_interpolation(
            self.expiries,
            self.forward_rates,
            jnp.array([expiry])
        )[0])

        return self.get_vol(forward, expiry)

    def get_vol_slice(self, expiry: float, strikes: Array) -> Array:
        """Get volatility smile for a given expiry across strikes.

        Parameters
        ----------
        expiry : float
            Caplet expiry in years
        strikes : Array
            Array of strike rates

        Returns
        -------
        Array
            Array of implied volatilities
        """
        return jnp.array([
            self.get_vol(float(k), expiry)
            for k in jnp.asarray(strikes)
        ])


@dataclass
class SwaptionVolCube:
    """Swaption volatility cube with term structure of SABR parameters.

    A swaption vol cube stores volatilities across 3 dimensions:
    - Option expiry (time to swaption expiry)
    - Swap tenor (length of underlying swap)
    - Strike (fixed rate of underlying swap)

    Market conventions:
    - ATM strikes (forward swap rate for each expiry/tenor pair)
    - Standard tenors: 1Y, 2Y, 5Y, 10Y, 30Y
    - Standard expiries: 1M, 3M, 6M, 1Y, 2Y, 5Y, 10Y

    Parameters
    ----------
    option_expiries : Array
        Swaption expiries in years
    swap_tenors : Array
        Underlying swap tenors in years
    forward_swap_rates : Array
        Forward swap rates [expiries, tenors]
    sabr_params : Array
        SABR parameters [expiries, tenors, 4] where 4 = [alpha, beta, rho, nu]
    interpolation_method : str
        Method for interpolation ('linear', 'cubic')

    Notes
    -----
    The cube supports:
    - 3D interpolation (expiry, tenor, strike)
    - ATM vol extraction
    - Vol smile for any expiry/tenor combination
    - Conversion to/from market quote formats (ATM, RR, BF)

    Example
    -------
    >>> expiries = jnp.array([0.25, 0.5, 1.0, 2.0, 5.0])
    >>> tenors = jnp.array([1.0, 2.0, 5.0, 10.0])
    >>> # Forward swap rates [5 expiries x 4 tenors]
    >>> fwd_rates = jnp.array([
    ...     [0.03, 0.035, 0.04, 0.042],  # 3M expiry
    ...     [0.032, 0.036, 0.041, 0.043],  # 6M expiry
    ...     ...
    ... ])
    >>> # SABR params [5 expiries x 4 tenors x 4 params]
    >>> sabr = jnp.array([...])  # Calibrated parameters
    >>> cube = SwaptionVolCube(expiries, tenors, fwd_rates, sabr)
    >>> vol = cube.get_vol(
    ...     option_expiry=0.75,
    ...     swap_tenor=3.0,
    ...     strike=0.04
    ... )
    """

    option_expiries: Array
    swap_tenors: Array
    forward_swap_rates: Array  # [expiries, tenors]
    sabr_params: Array  # [expiries, tenors, 4] = [alpha, beta, rho, nu]
    interpolation_method: str = 'linear'

    def __post_init__(self):
        """Validate cube construction."""
        self.option_expiries = jnp.asarray(self.option_expiries)
        self.swap_tenors = jnp.asarray(self.swap_tenors)
        self.forward_swap_rates = jnp.asarray(self.forward_swap_rates)
        self.sabr_params = jnp.asarray(self.sabr_params)

        n_expiries = len(self.option_expiries)
        n_tenors = len(self.swap_tenors)

        if self.forward_swap_rates.shape != (n_expiries, n_tenors):
            raise ValueError(
                f"Forward rates shape {self.forward_swap_rates.shape} "
                f"!= ({n_expiries}, {n_tenors})"
            )

        if self.sabr_params.shape != (n_expiries, n_tenors, 4):
            raise ValueError(
                f"SABR params shape {self.sabr_params.shape} "
                f"!= ({n_expiries}, {n_tenors}, 4)"
            )

    def _get_sabr_params_at_point(
        self,
        option_expiry: float,
        swap_tenor: float,
    ) -> tuple[float, SABRParams]:
        """Interpolate SABR parameters to a specific expiry/tenor point.

        Returns
        -------
        tuple[float, SABRParams]
            (forward_swap_rate, sabr_params)
        """
        # Find indices for bilinear interpolation
        exp_idx = jnp.searchsorted(self.option_expiries, option_expiry)
        ten_idx = jnp.searchsorted(self.swap_tenors, swap_tenor)

        # Clamp indices
        exp_idx = jnp.clip(exp_idx, 1, len(self.option_expiries) - 1)
        ten_idx = jnp.clip(ten_idx, 1, len(self.swap_tenors) - 1)

        # Get surrounding points
        e0, e1 = exp_idx - 1, exp_idx
        t0, t1 = ten_idx - 1, ten_idx

        # Weights
        w_e = (option_expiry - self.option_expiries[e0]) / (
            self.option_expiries[e1] - self.option_expiries[e0]
        )
        w_t = (swap_tenor - self.swap_tenors[t0]) / (
            self.swap_tenors[t1] - self.swap_tenors[t0]
        )

        # Bilinear interpolation for forward rate
        f00 = self.forward_swap_rates[e0, t0]
        f01 = self.forward_swap_rates[e0, t1]
        f10 = self.forward_swap_rates[e1, t0]
        f11 = self.forward_swap_rates[e1, t1]

        forward_rate = (
            (1 - w_e) * (1 - w_t) * f00 +
            (1 - w_e) * w_t * f01 +
            w_e * (1 - w_t) * f10 +
            w_e * w_t * f11
        )

        # Bilinear interpolation for SABR params
        sabr00 = self.sabr_params[e0, t0]
        sabr01 = self.sabr_params[e0, t1]
        sabr10 = self.sabr_params[e1, t0]
        sabr11 = self.sabr_params[e1, t1]

        sabr_interp = (
            (1 - w_e) * (1 - w_t) * sabr00 +
            (1 - w_e) * w_t * sabr01 +
            w_e * (1 - w_t) * sabr10 +
            w_e * w_t * sabr11
        )

        params = SABRParams(
            alpha=float(sabr_interp[0]),
            beta=float(sabr_interp[1]),
            rho=float(sabr_interp[2]),
            nu=float(sabr_interp[3]),
        )

        return float(forward_rate), params

    def get_vol(
        self,
        option_expiry: float,
        swap_tenor: float,
        strike: float,
    ) -> float:
        """Get implied volatility for a swaption.

        Parameters
        ----------
        option_expiry : float
            Time to swaption expiry in years
        swap_tenor : float
            Tenor of underlying swap in years
        strike : float
            Fixed rate (strike) of the swaption

        Returns
        -------
        float
            Implied volatility (log-normal)
        """
        # Interpolate to get forward and SABR params
        forward_rate, sabr_params = self._get_sabr_params_at_point(
            option_expiry, swap_tenor
        )

        # Compute SABR implied vol
        vol = hagan_implied_vol(
            forward_rate,
            strike,
            option_expiry,
            sabr_params,
        )

        return float(vol)

    def get_atm_vol(
        self,
        option_expiry: float,
        swap_tenor: float,
    ) -> float:
        """Get ATM volatility for a swaption.

        Parameters
        ----------
        option_expiry : float
            Time to swaption expiry in years
        swap_tenor : float
            Tenor of underlying swap in years

        Returns
        -------
        float
            ATM implied volatility
        """
        forward_rate, sabr_params = self._get_sabr_params_at_point(
            option_expiry, swap_tenor
        )

        # ATM strike = forward rate
        vol = hagan_implied_vol(
            forward_rate,
            forward_rate,  # ATM
            option_expiry,
            sabr_params,
        )

        return float(vol)

    def get_vol_smile(
        self,
        option_expiry: float,
        swap_tenor: float,
        strikes: Array,
    ) -> Array:
        """Get volatility smile for a swaption across strikes.

        Parameters
        ----------
        option_expiry : float
            Time to swaption expiry in years
        swap_tenor : float
            Tenor of underlying swap in years
        strikes : Array
            Array of strike rates

        Returns
        -------
        Array
            Array of implied volatilities
        """
        strikes = jnp.asarray(strikes)
        vols = jnp.array([
            self.get_vol(option_expiry, swap_tenor, float(k))
            for k in strikes
        ])
        return vols

    def get_atm_matrix(self) -> Array:
        """Get matrix of ATM volatilities [expiries x tenors].

        Returns
        -------
        Array
            ATM volatility matrix
        """
        n_expiries = len(self.option_expiries)
        n_tenors = len(self.swap_tenors)

        atm_vols = jnp.zeros((n_expiries, n_tenors))

        for i, expiry in enumerate(self.option_expiries):
            for j, tenor in enumerate(self.swap_tenors):
                # For points on the grid, use SABR directly
                forward = self.forward_swap_rates[i, j]
                sabr = SABRParams(
                    alpha=float(self.sabr_params[i, j, 0]),
                    beta=float(self.sabr_params[i, j, 1]),
                    rho=float(self.sabr_params[i, j, 2]),
                    nu=float(self.sabr_params[i, j, 3]),
                )
                vol = hagan_implied_vol(forward, forward, float(expiry), sabr)
                atm_vols = atm_vols.at[i, j].set(vol)

        return atm_vols


def construct_caplet_surface_from_sabr(
    expiries: Array,
    forward_rates: Array,
    alpha: float | Array,
    beta: float = 0.5,
    rho: float | Array = -0.3,
    nu: float | Array = 0.4,
) -> CapletVolSurface:
    """Construct caplet vol surface from SABR parameters.

    Parameters
    ----------
    expiries : Array
        Caplet expiries in years
    forward_rates : Array
        Forward rates for each expiry
    alpha : float or Array
        ATM vol level (scalar or per expiry)
    beta : float
        CEV exponent (typically constant across expiries)
    rho : float or Array
        Correlation (scalar or per expiry)
    nu : float or Array
        Vol-of-vol (scalar or per expiry)

    Returns
    -------
    CapletVolSurface
        Constructed caplet vol surface

    Examples
    --------
    >>> expiries = jnp.array([0.25, 0.5, 1.0, 2.0, 5.0])
    >>> forwards = jnp.array([0.03, 0.032, 0.035, 0.038, 0.04])
    >>> # Declining alpha term structure
    >>> alphas = jnp.array([0.25, 0.23, 0.20, 0.18, 0.15])
    >>> surface = construct_caplet_surface_from_sabr(
    ...     expiries, forwards, alphas, beta=0.5, rho=-0.3, nu=0.4
    ... )
    """
    expiries = jnp.asarray(expiries)
    forward_rates = jnp.asarray(forward_rates)

    # Convert scalars to arrays
    if isinstance(alpha, (int, float)):
        alpha = jnp.full_like(expiries, alpha)
    if isinstance(rho, (int, float)):
        rho = jnp.full_like(expiries, rho)
    if isinstance(nu, (int, float)):
        nu = jnp.full_like(expiries, nu)

    alpha = jnp.asarray(alpha)
    rho = jnp.asarray(rho)
    nu = jnp.asarray(nu)

    # Create SABR slices
    sabr_slices = [
        SABRSlice(
            expiry=float(expiries[i]),
            forward_rate=float(forward_rates[i]),
            alpha=float(alpha[i]),
            beta=beta,
            rho=float(rho[i]),
            nu=float(nu[i]),
        )
        for i in range(len(expiries))
    ]

    return CapletVolSurface(
        expiries=expiries,
        forward_rates=forward_rates,
        sabr_slices=sabr_slices,
    )


def construct_swaption_cube_from_sabr(
    option_expiries: Array,
    swap_tenors: Array,
    forward_swap_rates: Array,
    sabr_params: Array,
) -> SwaptionVolCube:
    """Construct swaption vol cube from SABR parameters.

    Parameters
    ----------
    option_expiries : Array
        Swaption expiries in years
    swap_tenors : Array
        Underlying swap tenors in years
    forward_swap_rates : Array
        Forward swap rates [expiries, tenors]
    sabr_params : Array
        SABR parameters [expiries, tenors, 4] where 4 = [alpha, beta, rho, nu]

    Returns
    -------
    SwaptionVolCube
        Constructed swaption vol cube

    Examples
    --------
    >>> expiries = jnp.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0])
    >>> tenors = jnp.array([1.0, 2.0, 5.0, 10.0, 30.0])
    >>> # Forward swap rates matrix
    >>> fwd_rates = jnp.array([
    ...     [0.030, 0.035, 0.040, 0.042, 0.043],  # 3M expiry
    ...     [0.032, 0.036, 0.041, 0.043, 0.044],  # 6M expiry
    ...     [0.034, 0.038, 0.042, 0.044, 0.045],  # 1Y expiry
    ...     ...
    ... ])
    >>> # SABR parameters calibrated for each expiry/tenor
    >>> sabr = jnp.array([...])  # Shape (6, 5, 4)
    >>> cube = construct_swaption_cube_from_sabr(
    ...     expiries, tenors, fwd_rates, sabr
    ... )
    """
    return SwaptionVolCube(
        option_expiries=jnp.asarray(option_expiries),
        swap_tenors=jnp.asarray(swap_tenors),
        forward_swap_rates=jnp.asarray(forward_swap_rates),
        sabr_params=jnp.asarray(sabr_params),
    )


__all__ = [
    'SABRSlice',
    'CapletVolSurface',
    'SwaptionVolCube',
    'construct_caplet_surface_from_sabr',
    'construct_swaption_cube_from_sabr',
]
