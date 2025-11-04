"""Advanced Greeks computation with full Hessian and cross-Greeks.

This module provides comprehensive Greeks calculation including:
- Full Hessian matrix (all second-order derivatives)
- Cross-Greeks (mixed partial derivatives)
- Vega bucketing by tenor and strike
- Multi-asset cross-gamma
- Portfolio-level Greeks aggregation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array


@dataclass
class GreeksProfile:
    """Complete Greeks profile for an option or portfolio.

    Attributes:
        price: Option/portfolio value
        delta: First derivative w.r.t. spot
        gamma: Second derivative w.r.t. spot
        vega: Derivative w.r.t. volatility
        theta: Derivative w.r.t. time
        rho: Derivative w.r.t. interest rate
        vanna: Mixed derivative ∂²V/∂S∂σ
        volga: Second derivative ∂²V/∂σ²
        charm: Mixed derivative ∂²V/∂S∂t
        veta: Mixed derivative ∂²V/∂σ∂t
        speed: Third derivative ∂³V/∂S³
        zomma: Mixed third derivative ∂³V/∂S²∂σ
        color: Mixed third derivative ∂³V/∂S²∂t
    """

    price: float
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0
    rho: float = 0.0
    # Second-order cross Greeks
    vanna: float = 0.0  # ∂²V/∂S∂σ
    volga: float = 0.0  # ∂²V/∂σ² (also called vomma)
    charm: float = 0.0  # ∂²V/∂S∂t (also called delta decay)
    veta: float = 0.0  # ∂²V/∂σ∂t (also called vega decay)
    # Third-order Greeks
    speed: float = 0.0  # ∂³V/∂S³
    zomma: float = 0.0  # ∂³V/∂S²∂σ
    color: float = 0.0  # ∂³V/∂S²∂t


def compute_full_hessian(
    pricer: Callable,
    params: Array,
    param_names: List[str]
) -> Tuple[Array, Array]:
    """Compute full Hessian matrix of second derivatives.

    Args:
        pricer: Pricing function that takes parameter array
        params: Parameter values [n_params]
        param_names: Names of parameters (e.g., ["S", "sigma", "r"])

    Returns:
        Tuple of (gradient, hessian)
        - gradient: First derivatives [n_params]
        - hessian: Second derivatives [n_params, n_params]

    Example:
        >>> def pricer(params):
        ...     S, sigma, r = params
        ...     return black_scholes(S, K, T, r, q, sigma)
        >>> params = jnp.array([100.0, 0.2, 0.05])
        >>> grad, hess = compute_full_hessian(pricer, params, ["S", "sigma", "r"])
    """
    # Gradient (first derivatives)
    grad_fn = jax.grad(pricer)
    gradient = grad_fn(params)

    # Hessian (second derivatives)
    hess_fn = jax.hessian(pricer)
    hessian = hess_fn(params)

    return gradient, hessian


def compute_cross_greeks(
    pricer: Callable,
    S: float,
    sigma: float,
    r: float,
    T: float,
    **other_params
) -> GreeksProfile:
    """Compute all Greeks including cross-Greeks using autodiff.

    Args:
        pricer: Pricing function(S, sigma, r, T, **kwargs) -> price
        S: Spot price
        sigma: Volatility
        r: Interest rate
        T: Time to maturity
        **other_params: Additional parameters

    Returns:
        Complete Greeks profile
    """
    # Define parameter vector
    params = jnp.array([S, sigma, r, T])

    def price_fn(p):
        return pricer(p[0], p[1], p[2], p[3], **other_params)

    # Price
    price = float(price_fn(params))

    # First derivatives (gradient)
    grad = jax.grad(price_fn)(params)
    delta = float(grad[0])
    vega = float(grad[1])
    rho = float(grad[2])
    theta = float(grad[3])

    # Second derivatives (Hessian)
    hess = jax.hessian(price_fn)(params)

    gamma = float(hess[0, 0])  # ∂²V/∂S²
    vanna = float(hess[0, 1])  # ∂²V/∂S∂σ
    volga = float(hess[1, 1])  # ∂²V/∂σ²
    charm = float(hess[0, 3])  # ∂²V/∂S∂t
    veta = float(hess[1, 3])  # ∂²V/∂σ∂t

    # Third derivatives
    def gamma_fn(p):
        h = jax.hessian(price_fn)(p)
        return h[0, 0]

    third_derivs = jax.grad(gamma_fn)(params)
    speed = float(third_derivs[0])  # ∂³V/∂S³
    zomma = float(third_derivs[1])  # ∂³V/∂S²∂σ
    color = float(third_derivs[3])  # ∂³V/∂S²∂t

    return GreeksProfile(
        price=price,
        delta=delta,
        gamma=gamma,
        vega=vega,
        theta=theta,
        rho=rho,
        vanna=vanna,
        volga=volga,
        charm=charm,
        veta=veta,
        speed=speed,
        zomma=zomma,
        color=color,
    )


@dataclass
class VegaBucket:
    """Vega bucketing for a specific tenor/strike.

    Attributes:
        tenor: Time to expiry
        strike: Strike price (None for ATM)
        vega: Vega value
        weight: Weight in portfolio
    """

    tenor: float
    strike: Optional[float]
    vega: float
    weight: float = 1.0


class VegaBucketing:
    """Vega bucketing by tenor and strike for risk management.

    Decomposes portfolio vega into buckets by expiry and moneyness,
    enabling more granular volatility risk management.
    """

    def __init__(self, tenor_buckets: List[float], strike_buckets: Optional[List[float]] = None):
        """Initialize vega bucketing.

        Args:
            tenor_buckets: Tenor pillars (e.g., [0.25, 0.5, 1, 2, 5])
            strike_buckets: Strike buckets as % of ATM (e.g., [0.9, 0.95, 1.0, 1.05, 1.1])
        """
        self.tenor_buckets = jnp.array(tenor_buckets)

        if strike_buckets is not None:
            self.strike_buckets = jnp.array(strike_buckets)
        else:
            # Default: ATM, ±5%, ±10%, ±20%
            self.strike_buckets = jnp.array([0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2])

    def compute_vega_ladder(
        self,
        pricer: Callable,
        spot: float,
        base_vol: float,
        bump_size: float = 0.01,
        **fixed_params
    ) -> Dict[Tuple[float, float], float]:
        """Compute vega ladder (vega by tenor and strike).

        Args:
            pricer: Pricing function(S, K, T, sigma, **params)
            spot: Current spot price
            base_vol: Base volatility
            bump_size: Volatility bump size (e.g., 0.01 = 1%)
            **fixed_params: Fixed parameters (r, q, etc.)

        Returns:
            Dictionary mapping (tenor, strike_moneyness) -> vega
        """
        vega_ladder = {}

        for tenor in self.tenor_buckets:
            for strike_pct in self.strike_buckets:
                strike = spot * strike_pct

                # Price with base vol
                price_base = pricer(spot, strike, float(tenor), base_vol, **fixed_params)

                # Price with bumped vol
                price_bumped = pricer(spot, strike, float(tenor), base_vol + bump_size, **fixed_params)

                # Vega: ∂V/∂σ
                vega = (price_bumped - price_base) / bump_size

                vega_ladder[(float(tenor), float(strike_pct))] = float(vega)

        return vega_ladder

    def aggregate_vegas(
        self,
        vega_ladder: Dict[Tuple[float, float], float]
    ) -> Dict[str, float]:
        """Aggregate vegas by tenor and strike dimension.

        Args:
            vega_ladder: Vega ladder from compute_vega_ladder

        Returns:
            Dictionary with aggregated vegas
        """
        # Total vega
        total_vega = sum(vega_ladder.values())

        # Vega by tenor
        vega_by_tenor = {}
        for (tenor, strike_pct), vega in vega_ladder.items():
            if tenor not in vega_by_tenor:
                vega_by_tenor[tenor] = 0.0
            vega_by_tenor[tenor] += vega

        # Vega by moneyness
        vega_by_moneyness = {}
        for (tenor, strike_pct), vega in vega_ladder.items():
            if strike_pct not in vega_by_moneyness:
                vega_by_moneyness[strike_pct] = 0.0
            vega_by_moneyness[strike_pct] += vega

        return {
            "total_vega": total_vega,
            "vega_by_tenor": vega_by_tenor,
            "vega_by_moneyness": vega_by_moneyness,
        }


@dataclass
class MultiAssetGreeks:
    """Multi-asset Greeks including cross-gammas.

    Attributes:
        n_assets: Number of assets
        deltas: Delta for each asset [n_assets]
        gammas: Gamma matrix [n_assets, n_assets]
        cross_gammas: Off-diagonal cross-gammas
        vegas: Vega for each asset [n_assets]
    """

    n_assets: int
    deltas: Array
    gammas: Array  # Full gamma matrix including cross-gammas
    vegas: Array
    cross_gammas: Optional[Dict[Tuple[int, int], float]] = None

    def total_delta(self) -> float:
        """Compute total portfolio delta.

        Returns:
            Sum of deltas
        """
        return float(jnp.sum(self.deltas))

    def total_gamma(self) -> float:
        """Compute total portfolio gamma (diagonal sum).

        Returns:
            Sum of diagonal gammas
        """
        return float(jnp.sum(jnp.diag(self.gammas)))

    def get_cross_gamma(self, asset1: int, asset2: int) -> float:
        """Get cross-gamma between two assets.

        Args:
            asset1: Index of first asset
            asset2: Index of second asset

        Returns:
            Cross-gamma ∂²V/∂S₁∂S₂
        """
        return float(self.gammas[asset1, asset2])


def compute_multi_asset_greeks(
    pricer: Callable,
    spot_prices: Array,
    **other_params
) -> MultiAssetGreeks:
    """Compute multi-asset Greeks including full gamma matrix.

    Args:
        pricer: Pricing function taking array of spot prices
        spot_prices: Current spot prices [n_assets]
        **other_params: Additional fixed parameters

    Returns:
        Multi-asset Greeks profile

    Example:
        >>> def basket_pricer(spots):
        ...     # Basket option on 3 assets
        ...     weights = jnp.array([0.4, 0.3, 0.3])
        ...     basket = jnp.sum(spots * weights)
        ...     return jnp.maximum(basket - K, 0)
        >>> spots = jnp.array([100.0, 50.0, 200.0])
        >>> greeks = compute_multi_asset_greeks(basket_pricer, spots)
        >>> print(f"Cross-gamma(0,1): {greeks.get_cross_gamma(0, 1)}")
    """
    n_assets = len(spot_prices)

    def price_fn(S):
        return pricer(S, **other_params)

    # Deltas (first derivatives)
    deltas = jax.grad(price_fn)(spot_prices)

    # Gamma matrix (Hessian: all second derivatives)
    gammas = jax.hessian(price_fn)(spot_prices)

    # For vega, need volatility parameters
    # This is a simplified version; full implementation would include vol parameters
    vegas = jnp.zeros(n_assets)

    # Extract cross-gammas
    cross_gammas = {}
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            cross_gammas[(i, j)] = float(gammas[i, j])

    return MultiAssetGreeks(
        n_assets=n_assets,
        deltas=deltas,
        gammas=gammas,
        vegas=vegas,
        cross_gammas=cross_gammas,
    )


@dataclass
class PortfolioGreeks:
    """Aggregated Greeks for a portfolio of instruments.

    Attributes:
        instruments: List of instrument names
        individual_greeks: Greeks for each instrument
        aggregated: Aggregated portfolio Greeks
        correlations: Asset correlations for cross-gamma adjustments
    """

    instruments: List[str]
    individual_greeks: List[GreeksProfile]
    aggregated: GreeksProfile
    correlations: Optional[Array] = None

    @classmethod
    def from_instruments(
        cls,
        instruments: List[str],
        greeks_list: List[GreeksProfile],
        correlations: Optional[Array] = None
    ) -> PortfolioGreeks:
        """Aggregate Greeks from individual instruments.

        Args:
            instruments: Instrument names
            greeks_list: List of Greeks profiles
            correlations: Optional correlation matrix for cross-gamma

        Returns:
            Portfolio Greeks
        """
        # Sum all Greeks
        total_price = sum(g.price for g in greeks_list)
        total_delta = sum(g.delta for g in greeks_list)
        total_gamma = sum(g.gamma for g in greeks_list)
        total_vega = sum(g.vega for g in greeks_list)
        total_theta = sum(g.theta for g in greeks_list)
        total_rho = sum(g.rho for g in greeks_list)
        total_vanna = sum(g.vanna for g in greeks_list)
        total_volga = sum(g.volga for g in greeks_list)

        aggregated = GreeksProfile(
            price=total_price,
            delta=total_delta,
            gamma=total_gamma,
            vega=total_vega,
            theta=total_theta,
            rho=total_rho,
            vanna=total_vanna,
            volga=total_volga,
        )

        return cls(
            instruments=instruments,
            individual_greeks=greeks_list,
            aggregated=aggregated,
            correlations=correlations,
        )

    def delta_hedged_gamma(self) -> float:
        """Compute residual gamma after delta hedging.

        For a delta-hedged portfolio, gamma represents the main risk.

        Returns:
            Total gamma
        """
        return self.aggregated.gamma

    def vanna_volga_risk(self) -> Tuple[float, float]:
        """Compute vanna-volga risk measures.

        Returns:
            Tuple of (vanna, volga) for portfolio
        """
        return self.aggregated.vanna, self.aggregated.volga


def taylor_expansion_pnl(
    greeks: GreeksProfile,
    dS: float,
    dsigma: float,
    dt: float,
    dr: float = 0.0,
    include_third_order: bool = False
) -> float:
    """Estimate P&L using Taylor expansion of Greeks.

    ΔV ≈ Δ·ΔS + ½Γ·(ΔS)² + ν·Δσ + θ·Δt + ρ·Δr + higher-order terms

    Args:
        greeks: Greeks profile
        dS: Change in spot price
        dsigma: Change in volatility
        dt: Time elapsed
        dr: Change in interest rate
        include_third_order: Include third-order terms

    Returns:
        Estimated P&L
    """
    # First-order terms
    pnl = greeks.delta * dS
    pnl += greeks.vega * dsigma
    pnl += greeks.theta * dt
    pnl += greeks.rho * dr

    # Second-order terms
    pnl += 0.5 * greeks.gamma * (dS ** 2)
    pnl += 0.5 * greeks.volga * (dsigma ** 2)
    pnl += greeks.vanna * dS * dsigma
    pnl += greeks.charm * dS * dt
    pnl += greeks.veta * dsigma * dt

    # Third-order terms
    if include_third_order:
        pnl += (1.0 / 6.0) * greeks.speed * (dS ** 3)
        pnl += greeks.zomma * (dS ** 2) * dsigma
        pnl += greeks.color * (dS ** 2) * dt

    return pnl
