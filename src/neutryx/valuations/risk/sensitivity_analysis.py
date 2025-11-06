"""Comprehensive sensitivity analysis framework for derivatives pricing.

This module provides a unified interface for calculating various risk sensitivities:
- DV01: Dollar value of 1 basis point change in interest rates
- CS01: Dollar value of 1 basis point change in credit spreads
- Vega: Sensitivity to volatility, with tenor/strike decomposition
- FX Greeks: Delta, Gamma, and cross-sensitivities
- Higher-order Greeks: Vanna, Volga, Vomma for volatility risk

Calculation Methods:
    - Analytical: Closed-form formulas where available
    - Finite Difference: Bump-and-revalue (central, forward, backward)
    - Automatic Differentiation: JAX grad/hessian for complex models
    - Algorithmic Adjoint: Efficient Hessian-vector products

References:
    - Hull, J. (2018). Options, Futures, and Other Derivatives (10th ed.)
    - Wilmott, P. (2006). Paul Wilmott on Quantitative Finance
    - Joshi, M. (2008). C++ Design Patterns and Derivatives Pricing
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax import Array

# Type aliases
PricingFunction = Callable[[Array], float]
SensitivityDict = Dict[str, Union[float, Array, Dict[str, float]]]


class SensitivityMethod(Enum):
    """Method for computing sensitivities."""

    ANALYTICAL = "analytical"
    FINITE_DIFF = "finite_difference"
    AUTODIFF = "automatic_differentiation"
    ADJOINT = "adjoint"


class FiniteDiffScheme(Enum):
    """Finite difference scheme."""

    CENTRAL = "central"  # O(h²) accurate
    FORWARD = "forward"  # O(h) accurate
    BACKWARD = "backward"  # O(h) accurate


@dataclass
class SensitivityConfig:
    """Configuration for sensitivity calculations.

    Args:
        method: Calculation method (analytical, finite_diff, autodiff, adjoint)
        fd_scheme: Finite difference scheme (central, forward, backward)
        bump_size: Bump size for finite difference (default: 1bp = 0.0001)
        use_parallel: Parallelize calculations across parameters
        epsilon: Tolerance for numerical stability checks
    """

    method: SensitivityMethod = SensitivityMethod.AUTODIFF
    fd_scheme: FiniteDiffScheme = FiniteDiffScheme.CENTRAL
    bump_size: float = 1e-4  # 1 basis point
    use_parallel: bool = True
    epsilon: float = 1e-10


# ============================================================================
# Core Sensitivity Calculators
# ============================================================================


def calculate_dv01(
    pricing_func: PricingFunction,
    params: Array,
    rate_indices: Union[int, List[int]],
    config: Optional[SensitivityConfig] = None
) -> Union[float, Array]:
    """Calculate DV01 (Dollar Value of 1bp) for interest rate products.

    DV01 measures the dollar change in value for a 1bp (0.01%) change in
    interest rates. Also known as PV01 or IR01.

    Args:
        pricing_func: Function that prices the instrument given parameters
        params: Current market parameters (rates, spreads, vols, etc.)
        rate_indices: Index/indices of rate parameters to bump
        config: Sensitivity calculation configuration

    Returns:
        DV01 value(s). Scalar if single rate, array if multiple rates.

    Example:
        >>> # Price a 5Y swap
        >>> def price_swap(params):
        ...     rates = params[:10]  # Curve rates
        ...     return swap_npv(rates, notional=10_000_000, tenor=5.0)
        >>>
        >>> rates = jnp.array([0.02, 0.022, 0.024, ...])  # Spot curve
        >>> dv01 = calculate_dv01(price_swap, rates, rate_indices=[0, 1, 2])
        >>> # dv01[0] = change in value for 1bp bump to 1Y rate

    Note:
        - Positive DV01 means gain value when rates rise (short position)
        - Negative DV01 means lose value when rates rise (long position)
        - Convention: report absolute value and specify long/short
    """
    if config is None:
        config = SensitivityConfig()

    if not isinstance(rate_indices, list):
        rate_indices = [rate_indices]

    if config.method == SensitivityMethod.AUTODIFF:
        return _dv01_autodiff(pricing_func, params, rate_indices)
    elif config.method == SensitivityMethod.FINITE_DIFF:
        return _dv01_finite_diff(pricing_func, params, rate_indices, config)
    else:
        raise ValueError(f"Method {config.method} not supported for DV01")


def _dv01_autodiff(
    pricing_func: PricingFunction,
    params: Array,
    rate_indices: List[int]
) -> Array:
    """Calculate DV01 using JAX automatic differentiation."""
    # Compute gradient with respect to all parameters
    grad_func = jax.grad(pricing_func)
    gradient = grad_func(params)

    # Extract DV01 for specified rates (convert from per-unit to per-bp)
    dv01_values = jnp.array([gradient[idx] * 1e-4 for idx in rate_indices])

    return dv01_values[0] if len(rate_indices) == 1 else dv01_values


def _dv01_finite_diff(
    pricing_func: PricingFunction,
    params: Array,
    rate_indices: List[int],
    config: SensitivityConfig
) -> Array:
    """Calculate DV01 using finite difference."""
    base_price = pricing_func(params)
    bump = config.bump_size  # 1bp = 0.0001

    dv01_values = []
    for idx in rate_indices:
        if config.fd_scheme == FiniteDiffScheme.CENTRAL:
            # Central difference: (P(r+h) - P(r-h)) / 2h
            params_up = params.at[idx].add(bump)
            params_down = params.at[idx].add(-bump)
            dv01 = (pricing_func(params_up) - pricing_func(params_down)) / 2.0
        elif config.fd_scheme == FiniteDiffScheme.FORWARD:
            # Forward difference: (P(r+h) - P(r)) / h
            params_up = params.at[idx].add(bump)
            dv01 = (pricing_func(params_up) - base_price)
        else:  # BACKWARD
            # Backward difference: (P(r) - P(r-h)) / h
            params_down = params.at[idx].add(-bump)
            dv01 = (base_price - pricing_func(params_down))

        dv01_values.append(dv01)

    dv01_values = jnp.array(dv01_values)
    return dv01_values[0] if len(rate_indices) == 1 else dv01_values


def calculate_cs01(
    pricing_func: PricingFunction,
    params: Array,
    spread_indices: Union[int, List[int]],
    config: Optional[SensitivityConfig] = None
) -> Union[float, Array]:
    """Calculate CS01 (Credit Spread 01) for credit products.

    CS01 measures the dollar change in value for a 1bp change in credit spread.
    Essential for CDS, CDX, bonds, and credit derivatives portfolios.

    Args:
        pricing_func: Function that prices the instrument given parameters
        params: Current market parameters (spreads, recovery, hazard rates)
        spread_indices: Index/indices of spread parameters to bump
        config: Sensitivity calculation configuration

    Returns:
        CS01 value(s). Scalar if single spread, array if multiple spreads.

    Example:
        >>> # Price a 5Y CDS
        >>> def price_cds(params):
        ...     spread = params[0]
        ...     recovery = params[1]
        ...     return cds_npv(spread, recovery, notional=10_000_000)
        >>>
        >>> params = jnp.array([0.0150, 0.40])  # 150bps spread, 40% recovery
        >>> cs01 = calculate_cs01(price_cds, params, spread_indices=0)
        >>> # cs01 = change in CDS value for 1bp spread widening

    Note:
        - Positive CS01: gain value when spreads widen (long protection)
        - Negative CS01: lose value when spreads widen (short protection)
        - For CDS indices, can compute CS01 per constituent
    """
    if config is None:
        config = SensitivityConfig()

    if not isinstance(spread_indices, list):
        spread_indices = [spread_indices]

    # CS01 uses same calculation as DV01, just different interpretation
    if config.method == SensitivityMethod.AUTODIFF:
        return _dv01_autodiff(pricing_func, params, spread_indices)
    elif config.method == SensitivityMethod.FINITE_DIFF:
        return _dv01_finite_diff(pricing_func, params, spread_indices, config)
    else:
        raise ValueError(f"Method {config.method} not supported for CS01")


def calculate_vega_surface(
    pricing_func: PricingFunction,
    params: Array,
    vol_indices: List[int],
    tenors: Optional[List[float]] = None,
    strikes: Optional[List[float]] = None,
    config: Optional[SensitivityConfig] = None
) -> Dict[str, Union[Array, Dict[Tuple[float, float], float]]]:
    """Calculate vega sensitivities across volatility surface.

    Computes sensitivity to volatility changes at different tenors and strikes.
    Essential for options portfolios and volatility trading strategies.

    Args:
        pricing_func: Function that prices the instrument
        params: Market parameters (rates, spot, vols)
        vol_indices: Indices of volatility parameters in params
        tenors: Option tenors (years) for each vol parameter
        strikes: Option strikes for each vol parameter
        config: Sensitivity calculation configuration

    Returns:
        Dictionary containing:
            - 'total_vega': Total vega across all volatilities
            - 'vega_vector': Vega for each volatility parameter
            - 'vega_surface': Dict mapping (tenor, strike) -> vega (if provided)
            - 'vega_by_tenor': Vega aggregated by tenor
            - 'vega_by_strike': Vega aggregated by strike

    Example:
        >>> # Portfolio of options across strikes and tenors
        >>> def price_portfolio(params):
        ...     spot = params[0]
        ...     vols = params[1:]  # Volatility surface points
        ...     return sum([price_option(spot, K, T, vol)
        ...                 for K, T, vol in zip(strikes, tenors, vols)])
        >>>
        >>> params = jnp.array([100.0, 0.20, 0.22, 0.25, 0.23, ...])
        >>> tenors = [0.25, 0.5, 1.0, 2.0, ...]
        >>> strikes = [95, 100, 105, 100, ...]
        >>>
        >>> vega_data = calculate_vega_surface(
        ...     price_portfolio, params,
        ...     vol_indices=list(range(1, len(params))),
        ...     tenors=tenors, strikes=strikes
        ... )
        >>> print(f"Total vega: {vega_data['total_vega']:.2f}")
        >>> print(f"1Y ATM vega: {vega_data['vega_surface'][(1.0, 100)]:.2f}")

    Note:
        - Vega is typically quoted per 1% (vol point) change
        - Options close to ATM have highest vega
        - Vega peaks around T/2 for European options
    """
    if config is None:
        config = SensitivityConfig()

    # Calculate vega for each volatility parameter
    if config.method == SensitivityMethod.AUTODIFF:
        grad_func = jax.grad(pricing_func)
        gradient = grad_func(params)
        vega_vector = jnp.array([gradient[idx] * 0.01 for idx in vol_indices])  # Per 1% vol
    else:
        # Finite difference
        bump = 0.01  # 1% volatility
        base_price = pricing_func(params)
        vega_list = []

        for idx in vol_indices:
            params_up = params.at[idx].add(bump)
            if config.fd_scheme == FiniteDiffScheme.CENTRAL:
                params_down = params.at[idx].add(-bump)
                vega = (pricing_func(params_up) - pricing_func(params_down)) / 2.0
            else:
                vega = pricing_func(params_up) - base_price
            vega_list.append(vega)

        vega_vector = jnp.array(vega_list)

    result = {
        'total_vega': float(jnp.sum(vega_vector)),
        'vega_vector': vega_vector
    }

    # Create surface mapping if tenor/strike info provided
    if tenors is not None and strikes is not None:
        vega_surface = {}
        vega_by_tenor = {}
        vega_by_strike = {}

        for i, (tenor, strike, vega) in enumerate(zip(tenors, strikes, vega_vector)):
            vega_surface[(tenor, strike)] = float(vega)

            # Aggregate by tenor
            if tenor not in vega_by_tenor:
                vega_by_tenor[tenor] = 0.0
            vega_by_tenor[tenor] += float(vega)

            # Aggregate by strike
            if strike not in vega_by_strike:
                vega_by_strike[strike] = 0.0
            vega_by_strike[strike] += float(vega)

        result['vega_surface'] = vega_surface
        result['vega_by_tenor'] = vega_by_tenor
        result['vega_by_strike'] = vega_by_strike

    return result


def calculate_fx_greeks(
    pricing_func: PricingFunction,
    params: Array,
    spot_index: int,
    vol_index: Optional[int] = None,
    config: Optional[SensitivityConfig] = None
) -> Dict[str, float]:
    """Calculate FX option Greeks: delta, gamma, vega, theta, rho.

    Computes first and second-order sensitivities for FX derivatives.

    Args:
        pricing_func: Function that prices the FX product
        params: Market parameters [spot, vol, r_domestic, r_foreign, T, ...]
        spot_index: Index of spot FX rate in params
        vol_index: Index of volatility in params (if applicable)
        config: Sensitivity calculation configuration

    Returns:
        Dictionary containing:
            - 'delta': ∂V/∂S (per unit spot change)
            - 'gamma': ∂²V/∂S² (delta change per unit spot)
            - 'vega': ∂V/∂σ (per 1% vol change)
            - 'theta': -∂V/∂T (per day time decay)
            - 'rho_domestic': ∂V/∂r_d (per 1% rate change)
            - 'rho_foreign': ∂V/∂r_f (per 1% rate change)

    Example:
        >>> # FX call option
        >>> def price_fx_call(params):
        ...     S, vol, r_d, r_f, T = params
        ...     return garman_kohlhagen(S, K=1.10, T=T, r_d=r_d, r_f=r_f,
        ...                             vol=vol, is_call=True)
        >>>
        >>> params = jnp.array([1.10, 0.12, 0.05, 0.02, 1.0])
        >>> greeks = calculate_fx_greeks(price_fx_call, params,
        ...                               spot_index=0, vol_index=1)
        >>> print(f"Delta: {greeks['delta']:.4f}")
        >>> print(f"Gamma: {greeks['gamma']:.6f}")
        >>> print(f"Vega: {greeks['vega']:.2f}")

    Note:
        - Delta: Hedge ratio (units of foreign currency per unit spot)
        - Gamma: Convexity, measures delta hedging error
        - Vega: Exposure to implied vol changes
        - For FX options, both domestic and foreign rho are relevant
    """
    if config is None:
        config = SensitivityConfig()

    greeks = {}

    if config.method == SensitivityMethod.AUTODIFF:
        # Use JAX autodiff for exact derivatives
        grad_func = jax.grad(pricing_func)
        hess_func = jax.hessian(pricing_func)

        gradient = grad_func(params)
        hessian = hess_func(params)

        # Delta: first derivative w.r.t. spot
        greeks['delta'] = float(gradient[spot_index])

        # Gamma: second derivative w.r.t. spot
        greeks['gamma'] = float(hessian[spot_index, spot_index])

        # Vega: first derivative w.r.t. volatility (if provided)
        if vol_index is not None:
            greeks['vega'] = float(gradient[vol_index] * 0.01)  # Per 1% vol

    else:
        # Finite difference method
        base_price = pricing_func(params)
        bump_spot = params[spot_index] * 0.01  # 1% bump

        # Delta (central difference)
        params_spot_up = params.at[spot_index].add(bump_spot)
        params_spot_down = params.at[spot_index].add(-bump_spot)
        price_up = pricing_func(params_spot_up)
        price_down = pricing_func(params_spot_down)

        greeks['delta'] = (price_up - price_down) / (2 * bump_spot)

        # Gamma (second derivative via finite difference)
        greeks['gamma'] = (price_up - 2 * base_price + price_down) / (bump_spot ** 2)

        # Vega (if vol index provided)
        if vol_index is not None:
            bump_vol = 0.01  # 1% vol bump
            params_vol_up = params.at[vol_index].add(bump_vol)
            params_vol_down = params.at[vol_index].add(-bump_vol)
            greeks['vega'] = (pricing_func(params_vol_up) -
                              pricing_func(params_vol_down)) / 2.0

    return greeks


def calculate_higher_order_greeks(
    pricing_func: PricingFunction,
    params: Array,
    spot_index: int,
    vol_index: int,
    config: Optional[SensitivityConfig] = None
) -> Dict[str, float]:
    """Calculate higher-order Greeks: vanna, volga, vomma.

    These cross-derivatives capture interactions between spot, volatility,
    and time. Critical for volatility trading and exotic options.

    Greeks:
        - Vanna: ∂²V/∂S∂σ (delta sensitivity to vol, vega sensitivity to spot)
        - Volga: ∂²V/∂σ² (vega sensitivity to vol, vega convexity)
        - Vomma: Same as Volga (alternative name)

    Args:
        pricing_func: Function that prices the instrument
        params: Market parameters (must include spot and vol)
        spot_index: Index of spot price in params
        vol_index: Index of volatility in params
        config: Sensitivity calculation configuration

    Returns:
        Dictionary containing:
            - 'vanna': ∂²V/∂S∂σ
            - 'volga': ∂²V/∂σ² (also called vomma)
            - 'vomma': Same as volga

    Example:
        >>> # Exotic option with vol sensitivity
        >>> def price_barrier_option(params):
        ...     S, vol, barrier, T = params
        ...     return barrier_option_price(S, barrier, vol, T)
        >>>
        >>> params = jnp.array([100.0, 0.20, 110.0, 1.0])
        >>> higher_greeks = calculate_higher_order_greeks(
        ...     price_barrier_option, params, spot_index=0, vol_index=1
        ... )
        >>> print(f"Vanna: {higher_greeks['vanna']:.4f}")
        >>> print(f"Volga: {higher_greeks['volga']:.4f}")

    Note:
        - Vanna: Important for managing delta-hedged volatility positions
        - Volga: Captures gamma-gamma (convexity of vega)
        - Both are zero for Black-Scholes at ATM forward
        - Critical for second-order hedging of exotic options

    References:
        - Derman, E. (1999). Regimes of Volatility
        - Taleb, N. (1997). Dynamic Hedging
    """
    if config is None:
        config = SensitivityConfig()

    if config.method == SensitivityMethod.AUTODIFF:
        # Use JAX Hessian for exact second derivatives
        hess_func = jax.hessian(pricing_func)
        hessian = hess_func(params)

        # Vanna: ∂²V/∂S∂σ
        vanna = float(hessian[spot_index, vol_index] * 0.01)  # Scale for 1% vol

        # Volga (Vomma): ∂²V/∂σ²
        volga = float(hessian[vol_index, vol_index] * 0.01 ** 2)  # Scale for 1% vol

    else:
        # Finite difference for mixed derivatives
        h_spot = params[spot_index] * 0.01
        h_vol = 0.01

        # Vanna via central difference: (∂²V/∂S∂σ)
        # = [V(S+h, σ+h) - V(S+h, σ-h) - V(S-h, σ+h) + V(S-h, σ-h)] / (4*h_S*h_σ)

        params_pp = params.at[spot_index].add(h_spot).at[vol_index].add(h_vol)
        params_pm = params.at[spot_index].add(h_spot).at[vol_index].add(-h_vol)
        params_mp = params.at[spot_index].add(-h_spot).at[vol_index].add(h_vol)
        params_mm = params.at[spot_index].add(-h_spot).at[vol_index].add(-h_vol)

        vanna = (pricing_func(params_pp) - pricing_func(params_pm) -
                 pricing_func(params_mp) + pricing_func(params_mm)) / (4 * h_spot * h_vol)

        # Volga via second derivative w.r.t. vol
        params_vol_up = params.at[vol_index].add(h_vol)
        params_vol_down = params.at[vol_index].add(-h_vol)
        base_price = pricing_func(params)

        volga = (pricing_func(params_vol_up) - 2 * base_price +
                 pricing_func(params_vol_down)) / (h_vol ** 2)

    return {
        'vanna': vanna,
        'volga': volga,
        'vomma': volga  # Same as volga
    }


# ============================================================================
# Portfolio-Level Sensitivity Aggregation
# ============================================================================


@dataclass
class PortfolioSensitivity:
    """Portfolio-level sensitivity aggregation.

    Attributes:
        total_dv01: Sum of DV01 across all positions
        total_cs01: Sum of CS01 across all credit positions
        total_vega: Sum of vega across all volatility positions
        total_delta: Sum of delta across all positions
        net_gamma: Sum of gamma (convexity measure)
        dv01_by_curve: DV01 broken down by yield curve
        cs01_by_entity: CS01 broken down by credit entity
        vega_by_tenor: Vega broken down by option tenor
        sensitivities_by_position: Individual position sensitivities
    """

    total_dv01: float = 0.0
    total_cs01: float = 0.0
    total_vega: float = 0.0
    total_delta: float = 0.0
    net_gamma: float = 0.0

    dv01_by_curve: Dict[str, float] = None
    cs01_by_entity: Dict[str, float] = None
    vega_by_tenor: Dict[float, float] = None
    sensitivities_by_position: Dict[str, Dict[str, float]] = None

    def __post_init__(self):
        if self.dv01_by_curve is None:
            self.dv01_by_curve = {}
        if self.cs01_by_entity is None:
            self.cs01_by_entity = {}
        if self.vega_by_tenor is None:
            self.vega_by_tenor = {}
        if self.sensitivities_by_position is None:
            self.sensitivities_by_position = {}


def aggregate_portfolio_sensitivities(
    positions: List[Dict[str, Any]],
    config: Optional[SensitivityConfig] = None
) -> PortfolioSensitivity:
    """Aggregate sensitivities across a portfolio of positions.

    Args:
        positions: List of position dictionaries, each containing:
            - 'name': Position identifier
            - 'pricing_func': Pricing function
            - 'params': Market parameters
            - 'type': 'rates', 'credit', 'fx', 'equity'
            - 'quantity': Number of contracts/notional
            - Additional metadata (curve_name, entity, tenor, etc.)
        config: Sensitivity calculation configuration

    Returns:
        PortfolioSensitivity object with aggregated risk metrics

    Example:
        >>> positions = [
        ...     {
        ...         'name': '5Y IRS',
        ...         'pricing_func': price_swap,
        ...         'params': jnp.array([0.02, 0.022, ...]),
        ...         'type': 'rates',
        ...         'quantity': 10_000_000,
        ...         'curve_name': 'USD-LIBOR'
        ...     },
        ...     {
        ...         'name': 'AAPL 5Y CDS',
        ...         'pricing_func': price_cds,
        ...         'params': jnp.array([0.0150, 0.40]),
        ...         'type': 'credit',
        ...         'quantity': 5_000_000,
        ...         'entity': 'AAPL'
        ...     },
        ...     # ... more positions
        ... ]
        >>>
        >>> portfolio_sens = aggregate_portfolio_sensitivities(positions)
        >>> print(f"Portfolio DV01: ${portfolio_sens.total_dv01:,.0f}")
        >>> print(f"Portfolio CS01: ${portfolio_sens.total_cs01:,.0f}")
        >>> print(f"Net Delta: {portfolio_sens.total_delta:.2f}")
    """
    if config is None:
        config = SensitivityConfig()

    portfolio = PortfolioSensitivity()

    for position in positions:
        name = position['name']
        pricing_func = position['pricing_func']
        params = position['params']
        quantity = position.get('quantity', 1.0)
        pos_type = position['type']

        position_sens = {}

        if pos_type == 'rates':
            # Calculate DV01
            rate_indices = position.get('rate_indices', [0])
            dv01 = calculate_dv01(pricing_func, params, rate_indices, config)
            dv01_total = float(jnp.sum(dv01)) * quantity

            portfolio.total_dv01 += dv01_total
            position_sens['dv01'] = dv01_total

            # Aggregate by curve if provided
            curve_name = position.get('curve_name', 'default')
            if curve_name not in portfolio.dv01_by_curve:
                portfolio.dv01_by_curve[curve_name] = 0.0
            portfolio.dv01_by_curve[curve_name] += dv01_total

        elif pos_type == 'credit':
            # Calculate CS01
            spread_indices = position.get('spread_indices', [0])
            cs01 = calculate_cs01(pricing_func, params, spread_indices, config)
            cs01_total = float(jnp.sum(cs01)) * quantity

            portfolio.total_cs01 += cs01_total
            position_sens['cs01'] = cs01_total

            # Aggregate by entity if provided
            entity = position.get('entity', 'default')
            if entity not in portfolio.cs01_by_entity:
                portfolio.cs01_by_entity[entity] = 0.0
            portfolio.cs01_by_entity[entity] += cs01_total

        elif pos_type in ['fx', 'equity']:
            # Calculate Greeks
            spot_index = position.get('spot_index', 0)
            vol_index = position.get('vol_index')

            greeks = calculate_fx_greeks(pricing_func, params, spot_index,
                                          vol_index, config)

            delta = greeks['delta'] * quantity
            gamma = greeks['gamma'] * quantity

            portfolio.total_delta += delta
            portfolio.net_gamma += gamma

            position_sens['delta'] = delta
            position_sens['gamma'] = gamma

            if 'vega' in greeks:
                vega = greeks['vega'] * quantity
                portfolio.total_vega += vega
                position_sens['vega'] = vega

                # Aggregate by tenor if provided
                tenor = position.get('tenor')
                if tenor is not None:
                    if tenor not in portfolio.vega_by_tenor:
                        portfolio.vega_by_tenor[tenor] = 0.0
                    portfolio.vega_by_tenor[tenor] += vega

        # Store individual position sensitivities
        portfolio.sensitivities_by_position[name] = position_sens

    return portfolio


def format_sensitivity_report(
    portfolio_sens: PortfolioSensitivity,
    include_breakdown: bool = True
) -> str:
    """Format portfolio sensitivities as human-readable report.

    Args:
        portfolio_sens: Portfolio sensitivity object
        include_breakdown: Include detailed breakdowns by curve/entity/tenor

    Returns:
        Formatted string report
    """
    lines = []
    lines.append("=" * 70)
    lines.append("PORTFOLIO SENSITIVITY REPORT")
    lines.append("=" * 70)
    lines.append("")

    # Summary
    lines.append("SUMMARY")
    lines.append("-" * 70)
    lines.append(f"Total DV01:       ${portfolio_sens.total_dv01:>15,.2f}")
    lines.append(f"Total CS01:       ${portfolio_sens.total_cs01:>15,.2f}")
    lines.append(f"Total Vega:       ${portfolio_sens.total_vega:>15,.2f}")
    lines.append(f"Total Delta:      {portfolio_sens.total_delta:>16,.2f}")
    lines.append(f"Net Gamma:        {portfolio_sens.net_gamma:>16,.4f}")
    lines.append("")

    if include_breakdown:
        # DV01 by curve
        if portfolio_sens.dv01_by_curve:
            lines.append("DV01 BY CURVE")
            lines.append("-" * 70)
            for curve, dv01 in sorted(portfolio_sens.dv01_by_curve.items()):
                lines.append(f"  {curve:<30} ${dv01:>15,.2f}")
            lines.append("")

        # CS01 by entity
        if portfolio_sens.cs01_by_entity:
            lines.append("CS01 BY ENTITY")
            lines.append("-" * 70)
            for entity, cs01 in sorted(portfolio_sens.cs01_by_entity.items()):
                lines.append(f"  {entity:<30} ${cs01:>15,.2f}")
            lines.append("")

        # Vega by tenor
        if portfolio_sens.vega_by_tenor:
            lines.append("VEGA BY TENOR")
            lines.append("-" * 70)
            for tenor, vega in sorted(portfolio_sens.vega_by_tenor.items()):
                lines.append(f"  {tenor:<5.2f}Y {'':<24} ${vega:>15,.2f}")
            lines.append("")

    lines.append("=" * 70)

    return "\n".join(lines)


# ============================================================================
# Utility Functions
# ============================================================================


def validate_sensitivity_params(
    params: Array,
    indices: List[int]
) -> None:
    """Validate that parameter indices are within bounds."""
    if not all(0 <= idx < len(params) for idx in indices):
        raise IndexError(f"Indices {indices} out of bounds for params of length {len(params)}")


def benchmark_sensitivity_methods(
    pricing_func: PricingFunction,
    params: Array,
    param_index: int,
    n_runs: int = 100
) -> Dict[str, float]:
    """Benchmark different sensitivity calculation methods.

    Compares accuracy and performance of analytical, finite difference,
    and automatic differentiation methods.

    Args:
        pricing_func: Pricing function to test
        params: Market parameters
        param_index: Parameter index to compute sensitivity for
        n_runs: Number of runs for timing

    Returns:
        Dictionary with timing and accuracy results
    """
    import time

    results = {}

    # Autodiff (reference)
    start = time.time()
    for _ in range(n_runs):
        grad_func = jax.grad(pricing_func)
        autodiff_sens = grad_func(params)[param_index]
    autodiff_time = (time.time() - start) / n_runs

    results['autodiff'] = {
        'time': autodiff_time,
        'value': float(autodiff_sens)
    }

    # Finite difference (central)
    bump = 1e-4
    start = time.time()
    for _ in range(n_runs):
        params_up = params.at[param_index].add(bump)
        params_down = params.at[param_index].add(-bump)
        fd_sens = (pricing_func(params_up) - pricing_func(params_down)) / (2 * bump)
    fd_time = (time.time() - start) / n_runs

    results['finite_diff'] = {
        'time': fd_time,
        'value': float(fd_sens),
        'error': float(jnp.abs(fd_sens - autodiff_sens))
    }

    return results
