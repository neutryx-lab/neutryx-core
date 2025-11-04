"""P&L Attribution System for derivatives portfolios.

This module provides comprehensive P&L attribution capabilities to decompose
portfolio P&L into risk factor contributions, enabling performance analysis
and risk management.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array


class AttributionMethod(Enum):
    """P&L attribution method."""
    GREEKS = "greeks"  # Greeks-based (Taylor expansion)
    REVALUATION = "revaluation"  # Full revaluation
    HYBRID = "hybrid"  # Combination of both


@dataclass
class MarketState:
    """Market state snapshot.

    Attributes:
        timestamp: Time of snapshot
        spot_prices: Spot prices by asset
        volatilities: Implied volatilities by asset
        interest_rates: Interest rates by currency
        fx_rates: FX rates by currency pair
        credit_spreads: Credit spreads by name/index
        dividend_yields: Dividend yields by asset
    """

    timestamp: float
    spot_prices: Dict[str, float] = field(default_factory=dict)
    volatilities: Dict[str, float] = field(default_factory=dict)
    interest_rates: Dict[str, float] = field(default_factory=dict)
    fx_rates: Dict[str, float] = field(default_factory=dict)
    credit_spreads: Dict[str, float] = field(default_factory=dict)
    dividend_yields: Dict[str, float] = field(default_factory=dict)

    def get_all_factors(self) -> Dict[str, float]:
        """Get all risk factors as a flat dictionary.

        Returns:
            Dictionary of all risk factors
        """
        factors = {}

        for asset, value in self.spot_prices.items():
            factors[f"spot_{asset}"] = value

        for asset, value in self.volatilities.items():
            factors[f"vol_{asset}"] = value

        for ccy, value in self.interest_rates.items():
            factors[f"rate_{ccy}"] = value

        for pair, value in self.fx_rates.items():
            factors[f"fx_{pair}"] = value

        for name, value in self.credit_spreads.items():
            factors[f"spread_{name}"] = value

        for asset, value in self.dividend_yields.items():
            factors[f"div_{asset}"] = value

        return factors


@dataclass
class PnLAttribution:
    """P&L attribution result.

    Attributes:
        total_pnl: Total P&L
        theta_pnl: P&L from time decay
        spot_pnl: P&L from spot moves (by asset)
        vol_pnl: P&L from volatility moves (by asset)
        rate_pnl: P&L from interest rate moves (by currency)
        fx_pnl: P&L from FX moves
        spread_pnl: P&L from credit spread moves
        gamma_pnl: P&L from gamma effects
        vega_pnl: P&L from vega effects
        cross_gamma_pnl: P&L from cross-gamma effects
        unexplained_pnl: Residual unexplained P&L
    """

    total_pnl: float
    theta_pnl: float = 0.0
    spot_pnl: Dict[str, float] = field(default_factory=dict)
    vol_pnl: Dict[str, float] = field(default_factory=dict)
    rate_pnl: Dict[str, float] = field(default_factory=dict)
    fx_pnl: Dict[str, float] = field(default_factory=dict)
    spread_pnl: Dict[str, float] = field(default_factory=dict)
    gamma_pnl: Dict[str, float] = field(default_factory=dict)
    vega_pnl: Dict[str, float] = field(default_factory=dict)
    cross_gamma_pnl: Dict[Tuple[str, str], float] = field(default_factory=dict)
    unexplained_pnl: float = 0.0

    def total_spot_pnl(self) -> float:
        """Total P&L from spot moves."""
        return sum(self.spot_pnl.values())

    def total_vol_pnl(self) -> float:
        """Total P&L from volatility moves."""
        return sum(self.vol_pnl.values())

    def total_rate_pnl(self) -> float:
        """Total P&L from rate moves."""
        return sum(self.rate_pnl.values())

    def explained_pnl(self) -> float:
        """Total explained P&L."""
        return (
            self.theta_pnl
            + self.total_spot_pnl()
            + self.total_vol_pnl()
            + self.total_rate_pnl()
            + sum(self.fx_pnl.values())
            + sum(self.spread_pnl.values())
            + sum(self.gamma_pnl.values())
            + sum(self.vega_pnl.values())
        )

    def explanation_ratio(self) -> float:
        """Ratio of explained to total P&L."""
        if abs(self.total_pnl) < 1e-8:
            return 1.0
        return self.explained_pnl() / self.total_pnl


class PnLAttributionEngine:
    """P&L attribution engine for derivatives portfolios.

    Performs P&L attribution using various methods to decompose total P&L
    into contributions from different risk factors.
    """

    def __init__(
        self,
        portfolio_pricer: Callable,
        greeks_calculator: Optional[Callable] = None,
        method: AttributionMethod = AttributionMethod.HYBRID
    ):
        """Initialize P&L attribution engine.

        Args:
            portfolio_pricer: Function to price portfolio given market state
            greeks_calculator: Optional function to compute Greeks
            method: Attribution method to use
        """
        self.portfolio_pricer = portfolio_pricer
        self.greeks_calculator = greeks_calculator
        self.method = method

    def attribute_pnl(
        self,
        start_state: MarketState,
        end_state: MarketState,
        start_portfolio_value: Optional[float] = None
    ) -> PnLAttribution:
        """Perform P&L attribution between two market states.

        Args:
            start_state: Market state at start of period
            end_state: Market state at end of period
            start_portfolio_value: Optional pre-computed portfolio value at start

        Returns:
            P&L attribution breakdown
        """
        # Compute portfolio values
        if start_portfolio_value is None:
            start_value = self.portfolio_pricer(start_state)
        else:
            start_value = start_portfolio_value

        end_value = self.portfolio_pricer(end_state)

        total_pnl = end_value - start_value

        if self.method == AttributionMethod.GREEKS:
            return self._attribute_greeks(start_state, end_state, total_pnl)
        elif self.method == AttributionMethod.REVALUATION:
            return self._attribute_revaluation(start_state, end_state, start_value, total_pnl)
        else:  # HYBRID
            return self._attribute_hybrid(start_state, end_state, start_value, total_pnl)

    def _attribute_greeks(
        self,
        start_state: MarketState,
        end_state: MarketState,
        total_pnl: float
    ) -> PnLAttribution:
        """Attribute P&L using Greeks (Taylor expansion).

        Args:
            start_state: Start market state
            end_state: End market state
            total_pnl: Total P&L

        Returns:
            P&L attribution
        """
        if self.greeks_calculator is None:
            raise ValueError("Greeks calculator required for Greeks-based attribution")

        # Compute Greeks at start state
        greeks = self.greeks_calculator(start_state)

        # Time decay
        dt = end_state.timestamp - start_state.timestamp
        theta_pnl = greeks.get("theta", 0.0) * dt

        # Spot P&L (delta + gamma)
        spot_pnl = {}
        gamma_pnl = {}

        for asset in start_state.spot_prices.keys():
            dS = end_state.spot_prices.get(asset, 0) - start_state.spot_prices.get(asset, 0)

            delta = greeks.get(f"delta_{asset}", 0.0)
            gamma = greeks.get(f"gamma_{asset}", 0.0)

            # First-order: Delta × dS
            spot_pnl[asset] = delta * dS

            # Second-order: 0.5 × Gamma × dS²
            gamma_pnl[asset] = 0.5 * gamma * (dS ** 2)

        # Vol P&L (vega + volga)
        vol_pnl = {}
        vega_pnl = {}

        for asset in start_state.volatilities.keys():
            dvol = end_state.volatilities.get(asset, 0) - start_state.volatilities.get(asset, 0)

            vega = greeks.get(f"vega_{asset}", 0.0)
            volga = greeks.get(f"volga_{asset}", 0.0)

            # First-order: Vega × dσ
            vol_pnl[asset] = vega * dvol

            # Second-order: 0.5 × Volga × dσ²
            vega_pnl[asset] = 0.5 * volga * (dvol ** 2)

        # Rate P&L
        rate_pnl = {}
        for ccy in start_state.interest_rates.keys():
            dr = end_state.interest_rates.get(ccy, 0) - start_state.interest_rates.get(ccy, 0)
            rho = greeks.get(f"rho_{ccy}", 0.0)
            rate_pnl[ccy] = rho * dr

        # Calculate explained P&L
        explained = (
            theta_pnl
            + sum(spot_pnl.values())
            + sum(gamma_pnl.values())
            + sum(vol_pnl.values())
            + sum(vega_pnl.values())
            + sum(rate_pnl.values())
        )

        unexplained = total_pnl - explained

        return PnLAttribution(
            total_pnl=total_pnl,
            theta_pnl=theta_pnl,
            spot_pnl=spot_pnl,
            vol_pnl=vol_pnl,
            rate_pnl=rate_pnl,
            gamma_pnl=gamma_pnl,
            vega_pnl=vega_pnl,
            unexplained_pnl=unexplained,
        )

    def _attribute_revaluation(
        self,
        start_state: MarketState,
        end_state: MarketState,
        start_value: float,
        total_pnl: float
    ) -> PnLAttribution:
        """Attribute P&L using full revaluation.

        Bumps each risk factor individually and revalues portfolio.

        Args:
            start_state: Start market state
            end_state: End market state
            start_value: Portfolio value at start
            total_pnl: Total P&L

        Returns:
            P&L attribution
        """
        spot_pnl = {}
        vol_pnl = {}
        rate_pnl = {}
        fx_pnl = {}
        spread_pnl = {}

        # Create intermediate states for each factor
        current_state = MarketState(
            timestamp=end_state.timestamp,
            spot_prices=start_state.spot_prices.copy(),
            volatilities=start_state.volatilities.copy(),
            interest_rates=start_state.interest_rates.copy(),
            fx_rates=start_state.fx_rates.copy(),
            credit_spreads=start_state.credit_spreads.copy(),
            dividend_yields=start_state.dividend_yields.copy(),
        )

        current_value = start_value

        # Time decay (move time forward with all else constant)
        time_only_state = MarketState(
            timestamp=end_state.timestamp,
            spot_prices=start_state.spot_prices.copy(),
            volatilities=start_state.volatilities.copy(),
            interest_rates=start_state.interest_rates.copy(),
            fx_rates=start_state.fx_rates.copy(),
            credit_spreads=start_state.credit_spreads.copy(),
            dividend_yields=start_state.dividend_yields.copy(),
        )
        time_value = self.portfolio_pricer(time_only_state)
        theta_pnl = time_value - current_value
        current_value = time_value

        # Spot moves
        for asset, end_spot in end_state.spot_prices.items():
            current_state.spot_prices[asset] = end_spot
            new_value = self.portfolio_pricer(current_state)
            spot_pnl[asset] = new_value - current_value
            current_value = new_value

        # Vol moves
        for asset, end_vol in end_state.volatilities.items():
            current_state.volatilities[asset] = end_vol
            new_value = self.portfolio_pricer(current_state)
            vol_pnl[asset] = new_value - current_value
            current_value = new_value

        # Rate moves
        for ccy, end_rate in end_state.interest_rates.items():
            current_state.interest_rates[ccy] = end_rate
            new_value = self.portfolio_pricer(current_state)
            rate_pnl[ccy] = new_value - current_value
            current_value = new_value

        # FX moves
        for pair, end_fx in end_state.fx_rates.items():
            current_state.fx_rates[pair] = end_fx
            new_value = self.portfolio_pricer(current_state)
            fx_pnl[pair] = new_value - current_value
            current_value = new_value

        # Credit spread moves
        for name, end_spread in end_state.credit_spreads.items():
            current_state.credit_spreads[name] = end_spread
            new_value = self.portfolio_pricer(current_state)
            spread_pnl[name] = new_value - current_value
            current_value = new_value

        # Unexplained is residual
        end_value = self.portfolio_pricer(end_state)
        unexplained = end_value - current_value

        return PnLAttribution(
            total_pnl=total_pnl,
            theta_pnl=theta_pnl,
            spot_pnl=spot_pnl,
            vol_pnl=vol_pnl,
            rate_pnl=rate_pnl,
            fx_pnl=fx_pnl,
            spread_pnl=spread_pnl,
            unexplained_pnl=unexplained,
        )

    def _attribute_hybrid(
        self,
        start_state: MarketState,
        end_state: MarketState,
        start_value: float,
        total_pnl: float
    ) -> PnLAttribution:
        """Attribute P&L using hybrid approach.

        Uses Greeks for small moves, revaluation for large moves.

        Args:
            start_state: Start market state
            end_state: End market state
            start_value: Portfolio value at start
            total_pnl: Total P&L

        Returns:
            P&L attribution
        """
        # Determine if moves are large
        large_move_threshold = 0.05  # 5%

        has_large_moves = False

        for asset in start_state.spot_prices.keys():
            start_spot = start_state.spot_prices[asset]
            end_spot = end_state.spot_prices.get(asset, start_spot)
            if abs(end_spot / start_spot - 1.0) > large_move_threshold:
                has_large_moves = True
                break

        if has_large_moves:
            # Use revaluation for accuracy
            return self._attribute_revaluation(start_state, end_state, start_value, total_pnl)
        else:
            # Use Greeks for efficiency
            return self._attribute_greeks(start_state, end_state, total_pnl)


@dataclass
class DailyPnLTracker:
    """Track daily P&L and attribution over time.

    Attributes:
        history: List of (date, attribution) tuples
    """

    history: List[Tuple[float, PnLAttribution]] = field(default_factory=list)

    def add_attribution(self, date: float, attribution: PnLAttribution):
        """Add attribution for a date.

        Args:
            date: Date timestamp
            attribution: P&L attribution
        """
        self.history.append((date, attribution))

    def total_pnl(self, start_date: Optional[float] = None, end_date: Optional[float] = None) -> float:
        """Compute total P&L over a period.

        Args:
            start_date: Start date (None = from beginning)
            end_date: End date (None = to end)

        Returns:
            Total P&L
        """
        total = 0.0

        for date, attribution in self.history:
            if start_date is not None and date < start_date:
                continue
            if end_date is not None and date > end_date:
                continue

            total += attribution.total_pnl

        return total

    def cumulative_attribution(self) -> Dict[str, List[float]]:
        """Compute cumulative attribution by risk factor.

        Returns:
            Dictionary mapping risk factor to cumulative P&L series
        """
        cumulative = {
            "theta": [],
            "spot": [],
            "vol": [],
            "rate": [],
            "unexplained": [],
        }

        theta_cum = 0.0
        spot_cum = 0.0
        vol_cum = 0.0
        rate_cum = 0.0
        unexplained_cum = 0.0

        for _, attribution in self.history:
            theta_cum += attribution.theta_pnl
            spot_cum += attribution.total_spot_pnl()
            vol_cum += attribution.total_vol_pnl()
            rate_cum += attribution.total_rate_pnl()
            unexplained_cum += attribution.unexplained_pnl

            cumulative["theta"].append(theta_cum)
            cumulative["spot"].append(spot_cum)
            cumulative["vol"].append(vol_cum)
            cumulative["rate"].append(rate_cum)
            cumulative["unexplained"].append(unexplained_cum)

        return cumulative


def analyze_pnl_drivers(attribution: PnLAttribution, threshold: float = 0.01) -> List[Tuple[str, float]]:
    """Analyze main P&L drivers.

    Args:
        attribution: P&L attribution
        threshold: Minimum contribution (as fraction of total P&L)

    Returns:
        List of (driver_name, contribution) sorted by magnitude
    """
    total = abs(attribution.total_pnl)
    if total < 1e-8:
        return []

    drivers = []

    # Theta
    if abs(attribution.theta_pnl / total) > threshold:
        drivers.append(("Theta", attribution.theta_pnl))

    # Spot by asset
    for asset, pnl in attribution.spot_pnl.items():
        if abs(pnl / total) > threshold:
            drivers.append((f"Spot_{asset}", pnl))

    # Vol by asset
    for asset, pnl in attribution.vol_pnl.items():
        if abs(pnl / total) > threshold:
            drivers.append((f"Vol_{asset}", pnl))

    # Rates by currency
    for ccy, pnl in attribution.rate_pnl.items():
        if abs(pnl / total) > threshold:
            drivers.append((f"Rate_{ccy}", pnl))

    # Unexplained
    if abs(attribution.unexplained_pnl / total) > threshold:
        drivers.append(("Unexplained", attribution.unexplained_pnl))

    # Sort by absolute magnitude
    drivers.sort(key=lambda x: abs(x[1]), reverse=True)

    return drivers
