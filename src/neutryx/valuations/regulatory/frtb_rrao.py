"""FRTB Residual Risk Add-On (RRAO) calculations.

This module implements the Residual Risk Add-On under FRTB, which captures risks not
adequately covered by the delta, vega, and curvature components.

RRAO applies to:
- Exotic underlyings (e.g., longevity, weather, natural catastrophe)
- Exotic payoffs (digital options, barrier options with discontinuous payoffs)
- Correlation risk not captured by standard sensitivities
- Basis risk between related but non-identical risk factors
- Gap risk in hedging strategies

The RRAO is a notional-based charge scaled by risk factors determined by:
- Product complexity
- Market liquidity of underlyings
- Hedging effectiveness

References:
    - BCBS d352: Minimum capital requirements for market risk (MAR21.93-21.98)
    - MAR21: Standardised approach - RRAO
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import jax.numpy as jnp


# ==============================================================================
# Enumerations
# ==============================================================================


class ExoticUnderlying(str, Enum):
    """Types of exotic underlyings subject to RRAO."""

    LONGEVITY = "longevity"  # Mortality/longevity risk
    WEATHER = "weather"  # Temperature, precipitation, wind
    NATURAL_CATASTROPHE = "nat_cat"  # Earthquakes, hurricanes
    FREIGHT = "freight"  # Shipping costs
    INFLATION = "inflation"  # CPI, RPI (if not modeled)
    VOLATILITY = "volatility"  # Realized/implied vol spread
    CORRELATION = "correlation"  # Correlation products
    DIVIDEND = "dividend"  # Single-stock dividends
    BASIS = "basis"  # Basis risk between related risk factors
    OTHER = "other"


class PayoffComplexity(str, Enum):
    """Complexity level of payoff structure."""

    LINEAR = "linear"  # Standard linear payoffs
    VANILLA_OPTION = "vanilla"  # European calls/puts
    EXOTIC_LOW = "exotic_low"  # Barriers, digitals (continuous monitoring)
    EXOTIC_MEDIUM = "exotic_medium"  # Path-dependent (Asians, lookbacks)
    EXOTIC_HIGH = "exotic_high"  # Multi-asset, correlation-dependent
    EXOTIC_VERY_HIGH = "exotic_very_high"  # Highly structured, discontinuous


class LiquidityClass(str, Enum):
    """Liquidity classification of underlying market."""

    LIQUID = "liquid"  # G7 rates, major FX, equity indices
    MODERATELY_LIQUID = "moderate"  # EM, small caps, some commodities
    ILLIQUID = "illiquid"  # Exotic commodities, bespoke structures
    VERY_ILLIQUID = "very_illiquid"  # Weather, longevity, nat cat


# ==============================================================================
# Data Structures
# ==============================================================================


@dataclass(frozen=True)
class RRAOExposure:
    """Exposure subject to RRAO.

    Attributes
    ----------
    instrument_id : str
        Unique identifier
    instrument_type : str
        Product type (e.g., "weather_derivative", "longevity_swap")
    underlying_type : ExoticUnderlying
        Type of exotic underlying
    payoff_complexity : PayoffComplexity
        Complexity of payoff structure
    liquidity_class : LiquidityClass
        Liquidity of the underlying market
    notional : float
        Notional amount or market value
    tenor_years : float
        Tenor in years
    is_hedged : bool
        Whether position has offsetting hedge
    hedge_effectiveness : float
        Hedge effectiveness ratio (0-1) if hedged
    risk_factor_override : Optional[float]
        Custom risk factor if not using standard calibration
    """

    instrument_id: str
    instrument_type: str
    underlying_type: ExoticUnderlying
    payoff_complexity: PayoffComplexity
    liquidity_class: LiquidityClass
    notional: float
    tenor_years: float
    is_hedged: bool = False
    hedge_effectiveness: float = 0.0
    risk_factor_override: Optional[float] = None


@dataclass(frozen=True)
class RRAOResult:
    """Result of RRAO calculation.

    Attributes
    ----------
    total_rrao : float
        Total residual risk add-on
    rrao_by_underlying : Dict[ExoticUnderlying, float]
        Breakdown by underlying type
    rrao_by_complexity : Dict[PayoffComplexity, float]
        Breakdown by payoff complexity
    gross_notional : float
        Total gross notional exposed to residual risk
    net_notional : float
        Net notional after hedging offsets
    """

    total_rrao: float
    rrao_by_underlying: Dict[ExoticUnderlying, float]
    rrao_by_complexity: Dict[PayoffComplexity, float]
    gross_notional: float
    net_notional: float


# ==============================================================================
# Risk Factor Calibrations
# ==============================================================================


# Base risk factors by underlying type (% of notional)
BASE_RISK_FACTORS: Dict[ExoticUnderlying, float] = {
    ExoticUnderlying.LONGEVITY: 0.10,  # 10% of notional
    ExoticUnderlying.WEATHER: 0.15,  # 15%
    ExoticUnderlying.NATURAL_CATASTROPHE: 0.20,  # 20%
    ExoticUnderlying.FREIGHT: 0.08,  # 8%
    ExoticUnderlying.INFLATION: 0.05,  # 5% (if not in delta)
    ExoticUnderlying.VOLATILITY: 0.12,  # 12%
    ExoticUnderlying.CORRELATION: 0.10,  # 10%
    ExoticUnderlying.DIVIDEND: 0.06,  # 6%
    ExoticUnderlying.BASIS: 0.04,  # 4%
    ExoticUnderlying.OTHER: 0.10,  # 10% default
}

# Multipliers by payoff complexity
COMPLEXITY_MULTIPLIERS: Dict[PayoffComplexity, float] = {
    PayoffComplexity.LINEAR: 1.0,
    PayoffComplexity.VANILLA_OPTION: 1.0,
    PayoffComplexity.EXOTIC_LOW: 1.25,  # +25% for barriers, digitals
    PayoffComplexity.EXOTIC_MEDIUM: 1.50,  # +50% for path-dependent
    PayoffComplexity.EXOTIC_HIGH: 1.75,  # +75% for multi-asset
    PayoffComplexity.EXOTIC_VERY_HIGH: 2.00,  # 2x for highly structured
}

# Multipliers by liquidity class
LIQUIDITY_MULTIPLIERS: Dict[LiquidityClass, float] = {
    LiquidityClass.LIQUID: 0.50,  # Reduce by 50% for liquid markets
    LiquidityClass.MODERATELY_LIQUID: 1.00,  # Base case
    LiquidityClass.ILLIQUID: 1.50,  # +50% for illiquid
    LiquidityClass.VERY_ILLIQUID: 2.00,  # 2x for very illiquid
}

# Tenor adjustment (longer tenor = higher risk)
def tenor_adjustment(tenor_years: float) -> float:
    """Calculate tenor adjustment factor.

    Parameters
    ----------
    tenor_years : float
        Tenor in years

    Returns
    -------
    float
        Adjustment factor (1.0 = 1 year baseline)
    """
    # Square root of time scaling
    return float(jnp.sqrt(jnp.maximum(tenor_years, 0.1)))


# ==============================================================================
# RRAO Calculator
# ==============================================================================


class FRTBResidualRiskAddOn:
    """Calculate Residual Risk Add-On under FRTB.

    The RRAO applies to positions with:
    1. Exotic underlyings not covered by standard risk factors
    2. Exotic payoffs with discontinuities or gaps
    3. Basis risk between related but non-identical factors
    4. Correlation risk not captured by delta/vega/curvature

    Calculation:
        RRAO = Σ (Notional × Base_Risk_Factor × Complexity_Mult ×
                  Liquidity_Mult × Tenor_Adj × (1 - Hedge_Effectiveness))
    """

    def __init__(
        self,
        base_risk_factors: Optional[Dict[ExoticUnderlying, float]] = None,
        complexity_multipliers: Optional[Dict[PayoffComplexity, float]] = None,
        liquidity_multipliers: Optional[Dict[LiquidityClass, float]] = None,
        netting_factor: float = 0.8,
    ):
        """Initialize RRAO calculator.

        Parameters
        ----------
        base_risk_factors : Optional[Dict[ExoticUnderlying, float]]
            Base risk factors by underlying type
        complexity_multipliers : Optional[Dict[PayoffComplexity, float]]
            Multipliers by payoff complexity
        liquidity_multipliers : Optional[Dict[LiquidityClass, float]]
            Multipliers by liquidity class
        netting_factor : float
            Factor for recognizing offsets (0.8 = 20% netting)
        """
        self.base_risk_factors = base_risk_factors or BASE_RISK_FACTORS
        self.complexity_multipliers = complexity_multipliers or COMPLEXITY_MULTIPLIERS
        self.liquidity_multipliers = liquidity_multipliers or LIQUIDITY_MULTIPLIERS
        self.netting_factor = netting_factor

    def calculate(self, exposures: List[RRAOExposure]) -> RRAOResult:
        """Calculate total RRAO.

        Parameters
        ----------
        exposures : List[RRAOExposure]
            Exposures subject to residual risk

        Returns
        -------
        RRAOResult
            Residual risk add-on components
        """
        if not exposures:
            return RRAOResult(
                total_rrao=0.0,
                rrao_by_underlying={},
                rrao_by_complexity={},
                gross_notional=0.0,
                net_notional=0.0,
            )

        # Calculate individual RRAOs
        individual_rraos = []
        gross_notional = 0.0
        net_notional = 0.0

        rrao_by_underlying: Dict[ExoticUnderlying, float] = {}
        rrao_by_complexity: Dict[PayoffComplexity, float] = {}

        for exp in exposures:
            rrao = self._calculate_single_rrao(exp)
            individual_rraos.append(rrao)

            # Accumulate notional
            gross_notional += abs(exp.notional)
            net_notional += abs(exp.notional) * (1.0 - exp.hedge_effectiveness)

            # Breakdown by underlying
            rrao_by_underlying[exp.underlying_type] = (
                rrao_by_underlying.get(exp.underlying_type, 0.0) + rrao
            )

            # Breakdown by complexity
            rrao_by_complexity[exp.payoff_complexity] = (
                rrao_by_complexity.get(exp.payoff_complexity, 0.0) + rrao
            )

        # Total RRAO with limited netting
        total_rrao = self._aggregate_rraos(individual_rraos)

        return RRAOResult(
            total_rrao=total_rrao,
            rrao_by_underlying=rrao_by_underlying,
            rrao_by_complexity=rrao_by_complexity,
            gross_notional=gross_notional,
            net_notional=net_notional,
        )

    def _calculate_single_rrao(self, exposure: RRAOExposure) -> float:
        """Calculate RRAO for a single exposure.

        RRAO = Notional × Risk_Factor × Multipliers × (1 - Hedge_Effectiveness)
        """
        # Use custom risk factor if provided, otherwise lookup
        if exposure.risk_factor_override is not None:
            base_rf = exposure.risk_factor_override
        else:
            base_rf = self.base_risk_factors[exposure.underlying_type]

        # Get multipliers
        complexity_mult = self.complexity_multipliers[exposure.payoff_complexity]
        liquidity_mult = self.liquidity_multipliers[exposure.liquidity_class]
        tenor_adj = tenor_adjustment(exposure.tenor_years)

        # Hedge adjustment
        hedge_factor = 1.0 - exposure.hedge_effectiveness if exposure.is_hedged else 1.0

        # Calculate RRAO
        rrao = (
            abs(exposure.notional)
            * base_rf
            * complexity_mult
            * liquidity_mult
            * tenor_adj
            * hedge_factor
        )

        return rrao

    def _aggregate_rraos(self, individual_rraos: List[float]) -> float:
        """Aggregate individual RRAOs with partial netting.

        Uses simple linear aggregation with netting factor to recognize
        some offset potential between different residual risks.
        """
        if not individual_rraos:
            return 0.0

        # Simple sum with netting factor
        gross_rrao = sum(individual_rraos)
        netted_rrao = gross_rrao * self.netting_factor

        return netted_rrao


# ==============================================================================
# Utility Functions
# ==============================================================================


def classify_payoff_complexity(
    has_barrier: bool = False,
    has_digital: bool = False,
    is_path_dependent: bool = False,
    is_multi_asset: bool = False,
    is_correlation_dependent: bool = False,
) -> PayoffComplexity:
    """Classify payoff complexity based on features.

    Parameters
    ----------
    has_barrier : bool
        Has barrier feature
    has_digital : bool
        Has digital/binary payoff
    is_path_dependent : bool
        Path-dependent (Asian, lookback)
    is_multi_asset : bool
        Depends on multiple assets
    is_correlation_dependent : bool
        Depends on correlation

    Returns
    -------
    PayoffComplexity
        Classified complexity level
    """
    if is_correlation_dependent or (is_multi_asset and is_path_dependent):
        return PayoffComplexity.EXOTIC_VERY_HIGH
    elif is_multi_asset or (is_path_dependent and (has_barrier or has_digital)):
        return PayoffComplexity.EXOTIC_HIGH
    elif is_path_dependent:
        return PayoffComplexity.EXOTIC_MEDIUM
    elif has_barrier or has_digital:
        return PayoffComplexity.EXOTIC_LOW
    else:
        return PayoffComplexity.VANILLA_OPTION


def estimate_hedge_effectiveness(
    hedge_pnl_correlation: float,
    hedge_ratio: float = 1.0,
) -> float:
    """Estimate hedge effectiveness from historical P&L correlation.

    Parameters
    ----------
    hedge_pnl_correlation : float
        Correlation between position and hedge P&L (0-1)
    hedge_ratio : float
        Notional ratio of hedge to position

    Returns
    -------
    float
        Hedge effectiveness (0-1)
    """
    # Effectiveness = min(correlation × hedge_ratio, 1.0)
    effectiveness = min(abs(hedge_pnl_correlation) * min(hedge_ratio, 1.0), 1.0)
    return effectiveness


def calculate_basis_risk_rrao(
    notional: float,
    basis_volatility: float,
    tenor_years: float = 1.0,
) -> float:
    """Calculate RRAO for basis risk.

    Parameters
    ----------
    notional : float
        Notional amount
    basis_volatility : float
        Annualized volatility of basis (e.g., 0.05 = 5%)
    tenor_years : float
        Tenor in years

    Returns
    -------
    float
        RRAO charge for basis risk
    """
    # Simplified: RRAO = Notional × Basis_Vol × sqrt(Tenor) × 2.33 (99th percentile)
    confidence_factor = 2.33  # 99% VaR
    rrao = notional * basis_volatility * jnp.sqrt(tenor_years) * confidence_factor
    return float(rrao)


__all__ = [
    # Enums
    "ExoticUnderlying",
    "PayoffComplexity",
    "LiquidityClass",
    # Data Structures
    "RRAOExposure",
    "RRAOResult",
    # Calculator
    "FRTBResidualRiskAddOn",
    # Utility Functions
    "classify_payoff_complexity",
    "estimate_hedge_effectiveness",
    "calculate_basis_risk_rrao",
    "tenor_adjustment",
    # Constants
    "BASE_RISK_FACTORS",
    "COMPLEXITY_MULTIPLIERS",
    "LIQUIDITY_MULTIPLIERS",
]
