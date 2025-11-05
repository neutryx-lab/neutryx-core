"""Expected Shortfall (ES) calculation for Internal Models Approach (IMA).

Expected Shortfall (also known as Conditional VaR or CVaR) is a coherent risk measure
used in the Basel III market risk framework. It measures the expected loss given that
the loss exceeds the VaR threshold.

For IMA under Basel III:
- Confidence level: 97.5% (tail probability: 2.5%)
- Liquidity horizons: 10 days to 1 year depending on risk factor
- Full revaluation required (not delta-gamma approximations)

References
----------
- Basel Committee on Banking Supervision (2016). "Minimum capital requirements for
  market risk" (FRTB standard)
- Basel Committee on Banking Supervision (2019). "Explanatory note on the minimum
  capital requirements for market risk"
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from ...core.engine import MCConfig


class LiquidityHorizon(str, Enum):
    """Liquidity horizons for IMA risk factors (Basel III)."""
    DAYS_10 = "10d"      # Most liquid: G10 rates, FX, equity indices
    DAYS_20 = "20d"      # Liquid: Other rates, FX, major equity names
    DAYS_40 = "40d"      # Less liquid: Credit spreads, vol surfaces
    DAYS_60 = "60d"      # Illiquid: Emerging markets, small caps
    DAYS_120 = "120d"    # Very illiquid: Structured products
    DAYS_250 = "250d"    # Least liquid: Bespoke derivatives, private assets

    def to_days(self) -> int:
        """Convert to number of days."""
        return int(self.value.replace('d', ''))


class RiskFactorCategory(str, Enum):
    """Risk factor categories for liquidity horizon mapping."""
    # Interest rates
    RATES_G10 = "rates_g10"
    RATES_EM = "rates_em"

    # Foreign exchange
    FX_G10 = "fx_g10"
    FX_EM = "fx_em"

    # Equities
    EQUITY_LARGE_CAP = "equity_large_cap"
    EQUITY_SMALL_CAP = "equity_small_cap"
    EQUITY_INDEX = "equity_index"

    # Credit
    CREDIT_IG = "credit_ig"
    CREDIT_HY = "credit_hy"
    CREDIT_SOVEREIGN = "credit_sovereign"

    # Commodities
    COMMODITY_ENERGY = "commodity_energy"
    COMMODITY_METALS = "commodity_metals"
    COMMODITY_AGRICULTURE = "commodity_agriculture"

    # Volatility
    VOLATILITY = "volatility"

    # Other
    STRUCTURED = "structured"
    BESPOKE = "bespoke"


# Default liquidity horizon mapping (Basel III guidelines)
DEFAULT_LIQUIDITY_HORIZONS: Dict[RiskFactorCategory, LiquidityHorizon] = {
    # Interest rates
    RiskFactorCategory.RATES_G10: LiquidityHorizon.DAYS_10,
    RiskFactorCategory.RATES_EM: LiquidityHorizon.DAYS_40,

    # FX
    RiskFactorCategory.FX_G10: LiquidityHorizon.DAYS_10,
    RiskFactorCategory.FX_EM: LiquidityHorizon.DAYS_40,

    # Equities
    RiskFactorCategory.EQUITY_LARGE_CAP: LiquidityHorizon.DAYS_20,
    RiskFactorCategory.EQUITY_SMALL_CAP: LiquidityHorizon.DAYS_60,
    RiskFactorCategory.EQUITY_INDEX: LiquidityHorizon.DAYS_10,

    # Credit
    RiskFactorCategory.CREDIT_IG: LiquidityHorizon.DAYS_40,
    RiskFactorCategory.CREDIT_HY: LiquidityHorizon.DAYS_60,
    RiskFactorCategory.CREDIT_SOVEREIGN: LiquidityHorizon.DAYS_20,

    # Commodities
    RiskFactorCategory.COMMODITY_ENERGY: LiquidityHorizon.DAYS_20,
    RiskFactorCategory.COMMODITY_METALS: LiquidityHorizon.DAYS_20,
    RiskFactorCategory.COMMODITY_AGRICULTURE: LiquidityHorizon.DAYS_60,

    # Volatility and other
    RiskFactorCategory.VOLATILITY: LiquidityHorizon.DAYS_60,
    RiskFactorCategory.STRUCTURED: LiquidityHorizon.DAYS_120,
    RiskFactorCategory.BESPOKE: LiquidityHorizon.DAYS_250,
}


def get_liquidity_horizon(asset_class: str, sub_class: str) -> LiquidityHorizon:
    """Get liquidity horizon for a given asset class and sub-class.

    This is a simplified helper function for mapping asset classes to liquidity
    horizons per Basel III guidelines.

    Args:
        asset_class: Asset class (equity, rates, fx, credit, commodity)
        sub_class: Sub-classification within asset class

    Returns:
        LiquidityHorizon enum value

    Examples:
        >>> get_liquidity_horizon("equity", "large_cap")
        LiquidityHorizon.DAYS_10
        >>> get_liquidity_horizon("credit", "hy_single")
        LiquidityHorizon.DAYS_120
    """
    # Mapping based on Basel III FRTB guidelines
    mappings = {
        # Equity
        ("equity", "large_cap"): LiquidityHorizon.DAYS_10,
        ("equity", "small_cap"): LiquidityHorizon.DAYS_20,
        ("equity", "emerging"): LiquidityHorizon.DAYS_40,
        ("equity", "other"): LiquidityHorizon.DAYS_20,

        # Interest rates
        ("rates", "major"): LiquidityHorizon.DAYS_10,
        ("rates", "other"): LiquidityHorizon.DAYS_20,
        ("rates", "emerging"): LiquidityHorizon.DAYS_40,

        # FX
        ("fx", "major_pairs"): LiquidityHorizon.DAYS_10,
        ("fx", "other"): LiquidityHorizon.DAYS_40,

        # Credit
        ("credit", "ig_index"): LiquidityHorizon.DAYS_20,
        ("credit", "ig_single"): LiquidityHorizon.DAYS_40,
        ("credit", "hy_index"): LiquidityHorizon.DAYS_60,
        ("credit", "hy_single"): LiquidityHorizon.DAYS_120,
        ("credit", "structured"): LiquidityHorizon.DAYS_120,

        # Commodities
        ("commodity", "energy"): LiquidityHorizon.DAYS_20,
        ("commodity", "metals"): LiquidityHorizon.DAYS_20,
        ("commodity", "agriculture"): LiquidityHorizon.DAYS_60,
        ("commodity", "electricity"): LiquidityHorizon.DAYS_250,
    }

    # Look up the mapping
    key = (asset_class.lower(), sub_class.lower())
    if key in mappings:
        return mappings[key]

    # Default to 20-day horizon for unknown combinations
    return LiquidityHorizon.DAYS_20


@dataclass
class ESResult:
    """Expected Shortfall calculation result."""

    # Core ES metrics
    expected_shortfall: float
    var_97_5: float  # VaR at 97.5% for reference
    confidence_level: float = 0.975

    # Additional statistics
    tail_observations: int = 0
    mean_excess_loss: float = 0.0
    max_loss: float = 0.0

    # Liquidity horizon adjustments
    base_horizon_days: int = 10
    adjusted_horizon_days: Optional[int] = None
    horizon_adjustment_factor: float = 1.0

    # Backtesting statistics
    num_scenarios: int = 0
    historical_window_days: int = 0

    # Metadata
    calculation_timestamp: Optional[str] = None
    risk_factors: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "expected_shortfall": self.expected_shortfall,
            "var_97_5": self.var_97_5,
            "confidence_level": self.confidence_level,
            "tail_observations": self.tail_observations,
            "mean_excess_loss": self.mean_excess_loss,
            "max_loss": self.max_loss,
            "base_horizon_days": self.base_horizon_days,
            "adjusted_horizon_days": self.adjusted_horizon_days,
            "horizon_adjustment_factor": self.horizon_adjustment_factor,
            "num_scenarios": self.num_scenarios,
            "calculation_timestamp": self.calculation_timestamp,
        }


def calculate_expected_shortfall(
    pnl_scenarios: Array,
    confidence_level: float = 0.975,
    return_var: bool = True
) -> Tuple[float, Optional[float], Dict[str, Any]]:
    """Calculate Expected Shortfall (ES) from P&L scenarios.

    ES is the average of losses in the tail beyond the VaR threshold.
    For a 97.5% confidence level, ES is the average of the worst 2.5% outcomes.

    Parameters
    ----------
    pnl_scenarios : Array
        P&L scenarios, shape (n_scenarios,)
        Losses should be negative values
    confidence_level : float, optional
        Confidence level (default: 0.975 for Basel III)
    return_var : bool, optional
        Whether to also return VaR

    Returns
    -------
    es : float
        Expected Shortfall (positive value indicates loss)
    var : float or None
        Value at Risk (if return_var=True)
    stats : dict
        Additional statistics

    Examples
    --------
    >>> pnl = jnp.array([-100, -50, -10, 0, 10, 50, 100])  # Losses are negative
    >>> es, var, stats = calculate_expected_shortfall(pnl, confidence_level=0.95)
    >>> print(f"ES: {es}, VaR: {var}")
    """
    # Ensure losses are represented as negative values
    # Sort in ascending order (most negative first)
    sorted_pnl = jnp.sort(pnl_scenarios)
    n_scenarios = len(sorted_pnl)

    # Calculate VaR threshold
    # For 97.5% confidence, we look at the 2.5% quantile (worst outcomes)
    alpha = 1 - confidence_level
    var_index = int(jnp.ceil(alpha * n_scenarios)) - 1
    var_index = jnp.maximum(0, var_index)

    var_threshold = sorted_pnl[var_index]

    # Calculate ES: average of losses beyond VaR threshold
    # All losses worse than (more negative than) VaR
    tail_scenarios = sorted_pnl[:(var_index + 1)]
    expected_shortfall = jnp.mean(tail_scenarios)

    # Convert to positive values for reporting (ES convention)
    es_value = float(-expected_shortfall)
    var_value = float(-var_threshold) if return_var else None

    # Calculate additional statistics
    tail_count = len(tail_scenarios)
    max_loss_value = float(-sorted_pnl[0])  # Most negative = biggest loss

    # Mean excess loss beyond VaR
    if tail_count > 0:
        mean_excess = float(-jnp.mean(tail_scenarios - var_threshold))
    else:
        mean_excess = 0.0

    stats = {
        "tail_observations": tail_count,
        "mean_excess_loss": mean_excess,
        "max_loss": max_loss_value,
        "num_scenarios": n_scenarios,
    }

    return es_value, var_value, stats


def calculate_es_with_liquidity_adjustment(
    pnl_scenarios: Array,
    risk_factor_categories: List[RiskFactorCategory],
    base_horizon_days: int = 10,
    confidence_level: float = 0.975,
    custom_horizons: Optional[Dict[RiskFactorCategory, LiquidityHorizon]] = None
) -> ESResult:
    """Calculate ES with liquidity horizon adjustments per Basel III.

    Basel III requires scaling ES to appropriate liquidity horizons for different
    risk factor categories. The adjustment uses square-root-of-time scaling:

    ES(T) = ES(10d) * sqrt(T / 10)

    Parameters
    ----------
    pnl_scenarios : Array
        P&L scenarios at base horizon
    risk_factor_categories : List[RiskFactorCategory]
        Risk factor categories in the portfolio
    base_horizon_days : int, optional
        Base horizon for scenarios (typically 10 days)
    confidence_level : float, optional
        Confidence level for ES calculation
    custom_horizons : dict, optional
        Custom liquidity horizon mapping (overrides defaults)

    Returns
    -------
    ESResult
        ES calculation with liquidity adjustments
    """
    # Calculate base ES
    es_base, var_base, stats = calculate_expected_shortfall(
        pnl_scenarios, confidence_level=confidence_level
    )

    # Determine target liquidity horizon
    horizon_map = custom_horizons or DEFAULT_LIQUIDITY_HORIZONS

    # Take the longest liquidity horizon applicable to the portfolio
    applicable_horizons = [
        horizon_map.get(cat, LiquidityHorizon.DAYS_60)
        for cat in risk_factor_categories
    ]

    max_horizon = max(
        [h.to_days() for h in applicable_horizons],
        default=base_horizon_days
    )

    # Apply liquidity horizon scaling (sqrt of time)
    if max_horizon != base_horizon_days:
        adjustment_factor = jnp.sqrt(max_horizon / base_horizon_days)
        es_adjusted = es_base * float(adjustment_factor)
        var_adjusted = var_base * float(adjustment_factor) if var_base else None
    else:
        adjustment_factor = 1.0
        es_adjusted = es_base
        var_adjusted = var_base

    # Create result
    result = ESResult(
        expected_shortfall=es_adjusted,
        var_97_5=var_adjusted or 0.0,
        confidence_level=confidence_level,
        tail_observations=stats["tail_observations"],
        mean_excess_loss=stats["mean_excess_loss"],
        max_loss=stats["max_loss"],
        base_horizon_days=base_horizon_days,
        adjusted_horizon_days=max_horizon,
        horizon_adjustment_factor=float(adjustment_factor),
        num_scenarios=stats["num_scenarios"],
    )

    return result


def calculate_stressed_es(
    pnl_scenarios: Array,
    stress_period_indices: Optional[Array] = None,
    confidence_level: float = 0.975
) -> Tuple[float, ESResult, ESResult]:
    """Calculate both standard and stressed ES per Basel III requirements.

    Basel III requires calculating ES under both:
    1. Current market conditions (12-month rolling window)
    2. Stressed market conditions (period of significant stress)

    The stressed ES is typically from a historical stress period identified
    by the bank (e.g., 2008 financial crisis, COVID-19, etc.)

    Parameters
    ----------
    pnl_scenarios : Array
        P&L scenarios, shape (n_scenarios,)
    stress_period_indices : Array, optional
        Boolean mask or indices indicating stress period scenarios
        If None, uses worst 20% of scenarios as stress period
    confidence_level : float, optional
        Confidence level

    Returns
    -------
    total_es : float
        Total ES requirement (max of standard and stressed ES)
    standard_es : ESResult
        Standard ES result
    stressed_es : ESResult
        Stressed ES result
    """
    # Calculate standard ES (all scenarios)
    es_std, var_std, stats_std = calculate_expected_shortfall(
        pnl_scenarios, confidence_level=confidence_level
    )

    standard_result = ESResult(
        expected_shortfall=es_std,
        var_97_5=var_std or 0.0,
        confidence_level=confidence_level,
        tail_observations=stats_std["tail_observations"],
        mean_excess_loss=stats_std["mean_excess_loss"],
        max_loss=stats_std["max_loss"],
        num_scenarios=stats_std["num_scenarios"],
    )

    # Identify stress period scenarios
    if stress_period_indices is None:
        # Default: use worst 20% of scenarios as "stress period"
        n_scenarios = len(pnl_scenarios)
        n_stress = int(0.2 * n_scenarios)
        sorted_indices = jnp.argsort(pnl_scenarios)
        stress_scenarios = pnl_scenarios[sorted_indices[:n_stress]]
    else:
        if stress_period_indices.dtype == jnp.bool_:
            stress_scenarios = pnl_scenarios[stress_period_indices]
        else:
            stress_scenarios = pnl_scenarios[stress_period_indices]

    # Calculate stressed ES
    if len(stress_scenarios) > 0:
        es_stress, var_stress, stats_stress = calculate_expected_shortfall(
            stress_scenarios, confidence_level=confidence_level
        )

        stressed_result = ESResult(
            expected_shortfall=es_stress,
            var_97_5=var_stress or 0.0,
            confidence_level=confidence_level,
            tail_observations=stats_stress["tail_observations"],
            mean_excess_loss=stats_stress["mean_excess_loss"],
            max_loss=stats_stress["max_loss"],
            num_scenarios=stats_stress["num_scenarios"],
        )
    else:
        # No stress scenarios available
        stressed_result = standard_result

    # Basel III requires the higher of standard and stressed ES
    total_es = max(es_std, stressed_result.expected_shortfall)

    return total_es, standard_result, stressed_result


@jax.jit
def expected_shortfall_jax(
    pnl_scenarios: Array,
    confidence_level: float = 0.975
) -> Tuple[Array, Array]:
    """JAX-optimized Expected Shortfall calculation.

    Parameters
    ----------
    pnl_scenarios : Array
        P&L scenarios (losses as negative values)
    confidence_level : float
        Confidence level

    Returns
    -------
    es : Array
        Expected Shortfall
    var : Array
        Value at Risk
    """
    sorted_pnl = jnp.sort(pnl_scenarios)
    n_scenarios = len(sorted_pnl)

    alpha = 1 - confidence_level
    var_index = jnp.maximum(0, int(jnp.ceil(alpha * n_scenarios)) - 1)

    var_threshold = sorted_pnl[var_index]
    tail_scenarios = sorted_pnl[:(var_index + 1)]
    expected_shortfall = jnp.mean(tail_scenarios)

    # Return as positive values (loss convention)
    return -expected_shortfall, -var_threshold


__all__ = [
    "LiquidityHorizon",
    "RiskFactorCategory",
    "ESResult",
    "calculate_expected_shortfall",
    "calculate_es_with_liquidity_adjustment",
    "calculate_stressed_es",
    "expected_shortfall_jax",
    "DEFAULT_LIQUIDITY_HORIZONS",
]
