"""
Credit Valuation Adjustment (CVA) calculations.

Enhanced with multi-currency support and MarketDataEnvironment integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import jax.numpy as jnp
from jax import Array

# Optional market environment import
try:
    from neutryx.market.environment import MarketDataEnvironment
    _HAS_MARKET_ENV = True
except ImportError:
    MarketDataEnvironment = None  # type: ignore
    _HAS_MARKET_ENV = False


def cva(epe_t: Array, df_t: Array, pd_t: Array, lgd: float = 0.6) -> float:
    """
    Calculate single-currency Credit Valuation Adjustment (CVA).

    CVA represents the expected loss due to counterparty default:
        CVA = LGD * Σ DF(t) * EPE(t) * ΔPD(t)

    Args:
        epe_t: Expected Positive Exposure at each time point
        df_t: Discount factors at each time point
        pd_t: Cumulative default probabilities at each time point
        lgd: Loss Given Default (default 0.6 = 60%)

    Returns:
        CVA value

    Example:
        >>> times = jnp.array([0.5, 1.0, 1.5, 2.0])
        >>> epe = jnp.array([100, 120, 110, 95])
        >>> df = jnp.array([0.975, 0.951, 0.928, 0.905])
        >>> pd = jnp.array([0.01, 0.02, 0.03, 0.04])
        >>> cva_value = cva(epe, df, pd, lgd=0.6)
    """
    # Discrete sum over time buckets: sum DF(t) * EPE(t) * dPD(t) * LGD
    dPD = jnp.diff(jnp.concatenate([jnp.array([0.0]), pd_t]))
    return float((df_t * epe_t * dPD * lgd).sum())


def dva(ene_t: Array, df_t: Array, pd_t: Array, lgd: float = 0.6) -> float:
    """
    Calculate single-currency Debit Valuation Adjustment (DVA).

    DVA represents the benefit from the possibility of own default:
        DVA = LGD * Σ DF(t) * ENE(t) * ΔPD(t)

    Args:
        ene_t: Expected Negative Exposure at each time point
        df_t: Discount factors at each time point
        pd_t: Cumulative default probabilities (own entity) at each time point
        lgd: Loss Given Default (default 0.6 = 60%)

    Returns:
        DVA value
    """
    dPD = jnp.diff(jnp.concatenate([jnp.array([0.0]), pd_t]))
    return float((df_t * ene_t * dPD * lgd).sum())


def bilateral_cva(
    epe_t: Array,
    ene_t: Array,
    df_t: Array,
    pd_counterparty_t: Array,
    pd_own_t: Array,
    lgd_counterparty: float = 0.6,
    lgd_own: float = 0.6
) -> tuple[float, float, float]:
    """
    Calculate bilateral CVA (CVA - DVA).

    Args:
        epe_t: Expected Positive Exposure
        ene_t: Expected Negative Exposure
        df_t: Discount factors
        pd_counterparty_t: Counterparty default probabilities
        pd_own_t: Own default probabilities
        lgd_counterparty: Counterparty LGD
        lgd_own: Own LGD

    Returns:
        Tuple of (CVA, DVA, Bilateral CVA = CVA - DVA)
    """
    cva_value = cva(epe_t, df_t, pd_counterparty_t, lgd_counterparty)
    dva_value = dva(ene_t, df_t, pd_own_t, lgd_own)
    bilateral = cva_value - dva_value
    return cva_value, dva_value, bilateral


@dataclass
class MultiCurrencyCVA:
    """
    Multi-currency CVA calculator with FX exposure.

    Handles CVA calculation for portfolios with multiple currencies,
    properly accounting for FX exposure and collateral currency.

    Attributes:
        collateral_currency: Currency for CVA reporting (e.g., "USD")
        counterparty_name: Counterparty identifier
        lgd: Loss Given Default

    Example:
        >>> from datetime import date
        >>> from neutryx.market import MarketDataEnvironment, FlatCurve
        >>> env = MarketDataEnvironment(
        ...     reference_date=date(2024, 1, 1),
        ...     discount_curves={'USD': FlatCurve(0.05), 'EUR': FlatCurve(0.02)},
        ...     fx_spots={('EUR', 'USD'): 1.10}
        ... )
        >>> calculator = MultiCurrencyCVA(
        ...     collateral_currency='USD',
        ...     counterparty_name='BANK_A',
        ...     lgd=0.6
        ... )
        >>> cva_value = calculator.calculate(env, exposures_by_ccy, times, pd_t)
    """

    collateral_currency: str
    counterparty_name: str = "COUNTERPARTY"
    lgd: float = 0.6

    def calculate(
        self,
        market_env: MarketDataEnvironment,
        exposures_by_currency: Dict[str, Array],
        times: Array,
        pd_t: Array,
        fx_exposures: Optional[Dict[tuple[str, str], Array]] = None
    ) -> float:
        """
        Calculate multi-currency CVA.

        Args:
            market_env: Market data environment with curves and FX rates
            exposures_by_currency: Dict mapping currency to EPE time series
            times: Time grid for exposures
            pd_t: Counterparty default probabilities at each time
            fx_exposures: Optional FX option exposures by currency pair

        Returns:
            Total CVA in collateral currency
        """
        total_cva = 0.0

        # CVA for each currency exposure
        for currency, epe_t in exposures_by_currency.items():
            if currency == self.collateral_currency:
                # Same currency: no FX conversion needed
                df_t = jnp.array([
                    market_env.get_discount_factor(currency, t)
                    for t in times
                ])
                cva_ccy = cva(epe_t, df_t, pd_t, self.lgd)
                total_cva += cva_ccy
            else:
                # Foreign currency: convert to collateral currency
                # EPE in foreign CCY -> EPE in collateral CCY using forward FX rates
                fx_forwards = jnp.array([
                    market_env.get_fx_forward(currency, self.collateral_currency, t)
                    for t in times
                ])
                epe_collateral = epe_t * fx_forwards

                # Discount in collateral currency
                df_t = jnp.array([
                    market_env.get_discount_factor(self.collateral_currency, t)
                    for t in times
                ])

                cva_ccy = cva(epe_collateral, df_t, pd_t, self.lgd)
                total_cva += cva_ccy

        # Add FX option exposure if present
        if fx_exposures:
            for (from_ccy, to_ccy), epe_fx in fx_exposures.items():
                # FX exposures are typically already in one of the currencies
                # Convert to collateral currency if needed
                if to_ccy == self.collateral_currency:
                    # Already in collateral currency
                    epe_collateral = epe_fx
                else:
                    # Need to convert
                    fx_forwards = jnp.array([
                        market_env.get_fx_forward(to_ccy, self.collateral_currency, t)
                        for t in times
                    ])
                    epe_collateral = epe_fx * fx_forwards

                df_t = jnp.array([
                    market_env.get_discount_factor(self.collateral_currency, t)
                    for t in times
                ])

                cva_fx = cva(epe_collateral, df_t, pd_t, self.lgd)
                total_cva += cva_fx

        return float(total_cva)

    def calculate_bilateral(
        self,
        market_env: MarketDataEnvironment,
        exposures_by_currency: Dict[str, tuple[Array, Array]],  # (EPE, ENE) per currency
        times: Array,
        pd_counterparty_t: Array,
        pd_own_t: Array,
        lgd_own: Optional[float] = None
    ) -> tuple[float, float, float]:
        """
        Calculate bilateral multi-currency CVA (CVA - DVA).

        Args:
            market_env: Market data environment
            exposures_by_currency: Dict mapping currency to (EPE, ENE) tuples
            times: Time grid
            pd_counterparty_t: Counterparty default probabilities
            pd_own_t: Own default probabilities
            lgd_own: Own LGD (defaults to same as counterparty)

        Returns:
            Tuple of (CVA, DVA, Bilateral CVA)
        """
        if lgd_own is None:
            lgd_own = self.lgd

        # Calculate CVA (counterparty defaults)
        epe_dict = {ccy: epe for ccy, (epe, _) in exposures_by_currency.items()}
        cva_value = self.calculate(market_env, epe_dict, times, pd_counterparty_t)

        # Calculate DVA (we default)
        ene_dict = {ccy: ene for ccy, (_, ene) in exposures_by_currency.items()}

        # For DVA, use own LGD
        original_lgd = self.lgd
        self.lgd = lgd_own
        dva_value = self.calculate(market_env, ene_dict, times, pd_own_t)
        self.lgd = original_lgd

        bilateral = cva_value - dva_value
        return cva_value, dva_value, bilateral
