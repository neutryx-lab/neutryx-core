"""Capital Valuation Adjustment (KVA) calculation.

KVA represents the cost of holding regulatory capital to support the
counterparty credit risk of a derivative portfolio. It captures the
present value of the cost of funding the capital requirements over
the life of the trades.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import jax.numpy as jnp
from jax import Array


def kva(
    capital_profile: Array,
    df_t: Array,
    hurdle_rate: float = 0.10,
    tax_rate: float = 0.30
) -> float:
    """Calculate Capital Valuation Adjustment (KVA).

    KVA quantifies the cost of holding regulatory capital for counterparty
    credit risk. It represents the present value of the funding costs associated
    with the capital that must be held against the exposure.

    Args:
        capital_profile: Expected regulatory capital requirements at each time step
        df_t: Discount factors at each time step
        hurdle_rate: Return on capital required by shareholders (e.g., 10%)
        tax_rate: Corporate tax rate (e.g., 30%)

    Returns:
        Capital Valuation Adjustment (present value of capital costs)

    Notes:
        KVA = sum_t [ DF(t) * Capital(t) * HurdleRate * (1 - TaxRate) ]

        The formula adjusts for tax deductibility of funding costs.

        Regulatory capital can be calculated using:
        - SA-CCR (Standardized Approach for Counterparty Credit Risk)
        - IMM (Internal Model Method)
        - CVA capital charge under Basel III/IV

    Example:
        >>> import jax.numpy as jnp
        >>> # 10 time steps over 5 years
        >>> capital = jnp.array([1000, 1050, 1100, 1150, 1200,
        ...                      1180, 1150, 1100, 1050, 1000])
        >>> df = jnp.exp(-0.05 * jnp.linspace(0.5, 5.0, 10))
        >>> kva_value = kva(capital, df, hurdle_rate=0.12, tax_rate=0.25)
    """
    # After-tax cost of capital
    after_tax_hurdle = hurdle_rate * (1.0 - tax_rate)

    return float((df_t * capital_profile * after_tax_hurdle).sum())


def kva_with_cva_capital(
    epe_t: Array,
    df_t: Array,
    pd_t: Array,
    lgd: float = 0.6,
    capital_multiplier: float = 1.4,
    hurdle_rate: float = 0.10,
    tax_rate: float = 0.30
) -> tuple[float, Array]:
    """Calculate KVA using CVA-based capital approximation.

    This function estimates regulatory capital requirements based on
    the CVA exposure profile, then calculates KVA.

    Args:
        epe_t: Expected Positive Exposure at each time step
        df_t: Discount factors at each time step
        pd_t: Cumulative default probabilities at each time step
        lgd: Loss Given Default
        capital_multiplier: Regulatory multiplier for CVA capital (typically 1.4-2.0)
        hurdle_rate: Return on capital required
        tax_rate: Corporate tax rate

    Returns:
        Tuple of (KVA value, capital profile array)

    Notes:
        Capital(t) ≈ Multiplier × CVA_profile(t)

        This is a simplified approach. More sophisticated methods include:
        - Basel III CVA VaR capital charge
        - Basel IV SA-CVA (Standardized Approach)
        - IMM approach with Monte Carlo
    """
    # Compute incremental PD
    dPD = jnp.diff(jnp.concatenate([jnp.array([0.0]), pd_t]))

    # CVA profile at each time bucket
    cva_profile = df_t * epe_t * dPD * lgd

    # Capital approximation: cumulative CVA exposure × multiplier
    capital_profile = jnp.cumsum(cva_profile) * capital_multiplier

    # Calculate KVA
    kva_value = kva(capital_profile, df_t, hurdle_rate, tax_rate)

    return kva_value, capital_profile


@dataclass
class CapitalCalculator:
    """Calculate regulatory capital for counterparty credit risk.

    Supports multiple methodologies for computing capital requirements:
    - SA-CCR: Standardized Approach for Counterparty Credit Risk
    - BA-CVA: Basic Approach for CVA capital
    - Advanced approaches using PFE/EPE profiles

    Attributes:
        method: Capital calculation method ('sa-ccr', 'ba-cva', 'advanced')
        alpha: Supervisory alpha factor (typically 1.4)
        maturity_adjustment: Whether to apply maturity adjustments
    """

    method: str = "advanced"
    alpha: float = 1.4
    maturity_adjustment: bool = True

    def compute_ead(
        self,
        epe_profile: Array,
        pfe_profile: Optional[Array] = None
    ) -> Array:
        """Compute Exposure at Default (EAD) profile.

        Args:
            epe_profile: Expected Positive Exposure profile
            pfe_profile: Potential Future Exposure (e.g., 97.5th percentile)

        Returns:
            EAD profile for capital calculation

        Notes:
            Under IMM: EAD = alpha × Effective EPE
            Under SA-CCR: EAD = alpha × RC (Replacement Cost) + PFE
        """
        if pfe_profile is not None:
            # Use PFE if available (more conservative)
            ead = self.alpha * jnp.maximum(epe_profile, pfe_profile)
        else:
            # Use EPE with alpha multiplier
            ead = self.alpha * epe_profile

        return ead

    def compute_capital(
        self,
        epe_profile: Array,
        pfe_profile: Optional[Array] = None,
        risk_weight: float = 1.0,
        capital_ratio: float = 0.08
    ) -> Array:
        """Compute regulatory capital profile.

        Args:
            epe_profile: Expected Positive Exposure profile
            pfe_profile: Potential Future Exposure profile
            risk_weight: Risk weight for counterparty (0-1, default 1.0)
            capital_ratio: Minimum capital ratio (default 8%)

        Returns:
            Regulatory capital requirement profile

        Notes:
            Capital = EAD × RiskWeight × CapitalRatio
        """
        ead = self.compute_ead(epe_profile, pfe_profile)
        capital = ead * risk_weight * capital_ratio

        return capital

    def compute_cva_capital(
        self,
        epe_t: Array,
        df_t: Array,
        pd_t: Array,
        lgd: float = 0.6,
        method: str = "ba-cva"
    ) -> Array:
        """Compute CVA capital charge.

        Args:
            epe_t: Expected Positive Exposure
            df_t: Discount factors
            pd_t: Default probabilities
            lgd: Loss Given Default
            method: 'ba-cva' (Basic Approach) or 'sa-cva' (Standardized)

        Returns:
            CVA capital charge profile

        Notes:
            BA-CVA: Simple approach based on EAD
            SA-CVA: More risk-sensitive, uses credit spread sensitivities
        """
        if method == "ba-cva":
            # Basic Approach: CVA capital = 0.5 × h × EAD
            # where h is a supervisory factor (typically 1.4-2.0)
            ead = self.compute_ead(epe_t)
            h_factor = 1.4  # Supervisory factor
            cva_capital = 0.5 * h_factor * ead

        elif method == "sa-cva":
            # Standardized Approach: Based on credit spread sensitivities
            # This is a simplified version
            dPD = jnp.diff(jnp.concatenate([jnp.array([0.0]), pd_t]))
            cva_profile = df_t * epe_t * dPD * lgd

            # Capital based on credit spread sensitivity
            # Simplified: Capital ≈ CVA × RW_CVA
            rw_cva = 5.0  # Risk weight for CVA (simplified)
            cva_capital = jnp.abs(cva_profile) * rw_cva

        else:
            raise ValueError(f"Unknown CVA capital method: {method}")

        return cva_capital


def kva_advanced(
    epe_profile: Array,
    pfe_profile: Array,
    df_t: Array,
    pd_t: Array,
    lgd: float = 0.6,
    hurdle_rate: float = 0.10,
    tax_rate: float = 0.30,
    risk_weight: float = 1.0,
    include_cva_capital: bool = True
) -> tuple[float, Array, Array]:
    """Advanced KVA calculation with full capital treatment.

    Args:
        epe_profile: Expected Positive Exposure profile
        pfe_profile: Potential Future Exposure profile (e.g., 97.5%)
        df_t: Discount factors
        pd_t: Default probabilities
        lgd: Loss Given Default
        hurdle_rate: Return on capital required
        tax_rate: Corporate tax rate
        risk_weight: Counterparty risk weight
        include_cva_capital: Whether to include CVA capital charge

    Returns:
        Tuple of (KVA value, CCR capital profile, CVA capital profile)

    Notes:
        Total Capital = CCR Capital + CVA Capital

        CCR Capital: Capital for counterparty credit risk exposure
        CVA Capital: Capital for CVA risk (mark-to-market changes)
    """
    calculator = CapitalCalculator()

    # Counterparty credit risk capital
    ccr_capital = calculator.compute_capital(
        epe_profile,
        pfe_profile,
        risk_weight=risk_weight
    )

    # CVA capital charge
    if include_cva_capital:
        cva_capital = calculator.compute_cva_capital(
            epe_profile,
            df_t,
            pd_t,
            lgd=lgd
        )
    else:
        cva_capital = jnp.zeros_like(ccr_capital)

    # Total capital
    total_capital = ccr_capital + cva_capital

    # Calculate KVA
    kva_value = kva(total_capital, df_t, hurdle_rate, tax_rate)

    return kva_value, ccr_capital, cva_capital
