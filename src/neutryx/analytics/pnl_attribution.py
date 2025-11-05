"""P&L Attribution Framework for daily P&L explain and risk decomposition.

This module provides comprehensive P&L attribution capabilities including:
- Daily P&L explain decomposed by Greeks
- Risk factor attribution (rates, credit, FX, equity)
- Carry P&L, Delta P&L, Gamma P&L, Vega P&L, Theta decay
- Unexplained P&L tracking and analysis
- Model risk quantification

P&L attribution is essential for:
- Understanding portfolio performance drivers
- Risk management and control
- Trader P&L validation
- Regulatory reporting (e.g., FRTB P&L attribution test)
- Model validation

References
----------
Hull, J. C. (2018). "Risk Management and Financial Institutions."
Wiley Finance. (Chapter on P&L Attribution)

Basel Committee on Banking Supervision (2019). "Minimum capital requirements
for market risk." (P&L attribution test for internal models approach)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import jax.numpy as jnp
from jax import Array


@dataclass
class PLComponent:
    """Single P&L component from attribution.

    Attributes
    ----------
    name : str
        Component name (e.g., "Delta P&L", "Carry P&L")
    value : float
        P&L contribution in base currency
    risk_factor : Optional[str]
        Associated risk factor (e.g., "USD.3M", "AAPL")
    description : Optional[str]
        Human-readable description
    """
    name: str
    value: float
    risk_factor: Optional[str] = None
    description: Optional[str] = None


@dataclass
class DailyPLExplain:
    """Daily P&L explain decomposition.

    Attributes
    ----------
    date : str
        Business date (YYYY-MM-DD)
    total_pnl : float
        Total P&L for the day
    carry_pnl : float
        Carry P&L (theta decay, time value)
    delta_pnl : float
        P&L from delta (first-order price moves)
    gamma_pnl : float
        P&L from gamma (convexity)
    vega_pnl : float
        P&L from vega (volatility changes)
    theta_pnl : float
        P&L from theta (pure time decay)
    rho_pnl : float
        P&L from rho (interest rate sensitivity)
    unexplained_pnl : float
        Unexplained residual P&L
    components : List[PLComponent]
        Detailed component breakdown
    """
    date: str
    total_pnl: float
    carry_pnl: float
    delta_pnl: float
    gamma_pnl: float
    vega_pnl: float
    theta_pnl: float
    rho_pnl: float
    unexplained_pnl: float
    components: List[PLComponent]

    @property
    def explained_pnl(self) -> float:
        """Total explained P&L (sum of Greeks)."""
        return (self.carry_pnl + self.delta_pnl + self.gamma_pnl +
                self.vega_pnl + self.theta_pnl + self.rho_pnl)

    @property
    def explanation_ratio(self) -> float:
        """Ratio of explained to total P&L."""
        if abs(self.total_pnl) < 1e-10:
            return 1.0
        return self.explained_pnl / self.total_pnl

    def passes_attribution_test(self, threshold: float = 0.9) -> bool:
        """Check if P&L attribution passes regulatory test.

        Parameters
        ----------
        threshold : float, optional
            Minimum explanation ratio (default: 0.9 for Basel)

        Returns
        -------
        bool
            True if explanation ratio >= threshold
        """
        return abs(self.explanation_ratio) >= threshold


class GreekPLCalculator:
    """Calculate P&L attribution using Greeks.

    Uses Taylor expansion of portfolio value:
    ΔV ≈ Carry + Delta·ΔS + 0.5·Gamma·(ΔS)² + Vega·Δσ + Theta·Δt + Rho·Δr
    """

    def compute_carry_pnl(
        self,
        position_value_t0: float,
        position_value_t1: float,
        funding_cost: float = 0.0,
    ) -> float:
        """Compute carry P&L.

        Carry P&L includes:
        - Funding costs/benefits
        - Accrued interest
        - Dividend accruals

        Parameters
        ----------
        position_value_t0 : float
            Position value at T₀
        position_value_t1 : float
            Position value at T₁ (no market moves)
        funding_cost : float, optional
            Overnight funding cost

        Returns
        -------
        float
            Carry P&L
        """
        carry = position_value_t1 - position_value_t0 - funding_cost
        return float(carry)

    def compute_delta_pnl(
        self,
        delta: float,
        price_change: float,
    ) -> float:
        """Compute Delta P&L.

        Delta P&L = Delta × ΔS

        Parameters
        ----------
        delta : float
            Position delta
        price_change : float
            Change in underlying price

        Returns
        -------
        float
            Delta P&L
        """
        return float(delta * price_change)

    def compute_gamma_pnl(
        self,
        gamma: float,
        price_change: float,
    ) -> float:
        """Compute Gamma P&L.

        Gamma P&L = 0.5 × Gamma × (ΔS)²

        Parameters
        ----------
        gamma : float
            Position gamma
        price_change : float
            Change in underlying price

        Returns
        -------
        float
            Gamma P&L
        """
        return float(0.5 * gamma * price_change ** 2)

    def compute_vega_pnl(
        self,
        vega: float,
        vol_change: float,
    ) -> float:
        """Compute Vega P&L.

        Vega P&L = Vega × Δσ

        Parameters
        ----------
        vega : float
            Position vega (per 1% vol change)
        vol_change : float
            Change in volatility (in %, e.g., 0.02 for 2% to 4%)

        Returns
        -------
        float
            Vega P&L
        """
        return float(vega * vol_change)

    def compute_theta_pnl(
        self,
        theta: float,
        time_elapsed: float = 1.0,
    ) -> float:
        """Compute Theta P&L.

        Theta P&L = Theta × Δt

        Parameters
        ----------
        theta : float
            Position theta (per day)
        time_elapsed : float, optional
            Time elapsed in days (default: 1)

        Returns
        -------
        float
            Theta P&L
        """
        return float(theta * time_elapsed)

    def compute_rho_pnl(
        self,
        rho: float,
        rate_change: float,
    ) -> float:
        """Compute Rho P&L.

        Rho P&L = Rho × Δr

        Parameters
        ----------
        rho : float
            Position rho (per 1% rate change)
        rate_change : float
            Change in interest rate (in %, e.g., 0.0025 for 25bp)

        Returns
        -------
        float
            Rho P&L
        """
        return float(rho * rate_change)

    def compute_daily_pnl_explain(
        self,
        date: str,
        total_pnl: float,
        delta: float,
        gamma: float,
        vega: float,
        theta: float,
        rho: float,
        price_change: float,
        vol_change: float,
        rate_change: float,
        carry_pnl: float = 0.0,
    ) -> DailyPLExplain:
        """Compute complete daily P&L explain.

        Parameters
        ----------
        date : str
            Business date
        total_pnl : float
            Actual total P&L
        delta, gamma, vega, theta, rho : float
            Position Greeks at T₀
        price_change : float
            Change in underlying price
        vol_change : float
            Change in volatility
        rate_change : float
            Change in interest rate
        carry_pnl : float, optional
            Pre-computed carry P&L

        Returns
        -------
        DailyPLExplain
            Complete P&L attribution
        """
        # Compute Greek P&Ls
        delta_pnl = self.compute_delta_pnl(delta, price_change)
        gamma_pnl = self.compute_gamma_pnl(gamma, price_change)
        vega_pnl = self.compute_vega_pnl(vega, vol_change)
        theta_pnl = self.compute_theta_pnl(theta)
        rho_pnl = self.compute_rho_pnl(rho, rate_change)

        # Unexplained residual
        explained = carry_pnl + delta_pnl + gamma_pnl + vega_pnl + theta_pnl + rho_pnl
        unexplained = total_pnl - explained

        # Build components
        components = [
            PLComponent("Carry", carry_pnl, description="Funding and accruals"),
            PLComponent("Delta", delta_pnl, description="First-order price sensitivity"),
            PLComponent("Gamma", gamma_pnl, description="Convexity"),
            PLComponent("Vega", vega_pnl, description="Volatility sensitivity"),
            PLComponent("Theta", theta_pnl, description="Time decay"),
            PLComponent("Rho", rho_pnl, description="Interest rate sensitivity"),
            PLComponent("Unexplained", unexplained, description="Residual"),
        ]

        return DailyPLExplain(
            date=date,
            total_pnl=total_pnl,
            carry_pnl=carry_pnl,
            delta_pnl=delta_pnl,
            gamma_pnl=gamma_pnl,
            vega_pnl=vega_pnl,
            theta_pnl=theta_pnl,
            rho_pnl=rho_pnl,
            unexplained_pnl=unexplained,
            components=components,
        )


@dataclass
class RiskFactorAttribution:
    """P&L attribution by risk factor.

    Attributes
    ----------
    risk_factor : str
        Risk factor identifier (e.g., "USD.3M", "SPX", "AAPL")
    asset_class : str
        Asset class (IR, Credit, FX, Equity, Commodity)
    delta_pnl : float
        Delta P&L from this factor
    gamma_pnl : float
        Gamma P&L from this factor
    vega_pnl : float
        Vega P&L from this factor
    basis_pnl : float
        Basis risk P&L (e.g., swap spread)
    """
    risk_factor: str
    asset_class: str
    delta_pnl: float
    gamma_pnl: float = 0.0
    vega_pnl: float = 0.0
    basis_pnl: float = 0.0

    @property
    def total_pnl(self) -> float:
        """Total P&L from this risk factor."""
        return self.delta_pnl + self.gamma_pnl + self.vega_pnl + self.basis_pnl


class RiskFactorPLCalculator:
    """Calculate P&L attribution by risk factor.

    Decomposes portfolio P&L into contributions from:
    - Interest rate risk factors
    - Credit spread risk factors
    - FX risk factors
    - Equity risk factors
    - Basis risk factors
    """

    def compute_ir_attribution(
        self,
        rate_sensitivities: Dict[str, float],
        rate_changes: Dict[str, float],
    ) -> List[RiskFactorAttribution]:
        """Compute interest rate P&L attribution.

        Parameters
        ----------
        rate_sensitivities : dict
            Delta sensitivities by IR curve point {"USD.3M": 10000, ...}
        rate_changes : dict
            Rate changes by curve point {"USD.3M": 0.0025, ...}

        Returns
        -------
        List[RiskFactorAttribution]
            Attribution by IR risk factor
        """
        attributions = []

        for factor, sensitivity in rate_sensitivities.items():
            if factor in rate_changes:
                delta_pnl = sensitivity * rate_changes[factor]

                attributions.append(RiskFactorAttribution(
                    risk_factor=factor,
                    asset_class="IR",
                    delta_pnl=delta_pnl,
                ))

        return attributions

    def compute_credit_attribution(
        self,
        credit_sensitivities: Dict[str, float],
        spread_changes: Dict[str, float],
    ) -> List[RiskFactorAttribution]:
        """Compute credit spread P&L attribution.

        Parameters
        ----------
        credit_sensitivities : dict
            CS01 by credit curve {"AAPL.5Y": 5000, ...}
        spread_changes : dict
            Spread changes {"AAPL.5Y": 0.0010, ...}

        Returns
        -------
        List[RiskFactorAttribution]
            Attribution by credit risk factor
        """
        attributions = []

        for factor, cs01 in credit_sensitivities.items():
            if factor in spread_changes:
                delta_pnl = cs01 * spread_changes[factor]

                attributions.append(RiskFactorAttribution(
                    risk_factor=factor,
                    asset_class="Credit",
                    delta_pnl=delta_pnl,
                ))

        return attributions

    def compute_fx_attribution(
        self,
        fx_deltas: Dict[str, float],
        fx_spot_changes: Dict[str, float],
    ) -> List[RiskFactorAttribution]:
        """Compute FX P&L attribution.

        Parameters
        ----------
        fx_deltas : dict
            FX delta by currency pair {"EURUSD": 100000, ...}
        fx_spot_changes : dict
            Spot changes {"EURUSD": 0.0050, ...}

        Returns
        -------
        List[RiskFactorAttribution]
            Attribution by FX risk factor
        """
        attributions = []

        for factor, delta in fx_deltas.items():
            if factor in fx_spot_changes:
                delta_pnl = delta * fx_spot_changes[factor]

                attributions.append(RiskFactorAttribution(
                    risk_factor=factor,
                    asset_class="FX",
                    delta_pnl=delta_pnl,
                ))

        return attributions

    def compute_equity_attribution(
        self,
        equity_deltas: Dict[str, float],
        spot_changes: Dict[str, float],
        equity_gammas: Optional[Dict[str, float]] = None,
    ) -> List[RiskFactorAttribution]:
        """Compute equity P&L attribution.

        Parameters
        ----------
        equity_deltas : dict
            Equity delta by ticker {"AAPL": 1000, ...}
        spot_changes : dict
            Spot price changes {"AAPL": 2.50, ...}
        equity_gammas : dict, optional
            Equity gamma by ticker

        Returns
        -------
        List[RiskFactorAttribution]
            Attribution by equity risk factor
        """
        attributions = []

        for factor, delta in equity_deltas.items():
            if factor in spot_changes:
                delta_pnl = delta * spot_changes[factor]

                # Add gamma contribution if available
                gamma_pnl = 0.0
                if equity_gammas and factor in equity_gammas:
                    gamma_pnl = 0.5 * equity_gammas[factor] * spot_changes[factor] ** 2

                attributions.append(RiskFactorAttribution(
                    risk_factor=factor,
                    asset_class="Equity",
                    delta_pnl=delta_pnl,
                    gamma_pnl=gamma_pnl,
                ))

        return attributions


@dataclass
class ModelRiskMetrics:
    """Model risk quantification metrics.

    Attributes
    ----------
    mark_to_model_reserve : float
        Reserve for marking to model (vs market)
    parameter_uncertainty : float
        P&L uncertainty from parameter estimation
    model_replacement_impact : float
        Estimated P&L impact of model change
    unexplained_pnl_avg : float
        Average unexplained P&L over period
    unexplained_pnl_std : float
        Std dev of unexplained P&L
    """
    mark_to_model_reserve: float
    parameter_uncertainty: float
    model_replacement_impact: float
    unexplained_pnl_avg: float
    unexplained_pnl_std: float

    @property
    def total_model_risk(self) -> float:
        """Total model risk charge."""
        return (self.mark_to_model_reserve +
                self.parameter_uncertainty +
                abs(self.model_replacement_impact))


class ModelRiskCalculator:
    """Quantify model risk and compute reserves.

    Model risk arises from:
    - Mark-to-model vs mark-to-market differences
    - Parameter estimation uncertainty
    - Model specification risk
    - Unexplained P&L patterns
    """

    def compute_mark_to_model_reserve(
        self,
        model_price: float,
        market_price: Optional[float] = None,
        bid_ask_spread: float = 0.0,
    ) -> float:
        """Compute mark-to-model reserve.

        Parameters
        ----------
        model_price : float
            Model valuation
        market_price : float, optional
            Observed market price (if available)
        bid_ask_spread : float, optional
            Bid-ask spread for liquidity reserve

        Returns
        -------
        float
            Required reserve
        """
        if market_price is not None:
            # Reserve based on model vs market difference
            reserve = abs(model_price - market_price)
        else:
            # Reserve based on bid-ask spread (illiquid)
            reserve = 0.5 * bid_ask_spread

        return float(reserve)

    def compute_parameter_uncertainty(
        self,
        parameter_std: Array,
        sensitivity_to_param: Array,
    ) -> float:
        """Compute P&L uncertainty from parameter estimation.

        Uses delta method: Var(V(θ)) ≈ (∂V/∂θ)ᵀ Σ (∂V/∂θ)

        Parameters
        ----------
        parameter_std : Array
            Standard errors of parameter estimates
        sensitivity_to_param : Array
            Sensitivities ∂V/∂θ for each parameter

        Returns
        -------
        float
            P&L standard deviation from parameter uncertainty
        """
        # Simplified: assume independent parameters
        variance = jnp.sum((sensitivity_to_param * parameter_std) ** 2)
        return float(jnp.sqrt(variance))

    def compute_model_replacement_impact(
        self,
        current_model_value: float,
        alternative_model_value: float,
    ) -> float:
        """Estimate P&L impact of model replacement.

        Parameters
        ----------
        current_model_value : float
            Valuation under current model
        alternative_model_value : float
            Valuation under alternative/benchmark model

        Returns
        -------
        float
            P&L impact of model change
        """
        return float(alternative_model_value - current_model_value)

    def analyze_unexplained_pnl(
        self,
        unexplained_pnl_history: Array,
    ) -> tuple[float, float]:
        """Analyze unexplained P&L patterns.

        Parameters
        ----------
        unexplained_pnl_history : Array
            Historical unexplained P&L

        Returns
        -------
        tuple[float, float]
            (mean, std) of unexplained P&L
        """
        mean = float(jnp.mean(unexplained_pnl_history))
        std = float(jnp.std(unexplained_pnl_history))

        return mean, std

    def compute_model_risk_metrics(
        self,
        model_price: float,
        market_price: Optional[float],
        bid_ask_spread: float,
        parameter_std: Array,
        sensitivity_to_param: Array,
        alternative_model_value: float,
        unexplained_pnl_history: Array,
    ) -> ModelRiskMetrics:
        """Compute comprehensive model risk metrics.

        Parameters
        ----------
        model_price : float
            Current model valuation
        market_price : float or None
            Market price if available
        bid_ask_spread : float
            Bid-ask spread
        parameter_std : Array
            Parameter standard errors
        sensitivity_to_param : Array
            Sensitivities to parameters
        alternative_model_value : float
            Alternative model valuation
        unexplained_pnl_history : Array
            Historical unexplained P&L

        Returns
        -------
        ModelRiskMetrics
            Complete model risk metrics
        """
        mtm_reserve = self.compute_mark_to_model_reserve(
            model_price, market_price, bid_ask_spread
        )

        param_uncertainty = self.compute_parameter_uncertainty(
            parameter_std, sensitivity_to_param
        )

        model_impact = self.compute_model_replacement_impact(
            model_price, alternative_model_value
        )

        unexp_mean, unexp_std = self.analyze_unexplained_pnl(
            unexplained_pnl_history
        )

        return ModelRiskMetrics(
            mark_to_model_reserve=mtm_reserve,
            parameter_uncertainty=param_uncertainty,
            model_replacement_impact=model_impact,
            unexplained_pnl_avg=unexp_mean,
            unexplained_pnl_std=unexp_std,
        )


__all__ = [
    "PLComponent",
    "DailyPLExplain",
    "GreekPLCalculator",
    "RiskFactorAttribution",
    "RiskFactorPLCalculator",
    "ModelRiskMetrics",
    "ModelRiskCalculator",
]
