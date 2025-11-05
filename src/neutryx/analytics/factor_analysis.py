"""Factor Analysis Framework for Risk Models and Portfolio Attribution.

This module provides comprehensive factor analysis capabilities including:
- Principal Component Analysis (PCA) for dimension reduction
- Barra-style multi-factor risk models
- Style attribution (value, growth, momentum, quality, size, volatility)
- Factor timing and allocation strategies
- Factor performance attribution

Factor models decompose portfolio returns and risk into:
- Common factor risk (systematic risk shared across assets)
- Specific risk (idiosyncratic risk unique to each asset)

Key Applications:
- Risk budgeting and decomposition
- Portfolio construction and optimization
- Performance attribution
- Risk forecasting
- Factor timing strategies

References
----------
Grinold, R. C., & Kahn, R. N. (2000). "Active Portfolio Management."
McGraw-Hill. (Chapter on Factor Models)

Menchero, J., Morozov, A., & Shepard, P. (2008). "The Barra US Equity Model."
MSCI Barra Research.

Fama, E. F., & French, K. R. (1993). "Common risk factors in the returns on
stocks and bonds." Journal of Financial Economics, 33(1), 3-56.

Carhart, M. M. (1997). "On persistence in mutual fund performance."
Journal of Finance, 52(1), 57-82.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import jax.numpy as jnp
from jax import Array

# ==============================================================================
# Enumerations
# ==============================================================================


class StyleFactor(str, Enum):
    """Style factors for equity attribution."""

    VALUE = "value"  # Book-to-market, earnings yield, dividend yield
    GROWTH = "growth"  # Earnings growth, sales growth
    MOMENTUM = "momentum"  # Price momentum, earnings momentum
    QUALITY = "quality"  # ROE, earnings quality, leverage
    SIZE = "size"  # Market capitalization
    VOLATILITY = "volatility"  # Historical volatility, beta
    LIQUIDITY = "liquidity"  # Trading volume, bid-ask spread
    DIVIDEND = "dividend"  # Dividend yield


class IndustryFactor(str, Enum):
    """Industry/sector factors (GICS Level 1)."""

    ENERGY = "energy"
    MATERIALS = "materials"
    INDUSTRIALS = "industrials"
    CONSUMER_DISCRETIONARY = "consumer_discretionary"
    CONSUMER_STAPLES = "consumer_staples"
    HEALTH_CARE = "health_care"
    FINANCIALS = "financials"
    INFORMATION_TECHNOLOGY = "information_technology"
    COMMUNICATION_SERVICES = "communication_services"
    UTILITIES = "utilities"
    REAL_ESTATE = "real_estate"


# ==============================================================================
# Data Structures - PCA
# ==============================================================================


@dataclass(frozen=True)
class PCAResult:
    """Result of Principal Component Analysis.

    Attributes
    ----------
    principal_components : Array
        Principal component vectors (eigenvectors), shape (n_features, n_components)
    explained_variance : Array
        Variance explained by each component, shape (n_components,)
    explained_variance_ratio : Array
        Proportion of variance explained by each component, shape (n_components,)
    cumulative_variance_ratio : Array
        Cumulative proportion of variance, shape (n_components,)
    eigenvalues : Array
        Eigenvalues of covariance matrix, shape (n_components,)
    mean : Array
        Mean of original data, shape (n_features,)
    n_components : int
        Number of components retained
    n_features : int
        Number of original features
    """

    principal_components: Array
    explained_variance: Array
    explained_variance_ratio: Array
    cumulative_variance_ratio: Array
    eigenvalues: Array
    mean: Array
    n_components: int
    n_features: int


@dataclass(frozen=True)
class PCTransform:
    """Transformed data using principal components.

    Attributes
    ----------
    transformed_data : Array
        Data projected onto principal components, shape (n_samples, n_components)
    reconstruction : Array
        Reconstructed data from components, shape (n_samples, n_features)
    reconstruction_error : float
        Mean squared reconstruction error
    """

    transformed_data: Array
    reconstruction: Array
    reconstruction_error: float


# ==============================================================================
# Data Structures - Factor Models
# ==============================================================================


@dataclass(frozen=True)
class FactorExposure:
    """Factor exposure (beta) for an asset.

    Attributes
    ----------
    asset_id : str
        Asset identifier
    style_exposures : Dict[StyleFactor, float]
        Exposures to style factors
    industry_exposures : Dict[IndustryFactor, float]
        Exposures to industry factors (typically binary 0/1)
    country_exposures : Optional[Dict[str, float]]
        Exposures to country factors
    custom_exposures : Optional[Dict[str, float]]
        Custom factor exposures
    """

    asset_id: str
    style_exposures: Dict[StyleFactor, float]
    industry_exposures: Dict[IndustryFactor, float]
    country_exposures: Optional[Dict[str, float]] = None
    custom_exposures: Optional[Dict[str, float]] = None


@dataclass(frozen=True)
class FactorReturn:
    """Factor returns for a period.

    Attributes
    ----------
    date : str
        Date (YYYY-MM-DD)
    style_returns : Dict[StyleFactor, float]
        Returns to style factors
    industry_returns : Dict[IndustryFactor, float]
        Returns to industry factors
    country_returns : Optional[Dict[str, float]]
        Returns to country factors
    custom_returns : Optional[Dict[str, float]]
        Custom factor returns
    """

    date: str
    style_returns: Dict[StyleFactor, float]
    industry_returns: Dict[IndustryFactor, float]
    country_returns: Optional[Dict[str, float]] = None
    custom_returns: Optional[Dict[str, float]] = None


@dataclass(frozen=True)
class FactorRiskModel:
    """Complete factor risk model (Barra-style).

    Attributes
    ----------
    factor_covariance : Array
        Covariance matrix of factor returns, shape (n_factors, n_factors)
    specific_variances : Dict[str, float]
        Asset-specific variances (idiosyncratic risk)
    factor_names : List[str]
        Ordered list of factor names
    estimation_date : str
        Date of risk model estimation
    estimation_window : int
        Number of days used in estimation
    """

    factor_covariance: Array
    specific_variances: Dict[str, float]
    factor_names: List[str]
    estimation_date: str
    estimation_window: int


@dataclass(frozen=True)
class AssetRiskDecomposition:
    """Risk decomposition for a single asset.

    Attributes
    ----------
    asset_id : str
        Asset identifier
    total_risk : float
        Total volatility (annualized)
    factor_risk : float
        Risk from common factors
    specific_risk : float
        Idiosyncratic risk
    factor_contributions : Dict[str, float]
        Risk contribution by factor
    """

    asset_id: str
    total_risk: float
    factor_risk: float
    specific_risk: float
    factor_contributions: Dict[str, float]


# ==============================================================================
# Data Structures - Style Attribution
# ==============================================================================


@dataclass(frozen=True)
class StyleAttribution:
    """Performance attribution by style factors.

    Attributes
    ----------
    period_start : str
        Start date (YYYY-MM-DD)
    period_end : str
        End date (YYYY-MM-DD)
    total_return : float
        Total portfolio return
    factor_returns : Dict[StyleFactor, float]
        Return contribution by factor
    specific_return : float
        Asset-specific return (alpha)
    factor_exposures : Dict[StyleFactor, float]
        Average factor exposures over period
    """

    period_start: str
    period_end: str
    total_return: float
    factor_returns: Dict[StyleFactor, float]
    specific_return: float
    factor_exposures: Dict[StyleFactor, float]


# ==============================================================================
# Data Structures - Factor Timing
# ==============================================================================


@dataclass(frozen=True)
class FactorTimingSignal:
    """Factor timing signal.

    Attributes
    ----------
    factor : StyleFactor
        Style factor
    date : str
        Signal date
    signal_value : float
        Signal strength (-1 to +1, or unbounded)
    expected_return : float
        Expected factor return
    confidence : float
        Signal confidence (0 to 1)
    regime : str
        Market regime (e.g., "risk_on", "risk_off", "neutral")
    """

    factor: StyleFactor
    date: str
    signal_value: float
    expected_return: float
    confidence: float
    regime: str


@dataclass(frozen=True)
class FactorAllocation:
    """Optimal factor allocation.

    Attributes
    ----------
    date : str
        Allocation date
    factor_weights : Dict[StyleFactor, float]
        Target weights to factors
    expected_return : float
        Expected portfolio return
    expected_volatility : float
        Expected portfolio volatility
    sharpe_ratio : float
        Expected Sharpe ratio
    optimization_method : str
        Method used (e.g., "mean_variance", "risk_parity")
    """

    date: str
    factor_weights: Dict[StyleFactor, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    optimization_method: str


# ==============================================================================
# PCA Implementation
# ==============================================================================


class PrincipalComponentAnalysis:
    """Principal Component Analysis for dimension reduction.

    PCA transforms correlated variables into uncorrelated principal components,
    ordered by the amount of variance they explain.

    Applications:
    - Dimension reduction for large covariance matrices
    - Identifying dominant risk factors
    - Noise reduction
    - Visualization of high-dimensional data
    """

    def __init__(self, n_components: Optional[int] = None, variance_threshold: float = 0.95):
        """Initialize PCA.

        Parameters
        ----------
        n_components : Optional[int]
            Number of components to retain (if None, use variance_threshold)
        variance_threshold : float
            Retain components explaining this fraction of variance (default 0.95)
        """
        self.n_components = n_components
        self.variance_threshold = variance_threshold

    def fit(self, data: Array) -> PCAResult:
        """Fit PCA to data.

        Parameters
        ----------
        data : Array
            Input data, shape (n_samples, n_features)

        Returns
        -------
        PCAResult
            PCA results including components and explained variance
        """
        n_samples, n_features = data.shape

        # Center data
        mean = jnp.mean(data, axis=0)
        centered_data = data - mean

        # Compute covariance matrix
        cov_matrix = jnp.cov(centered_data.T)

        # Eigendecomposition
        eigenvalues, eigenvectors = jnp.linalg.eigh(cov_matrix)

        # Sort by eigenvalue (descending)
        idx = jnp.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Determine number of components
        explained_variance = eigenvalues
        total_variance = jnp.sum(eigenvalues)
        explained_variance_ratio = eigenvalues / total_variance
        cumulative_variance = jnp.cumsum(explained_variance_ratio)

        if self.n_components is not None:
            n_components = min(self.n_components, n_features)
        else:
            # Use variance threshold
            n_components = int(jnp.sum(cumulative_variance < self.variance_threshold) + 1)
            n_components = min(n_components, n_features)

        # Extract principal components
        principal_components = eigenvectors[:, :n_components]
        explained_variance = explained_variance[:n_components]
        explained_variance_ratio = explained_variance_ratio[:n_components]
        cumulative_variance_ratio = cumulative_variance[:n_components]
        eigenvalues_retained = eigenvalues[:n_components]

        return PCAResult(
            principal_components=principal_components,
            explained_variance=explained_variance,
            explained_variance_ratio=explained_variance_ratio,
            cumulative_variance_ratio=cumulative_variance_ratio,
            eigenvalues=eigenvalues_retained,
            mean=mean,
            n_components=n_components,
            n_features=n_features,
        )

    def transform(self, data: Array, pca_result: PCAResult) -> PCTransform:
        """Transform data using PCA.

        Parameters
        ----------
        data : Array
            Input data, shape (n_samples, n_features)
        pca_result : PCAResult
            Fitted PCA result

        Returns
        -------
        PCTransform
            Transformed data and reconstruction
        """
        # Center data
        centered_data = data - pca_result.mean

        # Project onto principal components
        transformed = centered_data @ pca_result.principal_components

        # Reconstruct
        reconstruction = transformed @ pca_result.principal_components.T + pca_result.mean

        # Reconstruction error
        reconstruction_error = float(jnp.mean((data - reconstruction) ** 2))

        return PCTransform(
            transformed_data=transformed,
            reconstruction=reconstruction,
            reconstruction_error=reconstruction_error,
        )


# ==============================================================================
# Barra-Style Factor Risk Model
# ==============================================================================


class FactorRiskModelEstimator:
    """Estimate Barra-style multi-factor risk model.

    Factor models decompose asset returns:
        r_i = Σ_k β_ik * f_k + ε_i

    where:
    - r_i = return of asset i
    - β_ik = exposure of asset i to factor k
    - f_k = return of factor k
    - ε_i = asset-specific return

    Risk decomposition:
        Var(r_i) = Σ_k Σ_l β_ik β_il Cov(f_k, f_l) + Var(ε_i)
    """

    def __init__(
        self,
        estimation_window: int = 252,
        halflife: int = 90,
        min_observations: int = 120,
    ):
        """Initialize factor risk model estimator.

        Parameters
        ----------
        estimation_window : int
            Number of days for estimation (default 252 = 1 year)
        halflife : int
            Half-life for exponential weighting (default 90 days)
        min_observations : int
            Minimum observations required (default 120)
        """
        self.estimation_window = estimation_window
        self.halflife = halflife
        self.min_observations = min_observations

    def estimate_factor_model(
        self,
        returns: Array,
        exposures: Array,
        asset_ids: List[str],
        factor_names: List[str],
        estimation_date: str,
    ) -> FactorRiskModel:
        """Estimate factor risk model from returns and exposures.

        Parameters
        ----------
        returns : Array
            Asset returns, shape (n_periods, n_assets)
        exposures : Array
            Factor exposures, shape (n_assets, n_factors)
        asset_ids : List[str]
            Asset identifiers
        factor_names : List[str]
            Factor names
        estimation_date : str
            Estimation date

        Returns
        -------
        FactorRiskModel
            Estimated factor risk model
        """
        n_periods, n_assets = returns.shape
        n_factors = len(factor_names)

        # Estimate factor returns via cross-sectional regression
        # f_t = (X'X)^-1 X' r_t
        factor_returns = jnp.linalg.lstsq(exposures, returns.T)[0].T  # (n_periods, n_factors)

        # Compute factor covariance with exponential weighting
        weights = self._exponential_weights(n_periods)
        weighted_factor_returns = factor_returns * jnp.sqrt(weights[:, None])
        factor_mean = jnp.average(factor_returns, axis=0, weights=weights)
        centered_factor_returns = factor_returns - factor_mean
        factor_covariance = (
            centered_factor_returns.T
            @ jnp.diag(weights)
            @ centered_factor_returns
            / jnp.sum(weights)
        )

        # Estimate specific returns (residuals)
        predicted_returns = (exposures @ factor_returns.T).T  # (n_periods, n_assets)
        specific_returns = returns - predicted_returns

        # Compute specific variances
        specific_variances = {}
        for i, asset_id in enumerate(asset_ids):
            asset_specific = specific_returns[:, i]
            # Exponentially weighted variance
            weighted_specific = asset_specific * jnp.sqrt(weights)
            spec_var = float(jnp.sum(weights * asset_specific**2) / jnp.sum(weights))
            specific_variances[asset_id] = spec_var

        return FactorRiskModel(
            factor_covariance=factor_covariance,
            specific_variances=specific_variances,
            factor_names=factor_names,
            estimation_date=estimation_date,
            estimation_window=n_periods,
        )

    def decompose_asset_risk(
        self,
        asset_id: str,
        exposures: Dict[str, float],
        risk_model: FactorRiskModel,
    ) -> AssetRiskDecomposition:
        """Decompose asset risk into factor and specific components.

        Parameters
        ----------
        asset_id : str
            Asset identifier
        exposures : Dict[str, float]
            Factor exposures for the asset
        risk_model : FactorRiskModel
            Factor risk model

        Returns
        -------
        AssetRiskDecomposition
            Risk decomposition
        """
        # Build exposure vector
        exposure_vector = jnp.array([exposures.get(f, 0.0) for f in risk_model.factor_names])

        # Factor variance
        factor_variance = float(
            exposure_vector @ risk_model.factor_covariance @ exposure_vector
        )
        factor_risk = float(jnp.sqrt(factor_variance)) * jnp.sqrt(252)  # Annualize

        # Specific variance
        specific_variance = risk_model.specific_variances.get(asset_id, 0.0)
        specific_risk = float(jnp.sqrt(specific_variance)) * jnp.sqrt(252)  # Annualize

        # Total risk
        total_variance = factor_variance + specific_variance
        total_risk = float(jnp.sqrt(total_variance)) * jnp.sqrt(252)

        # Factor contributions (marginal contribution to risk)
        factor_contributions = {}
        for i, factor_name in enumerate(risk_model.factor_names):
            marginal_var = 2 * exposure_vector[i] * (
                risk_model.factor_covariance[i, :] @ exposure_vector
            )
            contribution = float(marginal_var / (2 * jnp.sqrt(total_variance))) * jnp.sqrt(252)
            factor_contributions[factor_name] = contribution

        return AssetRiskDecomposition(
            asset_id=asset_id,
            total_risk=total_risk,
            factor_risk=factor_risk,
            specific_risk=specific_risk,
            factor_contributions=factor_contributions,
        )

    def _exponential_weights(self, n_periods: int) -> Array:
        """Generate exponential weights for time series.

        Parameters
        ----------
        n_periods : int
            Number of periods

        Returns
        -------
        Array
            Weights, shape (n_periods,), most recent period has highest weight
        """
        decay = jnp.log(2) / self.halflife
        t = jnp.arange(n_periods)
        weights = jnp.exp(-decay * (n_periods - 1 - t))
        weights = weights / jnp.sum(weights)
        return weights


# ==============================================================================
# Style Attribution
# ==============================================================================


class StyleAttributionAnalyzer:
    """Analyze performance attribution by style factors.

    Decomposes portfolio returns into contributions from:
    - Style factor exposures (systematic)
    - Asset-specific returns (alpha)
    """

    def attribute_performance(
        self,
        portfolio_return: float,
        portfolio_exposures: Dict[StyleFactor, float],
        factor_returns: Dict[StyleFactor, float],
        period_start: str,
        period_end: str,
    ) -> StyleAttribution:
        """Attribute portfolio performance to style factors.

        Parameters
        ----------
        portfolio_return : float
            Total portfolio return
        portfolio_exposures : Dict[StyleFactor, float]
            Average factor exposures over period
        factor_returns : Dict[StyleFactor, float]
            Factor returns over period
        period_start : str
            Start date
        period_end : str
            End date

        Returns
        -------
        StyleAttribution
            Performance attribution
        """
        # Factor contributions
        factor_return_contributions = {}
        total_factor_return = 0.0

        for factor in StyleFactor:
            exposure = portfolio_exposures.get(factor, 0.0)
            f_return = factor_returns.get(factor, 0.0)
            contribution = exposure * f_return
            factor_return_contributions[factor] = contribution
            total_factor_return += contribution

        # Specific return (alpha)
        specific_return = portfolio_return - total_factor_return

        return StyleAttribution(
            period_start=period_start,
            period_end=period_end,
            total_return=portfolio_return,
            factor_returns=factor_return_contributions,
            specific_return=specific_return,
            factor_exposures=portfolio_exposures,
        )


# ==============================================================================
# Factor Timing and Allocation
# ==============================================================================


class FactorTimingStrategy:
    """Generate factor timing signals based on market conditions.

    Timing strategies exploit time-varying factor premiums by:
    - Identifying factor regimes
    - Forecasting factor returns
    - Dynamically tilting factor exposures
    """

    def __init__(
        self,
        lookback_window: int = 63,  # ~3 months
        momentum_window: int = 126,  # ~6 months
    ):
        """Initialize factor timing strategy.

        Parameters
        ----------
        lookback_window : int
            Window for recent factor performance (default 63 days)
        momentum_window : int
            Window for momentum signals (default 126 days)
        """
        self.lookback_window = lookback_window
        self.momentum_window = momentum_window

    def generate_timing_signal(
        self,
        factor: StyleFactor,
        factor_returns_history: Array,
        market_regime_indicators: Dict[str, float],
        date: str,
    ) -> FactorTimingSignal:
        """Generate timing signal for a factor.

        Parameters
        ----------
        factor : StyleFactor
            Style factor
        factor_returns_history : Array
            Historical factor returns, most recent last
        market_regime_indicators : Dict[str, float]
            Market regime indicators (e.g., VIX, credit spreads)
        date : str
            Signal date

        Returns
        -------
        FactorTimingSignal
            Factor timing signal
        """
        # Momentum signal (past returns)
        if len(factor_returns_history) >= self.momentum_window:
            momentum_return = float(
                jnp.sum(factor_returns_history[-self.momentum_window :])
            )
        else:
            momentum_return = 0.0

        # Recent performance
        if len(factor_returns_history) >= self.lookback_window:
            recent_return = float(jnp.mean(factor_returns_history[-self.lookback_window :]))
            recent_vol = float(jnp.std(factor_returns_history[-self.lookback_window :]))
        else:
            recent_return = 0.0
            recent_vol = 0.01

        # Signal value (simple momentum-based)
        signal_value = momentum_return / (recent_vol * jnp.sqrt(self.momentum_window))
        signal_value = float(jnp.clip(signal_value, -3.0, 3.0))  # Cap at +/- 3

        # Expected return (simple forecast)
        expected_return = signal_value * recent_vol * jnp.sqrt(252) * 0.1  # Scale

        # Confidence (based on consistency)
        if len(factor_returns_history) >= self.lookback_window:
            sign_consistency = jnp.mean(jnp.sign(factor_returns_history[-self.lookback_window :]) == jnp.sign(momentum_return))
            confidence = float(jnp.abs(sign_consistency - 0.5) * 2)  # 0 to 1
        else:
            confidence = 0.0

        # Regime classification (simple)
        vix = market_regime_indicators.get("vix", 20.0)
        if vix < 15:
            regime = "risk_on"
        elif vix > 25:
            regime = "risk_off"
        else:
            regime = "neutral"

        return FactorTimingSignal(
            factor=factor,
            date=date,
            signal_value=signal_value,
            expected_return=float(expected_return),
            confidence=confidence,
            regime=regime,
        )


class FactorAllocationOptimizer:
    """Optimize factor allocation using modern portfolio theory.

    Determines optimal weights to style factors to maximize risk-adjusted returns.
    """

    def __init__(self, risk_aversion: float = 2.5):
        """Initialize factor allocation optimizer.

        Parameters
        ----------
        risk_aversion : float
            Risk aversion parameter (default 2.5)
        """
        self.risk_aversion = risk_aversion

    def optimize_mean_variance(
        self,
        expected_returns: Dict[StyleFactor, float],
        covariance_matrix: Array,
        factor_order: List[StyleFactor],
        date: str,
        constraints: Optional[Dict[str, float]] = None,
    ) -> FactorAllocation:
        """Optimize factor allocation using mean-variance optimization.

        Parameters
        ----------
        expected_returns : Dict[StyleFactor, float]
            Expected returns for each factor
        covariance_matrix : Array
            Factor covariance matrix, shape (n_factors, n_factors)
        factor_order : List[StyleFactor]
            Ordering of factors in covariance matrix
        date : str
            Allocation date
        constraints : Optional[Dict[str, float]]
            Weight constraints (e.g., {"max_weight": 0.3})

        Returns
        -------
        FactorAllocation
            Optimal factor allocation
        """
        n_factors = len(factor_order)

        # Expected return vector
        mu = jnp.array([expected_returns.get(f, 0.0) for f in factor_order])

        # Mean-variance optimization: w* = (1/λ) Σ^-1 μ
        try:
            cov_inv = jnp.linalg.inv(covariance_matrix)
            optimal_weights = (1.0 / self.risk_aversion) * (cov_inv @ mu)

            # Normalize to sum to 1
            weight_sum = jnp.sum(optimal_weights)
            if jnp.abs(weight_sum) > 1e-10:
                optimal_weights = optimal_weights / weight_sum
            else:
                optimal_weights = jnp.ones(n_factors) / n_factors

        except Exception:
            # Fallback: equal weights
            optimal_weights = jnp.ones(n_factors) / n_factors

        # Apply constraints (simple box constraints)
        if constraints and "max_weight" in constraints:
            max_weight = constraints["max_weight"]
            optimal_weights = jnp.clip(optimal_weights, -max_weight, max_weight)
            # Renormalize
            optimal_weights = optimal_weights / jnp.sum(jnp.abs(optimal_weights))

        # Build factor weights dict
        factor_weights = {
            factor: float(optimal_weights[i]) for i, factor in enumerate(factor_order)
        }

        # Expected return and risk
        expected_return = float(optimal_weights @ mu)
        expected_variance = float(optimal_weights @ covariance_matrix @ optimal_weights)
        expected_volatility = float(jnp.sqrt(expected_variance))

        # Sharpe ratio (assuming 0 risk-free rate for simplicity)
        sharpe_ratio = expected_return / expected_volatility if expected_volatility > 0 else 0.0

        return FactorAllocation(
            date=date,
            factor_weights=factor_weights,
            expected_return=expected_return,
            expected_volatility=expected_volatility,
            sharpe_ratio=sharpe_ratio,
            optimization_method="mean_variance",
        )

    def optimize_risk_parity(
        self,
        covariance_matrix: Array,
        factor_order: List[StyleFactor],
        date: str,
    ) -> FactorAllocation:
        """Optimize factor allocation using risk parity.

        Risk parity allocates equal risk contribution to each factor.

        Parameters
        ----------
        covariance_matrix : Array
            Factor covariance matrix
        factor_order : List[StyleFactor]
            Ordering of factors
        date : str
            Allocation date

        Returns
        -------
        FactorAllocation
            Risk parity allocation
        """
        n_factors = len(factor_order)

        # Simple risk parity: w_i ∝ 1/σ_i
        factor_vols = jnp.sqrt(jnp.diag(covariance_matrix))
        weights = 1.0 / factor_vols
        weights = weights / jnp.sum(weights)

        # Build factor weights dict
        factor_weights = {factor: float(weights[i]) for i, factor in enumerate(factor_order)}

        # Expected return (assume equal for risk parity)
        expected_return = 0.0

        # Portfolio volatility
        expected_variance = float(weights @ covariance_matrix @ weights)
        expected_volatility = float(jnp.sqrt(expected_variance))

        return FactorAllocation(
            date=date,
            factor_weights=factor_weights,
            expected_return=expected_return,
            expected_volatility=expected_volatility,
            sharpe_ratio=0.0,
            optimization_method="risk_parity",
        )


__all__ = [
    # Enums
    "StyleFactor",
    "IndustryFactor",
    # PCA
    "PCAResult",
    "PCTransform",
    "PrincipalComponentAnalysis",
    # Factor Models
    "FactorExposure",
    "FactorReturn",
    "FactorRiskModel",
    "AssetRiskDecomposition",
    "FactorRiskModelEstimator",
    # Style Attribution
    "StyleAttribution",
    "StyleAttributionAnalyzer",
    # Factor Timing
    "FactorTimingSignal",
    "FactorAllocation",
    "FactorTimingStrategy",
    "FactorAllocationOptimizer",
]
