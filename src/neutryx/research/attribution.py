"""Performance attribution and risk decomposition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import Array


@dataclass
class AttributionResult:
    """Results of performance attribution analysis."""

    # Factor-based attribution
    factor_returns: Dict[str, float]
    factor_contributions: Dict[str, float]
    factor_percentages: Dict[str, float]

    # Sector/asset attribution
    asset_returns: Dict[str, float]
    asset_contributions: Dict[str, float]

    # Selection vs allocation
    allocation_effect: float
    selection_effect: float
    interaction_effect: float

    # Risk attribution
    risk_contributions: Dict[str, float]
    marginal_risks: Dict[str, float]

    # Total
    total_return: float
    explained_return: float
    unexplained_return: float

    def summary(self) -> pd.DataFrame:
        """Get summary as DataFrame."""
        data = []

        for factor, contribution in self.factor_contributions.items():
            data.append({
                "Factor": factor,
                "Return": self.factor_returns.get(factor, 0.0),
                "Contribution": contribution,
                "Percentage": self.factor_percentages.get(factor, 0.0),
            })

        return pd.DataFrame(data)


class PerformanceAttribution:
    """Performance attribution analyzer."""

    def __init__(
        self,
        returns: pd.Series,
        factor_returns: Optional[pd.DataFrame] = None,
        positions: Optional[pd.DataFrame] = None,
    ):
        """Initialize attribution analyzer.

        Args:
            returns: Portfolio returns
            factor_returns: Factor returns (columns are factors)
            positions: Position weights over time
        """
        self.returns = returns
        self.factor_returns = factor_returns
        self.positions = positions

    def factor_attribution(
        self,
        factor_exposures: pd.DataFrame,
    ) -> AttributionResult:
        """Perform factor-based attribution.

        Args:
            factor_exposures: Factor exposures (beta) for each factor

        Returns:
            AttributionResult with factor attribution
        """
        if self.factor_returns is None:
            raise ValueError("Factor returns required for factor attribution")

        # Align indices
        common_idx = self.returns.index.intersection(self.factor_returns.index)
        returns = self.returns.loc[common_idx]
        factor_rets = self.factor_returns.loc[common_idx]

        # Calculate factor contributions
        factor_contributions = {}
        factor_percentages = {}
        total_return = float(returns.sum())

        for factor in factor_rets.columns:
            # Factor contribution = exposure * factor_return
            exposure = factor_exposures.get(factor, 0.0)
            factor_ret = float(factor_rets[factor].sum())
            contribution = exposure * factor_ret

            factor_contributions[factor] = contribution
            factor_percentages[factor] = (
                contribution / total_return if abs(total_return) > 1e-10 else 0.0
            )

        # Factor returns
        factor_returns_dict = {
            factor: float(factor_rets[factor].sum())
            for factor in factor_rets.columns
        }

        explained = sum(factor_contributions.values())
        unexplained = total_return - explained

        return AttributionResult(
            factor_returns=factor_returns_dict,
            factor_contributions=factor_contributions,
            factor_percentages=factor_percentages,
            asset_returns={},
            asset_contributions={},
            allocation_effect=0.0,
            selection_effect=0.0,
            interaction_effect=0.0,
            risk_contributions={},
            marginal_risks={},
            total_return=total_return,
            explained_return=explained,
            unexplained_return=unexplained,
        )

    def brinson_attribution(
        self,
        portfolio_weights: pd.DataFrame,
        benchmark_weights: pd.DataFrame,
        asset_returns: pd.DataFrame,
    ) -> AttributionResult:
        """Perform Brinson attribution (allocation vs selection).

        Args:
            portfolio_weights: Portfolio weights by sector/asset
            benchmark_weights: Benchmark weights by sector/asset
            asset_returns: Returns by sector/asset

        Returns:
            AttributionResult with Brinson attribution
        """
        # Align data
        common_idx = (
            portfolio_weights.index
            .intersection(benchmark_weights.index)
            .intersection(asset_returns.index)
        )

        pw = portfolio_weights.loc[common_idx]
        bw = benchmark_weights.loc[common_idx]
        rets = asset_returns.loc[common_idx]

        # Weight differences
        weight_diff = pw - bw

        # Benchmark return for each asset
        benchmark_return = (bw * rets).sum(axis=1)

        # Allocation effect: (weight_diff) * (benchmark_return - portfolio_return)
        allocation = {}
        for asset in pw.columns:
            allocation[asset] = float(
                (weight_diff[asset] * (rets[asset] - benchmark_return)).sum()
            )

        allocation_effect = sum(allocation.values())

        # Selection effect: (benchmark_weight) * (asset_return - benchmark_return)
        selection = {}
        for asset in pw.columns:
            selection[asset] = float(
                (bw[asset] * (rets[asset] - benchmark_return)).sum()
            )

        selection_effect = sum(selection.values())

        # Interaction effect: (weight_diff) * (asset_return - benchmark_return)
        interaction = {}
        for asset in pw.columns:
            interaction[asset] = float(
                (weight_diff[asset] * (rets[asset] - benchmark_return)).sum()
            )

        interaction_effect = sum(interaction.values())

        # Asset contributions
        asset_contributions = {}
        for asset in pw.columns:
            asset_contributions[asset] = (
                allocation.get(asset, 0.0)
                + selection.get(asset, 0.0)
                + interaction.get(asset, 0.0)
            )

        # Asset returns
        asset_returns_dict = {
            asset: float(rets[asset].sum()) for asset in rets.columns
        }

        total_return = (pw * rets).sum().sum()

        return AttributionResult(
            factor_returns={},
            factor_contributions={},
            factor_percentages={},
            asset_returns=asset_returns_dict,
            asset_contributions=asset_contributions,
            allocation_effect=allocation_effect,
            selection_effect=selection_effect,
            interaction_effect=interaction_effect,
            risk_contributions={},
            marginal_risks={},
            total_return=float(total_return),
            explained_return=allocation_effect + selection_effect + interaction_effect,
            unexplained_return=0.0,
        )


def risk_decomposition(
    returns: pd.DataFrame,
    weights: pd.Series | Array,
) -> Dict[str, float]:
    """Decompose portfolio risk into component contributions.

    Uses covariance matrix to attribute risk to individual positions.

    Args:
        returns: Asset returns (columns are assets)
        weights: Portfolio weights

    Returns:
        Dictionary mapping asset names to risk contributions
    """
    if isinstance(weights, pd.Series):
        weights_array = weights.values
    else:
        weights_array = np.array(weights)

    # Calculate covariance matrix
    cov_matrix = returns.cov().values

    # Portfolio variance
    portfolio_variance = weights_array @ cov_matrix @ weights_array.T
    portfolio_vol = np.sqrt(portfolio_variance)

    if portfolio_vol < 1e-10:
        return {asset: 0.0 for asset in returns.columns}

    # Marginal contribution to risk (MCR)
    mcr = (cov_matrix @ weights_array) / portfolio_vol

    # Component contribution to risk (CCR)
    ccr = weights_array * mcr

    # Risk contributions as percentage
    risk_contrib = {
        asset: float(ccr[i])
        for i, asset in enumerate(returns.columns)
    }

    return risk_contrib


def marginal_risk_contribution(
    returns: pd.DataFrame,
    weights: pd.Series | Array,
) -> Dict[str, float]:
    """Calculate marginal risk contribution for each asset.

    Measures how portfolio risk changes with small changes in position size.

    Args:
        returns: Asset returns
        weights: Portfolio weights

    Returns:
        Dictionary mapping assets to marginal risk contributions
    """
    if isinstance(weights, pd.Series):
        weights_array = weights.values
    else:
        weights_array = np.array(weights)

    # Covariance matrix
    cov_matrix = returns.cov().values

    # Portfolio variance
    portfolio_variance = weights_array @ cov_matrix @ weights_array.T
    portfolio_vol = np.sqrt(portfolio_variance)

    if portfolio_vol < 1e-10:
        return {asset: 0.0 for asset in returns.columns}

    # Marginal contribution
    marginal = (cov_matrix @ weights_array) / portfolio_vol

    marginal_contrib = {
        asset: float(marginal[i])
        for i, asset in enumerate(returns.columns)
    }

    return marginal_contrib


def style_attribution(
    portfolio_returns: pd.Series,
    style_factor_returns: pd.DataFrame,
) -> Dict[str, float]:
    """Perform style-based attribution using regression.

    Args:
        portfolio_returns: Portfolio returns
        style_factor_returns: Style factor returns (e.g., size, value, momentum)

    Returns:
        Dictionary mapping style factors to contributions
    """
    # Align data
    common_idx = portfolio_returns.index.intersection(style_factor_returns.index)
    y = portfolio_returns.loc[common_idx].values
    X = style_factor_returns.loc[common_idx].values

    # Add intercept
    X_with_intercept = np.column_stack([np.ones(len(X)), X])

    # OLS regression
    try:
        betas = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return {factor: 0.0 for factor in style_factor_returns.columns}

    # Calculate contributions
    contributions = {}
    contributions["Alpha"] = float(betas[0])

    for i, factor in enumerate(style_factor_returns.columns):
        factor_return = float(style_factor_returns[factor].sum())
        contributions[factor] = float(betas[i + 1] * factor_return)

    return contributions


def transaction_cost_attribution(
    gross_returns: pd.Series,
    net_returns: pd.Series,
    turnover: pd.Series,
) -> Dict[str, float]:
    """Attribute performance difference between gross and net returns to costs.

    Args:
        gross_returns: Returns before costs
        net_returns: Returns after costs
        turnover: Portfolio turnover

    Returns:
        Attribution of cost drag
    """
    # Cost drag
    cost_drag = gross_returns - net_returns

    # Average cost per transaction
    total_turnover = turnover.sum()
    if total_turnover > 0:
        avg_cost_bps = (cost_drag.sum() / total_turnover) * 10000
    else:
        avg_cost_bps = 0.0

    attribution = {
        "Total Cost Drag": float(cost_drag.sum()),
        "Total Turnover": float(total_turnover),
        "Average Cost (bps)": float(avg_cost_bps),
        "Cost as % of Gross Return": (
            float(cost_drag.sum() / gross_returns.sum())
            if gross_returns.sum() > 1e-10
            else 0.0
        ),
    }

    return attribution


__all__ = [
    "PerformanceAttribution",
    "AttributionResult",
    "risk_decomposition",
    "marginal_risk_contribution",
    "style_attribution",
    "transaction_cost_attribution",
]
