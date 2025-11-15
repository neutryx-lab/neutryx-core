"""Factor Analysis Examples - PCA, Risk Models, Style Attribution, Factor Timing.

Demonstrates:
1. PCA for dimension reduction of covariance matrices
2. Barra-style multi-factor risk models
3. Style attribution (value, growth, momentum)
4. Factor timing strategies
5. Factor allocation optimization
"""
import jax
import jax.numpy as jnp

from neutryx.analytics.factor_analysis import (
    FactorAllocationOptimizer,
    FactorRiskModelEstimator,
    FactorTimingStrategy,
    PrincipalComponentAnalysis,
    StyleAttributionAnalyzer,
    StyleFactor,
)


def print_section(title: str):
    """Print section separator."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80 + "\n")


# ==============================================================================
# Example 1: PCA for Dimension Reduction
# ==============================================================================


def example_pca_dimension_reduction():
    """Example 1: Use PCA to reduce dimensionality of returns."""
    print_section("Example 1: PCA for Dimension Reduction")

    # Simulate 50 correlated asset returns
    key = jax.random.PRNGKey(42)
    n_periods = 252  # 1 year of daily data
    n_assets = 50

    # Generate factor structure
    n_true_factors = 5
    true_factors = jax.random.normal(key, (n_periods, n_true_factors)) * 0.015

    # Asset factor loadings
    loadings = jax.random.normal(jax.random.PRNGKey(43), (n_assets, n_true_factors))
    loadings = loadings / jnp.linalg.norm(loadings, axis=1, keepdims=True)

    # Generate returns with factor structure
    returns = (loadings @ true_factors.T).T
    # Add idiosyncratic noise
    noise = jax.random.normal(jax.random.PRNGKey(44), (n_periods, n_assets)) * 0.005
    returns = returns + noise

    print(f"Original Data:")
    print(f"  Number of Assets: {n_assets}")
    print(f"  Number of Periods: {n_periods}")
    print(f"  True Factors: {n_true_factors}")

    # Apply PCA
    pca = PrincipalComponentAnalysis(variance_threshold=0.80)
    pca_result = pca.fit(returns)

    print(f"\nPCA Results:")
    print(f"  Components Retained: {pca_result.n_components} (to explain 80% variance)")
    print(f"  Actual Variance Explained: {pca_result.cumulative_variance_ratio[-1]:.2%}")
    print(f"\nExplained Variance by Component:")
    for i in range(pca_result.n_components):
        print(
            f"    PC{i+1}: {pca_result.explained_variance_ratio[i]:.2%} "
            f"(Cumulative: {pca_result.cumulative_variance_ratio[i]:.2%})"
        )

    # Transform data
    transform = pca.transform(returns, pca_result)
    print(f"\nTransformation:")
    print(f"  Transformed Data Shape: {transform.transformed_data.shape}")
    print(f"  Reconstruction Error (MSE): {transform.reconstruction_error:.6f}")

    # Dimension reduction achieved
    compression_ratio = (n_assets / pca_result.n_components)
    print(f"\nDimension Reduction:")
    print(f"  Original Dimensions: {n_assets}")
    print(f"  Reduced Dimensions: {pca_result.n_components}")
    print(f"  Compression Ratio: {compression_ratio:.1f}x")


# ==============================================================================
# Example 2: Barra-Style Factor Risk Model
# ==============================================================================


def example_factor_risk_model():
    """Example 2: Estimate Barra-style multi-factor risk model."""
    print_section("Example 2: Barra-Style Factor Risk Model")

    # Simulate factor returns and asset returns
    key = jax.random.PRNGKey(123)
    n_periods = 252
    n_assets = 20
    n_factors = 4

    # True factor returns (VALUE, MOMENTUM, SIZE, QUALITY)
    factor_returns = jax.random.normal(key, (n_periods, n_factors)) * 0.012

    # Asset factor exposures (betas)
    exposures = jax.random.normal(jax.random.PRNGKey(124), (n_assets, n_factors))
    # Normalize exposures
    exposures = exposures * jnp.array([1.5, 0.8, 1.2, 1.0])  # Scale factors differently

    # Generate asset returns from factor model
    returns = (exposures @ factor_returns.T).T

    # Add asset-specific returns
    specific_returns = jax.random.normal(jax.random.PRNGKey(125), (n_periods, n_assets)) * 0.008
    returns = returns + specific_returns

    print("Estimating Factor Risk Model...")
    print(f"  Number of Assets: {n_assets}")
    print(f"  Number of Factors: {n_factors}")
    print(f"  Estimation Window: {n_periods} days")

    # Estimate factor model
    estimator = FactorRiskModelEstimator(estimation_window=252, halflife=90)
    asset_ids = [f"STOCK_{i:02d}" for i in range(n_assets)]
    factor_names = ["VALUE", "MOMENTUM", "SIZE", "QUALITY"]

    risk_model = estimator.estimate_factor_model(
        returns=returns,
        exposures=exposures,
        asset_ids=asset_ids,
        factor_names=factor_names,
        estimation_date="2024-12-31",
    )

    print(f"\nFactor Covariance Matrix (Daily):")
    for i, f1 in enumerate(factor_names):
        row_str = f"  {f1:10s}:"
        for j, f2 in enumerate(factor_names):
            row_str += f" {risk_model.factor_covariance[i, j]:8.6f}"
        print(row_str)

    # Annualized factor volatilities
    print(f"\nFactor Volatilities (Annualized):")
    for i, factor_name in enumerate(factor_names):
        daily_vol = jnp.sqrt(risk_model.factor_covariance[i, i])
        annual_vol = daily_vol * jnp.sqrt(252)
        print(f"  {factor_name:10s}: {annual_vol:.2%}")

    # Decompose risk for first 3 assets
    print(f"\nRisk Decomposition (First 3 Assets):")
    for i in range(3):
        asset_exposures = {factor_names[j]: float(exposures[i, j]) for j in range(n_factors)}
        decomp = estimator.decompose_asset_risk(
            asset_id=asset_ids[i], exposures=asset_exposures, risk_model=risk_model
        )

        print(f"\n  {decomp.asset_id}:")
        print(f"    Total Risk: {decomp.total_risk:.2%}")
        print(f"    Factor Risk: {decomp.factor_risk:.2%} ({decomp.factor_risk/decomp.total_risk:.1%} of total)")
        print(f"    Specific Risk: {decomp.specific_risk:.2%} ({decomp.specific_risk/decomp.total_risk:.1%} of total)")
        print(f"    Factor Contributions:")
        for factor_name, contribution in decomp.factor_contributions.items():
            print(f"      {factor_name:10s}: {contribution:6.2%}")


# ==============================================================================
# Example 3: Style Attribution
# ==============================================================================


def example_style_attribution():
    """Example 3: Attribute portfolio performance to style factors."""
    print_section("Example 3: Style Attribution")

    # Portfolio with style tilts
    print("Portfolio Strategy: Value-Quality with Momentum Overlay")
    print("\nPortfolio Factor Exposures:")

    portfolio_exposures = {
        StyleFactor.VALUE: 1.2,  # Strong value tilt
        StyleFactor.MOMENTUM: 0.5,  # Moderate momentum
        StyleFactor.SIZE: -0.3,  # Small-cap bias
        StyleFactor.QUALITY: 0.8,  # Quality bias
        StyleFactor.VOLATILITY: -0.4,  # Low volatility preference
        StyleFactor.GROWTH: -0.2,  # Anti-growth
    }

    for factor, exposure in portfolio_exposures.items():
        print(f"  {factor.value.upper():12s}: {exposure:+.2f}")

    # Factor returns over the year
    factor_returns = {
        StyleFactor.VALUE: 0.18,  # Strong year for value
        StyleFactor.MOMENTUM: 0.08,  # Moderate momentum returns
        StyleFactor.SIZE: -0.05,  # Large caps outperformed
        StyleFactor.QUALITY: 0.12,  # Quality premium
        StyleFactor.VOLATILITY: -0.03,  # Low vol underperformed
        StyleFactor.GROWTH: -0.10,  # Growth declined
    }

    print(f"\nFactor Returns (2024):")
    for factor, ret in factor_returns.items():
        print(f"  {factor.value.upper():12s}: {ret:+.2%}")

    # Total portfolio return
    total_return = 0.25  # 25%

    # Attribute performance
    analyzer = StyleAttributionAnalyzer()
    attribution = analyzer.attribute_performance(
        portfolio_return=total_return,
        portfolio_exposures=portfolio_exposures,
        factor_returns=factor_returns,
        period_start="2024-01-01",
        period_end="2024-12-31",
    )

    print(f"\nPerformance Attribution:")
    print(f"  Total Return: {attribution.total_return:+.2%}")
    print(f"\n  Factor Contributions:")

    # Sort by absolute contribution
    sorted_factors = sorted(
        attribution.factor_returns.items(), key=lambda x: abs(x[1]), reverse=True
    )

    for factor, contribution in sorted_factors:
        pct_of_return = contribution / total_return * 100
        print(f"    {factor.value.upper():12s}: {contribution:+.2%} ({pct_of_return:+.1f}% of return)")

    print(f"\n  Alpha (Specific Return): {attribution.specific_return:+.2%}")
    explained_return = sum(attribution.factor_returns.values())
    print(f"  Explained Return: {explained_return:+.2%}")

    # Key insights
    print(f"\nKey Insights:")
    value_contribution = attribution.factor_returns[StyleFactor.VALUE]
    quality_contribution = attribution.factor_returns[StyleFactor.QUALITY]
    print(f"  - Value exposure (1.2x) × Strong value returns (18%) = {value_contribution:.2%} contribution")
    print(f"  - Quality tilt captured {quality_contribution:.2%} of returns")
    print(f"  - Portfolio generated {attribution.specific_return:.2%} alpha beyond factors")


# ==============================================================================
# Example 4: Factor Timing Strategy
# ==============================================================================


def example_factor_timing():
    """Example 4: Generate factor timing signals."""
    print_section("Example 4: Factor Timing Strategy")

    # Simulate factor returns with different regimes
    key = jax.random.PRNGKey(456)

    # Three factors: VALUE, MOMENTUM, QUALITY
    factors_to_time = [StyleFactor.VALUE, StyleFactor.MOMENTUM, StyleFactor.QUALITY]

    # Simulate different factor return patterns
    n_periods = 252

    # VALUE: Cyclical with recent upturn
    value_returns = jax.random.normal(key, (n_periods,)) * 0.015
    value_returns = value_returns.at[-60:].set(value_returns[-60:] + 0.01)  # Recent outperformance

    # MOMENTUM: Consistent trend
    momentum_returns = jax.random.normal(jax.random.PRNGKey(457), (n_periods,)) * 0.012 + 0.005

    # QUALITY: Recent decline
    quality_returns = jax.random.normal(jax.random.PRNGKey(458), (n_periods,)) * 0.010
    quality_returns = quality_returns.at[-60:].set(quality_returns[-60:] - 0.008)

    factor_returns_dict = {
        StyleFactor.VALUE: value_returns,
        StyleFactor.MOMENTUM: momentum_returns,
        StyleFactor.QUALITY: quality_returns,
    }

    # Market regime indicators
    market_regimes = {
        "low_vol": {"vix": 12.0, "credit_spread": 1.0, "name": "Risk-On"},
        "normal": {"vix": 18.0, "credit_spread": 1.5, "name": "Neutral"},
        "high_vol": {"vix": 28.0, "credit_spread": 2.5, "name": "Risk-Off"},
    }

    strategy = FactorTimingStrategy(lookback_window=63, momentum_window=126)

    for regime_name, regime_indicators in market_regimes.items():
        print(f"\nMarket Regime: {regime_indicators['name']}")
        print(f"  VIX: {regime_indicators['vix']:.1f}")
        print(f"  Credit Spread: {regime_indicators['credit_spread']:.2f}%")
        print(f"\n  Factor Timing Signals:")

        for factor in factors_to_time:
            signal = strategy.generate_timing_signal(
                factor=factor,
                factor_returns_history=factor_returns_dict[factor],
                market_regime_indicators=regime_indicators,
                date="2024-12-31",
            )

            # Interpret signal
            if signal.signal_value > 0.5:
                recommendation = "OVERWEIGHT"
            elif signal.signal_value < -0.5:
                recommendation = "UNDERWEIGHT"
            else:
                recommendation = "NEUTRAL"

            print(f"\n    {factor.value.upper()}:")
            print(f"      Signal: {signal.signal_value:+.2f} → {recommendation}")
            print(f"      Expected Return: {signal.expected_return:+.2%}")
            print(f"      Confidence: {signal.confidence:.0%}")

    # Overall recommendations
    print(f"\n{'='*80}")
    print("Factor Timing Recommendations (Risk-Off Regime):")
    print("="*80)
    print("\n  Recommended Tilts:")
    print("    • QUALITY: Defensive positioning in volatile markets")
    print("    • MOMENTUM: Ride the trend, but monitor for reversal")
    print("    • VALUE: Reduce exposure until market stabilizes")


# ==============================================================================
# Example 5: Factor Allocation Optimization
# ==============================================================================


def example_factor_allocation():
    """Example 5: Optimize factor allocation."""
    print_section("Example 5: Factor Allocation Optimization")

    # Factor universe
    factors = [
        StyleFactor.VALUE,
        StyleFactor.MOMENTUM,
        StyleFactor.QUALITY,
        StyleFactor.SIZE,
    ]

    # Expected returns (forward-looking forecasts)
    expected_returns = {
        StyleFactor.VALUE: 0.09,
        StyleFactor.MOMENTUM: 0.07,
        StyleFactor.QUALITY: 0.06,
        StyleFactor.SIZE: 0.05,
    }

    # Historical factor covariance (annualized)
    factor_covariance = jnp.array([
        [0.0400, 0.0050, 0.0020, 0.0100],  # VALUE
        [0.0050, 0.0300, 0.0080, 0.0060],  # MOMENTUM
        [0.0020, 0.0080, 0.0250, 0.0040],  # QUALITY
        [0.0100, 0.0060, 0.0040, 0.0450],  # SIZE
    ])

    print("Factor Universe:")
    for factor in factors:
        exp_ret = expected_returns[factor]
        idx = factors.index(factor)
        vol = jnp.sqrt(factor_covariance[idx, idx])
        sharpe = exp_ret / vol
        print(f"  {factor.value.upper():10s}: E[R]={exp_ret:.2%}, σ={vol:.2%}, Sharpe={sharpe:.2f}")

    # Strategy 1: Mean-Variance Optimization
    print(f"\n{'='*80}")
    print("Strategy 1: Mean-Variance Optimization (Risk Aversion = 2.5)")
    print("="*80)

    optimizer_mv = FactorAllocationOptimizer(risk_aversion=2.5)
    allocation_mv = optimizer_mv.optimize_mean_variance(
        expected_returns=expected_returns,
        covariance_matrix=factor_covariance,
        factor_order=factors,
        date="2024-12-31",
    )

    print(f"\nOptimal Weights:")
    for factor in factors:
        weight = allocation_mv.factor_weights[factor]
        print(f"  {factor.value.upper():10s}: {weight:+.2%}")

    print(f"\nPortfolio Characteristics:")
    print(f"  Expected Return: {allocation_mv.expected_return:.2%}")
    print(f"  Expected Volatility: {allocation_mv.expected_volatility:.2%}")
    print(f"  Sharpe Ratio: {allocation_mv.sharpe_ratio:.2f}")

    # Strategy 2: Risk Parity
    print(f"\n{'='*80}")
    print("Strategy 2: Risk Parity (Equal Risk Contribution)")
    print("="*80)

    optimizer_rp = FactorAllocationOptimizer()
    allocation_rp = optimizer_rp.optimize_risk_parity(
        covariance_matrix=factor_covariance, factor_order=factors, date="2024-12-31"
    )

    print(f"\nRisk Parity Weights:")
    for factor in factors:
        weight = allocation_rp.factor_weights[factor]
        idx = factors.index(factor)
        factor_vol = jnp.sqrt(factor_covariance[idx, idx])
        print(f"  {factor.value.upper():10s}: {weight:.2%} (Factor Vol: {factor_vol:.2%})")

    print(f"\nPortfolio Volatility: {allocation_rp.expected_volatility:.2%}")

    # Comparison
    print(f"\n{'='*80}")
    print("Strategy Comparison")
    print("="*80)
    print(f"\n  Mean-Variance:")
    print(f"    - Targets highest risk-adjusted returns")
    print(f"    - Sharpe Ratio: {allocation_mv.sharpe_ratio:.2f}")
    print(f"    - May concentrate in high-return factors")

    print(f"\n  Risk Parity:")
    print(f"    - Balances risk contribution across factors")
    print(f"    - More diversified allocation")
    print(f"    - Reduces concentration risk")

    # Concentration metric
    mv_concentration = max(abs(w) for w in allocation_mv.factor_weights.values())
    rp_concentration = max(abs(w) for w in allocation_rp.factor_weights.values())
    print(f"\n  Max Single Factor Weight:")
    print(f"    Mean-Variance: {mv_concentration:.2%}")
    print(f"    Risk Parity: {rp_concentration:.2%}")


# ==============================================================================
# Example 6: Integrated Factor Analysis Workflow
# ==============================================================================


def example_integrated_workflow():
    """Example 6: Complete factor analysis workflow."""
    print_section("Example 6: Integrated Factor Analysis Workflow")

    print("Portfolio Management Process:")
    print("  1. Estimate factor risk model from historical data")
    print("  2. Decompose portfolio risk by factors")
    print("  3. Generate factor timing signals")
    print("  4. Optimize factor allocation")
    print("  5. Attribute realized performance to factors")

    # Setup
    key = jax.random.PRNGKey(999)
    n_periods = 252
    n_assets = 25
    n_factors = 4

    # Generate synthetic data
    factor_returns = jax.random.normal(key, (n_periods, n_factors)) * 0.01
    exposures = jax.random.normal(jax.random.PRNGKey(1000), (n_assets, n_factors))
    returns = (exposures @ factor_returns.T).T
    specific = jax.random.normal(jax.random.PRNGKey(1001), (n_periods, n_assets)) * 0.005
    returns = returns + specific

    # Step 1: Estimate factor model
    print(f"\n{'='*80}")
    print("Step 1: Estimate Factor Risk Model")
    print("="*80)

    estimator = FactorRiskModelEstimator()
    asset_ids = [f"ASSET_{i:02d}" for i in range(n_assets)]
    factor_names = ["VALUE", "MOMENTUM", "SIZE", "QUALITY"]

    risk_model = estimator.estimate_factor_model(
        returns=returns,
        exposures=exposures,
        asset_ids=asset_ids,
        factor_names=factor_names,
        estimation_date="2024-12-31",
    )

    factor_vols = [
        jnp.sqrt(risk_model.factor_covariance[i, i]) * jnp.sqrt(252)
        for i in range(n_factors)
    ]
    print(f"\nFactor Volatilities: {', '.join(f'{v:.1%}' for v in factor_vols)}")

    # Step 2: Portfolio risk decomposition
    print(f"\n{'='*80}")
    print("Step 2: Portfolio Risk Decomposition")
    print("="*80)

    # Equal-weight portfolio
    portfolio_exposures = {
        factor_names[i]: float(jnp.mean(exposures[:, i])) for i in range(n_factors)
    }

    print(f"\nPortfolio Factor Exposures:")
    for factor, exposure in portfolio_exposures.items():
        print(f"  {factor}: {exposure:+.2f}")

    # Aggregate portfolio risk
    exposure_vector = jnp.array(list(portfolio_exposures.values()))
    portfolio_factor_var = float(
        exposure_vector @ risk_model.factor_covariance @ exposure_vector
    )
    portfolio_factor_risk = jnp.sqrt(portfolio_factor_var) * jnp.sqrt(252)

    avg_specific_var = jnp.mean(jnp.array(list(risk_model.specific_variances.values())))
    portfolio_specific_risk = jnp.sqrt(avg_specific_var / n_assets) * jnp.sqrt(252)

    total_risk = jnp.sqrt(portfolio_factor_var + avg_specific_var / n_assets) * jnp.sqrt(252)

    print(f"\nPortfolio Risk:")
    print(f"  Total Risk: {total_risk:.2%}")
    print(f"  Factor Risk: {portfolio_factor_risk:.2%} ({portfolio_factor_risk/total_risk:.1%})")
    print(f"  Specific Risk: {portfolio_specific_risk:.2%} ({portfolio_specific_risk/total_risk:.1%})")

    # Step 3: Summary
    print(f"\n{'='*80}")
    print("Workflow Summary")
    print("="*80)
    print(f"\n✓ Factor model estimated with {n_factors} factors")
    print(f"✓ Portfolio risk decomposed: {portfolio_factor_risk/total_risk:.0%} from factors")
    print(f"✓ Ready for factor timing and allocation decisions")


# ==============================================================================
# Main
# ==============================================================================


def main():
    """Run all factor analysis examples."""
    print("=" * 80)
    print(" FACTOR ANALYSIS FRAMEWORK")
    print(" Comprehensive Examples")
    print("=" * 80)

    example_pca_dimension_reduction()
    example_factor_risk_model()
    example_style_attribution()
    example_factor_timing()
    example_factor_allocation()
    example_integrated_workflow()

    print(f"\n{'='*80}")
    print("Examples Completed!")
    print("="*80)
    print("\nKey Takeaways:")
    print("  • PCA reduces dimensionality while preserving variance structure")
    print("  • Factor models decompose risk into systematic and specific components")
    print("  • Style attribution explains performance through factor exposures")
    print("  • Factor timing exploits time-varying factor premiums")
    print("  • Factor allocation optimizes risk-return tradeoffs")
    print("\nApplications:")
    print("  • Risk management and budgeting")
    print("  • Portfolio construction and optimization")
    print("  • Performance attribution and analysis")
    print("  • Factor-based investment strategies")
    print("  • Dimension reduction for large universes")


if __name__ == "__main__":
    main()
