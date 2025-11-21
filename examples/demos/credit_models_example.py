"""Comprehensive Credit Models Demonstration.

This example showcases all credit risk models implemented in Neutryx:
1. Student-t copula for tail dependence modeling
2. Large Portfolio Approximation (LPA/Vasicek) for CDOs
3. CreditMetrics framework for rating migrations
4. Structural models (Merton and Black-Cox)

These models are fundamental for:
- CDO tranche pricing
- Portfolio credit risk measurement
- Credit correlation trading
- Counterparty credit risk (CVA/DVA)
- Economic capital allocation
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from neutryx.models.credit_models import (
    # Copula models
    GaussianCopulaParams,
    StudentTCopulaParams,
    simulate_gaussian_copula,
    simulate_student_t_copula,
    # Large Portfolio Approximation
    LPAParams,
    vasicek_loss_distribution,
    lpa_expected_loss,
    lpa_unexpected_loss,
    # CreditMetrics
    CreditMetricsParams,
    simulate_credit_migrations,
    # Structural models
    MertonModelParams,
    merton_default_probability,
    merton_distance_to_default,
    merton_equity_value,
    BlackCoxParams,
    black_cox_default_probability,
    credit_spread_from_default_prob,
)

# Set random seed for reproducibility
key = jax.random.PRNGKey(42)


def demo_student_t_copula_tail_dependence():
    """Demonstrate Student-t copula's tail dependence vs Gaussian copula."""
    print("=" * 80)
    print("1. STUDENT-T COPULA FOR TAIL DEPENDENCE")
    print("=" * 80)
    print("\nThe Student-t copula captures tail dependence, making it more realistic")
    print("for modeling default clustering during financial crises.\n")

    n_names = 100
    corr = jnp.eye(n_names) * 0.6 + jnp.ones((n_names, n_names)) * 0.25

    # Gaussian copula (no tail dependence)
    gaussian_params = GaussianCopulaParams(
        correlation_matrix=corr,
        default_probabilities=jnp.full(n_names, 0.02),  # 2% PD
        recovery_rates=jnp.full(n_names, 0.40),  # 40% recovery
    )

    # Student-t copula (with tail dependence)
    student_t_params = StudentTCopulaParams(
        correlation_matrix=corr,
        default_probabilities=jnp.full(n_names, 0.02),
        recovery_rates=jnp.full(n_names, 0.40),
        degrees_of_freedom=4.0,  # Heavy tails (lower = heavier)
    )

    # Simulate both models
    key1, key2 = jax.random.split(key)
    defaults_g, losses_g = simulate_gaussian_copula(key1, gaussian_params, n_simulations=10000)
    defaults_t, losses_t = simulate_student_t_copula(key2, student_t_params, n_simulations=10000)

    # Compare tail risk
    print(f"Portfolio size: {n_names} names")
    print(f"Average correlation: {0.25:.2%}")
    print(f"Individual default probability: {0.02:.2%}")
    print(f"\nGaussian Copula Results:")
    print(f"  Expected Loss:        {losses_g.mean():.4f}")
    print(f"  95th percentile:      {jnp.percentile(losses_g, 95):.4f}")
    print(f"  99th percentile:      {jnp.percentile(losses_g, 99):.4f}")
    print(f"  99.9th percentile:    {jnp.percentile(losses_g, 99.9):.4f}")

    print(f"\nStudent-t Copula Results (df={student_t_params.degrees_of_freedom}):")
    print(f"  Expected Loss:        {losses_t.mean():.4f}")
    print(f"  95th percentile:      {jnp.percentile(losses_t, 95):.4f}")
    print(f"  99th percentile:      {jnp.percentile(losses_t, 99):.4f}")
    print(f"  99.9th percentile:    {jnp.percentile(losses_t, 99.9):.4f}")

    # Calculate tail dependence ratio
    tail_ratio = jnp.percentile(losses_t, 99) / jnp.percentile(losses_g, 99)
    print(f"\nTail Risk Ratio (99th %ile): {tail_ratio:.2f}x")
    print(f"→ Student-t exhibits {(tail_ratio-1)*100:.1f}% more tail risk\n")


def demo_large_portfolio_approximation():
    """Demonstrate LPA/Vasicek model for fast portfolio loss computation."""
    print("=" * 80)
    print("2. LARGE PORTFOLIO APPROXIMATION (LPA) FOR CDOS")
    print("=" * 80)
    print("\nThe LPA (Vasicek model) provides fast semi-analytical computation of")
    print("loss distributions for large homogeneous portfolios. This is the")
    print("foundation of Basel II/III capital requirements.\n")

    params = LPAParams(
        default_probability=0.015,  # 1.5% PD
        correlation=0.20,  # 20% asset correlation
        recovery_rate=0.40,  # 40% recovery
        n_names=1000,  # Large portfolio
    )

    # Calculate key risk metrics
    el = lpa_expected_loss(params)
    ul_99 = lpa_unexpected_loss(params, confidence_level=0.99)
    ul_999 = lpa_unexpected_loss(params, confidence_level=0.999)  # Basel III

    print(f"Portfolio Parameters:")
    print(f"  Size:                 {params.n_names} names")
    print(f"  Default probability:  {params.default_probability:.2%}")
    print(f"  Asset correlation:    {params.correlation:.2%}")
    print(f"  Recovery rate:        {params.recovery_rate:.2%}")
    print(f"  LGD:                  {(1-params.recovery_rate):.2%}")

    print(f"\nRisk Metrics:")
    print(f"  Expected Loss (EL):           {el:.4f} ({el*100:.2f}%)")
    print(f"  Unexpected Loss (99.0% VaR):  {ul_99:.4f} ({ul_99*100:.2f}%)")
    print(f"  Unexpected Loss (99.9% VaR):  {ul_999:.4f} ({ul_999*100:.2f}%)")
    print(f"  Regulatory Capital (UL):      {ul_999*100:.2f}%")

    # Compute full loss distribution
    loss_grid = jnp.linspace(0, 0.15, 200)
    loss_dist = vasicek_loss_distribution(params, loss_grid)

    print(f"\nLoss Distribution Statistics:")
    expected_loss_from_dist = jnp.sum(loss_grid * loss_dist) * (loss_grid[1] - loss_grid[0])
    print(f"  Mean:                 {expected_loss_from_dist:.4f}")
    print(f"  Mode:                 {loss_grid[jnp.argmax(loss_dist)]:.4f}")

    # Calculate VaR and CVaR from distribution
    cumulative_dist = jnp.cumsum(loss_dist) * (loss_grid[1] - loss_grid[0])
    var_99_idx = jnp.searchsorted(cumulative_dist, 0.99)
    var_99 = loss_grid[var_99_idx]
    print(f"  VaR 99%:              {var_99:.4f}")
    print()


def demo_creditmetrics_migrations():
    """Demonstrate CreditMetrics rating migration framework."""
    print("=" * 80)
    print("3. CREDITMETRICS FRAMEWORK INTEGRATION")
    print("=" * 80)
    print("\nCreditMetrics models not just defaults but full rating migrations,")
    print("allowing mark-to-market portfolio valuation and credit VaR.\n")

    # 5-rating system: AAA (0), AA (1), BBB (2), B (3), Default (4)
    transition_matrix = jnp.array([
        [0.90, 0.07, 0.02, 0.005, 0.005],  # From AAA
        [0.02, 0.88, 0.08, 0.015, 0.005],  # From AA
        [0.01, 0.03, 0.88, 0.06, 0.02],    # From BBB
        [0.005, 0.01, 0.05, 0.90, 0.035],  # From B
        [0.00, 0.00, 0.00, 0.00, 1.00],    # From Default (absorbing)
    ])

    # Portfolio: 20 obligors across different ratings
    n_obligors = 20
    current_ratings = jnp.array([
        0, 0, 0, 0, 0,  # 5 AAA
        1, 1, 1, 1, 1,  # 5 AA
        2, 2, 2, 2, 2,  # 5 BBB
        3, 3, 3, 3, 3,  # 5 B
    ])

    # Values in each rating state (declining with credit quality)
    values_by_rating = jnp.array([100.0, 98.0, 92.0, 80.0, 30.0])  # 30% recovery

    # Moderate correlation
    correlation_matrix = jnp.eye(n_obligors) * 0.75 + jnp.ones((n_obligors, n_obligors)) * 0.15

    params = CreditMetricsParams(
        transition_matrix=transition_matrix,
        current_ratings=current_ratings,
        exposures=jnp.ones(n_obligors) * 1000.0,  # $1,000 each
        values_by_rating=values_by_rating,
        correlation_matrix=correlation_matrix,
    )

    # Simulate migrations
    key_cm = jax.random.split(key, 1)[0]
    ratings, portfolio_values = simulate_credit_migrations(key_cm, params, n_simulations=5000)

    # Analyze results
    print(f"Portfolio Composition:")
    print(f"  Total size:           {n_obligors} obligors")
    print(f"  Exposure per name:    $1,000")
    print(f"  Total exposure:       ${n_obligors * 1000:,}")

    print(f"\nInitial Rating Distribution:")
    rating_names = ["AAA", "AA", "BBB", "B", "Default"]
    for i, name in enumerate(rating_names[:-1]):
        count = jnp.sum(current_ratings == i)
        print(f"  {name}:  {int(count)} obligors")

    print(f"\nPortfolio Value Statistics (1-year horizon):")
    initial_value = jnp.sum(params.exposures) * (values_by_rating[current_ratings.astype(int)].mean() / 100.0)
    mean_value = portfolio_values.mean()
    std_value = portfolio_values.std()
    var_95 = jnp.percentile(portfolio_values, 5)  # 5th percentile
    cvar_95 = portfolio_values[portfolio_values <= var_95].mean()

    print(f"  Current value:        ${initial_value:,.0f}")
    print(f"  Expected value:       ${mean_value:,.0f}")
    print(f"  Std deviation:        ${std_value:,.0f}")
    print(f"  VaR (95%):            ${initial_value - var_95:,.0f} loss")
    print(f"  CVaR (95%):           ${initial_value - cvar_95:,.0f} loss")

    # Count defaults
    final_defaults = (ratings == 4).sum(axis=1)
    print(f"\nDefault Analysis:")
    print(f"  Expected defaults:    {final_defaults.mean():.2f}")
    print(f"  Max defaults:         {final_defaults.max()}")
    print(f"  Prob of any default:  {(final_defaults > 0).mean():.2%}")
    print()


def demo_structural_models():
    """Demonstrate Merton and Black-Cox structural models."""
    print("=" * 80)
    print("4. STRUCTURAL MODELS (MERTON & BLACK-COX)")
    print("=" * 80)
    print("\nStructural models link default to firm fundamentals (asset value, debt).")
    print("The Merton model assumes default at maturity, while Black-Cox allows")
    print("continuous monitoring with first-passage time.\n")

    # Company fundamentals
    asset_value = 500.0  # $500M
    debt_value = 400.0   # $400M debt
    maturity = 5.0       # 5 years
    asset_vol = 0.30     # 30% asset volatility
    risk_free_rate = 0.04
    dividend_yield = 0.02

    # --- Merton Model ---
    print("MERTON MODEL")
    print("-" * 40)

    merton_params = MertonModelParams(
        asset_value=asset_value,
        debt_value=debt_value,
        volatility=asset_vol,
        maturity=maturity,
        risk_free_rate=risk_free_rate,
    )

    merton_pd = merton_default_probability(merton_params)
    merton_dd = merton_distance_to_default(merton_params)
    merton_equity = merton_equity_value(merton_params)
    merton_spread = credit_spread_from_default_prob(merton_pd, 0.40, maturity)

    print(f"Firm Parameters:")
    print(f"  Asset value:          ${asset_value}M")
    print(f"  Debt (face value):    ${debt_value}M")
    print(f"  Asset volatility:     {asset_vol:.1%}")
    print(f"  Maturity:             {maturity} years")
    print(f"  Leverage:             {debt_value/asset_value:.1%}")

    print(f"\nMerton Model Results:")
    print(f"  Distance-to-default:  {merton_dd:.3f} std devs")
    print(f"  Default probability:  {merton_pd:.2%}")
    print(f"  Equity value:         ${merton_equity:.2f}M")
    print(f"  Implied spread:       {merton_spread*10000:.0f} bps")

    # --- Black-Cox Model ---
    print(f"\nBLACK-COX MODEL")
    print("-" * 40)

    # Set barrier at 80% of debt (typical covenant level)
    barrier = 0.80 * debt_value

    black_cox_params = BlackCoxParams(
        asset_value=asset_value,
        barrier=barrier,
        volatility=asset_vol,
        maturity=maturity,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
    )

    bc_pd = black_cox_default_probability(black_cox_params)
    bc_spread = credit_spread_from_default_prob(bc_pd, 0.40, maturity)

    print(f"Model Parameters:")
    print(f"  Asset value:          ${asset_value}M")
    print(f"  Barrier level:        ${barrier:.1f}M ({barrier/debt_value:.0%} of debt)")
    print(f"  Asset volatility:     {asset_vol:.1%}")
    print(f"  Maturity:             {maturity} years")

    print(f"\nBlack-Cox Results:")
    print(f"  First-passage PD:     {bc_pd:.2%}")
    print(f"  Implied spread:       {bc_spread*10000:.0f} bps")

    # --- Comparison ---
    print(f"\nMODEL COMPARISON")
    print("-" * 40)
    print(f"  Merton PD:            {merton_pd:.2%}")
    print(f"  Black-Cox PD:         {bc_pd:.2%}")
    print(f"  Difference:           {(bc_pd - merton_pd):.2%} ({(bc_pd/merton_pd - 1)*100:+.1f}%)")
    print()
    print("→ Black-Cox typically gives higher PD due to continuous monitoring")
    print("  (default can occur before maturity at first passage)\n")

    # Sensitivity analysis
    print("SENSITIVITY ANALYSIS")
    print("-" * 40)

    leverage_ratios = jnp.linspace(0.5, 0.95, 10)
    merton_pds = []
    bc_pds = []

    for lev in leverage_ratios:
        debt = asset_value * lev

        m_params = MertonModelParams(
            asset_value=asset_value,
            debt_value=debt,
            volatility=asset_vol,
            maturity=maturity,
            risk_free_rate=risk_free_rate,
        )
        merton_pds.append(merton_default_probability(m_params))

        bc_params = BlackCoxParams(
            asset_value=asset_value,
            barrier=0.80 * debt,
            volatility=asset_vol,
            maturity=maturity,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
        )
        bc_pds.append(black_cox_default_probability(bc_params))

    print(f"Default Probability vs Leverage:")
    print(f"{'Leverage':<12} {'Merton PD':<12} {'Black-Cox PD':<12}")
    print("-" * 40)
    for lev, m_pd, bc_pd in zip(leverage_ratios, merton_pds, bc_pds):
        print(f"{lev:<12.1%} {m_pd:<12.2%} {bc_pd:<12.2%}")
    print()


def demo_cdo_tranche_pricing():
    """Demonstrate CDO tranche analysis using LPA."""
    print("=" * 80)
    print("5. CDO TRANCHE PRICING WITH LPA")
    print("=" * 80)
    print("\nCDO tranches are slices of the portfolio loss distribution.")
    print("LPA provides fast tranche pricing for large portfolios.\n")

    params = LPAParams(
        default_probability=0.02,
        correlation=0.30,  # Higher correlation for credit crisis scenario
        recovery_rate=0.40,
        n_names=500,
    )

    # Define CDO tranches
    tranches = [
        ("Equity", 0.00, 0.03),
        ("Mezzanine", 0.03, 0.07),
        ("Senior", 0.07, 0.15),
        ("Super Senior", 0.15, 1.00),
    ]

    print(f"Portfolio Parameters:")
    print(f"  Size:                 {params.n_names} names")
    print(f"  Default probability:  {params.default_probability:.2%}")
    print(f"  Asset correlation:    {params.correlation:.2%}")
    print(f"  Recovery rate:        {params.recovery_rate:.2%}")

    # Compute loss distribution
    loss_grid = jnp.linspace(0, 0.25, 500)
    loss_dist = vasicek_loss_distribution(params, loss_grid)
    dx = loss_grid[1] - loss_grid[0]

    print(f"\nCDO Tranche Analysis:")
    print(f"{'Tranche':<15} {'Attachment':<12} {'Expected Loss':<15} {'Risk':<10}")
    print("-" * 55)

    for name, lower, upper in tranches:
        # Calculate expected loss for this tranche
        tranche_losses = jnp.maximum(0, jnp.minimum(loss_grid, upper) - lower) / (upper - lower)
        expected_loss = jnp.sum(tranche_losses * loss_dist) * dx

        # Calculate probability of attachment (tranche gets hit)
        prob_attachment = jnp.sum(loss_dist[loss_grid > lower]) * dx

        print(f"{name:<15} [{lower:.0%}-{upper:.0%}]     {expected_loss:<15.2%} {prob_attachment:<10.2%}")

    print()


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "NEUTRYX CREDIT MODELS DEMONSTRATION" + " " * 23 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    # Run all demonstrations
    demo_student_t_copula_tail_dependence()
    demo_large_portfolio_approximation()
    demo_creditmetrics_migrations()
    demo_structural_models()
    demo_cdo_tranche_pricing()

    print("=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nAll credit models are production-ready and fully tested.")
    print("See tests/models/test_credit_models.py for comprehensive test suite.")
    print()
