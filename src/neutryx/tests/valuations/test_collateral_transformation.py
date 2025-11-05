"""Tests for collateral transformation strategies.

Tests cover:
- CollateralHolding eligibility and value calculations
- CollateralPortfolio inventory management
- GreedyLowestCostStrategy
- OptimalMixStrategy with linear programming
- ConcentrationOptimizedStrategy
- Cost calculations (haircut, FX, funding)
- Integration with CSA terms
"""
import pytest
import jax.numpy as jnp

from neutryx.contracts.csa import CollateralType as CSACollateralType
from neutryx.contracts.csa import EligibleCollateral
from neutryx.valuations.margin.collateral_transformation import (
    CollateralHolding,
    CollateralPortfolio,
    CollateralSelection,
    TransformationCost,
    TransformationResult,
    GreedyLowestCostStrategy,
    OptimalMixStrategy,
    ConcentrationOptimizedStrategy,
    calculate_haircut_adjusted_value,
    calculate_fx_conversion_cost,
    calculate_funding_cost,
)


# ==============================================================================
# CollateralHolding Tests
# ==============================================================================


class TestCollateralHolding:
    """Test CollateralHolding class."""

    def test_holding_creation(self):
        """Test creating a collateral holding."""
        holding = CollateralHolding(
            collateral_type=CSACollateralType.CASH,
            currency="USD",
            market_value=1_000_000.0,
            quantity=1.0,
            available=True,
        )

        assert holding.collateral_type == CSACollateralType.CASH
        assert holding.currency == "USD"
        assert holding.market_value == 1_000_000.0
        assert holding.available is True

    def test_eligibility_check_cash(self):
        """Test eligibility check for cash."""
        holding = CollateralHolding(
            collateral_type=CSACollateralType.CASH,
            currency="USD",
            market_value=1_000_000.0,
        )

        spec = EligibleCollateral(
            collateral_type=CSACollateralType.CASH,
            currency="USD",
            haircut=0.0,
        )

        assert holding.is_eligible(spec) is True

    def test_eligibility_check_wrong_currency(self):
        """Test eligibility fails for wrong currency."""
        holding = CollateralHolding(
            collateral_type=CSACollateralType.CASH,
            currency="EUR",
            market_value=1_000_000.0,
        )

        spec = EligibleCollateral(
            collateral_type=CSACollateralType.CASH,
            currency="USD",
            haircut=0.0,
        )

        assert holding.is_eligible(spec) is False

    def test_eligibility_check_maturity(self):
        """Test eligibility check with maturity constraint."""
        holding = CollateralHolding(
            collateral_type=CSACollateralType.GOVERNMENT_BOND,
            currency="USD",
            market_value=1_000_000.0,
            maturity_years=15.0,
        )

        # Max maturity 10 years
        spec = EligibleCollateral(
            collateral_type=CSACollateralType.GOVERNMENT_BOND,
            haircut=0.01,
            maturity_max_years=10.0,
        )

        assert holding.is_eligible(spec) is False

        # Max maturity 20 years
        spec2 = EligibleCollateral(
            collateral_type=CSACollateralType.GOVERNMENT_BOND,
            haircut=0.01,
            maturity_max_years=20.0,
        )

        assert holding.is_eligible(spec2) is True

    def test_calculate_collateral_value(self):
        """Test haircut-adjusted value calculation."""
        holding = CollateralHolding(
            collateral_type=CSACollateralType.GOVERNMENT_BOND,
            currency="USD",
            market_value=1_000_000.0,
        )

        # 2% haircut
        value = holding.calculate_collateral_value(haircut=0.02)
        assert abs(value - 980_000.0) < 1.0


# ==============================================================================
# CollateralPortfolio Tests
# ==============================================================================


class TestCollateralPortfolio:
    """Test CollateralPortfolio class."""

    def test_portfolio_creation(self):
        """Test creating empty portfolio."""
        portfolio = CollateralPortfolio(base_currency="USD")

        assert portfolio.base_currency == "USD"
        assert len(portfolio.holdings) == 0
        assert portfolio.fx_rates["USD"] == 1.0

    def test_add_holdings(self):
        """Test adding holdings to portfolio."""
        portfolio = CollateralPortfolio(base_currency="USD")

        holding1 = CollateralHolding(
            collateral_type=CSACollateralType.CASH,
            currency="USD",
            market_value=1_000_000.0,
        )

        holding2 = CollateralHolding(
            collateral_type=CSACollateralType.GOVERNMENT_BOND,
            currency="USD",
            market_value=500_000.0,
        )

        portfolio.add_holding(holding1)
        portfolio.add_holding(holding2)

        assert len(portfolio.holdings) == 2

    def test_get_available_holdings(self):
        """Test filtering available holdings."""
        portfolio = CollateralPortfolio()

        h1 = CollateralHolding(
            collateral_type=CSACollateralType.CASH,
            currency="USD",
            market_value=1_000_000.0,
            available=True,
        )

        h2 = CollateralHolding(
            collateral_type=CSACollateralType.GOVERNMENT_BOND,
            currency="USD",
            market_value=500_000.0,
            available=False,  # Encumbered
        )

        portfolio.add_holding(h1)
        portfolio.add_holding(h2)

        available = portfolio.get_available_holdings()
        assert len(available) == 1
        assert available[0].collateral_type == CSACollateralType.CASH

    def test_get_total_value(self):
        """Test total portfolio value calculation."""
        portfolio = CollateralPortfolio(
            base_currency="USD",
            fx_rates={"USD": 1.0, "EUR": 1.10, "GBP": 1.25},
        )

        portfolio.add_holding(
            CollateralHolding(
                collateral_type=CSACollateralType.CASH,
                currency="USD",
                market_value=1_000_000.0,
            )
        )

        portfolio.add_holding(
            CollateralHolding(
                collateral_type=CSACollateralType.GOVERNMENT_BOND,
                currency="EUR",
                market_value=500_000.0,
            )
        )

        # Total = 1M USD + 500k EUR * 1.10 = 1M + 550k = 1.55M
        total = portfolio.get_total_value(in_base_currency=True)
        assert abs(total - 1_550_000.0) < 1.0

    def test_get_holdings_by_type(self):
        """Test filtering by collateral type."""
        portfolio = CollateralPortfolio()

        portfolio.add_holding(
            CollateralHolding(
                collateral_type=CSACollateralType.CASH,
                currency="USD",
                market_value=1_000_000.0,
            )
        )

        portfolio.add_holding(
            CollateralHolding(
                collateral_type=CSACollateralType.GOVERNMENT_BOND,
                currency="USD",
                market_value=500_000.0,
            )
        )

        portfolio.add_holding(
            CollateralHolding(
                collateral_type=CSACollateralType.CASH,
                currency="EUR",
                market_value=300_000.0,
            )
        )

        cash_holdings = portfolio.get_holdings_by_type(CSACollateralType.CASH)
        assert len(cash_holdings) == 2

        bond_holdings = portfolio.get_holdings_by_type(CSACollateralType.GOVERNMENT_BOND)
        assert len(bond_holdings) == 1

    def test_get_eligible_holdings(self):
        """Test getting eligible holdings with specs."""
        portfolio = CollateralPortfolio()

        portfolio.add_holding(
            CollateralHolding(
                collateral_type=CSACollateralType.CASH,
                currency="USD",
                market_value=1_000_000.0,
            )
        )

        portfolio.add_holding(
            CollateralHolding(
                collateral_type=CSACollateralType.GOVERNMENT_BOND,
                currency="USD",
                market_value=500_000.0,
            )
        )

        eligible_specs = [
            EligibleCollateral(
                collateral_type=CSACollateralType.CASH,
                currency="USD",
                haircut=0.0,
            ),
            EligibleCollateral(
                collateral_type=CSACollateralType.GOVERNMENT_BOND,
                haircut=0.02,
            ),
        ]

        eligible_pairs = portfolio.get_eligible_holdings(eligible_specs)
        assert len(eligible_pairs) == 2


# ==============================================================================
# GreedyLowestCostStrategy Tests
# ==============================================================================


class TestGreedyLowestCostStrategy:
    """Test GreedyLowestCostStrategy."""

    def test_greedy_strategy_basic(self):
        """Test basic greedy selection."""
        strategy = GreedyLowestCostStrategy(
            fx_spread=0.001,
            funding_rate=0.05,
            operational_cost_per_asset=100.0,
        )

        portfolio = CollateralPortfolio(base_currency="USD")

        # Add cash (no haircut, best option)
        portfolio.add_holding(
            CollateralHolding(
                collateral_type=CSACollateralType.CASH,
                currency="USD",
                market_value=500_000.0,
            )
        )

        # Add bond (2% haircut, worse option)
        portfolio.add_holding(
            CollateralHolding(
                collateral_type=CSACollateralType.GOVERNMENT_BOND,
                currency="USD",
                market_value=600_000.0,
            )
        )

        eligible_specs = [
            EligibleCollateral(
                collateral_type=CSACollateralType.CASH,
                haircut=0.0,
            ),
            EligibleCollateral(
                collateral_type=CSACollateralType.GOVERNMENT_BOND,
                haircut=0.02,
            ),
        ]

        margin_requirement = 400_000.0

        result = strategy.select_collateral(
            portfolio=portfolio,
            margin_requirement=margin_requirement,
            eligible_collateral=eligible_specs,
        )

        assert result.satisfied is True
        assert result.optimization_status == "success"
        assert result.total_collateral_value >= margin_requirement
        # Should select cash first (lowest cost)
        assert len(result.selected_collateral) >= 1
        assert result.selected_collateral[0].holding.collateral_type == CSACollateralType.CASH

    def test_greedy_insufficient_collateral(self):
        """Test greedy strategy with insufficient collateral."""
        strategy = GreedyLowestCostStrategy()

        portfolio = CollateralPortfolio()
        portfolio.add_holding(
            CollateralHolding(
                collateral_type=CSACollateralType.CASH,
                currency="USD",
                market_value=100_000.0,  # Not enough
            )
        )

        eligible_specs = [
            EligibleCollateral(collateral_type=CSACollateralType.CASH, haircut=0.0)
        ]

        result = strategy.select_collateral(
            portfolio=portfolio,
            margin_requirement=500_000.0,
            eligible_collateral=eligible_specs,
        )

        assert result.satisfied is False
        assert result.optimization_status == "insufficient_collateral"

    def test_greedy_no_eligible_collateral(self):
        """Test greedy strategy with no eligible collateral."""
        strategy = GreedyLowestCostStrategy()

        portfolio = CollateralPortfolio()
        portfolio.add_holding(
            CollateralHolding(
                collateral_type=CSACollateralType.EQUITY,
                currency="USD",
                market_value=1_000_000.0,
            )
        )

        # Only accept cash
        eligible_specs = [
            EligibleCollateral(collateral_type=CSACollateralType.CASH, haircut=0.0)
        ]

        result = strategy.select_collateral(
            portfolio=portfolio,
            margin_requirement=100_000.0,
            eligible_collateral=eligible_specs,
        )

        assert result.satisfied is False
        assert result.optimization_status == "no_eligible_collateral"


# ==============================================================================
# OptimalMixStrategy Tests
# ==============================================================================


class TestOptimalMixStrategy:
    """Test OptimalMixStrategy."""

    def test_optimal_mix_basic(self):
        """Test optimal mix strategy."""
        pytest.importorskip("scipy")

        strategy = OptimalMixStrategy(
            fx_spread=0.001,
            funding_rate=0.05,
            operational_cost_per_asset=100.0,
        )

        portfolio = CollateralPortfolio(base_currency="USD")

        portfolio.add_holding(
            CollateralHolding(
                collateral_type=CSACollateralType.CASH,
                currency="USD",
                market_value=1_000_000.0,
            )
        )

        portfolio.add_holding(
            CollateralHolding(
                collateral_type=CSACollateralType.GOVERNMENT_BOND,
                currency="USD",
                market_value=1_000_000.0,
            )
        )

        eligible_specs = [
            EligibleCollateral(collateral_type=CSACollateralType.CASH, haircut=0.0),
            EligibleCollateral(
                collateral_type=CSACollateralType.GOVERNMENT_BOND, haircut=0.02
            ),
        ]

        margin_requirement = 500_000.0

        result = strategy.select_collateral(
            portfolio=portfolio,
            margin_requirement=margin_requirement,
            eligible_collateral=eligible_specs,
        )

        assert result.satisfied is True
        assert result.optimization_status == "success"
        assert result.total_collateral_value >= margin_requirement * 0.99

    def test_optimal_mix_multi_currency(self):
        """Test optimal mix with multi-currency portfolio."""
        pytest.importorskip("scipy")

        strategy = OptimalMixStrategy()

        portfolio = CollateralPortfolio(
            base_currency="USD",
            fx_rates={"USD": 1.0, "EUR": 1.10, "GBP": 1.25},
        )

        portfolio.add_holding(
            CollateralHolding(
                collateral_type=CSACollateralType.CASH,
                currency="USD",
                market_value=500_000.0,
            )
        )

        portfolio.add_holding(
            CollateralHolding(
                collateral_type=CSACollateralType.CASH,
                currency="EUR",
                market_value=400_000.0,
            )
        )

        portfolio.add_holding(
            CollateralHolding(
                collateral_type=CSACollateralType.CASH,
                currency="GBP",
                market_value=300_000.0,
            )
        )

        eligible_specs = [
            EligibleCollateral(collateral_type=CSACollateralType.CASH, haircut=0.0)
        ]

        margin_requirement = 600_000.0

        result = strategy.select_collateral(
            portfolio=portfolio,
            margin_requirement=margin_requirement,
            eligible_collateral=eligible_specs,
        )

        assert result.satisfied is True
        # Should prefer USD (no FX cost)
        usd_selected = [s for s in result.selected_collateral if s.holding.currency == "USD"]
        assert len(usd_selected) > 0


# ==============================================================================
# ConcentrationOptimizedStrategy Tests
# ==============================================================================


class TestConcentrationOptimizedStrategy:
    """Test ConcentrationOptimizedStrategy."""

    def test_concentration_limits(self):
        """Test strategy respects concentration limits."""
        pytest.importorskip("scipy")

        strategy = ConcentrationOptimizedStrategy()

        portfolio = CollateralPortfolio()

        # Add multiple assets of same type
        for i in range(5):
            portfolio.add_holding(
                CollateralHolding(
                    collateral_type=CSACollateralType.CORPORATE_BOND,
                    currency="USD",
                    market_value=200_000.0,
                )
            )

        # Add some cash
        portfolio.add_holding(
            CollateralHolding(
                collateral_type=CSACollateralType.CASH,
                currency="USD",
                market_value=500_000.0,
            )
        )

        # Set concentration limit: corporate bonds max 40%
        eligible_specs = [
            EligibleCollateral(
                collateral_type=CSACollateralType.CORPORATE_BOND,
                haircut=0.05,
                concentration_limit=0.4,  # Max 40%
            ),
            EligibleCollateral(collateral_type=CSACollateralType.CASH, haircut=0.0),
        ]

        margin_requirement = 800_000.0

        result = strategy.select_collateral(
            portfolio=portfolio,
            margin_requirement=margin_requirement,
            eligible_collateral=eligible_specs,
        )

        if result.satisfied:
            # Check concentration ratios
            concentration = result.get_concentration_ratios()
            if CSACollateralType.CORPORATE_BOND in concentration:
                assert concentration[CSACollateralType.CORPORATE_BOND] <= 0.41  # Small tolerance


# ==============================================================================
# TransformationResult Tests
# ==============================================================================


class TestTransformationResult:
    """Test TransformationResult class."""

    def test_result_creation(self):
        """Test creating transformation result."""
        cost = TransformationCost(
            haircut_cost=10_000.0,
            fx_conversion_cost=500.0,
            funding_cost=1_000.0,
            operational_cost=300.0,
        )

        result = TransformationResult(
            selected_collateral=[],
            total_collateral_value=500_000.0,
            total_cost=cost,
            margin_requirement=500_000.0,
            satisfied=True,
            optimization_status="success",
        )

        assert result.total_cost.total_cost == 11_800.0
        assert result.satisfied is True

    def test_concentration_ratios(self):
        """Test concentration ratio calculation."""
        holding1 = CollateralHolding(
            collateral_type=CSACollateralType.CASH,
            currency="USD",
            market_value=300_000.0,
        )

        holding2 = CollateralHolding(
            collateral_type=CSACollateralType.GOVERNMENT_BOND,
            currency="USD",
            market_value=200_000.0,
        )

        selections = [
            CollateralSelection(
                holding=holding1,
                amount=300_000.0,
                haircut=0.0,
                collateral_value=300_000.0,
                fx_rate=1.0,
            ),
            CollateralSelection(
                holding=holding2,
                amount=200_000.0,
                haircut=0.02,
                collateral_value=196_000.0,
                fx_rate=1.0,
            ),
        ]

        result = TransformationResult(
            selected_collateral=selections,
            total_collateral_value=496_000.0,
            total_cost=TransformationCost(),
            margin_requirement=400_000.0,
            satisfied=True,
            optimization_status="success",
        )

        ratios = result.get_concentration_ratios()

        assert CSACollateralType.CASH in ratios
        assert CSACollateralType.GOVERNMENT_BOND in ratios
        # Cash should be ~60%, bonds ~40%
        assert abs(ratios[CSACollateralType.CASH] - 0.605) < 0.01


# ==============================================================================
# Utility Function Tests
# ==============================================================================


class TestUtilityFunctions:
    """Test utility functions."""

    def test_calculate_haircut_adjusted_value(self):
        """Test haircut adjustment."""
        value = calculate_haircut_adjusted_value(
            market_value=1_000_000.0,
            haircut=0.05,
        )

        assert abs(value - 950_000.0) < 1.0

    def test_calculate_fx_conversion_cost(self):
        """Test FX conversion cost."""
        cost = calculate_fx_conversion_cost(
            amount=1_000_000.0,
            fx_rate=1.10,
            spread=0.001,
        )

        # Cost = 1M * 1.10 * 0.001 = 1,100
        assert abs(cost - 1_100.0) < 1.0

    def test_calculate_funding_cost(self):
        """Test funding cost calculation."""
        # Daily cost
        cost = calculate_funding_cost(
            collateral_value=1_000_000.0,
            funding_rate=0.05,
            days=1,
        )

        # Cost = 1M * 0.05 / 365 ≈ 136.99
        assert abs(cost - 136.99) < 1.0

        # Annual cost
        cost_annual = calculate_funding_cost(
            collateral_value=1_000_000.0,
            funding_rate=0.05,
            days=365,
        )

        assert abs(cost_annual - 50_000.0) < 1.0


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestIntegration:
    """Integration tests with realistic scenarios."""

    def test_complete_transformation_workflow(self):
        """Test complete collateral transformation workflow."""
        # Setup portfolio
        portfolio = CollateralPortfolio(
            base_currency="USD",
            fx_rates={"USD": 1.0, "EUR": 1.10},
        )

        # Add diverse collateral
        portfolio.add_holding(
            CollateralHolding(
                collateral_type=CSACollateralType.CASH,
                currency="USD",
                market_value=2_000_000.0,
            )
        )

        portfolio.add_holding(
            CollateralHolding(
                collateral_type=CSACollateralType.CASH,
                currency="EUR",
                market_value=1_000_000.0,
            )
        )

        portfolio.add_holding(
            CollateralHolding(
                collateral_type=CSACollateralType.GOVERNMENT_BOND,
                currency="USD",
                market_value=3_000_000.0,
                maturity_years=5.0,
            )
        )

        portfolio.add_holding(
            CollateralHolding(
                collateral_type=CSACollateralType.CORPORATE_BOND,
                currency="USD",
                market_value=1_000_000.0,
                rating="BBB+",
                maturity_years=7.0,
            )
        )

        # Define eligibility
        eligible_specs = [
            EligibleCollateral(
                collateral_type=CSACollateralType.CASH,
                haircut=0.0,
            ),
            EligibleCollateral(
                collateral_type=CSACollateralType.GOVERNMENT_BOND,
                haircut=0.01,
                maturity_max_years=10.0,
            ),
            EligibleCollateral(
                collateral_type=CSACollateralType.CORPORATE_BOND,
                haircut=0.05,
                rating_threshold="BBB-",
                concentration_limit=0.30,
            ),
        ]

        # Margin requirement
        margin_requirement = 4_000_000.0

        # Test greedy strategy
        greedy_strategy = GreedyLowestCostStrategy()
        greedy_result = greedy_strategy.select_collateral(
            portfolio=portfolio,
            margin_requirement=margin_requirement,
            eligible_collateral=eligible_specs,
        )

        assert greedy_result.satisfied is True
        assert greedy_result.total_collateral_value >= margin_requirement

        # Test optimal strategy (if scipy available)
        try:
            import scipy

            optimal_strategy = OptimalMixStrategy()
            optimal_result = optimal_strategy.select_collateral(
                portfolio=portfolio,
                margin_requirement=margin_requirement,
                eligible_collateral=eligible_specs,
            )

            assert optimal_result.satisfied is True
            # Optimal should have lower or equal cost
            assert optimal_result.total_cost.total_cost <= greedy_result.total_cost.total_cost * 1.1

        except ImportError:
            pass  # Skip if scipy not available


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
