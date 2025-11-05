"""Tests for position limits and risk controls.

Tests cover:
- Position limits (notional, VaR, concentration, issuer)
- Limit checking and breach detection
- Pre-trade controls
- What-if scenario analysis
- Limit utilization tracking
"""
import pytest

from neutryx.risk.limits import (
    BreachSeverity,
    ConcentrationLimit,
    IssuerLimit,
    LimitBreach,
    LimitManager,
    LimitStatus,
    LimitType,
    NotionalLimit,
    PreTradeCheck,
    VaRLimit,
    WhatIfScenario,
    pre_trade_control,
    what_if_analysis,
)


# ==============================================================================
# Limit Tests
# ==============================================================================


class TestNotionalLimit:
    """Test notional limit functionality."""

    def test_notional_limit_creation(self):
        """Test creating a notional limit."""
        limit = NotionalLimit(
            name="Rates Swaps",
            hard_limit=1e9,
            soft_limit=800e6,
            scope={"desk": "rates", "product": "swaps"},
        )

        assert limit.name == "Rates Swaps"
        assert limit.limit_type == LimitType.NOTIONAL
        assert limit.hard_limit == 1e9
        assert limit.soft_limit == 800e6
        assert limit.scope["desk"] == "rates"

    def test_limit_status_ok(self):
        """Test limit status when within limits."""
        limit = NotionalLimit("Test", hard_limit=1000, soft_limit=800)

        status = limit.check_utilization(500)
        assert status == LimitStatus.OK

    def test_limit_status_warning(self):
        """Test limit status at warning threshold."""
        limit = NotionalLimit("Test", hard_limit=1000, soft_limit=800, warning_threshold=0.7)

        # 75% utilization should trigger warning (above 70% threshold, below 80% soft limit)
        status = limit.check_utilization(750)
        assert status == LimitStatus.WARNING

    def test_limit_status_soft_breach(self):
        """Test soft limit breach."""
        limit = NotionalLimit("Test", hard_limit=1000, soft_limit=800)

        status = limit.check_utilization(900)
        assert status == LimitStatus.SOFT_BREACH

    def test_limit_status_hard_breach(self):
        """Test hard limit breach."""
        limit = NotionalLimit("Test", hard_limit=1000, soft_limit=800)

        status = limit.check_utilization(1100)
        assert status == LimitStatus.HARD_BREACH

    def test_utilization_percentage(self):
        """Test utilization percentage calculation."""
        limit = NotionalLimit("Test", hard_limit=1000)

        utilization = limit.utilization_pct(500)
        assert utilization == 50.0

        utilization = limit.utilization_pct(1200)
        assert utilization == 120.0

    def test_available_capacity(self):
        """Test available capacity calculation."""
        limit = NotionalLimit("Test", hard_limit=1000)

        capacity = limit.available_capacity(600)
        assert capacity == 400

        # Over limit should return 0
        capacity = limit.available_capacity(1200)
        assert capacity == 0


class TestVaRLimit:
    """Test VaR limit functionality."""

    def test_var_limit_creation(self):
        """Test creating a VaR limit."""
        limit = VaRLimit(
            name="Equity Desk VaR",
            hard_limit=10e6,
            confidence_level=0.99,
            horizon_days=1,
            scope={"desk": "equity"},
        )

        assert limit.limit_type == LimitType.VAR
        assert limit.confidence_level == 0.99
        assert limit.horizon_days == 1

    def test_var_limit_checking(self):
        """Test VaR limit checking."""
        limit = VaRLimit("VaR", hard_limit=10e6, soft_limit=8e6)

        # Within limits
        assert limit.check_utilization(7e6) == LimitStatus.OK

        # Soft breach
        assert limit.check_utilization(9e6) == LimitStatus.SOFT_BREACH

        # Hard breach
        assert limit.check_utilization(11e6) == LimitStatus.HARD_BREACH


class TestConcentrationLimit:
    """Test concentration limit functionality."""

    def test_concentration_limit_creation(self):
        """Test creating concentration limit."""
        limit = ConcentrationLimit(
            name="Single Name",
            hard_limit=0.10,  # 10% max
            concentration_type="single_name",
            scope={"portfolio": "credit"},
        )

        assert limit.limit_type == LimitType.CONCENTRATION
        assert limit.concentration_type == "single_name"
        assert limit.max_percentage == 0.10

    def test_concentration_limit_checking(self):
        """Test concentration limit checking."""
        limit = ConcentrationLimit("Single Name", hard_limit=0.10, soft_limit=0.08)

        # 5% concentration - OK
        assert limit.check_utilization(0.05) == LimitStatus.OK

        # 9% concentration - soft breach
        assert limit.check_utilization(0.09) == LimitStatus.SOFT_BREACH

        # 12% concentration - hard breach
        assert limit.check_utilization(0.12) == LimitStatus.HARD_BREACH


class TestIssuerLimit:
    """Test issuer limit functionality."""

    def test_issuer_limit_creation(self):
        """Test creating issuer limit."""
        limit = IssuerLimit(
            name="Bank XYZ Exposure",
            issuer_id="BANK_XYZ",
            hard_limit=500e6,
            credit_rating="AA",
        )

        assert limit.limit_type == LimitType.ISSUER
        assert limit.issuer_id == "BANK_XYZ"
        assert limit.credit_rating == "AA"

    def test_issuer_limit_checking(self):
        """Test issuer limit checking."""
        limit = IssuerLimit("Bank XYZ", issuer_id="XYZ", hard_limit=500e6)

        assert limit.check_utilization(400e6) == LimitStatus.OK
        assert limit.check_utilization(600e6) == LimitStatus.HARD_BREACH


# ==============================================================================
# Limit Manager Tests
# ==============================================================================


class TestLimitManager:
    """Test limit manager functionality."""

    def test_limit_manager_creation(self):
        """Test creating limit manager."""
        manager = LimitManager()

        assert len(manager.limits) == 0
        assert len(manager.breaches) == 0

    def test_add_remove_limits(self):
        """Test adding and removing limits."""
        manager = LimitManager()

        limit = NotionalLimit("Test", hard_limit=1000)
        manager.add_limit(limit)

        assert "Test" in manager.limits
        assert len(manager.limits) == 1

        manager.remove_limit("Test")
        assert "Test" not in manager.limits
        assert len(manager.limits) == 0

    def test_check_single_limit(self):
        """Test checking a single limit."""
        manager = LimitManager()
        limit = NotionalLimit("Test", hard_limit=1000, soft_limit=800)
        manager.add_limit(limit)

        # Within limits - no breach
        breach = manager.check_limit("Test", 700)
        assert breach is None

        # Soft breach
        breach = manager.check_limit("Test", 900)
        assert breach is not None
        assert breach.status == LimitStatus.SOFT_BREACH
        assert len(manager.breaches) == 1

        # Hard breach
        breach = manager.check_limit("Test", 1100)
        assert breach is not None
        assert breach.status == LimitStatus.HARD_BREACH
        assert len(manager.breaches) == 2

    def test_check_all_limits(self):
        """Test checking multiple limits."""
        manager = LimitManager()
        manager.add_limit(NotionalLimit("Limit1", hard_limit=1000))
        manager.add_limit(NotionalLimit("Limit2", hard_limit=500))
        manager.add_limit(NotionalLimit("Limit3", hard_limit=2000))

        exposures = {
            "Limit1": 800,  # OK
            "Limit2": 600,  # Breach
            "Limit3": 1500,  # OK
        }

        breaches = manager.check_all_limits(exposures)

        assert len(breaches) == 1
        assert breaches[0].limit.name == "Limit2"

    def test_limit_status_summary(self):
        """Test generating limit status summary."""
        manager = LimitManager()
        manager.add_limit(NotionalLimit("Limit1", hard_limit=1000, soft_limit=800))
        manager.add_limit(NotionalLimit("Limit2", hard_limit=500, soft_limit=400))
        manager.add_limit(NotionalLimit("Limit3", hard_limit=2000, soft_limit=1600, warning_threshold=0.4))

        exposures = {
            "Limit1": 700,  # OK
            "Limit2": 550,  # Hard breach
            "Limit3": 1000,  # Warning (above 40% = 800, below soft limit 1600)
        }

        summary = manager.get_limit_status_summary(exposures)

        assert summary["total_limits"] == 3
        assert summary["breached_limits"] == 1
        assert summary["warning_limits"] == 1
        assert "Limit2" in summary["breaches"]
        assert "Limit3" in summary["warnings"]

    def test_get_available_capacity(self):
        """Test getting available capacity."""
        manager = LimitManager()
        manager.add_limit(NotionalLimit("Test", hard_limit=1000))

        capacity = manager.get_available_capacity("Test", 600)
        assert capacity == 400

    def test_get_breaches_with_filtering(self):
        """Test filtering breach history."""
        manager = LimitManager()
        manager.add_limit(NotionalLimit("Limit1", hard_limit=1000, soft_limit=800))
        manager.add_limit(VaRLimit("VaRLimit", hard_limit=5e6, soft_limit=4e6))

        # Generate breaches
        manager.check_limit("Limit1", 900)  # Soft breach
        manager.check_limit("Limit1", 1100)  # Hard breach
        manager.check_limit("VaRLimit", 4.5e6)  # Soft breach

        # Get all breaches
        all_breaches = manager.get_breaches()
        assert len(all_breaches) == 3

        # Get critical breaches only
        critical = manager.get_breaches(severity=BreachSeverity.CRITICAL)
        assert len(critical) == 1

        # Get VaR limit breaches
        var_breaches = manager.get_breaches(limit_type=LimitType.VAR)
        assert len(var_breaches) == 1

    def test_clear_breach_history(self):
        """Test clearing breach history."""
        manager = LimitManager()
        manager.add_limit(NotionalLimit("Test", hard_limit=1000))

        manager.check_limit("Test", 1100)
        assert len(manager.breaches) == 1

        manager.clear_breach_history()
        assert len(manager.breaches) == 0


# ==============================================================================
# Limit Breach Tests
# ==============================================================================


class TestLimitBreach:
    """Test limit breach functionality."""

    def test_breach_creation(self):
        """Test creating a limit breach."""
        limit = NotionalLimit("Test", hard_limit=1000)
        breach = LimitBreach(
            limit=limit,
            current_exposure=1100,
            excess_amount=100,
            status=LimitStatus.HARD_BREACH,
        )

        assert breach.limit.name == "Test"
        assert breach.current_exposure == 1100
        assert breach.excess_amount == 100
        assert breach.status == LimitStatus.HARD_BREACH

    def test_breach_severity(self):
        """Test breach severity determination."""
        limit = NotionalLimit("Test", hard_limit=1000)

        hard_breach = LimitBreach(
            limit=limit,
            current_exposure=1100,
            excess_amount=100,
            status=LimitStatus.HARD_BREACH,
        )
        assert hard_breach.severity == BreachSeverity.CRITICAL

        soft_breach = LimitBreach(
            limit=limit,
            current_exposure=900,
            excess_amount=100,
            status=LimitStatus.SOFT_BREACH,
        )
        assert soft_breach.severity == BreachSeverity.WARNING

    def test_breach_to_alert(self):
        """Test converting breach to alert."""
        limit = NotionalLimit("Test", hard_limit=1000, scope={"desk": "rates"})
        breach = LimitBreach(
            limit=limit,
            current_exposure=1100,
            excess_amount=100,
            status=LimitStatus.HARD_BREACH,
            trade_details={"trade_id": "T123"},
        )

        alert = breach.to_alert()

        assert alert["limit_name"] == "Test"
        assert alert["limit_type"] == "notional"
        assert alert["hard_limit"] == 1000
        assert alert["current_exposure"] == 1100
        assert alert["excess_amount"] == 100
        assert alert["status"] == "hard_breach"
        assert alert["severity"] == "critical"
        assert alert["scope"]["desk"] == "rates"
        assert alert["trade_details"]["trade_id"] == "T123"


# ==============================================================================
# Pre-Trade Control Tests
# ==============================================================================


class TestPreTradeControl:
    """Test pre-trade control functionality."""

    def test_pre_trade_check_approved(self):
        """Test pre-trade check with approved trade."""
        manager = LimitManager()
        manager.add_limit(NotionalLimit("Swaps", hard_limit=1e9, soft_limit=800e6))

        current_exposures = {"Swaps": 600e6}
        trade_impact = {"Swaps": 150e6}  # Proposed $150M trade

        check = pre_trade_control(manager, current_exposures, trade_impact)

        assert check.approved is True
        assert len(check.breaches) == 0
        assert check.has_hard_breaches is False
        assert check.has_soft_breaches is False

    def test_pre_trade_check_soft_breach_allowed(self):
        """Test pre-trade check with soft breach (allowed)."""
        manager = LimitManager()
        manager.add_limit(NotionalLimit("Swaps", hard_limit=1e9, soft_limit=800e6))

        current_exposures = {"Swaps": 700e6}
        trade_impact = {"Swaps": 200e6}  # Would breach soft limit

        check = pre_trade_control(
            manager,
            current_exposures,
            trade_impact,
            allow_soft_breaches=True,
        )

        assert check.approved is True  # Soft breaches allowed
        assert len(check.breaches) == 1
        assert check.has_soft_breaches is True
        assert check.has_hard_breaches is False

    def test_pre_trade_check_soft_breach_rejected(self):
        """Test pre-trade check with soft breach (not allowed)."""
        manager = LimitManager()
        manager.add_limit(NotionalLimit("Swaps", hard_limit=1e9, soft_limit=800e6))

        current_exposures = {"Swaps": 700e6}
        trade_impact = {"Swaps": 200e6}

        check = pre_trade_control(
            manager,
            current_exposures,
            trade_impact,
            allow_soft_breaches=False,
        )

        assert check.approved is False  # Soft breaches not allowed
        assert check.has_soft_breaches is True

    def test_pre_trade_check_hard_breach(self):
        """Test pre-trade check with hard breach."""
        manager = LimitManager()
        manager.add_limit(NotionalLimit("Swaps", hard_limit=1e9))

        current_exposures = {"Swaps": 900e6}
        trade_impact = {"Swaps": 200e6}  # Would breach hard limit

        check = pre_trade_control(manager, current_exposures, trade_impact)

        assert check.approved is False
        assert check.has_hard_breaches is True
        assert len(check.breaches) == 1

    def test_pre_trade_check_warnings(self):
        """Test pre-trade check warning generation."""
        manager = LimitManager()
        manager.add_limit(NotionalLimit("Swaps", hard_limit=1e9, soft_limit=900e6, warning_threshold=0.7))

        current_exposures = {"Swaps": 600e6}
        trade_impact = {"Swaps": 200e6}  # 80% utilization (above warning 70%, below soft 90%)

        check = pre_trade_control(manager, current_exposures, trade_impact)

        assert check.approved is True
        assert len(check.warnings) > 0
        assert "approaching limit" in check.warnings[0]

    def test_pre_trade_check_available_capacity(self):
        """Test available capacity in pre-trade check."""
        manager = LimitManager()
        manager.add_limit(NotionalLimit("Swaps", hard_limit=1e9))

        current_exposures = {"Swaps": 600e6}
        trade_impact = {"Swaps": 150e6}

        check = pre_trade_control(manager, current_exposures, trade_impact)

        assert "Swaps" in check.available_capacity
        assert check.available_capacity["Swaps"] == 250e6  # 1B - 750M

    def test_pre_trade_check_to_dict(self):
        """Test converting pre-trade check to dict."""
        manager = LimitManager()
        manager.add_limit(NotionalLimit("Swaps", hard_limit=1e9))

        current_exposures = {"Swaps": 900e6}
        trade_impact = {"Swaps": 200e6}

        check = pre_trade_control(manager, current_exposures, trade_impact)
        result = check.to_dict()

        assert "approved" in result
        assert "has_hard_breaches" in result
        assert "breaches" in result
        assert "warnings" in result
        assert "available_capacity" in result


# ==============================================================================
# What-If Analysis Tests
# ==============================================================================


class TestWhatIfAnalysis:
    """Test what-if scenario analysis."""

    def test_what_if_single_scenario(self):
        """Test what-if analysis with single scenario."""
        manager = LimitManager()
        manager.add_limit(NotionalLimit("Swaps", hard_limit=1e9))

        current_exposures = {"Swaps": 700e6}

        scenarios = [
            WhatIfScenario("Small Trade", {"Swaps": 100e6}),
        ]

        results = what_if_analysis(manager, current_exposures, scenarios)

        assert "Small Trade" in results
        assert results["Small Trade"].approved is True

    def test_what_if_multiple_scenarios(self):
        """Test what-if analysis with multiple scenarios."""
        manager = LimitManager()
        manager.add_limit(NotionalLimit("Swaps", hard_limit=1e9, soft_limit=800e6))

        current_exposures = {"Swaps": 600e6}

        scenarios = [
            WhatIfScenario("Small Trade", {"Swaps": 100e6}, "Add $100M position"),
            WhatIfScenario("Medium Trade", {"Swaps": 250e6}, "Add $250M position"),
            WhatIfScenario("Large Trade", {"Swaps": 500e6}, "Add $500M position"),
        ]

        results = what_if_analysis(manager, current_exposures, scenarios)

        assert len(results) == 3
        assert results["Small Trade"].approved is True
        assert results["Medium Trade"].approved is True  # Soft breach but allowed
        assert results["Large Trade"].approved is False  # Hard breach

    def test_what_if_breach_comparison(self):
        """Test comparing breaches across scenarios."""
        manager = LimitManager()
        manager.add_limit(NotionalLimit("Swaps", hard_limit=1e9, soft_limit=800e6))

        current_exposures = {"Swaps": 700e6}

        scenarios = [
            WhatIfScenario("Scenario A", {"Swaps": 50e6}),
            WhatIfScenario("Scenario B", {"Swaps": 150e6}),
            WhatIfScenario("Scenario C", {"Swaps": 400e6}),
        ]

        results = what_if_analysis(manager, current_exposures, scenarios)

        # Scenario A: no breach
        assert len(results["Scenario A"].breaches) == 0

        # Scenario B: soft breach
        assert len(results["Scenario B"].breaches) == 1
        assert results["Scenario B"].has_soft_breaches is True

        # Scenario C: hard breach
        assert len(results["Scenario C"].breaches) == 1
        assert results["Scenario C"].has_hard_breaches is True


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestIntegration:
    """Test integration of limits and controls."""

    def test_full_workflow(self):
        """Test complete limit management workflow."""
        # Setup limits
        manager = LimitManager()
        manager.add_limit(NotionalLimit("Rates", hard_limit=1e9, soft_limit=800e6))
        manager.add_limit(VaRLimit("VaR", hard_limit=10e6, soft_limit=8e6))
        manager.add_limit(ConcentrationLimit("SingleName", hard_limit=0.10))

        # Current exposures
        current_exposures = {
            "Rates": 600e6,
            "VaR": 7e6,
            "SingleName": 0.08,
        }

        # Get status summary
        summary = manager.get_limit_status_summary(current_exposures)
        assert summary["breached_limits"] == 0

        # Propose a trade
        trade_impact = {
            "Rates": 250e6,
            "VaR": 1.5e6,
            "SingleName": 0.01,
        }

        # Pre-trade check
        check = pre_trade_control(manager, current_exposures, trade_impact)

        # Should trigger soft breach on Rates
        assert check.approved is True  # Soft breaches allowed by default
        assert check.has_soft_breaches is True

        # What-if analysis for alternative sizes
        scenarios = [
            WhatIfScenario("Half Size", {"Rates": 125e6, "VaR": 0.75e6, "SingleName": 0.005}),
            WhatIfScenario("Full Size", trade_impact),
            WhatIfScenario("Double Size", {"Rates": 500e6, "VaR": 3e6, "SingleName": 0.02}),
        ]

        results = what_if_analysis(manager, current_exposures, scenarios)

        assert results["Half Size"].approved is True
        assert results["Full Size"].approved is True
        assert results["Double Size"].approved is False  # Would breach hard limit

    def test_multiple_limit_types(self):
        """Test managing multiple limit types simultaneously."""
        manager = LimitManager()

        # Add various limit types
        manager.add_limit(NotionalLimit("Swaps", hard_limit=1e9))
        manager.add_limit(VaRLimit("EquityVaR", hard_limit=5e6, confidence_level=0.99))
        manager.add_limit(ConcentrationLimit("Sector", hard_limit=0.15, concentration_type="sector"))
        manager.add_limit(IssuerLimit("BankXYZ", issuer_id="XYZ", hard_limit=500e6, credit_rating="AA"))

        exposures = {
            "Swaps": 800e6,
            "EquityVaR": 4e6,
            "Sector": 0.12,
            "BankXYZ": 400e6,
        }

        summary = manager.get_limit_status_summary(exposures)

        assert summary["total_limits"] == 4
        assert all(summary["limits"][name]["status"] == "ok" for name in exposures.keys())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
