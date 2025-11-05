"""Comprehensive tests for IFRS 9/13 accounting standards compliance."""

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from neutryx.accounting.ifrs9 import (
    BusinessModel,
    ECLModel,
    ECLResult,
    ECLStage,
    FinancialInstrumentCategory,
    HedgeEffectivenessTest,
    HedgeRelationship,
    HedgeType,
    IFRS9Classifier,
    sppi_test,
)
from neutryx.accounting.ifrs13 import (
    FairValueHierarchy,
    FairValueInput,
    FairValueMeasurement,
    IFRS13Disclosure,
    InputObservability,
    ValuationTechnique,
    classify_input_observability,
    determine_hierarchy_level,
)
from neutryx.accounting.xva import (
    CVACalculator,
    DVACalculator,
    FVACalculator,
    KVACalculator,
    MVACalculator,
    XVAEngine,
)


# IFRS 9 Tests
class TestIFRS9Classifier:
    def test_classify_amortized_cost(self):
        """Test classification as Amortized Cost."""
        category = IFRS9Classifier.classify(
            business_model=BusinessModel.HOLD_TO_COLLECT,
            cash_flows_solely_payments_principal_interest=True,
        )
        assert category == FinancialInstrumentCategory.AMORTIZED_COST

    def test_classify_fvoci(self):
        """Test classification as FVOCI."""
        category = IFRS9Classifier.classify(
            business_model=BusinessModel.HOLD_AND_SELL,
            cash_flows_solely_payments_principal_interest=True,
        )
        assert category == FinancialInstrumentCategory.FVOCI

    def test_classify_fvpl_failed_sppi(self):
        """Test classification as FVPL when SPPI fails."""
        category = IFRS9Classifier.classify(
            business_model=BusinessModel.HOLD_TO_COLLECT,
            cash_flows_solely_payments_principal_interest=False,  # SPPI fails
        )
        assert category == FinancialInstrumentCategory.FVPL

    def test_classify_equity_fvoci_election(self):
        """Test equity classification with FVOCI election."""
        category = IFRS9Classifier.classify(
            business_model=BusinessModel.OTHER,
            cash_flows_solely_payments_principal_interest=False,
            equity_instrument=True,
            elected_fvoci=True,
        )
        assert category == FinancialInstrumentCategory.FVOCI

    def test_classify_equity_default_fvpl(self):
        """Test equity default classification as FVPL."""
        category = IFRS9Classifier.classify(
            business_model=BusinessModel.OTHER,
            cash_flows_solely_payments_principal_interest=False,
            equity_instrument=True,
            elected_fvoci=False,
        )
        assert category == FinancialInstrumentCategory.FVPL


class TestECLModel:
    def test_calculate_12_month_ecl(self):
        """Test 12-month ECL calculation."""
        model = ECLModel(
            exposure_at_default=Decimal("1000000"),
            probability_of_default=0.02,  # 2%
            loss_given_default=0.45,
        )

        ecl = model.calculate_12_month_ecl()
        expected = Decimal("1000000") * Decimal("0.02") * Decimal("0.45")
        assert ecl == expected  # 9,000

    def test_calculate_lifetime_ecl(self):
        """Test lifetime ECL calculation."""
        model = ECLModel(
            exposure_at_default=Decimal("1000000"),
            probability_of_default=0.02,
            loss_given_default=0.45,
            discount_rate=0.05,
        )

        ecl = model.calculate_lifetime_ecl(maturity_years=3.0)
        assert ecl > Decimal("0")
        # Lifetime ECL should be higher than 12-month
        ecl_12m = model.calculate_12_month_ecl()
        assert ecl > ecl_12m


class TestECLResult:
    def test_determine_stage_1(self):
        """Test Stage 1 determination (performing)."""
        result = ECLResult(
            instrument_id="LOAN001",
            calculation_date=datetime.utcnow(),
            ecl_stage=ECLStage.STAGE_1,
            exposure_at_default=Decimal("1000000"),
            probability_of_default=0.01,
            loss_given_default=0.45,
            ecl_12_month=Decimal("4500"),
            ecl_lifetime=Decimal("15000"),
            days_past_due=0,
            credit_impaired=False,
            significant_increase_in_credit_risk=False,
        )

        stage = result.determine_stage()
        assert stage == ECLStage.STAGE_1
        assert result.ecl_provision == result.ecl_12_month

    def test_determine_stage_2(self):
        """Test Stage 2 determination (SICR)."""
        result = ECLResult(
            instrument_id="LOAN002",
            calculation_date=datetime.utcnow(),
            ecl_stage=ECLStage.STAGE_1,
            exposure_at_default=Decimal("1000000"),
            probability_of_default=0.05,
            loss_given_default=0.45,
            ecl_12_month=Decimal("22500"),
            ecl_lifetime=Decimal("75000"),
            days_past_due=45,  # > 30 days
            credit_impaired=False,
            significant_increase_in_credit_risk=True,
        )

        stage = result.determine_stage()
        assert stage == ECLStage.STAGE_2
        assert result.ecl_provision == result.ecl_lifetime

    def test_determine_stage_3(self):
        """Test Stage 3 determination (credit-impaired)."""
        result = ECLResult(
            instrument_id="LOAN003",
            calculation_date=datetime.utcnow(),
            ecl_stage=ECLStage.STAGE_1,
            exposure_at_default=Decimal("1000000"),
            probability_of_default=0.15,
            loss_given_default=0.45,
            ecl_12_month=Decimal("67500"),
            ecl_lifetime=Decimal("200000"),
            days_past_due=120,  # > 90 days
            credit_impaired=True,
        )

        stage = result.determine_stage()
        assert stage == ECLStage.STAGE_3
        assert result.ecl_provision == result.ecl_lifetime


class TestHedgeEffectivenessTest:
    def test_retrospective_effectiveness_pass(self):
        """Test passing retrospective effectiveness test."""
        hedge_rel = HedgeRelationship(
            hedge_id="HEDGE001",
            hedge_type=HedgeType.FAIR_VALUE,
            designation_date=datetime.utcnow(),
            hedged_item_id="BOND001",
            hedged_item_description="Fixed rate bond",
            hedged_risk="Interest Rate Risk",
            hedging_instrument_id="IRS001",
            hedging_instrument_type="Interest Rate Swap",
        )

        test = HedgeEffectivenessTest(
            hedge_relationship=hedge_rel,
            test_date=datetime.utcnow(),
            hedged_item_change=Decimal("100000"),
            hedging_instrument_change=Decimal("-95000"),  # Offsetting
        )

        is_effective = test.test_retrospective_effectiveness()
        assert is_effective
        # Ratio = 95k / 100k = 0.95, within 80%-125%
        assert test.calculate_effectiveness_ratio() == pytest.approx(Decimal("0.95"))

    def test_retrospective_effectiveness_fail(self):
        """Test failing retrospective effectiveness test."""
        hedge_rel = HedgeRelationship(
            hedge_id="HEDGE002",
            hedge_type=HedgeType.CASH_FLOW,
            designation_date=datetime.utcnow(),
            hedged_item_id="FORECAST001",
            hedged_item_description="Forecasted transaction",
            hedged_risk="FX Risk",
            hedging_instrument_id="FWD001",
            hedging_instrument_type="FX Forward",
        )

        test = HedgeEffectivenessTest(
            hedge_relationship=hedge_rel,
            test_date=datetime.utcnow(),
            hedged_item_change=Decimal("100000"),
            hedging_instrument_change=Decimal("-50000"),  # Poor offset
        )

        is_effective = test.test_retrospective_effectiveness()
        assert not is_effective  # 50% ratio outside 80%-125%


# IFRS 13 Tests
class TestFairValueHierarchy:
    def test_determine_level_1(self):
        """Test Level 1 determination (quoted prices)."""
        level = determine_hierarchy_level(
            has_active_market_quotes=True,
            has_observable_inputs=True,
        )
        assert level == FairValueHierarchy.LEVEL_1

    def test_determine_level_2(self):
        """Test Level 2 determination (observable inputs)."""
        level = determine_hierarchy_level(
            has_active_market_quotes=False,
            has_observable_inputs=True,
            observable_inputs_adjustments_significant=False,
        )
        assert level == FairValueHierarchy.LEVEL_2

    def test_determine_level_3(self):
        """Test Level 3 determination (unobservable inputs)."""
        level = determine_hierarchy_level(
            has_active_market_quotes=False,
            has_observable_inputs=False,
        )
        assert level == FairValueHierarchy.LEVEL_3


class TestFairValueMeasurement:
    def test_create_level_1_measurement(self):
        """Test Level 1 fair value measurement."""
        measurement = FairValueMeasurement(
            instrument_id="EQUITY001",
            measurement_date=datetime.utcnow(),
            fair_value=Decimal("150.50"),
            hierarchy_level=FairValueHierarchy.LEVEL_1,
            valuation_technique=ValuationTechnique.MARKET_APPROACH,
            transaction_price=Decimal("150.00"),
        )

        day1_gl = measurement.calculate_day1_gain_loss()
        assert day1_gl == Decimal("0.50")  # Immediate recognition for Level 1

    def test_total_adjustments(self):
        """Test total valuation adjustments calculation."""
        measurement = FairValueMeasurement(
            instrument_id="DERIV001",
            measurement_date=datetime.utcnow(),
            fair_value=Decimal("1000000"),
            hierarchy_level=FairValueHierarchy.LEVEL_2,
            valuation_technique=ValuationTechnique.INCOME_APPROACH,
            bid_ask_adjustment=Decimal("5000"),
            liquidity_adjustment=Decimal("10000"),
            credit_adjustment=Decimal("15000"),
        )

        total_adj = measurement.total_adjustments()
        assert total_adj == Decimal("30000")

    def test_level3_reconciliation(self):
        """Test Level 3 reconciliation."""
        measurement = FairValueMeasurement(
            instrument_id="ILLIQUID001",
            measurement_date=datetime.utcnow(),
            fair_value=Decimal("2000000"),
            hierarchy_level=FairValueHierarchy.LEVEL_3,
            valuation_technique=ValuationTechnique.INCOME_APPROACH,
            beginning_balance=Decimal("1800000"),
            purchases=Decimal("100000"),
            unrealized_gains_losses_pnl=Decimal("100000"),
        )

        ending = measurement.level3_reconciliation()
        assert ending == Decimal("2000000")


class TestIFRS13Disclosure:
    def test_fair_value_hierarchy_table(self):
        """Test fair value hierarchy disclosure table."""
        disclosure = IFRS13Disclosure(
            reporting_date=datetime.utcnow(),
            entity_name="Test Bank",
            level1_assets=Decimal("1000000"),
            level1_liabilities=Decimal("500000"),
            level2_assets=Decimal("2000000"),
            level2_liabilities=Decimal("1500000"),
            level3_assets=Decimal("500000"),
            level3_liabilities=Decimal("300000"),
        )

        table = disclosure.fair_value_hierarchy_table()
        assert table["fair_value_hierarchy"]["Level 1"]["assets"] == "1000000"
        assert table["total"]["assets"] == "3500000"  # 1M + 2M + 500k

    def test_total_assets_at_fair_value(self):
        """Test total assets calculation."""
        disclosure = IFRS13Disclosure(
            reporting_date=datetime.utcnow(),
            entity_name="Test Bank",
            level1_assets=Decimal("1000000"),
            level2_assets=Decimal("2000000"),
            level3_assets=Decimal("500000"),
        )

        total = disclosure.total_assets_at_fair_value()
        assert total == Decimal("3500000")


# XVA Tests
class TestCVACalculator:
    def test_calculate_cva(self):
        """Test CVA calculation."""
        calc = CVACalculator(
            counterparty_id="CP001",
            loss_given_default=0.40,
            expected_exposure=[Decimal("1000000"), Decimal("950000"), Decimal("900000")],
            time_points=[1.0, 2.0, 3.0],
            survival_probabilities=[1.0, 0.98, 0.96, 0.94],  # Need n+1
            discount_factors=[Decimal("0.95"), Decimal("0.90"), Decimal("0.86")],
        )

        cva = calc.calculate_cva()
        assert cva > Decimal("0")
        # CVA should reflect credit risk cost

    def test_calculate_cva_from_cds_spread(self):
        """Test CVA from CDS spread."""
        calc = CVACalculator(counterparty_id="CP002")

        cva = calc.calculate_cva_from_cds_spread(
            cds_spread_bps=100,  # 100 bps
            maturity_years=5.0,
            expected_positive_exposure=Decimal("1000000"),
        )

        assert cva > Decimal("0")


class TestDVACalculator:
    def test_calculate_dva(self):
        """Test DVA calculation."""
        calc = DVACalculator(
            own_entity_id="BANK001",
            loss_given_default=0.40,
            expected_negative_exposure=[Decimal("500000"), Decimal("450000")],
            time_points=[1.0, 2.0],
            own_survival_probabilities=[1.0, 0.99, 0.98],
            discount_factors=[Decimal("0.95"), Decimal("0.90")],
        )

        dva = calc.calculate_dva()
        assert dva > Decimal("0")
        # DVA is a benefit from own credit risk


class TestFVACalculator:
    def test_calculate_fva(self):
        """Test FVA calculation."""
        calc = FVACalculator(
            funding_spread=50.0,  # 50 bps over risk-free
            expected_positive_exposure=[Decimal("1000000"), Decimal("900000")],
            expected_negative_exposure=[Decimal("200000"), Decimal("100000")],
            time_points=[1.0, 2.0],
            discount_factors=[Decimal("0.95"), Decimal("0.90")],
        )

        fva = calc.calculate_fva()
        # FVA = FCA - FBA (cost - benefit)
        assert isinstance(fva, Decimal)


class TestXVAEngine:
    def test_calculate_all_xva(self):
        """Test comprehensive XVA calculation."""
        # Create component calculators
        cva_calc = CVACalculator(
            counterparty_id="CP001",
            expected_exposure=[Decimal("1000000")],
            time_points=[1.0],
            survival_probabilities=[1.0, 0.98],
            discount_factors=[Decimal("0.95")],
        )

        dva_calc = DVACalculator(
            own_entity_id="BANK001",
            expected_negative_exposure=[Decimal("200000")],
            time_points=[1.0],
            own_survival_probabilities=[1.0, 0.99],
            discount_factors=[Decimal("0.95")],
        )

        engine = XVAEngine(
            instrument_id="SWAP001",
            counterparty_id="CP001",
            cva_calculator=cva_calc,
            dva_calculator=dva_calc,
        )

        results = engine.calculate_all_xva()

        assert "cva" in results
        assert "dva" in results
        assert "total_xva" in results
        assert results["cva"] > Decimal("0")

    def test_ifrs13_credit_adjustment(self):
        """Test IFRS 13 credit adjustment extraction."""
        engine = XVAEngine(
            instrument_id="DERIV001",
            counterparty_id="CP001",
        )

        engine.cva = Decimal("50000")
        engine.dva = Decimal("10000")

        credit_adj = engine.get_ifrs13_credit_adjustment()
        assert credit_adj == Decimal("40000")  # CVA - DVA


def test_sppi_test_pass():
    """Test SPPI test passing."""
    result = sppi_test(
        contractual_cash_flows=[Decimal("100"), Decimal("100"), Decimal("1100")],
        principal_amount=Decimal("1000"),
        interest_rate=0.10,
        time_value_of_money=True,
        credit_risk=True,
        other_basic_lending_risks=True,
    )
    assert result is True


def test_sppi_test_fail():
    """Test SPPI test failing."""
    result = sppi_test(
        contractual_cash_flows=[],
        principal_amount=Decimal("1000"),
        interest_rate=0.10,
        time_value_of_money=False,  # No time value of money
        credit_risk=True,
        other_basic_lending_risks=True,
    )
    assert result is False


def test_classify_input_observability():
    """Test input observability classification."""
    # Observable
    obs = classify_input_observability(
        input_source="Bloomberg",
        input_type="Interest Rate",
    )
    assert obs == InputObservability.OBSERVABLE

    # Unobservable
    unobs = classify_input_observability(
        input_source="Internal Model",
        input_type="Correlation",
    )
    assert unobs == InputObservability.UNOBSERVABLE
