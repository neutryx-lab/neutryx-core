"""Tests for confirmation matching and affirmation."""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal

from neutryx.integrations.clearing.base import Party, ProductType
from neutryx.integrations.clearing.confirmation import (
    Confirmation,
    ConfirmationDetails,
    ConfirmationStatus,
    ConfirmationMatcher,
    ToleranceConfig,
    BreakType,
    BreakSeverity,
    AffirmationMethod,
)


@pytest.fixture
def buyer():
    """Buyer party."""
    return Party(party_id="BUYER001", name="Buyer Bank", bic="BUYRBANKXXX")


@pytest.fixture
def seller():
    """Seller party."""
    return Party(party_id="SELLER001", name="Seller Bank", bic="SELLBANKXXX")


@pytest.fixture
def conf_details(buyer, seller):
    """Sample confirmation details."""
    return ConfirmationDetails(
        trade_id="TRD-001",
        trade_date=datetime.now(),
        settlement_date=datetime.now() + timedelta(days=2),
        buyer=buyer,
        seller=seller,
        product_type=ProductType.IRS,
        product_description="5Y USD IRS",
        quantity=Decimal("10000000"),
        price=Decimal("2.5"),
        notional=Decimal("10000000"),
        currency="USD"
    )


class TestConfirmationMatching:
    """Test confirmation matching."""

    def test_perfect_match(self, buyer, seller, conf_details):
        """Test perfect matching confirmations."""
        matcher = ConfirmationMatcher()

        # Buyer's confirmation
        conf_buy = Confirmation(
            originator=buyer,
            recipient=seller,
            direction="buy",
            details=conf_details
        )

        # Seller's confirmation (same details)
        conf_sell = Confirmation(
            originator=seller,
            recipient=buyer,
            direction="sell",
            details=conf_details
        )

        matcher.add_confirmation(conf_buy)
        matcher.add_confirmation(conf_sell)

        # Should auto-match
        assert conf_buy.status == ConfirmationStatus.MATCHED
        assert conf_sell.status == ConfirmationStatus.MATCHED
        assert conf_buy.match_score == 100.0
        assert len(conf_buy.breaks) == 0

    def test_price_mismatch(self, buyer, seller, conf_details):
        """Test price mismatch detection."""
        matcher = ConfirmationMatcher(
            tolerance=ToleranceConfig(price_tolerance=Decimal("0.01"), strict_matching=True)
        )

        # Buyer's confirmation
        conf_buy = Confirmation(
            originator=buyer,
            recipient=seller,
            direction="buy",
            details=conf_details
        )

        # Seller's confirmation with different price
        details_sell = conf_details.model_copy(deep=True)
        details_sell.price = Decimal("2.55")  # Different price

        conf_sell = Confirmation(
            originator=seller,
            recipient=buyer,
            direction="sell",
            details=details_sell
        )

        matcher.add_confirmation(conf_buy)
        matcher.add_confirmation(conf_sell)

        # Should detect mismatch
        assert conf_buy.status == ConfirmationStatus.MISMATCHED
        assert len(conf_buy.breaks) > 0

        # Check break type
        breaks = [matcher.breaks[b_id] for b_id in conf_buy.breaks]
        assert any(b.break_type == BreakType.PRICE_MISMATCH for b in breaks)

    def test_quantity_mismatch(self, buyer, seller, conf_details):
        """Test quantity mismatch detection."""
        matcher = ConfirmationMatcher()

        conf_buy = Confirmation(
            originator=buyer,
            recipient=seller,
            direction="buy",
            details=conf_details
        )

        details_sell = conf_details.model_copy(deep=True)
        details_sell.quantity = Decimal("11000000")  # Different quantity

        conf_sell = Confirmation(
            originator=seller,
            recipient=buyer,
            direction="sell",
            details=details_sell
        )

        matcher.add_confirmation(conf_buy)
        matcher.add_confirmation(conf_sell)

        assert len(conf_buy.breaks) > 0
        breaks = [matcher.breaks[b_id] for b_id in conf_buy.breaks]
        assert any(b.break_type == BreakType.QUANTITY_MISMATCH for b in breaks)

    def test_counterparty_mismatch(self, buyer, seller, conf_details):
        """Test counterparty mismatch detection."""
        matcher = ConfirmationMatcher()

        # Wrong buyer in seller's confirmation
        wrong_buyer = Party(party_id="WRONGBUYER", name="Wrong Buyer", bic="WRONGXXX")

        conf_buy = Confirmation(
            originator=buyer,
            recipient=seller,
            direction="buy",
            details=conf_details
        )

        details_sell = conf_details.model_copy(deep=True)
        details_sell.buyer = wrong_buyer

        conf_sell = Confirmation(
            originator=seller,
            recipient=buyer,
            direction="sell",
            details=details_sell
        )

        matcher.add_confirmation(conf_buy)
        matcher.add_confirmation(conf_sell)

        breaks = [matcher.breaks[b_id] for b_id in conf_buy.breaks]
        critical_breaks = [b for b in breaks if b.severity == BreakSeverity.CRITICAL]
        assert len(critical_breaks) > 0

    def test_affirmation(self, buyer, seller, conf_details):
        """Test confirmation affirmation."""
        matcher = ConfirmationMatcher()

        conf_buy = Confirmation(
            originator=buyer,
            recipient=seller,
            direction="buy",
            details=conf_details
        )

        conf_sell = Confirmation(
            originator=seller,
            recipient=buyer,
            direction="sell",
            details=conf_details
        )

        matcher.add_confirmation(conf_buy)
        matcher.add_confirmation(conf_sell)

        # Affirm
        affirmed = matcher.affirm_confirmation(
            conf_buy.confirmation_id,
            AffirmationMethod.ELECTRONIC,
            "CCP001"
        )

        assert affirmed.status == ConfirmationStatus.AFFIRMED
        assert affirmed.affirmation_method == AffirmationMethod.ELECTRONIC
        assert affirmed.affirmed_time is not None

    def test_break_resolution(self, buyer, seller, conf_details):
        """Test break resolution."""
        matcher = ConfirmationMatcher()

        conf_buy = Confirmation(
            originator=buyer,
            recipient=seller,
            direction="buy",
            details=conf_details
        )

        details_sell = conf_details.model_copy(deep=True)
        details_sell.price = Decimal("2.55")

        conf_sell = Confirmation(
            originator=seller,
            recipient=buyer,
            direction="sell",
            details=details_sell
        )

        matcher.add_confirmation(conf_buy)
        matcher.add_confirmation(conf_sell)

        # Resolve break
        break_id = conf_buy.breaks[0]
        resolved = matcher.resolve_break(
            break_id,
            "Agreed on price 2.5",
            "OPS001"
        )

        assert resolved.resolved
        assert resolved.resolution is not None


class TestStatistics:
    """Test matching statistics."""

    def test_statistics_tracking(self, buyer, seller, conf_details):
        """Test statistics are tracked correctly."""
        matcher = ConfirmationMatcher()

        conf_buy = Confirmation(
            originator=buyer,
            recipient=seller,
            direction="buy",
            details=conf_details
        )

        conf_sell = Confirmation(
            originator=seller,
            recipient=buyer,
            direction="sell",
            details=conf_details
        )

        matcher.add_confirmation(conf_buy)
        matcher.add_confirmation(conf_sell)

        stats = matcher.get_statistics()

        assert stats["total_confirmations"] == 2
        assert stats["matched_confirmations"] == 2
        assert stats["match_rate"] == 100.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
