from datetime import date, timedelta
from math import exp

import pytest

from neutryx.portfolio.contracts.trade import ProductType, Trade, TradeStatus
from neutryx.portfolio.pricing_bridge import MarketData, PricingBridge
from neutryx.products.credit_derivatives import CreditDefaultSwap
from neutryx.products.equity import equity_forward_value, variance_swap_value
from neutryx.products.swaptions import european_swaption_black


def _make_trade(**kwargs) -> Trade:
    defaults = dict(
        id="TRD",
        counterparty_id="CP",
        trade_date=date(2024, 1, 1),
        status=TradeStatus.ACTIVE,
    )
    defaults.update(kwargs)
    return Trade(**defaults)


@pytest.fixture
def bridge() -> PricingBridge:
    return PricingBridge()


@pytest.fixture
def market_data() -> MarketData:
    return MarketData(
        pricing_date=date(2024, 1, 1),
        spot_prices={"AAPL": 100.0, "SPX": 4000.0},
        volatilities={"AAPL": 0.20, "SPX": 0.18},
        interest_rates={"USD": 0.03},
        dividend_yields={"AAPL": 0.02},
    )


def test_price_forward(bridge: PricingBridge, market_data: MarketData) -> None:
    maturity = market_data.pricing_date + timedelta(days=365)
    trade = _make_trade(
        id="FWD1",
        product_type=ProductType.FORWARD,
        maturity_date=maturity,
        currency="USD",
        product_details={
            "underlying": "AAPL",
            "strike": 102.0,
            "position": "long",
        },
    )

    result = bridge.price_trade(trade, market_data)

    maturity = (trade.maturity_date - market_data.pricing_date).days / 365.0
    expected = equity_forward_value(
        spot=market_data.spot_prices["AAPL"],
        strike=102.0,
        maturity=maturity,
        risk_free_rate=market_data.interest_rates["USD"],
        dividend_yield=market_data.dividend_yields["AAPL"],
        position="long",
    )

    assert result.success
    assert pytest.approx(result.price, rel=1e-6) == expected


def test_price_swaption(bridge: PricingBridge, market_data: MarketData) -> None:
    maturity = market_data.pricing_date + timedelta(days=365)
    trade = _make_trade(
        id="SWP1",
        product_type=ProductType.SWAPTION,
        maturity_date=maturity,
        notional=1_000_000.0,
        currency="USD",
        product_details={
            "strike": 0.05,
            "volatility": 0.2,
            "swap_maturity": 5.0,
            "payment_frequency": 2,
            "is_payer": True,
        },
    )

    result = bridge.price_trade(trade, market_data)

    option_maturity = (trade.maturity_date - market_data.pricing_date).days / 365.0
    expected = european_swaption_black(
        strike=0.05,
        option_maturity=option_maturity,
        swap_maturity=5.0,
        volatility=0.2,
        discount_rate=market_data.interest_rates["USD"],
        notional=trade.notional,
        payment_frequency=2,
        is_payer=True,
    )

    assert result.success
    assert pytest.approx(result.price, rel=1e-6) == expected


def test_price_variance_swap(bridge: PricingBridge, market_data: MarketData) -> None:
    maturity = market_data.pricing_date + timedelta(days=365)
    trade = _make_trade(
        id="VAR1",
        product_type=ProductType.VARIANCE_SWAP,
        maturity_date=maturity,
        notional=10_000.0,
        currency="USD",
        product_details={
            "underlying": "SPX",
            "strike": 0.035,
            "realized_variance": 0.03,
            "expected_variance": 0.05,
        },
    )

    result = bridge.price_trade(trade, market_data)

    discount_factor = exp(-market_data.interest_rates["USD"] * 1.0)
    expected = variance_swap_value(
        notional=trade.notional,
        strike=0.035,
        realized_variance=0.03,
        expected_variance=0.05,
        discount_factor=discount_factor,
    )

    assert result.success
    assert pytest.approx(result.price, rel=1e-6) == expected


def test_price_credit_default_swap(bridge: PricingBridge, market_data: MarketData) -> None:
    maturity = market_data.pricing_date + timedelta(days=5 * 365)
    trade = _make_trade(
        id="CDS1",
        product_type=ProductType.CREDIT_DEFAULT_SWAP,
        maturity_date=maturity,
        notional=5_000_000.0,
        currency="USD",
        product_details={
            "spread": 100.0,
            "hazard_rate": 0.02,
            "recovery_rate": 0.4,
            "coupon_frequency": 4,
        },
    )

    result = bridge.price_trade(trade, market_data)

    option_maturity = (trade.maturity_date - market_data.pricing_date).days / 365.0
    cds = CreditDefaultSwap(
        T=option_maturity,
        notional=trade.notional,
        spread=100.0,
        recovery_rate=0.4,
        coupon_freq=4,
    )
    protection = cds.protection_leg_pv(0.02, market_data.interest_rates["USD"])
    premium = cds.premium_leg_pv(0.02, market_data.interest_rates["USD"])
    expected = float(protection - premium)

    assert result.success
    assert pytest.approx(result.price, rel=1e-6) == expected


def test_price_credit_default_swap_missing_hazard_rate(
    bridge: PricingBridge, market_data: MarketData
) -> None:
    maturity = market_data.pricing_date + timedelta(days=365)
    trade = _make_trade(
        id="CDS_ERR",
        product_type=ProductType.CREDIT_DEFAULT_SWAP,
        maturity_date=maturity,
        notional=1_000_000.0,
        currency="USD",
        product_details={"spread": 120.0},
    )

    result = bridge.price_trade(trade, market_data)

    assert not result.success
    assert "hazard_rate" in (result.error_message or "")
