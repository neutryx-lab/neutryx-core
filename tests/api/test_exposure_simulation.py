import datetime

import sys
import types

import jax
import jax.numpy as jnp
import pytest


_prometheus_stub = types.ModuleType("prometheus_client")


class _MetricStub:
    def labels(self, *args, **kwargs):  # pragma: no cover - simple stub
        return self

    def observe(self, *args, **kwargs) -> None:  # pragma: no cover - simple stub
        return None

    def inc(self, *args, **kwargs) -> None:  # pragma: no cover - simple stub
        return None


def _metric_factory(*args, **kwargs):  # pragma: no cover - simple stub
    return _MetricStub()


_prometheus_stub.CONTENT_TYPE_LATEST = "text/plain"
_prometheus_stub.Counter = _metric_factory
_prometheus_stub.Histogram = _metric_factory
_prometheus_stub.REGISTRY = None
_prometheus_stub.generate_latest = lambda *args, **kwargs: b""  # pragma: no cover

sys.modules.setdefault("prometheus_client", _prometheus_stub)

from neutryx.api.rest import (
    MarketDataView,
    _simulate_trade_exposure,
)
from neutryx.core.engine import MCConfig
from neutryx.portfolio.contracts.trade import ProductType, Trade, TradeStatus


@pytest.fixture
def valuation_date() -> datetime.date:
    return datetime.date(2024, 1, 1)


def _build_trade(**kwargs) -> Trade:
    defaults = {
        "id": "TRD-1",
        "counterparty_id": "CP-1",
        "product_type": ProductType.EQUITY_OPTION,
        "trade_date": datetime.date(2023, 1, 1),
        "status": TradeStatus.ACTIVE,
    }
    defaults.update(kwargs)
    return Trade(**defaults)


def _expected_profiles(exposures: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    epe = jnp.maximum(exposures, 0.0).mean(axis=0)
    ene = jnp.maximum(-exposures, 0.0).mean(axis=0)
    return epe, ene


def test_equity_option_exposure_profile(valuation_date: datetime.date) -> None:
    trade = _build_trade(
        id="EQ-1",
        product_type=ProductType.EQUITY_OPTION,
        maturity_date=valuation_date + datetime.timedelta(days=365),
        notional=5.0,
        currency="USD",
        product_details={
            "underlying": "AAPL",
            "strike": 95.0,
            "is_call": True,
            "is_long": True,
        },
    )
    times = jnp.linspace(0.0, 1.2, 7, dtype=jnp.float32)
    cfg = MCConfig(steps=6, paths=4096)
    market = MarketDataView.from_payload(
        {
            "equities": {"AAPL": {"spot": 100.0, "volatility": 0.2, "dividend": 0.01}},
            "rates": {"USD": {"rate": 0.03}},
        }
    )
    key = jax.random.PRNGKey(0)
    exposures = _simulate_trade_exposure(
        trade,
        key=key,
        cfg=cfg,
        times=times,
        valuation_date=valuation_date,
        market=market,
    )
    assert exposures.shape == (cfg.paths, times.shape[0])
    epe, ene = _expected_profiles(exposures)
    assert float(epe[0]) > 0.0
    assert pytest.approx(float(epe[-1]), abs=1e-5) == 0.0
    assert pytest.approx(float(ene[-1]), abs=1e-5) == 0.0


def test_fx_option_exposure_profile(valuation_date: datetime.date) -> None:
    trade = _build_trade(
        id="FX-1",
        product_type=ProductType.FX_OPTION,
        maturity_date=valuation_date + datetime.timedelta(days=365),
        notional=2.0,
        product_details={
            "currency_pair": "EURUSD",
            "strike": 1.05,
            "option_type": "call",
            "is_long": True,
        },
    )
    times = jnp.linspace(0.0, 1.5, 8, dtype=jnp.float32)
    cfg = MCConfig(steps=7, paths=4096)
    market = MarketDataView.from_payload(
        {
            "fx": {
                "EURUSD": {
                    "spot": 1.1,
                    "volatility": 0.15,
                    "domestic_currency": "USD",
                    "foreign_currency": "EUR",
                    "domestic_rate": 0.03,
                    "foreign_rate": 0.01,
                }
            }
        }
    )
    key = jax.random.PRNGKey(1)
    exposures = _simulate_trade_exposure(
        trade,
        key=key,
        cfg=cfg,
        times=times,
        valuation_date=valuation_date,
        market=market,
    )
    assert exposures.shape == (cfg.paths, times.shape[0])
    epe, ene = _expected_profiles(exposures)
    assert float(epe[0]) > 0.0
    assert pytest.approx(float(epe[-1]), abs=1e-5) == 0.0
    assert pytest.approx(float(ene[-1]), abs=1e-5) == 0.0


def test_interest_rate_swap_exposure_profile(valuation_date: datetime.date) -> None:
    trade = _build_trade(
        id="IRS-1",
        product_type=ProductType.INTEREST_RATE_SWAP,
        maturity_date=valuation_date + datetime.timedelta(days=5 * 365),
        notional=1_000_000.0,
        currency="USD",
        product_details={
            "fixed_rate": 0.03,
            "floating_rate": 0.025,
            "payment_frequency": 2,
            "pay_fixed": True,
        },
    )
    times = jnp.linspace(0.0, 5.5, 12, dtype=jnp.float32)
    cfg = MCConfig(steps=11, paths=4096)
    market = MarketDataView.from_payload(
        {"rates": {"USD": {"rate": 0.025, "volatility": 0.01}}}
    )
    key = jax.random.PRNGKey(2)
    exposures = _simulate_trade_exposure(
        trade,
        key=key,
        cfg=cfg,
        times=times,
        valuation_date=valuation_date,
        market=market,
    )
    assert exposures.shape == (cfg.paths, times.shape[0])
    epe, ene = _expected_profiles(exposures)
    assert float(jnp.mean(jnp.abs(exposures[:, 0]))) > 0.0
    assert float(ene[0]) > 0.0
    assert pytest.approx(float(epe[-1]), abs=1e-4) == 0.0
    assert pytest.approx(float(ene[-1]), abs=1e-4) == 0.0
