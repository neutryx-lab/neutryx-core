from datetime import date
import sys
import types

import jax.numpy as jnp
import pytest
from fastapi.testclient import TestClient

if "prometheus_client" not in sys.modules:
    class _DummyMetric:
        def __init__(self, *args, **kwargs):
            self._labels = {}

        def labels(self, **kwargs):
            self._labels.update(kwargs)
            return self

        def observe(self, *args, **kwargs):
            return None

        def inc(self, *args, **kwargs):
            return None

        def time(self):  # pragma: no cover - unused in tests
            return self

        def __call__(self, *args, **kwargs):
            return self

    prometheus_stub = types.SimpleNamespace(
        CONTENT_TYPE_LATEST="text/plain",
        Counter=_DummyMetric,
        Histogram=_DummyMetric,
        REGISTRY=types.SimpleNamespace(_names_to_collectors={}),
        generate_latest=lambda registry=None: b"",
    )
    sys.modules["prometheus_client"] = prometheus_stub

from neutryx.api.rest import create_app
from neutryx.portfolio.portfolio import Portfolio
from neutryx.portfolio.contracts.counterparty import Counterparty, CounterpartyCredit, EntityType
from neutryx.portfolio.contracts.trade import ProductType, Trade, TradeStatus


def _build_portfolio() -> Portfolio:
    trade = Trade(
        id="TRD-1",
        counterparty_id="CP1",
        product_type=ProductType.EQUITY_OPTION,
        trade_date=date(2023, 1, 1),
        maturity_date=date(2024, 1, 1),
        status=TradeStatus.ACTIVE,
        notional=1.0,
        currency="USD",
        product_details={"underlying": "AAPL", "strike": 100.0, "is_long": True},
    )
    counterparty = Counterparty(
        id="CP1",
        name="Test Counterparty",
        entity_type=EntityType.FINANCIAL,
        credit=CounterpartyCredit(lgd=0.4),
    )
    return Portfolio(name="TEST_PORT", counterparties={"CP1": counterparty}, trades={trade.id: trade})


def test_portfolio_xva_time_profiles():
    app = create_app()
    client = TestClient(app)
    portfolio = _build_portfolio()

    register = client.post("/portfolio/register", json=portfolio.model_dump(mode="json"))
    assert register.status_code == 200

    payload = {
        "portfolio_id": portfolio.name,
        "valuation_date": "2024-01-01",
        "compute_cva": True,
        "compute_dva": True,
        "compute_fva": True,
        "compute_mva": True,
        "lgd": 0.4,
        "funding_spread_bps": 20.0,
        "time_grid": [0.0, 0.5, 1.0],
        "monte_carlo": {"steps": 2, "paths": 2048, "seed": 123},
        "market_data": {
            "spots": {"AAPL": 100.0},
            "vols": {"AAPL": 0.2},
            "rates": {"USD": 0.02},
            "dividends": {"AAPL": 0.0},
            "discount_curve": {"values": [1.0, 0.99, 0.98]},
            "funding_curve": [0.002, 0.002, 0.002],
            "initial_margin": [0.4, 0.4, 0.4],
            "im_spread": [0.01, 0.01, 0.01],
        },
        "counterparty_pd": {"values": [0.0, 0.02, 0.05]},
        "own_pd": {"values": [0.0, 0.01, 0.03]},
        "lgd_curve": {"values": [0.4, 0.4, 0.4]},
        "own_lgd_curve": {"values": [0.6, 0.6, 0.6]},
        "funding_curve": {"values": [0.002, 0.002, 0.002]},
        "initial_margin": {"values": [0.4, 0.4, 0.4]},
    }

    response = client.post("/portfolio/xva", json=payload)
    assert response.status_code == 200
    data = response.json()

    times = jnp.asarray(data["times"], dtype=jnp.float32)
    assert times.shape[0] == len(payload["time_grid"])
    assert jnp.allclose(times, jnp.asarray(payload["time_grid"], dtype=jnp.float32))

    epe = jnp.asarray(data["epe_profile"], dtype=jnp.float32)
    ene = jnp.asarray(data["ene_profile"], dtype=jnp.float32)
    discount = jnp.asarray(payload["market_data"]["discount_curve"]["values"], dtype=jnp.float32)
    cp_pd = jnp.asarray(payload["counterparty_pd"]["values"], dtype=jnp.float32)
    own_pd = jnp.asarray(payload["own_pd"]["values"], dtype=jnp.float32)
    lgd_curve = jnp.asarray(payload["lgd_curve"]["values"], dtype=jnp.float32)
    own_lgd_curve = jnp.asarray(payload["own_lgd_curve"]["values"], dtype=jnp.float32)
    funding_curve = jnp.asarray(payload["funding_curve"]["values"], dtype=jnp.float32)
    initial_margin = jnp.asarray(payload["initial_margin"]["values"], dtype=jnp.float32)
    im_spread = jnp.asarray(payload["market_data"]["im_spread"], dtype=jnp.float32)

    d_pd = jnp.diff(jnp.concatenate([jnp.array([0.0], dtype=jnp.float32), cp_pd]))
    d_own_pd = jnp.diff(jnp.concatenate([jnp.array([0.0], dtype=jnp.float32), own_pd]))

    expected_cva = float((discount * epe * d_pd * lgd_curve).sum())
    expected_dva = float((discount * ene * d_own_pd * own_lgd_curve).sum())
    expected_fva = float((discount * epe * funding_curve).sum())
    expected_mva = float((discount * initial_margin * im_spread).sum())

    assert data["has_csa"] is False
    assert pytest.approx(data["cva"], rel=1e-3) == expected_cva
    assert pytest.approx(data["dva"], rel=1e-3) == expected_dva
    assert pytest.approx(data["fva"], rel=1e-3) == expected_fva
    assert pytest.approx(data["mva"], rel=1e-3) == expected_mva
