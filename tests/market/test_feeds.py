import asyncio
from datetime import timedelta

from neutryx.data.validation import DataValidator, RangeRule, RequiredFieldRule, StalenessRule
from neutryx.integrations.databases.memory import InMemoryConnector
from neutryx.market.adapters.simulated import SimulatedAdapter, SimulatedConfig
from neutryx.market.data_models import AssetClass, DataQuality, FXQuote
from neutryx.market.feeds import PollingMarketDataFeed
from neutryx.market.market_data import create_market_data_feed


def test_polling_feed_persists_fx_quotes():
    asyncio.run(_run_polling_feed_test())


async def _run_polling_feed_test():
    config = SimulatedConfig(adapter_name="simulated", seed=123)
    adapter = SimulatedAdapter(config)
    storage = InMemoryConnector()
    await storage.connect()

    validator = DataValidator(
        [
            RequiredFieldRule(["timestamp"]),
            RangeRule("spot", minimum=0.0),
            StalenessRule(timedelta(seconds=2)),
        ]
    )

    feed = PollingMarketDataFeed(adapter, validator=validator, storage=storage)

    subscription_id = feed.subscribe_instrument(
        "EURUSD",
        asset_class=AssetClass.FX,
        adapter_method="get_fx_quote",
        polling_interval=0.05,
        params={"base_currency": "EUR", "quote_currency": "USD"},
    )

    await feed.poll_once(subscription_id)
    await feed.poll_once(subscription_id)
    await storage.disconnect()

    records = storage.get_records()
    assert records, "Expected at least one persisted record"
    assert records[0]["instrument_id"] == "EURUSD"
    assert "payload" in records[0]


def test_simulated_adapter_generates_quotes():
    config = SimulatedConfig(adapter_name="simulated", seed=1)
    adapter = SimulatedAdapter(config)
    adapter.connect()

    quote = adapter.get_fx_quote("EUR", "USD")

    assert isinstance(quote, FXQuote)
    assert quote.quality == DataQuality.REALTIME
    assert quote.spot > 0


def test_create_market_data_feed_returns_feed():
    feed = create_market_data_feed(
        "simulated",
        adapter_kwargs={"seed": 1},
    )

    assert isinstance(feed, PollingMarketDataFeed)
