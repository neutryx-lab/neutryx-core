import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from neutryx.data.validation import DataValidator, RequiredFieldRule  # noqa: E402
from neutryx.market.adapters.simulated import SimulatedAdapter, SimulatedConfig  # noqa: E402
from neutryx.market.data_models import AssetClass  # noqa: E402
from neutryx.market.feeds import FeedState  # noqa: E402
from neutryx.market.feeds.vendors import (  # noqa: E402
    SimulatedMarketDataFeed,
    ValidationCallback,
    VendorSubscription,
)


@pytest.mark.asyncio
async def test_simulated_vendor_feed_lifecycle():
    adapter = SimulatedAdapter(SimulatedConfig(adapter_name="simulator", seed=123))
    validator = DataValidator([RequiredFieldRule(["price"])])
    callback = ValidationCallback()

    subscription = VendorSubscription(
        name="equity:aapl",
        instrument_id="AAPL US Equity",
        asset_class=AssetClass.EQUITY,
        adapter_method="get_equity_quote",
        polling_interval=0.01,
        params={"ticker": "AAPL US Equity"},
        callback=callback,
    )

    feed = SimulatedMarketDataFeed(
        adapter,
        validator=validator,
        loop=asyncio.get_running_loop(),
        default_subscriptions=[subscription],
    )

    feed.start()

    assert feed.state == FeedState.RUNNING
    assert adapter.is_connected()
    assert "equity:aapl" in feed.default_subscription_ids

    subscription_id = feed.default_subscription_ids["equity:aapl"]
    delivered = await feed.poll_once(subscription_id)

    assert delivered >= 1
    assert len(callback.results) == 1

    data_point, validation_result = callback.results[0]
    assert validation_result.is_valid
    assert getattr(data_point, "ticker", None) == "AAPL US Equity"

    await feed.stop()

    assert feed.state == FeedState.STOPPED
    assert not adapter.is_connected()
    assert not feed.default_subscription_ids
    assert not feed._subscriptions  # type: ignore[attr-defined]
    assert not feed._tasks  # type: ignore[attr-defined]
