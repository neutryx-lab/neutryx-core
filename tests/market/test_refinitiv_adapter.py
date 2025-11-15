from datetime import datetime

import pytest

from neutryx.market.adapters.base import ConnectionState
from neutryx.market.adapters.refinitiv import RefinitivAdapter, RefinitivConfig


class FakeDataFrame:
    def __init__(self, records):
        self._records = records

    def to_dict(self, orient="records"):
        if orient != "records":
            raise TypeError("Unsupported orientation")
        return self._records

    @property
    def empty(self):
        return not self._records


class FakePricingAPI:
    def __init__(self, responses):
        self._responses = responses
        self.calls = 0

    def get_snapshots(self, universe, fields):
        self.calls += 1
        ric = universe[0]
        response = self._responses[self.calls - 1]
        if callable(response):
            response = response(ric, fields)
        return response


class FakeResponse:
    def __init__(self, raw=None, df_records=None):
        class _Data:
            def __init__(self, raw_payload, records):
                self.raw = raw_payload
                self.df = FakeDataFrame(records) if records is not None else None

        self.data = _Data(raw, df_records)


class FakeRDP:
    def __init__(self, pricing_api):
        class _Content:
            def __init__(self, pricing):
                self.pricing = pricing

        self.content = _Content(pricing_api)


@pytest.fixture
def connected_adapter():
    config = RefinitivConfig(
        adapter_name="refinitiv",
        use_desktop=False,
        cache_ttl_seconds=60,
    )
    adapter = RefinitivAdapter(config)
    adapter.connection_state = ConnectionState.CONNECTED
    adapter._session = object()
    return adapter


def test_rdp_snapshot_parsing(connected_adapter):
    responses = [
        FakeResponse(
            raw={
                "snapshots": [
                    {
                        "ric": "EUR=",
                        "fields": {
                            "CF_LAST": 1.05,
                            "CF_BID": 1.04,
                        },
                    }
                ]
            }
        )
    ]
    pricing_api = FakePricingAPI(responses)
    connected_adapter._rdp = FakeRDP(pricing_api)

    data = connected_adapter._get_data("EUR=", ["CF_LAST", "CF_BID"])

    assert data == {"CF_LAST": 1.05, "CF_BID": 1.04}
    assert pricing_api.calls == 1


def test_equity_quote_uses_cache(connected_adapter):
    quote_payload = {
        "CF_LAST": 180.0,
        "CF_BID": 179.5,
        "CF_ASK": 180.5,
        "CF_VOLUME": 100,
        "OPEN_PRC": 175.0,
        "HIGH_1": 181.0,
        "LOW_1": 174.0,
        "CF_CLOSE": 178.0,
        "CURRENCY": "USD",
    }

    responses = [
        FakeResponse(
            raw={
                "snapshots": [
                    {
                        "RIC": "AAPL.O",
                        "fields": quote_payload,
                    }
                ]
            }
        )
    ]
    pricing_api = FakePricingAPI(responses)
    connected_adapter._rdp = FakeRDP(pricing_api)

    first_quote = connected_adapter.get_equity_quote("AAPL.O")
    second_quote = connected_adapter.get_equity_quote("AAPL.O")

    assert pricing_api.calls == 1
    assert first_quote is second_quote
    assert first_quote.price == pytest.approx(180.0)
    assert first_quote.timestamp <= datetime.utcnow()
