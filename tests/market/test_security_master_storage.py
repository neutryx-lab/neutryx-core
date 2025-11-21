from datetime import date

import pytest

from neutryx.market.storage.security_master import (
    SecurityMaster,
    SecurityIdentifier,
    SecurityIdentifierType,
)
from neutryx.market.adapters.corporate_actions import (
    BloombergCorporateActionParser,
    RefinitivCorporateActionParser,
)


@pytest.fixture
def security_master() -> SecurityMaster:
    return SecurityMaster()


def test_registration_update_and_corporate_actions(security_master: SecurityMaster) -> None:
    record = security_master.register_security(
        security_id="SEC-001",
        name="Neutryx Holdings",
        asset_class="equity",
        identifiers=[
            SecurityIdentifier(SecurityIdentifierType.TICKER, "NTX", primary=True),
            SecurityIdentifier(SecurityIdentifierType.ISIN, "US1234567890"),
        ],
        attributes={"ticker": "NTX", "currency": "USD"},
        effective_date=date(2023, 1, 2),
    )

    assert security_master.get_security("SEC-001") is record
    assert (
        security_master.get_security_by_identifier("ntx", SecurityIdentifierType.TICKER)
        is record
    )
    assert record.versions[-1].version == 1
    assert record.get_attributes()["ticker"] == "NTX"

    # Update record and ensure versioning retains previous attributes
    updated_version = security_master.update_security(
        "SEC-001", {"industry": "Technology"}, effective_date=date(2023, 6, 1)
    )
    assert updated_version.version == 2
    assert record.get_attributes()["industry"] == "Technology"
    assert record.get_attributes(date(2023, 3, 1))["ticker"] == "NTX"

    # Apply Bloomberg symbol change event
    bloomberg_parser = BloombergCorporateActionParser()
    bloomberg_event = bloomberg_parser.parse(
        {
            "event_type": "Name/Ticker Change",
            "effective_date": "2023-10-01",
            "old_ticker": "NTX",
            "new_ticker": "NYX",
            "description": "Ticker updated",
        }
    )
    security_master.apply_corporate_action("SEC-001", bloomberg_event)

    record_after_symbol_change = security_master.get_security("SEC-001")
    assert record_after_symbol_change.corporate_actions[-1] == bloomberg_event
    assert record_after_symbol_change.versions[-1].get("ticker") == "NYX"
    assert (
        security_master.get_security_by_identifier("NYX", SecurityIdentifierType.TICKER)
        is record
    )

    # Apply Refinitiv cash dividend event and ensure dividend history recorded
    refinitiv_parser = RefinitivCorporateActionParser()
    refinitiv_event = refinitiv_parser.parse(
        {
            "type": "DVD_CASH",
            "effectiveDate": "2023-12-15",
            "cashAmount": 0.25,
            "currency": "USD",
            "payDate": "2023-12-30",
            "text": "Quarterly dividend",
        }
    )
    security_master.apply_corporate_action("SEC-001", refinitiv_event)

    final_record = security_master.get_security("SEC-001")
    latest_attrs = final_record.get_attributes()
    assert pytest.approx(latest_attrs["dividends"][0]["amount"]) == 0.25
    assert latest_attrs["dividends"][0]["pay_date"] == "2023-12-30"
    assert final_record.corporate_actions[-1] == refinitiv_event
    assert len(final_record.versions) == 4
