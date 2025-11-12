from datetime import datetime

from neutryx.data.security_master import (
    InMemorySecurityMasterStorage,
    SecurityMaster,
    SecurityRecord,
)
from neutryx.data.validation import DataValidator
from neutryx.market.data_models import AssetClass, EquityQuote


def test_security_master_registration_and_lookup():
    storage = InMemorySecurityMasterStorage()
    master = SecurityMaster(storage)

    record = SecurityRecord(
        security_id="EQ-XYZ",
        asset_class=AssetClass.EQUITY,
        ticker="XYZ",
        isin="US1234567890",
        metadata={"currency": "USD"},
    )
    master.register(record)

    found = master.lookup("XYZ", identifier_type="ticker")
    assert found.security_id == "EQ-XYZ"

    found_by_id = master.get("EQ-XYZ")
    assert found_by_id == record


def test_data_validator_integrates_security_master():
    master = SecurityMaster(InMemorySecurityMasterStorage())
    master.register(
        SecurityRecord(
            security_id="EQ-XYZ",
            asset_class=AssetClass.EQUITY,
            ticker="XYZ",
            metadata={"currency": "USD"},
        )
    )

    validator = DataValidator(security_master=master)

    valid_quote = EquityQuote(
        timestamp=datetime.utcnow(),
        source="feed",
        ticker="XYZ",
        exchange="NYSE",
        price=101.0,
    )
    valid_result = validator.validate(valid_quote)

    assert valid_result.is_valid
    assert valid_result.metadata["security_master"]["security_id"] == "EQ-XYZ"

    invalid_quote = EquityQuote(
        timestamp=datetime.utcnow(),
        source="feed",
        ticker="MISSING",
        exchange="NYSE",
        price=10.0,
    )
    invalid_result = validator.validate(invalid_quote)

    assert not invalid_result.is_valid
    assert any(
        issue.rule == "security_master_reference" for issue in invalid_result.issues
    )
