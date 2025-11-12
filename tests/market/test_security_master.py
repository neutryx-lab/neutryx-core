from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from neutryx.market.data_models import AssetClass
from neutryx.market.storage.security_master import (
    CorporateActionEvent,
    CorporateActionType,
    SecurityMaster,
)


def _dt(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day)


def test_register_and_lookup_security():
    master = SecurityMaster()
    inception = _dt(2024, 1, 1)

    record = master.register_security(
        security_id="SEC-1",
        asset_class=AssetClass.EQUITY,
        name="Acme Corp",
        identifiers={"isin": "US0000000001", "ticker": "ACME"},
        metadata={"country": "US"},
        effective_from=inception,
    )

    assert record.version == 1
    assert record.metadata["country"] == "US"

    lookup_by_id = master.get_security_by_id("SEC-1")
    assert lookup_by_id is record

    lookup_by_identifier = master.get_security_by_identifier("isin", "US0000000001")
    assert lookup_by_identifier is record


def test_update_creates_new_version():
    master = SecurityMaster()
    inception = _dt(2024, 1, 1)
    master.register_security(
        security_id="SEC-2",
        asset_class=AssetClass.EQUITY,
        name="Beta Corp",
        identifiers={"ticker": "BETA"},
        effective_from=inception,
    )

    effective_update = inception + timedelta(days=30)
    updated = master.update_security(
        "SEC-2",
        name="Beta Holdings",
        metadata={"sector": "Technology"},
        effective_from=effective_update,
    )

    assert updated.version == 2
    assert updated.name == "Beta Holdings"
    assert updated.metadata["sector"] == "Technology"

    prior = master.get_security_by_id("SEC-2", as_of=inception + timedelta(days=1))
    assert prior.version == 1
    assert prior.name == "Beta Corp"
    assert prior.effective_to == effective_update


def test_apply_corporate_actions():
    master = SecurityMaster()
    inception = _dt(2024, 1, 1)
    master.register_security(
        security_id="SEC-3",
        asset_class=AssetClass.EQUITY,
        name="Gamma Industries",
        identifiers={"ticker": "GAM"},
        effective_from=inception,
    )

    name_change_date = inception + timedelta(days=60)
    name_change = CorporateActionEvent(
        event_id="evt-1",
        security_id="SEC-3",
        action_type=CorporateActionType.NAME_CHANGE,
        effective_date=name_change_date,
        details={"new_name": "Gamma Global"},
        announcement_date=name_change_date - timedelta(days=5),
        source="simulated",
    )

    after_name_change = master.apply_corporate_action(name_change)
    assert after_name_change.version == 2
    assert after_name_change.name == "Gamma Global"
    assert after_name_change.events[-1] == name_change

    identifier_change_date = inception + timedelta(days=120)
    identifier_change = CorporateActionEvent(
        event_id="evt-2",
        security_id="SEC-3",
        action_type=CorporateActionType.IDENTIFIER_CHANGE,
        effective_date=identifier_change_date,
        details={"identifiers": {"ticker": "GGL"}, "removed_identifiers": ["ticker"]},
        source="simulated",
    )

    after_identifier_change = master.apply_corporate_action(identifier_change)
    assert after_identifier_change.version == 3
    assert after_identifier_change.identifiers["ticker"] == "GGL"
    assert master.get_security_by_identifier("ticker", "GGL").version == 3

    previous_version = master.get_security_by_id("SEC-3", as_of=name_change_date + timedelta(days=1))
    assert previous_version.version == 2

    all_versions = master.get_versions("SEC-3")
    assert len(all_versions) == 3
    assert all_versions[-1].events[-1] == identifier_change
