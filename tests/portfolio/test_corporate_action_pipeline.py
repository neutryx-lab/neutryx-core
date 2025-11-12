from datetime import date
from decimal import Decimal

from neutryx.data.security_master import (
    InMemorySecurityMasterStorage,
    SecurityMaster,
    SecurityRecord,
)
from neutryx.integrations.clearing.corporate_actions import (
    CorporateActionEvent,
    CorporateActionScheduler,
    CorporateActionType,
    DividendTerms,
    ElectionType,
    MergerTerms,
    SplitTerms,
)
from neutryx.market.data_models import AssetClass
from neutryx.portfolio import Portfolio, PortfolioUpdatePipeline


def _event_base_kwargs(event_type: CorporateActionType) -> dict:
    return {
        "event_type": event_type,
        "security_id": "EQ-XYZ",
        "security_name": "XYZ Corp",
        "issuer": "XYZ Holdings",
        "announcement_date": date(2024, 1, 1),
        "ex_date": date(2024, 1, 15),
        "record_date": date(2024, 1, 15),
        "payment_date": date(2024, 1, 20),
        "election_type": ElectionType.MANDATORY,
        "terms": {},
        "description": f"{event_type.value} event",
    }


def test_portfolio_update_pipeline_handles_split_dividend_and_merger():
    security_master = SecurityMaster(InMemorySecurityMasterStorage())
    security_master.register(
        SecurityRecord(
            security_id="EQ-XYZ",
            asset_class=AssetClass.EQUITY,
            ticker="XYZ",
            metadata={"currency": "USD"},
        )
    )
    security_master.register(
        SecurityRecord(
            security_id="EQ-NEW",
            asset_class=AssetClass.EQUITY,
            ticker="XYZN",
            metadata={"currency": "USD"},
        )
    )

    portfolio = Portfolio(name="Test Portfolio")
    portfolio.upsert_position("EQ-XYZ", quantity=Decimal("100"))

    scheduler = CorporateActionScheduler()

    split_terms = SplitTerms(split_ratio="2:1", old_shares=1, new_shares=2)
    split_event = CorporateActionEvent(
        **{
            **_event_base_kwargs(CorporateActionType.STOCK_SPLIT),
            "payment_date": date(2024, 1, 20),
            "terms": split_terms.model_dump(),
        }
    )
    scheduler.schedule(split_event)

    dividend_terms = DividendTerms(dividend_rate=Decimal("1.50"), currency="USD")
    dividend_event = CorporateActionEvent(
        **{
            **_event_base_kwargs(CorporateActionType.CASH_DIVIDEND),
            "payment_date": date(2024, 2, 1),
            "terms": dividend_terms.model_dump(),
        }
    )
    scheduler.schedule(dividend_event)

    merger_terms = MergerTerms(
        acquirer_security_id="EQ-NEW",
        acquirer_name="New Corp",
        stock_consideration=Decimal("0.5"),
        cash_consideration=Decimal("5"),
    )
    merger_event = CorporateActionEvent(
        **{
            **_event_base_kwargs(CorporateActionType.MERGER),
            "payment_date": date(2024, 3, 1),
            "terms": merger_terms.model_dump(),
            "new_security_id": "EQ-NEW",
            "new_security_name": "New Corp",
        }
    )
    scheduler.schedule(merger_event)

    pipeline = PortfolioUpdatePipeline(
        portfolio,
        scheduler=scheduler,
        security_master=security_master,
    )

    split_results = pipeline.process_corporate_actions(date(2024, 1, 21))
    assert split_results and split_results[0].applied
    updated_position = portfolio.get_position("EQ-XYZ")
    assert updated_position is not None
    assert updated_position.quantity == Decimal("200")

    dividend_results = pipeline.process_corporate_actions(date(2024, 2, 2))
    assert dividend_results and dividend_results[0].applied
    assert portfolio.get_cash_balance("USD") == Decimal("300")

    merger_results = pipeline.process_corporate_actions(date(2024, 3, 2))
    assert merger_results and merger_results[0].applied
    assert portfolio.get_position("EQ-XYZ") is not None
    assert portfolio.get_position("EQ-XYZ").quantity == Decimal("0")
    new_position = portfolio.get_position("EQ-NEW")
    assert new_position is not None
    assert new_position.quantity == Decimal("100")  # 200 shares * 0.5 ratio
    # Cash balance includes dividend (150) + merger cash (200 * 5)
    assert portfolio.get_cash_balance("USD") == Decimal("1300")
