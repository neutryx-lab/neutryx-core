from __future__ import annotations

import json
from datetime import date, datetime
from decimal import Decimal
from typing import Dict, Tuple

import httpx
import pytest

from neutryx.integrations.clearing.base import (
    CCPConfig,
    CCPTimeoutError,
    CCPTradeRejectionError,
    Party,
    ProductType,
    Trade,
    TradeEconomics,
    TradeStatus,
)
from neutryx.integrations.clearing.euroclear import EuroclearConnector
from neutryx.integrations.clearing.euroclear.messages import (
    EuroclearSettlementInstruction,
    SettlementStatus,
    SettlementType,
)


class MockEuroclearAPI:
    """Stateful Euroclear API mock."""

    def __init__(self) -> None:
        self.instructions: Dict[str, Dict[str, str]] = {}
        self.trades: Dict[str, Dict[str, str]] = {}
        self.submit_attempts = 0
        self.fail_first_instruction = False
        self.timeout_next_instruction = False
        self.reject_trades = False

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path

        if path == "/auth/session" and request.method == "POST":
            return httpx.Response(
                200,
                json={"session_id": "EUROCLEAR-SESSION", "session_token": "TOKEN"},
            )

        if path == "/trades" and request.method == "POST":
            payload = json.loads(request.content.decode())
            trade_id = payload["trade_id"]
            if self.reject_trades:
                return httpx.Response(422, json={"message": "trade rejected", "code": "EC-REJ"})
            submission = {
                "submission_id": f"EC-{trade_id}",
                "trade_id": trade_id,
                "status": "accepted",
                "timestamp": datetime.utcnow().isoformat(),
                "ccp_trade_id": f"EUROCLEAR-{trade_id}",
            }
            self.trades[trade_id] = submission
            return httpx.Response(202, json=submission)

        if path.startswith("/trades/") and path.endswith("/status"):
            trade_id = path.split("/")[2]
            status = self.trades.get(trade_id, {"status": "pending"})
            return httpx.Response(200, json={"trade_id": trade_id, "status": status["status"]})

        if path.startswith("/trades/") and path.endswith("/cancel"):
            trade_id = path.split("/")[2]
            self.trades.setdefault(trade_id, {})["status"] = "cancelled"
            return httpx.Response(200, json={"trade_id": trade_id, "status": "cancelled"})

        if path == "/settlements/securities" and request.method == "POST":
            if self.timeout_next_instruction:
                self.timeout_next_instruction = False
                raise httpx.TimeoutException("Euroclear timeout", request=request)

            self.submit_attempts += 1
            if self.fail_first_instruction and self.submit_attempts == 1:
                return httpx.Response(500, json={"message": "temporary failure"})

            payload = json.loads(request.content.decode())
            instruction_id = payload["instruction_id"]
            self.instructions[instruction_id] = {
                "sender_reference": payload["sender_reference"],
                "status": "matched",
                "settlement_date": payload["settlement_date"],
                "quantity": payload["quantity"],
                "settlement_amount": payload.get("settlement_amount"),
                "settlement_currency": payload.get("settlement_currency"),
            }
            confirmation = {
                "confirmation_id": f"CONF-{instruction_id}",
                "instruction_id": instruction_id,
                "sender_reference": payload["sender_reference"],
                "settlement_date": payload["settlement_date"],
                "actual_settlement_time": datetime.utcnow().isoformat(),
                "isin": payload["isin"],
                "quantity_settled": "0",
                "quantity_instructed": payload["quantity"],
                "settlement_amount": payload.get("settlement_amount"),
                "settlement_currency": payload.get("settlement_currency"),
                "status": "matched",
                "euroclear_reference": f"EUR-{instruction_id}",
            }
            return httpx.Response(202, json=confirmation)

        if path.startswith("/settlements/securities/") and path.endswith("/cancel"):
            instruction_id = path.split("/")[3]
            if instruction_id in self.instructions:
                self.instructions[instruction_id]["status"] = "cancelled"
            return httpx.Response(200, json={"instruction_id": instruction_id, "status": "cancelled"})

        if path.startswith("/settlements/securities/") and request.method == "PATCH":
            instruction_id = path.split("/")[3]
            updates = json.loads(request.content.decode())
            inst = self.instructions[instruction_id]
            inst.update(updates)
            confirmation = {
                "confirmation_id": f"CONF-{instruction_id}",
                "instruction_id": instruction_id,
                "sender_reference": inst["sender_reference"],
                "settlement_date": inst["settlement_date"],
                "actual_settlement_time": datetime.utcnow().isoformat(),
                "isin": "US0000000000",
                "quantity_settled": "0",
                "quantity_instructed": inst["quantity"],
                "settlement_amount": inst.get("settlement_amount"),
                "settlement_currency": inst.get("settlement_currency"),
                "status": inst.get("status", "matched"),
                "euroclear_reference": f"EUR-{instruction_id}",
            }
            return httpx.Response(200, json=confirmation)

        if path.startswith("/settlements/securities/") and request.method == "GET":
            instruction_id = path.split("/")[3]
            inst = self.instructions.get(instruction_id)
            if not inst:
                return httpx.Response(404, json={"message": "not found"})
            status = {
                "instruction_id": instruction_id,
                "sender_reference": inst["sender_reference"],
                "status": inst["status"],
                "last_updated": datetime.utcnow().isoformat(),
                "matched": inst["status"] in ("matched", "affirmed", "settled"),
                "affirmed": inst["status"] in ("affirmed", "settled"),
                "securities_delivered": inst["status"] == "settled",
                "payment_received": inst["status"] == "settled",
                "status_message": "OK",
                "expected_settlement_date": inst["settlement_date"],
            }
            return httpx.Response(200, json=status)

        if path.startswith("/accounts/") and path.endswith("/holdings"):
            account_id = path.split("/")[2]
            return httpx.Response(
                200,
                json={
                    "account_id": account_id,
                    "holdings": [
                        {"isin": "US0378331005", "quantity": "1000", "currency": "USD"}
                    ],
                },
            )

        if path == "/margin":
            return httpx.Response(200, json={"member_id": "EUROCLEAR001", "initial_margin": 0})

        if path == "/reports/positions":
            return httpx.Response(200, json={"report_id": "EC-REPORT", "positions": []})

        if path == "/health":
            return httpx.Response(200, json={"status": "ok"})

        return httpx.Response(404, json={"message": "unknown"})


@pytest.fixture
def euroclear_config() -> CCPConfig:
    return CCPConfig(
        ccp_name="Euroclear",
        member_id="EUROCLEAR001",
        api_endpoint="https://mock.euroclear",
        api_key="test",
        environment="test",
        max_retries=2,
        retry_delay=0,
    )


@pytest.fixture
def mock_euroclear_api() -> MockEuroclearAPI:
    return MockEuroclearAPI()


@pytest.fixture
async def euroclear_connector(
    euroclear_config: CCPConfig, mock_euroclear_api: MockEuroclearAPI
) -> Tuple[EuroclearConnector, MockEuroclearAPI]:
    transport = httpx.MockTransport(mock_euroclear_api)
    connector = EuroclearConnector(euroclear_config, transport=transport)
    await connector.connect()
    try:
        yield connector, mock_euroclear_api
    finally:
        await connector.disconnect()


def _make_trade(trade_id: str) -> Trade:
    return Trade(
        trade_id=trade_id,
        product_type=ProductType.EQUITY_OPTION,
        trade_date=datetime.utcnow(),
        effective_date=datetime.utcnow(),
        maturity_date=datetime.utcnow(),
        buyer=Party(party_id="BUYER", name="Buyer"),
        seller=Party(party_id="SELLER", name="Seller"),
        economics=TradeEconomics(notional=Decimal("500000"), currency="USD"),
    )


def _make_instruction(instruction_id: str) -> EuroclearSettlementInstruction:
    return EuroclearSettlementInstruction(
        instruction_id=instruction_id,
        sender_reference=f"SR-{instruction_id}",
        settlement_type=SettlementType.DVP,
        settlement_date=date(2025, 1, 17),
        trade_date=date(2025, 1, 15),
        isin="US0378331005",
        quantity=Decimal("1000"),
        delivering_party="DELIVER",
        receiving_party="RECEIVE",
        participant_bic="BANKGB2LXXX",
        settlement_amount=Decimal("150000"),
        settlement_currency="USD",
    )


@pytest.mark.anyio
async def test_submit_trade_success(
    euroclear_connector: Tuple[EuroclearConnector, MockEuroclearAPI]
) -> None:
    connector, _ = euroclear_connector
    trade = _make_trade("EC-TRADE-1")

    response = await connector.submit_trade(trade)

    assert response.status == TradeStatus.ACCEPTED
    assert connector.metrics.successful_submissions == 1


@pytest.mark.anyio
async def test_submit_trade_rejection(
    euroclear_connector: Tuple[EuroclearConnector, MockEuroclearAPI]
) -> None:
    connector, api = euroclear_connector
    api.reject_trades = True

    with pytest.raises(CCPTradeRejectionError):
        await connector.submit_trade(_make_trade("EC-TRADE-2"))

    assert connector.metrics.rejected_submissions == 1


@pytest.mark.anyio
async def test_submit_settlement_instruction_with_retry(
    euroclear_connector: Tuple[EuroclearConnector, MockEuroclearAPI]
) -> None:
    connector, api = euroclear_connector
    api.fail_first_instruction = True

    confirmation = await connector.submit_settlement_instruction(_make_instruction("ECI-1"))

    assert confirmation.status == SettlementStatus.MATCHED
    assert api.submit_attempts == 2


@pytest.mark.anyio
async def test_submit_settlement_instruction_timeout(
    euroclear_connector: Tuple[EuroclearConnector, MockEuroclearAPI]
) -> None:
    connector, api = euroclear_connector
    api.timeout_next_instruction = True
    connector.config.max_retries = 1

    with pytest.raises(CCPTimeoutError):
        await connector.submit_settlement_instruction(_make_instruction("ECI-2"))


@pytest.mark.anyio
async def test_amend_and_cancel_instruction(
    euroclear_connector: Tuple[EuroclearConnector, MockEuroclearAPI]
) -> None:
    connector, _ = euroclear_connector
    instruction = _make_instruction("ECI-3")
    await connector.submit_settlement_instruction(instruction)

    amended = await connector.amend_settlement_instruction(
        instruction.instruction_id,
        new_settlement_date=date(2025, 1, 20),
        new_quantity=Decimal("1200"),
    )
    assert amended.instruction_id == instruction.instruction_id

    assert await connector.cancel_settlement_instruction(instruction.instruction_id) is True
    status = await connector.get_settlement_status(instruction.instruction_id)
    assert status.status == SettlementStatus.CANCELLED


@pytest.mark.anyio
async def test_get_holdings(euroclear_connector: Tuple[EuroclearConnector, MockEuroclearAPI]) -> None:
    connector, _ = euroclear_connector
    holdings = await connector.get_holdings("ACCOUNT-1")
    assert holdings["account_id"] == "ACCOUNT-1"
    assert len(holdings["holdings"]) == 1


@pytest.mark.anyio
async def test_healthcheck(euroclear_connector: Tuple[EuroclearConnector, MockEuroclearAPI]) -> None:
    connector, _ = euroclear_connector
    assert await connector.healthcheck() is True
