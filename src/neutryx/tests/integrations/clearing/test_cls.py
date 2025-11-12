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
from neutryx.integrations.clearing.cls import CLSConnector
from neutryx.integrations.clearing.cls.messages import (
    CLSCurrency,
    CLSSettlementInstruction,
)


class MockCLSAPI:
    """Stateful mock of the CLS HTTP API."""

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
                json={
                    "session_id": "CLS-SESSION",
                    "session_token": "CLS-TOKEN",
                },
            )

        if path == "/trades" and request.method == "POST":
            body = json.loads(request.content.decode())
            trade_id = body["trade_id"]

            if self.reject_trades:
                return httpx.Response(422, json={"message": "Trade rejected", "code": "CLS-REJ"})

            submission = {
                "submission_id": f"SUB-{trade_id}",
                "trade_id": trade_id,
                "status": "accepted",
                "timestamp": datetime.utcnow().isoformat(),
                "ccp_trade_id": f"CLS-{trade_id}",
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

        if path == "/settlements/instructions" and request.method == "POST":
            if self.timeout_next_instruction:
                self.timeout_next_instruction = False
                raise httpx.TimeoutException("CLS instruction timeout", request=request)

            self.submit_attempts += 1
            if self.fail_first_instruction and self.submit_attempts == 1:
                return httpx.Response(500, json={"message": "temporary failure"})

            body = json.loads(request.content.decode())
            instruction_id = body["instruction_id"]
            trade_id = body["trade_id"]
            self.instructions[instruction_id] = {
                "trade_id": trade_id,
                "status": "matched",
                "settlement_date": body["settlement_date"],
                "buy_currency": body["buy_currency"],
                "sell_currency": body["sell_currency"],
                "buy_amount": body["buy_amount"],
                "sell_amount": body["sell_amount"],
            }
            confirmation = {
                "confirmation_id": f"CONF-{instruction_id}",
                "instruction_id": instruction_id,
                "trade_id": trade_id,
                "settlement_date": body["settlement_date"],
                "settlement_time": datetime.utcnow().isoformat(),
                "buy_currency": body["buy_currency"],
                "buy_amount": body["buy_amount"],
                "sell_currency": body["sell_currency"],
                "sell_amount": body["sell_amount"],
                "status": "matched",
            }
            return httpx.Response(202, json=confirmation)

        if path.startswith("/settlements/instructions/") and path.endswith("/cancel"):
            instruction_id = path.split("/")[3]
            if instruction_id in self.instructions:
                self.instructions[instruction_id]["status"] = "cancelled"
            return httpx.Response(200, json={"instruction_id": instruction_id, "status": "cancelled"})

        if path.startswith("/settlements/instructions/") and request.method == "GET":
            instruction_id = path.split("/")[3]
            instruction = self.instructions.get(instruction_id)
            if not instruction:
                return httpx.Response(404, json={"message": "instruction not found"})
            status = {
                "instruction_id": instruction_id,
                "trade_id": instruction["trade_id"],
                "status": instruction["status"],
                "last_updated": datetime.utcnow().isoformat(),
                "pay_in_complete": instruction["status"] in ("matched", "settled"),
                "pay_out_complete": instruction["status"] == "settled",
                "status_message": "OK",
            }
            return httpx.Response(200, json=status)

        if path.startswith("/settlements/sessions/") and path.endswith("/liquidity"):
            session_id = path.split("/")[3]
            return httpx.Response(200, json={"session_id": session_id, "currencies": {}})

        if path == "/margin":
            return httpx.Response(200, json={"member_id": "CLSMEMBER001", "initial_margin": 0})

        if path == "/reports/positions":
            return httpx.Response(200, json={"report_id": "CLS-REPORT", "positions": []})

        if path == "/health":
            return httpx.Response(200, json={"status": "ok"})

        return httpx.Response(404, json={"message": "unknown endpoint"})


@pytest.fixture
def cls_config() -> CCPConfig:
    return CCPConfig(
        ccp_name="CLS",
        member_id="CLSMEMBER001",
        api_endpoint="https://mock.cls",
        api_key="test",
        environment="test",
        max_retries=2,
        retry_delay=0,
    )


@pytest.fixture
def mock_cls_api() -> MockCLSAPI:
    return MockCLSAPI()


@pytest.fixture
async def cls_connector(
    cls_config: CCPConfig, mock_cls_api: MockCLSAPI
) -> Tuple[CLSConnector, MockCLSAPI]:
    transport = httpx.MockTransport(mock_cls_api)
    connector = CLSConnector(cls_config, transport=transport)
    await connector.connect()
    try:
        yield connector, mock_cls_api
    finally:
        await connector.disconnect()


def _make_trade(trade_id: str) -> Trade:
    return Trade(
        trade_id=trade_id,
        product_type=ProductType.FX_FORWARD,
        trade_date=datetime.utcnow(),
        effective_date=datetime.utcnow(),
        maturity_date=datetime.utcnow(),
        buyer=Party(party_id="BUYER", name="Buyer Corp"),
        seller=Party(party_id="SELLER", name="Seller Corp"),
        economics=TradeEconomics(notional=Decimal("1000000"), currency="USD"),
    )


def _make_instruction(instruction_id: str) -> CLSSettlementInstruction:
    return CLSSettlementInstruction(
        instruction_id=instruction_id,
        trade_id=f"TRADE-{instruction_id[-3:]}",
        settlement_session_id="CLS_SESSION",
        buy_currency=CLSCurrency.USD,
        buy_amount=Decimal("1000000"),
        sell_currency=CLSCurrency.EUR,
        sell_amount=Decimal("900000"),
        fx_rate=Decimal("1.1111"),
        value_date=date(2025, 1, 17),
        settlement_date=date(2025, 1, 17),
        submitter_bic="BANKGB2LXXX",
        counterparty_bic="BANKUS33XXX",
        settlement_member="MEMBER001",
    )


@pytest.mark.anyio
async def test_submit_trade_success(cls_connector: Tuple[CLSConnector, MockCLSAPI]) -> None:
    connector, _ = cls_connector
    trade = _make_trade("TRADE-001")

    response = await connector.submit_trade(trade)

    assert response.status == TradeStatus.ACCEPTED
    assert connector.metrics.successful_submissions == 1


@pytest.mark.anyio
async def test_submit_trade_rejection(cls_connector: Tuple[CLSConnector, MockCLSAPI]) -> None:
    connector, api = cls_connector
    trade = _make_trade("TRADE-002")
    api.reject_trades = True

    with pytest.raises(CCPTradeRejectionError):
        await connector.submit_trade(trade)

    assert connector.metrics.rejected_submissions == 1


@pytest.mark.anyio
async def test_submit_settlement_instruction_with_retry(
    cls_connector: Tuple[CLSConnector, MockCLSAPI]
) -> None:
    connector, api = cls_connector
    api.fail_first_instruction = True

    confirmation = await connector.submit_settlement_instruction(_make_instruction("CLSI-001"))

    assert confirmation.status.value == "matched"
    assert api.submit_attempts == 2


@pytest.mark.anyio
async def test_submit_settlement_instruction_timeout(
    cls_connector: Tuple[CLSConnector, MockCLSAPI]
) -> None:
    connector, api = cls_connector
    api.timeout_next_instruction = True
    connector.config.max_retries = 1

    with pytest.raises(CCPTimeoutError):
        await connector.submit_settlement_instruction(_make_instruction("CLSI-002"))


@pytest.mark.anyio
async def test_get_and_cancel_settlement_status(
    cls_connector: Tuple[CLSConnector, MockCLSAPI]
) -> None:
    connector, _ = cls_connector
    instruction = _make_instruction("CLSI-003")
    await connector.submit_settlement_instruction(instruction)

    status = await connector.get_settlement_status(instruction.instruction_id)
    assert status.status.value == "matched"

    assert await connector.cancel_settlement_instruction(instruction.instruction_id) is True
    status_after_cancel = await connector.get_settlement_status(instruction.instruction_id)
    assert status_after_cancel.status.value == "cancelled"


@pytest.mark.anyio
async def test_healthcheck(cls_connector: Tuple[CLSConnector, MockCLSAPI]) -> None:
    connector, _ = cls_connector
    assert await connector.healthcheck() is True
