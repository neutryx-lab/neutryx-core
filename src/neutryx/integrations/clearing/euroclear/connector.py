"""Euroclear connectivity and API integration."""

from __future__ import annotations

import asyncio
import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, Optional, Sequence

import httpx

from ..base import (
    CCPAuthenticationError,
    CCPConfig,
    CCPConnector,
    CCPConnectionError,
    CCPError,
    CCPMetrics,
    CCPTimeoutError,
    CCPTradeRejectionError,
    Trade,
    TradeStatus,
    TradeSubmissionResponse,
)
from .messages import (
    EuroclearConfirmation,
    EuroclearSettlementInstruction,
    EuroclearStatus,
    SettlementStatus,
)


class EuroclearConnector(CCPConnector):
    """Connector for Euroclear integration."""

    def __init__(
        self,
        config: CCPConfig,
        *,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        super().__init__(config)
        self._session_token: Optional[str] = None
        self._instructions: Dict[str, EuroclearSettlementInstruction] = {}
        self._client: Optional[httpx.AsyncClient] = None
        self._transport = transport
        self.metrics = CCPMetrics()

    async def connect(self) -> bool:
        """Establish connection to Euroclear."""

        try:
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-Euroclear-Member-ID": self.config.member_id,
            }

            if self.config.api_key:
                headers["X-API-Key"] = self.config.api_key

            timeout = httpx.Timeout(self.config.timeout, connect=10.0)

            self._client = httpx.AsyncClient(
                base_url=self.config.api_endpoint,
                headers=headers,
                timeout=timeout,
                transport=self._transport,
            )

            auth_payload = {
                "member_id": self.config.member_id,
                "environment": self.config.environment,
                "timestamp": datetime.utcnow().isoformat(),
            }

            response = await self._perform_request(
                "POST",
                "/auth/session",
                json=auth_payload,
                expected_status=(200, 201),
                allow_retries=False,
            )

            data = response.json()
            self._session_id = data.get("session_id") or str(uuid.uuid4())
            self._session_token = data.get("session_token")
            self._connected = True
            self.metrics.last_heartbeat = datetime.utcnow()
            return True

        except CCPAuthenticationError:
            raise
        except CCPTimeoutError as exc:
            raise CCPConnectionError(f"Euroclear authentication timed out: {exc}")
        except CCPConnectionError:
            raise
        except Exception as exc:  # pragma: no cover - defensive fallback
            raise CCPConnectionError(f"Failed to connect to Euroclear: {exc}")

    async def disconnect(self) -> bool:
        """Disconnect from Euroclear."""

        try:
            if self._client:
                await self._client.aclose()
            self._client = None
            self._connected = False
            self._session_id = None
            self._session_token = None
            return True

        except Exception as e:  # pragma: no cover - defensive fallback
            raise CCPError(f"Failed to disconnect from Euroclear: {e}")

    async def submit_settlement_instruction(
        self,
        instruction: EuroclearSettlementInstruction,
    ) -> EuroclearConfirmation:
        """Submit securities settlement instruction to Euroclear."""

        if not self._connected or not self._client:
            raise CCPConnectionError("Not connected to Euroclear")

        try:
            instruction.validate_dvp_fields()
            instruction.validate_dfp_fields()

            payload = instruction.model_dump(mode="json")
            if self._session_id:
                payload["session_id"] = self._session_id

            response = await self._perform_request(
                "POST",
                "/settlements/securities",
                json=payload,
                expected_status=(200, 201, 202),
            )

            confirmation = EuroclearConfirmation(**response.json())
            instruction.status = confirmation.status
            self._instructions[instruction.instruction_id] = instruction
            return confirmation

        except CCPTradeRejectionError:
            raise
        except CCPTimeoutError:
            raise
        except Exception as e:
            raise CCPError(f"Failed to submit settlement instruction: {e}")

    async def get_settlement_status(self, instruction_id: str) -> EuroclearStatus:
        """Get current settlement status."""

        if not self._connected or not self._client:
            raise CCPConnectionError("Not connected to Euroclear")

        try:
            response = await self._perform_request(
                "GET",
                f"/settlements/securities/{instruction_id}",
                expected_status=(200,),
            )

            status = EuroclearStatus(**response.json())
            if instruction_id in self._instructions:
                self._instructions[instruction_id].status = status.status
            return status

        except Exception as e:
            raise CCPError(f"Failed to get settlement status: {e}")

    async def cancel_settlement_instruction(self, instruction_id: str) -> bool:
        """Cancel a pending settlement instruction."""

        if not self._connected or not self._client:
            raise CCPConnectionError("Not connected to Euroclear")

        try:
            response = await self._perform_request(
                "POST",
                f"/settlements/securities/{instruction_id}/cancel",
                expected_status=(200, 202, 204),
            )

            if instruction_id in self._instructions:
                self._instructions[instruction_id].status = SettlementStatus.CANCELLED

            return response.status_code in (200, 202, 204)

        except Exception as e:
            raise CCPError(f"Failed to cancel instruction: {e}")

    async def amend_settlement_instruction(
        self,
        instruction_id: str,
        new_settlement_date: Optional[date] = None,
        new_quantity: Optional[Decimal] = None,
    ) -> EuroclearConfirmation:
        """Amend a pending settlement instruction."""

        if not self._connected or not self._client:
            raise CCPConnectionError("Not connected to Euroclear")

        try:
            payload: Dict[str, Any] = {}
            if new_settlement_date is not None:
                payload["settlement_date"] = new_settlement_date.isoformat()
            if new_quantity is not None:
                payload["quantity"] = str(new_quantity)

            response = await self._perform_request(
                "PATCH",
                f"/settlements/securities/{instruction_id}",
                json=payload,
                expected_status=(200, 202),
            )

            confirmation = EuroclearConfirmation(**response.json())
            if instruction_id in self._instructions:
                instruction = self._instructions[instruction_id]
                instruction.settlement_date = confirmation.settlement_date
                instruction.quantity = confirmation.quantity_instructed
                instruction.status = confirmation.status
            return confirmation

        except Exception as e:
            raise CCPError(f"Failed to amend instruction: {e}")

    async def get_holdings(
        self,
        account_id: str,
        as_of_date: Optional[date] = None,
    ) -> Dict[str, Any]:
        """Get securities holdings for an account."""

        if not self._connected or not self._client:
            raise CCPConnectionError("Not connected to Euroclear")

        params: Dict[str, Any] = {}
        if as_of_date:
            params["as_of_date"] = as_of_date.isoformat()

        response = await self._perform_request(
            "GET",
            f"/accounts/{account_id}/holdings",
            params=params,
            expected_status=(200,),
        )

        return response.json()

    async def submit_trade(self, trade: Trade) -> TradeSubmissionResponse:
        """Submit trade for clearing."""

        if not self._connected or not self._client:
            raise CCPConnectionError("Not connected to Euroclear")

        start_time = datetime.utcnow()

        try:
            payload = trade.to_dict()
            payload["member_id"] = self.config.member_id
            if self._session_id:
                payload["session_id"] = self._session_id

            response = await self._perform_request(
                "POST",
                "/trades",
                json=payload,
                expected_status=(200, 201, 202),
            )

            duration = (datetime.utcnow() - start_time).total_seconds() * 1000
            submission = TradeSubmissionResponse(**response.json())
            self.metrics.record_submission(True, duration)
            return submission

        except CCPTradeRejectionError as exc:
            duration = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.metrics.record_submission(False, duration, rejected=True)
            raise exc
        except (CCPTimeoutError, CCPConnectionError) as exc:
            duration = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.metrics.record_submission(False, duration)
            raise exc

    async def get_trade_status(self, trade_id: str) -> TradeStatus:
        """Get current trade status."""

        if not self._connected or not self._client:
            raise CCPConnectionError("Not connected to Euroclear")

        response = await self._perform_request(
            "GET",
            f"/trades/{trade_id}/status",
            expected_status=(200,),
        )

        data = response.json()
        status_value = data.get("status", TradeStatus.PENDING.value)
        return TradeStatus(status_value)

    async def cancel_trade(self, trade_id: str) -> bool:
        """Cancel a pending trade."""

        if not self._connected or not self._client:
            raise CCPConnectionError("Not connected to Euroclear")

        response = await self._perform_request(
            "POST",
            f"/trades/{trade_id}/cancel",
            expected_status=(200, 202, 204),
        )

        return response.status_code in (200, 202, 204)

    async def get_margin_requirements(
        self,
        member_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get margin requirements for the member."""

        if not self._connected or not self._client:
            raise CCPConnectionError("Not connected to Euroclear")

        response = await self._perform_request(
            "GET",
            "/margin",
            params={"member_id": member_id or self.config.member_id},
            expected_status=(200,),
        )

        return response.json()

    async def get_position_report(
        self,
        as_of_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Get position report from Euroclear."""

        if not self._connected or not self._client:
            raise CCPConnectionError("Not connected to Euroclear")

        params: Dict[str, Any] = {}
        if as_of_date:
            params["as_of_date"] = as_of_date.isoformat()

        response = await self._perform_request(
            "GET",
            "/reports/positions",
            params=params,
            expected_status=(200,),
        )

        return response.json()

    async def healthcheck(self) -> bool:
        """Run Euroclear healthcheck."""

        if not self._connected or not self._client:
            return False

        try:
            response = await self._perform_request(
                "GET",
                "/health",
                expected_status=(200,),
            )
            return response.status_code == 200
        except CCPError:
            return False

    async def _perform_request(
        self,
        method: str,
        endpoint: str,
        *,
        expected_status: Sequence[int],
        allow_retries: bool = True,
        **kwargs: Any,
    ) -> httpx.Response:
        """Execute HTTP request with retry and error handling."""

        if not self._client:
            raise CCPConnectionError("HTTP client not initialized")

        attempts = self.config.max_retries if allow_retries else 1
        last_error: Optional[Exception] = None

        for attempt in range(1, attempts + 1):
            try:
                response = await self._client.request(method, endpoint, **kwargs)

                if response.status_code in expected_status:
                    return response

                if response.status_code in (400, 422):
                    payload = self._safe_json(response)
                    message = payload.get("message") if isinstance(payload, dict) else response.text
                    code = payload.get("code") if isinstance(payload, dict) else None
                    raise CCPTradeRejectionError(message or "Euroclear request rejected", code)

                if response.status_code in (401, 403):
                    raise CCPAuthenticationError("Euroclear authentication failed")

                if response.status_code == 408:
                    raise CCPTimeoutError("Euroclear request timed out")

                if response.status_code >= 500:
                    response.raise_for_status()

                raise CCPConnectionError(
                    f"Unexpected response {response.status_code}: {response.text}"
                )

            except CCPTradeRejectionError:
                raise
            except CCPAuthenticationError:
                raise
            except CCPTimeoutError as exc:
                last_error = exc
            except httpx.TimeoutException as exc:
                last_error = CCPTimeoutError(f"Euroclear request timed out: {exc}")
            except httpx.HTTPStatusError as exc:
                last_error = CCPConnectionError(
                    f"Euroclear server error {exc.response.status_code}: {exc.response.text}"
                )
            except httpx.RequestError as exc:
                last_error = CCPConnectionError(f"Euroclear request failed: {exc}")

            if attempt < attempts:
                await asyncio.sleep(self.config.retry_delay * attempt)

        if isinstance(last_error, CCPTimeoutError):
            raise last_error
        if isinstance(last_error, CCPConnectionError):
            raise last_error
        if last_error is not None:
            raise CCPConnectionError(str(last_error))

        raise CCPConnectionError("Euroclear request failed without response")

    @staticmethod
    def _safe_json(response: httpx.Response) -> Any:
        """Safely parse JSON payload."""

        try:
            return response.json()
        except Exception:  # pragma: no cover - defensive fallback
            return response.text


__all__ = ["EuroclearConnector"]
