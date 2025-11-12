"""CLS connectivity and API integration."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
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
    CLSConfirmation,
    CLSSettlementInstruction,
    CLSSettlementStatus,
    CLSStatus,
)


class CLSConnector(CCPConnector):
    """Connector for CLS (Continuous Linked Settlement) integration.

    Provides connectivity to CLS for FX settlement instructions,
    status queries, and confirmation processing.
    """

    def __init__(
        self,
        config: CCPConfig,
        *,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        """Initialize CLS connector."""

        super().__init__(config)
        self._session_token: Optional[str] = None
        self._instructions: Dict[str, CLSSettlementInstruction] = {}
        self._client: Optional[httpx.AsyncClient] = None
        self._transport = transport
        self.metrics = CCPMetrics()

    async def connect(self) -> bool:
        """Establish connection to CLS."""

        try:
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-CLS-Member-ID": self.config.member_id,
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
            raise CCPConnectionError(f"CLS authentication timed out: {exc}")
        except CCPConnectionError:
            raise
        except Exception as exc:  # pragma: no cover - defensive fallback
            raise CCPConnectionError(f"Failed to connect to CLS: {exc}")

    async def disconnect(self) -> bool:
        """Disconnect from CLS."""

        try:
            if self._client:
                await self._client.aclose()
            self._client = None
            self._connected = False
            self._session_id = None
            self._session_token = None
            return True

        except Exception as e:  # pragma: no cover - defensive fallback
            raise CCPError(f"Failed to disconnect from CLS: {e}")

    async def submit_settlement_instruction(
        self,
        instruction: CLSSettlementInstruction,
    ) -> CLSConfirmation:
        """Submit FX settlement instruction to CLS."""

        if not self._connected or not self._client:
            raise CCPConnectionError("Not connected to CLS")

        try:
            instruction.validate_currencies()

            payload = instruction.model_dump(mode="json")
            if self._session_id:
                payload["session_id"] = self._session_id

            response = await self._perform_request(
                "POST",
                "/settlements/instructions",
                json=payload,
                expected_status=(200, 201, 202),
            )

            confirmation = CLSConfirmation(**response.json())
            instruction.status = confirmation.status
            self._instructions[instruction.instruction_id] = instruction
            return confirmation

        except CCPTimeoutError:
            raise
        except CCPTradeRejectionError:
            raise
        except Exception as e:
            raise CCPError(f"Failed to submit settlement instruction: {e}")

    async def get_settlement_status(self, instruction_id: str) -> CLSStatus:
        """Get current settlement status."""

        if not self._connected or not self._client:
            raise CCPConnectionError("Not connected to CLS")

        try:
            response = await self._perform_request(
                "GET",
                f"/settlements/instructions/{instruction_id}",
                expected_status=(200,),
            )

            status = CLSStatus(**response.json())
            if instruction_id in self._instructions:
                self._instructions[instruction_id].status = status.status
            return status

        except Exception as e:
            raise CCPError(f"Failed to get settlement status: {e}")

    async def cancel_settlement_instruction(self, instruction_id: str) -> bool:
        """Cancel a pending settlement instruction."""

        if not self._connected or not self._client:
            raise CCPConnectionError("Not connected to CLS")

        try:
            response = await self._perform_request(
                "POST",
                f"/settlements/instructions/{instruction_id}/cancel",
                expected_status=(200, 202, 204),
            )

            if instruction_id in self._instructions:
                self._instructions[instruction_id].status = CLSSettlementStatus.CANCELLED

            return response.status_code in (200, 202, 204)

        except Exception as e:
            raise CCPError(f"Failed to cancel instruction: {e}")

    async def get_session_liquidity(self, settlement_session_id: str) -> Dict[str, Any]:
        """Get liquidity information for settlement session."""

        if not self._connected or not self._client:
            raise CCPConnectionError("Not connected to CLS")

        response = await self._perform_request(
            "GET",
            f"/settlements/sessions/{settlement_session_id}/liquidity",
            expected_status=(200,),
        )

        return response.json()

    # Implement abstract methods from CCPConnector

    async def submit_trade(self, trade: Trade) -> TradeSubmissionResponse:
        """Submit trade to CLS trade capture API."""

        if not self._connected or not self._client:
            raise CCPConnectionError("Not connected to CLS")

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
        """Get trade status."""

        if not self._connected or not self._client:
            raise CCPConnectionError("Not connected to CLS")

        response = await self._perform_request(
            "GET",
            f"/trades/{trade_id}/status",
            expected_status=(200,),
        )

        data = response.json()
        status_value = data.get("status", TradeStatus.PENDING.value)
        return TradeStatus(status_value)

    async def cancel_trade(self, trade_id: str) -> bool:
        """Cancel trade by trade_id."""

        if not self._connected or not self._client:
            raise CCPConnectionError("Not connected to CLS")

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
        """Get margin requirements - not typically required for CLS."""

        if not self._connected or not self._client:
            raise CCPConnectionError("Not connected to CLS")

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
        """Get position report - returns settlement positions."""

        if not self._connected or not self._client:
            raise CCPConnectionError("Not connected to CLS")

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
        """Check CLS connectivity health."""

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
        """Execute HTTP request with retry and robust error handling."""

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
                    raise CCPTradeRejectionError(message or "CLS request rejected", code)

                if response.status_code in (401, 403):
                    raise CCPAuthenticationError("CLS authentication failed")

                if response.status_code == 408:
                    raise CCPTimeoutError("CLS request timed out")

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
                last_error = CCPTimeoutError(f"CLS request timed out: {exc}")
            except httpx.HTTPStatusError as exc:
                last_error = CCPConnectionError(
                    f"CLS server error {exc.response.status_code}: {exc.response.text}"
                )
            except httpx.RequestError as exc:
                last_error = CCPConnectionError(f"CLS request failed: {exc}")

            if attempt < attempts:
                await asyncio.sleep(self.config.retry_delay * attempt)

        if isinstance(last_error, CCPTimeoutError):
            raise last_error
        if isinstance(last_error, CCPConnectionError):
            raise last_error
        if last_error is not None:
            raise CCPConnectionError(str(last_error))

        raise CCPConnectionError("CLS request failed without response")

    @staticmethod
    def _safe_json(response: httpx.Response) -> Any:
        """Safely parse JSON payload."""

        try:
            return response.json()
        except Exception:  # pragma: no cover - defensive fallback
            return response.text


__all__ = ["CLSConnector"]
