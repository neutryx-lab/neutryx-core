"""Tests for SMS and Email MFA handlers."""

from __future__ import annotations

import sys
import types

if "prometheus_client" not in sys.modules:
    stub = types.ModuleType("prometheus_client")
    stub.CONTENT_TYPE_LATEST = "text/plain"

    def _noop(*args, **kwargs):  # pragma: no cover - test stub
        return None

    stub.Counter = _noop
    stub.Histogram = _noop
    stub.REGISTRY = None
    stub.generate_latest = _noop
    sys.modules["prometheus_client"] = stub

from dataclasses import dataclass
from typing import List

import pytest

from neutryx.api.auth.mfa import EmailMFAHandler, EmailProvider, SMSMFAHandler, SMSProvider
from neutryx.api.auth.models import EmailProviderConfig, SMSProviderConfig, VerificationCodeSettings


@dataclass
class RecordedSMS:
    to: str
    body: str
    from_number: str


class MockSMSProvider(SMSProvider):
    """Mock SMS provider that records outbound messages."""

    def __init__(self) -> None:
        self.messages: List[RecordedSMS] = []

    async def send_sms(self, to: str, body: str, from_number: str) -> None:  # pragma: no cover - interface compliance
        self.messages.append(RecordedSMS(to=to, body=body, from_number=from_number))


@dataclass
class RecordedEmail:
    to: str
    subject: str
    body: str
    from_address: str


class MockEmailProvider(EmailProvider):
    """Mock Email provider that records outbound messages."""

    def __init__(self) -> None:
        self.messages: List[RecordedEmail] = []

    async def send_email(self, to: str, subject: str, body: str, from_address: str) -> None:  # pragma: no cover - interface compliance
        self.messages.append(RecordedEmail(to=to, subject=subject, body=body, from_address=from_address))


class DeterministicClock:
    """Deterministic clock helper for simulating time progression."""

    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


@pytest.mark.asyncio
async def test_sms_mfa_handler_send_and_verify() -> None:
    """SMS handler should send and verify codes using the mock provider."""

    provider = MockSMSProvider()
    config = SMSProviderConfig(
        from_number="+15550000000",
        message_template="Code: {code}",
        code_settings=VerificationCodeSettings(ttl_seconds=120, max_attempts=3),
    )
    handler = SMSMFAHandler(config, provider)

    message_id = await handler.send_verification_code("+15551112222")
    assert isinstance(message_id, str) and message_id
    assert len(provider.messages) == 1
    message = provider.messages[0]
    assert message.to == "+15551112222"
    assert message.from_number == "+15550000000"
    code = message.body.split(": ")[-1]

    assert await handler.verify_code("+15551112222", code) is True
    # Code should be removed after successful verification
    assert await handler.verify_code("+15551112222", code) is False


@pytest.mark.asyncio
async def test_sms_mfa_handler_expiration_and_attempts() -> None:
    """Expired or repeatedly invalid codes should fail verification."""

    clock = DeterministicClock(start=100.0)
    provider = MockSMSProvider()
    config = SMSProviderConfig(
        from_number="+15550000000",
        message_template="Code: {code}",
        code_settings=VerificationCodeSettings(ttl_seconds=10, max_attempts=2),
    )
    handler = SMSMFAHandler(config, provider, clock=clock.__call__)

    await handler.send_verification_code("+15553334444")
    code = provider.messages[0].body.split(": ")[-1]

    # Wrong attempts should eventually invalidate the code
    assert await handler.verify_code("+15553334444", "000000") is False
    assert await handler.verify_code("+15553334444", "111111") is False
    # Even the correct code should now fail because attempts exceeded max_attempts
    assert await handler.verify_code("+15553334444", code) is False

    # Reset by sending another code and let it expire
    await handler.send_verification_code("+15553334444")
    code = provider.messages[-1].body.split(": ")[-1]
    clock.advance(11)
    assert await handler.verify_code("+15553334444", code) is False


@pytest.mark.asyncio
async def test_email_mfa_handler_send_and_verify() -> None:
    """Email handler should send and verify codes using the mock provider."""

    provider = MockEmailProvider()
    config = EmailProviderConfig(
        from_address="no-reply@example.com",
        subject_template="Your code {code}",
        body_template="Hello, your code is {code}",
        code_settings=VerificationCodeSettings(ttl_seconds=60, max_attempts=3),
    )
    handler = EmailMFAHandler(config, provider)

    await handler.send_verification_code("user@example.com")
    assert len(provider.messages) == 1
    message = provider.messages[0]
    assert message.to == "user@example.com"
    assert message.from_address == "no-reply@example.com"
    assert "Your code" in message.subject
    code = message.body.split(" ")[-1]

    assert await handler.verify_code("user@example.com", code) is True
    assert await handler.verify_code("user@example.com", code) is False


@pytest.mark.asyncio
async def test_email_mfa_handler_invalid_then_valid() -> None:
    """An incorrect email code should not prevent a later valid attempt within limits."""

    clock = DeterministicClock(start=50.0)
    provider = MockEmailProvider()
    config = EmailProviderConfig(
        from_address="no-reply@example.com",
        body_template="Code={code}",
        code_settings=VerificationCodeSettings(ttl_seconds=30, max_attempts=3),
    )
    handler = EmailMFAHandler(config, provider, clock=clock.__call__)

    await handler.send_verification_code("user@example.com")
    code = provider.messages[0].body.split("=")[-1]

    assert await handler.verify_code("user@example.com", "badcode") is False
    # Advance time but remain within TTL
    clock.advance(5)
    assert await handler.verify_code("user@example.com", code) is True

