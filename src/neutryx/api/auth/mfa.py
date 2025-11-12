"""Multi-factor authentication (MFA) support."""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Protocol

import pyotp

from .models import (
    EmailProviderConfig,
    MFASetupResponse,
    SMSProviderConfig,
    User,
    VerificationCodeSettings,
)


class MFAHandler:
    """Handle multi-factor authentication (TOTP-based)."""

    def __init__(self, issuer_name: str = "Neutryx"):
        """Initialize MFA handler.

        Args:
            issuer_name: Name shown in authenticator apps
        """
        self.issuer_name = issuer_name

    def generate_secret(self) -> str:
        """Generate a new TOTP secret.

        Returns:
            Base32-encoded secret string
        """
        return pyotp.random_base32()

    def get_provisioning_uri(self, user: User, secret: str) -> str:
        """Get provisioning URI for QR code generation.

        Args:
            user: User object
            secret: TOTP secret

        Returns:
            Provisioning URI string for QR code
        """
        totp = pyotp.TOTP(secret)
        return totp.provisioning_uri(
            name=user.email or user.username,
            issuer_name=self.issuer_name,
        )

    def verify_code(self, secret: str, code: str, valid_window: int = 1) -> bool:
        """Verify TOTP code.

        Args:
            secret: TOTP secret
            code: 6-digit code from authenticator
            valid_window: Number of time windows to check (default: 1)

        Returns:
            True if code is valid, False otherwise
        """
        totp = pyotp.TOTP(secret)
        return totp.verify(code, valid_window=valid_window)

    def generate_backup_codes(self, count: int = 10) -> List[str]:
        """Generate backup codes for MFA recovery.

        Args:
            count: Number of backup codes to generate

        Returns:
            List of backup codes
        """
        return [self._generate_backup_code() for _ in range(count)]

    def _generate_backup_code(self) -> str:
        """Generate a single backup code.

        Returns:
            8-character backup code
        """
        # Generate 8-character alphanumeric code
        return secrets.token_hex(4).upper()

    async def setup_mfa(self, user: User) -> MFASetupResponse:
        """Set up MFA for a user.

        Args:
            user: User object

        Returns:
            MFASetupResponse with secret, QR code URI, and backup codes
        """
        secret = self.generate_secret()
        qr_uri = self.get_provisioning_uri(user, secret)
        backup_codes = self.generate_backup_codes()

        return MFASetupResponse(
            secret=secret,
            qr_code_uri=qr_uri,
            backup_codes=backup_codes,
        )

    async def enable_mfa(self, user: User, secret: str, verification_code: str) -> bool:
        """Enable MFA for a user after verifying initial code.

        Args:
            user: User object
            secret: TOTP secret
            verification_code: Initial verification code

        Returns:
            True if verification successful and MFA enabled
        """
        if not self.verify_code(secret, verification_code):
            return False

        # In a real implementation, this would update the database
        user.mfa_enabled = True
        user.mfa_secret = secret

        return True

    async def disable_mfa(self, user: User, verification_code: str) -> bool:
        """Disable MFA for a user.

        Args:
            user: User object
            verification_code: MFA code to verify before disabling

        Returns:
            True if verification successful and MFA disabled
        """
        if not user.mfa_enabled or not user.mfa_secret:
            return False

        if not self.verify_code(user.mfa_secret, verification_code):
            return False

        # In a real implementation, this would update the database
        user.mfa_enabled = False
        user.mfa_secret = None

        return True

    async def verify_mfa(self, user: User, code: str) -> bool:
        """Verify MFA code for a user.

        Args:
            user: User object
            code: MFA code from authenticator

        Returns:
            True if code is valid
        """
        if not user.mfa_enabled or not user.mfa_secret:
            return False

        return self.verify_code(user.mfa_secret, code)


# SMS/Email MFA support
@dataclass
class _VerificationState:
    """Track outstanding verification codes for a destination."""

    code: str
    expires_at: float
    attempts: int = 0


class _BaseCodeHandler:
    """Shared functionality for SMS and email MFA handlers."""

    def __init__(
        self,
        code_settings: VerificationCodeSettings,
        clock: Optional[Callable[[], float]] = None,
        code_generator: Optional[Callable[[int], str]] = None,
    ) -> None:
        self._code_settings = code_settings
        self._clock = clock or time.time
        self._generate_code = code_generator or self._default_code_generator
        self._pending_codes: Dict[str, _VerificationState] = {}

    def _default_code_generator(self, length: int) -> str:
        return f"{secrets.randbelow(10 ** length):0{length}d}"

    def _record_code(self, destination: str, code: str) -> str:
        expires_at = self._clock() + self._code_settings.ttl_seconds
        self._pending_codes[destination] = _VerificationState(code=code, expires_at=expires_at)
        # A unique identifier can be useful for observability/logging purposes.
        return secrets.token_urlsafe(16)

    def _validate_code(self, destination: str, code: str) -> bool:
        state = self._pending_codes.get(destination)
        if not state:
            return False

        now = self._clock()
        if now >= state.expires_at:
            # Expired codes are removed immediately.
            self._pending_codes.pop(destination, None)
            return False

        if secrets.compare_digest(code, state.code):
            self._pending_codes.pop(destination, None)
            return True

        state.attempts += 1
        if state.attempts >= self._code_settings.max_attempts:
            self._pending_codes.pop(destination, None)
        return False


class SMSProvider(Protocol):
    """Abstract SMS provider for MFA code delivery."""

    async def send_sms(self, to: str, body: str, from_number: str) -> None:
        """Send an SMS message."""


class EmailProvider(Protocol):
    """Abstract Email provider for MFA code delivery."""

    async def send_email(self, to: str, subject: str, body: str, from_address: str) -> None:
        """Send an email message."""


class SMSMFAHandler(_BaseCodeHandler):
    """Handle SMS-based MFA using an injected SMS provider."""

    def __init__(
        self,
        sms_provider_config: SMSProviderConfig | dict,
        sms_provider: SMSProvider,
        clock: Optional[Callable[[], float]] = None,
        code_generator: Optional[Callable[[int], str]] = None,
    ) -> None:
        """Initialize SMS MFA handler.

        Args:
            sms_provider_config: SMS provider configuration (Twilio, etc.).
            sms_provider: Provider implementation responsible for sending SMS messages.
            clock: Optional callable returning the current timestamp (for testing).
            code_generator: Optional deterministic code generator (for testing).
        """

        config = sms_provider_config
        if isinstance(config, dict):
            config = SMSProviderConfig(**config)

        self.config = config
        self._provider = sms_provider
        super().__init__(config.code_settings, clock=clock, code_generator=code_generator)

    async def send_verification_code(self, phone_number: str) -> str:
        """Generate and send a verification code via SMS."""

        code = self._generate_code(self._code_settings.length)
        message = self.config.message_template.format(code=code)
        message_id = self._record_code(phone_number, code)
        await self._provider.send_sms(phone_number, message, self.config.from_number)
        return message_id

    async def verify_code(self, phone_number: str, code: str) -> bool:
        """Verify an SMS-delivered MFA code."""

        return self._validate_code(phone_number, code)


class EmailMFAHandler(_BaseCodeHandler):
    """Handle Email-based MFA using an injected email provider."""

    def __init__(
        self,
        email_provider_config: EmailProviderConfig | dict,
        email_provider: EmailProvider,
        clock: Optional[Callable[[], float]] = None,
        code_generator: Optional[Callable[[int], str]] = None,
    ) -> None:
        """Initialize Email MFA handler.

        Args:
            email_provider_config: Email provider configuration.
            email_provider: Provider implementation responsible for sending emails.
            clock: Optional callable returning the current timestamp (for testing).
            code_generator: Optional deterministic code generator (for testing).
        """

        config = email_provider_config
        if isinstance(config, dict):
            config = EmailProviderConfig(**config)

        self.config = config
        self._provider = email_provider
        super().__init__(config.code_settings, clock=clock, code_generator=code_generator)

    async def send_verification_code(self, email: str) -> str:
        """Generate and send a verification code via email."""

        code = self._generate_code(self._code_settings.length)
        subject = self.config.subject_template.format(code=code)
        body = self.config.body_template.format(code=code)
        message_id = self._record_code(email, code)
        await self._provider.send_email(email, subject, body, self.config.from_address)
        return message_id

    async def verify_code(self, email: str, code: str) -> bool:
        """Verify an email-delivered MFA code."""

        return self._validate_code(email, code)


__all__ = [
    "MFAHandler",
    "SMSMFAHandler",
    "EmailMFAHandler",
]
