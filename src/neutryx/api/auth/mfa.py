"""Multi-factor authentication (MFA) support."""

from __future__ import annotations

import secrets
from typing import List
from urllib.parse import quote

import pyotp

from .models import User, MFASetupResponse


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


# SMS/Email MFA Support (placeholder for future implementation)
class SMSMFAHandler:
    """Handle SMS-based MFA (future implementation)."""

    def __init__(self, sms_provider_config: dict):
        """Initialize SMS MFA handler.

        Args:
            sms_provider_config: SMS provider configuration (Twilio, etc.)
        """
        self.config = sms_provider_config
        raise NotImplementedError("SMS MFA not yet implemented")

    async def send_verification_code(self, phone_number: str) -> str:
        """Send verification code via SMS."""
        raise NotImplementedError()

    async def verify_code(self, phone_number: str, code: str) -> bool:
        """Verify SMS code."""
        raise NotImplementedError()


class EmailMFAHandler:
    """Handle Email-based MFA (future implementation)."""

    def __init__(self, email_provider_config: dict):
        """Initialize Email MFA handler.

        Args:
            email_provider_config: Email provider configuration
        """
        self.config = email_provider_config
        raise NotImplementedError("Email MFA not yet implemented")

    async def send_verification_code(self, email: str) -> str:
        """Send verification code via email."""
        raise NotImplementedError()

    async def verify_code(self, email: str, code: str) -> bool:
        """Verify email code."""
        raise NotImplementedError()


__all__ = [
    "MFAHandler",
    "SMSMFAHandler",
    "EmailMFAHandler",
]
