"""Tests for authentication and authorization."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from fastapi import HTTPException
from jose import jwt

from neutryx.api.auth.jwt_handler import JWTHandler, ALGORITHM, SECRET_KEY
from neutryx.api.auth.models import User, Token, TokenData, AuthProvider
from neutryx.api.auth.oauth2 import (
    OAuth2Config,
    get_google_oauth_config,
    get_azure_oauth_config,
    get_github_oauth_config,
)
from neutryx.api.auth.mfa import MFAHandler
from neutryx.api.auth.dependencies import (
    get_current_user,
    get_current_active_user,
    add_user_to_store,
    get_user_from_store,
)


class TestJWTHandler:
    """Test JWT token handling."""

    @pytest.fixture
    def handler(self):
        """Create JWT handler."""
        return JWTHandler()

    @pytest.fixture
    def test_user(self):
        """Create test user."""
        return User(
            user_id="test123",
            username="testuser",
            email="test@example.com",
            full_name="Test User",
            roles={"trader", "analyst"},
            permissions={"pricing.vanilla", "xva.cva"},
            tenant_id="tenant001",
        )

    def test_create_access_token(self, handler, test_user):
        """Test creating access token."""
        token = handler.create_access_token(test_user)

        assert isinstance(token, str)
        assert len(token) > 0

        # Decode and verify
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        assert payload["sub"] == test_user.user_id
        assert payload["username"] == test_user.username
        assert payload["tenant_id"] == test_user.tenant_id
        assert set(payload["roles"]) == test_user.roles
        assert set(payload["permissions"]) == test_user.permissions
        assert payload["type"] == "access"

    def test_create_refresh_token(self, handler, test_user):
        """Test creating refresh token."""
        token = handler.create_refresh_token(test_user)

        assert isinstance(token, str)
        assert len(token) > 0

        # Decode and verify
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        assert payload["sub"] == test_user.user_id
        assert payload["username"] == test_user.username
        assert payload["type"] == "refresh"

    def test_verify_access_token(self, handler, test_user):
        """Test verifying access token."""
        token = handler.create_access_token(test_user)
        token_data = handler.verify_token(token, token_type="access")

        assert isinstance(token_data, TokenData)
        assert token_data.user_id == test_user.user_id
        assert token_data.username == test_user.username
        assert token_data.tenant_id == test_user.tenant_id
        assert token_data.roles == test_user.roles
        assert token_data.permissions == test_user.permissions

    def test_verify_refresh_token(self, handler, test_user):
        """Test verifying refresh token."""
        token = handler.create_refresh_token(test_user)
        token_data = handler.verify_token(token, token_type="refresh")

        assert isinstance(token_data, TokenData)
        assert token_data.user_id == test_user.user_id
        assert token_data.username == test_user.username

    def test_verify_token_wrong_type(self, handler, test_user):
        """Test verifying token with wrong type."""
        access_token = handler.create_access_token(test_user)

        with pytest.raises(Exception):  # JWTError
            handler.verify_token(access_token, token_type="refresh")

    def test_verify_expired_token(self, handler, test_user):
        """Test verifying expired token."""
        # Create token with negative expiration
        token = handler.create_access_token(
            test_user,
            expires_delta=timedelta(seconds=-1),
        )

        with pytest.raises(Exception):  # JWTError for expired token
            handler.verify_token(token, token_type="access")

    def test_verify_invalid_token(self, handler):
        """Test verifying invalid token."""
        with pytest.raises(Exception):  # JWTError
            handler.verify_token("invalid.token.here", token_type="access")

    def test_token_custom_expiration(self, handler, test_user):
        """Test creating token with custom expiration."""
        custom_delta = timedelta(hours=2)
        token = handler.create_access_token(test_user, expires_delta=custom_delta)

        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        exp = datetime.fromtimestamp(payload["exp"])
        iat = datetime.fromtimestamp(payload["iat"])

        # Check expiration is approximately 2 hours from issuance
        assert (exp - iat).total_seconds() == pytest.approx(custom_delta.total_seconds(), abs=1)


class TestUser:
    """Test User model."""

    def test_create_user(self):
        """Test creating user."""
        user = User(
            user_id="user123",
            username="john.doe",
            email="john@example.com",
            full_name="John Doe",
            roles={"trader"},
            tenant_id="tenant001",
        )

        assert user.user_id == "user123"
        assert user.username == "john.doe"
        assert user.email == "john@example.com"
        assert "trader" in user.roles
        assert not user.disabled
        assert not user.mfa_enabled

    def test_user_defaults(self):
        """Test user default values."""
        user = User(user_id="user123", username="john.doe")

        assert user.disabled is False
        assert user.mfa_enabled is False
        assert user.auth_provider == AuthProvider.LOCAL
        assert len(user.roles) == 0
        assert len(user.permissions) == 0
        assert user.tenant_id is None

    def test_user_with_oauth(self):
        """Test user with OAuth provider."""
        user = User(
            user_id="oauth_user",
            username="oauth@example.com",
            auth_provider=AuthProvider.OAUTH2,
        )

        assert user.auth_provider == AuthProvider.OAUTH2


class TestOAuth2Config:
    """Test OAuth2 configuration."""

    def test_google_oauth_config(self):
        """Test Google OAuth configuration."""
        config = get_google_oauth_config(
            client_id="test_client_id",
            client_secret="test_secret",
            redirect_uri="http://localhost:8000/callback",
        )

        assert config.provider_name == "google"
        assert config.client_id == "test_client_id"
        assert "accounts.google.com" in config.authorization_endpoint
        assert "oauth2.googleapis.com" in config.token_endpoint
        assert "googleapis.com" in config.userinfo_endpoint

    def test_azure_oauth_config(self):
        """Test Azure OAuth configuration."""
        config = get_azure_oauth_config(
            tenant_id="test_tenant",
            client_id="test_client_id",
            client_secret="test_secret",
            redirect_uri="http://localhost:8000/callback",
        )

        assert config.provider_name == "azure"
        assert "test_tenant" in config.authorization_endpoint
        assert "microsoftonline.com" in config.token_endpoint

    def test_github_oauth_config(self):
        """Test GitHub OAuth configuration."""
        config = get_github_oauth_config(
            client_id="test_client_id",
            client_secret="test_secret",
            redirect_uri="http://localhost:8000/callback",
        )

        assert config.provider_name == "github"
        assert "github.com" in config.authorization_endpoint


class TestMFAHandler:
    """Test MFA handler."""

    @pytest.fixture
    def handler(self):
        """Create MFA handler."""
        return MFAHandler(issuer_name="Neutryx Test")

    @pytest.fixture
    def test_user(self):
        """Create test user."""
        return User(
            user_id="test123",
            username="testuser",
            email="test@example.com",
        )

    def test_generate_secret(self, handler):
        """Test generating TOTP secret."""
        secret = handler.generate_secret()

        assert isinstance(secret, str)
        assert len(secret) == 32  # Base32 secret is 32 characters

    def test_get_provisioning_uri(self, handler, test_user):
        """Test getting provisioning URI."""
        secret = handler.generate_secret()
        uri = handler.get_provisioning_uri(test_user, secret)

        assert isinstance(uri, str)
        assert uri.startswith("otpauth://totp/")
        assert "Neutryx" in uri
        # Email is URL-encoded in the URI
        assert "test%40example.com" in uri or test_user.email in uri

    def test_verify_code_valid(self, handler):
        """Test verifying valid TOTP code."""
        import pyotp

        secret = handler.generate_secret()
        totp = pyotp.TOTP(secret)
        code = totp.now()

        assert handler.verify_code(secret, code)

    def test_verify_code_invalid(self, handler):
        """Test verifying invalid TOTP code."""
        secret = handler.generate_secret()

        assert not handler.verify_code(secret, "000000")

    def test_generate_backup_codes(self, handler):
        """Test generating backup codes."""
        codes = handler.generate_backup_codes(count=10)

        assert len(codes) == 10
        assert all(isinstance(code, str) for code in codes)
        assert all(len(code) == 8 for code in codes)
        # All codes should be unique
        assert len(set(codes)) == 10

    @pytest.mark.asyncio
    async def test_setup_mfa(self, handler, test_user):
        """Test setting up MFA."""
        response = await handler.setup_mfa(test_user)

        assert isinstance(response.secret, str)
        assert isinstance(response.qr_code_uri, str)
        assert len(response.backup_codes) == 10

    @pytest.mark.asyncio
    async def test_enable_mfa(self, handler, test_user):
        """Test enabling MFA."""
        import pyotp

        secret = handler.generate_secret()
        totp = pyotp.TOTP(secret)
        code = totp.now()

        success = await handler.enable_mfa(test_user, secret, code)

        assert success
        assert test_user.mfa_enabled
        assert test_user.mfa_secret == secret

    @pytest.mark.asyncio
    async def test_enable_mfa_invalid_code(self, handler, test_user):
        """Test enabling MFA with invalid code."""
        secret = handler.generate_secret()

        success = await handler.enable_mfa(test_user, secret, "000000")

        assert not success
        assert not test_user.mfa_enabled

    @pytest.mark.asyncio
    async def test_disable_mfa(self, handler, test_user):
        """Test disabling MFA."""
        import pyotp

        # First enable MFA
        secret = handler.generate_secret()
        test_user.mfa_enabled = True
        test_user.mfa_secret = secret

        # Now disable it
        totp = pyotp.TOTP(secret)
        code = totp.now()

        success = await handler.disable_mfa(test_user, code)

        assert success
        assert not test_user.mfa_enabled
        assert test_user.mfa_secret is None


class TestUserStore:
    """Test user store functions."""

    def test_add_and_get_user(self):
        """Test adding and getting user from store."""
        user = User(
            user_id="store_test",
            username="storeuser",
            email="store@example.com",
        )

        add_user_to_store(user)
        retrieved = get_user_from_store("store_test")

        assert retrieved is not None
        assert retrieved.user_id == user.user_id
        assert retrieved.username == user.username

    def test_get_nonexistent_user(self):
        """Test getting nonexistent user."""
        retrieved = get_user_from_store("nonexistent_user_id_xyz")
        assert retrieved is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
