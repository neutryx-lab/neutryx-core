"""Tests for authentication endpoints."""

from __future__ import annotations

import sys
import types

import pytest
import pyotp
from fastapi.testclient import TestClient

# Provide a lightweight prometheus_client stub if the dependency is unavailable.
if "prometheus_client" not in sys.modules:
    prometheus_stub = types.ModuleType("prometheus_client")

    class _DummyMetric:
        def labels(self, *args, **kwargs):
            return self

        def observe(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            return None

        def inc(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            return None

    class _Registry:
        def __init__(self) -> None:
            self._names_to_collectors: dict[str, _DummyMetric] = {}

        def register(self, collector):  # type: ignore[no-untyped-def]
            name = getattr(collector, "_metric_id", getattr(collector, "name", str(id(collector))))
            self._names_to_collectors[name] = collector

        def unregister(self, collector):  # type: ignore[no-untyped-def]
            name = getattr(collector, "_metric_id", getattr(collector, "name", str(id(collector))))
            self._names_to_collectors.pop(name, None)

    def _metric_factory(*args, **kwargs):  # type: ignore[no-untyped-def]
        metric = _DummyMetric()
        namespace = kwargs.get("namespace", "")
        subsystem = kwargs.get("subsystem", "")
        name = args[0] if args else kwargs.get("name", "")
        metric_id_parts = [part for part in (namespace, subsystem, name) if part]
        metric._metric_id = "_".join(metric_id_parts) if metric_id_parts else name  # type: ignore[attr-defined]
        prometheus_stub.REGISTRY._names_to_collectors[metric._metric_id] = metric  # type: ignore[attr-defined]
        return metric

    prometheus_stub.CONTENT_TYPE_LATEST = "text/plain"
    prometheus_stub.Counter = _metric_factory  # type: ignore[attr-defined]
    prometheus_stub.Histogram = _metric_factory  # type: ignore[attr-defined]
    prometheus_stub.REGISTRY = _Registry()
    prometheus_stub.generate_latest = lambda *args, **kwargs: b""  # type: ignore[attr-defined]

    sys.modules["prometheus_client"] = prometheus_stub

from neutryx.api.rest import create_app
from neutryx.api.auth.models import User
from neutryx.api.auth.jwt_handler import JWTHandler
from neutryx.api.auth.dependencies import add_user_to_store, register_local_user


@pytest.fixture
def app():
    """Create test app."""
    return create_app()


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_user_store():
    """Ensure the in-memory user store is isolated across tests."""

    clear_user_store()
    yield
    clear_user_store()


@pytest.fixture
def test_user():
    """Create and register test user."""
    user = User(
        user_id="test123",
        username="testuser",
        email="test@example.com",
        full_name="Test User",
        roles={"trader"},
        permissions={"pricing.vanilla"},
        tenant_id="tenant001",
    )
    add_user_to_store(user)
    return user


@pytest.fixture
def auth_token(test_user):
    """Create auth token for test user."""
    handler = JWTHandler()
    return handler.create_access_token(test_user)


class TestAuthEndpoints:
    """Test authentication endpoints."""

    def test_login_local_user(self, client):
        """Local users can authenticate via the token endpoint."""

        password = "Sup3rSecret!"
        user = User(
            user_id="local123",
            username="localuser",
            email="local@example.com",
            full_name="Local User",
            roles={"viewer"},
        )
        register_local_user(user, password)

        response = client.post(
            "/auth/token",
            data={"username": user.username, "password": password},
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["token_type"] == "bearer"
        assert payload["access_token"]
        assert payload["refresh_token"]

    def test_login_local_user_invalid_password(self, client):
        """Invalid credentials for a local user should fail."""

        user = User(user_id="local124", username="localuser2")
        register_local_user(user, "CorrectHorseBatteryStaple")

        response = client.post(
            "/auth/token",
            data={"username": user.username, "password": "wrong"},
        )

        assert response.status_code == 401
        assert response.json()["detail"] == "Incorrect username or password"

    def test_login_local_user_mfa(self, client):
        """Local users with MFA must provide the correct TOTP code."""

        password = "Another$ecret1"
        secret = pyotp.random_base32()
        user = User(
            user_id="local125",
            username="localmfa",
            mfa_enabled=True,
            mfa_secret=secret,
        )
        register_local_user(user, password)

        totp = pyotp.TOTP(secret)

        response = client.post(
            "/auth/token",
            data={
                "username": user.username,
                "password": f"{password}:{totp.now()}",
            },
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["access_token"]

    def test_login_local_user_missing_mfa_code(self, client):
        """Fail login when MFA enabled but code is missing."""

        password = "NoMFACode123"
        user = User(
            user_id="local126",
            username="localmissingmfa",
            mfa_enabled=True,
            mfa_secret=pyotp.random_base32(),
        )
        register_local_user(user, password)

        response = client.post(
            "/auth/token",
            data={"username": user.username, "password": password},
        )

        assert response.status_code == 401
        assert response.json()["detail"] == "MFA code required. Format: password:mfa_code"

    def test_get_current_user(self, client, test_user, auth_token):
        """Test getting current user info."""
        response = client.get(
            "/auth/me",
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == test_user.user_id
        assert data["username"] == test_user.username
        assert data["email"] == test_user.email

    def test_get_current_user_no_token(self, client):
        """Test getting current user without token."""
        response = client.get("/auth/me")

        assert response.status_code == 403  # Forbidden (no auth header)

    def test_get_current_user_invalid_token(self, client):
        """Test getting current user with invalid token."""
        response = client.get(
            "/auth/me",
            headers={"Authorization": "Bearer invalid.token.here"},
        )

        assert response.status_code == 401  # Unauthorized

    def test_refresh_token_endpoint(self, client, test_user):
        """Test refresh token endpoint."""
        handler = JWTHandler()
        refresh_token = handler.create_refresh_token(test_user)

        response = client.post(
            "/auth/refresh",
            params={"refresh_token": refresh_token},
        )

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"

    def test_local_login_without_ldap(self, client):
        """Local users can obtain tokens when LDAP is not configured."""

        user = User(
            user_id="local-user-001",
            username="localuser",
            email="local@example.com",
            full_name="Local User",
            roles={"viewer"},
            permissions=set(),
            tenant_id="tenant-local",
        )
        register_local_user(user, "s3cret123")

        response = client.post(
            "/auth/token",
            data={
                "username": "localuser",
                "password": "s3cret123",
                "grant_type": "password",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_mfa_setup(self, client, auth_token):
        """Test MFA setup endpoint."""
        response = client.post(
            "/auth/mfa/setup",
            headers={"Authorization": f"Bearer {auth_token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "secret" in data
        assert "qr_code_uri" in data
        assert "backup_codes" in data
        assert len(data["backup_codes"]) == 10

    def test_mfa_verify(self, client, test_user, auth_token):
        """Test MFA verification endpoint."""
        # Invalid code should return false
        response = client.post(
            "/auth/mfa/verify",
            headers={"Authorization": f"Bearer {auth_token}"},
            params={"code": "000000"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False


class TestProtectedEndpoints:
    """Test protected endpoints require authentication."""

    def test_price_vanilla_without_auth(self, client):
        """Test that pricing endpoints work without auth (not protected by default)."""
        # By default, endpoints are not protected
        # This test documents current behavior
        payload = {
            "spot": 100.0,
            "strike": 100.0,
            "maturity": 1.0,
            "rate": 0.05,
            "dividend": 0.0,
            "volatility": 0.2,
            "call": True,
            "steps": 100,
            "paths": 1000,
        }

        response = client.post("/price/vanilla", json=payload)

        # Should work without authentication (current behavior)
        assert response.status_code == 200
        assert "price" in response.json()


class TestAuthIntegration:
    """Test authentication integration with RBAC."""

    def test_user_with_roles(self, test_user):
        """Test user has correct roles."""
        assert "trader" in test_user.roles
        assert "admin" not in test_user.roles

    def test_user_with_permissions(self, test_user):
        """Test user has correct permissions."""
        assert "pricing.vanilla" in test_user.permissions

    def test_token_contains_roles(self, test_user):
        """Test JWT token contains roles."""
        handler = JWTHandler()
        token = handler.create_access_token(test_user)
        token_data = handler.verify_token(token)

        assert token_data.roles == test_user.roles
        assert token_data.permissions == test_user.permissions


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
