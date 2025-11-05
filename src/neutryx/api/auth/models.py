"""Authentication data models."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional, Set

from pydantic import BaseModel, EmailStr, Field


class UserRole(str, Enum):
    """User roles."""
    ADMIN = "admin"
    TRADER = "trader"
    RISK_MANAGER = "risk_manager"
    ANALYST = "analyst"
    VIEWER = "viewer"


class AuthProvider(str, Enum):
    """Authentication provider types."""
    LOCAL = "local"
    OAUTH2 = "oauth2"
    LDAP = "ldap"
    SAML = "saml"


class User(BaseModel):
    """User model with authentication and authorization data."""

    user_id: str = Field(..., description="Unique user identifier")
    username: str = Field(..., description="Username for login")
    email: Optional[EmailStr] = Field(None, description="User email address")
    full_name: Optional[str] = Field(None, description="Full name")

    # Authentication
    disabled: bool = Field(default=False, description="Whether user is disabled")
    auth_provider: AuthProvider = Field(default=AuthProvider.LOCAL, description="Authentication provider")

    # Authorization
    roles: Set[str] = Field(default_factory=set, description="User roles")
    permissions: Set[str] = Field(default_factory=set, description="User permissions")
    tenant_id: Optional[str] = Field(None, description="Tenant ID for multi-tenancy")

    # MFA
    mfa_enabled: bool = Field(default=False, description="Whether MFA is enabled")
    mfa_secret: Optional[str] = Field(None, description="MFA secret (TOTP)")

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user123",
                "username": "john.doe",
                "email": "john.doe@example.com",
                "full_name": "John Doe",
                "roles": ["trader", "analyst"],
                "tenant_id": "tenant001",
            }
        }


class Token(BaseModel):
    """OAuth 2.0 token response."""

    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration in seconds")
    refresh_token: Optional[str] = Field(None, description="Refresh token")
    scope: Optional[str] = Field(None, description="Token scope")


class TokenData(BaseModel):
    """Data extracted from JWT token."""

    user_id: str
    username: str
    tenant_id: Optional[str] = None
    roles: Set[str] = Field(default_factory=set)
    permissions: Set[str] = Field(default_factory=set)
    exp: Optional[datetime] = None


class LoginRequest(BaseModel):
    """Login request payload."""

    username: str = Field(..., description="Username")
    password: str = Field(..., description="Password")
    mfa_code: Optional[str] = Field(None, description="MFA code (if MFA enabled)")


class OAuth2Config(BaseModel):
    """OAuth 2.0 provider configuration."""

    provider_name: str = Field(..., description="Provider name (e.g., 'google', 'azure')")
    client_id: str = Field(..., description="OAuth 2.0 client ID")
    client_secret: str = Field(..., description="OAuth 2.0 client secret")

    authorization_endpoint: str = Field(..., description="Authorization endpoint URL")
    token_endpoint: str = Field(..., description="Token endpoint URL")
    userinfo_endpoint: str = Field(..., description="User info endpoint URL")

    redirect_uri: str = Field(..., description="Redirect URI after authentication")
    scope: str = Field(default="openid profile email", description="OAuth 2.0 scope")


class LDAPConfig(BaseModel):
    """LDAP/Active Directory configuration."""

    server_uri: str = Field(..., description="LDAP server URI (e.g., ldap://ldap.example.com)")
    bind_dn: str = Field(..., description="Bind DN for LDAP connection")
    bind_password: str = Field(..., description="Bind password")

    user_search_base: str = Field(..., description="Base DN for user search")
    user_search_filter: str = Field(default="(uid={username})", description="LDAP search filter")

    group_search_base: Optional[str] = Field(None, description="Base DN for group search")
    group_search_filter: Optional[str] = Field(default="(member={user_dn})", description="Group search filter")

    # Attribute mappings
    username_attribute: str = Field(default="uid", description="Username attribute")
    email_attribute: str = Field(default="mail", description="Email attribute")
    name_attribute: str = Field(default="cn", description="Full name attribute")

    use_ssl: bool = Field(default=True, description="Use SSL/TLS")
    timeout: int = Field(default=10, description="Connection timeout in seconds")


class MFASetupResponse(BaseModel):
    """MFA setup response with QR code data."""

    secret: str = Field(..., description="TOTP secret")
    qr_code_uri: str = Field(..., description="QR code URI for authenticator apps")
    backup_codes: list[str] = Field(..., description="Backup codes for MFA recovery")


__all__ = [
    "User",
    "Token",
    "TokenData",
    "LoginRequest",
    "OAuth2Config",
    "LDAPConfig",
    "MFASetupResponse",
    "UserRole",
    "AuthProvider",
]
