"""Authentication endpoints for FastAPI."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.security import OAuth2PasswordRequestForm

from .dependencies import (
    get_current_active_user,
    get_current_user,
    add_user_to_store,
    authenticate_local_user,
)
from .jwt_handler import JWTHandler
from .models import (
    User,
    Token,
    LoginRequest,
    OAuth2Config,
    LDAPConfig,
    MFASetupResponse,
)
from .oauth2 import OAuth2Handler
from .mfa import MFAHandler
from .ldap import LDAPHandler


# Create router
router = APIRouter(prefix="/auth", tags=["authentication"])


# Global handlers (should be configured from app state in production)
_jwt_handler = JWTHandler()
_mfa_handler = MFAHandler()
_oauth_handlers: dict[str, OAuth2Handler] = {}
_ldap_handler: Optional[LDAPHandler] = None


def configure_oauth_provider(provider_name: str, config: OAuth2Config):
    """Configure OAuth2 provider.

    Args:
        provider_name: Provider name
        config: OAuth2 configuration
    """
    _oauth_handlers[provider_name] = OAuth2Handler(config)


def configure_ldap(config: LDAPConfig):
    """Configure LDAP handler.

    Args:
        config: LDAP configuration
    """
    global _ldap_handler
    _ldap_handler = LDAPHandler(config)


@router.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()) -> Token:
    """Login with username and password (OAuth2 password flow).

    Returns access token for subsequent API calls.
    """
    # Separate MFA code if provided in password field
    raw_password = form_data.password
    password = raw_password
    provided_mfa = None
    if ":" in raw_password:
        password, provided_mfa = raw_password.rsplit(":", 1)

    # Try LDAP authentication first if configured
    user = None
    if _ldap_handler:
        try:
            user = _ldap_handler.authenticate_user(form_data.username, password)
        except Exception:
            user = None  # Fall through to local auth

    # Local authentication fallback using credential store
    if not user:
        user = authenticate_local_user(form_data.username, password)

    # Local authentication fallback (for demo - in production use proper user db)
    if not user:
        local_user = get_user_by_username(form_data.username)
        if local_user and verify_local_user_password(local_user, password):
            user = local_user

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check MFA if enabled
    if user.mfa_enabled:
        # In a real implementation, this would be a two-step process
        # For now, we'll require MFA code in the password field format: password:mfa_code
        if provided_mfa is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="MFA code required. Format: password:mfa_code",
            )

        if not await _mfa_handler.verify_mfa(user, provided_mfa):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid MFA code",
            )

    # Store user and create tokens
    user.last_login = datetime.utcnow()
    add_user_to_store(user)

    access_token = _jwt_handler.create_access_token(user)
    refresh_token = _jwt_handler.create_refresh_token(user)

    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=_jwt_handler.access_token_expire_minutes * 60,
        refresh_token=refresh_token,
    )


@router.post("/refresh", response_model=Token)
async def refresh_token(refresh_token: str) -> Token:
    """Refresh access token using refresh token."""
    try:
        token_data = _jwt_handler.verify_token(refresh_token, token_type="refresh")

        # Get user from store
        from .dependencies import get_user_from_store

        user = get_user_from_store(token_data.user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
            )

        # Create new tokens
        new_access_token = _jwt_handler.create_access_token(user)
        new_refresh_token = _jwt_handler.create_refresh_token(user)

        return Token(
            access_token=new_access_token,
            token_type="bearer",
            expires_in=_jwt_handler.access_token_expire_minutes * 60,
            refresh_token=new_refresh_token,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid refresh token: {e}",
        )


@router.get("/me", response_model=User)
async def get_current_user_info(
    current_user: User = Depends(get_current_active_user),
) -> User:
    """Get current authenticated user information."""
    return current_user


# OAuth2 endpoints
@router.get("/oauth/{provider}/authorize")
async def oauth_authorize(provider: str) -> dict:
    """Get OAuth2 authorization URL.

    Args:
        provider: OAuth provider name (google, azure, github)

    Returns:
        Authorization URL and state
    """
    if provider not in _oauth_handlers:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"OAuth provider '{provider}' not configured",
        )

    handler = _oauth_handlers[provider]
    auth_url, state = await handler.get_authorization_url()

    return {
        "authorization_url": auth_url,
        "state": state,
    }


@router.get("/oauth/{provider}/callback")
async def oauth_callback(
    provider: str,
    code: str = Query(..., description="Authorization code"),
    state: str = Query(..., description="State parameter"),
) -> Token:
    """OAuth2 callback endpoint.

    Args:
        provider: OAuth provider name
        code: Authorization code from provider
        state: State parameter for CSRF protection

    Returns:
        Access token
    """
    if provider not in _oauth_handlers:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"OAuth provider '{provider}' not configured",
        )

    handler = _oauth_handlers[provider]

    try:
        # Authenticate user via OAuth
        user = await handler.authenticate_user(code)

        # Store user
        add_user_to_store(user)

        # Create tokens
        access_token = _jwt_handler.create_access_token(user)
        refresh_token = _jwt_handler.create_refresh_token(user)

        return Token(
            access_token=access_token,
            token_type="bearer",
            expires_in=_jwt_handler.access_token_expire_minutes * 60,
            refresh_token=refresh_token,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"OAuth authentication failed: {e}",
        )


# MFA endpoints
@router.post("/mfa/setup", response_model=MFASetupResponse)
async def setup_mfa(
    current_user: User = Depends(get_current_active_user),
) -> MFASetupResponse:
    """Set up MFA for current user.

    Returns QR code and backup codes.
    """
    if current_user.mfa_enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="MFA is already enabled",
        )

    return await _mfa_handler.setup_mfa(current_user)


@router.post("/mfa/enable")
async def enable_mfa(
    secret: str,
    verification_code: str,
    current_user: User = Depends(get_current_active_user),
) -> dict:
    """Enable MFA after verifying initial code.

    Args:
        secret: TOTP secret from setup
        verification_code: 6-digit code from authenticator

    Returns:
        Success status
    """
    success = await _mfa_handler.enable_mfa(current_user, secret, verification_code)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid verification code",
        )

    return {"status": "MFA enabled successfully"}


@router.post("/mfa/disable")
async def disable_mfa(
    verification_code: str,
    current_user: User = Depends(get_current_active_user),
) -> dict:
    """Disable MFA for current user.

    Args:
        verification_code: 6-digit code from authenticator

    Returns:
        Success status
    """
    success = await _mfa_handler.disable_mfa(current_user, verification_code)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid verification code or MFA not enabled",
        )

    return {"status": "MFA disabled successfully"}


@router.post("/mfa/verify")
async def verify_mfa_code(
    code: str,
    current_user: User = Depends(get_current_active_user),
) -> dict:
    """Verify MFA code.

    Args:
        code: 6-digit code from authenticator

    Returns:
        Verification status
    """
    valid = await _mfa_handler.verify_mfa(current_user, code)

    return {
        "valid": valid,
        "message": "MFA code is valid" if valid else "Invalid MFA code",
    }


# LDAP endpoints
@router.get("/ldap/user/{username}")
async def get_ldap_user(
    username: str,
    current_user: User = Depends(get_current_active_user),
) -> User:
    """Get user information from LDAP.

    Args:
        username: LDAP username

    Returns:
        User object
    """
    if not _ldap_handler:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="LDAP not configured",
        )

    # Require admin role to query LDAP
    if "admin" not in current_user.roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required",
        )

    user = _ldap_handler.get_user_by_username(username)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User '{username}' not found in LDAP",
        )

    return user


@router.post("/ldap/sync")
async def sync_ldap_users(
    current_user: User = Depends(get_current_active_user),
) -> dict:
    """Sync users from LDAP to local database.

    Returns:
        Sync statistics
    """
    if not _ldap_handler:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="LDAP not configured",
        )

    # Require admin role
    if "admin" not in current_user.roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required",
        )

    try:
        users = _ldap_handler.sync_users()

        # Store synced users
        for user in users:
            add_user_to_store(user)

        return {
            "status": "success",
            "users_synced": len(users),
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LDAP sync failed: {e}",
        )


__all__ = [
    "router",
    "configure_oauth_provider",
    "configure_ldap",
]
