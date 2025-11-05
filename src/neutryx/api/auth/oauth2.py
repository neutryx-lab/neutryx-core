"""OAuth 2.0 and OpenID Connect integration."""

from __future__ import annotations

import secrets
from typing import Any, Dict, Optional
from urllib.parse import urlencode

import httpx

from .models import OAuth2Config, User, AuthProvider


class OAuth2Handler:
    """Handle OAuth 2.0 / OpenID Connect flows."""

    def __init__(self, config: OAuth2Config):
        """Initialize OAuth2 handler.

        Args:
            config: OAuth2Config with provider settings
        """
        self.config = config
        self._http_client = httpx.AsyncClient(timeout=30.0)

    async def get_authorization_url(self, state: Optional[str] = None) -> tuple[str, str]:
        """Get authorization URL for OAuth 2.0 authorization code flow.

        Args:
            state: Optional state parameter for CSRF protection

        Returns:
            Tuple of (authorization_url, state)
        """
        if state is None:
            state = secrets.token_urlsafe(32)

        params = {
            "client_id": self.config.client_id,
            "redirect_uri": self.config.redirect_uri,
            "response_type": "code",
            "scope": self.config.scope,
            "state": state,
        }

        auth_url = f"{self.config.authorization_endpoint}?{urlencode(params)}"
        return auth_url, state

    async def exchange_code_for_token(self, code: str) -> Dict[str, Any]:
        """Exchange authorization code for access token.

        Args:
            code: Authorization code from OAuth provider

        Returns:
            Token response dictionary

        Raises:
            httpx.HTTPError: If token exchange fails
        """
        token_data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.config.redirect_uri,
            "client_id": self.config.client_id,
            "client_secret": self.config.client_secret,
        }

        response = await self._http_client.post(
            self.config.token_endpoint,
            data=token_data,
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()

        return response.json()

    async def get_user_info(self, access_token: str) -> Dict[str, Any]:
        """Get user information from OAuth provider.

        Args:
            access_token: Access token from provider

        Returns:
            User information dictionary

        Raises:
            httpx.HTTPError: If user info request fails
        """
        response = await self._http_client.get(
            self.config.userinfo_endpoint,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/json",
            },
        )
        response.raise_for_status()

        return response.json()

    async def refresh_access_token(self, refresh_token: str) -> Dict[str, Any]:
        """Refresh access token using refresh token.

        Args:
            refresh_token: Refresh token from provider

        Returns:
            New token response dictionary

        Raises:
            httpx.HTTPError: If token refresh fails
        """
        token_data = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": self.config.client_id,
            "client_secret": self.config.client_secret,
        }

        response = await self._http_client.post(
            self.config.token_endpoint,
            data=token_data,
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()

        return response.json()

    async def authenticate_user(self, code: str) -> User:
        """Complete OAuth 2.0 authentication flow.

        Exchange authorization code for tokens and create User object.

        Args:
            code: Authorization code from OAuth provider

        Returns:
            User object with OAuth provider information

        Raises:
            httpx.HTTPError: If authentication fails
        """
        # Exchange code for token
        token_response = await self.exchange_code_for_token(code)
        access_token = token_response.get("access_token")

        if not access_token:
            raise ValueError("No access token in response")

        # Get user info
        user_info = await self.get_user_info(access_token)

        # Map provider user info to our User model
        user = self._map_user_info_to_user(user_info)

        return user

    def _map_user_info_to_user(self, user_info: Dict[str, Any]) -> User:
        """Map OAuth provider user info to User model.

        Args:
            user_info: User info from OAuth provider

        Returns:
            User object
        """
        # Common OpenID Connect claims
        user_id = user_info.get("sub") or user_info.get("id")
        email = user_info.get("email")
        username = email or user_id

        # Provider-specific mappings
        if self.config.provider_name.lower() == "google":
            full_name = user_info.get("name")
        elif self.config.provider_name.lower() == "azure":
            full_name = user_info.get("name") or user_info.get("displayName")
        elif self.config.provider_name.lower() == "github":
            full_name = user_info.get("name")
            username = user_info.get("login") or email or user_id
        else:
            # Generic mapping
            full_name = user_info.get("name") or user_info.get("displayName")

        return User(
            user_id=str(user_id),
            username=str(username),
            email=email,
            full_name=full_name,
            auth_provider=AuthProvider.OAUTH2,
            disabled=False,
        )

    async def close(self):
        """Close HTTP client."""
        await self._http_client.aclose()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Predefined OAuth2 configurations for common providers
def get_google_oauth_config(
    client_id: str,
    client_secret: str,
    redirect_uri: str,
) -> OAuth2Config:
    """Get OAuth2Config for Google.

    Args:
        client_id: Google OAuth 2.0 client ID
        client_secret: Google OAuth 2.0 client secret
        redirect_uri: Redirect URI registered with Google

    Returns:
        OAuth2Config for Google
    """
    return OAuth2Config(
        provider_name="google",
        client_id=client_id,
        client_secret=client_secret,
        authorization_endpoint="https://accounts.google.com/o/oauth2/v2/auth",
        token_endpoint="https://oauth2.googleapis.com/token",
        userinfo_endpoint="https://www.googleapis.com/oauth2/v2/userinfo",
        redirect_uri=redirect_uri,
        scope="openid profile email",
    )


def get_azure_oauth_config(
    tenant_id: str,
    client_id: str,
    client_secret: str,
    redirect_uri: str,
) -> OAuth2Config:
    """Get OAuth2Config for Azure AD.

    Args:
        tenant_id: Azure AD tenant ID
        client_id: Azure AD application client ID
        client_secret: Azure AD application client secret
        redirect_uri: Redirect URI registered with Azure AD

    Returns:
        OAuth2Config for Azure AD
    """
    base_url = f"https://login.microsoftonline.com/{tenant_id}"

    return OAuth2Config(
        provider_name="azure",
        client_id=client_id,
        client_secret=client_secret,
        authorization_endpoint=f"{base_url}/oauth2/v2.0/authorize",
        token_endpoint=f"{base_url}/oauth2/v2.0/token",
        userinfo_endpoint="https://graph.microsoft.com/v1.0/me",
        redirect_uri=redirect_uri,
        scope="openid profile email User.Read",
    )


def get_github_oauth_config(
    client_id: str,
    client_secret: str,
    redirect_uri: str,
) -> OAuth2Config:
    """Get OAuth2Config for GitHub.

    Args:
        client_id: GitHub OAuth app client ID
        client_secret: GitHub OAuth app client secret
        redirect_uri: Redirect URI registered with GitHub

    Returns:
        OAuth2Config for GitHub
    """
    return OAuth2Config(
        provider_name="github",
        client_id=client_id,
        client_secret=client_secret,
        authorization_endpoint="https://github.com/login/oauth/authorize",
        token_endpoint="https://github.com/login/oauth/access_token",
        userinfo_endpoint="https://api.github.com/user",
        redirect_uri=redirect_uri,
        scope="read:user user:email",
    )


__all__ = [
    "OAuth2Handler",
    "get_google_oauth_config",
    "get_azure_oauth_config",
    "get_github_oauth_config",
]
