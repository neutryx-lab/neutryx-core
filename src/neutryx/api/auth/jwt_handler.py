"""JWT token handling for authentication."""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Optional

from jose import JWTError, jwt
from jose.constants import ALGORITHMS

from .models import TokenData, User


# JWT Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = ALGORITHMS.HS256
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("JWT_REFRESH_TOKEN_EXPIRE_DAYS", "7"))


class JWTHandler:
    """Handle JWT token creation and verification."""

    def __init__(
        self,
        secret_key: str = SECRET_KEY,
        algorithm: str = ALGORITHM,
        access_token_expire_minutes: int = ACCESS_TOKEN_EXPIRE_MINUTES,
        refresh_token_expire_days: int = REFRESH_TOKEN_EXPIRE_DAYS,
    ):
        """Initialize JWT handler.

        Args:
            secret_key: Secret key for JWT signing
            algorithm: JWT algorithm (default: HS256)
            access_token_expire_minutes: Access token expiration in minutes
            refresh_token_expire_days: Refresh token expiration in days
        """
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        self.refresh_token_expire_days = refresh_token_expire_days

    def create_access_token(
        self,
        user: User,
        expires_delta: Optional[timedelta] = None,
    ) -> str:
        """Create JWT access token for user.

        Args:
            user: User object
            expires_delta: Optional custom expiration time

        Returns:
            Encoded JWT token string
        """
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)

        to_encode = {
            "sub": user.user_id,
            "username": user.username,
            "tenant_id": user.tenant_id,
            "roles": list(user.roles),
            "permissions": list(user.permissions),
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access",
        }

        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt

    def create_refresh_token(self, user: User) -> str:
        """Create JWT refresh token for user.

        Args:
            user: User object

        Returns:
            Encoded JWT refresh token string
        """
        expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)

        to_encode = {
            "sub": user.user_id,
            "username": user.username,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh",
        }

        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt

    def verify_token(self, token: str, token_type: str = "access") -> TokenData:
        """Verify JWT token and extract data.

        Args:
            token: JWT token string
            token_type: Expected token type ('access' or 'refresh')

        Returns:
            TokenData object with user information

        Raises:
            JWTError: If token is invalid or expired
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            # Verify token type
            if payload.get("type") != token_type:
                raise JWTError(f"Invalid token type. Expected '{token_type}'")

            user_id: str = payload.get("sub")
            username: str = payload.get("username")

            if user_id is None or username is None:
                raise JWTError("Invalid token payload")

            token_data = TokenData(
                user_id=user_id,
                username=username,
                tenant_id=payload.get("tenant_id"),
                roles=set(payload.get("roles", [])),
                permissions=set(payload.get("permissions", [])),
                exp=datetime.fromtimestamp(payload.get("exp")) if payload.get("exp") else None,
            )

            return token_data

        except JWTError as e:
            raise JWTError(f"Token verification failed: {e}") from e

    def decode_token_without_verification(self, token: str) -> dict:
        """Decode token without verification (for debugging).

        Args:
            token: JWT token string

        Returns:
            Decoded payload dictionary
        """
        return jwt.decode(token, options={"verify_signature": False})


# Convenience functions for backward compatibility
_default_handler = JWTHandler()


def create_access_token(user: User, expires_delta: Optional[timedelta] = None) -> str:
    """Create access token using default handler."""
    return _default_handler.create_access_token(user, expires_delta)


def create_refresh_token(user: User) -> str:
    """Create refresh token using default handler."""
    return _default_handler.create_refresh_token(user)


def verify_token(token: str, token_type: str = "access") -> TokenData:
    """Verify token using default handler."""
    return _default_handler.verify_token(token, token_type)


__all__ = [
    "JWTHandler",
    "create_access_token",
    "create_refresh_token",
    "verify_token",
    "SECRET_KEY",
    "ALGORITHM",
    "ACCESS_TOKEN_EXPIRE_MINUTES",
    "REFRESH_TOKEN_EXPIRE_DAYS",
]
