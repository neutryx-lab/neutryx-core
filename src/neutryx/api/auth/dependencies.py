"""FastAPI dependencies for authentication and authorization."""

from __future__ import annotations

from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError
import warnings

# Workaround for bcrypt 4.0+ compatibility with passlib
# Patch bcrypt.hashpw to handle >72 byte passwords before passlib loads it
try:
    import bcrypt as _bcrypt_module

    _original_hashpw = _bcrypt_module.hashpw

    def _patched_hashpw(password: bytes, salt: bytes) -> bytes:
        """Wrapper for bcrypt.hashpw that truncates long passwords."""
        if len(password) > 72:
            password = password[:72]
        return _original_hashpw(password, salt)

    _bcrypt_module.hashpw = _patched_hashpw
except ImportError:
    pass

from passlib.context import CryptContext

from neutryx.infrastructure.governance.rbac import RBACManager

from .jwt_handler import verify_token
from .models import User, TokenData


# HTTP Bearer token authentication
security = HTTPBearer()


# In-memory user storage (for demo purposes)
# In production, this would be a database
_user_store: dict[str, User] = {}
_user_store_by_username: dict[str, str] = {}
_credentials_store: dict[str, str] = {}


# Initialize password context
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    _password_context = CryptContext(
        schemes=["bcrypt"],
        deprecated="auto",
    )


# RBAC manager (should be injected from app state in production)
_rbac_manager: Optional[RBACManager] = None


def set_rbac_manager(rbac_manager: RBACManager):
    """Set global RBAC manager instance.

    Args:
        rbac_manager: RBAC manager instance
    """
    global _rbac_manager
    _rbac_manager = rbac_manager


def get_user_from_store(user_id: str) -> Optional[User]:
    """Get user from in-memory store.

    Args:
        user_id: User ID

    Returns:
        User object if found, None otherwise
    """
    return _user_store.get(user_id)


def get_user_by_username(username: str) -> Optional[User]:
    """Get user from in-memory store by username."""

    user_id = _user_store_by_username.get(username)
    if not user_id:
        return None
    return _user_store.get(user_id)


def add_user_to_store(user: User):
    """Add user to in-memory store.

    Args:
        user: User object
    """
    _user_store[user.user_id] = user
    _user_store_by_username[user.username] = user.user_id


def clear_user_store() -> None:
    """Clear all in-memory user stores.

    Used primarily for testing to ensure test isolation.
    """
    global _user_store, _user_store_by_username, _credentials_store
    _user_store.clear()
    _user_store_by_username.clear()
    _credentials_store.clear()


def register_local_user(user: User, password: str) -> None:
    """Register a local user with hashed credentials."""

    add_user_to_store(user)
    hashed_password = _password_context.hash(password)
    _credentials_store[user.username] = hashed_password


def authenticate_local_user(username: str, password: str) -> Optional[User]:
    """Authenticate a local user against the credential store."""

    hashed_password = _credentials_store.get(username)
    if not hashed_password:
        return None

    try:
        if not _password_context.verify(password, hashed_password):
            return None
    except ValueError:
        return None

    return get_user_by_username(username)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> User:
    """Get current authenticated user from JWT token.

    Args:
        credentials: HTTP authorization credentials

    Returns:
        Current user

    Raises:
        HTTPException: If authentication fails
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        token = credentials.credentials
        token_data: TokenData = verify_token(token, token_type="access")

        # Get user from store
        user = get_user_from_store(token_data.user_id)

        if user is None:
            # Create minimal user from token data if not in store
            user = User(
                user_id=token_data.user_id,
                username=token_data.username,
                tenant_id=token_data.tenant_id,
                roles=token_data.roles,
                permissions=token_data.permissions,
            )

        return user

    except JWTError:
        raise credentials_exception


async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """Get current active user (not disabled).

    Args:
        current_user: Current authenticated user

    Returns:
        Current active user

    Raises:
        HTTPException: If user is disabled
    """
    if current_user.disabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled",
        )
    return current_user


def require_role(required_role: str):
    """Dependency factory to require specific role.

    Args:
        required_role: Required role name

    Returns:
        Dependency function
    """

    async def role_checker(
        current_user: User = Depends(get_current_active_user),
    ) -> User:
        """Check if user has required role."""
        if required_role not in current_user.roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"User does not have required role: {required_role}",
            )
        return current_user

    return role_checker


def require_permission(required_permission: str):
    """Dependency factory to require specific permission.

    Args:
        required_permission: Required permission

    Returns:
        Dependency function
    """

    async def permission_checker(
        current_user: User = Depends(get_current_active_user),
    ) -> User:
        """Check if user has required permission."""
        # Check direct permissions
        if required_permission in current_user.permissions:
            return current_user

        # Check RBAC manager for role-based permissions
        if _rbac_manager:
            for role in current_user.roles:
                if _rbac_manager.check_permission(
                    current_user.user_id,
                    required_permission,
                    tenant_id=current_user.tenant_id,
                ):
                    return current_user

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"User does not have required permission: {required_permission}",
        )

    return permission_checker


def require_any_role(*roles: str):
    """Dependency factory to require any of the specified roles.

    Args:
        roles: List of acceptable role names

    Returns:
        Dependency function
    """

    async def any_role_checker(
        current_user: User = Depends(get_current_active_user),
    ) -> User:
        """Check if user has any of the required roles."""
        if not any(role in current_user.roles for role in roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"User must have one of these roles: {', '.join(roles)}",
            )
        return current_user

    return any_role_checker


def require_tenant(tenant_id: str):
    """Dependency factory to require specific tenant.

    Args:
        tenant_id: Required tenant ID

    Returns:
        Dependency function
    """

    async def tenant_checker(
        current_user: User = Depends(get_current_active_user),
    ) -> User:
        """Check if user belongs to required tenant."""
        if current_user.tenant_id != tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"User does not belong to tenant: {tenant_id}",
            )
        return current_user

    return tenant_checker


__all__ = [
    "security",
    "get_current_user",
    "get_current_active_user",
    "require_role",
    "require_permission",
    "require_any_role",
    "require_tenant",
    "set_rbac_manager",
    "get_user_from_store",
    "get_user_by_username",
    "add_user_to_store",
    "clear_user_store",
    "register_local_user",
    "authenticate_local_user",
]
