"""Authentication and authorization components.

This package provides:
- OAuth 2.0/OpenID Connect SSO
- Multi-factor authentication (MFA)
- LDAP/Active Directory integration
- JWT token management
- FastAPI authentication dependencies
- gRPC authentication interceptors
"""

from .dependencies import (
    get_current_user,
    get_current_active_user,
    require_permission,
    require_role,
)
from .jwt_handler import JWTHandler, create_access_token, verify_token
from .models import User, Token, TokenData
from .oauth2 import OAuth2Handler
from .mfa import MFAHandler
from .ldap import LDAPHandler
from .grpc_interceptor import (
    AuthenticationInterceptor,
    RBACInterceptor,
    create_authenticated_server,
)

__all__ = [
    "get_current_user",
    "get_current_active_user",
    "require_permission",
    "require_role",
    "JWTHandler",
    "create_access_token",
    "verify_token",
    "User",
    "Token",
    "TokenData",
    "OAuth2Handler",
    "MFAHandler",
    "LDAPHandler",
    "AuthenticationInterceptor",
    "RBACInterceptor",
    "create_authenticated_server",
]
