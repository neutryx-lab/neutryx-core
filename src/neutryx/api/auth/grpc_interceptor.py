"""gRPC authentication interceptor."""

from __future__ import annotations

from typing import Callable, Any

import grpc
from jose import JWTError

from .jwt_handler import verify_token
from .dependencies import get_user_from_store
from .models import User


try:  # pragma: no cover - fallback for environments without grpc.aio
    from grpc.aio import ServerInterceptor as _BaseServerInterceptor
except ImportError:  # pragma: no cover - synchronous fallback
    from grpc import ServerInterceptor as _BaseServerInterceptor


class AuthenticationInterceptor(_BaseServerInterceptor):
    """gRPC server interceptor for JWT authentication.

    Intercepts all gRPC calls and validates JWT tokens from metadata.
    """

    def __init__(self, exempt_methods: set[str] | None = None):
        """Initialize authentication interceptor.

        Args:
            exempt_methods: Set of method names to exempt from authentication
        """
        self.exempt_methods = exempt_methods or set()

    async def intercept_service(self, continuation: Callable, handler_call_details: grpc.HandlerCallDetails):
        """Intercept gRPC service call.

        Args:
            continuation: Function to invoke next interceptor or handler
            handler_call_details: Details about the RPC call

        Returns:
            RPC handler or error
        """
        # Extract method name
        method = handler_call_details.method

        # Check if method is exempt from authentication
        if method in self.exempt_methods:
            return await continuation(handler_call_details)

        # Extract metadata
        metadata = dict(handler_call_details.invocation_metadata)

        # Get authorization token from metadata
        auth_header = metadata.get("authorization")

        if not auth_header:
            return self._unauthenticated_handler()

        # Extract token (format: "Bearer <token>")
        try:
            scheme, token = auth_header.split(" ", 1)
            if scheme.lower() != "bearer":
                return self._unauthenticated_handler()
        except ValueError:
            return self._unauthenticated_handler()

        # Verify token
        try:
            token_data = verify_token(token, token_type="access")

            # Get user from store
            user = get_user_from_store(token_data.user_id)

            if user is None:
                # Create minimal user from token data
                user = User(
                    user_id=token_data.user_id,
                    username=token_data.username,
                    tenant_id=token_data.tenant_id,
                    roles=token_data.roles,
                    permissions=token_data.permissions,
                )

            # Check if user is disabled
            if user.disabled:
                return self._permission_denied_handler("User account is disabled")

            # Store user in context (for access in RPC handlers)
            # This would typically be done via context propagation
            # For now, we just validate the token

            return await continuation(handler_call_details)

        except JWTError as e:
            return self._unauthenticated_handler(str(e))

    def _unauthenticated_handler(self, message: str = "Missing or invalid authentication token"):
        """Return unauthenticated error handler.

        Args:
            message: Error message

        Returns:
            RPC handler that returns UNAUTHENTICATED status
        """

        async def handler(request, context):
            await context.abort(grpc.StatusCode.UNAUTHENTICATED, message)

        return grpc.unary_unary_rpc_method_handler(handler)

    def _permission_denied_handler(self, message: str = "Permission denied"):
        """Return permission denied error handler.

        Args:
            message: Error message

        Returns:
            RPC handler that returns PERMISSION_DENIED status
        """

        async def handler(request, context):
            await context.abort(grpc.StatusCode.PERMISSION_DENIED, message)

        return grpc.unary_unary_rpc_method_handler(handler)


class RBACInterceptor(_BaseServerInterceptor):
    """gRPC server interceptor for role-based access control.

    Intercepts gRPC calls and validates user roles/permissions.
    """

    def __init__(self, method_permissions: dict[str, str] | None = None):
        """Initialize RBAC interceptor.

        Args:
            method_permissions: Mapping of method names to required permissions
                                Example: {"PriceVanilla": "pricing.vanilla"}
        """
        self.method_permissions = method_permissions or {}

    async def intercept_service(self, continuation: Callable, handler_call_details: grpc.HandlerCallDetails):
        """Intercept gRPC service call.

        Args:
            continuation: Function to invoke next interceptor or handler
            handler_call_details: Details about the RPC call

        Returns:
            RPC handler or error
        """
        method = handler_call_details.method

        # Check if method requires specific permission
        required_permission = self.method_permissions.get(method)

        if not required_permission:
            # No permission required
            return await continuation(handler_call_details)

        # Extract metadata and get user (assumes AuthenticationInterceptor ran first)
        metadata = dict(handler_call_details.invocation_metadata)
        auth_header = metadata.get("authorization")

        if not auth_header:
            return self._permission_denied_handler("Authentication required")

        try:
            scheme, token = auth_header.split(" ", 1)
            token_data = verify_token(token, token_type="access")

            # Check if user has required permission
            if required_permission not in token_data.permissions:
                # Check if any of user's roles grant the permission
                # In a real implementation, this would check RBAC manager
                return self._permission_denied_handler(
                    f"Missing required permission: {required_permission}"
                )

            return await continuation(handler_call_details)

        except (ValueError, JWTError):
            return self._permission_denied_handler("Invalid authentication token")

    def _permission_denied_handler(self, message: str):
        """Return permission denied error handler.

        Args:
            message: Error message

        Returns:
            RPC handler that returns PERMISSION_DENIED status
        """

        async def handler(request, context):
            await context.abort(grpc.StatusCode.PERMISSION_DENIED, message)

        return grpc.unary_unary_rpc_method_handler(handler)


def create_authenticated_server(
    *,
    enable_auth: bool = True,
    exempt_methods: set[str] | None = None,
    method_permissions: dict[str, str] | None = None,
) -> grpc.aio.Server:
    """Create gRPC server with authentication interceptors.

    Args:
        enable_auth: Whether to enable authentication
        exempt_methods: Methods to exempt from authentication
        method_permissions: Method-to-permission mapping for RBAC

    Returns:
        gRPC server with authentication interceptors
    """
    interceptors = []

    if enable_auth:
        # Add authentication interceptor
        auth_interceptor = AuthenticationInterceptor(exempt_methods=exempt_methods)
        interceptors.append(auth_interceptor)

        # Add RBAC interceptor if permissions are defined
        if method_permissions:
            rbac_interceptor = RBACInterceptor(method_permissions=method_permissions)
            interceptors.append(rbac_interceptor)

    server = grpc.aio.server(interceptors=interceptors)
    return server


__all__ = [
    "AuthenticationInterceptor",
    "RBACInterceptor",
    "create_authenticated_server",
]
