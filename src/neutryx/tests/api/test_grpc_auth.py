"""Authentication and RBAC tests for the gRPC pricing service."""

from __future__ import annotations

import sys
import types

import grpc
import pytest
import pytest_asyncio
from google.protobuf.struct_pb2 import Struct

if "prometheus_client" not in sys.modules:  # pragma: no cover - test shim
    prometheus_stub = types.ModuleType("prometheus_client")

    class _DummyCollector:
        def __init__(self, *args, **kwargs):
            self._labels = kwargs.get("labelnames", ())

        def labels(self, *args, **kwargs):
            return self

        def observe(self, *args, **kwargs):
            return None

        def inc(self, *args, **kwargs):
            return None

    prometheus_stub.CONTENT_TYPE_LATEST = "text/plain"
    prometheus_stub.Counter = _DummyCollector
    prometheus_stub.Histogram = _DummyCollector
    prometheus_stub.REGISTRY = types.SimpleNamespace(_names_to_collectors={})

    def _generate_latest(*args, **kwargs):
        return b""

    prometheus_stub.generate_latest = _generate_latest
    sys.modules["prometheus_client"] = prometheus_stub

from neutryx.api.auth.grpc_interceptor import create_authenticated_server
from neutryx.api.auth.jwt_handler import JWTHandler
from neutryx.api.auth.models import User
from neutryx.api.grpc import SERVICE_NAME, add_servicer

FULL_METHOD_NAME = f"/{SERVICE_NAME}/PriceVanilla"


@pytest_asyncio.fixture
async def secured_pricing_server():
    server = create_authenticated_server(
        enable_auth=True,
        exempt_methods=None,
        method_permissions={FULL_METHOD_NAME: "pricing.vanilla:read"},
    )
    registered_methods = add_servicer(server)
    assert FULL_METHOD_NAME in registered_methods
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    try:
        yield f"127.0.0.1:{port}"
    finally:
        await server.stop(grace=None)


@pytest.mark.asyncio
async def test_request_without_token_is_rejected(secured_pricing_server):
    async with grpc.aio.insecure_channel(secured_pricing_server) as channel:
        stub = channel.unary_unary(
            FULL_METHOD_NAME,
            request_serializer=Struct.SerializeToString,
            response_deserializer=Struct.FromString,
        )
        with pytest.raises(grpc.aio.AioRpcError) as excinfo:
            await stub(Struct())
        assert excinfo.value.code() == grpc.StatusCode.UNAUTHENTICATED


@pytest.mark.asyncio
async def test_request_without_required_permission_is_rejected(secured_pricing_server):
    handler = JWTHandler()
    user = User(
        user_id="user-123",
        username="alice",
        permissions=set(),
        roles={"viewer"},
    )
    token = handler.create_access_token(user)

    request = Struct()
    request.update({
        "spot": 100.0,
        "strike": 100.0,
        "maturity": 1.0,
        "volatility": 0.2,
    })

    async with grpc.aio.insecure_channel(secured_pricing_server) as channel:
        stub = channel.unary_unary(
            FULL_METHOD_NAME,
            request_serializer=Struct.SerializeToString,
            response_deserializer=Struct.FromString,
        )
        with pytest.raises(grpc.aio.AioRpcError) as excinfo:
            await stub(request, metadata=(("authorization", f"Bearer {token}"),))
        assert excinfo.value.code() == grpc.StatusCode.PERMISSION_DENIED
        assert "Missing required permission" in excinfo.value.details()
