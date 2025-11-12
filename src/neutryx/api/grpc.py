"""gRPC service definitions for Neutryx pricing workflows."""
from __future__ import annotations

import asyncio
from collections.abc import Iterable, Mapping
from typing import Any

import grpc
import jax
import jax.numpy as jnp
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Struct

from neutryx.core.engine import MCConfig, price_vanilla_mc
from neutryx.valuations.xva.cva import cva
from neutryx.valuations.xva.fva import fva
from neutryx.valuations.xva.mva import mva
from neutryx.infrastructure.observability import setup_observability
from neutryx.api.auth import create_authenticated_server

SERVICE_NAME = "neutryx.api.PricingService"


def _struct_to_dict(message: Struct) -> Mapping[str, Any]:
    return json_format.MessageToDict(message, preserving_proto_field_name=True)


def _sequence(values: Any, *, match: int) -> jnp.ndarray:
    if isinstance(values, (int, float)):
        return jnp.full((match,), float(values), dtype=jnp.float32)
    seq = list(values)
    if len(seq) != match:
        raise ValueError("sequence length mismatch")
    return jnp.asarray(seq, dtype=jnp.float32)


class PricingServicer:
    """Implements RPC handlers for pricing workflows."""

    def __init__(self):
        self._observability = setup_observability()
        self._metrics = self._observability.metrics

    async def PriceVanilla(self, request: Struct, context: grpc.aio.ServicerContext) -> Struct:  # noqa: N802
        payload = _struct_to_dict(request)
        with self._metrics.time(
            "pricing.vanilla",
            labels={"channel": "grpc", "product": "vanilla_option"},
            kind="grpc",
        ):
            try:
                mc_cfg = payload.get("mc", {})
                steps = int(mc_cfg.get("steps", payload.get("steps", 64)))
                paths = int(mc_cfg.get("paths", payload.get("paths", 8192)))
                antithetic = bool(mc_cfg.get("antithetic", payload.get("antithetic", False)))
                mc = MCConfig(steps=steps, paths=paths, antithetic=antithetic)
                seed = int(payload.get("seed", 0))
                key = jax.random.PRNGKey(seed)
                price = price_vanilla_mc(
                    key,
                    float(payload["spot"]),
                    float(payload["strike"]),
                    float(payload["maturity"]),
                    float(payload.get("rate", 0.0)),
                    float(payload.get("dividend", 0.0)),
                    float(payload["volatility"]),
                    mc,
                    is_call=bool(payload.get("call", True)),
                )
            except KeyError as exc:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"missing field: {exc.args[0]}")
            except Exception as exc:  # pragma: no cover - defensive
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        response = Struct()
        response.update({"price": float(price)})
        return response

    async def ComputeCVA(self, request: Struct, context: grpc.aio.ServicerContext) -> Struct:  # noqa: N802
        payload = _struct_to_dict(request)
        with self._metrics.time(
            "xva.cva",
            labels={"channel": "grpc", "product": "cva"},
            kind="grpc",
        ):
            try:
                epe = jnp.asarray(payload["epe"], dtype=jnp.float32)
                discount = jnp.asarray(payload["discount"], dtype=jnp.float32)
                pd = jnp.asarray(payload["default_probability"], dtype=jnp.float32)
                if epe.shape[0] != discount.shape[0] or epe.shape[0] != pd.shape[0]:
                    raise ValueError("profiles must share the same length")
                value = cva(epe, discount, pd, lgd=float(payload.get("lgd", 0.6)))
            except KeyError as exc:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"missing field: {exc.args[0]}")
            except Exception as exc:  # pragma: no cover - defensive
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        response = Struct()
        response.update({"cva": float(value)})
        return response

    async def ComputeFVA(self, request: Struct, context: grpc.aio.ServicerContext) -> Struct:  # noqa: N802
        payload = _struct_to_dict(request)
        with self._metrics.time(
            "xva.fva",
            labels={"channel": "grpc", "product": "fva"},
            kind="grpc",
        ):
            try:
                epe = jnp.asarray(payload["epe"], dtype=jnp.float32)
                discount = jnp.asarray(payload["discount"], dtype=jnp.float32)
                spread = _sequence(payload.get("funding_spread"), match=epe.shape[0])
                value = fva(epe, spread, discount)
            except KeyError as exc:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"missing field: {exc.args[0]}")
            except Exception as exc:  # pragma: no cover - defensive
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        response = Struct()
        response.update({"fva": float(value)})
        return response

    async def ComputeMVA(self, request: Struct, context: grpc.aio.ServicerContext) -> Struct:  # noqa: N802
        payload = _struct_to_dict(request)
        with self._metrics.time(
            "xva.mva",
            labels={"channel": "grpc", "product": "mva"},
            kind="grpc",
        ):
            try:
                margin = jnp.asarray(payload["initial_margin"], dtype=jnp.float32)
                discount = jnp.asarray(payload["discount"], dtype=jnp.float32)
                spread = _sequence(payload.get("spread"), match=margin.shape[0])
                value = mva(margin, discount, spread)
            except KeyError as exc:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"missing field: {exc.args[0]}")
            except Exception as exc:  # pragma: no cover - defensive
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        response = Struct()
        response.update({"mva": float(value)})
        return response


def _normalize_method_name(method: str) -> str:
    """Convert a method identifier into its fully-qualified gRPC name."""

    if not method:
        raise ValueError("method name must be a non-empty string")

    if method.startswith("/"):
        return method

    if "/" in method:
        service, rpc = method.split("/", 1)
        if "." not in service:
            service = SERVICE_NAME
        return f"/{service}/{rpc}"

    return f"/{SERVICE_NAME}/{method}"


def _normalize_method_set(methods: Iterable[str] | None) -> set[str] | None:
    if methods is None:
        return None
    return {_normalize_method_name(method) for method in methods}


def _normalize_method_mapping(mapping: Mapping[str, str] | None) -> dict[str, str] | None:
    if mapping is None:
        return None
    return {_normalize_method_name(method): permission for method, permission in mapping.items()}


def add_servicer(server: grpc.aio.Server, servicer: PricingServicer | None = None) -> set[str]:
    servicer = servicer or PricingServicer()
    method_handlers = {
        "PriceVanilla": grpc.unary_unary_rpc_method_handler(
            servicer.PriceVanilla,
            request_deserializer=Struct.FromString,
            response_serializer=lambda msg: msg.SerializeToString(),
        ),
        "ComputeCVA": grpc.unary_unary_rpc_method_handler(
            servicer.ComputeCVA,
            request_deserializer=Struct.FromString,
            response_serializer=lambda msg: msg.SerializeToString(),
        ),
        "ComputeFVA": grpc.unary_unary_rpc_method_handler(
            servicer.ComputeFVA,
            request_deserializer=Struct.FromString,
            response_serializer=lambda msg: msg.SerializeToString(),
        ),
        "ComputeMVA": grpc.unary_unary_rpc_method_handler(
            servicer.ComputeMVA,
            request_deserializer=Struct.FromString,
            response_serializer=lambda msg: msg.SerializeToString(),
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(SERVICE_NAME, method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    return {_normalize_method_name(name) for name in method_handlers}


async def serve(
    address: str = "0.0.0.0:50051",
    *,
    enable_auth: bool = False,
    exempt_methods: Iterable[str] | None = None,
    method_permissions: Mapping[str, str] | None = None,
) -> None:
    """Start the gRPC pricing server.

    Args:
        address: Socket address to bind the server to.
        enable_auth: Whether to enable authentication interceptors.
        exempt_methods: Collection of RPC method names that should bypass authentication.
        method_permissions: Mapping of RPC methods to the permission required to execute them.
    """

    normalized_exempt = _normalize_method_set(exempt_methods)
    normalized_permissions = _normalize_method_mapping(method_permissions)

    server = create_authenticated_server(
        enable_auth=enable_auth,
        exempt_methods=normalized_exempt,
        method_permissions=normalized_permissions,
    )
    add_servicer(server)
    server.add_insecure_port(address)
    await server.start()
    try:
        await server.wait_for_termination()
    finally:  # pragma: no cover - server shutdown path
        await server.stop(grace=None)


def run_server(
    address: str = "0.0.0.0:50051",
    *,
    enable_auth: bool = False,
    exempt_methods: Iterable[str] | None = None,
    method_permissions: Mapping[str, str] | None = None,
) -> None:
    """Blocking wrapper that starts the asyncio-based gRPC server."""

    asyncio.run(
        serve(
            address,
            enable_auth=enable_auth,
            exempt_methods=exempt_methods,
            method_permissions=method_permissions,
        )
    )


__all__ = ["PricingServicer", "add_servicer", "serve", "run_server"]
