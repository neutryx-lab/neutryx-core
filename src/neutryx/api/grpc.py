"""gRPC service definitions for Neutryx pricing workflows."""
from __future__ import annotations

import asyncio
from typing import Any, Mapping

import grpc
import jax
import jax.numpy as jnp
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Struct

from neutryx.core.engine import MCConfig, price_vanilla_mc
from neutryx.valuations.xva.cva import cva
from neutryx.valuations.xva.fva import fva
from neutryx.valuations.xva.mva import mva

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

    async def PriceVanilla(self, request: Struct, context: grpc.aio.ServicerContext) -> Struct:  # noqa: N802
        payload = _struct_to_dict(request)
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


def add_servicer(server: grpc.aio.Server, servicer: PricingServicer | None = None) -> None:
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


async def serve(address: str = "0.0.0.0:50051") -> None:
    server = grpc.aio.server()
    add_servicer(server)
    server.add_insecure_port(address)
    await server.start()
    try:
        await server.wait_for_termination()
    finally:  # pragma: no cover - server shutdown path
        await server.stop(grace=None)


def run_server(address: str = "0.0.0.0:50051") -> None:
    asyncio.run(serve(address))


__all__ = ["PricingServicer", "add_servicer", "serve", "run_server"]
