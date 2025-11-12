"""Helper script for launching the Neutryx gRPC pricing server."""

from __future__ import annotations

import argparse
from typing import Iterable

from neutryx.api.grpc import run_server


def _parse_permissions(values: Iterable[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                f"invalid permission mapping '{item}'. Expected format METHOD=PERMISSION"
            )
        method, permission = item.split("=", 1)
        method = method.strip()
        permission = permission.strip()
        if not method or not permission:
            raise argparse.ArgumentTypeError(
                f"invalid permission mapping '{item}'. Expected format METHOD=PERMISSION"
            )
        mapping[method] = permission
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--address",
        default="0.0.0.0:50051",
        help="Address the gRPC server should bind to (default: 0.0.0.0:50051)",
    )
    parser.add_argument(
        "--enable-auth",
        action="store_true",
        help="Enable JWT authentication and RBAC interceptors",
    )
    parser.add_argument(
        "--exempt-method",
        action="append",
        default=[],
        help=(
            "RPC method name that should bypass authentication. "
            "Repeat this option to exempt multiple methods."
        ),
    )
    parser.add_argument(
        "--require-permission",
        action="append",
        default=[],
        metavar="METHOD=PERMISSION",
        help=(
            "Map an RPC method to a permission string enforced by RBAC. "
            "Example: --require-permission PriceVanilla=pricing.vanilla:read"
        ),
    )

    args = parser.parse_args()

    method_permissions = _parse_permissions(args.require_permission)

    run_server(
        args.address,
        enable_auth=args.enable_auth,
        exempt_methods=args.exempt_method,
        method_permissions=method_permissions if method_permissions else None,
    )


if __name__ == "__main__":
    main()
