"""Command line interface for Neutryx pricing workflows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import jax
import jax.numpy as jnp
import yaml

from neutryx.core.engine import MCConfig, price_vanilla_mc
from neutryx.valuations.cva import cva
from neutryx.valuations.fva import fva
from neutryx.valuations.mva import mva

JsonDict = dict[str, Any]


def _float_sequence(text: str) -> list[float]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("expected at least one numeric value")
    try:
        return [float(p) for p in parts]
    except ValueError as exc:  # pragma: no cover - defensive
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _load_config(path: str | None) -> Mapping[str, Any]:
    if path is None:
        return {}
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"configuration file '{path}' not found")
    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, Mapping):
        raise ValueError("configuration root must be a mapping")
    return data


def _array(values: Iterable[float] | None, name: str, *, match: int | None = None) -> jnp.ndarray:
    if values is None:
        raise ValueError(f"missing required sequence '{name}'")
    arr = jnp.asarray(list(values), dtype=jnp.float32)
    if arr.ndim != 1:
        raise ValueError(f"'{name}' must be a 1-D sequence")
    if match is not None and arr.shape[0] != match:
        raise ValueError(f"'{name}' must have length {match}")
    return arr


def _handle_price_vanilla(args: argparse.Namespace) -> JsonDict:
    config = dict(_load_config(args.config))
    mc_cfg = config.pop("mc", {}) if isinstance(config.get("mc"), Mapping) else {}

    def _override(key: str, current: Any) -> Any:
        value = getattr(args, key)
        return value if value is not None else current

    spot = _override("spot", config.get("spot"))
    strike = _override("strike", config.get("strike"))
    maturity = _override("maturity", config.get("maturity"))
    rate = _override("rate", config.get("rate", 0.0))
    dividend = _override("dividend", config.get("dividend", 0.0))
    vol = _override("volatility", config.get("volatility"))
    is_call = not bool(_override("put", config.get("put", False)))
    steps = int(_override("steps", mc_cfg.get("steps", config.get("steps", 64))))
    paths = int(_override("paths", mc_cfg.get("paths", config.get("paths", 8192))))
    antithetic = bool(_override("antithetic", mc_cfg.get("antithetic", config.get("antithetic", False))))
    seed = _override("seed", config.get("seed"))

    required = {
        "spot": spot,
        "strike": strike,
        "maturity": maturity,
        "volatility": vol,
    }
    missing = [key for key, value in required.items() if value is None]
    if missing:
        raise ValueError(f"missing required parameters: {', '.join(missing)}")

    mc = MCConfig(steps=steps, paths=paths, antithetic=antithetic)
    key_seed = int(seed) if seed is not None else 0
    key = jax.random.PRNGKey(key_seed)
    price = price_vanilla_mc(
        key,
        float(spot),
        float(strike),
        float(maturity),
        float(rate),
        float(dividend),
        float(vol),
        mc,
        is_call=is_call,
    )
    return {"price": float(price)}


def _handle_xva_cva(args: argparse.Namespace) -> JsonDict:
    config = dict(_load_config(args.config))
    epe = args.epe or config.get("epe")
    discount = args.discount or config.get("discount")
    default_prob = args.default_probability or config.get("default_probability")
    lgd = args.lgd if args.lgd is not None else config.get("lgd", 0.6)

    epe_arr = _array(epe, "epe")
    disc_arr = _array(discount, "discount", match=epe_arr.shape[0])
    pd_arr = _array(default_prob, "default_probability", match=epe_arr.shape[0])

    value = cva(epe_arr, disc_arr, pd_arr, lgd=float(lgd))
    return {"cva": float(value)}


def _handle_xva_fva(args: argparse.Namespace) -> JsonDict:
    config = dict(_load_config(args.config))
    epe = args.epe or config.get("epe")
    discount = args.discount or config.get("discount")
    spread = config.get("funding_spread")
    if args.funding_spread is not None:
        spread = args.funding_spread

    epe_arr = _array(epe, "epe")
    disc_arr = _array(discount, "discount", match=epe_arr.shape[0])

    if isinstance(spread, Iterable) and not isinstance(spread, (str, bytes)):
        spread_arr = _array(spread, "funding_spread", match=epe_arr.shape[0])
    elif spread is not None:
        spread_arr = jnp.full_like(epe_arr, float(spread))
    else:
        raise ValueError("missing required parameter 'funding_spread'")

    value = fva(epe_arr, spread_arr, disc_arr)
    return {"fva": float(value)}


def _handle_xva_mva(args: argparse.Namespace) -> JsonDict:
    config = dict(_load_config(args.config))
    margin = args.initial_margin or config.get("initial_margin")
    discount = args.discount or config.get("discount")
    spread = args.spread if args.spread is not None else config.get("spread")

    margin_arr = _array(margin, "initial_margin")
    disc_arr = _array(discount, "discount", match=margin_arr.shape[0])
    spread_arr = _array(spread, "spread", match=margin_arr.shape[0]) if isinstance(spread, Iterable) and not isinstance(spread, (str, bytes)) else None

    if spread_arr is None:
        if spread is None:
            raise ValueError("missing required parameter 'spread'")
        spread_arr = jnp.full_like(margin_arr, float(spread))

    value = mva(margin_arr, disc_arr, spread_arr)
    return {"mva": float(value)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="neutryx", description="Neutryx pricing CLI")
    subparsers = parser.add_subparsers(dest="command")

    price_parser = subparsers.add_parser("price", help="Pricing workflows")
    price_sub = price_parser.add_subparsers(dest="workflow")
    vanilla = price_sub.add_parser("vanilla", help="Monte Carlo vanilla option pricing")
    vanilla.add_argument("--config", help="Path to YAML configuration")
    vanilla.add_argument("--spot", type=float, help="Spot price")
    vanilla.add_argument("--strike", type=float, help="Strike price")
    vanilla.add_argument("--maturity", type=float, help="Time to maturity (years)")
    vanilla.add_argument("--rate", type=float, help="Risk-free rate", default=None)
    vanilla.add_argument("--dividend", type=float, help="Dividend yield", default=None)
    vanilla.add_argument("--volatility", type=float, help="Volatility")
    vanilla.add_argument("--steps", type=int, help="Number of Monte Carlo steps")
    vanilla.add_argument("--paths", type=int, help="Number of Monte Carlo paths")
    vanilla.add_argument("--antithetic", action="store_true", help="Enable antithetic sampling")
    vanilla.add_argument("--put", action="store_true", help="Price a put instead of a call")
    vanilla.add_argument("--seed", type=int, help="Random seed")
    vanilla.set_defaults(handler=_handle_price_vanilla)

    xva_parser = subparsers.add_parser("xva", help="XVA valuation workflows")
    xva_sub = xva_parser.add_subparsers(dest="workflow")

    cva_parser = xva_sub.add_parser("cva", help="Credit Valuation Adjustment")
    cva_parser.add_argument("--config", help="Path to YAML configuration")
    cva_parser.add_argument("--epe", type=_float_sequence, help="Expected positive exposure profile")
    cva_parser.add_argument("--discount", type=_float_sequence, help="Discount factors over time")
    cva_parser.add_argument("--default-probability", dest="default_probability", type=_float_sequence, help="Cumulative default probabilities")
    cva_parser.add_argument("--lgd", type=float, help="Loss given default")
    cva_parser.set_defaults(handler=_handle_xva_cva)

    fva_parser = xva_sub.add_parser("fva", help="Funding Valuation Adjustment")
    fva_parser.add_argument("--config", help="Path to YAML configuration")
    fva_parser.add_argument("--epe", type=_float_sequence, help="Expected positive exposure profile")
    fva_parser.add_argument("--discount", type=_float_sequence, help="Discount factors over time")
    fva_parser.add_argument("--funding-spread", dest="funding_spread", type=float, help="Funding spread (constant)")
    fva_parser.set_defaults(handler=_handle_xva_fva)

    mva_parser = xva_sub.add_parser("mva", help="Margin Valuation Adjustment")
    mva_parser.add_argument("--config", help="Path to YAML configuration")
    mva_parser.add_argument("--initial-margin", dest="initial_margin", type=_float_sequence, help="Initial margin profile")
    mva_parser.add_argument("--discount", type=_float_sequence, help="Discount factors over time")
    mva_parser.add_argument("--spread", type=float, help="Margin funding spread (constant)")
    mva_parser.set_defaults(handler=_handle_xva_mva)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)

    if handler is None:
        parser.print_help()
        return 1

    result = handler(args)
    print(json.dumps(result, indent=2))
    return 0

