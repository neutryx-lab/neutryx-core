"""Converters from legacy Pydantic models to high-performance SoA format.

Provides utilities to migrate from dict/Pydantic-based portfolio representations
to vectorized Struct-of-Arrays format for batch processing.
"""

from __future__ import annotations

from typing import Any, Sequence

import jax.numpy as jnp

from neutryx.data.indices import IndexMapping, build_index_mapping
from neutryx.data.portfolio_batch import PortfolioBatch, TradeArrays


def trades_to_portfolio_batch(
    trades: Sequence[dict[str, Any]],
    default_spot: float = 100.0,
    default_strike: float = 100.0,
) -> PortfolioBatch:
    """Convert a sequence of trade dictionaries to PortfolioBatch.

    Extracts numerical parameters and categorical fields from trade dicts,
    building index mappings and vectorized arrays.

    Parameters
    ----------
    trades : Sequence[dict[str, Any]]
        List of trade dictionaries with fields:
        - spot (float, optional): Current spot/rate
        - strike (float, optional): Strike price/rate
        - maturity (float): Time to maturity in years
        - notional (float): Notional amount
        - option_type (str, optional): "call", "put", or None
        - currency (str): Currency code (e.g., "USD")
        - counterparty_id (str): Counterparty identifier
        - product_type (str): Product type identifier
    default_spot : float, optional
        Default spot value if not specified in trade
    default_strike : float, optional
        Default strike value if not specified in trade

    Returns
    -------
    PortfolioBatch
        Vectorized portfolio representation

    Examples
    --------
    >>> trades = [
    ...     {"spot": 100, "strike": 110, "maturity": 1.0, "notional": 1e6,
    ...      "option_type": "call", "currency": "USD", "counterparty_id": "CP1",
    ...      "product_type": "VanillaOption"},
    ...     {"spot": 105, "strike": 115, "maturity": 2.0, "notional": 2e6,
    ...      "option_type": "put", "currency": "EUR", "counterparty_id": "CP1",
    ...      "product_type": "VanillaOption"},
    ... ]
    >>> portfolio = trades_to_portfolio_batch(trades)
    >>> portfolio.n_trades
    2
    >>> portfolio.n_currencies
    2
    """
    if not trades:
        raise ValueError("Cannot convert empty trades list to PortfolioBatch")

    # Extract numerical parameters
    spots = []
    strikes = []
    maturities = []
    notionals = []
    option_types_encoded = []
    currencies = []
    counterparty_ids = []
    product_types = []

    option_type_map = {"call": 0, "put": 1, None: -1, "": -1}

    for trade in trades:
        # Numerical fields
        spots.append(trade.get("spot", default_spot))
        strikes.append(trade.get("strike", default_strike))
        maturities.append(trade["maturity"])
        notionals.append(trade["notional"])

        # Option type encoding
        option_type_str = trade.get("option_type", "").lower()
        option_types_encoded.append(option_type_map.get(option_type_str, -1))

        # Categorical fields
        currencies.append(trade["currency"])
        counterparty_ids.append(trade["counterparty_id"])
        product_types.append(trade["product_type"])

    # Build index mappings
    currency_mapping = build_index_mapping(currencies, name="currency")
    counterparty_mapping = build_index_mapping(counterparty_ids, name="counterparty")
    product_type_mapping = build_index_mapping(product_types, name="product_type")

    # Encode categorical fields
    currency_idx = currency_mapping.encode_batch(currencies)
    counterparty_idx = counterparty_mapping.encode_batch(counterparty_ids)
    product_type_idx = product_type_mapping.encode_batch(product_types)

    # Create trade arrays
    trade_arrays = TradeArrays(
        spots=jnp.array(spots, dtype=jnp.float32),
        strikes=jnp.array(strikes, dtype=jnp.float32),
        maturities=jnp.array(maturities, dtype=jnp.float32),
        notionals=jnp.array(notionals, dtype=jnp.float32),
        option_types=jnp.array(option_types_encoded, dtype=jnp.int32),
    )

    return PortfolioBatch(
        trade_arrays=trade_arrays,
        currency_idx=currency_idx,
        counterparty_idx=counterparty_idx,
        product_type_idx=product_type_idx,
        currency_mapping=currency_mapping,
        counterparty_mapping=counterparty_mapping,
        product_type_mapping=product_type_mapping,
        metadata={"source": "dict_conversion", "n_original_trades": len(trades)},
    )


def pydantic_portfolio_to_batch(portfolio: Any) -> PortfolioBatch:
    """Convert a Pydantic-based Portfolio to PortfolioBatch.

    Parameters
    ----------
    portfolio : Portfolio
        Legacy Pydantic portfolio model (from neutryx.portfolio.portfolio)

    Returns
    -------
    PortfolioBatch
        Vectorized portfolio representation

    Notes
    -----
    This function is designed to work with the legacy Portfolio class but
    does not import it to avoid circular dependencies. It works with any
    object that has a `.trades` dict attribute.
    """
    # Extract trade dicts from Pydantic models
    trade_dicts = []

    for trade_id, trade in portfolio.trades.items():
        trade_dict = {
            "spot": trade.product_details.get("spot") if trade.product_details else 100.0,
            "strike": trade.product_details.get("strike") if trade.product_details else 100.0,
            "maturity": (trade.maturity_date - trade.trade_date).days / 365.25
            if hasattr(trade, "maturity_date") and hasattr(trade, "trade_date")
            else 1.0,
            "notional": trade.notional if hasattr(trade, "notional") else 1e6,
            "option_type": trade.product_details.get("option_type") if trade.product_details else "",
            "currency": trade.product_details.get("currency", "USD") if trade.product_details else "USD",
            "counterparty_id": trade.counterparty_id if hasattr(trade, "counterparty_id") else "UNKNOWN",
            "product_type": trade.product_type if hasattr(trade, "product_type") else "UNKNOWN",
            "trade_id": trade_id,
        }
        trade_dicts.append(trade_dict)

    portfolio_batch = trades_to_portfolio_batch(trade_dicts)

    # Add portfolio-level metadata
    portfolio_batch.metadata.update({
        "source": "pydantic_conversion",
        "original_portfolio_id": getattr(portfolio, "id", None),
    })

    return portfolio_batch


def batch_to_trade_dicts(portfolio_batch: PortfolioBatch) -> list[dict[str, Any]]:
    """Convert PortfolioBatch back to list of trade dictionaries.

    Useful for debugging, serialization, or interfacing with legacy systems.

    Parameters
    ----------
    portfolio_batch : PortfolioBatch
        Vectorized portfolio

    Returns
    -------
    list[dict[str, Any]]
        List of trade dictionaries

    Examples
    --------
    >>> # Round-trip conversion
    >>> original_trades = [...]
    >>> batch = trades_to_portfolio_batch(original_trades)
    >>> recovered_trades = batch_to_trade_dicts(batch)
    """
    trades = []
    option_type_decode = {0: "call", 1: "put", -1: None}

    for i in range(portfolio_batch.n_trades):
        trade_dict = {
            "spot": float(portfolio_batch.trade_arrays.spots[i]),
            "strike": float(portfolio_batch.trade_arrays.strikes[i]),
            "maturity": float(portfolio_batch.trade_arrays.maturities[i]),
            "notional": float(portfolio_batch.trade_arrays.notionals[i]),
            "option_type": option_type_decode[int(portfolio_batch.trade_arrays.option_types[i])],
            "currency": portfolio_batch.currency_mapping.decode([portfolio_batch.currency_idx[i]])[0],
            "counterparty_id": portfolio_batch.counterparty_mapping.decode(
                [portfolio_batch.counterparty_idx[i]]
            )[0],
            "product_type": portfolio_batch.product_type_mapping.decode(
                [portfolio_batch.product_type_idx[i]]
            )[0],
        }
        trades.append(trade_dict)

    return trades
