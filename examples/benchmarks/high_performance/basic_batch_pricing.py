"""Basic example of high-performance batch portfolio pricing.

This example demonstrates:
1. Converting trades to Struct-of-Arrays format
2. Building pre-computed market data grids
3. Batch pricing 10K trades in ~50ms
4. Aggregating to counterparty level
"""

import jax
import jax.numpy as jnp

from neutryx.data import (
    MarketDataGrid,
    PortfolioBatch,
    aggregate_to_counterparties,
    build_market_data_grid_simple,
    build_time_grid,
    compute_concentration_metrics,
    price_portfolio_batch,
    trades_to_portfolio_batch,
)


def create_sample_portfolio(n_trades: int = 10_000) -> list[dict]:
    """Create a sample portfolio of vanilla options.

    Parameters
    ----------
    n_trades : int
        Number of trades to generate

    Returns
    -------
    list[dict]
        List of trade dictionaries
    """
    import random

    trades = []
    currencies = ["USD", "EUR", "GBP"]
    counterparties = [f"CP_{i:03d}" for i in range(100)]  # 100 counterparties

    for i in range(n_trades):
        trade = {
            "spot": random.uniform(90, 110),
            "strike": random.uniform(95, 105),
            "maturity": random.uniform(0.25, 5.0),
            "notional": random.choice([1e6, 2e6, 5e6]),
            "option_type": random.choice(["call", "put"]),
            "currency": random.choice(currencies),
            "counterparty_id": random.choice(counterparties),
            "product_type": "VanillaOption",
        }
        trades.append(trade)

    return trades


def main():
    """Run basic batch pricing example."""
    print("=" * 80)
    print("High-Performance Batch Portfolio Pricing Example")
    print("=" * 80)

    # Step 1: Create sample portfolio (10K trades)
    print("\n1. Creating sample portfolio (10,000 trades)...")
    trades_list = create_sample_portfolio(n_trades=10_000)
    print(f"   ✓ Created {len(trades_list):,} trades")

    # Step 2: Convert to Struct-of-Arrays format
    print("\n2. Converting to Struct-of-Arrays format...")
    portfolio = trades_to_portfolio_batch(trades_list)
    print(f"   ✓ Portfolio batch created")
    print(f"     - Trades: {portfolio.n_trades:,}")
    print(f"     - Counterparties: {portfolio.n_counterparties:,}")
    print(f"     - Currencies: {portfolio.n_currencies}")

    # Step 3: Build market data grid
    print("\n3. Building pre-computed market data grid...")
    time_grid = build_time_grid(max_maturity=5.0, n_steps=600)
    market_grid = build_market_data_grid_simple(
        time_grid=time_grid,
        currencies=["EUR", "GBP", "USD"],  # Sorted order
        assets=["SPX"],
        flat_rate=0.05,
        flat_vol=0.20,
    )
    print(f"   ✓ Market grid built")
    print(f"     - Time points: {market_grid.n_times}")
    print(f"     - Currencies: {market_grid.n_currencies}")
    print(f"     - Memory: {(market_grid.discount_factors.nbytes / 1024):.1f} KB")

    # Step 4: Batch price portfolio
    print("\n4. Batch pricing portfolio...")
    import time

    start = time.perf_counter()
    mtms = price_portfolio_batch(portfolio, market_grid, use_notional=True)
    elapsed_ms = (time.perf_counter() - start) * 1000

    print(f"   ✓ Priced {portfolio.n_trades:,} trades in {elapsed_ms:.1f}ms")
    print(f"     - Speed: {portfolio.n_trades / elapsed_ms * 1000:.0f} trades/second")
    print(f"     - Total PV: ${float(jnp.sum(mtms)):,.0f}")
    print(f"     - Average trade value: ${float(jnp.mean(mtms)):,.0f}")

    # Step 5: Aggregate to counterparty level
    print("\n5. Aggregating to counterparty level...")
    cp_exposures = aggregate_to_counterparties(portfolio, mtms)
    print(f"   ✓ Aggregated to {portfolio.n_counterparties} counterparties")
    print(f"     - Total CP exposure: ${float(jnp.sum(cp_exposures)):,.0f}")
    print(f"     - Max CP exposure: ${float(jnp.max(cp_exposures)):,.0f}")
    print(f"     - Min CP exposure: ${float(jnp.min(cp_exposures)):,.0f}")

    # Step 6: Compute concentration metrics
    print("\n6. Computing portfolio concentration metrics...")
    metrics = compute_concentration_metrics(portfolio, mtms)
    print(f"   ✓ Concentration analysis:")
    print(f"     - Herfindahl Index: {metrics['herfindahl_index']:.4f}")
    print(f"     - Top 5 concentration: {metrics['top_5_concentration']:.1%}")
    print(f"     - Top 10 concentration: {metrics['top_10_concentration']:.1%}")
    print(f"     - Max single CP: {metrics['max_concentration']:.1%}")

    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    # Set random seed for reproducibility
    import random

    random.seed(42)

    main()
