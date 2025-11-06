"""Comprehensive showcase of asset class implementations in Neutryx.

This demo showcases various asset classes and their derivatives:
- Equity: forwards, dividend swaps, variance swaps, total return swaps
- Commodities: forwards with convenience yield, options, swaps
- Inflation: linked bonds, zero-coupon swaps, caps/floors
- Volatility: VIX futures, variance swaps, realized variance
- Convertible bonds: pricing and analytics
"""
import jax.numpy as jnp

from neutryx.products.commodity import (
    commodity_forward_price,
    commodity_option_price,
    commodity_swap_value,
)
from neutryx.products.convertible import (
    convertible_bond_parity,
    convertible_bond_simple_price,
)
from neutryx.products.equity import (
    dividend_swap_value,
    equity_forward_price,
    total_return_swap_value,
    variance_swap_value,
)
from neutryx.products.inflation import (
    breakeven_inflation,
    inflation_linked_bond_price,
    zero_coupon_inflation_swap_value,
)
from neutryx.products.volatility import (
    realized_variance,
    variance_swap_payoff,
    vix_futures_price,
)


def equity_examples():
    """Demonstrate equity products."""
    print("\n" + "=" * 60)
    print("EQUITY PRODUCTS")
    print("=" * 60)

    # 1. Equity Forward
    print("\n1. Equity Forward Contract")
    spot = 100.0
    maturity = 1.0
    risk_free_rate = 0.05
    dividend_yield = 0.02

    forward = equity_forward_price(spot, maturity, risk_free_rate, dividend_yield)
    print(f"   Spot Price: ${spot:.2f}")
    print(f"   Forward Price (1Y): ${forward:.2f}")
    print(f"   Cost of Carry: {(risk_free_rate - dividend_yield)*100:.2f}%")

    # 2. Dividend Swap
    print("\n2. Dividend Swap")
    notional = 10_000  # shares
    strike = 5.0  # expected dividend per share
    expected_divs = 6.5
    discount_factor = 0.95

    div_swap_value = dividend_swap_value(notional, strike, expected_divs, discount_factor)
    print(f"   Notional: {notional:,} shares")
    print(f"   Strike: ${strike:.2f}/share")
    print(f"   Expected Dividends: ${expected_divs:.2f}/share")
    print(f"   Swap Value: ${div_swap_value:,.2f}")

    # 3. Variance Swap
    print("\n3. Variance Swap")
    var_notional = 10_000
    var_strike = 0.04  # 20% vol
    realized_var = 0.035
    expected_var = 0.045
    discount_factor = 0.95

    var_swap_value = variance_swap_value(
        var_notional, var_strike, realized_var, expected_var, discount_factor
    )
    print(f"   Variance Notional: ${var_notional:,}")
    print(f"   Strike: {var_strike*100:.1f}% variance (20% vol)")
    print(f"   Expected Variance: {expected_var*100:.1f}%")
    print(f"   Swap Value: ${var_swap_value:,.2f}")

    # 4. Total Return Swap
    print("\n4. Total Return Swap")
    trs_notional = 1_000_000
    spot_initial = 100.0
    spot_current = 108.0
    dividends = 3.0
    funding_rate = 0.03
    time_elapsed = 0.5

    trs_value = total_return_swap_value(
        trs_notional,
        spot_initial,
        spot_current,
        dividends,
        funding_rate,
        time_elapsed,
        discount_factor,
    )
    equity_return = ((spot_current / spot_initial) - 1 + dividends / spot_initial) * 100
    print(f"   Notional: ${trs_notional:,}")
    print(f"   Equity Return: {equity_return:.2f}%")
    print(f"   Funding Cost: {funding_rate * time_elapsed * 100:.2f}%")
    print(f"   TRS Value: ${trs_value:,.2f}")


def commodity_examples():
    """Demonstrate commodity products."""
    print("\n" + "=" * 60)
    print("COMMODITY PRODUCTS")
    print("=" * 60)

    # 1. Commodity Forward with Convenience Yield
    print("\n1. Crude Oil Forward Contract")
    spot = 75.0  # $/barrel
    maturity = 1.0
    risk_free_rate = 0.05
    storage_cost = 0.03
    convenience_yield = 0.02

    forward = commodity_forward_price(spot, maturity, risk_free_rate, storage_cost, convenience_yield)
    contango = forward - spot
    print(f"   Spot Price: ${spot:.2f}/barrel")
    print(f"   Storage Cost: {storage_cost*100:.1f}%")
    print(f"   Convenience Yield: {convenience_yield*100:.1f}%")
    print(f"   1Y Forward: ${forward:.2f}/barrel")
    print(f"   Market Structure: {'Contango' if contango > 0 else 'Backwardation'} (${contango:.2f})")

    # 2. Commodity Option
    print("\n2. Gold Call Option")
    spot = 1800.0  # $/oz
    strike = 1850.0
    maturity = 0.5
    volatility = 0.15

    call_price = commodity_option_price(
        spot, strike, maturity, risk_free_rate, volatility, storage_cost=0.01, convenience_yield=0.0
    )
    print(f"   Spot Gold: ${spot:.2f}/oz")
    print(f"   Strike: ${strike:.2f}/oz")
    print(f"   Volatility: {volatility*100:.0f}%")
    print(f"   Call Option Price: ${call_price:.2f}/oz")

    # 3. Commodity Swap
    print("\n3. Natural Gas Swap")
    quantity = 10_000  # MMBtu
    fixed_price = 3.50
    floating_price = 4.20
    discount_factor = 0.98

    swap_value = commodity_swap_value(quantity, fixed_price, floating_price, discount_factor, "fixed_payer")
    print(f"   Quantity: {quantity:,} MMBtu")
    print(f"   Fixed Price: ${fixed_price:.2f}/MMBtu")
    print(f"   Floating Price: ${floating_price:.2f}/MMBtu")
    print(f"   Swap Value (to fixed payer): ${swap_value:,.2f}")


def inflation_examples():
    """Demonstrate inflation-linked products."""
    print("\n" + "=" * 60)
    print("INFLATION-LINKED PRODUCTS")
    print("=" * 60)

    # 1. Inflation-Linked Bond (TIPS)
    print("\n1. Treasury Inflation-Protected Security (TIPS)")
    face_value = 1000.0
    real_coupon = 0.005  # 0.5% real coupon
    real_yield = 0.003  # 0.3% real yield
    maturity = 10.0
    index_ratio = 1.25  # 25% cumulative inflation since issuance

    tips_price = inflation_linked_bond_price(
        face_value, real_coupon, real_yield, maturity, index_ratio, frequency=2
    )
    adjusted_par = face_value * index_ratio
    print(f"   Original Par: ${face_value:.2f}")
    print(f"   Index Ratio: {index_ratio:.3f}")
    print(f"   Inflation-Adjusted Par: ${adjusted_par:.2f}")
    print(f"   Real Coupon: {real_coupon*100:.2f}%")
    print(f"   Real Yield: {real_yield*100:.2f}%")
    print(f"   TIPS Price: ${tips_price:.2f}")

    # 2. Zero-Coupon Inflation Swap
    print("\n2. Zero-Coupon Inflation Swap (5Y)")
    notional = 10_000_000
    strike = 0.12  # 12% total inflation expected
    current_cpi = 280.0
    forward_cpi = 315.0
    base_cpi = 280.0
    discount_factor = 0.92

    zcis_value = zero_coupon_inflation_swap_value(
        notional, strike, 5.0, current_cpi, forward_cpi, base_cpi, discount_factor
    )
    expected_inflation = (forward_cpi / base_cpi - 1) * 100
    print(f"   Notional: ${notional:,}")
    print(f"   Strike: {strike*100:.1f}%")
    print(f"   Expected 5Y Inflation: {expected_inflation:.1f}%")
    print(f"   Swap Value: ${zcis_value:,.2f}")

    # 3. Breakeven Inflation
    print("\n3. Breakeven Inflation Analysis")
    nominal_yield = 0.045  # 10Y Treasury
    real_yield = 0.015  # 10Y TIPS

    breakeven = breakeven_inflation(nominal_yield, real_yield)
    print(f"   10Y Treasury Yield: {nominal_yield*100:.2f}%")
    print(f"   10Y TIPS Yield: {real_yield*100:.2f}%")
    print(f"   Breakeven Inflation: {breakeven*100:.2f}%")
    print(f"   Interpretation: Market expects {breakeven*100:.2f}% avg annual inflation")


def volatility_examples():
    """Demonstrate volatility products."""
    print("\n" + "=" * 60)
    print("VOLATILITY PRODUCTS")
    print("=" * 60)

    # 1. VIX Futures
    print("\n1. VIX Futures")
    vix_spot = 18.5
    maturity = 0.25  # 3 months
    mean_reversion = 1.5
    long_term_vol = 20.0

    vix_futures = vix_futures_price(vix_spot, maturity, 0.0, mean_reversion, long_term_vol)
    roll_yield = ((vix_futures / vix_spot) - 1) * 100
    print(f"   VIX Spot: {vix_spot:.2f}")
    print(f"   3M VIX Future: {vix_futures:.2f}")
    print(f"   Roll Yield: {roll_yield:+.2f}%")
    print(f"   Market Structure: {'Contango' if roll_yield > 0 else 'Backwardation'}")

    # 2. Realized Variance
    print("\n2. Realized Variance Calculation")
    # Simulate some price data
    prices = jnp.array([100.0, 101.5, 99.8, 102.3, 101.2, 103.5, 102.8, 104.2])

    realized_var = realized_variance(prices, annualization_factor=252.0)
    realized_vol = jnp.sqrt(realized_var) * 100
    print(f"   Sample Prices: {prices[:4]}...")
    print(f"   Realized Variance: {realized_var:.6f}")
    print(f"   Realized Volatility: {realized_vol:.2f}%")

    # 3. Variance Swap P&L
    print("\n3. Variance Swap Settlement")
    var_notional = 50_000
    var_strike = 0.0400  # 20% vol
    realized_var = 0.0529  # 23% realized vol

    payoff = variance_swap_payoff(var_notional, var_strike, realized_var)
    realized_vol_pct = jnp.sqrt(realized_var) * 100
    strike_vol_pct = jnp.sqrt(var_strike) * 100
    print(f"   Variance Notional: ${var_notional:,}")
    print(f"   Strike: {strike_vol_pct:.1f}% vol ({var_strike:.4f} variance)")
    print(f"   Realized: {realized_vol_pct:.1f}% vol ({realized_var:.4f} variance)")
    print(f"   Payoff: ${payoff:,.2f}")


def convertible_examples():
    """Demonstrate convertible bonds."""
    print("\n" + "=" * 60)
    print("CONVERTIBLE BONDS")
    print("=" * 60)

    # 1. Convertible Bond Pricing
    print("\n1. Convertible Bond Analysis")
    face_value = 1000.0
    coupon_rate = 0.03
    yield_rate = 0.05
    maturity = 5.0
    stock_price = 42.0
    conversion_ratio = 23.0
    stock_volatility = 0.35
    risk_free_rate = 0.04

    cb_result = convertible_bond_simple_price(
        face_value,
        coupon_rate,
        yield_rate,
        maturity,
        stock_price,
        conversion_ratio,
        stock_volatility,
        risk_free_rate,
        2,
    )

    conversion_value = convertible_bond_parity(stock_price, conversion_ratio)
    conversion_premium = ((cb_result["convertible_value"] / conversion_value) - 1) * 100

    print(f"   Face Value: ${face_value:.2f}")
    print(f"   Stock Price: ${stock_price:.2f}")
    print(f"   Conversion Ratio: {conversion_ratio:.1f} shares")
    print(f"   Conversion Price: ${face_value/conversion_ratio:.2f}")
    print(f"\n   Components:")
    print(f"   - Straight Bond Value: ${cb_result['straight_bond']:.2f}")
    print(f"   - Conversion Value: ${cb_result['conversion_value']:.2f}")
    print(f"   - Option Value: ${cb_result['option_value']:.2f}")
    print(f"\n   Convertible Bond Value: ${cb_result['convertible_value']:.2f}")
    print(f"   Conversion Premium: {conversion_premium:.2f}%")


def portfolio_example():
    """Demonstrate a multi-asset portfolio."""
    print("\n" + "=" * 60)
    print("MULTI-ASSET PORTFOLIO EXAMPLE")
    print("=" * 60)

    portfolio = {
        "Equity Forwards": 1_250_000,
        "Commodity Swaps": 875_000,
        "Inflation Swaps": 450_000,
        "Variance Swaps": 325_000,
        "Convertible Bonds": 2_100_000,
    }

    total = sum(portfolio.values())

    print("\n   Portfolio Composition:")
    for asset_class, value in portfolio.items():
        weight = (value / total) * 100
        print(f"   {asset_class:.<25} ${value:>12,}  ({weight:>5.1f}%)")

    print(f"   {'Total Portfolio Value':.<25} ${total:>12,}  (100.0%)")

    print("\n   Diversification Benefits:")
    print("   - Cross-asset hedging opportunities")
    print("   - Inflation protection through TIPS and inflation swaps")
    print("   - Volatility exposure via variance swaps")
    print("   - Commodity exposure provides inflation hedge")
    print("   - Convertibles offer equity upside with downside protection")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("NEUTRYX ASSET CLASS SHOWCASE")
    print("=" * 60)
    print("\nDemonstrating pricing and analytics across asset classes:")
    print("- Equity derivatives")
    print("- Commodity products")
    print("- Inflation-linked securities")
    print("- Volatility products")
    print("- Convertible bonds")

    equity_examples()
    commodity_examples()
    inflation_examples()
    volatility_examples()
    convertible_examples()
    portfolio_example()

    print("\n" + "=" * 60)
    print("SHOWCASE COMPLETE")
    print("=" * 60)
    print("\nFor more details, see:")
    print("- src/neutryx/products/equity.py")
    print("- src/neutryx/products/commodity.py")
    print("- src/neutryx/products/inflation.py")
    print("- src/neutryx/products/volatility.py")
    print("- src/neutryx/products/convertible.py")
    print()


if __name__ == "__main__":
    main()
