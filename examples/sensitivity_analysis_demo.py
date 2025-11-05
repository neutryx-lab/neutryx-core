"""Comprehensive examples for sensitivity analysis framework.

This script demonstrates how to use the sensitivity analysis framework for:
- DV01: Interest rate risk management
- CS01: Credit spread risk management
- Vega surfaces: Volatility risk across tenors and strikes
- FX Greeks: Delta, gamma, and higher-order sensitivities
- Portfolio aggregation: Risk reporting across asset classes

Run this script to see practical examples and formatted outputs.
"""
import jax.numpy as jnp

from neutryx.risk.sensitivity_analysis import (
    SensitivityConfig,
    SensitivityMethod,
    aggregate_portfolio_sensitivities,
    calculate_cs01,
    calculate_dv01,
    calculate_fx_greeks,
    calculate_higher_order_greeks,
    calculate_vega_surface,
    format_sensitivity_report,
)


def example_1_dv01_interest_rate_swap():
    """Example 1: DV01 for an interest rate swap portfolio."""
    print("=" * 80)
    print("EXAMPLE 1: DV01 for Interest Rate Swap")
    print("=" * 80)
    print()

    # Define a 5-year payer swap pricing function
    def price_5y_payer_swap(params):
        """Price a 5Y payer swap (pay fixed, receive floating).

        Params:
            params[0]: Current floating rate (e.g., SOFR, LIBOR)
            params[1:]: Forward rates for 1Y, 2Y, 3Y, 4Y, 5Y
        """
        floating_rate = params[0]
        forward_rates = params[1:6]
        fixed_rate = 0.03  # 3% fixed rate
        notional = 10_000_000

        # Simplified swap PV: sum of fixed payments - floating payments
        fixed_pv = sum([
            fixed_rate * notional * 0.25 * jnp.exp(-forward_rates[i//4] * (i+1) * 0.25)
            for i in range(20)  # Quarterly payments
        ])

        floating_pv = notional * (1 - jnp.exp(-floating_rate * 5.0))

        return fixed_pv - floating_pv

    # Current market rates (spot rate + forward curve)
    rates = jnp.array([
        0.0300,  # Spot rate
        0.0310,  # 1Y forward
        0.0320,  # 2Y forward
        0.0325,  # 3Y forward
        0.0330,  # 4Y forward
        0.0335,  # 5Y forward
    ])

    # Calculate DV01 for all rates
    dv01_vector = calculate_dv01(
        price_5y_payer_swap,
        rates,
        rate_indices=list(range(len(rates)))
    )

    print(f"Position: $10M notional 5Y payer swap (pay 3% fixed)")
    print(f"Current rates: {rates}")
    print()
    print("DV01 by tenor:")
    tenors = ["Spot", "1Y", "2Y", "3Y", "4Y", "5Y"]
    for tenor, dv01 in zip(tenors, dv01_vector):
        print(f"  {tenor:<6} DV01 = ${dv01:>12,.2f}")
    print()
    print(f"Total DV01: ${jnp.sum(dv01_vector):>12,.2f}")
    print()
    print("Interpretation:")
    print("  - Negative DV01: Lose money when rates rise (paying fixed)")
    print("  - Hedge: Buy bonds or enter receiver swaps")
    print("  - Largest exposure at longer tenors (4Y-5Y)")
    print()


def example_2_cs01_cds_portfolio():
    """Example 2: CS01 for a CDS portfolio."""
    print("=" * 80)
    print("EXAMPLE 2: CS01 for CDS Portfolio")
    print("=" * 80)
    print()

    # Define CDS pricing functions
    def price_cds_protection(spread, recovery, notional, tenor):
        """Price CDS protection (buy protection = long credit)."""
        # Simplified: PV of protection leg - PV of premium leg
        protection_pv = notional * (1 - recovery) * (1 - jnp.exp(-spread * tenor))
        premium_pv = spread * notional * tenor / 2  # Approximation
        return protection_pv - premium_pv

    # Portfolio: 3 CDS contracts
    cds_portfolio = [
        {'name': 'AAPL 5Y CDS', 'spread': 0.0120, 'notional': 5_000_000},
        {'name': 'MSFT 5Y CDS', 'spread': 0.0100, 'notional': 3_000_000},
        {'name': 'TSLA 5Y CDS', 'spread': 0.0350, 'notional': 2_000_000},
    ]

    recovery = 0.40  # 40% recovery assumption

    print("Portfolio:")
    for cds in cds_portfolio:
        print(f"  {cds['name']}: ${cds['notional']:,} notional @ {cds['spread']*10000:.0f} bps")
    print()

    print("CS01 by position:")
    total_cs01 = 0.0

    for cds in cds_portfolio:
        # Create pricing function for this CDS
        def price_this_cds(params):
            return price_cds_protection(
                params[0], recovery, cds['notional'], 5.0
            )

        params = jnp.array([cds['spread']])
        cs01 = calculate_cs01(price_this_cds, params, spread_indices=0)

        print(f"  {cds['name']:<20} CS01 = ${cs01:>12,.2f}")
        total_cs01 += cs01

    print()
    print(f"Portfolio CS01: ${total_cs01:>12,.2f}")
    print()
    print("Interpretation:")
    print("  - Positive CS01: Gain when spreads widen (long protection)")
    print("  - TSLA has highest CS01 due to higher spread")
    print("  - Hedge: Sell protection on index (CDX, iTraxx)")
    print()


def example_3_vega_surface_options():
    """Example 3: Vega surface for options portfolio."""
    print("=" * 80)
    print("EXAMPLE 3: Vega Surface for Options Portfolio")
    print("=" * 80)
    print()

    def price_options_portfolio(params):
        """Price a portfolio of options across strikes and tenors."""
        S = 100.0  # Spot price
        r = 0.05
        vols = params  # Volatility for each option

        # Portfolio specification
        positions = [
            {'K': 95, 'T': 0.25, 'vol_idx': 0, 'qty': 100},   # 3M 95 call
            {'K': 100, 'T': 0.25, 'vol_idx': 1, 'qty': 200},  # 3M ATM call
            {'K': 105, 'T': 0.25, 'vol_idx': 2, 'qty': 100},  # 3M 105 call
            {'K': 95, 'T': 1.0, 'vol_idx': 3, 'qty': 150},    # 1Y 95 call
            {'K': 100, 'T': 1.0, 'vol_idx': 4, 'qty': 300},   # 1Y ATM call
            {'K': 105, 'T': 1.0, 'vol_idx': 5, 'qty': 150},   # 1Y 105 call
        ]

        total_value = 0.0
        for pos in positions:
            vol = vols[pos['vol_idx']]
            K = pos['K']
            T = pos['T']
            qty = pos['qty']

            # Simplified Black-Scholes
            d1 = (jnp.log(S/K) + (r + 0.5 * vol**2) * T) / (vol * jnp.sqrt(T))
            d2 = d1 - vol * jnp.sqrt(T)

            from jax.scipy.stats import norm
            call_price = S * norm.cdf(d1) - K * jnp.exp(-r*T) * norm.cdf(d2)
            total_value += call_price * qty

        return total_value

    # Current volatility surface (by tenor and strike)
    vols = jnp.array([
        0.18,  # 3M 95 call
        0.16,  # 3M ATM call
        0.19,  # 3M 105 call
        0.20,  # 1Y 95 call
        0.18,  # 1Y ATM call
        0.21,  # 1Y 105 call
    ])

    tenors = [0.25, 0.25, 0.25, 1.0, 1.0, 1.0]
    strikes = [95.0, 100.0, 105.0, 95.0, 100.0, 105.0]

    # Calculate vega surface
    vega_data = calculate_vega_surface(
        price_options_portfolio,
        vols,
        vol_indices=list(range(len(vols))),
        tenors=tenors,
        strikes=strikes
    )

    print("Vega Surface ($ per 1% vol move):")
    print()
    print("By Tenor and Strike:")
    for tenor, strike, vega in zip(tenors, strikes, vega_data['vega_vector']):
        print(f"  T={tenor:.2f}Y, K={strike:>5.0f}: ${vega:>10,.2f}")

    print()
    print("Aggregated by Tenor:")
    for tenor, vega in sorted(vega_data['vega_by_tenor'].items()):
        print(f"  {tenor:.2f}Y: ${vega:>10,.2f}")

    print()
    print("Aggregated by Strike:")
    for strike, vega in sorted(vega_data['vega_by_strike'].items()):
        print(f"  K={strike:>5.0f}: ${vega:>10,.2f}")

    print()
    print(f"Total Portfolio Vega: ${vega_data['total_vega']:,.2f}")
    print()
    print("Interpretation:")
    print("  - Positive vega: Gain when volatility rises")
    print("  - 1Y options have more vega than 3M (more time value)")
    print("  - ATM strikes (K=100) have highest vega")
    print()


def example_4_fx_greeks():
    """Example 4: FX option Greeks."""
    print("=" * 80)
    print("EXAMPLE 4: FX Option Greeks (EUR/USD)")
    print("=" * 80)
    print()

    def price_eurusd_call(params):
        """Price EUR/USD call option using Garman-Kohlhagen."""
        S = params[0]  # Spot EUR/USD
        vol = params[1]
        K = 1.10
        T = 1.0
        r_usd = 0.05  # USD domestic rate
        r_eur = 0.02  # EUR foreign rate

        d1 = (jnp.log(S/K) + (r_usd - r_eur + 0.5 * vol**2) * T) / (vol * jnp.sqrt(T))
        d2 = d1 - vol * jnp.sqrt(T)

        from jax.scipy.stats import norm
        call = (S * jnp.exp(-r_eur * T) * norm.cdf(d1) -
                K * jnp.exp(-r_usd * T) * norm.cdf(d2))

        return call * 1_000_000  # $1M notional

    # Market parameters
    params = jnp.array([1.10, 0.12])  # Spot = 1.10, Vol = 12%

    # Calculate Greeks
    greeks = calculate_fx_greeks(
        price_eurusd_call,
        params,
        spot_index=0,
        vol_index=1
    )

    print("Position: EUR/USD 1Y ATM Call, $1M Notional")
    print(f"Spot: {params[0]:.4f}")
    print(f"Volatility: {params[1]*100:.1f}%")
    print(f"Strike: 1.1000")
    print()
    print("Greeks:")
    print(f"  Delta:  {greeks['delta']:>10,.2f}  (spot sensitivity)")
    print(f"  Gamma:  {greeks['gamma']:>10,.6f}  (delta change per spot move)")
    print(f"  Vega:   ${greeks['vega']:>10,.2f}  (per 1% vol)")
    print()
    print("Interpretation:")
    print(f"  - Delta {greeks['delta']:.2f}: For 1 pip move, P&L changes by ${greeks['delta']*0.0001:,.0f}")
    print(f"  - Vega ${greeks['vega']:.0f}: If vol rises 1%, gain ${greeks['vega']:.0f}")
    print()

    # Calculate higher-order Greeks
    higher_greeks = calculate_higher_order_greeks(
        price_eurusd_call,
        params,
        spot_index=0,
        vol_index=1
    )

    print("Higher-Order Greeks:")
    print(f"  Vanna:  {higher_greeks['vanna']:>10,.4f}  (delta change per vol)")
    print(f"  Volga:  {higher_greeks['volga']:>10,.4f}  (vega change per vol)")
    print()
    print("Interpretation:")
    print("  - Vanna: Delta hedging changes when volatility changes")
    print("  - Volga: Vega exposure increases when volatility increases")
    print()


def example_5_portfolio_aggregation():
    """Example 5: Portfolio-level risk aggregation."""
    print("=" * 80)
    print("EXAMPLE 5: Portfolio Risk Aggregation")
    print("=" * 80)
    print()

    # Define pricing functions for each position
    def price_10y_treasury(params):
        r = params[0]
        return 10_000_000 * jnp.exp(-r * 10.0)

    def price_5y_swap(params):
        r = params[0]
        return -2_000_000 * r * 5.0

    def price_aapl_cds(params):
        spread = params[0]
        return 5_000_000 * (1 - 0.40) * (1 - jnp.exp(-spread * 5.0))

    def price_msft_cds(params):
        spread = params[0]
        return 3_000_000 * (1 - 0.40) * (1 - jnp.exp(-spread * 5.0))

    def price_eurusd_option(params):
        S = params[0]
        return S * 1_000_000 * 0.55  # Delta of 0.55

    def price_gbpusd_option(params):
        S = params[0]
        return S * 500_000 * 0.48  # Delta of 0.48

    # Define portfolio
    portfolio = [
        {
            'name': '10Y US Treasury',
            'pricing_func': price_10y_treasury,
            'params': jnp.array([0.035]),
            'type': 'rates',
            'quantity': 1.0,
            'rate_indices': [0],
            'curve_name': 'USD-Treasury'
        },
        {
            'name': '5Y IRS (Payer)',
            'pricing_func': price_5y_swap,
            'params': jnp.array([0.030]),
            'type': 'rates',
            'quantity': 1.0,
            'rate_indices': [0],
            'curve_name': 'USD-SOFR'
        },
        {
            'name': 'AAPL 5Y CDS',
            'pricing_func': price_aapl_cds,
            'params': jnp.array([0.0120]),
            'type': 'credit',
            'quantity': 1.0,
            'spread_indices': [0],
            'entity': 'AAPL'
        },
        {
            'name': 'MSFT 5Y CDS',
            'pricing_func': price_msft_cds,
            'params': jnp.array([0.0100]),
            'type': 'credit',
            'quantity': 1.0,
            'spread_indices': [0],
            'entity': 'MSFT'
        },
        {
            'name': 'EUR/USD 1Y Call',
            'pricing_func': price_eurusd_option,
            'params': jnp.array([1.10]),
            'type': 'fx',
            'quantity': 1.0,
            'spot_index': 0,
            'tenor': 1.0
        },
        {
            'name': 'GBP/USD 6M Call',
            'pricing_func': price_gbpusd_option,
            'params': jnp.array([1.25]),
            'type': 'fx',
            'quantity': 1.0,
            'spot_index': 0,
            'tenor': 0.5
        },
    ]

    # Aggregate sensitivities
    portfolio_sens = aggregate_portfolio_sensitivities(portfolio)

    # Print formatted report
    report = format_sensitivity_report(portfolio_sens, include_breakdown=True)
    print(report)

    print()
    print("Risk Analysis:")
    print(f"  - DV01: ${portfolio_sens.total_dv01:,.2f}")
    print("    → Mixed rate exposure across Treasury and SOFR curves")
    print(f"  - CS01: ${portfolio_sens.total_cs01:,.2f}")
    print("    → Credit risk concentrated in tech names (AAPL, MSFT)")
    print(f"  - FX Delta: {portfolio_sens.total_delta:,.2f}")
    print("    → Net long EUR and GBP against USD")
    print()


def example_6_method_comparison():
    """Example 6: Comparing sensitivity calculation methods."""
    print("=" * 80)
    print("EXAMPLE 6: Comparing Calculation Methods")
    print("=" * 80)
    print()

    def price_option(params):
        S = params[0]
        K = 100.0
        r = 0.05
        T = 1.0
        vol = 0.20

        d1 = (jnp.log(S/K) + (r + 0.5 * vol**2) * T) / (vol * jnp.sqrt(T))
        d2 = d1 - vol * jnp.sqrt(T)

        from jax.scipy.stats import norm
        return S * norm.cdf(d1) - K * jnp.exp(-r*T) * norm.cdf(d2)

    params = jnp.array([100.0])

    # Method 1: Automatic Differentiation (JAX)
    config_autodiff = SensitivityConfig(method=SensitivityMethod.AUTODIFF)
    delta_autodiff = calculate_fx_greeks(price_option, params, spot_index=0,
                                          config=config_autodiff)

    # Method 2: Central Finite Difference
    from neutryx.risk.sensitivity_analysis import FiniteDiffScheme
    config_fd_central = SensitivityConfig(
        method=SensitivityMethod.FINITE_DIFF,
        fd_scheme=FiniteDiffScheme.CENTRAL,
        bump_size=0.01
    )
    delta_fd_central = calculate_fx_greeks(price_option, params, spot_index=0,
                                            config=config_fd_central)

    # Method 3: Forward Finite Difference
    config_fd_forward = SensitivityConfig(
        method=SensitivityMethod.FINITE_DIFF,
        fd_scheme=FiniteDiffScheme.FORWARD,
        bump_size=0.01
    )
    delta_fd_forward = calculate_fx_greeks(price_option, params, spot_index=0,
                                            config=config_fd_forward)

    print("Option: ATM Call, S=K=100, T=1Y, vol=20%")
    print()
    print("Delta calculation methods:")
    print(f"  Automatic Differentiation (JAX):  {delta_autodiff['delta']:.6f}")
    print(f"  Central Finite Difference:        {delta_fd_central['delta']:.6f}")
    print(f"  Forward Finite Difference:        {delta_fd_forward['delta']:.6f}")
    print()
    print("Gamma calculation methods:")
    print(f"  Automatic Differentiation (JAX):  {delta_autodiff['gamma']:.8f}")
    print(f"  Central Finite Difference:        {delta_fd_central['gamma']:.8f}")
    print()
    print("Method Comparison:")
    print("  - Autodiff: Exact derivatives, fast for complex models")
    print("  - Central FD: Good accuracy, requires 2 function evaluations")
    print("  - Forward FD: Less accurate, only 1 extra function evaluation")
    print("  - Use autodiff when available (JAX), fall back to central FD")
    print()


if __name__ == "__main__":
    """Run all examples."""
    examples = [
        example_1_dv01_interest_rate_swap,
        example_2_cs01_cds_portfolio,
        example_3_vega_surface_options,
        example_4_fx_greeks,
        example_5_portfolio_aggregation,
        example_6_method_comparison,
    ]

    for example_func in examples:
        example_func()
        print()

    print("=" * 80)
    print("All examples completed!")
    print("=" * 80)
