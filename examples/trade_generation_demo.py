"""
Convention-Based Trade Generation - Demo

このデモは、マーケット慣行に基づいた取引生成システムの使用方法を示します。
"""

from datetime import date
from neutryx.portfolio.trade_generation.generators import (
    generate_irs_trade,
    generate_ois_trade,
)

print("=" * 70)
print("Convention-Based Trade Generation Demo")
print("=" * 70)

# Example 1: USD 金利スワップ（標準的な市場慣行）
print("\n[Example 1] USD Interest Rate Swap (Standard Conventions)")
print("-" * 70)

trade, product, result = generate_irs_trade(
    currency="USD",
    trade_date=date(2024, 1, 15),
    tenor="5Y",
    notional=10_000_000,
    fixed_rate=0.045,  # 4.5%
    counterparty_id="CP-001",
    swap_type="PAYER",
)

print(f"[OK] Trade Created: {trade.id}")
print(f"   Product Type: {trade.product_type.value}")
print(f"   Convention Profile: {trade.convention_profile_id}")
print(f"   Currency: {trade.currency}")
print(f"   Notional: ${trade.notional:,.0f}")
print(f"   Trade Date: {trade.trade_date}")
print(f"   Effective Date: {trade.effective_date}")
print(f"   Maturity Date: {trade.maturity_date}")
print(f"\n   Fixed Leg:")
print(f"      Frequency: Semi-annual (2 payments/year)")
print(f"      Day Count: 30/360")
print(f"   Floating Leg:")
print(f"      Frequency: Quarterly (4 payments/year)")
print(f"      Day Count: ACT/360")
print(f"      Index: SOFR (compounded)")

if result.has_warnings():
    print(f"\n[WARNING]  Warnings: {len(result.get_warnings())}")
else:
    print(f"\n[CHECK] No convention warnings (standard market conventions used)")

# Example 2: EUR OIS スワップ
print("\n\n[Example 2] EUR Overnight Index Swap (ESTR)")
print("-" * 70)

trade2, product2, result2 = generate_ois_trade(
    currency="EUR",
    trade_date=date(2024, 1, 15),
    tenor="2Y",
    notional=5_000_000,
    fixed_rate=0.035,  # 3.5%
    counterparty_id="CP-002",
    swap_type="RECEIVER",
)

print(f"[OK] Trade Created: {trade2.id}")
print(f"   Product Type: {trade2.product_type.value}")
print(f"   Convention Profile: {trade2.convention_profile_id}")
print(f"   Currency: {trade2.currency}")
print(f"   Notional: €{trade2.notional:,.0f}")
print(f"   Maturity: {trade2.maturity_date}")
print(f"\n   Both Legs: Annual, ACT/360")
print(f"   Floating Index: ESTR (compounded)")

# Example 3: 非標準的な慣行（警告付き）
print("\n\n[Example 3] USD IRS with Non-Standard Conventions")
print("-" * 70)

from neutryx.core.dates.schedule import Frequency
from neutryx.core.dates.day_count import ACT_365

trade3, product3, result3 = generate_irs_trade(
    currency="USD",
    trade_date=date(2024, 1, 15),
    tenor="3Y",
    notional=8_000_000,
    fixed_rate=0.042,
    counterparty_id="CP-003",
    # 非標準的な設定
    fixed_leg_frequency=Frequency.QUARTERLY,  # 通常はSemi-annual
    floating_leg_day_count=ACT_365,  # 通常はACT/360
)

print(f"[OK] Trade Created: {trade3.id}")
print(f"   Convention Profile: {trade3.convention_profile_id}")

if result3.has_warnings():
    print(f"\n[WARNING]  Convention Warnings ({len(result3.get_warnings())}):")
    for warning in result3.get_warnings():
        print(f"   [{warning.severity.value.upper()}] {warning.field}")
        print(f"      {warning.message}")
        if warning.expected and warning.actual:
            print(f"      Expected: {warning.expected}")
            print(f"      Actual: {warning.actual}")

print("\n" + "=" * 70)
print("Demo Complete!")
print("=" * 70)
print("\n[TIP] Key Features:")
print("   • Automatic application of market conventions")
print("   • Support for convention overrides")
print("   • Warning system for non-standard trades")
print("   • Multi-currency support (USD, EUR, GBP, JPY, CHF)")
print("   • Multiple product types (IRS, OIS, and more coming)")
