## Fictional Bank - Complete Trading Infrastructure

## 概要

Fictional Bankは、Neutryx Coreの包括的な参照実装であり、実際の銀行トレーディングシステムをシミュレートします。Bank Trading Systemをベースに構築されており、以下の機能を提供します：

- **完全なポートフォリオ管理**：複数デスク、トレーダー、取引先の管理
- **データベース永続化**：PostgreSQLによる取引、クライアント、CSAの永続化
- **取引実行ワークフロー**：予約、確認、キャンセル、終了の完全なライフサイクル
- **リアルタイムモニタリング**：エクスポージャー、リスク集中度の監視
- **包括的レポート**：P&L、リスク、パフォーマンス、規制レポート
- **トレーディングシナリオ**：日次取引、オンボーディング、ストレステスト

## アーキテクチャ

```
┌─────────────────────────────────────────────────────────────┐
│              Fictional Bank                                 │
│  (統合されたトレーディングバンク実装)                           │
└────────┬────────────────────────────────────────────────────┘
         │
         ├──► Bank Trading System
         │    ├── TradeRepository (PostgreSQL)
         │    ├── CounterpartyRepository (PostgreSQL)
         │    └── CSARepository (PostgreSQL)
         │
         ├──► Portfolio Management
         │    ├── In-memory Portfolio
         │    ├── Book Hierarchy
         │    └── Netting Sets
         │
         ├──► Trading Scenarios
         │    ├── Daily Trading
         │    ├── Counterparty Onboarding
         │    ├── Portfolio Rebalancing
         │    └── Stress Testing
         │
         └──► Reporting & Analytics
              ├── Daily P&L Reports
              ├── Risk Reports
              ├── Desk Performance
              └── Executive Dashboard
```

## 主要コンポーネント

### 1. FictionalBank

メインの銀行クラス。すべての機能を統合します。

**場所**: `src/neutryx/portfolio/fictional_bank.py`

**主要機能**:
- 銀行の初期化とセットアップ
- 架空のポートフォリオのロード
- 取引先とCSAの管理
- 取引実行
- エクスポージャー計算
- 日次レポート生成

**使用例**:
```python
from neutryx.integrations.databases.base import DatabaseConfig
from neutryx.portfolio.fictional_bank import create_fictional_bank

# データベース設定
config = DatabaseConfig(
    host="localhost",
    port=5432,
    database="neutryx_bank",
    user="postgres",
    password="postgres",
)

# 架空銀行の作成と初期化
bank = await create_fictional_bank(
    database_config=config,
    load_portfolio=True,  # 標準的な架空ポートフォリオをロード
)

# 健全性チェック
health = await bank.health_check()
print(f"Status: {health}")
```

### 2. Trading Scenarios

様々なトレーディングシナリオの実装。

**場所**: `src/neutryx/portfolio/trading_scenarios.py`

**利用可能なシナリオ**:

#### DailyTradingScenario
通常の取引日をシミュレート。全デスクで複数の取引を実行。

```python
from neutryx.portfolio.trading_scenarios import DailyTradingScenario

scenario = DailyTradingScenario()
result = await scenario.execute(bank)
```

#### CounterpartyOnboardingScenario
新規取引先のオンボーディング：
1. 新規取引先の作成
2. CSA契約の締結
3. 初回取引の実行

```python
from neutryx.portfolio.trading_scenarios import CounterpartyOnboardingScenario

scenario = CounterpartyOnboardingScenario()
result = await scenario.execute(bank)
```

#### PortfolioRebalancingScenario
ポートフォリオのリバランシング：
- 古いポジションの終了
- 新規ポジションの確立

```python
from neutryx.portfolio.trading_scenarios import PortfolioRebalancingScenario

scenario = PortfolioRebalancingScenario()
result = await scenario.execute(bank)
```

#### StressTestScenario
高負荷取引のストレステスト。

```python
from neutryx.portfolio.trading_scenarios import StressTestScenario

scenario = StressTestScenario(num_trades=50)
result = await scenario.execute(bank)
```

#### ExposureMonitoringScenario
取引先エクスポージャーの監視と報告。

```python
from neutryx.portfolio.trading_scenarios import ExposureMonitoringScenario

scenario = ExposureMonitoringScenario()
result = await scenario.execute(bank)
```

### 3. Bank Reports

包括的なレポート生成機能。

**場所**: `src/neutryx/portfolio/bank_reports.py`

**利用可能なレポート**:

#### 日次P&Lレポート
```python
from neutryx.portfolio.bank_reports import BankReportGenerator

report_gen = BankReportGenerator(bank)
pnl_report = await report_gen.generate_daily_pnl_report()

# レポート内容:
# - 総MTM
# - 総想定元本
# - デスク別P&L
# - 商品別P&L
# - 取引先別P&L
```

#### リスクレポート
```python
risk_report = await report_gen.generate_risk_report()

# レポート内容:
# - 取引先エクスポージャー
# - 集中度メトリクス
# - 信用メトリクス
# - 格付別分布
# - トップエクスポージャー
```

#### デスクパフォーマンスレポート
```python
desk_report = await report_gen.generate_desk_performance_report()

# レポート内容:
# - デスク別統計
# - ブック情報
# - トレーダー情報
# - パフォーマンス指標
```

#### 取引先信用レポート
```python
credit_report = await report_gen.generate_counterparty_credit_report()

# レポート内容:
# - 取引先プロファイル
# - 信用格付
# - CSAステータス
# - エクスポージャー
```

#### エグゼクティブダッシュボード
```python
dashboard = await report_gen.generate_executive_dashboard()
report_gen.print_executive_dashboard(dashboard)

# ダッシュボード内容:
# - 主要指標
# - デスクパフォーマンス
# - トップエクスポージャー
# - 格付分布
```

## 標準的な架空ポートフォリオ

Fictional Bankは、包括的なテストポートフォリオを含みます：

### 組織構造

**Legal Entity**:
- Global Investment Bank Ltd (LEI: 529900T8BM49AURSDO55)

**Business Unit**:
- Global Trading

**デスク** (3つ):
1. **Interest Rates Desk** (DESK_RATES)
   - USD IRS Book
   - EUR IRS Book
   - Swaptions Book

2. **Foreign Exchange Desk** (DESK_FX)
   - FX Majors Book
   - FX Emerging Markets Book

3. **Equity Derivatives Desk** (DESK_EQUITY)
   - Equity Vanilla Options Book
   - Equity Exotic Options Book

**トレーダー** (6名):
- Alice Chen (Rates)
- Bob Martinez (Rates)
- Carol Zhang (FX)
- David Kim (FX)
- Emma Wilson (Equity)
- Frank Johnson (Equity)

### 取引先 (6社)

1. **AAA Global Bank** (CP_BANK_AAA)
   - Type: Financial
   - Rating: AAA
   - CSA: Yes

2. **Tech Corporation A** (CP_CORP_A)
   - Type: Corporate
   - Rating: A
   - CSA: Yes

3. **Industrial Group BBB** (CP_CORP_BBB)
   - Type: Corporate
   - Rating: BBB
   - CSA: No

4. **Alpha Strategies Fund** (CP_HEDGE_FUND)
   - Type: Fund
   - Rating: A-
   - CSA: No

5. **Republic Investment Authority** (CP_SOVEREIGN)
   - Type: Sovereign
   - Rating: AA+
   - CSA: Yes

6. **Global Insurance Group** (CP_INSURANCE)
   - Type: Financial
   - Rating: AA
   - CSA: Yes

### 取引 (11件)

- **金利スワップ** (3件): USD、EUR
- **スワプション** (1件): USD
- **FXオプション** (3件): EUR/USD、USD/JPY、USD/BRL
- **株式オプション** (3件): SPX、AAPL、TSLA
- **バリアンススワップ** (1件): SPX

## 使用方法

### 基本的な使用例

```python
import asyncio
from neutryx.integrations.databases.base import DatabaseConfig
from neutryx.portfolio.fictional_bank import create_fictional_bank

async def main():
    # 1. データベース設定
    config = DatabaseConfig(
        host="localhost",
        port=5432,
        database="neutryx_bank",
        user="postgres",
        password="postgres",
    )

    # 2. 銀行の作成と初期化
    bank = await create_fictional_bank(
        database_config=config,
        load_portfolio=True,
    )

    # 3. 取引の実行
    from neutryx.portfolio.contracts.trade import ProductType
    from datetime import date, timedelta

    result = await bank.book_trade(
        counterparty_id="CP_BANK_AAA",
        product_type=ProductType.INTEREST_RATE_SWAP,
        trade_date=date.today(),
        notional=10_000_000.0,
        currency="USD",
        maturity_date=date.today() + timedelta(days=365*5),
        auto_confirm=True,
    )

    print(f"Trade result: {result.status}")

    # 4. エクスポージャーの照会
    exposure = await bank.get_counterparty_exposure("CP_BANK_AAA")
    print(f"Exposure: ${exposure['total_mtm']:,.2f}")

    # 5. レポート生成
    from neutryx.portfolio.bank_reports import BankReportGenerator

    report_gen = BankReportGenerator(bank)
    dashboard = await report_gen.generate_executive_dashboard()
    report_gen.print_executive_dashboard(dashboard)

    # 6. クリーンアップ
    await bank.shutdown()

asyncio.run(main())
```

### トレーディングシナリオの実行

```python
from neutryx.portfolio.trading_scenarios import run_all_scenarios

async def run_scenarios():
    bank = await create_fictional_bank(
        database_config=config,
        load_portfolio=True,
    )

    # すべてのシナリオを実行
    results = await run_all_scenarios(bank)

    print("Scenario Results:")
    for scenario_name, result in results.items():
        print(f"  {scenario_name}: {result}")

    await bank.shutdown()
```

### カスタム取引先の追加

```python
from neutryx.portfolio.contracts.counterparty import (
    Counterparty, CounterpartyCredit, CreditRating, EntityType
)

async def add_custom_counterparty():
    bank = await create_fictional_bank(database_config=config)

    # 新規取引先の作成
    new_cp = Counterparty(
        id="CP_CUSTOM",
        name="Custom Client Corp",
        entity_type=EntityType.CORPORATE,
        lei="CUSTOM12345678901234",
        jurisdiction="JP",
        credit=CounterpartyCredit(
            rating=CreditRating.A_PLUS,
            lgd=0.45,
            credit_spread_bps=90.0,
        ),
    )

    # 銀行に追加（データベースに永続化）
    await bank.add_counterparty(new_cp, persist=True)

    await bank.shutdown()
```

## 完全なデモンストレーション

包括的なデモが用意されています：

**ファイル**: `examples/fictional_bank_demo.py`

このデモには以下が含まれます：
1. 銀行の初期化とセットアップ
2. トレーディングシナリオの実行
3. 包括的レポートの生成
4. 取引先分析
5. データベース操作
6. リアルタイムモニタリング

実行方法:
```bash
# データベースが起動していることを確認
# PostgreSQL 12+ が必要

# データベースの作成
createdb neutryx_bank

# デモの実行
python examples/fictional_bank_demo.py
```

出力例:
```
================================================================================
FICTIONAL BANK - COMPREHENSIVE DEMONSTRATION
================================================================================

DEMO 1: Bank Initialization
================================================================================
✓ Global Investment Bank Ltd initialized
✓ Loaded fictional portfolio:
  - 6 counterparties
  - 4 CSA agreements
  - 11 trades
  - 7 books

📊 Health Check:
  Status: OK
  Database Connected: True
  Counterparties: 6
  Trades: 11
  CSAs: 4

...
```

## データモデル

### FictionalBank

```python
class FictionalBank:
    name: str                           # 銀行名
    lei: str                            # Legal Entity Identifier
    jurisdiction: str                   # 管轄
    manager: BankConnectionManager      # データベース接続マネージャー
    execution_service: TradeExecutionService  # 取引実行サービス
    portfolio: Portfolio                # インメモリポートフォlio
    book_hierarchy: BookHierarchy       # 組織構造
```

### Portfolio Structure

標準的なポートフォリオ構造：
- 1 Legal Entity
- 1 Business Unit
- 3 Desks
- 6 Traders
- 7 Books
- 6 Counterparties
- 4 CSA Agreements
- 6 Netting Sets
- 11+ Trades

## パフォーマンス

### ストレステスト結果

典型的なパフォーマンス（PostgreSQL、ローカルマシン）：

- **取引実行**: 20-30 取引/秒
- **データベースクエリ**: < 10ms（単一取引）
- **レポート生成**: 50-100ms（エグゼクティブダッシュボード）
- **バッチ取引**: 100件の取引を3-5秒で実行

### メモリ使用量

- ベースライン: ~50MB
- 1000件の取引: ~100MB
- 10000件の取引: ~500MB

## 拡張とカスタマイズ

### カスタムシナリオの作成

```python
from neutryx.portfolio.trading_scenarios import TradingScenario

class CustomScenario(TradingScenario):
    def __init__(self):
        super().__init__(
            name="Custom Scenario",
            description="Your custom trading scenario",
        )

    async def execute(self, bank: FictionalBank) -> Dict:
        # カスタムロジックを実装
        self.results = []

        # 取引を実行
        result = await bank.book_trade(...)
        self.results.append(result)

        self.print_results()
        return {"scenario": self.name}
```

### カスタムレポートの作成

```python
from neutryx.portfolio.bank_reports import BankReportGenerator

class CustomReportGenerator(BankReportGenerator):
    async def generate_custom_report(self) -> Dict:
        # カスタムレポートロジック
        report = {
            "report_type": "Custom Report",
            "bank_name": self.bank.name,
            # カスタムデータ
        }
        return report
```

## トラブルシューティング

### データベース接続エラー

```python
# エラー: Could not connect to database
# 解決策:
# 1. PostgreSQLが起動しているか確認
# 2. 接続情報が正しいか確認
# 3. データベースが存在するか確認

# 確認コマンド:
# psql -h localhost -U postgres -d neutryx_bank
```

### 取引実行エラー

```python
# エラー: Counterparty not found
# 解決策: 取引先が存在することを確認

# 確認:
counterparty = await bank.manager.counterparty_repo.find_by_id_async("CP_ID")
if not counterparty:
    print("Counterparty not found. Create it first.")
```

### メモリエラー

```python
# 大量の取引を処理する場合
# 解決策: バッチサイズを制限

# 良い例:
for batch in chunks(large_trade_list, batch_size=100):
    await bank.execution_service.batch_execute_trades(batch)
```

## ベストプラクティス

1. **接続管理**:
   - 常に`async with`またはtry/finallyを使用
   - `shutdown()`を呼び出してクリーンアップ

2. **取引実行**:
   - 本番環境では`validate_counterparty=True`を使用
   - CSAが必要な場合は`validate_csa=True`を設定

3. **レポート生成**:
   - 大規模ポートフォリオでは非同期レポート生成を使用
   - レポートをJSONで保存して後で分析

4. **データベース**:
   - 定期的にバックアップ
   - インデックスを適切に設定
   - クエリパフォーマンスを監視

5. **テスト**:
   - ユニットテストには`in_memory`モードを使用
   - 統合テストにはデータベースを使用
   - トランザクションロールバックでテストデータをクリーンアップ

## 統合

### XVA計算との統合

```python
# Fictional Bankポートフォリオを使用してXVA計算
from neutryx.xva import CVACalculator

async def calculate_portfolio_cva():
    bank = await create_fictional_bank(database_config=config)

    cva_calculator = CVACalculator()

    for cp_id in bank.portfolio.counterparties.keys():
        trades = bank.portfolio.get_trades_by_counterparty(cp_id)
        counterparty = bank.portfolio.get_counterparty(cp_id)

        cva = cva_calculator.calculate_cva(
            trades=trades,
            counterparty=counterparty,
        )

        print(f"CVA for {counterparty.name}: ${cva:,.2f}")
```

### リスク計算との統合

```python
# Fictional Bankポートフォリオを使用してリスク計算
from neutryx.risk import VaRCalculator

async def calculate_portfolio_var():
    bank = await create_fictional_bank(database_config=config)

    var_calculator = VaRCalculator()

    all_trades = list(bank.portfolio.trades.values())

    var_95 = var_calculator.calculate_var(
        trades=all_trades,
        confidence_level=0.95,
    )

    print(f"Portfolio VaR (95%): ${var_95:,.2f}")
```

## 参考資料

- [Bank Trading System Documentation](BANK_TRADING_SYSTEM.md)
- [Portfolio Management Guide](https://github.com/neutryx-lab/neutryx-core/blob/main/src/neutryx/portfolio/README.md)
- [XVA Calculations](https://github.com/neutryx-lab/neutryx-core/blob/main/src/neutryx/xva/README.md)
- [Fictional Portfolio Source](https://github.com/neutryx-lab/neutryx-core/blob/main/tests/fixtures/fictional_portfolio.py)

## サポートとフィードバック

問題や質問がある場合:
1. [GitHub Issues](https://github.com/neutryx-lab/neutryx-core/issues)
2. ドキュメントを確認
3. 例を参照

## まとめ

Fictional Bankは、Neutryx Coreの完全な参照実装であり、以下を提供します：

✅ **完全なトレーディングインフラ**: 取引実行からレポート生成まで
✅ **データベース統合**: PostgreSQLによる永続化
✅ **リアルなシナリオ**: 日次取引からストレステストまで
✅ **包括的レポート**: P&L、リスク、パフォーマンス分析
✅ **拡張可能**: カスタムシナリオとレポートの追加が容易
✅ **本番環境準備完了**: スケーラブルで高パフォーマンス

これにより、開発者はリアルな銀行環境でXVAやリスク計算をテストし、検証することができます。