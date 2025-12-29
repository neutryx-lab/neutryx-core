# Bank Trading System - Repository and Execution Framework

## 概要

このドキュメントでは、Neutryx Coreに実装されたBank Trading System（銀行取引システム）について説明します。このシステムは、銀行のトレーディング業務に必要な以下の機能を提供します：

1. **TradeRepository** - 取引データの永続化
2. **ClientRepository** - クライアント（取引相手）とCSA契約の管理
3. **Connection Manager** - データベース接続の統合管理
4. **Trade Execution Service** - 取引実行とライフサイクル管理

## アーキテクチャ

```
┌─────────────────────────────────────────────────────────┐
│           Trade Execution Service                       │
│  (取引実行、検証、ライフサイクル管理)                      │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│        Bank Connection Manager                          │
│  (統合接続管理、トランザクション調整)                      │
└──┬─────────────┬──────────────┬────────────────────────┘
   │             │              │
┌──▼──────────┐ ┌▼──────────┐ ┌▼─────────────────────┐
│Trade        │ │Counterparty│ │CSA Repository        │
│Repository   │ │Repository  │ │(CSA契約管理)          │
└──┬──────────┘ └┬───────────┘ └┬─────────────────────┘
   │             │              │
┌──▼─────────────▼──────────────▼─────────────────────┐
│           PostgreSQL Database                       │
│  Schema: trading, clients                           │
└─────────────────────────────────────────────────────┘
```

## 主要コンポーネント

### 1. PostgresTradeRepository

PostgreSQLベースの取引リポジトリ実装。

**場所**: `src/neutryx/portfolio/repository_postgres.py`

**機能**:
- 取引データの永続化（ACID準拠）
- 高度なクエリ機能
- インデックスによる高速検索
- JSONB形式での商品詳細の保存

**主要メソッド**:
```python
async def save_async(trade: Trade) -> None
async def find_by_id_async(trade_id: str) -> Optional[Trade]
async def find_by_counterparty_async(counterparty_id: str) -> List[Trade]
async def find_by_book_async(book_id: str) -> List[Trade]
async def find_by_status_async(status: TradeStatus) -> List[Trade]
async def delete_async(trade_id: str) -> bool
async def count_async() -> int
```

**データベーススキーマ**:
```sql
CREATE TABLE trading.trades (
    id TEXT PRIMARY KEY,
    trade_number TEXT,
    external_id TEXT,
    usi TEXT,
    counterparty_id TEXT NOT NULL,
    netting_set_id TEXT,
    book_id TEXT,
    desk_id TEXT,
    trader_id TEXT,
    product_type TEXT NOT NULL,
    trade_date DATE NOT NULL,
    effective_date DATE,
    maturity_date DATE,
    status TEXT NOT NULL,
    notional NUMERIC,
    currency TEXT,
    settlement_type TEXT,
    product_details JSONB,
    mtm NUMERIC,
    last_valuation_date DATE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

### 2. PostgresCounterpartyRepository

取引相手（クライアント）の管理リポジトリ。

**場所**: `src/neutryx/portfolio/client_repository.py`

**機能**:
- 取引相手情報の永続化
- 信用プロファイルの管理
- LEI（Legal Entity Identifier）によるインデックス
- 親子エンティティ関係のサポート

**主要メソッド**:
```python
async def save_async(counterparty: Counterparty) -> None
async def find_by_id_async(counterparty_id: str) -> Optional[Counterparty]
async def find_by_lei_async(lei: str) -> Optional[Counterparty]
async def find_all_async() -> List[Counterparty]
async def find_banks_async() -> List[Counterparty]
```

**データベーススキーマ**:
```sql
CREATE TABLE clients.counterparties (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    lei TEXT UNIQUE,
    jurisdiction TEXT,
    parent_id TEXT,
    is_bank BOOLEAN DEFAULT FALSE,
    is_clearinghouse BOOLEAN DEFAULT FALSE,
    credit_rating TEXT,
    internal_rating TEXT,
    lgd NUMERIC,
    recovery_rate NUMERIC,
    credit_spread_bps NUMERIC,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

### 3. PostgresCSARepository

CSA（Credit Support Annex）契約の管理リポジトリ。

**場所**: `src/neutryx/portfolio/client_repository.py`

**機能**:
- CSA契約条件の永続化
- 担保条件（Threshold、MTA）の管理
- 適格担保仕様の管理
- 両者間関係の管理

**主要メソッド**:
```python
async def save_async(csa: CSA) -> None
async def find_by_id_async(csa_id: str) -> Optional[CSA]
async def find_by_counterparty_async(counterparty_id: str) -> List[CSA]
async def find_by_parties_async(party_a_id: str, party_b_id: str) -> Optional[CSA]
async def find_all_async() -> List[CSA]
async def delete_async(csa_id: str) -> bool
```

**データベーススキーマ**:
```sql
CREATE TABLE clients.csa_agreements (
    id TEXT PRIMARY KEY,
    party_a_id TEXT NOT NULL,
    party_b_id TEXT NOT NULL,
    effective_date TEXT NOT NULL,
    threshold_party_a NUMERIC DEFAULT 0,
    threshold_party_b NUMERIC DEFAULT 0,
    mta_party_a NUMERIC DEFAULT 0,
    mta_party_b NUMERIC DEFAULT 0,
    independent_amount_party_a NUMERIC DEFAULT 0,
    independent_amount_party_b NUMERIC DEFAULT 0,
    rounding NUMERIC DEFAULT 100000,
    base_currency TEXT NOT NULL,
    valuation_frequency TEXT DEFAULT 'Daily',
    initial_margin_required BOOLEAN DEFAULT FALSE,
    variation_margin_required BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE clients.eligible_collateral (
    id SERIAL PRIMARY KEY,
    csa_id TEXT REFERENCES clients.csa_agreements(id),
    collateral_type TEXT NOT NULL,
    currency TEXT,
    haircut NUMERIC DEFAULT 0,
    concentration_limit NUMERIC,
    rating_threshold TEXT,
    maturity_max_years NUMERIC
);
```

### 4. BankConnectionManager

すべてのリポジトリの統合接続管理。

**場所**: `src/neutryx/portfolio/bank_connection_manager.py`

**機能**:
- 統一された接続管理
- トランザクション調整
- 優雅なライフサイクル管理
- 設定の共有

**使用例**:
```python
from neutryx.integrations.databases.base import DatabaseConfig
from neutryx.portfolio.bank_connection_manager import BankConnectionManager

# データベース設定
config = DatabaseConfig(
    host="localhost",
    port=5432,
    database="neutryx_bank",
    user="postgres",
    password="postgres",
)

# 接続マネージャーの作成
manager = BankConnectionManager(
    config=config,
    trade_schema="trading",
    client_schema="clients",
)

# 接続と使用
async with manager.connection():
    # 取引の保存
    await manager.trade_repo.save_async(trade)

    # 取引相手の取得
    counterparty = await manager.counterparty_repo.find_by_id_async("CP001")

    # CSA契約の検索
    csas = await manager.csa_repo.find_by_counterparty_async("CP001")
```

### 5. TradeExecutionService

取引実行とライフサイクル管理サービス。

**場所**: `src/neutryx/portfolio/trade_execution_service.py`

**機能**:
- 取引の検証と実行
- 取引相手とCSAの検証
- 取引ライフサイクル管理（確認、キャンセル、終了）
- バッチ実行のサポート
- エクスポージャー計算

**主要メソッド**:
```python
async def execute_trade(
    trade: Trade,
    validate_counterparty: bool = True,
    validate_csa: bool = False,
    auto_confirm: bool = False,
) -> ExecutionResult

async def book_trade(
    counterparty_id: str,
    product_type: ProductType,
    trade_date: date,
    notional: float,
    currency: str,
    ...
) -> ExecutionResult

async def confirm_trade(trade_id: str) -> ExecutionResult
async def cancel_trade(trade_id: str) -> ExecutionResult
async def terminate_trade(trade_id: str, termination_date: date) -> ExecutionResult
async def get_counterparty_exposure(counterparty_id: str) -> Dict
```

## 使用方法

### セットアップ

1. **依存関係のインストール**:
```bash
pip install asyncpg
```

2. **データベースの作成**:
```sql
CREATE DATABASE neutryx_bank;
```

3. **環境変数の設定**（オプション）:
```bash
export DATABASE_HOST=localhost
export DATABASE_PORT=5432
export DATABASE_NAME=neutryx_bank
export DATABASE_USER=postgres
export DATABASE_PASSWORD=postgres
```

### 基本的な使用例

#### 1. データベースのセットアップ

```python
import asyncio
from neutryx.integrations.databases.base import DatabaseConfig
from neutryx.portfolio.bank_connection_manager import BankConnectionManager

async def setup_database():
    config = DatabaseConfig(
        host="localhost",
        port=5432,
        database="neutryx_bank",
        user="postgres",
        password="postgres",
    )

    manager = BankConnectionManager(config=config)

    try:
        await manager.initialize_schemas()
        print("✓ Database initialized")

        health = await manager.health_check()
        print(f"✓ Health: {health}")
    finally:
        await manager.disconnect()

asyncio.run(setup_database())
```

#### 2. 取引相手の作成

```python
from neutryx.portfolio.contracts.counterparty import (
    Counterparty, CounterpartyCredit, CreditRating, EntityType
)

async def create_counterparty():
    manager = BankConnectionManager(config=config)

    try:
        await manager.connect()

        counterparty = Counterparty(
            id="CP001",
            name="ABC Corporation",
            entity_type=EntityType.CORPORATE,
            lei="ABC123456789CORPORATE",
            jurisdiction="US",
            credit=CounterpartyCredit(
                rating=CreditRating.A,
                lgd=0.4,
                credit_spread_bps=150.0,
            ),
        )

        await manager.counterparty_repo.save_async(counterparty)
        print(f"✓ Created: {counterparty}")
    finally:
        await manager.disconnect()
```

#### 3. CSA契約の作成

```python
from neutryx.portfolio.contracts.csa import (
    CSA, CollateralTerms, ThresholdTerms, EligibleCollateral,
    CollateralType, ValuationFrequency
)

async def create_csa():
    manager = BankConnectionManager(config=config)

    try:
        await manager.connect()

        csa = CSA(
            id="CSA001",
            party_a_id="BANK001",  # 自行
            party_b_id="CP001",    # 取引相手
            effective_date="2025-01-01",
            threshold_terms=ThresholdTerms(
                threshold_party_a=0.0,
                threshold_party_b=5_000_000.0,
                mta_party_a=250_000.0,
                mta_party_b=250_000.0,
            ),
            collateral_terms=CollateralTerms(
                base_currency="USD",
                valuation_frequency=ValuationFrequency.DAILY,
                eligible_collateral=[
                    EligibleCollateral(
                        collateral_type=CollateralType.CASH,
                        currency="USD",
                        haircut=0.0,
                    ),
                ],
            ),
        )

        await manager.csa_repo.save_async(csa)
        print(f"✓ Created CSA: {csa}")
    finally:
        await manager.disconnect()
```

#### 4. 取引の実行

```python
from neutryx.portfolio.trade_execution_service import TradeExecutionService
from neutryx.portfolio.contracts.trade import ProductType
from datetime import date, timedelta

async def execute_trade():
    manager = BankConnectionManager(config=config)

    try:
        await manager.connect()

        service = TradeExecutionService(manager)

        # 新規取引の予約
        result = await service.book_trade(
            counterparty_id="CP001",
            product_type=ProductType.INTEREST_RATE_SWAP,
            trade_date=date.today(),
            notional=10_000_000.0,
            currency="USD",
            maturity_date=date.today() + timedelta(days=365*5),
            product_details={
                "fixed_rate": 0.045,
                "floating_index": "USD-LIBOR-3M",
            },
            validate_counterparty=True,
            validate_csa=True,
        )

        if result.is_success():
            print(f"✓ Trade booked: {result.trade_id}")

            # 取引の確認
            confirm_result = await service.confirm_trade(result.trade_id)
            print(f"✓ Trade confirmed: {confirm_result.is_success()}")
        else:
            print(f"✗ Error: {result.error_message}")

    finally:
        await manager.disconnect()
```

#### 5. エクスポージャーの照会

```python
async def query_exposure():
    manager = BankConnectionManager(config=config)
    service = TradeExecutionService(manager)

    try:
        await manager.connect()

        # 取引相手のエクスポージャーを取得
        exposure = await service.get_counterparty_exposure("CP001")

        print(f"Counterparty: {exposure['counterparty'].name}")
        print(f"Active Trades: {exposure['trade_count']}")
        print(f"Total MTM: ${exposure['total_mtm']:,.2f}")
        print(f"CSA Agreements: {len(exposure['csas'])}")

    finally:
        await manager.disconnect()
```

## 完全な例

詳細な使用例は以下を参照してください：

**ファイル**: `examples/bank_trading_example.py`

このファイルには以下の例が含まれています：
1. データベースのセットアップと初期化
2. 取引相手の作成と管理
3. CSA契約の作成
4. 取引の実行
5. 取引とエクスポージャーの照会
6. 取引ライフサイクル管理

実行方法:
```bash
python examples/bank_trading_example.py
```

## データモデル

### Trade（取引）

主要な属性:
- `id`: 取引ID
- `counterparty_id`: 取引相手ID
- `product_type`: 商品タイプ（IRS、FXオプション等）
- `trade_date`: 取引日
- `maturity_date`: 満期日
- `notional`: 想定元本
- `currency`: 通貨
- `status`: ステータス（Pending, Active, Terminated等）
- `mtm`: 時価評価額

### Counterparty（取引相手）

主要な属性:
- `id`: 取引相手ID
- `name`: 名称
- `entity_type`: エンティティタイプ
- `lei`: Legal Entity Identifier
- `jurisdiction`: 管轄
- `credit`: 信用プロファイル（格付け、LGD等）
- `is_bank`: 銀行フラグ

### CSA（Credit Support Annex）

主要な属性:
- `id`: CSA ID
- `party_a_id`: 当事者A
- `party_b_id`: 当事者B
- `threshold_terms`: 閾値条件（Threshold、MTA等）
- `collateral_terms`: 担保条件
- `initial_margin_required`: 当初証拠金要否
- `variation_margin_required`: 変動証拠金要否

## セキュリティとベストプラクティス

1. **接続情報の保護**:
   - データベース認証情報は環境変数で管理
   - 本番環境ではシークレット管理サービスを使用

2. **トランザクション管理**:
   - 関連する操作は単一トランザクションで実行
   - エラー時の適切なロールバック

3. **バリデーション**:
   - 取引実行前の取引相手検証
   - CSA契約の存在確認
   - 重複取引IDのチェック

4. **監査ログ**:
   - すべてのテーブルに`created_at`、`updated_at`タイムスタンプ
   - 取引ステータス変更の追跡

5. **接続プーリング**:
   - asyncpgの接続プールを活用
   - 効率的なリソース管理

## トラブルシューティング

### 接続エラー

```python
# エラー: asyncpg not found
# 解決: pip install asyncpg

# エラー: Database connection failed
# 解決: データベースが起動しているか確認
# psql -h localhost -U postgres -d neutryx_bank
```

### スキーマ作成エラー

```python
# エラー: Permission denied
# 解決: ユーザーにスキーマ作成権限を付与
# GRANT CREATE ON DATABASE neutryx_bank TO postgres;
```

## 今後の拡張

1. **追加機能**:
   - トレードのバージョニング
   - 監査ログの詳細化
   - リアルタイム通知
   - ワークフロー承認機能

2. **パフォーマンス最適化**:
   - パーティショニング（取引日による）
   - マテリアライズドビュー
   - キャッシング戦略

3. **統合**:
   - FpMLインポート/エクスポート
   - 外部決済システム連携
   - リスク計算エンジン統合

## 参考資料

- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [asyncpg Documentation](https://magicstack.github.io/asyncpg/)
- [ISDA Documentation](https://www.isda.org/)
- [Neutryx Core Documentation](../README.md)
