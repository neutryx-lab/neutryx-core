## Real-Time Market Data Infrastructure

Comprehensive guide to Neutryx's real-time market data feeds, database connectors, and data validation framework.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Market Data Adapters](#market-data-adapters)
- [Database Storage](#database-storage)
- [Data Validation](#data-validation)
- [Real-Time Feeds](#real-time-feeds)
- [Examples](#examples)
- [Production Deployment](#production-deployment)

## Overview

Neutryx provides enterprise-grade market data infrastructure with:

- **Bloomberg & Refinitiv Integration**: Native support for major data vendors
- **Multi-Database Storage**: PostgreSQL, MongoDB, and TimescaleDB connectors
- **Data Validation**: Comprehensive quality checks and anomaly detection
- **Real-Time Feeds**: Streaming data with automatic failover
- **Production-Ready**: Built for high-throughput, low-latency applications

## Quick Start

### Install Dependencies

```bash
# Core dependencies (already in requirements.txt)
pip install asyncpg motor

# Optional: Bloomberg API
pip install blpapi

# Optional: Refinitiv API
pip install refinitiv-dataplatform eikon
```

### Basic Usage

```python
from neutryx.market.adapters import BloombergAdapter, BloombergConfig
from neutryx.market.storage import TimescaleDBStorage, TimescaleDBConfig
from neutryx.market.validation import ValidationPipeline, PriceRangeValidator
from neutryx.market.feeds import FeedManager

# Configure Bloomberg
bloomberg_config = BloombergConfig(
    adapter_name="bloomberg",
    host="localhost",
    port=8194
)

# Configure TimescaleDB
storage_config = TimescaleDBConfig(
    host="localhost",
    port=5432,
    database="market_data",
    username="trader",
    password="secret",
    compression_enabled=True
)

# Create components
adapter = BloombergAdapter(bloomberg_config)
storage = TimescaleDBStorage(storage_config)

# Setup validation
pipeline = ValidationPipeline()
pipeline.add_validator(PriceRangeValidator(min_price=0, max_price=10000))

# Create feed manager
manager = FeedManager(
    adapters=[adapter],
    storage=storage,
    validation_pipeline=pipeline
)

# Start and subscribe
await manager.start()
await manager.subscribe("equity", ["AAPL", "MSFT", "GOOGL"])
```

## Market Data Adapters

### Bloomberg Adapter

Connect to Bloomberg Terminal or Bloomberg Server API (SAPI/BPIPE).

```python
from neutryx.market.adapters import BloombergAdapter, BloombergConfig

config = BloombergConfig(
    adapter_name="bloomberg",
    host="localhost",  # Bloomberg API host
    port=8194,         # Bloomberg API port
    application_name="Neutryx",
    timeout_ms=5000,
    cache_enabled=True
)

adapter = BloombergAdapter(config)
await asyncio.to_thread(adapter.connect)

# Get equity quote
quote = adapter.get_equity_quote("AAPL US Equity")
print(f"Price: {quote.price}, Volume: {quote.volume}")

# Get FX quote
fx_quote = adapter.get_fx_quote("EUR", "USD")
print(f"EUR/USD: {fx_quote.spot}")

# Get bond quote
bond_quote = adapter.get_bond_quote("US912828ZG94", id_type="isin")
print(f"Yield: {bond_quote.yield_to_maturity:.2%}")
```

**Bloomberg Field Reference**:
- `PX_LAST`: Last price
- `PX_BID`: Bid price
- `PX_ASK`: Ask price
- `PX_VOLUME`: Volume
- `YLD_YTM_MID`: Yield to maturity
- `Z_SPRD_MID`: Credit spread

### Refinitiv Adapter

Connect to Refinitiv Data Platform (RDP) or Eikon Desktop.

```python
from neutryx.market.adapters import RefinitivAdapter, RefinitivConfig

# For Eikon Desktop
config = RefinitivConfig(
    adapter_name="refinitiv",
    app_key="your_app_key",
    use_desktop=True
)

# For Refinitiv Data Platform
config_rdp = RefinitivConfig(
    adapter_name="refinitiv",
    app_key="your_app_key",
    username="your_username",
    password="your_password",
    use_desktop=False
)

adapter = RefinitivAdapter(config)
adapter.connect()

# Get equity quote (using RIC)
quote = adapter.get_equity_quote("AAPL.O")  # Apple on NASDAQ

# Get FX quote
fx_quote = adapter.get_fx_quote("EUR", "USD")  # EUR=

# Get commodity quote
commodity = adapter.get_commodity_quote("LCOc1")  # Brent crude
```

**Reuters Instrument Code (RIC) Examples**:
- Equity: `AAPL.O` (Apple on NASDAQ), `MSFT.O`
- FX: `EUR=` (EUR/USD), `JPY=` (USD/JPY)
- Commodities: `LCOc1` (Brent crude), `CLc1` (WTI crude)
- Bonds: `US10YT=RR` (US 10Y Treasury yield)

## Database Storage

### PostgreSQL Storage

High-performance relational storage with time-series optimizations.

```python
from neutryx.market.storage import PostgreSQLStorage, PostgreSQLConfig

config = PostgreSQLConfig(
    host="localhost",
    port=5432,
    database="market_data",
    username="trader",
    password="secret",
    schema="market_data",
    table_prefix="md_",
    connection_pool_size=20
)

storage = PostgreSQLStorage(config)
await storage.connect()

# Create indexes
await storage.create_indexes()

# Store quote
await storage.store_quote(
    data_type="equity",
    symbol="AAPL",
    timestamp=datetime.utcnow(),
    data={
        "price": 150.25,
        "bid": 150.20,
        "ask": 150.30,
        "volume": 1000000
    }
)

# Store batch
records = [
    {
        "symbol": "AAPL",
        "timestamp": datetime.utcnow(),
        "price": 150.25,
        "volume": 1000000
    },
    # ... more records
]
await storage.store_batch("equity", records)

# Query latest
latest = await storage.query_latest("equity", "AAPL")
print(f"Latest price: {latest['price']}")

# Query time range
data = await storage.query_time_range(
    "equity",
    "AAPL",
    start_time=datetime(2025, 1, 1),
    end_time=datetime(2025, 1, 31),
    limit=1000
)

# Query aggregated (OHLCV)
ohlcv = await storage.query_aggregated(
    "equity",
    "AAPL",
    start_time=datetime(2025, 1, 1),
    end_time=datetime(2025, 1, 31),
    interval="1min"
)
```

### MongoDB Storage

Flexible document storage for heterogeneous market data.

```python
from neutryx.market.storage import MongoDBStorage, MongoDBConfig

config = MongoDBConfig(
    host="localhost",
    port=27017,
    database="market_data",
    username="trader",
    password="secret",
    collection_prefix="md_",
    use_time_series=True  # MongoDB 5.0+ time-series collections
)

storage = MongoDBStorage(config)
await storage.connect()

# Store with flexible schema
await storage.store_quote(
    data_type="equity",
    symbol="AAPL",
    timestamp=datetime.utcnow(),
    data={
        "price": 150.25,
        "volume": 1000000,
        "custom_field": "value",  # Flexible schema
        "metadata": {"exchange": "NASDAQ"}
    }
)

# Aggregation pipeline
aggregated = await storage.query_aggregated(
    "equity",
    "AAPL",
    start_time=datetime(2025, 1, 1),
    end_time=datetime(2025, 1, 31),
    interval="1min"
)
```

### TimescaleDB Storage

Optimized time-series database with automatic partitioning and compression.

```python
from neutryx.market.storage import TimescaleDBStorage, TimescaleDBConfig

config = TimescaleDBConfig(
    host="localhost",
    port=5432,
    database="market_data",
    username="trader",
    password="secret",
    chunk_time_interval="1 day",       # Partition by day
    compression_enabled=True,           # Auto-compress old data
    compression_after="7 days",         # Compress after 7 days
    retention_policy_days=90            # Auto-delete after 90 days
)

storage = TimescaleDBStorage(config)
await storage.connect()

# Create hypertables and continuous aggregates
await storage.create_indexes()

# Query uses continuous aggregates for speed
ohlcv = await storage.query_aggregated(
    "equity",
    "AAPL",
    start_time=datetime(2025, 1, 1),
    end_time=datetime(2025, 1, 31),
    interval="1min"
)

# Get TimescaleDB-specific statistics
stats = await storage.get_statistics()
print(f"Compression ratio: {stats['timescaledb']['compression_ratio']:.1%}")

# Maintenance
await storage.vacuum_and_analyze()
```

**TimescaleDB Features**:
- **Hypertables**: Automatic partitioning by time
- **Compression**: Up to 90% storage reduction
- **Continuous Aggregates**: Pre-computed OHLCV bars
- **Retention Policies**: Automatic data cleanup

## Data Validation

### Validation Pipeline

Comprehensive data quality checks with customizable validators.

```python
from neutryx.market.validation import (
    ValidationPipeline,
    PriceRangeValidator,
    SpreadValidator,
    VolumeValidator,
    VolatilityValidator,
    TimeSeriesValidator
)

# Create pipeline
pipeline = ValidationPipeline()

# Add validators
pipeline.add_validator(PriceRangeValidator(
    min_price=0.0,
    max_price=100000.0,
    max_jump_pct=0.20  # 20% max price jump
))

pipeline.add_validator(SpreadValidator(
    max_spread_pct=0.05,  # 5% max spread
    check_mid_consistency=True
))

pipeline.add_validator(VolumeValidator(
    min_volume=0.0,
    max_volume_multiplier=10.0  # 10x average volume
))

pipeline.add_validator(VolatilityValidator(
    min_vol=0.01,  # 1%
    max_vol=3.0    # 300%
))

pipeline.add_validator(TimeSeriesValidator(
    max_gap_seconds=300  # 5 minutes max gap
))

# Validate data
data = {
    "symbol": "AAPL",
    "timestamp": datetime.utcnow(),
    "price": 150.25,
    "bid": 150.20,
    "ask": 150.30,
    "volume": 1000000,
    "volatility": 0.25
}

results = pipeline.validate(data)

# Check results
for result in results:
    if not result.passed:
        print(f"[{result.severity.value.upper()}] {result.validator_name}: {result.message}")

# Get quality report
report = pipeline.get_quality_report()
print(f"Quality Score: {report.metrics.quality_score:.1%}")
print(f"Total Records: {report.metrics.total_records}")
print(f"Errors: {report.metrics.error_records}")
print(f"Warnings: {report.metrics.warning_records}")

# Get recommendations
for recommendation in report.recommendations:
    print(f"- {recommendation}")
```

### Custom Validators

Create custom validators for specific requirements.

```python
from neutryx.market.validation import BaseValidator, ValidationResult, ValidationSeverity

class CustomValidator(BaseValidator):
    def __init__(self):
        super().__init__("CustomValidator")

    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        # Custom validation logic
        price = data.get("price")

        if price and price % 1 != 0:
            return ValidationResult(
                validator_name=self.name,
                passed=False,
                severity=ValidationSeverity.WARNING,
                message="Price has fractional component",
                details={"price": price}
            )

        return ValidationResult(
            validator_name=self.name,
            passed=True,
            message="Custom validation passed"
        )

# Add to pipeline
pipeline.add_validator(CustomValidator())
```

## Real-Time Feeds

### Feed Manager

Orchestrate real-time data feeds with automatic failover and storage.

```python
from neutryx.market.feeds import FeedManager, FeedConfig

# Configure feed manager
feed_config = FeedConfig(
    enable_validation=True,
    enable_storage=True,
    buffer_size=1000,
    flush_interval_seconds=60,
    max_errors=10,
    enable_failover=True
)

# Create manager with multiple adapters (failover)
manager = FeedManager(
    adapters=[bloomberg_adapter, refinitiv_adapter],  # Ordered by priority
    storage=storage,
    validation_pipeline=pipeline,
    config=feed_config
)

# Start feed
await manager.start()

# Subscribe to symbols
await manager.subscribe("equity", ["AAPL", "MSFT", "GOOGL"])
await manager.subscribe("fx", ["EURUSD", "GBPUSD"])

# Add callback for real-time updates
def on_data_update(data):
    print(f"Update: {data['symbol']} @ {data['price']}")

manager.add_callback(on_data_update)

# Get statistics
stats = manager.get_statistics()
print(f"Status: {stats['status']}")
print(f"Current Adapter: {stats['current_adapter']}")
print(f"Quality Score: {stats['quality_score']:.1%}")

# Stop feed
await manager.stop()
```

### Data Subscriber

Manage subscriptions with callbacks.

```python
from neutryx.market.feeds import DataSubscriber, SubscriptionRequest

subscriber = DataSubscriber()

# Subscribe with callback
def on_equity_update(data):
    print(f"Equity update: {data}")

subscription_id = subscriber.subscribe(
    SubscriptionRequest(
        data_type="equity",
        symbols=["AAPL", "MSFT"],
        callback=on_equity_update
    )
)

# Add global callback
def on_any_update(data):
    print(f"Data update: {data}")

subscriber.add_global_callback(on_any_update)

# Notify (typically called by feed manager)
subscriber.notify({
    "data_type": "equity",
    "symbol": "AAPL",
    "price": 150.25
})

# Unsubscribe
subscriber.unsubscribe(subscription_id)
```

## Examples

### Complete Real-Time Pipeline

```python
import asyncio
from datetime import datetime
from neutryx.market.adapters import BloombergAdapter, BloombergConfig
from neutryx.market.storage import TimescaleDBStorage, TimescaleDBConfig
from neutryx.market.validation import (
    ValidationPipeline,
    PriceRangeValidator,
    SpreadValidator,
    VolumeValidator
)
from neutryx.market.feeds import FeedManager, FeedConfig

async def main():
    # Configure Bloomberg
    bloomberg_config = BloombergConfig(
        adapter_name="bloomberg",
        host="localhost",
        port=8194
    )

    # Configure TimescaleDB
    storage_config = TimescaleDBConfig(
        host="localhost",
        port=5432,
        database="market_data",
        username="trader",
        password="secret",
        compression_enabled=True,
        retention_policy_days=90
    )

    # Initialize components
    adapter = BloombergAdapter(bloomberg_config)
    storage = TimescaleDBStorage(storage_config)

    # Connect to storage
    await storage.connect()
    await storage.create_indexes()

    # Setup validation pipeline
    pipeline = ValidationPipeline()
    pipeline.add_validator(PriceRangeValidator(min_price=0, max_jump_pct=0.20))
    pipeline.add_validator(SpreadValidator(max_spread_pct=0.05))
    pipeline.add_validator(VolumeValidator(max_volume_multiplier=10.0))

    # Create feed manager
    feed_config = FeedConfig(
        enable_validation=True,
        enable_storage=True,
        flush_interval_seconds=60
    )

    manager = FeedManager(
        adapters=[adapter],
        storage=storage,
        validation_pipeline=pipeline,
        config=feed_config
    )

    # Start and subscribe
    await manager.start()
    await manager.subscribe("equity", ["AAPL", "MSFT", "GOOGL", "AMZN"])

    # Run for some time
    await asyncio.sleep(3600)  # 1 hour

    # Get quality report
    report = pipeline.get_quality_report()
    print(f"Quality Score: {report.metrics.quality_score:.1%}")
    print(f"Total Records: {report.metrics.total_records}")

    # Stop
    await manager.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

### Multi-Source Aggregation

```python
# Use multiple adapters with automatic failover
bloomberg_adapter = BloombergAdapter(bloomberg_config)
refinitiv_adapter = RefinitivAdapter(refinitiv_config)

# Feed manager automatically fails over
manager = FeedManager(
    adapters=[bloomberg_adapter, refinitiv_adapter],  # Priority order
    storage=storage,
    validation_pipeline=pipeline,
    config=FeedConfig(enable_failover=True)
)
```

## Production Deployment

### High Availability Setup

```bash
# TimescaleDB cluster
docker run -d \
  --name timescaledb \
  -p 5432:5432 \
  -e POSTGRES_PASSWORD=secret \
  -e POSTGRES_DB=market_data \
  timescale/timescaledb:latest-pg14

# MongoDB replica set
docker run -d \
  --name mongodb \
  -p 27017:27017 \
  -e MONGO_INITDB_ROOT_USERNAME=admin \
  -e MONGO_INITDB_ROOT_PASSWORD=secret \
  mongo:6.0
```

### Environment Variables

```bash
# Bloomberg
export BLOOMBERG_HOST=localhost
export BLOOMBERG_PORT=8194

# Refinitiv
export REFINITIV_APP_KEY=your_app_key
export REFINITIV_USERNAME=your_username
export REFINITIV_PASSWORD=your_password

# TimescaleDB
export TIMESCALEDB_HOST=localhost
export TIMESCALEDB_PORT=5432
export TIMESCALEDB_DATABASE=market_data
export TIMESCALEDB_USERNAME=trader
export TIMESCALEDB_PASSWORD=secret
```

### Performance Tuning

```python
# Optimize storage configuration
storage_config = TimescaleDBConfig(
    host="localhost",
    port=5432,
    database="market_data",
    username="trader",
    password="secret",
    connection_pool_size=50,           # Increase pool size
    chunk_time_interval="1 day",       # Adjust chunk size
    compression_enabled=True,
    compression_after="1 day",         # Compress aggressively
    retention_policy_days=30           # Shorter retention
)

# Optimize feed configuration
feed_config = FeedConfig(
    buffer_size=5000,                  # Larger buffer
    flush_interval_seconds=30,         # More frequent flushes
    max_errors=20,                     # Higher tolerance
    enable_failover=True
)
```

### Monitoring

```python
# Get statistics
storage_stats = await storage.get_statistics()
print(f"Total Size: {storage_stats['total_size_bytes'] / 1e9:.2f} GB")
print(f"Total Rows: {storage_stats['total_rows']:,}")

if "timescaledb" in storage_stats:
    print(f"Compression Ratio: {storage_stats['timescaledb']['compression_ratio']:.1%}")
    print(f"Total Chunks: {storage_stats['timescaledb']['total_chunks']}")

# Feed statistics
feed_stats = manager.get_statistics()
print(f"Feed Status: {feed_stats['status']}")
print(f"Quality Score: {feed_stats.get('quality_score', 0):.1%}")
```

## Best Practices

1. **Use TimescaleDB for Time-Series**: Best performance for tick data
2. **Enable Compression**: Reduces storage by up to 90%
3. **Set Retention Policies**: Auto-delete old data
4. **Validate All Data**: Catch issues early
5. **Use Connection Pooling**: Better performance
6. **Monitor Quality Scores**: Track data quality
7. **Enable Failover**: Ensure high availability
8. **Flush Regularly**: Don't lose data

## Troubleshooting

### Bloomberg Connection Issues

```python
# Check Bloomberg service
import blpapi
session_options = blpapi.SessionOptions()
session_options.setServerHost("localhost")
session_options.setServerPort(8194)
session = blpapi.Session(session_options)
print(session.start())  # Should return True
```

### Storage Connection Issues

```python
# Test PostgreSQL connection
import asyncpg
conn = await asyncpg.connect(
    host="localhost",
    port=5432,
    database="market_data",
    user="trader",
    password="secret"
)
print(await conn.fetchval("SELECT version()"))
await conn.close()
```

### Performance Issues

```sql
-- Check TimescaleDB chunk statistics
SELECT * FROM timescaledb_information.chunks;

-- Check table sizes
SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'market_data'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```
