# Neutryx API Integration Examples

This guide demonstrates how to integrate with the Neutryx REST API for portfolio management and XVA calculations.

## Table of Contents

1. [API Setup](#api-setup)
2. [Authentication](#authentication)
3. [Portfolio Management](#portfolio-management)
4. [XVA Calculations](#xva-calculations)
5. [Market Data](#market-data)
6. [Error Handling](#error-handling)
7. [Best Practices](#best-practices)

## API Setup

### Starting the API

```bash
# Development mode with auto-reload
uvicorn neutryx.api.rest:create_app --factory --reload

# Production mode
uvicorn neutryx.api.rest:create_app --factory --host 0.0.0.0 --port 8000

# With workers (production)
gunicorn -w 4 -k uvicorn.workers.UvicornWorker neutryx.api.rest:create_app
```

### API Documentation

Once the API is running, access interactive documentation at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

### Health Check

```bash
curl http://localhost:8000/health
```

## Authentication

*Note: Current version uses no authentication. Production deployments should implement OAuth2/JWT.*

## Portfolio Management

### 1. Register Portfolio

Register a portfolio with the Neutryx system.

**Endpoint:** `POST /portfolio/register`

**Python Example:**

```python
import requests
from neutryx.tests.fixtures.fictional_portfolio import create_fictional_portfolio

# Create portfolio
portfolio, book_hierarchy = create_fictional_portfolio()

# Convert to JSON
portfolio_data = portfolio.model_dump(mode="json")

# Register with API
response = requests.post(
    "http://localhost:8000/portfolio/register",
    json=portfolio_data,
    headers={"Content-Type": "application/json"},
)

result = response.json()
portfolio_id = result["portfolio_id"]
print(f"Portfolio registered: {portfolio_id}")
```

**cURL Example:**

```bash
curl -X POST "http://localhost:8000/portfolio/register" \
  -H "Content-Type: application/json" \
  -d @portfolio_data.json
```

**Response:**

```json
{
  "portfolio_id": "fictional_global_trading",
  "status": "registered",
  "summary": {
    "trades": 13,
    "counterparties": 6,
    "netting_sets": 6
  }
}
```

### 2. Get Portfolio Summary

Retrieve portfolio summary statistics.

**Endpoint:** `GET /portfolio/{portfolio_id}/summary`

**Python Example:**

```python
response = requests.get(
    f"http://localhost:8000/portfolio/{portfolio_id}/summary"
)

summary = response.json()
print(f"Total Trades: {summary['trades']}")
print(f"Total MTM: ${summary['total_mtm']:,.2f}")
```

**Response:**

```json
{
  "portfolio_id": "fictional_global_trading",
  "counterparties": 6,
  "netting_sets": 6,
  "trades": 13,
  "total_mtm": 1247380.00,
  "gross_notional": 152000000.00,
  "base_currency": "USD"
}
```

### 3. Get Netting Sets

List all netting sets in the portfolio.

**Endpoint:** `GET /portfolio/{portfolio_id}/netting-sets`

**Python Example:**

```python
response = requests.get(
    f"http://localhost:8000/portfolio/{portfolio_id}/netting-sets"
)

netting_sets = response.json()
for ns in netting_sets["netting_sets"]:
    print(f"Netting Set: {ns['netting_set_id']}")
    print(f"  Counterparty: {ns['counterparty_name']}")
    print(f"  CSA: {'Yes' if ns['has_csa'] else 'No'}")
    print(f"  Trades: {ns['num_trades']}")
    print(f"  Net MTM: ${ns['net_mtm']:,.2f}")
    print()
```

## XVA Calculations

### 1. Portfolio-Level XVA

Calculate XVA for the entire portfolio.

**Endpoint:** `POST /portfolio/xva`

**Python Example:**

```python
xva_request = {
    "portfolio_id": portfolio_id,
    "valuation_date": "2024-01-15",
    "compute_cva": True,
    "compute_dva": True,
    "compute_fva": True,
    "compute_mva": True,
    "lgd": 0.6,  # Loss Given Default
    "funding_spread_bps": 50.0,
}

response = requests.post(
    "http://localhost:8000/portfolio/xva",
    json=xva_request,
    headers={"Content-Type": "application/json"},
)

xva = response.json()
print(f"CVA: ${xva['cva']:,.2f}")
print(f"DVA: ${xva['dva']:,.2f}")
print(f"FVA: ${xva['fva']:,.2f}")
print(f"MVA: ${xva['mva']:,.2f}")
print(f"Total XVA: ${xva['total_xva']:,.2f}")
```

**Response:**

```json
{
  "portfolio_id": "fictional_global_trading",
  "valuation_date": "2024-01-15",
  "scope": "portfolio",
  "num_trades": 13,
  "net_mtm": 1247380.00,
  "positive_exposure": 2450000.00,
  "negative_exposure": -1202620.00,
  "cva": 123400.00,
  "dva": -45600.00,
  "fva": 78900.00,
  "mva": 34500.00,
  "total_xva": 191200.00,
  "computation_time_ms": 1234
}
```

### 2. Netting Set-Level XVA

Calculate XVA for a specific netting set.

**Python Example:**

```python
xva_request = {
    "portfolio_id": portfolio_id,
    "netting_set_id": "NS_AAAGLOBALBANK",
    "valuation_date": "2024-01-15",
    "compute_cva": True,
    "compute_dva": True,
    "compute_fva": True,
    "compute_mva": True,
    "lgd": 0.6,
    "funding_spread_bps": 50.0,
}

response = requests.post(
    "http://localhost:8000/portfolio/xva",
    json=xva_request,
)

xva = response.json()
print(f"Netting Set: {xva['netting_set_id']}")
print(f"Total XVA: ${xva['total_xva']:,.2f}")
```

### 3. Batch XVA Calculation

Calculate XVA for all netting sets.

**Python Example:**

```python
# Get all netting sets
response = requests.get(
    f"http://localhost:8000/portfolio/{portfolio_id}/netting-sets"
)
netting_sets = response.json()["netting_sets"]

# Calculate XVA for each
xva_results = []
for ns in netting_sets:
    xva_request = {
        "portfolio_id": portfolio_id,
        "netting_set_id": ns["netting_set_id"],
        "valuation_date": "2024-01-15",
        "compute_cva": True,
        "compute_dva": True,
        "compute_fva": True,
        "compute_mva": True,
        "lgd": 0.6,
        "funding_spread_bps": 50.0,
    }

    response = requests.post(
        "http://localhost:8000/portfolio/xva",
        json=xva_request,
    )

    xva_results.append(response.json())

# Aggregate results
total_cva = sum(r["cva"] for r in xva_results)
total_xva = sum(r["total_xva"] for r in xva_results)
print(f"Total Portfolio CVA: ${total_cva:,.2f}")
print(f"Total Portfolio XVA: ${total_xva:,.2f}")
```

## Market Data

### 1. Upload Market Data

*Note: This endpoint may vary based on your Neutryx configuration.*

**Python Example:**

```python
import yaml

# Load market data from config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

market_data = config["market_data"]

response = requests.post(
    "http://localhost:8000/market-data/upload",
    json=market_data,
)

print(f"Market data uploaded: {response.json()}")
```

### 2. Get Current Market Data

```python
response = requests.get(
    "http://localhost:8000/market-data/current"
)

market_data = response.json()
print(f"Valuation Date: {market_data['valuation_date']}")
print(f"USD 10Y Rate: {market_data['rates']['USD']['10Y']}%")
```

## Error Handling

### Common HTTP Status Codes

- **200**: Success
- **400**: Bad Request (invalid parameters)
- **404**: Not Found (portfolio/resource not found)
- **422**: Validation Error (invalid data format)
- **500**: Internal Server Error

### Error Response Format

```json
{
  "detail": "Error message description",
  "status_code": 400,
  "error_type": "ValidationError"
}
```

### Python Error Handling

```python
import requests

try:
    response = requests.post(
        "http://localhost:8000/portfolio/xva",
        json=xva_request,
        timeout=30,  # 30 second timeout
    )
    response.raise_for_status()  # Raise exception for 4xx/5xx

    xva = response.json()
    # Process results...

except requests.exceptions.Timeout:
    print("Request timed out")
except requests.exceptions.ConnectionError:
    print("Cannot connect to API. Is it running?")
except requests.exceptions.HTTPError as e:
    print(f"HTTP Error: {e.response.status_code}")
    print(f"Details: {e.response.json()}")
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
```

## Best Practices

### 1. Connection Management

Use sessions for multiple requests:

```python
import requests

session = requests.Session()
session.headers.update({"Content-Type": "application/json"})

# Reuse session for all requests
response1 = session.get("http://localhost:8000/portfolio/...")
response2 = session.post("http://localhost:8000/portfolio/xva", json=...)

session.close()
```

### 2. Asynchronous Requests

For better performance with multiple calculations:

```python
import asyncio
import aiohttp

async def calculate_xva(session, portfolio_id, netting_set_id):
    """Calculate XVA asynchronously."""
    xva_request = {
        "portfolio_id": portfolio_id,
        "netting_set_id": netting_set_id,
        "valuation_date": "2024-01-15",
        "compute_cva": True,
        "compute_dva": True,
        "compute_fva": True,
        "compute_mva": True,
    }

    async with session.post(
        "http://localhost:8000/portfolio/xva",
        json=xva_request,
    ) as response:
        return await response.json()

async def calculate_all_xvas(portfolio_id, netting_set_ids):
    """Calculate XVAs for all netting sets in parallel."""
    async with aiohttp.ClientSession() as session:
        tasks = [
            calculate_xva(session, portfolio_id, ns_id)
            for ns_id in netting_set_ids
        ]
        results = await asyncio.gather(*tasks)
        return results

# Usage
netting_set_ids = ["NS_1", "NS_2", "NS_3", "NS_4", "NS_5", "NS_6"]
xva_results = asyncio.run(calculate_all_xvas(portfolio_id, netting_set_ids))
```

### 3. Caching Results

Cache expensive calculations:

```python
from functools import lru_cache
import hashlib
import json

def hash_request(request_dict):
    """Create hash of request for caching."""
    request_str = json.dumps(request_dict, sort_keys=True)
    return hashlib.md5(request_str.encode()).hexdigest()

class XVACalculator:
    def __init__(self, api_url):
        self.api_url = api_url
        self.cache = {}

    def calculate_xva(self, xva_request):
        """Calculate XVA with caching."""
        cache_key = hash_request(xva_request)

        if cache_key in self.cache:
            print("Returning cached result")
            return self.cache[cache_key]

        response = requests.post(
            f"{self.api_url}/portfolio/xva",
            json=xva_request,
        )

        result = response.json()
        self.cache[cache_key] = result
        return result

# Usage
calculator = XVACalculator("http://localhost:8000")
xva1 = calculator.calculate_xva(request1)  # Calls API
xva2 = calculator.calculate_xva(request1)  # Returns cached
```

### 4. Retry Logic

Implement retry logic for transient failures:

```python
import time
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

def create_session_with_retries():
    """Create session with automatic retries."""
    session = requests.Session()

    retry_strategy = Retry(
        total=3,  # Total retries
        backoff_factor=1,  # Wait 1, 2, 4 seconds between retries
        status_forcelist=[429, 500, 502, 503, 504],  # Retry on these status codes
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session

# Usage
session = create_session_with_retries()
response = session.post("http://localhost:8000/portfolio/xva", json=xva_request)
```

### 5. Monitoring & Logging

Track API performance:

```python
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MonitoredXVAClient:
    def __init__(self, api_url):
        self.api_url = api_url

    def calculate_xva(self, xva_request):
        """Calculate XVA with monitoring."""
        start_time = time.time()

        logger.info(f"Starting XVA calculation for {xva_request['portfolio_id']}")

        try:
            response = requests.post(
                f"{self.api_url}/portfolio/xva",
                json=xva_request,
            )
            response.raise_for_status()

            elapsed = time.time() - start_time
            logger.info(f"XVA calculation completed in {elapsed:.2f}s")

            return response.json()

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"XVA calculation failed after {elapsed:.2f}s: {e}")
            raise
```

## Complete Example Script

```python
#!/usr/bin/env python3
"""Complete example of Neutryx API usage."""

import requests
import sys

API_URL = "http://localhost:8000"

def check_api_health():
    """Check if API is available."""
    try:
        response = requests.get(f"{API_URL}/docs", timeout=2)
        return response.status_code == 200
    except:
        return False

def main():
    # 1. Check API
    if not check_api_health():
        print("Error: API not available at", API_URL)
        sys.exit(1)

    print("✓ API is running")

    # 2. Load portfolio
    from neutryx.tests.fixtures.fictional_portfolio import create_fictional_portfolio

    portfolio, book_hierarchy = create_fictional_portfolio()
    print(f"✓ Portfolio created: {portfolio.name}")

    # 3. Register portfolio
    portfolio_data = portfolio.model_dump(mode="json")
    response = requests.post(
        f"{API_URL}/portfolio/register",
        json=portfolio_data,
    )
    portfolio_id = response.json()["portfolio_id"]
    print(f"✓ Portfolio registered: {portfolio_id}")

    # 4. Get summary
    response = requests.get(f"{API_URL}/portfolio/{portfolio_id}/summary")
    summary = response.json()
    print(f"✓ Trades: {summary['trades']}, MTM: ${summary['total_mtm']:,.2f}")

    # 5. Calculate XVA
    xva_request = {
        "portfolio_id": portfolio_id,
        "valuation_date": "2024-01-15",
        "compute_cva": True,
        "compute_dva": True,
        "compute_fva": True,
        "compute_mva": True,
    }

    response = requests.post(f"{API_URL}/portfolio/xva", json=xva_request)
    xva = response.json()
    print(f"✓ Total XVA: ${xva['total_xva']:,.2f}")
    print(f"  CVA: ${xva['cva']:,.2f}")
    print(f"  DVA: ${xva['dva']:,.2f}")
    print(f"  FVA: ${xva['fva']:,.2f}")
    print(f"  MVA: ${xva['mva']:,.2f}")

if __name__ == "__main__":
    main()
```

---

For more examples, see the scripts in this directory:
- `compute_xva.py` - Full XVA workflow
- `cli.py` - CLI implementation with API calls
