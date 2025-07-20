# CoinGecko Pro API: Fetching Circulating and Total Supply Data

A comprehensive guide for fetching cryptocurrency supply data at any point in time using the CoinGecko Pro API and Python SDK.

## Table of Contents

- [Overview](#overview)
- [Supply Data Endpoints](#supply-data-endpoints)
- [Key Differences Between Supply Types](#key-differences-between-supply-types)
- [Python SDK Setup](#python-sdk-setup)
- [Usage Examples](#usage-examples)
  - [Method 1: Get Supply Data by Days Back](#method-1-get-supply-data-by-days-back)
  - [Method 2: Get Supply Data for Specific Time Range](#method-2-get-supply-data-for-specific-time-range)
  - [Method 3: Get Supply at a Specific Point in Time](#method-3-get-supply-at-a-specific-point-in-time)
- [Data Structure and Response Format](#data-structure-and-response-format)
- [Data Granularity and Limitations](#data-granularity-and-limitations)
- [Complete Working Example](#complete-working-example)
- [Error Handling](#error-handling)
- [Best Practices](#best-practices)

## Overview

The CoinGecko Pro API provides historical supply data through specialized endpoints that allow you to fetch circulating and total supply numbers for any cryptocurrency at any given point in time. This data is essential for calculating metrics like circulation ratios, supply inflation rates, and historical market capitalizations.

## Supply Data Endpoints

The CoinGecko Pro API provides **four main endpoints** for historical supply data (marked with 👑 indicating Pro-only features):

1. **Circulating Supply Chart by ID** - `/coins/{id}/circulating_supply_chart`
2. **Circulating Supply Chart within Time Range** - `/coins/{id}/circulating_supply_chart/range`
3. **Total Supply Chart by ID** - `/coins/{id}/total_supply_chart`
4. **Total Supply Chart within Time Range** - `/coins/{id}/total_supply_chart/range`

## Key Differences Between Supply Types

- **Circulating Supply**: The number of coins/tokens that are publicly available and circulating in the market
- **Total Supply**: The total number of coins/tokens that exist right now (excluding burned tokens)
- **Max Supply**: The maximum number of coins/tokens that will ever exist (not covered by these endpoints)

## Python SDK Setup

### Installation

```bash
# Install the CoinGecko SDK
pip install coingecko_sdk

# Optional: Install with aiohttp for better async performance
pip install coingecko_sdk[aiohttp]
```

### Basic Setup

```python
import os
from coingecko_sdk import Coingecko
from datetime import datetime, timezone

# Initialize the client with your Pro API key
client = Coingecko(
    pro_api_key=os.environ.get("COINGECKO_PRO_API_KEY"),
    environment="pro"
)

# For async usage
from coingecko_sdk import AsyncCoingecko

async_client = AsyncCoingecko(
    pro_api_key=os.environ.get("COINGECKO_PRO_API_KEY"),
    environment="pro"
)
```

## Usage Examples

### Method 1: Get Supply Data by Days Back

For fetching supply data a certain number of days back from now:

```python
# Get Bitcoin circulating supply for the last 30 days
circulating_supply = client.coins.circulating_supply_chart.get(
    id="bitcoin",
    days="30",  # Can be any integer or "max" for all available data
    interval="daily"  # Optional: specify interval
)

# Get Bitcoin total supply for the last 90 days
total_supply = client.coins.total_supply_chart.get(
    id="bitcoin", 
    days="90",
    interval="daily"
)

# Access the data
print("Circulating Supply Data:", circulating_supply.circulating_supply)
print("Total Supply Data:", total_supply.total_supply)

# Example output format:
# [[timestamp1, supply_value1], [timestamp2, supply_value2], ...]
```

### Method 2: Get Supply Data for Specific Time Range

For fetching supply data between specific timestamps (more precise control):

```python
import time
from datetime import datetime, timezone

# Define your time range using UNIX timestamps
start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
end_date = datetime(2024, 6, 30, tzinfo=timezone.utc)

start_timestamp = start_date.timestamp()
end_timestamp = end_date.timestamp()

# Get circulating supply within specific time range
circulating_supply_range = client.coins.circulating_supply_chart.get_range(
    id="ethereum",
    from_=start_timestamp,
    to=end_timestamp
)

# Get total supply within specific time range  
total_supply_range = client.coins.total_supply_chart.get_range(
    id="ethereum",
    from_=start_timestamp, 
    to=end_timestamp
)

# The response contains arrays of [timestamp, supply_value] pairs
print("Circulating Supply Range:", circulating_supply_range.circulating_supply)
print("Total Supply Range:", total_supply_range.total_supply)
```

### Method 3: Get Supply at a Specific Point in Time

To get supply data for a very specific date:

```python
# Get supply data for a specific day (January 1, 2024)
target_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
target_timestamp = target_date.timestamp()

# Use a small range around your target date
one_day = 86400  # seconds in a day
start_ts = target_timestamp - one_day
end_ts = target_timestamp + one_day

# Fetch the data
supply_data = client.coins.circulating_supply_chart.get_range(
    id="bitcoin",
    from_=start_ts,
    to=end_ts
)

# Find the closest data point to your target
if supply_data.circulating_supply:
    closest_data = min(
        supply_data.circulating_supply,
        key=lambda x: abs(x[0] - target_timestamp)
    )
    timestamp, supply_value = closest_data
    actual_date = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    print(f"Circulating supply on {actual_date}: {supply_value:,.0f}")
```

## Data Structure and Response Format

The API returns data in this format:

```python
# Response structure for circulating supply
{
    "circulating_supply": [
        [timestamp, supply_value],
        [timestamp, supply_value],
        # ... more data points
    ]
}

# Response structure for total supply
{
    "total_supply": [
        [timestamp, supply_value],
        [timestamp, supply_value],
        # ... more data points
    ]
}

# Example actual response
{
    "circulating_supply": [
        [1704067200, 19675962],  # [UNIX timestamp, supply amount]
        [1704153600, 19676000],
        [1704240000, 19676037]
    ]
}
```

Each data point is an array where:
- `[0]` = UNIX timestamp (seconds since epoch)
- `[1]` = Supply value (number of coins/tokens)

## Data Granularity and Limitations

### Data Interval Rules

The API automatically determines data granularity based on the requested time range:

- **1 day from current time**: 5-minute intervals
- **1 day from any other time**: hourly intervals  
- **2-90 days**: hourly intervals
- **Above 90 days**: daily intervals (00:00 UTC)

### Important Considerations

1. **Pro API Required**: Supply chart endpoints require a Pro API subscription
2. **Rate Limits**: Respect API rate limits to avoid being throttled
3. **Data Availability**: Not all coins have complete historical supply data
4. **UNIX Timestamps**: All timestamps are in UNIX format (seconds since epoch)
5. **Data Points**: The API returns the closest available data points to your requested range
6. **Timezone**: All timestamps are in UTC

## Complete Working Example

```python
import os
from datetime import datetime, timezone
from coingecko_sdk import Coingecko

# Initialize client
client = Coingecko(
    pro_api_key=os.environ.get("COINGECKO_PRO_API_KEY"),
    environment="pro"
)

def get_supply_at_date(coin_id: str, target_date: datetime):
    """
    Get both circulating and total supply for a specific date
    
    Args:
        coin_id: CoinGecko coin ID (e.g., 'bitcoin', 'ethereum')
        target_date: Target date as datetime object
        
    Returns:
        Dictionary with supply data or None if error
    """
    target_timestamp = target_date.timestamp()
    
    # Use a 2-day window around target date for better data coverage
    two_days = 2 * 86400
    start_ts = target_timestamp - two_days
    end_ts = target_timestamp + two_days
    
    try:
        # Fetch both supply types
        circulating = client.coins.circulating_supply_chart.get_range(
            id=coin_id,
            from_=start_ts,
            to=end_ts
        )
        
        total = client.coins.total_supply_chart.get_range(
            id=coin_id,
            from_=start_ts,
            to=end_ts
        )
        
        # Find closest data points
        results = {}
        
        if circulating.circulating_supply:
            closest_circ = min(
                circulating.circulating_supply,
                key=lambda x: abs(x[0] - target_timestamp)
            )
            results['circulating_supply'] = closest_circ[1]
            results['circulating_timestamp'] = closest_circ[0]
            results['circulating_date'] = datetime.fromtimestamp(
                closest_circ[0], tz=timezone.utc
            )
        
        if total.total_supply:
            closest_total = min(
                total.total_supply,
                key=lambda x: abs(x[0] - target_timestamp)
            )
            results['total_supply'] = closest_total[1]
            results['total_timestamp'] = closest_total[0]
            results['total_date'] = datetime.fromtimestamp(
                closest_total[0], tz=timezone.utc
            )
            
        return results
        
    except Exception as e:
        print(f"Error fetching data for {coin_id}: {e}")
        return None

def analyze_supply_data(coin_id: str, target_date: datetime):
    """
    Analyze supply data and calculate useful metrics
    """
    supply_data = get_supply_at_date(coin_id, target_date)
    
    if not supply_data:
        print(f"Could not fetch supply data for {coin_id}")
        return
    
    print(f"\n{coin_id.upper()} Supply Analysis for {target_date.date()}:")
    print("=" * 50)
    
    if 'circulating_supply' in supply_data:
        circ_supply = supply_data['circulating_supply']
        circ_date = supply_data['circulating_date']
        print(f"Circulating Supply: {circ_supply:,.0f}")
        print(f"Circulating Data Date: {circ_date.date()}")
    
    if 'total_supply' in supply_data:
        total_supply = supply_data['total_supply']
        total_date = supply_data['total_date']
        print(f"Total Supply: {total_supply:,.0f}")
        print(f"Total Data Date: {total_date.date()}")
    
    # Calculate circulation ratio if both values are available
    if 'circulating_supply' in supply_data and 'total_supply' in supply_data:
        circulation_ratio = (supply_data['circulating_supply'] / supply_data['total_supply']) * 100
        print(f"Circulation Ratio: {circulation_ratio:.2f}%")
        
        # Calculate how many tokens are not circulating
        locked_supply = supply_data['total_supply'] - supply_data['circulating_supply']
        print(f"Locked/Reserved Supply: {locked_supply:,.0f}")

# Example usage
if __name__ == "__main__":
    # Example 1: Get Bitcoin supply data for a specific date
    target_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
    analyze_supply_data("bitcoin", target_date)
    
    # Example 2: Get Ethereum supply data for another date
    target_date = datetime(2023, 9, 15, tzinfo=timezone.utc)
    analyze_supply_data("ethereum", target_date)
    
    # Example 3: Get historical data for multiple coins
    coins = ["bitcoin", "ethereum", "cardano", "solana"]
    target_date = datetime(2024, 6, 1, tzinfo=timezone.utc)
    
    for coin in coins:
        analyze_supply_data(coin, target_date)
```

## Error Handling

```python
import coingecko_sdk

def safe_get_supply_data(coin_id: str, days: str = "30"):
    """
    Safely fetch supply data with comprehensive error handling
    """
    try:
        supply_data = client.coins.circulating_supply_chart.get(
            id=coin_id,
            days=days
        )
        return supply_data
        
    except coingecko_sdk.RateLimitError as e:
        print("Rate limit exceeded. Please wait before making more requests.")
        print(f"Details: {e}")
        return None
        
    except coingecko_sdk.AuthenticationError as e:
        print("Invalid API key. Check your Pro API key.")
        print(f"Details: {e}")
        return None
        
    except coingecko_sdk.NotFoundError as e:
        print(f"Coin '{coin_id}' not found. Check the coin ID.")
        print(f"Details: {e}")
        return None
        
    except coingecko_sdk.PermissionDeniedError as e:
        print("Access denied. This endpoint requires a Pro API subscription.")
        print(f"Details: {e}")
        return None
        
    except coingecko_sdk.APIStatusError as e:
        print(f"API returned status code {e.status_code}")
        print(f"Response: {e.response}")
        return None
        
    except coingecko_sdk.APIConnectionError as e:
        print("Connection error. Check your network connection.")
        print(f"Details: {e}")
        return None
        
    except coingecko_sdk.APIError as e:
        print(f"General API error occurred: {e}")
        return None
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
```

## Best Practices

### 1. API Key Management

```python
# Use environment variables for API keys
import os
from dotenv import load_dotenv

load_dotenv()  # Load from .env file

client = Coingecko(
    pro_api_key=os.environ.get("COINGECKO_PRO_API_KEY"),
    environment="pro"
)
```

### 2. Rate Limiting

```python
import time
from typing import List

def batch_get_supply_data(coin_ids: List[str], delay: float = 0.1):
    """
    Fetch supply data for multiple coins with rate limiting
    """
    results = {}
    
    for coin_id in coin_ids:
        try:
            data = client.coins.circulating_supply_chart.get(
                id=coin_id,
                days="7"
            )
            results[coin_id] = data
            
            # Add delay to respect rate limits
            time.sleep(delay)
            
        except Exception as e:
            print(f"Error fetching {coin_id}: {e}")
            results[coin_id] = None
            
    return results
```

### 3. Data Caching

```python
import json
from pathlib import Path
from datetime import datetime, timedelta

class SupplyDataCache:
    def __init__(self, cache_dir: str = "supply_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def get_cache_key(self, coin_id: str, start_ts: float, end_ts: float) -> str:
        return f"{coin_id}_{int(start_ts)}_{int(end_ts)}"
        
    def get_cached_data(self, coin_id: str, start_ts: float, end_ts: float):
        cache_key = self.get_cache_key(coin_id, start_ts, end_ts)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            # Check if cache is less than 1 hour old
            if datetime.now().timestamp() - cache_file.stat().st_mtime < 3600:
                with open(cache_file, 'r') as f:
                    return json.load(f)
        return None
        
    def cache_data(self, coin_id: str, start_ts: float, end_ts: float, data):
        cache_key = self.get_cache_key(coin_id, start_ts, end_ts)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        with open(cache_file, 'w') as f:
            json.dump(data, f)
```

### 4. Async Usage for Better Performance

```python
import asyncio
from coingecko_sdk import AsyncCoingecko

async def get_multiple_supply_data(coin_ids: List[str]):
    """
    Fetch supply data for multiple coins concurrently
    """
    async_client = AsyncCoingecko(
        pro_api_key=os.environ.get("COINGECKO_PRO_API_KEY"),
        environment="pro"
    )
    
    async def fetch_single_coin(coin_id: str):
        try:
            return await async_client.coins.circulating_supply_chart.get(
                id=coin_id,
                days="30"
            )
        except Exception as e:
            print(f"Error fetching {coin_id}: {e}")
            return None
    
    # Fetch all coins concurrently
    tasks = [fetch_single_coin(coin_id) for coin_id in coin_ids]
    results = await asyncio.gather(*tasks)
    
    await async_client.close()
    return dict(zip(coin_ids, results))

# Usage
async def main():
    coins = ["bitcoin", "ethereum", "cardano", "solana"]
    results = await get_multiple_supply_data(coins)
    
    for coin_id, data in results.items():
        if data:
            print(f"{coin_id}: {len(data.circulating_supply)} data points")

# Run the async function
asyncio.run(main())
```

---

## Additional Resources

- [CoinGecko Pro API Documentation](https://docs.coingecko.com)
- [CoinGecko Python SDK GitHub](https://github.com/coingecko/coingecko-python)
- [CoinGecko API Pricing](https://www.coingecko.com/en/api/pricing)
- [Epoch Converter](https://www.epochconverter.com/) - For UNIX timestamp conversion

---

*This guide was created to help developers efficiently fetch and analyze cryptocurrency supply data using the CoinGecko Pro API. For the most up-to-date information, always refer to the official CoinGecko API documentation.*