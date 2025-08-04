# CoinGecko Pro API: Best Practices for Batched API Calls

A comprehensive guide for efficiently making batched API calls to the CoinGecko Pro API, covering all types of batch operations, rate limiting strategies, and performance optimization techniques.

## Table of Contents

- [Overview](#overview)
- [Types of Batch Operations](#types-of-batch-operations)
- [Core Batching Endpoints](#core-batching-endpoints)
- [Batch Operation Patterns](#batch-operation-patterns)
- [Rate Limiting and Performance](#rate-limiting-and-performance)
- [Error Handling Strategies](#error-handling-strategies)
- [Advanced Batching Techniques](#advanced-batching-techniques)
- [Production Implementation Examples](#production-implementation-examples)
- [Monitoring and Optimization](#monitoring-and-optimization)

## Overview

The CoinGecko Pro API supports various types of batch operations that allow you to efficiently retrieve data for multiple assets, pools, exchanges, or time periods in single API calls. Proper batching is essential for:

- **Minimizing API calls** and staying within rate limits
- **Reducing latency** by consolidating requests
- **Improving application performance** 
- **Optimizing cost** for metered API plans

## Types of Batch Operations

### 1. **Multi-Asset Batching**
Retrieve data for multiple coins/tokens in a single request.

### 2. **Paginated Batching**
Handle large datasets using pagination parameters.

### 3. **Time-Range Batching**
Fetch historical data across multiple time periods.

### 4. **Multi-Network Batching**
Query data across different blockchain networks.

### 5. **Cross-Category Batching**
Combine different data types (coins, pools, exchanges) strategically.

## Core Batching Endpoints

### 1. Multi-Asset Price Data

#### `/simple/price` - Multiple Coin Prices
**Best for:** Real-time price data for multiple coins

```python
# Single call for multiple coins
import os
import requests
from typing import List, Dict, Optional
import time

def get_multiple_coin_prices(
    api_key: str, 
    coin_ids: List[str], 
    vs_currencies: List[str] = ["usd"],
    include_market_cap: bool = True,
    include_24hr_vol: bool = True,
    include_24hr_change: bool = True
) -> Optional[Dict]:
    """Get prices for multiple coins in a single API call."""
    
    # CoinGecko allows up to ~1000 characters in the ids parameter
    # Typically 50-100 coin IDs depending on length
    if len(coin_ids) > 100:
        raise ValueError("Too many coin IDs. Consider splitting into smaller batches.")
    
    url = "https://pro-api.coingecko.com/api/v3/simple/price"
    params = {
        "ids": ",".join(coin_ids),
        "vs_currencies": ",".join(vs_currencies),
        "include_market_cap": str(include_market_cap).lower(),
        "include_24hr_vol": str(include_24hr_vol).lower(),
        "include_24hr_change": str(include_24hr_change).lower(),
        "x_cg_pro_api_key": api_key
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching prices: {e}")
        return None

# Example usage
api_key = os.environ.get("COINGECKO_PRO_API_KEY")
major_coins = [
    "bitcoin", "ethereum", "solana", "cardano", "polygon-ecosystem-token",
    "chainlink", "avalanche-2", "polkadot", "uniswap", "litecoin"
]

prices = get_multiple_coin_prices(api_key, major_coins)
if prices:
    for coin_id, data in prices.items():
        print(f"{coin_id}: ${data['usd']} (24h: {data.get('usd_24h_change', 0):.2f}%)")
```

#### `/simple/token_price/{id}` - Multiple Token Addresses
**Best for:** Multiple tokens on same blockchain

```python
def get_multiple_token_prices(
    api_key: str,
    platform_id: str,  # e.g., "ethereum", "binance-smart-chain"
    contract_addresses: List[str],
    vs_currencies: List[str] = ["usd"]
) -> Optional[Dict]:
    """Get prices for multiple tokens by contract address."""
    
    if len(contract_addresses) > 50:
        raise ValueError("Too many contract addresses. Consider splitting into smaller batches.")
    
    url = f"https://pro-api.coingecko.com/api/v3/simple/token_price/{platform_id}"
    params = {
        "contract_addresses": ",".join(contract_addresses),
        "vs_currencies": ",".join(vs_currencies),
        "include_market_cap": "true",
        "include_24hr_vol": "true",
        "include_24hr_change": "true",
        "x_cg_pro_api_key": api_key
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching token prices: {e}")
        return None

# Example: Multiple Ethereum tokens
ethereum_tokens = [
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",  # USDC
    "0xdac17f958d2ee523a2206206994597c13d831ec7",  # USDT
    "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599",  # WBTC
    "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984",  # UNI
    "0x514910771af9ca656af840dff83e8264ecf986ca"   # LINK
]

token_prices = get_multiple_token_prices(api_key, "ethereum", ethereum_tokens)
```

### 2. Market Data Batching

#### `/coins/markets` - Paginated Market Data
**Best for:** Large-scale market data with pagination

```python
def get_market_data_batch(
    api_key: str,
    vs_currency: str = "usd",
    order: str = "market_cap_desc",
    per_page: int = 250,  # Maximum allowed
    page: int = 1,
    sparkline: bool = False,
    price_change_percentage: str = "24h,7d,30d",
    coin_ids: Optional[List[str]] = None,
    category: Optional[str] = None
) -> Optional[List[Dict]]:
    """Get market data for multiple coins with pagination."""
    
    url = "https://pro-api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": vs_currency,
        "order": order,
        "per_page": per_page,
        "page": page,
        "sparkline": str(sparkline).lower(),
        "price_change_percentage": price_change_percentage,
        "x_cg_pro_api_key": api_key
    }
    
    # Optional filters
    if coin_ids:
        params["ids"] = ",".join(coin_ids)
    if category:
        params["category"] = category
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching market data: {e}")
        return None

def get_all_market_data(
    api_key: str,
    max_coins: int = 1000,
    **kwargs
) -> List[Dict]:
    """Get market data for multiple pages."""
    all_data = []
    page = 1
    per_page = min(250, max_coins)  # API max is 250
    
    while len(all_data) < max_coins:
        remaining = max_coins - len(all_data)
        current_per_page = min(per_page, remaining)
        
        data = get_market_data_batch(
            api_key, 
            per_page=current_per_page, 
            page=page, 
            **kwargs
        )
        
        if not data or len(data) == 0:
            break
            
        all_data.extend(data)
        
        # Rate limiting: wait between requests
        if len(data) == current_per_page:  # More pages available
            time.sleep(0.2)  # 200ms delay
            page += 1
        else:
            break
    
    return all_data[:max_coins]

# Example: Get top 500 coins by market cap
market_data = get_all_market_data(api_key, max_coins=500)
print(f"Retrieved data for {len(market_data)} coins")
```

### 3. On-Chain Data Batching

#### `/onchain/networks/{network}/pools/multi/{addresses}` - Multiple Pool Data
**Best for:** Multiple DEX pools on same network

```python
def get_multiple_pools_data(
    api_key: str,
    network: str,  # e.g., "eth", "bsc", "polygon-pos"
    pool_addresses: List[str],
    include: Optional[str] = None  # e.g., "base_token,quote_token"
) -> Optional[Dict]:
    """Get data for multiple pools in a single call."""
    
    if len(pool_addresses) > 30:
        raise ValueError("Too many pool addresses. API typically supports ~30 pools per call.")
    
    addresses_param = ",".join(pool_addresses)
    url = f"https://pro-api.coingecko.com/api/v3/onchain/networks/{network}/pools/multi/{addresses_param}"
    
    params = {"x_cg_pro_api_key": api_key}
    if include:
        params["include"] = include
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching pool data: {e}")
        return None

# Example: Multiple Uniswap V3 pools on Ethereum
uniswap_pools = [
    "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640",  # USDC/WETH
    "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8",  # USDC/ETH
    "0x5777d92f208679db4b9778590fa3cab3ac9e2168",  # DAI/USDC
    "0x6c6bc977e13df9b0de53b251522280bb72383700",  # DAI/USDC
]

pools_data = get_multiple_pools_data(api_key, "eth", uniswap_pools)
```

#### `/onchain/networks/{network}/tokens/multi/{addresses}` - Multiple Token Data
**Best for:** Multiple tokens on same network

```python
def get_multiple_tokens_data(
    api_key: str,
    network: str,
    token_addresses: List[str],
    include: Optional[str] = None
) -> Optional[Dict]:
    """Get on-chain data for multiple tokens."""
    
    if len(token_addresses) > 30:
        raise ValueError("Too many token addresses for single call.")
    
    addresses_param = ",".join(token_addresses)
    url = f"https://pro-api.coingecko.com/api/v3/onchain/networks/{network}/tokens/multi/{addresses_param}"
    
    params = {"x_cg_pro_api_key": api_key}
    if include:
        params["include"] = include
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching token data: {e}")
        return None

# Example: Multiple tokens on Ethereum
ethereum_tokens = [
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",  # USDC
    "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",  # WETH
    "0x6b175474e89094c44da98b954eedeac495271d0f",  # DAI
]

tokens_data = get_multiple_tokens_data(api_key, "eth", ethereum_tokens)
```

## Batch Operation Patterns

### 1. **Chunking Strategy**
Split large datasets into optimal chunk sizes.

```python
def chunk_list(items: List, chunk_size: int) -> List[List]:
    """Split a list into chunks of specified size."""
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

def batch_process_coins(
    api_key: str,
    coin_ids: List[str],
    chunk_size: int = 50,
    delay_between_chunks: float = 0.2
) -> Dict:
    """Process large list of coins in chunks."""
    
    chunks = chunk_list(coin_ids, chunk_size)
    all_results = {}
    
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)} ({len(chunk)} coins)")
        
        result = get_multiple_coin_prices(api_key, chunk)
        if result:
            all_results.update(result)
        
        # Rate limiting delay between chunks
        if i < len(chunks) - 1:  # Don't sleep after last chunk
            time.sleep(delay_between_chunks)
    
    return all_results

# Example: Process 500 coins in chunks of 50
large_coin_list = [f"coin-{i}" for i in range(500)]  # Replace with real coin IDs
results = batch_process_coins(api_key, large_coin_list)
```

### 2. **Parallel Processing with Rate Limiting**
Use concurrent requests while respecting rate limits.

```python
import asyncio
import aiohttp
from typing import List, Dict, Optional
import time

class CoinGeckoBatchClient:
    """Async batch client with rate limiting."""
    
    def __init__(self, api_key: str, rate_limit: int = 500):
        self.api_key = api_key
        self.base_url = "https://pro-api.coingecko.com/api/v3"
        self.rate_limit = rate_limit  # requests per minute
        self.request_times = []
        
    async def _rate_limit_check(self):
        """Ensure we don't exceed rate limits."""
        now = time.time()
        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times if now - t < 60]
        
        if len(self.request_times) >= self.rate_limit:
            # Wait until we can make another request
            sleep_time = 60 - (now - self.request_times[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
                
        self.request_times.append(now)
    
    async def _make_request(self, session: aiohttp.ClientSession, endpoint: str, params: Dict) -> Optional[Dict]:
        """Make rate-limited async request."""
        await self._rate_limit_check()
        
        params["x_cg_pro_api_key"] = self.api_key
        url = f"{self.base_url}/{endpoint}"
        
        try:
            async with session.get(url, params=params, timeout=30) as response:
                if response.status == 429:
                    # Rate limit hit, wait and retry once
                    await asyncio.sleep(60)
                    async with session.get(url, params=params, timeout=30) as retry_response:
                        retry_response.raise_for_status()
                        return await retry_response.json()
                
                response.raise_for_status()
                return await response.json()
                
        except Exception as e:
            print(f"Request failed: {e}")
            return None
    
    async def batch_get_prices(self, coin_chunks: List[List[str]]) -> Dict:
        """Get prices for multiple chunks concurrently."""
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            for chunk in coin_chunks:
                params = {
                    "ids": ",".join(chunk),
                    "vs_currencies": "usd",
                    "include_market_cap": "true",
                    "include_24hr_vol": "true",
                    "include_24hr_change": "true"
                }
                
                task = self._make_request(session, "simple/price", params)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine results
            combined = {}
            for result in results:
                if isinstance(result, dict):
                    combined.update(result)
                elif isinstance(result, Exception):
                    print(f"Task failed: {result}")
            
            return combined

# Usage example
async def main():
    client = CoinGeckoBatchClient(api_key)
    
    # Split large coin list into chunks
    all_coins = ["bitcoin", "ethereum", "solana"] * 50  # Example large list
    chunks = chunk_list(all_coins, 30)  # 30 coins per chunk
    
    # Process chunks concurrently
    prices = await client.batch_get_prices(chunks)
    print(f"Retrieved prices for {len(prices)} coins")

# Run async batch processing
# asyncio.run(main())
```

### 3. **Smart Batching Strategy**
Combine different API endpoints strategically.

```python
class SmartBatcher:
    """Intelligent batching with endpoint optimization."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        
    def get_comprehensive_market_data(
        self,
        coin_ids: List[str],
        include_historical: bool = False,
        include_on_chain: bool = False
    ) -> Dict:
        """Get comprehensive data using optimal endpoint combination."""
        
        results = {
            "current_prices": {},
            "market_data": {},
            "historical_data": {},
            "on_chain_data": {}
        }
        
        # 1. Get current prices (most efficient for price-only data)
        if len(coin_ids) <= 100:
            prices = get_multiple_coin_prices(self.api_key, coin_ids)
            if prices:
                results["current_prices"] = prices
        
        # 2. Get detailed market data (includes additional metrics)
        market_data = get_market_data_batch(
            self.api_key,
            coin_ids=coin_ids,
            per_page=min(250, len(coin_ids))
        )
        if market_data:
            results["market_data"] = {coin["id"]: coin for coin in market_data}
        
        # 3. Optionally get historical data (separate calls needed)
        if include_historical:
            for coin_id in coin_ids[:10]:  # Limit to avoid too many calls
                historical = self._get_historical_data(coin_id)
                if historical:
                    results["historical_data"][coin_id] = historical
                time.sleep(0.1)  # Rate limiting
        
        return results
    
    def _get_historical_data(self, coin_id: str, days: int = 30) -> Optional[Dict]:
        """Get historical data for a single coin."""
        url = f"https://pro-api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {
            "vs_currency": "usd",
            "days": days,
            "x_cg_pro_api_key": self.api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching historical data for {coin_id}: {e}")
            return None

# Usage
batcher = SmartBatcher(api_key)
comprehensive_data = batcher.get_comprehensive_market_data(
    coin_ids=["bitcoin", "ethereum", "solana"],
    include_historical=True
)
```

## Rate Limiting and Performance

### 1. **Rate Limit Management**

```python
import time
from collections import deque
from typing import Deque

class RateLimiter:
    """Thread-safe rate limiter for API calls."""
    
    def __init__(self, max_calls: int = 500, time_window: int = 60):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls: Deque[float] = deque()
    
    def acquire(self) -> float:
        """Acquire permission to make an API call. Returns wait time."""
        now = time.time()
        
        # Remove old calls outside the time window
        while self.calls and self.calls[0] <= now - self.time_window:
            self.calls.popleft()
        
        if len(self.calls) >= self.max_calls:
            # Calculate wait time
            oldest_call = self.calls[0]
            wait_time = self.time_window - (now - oldest_call)
            return max(0, wait_time)
        
        self.calls.append(now)
        return 0

# Usage with rate limiter
rate_limiter = RateLimiter(max_calls=500, time_window=60)

def rate_limited_request(url: str, params: Dict) -> Optional[Dict]:
    """Make a rate-limited API request."""
    wait_time = rate_limiter.acquire()
    if wait_time > 0:
        print(f"Rate limit reached. Waiting {wait_time:.2f} seconds...")
        time.sleep(wait_time)
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Request failed: {e}")
        return None
```

### 2. **Adaptive Batching**

```python
class AdaptiveBatcher:
    """Dynamically adjust batch sizes based on performance."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.optimal_batch_size = 50
        self.performance_history = []
    
    def adaptive_batch_process(self, items: List[str]) -> Dict:
        """Process items with adaptive batch sizing."""
        results = {}
        remaining_items = items.copy()
        
        while remaining_items:
            batch_size = min(self.optimal_batch_size, len(remaining_items))
            batch = remaining_items[:batch_size]
            remaining_items = remaining_items[batch_size:]
            
            start_time = time.time()
            batch_result = get_multiple_coin_prices(self.api_key, batch)
            end_time = time.time()
            
            if batch_result:
                results.update(batch_result)
                
                # Track performance
                request_time = end_time - start_time
                throughput = len(batch) / request_time
                self.performance_history.append({
                    "batch_size": batch_size,
                    "request_time": request_time,
                    "throughput": throughput
                })
                
                # Adjust batch size based on performance
                self._adjust_batch_size()
            
            # Rate limiting
            if remaining_items:
                time.sleep(0.2)
        
        return results
    
    def _adjust_batch_size(self):
        """Adjust optimal batch size based on performance history."""
        if len(self.performance_history) < 3:
            return
        
        # Get recent performance data
        recent = self.performance_history[-3:]
        avg_throughput = sum(p["throughput"] for p in recent) / len(recent)
        avg_request_time = sum(p["request_time"] for p in recent) / len(recent)
        
        # Adjust batch size based on performance
        if avg_request_time > 5.0:  # Too slow, reduce batch size
            self.optimal_batch_size = max(10, int(self.optimal_batch_size * 0.8))
        elif avg_request_time < 1.0 and avg_throughput > 30:  # Fast, increase batch size
            self.optimal_batch_size = min(100, int(self.optimal_batch_size * 1.2))
        
        print(f"Adjusted batch size to: {self.optimal_batch_size}")
```

## Error Handling Strategies

### 1. **Comprehensive Error Handling**

```python
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Callable

class ErrorType(Enum):
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    NETWORK = "network"
    AUTHENTICATION = "authentication"
    NOT_FOUND = "not_found"
    SERVER_ERROR = "server_error"
    UNKNOWN = "unknown"

@dataclass
class APIError:
    error_type: ErrorType
    message: str
    status_code: Optional[int] = None
    retry_after: Optional[int] = None

class ErrorHandler:
    """Robust error handling for API calls."""
    
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self.logger = logging.getLogger(__name__)
    
    def handle_request_with_retry(
        self,
        request_func: Callable,
        *args,
        **kwargs
    ) -> tuple[Optional[Dict], Optional[APIError]]:
        """Execute request with intelligent retry logic."""
        
        for attempt in range(self.max_retries + 1):
            try:
                result = request_func(*args, **kwargs)
                return result, None
                
            except requests.exceptions.HTTPError as e:
                error = self._categorize_http_error(e)
                
                if error.error_type == ErrorType.RATE_LIMIT:
                    if attempt < self.max_retries:
                        wait_time = error.retry_after or 60
                        self.logger.warning(f"Rate limit hit. Waiting {wait_time}s (attempt {attempt + 1})")
                        time.sleep(wait_time)
                        continue
                
                elif error.error_type == ErrorType.SERVER_ERROR:
                    if attempt < self.max_retries:
                        wait_time = 2 ** attempt  # Exponential backoff
                        self.logger.warning(f"Server error. Retrying in {wait_time}s (attempt {attempt + 1})")
                        time.sleep(wait_time)
                        continue
                
                # Non-retryable errors or max retries reached
                return None, error
                
            except requests.exceptions.Timeout as e:
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt
                    self.logger.warning(f"Timeout. Retrying in {wait_time}s (attempt {attempt + 1})")
                    time.sleep(wait_time)
                    continue
                return None, APIError(ErrorType.TIMEOUT, str(e))
                
            except requests.exceptions.RequestException as e:
                return None, APIError(ErrorType.NETWORK, str(e))
                
            except Exception as e:
                return None, APIError(ErrorType.UNKNOWN, str(e))
        
        return None, APIError(ErrorType.UNKNOWN, "Max retries exceeded")
    
    def _categorize_http_error(self, error: requests.exceptions.HTTPError) -> APIError:
        """Categorize HTTP errors for appropriate handling."""
        status_code = error.response.status_code
        
        if status_code == 429:
            retry_after = error.response.headers.get("Retry-After")
            return APIError(
                ErrorType.RATE_LIMIT,
                "Rate limit exceeded",
                status_code,
                int(retry_after) if retry_after else None
            )
        elif status_code == 401:
            return APIError(ErrorType.AUTHENTICATION, "Invalid API key", status_code)
        elif status_code == 404:
            return APIError(ErrorType.NOT_FOUND, "Resource not found", status_code)
        elif 500 <= status_code < 600:
            return APIError(ErrorType.SERVER_ERROR, "Server error", status_code)
        else:
            return APIError(ErrorType.UNKNOWN, f"HTTP {status_code}", status_code)

# Usage with error handling
error_handler = ErrorHandler(max_retries=3)

def robust_get_prices(api_key: str, coin_ids: List[str]) -> Optional[Dict]:
    """Get prices with robust error handling."""
    
    def _request():
        return get_multiple_coin_prices(api_key, coin_ids)
    
    result, error = error_handler.handle_request_with_retry(_request)
    
    if error:
        logging.error(f"Failed to get prices: {error.message}")
        return None
    
    return result
```

### 2. **Circuit Breaker Pattern**

```python
import time
from enum import Enum

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit is open, failing fast
    HALF_OPEN = "half_open"  # Testing if service is recovered

class CircuitBreaker:
    """Circuit breaker for API resilience."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: int = 60,
        expected_exception: type = requests.exceptions.RequestException
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit."""
        return (
            self.last_failure_time and
            time.time() - self.last_failure_time >= self.timeout
        )
    
    def _on_success(self):
        """Handle successful request."""
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def _on_failure(self):
        """Handle failed request."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

# Usage with circuit breaker
circuit_breaker = CircuitBreaker(failure_threshold=3, timeout=60)

def protected_api_call(api_key: str, coin_ids: List[str]) -> Optional[Dict]:
    """API call protected by circuit breaker."""
    try:
        return circuit_breaker.call(get_multiple_coin_prices, api_key, coin_ids)
    except Exception as e:
        print(f"Circuit breaker prevented call: {e}")
        return None
```

## Advanced Batching Techniques

### 1. **Dynamic Endpoint Selection**

```python
class EndpointSelector:
    """Intelligently select optimal endpoints based on request characteristics."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.endpoint_performance = {
            "simple/price": {"avg_time": 0.8, "max_items": 100},
            "coins/markets": {"avg_time": 1.2, "max_items": 250},
            "simple/token_price": {"avg_time": 0.6, "max_items": 50}
        }
    
    def select_optimal_strategy(
        self,
        request_type: str,
        item_count: int,
        include_detailed_data: bool = False
    ) -> str:
        """Select the best endpoint and strategy for the request."""
        
        if request_type == "coin_prices":
            if item_count <= 50 and not include_detailed_data:
                return "simple/price"
            elif item_count <= 250:
                return "coins/markets"
            else:
                return "chunked_simple/price"
        
        elif request_type == "token_prices":
            if item_count <= 50:
                return "simple/token_price"
            else:
                return "chunked_simple/token_price"
        
        return "simple/price"  # Default fallback
    
    def execute_optimal_strategy(
        self,
        strategy: str,
        items: List[str],
        **kwargs
    ) -> Optional[Dict]:
        """Execute the optimal strategy for data retrieval."""
        
        if strategy == "simple/price":
            return get_multiple_coin_prices(self.api_key, items, **kwargs)
        
        elif strategy == "coins/markets":
            market_data = get_market_data_batch(self.api_key, coin_ids=items, **kwargs)
            if market_data:
                return {coin["id"]: {"usd": coin["current_price"]} for coin in market_data}
        
        elif strategy.startswith("chunked_"):
            base_strategy = strategy.replace("chunked_", "")
            return self._execute_chunked_strategy(base_strategy, items, **kwargs)
        
        return None
    
    def _execute_chunked_strategy(
        self,
        base_strategy: str,
        items: List[str],
        chunk_size: int = 50,
        **kwargs
    ) -> Dict:
        """Execute a chunked version of the strategy."""
        results = {}
        chunks = chunk_list(items, chunk_size)
        
        for chunk in chunks:
            if base_strategy == "simple/price":
                chunk_result = get_multiple_coin_prices(self.api_key, chunk, **kwargs)
            else:
                chunk_result = get_multiple_coin_prices(self.api_key, chunk, **kwargs)
            
            if chunk_result:
                results.update(chunk_result)
            
            time.sleep(0.2)  # Rate limiting between chunks
        
        return results
```

### 2. **Intelligent Caching Strategy**

```python
import json
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

class APICache:
    """Intelligent caching for API responses."""
    
    def __init__(self, cache_dir: str = "./api_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Different TTL for different data types
        self.ttl_config = {
            "prices": 60,      # 1 minute for prices
            "market_data": 300,  # 5 minutes for market data
            "historical": 3600,  # 1 hour for historical data
            "pools": 300,      # 5 minutes for pool data
            "static": 86400    # 1 day for static data (coin lists, etc.)
        }
    
    def _generate_cache_key(self, endpoint: str, params: Dict) -> str:
        """Generate a unique cache key for the request."""
        # Remove API key from cache key generation
        cache_params = {k: v for k, v in params.items() if k != "x_cg_pro_api_key"}
        
        key_string = f"{endpoint}:{json.dumps(cache_params, sort_keys=True)}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, endpoint: str, params: Dict, data_type: str = "prices") -> Optional[Dict]:
        """Get cached data if available and not expired."""
        cache_key = self._generate_cache_key(endpoint, params)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
            
            # Check if cache is expired
            cached_time = datetime.fromisoformat(cached_data["timestamp"])
            ttl = self.ttl_config.get(data_type, 300)
            
            if datetime.now() - cached_time > timedelta(seconds=ttl):
                cache_file.unlink()  # Remove expired cache
                return None
            
            return cached_data["data"]
            
        except Exception:
            # If there's any error reading cache, remove it
            cache_file.unlink()
            return None
    
    def set(self, endpoint: str, params: Dict, data: Dict, data_type: str = "prices"):
        """Cache the API response."""
        cache_key = self._generate_cache_key(endpoint, params)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        cached_data = {
            "timestamp": datetime.now().isoformat(),
            "data_type": data_type,
            "data": data
        }
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(cached_data, f)
        except Exception as e:
            print(f"Failed to cache data: {e}")
    
    def clear_expired(self):
        """Clear all expired cache entries."""
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                
                cached_time = datetime.fromisoformat(cached_data["timestamp"])
                data_type = cached_data.get("data_type", "prices")
                ttl = self.ttl_config.get(data_type, 300)
                
                if datetime.now() - cached_time > timedelta(seconds=ttl):
                    cache_file.unlink()
                    
            except Exception:
                cache_file.unlink()  # Remove corrupted cache files

# Cached API client
class CachedAPIClient:
    """API client with intelligent caching."""
    
    def __init__(self, api_key: str, cache_dir: str = "./api_cache"):
        self.api_key = api_key
        self.cache = APICache(cache_dir)
        self.base_url = "https://pro-api.coingecko.com/api/v3"
    
    def cached_request(
        self,
        endpoint: str,
        params: Dict,
        data_type: str = "prices",
        force_refresh: bool = False
    ) -> Optional[Dict]:
        """Make a cached API request."""
        
        # Check cache first (unless force refresh)
        if not force_refresh:
            cached_result = self.cache.get(endpoint, params, data_type)
            if cached_result:
                print(f"Cache hit for {endpoint}")
                return cached_result
        
        # Make fresh API request
        params["x_cg_pro_api_key"] = self.api_key
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Cache the result
            self.cache.set(endpoint, params, data, data_type)
            print(f"Fresh data cached for {endpoint}")
            
            return data
            
        except Exception as e:
            print(f"API request failed: {e}")
            return None
    
    def get_prices_with_cache(self, coin_ids: List[str]) -> Optional[Dict]:
        """Get coin prices with caching."""
        params = {
            "ids": ",".join(coin_ids),
            "vs_currencies": "usd",
            "include_market_cap": "true",
            "include_24hr_vol": "true",
            "include_24hr_change": "true"
        }
        
        return self.cached_request("simple/price", params, "prices")

# Usage with caching
cached_client = CachedAPIClient(api_key)
prices = cached_client.get_prices_with_cache(["bitcoin", "ethereum", "solana"])
```

## Production Implementation Examples

### 1. **Enterprise-Grade Batch Processor**

```python
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Optional, Callable
import json
from dataclasses import dataclass
from datetime import datetime

@dataclass
class BatchJob:
    """Represents a batch processing job."""
    job_id: str
    endpoint: str
    items: List[str]
    params: Dict
    priority: int = 0
    retry_count: int = 0
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

class EnterpriseBatchProcessor:
    """Production-ready batch processor for CoinGecko API."""
    
    def __init__(
        self,
        api_key: str,
        max_concurrent_jobs: int = 10,
        rate_limit: int = 500,
        cache_enabled: bool = True,
        persistence_dir: str = "./batch_jobs"
    ):
        self.api_key = api_key
        self.max_concurrent_jobs = max_concurrent_jobs
        self.rate_limiter = RateLimiter(rate_limit, 60)
        self.cache = APICache() if cache_enabled else None
        self.persistence_dir = Path(persistence_dir)
        self.persistence_dir.mkdir(exist_ok=True)
        
        # Job management
        self.job_queue = asyncio.Queue()
        self.active_jobs = {}
        self.completed_jobs = {}
        self.failed_jobs = {}
        
        # Setup logging
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging."""
        logger = logging.getLogger("batch_processor")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    async def submit_job(self, job: BatchJob) -> str:
        """Submit a batch job for processing."""
        await self.job_queue.put(job)
        self.active_jobs[job.job_id] = job
        
        # Persist job for recovery
        self._persist_job(job)
        
        self.logger.info(f"Job {job.job_id} submitted with {len(job.items)} items")
        return job.job_id
    
    async def process_jobs(self):
        """Main job processing loop."""
        workers = [
            asyncio.create_task(self._worker(f"worker-{i}"))
            for i in range(self.max_concurrent_jobs)
        ]
        
        try:
            await asyncio.gather(*workers)
        except Exception as e:
            self.logger.error(f"Job processing error: {e}")
        finally:
            for worker in workers:
                worker.cancel()
    
    async def _worker(self, worker_name: str):
        """Individual worker for processing jobs."""
        while True:
            try:
                job = await self.job_queue.get()
                self.logger.info(f"{worker_name} processing job {job.job_id}")
                
                result = await self._process_single_job(job)
                
                if result:
                    self.completed_jobs[job.job_id] = {
                        "job": job,
                        "result": result,
                        "completed_at": datetime.now()
                    }
                    self.logger.info(f"Job {job.job_id} completed successfully")
                else:
                    self._handle_job_failure(job)
                
                # Remove from active jobs
                self.active_jobs.pop(job.job_id, None)
                self.job_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Worker {worker_name} error: {e}")
                await asyncio.sleep(1)
    
    async def _process_single_job(self, job: BatchJob) -> Optional[Dict]:
        """Process a single batch job."""
        try:
            # Check cache first
            if self.cache:
                cached_result = self.cache.get(job.endpoint, job.params)
                if cached_result:
                    return cached_result
            
            # Rate limiting
            wait_time = self.rate_limiter.acquire()
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            
            # Execute the actual API call
            if job.endpoint == "simple/price":
                result = await self._async_get_prices(job.items, job.params)
            elif job.endpoint == "coins/markets":
                result = await self._async_get_market_data(job.items, job.params)
            else:
                raise ValueError(f"Unsupported endpoint: {job.endpoint}")
            
            # Cache the result
            if result and self.cache:
                self.cache.set(job.endpoint, job.params, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Job {job.job_id} failed: {e}")
            return None
    
    async def _async_get_prices(self, coin_ids: List[str], params: Dict) -> Optional[Dict]:
        """Async version of get_multiple_coin_prices."""
        url = "https://pro-api.coingecko.com/api/v3/simple/price"
        request_params = {
            **params,
            "ids": ",".join(coin_ids),
            "x_cg_pro_api_key": self.api_key
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=request_params) as response:
                response.raise_for_status()
                return await response.json()
    
    async def _async_get_market_data(self, coin_ids: List[str], params: Dict) -> Optional[List[Dict]]:
        """Async version of get_market_data_batch."""
        url = "https://pro-api.coingecko.com/api/v3/coins/markets"
        request_params = {
            **params,
            "ids": ",".join(coin_ids) if coin_ids else None,
            "x_cg_pro_api_key": self.api_key
        }
        
        # Remove None values
        request_params = {k: v for k, v in request_params.items() if v is not None}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=request_params) as response:
                response.raise_for_status()
                return await response.json()
    
    def _handle_job_failure(self, job: BatchJob):
        """Handle failed job with retry logic."""
        job.retry_count += 1
        
        if job.retry_count <= 3:
            # Exponential backoff retry
            delay = 2 ** job.retry_count
            self.logger.warning(f"Job {job.job_id} failed, retrying in {delay}s (attempt {job.retry_count})")
            
            # Re-queue the job
            asyncio.create_task(self._delayed_retry(job, delay))
        else:
            # Max retries exceeded
            self.failed_jobs[job.job_id] = {
                "job": job,
                "failed_at": datetime.now()
            }
            self.logger.error(f"Job {job.job_id} failed permanently after {job.retry_count} attempts")
    
    async def _delayed_retry(self, job: BatchJob, delay: float):
        """Retry a job after a delay."""
        await asyncio.sleep(delay)
        await self.submit_job(job)
    
    def _persist_job(self, job: BatchJob):
        """Persist job state for recovery."""
        job_file = self.persistence_dir / f"{job.job_id}.json"
        
        try:
            with open(job_file, 'w') as f:
                json.dump({
                    "job_id": job.job_id,
                    "endpoint": job.endpoint,
                    "items": job.items,
                    "params": job.params,
                    "priority": job.priority,
                    "retry_count": job.retry_count,
                    "created_at": job.created_at.isoformat()
                }, f)
        except Exception as e:
            self.logger.error(f"Failed to persist job {job.job_id}: {e}")
    
    def get_job_status(self, job_id: str) -> Dict:
        """Get the status of a job."""
        if job_id in self.completed_jobs:
            return {"status": "completed", "data": self.completed_jobs[job_id]}
        elif job_id in self.active_jobs:
            return {"status": "active", "data": self.active_jobs[job_id]}
        elif job_id in self.failed_jobs:
            return {"status": "failed", "data": self.failed_jobs[job_id]}
        else:
            return {"status": "not_found"}

# Usage example
async def enterprise_example():
    processor = EnterpriseBatchProcessor(api_key, max_concurrent_jobs=5)
    
    # Start the processor
    processor_task = asyncio.create_task(processor.process_jobs())
    
    # Submit various batch jobs
    jobs = []
    
    # Job 1: Get prices for major coins
    major_coins_job = BatchJob(
        job_id="major_coins_prices",
        endpoint="simple/price",
        items=["bitcoin", "ethereum", "solana", "cardano"],
        params={"vs_currencies": "usd", "include_market_cap": "true"}
    )
    job1_id = await processor.submit_job(major_coins_job)
    jobs.append(job1_id)
    
    # Job 2: Get market data for DeFi tokens
    defi_tokens_job = BatchJob(
        job_id="defi_market_data",
        endpoint="coins/markets",
        items=["uniswap", "chainlink", "aave", "compound-governance-token"],
        params={"vs_currency": "usd", "order": "market_cap_desc", "per_page": 10}
    )
    job2_id = await processor.submit_job(defi_tokens_job)
    jobs.append(job2_id)
    
    # Wait for jobs to complete
    await asyncio.sleep(10)
    
    # Check job statuses
    for job_id in jobs:
        status = processor.get_job_status(job_id)
        print(f"Job {job_id}: {status['status']}")
    
    # Stop the processor
    processor_task.cancel()

# Run the enterprise example
# asyncio.run(enterprise_example())
```

## Monitoring and Optimization

### 1. **Performance Monitoring**

```python
import time
from dataclasses import dataclass
from typing import Dict, List
import statistics

@dataclass
class RequestMetrics:
    """Metrics for individual API requests."""
    endpoint: str
    item_count: int
    response_time: float
    success: bool
    timestamp: float
    bytes_transferred: int = 0

class PerformanceMonitor:
    """Monitor API performance and optimize batch strategies."""
    
    def __init__(self):
        self.metrics: List[RequestMetrics] = []
        self.endpoint_stats = {}
    
    def record_request(
        self,
        endpoint: str,
        item_count: int,
        response_time: float,
        success: bool,
        bytes_transferred: int = 0
    ):
        """Record metrics for a request."""
        metric = RequestMetrics(
            endpoint=endpoint,
            item_count=item_count,
            response_time=response_time,
            success=success,
            timestamp=time.time(),
            bytes_transferred=bytes_transferred
        )
        
        self.metrics.append(metric)
        self._update_endpoint_stats(metric)
    
    def _update_endpoint_stats(self, metric: RequestMetrics):
        """Update aggregated statistics for endpoints."""
        if metric.endpoint not in self.endpoint_stats:
            self.endpoint_stats[metric.endpoint] = {
                "total_requests": 0,
                "successful_requests": 0,
                "total_response_time": 0,
                "total_items": 0,
                "response_times": []
            }
        
        stats = self.endpoint_stats[metric.endpoint]
        stats["total_requests"] += 1
        stats["total_response_time"] += metric.response_time
        stats["total_items"] += metric.item_count
        stats["response_times"].append(metric.response_time)
        
        if metric.success:
            stats["successful_requests"] += 1
    
    def get_performance_report(self) -> Dict:
        """Generate a comprehensive performance report."""
        report = {}
        
        for endpoint, stats in self.endpoint_stats.items():
            if stats["total_requests"] > 0:
                response_times = stats["response_times"]
                
                report[endpoint] = {
                    "total_requests": stats["total_requests"],
                    "success_rate": stats["successful_requests"] / stats["total_requests"],
                    "avg_response_time": stats["total_response_time"] / stats["total_requests"],
                    "avg_items_per_request": stats["total_items"] / stats["total_requests"],
                    "throughput": stats["total_items"] / stats["total_response_time"],
                    "p50_response_time": statistics.median(response_times),
                    "p95_response_time": statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else max(response_times),
                    "min_response_time": min(response_times),
                    "max_response_time": max(response_times)
                }
        
        return report
    
    def get_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on metrics."""
        recommendations = []
        report = self.get_performance_report()
        
        for endpoint, stats in report.items():
            # Check success rate
            if stats["success_rate"] < 0.9:
                recommendations.append(
                    f"{endpoint}: Low success rate ({stats['success_rate']:.2%}). "
                    "Consider reducing batch size or implementing better error handling."
                )
            
            # Check response times
            if stats["avg_response_time"] > 5.0:
                recommendations.append(
                    f"{endpoint}: High average response time ({stats['avg_response_time']:.2f}s). "
                    "Consider reducing batch size."
                )
            
            # Check throughput
            if stats["throughput"] < 10:
                recommendations.append(
                    f"{endpoint}: Low throughput ({stats['throughput']:.1f} items/s). "
                    "Consider increasing batch size if response times allow."
                )
        
        return recommendations

# Decorator for automatic performance monitoring
def monitor_performance(monitor: PerformanceMonitor, endpoint: str):
    """Decorator to automatically monitor API call performance."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = False
            item_count = 0
            
            try:
                # Try to determine item count from arguments
                if "coin_ids" in kwargs:
                    item_count = len(kwargs["coin_ids"])
                elif len(args) > 1 and isinstance(args[1], list):
                    item_count = len(args[1])
                
                result = func(*args, **kwargs)
                success = result is not None
                return result
                
            except Exception as e:
                print(f"Error in {func.__name__}: {e}")
                raise
            finally:
                response_time = time.time() - start_time
                monitor.record_request(endpoint, item_count, response_time, success)
        
        return wrapper
    return decorator

# Usage with performance monitoring
performance_monitor = PerformanceMonitor()

@monitor_performance(performance_monitor, "simple/price")
def monitored_get_prices(api_key: str, coin_ids: List[str]) -> Optional[Dict]:
    """Get prices with automatic performance monitoring."""
    return get_multiple_coin_prices(api_key, coin_ids)

# Example usage and reporting
def performance_example():
    # Make several API calls
    for i in range(10):
        coin_batch = ["bitcoin", "ethereum", "solana"]
        result = monitored_get_prices(api_key, coin_batch)
        time.sleep(1)  # Simulate real usage pattern
    
    # Generate performance report
    report = performance_monitor.get_performance_report()
    print("Performance Report:")
    print(json.dumps(report, indent=2))
    
    # Get optimization recommendations
    recommendations = performance_monitor.get_optimization_recommendations()
    print("\nOptimization Recommendations:")
    for rec in recommendations:
        print(f"- {rec}")
```

### 2. **Cost Optimization**

```python
class CostOptimizer:
    """Optimize API usage for cost efficiency."""
    
    def __init__(self, api_key: str, cost_per_request: float = 0.001):
        self.api_key = api_key
        self.cost_per_request = cost_per_request
        self.request_count = 0
        self.cache_hits = 0
        
    def estimate_cost(self, requests_count: int) -> float:
        """Estimate cost for a number of requests."""
        return requests_count * self.cost_per_request
    
    def optimize_batch_strategy(
        self,
        total_items: int,
        data_freshness_requirement: int = 300  # seconds
    ) -> Dict:
        """Determine optimal batching strategy for cost efficiency."""
        
        # Different strategies with their characteristics
        strategies = {
            "small_batches": {
                "batch_size": 20,
                "requests_needed": (total_items + 19) // 20,
                "cache_friendly": True,
                "freshness_control": "high"
            },
            "medium_batches": {
                "batch_size": 50,
                "requests_needed": (total_items + 49) // 50,
                "cache_friendly": True,
                "freshness_control": "medium"
            },
            "large_batches": {
                "batch_size": 100,
                "requests_needed": (total_items + 99) // 100,
                "cache_friendly": False,
                "freshness_control": "low"
            }
        }
        
        # Calculate cost for each strategy
        for name, strategy in strategies.items():
            base_cost = self.estimate_cost(strategy["requests_needed"])
            
            # Apply cache efficiency multiplier
            cache_multiplier = 0.3 if strategy["cache_friendly"] else 0.8
            estimated_cost = base_cost * cache_multiplier
            
            strategy["estimated_cost"] = estimated_cost
            strategy["cost_per_item"] = estimated_cost / total_items
        
        # Find most cost-effective strategy
        best_strategy = min(strategies.items(), key=lambda x: x[1]["estimated_cost"])
        
        return {
            "recommended_strategy": best_strategy[0],
            "strategies": strategies,
            "total_items": total_items
        }
    
    def track_usage(self):
        """Track API usage for cost monitoring."""
        self.request_count += 1
        
    def get_usage_report(self) -> Dict:
        """Generate usage and cost report."""
        total_cost = self.request_count * self.cost_per_request
        cache_efficiency = self.cache_hits / max(1, self.request_count + self.cache_hits)
        
        return {
            "total_requests": self.request_count,
            "cache_hits": self.cache_hits,
            "cache_efficiency": cache_efficiency,
            "total_cost": total_cost,
            "avg_cost_per_request": self.cost_per_request,
            "potential_savings": self.cache_hits * self.cost_per_request
        }

# Usage example
cost_optimizer = CostOptimizer(api_key, cost_per_request=0.002)

# Get optimization recommendation
optimization = cost_optimizer.optimize_batch_strategy(total_items=500)
print("Cost Optimization Recommendation:")
print(json.dumps(optimization, indent=2))
```

## Summary and Best Practices

### Key Takeaways

1. **Use the Right Endpoint**: Choose between `/simple/price`, `/coins/markets`, and multi-endpoint calls based on your data needs.

2. **Optimize Batch Sizes**: 
   - `/simple/price`: 50-100 coin IDs
   - `/coins/markets`: Up to 250 items with pagination
   - On-chain endpoints: 20-30 items per call

3. **Implement Intelligent Rate Limiting**: Respect the 500 calls/minute limit with proper spacing.

4. **Use Caching Strategically**: Cache static data longer, price data shorter.

5. **Handle Errors Gracefully**: Implement retry logic with exponential backoff.

6. **Monitor Performance**: Track metrics and optimize based on real usage patterns.

7. **Consider Costs**: Balance data freshness requirements with API call costs.

### Recommended Implementation Pattern

```python
# Complete recommended implementation
class OptimalBatchClient:
    """Production-ready batch client combining all best practices."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.rate_limiter = RateLimiter(450, 60)  # Conservative rate limiting
        self.cache = APICache()
        self.performance_monitor = PerformanceMonitor()
        self.error_handler = ErrorHandler()
        self.cost_optimizer = CostOptimizer(api_key)
        
    def batch_get_data(
        self,
        data_type: str,
        items: List[str],
        **kwargs
    ) -> Optional[Dict]:
        """Unified batch data retrieval with all optimizations."""
        
        # Optimize batch strategy
        optimization = self.cost_optimizer.optimize_batch_strategy(len(items))
        strategy = optimization["recommended_strategy"]
        batch_size = optimization["strategies"][strategy]["batch_size"]
        
        # Process in optimized chunks
        chunks = chunk_list(items, batch_size)
        all_results = {}
        
        for chunk in chunks:
            # Rate limiting
            wait_time = self.rate_limiter.acquire()
            if wait_time > 0:
                time.sleep(wait_time)
            
            # Try cache first
            cache_key = f"{data_type}:{','.join(chunk)}"
            cached_result = self.cache.get(cache_key, {})
            
            if cached_result:
                all_results.update(cached_result)
                self.cost_optimizer.cache_hits += 1
                continue
            
            # Make API call with error handling
            if data_type == "prices":
                result, error = self.error_handler.handle_request_with_retry(
                    get_multiple_coin_prices, self.api_key, chunk, **kwargs
                )
            else:
                result, error = self.error_handler.handle_request_with_retry(
                    get_market_data_batch, self.api_key, coin_ids=chunk, **kwargs
                )
            
            if result:
                all_results.update(result)
                self.cache.set(cache_key, {}, result)
                self.cost_optimizer.track_usage()
            elif error:
                print(f"Chunk failed: {error.message}")
        
        return all_results if all_results else None

# Usage
client = OptimalBatchClient(api_key)
prices = client.batch_get_data("prices", ["bitcoin", "ethereum", "solana"])
```

This comprehensive guide provides all the tools and patterns needed to implement efficient, robust, and cost-effective batch operations with the CoinGecko Pro API.