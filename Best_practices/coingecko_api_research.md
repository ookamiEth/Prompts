# CoinGecko API Research for Quantitative Analysis

## API Key Management & Setup

### Best Practices for API Key Security

Protecting your CoinGecko Pro API key is critical for security and preventing unauthorized usage. Follow these best practices:

### 1. Environment Variables with .env File

**Never hardcode API keys directly in your Python scripts.** Instead, use environment variables stored in a `.env` file.

#### Create a .env file:
```bash
# .env
COINGECKO_PRO_API_KEY=CG-your-actual-api-key-here
```

#### Create a .gitignore file:
```bash
# .gitignore
# Environment variables
.env
.env.local
.env.*.local

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Project specific
coingecko_coins.json  # Cache file with coin IDs
*.log
```

### 2. Repository Structure

```
your-project/
├── .env                    # API keys (NEVER commit this)
├── .env.example            # Template for other developers
├── .gitignore              # Excludes .env from version control
├── requirements.txt        # Python dependencies
├── coingecko_client.py     # Your API client code
└── README.md               # Setup instructions
```

#### Create .env.example:
```bash
# .env.example
# Copy this file to .env and add your actual API key
COINGECKO_PRO_API_KEY=your-api-key-here
```

### 3. Loading API Keys in Python

#### Install python-dotenv with conda:
```bash
# Using conda environment
conda activate your_env_name
conda install -c conda-forge python-dotenv requests -y

# Alternative: using pip within conda
conda activate your_env_name
pip install python-dotenv requests
```

#### Loading environment variables:
```python
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key (will be None if not set)
api_key = os.environ.get("COINGECKO_PRO_API_KEY")

if not api_key:
    raise ValueError("COINGECKO_PRO_API_KEY not found in environment variables")
```

#### Alternative loading methods:
```python
# Method 1: Using os.getenv with default
api_key = os.getenv("COINGECKO_PRO_API_KEY", "")

# Method 2: Direct environment access (raises KeyError if missing)
try:
    api_key = os.environ["COINGECKO_PRO_API_KEY"]
except KeyError:
    print("Please set COINGECKO_PRO_API_KEY environment variable")
    sys.exit(1)

# Method 3: Using python-dotenv with specific path
from pathlib import Path
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)
```

### 4. Complete Setup Example

```python
# setup.py - Initial setup script
import os
from pathlib import Path
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

class CoinGeckoConfig:
    """Configuration management for CoinGecko API."""
    
    def __init__(self):
        self.api_key = self._load_api_key()
        self.base_url = "https://pro-api.coingecko.com/api/v3"
        self.timeout = 30
        self.max_retries = 3
        
    def _load_api_key(self) -> str:
        """Load API key from environment with validation."""
        api_key = os.environ.get("COINGECKO_PRO_API_KEY")
        
        if not api_key:
            raise ValueError(
                "COINGECKO_PRO_API_KEY not found. "
                "Please set it in your .env file or environment variables."
            )
        
        # Basic validation
        if not api_key.startswith("CG-"):
            print("Warning: CoinGecko API keys typically start with 'CG-'")
        
        return api_key
    
    def test_connection(self) -> bool:
        """Test the API connection and key validity."""
        url = f"{self.base_url}/ping"
        params = {"x_cg_pro_api_key": self.api_key}
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            print("✓ API connection successful")
            return True
        except requests.exceptions.RequestException as e:
            print(f"✗ API connection failed: {e}")
            return False

# Usage
if __name__ == "__main__":
    config = CoinGeckoConfig()
    config.test_connection()
```

### 5. Security Best Practices

1. **Never commit .env files** - Always add .env to .gitignore
2. **Rotate keys regularly** - Change API keys periodically
3. **Use different keys for different environments** - Separate keys for dev/staging/production
4. **Limit key permissions** - Use read-only keys when possible
5. **Monitor usage** - Check CoinGecko dashboard for unusual activity
6. **Use secure storage in production** - Consider AWS Secrets Manager, Azure Key Vault, or similar

### 6. Production Deployment

For production environments, consider more secure alternatives:

```python
# Example: AWS Secrets Manager
import boto3
import json

def get_secret():
    secret_name = "coingecko/api-key"
    region_name = "us-east-1"
    
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )
    
    get_secret_value_response = client.get_secret_value(
        SecretId=secret_name
    )
    
    secret = get_secret_value_response['SecretString']
    return json.loads(secret)['COINGECKO_PRO_API_KEY']

# Example: Environment variables in Docker
# Dockerfile
# ENV COINGECKO_PRO_API_KEY=${COINGECKO_PRO_API_KEY}

# docker-compose.yml
# environment:
#   - COINGECKO_PRO_API_KEY=${COINGECKO_PRO_API_KEY}
```

### 7. Using the API Key in Requests

All examples in this document will use environment variables:

```python
import os
from dotenv import load_dotenv
import requests

# Load environment variables at the start of your script
load_dotenv()

# Use throughout your application
api_key = os.environ.get("COINGECKO_PRO_API_KEY")

def get_bitcoin_price():
    url = "https://pro-api.coingecko.com/api/v3/simple/price"
    params = {
        "ids": "bitcoin",
        "vs_currencies": "usd",
        "x_cg_pro_api_key": api_key  # Using environment variable
    }
    response = requests.get(url, params=params)
    return response.json()
```

## Overview
This document provides a comprehensive guide for using the CoinGecko Pro API to query USD price data for Bitcoin (BTC), Ethereum (ETH), and Solana (SOL). The research is focused on basic price retrieval tasks suitable for quantitative analysis.

## Key Endpoints for Price Data

### 1. `/simple/price` - Primary Price Endpoint
**Best for:** Current price queries for one or multiple coins
- **URL:** `https://pro-api.coingecko.com/api/v3/simple/price`
- **Authentication:** Requires Pro API Key
- **Parameters:**
  - `ids`: Coin IDs (comma-separated for multiple coins)
  - `vs_currencies`: Target currency (usd)
  - `x_cg_pro_api_key`: Your API key

### 2. `/coins/markets` - Market Data Endpoint
**Best for:** Bulk price data with additional market information
- **URL:** `https://pro-api.coingecko.com/api/v3/coins/markets`
- **Parameters:**
  - `vs_currency`: Target currency (usd)
  - `ids`: Specific coin IDs (optional)
  - `order`: Sorting order (market_cap_desc, etc.)
  - `per_page`: Results per page (max 250)
  - `page`: Page number

### 3. `/coins/{id}` - Detailed Coin Data
**Best for:** Comprehensive coin information including current price
- **URL:** `https://pro-api.coingecko.com/api/v3/coins/{id}`
- **Parameters:**
  - `{id}`: Coin ID
  - `localization`: false (for faster response)
  - `tickers`: false (for faster response)

## Finding Coin IDs

### Why Coin IDs Matter
CoinGecko uses unique coin IDs (not symbols) for API queries. For example:
- Bitcoin's ID is `bitcoin` (not `BTC`)
- Wrapped Bitcoin's ID is `wrapped-bitcoin` (not `WBTC`)
- Multiple coins can share the same symbol (many `USDT` tokens exist)

### Method 1: Query All Coins List (Recommended for Initial Setup)
```python
def get_all_coin_ids(api_key: str) -> List[Dict]:
    """Get complete list of all coins with their IDs, symbols, and names."""
    url = "https://pro-api.coingecko.com/api/v3/coins/list"
    params = {
        "include_platform": False,  # Set True to get contract addresses
        "x_cg_pro_api_key": api_key
    }
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

# Usage - Save this data locally to avoid repeated API calls
api_key = os.environ.get("COINGECKO_PRO_API_KEY")
all_coins = get_all_coin_ids(api_key)

# Save to file for future reference
import json
with open('coingecko_coin_ids.json', 'w') as f:
    json.dump(all_coins, f, indent=2)

# Example response format:
# [
#   {"id": "bitcoin", "symbol": "btc", "name": "Bitcoin"},
#   {"id": "ethereum", "symbol": "eth", "name": "Ethereum"},
#   {"id": "tether", "symbol": "usdt", "name": "Tether"},
#   ...
# ]
```

### Method 2: Create a Symbol-to-ID Mapping
```python
def create_symbol_mapping(api_key: str) -> Dict[str, List[Dict]]:
    """Create a mapping of symbols to coin IDs (handles duplicate symbols)."""
    all_coins = get_all_coin_ids(api_key)
    
    symbol_map = {}
    for coin in all_coins:
        symbol = coin['symbol'].upper()
        if symbol not in symbol_map:
            symbol_map[symbol] = []
        symbol_map[symbol].append({
            'id': coin['id'],
            'name': coin['name']
        })
    
    return symbol_map

# Usage
symbol_map = create_symbol_mapping(api_key)

# Look up Bitcoin
btc_coins = symbol_map.get('BTC', [])
print(f"Coins with symbol BTC: {btc_coins}")
# Output: [{'id': 'bitcoin', 'name': 'Bitcoin'}, {'id': 'bitcoin-bep2', 'name': 'Bitcoin BEP2'}, ...]
```

### Method 3: Search for Specific Coins
```python
def find_coin_by_name_or_symbol(coins_list: List[Dict], query: str) -> List[Dict]:
    """Search for coins by name or symbol."""
    query = query.lower()
    matches = []
    
    for coin in coins_list:
        if (query in coin['id'].lower() or 
            query in coin['symbol'].lower() or 
            query in coin['name'].lower()):
            matches.append(coin)
    
    return matches

# Usage
all_coins = get_all_coin_ids(api_key)
solana_results = find_coin_by_name_or_symbol(all_coins, "solana")
print(f"Found {len(solana_results)} matches for 'solana'")
```

### Using the Official SDK for Coin Discovery
```python
from coingecko_sdk import Coingecko

client = Coingecko(pro_api_key=os.environ.get("COINGECKO_PRO_API_KEY"), environment="pro")

# Get all coins list
coins_list = client.coins.list.get(include_platform=False)

# Convert to more usable format
coin_data = [{
    'id': coin.id,
    'symbol': coin.symbol,
    'name': coin.name
} for coin in coins_list]
```

## Common Coin IDs for Reference

| Cryptocurrency | CoinGecko ID | Symbol | Notes |
|---------------|--------------|--------|-------|
| Bitcoin | `bitcoin` | BTC | |
| Ethereum | `ethereum` | ETH | |
| Solana | `solana` | SOL | |
| Tether (Ethereum) | `tether` | USDT | Main USDT |
| USD Coin | `usd-coin` | USDC | |
| Binance Coin | `binancecoin` | BNB | |
| Cardano | `cardano` | ADA | |
| Polygon | `matic-network` | MATIC | Note: ID is not "polygon" |
| Wrapped Bitcoin | `wrapped-bitcoin` | WBTC | |
| Chainlink | `chainlink` | LINK | |

## Python Implementation Examples

### Required Dependencies
```python
import os
from dotenv import load_dotenv
import requests
import json
from typing import Dict, List, Optional

# Load environment variables at the start of your script
load_dotenv()
```

### Single Coin Price Query (Bitcoin)
```python
import requests

def get_bitcoin_price(api_key: str) -> Dict:
    """Get current Bitcoin price in USD."""
    url = "https://pro-api.coingecko.com/api/v3/simple/price"
    params = {
        "ids": "bitcoin",
        "vs_currencies": "usd",
        "x_cg_pro_api_key": api_key
    }
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

# Usage
api_key = os.environ.get("COINGECKO_PRO_API_KEY")
btc_data = get_bitcoin_price(api_key)
print(f"Bitcoin price: ${btc_data['bitcoin']['usd']}")
```

**Response:**
```json
{
  "bitcoin": {
    "usd": 42000.50
  }
}
```

### Multiple Coins Price Query (BTC, ETH, SOL)
```python
def get_multiple_coin_prices(api_key: str, coin_ids: List[str]) -> Dict:
    """Get current prices for multiple coins in USD."""
    url = "https://pro-api.coingecko.com/api/v3/simple/price"
    params = {
        "ids": ",".join(coin_ids),
        "vs_currencies": "usd",
        "x_cg_pro_api_key": api_key
    }
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

# Usage
api_key = os.environ.get("COINGECKO_PRO_API_KEY")
coins = ["bitcoin", "ethereum", "solana"]
prices = get_multiple_coin_prices(api_key, coins)

for coin, data in prices.items():
    print(f"{coin.capitalize()}: ${data['usd']}")
```

**Response:**
```json
{
  "bitcoin": {
    "usd": 42000.50
  },
  "ethereum": {
    "usd": 2500.75
  },
  "solana": {
    "usd": 95.25
  }
}
```

### Enhanced Price Data with Market Information
```python
def get_enhanced_price_data(api_key: str, coin_ids: List[str]) -> Dict:
    """Get enhanced price data including market cap, volume, and 24h change."""
    url = "https://pro-api.coingecko.com/api/v3/simple/price"
    params = {
        "ids": ",".join(coin_ids),
        "vs_currencies": "usd",
        "include_market_cap": "true",
        "include_24hr_vol": "true",
        "include_24hr_change": "true",
        "x_cg_pro_api_key": api_key
    }
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

# Usage
api_key = os.environ.get("COINGECKO_PRO_API_KEY")
coins = ["bitcoin", "ethereum", "solana"]
enhanced_data = get_enhanced_price_data(api_key, coins)

for coin, data in enhanced_data.items():
    print(f"{coin.capitalize()}:")
    print(f"  Price: ${data['usd']}")
    print(f"  Market Cap: ${data['usd_market_cap']:,}")
    print(f"  24h Volume: ${data['usd_24h_vol']:,}")
    print(f"  24h Change: {data['usd_24h_change']:.2f}%")
    print()
```

**Response:**
```json
{
  "bitcoin": {
    "usd": 42000.50,
    "usd_market_cap": 825000000000,
    "usd_24h_vol": 25000000000,
    "usd_24h_change": 2.5
  },
  "ethereum": {
    "usd": 2500.75,
    "usd_market_cap": 300000000000,
    "usd_24h_vol": 15000000000,
    "usd_24h_change": 1.8
  },
  "solana": {
    "usd": 95.25,
    "usd_market_cap": 42000000000,
    "usd_24h_vol": 2000000000,
    "usd_24h_change": -0.5
  }
}
```

### Bulk Market Data Query
```python
def get_market_data(api_key: str, coin_ids: List[str]) -> List[Dict]:
    """Get comprehensive market data for multiple coins."""
    url = "https://pro-api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "ids": ",".join(coin_ids),
        "order": "market_cap_desc",
        "per_page": len(coin_ids),
        "page": 1,
        "sparkline": "false",
        "x_cg_pro_api_key": api_key
    }
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

# Usage
api_key = os.environ.get("COINGECKO_PRO_API_KEY")
coins = ["bitcoin", "ethereum", "solana"]
market_data = get_market_data(api_key, coins)

for coin in market_data:
    print(f"{coin['name']} ({coin['symbol'].upper()}):")
    print(f"  Price: ${coin['current_price']}")
    print(f"  Market Cap Rank: #{coin['market_cap_rank']}")
    print(f"  Market Cap: ${coin['market_cap']:,}")
    print(f"  24h Volume: ${coin['total_volume']:,}")
    print(f"  24h Change: {coin['price_change_percentage_24h']:.2f}%")
    print()
```

## Rate Limits
- **Pro API:** 500 calls/minute
- **Enterprise:** Higher limits available
- **Free API:** 10-50 calls/minute (not recommended for production)

## Authentication
All Pro API requests require authentication via:
- **Header:** `x-cg-pro-api-key: YOUR_API_KEY`
- **Query Parameter:** `x_cg_pro_api_key=YOUR_API_KEY`

## Python Error Handling and Best Practices

### Error Handling Example
```python
import requests
import time
from typing import Dict, List, Optional

def safe_api_call(url: str, params: Dict, max_retries: int = 3) -> Optional[Dict]:
    """Make a safe API call with error handling and retries."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=10)
            
            # Handle rate limiting
            if response.status_code == 429:
                print(f"Rate limit exceeded. Waiting 60 seconds... (Attempt {attempt + 1})")
                time.sleep(60)
                continue
            
            # Handle other HTTP errors
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.Timeout:
            print(f"Request timeout (Attempt {attempt + 1})")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e} (Attempt {attempt + 1})")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                
        except Exception as e:
            print(f"Unexpected error: {e}")
            break
    
    return None

# Usage with error handling
def get_prices_safely(api_key: str, coin_ids: List[str]) -> Optional[Dict]:
    """Get coin prices with proper error handling."""
    url = "https://pro-api.coingecko.com/api/v3/simple/price"
    params = {
        "ids": ",".join(coin_ids),
        "vs_currencies": "usd",
        "x_cg_pro_api_key": api_key
    }
    
    return safe_api_call(url, params)
```

### Best Practices for Python Implementation

1. **Use `/simple/price` for basic price queries** - Most efficient for simple price data
2. **Batch requests** - Query multiple coins in a single request when possible
3. **Implement proper error handling** - Handle rate limits, timeouts, and API errors
4. **Add retry logic** - Implement exponential backoff for failed requests
5. **Cache responses** - Use caching to avoid unnecessary API calls
6. **Use type hints** - Make code more maintainable and readable
7. **Set request timeouts** - Prevent hanging requests
8. **Include additional parameters** when needed:
   - `include_market_cap=true` for market cap data
   - `include_24hr_vol=true` for volume data
   - `include_24hr_change=true` for price change data
   - `include_last_updated_at=true` for timestamp data

### Common HTTP Status Codes
- **200 - OK:** Request successful
- **401 - Unauthorized:** Invalid API key
- **404 - Not Found:** Invalid coin ID or endpoint
- **429 - Too Many Requests:** Rate limit exceeded (wait 60 seconds)
- **500 - Internal Server Error:** API server issues
- **503 - Service Unavailable:** API temporarily unavailable

## Complete Python Class Implementation

### CoinGecko API Client Class
```python
import os
from dotenv import load_dotenv
import requests
import time
import json
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

# Load environment variables
load_dotenv()

@dataclass
class CoinPrice:
    """Data class for coin price information."""
    coin_id: str
    price: float
    market_cap: Optional[float] = None
    volume_24h: Optional[float] = None
    change_24h: Optional[float] = None
    last_updated: Optional[int] = None

class CoinGeckoClient:
    """CoinGecko Pro API client for cryptocurrency data."""
    
    def __init__(self, api_key: str, cache_coin_list: bool = True):
        """Initialize the CoinGecko client with Pro API key."""
        self.api_key = api_key
        self.base_url = "https://pro-api.coingecko.com/api/v3"
        self.session = requests.Session()
        self._coin_list_cache = None
        self._symbol_map_cache = None
        
        if cache_coin_list:
            self._load_coin_list()
        
    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make a request to the CoinGecko API with error handling."""
        if params is None:
            params = {}
        
        params["x_cg_pro_api_key"] = self.api_key
        url = f"{self.base_url}/{endpoint}"
        
        for attempt in range(3):
            try:
                response = self.session.get(url, params=params, timeout=10)
                
                if response.status_code == 429:
                    print(f"Rate limit exceeded. Waiting 60 seconds...")
                    time.sleep(60)
                    continue
                    
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                print(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    
        return None
    
    def _load_coin_list(self):
        """Load and cache the coin list on initialization."""
        print("Loading CoinGecko coin list...")
        self._coin_list_cache = self.get_coins_list()
        if self._coin_list_cache:
            self._build_symbol_map()
            print(f"Loaded {len(self._coin_list_cache)} coins")
    
    def _build_symbol_map(self):
        """Build a symbol to coin ID mapping."""
        self._symbol_map_cache = {}
        for coin in self._coin_list_cache:
            symbol = coin['symbol'].upper()
            if symbol not in self._symbol_map_cache:
                self._symbol_map_cache[symbol] = []
            self._symbol_map_cache[symbol].append({
                'id': coin['id'],
                'name': coin['name']
            })
    
    def get_coins_list(self, include_platform: bool = False) -> Optional[List[Dict]]:
        """Get list of all supported coins."""
        params = {"include_platform": str(include_platform).lower()}
        return self._make_request("coins/list", params)
    
    def find_coin_id(self, symbol: str) -> Optional[str]:
        """Find coin ID by symbol (returns most popular if multiple matches)."""
        if not self._symbol_map_cache:
            self._load_coin_list()
        
        matches = self._symbol_map_cache.get(symbol.upper(), [])
        if matches:
            # Return first match (usually the most popular)
            return matches[0]['id']
        return None
    
    def search_coins(self, query: str) -> List[Dict]:
        """Search for coins by name, symbol, or ID."""
        if not self._coin_list_cache:
            self._load_coin_list()
        
        query = query.lower()
        matches = []
        
        for coin in self._coin_list_cache:
            if (query in coin['id'].lower() or 
                query in coin['symbol'].lower() or 
                query in coin['name'].lower()):
                matches.append(coin)
        
        return matches
    
    def get_price(self, coin_id: str) -> Optional[CoinPrice]:
        """Get current price for a single coin."""
        params = {
            "ids": coin_id,
            "vs_currencies": "usd",
            "include_market_cap": "true",
            "include_24hr_vol": "true",
            "include_24hr_change": "true",
            "include_last_updated_at": "true"
        }
        
        data = self._make_request("simple/price", params)
        if data and coin_id in data:
            coin_data = data[coin_id]
            return CoinPrice(
                coin_id=coin_id,
                price=coin_data["usd"],
                market_cap=coin_data.get("usd_market_cap"),
                volume_24h=coin_data.get("usd_24h_vol"),
                change_24h=coin_data.get("usd_24h_change"),
                last_updated=coin_data.get("last_updated_at")
            )
        return None
    
    def get_multiple_prices(self, coin_ids: List[str]) -> List[CoinPrice]:
        """Get current prices for multiple coins."""
        params = {
            "ids": ",".join(coin_ids),
            "vs_currencies": "usd",
            "include_market_cap": "true",
            "include_24hr_vol": "true",
            "include_24hr_change": "true",
            "include_last_updated_at": "true"
        }
        
        data = self._make_request("simple/price", params)
        prices = []
        
        if data:
            for coin_id in coin_ids:
                if coin_id in data:
                    coin_data = data[coin_id]
                    prices.append(CoinPrice(
                        coin_id=coin_id,
                        price=coin_data["usd"],
                        market_cap=coin_data.get("usd_market_cap"),
                        volume_24h=coin_data.get("usd_24h_vol"),
                        change_24h=coin_data.get("usd_24h_change"),
                        last_updated=coin_data.get("last_updated_at")
                    ))
        
        return prices
    
    def get_btc_eth_sol_prices(self) -> List[CoinPrice]:
        """Convenience method to get BTC, ETH, and SOL prices."""
        return self.get_multiple_prices(["bitcoin", "ethereum", "solana"])
    
    def get_market_data(self, coin_ids: List[str]) -> List[Dict]:
        """Get comprehensive market data for multiple coins."""
        params = {
            "vs_currency": "usd",
            "ids": ",".join(coin_ids),
            "order": "market_cap_desc",
            "per_page": len(coin_ids),
            "page": 1,
            "sparkline": "false"
        }
        
        return self._make_request("coins/markets", params) or []

# Usage Examples
def main():
    """Example usage of the CoinGecko client."""
    # Initialize client (will auto-load coin list)
    api_key = os.environ.get("COINGECKO_PRO_API_KEY")
    client = CoinGeckoClient(api_key)
    
    # Find coin IDs by symbol
    btc_id = client.find_coin_id("BTC")
    print(f"Bitcoin ID: {btc_id}")  # Output: bitcoin
    
    # Search for coins
    sol_coins = client.search_coins("solana")
    print(f"Found {len(sol_coins)} coins matching 'solana'")
    
    # Get price using symbol lookup
    btc_id = client.find_coin_id("BTC")
    if btc_id:
        btc_price = client.get_price(btc_id)
        if btc_price:
            print(f"Bitcoin: ${btc_price.price:,.2f}")
    
    # Get multiple coin prices
    prices = client.get_btc_eth_sol_prices()
    for price in prices:
        print(f"{price.coin_id.capitalize()}: ${price.price:,.2f}")
    
    # Get detailed market data
    market_data = client.get_market_data(["bitcoin", "ethereum", "solana"])
    for coin in market_data:
        print(f"{coin['name']}: ${coin['current_price']:,.2f} (Rank #{coin['market_cap_rank']})")

if __name__ == "__main__":
    main()
```

### Simple Function-Based Approach
```python
# For simpler use cases, here are standalone functions:

def get_crypto_prices(api_key: str) -> Dict[str, float]:
    """Get current prices for BTC, ETH, and SOL."""
    url = "https://pro-api.coingecko.com/api/v3/simple/price"
    params = {
        "ids": "bitcoin,ethereum,solana",
        "vs_currencies": "usd",
        "x_cg_pro_api_key": api_key
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        return {
            "btc": data["bitcoin"]["usd"],
            "eth": data["ethereum"]["usd"],
            "sol": data["solana"]["usd"]
        }
    except Exception as e:
        print(f"Error fetching prices: {e}")
        return {}

# Usage
api_key = os.environ.get("COINGECKO_PRO_API_KEY")
prices = get_crypto_prices(api_key)
print(f"BTC: ${prices.get('btc', 0):,.2f}")
print(f"ETH: ${prices.get('eth', 0):,.2f}")
print(f"SOL: ${prices.get('sol', 0):,.2f}")
```

## Using the Official CoinGecko Python SDK

### Installation
```bash
# Using conda environment
conda activate your_env_name
pip install coingecko_sdk  # Not available via conda, use pip

# Or create dedicated environment
conda create -n coingecko_env python=3.10 -y
conda activate coingecko_env
pip install coingecko_sdk
```

### Basic SDK Usage
```python
import os
from coingecko_sdk import Coingecko

# Initialize client
client = Coingecko(
    pro_api_key=os.environ.get("COINGECKO_PRO_API_KEY"),
    environment="pro"
)

# Get single coin price
price = client.simple.price.get(
    vs_currencies="usd",
    ids="bitcoin"
)
print(f"Bitcoin: ${price.bitcoin.usd}")

# Get multiple coins with market data
prices = client.simple.price.get(
    vs_currencies="usd",
    ids="bitcoin,ethereum,solana",
    include_market_cap=True,
    include_24hr_vol=True,
    include_24hr_change=True
)

# Access data with type safety
print(f"BTC: ${prices.bitcoin.usd}")
print(f"ETH: ${prices.ethereum.usd}")
print(f"SOL: ${prices.solana.usd}")
```

### Async SDK Usage
```python
import asyncio
from coingecko_sdk import AsyncCoingecko

async def get_prices():
    client = AsyncCoingecko(
        pro_api_key=os.environ.get("COINGECKO_PRO_API_KEY"),
        environment="pro"
    )
    
    prices = await client.simple.price.get(
        vs_currencies="usd",
        ids="bitcoin,ethereum,solana"
    )
    return prices

# Run async
prices = asyncio.run(get_prices())
```

### SDK Features
- **Type Safety:** Full autocomplete and type checking with Pydantic models
- **Error Handling:** Built-in retry logic and comprehensive error types
- **Professional Features:** Connection pooling, timeouts, rate limit handling
- **Both Sync & Async:** Choose based on your application needs
- **Automatic Updates:** SDK is auto-generated from API spec

### SDK Error Handling
```python
import coingecko_sdk
from coingecko_sdk import Coingecko

client = Coingecko(pro_api_key=os.environ.get("COINGECKO_PRO_API_KEY"))

try:
    price = client.simple.price.get(
        vs_currencies="usd",
        ids="bitcoin"
    )
except coingecko_sdk.RateLimitError:
    print("Rate limit hit - automatic retry will handle this")
except coingecko_sdk.AuthenticationError:
    print("Invalid API key")
except coingecko_sdk.APIStatusError as e:
    print(f"API error: {e.status_code}")
```

## Comparison: Custom Implementation vs Official SDK

### When to Use Custom Implementation
- **Learning purposes** - Understanding API mechanics
- **Minimal dependencies** - Only requires `requests`
- **Custom requirements** - Specific modifications needed
- **Simple scripts** - One-off data fetches

### When to Use Official SDK (Recommended)
- **Production applications** - Professional error handling
- **High-frequency queries** - Optimized performance
- **Type safety required** - IDE autocomplete and validation
- **Long-term maintenance** - Automatic API updates
- **Async operations** - Built-in async support

### Feature Comparison

| Feature | Custom Implementation | Official SDK |
|---------|---------------------|--------------|
| Dependencies | `requests` only | `coingecko_sdk` + deps |
| Type Safety | Manual type hints | Full Pydantic models |
| Error Handling | Basic | Comprehensive |
| Retry Logic | Manual | Automatic |
| Rate Limiting | Manual | Automatic |
| Async Support | Requires `aiohttp` | Built-in |
| Maintenance | Manual updates | Auto-generated |
| Performance | Good | Optimized |

## OpenAPI Specification Use Cases

The `coingecko-pro-api-v3.json` OpenAPI spec is useful for:

### 1. **API Documentation Reference**
- Complete endpoint documentation
- All parameter definitions and types
- Response schema definitions
- Error response formats

### 2. **Code Generation**
- Generate clients in any language (Java, C#, Go, etc.)
- Create TypeScript interfaces
- Generate API documentation
- Create mock servers for testing

### 3. **API Validation & Testing**
- Validate requests/responses against schema
- Generate test cases automatically
- API contract testing
- Integration with tools like Postman/Insomnia

### 4. **Development Tools Integration**
```bash
# Example: Generate a client in another language
openapi-generator generate -i coingecko-pro-api-v3.json -g python -o ./generated-client

# Import into Postman
# Import the JSON file directly into Postman for testing

# Use with API testing tools
# Many tools can import OpenAPI specs for automated testing
```

### 5. **Custom Client Development**
If you need to build a client in an unsupported language:
```python
# Parse the OpenAPI spec to understand endpoints
import json

with open('coingecko-pro-api-v3.json', 'r') as f:
    spec = json.load(f)
    
# Extract endpoint information
endpoints = spec['paths']
simple_price = endpoints['/simple/price']
print(simple_price['get']['parameters'])
```

## Additional Resources
- **Coin IDs List:** Use `/coins/list` endpoint to get all available coin IDs
- **Supported Currencies:** Use `/simple/supported_vs_currencies` endpoint
- **API Status:** Check `https://status.coingecko.com/` for service status
- **Documentation:** `https://docs.coingecko.com/`
- **Python requests library:** `conda install -c conda-forge requests` or `pip install requests`
- **Official SDK:** `pip install coingecko_sdk` (not available via conda)
- **SDK GitHub:** `https://github.com/coingecko/coingecko-python`

## Best Practices for Coin ID Management

1. **Cache the coin list locally** - The `/coins/list` endpoint returns ~13,000+ coins. Query it once and save:
   ```python
   # Run once and save
   all_coins = client.get_coins_list()
   with open('coingecko_coins.json', 'w') as f:
       json.dump(all_coins, f)
   ```

2. **Handle duplicate symbols** - Many coins share the same symbol:
   ```python
   # Example: Multiple USDT tokens
   usdt_coins = client.search_coins("USDT")
   # Returns: Tether, Tether (Wormhole), TetherUS, etc.
   ```

3. **Use exact IDs when possible** - If you know the exact coin, use its ID directly:
   ```python
   # Preferred: Use exact IDs
   prices = client.get_multiple_prices(["bitcoin", "ethereum", "solana"])
   
   # Less reliable: Symbol lookup (might get wrong coin)
   btc_id = client.find_coin_id("BTC")  # Could return bitcoin-bep2 instead
   ```

## Notes
- **CoinGecko has 13,000+ coins** - Not all are actively traded or have good data
- **IDs are stable** - Unlike symbols, coin IDs don't change
- **Symbols are NOT unique** - Always verify you have the correct coin
- SOL currency is now supported for CoinGecko endpoints as of recent updates
- Price data is updated frequently (typically every few minutes)
- For historical data, use `/coins/{id}/market_chart` or `/coins/{id}/history` endpoints
- All prices are in USD unless otherwise specified
- Always use proper error handling and respect rate limits in production code
- The official SDK handles rate limiting automatically with built-in retry logic