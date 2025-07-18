# yfinance Best Practices Guide for International Equity Price Queries

## Table of Contents
1. [Introduction](#introduction)
2. [Installation and Setup](#installation-and-setup)
3. [Understanding International Ticker Symbols](#understanding-international-ticker-symbols)
4. [Basic Price Queries](#basic-price-queries)
5. [Bulk Data Downloads](#bulk-data-downloads)
6. [Error Handling and Rate Limits](#error-handling-and-rate-limits)
7. [Timezone and Currency Considerations](#timezone-and-currency-considerations)
8. [Data Quality and Repair](#data-quality-and-repair)
9. [Production-Ready Code Patterns](#production-ready-code-patterns)
10. [Common Pitfalls and Solutions](#common-pitfalls-and-solutions)

## Introduction

This guide provides step-by-step instructions for using yfinance to query equity prices across different countries. Follow these instructions exactly as written to avoid common mistakes.

**IMPORTANT**: yfinance is an unofficial library that scrapes Yahoo Finance. It can break at any time. NEVER use it for real trading without backup data sources.

## Installation and Setup

### Step 1: Create Conda Environment and Install yfinance
```bash
# Create a new conda environment
conda create -n yfinance_env python=3.10 -y

# Activate the environment
conda activate yfinance_env

# Install yfinance and dependencies
conda install -c conda-forge pandas pytz -y
pip install yfinance --upgrade
```

### Alternative: Install in existing conda environment
```bash
# Activate your existing environment
conda activate your_env_name

# Install packages
pip install yfinance --upgrade
conda install -c conda-forge pandas pytz -y
```

### Step 2: Import Required Libraries
```python
import yfinance as yf
import pandas as pd
import time
from datetime import datetime, timedelta
import pytz
```

### Step 3: Verify Installation
```python
# Test with a simple US stock
apple = yf.Ticker("AAPL")
print(apple.info['longName'])  # Should print "Apple Inc."
```

## Understanding International Ticker Symbols

### The Golden Rule: Ticker + Exchange Suffix

For non-US stocks, you MUST add the exchange suffix to the ticker symbol. Here's the complete reference:

| Country/Region | Exchange | Suffix | Example | Company |
|----------------|----------|---------|---------|---------|
| United States | NYSE/NASDAQ | None | `AAPL` | Apple |
| United Kingdom | London Stock Exchange | `.L` | `BP.L` | BP |
| Canada | Toronto Stock Exchange | `.TO` | `RY.TO` | Royal Bank |
| Canada | TSX Venture | `.V` | `LA.V` | Lithium Americas |
| Germany | Frankfurt | `.DE` | `BMW.DE` | BMW |
| France | Euronext Paris | `.PA` | `MC.PA` | LVMH |
| Spain | Madrid | `.MC` | `IBE.MC` | Iberdrola |
| Italy | Milan | `.MI` | `ISP.MI` | Intesa Sanpaolo |
| Switzerland | SIX Swiss | `.SW` | `NESN.SW` | Nestlé |
| Netherlands | Euronext Amsterdam | `.AS` | `ASML.AS` | ASML |
| Sweden | Stockholm | `.ST` | `VOLV-B.ST` | Volvo |
| Japan | Tokyo | `.T` | `7203.T` | Toyota |
| Hong Kong | HKEX | `.HK` | `0005.HK` | HSBC |
| China | Shanghai | `.SS` | `600519.SS` | Kweichow Moutai |
| China | Shenzhen | `.SZ` | `000002.SZ` | China Vanke |
| Australia | ASX | `.AX` | `BHP.AX` | BHP Group |
| India | NSE | `.NS` | `RELIANCE.NS` | Reliance |
| India | BSE | `.BO` | `RELIANCE.BO` | Reliance |
| South Korea | KOSPI | `.KS` | `005930.KS` | Samsung |
| Taiwan | TWSE | `.TW` | `2330.TW` | TSMC |
| Israel | Tel Aviv | `.TA` | `ESLT.TA` | Elbit Systems |
| South Africa | Johannesburg | `.JO` | `AGL.JO` | Anglo American |
| Brazil | Bovespa | `.SA` | `PETR4.SA` | Petrobras |

### Finding the Correct Ticker Symbol

```python
# Method 1: Use the Search function
from yfinance import Search

# Search for a company
results = Search("Toyota", max_results=10)
for result in results.quotes:
    print(f"{result['symbol']} - {result['longname']} - {result['exchange']}")

# Method 2: Try common suffixes
company_base = "TOYOTA"
suffixes = ['.T', '.F', '.DE', '.L']
for suffix in suffixes:
    try:
        ticker = yf.Ticker(company_base + suffix)
        info = ticker.info
        if info.get('longName'):
            print(f"Found: {company_base + suffix} = {info['longName']}")
    except:
        pass
```

## Basic Price Queries

### Single Stock Query

```python
# ALWAYS follow this pattern for international stocks
def get_stock_data(ticker_symbol, period="1mo"):
    """
    Get stock data for a single ticker
    
    Args:
        ticker_symbol: Full ticker including exchange suffix (e.g., 'BP.L')
        period: Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
    
    Returns:
        pandas DataFrame with OHLCV data
    """
    try:
        stock = yf.Ticker(ticker_symbol)
        data = stock.history(period=period)
        
        # ALWAYS check if data is empty
        if data.empty:
            print(f"No data found for {ticker_symbol}")
            return None
            
        return data
    except Exception as e:
        print(f"Error fetching {ticker_symbol}: {e}")
        return None

# Examples for different countries
uk_stock = get_stock_data("BP.L")
japan_stock = get_stock_data("7203.T") 
german_stock = get_stock_data("BMW.DE")
```

### Getting Company Information

```python
def get_company_info(ticker_symbol):
    """Get detailed company information"""
    try:
        stock = yf.Ticker(ticker_symbol)
        info = stock.info
        
        # Key fields to extract
        important_info = {
            'Name': info.get('longName', 'N/A'),
            'Symbol': info.get('symbol', ticker_symbol),
            'Exchange': info.get('exchange', 'N/A'),
            'Currency': info.get('currency', 'N/A'),
            'Country': info.get('country', 'N/A'),
            'Sector': info.get('sector', 'N/A'),
            'MarketCap': info.get('marketCap', 0),
            'Price': info.get('currentPrice', 0)
        }
        return important_info
    except Exception as e:
        print(f"Error getting info for {ticker_symbol}: {e}")
        return None

# Example
bp_info = get_company_info("BP.L")
print(bp_info)
```

## Bulk Data Downloads

### Downloading Multiple International Stocks

```python
def download_multiple_stocks(ticker_list, start_date=None, end_date=None, period="1mo"):
    """
    Download data for multiple stocks efficiently
    
    IMPORTANT: Always use this for multiple stocks, NOT a loop!
    """
    # Add delay to avoid rate limiting
    time.sleep(0.5)
    
    try:
        # Method 1: Using period (recommended for beginners)
        if not start_date and not end_date:
            data = yf.download(
                tickers=ticker_list,
                period=period,
                group_by='ticker',
                threads=True,  # Enable parallel downloads
                progress=True  # Show progress bar
            )
        else:
            # Method 2: Using specific dates
            data = yf.download(
                tickers=ticker_list,
                start=start_date,
                end=end_date,
                group_by='ticker',
                threads=True,
                progress=True
            )
        
        return data
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

# Example: Global portfolio
global_stocks = [
    "AAPL",      # Apple (US)
    "BP.L",      # BP (UK)
    "NESN.SW",   # Nestlé (Switzerland)
    "7203.T",    # Toyota (Japan)
    "BHP.AX",    # BHP (Australia)
    "BABA.HK",   # Alibaba (Hong Kong)
    "SAP.DE"     # SAP (Germany)
]

# Download all at once
portfolio_data = download_multiple_stocks(global_stocks, period="3mo")

# Access individual stock data
if portfolio_data is not None and not portfolio_data.empty:
    # For single stock in result
    bp_close_prices = portfolio_data['Close']['BP.L']
    
    # For all close prices
    all_close_prices = portfolio_data['Close']
```

## Error Handling and Rate Limits

### Essential Rate Limit Rules

```python
import time
from functools import wraps

def rate_limit(calls_per_minute=30):
    """Decorator to enforce rate limits"""
    min_interval = 60.0 / calls_per_minute
    last_called = [0.0]
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret
        return wrapper
    return decorator

@rate_limit(calls_per_minute=30)  # Yahoo allows ~60/min, but be conservative
def safe_get_ticker_data(ticker):
    """Rate-limited ticker data fetch"""
    return yf.Ticker(ticker).history(period="1mo")
```

### Robust Error Handling Pattern

```python
def fetch_with_retry(ticker_symbol, max_retries=3, initial_delay=1):
    """
    Fetch data with exponential backoff retry
    
    ALWAYS use this pattern for production code!
    """
    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(ticker_symbol)
            data = stock.history(period="1mo")
            
            # Check for empty data
            if data.empty:
                print(f"Warning: No data for {ticker_symbol}")
                return None
                
            # Check for suspicious data
            if (data['Close'] == 0).any():
                print(f"Warning: Zero prices found for {ticker_symbol}")
                
            return data
            
        except Exception as e:
            wait_time = initial_delay * (2 ** attempt)  # Exponential backoff
            print(f"Attempt {attempt + 1} failed for {ticker_symbol}: {e}")
            
            if attempt < max_retries - 1:
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to fetch {ticker_symbol} after {max_retries} attempts")
                return None
```

## Timezone and Currency Considerations

### Understanding Market Hours

```python
def get_market_info(ticker_symbol):
    """Get timezone and trading hours for a stock"""
    stock = yf.Ticker(ticker_symbol)
    info = stock.info
    
    market_info = {
        'Exchange': info.get('exchange', 'Unknown'),
        'TimeZone': info.get('exchangeTimezoneName', 'Unknown'),
        'Currency': info.get('currency', 'Unknown'),
        'Market': info.get('market', 'Unknown')
    }
    
    # Get trading hours if available
    try:
        trading_period = stock.trading_period_info
        market_info['TradingHours'] = trading_period
    except:
        market_info['TradingHours'] = 'Not available'
    
    return market_info

# Example timezone mapping
EXCHANGE_TIMEZONES = {
    'LSE': 'Europe/London',
    'TSE': 'Asia/Tokyo', 
    'HKEX': 'Asia/Hong_Kong',
    'ASX': 'Australia/Sydney',
    'NYSE': 'America/New_York',
    'NASDAQ': 'America/New_York',
    'FRA': 'Europe/Berlin',
    'EPA': 'Europe/Paris'
}
```

### Converting Timestamps

```python
def normalize_to_utc(data, source_timezone):
    """Convert market data to UTC timezone"""
    if isinstance(data.index, pd.DatetimeIndex):
        if data.index.tz is None:
            # Localize to source timezone first
            data.index = data.index.tz_localize(source_timezone)
        # Then convert to UTC
        data.index = data.index.tz_convert('UTC')
    return data

# Example: Compare prices at same UTC time
uk_data = get_stock_data("BP.L")
us_data = get_stock_data("XOM")  # ExxonMobil

# Normalize both to UTC
uk_data = normalize_to_utc(uk_data, 'Europe/London')
us_data = normalize_to_utc(us_data, 'America/New_York')
```

### Currency Conversion

```python
def get_fx_rate(from_currency, to_currency="USD"):
    """Get exchange rate using Yahoo Finance"""
    fx_ticker = f"{from_currency}{to_currency}=X"
    try:
        fx = yf.Ticker(fx_ticker)
        rate = fx.info.get('regularMarketPrice', None)
        if rate:
            return rate
        else:
            # Try historical data
            hist = fx.history(period="1d")
            if not hist.empty:
                return hist['Close'].iloc[-1]
    except:
        pass
    return None

# Convert GBP stock price to USD
bp = yf.Ticker("BP.L")
bp_price_gbp = bp.info['currentPrice']
gbp_to_usd = get_fx_rate("GBP", "USD")
bp_price_usd = bp_price_gbp * gbp_to_usd if gbp_to_usd else None
```

## Data Quality and Repair

### Common Data Issues and Fixes

```python
def clean_price_data(data, ticker_symbol):
    """
    Clean common data quality issues
    
    ALWAYS run this on downloaded data!
    """
    if data is None or data.empty:
        return None
    
    # 1. Remove rows with all NaN values
    data = data.dropna(how='all')
    
    # 2. Forward fill missing values (max 5 days)
    data = data.fillna(method='ffill', limit=5)
    
    # 3. Check for price spikes (>50% daily move)
    if 'Close' in data.columns:
        daily_returns = data['Close'].pct_change()
        suspicious_days = daily_returns[daily_returns.abs() > 0.5]
        
        if len(suspicious_days) > 0:
            print(f"Warning: {ticker_symbol} has {len(suspicious_days)} suspicious price movements")
            
    # 4. Check for zero volumes
    if 'Volume' in data.columns:
        zero_volume_days = data[data['Volume'] == 0]
        if len(zero_volume_days) > 0:
            print(f"Warning: {ticker_symbol} has {len(zero_volume_days)} days with zero volume")
    
    return data

# Use the repair parameter
def download_with_repair(ticker_list, period="1mo"):
    """Always use repair=True for international stocks"""
    data = yf.download(
        ticker_list,
        period=period,
        repair=True,  # IMPORTANT: Fixes splits/dividends issues
        actions=True,  # Include dividends and splits
        prepost=False, # Exclude pre/post market data
        group_by='ticker'
    )
    return data
```

### Detecting Bad Data

```python
def validate_price_data(data, ticker_symbol):
    """
    Validate downloaded data quality
    
    Returns: (is_valid, issues_list)
    """
    issues = []
    
    if data is None or data.empty:
        return False, ["No data available"]
    
    # Check 1: Sufficient data points
    if len(data) < 10:
        issues.append(f"Only {len(data)} data points available")
    
    # Check 2: Recent data
    last_date = data.index[-1]
    days_old = (pd.Timestamp.now() - last_date).days
    if days_old > 5:
        issues.append(f"Latest data is {days_old} days old")
    
    # Check 3: Price reasonableness
    if 'Close' in data.columns:
        price_stats = data['Close'].describe()
        cv = price_stats['std'] / price_stats['mean']  # Coefficient of variation
        
        if cv > 2:
            issues.append(f"Extremely volatile (CV={cv:.2f})")
        
        if price_stats['min'] <= 0:
            issues.append("Contains zero or negative prices")
    
    # Check 4: Volume presence
    if 'Volume' in data.columns:
        zero_volume_pct = (data['Volume'] == 0).sum() / len(data)
        if zero_volume_pct > 0.1:
            issues.append(f"{zero_volume_pct:.1%} of days have zero volume")
    
    is_valid = len(issues) == 0
    return is_valid, issues

# Example usage
data = get_stock_data("BP.L")
is_valid, issues = validate_price_data(data, "BP.L")
if not is_valid:
    print(f"Data quality issues for BP.L: {issues}")
```

## Production-Ready Code Patterns

### Complete Working Example

```python
import yfinance as yf
import pandas as pd
import time
from datetime import datetime, timedelta
import json

class InternationalStockFetcher:
    """
    Production-ready class for fetching international stock data
    
    Copy this entire class for your projects!
    """
    
    def __init__(self, cache_dir="./cache"):
        self.cache_dir = cache_dir
        self.request_count = 0
        self.last_request_time = 0
        
    def _rate_limit(self):
        """Enforce rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        # Ensure at least 1 second between requests
        if time_since_last < 1.0:
            time.sleep(1.0 - time_since_last)
            
        self.last_request_time = time.time()
        self.request_count += 1
        
        # Every 50 requests, pause for 10 seconds
        if self.request_count % 50 == 0:
            print(f"Rate limit pause after {self.request_count} requests...")
            time.sleep(10)
    
    def fetch_single_stock(self, ticker_symbol, period="1mo", repair=True):
        """
        Fetch data for a single stock with all safety measures
        
        Args:
            ticker_symbol: Full ticker with exchange suffix
            period: Time period (1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max)
            repair: Whether to repair data issues
            
        Returns:
            dict with 'data' (DataFrame), 'info' (dict), and 'valid' (bool)
        """
        self._rate_limit()
        
        result = {
            'ticker': ticker_symbol,
            'data': None,
            'info': None,
            'valid': False,
            'errors': []
        }
        
        try:
            # Fetch ticker
            stock = yf.Ticker(ticker_symbol)
            
            # Get historical data
            data = stock.history(period=period, repair=repair)
            
            if data.empty:
                result['errors'].append("No historical data available")
                return result
                
            # Get info
            try:
                info = stock.info
                result['info'] = {
                    'name': info.get('longName', ticker_symbol),
                    'exchange': info.get('exchange', 'Unknown'),
                    'currency': info.get('currency', 'Unknown'),
                    'country': info.get('country', 'Unknown'),
                    'sector': info.get('sector', 'Unknown'),
                    'marketCap': info.get('marketCap', 0),
                    'currentPrice': info.get('currentPrice', 0)
                }
            except Exception as e:
                result['errors'].append(f"Could not fetch info: {e}")
                result['info'] = {'name': ticker_symbol}
            
            # Clean data
            data = self._clean_data(data)
            
            # Validate
            is_valid, issues = self._validate_data(data)
            
            result['data'] = data
            result['valid'] = is_valid
            if issues:
                result['errors'].extend(issues)
                
            return result
            
        except Exception as e:
            result['errors'].append(f"Fetch error: {str(e)}")
            return result
    
    def fetch_multiple_stocks(self, ticker_list, period="1mo", repair=True):
        """
        Fetch data for multiple stocks efficiently
        
        Returns:
            dict mapping ticker to result dict
        """
        results = {}
        
        # First try bulk download
        try:
            self._rate_limit()
            
            bulk_data = yf.download(
                ticker_list,
                period=period,
                repair=repair,
                group_by='ticker',
                threads=True,
                progress=False
            )
            
            # Process bulk results
            for ticker in ticker_list:
                try:
                    if len(ticker_list) == 1:
                        # Single ticker returns different structure
                        ticker_data = bulk_data
                    else:
                        # Multiple tickers
                        ticker_data = bulk_data[ticker] if ticker in bulk_data.columns.levels[1] else pd.DataFrame()
                    
                    if not ticker_data.empty:
                        results[ticker] = {
                            'ticker': ticker,
                            'data': ticker_data,
                            'info': None,  # Fetch separately if needed
                            'valid': True,
                            'errors': []
                        }
                    else:
                        results[ticker] = {
                            'ticker': ticker,
                            'data': None,
                            'info': None,
                            'valid': False,
                            'errors': ['No data in bulk download']
                        }
                except Exception as e:
                    results[ticker] = {
                        'ticker': ticker,
                        'data': None,
                        'info': None,
                        'valid': False,
                        'errors': [f'Processing error: {str(e)}']
                    }
                    
        except Exception as e:
            print(f"Bulk download failed: {e}")
            # Fall back to individual downloads
            for ticker in ticker_list:
                results[ticker] = self.fetch_single_stock(ticker, period, repair)
                
        return results
    
    def _clean_data(self, data):
        """Clean common data issues"""
        if data is None or data.empty:
            return data
            
        # Remove all-NaN rows
        data = data.dropna(how='all')
        
        # Forward fill missing values (max 5 days)
        data = data.fillna(method='ffill', limit=5)
        
        return data
    
    def _validate_data(self, data):
        """Validate data quality"""
        issues = []
        
        if data is None or data.empty:
            return False, ["No data"]
            
        # Check data recency
        last_date = data.index[-1]
        days_old = (pd.Timestamp.now() - last_date).days
        if days_old > 7:
            issues.append(f"Data is {days_old} days old")
            
        # Check for suspicious prices
        if 'Close' in data.columns:
            returns = data['Close'].pct_change()
            extreme_moves = returns[returns.abs() > 0.5]
            if len(extreme_moves) > 0:
                issues.append(f"{len(extreme_moves)} days with >50% price change")
                
        return len(issues) == 0, issues

# Example usage
fetcher = InternationalStockFetcher()

# Single stock
bp_result = fetcher.fetch_single_stock("BP.L", period="3mo")
if bp_result['valid']:
    print(f"Successfully fetched {bp_result['info']['name']}")
    print(f"Latest price: {bp_result['data']['Close'].iloc[-1]:.2f} {bp_result['info']['currency']}")
else:
    print(f"Issues with BP.L: {bp_result['errors']}")

# Multiple stocks
portfolio = ["AAPL", "BP.L", "7203.T", "NESN.SW", "BHP.AX"]
results = fetcher.fetch_multiple_stocks(portfolio, period="1mo")

for ticker, result in results.items():
    if result['valid']:
        latest_price = result['data']['Close'].iloc[-1]
        print(f"{ticker}: {latest_price:.2f}")
    else:
        print(f"{ticker}: Failed - {result['errors']}")
```

## Common Pitfalls and Solutions

### Pitfall 1: Wrong Ticker Format

```python
# WRONG - These will fail or return wrong data
wrong_examples = [
    "BP",          # Missing .L suffix for London
    "TOYOTA",      # Should be 7203.T
    "Nestle",      # Should be NESN.SW
    "BMW.FRA",     # Should be BMW.DE
]

# CORRECT
correct_examples = [
    "BP.L",        # BP on London Stock Exchange
    "7203.T",      # Toyota on Tokyo Stock Exchange  
    "NESN.SW",     # Nestlé on Swiss Exchange
    "BMW.DE",      # BMW on Frankfurt Exchange
]

# Always verify the ticker works
def verify_ticker(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        if info.get('symbol'):
            return True, info.get('longName', 'Unknown')
        return False, "No data"
    except:
        return False, "Error"

# Test your tickers first!
for ticker in ["BP", "BP.L"]:
    valid, name = verify_ticker(ticker)
    print(f"{ticker}: {valid} - {name}")
```

### Pitfall 2: Ignoring Empty Data

```python
# WRONG - This will crash
data = yf.download("INVALID.TICKER")
closing_price = data['Close'].iloc[-1]  # IndexError!

# CORRECT - Always check
data = yf.download("INVALID.TICKER")
if not data.empty:
    closing_price = data['Close'].iloc[-1]
else:
    print("No data available")
```

### Pitfall 3: Rate Limit Violations

```python
# WRONG - This will get you blocked
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"] * 20  # 80 requests
for ticker in tickers:
    data = yf.Ticker(ticker).history()  # No delay!

# CORRECT - Use bulk download or add delays
# Option 1: Bulk download (preferred)
data = yf.download(tickers, period="1mo", group_by='ticker')

# Option 2: Individual with delays
for ticker in tickers:
    data = yf.Ticker(ticker).history()
    time.sleep(1)  # Minimum 1 second delay
```

### Pitfall 4: Timezone Confusion

```python
# WRONG - Comparing prices at different times
uk_data = yf.download("BP.L", period="1d", interval="1h")
us_data = yf.download("XOM", period="1d", interval="1h")
# These timestamps are in different timezones!

# CORRECT - Normalize timezones
uk_data.index = uk_data.index.tz_localize('Europe/London').tz_convert('UTC')
us_data.index = us_data.index.tz_localize('America/New_York').tz_convert('UTC')
# Now you can compare them properly
```

### Pitfall 5: Not Handling Splits/Dividends

```python
# WRONG - Using raw close prices
data = yf.download("AAPL", period="10y")
total_return = (data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1

# CORRECT - Use adjusted close
data = yf.download("AAPL", period="10y")
total_return = (data['Adj Close'].iloc[-1] / data['Adj Close'].iloc[0]) - 1
# This accounts for splits and dividends
```

### Pitfall 6: Hardcoding Currency

```python
# WRONG - Assuming all prices are in USD
price = yf.Ticker("BP.L").info['currentPrice']
print(f"Price: ${price}")  # This is actually in GBP!

# CORRECT - Check currency
ticker = yf.Ticker("BP.L")
price = ticker.info['currentPrice']
currency = ticker.info['currency']
print(f"Price: {price} {currency}")
```

## Quick Reference Card

### Essential Code Snippets

```python
# 1. Import everything you need
import yfinance as yf
import pandas as pd
import time
from datetime import datetime, timedelta

# 2. Download single stock
stock = yf.Ticker("BP.L")
data = stock.history(period="1mo")

# 3. Download multiple stocks
tickers = ["AAPL", "BP.L", "7203.T"]
data = yf.download(tickers, period="1mo", group_by='ticker')

# 4. Get company info
info = yf.Ticker("BP.L").info
print(f"{info['longName']} - {info['currency']}")

# 5. Check if data exists
if not data.empty:
    latest_price = data['Close'].iloc[-1]

# 6. Always use try-except
try:
    data = yf.Ticker("BP.L").history()
except Exception as e:
    print(f"Error: {e}")

# 7. Add delays between requests
time.sleep(1)

# 8. Use adjusted close for returns
returns = data['Adj Close'].pct_change()

# 9. Download with repair
data = yf.download("BP.L", repair=True)

# 10. Search for tickers
from yfinance import Search
results = Search("British Petroleum")
```

### Emergency Fixes

If yfinance stops working:

1. **Update yfinance**: `pip install --upgrade yfinance`
2. **Clear cache**: `yf.Ticker("AAPL").history(period="1d", repair=True)`
3. **Try different period**: Some periods may work when others don't
4. **Check Yahoo Finance website**: If it's down, yfinance won't work
5. **Use VPN**: Sometimes geographic restrictions apply
6. **Reduce request rate**: Add longer delays between requests

## Summary

Remember these key points:

1. **ALWAYS** use exchange suffixes for non-US stocks
2. **ALWAYS** check if data is empty before using it
3. **ALWAYS** handle errors with try-except blocks
4. **ALWAYS** respect rate limits (1 request per second minimum)
5. **ALWAYS** use `repair=True` for better data quality
6. **NEVER** use yfinance for real-time trading
7. **NEVER** assume data will be available
8. **NEVER** ignore timezone differences
9. **PREFER** bulk downloads over loops
10. **PREFER** adjusted close prices for return calculations

When in doubt, test with a known good ticker like "AAPL" first, then try your international ticker.