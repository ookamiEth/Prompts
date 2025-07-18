#!/usr/bin/env python
import os
import sys
import time
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Crypto Holdings (Placeholder Values)
CRYPTO_HOLDINGS = {
    # Bitcoin Holdings - Core 15
    "MSTR": {"crypto": "bitcoin", "amount": 500000, "name": "MicroStrategy"},
    "MARA": {"crypto": "bitcoin", "amount": 40000, "name": "MARA Holdings"},
    "RIOT": {"crypto": "bitcoin", "amount": 15000, "name": "Riot Platforms"},
    "TSLA": {"crypto": "bitcoin", "amount": 10000, "name": "Tesla"},
    "CLSK": {"crypto": "bitcoin", "amount": 8000, "name": "CleanSpark"},
    
    # Bitcoin Holdings - Additional
    "CEP": {"crypto": "bitcoin", "amount": 35000, "name": "XXI (Twenty One Capital)"},
    "3350.T": {"crypto": "bitcoin", "amount": 15000, "name": "Metaplanet Inc"},
    "GLXY.TO": {"crypto": "bitcoin", "amount": 14000, "name": "Galaxy Digital Holdings"},
    "HUT": {"crypto": "bitcoin", "amount": 9000, "name": "Hut 8 Mining Corp"},
    "COIN": {"crypto": "bitcoin", "amount": 8500, "name": "Coinbase Global"},
    "SQ": {"crypto": "bitcoin", "amount": 8000, "name": "Block Inc"},
    "SMLR": {"crypto": "bitcoin", "amount": 4500, "name": "Semler Scientific"},
    "GME": {"crypto": "bitcoin", "amount": 4000, "name": "GameStop Corp"},
    "HIVE": {"crypto": "bitcoin", "amount": 2000, "name": "Hive Digital Technologies"},
    "BITF": {"crypto": "bitcoin", "amount": 700, "name": "Bitfarms Limited"},
    
    # Ethereum Holdings - Core 5
    "BMNR": {"crypto": "ethereum", "amount": 250000, "name": "BitMine Immersion"},
    "SBET": {"crypto": "ethereum", "amount": 200000, "name": "SharpLink Gaming"},
    "BTBT": {"crypto": "ethereum", "amount": 80000, "name": "Bit Digital"},
    "BTCS": {"crypto": "ethereum", "amount": 25000, "name": "BTCS Inc"},
    "GAME": {"crypto": "ethereum", "amount": 1500, "name": "GameSquare Holdings"},
    
    # Ethereum Holdings - Additional
    "BTCT": {"crypto": "ethereum", "amount": 45000, "name": "BTC Digital Ltd"},
    # Note: COIN also holds ETH but is already listed for BTC
    
    # Solana Holdings - Core 5
    "UPXI": {"crypto": "solana", "amount": 1500000, "name": "Upexi Inc"},
    "DFDV": {"crypto": "solana", "amount": 800000, "name": "DeFi Development Corp"},
    "HODL": {"crypto": "solana", "amount": 400000, "name": "Sol Strategies Inc"},
    "KIDZ": {"crypto": "solana", "amount": 6000, "name": "Classover Holdings"},
    "TORR": {"crypto": "solana", "amount": 35000, "name": "Torrent Capital Ltd"},
    
    # Solana Holdings - Additional
    "BTCM": {"crypto": "solana", "amount": 1200000, "name": "BIT Mining Limited"},
    "DTCK": {"crypto": "solana", "amount": 50000, "name": "Davis Commodities Limited"},
    "LGHL": {"crypto": "solana", "amount": 30000, "name": "Lion Group Holding Ltd"},
    "CLKH": {"crypto": "solana", "amount": 200000, "name": "Click Holdings"},
    "NDA.V": {"crypto": "solana", "amount": 20000, "name": "Neptune Digital Assets"},
    "DEFTF": {"crypto": "solana", "amount": 15000, "name": "DeFi Technologies"}
}

# Add COIN's ETH holdings as a separate entry
CRYPTO_HOLDINGS["COIN_ETH"] = {"crypto": "ethereum", "amount": 10000, "name": "Coinbase Global", "ticker": "COIN"}


def validate_environment():
    """Validate environment and load API key."""
    load_dotenv()
    api_key = os.environ.get("COINGECKO_PRO_API_KEY")
    if not api_key:
        raise ValueError("COINGECKO_PRO_API_KEY not found in environment")
    return api_key


def fetch_crypto_prices(api_key, max_retries=3):
    """Fetch current crypto prices from CoinGecko."""
    url = "https://pro-api.coingecko.com/api/v3/simple/price"
    headers = {"x-cg-pro-api-key": api_key}  # Use header for security
    params = {
        "ids": "bitcoin,ethereum,solana",
        "vs_currencies": "usd"
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Validate all expected cryptos present
            for crypto in ["bitcoin", "ethereum", "solana"]:
                if crypto not in data:
                    raise ValueError(f"Missing {crypto} in response")
            
            return data
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            raise Exception(f"Failed to fetch crypto prices after {max_retries} attempts: {e}")


def validate_all_tickers(tickers):
    """Validate all tickers are valid."""
    invalid_tickers = []
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Check both info and minimal history
            if not info.get('symbol'):
                invalid_tickers.append((ticker, "No symbol in info"))
                continue
                
            # Verify data availability
            hist = stock.history(period="5d")
            if hist.empty:
                invalid_tickers.append((ticker, "No historical data"))
                
        except Exception as e:
            invalid_tickers.append((ticker, str(e)))
    
    if invalid_tickers:
        for ticker, reason in invalid_tickers:
            print(f"Invalid ticker {ticker}: {reason}")
        raise ValueError(f"Found {len(invalid_tickers)} invalid tickers")


def calculate_mnav(market_cap, crypto_holdings, crypto_price):
    """Calculate mNAV (market cap to crypto holdings ratio)."""
    crypto_value = crypto_holdings * crypto_price
    if crypto_value > 0:
        return market_cap / crypto_value
    return None


def calculate_volume_metrics(hist_data, currency):
    """Calculate volume metrics for different periods."""
    if hist_data.empty:
        return None
    
    # Filter trading days only
    trading_data = hist_data[hist_data['Volume'] > 0].copy()
    
    if len(trading_data) == 0:
        return None
    
    # Convert to USD if needed (using current FX rate as approximation)
    if currency != 'USD':
        fx_ticker = f"{currency}USD=X"
        try:
            fx_rate = yf.Ticker(fx_ticker).info.get('regularMarketPrice', 1)
        except:
            fx_rate = 1  # Default to 1 if FX fetch fails
        trading_data['Volume_USD'] = trading_data['Volume'] * trading_data['Close'] * fx_rate
    else:
        trading_data['Volume_USD'] = trading_data['Volume'] * trading_data['Close']
    
    # Calculate metrics for trading days
    periods = {
        'volume_1d': 1,    # Last trading day
        'volume_5d': 5,    # ~1 week
        'volume_22d': 22,  # ~1 month
        'volume_63d': 63   # ~3 months
    }
    
    results = {}
    for period_name, days in periods.items():
        if len(trading_data) >= days:
            # Use median for outlier resistance
            results[period_name] = trading_data['Volume_USD'].tail(days).median()
        else:
            results[period_name] = None
            results[f'{period_name}_flag'] = 'insufficient_data'
    
    return results


def get_public_float(info):
    """Extract public float information."""
    shares_outstanding = info.get('sharesOutstanding')
    float_shares = info.get('floatShares')
    
    return {
        'shares_outstanding': shares_outstanding,
        'float_shares': float_shares,
        'float_percentage': (float_shares / shares_outstanding * 100) if shares_outstanding and float_shares else None,
        'float_data_quality': 'complete' if float_shares else 'incomplete'
    }


def calculate_liquidity_metrics(market_cap, volume_5d, volume_22d):
    """Calculate liquidity metrics based on volume and market cap."""
    if not market_cap:
        return {"score": "low", "ratio": None}
    
    # Use weighted average of different periods
    if volume_22d and volume_5d:
        weighted_volume = (0.7 * volume_22d) + (0.3 * volume_5d)
    elif volume_22d:
        weighted_volume = volume_22d
    elif volume_5d:
        weighted_volume = volume_5d
    else:
        return {"score": "low", "ratio": None}
    
    ratio = weighted_volume / market_cap
    
    if ratio > 0.02:
        score = "high"
    elif ratio > 0.005:
        score = "medium"
    else:
        score = "low"
    
    return {"score": score, "ratio": ratio}


def analyze_crypto_treasuries():
    """Main function to analyze crypto treasuries."""
    # Validate environment
    try:
        api_key = validate_environment()
    except ValueError as e:
        print(f"Environment error: {e}")
        return
    
    # Get unique tickers (handle COIN appearing twice)
    unique_tickers = set()
    for key, holdings_info in CRYPTO_HOLDINGS.items():
        if 'ticker' in holdings_info:
            unique_tickers.add(holdings_info['ticker'])
        else:
            unique_tickers.add(key)
    
    # Validate all tickers
    try:
        validate_all_tickers(list(unique_tickers))
    except ValueError as e:
        print(f"Ticker validation failed: {e}")
        return
    
    # Fetch crypto prices
    try:
        crypto_prices = fetch_crypto_prices(api_key)
        time.sleep(1)  # Rate limit compliance
    except Exception as e:
        print(f"Failed to fetch crypto prices: {e}")
        return
    
    results = []
    
    # Process each ticker
    for key, holdings_info in CRYPTO_HOLDINGS.items():
        try:
            # Get actual ticker symbol
            ticker = holdings_info.get('ticker', key)
            
            # Fetch equity data
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Get required fields
            market_cap = info.get('marketCap')
            if not market_cap:
                raise ValueError("No market cap data")
            
            currency = info.get('currency', 'USD')
            
            # Get historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=180)
            hist_data = stock.history(start=start_date, end=end_date)
            
            # Get crypto price
            crypto_type = holdings_info['crypto']
            crypto_price = crypto_prices[crypto_type]['usd']
            
            # Calculate metrics
            mnav = calculate_mnav(market_cap, holdings_info['amount'], crypto_price)
            volume_metrics = calculate_volume_metrics(hist_data, currency)
            float_data = get_public_float(info)
            
            # Calculate liquidity
            if volume_metrics:
                liquidity = calculate_liquidity_metrics(
                    market_cap,
                    volume_metrics.get('volume_5d'),
                    volume_metrics.get('volume_22d')
                )
            else:
                liquidity = {"score": "low", "ratio": None}
            
            # Compile results
            result = {
                'ticker': ticker,
                'company_name': holdings_info['name'],
                'crypto_type': crypto_type,
                'crypto_holdings': holdings_info['amount'],
                'crypto_price': crypto_price,
                'crypto_value': holdings_info['amount'] * crypto_price,
                'market_cap': market_cap,
                'mnav': mnav,
                'volume_1d': volume_metrics.get('volume_1d') if volume_metrics else None,
                'volume_5d': volume_metrics.get('volume_5d') if volume_metrics else None,
                'volume_22d': volume_metrics.get('volume_22d') if volume_metrics else None,
                'volume_63d': volume_metrics.get('volume_63d') if volume_metrics else None,
                'shares_outstanding': float_data['shares_outstanding'],
                'public_float': float_data['float_shares'],
                'float_percentage': float_data['float_percentage'],
                'liquidity_score': liquidity['score'],
                'liquidity_ratio': liquidity['ratio'],
                'float_data_quality': float_data['float_data_quality'],
                'volume_data_quality': 'complete' if volume_metrics and all(
                    v is not None for v in [volume_metrics.get(f'volume_{d}d') for d in [1,5,22,63]]
                ) else 'incomplete'
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"Failed to process {key}: {e}")
            # Clean up on failure (all-or-nothing)
            if os.path.exists('crypto_treasury_analysis.csv'):
                os.remove('crypto_treasury_analysis.csv')
            return
    
    # Save results
    df = pd.DataFrame(results)
    # Sort by market cap descending
    df = df.sort_values('market_cap', ascending=False)
    df.to_csv('crypto_treasury_analysis.csv', index=False)
    print("Analysis complete. Results saved to crypto_treasury_analysis.csv")


if __name__ == "__main__":
    analyze_crypto_treasuries()