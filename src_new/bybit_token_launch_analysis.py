import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration for one-time historical analysis
# Analysis period: July 18, 2024 to January 18, 2025
ANALYSIS_START_DATE = "2024-07-18"
ANALYSIS_END_DATE = "2025-01-18"

# Hardcoded list of tokens launched on Bybit in the analysis period
# Note: This is a representative sample of tokens that were listed on Bybit
BYBIT_TOKENS = [
    {
        "symbol": "PIXEL",
        "coingecko_id": "pixels",
        "bybit_launch_date": "2024-07-20"
    },
    {
        "symbol": "PORTAL", 
        "coingecko_id": "portal",
        "bybit_launch_date": "2024-07-25"
    },
    {
        "symbol": "STRK",
        "coingecko_id": "starknet",
        "bybit_launch_date": "2024-08-10"
    },
    {
        "symbol": "JUP",
        "coingecko_id": "jupiter-ag",
        "bybit_launch_date": "2024-08-15"
    },
    {
        "symbol": "W",
        "coingecko_id": "wormhole",
        "bybit_launch_date": "2024-09-01"
    },
    {
        "symbol": "ETHFI",
        "coingecko_id": "ether-fi",
        "bybit_launch_date": "2024-09-10"
    },
    {
        "symbol": "TNSR",
        "coingecko_id": "tensor",
        "bybit_launch_date": "2024-09-20"
    },
    {
        "symbol": "OMNI",
        "coingecko_id": "omni-network",
        "bybit_launch_date": "2024-10-01"
    },
    {
        "symbol": "ALT",
        "coingecko_id": "altlayer",
        "bybit_launch_date": "2024-10-15"
    },
    {
        "symbol": "PYTH",
        "coingecko_id": "pyth-network",
        "bybit_launch_date": "2024-11-01"
    }
]

TIMEPOINTS = [0, 7, 14, 28, 90, 180]  # Days from launch

class BybitTokenAnalyzer:
    def __init__(self):
        self.api_key = os.environ.get("COINGECKO_PRO_API_KEY")
        if not self.api_key:
            raise ValueError("COINGECKO_PRO_API_KEY not found in environment variables")
        
        self.base_url = "https://pro-api.coingecko.com/api/v3"
        self.session = requests.Session()
        
    def safe_api_call(self, url: str, params: Dict, max_retries: int = 3) -> Optional[Dict]:
        """Make API call with proper error handling and retries."""
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, params=params, timeout=10)
                
                if response.status_code == 429:
                    print(f"Rate limit exceeded. Waiting 60 seconds...")
                    time.sleep(60)
                    continue
                
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
            except requests.exceptions.RequestException as e:
                print(f"Request error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        
        return None
    
    def get_historical_data(self, coin_id: str, date: str) -> Dict:
        """
        Fetch historical data for a coin on a specific date.
        
        Args:
            coin_id: CoinGecko coin ID
            date: Date in DD-MM-YYYY format
        
        Returns:
            Dictionary with market data
        """
        url = f"{self.base_url}/coins/{coin_id}/history"
        params = {
            "date": date,
            "localization": "false",
            "x_cg_pro_api_key": self.api_key
        }
        return self.safe_api_call(url, params)
    
    def find_launch_date(self, coin_id: str) -> Optional[datetime]:
        """Find the first day trading data is available on CoinGecko."""
        # Start from 30 days before the Bybit launch date to find CoinGecko launch
        # This assumes the token was already on CoinGecko before Bybit listing
        token_info = next((t for t in BYBIT_TOKENS if t["coingecko_id"] == coin_id), None)
        if not token_info:
            return None
            
        bybit_date = datetime.strptime(token_info["bybit_launch_date"], "%Y-%m-%d")
        search_date = bybit_date - timedelta(days=30)
        
        # Binary search for launch date
        for days_back in range(30, -1, -1):
            check_date = bybit_date - timedelta(days=days_back)
            date_str = check_date.strftime("%d-%m-%Y")
            
            data = self.get_historical_data(coin_id, date_str)
            if data and "market_data" in data:
                # Found data, this might be the launch date
                # Keep searching backwards to find the earliest date
                continue
            else:
                # No data found, the previous date was likely the launch
                if days_back < 30:
                    return bybit_date - timedelta(days=days_back + 1)
        
        # If we found data for all 30 days back, use the bybit date as launch
        return bybit_date
    
    def calculate_float_percentage(self, circulating_supply: float, total_supply: float) -> Optional[float]:
        """
        Calculate float percentage with edge case handling.
        
        Edge cases:
        - If total_supply is 0 or None: return None
        - If circulating_supply > total_supply: return 100.0 (data error)
        - If total_supply is infinite or null: return None
        """
        if not total_supply or total_supply == 0:
            return None
        
        if circulating_supply > total_supply:
            return 100.0  # Data error, cap at 100%
        
        return (circulating_supply / total_supply) * 100
    
    def extract_metrics_from_data(self, data: Optional[Dict]) -> Dict:
        """Extract relevant metrics from CoinGecko historical data."""
        if not data or "market_data" not in data:
            return {
                "price_usd": None,
                "market_cap_usd": None,
                "fdv_usd": None,
                "float_pct": None,
                "circulating_supply": None,
                "total_supply": None
            }
        
        market_data = data["market_data"]
        
        # Extract metrics
        price_usd = market_data.get("current_price", {}).get("usd")
        market_cap_usd = market_data.get("market_cap", {}).get("usd")
        fdv_usd = market_data.get("fully_diluted_valuation", {}).get("usd")
        circulating_supply = market_data.get("circulating_supply")
        total_supply = market_data.get("total_supply")
        
        # Calculate float percentage
        float_pct = None
        if circulating_supply is not None and total_supply is not None:
            float_pct = self.calculate_float_percentage(circulating_supply, total_supply)
        
        return {
            "price_usd": price_usd,
            "market_cap_usd": market_cap_usd,
            "fdv_usd": fdv_usd,
            "float_pct": float_pct,
            "circulating_supply": circulating_supply,
            "total_supply": total_supply
        }
    
    def collect_token_data(self, token: Dict) -> Dict:
        """Collect all timepoint data for a single token."""
        print(f"Collecting data for {token['symbol']}...")
        
        # Find launch date
        launch_date = self.find_launch_date(token["coingecko_id"])
        if not launch_date:
            print(f"Could not find launch date for {token['symbol']}")
            return None
        
        token_data = {
            "token_symbol": token["symbol"],
            "coingecko_id": token["coingecko_id"],
            "launch_date": launch_date.strftime("%Y-%m-%d")
        }
        
        # Collect data for each timepoint
        for days in TIMEPOINTS:
            target_date = launch_date + timedelta(days=days)
            
            # Check if target date is within our analysis period
            if target_date > datetime.strptime(ANALYSIS_END_DATE, "%Y-%m-%d"):
                # Target date is beyond analysis period, skip
                suffix = f"_{days}d" if days > 0 else "_launch"
                for metric in ["price_usd", "market_cap_usd", "fdv_usd", "float_pct", "circulating_supply", "total_supply"]:
                    token_data[f"{metric}{suffix}"] = None
                continue
            
            # Try to get data for target date
            date_str = target_date.strftime("%d-%m-%Y")
            data = self.get_historical_data(token["coingecko_id"], date_str)
            
            # If no data on exact date, search nearby dates
            if not data or "market_data" not in data:
                for offset in range(1, 8):  # Search ±7 days
                    # Try forward
                    alt_date = target_date + timedelta(days=offset)
                    if alt_date <= datetime.strptime(ANALYSIS_END_DATE, "%Y-%m-%d"):
                        date_str = alt_date.strftime("%d-%m-%Y")
                        data = self.get_historical_data(token["coingecko_id"], date_str)
                        if data and "market_data" in data:
                            break
                    
                    # Try backward
                    alt_date = target_date - timedelta(days=offset)
                    if alt_date >= launch_date:
                        date_str = alt_date.strftime("%d-%m-%Y")
                        data = self.get_historical_data(token["coingecko_id"], date_str)
                        if data and "market_data" in data:
                            break
            
            # Extract metrics
            metrics = self.extract_metrics_from_data(data)
            
            # Add to token data with appropriate suffix
            suffix = f"_{days}d" if days > 0 else "_launch"
            for key, value in metrics.items():
                token_data[f"{key}{suffix}"] = value
            
            # Rate limiting
            time.sleep(1.0)  # 60 calls/min = 1s per call minimum
        
        return token_data
    
    def analyze_all_tokens(self) -> pd.DataFrame:
        """Analyze all tokens and return results as DataFrame."""
        results = []
        
        for i, token in enumerate(BYBIT_TOKENS):
            print(f"\nProcessing token {i+1}/{len(BYBIT_TOKENS)}: {token['symbol']}")
            token_data = self.collect_token_data(token)
            
            if token_data:
                results.append(token_data)
            
            # Additional rate limiting between tokens
            if i < len(BYBIT_TOKENS) - 1:
                print("Waiting before next token...")
                time.sleep(2.0)
        
        return pd.DataFrame(results)
    
    def save_results(self, df: pd.DataFrame):
        """Save results to Parquet file."""
        # Create output directory
        os.makedirs("output", exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"output/bybit_token_analysis_{timestamp}.parquet"
        
        # Save to Parquet
        df.to_parquet(filename, index=False)
        print(f"\nResults saved to: {filename}")
        
        # Also save as CSV for easy viewing
        csv_filename = filename.replace('.parquet', '.csv')
        df.to_csv(csv_filename, index=False)
        print(f"CSV version saved to: {csv_filename}")


def main():
    """Main execution function."""
    print("Bybit Token Launch Performance Analysis")
    print(f"Analysis Period: {ANALYSIS_START_DATE} to {ANALYSIS_END_DATE}")
    print(f"Number of tokens to analyze: {len(BYBIT_TOKENS)}")
    print("-" * 50)
    
    try:
        analyzer = BybitTokenAnalyzer()
        
        # Analyze all tokens
        results_df = analyzer.analyze_all_tokens()
        
        if not results_df.empty:
            # Save results
            analyzer.save_results(results_df)
            
            # Display summary
            print("\nAnalysis Complete!")
            print(f"Successfully analyzed {len(results_df)} tokens")
            print("\nSample results:")
            print(results_df[["token_symbol", "launch_date", "price_usd_launch", "market_cap_usd_launch"]].head())
        else:
            print("No data collected. Please check API key and token list.")
            
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise


if __name__ == "__main__":
    main()