# Crypto Treasury Analysis Tool

This tool analyzes publicly traded companies holding cryptocurrency (Bitcoin, Ethereum, or Solana) as treasury assets.

## Setup

### 1. Create Conda Environment

```bash
# Create a new conda environment
conda create -n crypto_treasury python=3.10 -y

# Activate the environment
conda activate crypto_treasury

# Install required packages
conda install -c conda-forge pandas=2.1.4 requests=2.31.0 python-dotenv=1.0.0 -y
pip install yfinance==0.2.40  # Not available via conda
```

### 2. Configure API Key

1. Copy `.env.example` to `.env` (if example exists) or edit `.env` directly
2. Add your CoinGecko Pro API key:
   ```
   COINGECKO_PRO_API_KEY=CG-your-actual-api-key-here
   ```

### 3. Run the Analysis

```bash
python analyze_crypto_treasuries.py
```

## Output

The script generates `crypto_treasury_analysis.csv` with the following columns:
- ticker
- company_name
- crypto_type
- crypto_holdings
- crypto_price
- crypto_value
- market_cap
- mnav (market cap to crypto holdings ratio)
- volume metrics (1d, 5d, 22d, 63d)
- shares_outstanding
- public_float
- float_percentage
- liquidity_score (low/medium/high)
- liquidity_ratio
- data quality indicators

## Notes

- The crypto holdings amounts are placeholder values and should be updated with actual holdings
- Some tickers may be OTC/delisted - verify all are valid exchange-listed equities
- COIN appears twice in the holdings (for both BTC and ETH)
- FX conversion uses current exchange rates as approximation for historical volumes