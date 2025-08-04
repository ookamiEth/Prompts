# Product Requirements Document: Bybit Token Launch Performance Analysis

## Executive Summary
This PRD outlines the requirements for a Python script that performs a one-time historical analysis of tokens launched on Bybit spot market between July 18, 2024 and January 18, 2025. The script will track key metrics at specific time intervals using the CoinGecko Pro API as the sole data source.

## Project Overview

### Objective
Create a Python script that generates a comprehensive analysis of tokens listed on Bybit in the past six months, tracking their price, market cap, FDV, float percentage, and token supply metrics at launch and predetermined intervals.

### Scope
- **Time Period**: Fixed historical analysis period - July 18, 2024 to January 18, 2025 (NOT dynamic)
- **Analysis Type**: One-time historical analysis (not a reusable script with dynamic dates)
- **Data Source**: CoinGecko Pro API (exclusively)
- **Output**: CSV or Parquet file containing all tracked metrics
- **Currency**: All values in USD

## Technical Requirements

### Data Source
- **API**: CoinGecko Pro API only
- **Authentication**: Use environment variable `COINGECKO_PRO_API_KEY`
- **Base URL**: `https://pro-api.coingecko.com/api/v3/`

### Token List
**IMPORTANT**: For this implementation, the list of tokens that launched on Bybit in the last six months will be HARDCODED in the script. The junior developer must research and create this list.

### Metrics to Track

For each token, capture the following metrics at each timepoint:

1. **Price** (USD)
2. **Market Cap** (USD)
3. **FDV (Fully Diluted Valuation)** (USD)
4. **Float Percentage** (calculated as: circulating_supply / total_supply * 100)
5. **Circulating Supply** (number of tokens)
6. **Total Supply** (number of tokens)

### Timepoints

Track metrics at exactly these intervals from the launch date:
1. **Launch Date** (Day 0) - First day of trading data available on CoinGecko
2. **7 Days After Launch**
3. **14 Days After Launch** (2 weeks)
4. **28 Days After Launch** (4 weeks)
5. **90 Days After Launch** (3 months)
6. **180 Days After Launch** (6 months)

## Implementation Requirements

### 1. Environment Setup
```python
# Required imports
import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.environ.get("COINGECKO_PRO_API_KEY")
```

### 2. Hardcoded Token List
The junior developer must create a list of dictionaries containing tokens that were listed on Bybit between July 18, 2024 and January 18, 2025:
```python
# Static list for one-time historical analysis
# Only include tokens listed on Bybit between 2024-07-18 and 2025-01-18
bybit_tokens = [
    {
        "symbol": "TOKEN_SYMBOL",
        "coingecko_id": "coingecko-id-here",
        "bybit_launch_date": "YYYY-MM-DD"  # Date listed on Bybit (must be within analysis period)
    },
    # ... more tokens
]
```

### 3. Data Collection Logic

**IMPORTANT**: With only 60 requests/minute, efficient batching is CRITICAL. Each token requires 6 API calls (one per timepoint), so plan accordingly.

#### Step 3.1: Determine Launch Date
For each token, the "launch date" is defined as the first day trading data is available on CoinGecko. Use the `/coins/{id}/history` endpoint to find the earliest available data.

#### Step 3.2: Collect Historical Data
For each timepoint, use the `/coins/{id}/history` endpoint with the specific date:
```
GET /coins/{id}/history?date={DD-MM-YYYY}
```

**Batching Strategy Example**:
- If analyzing 10 tokens with 6 timepoints each = 60 total API calls
- This would consume your entire minute's quota
- Plan token processing in batches to avoid hitting limits

#### Step 3.3: Handle Missing Data
- If exact date data is unavailable, use the **nearest available data point**
- Implement logic to search forward and backward from the target date
- Maximum search window: ±7 days from target date

### 4. Data Processing

#### Float Percentage Calculation
```python
def calculate_float_percentage(circulating_supply: float, total_supply: float) -> Optional[float]:
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
```

### 5. Output Requirements

#### File Format
- The junior developer must choose between CSV or Parquet format
- Recommendation: Use Parquet for better data type preservation and compression

#### File Location
- The junior developer must decide the output file location
- Suggested structure: `output/bybit_token_analysis_YYYYMMDD.{csv|parquet}`

#### Column Naming Convention
Use professional, unambiguous column names suitable for experienced traders:

```
token_symbol
coingecko_id
launch_date
price_usd_launch
market_cap_usd_launch
fdv_usd_launch
float_pct_launch
circulating_supply_launch
total_supply_launch
price_usd_7d
market_cap_usd_7d
fdv_usd_7d
float_pct_7d
circulating_supply_7d
total_supply_7d
price_usd_14d
market_cap_usd_14d
fdv_usd_14d
float_pct_14d
circulating_supply_14d
total_supply_14d
price_usd_28d
market_cap_usd_28d
fdv_usd_28d
float_pct_28d
circulating_supply_28d
total_supply_28d
price_usd_90d
market_cap_usd_90d
fdv_usd_90d
float_pct_90d
circulating_supply_90d
total_supply_90d
price_usd_180d
market_cap_usd_180d
fdv_usd_180d
float_pct_180d
circulating_supply_180d
total_supply_180d
```

### 6. Error Handling

Implement comprehensive error handling following CoinGecko best practices:
- API rate limit errors (429 status code) - Wait 60 seconds before retry
- Network timeouts - Use 10 second timeout for requests
- Invalid API responses - Implement retry with exponential backoff
- Missing data fields - Handle gracefully with None values
- Data type conversion errors - Validate before processing
- Maximum 3 retry attempts per request

Example implementation:
```python
def safe_api_call(url: str, params: Dict, max_retries: int = 3) -> Optional[Dict]:
    """Make API call with proper error handling and retries."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=10)
            
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
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    
    return None
```

### 7. Code Style Requirements

- **NO EMOJIS** in code or output
- Concise, professional code
- Clear variable names
- Minimal comments (only where absolutely necessary)
- Follow PEP 8 style guide

## API Implementation Details

### CoinGecko Historical Data Endpoint
```python
def get_historical_data(coin_id: str, date: str) -> Dict:
    """
    Fetch historical data for a coin on a specific date.
    
    Args:
        coin_id: CoinGecko coin ID
        date: Date in DD-MM-YYYY format
    
    Returns:
        Dictionary with market data
    """
    url = f"https://pro-api.coingecko.com/api/v3/coins/{coin_id}/history"
    params = {
        "date": date,
        "localization": "false",
        "x_cg_pro_api_key": api_key
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()
```

### Data Extraction
From the API response, extract:
- `market_data.current_price.usd` → price_usd
- `market_data.market_cap.usd` → market_cap_usd
- `market_data.fully_diluted_valuation.usd` → fdv_usd
- `market_data.circulating_supply` → circulating_supply
- `market_data.total_supply` → total_supply

## Performance Considerations

1. **Rate Limiting**: **CRITICAL - Limited to 60 requests per minute maximum**
   - Add a minimum delay of 1 second between requests (60 calls/60 seconds)
   - **MUST use batched calls wherever possible** - This is essential given the low rate limit
   - Implement exponential backoff for retries
   - Handle 429 status code by waiting 60 seconds before retry
2. **Batching Strategy**: 
   - The `/simple/price` endpoint supports multiple coin IDs in a single request
   - Group multiple tokens into single API calls to maximize efficiency
   - Example: Instead of 10 individual calls, make 1 call with 10 coin IDs
3. **Caching**: Consider caching API responses to avoid redundant calls during development.
4. **Sequential Processing**: Process batches sequentially with appropriate delays.

## Validation Requirements

1. Verify all hardcoded CoinGecko IDs are valid
2. Ensure all dates are within the analysis period
3. Validate numeric data types and handle None values
4. Check for logical inconsistencies (e.g., circulating > total supply)

## Example Implementation Structure

```python
# main.py
import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Configuration for one-time historical analysis
# Analysis period: July 18, 2024 to January 18, 2025
ANALYSIS_START_DATE = "2024-07-18"
ANALYSIS_END_DATE = "2025-01-18"

BYBIT_TOKENS = [
    # Junior developer must populate this list
    # Only tokens listed on Bybit between ANALYSIS_START_DATE and ANALYSIS_END_DATE
]

TIMEPOINTS = [0, 7, 14, 28, 90, 180]  # Days from launch

def main():
    """Main execution function."""
    results = []
    
    for token in BYBIT_TOKENS:
        token_data = collect_token_data(token)
        results.append(token_data)
        time.sleep(1.0)  # Rate limiting: 60 calls/min = 1s per call minimum
    
    # Convert to DataFrame and save
    df = pd.DataFrame(results)
    save_results(df)

def collect_token_data(token: Dict) -> Dict:
    """Collect all timepoint data for a single token."""
    # IMPORTANT: Each token requires 6 API calls (one per timepoint)
    # With 60 requests/min limit, you can only process 10 tokens per minute
    # Implementation here
    pass

def batch_get_current_prices(coin_ids: List[str]) -> Dict:
    """Get current prices for multiple coins in a single API call."""
    # Use this for any current price needs to minimize API calls
    url = "https://pro-api.coingecko.com/api/v3/simple/price"
    params = {
        "ids": ",".join(coin_ids),
        "vs_currencies": "usd",
        "include_market_cap": "true",
        "include_24hr_vol": "true",
        "x_cg_pro_api_key": api_key
    }
    return safe_api_call(url, params)

def save_results(df: pd.DataFrame):
    """Save results to CSV or Parquet."""
    # Junior developer decides location and format
    pass

if __name__ == "__main__":
    main()
```

## Deliverables

1. **Python Script**: Complete implementation following all requirements
2. **Output File**: CSV or Parquet file with all token metrics
3. **No Additional Documentation**: The code should be self-explanatory

## Important Reminders

1. **ONLY use CoinGecko Pro API** - Do not use Bybit API
2. **All values in USD** - No other currencies
3. **Professional output** - No emojis, clear column names
4. **Hardcode the token list** - Research and create the list yourself
5. **Handle all edge cases** - Especially for float percentage calculation
6. **Decide file location yourself** - Choose an appropriate output directory
7. **CRITICAL: Only 60 API requests per minute** - Plan your implementation accordingly
8. **Use batched API calls wherever possible** - Essential for staying within rate limits

## Success Criteria

The script successfully:
1. Retrieves historical data for all hardcoded tokens
2. Calculates metrics for all six timepoints
3. Handles missing data appropriately
4. Produces a clean, professional output file
5. Runs without errors for the complete token list

---
description: Interactive review protocol for Bybit token launch analysis
globs:
alwaysApply: true
---

## AI Agent Protocol: Interactive Final Review for Token Launch Analysis

### When to Apply This Protocol
This protocol should be activated when:
1. Building or modifying the Bybit token launch analysis tool
2. Running historical analysis that affects token performance data
3. Making significant changes to data collection methods or calculations
4. User requests verification of analysis results before finalizing

### Phase 0: Review Script Setup
The AI will ensure `final_review_gate.py` exists in your project root with the interactive review functionality.

### Phase 1: Primary Analysis Implementation
The AI will complete the requested analysis task (e.g., implementing data collection, calculating metrics, handling API rate limits).

### Phase 2: Mandatory Review Checkpoint
Before considering the task complete, the AI will:
1. Run `final_review_gate.py` to enter interactive review mode
2. Allow you to:
   - Review token list and date ranges
   - Test specific token calculations
   - Verify API rate limit handling
   - Check data quality and completeness
   - Request adjustments to output format

### Phase 3: Iterative Refinement
You can provide sub-prompts like:
- "Show me the float percentage calculation for token X"
- "Add validation for missing historical data"
- "Test the rate limiting with 15 tokens"
- "Include a data quality summary"

### Phase 4: Explicit Completion
The task is only complete when you type `TASK_COMPLETE` in the review terminal.

This ensures that complex token analysis tools meet your exact requirements before being finalized.