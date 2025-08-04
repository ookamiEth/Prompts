# NumPy Best Practices for Quantitative Research & Trading

A comprehensive guide to numerical computing, array operations, and mathematical analysis practices for quantitative finance professionals.

## Table of Contents
1. [Array Creation & Initialization](#array-creation--initialization)
2. [Data Types & Memory Management](#data-types--memory-management)
3. [Array Indexing & Slicing](#array-indexing--slicing)
4. [Mathematical Operations & Broadcasting](#mathematical-operations--broadcasting)
5. [Linear Algebra for Finance](#linear-algebra-for-finance)
6. [Random Number Generation & Monte Carlo](#random-number-generation--monte-carlo)
7. [Performance Optimization](#performance-optimization)
8. [Financial Calculations & Vectorization](#financial-calculations--vectorization)
9. [Time Series Operations](#time-series-operations)
10. [Risk Calculations & Statistics](#risk-calculations--statistics)
11. [Matrix Operations for Portfolio Theory](#matrix-operations-for-portfolio-theory)
12. [Advanced Numerical Techniques](#advanced-numerical-techniques)

---

## Array Creation & Initialization

### 1. Standard Import and Setup
```python
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings

# Set consistent random seed for reproducible research
np.random.seed(42)  # ALWAYS set seed for research reproducibility

# Configure numpy display options for better readability
np.set_printoptions(
    precision=4,      # 4 decimal places for numbers
    suppress=True,    # Suppress scientific notation for small numbers
    threshold=50,     # Show max 50 elements before truncation
    linewidth=120     # Wrap arrays at 120 characters
)

# IMPORTANT: Never ignore all warnings - be selective
# warnings.filterwarnings('ignore')  # BAD - masks important issues
# Instead, be specific about what to suppress
warnings.filterwarnings('ignore', message='.*divide by zero.*', category=RuntimeWarning)
```

### 2. Efficient Array Creation Patterns
```python
def create_price_array(n_assets=100, n_periods=252, initial_price=100.0):
    """
    Create price arrays efficiently for backtesting
    
    IMPORTANT: Always specify dtype explicitly to avoid memory waste
    """
    # BAD - Let numpy guess the dtype
    # prices = np.ones((n_periods, n_assets)) * initial_price
    
    # GOOD - Specify dtype explicitly
    prices = np.full((n_periods, n_assets), initial_price, dtype=np.float64)
    
    return prices

def create_returns_array(n_assets=100, n_periods=252):
    """
    Create returns array with proper initialization
    """
    # Initialize with NaN for first period (no returns on day 1)
    returns = np.full((n_periods, n_assets), np.nan, dtype=np.float64)
    
    # Fill remaining periods with zeros (will be calculated later)
    returns[1:] = 0.0
    
    return returns

# Create common financial data structures efficiently
def initialize_financial_arrays():
    """
    Standard initialization for financial data analysis
    """
    n_assets, n_periods = 50, 252
    
    # Price matrix (periods x assets)
    prices = np.full((n_periods, n_assets), 100.0, dtype=np.float64)
    
    # Returns matrix (NaN for first period)
    returns = np.full((n_periods, n_assets), np.nan, dtype=np.float64)
    
    # Weights vector (equal weight initially)
    weights = np.full(n_assets, 1.0/n_assets, dtype=np.float64)
    
    # Time array (business days)
    dates = pd.date_range('2023-01-01', periods=n_periods, freq='B')
    
    return {
        'prices': prices,
        'returns': returns, 
        'weights': weights,
        'dates': dates
    }
```

### 3. Array Type Validation and Conversion
```python
def validate_and_convert_array(data, expected_shape=None, dtype=np.float64):
    """
    Robust array conversion with validation for financial data
    """
    # Convert to numpy array if needed
    if not isinstance(data, np.ndarray):
        if hasattr(data, 'values'):  # pandas Series/DataFrame
            array = data.values
        else:
            array = np.asarray(data, dtype=dtype)
    else:
        array = data.astype(dtype) if array.dtype != dtype else data
    
    # Validate shape if specified
    if expected_shape is not None:
        if array.shape != expected_shape:
            raise ValueError(f"Expected shape {expected_shape}, got {array.shape}")
    
    # Check for common data issues
    if np.any(np.isnan(array)) and np.all(np.isnan(array)):
        raise ValueError("Array contains only NaN values")
    
    if np.any(np.isinf(array)):
        raise ValueError("Array contains infinite values")
    
    return array

# Example usage for price data
def load_price_data_to_array(price_df):
    """
    Convert price DataFrame to numpy array with validation
    """
    # Ensure we have numeric data only
    numeric_columns = price_df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) == 0:
        raise ValueError("No numeric columns found in price data")
    
    # Convert to array and validate
    price_array = validate_and_convert_array(
        price_df[numeric_columns], 
        dtype=np.float64
    )
    
    # Check for negative prices (common data error)
    if np.any(price_array <= 0):
        negative_mask = price_array <= 0
        n_negative = np.sum(negative_mask)
        print(f"Warning: Found {n_negative} non-positive prices")
        # Option 1: Replace with NaN
        # price_array[negative_mask] = np.nan
        # Option 2: Forward fill (be careful!)
        # price_array = forward_fill_array(price_array)
    
    return price_array
```

---

## Data Types & Memory Management

### 1. Financial Data Type Standards
```python
# Standard data types for financial applications
FINANCIAL_DTYPES = {
    'price': np.float64,        # High precision needed for price calculations
    'return': np.float32,       # Lower precision acceptable for returns
    'volume': np.int64,         # Large integers for volume
    'signal': np.int8,          # Small integers for signals (-1, 0, 1)
    'weight': np.float64,       # High precision for portfolio weights
    'correlation': np.float32,  # Acceptable precision for correlations
}

def optimize_array_memory(array, data_type='price'):
    """
    Optimize array memory usage based on data type
    
    CRITICAL: Always test precision requirements before downcasting
    """
    original_size = array.nbytes
    
    if data_type == 'price':
        # Keep float64 for price data - precision is critical
        optimized = array.astype(np.float64)
    
    elif data_type == 'return':
        # Check if we can safely downcast returns
        if np.all(np.abs(array) < 3.4e38):  # float32 max value
            optimized = array.astype(np.float32)
        else:
            print("Warning: Values too large for float32, keeping float64")
            optimized = array.astype(np.float64)
    
    elif data_type == 'signal':
        # Integer signals can use int8 if values are small
        if np.all(np.abs(array) <= 127):  # int8 range
            optimized = array.astype(np.int8)
        elif np.all(np.abs(array) <= 32767):  # int16 range
            optimized = array.astype(np.int16)
        else:
            optimized = array.astype(np.int32)
    
    else:
        optimized = array  # No optimization for unknown types
    
    new_size = optimized.nbytes
    reduction = (original_size - new_size) / original_size * 100
    
    print(f"Memory reduction: {reduction:.1f}% ({original_size:,} -> {new_size:,} bytes)")
    
    return optimized

def check_array_memory_usage(arrays_dict):
    """
    Analyze memory usage of multiple arrays
    """
    print("Array Memory Usage Report:")
    print("-" * 50)
    
    total_bytes = 0
    for name, array in arrays_dict.items():
        bytes_used = array.nbytes
        mb_used = bytes_used / (1024 * 1024)
        total_bytes += bytes_used
        
        print(f"{name:15} {array.shape} {array.dtype} {mb_used:8.2f} MB")
    
    total_mb = total_bytes / (1024 * 1024)
    print(f"{'Total:':15} {total_mb:8.2f} MB")
    
    return total_bytes
```

### 2. Memory-Efficient Array Operations
```python
def efficient_array_operations():
    """
    Examples of memory-efficient array operations
    """
    n_assets, n_periods = 1000, 252
    
    # BAD - Creates multiple intermediate arrays
    prices = np.random.randn(n_periods, n_assets) * 100
    returns1 = (prices[1:] - prices[:-1]) / prices[:-1]
    log_returns1 = np.log(1 + returns1)
    
    # GOOD - In-place operations where possible
    prices = np.random.randn(n_periods, n_assets) * 100
    
    # Calculate returns in-place (be careful with this approach)
    returns2 = np.empty((n_periods-1, n_assets), dtype=np.float64)
    np.divide(np.subtract(prices[1:], prices[:-1], out=returns2), 
              prices[:-1], out=returns2)
    
    # For log returns, use numpy's built-in functions
    log_returns2 = np.log1p(returns2)  # More numerically stable than log(1+x)
    
    return returns2, log_returns2

def memory_efficient_covariance(returns, chunk_size=1000):
    """
    Calculate covariance matrix for large datasets without memory overflow
    """
    n_periods, n_assets = returns.shape
    
    if n_assets <= chunk_size:
        # Small enough to compute directly
        return np.cov(returns.T)
    
    # For large datasets, use chunked computation
    print(f"Computing covariance for {n_assets} assets using chunking...")
    
    covariance = np.zeros((n_assets, n_assets))
    
    # Center the data first
    mean_returns = np.nanmean(returns, axis=0)
    centered_returns = returns - mean_returns
    
    # Compute covariance in chunks
    for i in range(0, n_assets, chunk_size):
        end_i = min(i + chunk_size, n_assets)
        for j in range(i, n_assets, chunk_size):
            end_j = min(j + chunk_size, n_assets)
            
            # Compute cross-covariance block
            block = np.dot(centered_returns[:, i:end_i].T, 
                          centered_returns[:, j:end_j]) / (n_periods - 1)
            
            covariance[i:end_i, j:end_j] = block
            if i != j:  # Fill symmetric part
                covariance[j:end_j, i:end_i] = block.T
    
    return covariance
```

---

## Array Indexing & Slicing

### 1. Efficient Financial Data Indexing
```python
def advanced_array_indexing_examples():
    """
    Financial data indexing patterns that every quant must know
    """
    # Create sample data: 252 days x 10 assets
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, (252, 10))
    
    # 1. Time-based indexing (most common in finance)
    # Get last 21 days (1 month)
    recent_returns = returns[-21:]
    print(f"Recent returns shape: {recent_returns.shape}")
    
    # Get every 5th day (weekly sampling from daily data)
    weekly_returns = returns[::5]
    print(f"Weekly returns shape: {weekly_returns.shape}")
    
    # 2. Asset selection (columns)
    # Select specific assets
    asset_indices = [0, 2, 5, 8]  # Asset selection
    selected_returns = returns[:, asset_indices]
    print(f"Selected assets shape: {selected_returns.shape}")
    
    # 3. Boolean indexing for filtering
    # Find days with high volatility (any asset > 3 std devs)
    vol_threshold = 3 * np.std(returns, axis=0)
    high_vol_days = np.any(np.abs(returns) > vol_threshold, axis=1)
    crisis_returns = returns[high_vol_days]
    print(f"Crisis days: {np.sum(high_vol_days)}")
    
    # 4. Fancy indexing for rebalancing
    # Reorder assets by some ranking (e.g., by average return)
    avg_returns = np.mean(returns, axis=0)
    ranking = np.argsort(avg_returns)[::-1]  # Descending order
    reordered_returns = returns[:, ranking]
    
    return {
        'recent': recent_returns,
        'weekly': weekly_returns,
        'selected': selected_returns,
        'crisis': crisis_returns,
        'reordered': reordered_returns
    }

def rolling_window_indexing(data, window_size=21):
    """
    Create rolling windows efficiently for time series analysis
    
    IMPORTANT: This creates a view, not a copy, for memory efficiency
    """
    n_periods = len(data)
    if window_size > n_periods:
        raise ValueError(f"Window size {window_size} larger than data length {n_periods}")
    
    # Method 1: Using stride_tricks (memory efficient but advanced)
    from numpy.lib.stride_tricks import sliding_window_view
    
    # Create rolling windows (each row is a window)
    windows = sliding_window_view(data, window_shape=window_size, axis=0)
    
    return windows

def boolean_indexing_for_signals(returns, signals):
    """
    Use boolean indexing for signal-based analysis
    """
    # signals: -1 (sell), 0 (hold), 1 (buy)
    
    # Get returns only when we had long positions
    long_returns = returns[signals == 1]
    long_performance = np.mean(long_returns, axis=0)
    
    # Get returns only when we had short positions  
    short_returns = returns[signals == -1]
    short_performance = np.mean(short_returns, axis=0)
    
    # Calculate signal effectiveness
    signal_pnl = np.where(signals == 1, returns, 
                         np.where(signals == -1, -returns, 0))
    
    return {
        'long_performance': long_performance,
        'short_performance': short_performance,
        'signal_pnl': signal_pnl
    }
```

### 2. Advanced Slicing for Portfolio Analysis
```python
def portfolio_slicing_examples(returns, weights):
    """
    Advanced slicing patterns for portfolio analysis
    """
    n_periods, n_assets = returns.shape
    
    # 1. Time period analysis
    # Split data into train/test periods
    split_point = int(n_periods * 0.8)
    train_returns = returns[:split_point]
    test_returns = returns[split_point:]
    
    # 2. Sector analysis (assuming assets are grouped by sector)
    # Example: First 3 assets are tech, next 3 are finance, etc.
    sector_size = 3
    sectors = {}
    for i in range(0, n_assets, sector_size):
        sector_name = f'sector_{i//sector_size}'
        sectors[sector_name] = returns[:, i:i+sector_size]
    
    # 3. Portfolio component analysis
    # Get top 5 holdings by weight
    top_holdings_idx = np.argsort(weights)[-5:]
    top_holdings_returns = returns[:, top_holdings_idx]
    top_holdings_weights = weights[top_holdings_idx]
    
    # 4. Risk factor analysis
    # Separate systematic vs idiosyncratic risk
    market_return = np.average(returns, axis=1, weights=weights)  # Market proxy
    
    # Calculate betas for each asset
    betas = np.zeros(n_assets)
    for i in range(n_assets):
        covariance = np.cov(returns[:, i], market_return)[0, 1]
        market_variance = np.var(market_return)
        betas[i] = covariance / market_variance
    
    # Separate high-beta and low-beta assets
    beta_median = np.median(betas)
    high_beta_mask = betas > beta_median
    
    high_beta_returns = returns[:, high_beta_mask]
    low_beta_returns = returns[:, ~high_beta_mask]
    
    return {
        'train_returns': train_returns,
        'test_returns': test_returns,
        'sectors': sectors,
        'top_holdings': top_holdings_returns,
        'high_beta_returns': high_beta_returns,
        'low_beta_returns': low_beta_returns,
        'betas': betas
    }

def conditional_indexing_patterns(returns, market_returns=None):
    """
    Conditional indexing for regime-dependent analysis
    """
    # 1. Market regime indexing
    if market_returns is not None:
        # Bull vs bear market analysis
        market_median = np.median(market_returns)
        bull_mask = market_returns > market_median
        bear_mask = ~bull_mask
        
        bull_returns = returns[bull_mask]
        bear_returns = returns[bear_mask]
    else:
        bull_returns = bear_returns = None
    
    # 2. Volatility regime indexing
    # Calculate rolling volatility
    window = 21
    rolling_vol = np.zeros(len(returns) - window + 1)
    for i in range(len(rolling_vol)):
        rolling_vol[i] = np.std(returns[i:i+window].flatten())
    
    vol_threshold = np.percentile(rolling_vol, 75)  # Top quartile
    high_vol_periods = rolling_vol > vol_threshold
    
    # Extend mask to match returns length
    high_vol_mask = np.zeros(len(returns), dtype=bool)
    high_vol_mask[window-1:] = high_vol_periods
    
    high_vol_returns = returns[high_vol_mask]
    low_vol_returns = returns[~high_vol_mask]
    
    return {
        'bull_returns': bull_returns,
        'bear_returns': bear_returns,
        'high_vol_returns': high_vol_returns,
        'low_vol_returns': low_vol_returns
    }
```

---

## Mathematical Operations & Broadcasting

### 1. Vectorized Financial Calculations
```python
def vectorized_financial_operations():
    """
    Essential vectorized operations for quantitative finance
    
    CRITICAL: Always use vectorized operations instead of loops
    """
    # Sample data
    np.random.seed(42)
    prices = np.random.uniform(50, 150, (252, 100))  # 252 days, 100 assets
    
    # 1. Returns calculations (multiple methods)
    # Simple returns
    simple_returns = (prices[1:] - prices[:-1]) / prices[:-1]
    
    # Log returns (more stable for compounding)
    log_returns = np.log(prices[1:] / prices[:-1])
    
    # Percentage changes (equivalent to simple returns)
    pct_returns = np.diff(prices, axis=0) / prices[:-1]
    
    print("Returns calculation shapes:")
    print(f"Simple returns: {simple_returns.shape}")
    print(f"Log returns: {log_returns.shape}")
    
    # 2. Cumulative operations
    # Cumulative product for wealth evolution
    wealth_simple = np.cumprod(1 + simple_returns, axis=0)
    wealth_log = np.exp(np.cumsum(log_returns, axis=0))
    
    # 3. Rolling operations using convolution (efficient)
    def rolling_mean_conv(data, window):
        """Rolling mean using convolution - faster for large arrays"""
        kernel = np.ones(window) / window
        # Pad the data to handle edges
        padded = np.pad(data, (window-1, 0), mode='edge')
        return np.convolve(padded, kernel, mode='valid')
    
    # Apply to each asset
    rolling_means = np.apply_along_axis(
        lambda x: rolling_mean_conv(x, 21), axis=0, arr=simple_returns
    )
    
    return {
        'simple_returns': simple_returns,
        'log_returns': log_returns,
        'wealth_simple': wealth_simple,
        'wealth_log': wealth_log,
        'rolling_means': rolling_means
    }

def broadcasting_examples_finance():
    """
    Broadcasting examples specifically for financial applications
    """
    # Data setup
    returns = np.random.normal(0.001, 0.02, (252, 50))  # Daily returns
    weights = np.random.uniform(0, 1, 50)
    weights = weights / np.sum(weights)  # Normalize to sum to 1
    
    # 1. Portfolio returns using broadcasting
    # Method 1: Explicit multiplication and sum
    portfolio_returns_1 = np.sum(returns * weights, axis=1)
    
    # Method 2: Matrix multiplication (more efficient)
    portfolio_returns_2 = np.dot(returns, weights)
    
    # Verify they're the same
    assert np.allclose(portfolio_returns_1, portfolio_returns_2)
    
    # 2. Risk attribution using broadcasting
    # Calculate contribution of each asset to portfolio volatility
    cov_matrix = np.cov(returns.T)
    portfolio_var = np.dot(weights.T, np.dot(cov_matrix, weights))
    portfolio_vol = np.sqrt(portfolio_var)
    
    # Marginal contributions (broadcasting used implicitly)
    marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
    component_contrib = weights * marginal_contrib
    
    # 3. Scenario analysis with broadcasting
    # Stress test: shock each asset by +/- 2 standard deviations
    asset_stds = np.std(returns, axis=0)
    
    # Create shock scenarios (broadcasting)
    shock_up = returns + 2 * asset_stds    # Add 2-sigma shock
    shock_down = returns - 2 * asset_stds  # Subtract 2-sigma shock
    
    # Portfolio impact (broadcasting in action)
    portfolio_shock_up = np.dot(shock_up, weights)
    portfolio_shock_down = np.dot(shock_down, weights)
    
    return {
        'portfolio_returns': portfolio_returns_2,
        'component_contrib': component_contrib,
        'portfolio_shock_up': portfolio_shock_up,
        'portfolio_shock_down': portfolio_shock_down
    }

def element_wise_vs_matrix_operations():
    """
    When to use element-wise vs matrix operations
    """
    n_assets = 100
    n_periods = 252
    
    returns = np.random.normal(0.001, 0.02, (n_periods, n_assets))
    weights = np.random.uniform(0, 1, n_assets)
    weights = weights / np.sum(weights)
    
    # Element-wise operations (broadcasting)
    # Good for: applying same operation to each element/row/column
    
    # 1. Standardization (z-score)
    mean_returns = np.mean(returns, axis=0)  # Mean for each asset
    std_returns = np.std(returns, axis=0)    # Std for each asset
    standardized = (returns - mean_returns) / std_returns  # Broadcasting
    
    # 2. Rebalancing costs
    target_weights = np.ones(n_assets) / n_assets  # Equal weight
    weight_changes = np.abs(weights - target_weights)
    transaction_costs = weight_changes * 0.001  # 10 bps per transaction
    
    # Matrix operations
    # Good for: linear combinations, transformations
    
    # 1. Portfolio construction
    portfolio_returns = np.dot(returns, weights)
    
    # 2. Principal component analysis setup
    correlation_matrix = np.corrcoef(returns.T)
    eigenvalues, eigenvectors = np.linalg.eig(correlation_matrix)
    
    # Sort by eigenvalues (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues_sorted = eigenvalues[idx]
    eigenvectors_sorted = eigenvectors[:, idx]
    
    # First 3 principal components
    pc_loadings = eigenvectors_sorted[:, :3]
    pc_returns = np.dot(returns, pc_loadings)
    
    return {
        'standardized_returns': standardized,
        'transaction_costs': transaction_costs,
        'portfolio_returns': portfolio_returns,
        'pc_returns': pc_returns,
        'eigenvalues': eigenvalues_sorted
    }
```

### 2. Advanced Broadcasting for Risk Management
```python
def risk_management_broadcasting():
    """
    Advanced broadcasting techniques for risk management
    """
    # Setup: Multiple portfolios across different time periods
    n_portfolios = 20
    n_assets = 50
    n_periods = 252
    
    # Returns matrix: (periods, assets)
    returns = np.random.normal(0.001, 0.02, (n_periods, n_assets))
    
    # Weights matrix: (portfolios, assets) - different portfolio allocations
    weights_matrix = np.random.uniform(0, 1, (n_portfolios, n_assets))
    weights_matrix = weights_matrix / np.sum(weights_matrix, axis=1, keepdims=True)
    
    # 1. Calculate all portfolio returns simultaneously
    # Shape: (periods, portfolios)
    all_portfolio_returns = np.dot(returns, weights_matrix.T)
    
    # 2. Risk metrics for all portfolios using broadcasting
    # Volatility for each portfolio
    portfolio_vols = np.std(all_portfolio_returns, axis=0) * np.sqrt(252)
    
    # Sharpe ratios (assuming risk-free rate = 2%)
    risk_free_rate = 0.02
    excess_returns = np.mean(all_portfolio_returns, axis=0) * 252 - risk_free_rate
    sharpe_ratios = excess_returns / portfolio_vols
    
    # 3. Rolling risk metrics using advanced broadcasting
    window = 21
    n_windows = n_periods - window + 1
    
    # Create rolling windows for all portfolios simultaneously
    rolling_vols = np.zeros((n_windows, n_portfolios))
    
    for i in range(n_windows):
        window_returns = all_portfolio_returns[i:i+window, :]
        rolling_vols[i, :] = np.std(window_returns, axis=0) * np.sqrt(252)
    
    # 4. Correlation analysis with broadcasting
    # Pairwise correlations between all portfolios
    portfolio_correlations = np.corrcoef(all_portfolio_returns.T)
    
    # 5. Value at Risk calculations
    # Historical VaR at 95% and 99% confidence levels
    var_95 = np.percentile(all_portfolio_returns, 5, axis=0)
    var_99 = np.percentile(all_portfolio_returns, 1, axis=0)
    
    # Expected Shortfall (Conditional VaR)
    es_95 = np.mean(all_portfolio_returns * (all_portfolio_returns <= var_95[:, np.newaxis].T), axis=0)
    es_99 = np.mean(all_portfolio_returns * (all_portfolio_returns <= var_99[:, np.newaxis].T), axis=0)
    
    return {
        'portfolio_returns': all_portfolio_returns,
        'portfolio_vols': portfolio_vols,
        'sharpe_ratios': sharpe_ratios,
        'rolling_vols': rolling_vols,
        'var_95': var_95,
        'var_99': var_99,
        'es_95': es_95,
        'es_99': es_99
    }

def broadcasting_performance_tips():
    """
    Performance tips for broadcasting in financial applications
    """
    # Setup large dataset
    n_assets = 1000
    n_periods = 2520  # 10 years daily
    
    returns = np.random.normal(0.0005, 0.015, (n_periods, n_assets))
    
    print("Broadcasting Performance Comparison:")
    print("-" * 50)
    
    import time
    
    # 1. BAD: Using Python loops
    start = time.time()
    mean_returns_loop = []
    for i in range(n_assets):
        mean_returns_loop.append(np.mean(returns[:, i]))
    loop_time = time.time() - start
    
    # 2. GOOD: Using broadcasting
    start = time.time()
    mean_returns_broadcast = np.mean(returns, axis=0)
    broadcast_time = time.time() - start
    
    # 3. Rolling calculations comparison
    window = 21
    
    # BAD: Nested loops
    start = time.time()
    rolling_std_loop = np.zeros((n_periods - window + 1, n_assets))
    for i in range(n_periods - window + 1):
        for j in range(n_assets):
            rolling_std_loop[i, j] = np.std(returns[i:i+window, j])
    nested_loop_time = time.time() - start
    
    # GOOD: Vectorized with list comprehension
    start = time.time()
    rolling_std_vectorized = np.array([
        np.std(returns[i:i+window, :], axis=0) 
        for i in range(n_periods - window + 1)
    ])
    vectorized_time = time.time() - start
    
    print(f"Mean calculation - Loop: {loop_time:.4f}s, Broadcast: {broadcast_time:.4f}s")
    print(f"Speedup: {loop_time/broadcast_time:.1f}x")
    print(f"Rolling std - Nested loops: {nested_loop_time:.4f}s, Vectorized: {vectorized_time:.4f}s")
    print(f"Speedup: {nested_loop_time/vectorized_time:.1f}x")
    
    # Verify results are the same
    assert np.allclose(mean_returns_loop, mean_returns_broadcast)
    assert np.allclose(rolling_std_loop, rolling_std_vectorized)
    
    return {
        'broadcast_speedup': loop_time/broadcast_time,
        'vectorized_speedup': nested_loop_time/vectorized_time
    }
```

---

## Linear Algebra for Finance

### 1. Portfolio Optimization with Linear Algebra
```python
def portfolio_optimization_linear_algebra():
    """
    Modern Portfolio Theory using NumPy's linear algebra functions
    """
    # Sample data: 10 assets, 252 days
    np.random.seed(42)
    n_assets = 10
    n_periods = 252
    
    # Generate correlated returns
    correlation_matrix = np.random.uniform(0.1, 0.7, (n_assets, n_assets))
    correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2  # Make symmetric
    np.fill_diagonal(correlation_matrix, 1.0)  # Diagonal = 1
    
    # Ensure positive semi-definite
    eigenvals, eigenvecs = np.linalg.eigh(correlation_matrix)
    eigenvals = np.maximum(eigenvals, 0.01)  # Floor eigenvalues
    correlation_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
    
    # Generate returns
    volatilities = np.random.uniform(0.1, 0.3, n_assets)
    expected_returns = np.random.uniform(0.05, 0.15, n_assets)
    
    # Covariance matrix
    std_matrix = np.outer(volatilities, volatilities)
    covariance_matrix = correlation_matrix * std_matrix
    
    # 1. Minimum Variance Portfolio
    # Solve: minimize w'Σw subject to w'1 = 1
    # Solution: w = (Σ^-1 * 1) / (1'Σ^-1 * 1)
    
    try:
        inv_cov = np.linalg.inv(covariance_matrix)
    except np.linalg.LinAlgError:
        # Use pseudo-inverse if singular
        inv_cov = np.linalg.pinv(covariance_matrix)
        print("Warning: Using pseudo-inverse for singular covariance matrix")
    
    ones = np.ones(n_assets)
    min_var_weights = inv_cov @ ones / (ones.T @ inv_cov @ ones)
    
    # 2. Maximum Sharpe Ratio Portfolio
    # Solve: maximize (μ'w - rf) / sqrt(w'Σw)
    risk_free_rate = 0.02
    excess_returns = expected_returns - risk_free_rate
    
    # Solution: w ∝ Σ^-1 * (μ - rf*1)
    tangency_weights_unnorm = inv_cov @ excess_returns
    tangency_weights = tangency_weights_unnorm / np.sum(tangency_weights_unnorm)
    
    # 3. Calculate portfolio metrics
    def portfolio_metrics(weights, exp_ret, cov_mat):
        """Calculate portfolio expected return, volatility, and Sharpe ratio"""
        port_return = np.dot(weights, exp_ret)
        port_var = np.dot(weights.T, np.dot(cov_mat, weights))
        port_vol = np.sqrt(port_var)
        sharpe = (port_return - risk_free_rate) / port_vol
        return port_return, port_vol, sharpe
    
    # Metrics for both portfolios
    minvar_return, minvar_vol, minvar_sharpe = portfolio_metrics(
        min_var_weights, expected_returns, covariance_matrix
    )
    
    tangency_return, tangency_vol, tangency_sharpe = portfolio_metrics(
        tangency_weights, expected_returns, covariance_matrix
    )
    
    # 4. Efficient Frontier
    # Generate points on efficient frontier
    n_points = 50
    min_return = min(expected_returns)
    max_return = max(expected_returns)
    target_returns = np.linspace(min_return, max_return, n_points)
    
    efficient_vols = []
    efficient_weights = []
    
    for target_return in target_returns:
        # Solve: minimize w'Σw subject to w'μ = target_return, w'1 = 1
        # Using method of Lagrange multipliers
        
        A = np.vstack([expected_returns, ones])  # Constraints matrix
        b = np.array([target_return, 1.0])       # Constraints vector
        
        try:
            # Solve the system
            C = A @ inv_cov @ A.T
            lambda_vec = np.linalg.solve(C, b)
            weights = inv_cov @ A.T @ lambda_vec
            
            portfolio_vol = np.sqrt(weights.T @ covariance_matrix @ weights)
            efficient_vols.append(portfolio_vol)
            efficient_weights.append(weights)
            
        except np.linalg.LinAlgError:
            # Skip this point if numerical issues
            efficient_vols.append(np.nan)
            efficient_weights.append(np.full(n_assets, np.nan))
    
    efficient_vols = np.array(efficient_vols)
    efficient_weights = np.array(efficient_weights)
    
    return {
        'min_var_weights': min_var_weights,
        'tangency_weights': tangency_weights,
        'min_var_metrics': (minvar_return, minvar_vol, minvar_sharpe),
        'tangency_metrics': (tangency_return, tangency_vol, tangency_sharpe),
        'efficient_frontier': (target_returns, efficient_vols),
        'covariance_matrix': covariance_matrix,
        'expected_returns': expected_returns
    }

def factor_model_linear_algebra():
    """
    Factor models using linear algebra operations
    """
    # Setup: 3-factor model (Market, Size, Value)
    np.random.seed(42)
    n_assets = 20
    n_periods = 252
    n_factors = 3
    
    # Generate factor returns
    factor_names = ['Market', 'Size', 'Value']
    factor_returns = np.random.multivariate_normal(
        mean=[0.01, 0.005, 0.003],  # Monthly factor returns
        cov=[[0.04, 0.01, 0.005],   # Factor covariance
             [0.01, 0.02, 0.002],
             [0.005, 0.002, 0.015]],
        size=n_periods
    )
    
    # Generate factor loadings (betas)
    factor_loadings = np.random.normal(0, 1, (n_assets, n_factors))
    factor_loadings[:, 0] = np.abs(factor_loadings[:, 0])  # Market beta > 0
    
    # Generate asset returns using factor model
    # R = α + β*F + ε
    alphas = np.random.normal(0, 0.001, n_assets)  # Small alphas
    idiosyncratic_vol = np.random.uniform(0.01, 0.03, n_assets)
    
    asset_returns = np.zeros((n_periods, n_assets))
    for t in range(n_periods):
        systematic_returns = factor_loadings @ factor_returns[t]
        idiosyncratic_returns = np.random.normal(0, idiosyncratic_vol)
        asset_returns[t] = alphas + systematic_returns + idiosyncratic_returns
    
    # 1. Estimate factor model using OLS (matrix form)
    # R = F*β' + ε  =>  β = (F'F)^-1 * F'R
    
    # Add intercept to factor matrix
    F = np.column_stack([np.ones(n_periods), factor_returns])  # [1, Market, Size, Value]
    
    # OLS estimation
    F_transpose_F_inv = np.linalg.inv(F.T @ F)
    estimated_coefficients = F_transpose_F_inv @ F.T @ asset_returns
    
    # Extract alphas and betas
    estimated_alphas = estimated_coefficients[0]
    estimated_betas = estimated_coefficients[1:]
    
    # 2. Model diagnostics
    # R-squared for each asset
    predicted_returns = F @ estimated_coefficients
    residuals = asset_returns - predicted_returns
    
    tss = np.sum((asset_returns - np.mean(asset_returns, axis=0))**2, axis=0)
    rss = np.sum(residuals**2, axis=0)
    r_squared = 1 - rss / tss
    
    # 3. Factor portfolio construction
    # Create factor-mimicking portfolios
    factor_portfolios = np.zeros((n_periods, n_factors))
    
    for f in range(n_factors):
        # Sort assets by factor loading
        factor_loading = estimated_betas[f]
        sorted_indices = np.argsort(factor_loading)
        
        # Long-short portfolio: long top decile, short bottom decile
        n_decile = n_assets // 10
        long_assets = sorted_indices[-n_decile:]
        short_assets = sorted_indices[:n_decile]
        
        # Equal weight within each decile
        long_weight = 1.0 / n_decile
        short_weight = -1.0 / n_decile
        
        portfolio_weights = np.zeros(n_assets)
        portfolio_weights[long_assets] = long_weight
        portfolio_weights[short_assets] = short_weight
        
        # Calculate portfolio returns
        factor_portfolios[:, f] = asset_returns @ portfolio_weights
    
    return {
        'true_factor_loadings': factor_loadings,
        'estimated_alphas': estimated_alphas,
        'estimated_betas': estimated_betas.T,  # Transpose for asset x factor
        'true_alphas': alphas,
        'r_squared': r_squared,
        'factor_portfolios': factor_portfolios,
        'factor_returns': factor_returns,
        'asset_returns': asset_returns
    }
```

### 2. Risk Decomposition and Attribution
```python
def risk_decomposition_linear_algebra():
    """
    Portfolio risk decomposition using linear algebra
    """
    # Setup
    np.random.seed(42)
    n_assets = 15
    n_periods = 252
    
    # Generate returns and covariance
    returns = np.random.multivariate_normal(
        mean=np.random.uniform(0.05, 0.12, n_assets),
        cov=np.random.uniform(0.1, 0.4, (n_assets, n_assets)),
        size=n_periods
    )
    
    # Portfolio weights
    weights = np.random.uniform(0, 1, n_assets)
    weights = weights / np.sum(weights)  # Normalize
    
    # Covariance matrix
    cov_matrix = np.cov(returns.T)
    
    # 1. Portfolio Risk Decomposition
    # Total portfolio variance: σ²_p = w'Σw
    portfolio_variance = weights.T @ cov_matrix @ weights
    portfolio_volatility = np.sqrt(portfolio_variance)
    
    # 2. Marginal Contribution to Risk (MCR)
    # MCR_i = ∂σ_p/∂w_i = (Σw)_i / σ_p
    marginal_contrib = (cov_matrix @ weights) / portfolio_volatility
    
    # 3. Component Contribution to Risk (CCR)
    # CCR_i = w_i * MCR_i
    component_contrib = weights * marginal_contrib
    
    # 4. Percentage Contribution to Risk
    pct_contrib = component_contrib / portfolio_volatility
    
    # Verify: sum of component contributions = portfolio volatility
    assert np.isclose(np.sum(component_contrib), portfolio_volatility)
    assert np.isclose(np.sum(pct_contrib), 1.0)
    
    # 5. Factor Risk Decomposition (using PCA)
    # Principal Component Analysis
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort by eigenvalues (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues_sorted = eigenvalues[idx]
    eigenvectors_sorted = eigenvectors[:, idx]
    
    # Explained variance ratio
    explained_var_ratio = eigenvalues_sorted / np.sum(eigenvalues_sorted)
    cumulative_var_ratio = np.cumsum(explained_var_ratio)
    
    # Transform portfolio to factor space
    factor_weights = eigenvectors_sorted.T @ weights
    factor_variances = eigenvalues_sorted * factor_weights**2
    
    # 6. Systematic vs Idiosyncratic Risk
    # Use first k factors as systematic risk
    n_systematic_factors = 5
    systematic_var = np.sum(factor_variances[:n_systematic_factors])
    idiosyncratic_var = np.sum(factor_variances[n_systematic_factors:])
    
    systematic_vol = np.sqrt(systematic_var)
    idiosyncratic_vol = np.sqrt(idiosyncratic_var)
    
    # 7. Sector Risk Analysis (assuming sectors)
    # Example: assets 0-4 are tech, 5-9 are finance, 10-14 are healthcare
    sector_indices = {
        'Tech': np.arange(0, 5),
        'Finance': np.arange(5, 10),
        'Healthcare': np.arange(10, 15)
    }
    
    sector_risk_contrib = {}
    for sector, indices in sector_indices.items():
        sector_weights = np.zeros(n_assets)
        sector_weights[indices] = weights[indices]
        
        # Sector contribution to portfolio risk
        sector_contrib = sector_weights.T @ cov_matrix @ weights / portfolio_volatility
        sector_risk_contrib[sector] = sector_contrib
    
    return {
        'portfolio_volatility': portfolio_volatility,
        'marginal_contrib': marginal_contrib,
        'component_contrib': component_contrib,
        'pct_contrib': pct_contrib,
        'eigenvalues': eigenvalues_sorted,
        'explained_var_ratio': explained_var_ratio,
        'systematic_vol': systematic_vol,
        'idiosyncratic_vol': idiosyncratic_vol,
        'sector_risk_contrib': sector_risk_contrib
    }

def advanced_linear_algebra_techniques():
    """
    Advanced linear algebra techniques for quantitative finance
    """
    np.random.seed(42)
    n_assets = 50
    n_periods = 1000
    
    # Generate realistic covariance structure
    # Use factor model to ensure positive semi-definite matrix
    n_factors = 5
    factor_loadings = np.random.normal(0, 1, (n_assets, n_factors))
    idiosyncratic_var = np.random.uniform(0.01, 0.04, n_assets)
    
    # Covariance matrix: Σ = ΛΛ' + D (where D is diagonal)
    covariance_matrix = (factor_loadings @ factor_loadings.T + 
                        np.diag(idiosyncratic_var))
    
    # 1. Matrix Condition Number Analysis
    condition_number = np.linalg.cond(covariance_matrix)
    print(f"Covariance matrix condition number: {condition_number:.2f}")
    
    if condition_number > 1e12:
        print("Warning: Matrix is ill-conditioned, consider regularization")
    
    # 2. Regularization Techniques
    # Ridge regularization
    lambda_ridge = 0.01
    regularized_cov = covariance_matrix + lambda_ridge * np.eye(n_assets)
    
    # Ledoit-Wolf shrinkage
    def ledoit_wolf_shrinkage(sample_cov):
        """Simplified Ledoit-Wolf shrinkage estimator"""
        n_samples = n_periods
        trace = np.trace(sample_cov)
        
        # Shrinkage target (identity scaled by average variance)
        target = (trace / n_assets) * np.eye(n_assets)
        
        # Optimal shrinkage intensity (simplified formula)
        shrinkage_intensity = min(1.0, 1.0 / n_samples)
        
        return (1 - shrinkage_intensity) * sample_cov + shrinkage_intensity * target
    
    shrunk_cov = ledoit_wolf_shrinkage(covariance_matrix)
    
    # 3. Robust Covariance Estimation
    # Generate sample data with outliers
    returns = np.random.multivariate_normal(
        mean=np.zeros(n_assets), 
        cov=covariance_matrix, 
        size=n_periods
    )
    
    # Add some outliers
    n_outliers = int(0.05 * n_periods)  # 5% outliers
    outlier_indices = np.random.choice(n_periods, n_outliers, replace=False)
    returns[outlier_indices] *= 5  # Make outliers 5x larger
    
    # Sample covariance (sensitive to outliers)
    sample_cov = np.cov(returns.T)
    
    # Robust covariance using Minimum Covariance Determinant (simplified)
    def robust_covariance_mcd(returns, support_fraction=0.8):
        """Simplified MCD estimator"""
        n_samples = len(returns)
        n_support = int(support_fraction * n_samples)
        
        # Find subset with minimum determinant
        best_det = np.inf
        best_indices = None
        
        # Random search (in practice, use more sophisticated algorithm)
        for _ in range(100):
            indices = np.random.choice(n_samples, n_support, replace=False)
            subset_cov = np.cov(returns[indices].T)
            
            try:
                det = np.linalg.det(subset_cov)
                if det < best_det and det > 0:
                    best_det = det
                    best_indices = indices
            except:
                continue
        
        if best_indices is not None:
            return np.cov(returns[best_indices].T)
        else:
            return sample_cov  # Fallback
    
    robust_cov = robust_covariance_mcd(returns)
    
    # 4. Low-rank approximation using SVD
    def low_rank_approximation(matrix, rank):
        """Low-rank approximation using SVD"""
        U, s, Vt = np.linalg.svd(matrix)
        
        # Keep only top 'rank' components
        U_reduced = U[:, :rank]
        s_reduced = s[:rank]
        Vt_reduced = Vt[:rank, :]
        
        # Reconstruct
        approx_matrix = U_reduced @ np.diag(s_reduced) @ Vt_reduced
        
        # Explained variance
        explained_var = np.sum(s_reduced**2) / np.sum(s**2)
        
        return approx_matrix, explained_var
    
    # Approximate covariance with rank-10 matrix
    low_rank_cov, explained_var = low_rank_approximation(covariance_matrix, rank=10)
    
    # 5. Sparse inverse covariance (Graphical Lasso concept)
    def sparse_inverse_regularized(cov_matrix, lambda_sparse):
        """Regularized inverse with sparsity penalty (simplified)"""
        # Add L1 penalty to encourage sparsity in inverse
        # This is a simplified version - use sklearn.covariance.GraphicalLasso in practice
        
        precision_matrix = np.linalg.inv(cov_matrix + 
                                       lambda_sparse * np.eye(len(cov_matrix)))
        
        # Soft thresholding for sparsity
        threshold = lambda_sparse * 0.1
        precision_matrix[np.abs(precision_matrix) < threshold] = 0
        
        return precision_matrix
    
    sparse_precision = sparse_inverse_regularized(covariance_matrix, 0.05)
    
    return {
        'condition_number': condition_number,
        'regularized_cov': regularized_cov,
        'shrunk_cov': shrunk_cov,
        'sample_cov': sample_cov,
        'robust_cov': robust_cov,
        'low_rank_cov': low_rank_cov,
        'explained_var_low_rank': explained_var,
        'sparse_precision': sparse_precision
    }
```

---

## Random Number Generation & Monte Carlo

### 1. Random Number Generation for Finance
```python
def financial_random_number_setup():
    """
    Proper random number generation setup for financial simulations
    
    CRITICAL: Always set seeds for reproducible research
    """
    # 1. Global seed setting
    np.random.seed(42)  # For NumPy
    
    # 2. Create RandomState for isolated randomness
    rng = np.random.RandomState(42)  # Isolated random state
    
    # 3. Modern approach: use Generator (NumPy 1.17+)
    from numpy.random import default_rng
    generator = default_rng(42)  # More flexible and faster
    
    print("Random number generators initialized with seed=42")
    
    return rng, generator

def monte_carlo_stock_price_simulation():
    """
    Monte Carlo simulation for stock price paths using different models
    """
    # Parameters
    S0 = 100.0          # Initial stock price
    T = 1.0             # Time to maturity (1 year)
    r = 0.05            # Risk-free rate
    sigma = 0.2         # Volatility
    n_steps = 252       # Number of time steps (daily)
    n_paths = 10000     # Number of simulation paths
    
    dt = T / n_steps
    rng = np.random.RandomState(42)
    
    # 1. Geometric Brownian Motion (Black-Scholes)
    # dS = rS*dt + σS*dW
    
    # Method 1: Direct simulation
    def gbm_simulation_direct():
        """Direct GBM simulation"""
        prices = np.zeros((n_steps + 1, n_paths))
        prices[0] = S0
        
        for t in range(1, n_steps + 1):
            Z = rng.standard_normal(n_paths)  # Random normal draws
            prices[t] = prices[t-1] * np.exp((r - 0.5 * sigma**2) * dt + 
                                           sigma * np.sqrt(dt) * Z)
        return prices
    
    # Method 2: Vectorized simulation (faster)
    def gbm_simulation_vectorized():
        """Vectorized GBM simulation"""
        # Generate all random numbers at once
        Z = rng.standard_normal((n_steps, n_paths))
        
        # Calculate log returns
        log_returns = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        
        # Convert to prices
        log_prices = np.log(S0) + np.cumsum(log_returns, axis=0)
        prices = np.exp(log_prices)
        
        # Add initial price
        prices = np.vstack([np.full(n_paths, S0), prices])
        
        return prices
    
    # Method 3: Antithetic variates (variance reduction)
    def gbm_simulation_antithetic():
        """GBM with antithetic variates for variance reduction"""
        half_paths = n_paths // 2
        Z = rng.standard_normal((n_steps, half_paths))
        Z_anti = -Z  # Antithetic variates
        
        # Combine normal and antithetic
        Z_combined = np.hstack([Z, Z_anti])
        
        log_returns = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z_combined
        log_prices = np.log(S0) + np.cumsum(log_returns, axis=0)
        prices = np.exp(log_prices)
        prices = np.vstack([np.full(n_paths, S0), prices])
        
        return prices
    
    # Run simulations
    import time
    
    start = time.time()
    prices_direct = gbm_simulation_direct()
    time_direct = time.time() - start
    
    start = time.time()
    prices_vectorized = gbm_simulation_vectorized()
    time_vectorized = time.time() - start
    
    start = time.time()
    prices_antithetic = gbm_simulation_antithetic()
    time_antithetic = time.time() - start
    
    print(f"Simulation Times:")
    print(f"Direct: {time_direct:.4f}s")
    print(f"Vectorized: {time_vectorized:.4f}s (Speedup: {time_direct/time_vectorized:.1f}x)")
    print(f"Antithetic: {time_antithetic:.4f}s")
    
    # 2. Jump Diffusion Model (Merton)
    def jump_diffusion_simulation():
        """Merton jump diffusion model simulation"""
        # Additional parameters for jumps
        lambda_j = 0.1      # Jump intensity (jumps per year)
        mu_j = -0.05        # Mean jump size
        sigma_j = 0.15      # Jump volatility
        
        prices = np.zeros((n_steps + 1, n_paths))
        prices[0] = S0
        
        for t in range(1, n_steps + 1):
            # Brownian motion component
            Z_diffusion = rng.standard_normal(n_paths)
            diffusion_return = (r - 0.5 * sigma**2 - lambda_j * 
                              (np.exp(mu_j + 0.5 * sigma_j**2) - 1)) * dt
            
            # Jump component
            N_jumps = rng.poisson(lambda_j * dt, n_paths)  # Number of jumps
            jump_return = np.zeros(n_paths)
            
            for i in range(n_paths):
                if N_jumps[i] > 0:
                    # Sum of jump sizes
                    jump_sizes = rng.normal(mu_j, sigma_j, N_jumps[i])
                    jump_return[i] = np.sum(jump_sizes)
            
            # Total return
            total_return = (diffusion_return + 
                          sigma * np.sqrt(dt) * Z_diffusion + 
                          jump_return)
            
            prices[t] = prices[t-1] * np.exp(total_return)
        
        return prices
    
    prices_jump_diffusion = jump_diffusion_simulation()
    
    return {
        'gbm_prices': prices_vectorized,
        'antithetic_prices': prices_antithetic,
        'jump_diffusion_prices': prices_jump_diffusion,
        'simulation_times': {
            'direct': time_direct,
            'vectorized': time_vectorized,
            'antithetic': time_antithetic
        }
    }

def monte_carlo_option_pricing():
    """
    Monte Carlo option pricing with Greeks calculation
    """
    # Option parameters
    S0 = 100.0      # Current stock price
    K = 105.0       # Strike price
    T = 0.25        # Time to expiration (3 months)
    r = 0.05        # Risk-free rate
    sigma = 0.2     # Volatility
    n_paths = 100000
    
    rng = np.random.RandomState(42)
    
    # 1. European Option Pricing
    def european_option_mc():
        """Monte Carlo pricing for European options"""
        # Generate terminal stock prices
        Z = rng.standard_normal(n_paths)
        ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
        
        # Payoffs
        call_payoffs = np.maximum(ST - K, 0)
        put_payoffs = np.maximum(K - ST, 0)
        
        # Present values
        call_price = np.exp(-r * T) * np.mean(call_payoffs)
        put_price = np.exp(-r * T) * np.mean(put_payoffs)
        
        # Standard errors
        call_se = np.exp(-r * T) * np.std(call_payoffs) / np.sqrt(n_paths)
        put_se = np.exp(-r * T) * np.std(put_payoffs) / np.sqrt(n_paths)
        
        return {
            'call_price': call_price,
            'put_price': put_price,
            'call_se': call_se,
            'put_se': put_se,
            'terminal_prices': ST
        }
    
    # 2. Greek Calculation using Finite Differences
    def calculate_greeks_fd():
        """Calculate Greeks using finite difference method"""
        bump_size = 0.01  # 1% bump
        
        # Base case
        base_result = european_option_mc()
        base_call = base_result['call_price']
        
        # Delta: ∂V/∂S
        S0_up = S0 * (1 + bump_size)
        Z = rng.standard_normal(n_paths)
        ST_up = S0_up * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
        call_payoffs_up = np.maximum(ST_up - K, 0)
        call_price_up = np.exp(-r * T) * np.mean(call_payoffs_up)
        
        delta = (call_price_up - base_call) / (S0_up - S0)
        
        # Gamma: ∂²V/∂S²
        S0_down = S0 * (1 - bump_size)
        ST_down = S0_down * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
        call_payoffs_down = np.maximum(ST_down - K, 0)
        call_price_down = np.exp(-r * T) * np.mean(call_payoffs_down)
        
        gamma = (call_price_up - 2 * base_call + call_price_down) / (bump_size * S0)**2
        
        # Theta: ∂V/∂t (using T - dt)
        T_down = T - 1/252  # One day less
        ST_theta = S0 * np.exp((r - 0.5 * sigma**2) * T_down + 
                              sigma * np.sqrt(T_down) * Z)
        call_payoffs_theta = np.maximum(ST_theta - K, 0)
        call_price_theta = np.exp(-r * T_down) * np.mean(call_payoffs_theta)
        
        theta = (call_price_theta - base_call) / (1/252)
        
        # Vega: ∂V/∂σ
        sigma_up = sigma * (1 + bump_size)
        ST_vega = S0 * np.exp((r - 0.5 * sigma_up**2) * T + 
                             sigma_up * np.sqrt(T) * Z)
        call_payoffs_vega = np.maximum(ST_vega - K, 0)
        call_price_vega = np.exp(-r * T) * np.mean(call_payoffs_vega)
        
        vega = (call_price_vega - base_call) / (sigma_up - sigma)
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega
        }
    
    # 3. Pathwise Greeks (more efficient)
    def calculate_greeks_pathwise():
        """Calculate Greeks using pathwise derivatives"""
        Z = rng.standard_normal(n_paths)
        ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
        
        # Call option indicators (1 if in-the-money, 0 otherwise)
        itm_indicator = (ST > K).astype(float)
        
        # Delta using pathwise method
        delta_pathwise = np.exp(-r * T) * np.mean(itm_indicator * ST / S0)
        
        # Vega using pathwise method
        vega_pathwise = np.exp(-r * T) * np.mean(
            itm_indicator * ST * (Z * np.sqrt(T) - sigma * T)
        )
        
        return {
            'delta_pathwise': delta_pathwise,
            'vega_pathwise': vega_pathwise
        }
    
    # Run calculations
    option_results = european_option_mc()
    greeks_fd = calculate_greeks_fd()
    greeks_pathwise = calculate_greeks_pathwise()
    
    # Compare with Black-Scholes (for validation)
    from scipy.stats import norm
    
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    bs_call = S0 * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    bs_delta = norm.cdf(d1)
    
    print(f"Option Pricing Comparison:")
    print(f"Monte Carlo Call Price: {option_results['call_price']:.4f} ± {option_results['call_se']:.4f}")
    print(f"Black-Scholes Call Price: {bs_call:.4f}")
    print(f"MC Delta: {greeks_fd['delta']:.4f}")
    print(f"Pathwise Delta: {greeks_pathwise['delta_pathwise']:.4f}")
    print(f"BS Delta: {bs_delta:.4f}")
    
    return {
        'option_prices': option_results,
        'greeks_fd': greeks_fd,
        'greeks_pathwise': greeks_pathwise,
        'black_scholes': {'call': bs_call, 'delta': bs_delta}
    }
```

### 2. Advanced Monte Carlo Techniques
```python
def variance_reduction_techniques():
    """
    Advanced variance reduction techniques for Monte Carlo
    """
    # Setup: Asian option pricing (path-dependent)
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.2
    n_steps = 12  # Monthly averaging
    n_paths = 50000
    
    dt = T / n_steps
    rng = np.random.RandomState(42)
    
    # 1. Crude Monte Carlo (baseline)
    def crude_monte_carlo():
        """Standard Monte Carlo without variance reduction"""
        payoffs = np.zeros(n_paths)
        
        for i in range(n_paths):
            # Generate price path
            prices = [S0]
            for t in range(n_steps):
                Z = rng.standard_normal()
                S_new = prices[-1] * np.exp((r - 0.5 * sigma**2) * dt + 
                                          sigma * np.sqrt(dt) * Z)
                prices.append(S_new)
            
            # Asian option payoff (arithmetic average)
            avg_price = np.mean(prices[1:])  # Exclude initial price
            payoffs[i] = max(avg_price - K, 0)
        
        option_value = np.exp(-r * T) * np.mean(payoffs)
        standard_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_paths)
        
        return option_value, standard_error
    
    # 2. Antithetic Variates
    def antithetic_variates():
        """Monte Carlo with antithetic variates"""
        half_paths = n_paths // 2
        payoffs = np.zeros(n_paths)
        
        for i in range(half_paths):
            # Generate random numbers
            Z = rng.standard_normal(n_steps)
            Z_anti = -Z  # Antithetic variates
            
            # Simulate both paths
            for j, random_nums in enumerate([Z, Z_anti]):
                prices = [S0]
                for t in range(n_steps):
                    S_new = prices[-1] * np.exp((r - 0.5 * sigma**2) * dt + 
                                              sigma * np.sqrt(dt) * random_nums[t])
                    prices.append(S_new)
                
                avg_price = np.mean(prices[1:])
                payoffs[i * 2 + j] = max(avg_price - K, 0)
        
        option_value = np.exp(-r * T) * np.mean(payoffs)
        standard_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_paths)
        
        return option_value, standard_error
    
    # 3. Control Variates
    def control_variates():
        """Monte Carlo with control variates"""
        payoffs_asian = np.zeros(n_paths)
        payoffs_european = np.zeros(n_paths)  # Control variate
        
        for i in range(n_paths):
            prices = [S0]
            for t in range(n_steps):
                Z = rng.standard_normal()
                S_new = prices[-1] * np.exp((r - 0.5 * sigma**2) * dt + 
                                          sigma * np.sqrt(dt) * Z)
                prices.append(S_new)
            
            # Asian option payoff
            avg_price = np.mean(prices[1:])
            payoffs_asian[i] = max(avg_price - K, 0)
            
            # European option payoff (control variate)
            payoffs_european[i] = max(prices[-1] - K, 0)
        
        # Black-Scholes value for European option (known)
        from scipy.stats import norm
        d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        bs_european = S0 * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
        
        # Control variate adjustment
        beta = np.cov(payoffs_asian, payoffs_european)[0,1] / np.var(payoffs_european)
        
        european_mc = np.mean(payoffs_european)
        adjusted_asian = payoffs_asian - beta * (payoffs_european - european_mc)
        
        option_value = np.exp(-r * T) * np.mean(adjusted_asian)
        standard_error = np.exp(-r * T) * np.std(adjusted_asian) / np.sqrt(n_paths)
        
        variance_reduction = np.var(payoffs_asian) / np.var(adjusted_asian)
        
        return option_value, standard_error, variance_reduction
    
    # 4. Importance Sampling
    def importance_sampling():
        """Monte Carlo with importance sampling"""
        # Shift the distribution to focus on in-the-money outcomes
        theta = 0.1  # Drift adjustment
        payoffs = np.zeros(n_paths)
        likelihood_ratios = np.zeros(n_paths)
        
        for i in range(n_paths):
            # Generate path under shifted measure
            prices = [S0]
            log_likelihood_ratio = 0
            
            for t in range(n_steps):
                Z = rng.standard_normal()
                Z_shifted = Z + theta * np.sqrt(dt)  # Importance sampling adjustment
                
                S_new = prices[-1] * np.exp((r - 0.5 * sigma**2) * dt + 
                                          sigma * np.sqrt(dt) * Z_shifted)
                prices.append(S_new)
                
                # Update likelihood ratio
                log_likelihood_ratio += -theta * Z * np.sqrt(dt) - 0.5 * theta**2 * dt
            
            avg_price = np.mean(prices[1:])
            payoffs[i] = max(avg_price - K, 0)
            likelihood_ratios[i] = np.exp(log_likelihood_ratio)
        
        # Weighted average
        weighted_payoffs = payoffs * likelihood_ratios
        option_value = np.exp(-r * T) * np.mean(weighted_payoffs)
        
        # Standard error calculation for importance sampling
        variance_is = np.var(weighted_payoffs)
        standard_error = np.exp(-r * T) * np.sqrt(variance_is / n_paths)
        
        return option_value, standard_error
    
    # Run all methods
    import time
    
    methods = [
        ("Crude MC", crude_monte_carlo),
        ("Antithetic", antithetic_variates),
        ("Control Variates", control_variates),
        ("Importance Sampling", importance_sampling)
    ]
    
    results = {}
    
    print("Asian Option Pricing - Variance Reduction Comparison:")
    print("-" * 60)
    
    for name, method in methods:
        start_time = time.time()
        
        if name == "Control Variates":
            value, se, var_reduction = method()
            compute_time = time.time() - start_time
            results[name] = {
                'value': value, 
                'se': se, 
                'time': compute_time,
                'var_reduction': var_reduction
            }
            print(f"{name:20}: {value:.4f} ± {se:.4f} (VR: {var_reduction:.2f}x) [{compute_time:.3f}s]")
        else:
            value, se = method()
            compute_time = time.time() - start_time
            results[name] = {'value': value, 'se': se, 'time': compute_time}
            print(f"{name:20}: {value:.4f} ± {se:.4f} [{compute_time:.3f}s]")
    
    return results

def quasi_monte_carlo():
    """
    Quasi-Monte Carlo using low-discrepancy sequences
    """
    # For high-dimensional problems, QMC can be more efficient
    
    # 1. Sobol sequence generation (simplified version)
    def sobol_sequence(n_dims, n_points):
        """
        Generate Sobol sequence (simplified implementation)
        In practice, use scipy.stats.qmc.Sobol
        """
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=n_dims, scramble=True)
            return sampler.random(n_points)
        except ImportError:
            # Fallback to pseudo-random if scipy not available
            print("Warning: Using pseudo-random instead of Sobol")
            return np.random.uniform(0, 1, (n_points, n_dims))
    
    # 2. Multi-asset option pricing with QMC
    def basket_option_qmc():
        """Price basket option using Quasi-Monte Carlo"""
        # Parameters
        n_assets = 5
        S0 = np.full(n_assets, 100.0)  # Initial prices
        K = 100.0                       # Strike
        T = 1.0                        # Maturity
        r = 0.05                       # Risk-free rate
        
        # Correlation matrix
        rho = 0.3
        corr_matrix = np.full((n_assets, n_assets), rho)
        np.fill_diagonal(corr_matrix, 1.0)
        
        # Volatilities
        sigmas = np.array([0.15, 0.20, 0.25, 0.18, 0.22])
        
        # Covariance matrix
        vol_matrix = np.outer(sigmas, sigmas)
        cov_matrix = corr_matrix * vol_matrix
        
        # Cholesky decomposition for correlation
        L = np.linalg.cholesky(corr_matrix)
        
        n_paths = 50000
        
        # Generate Sobol sequence
        uniform_randoms = sobol_sequence(n_assets, n_paths)
        
        # Convert to standard normal
        from scipy.stats import norm
        standard_normals = norm.ppf(uniform_randoms)
        
        # Apply correlation structure
        correlated_normals = standard_normals @ L.T
        
        # Generate terminal stock prices
        terminal_prices = np.zeros((n_paths, n_assets))
        for i in range(n_assets):
            terminal_prices[:, i] = S0[i] * np.exp(
                (r - 0.5 * sigmas[i]**2) * T + 
                sigmas[i] * np.sqrt(T) * correlated_normals[:, i]
            )
        
        # Basket option payoff (arithmetic average of assets)
        basket_values = np.mean(terminal_prices, axis=1)
        payoffs = np.maximum(basket_values - K, 0)
        
        # Option value
        option_value = np.exp(-r * T) * np.mean(payoffs)
        standard_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_paths)
        
        return {
            'option_value': option_value,
            'standard_error': standard_error,
            'terminal_prices': terminal_prices
        }
    
    # 3. Compare QMC vs Standard MC
    def compare_qmc_mc():
        """Compare Quasi-Monte Carlo vs regular Monte Carlo convergence"""
        n_runs = 10
        sample_sizes = [1000, 5000, 10000, 25000, 50000]
        
        qmc_results = []
        mc_results = []
        
        for n_samples in sample_sizes:
            qmc_values = []
            mc_values = []
            
            for run in range(n_runs):
                # QMC run
                np.random.seed(run)  # Different seed for each run
                qmc_result = basket_option_qmc()  # Uses QMC internally
                qmc_values.append(qmc_result['option_value'])
                
                # Standard MC run
                np.random.seed(run)
                # Regular MC would go here (simplified for example)
                mc_values.append(qmc_result['option_value'] + 
                               np.random.normal(0, 0.1))  # Add noise for comparison
            
            qmc_results.append({
                'n_samples': n_samples,
                'mean': np.mean(qmc_values),
                'std': np.std(qmc_values)
            })
            
            mc_results.append({
                'n_samples': n_samples,
                'mean': np.mean(mc_values),
                'std': np.std(mc_values)
            })
        
        return qmc_results, mc_results
    
    # Run QMC example
    basket_result = basket_option_qmc()
    convergence_comparison = compare_qmc_mc()
    
    print(f"Basket Option QMC Result: {basket_result['option_value']:.4f} ± {basket_result['standard_error']:.4f}")
    
    return {
        'basket_option': basket_result,
        'convergence_comparison': convergence_comparison
    }
```

---

## Performance Optimization

### 1. Vectorization vs Loops
```python
def vectorization_performance_examples():
    """
    Performance comparison: vectorized operations vs loops
    
    CRITICAL: Always prefer vectorized operations in NumPy
    """
    import time
    
    # Large dataset for meaningful benchmarks
    n_assets = 1000
    n_periods = 2520  # 10 years daily
    np.random.seed(42)
    
    # Generate test data
    prices = np.random.uniform(50, 150, (n_periods, n_assets))
    returns = np.random.normal(0.001, 0.02, (n_periods, n_assets))
    weights = np.random.uniform(0, 1, n_assets)
    weights = weights / np.sum(weights)
    
    print("Performance Comparison: Loops vs Vectorization")
    print("=" * 60)
    
    # 1. Returns Calculation
    print("\n1. Returns Calculation:")
    
    # BAD: Using explicit loops
    def returns_calculation_loop(prices):
        n_periods, n_assets = prices.shape
        returns = np.zeros((n_periods - 1, n_assets))
        
        for t in range(1, n_periods):
            for i in range(n_assets):
                returns[t-1, i] = (prices[t, i] - prices[t-1, i]) / prices[t-1, i]
        
        return returns
    
    # GOOD: Vectorized operation
    def returns_calculation_vectorized(prices):
        return (prices[1:] - prices[:-1]) / prices[:-1]
    
    # Benchmark
    start = time.time()
    returns_loop = returns_calculation_loop(prices[:100, :50])  # Smaller for loops
    loop_time = time.time() - start
    
    start = time.time()
    returns_vectorized = returns_calculation_vectorized(prices)
    vectorized_time = time.time() - start
    
    print(f"Loop method:       {loop_time:.4f}s (small subset)")
    print(f"Vectorized method: {vectorized_time:.4f}s (full dataset)")
    print(f"Estimated speedup: {loop_time * (n_periods * n_assets) / (100 * 50) / vectorized_time:.1f}x")
    
    # 2. Portfolio Returns
    print("\n2. Portfolio Returns Calculation:")
    
    # BAD: Element-wise multiplication and summation
    def portfolio_returns_loop(returns, weights):
        n_periods = len(returns)
        portfolio_returns = np.zeros(n_periods)
        
        for t in range(n_periods):
            for i in range(len(weights)):
                portfolio_returns[t] += returns[t, i] * weights[i]
        
        return portfolio_returns
    
    # GOOD: Matrix multiplication
    def portfolio_returns_vectorized(returns, weights):
        return np.dot(returns, weights)
    
    start = time.time()
    port_ret_loop = portfolio_returns_loop(returns[:100], weights)
    loop_time = time.time() - start
    
    start = time.time()
    port_ret_vectorized = portfolio_returns_vectorized(returns, weights)
    vectorized_time = time.time() - start
    
    print(f"Loop method:       {loop_time:.4f}s (subset)")
    print(f"Vectorized method: {vectorized_time:.4f}s")
    print(f"Speedup:          {loop_time / vectorized_time:.1f}x")
    
    # 3. Rolling Statistics
    print("\n3. Rolling Statistics (21-day volatility):")
    window = 21
    
    # BAD: Nested loops for rolling calculations
    def rolling_volatility_loop(returns, window):
        n_periods, n_assets = returns.shape
        rolling_vol = np.zeros((n_periods - window + 1, n_assets))
        
        for t in range(window - 1, n_periods):
            for i in range(n_assets):
                window_data = returns[t - window + 1:t + 1, i]
                rolling_vol[t - window + 1, i] = np.std(window_data)
        
        return rolling_vol
    
    # GOOD: Vectorized rolling (using pandas for comparison)
    def rolling_volatility_vectorized(returns, window):
        # Pure NumPy implementation using stride tricks
        from numpy.lib.stride_tricks import sliding_window_view
        
        # Create sliding windows
        windowed = sliding_window_view(returns, window_shape=window, axis=0)
        
        # Calculate standard deviation for each window
        return np.std(windowed, axis=2)
    
    start = time.time()
    roll_vol_loop = rolling_volatility_loop(returns[:100, :10], window)
    loop_time = time.time() - start
    
    start = time.time()
    roll_vol_vectorized = rolling_volatility_vectorized(returns, window)
    vectorized_time = time.time() - start
    
    print(f"Loop method:       {loop_time:.4f}s (subset)")
    print(f"Vectorized method: {vectorized_time:.4f}s")
    print(f"Speedup:          {loop_time / vectorized_time:.1f}x")
    
    # 4. Correlation Matrix Calculation
    print("\n4. Correlation Matrix:")
    
    # BAD: Manual correlation calculation
    def correlation_matrix_manual(returns):
        n_assets = returns.shape[1]
        corr_matrix = np.zeros((n_assets, n_assets))
        
        for i in range(n_assets):
            for j in range(n_assets):
                if i == j:
                    corr_matrix[i, j] = 1.0
                else:
                    corr_matrix[i, j] = np.corrcoef(returns[:, i], returns[:, j])[0, 1]
        
        return corr_matrix
    
    # GOOD: Built-in NumPy function
    def correlation_matrix_builtin(returns):
        return np.corrcoef(returns.T)
    
    start = time.time()
    corr_manual = correlation_matrix_manual(returns[:, :20])  # Subset for manual
    manual_time = time.time() - start
    
    start = time.time()
    corr_builtin = correlation_matrix_builtin(returns)
    builtin_time = time.time() - start
    
    print(f"Manual method:     {manual_time:.4f}s (subset)")
    print(f"Built-in method:   {builtin_time:.4f}s")
    print(f"Speedup:          {manual_time / builtin_time:.1f}x")
    
    return {
        'vectorization_speedups': {
            'returns_calculation': loop_time * (n_periods * n_assets) / (100 * 50) / vectorized_time,
            'portfolio_returns': loop_time / vectorized_time,
            'rolling_statistics': loop_time / vectorized_time,
            'correlation_matrix': manual_time / builtin_time
        }
    }

def memory_optimization_techniques():
    """
    Memory optimization techniques for large datasets
    """
    # Large dataset simulation
    n_assets = 2000
    n_periods = 5000  # 20 years daily
    
    print("Memory Optimization Techniques")
    print("=" * 40)
    
    # 1. Data type optimization
    print("\n1. Data Type Optimization:")
    
    # BAD: Default float64 for everything
    prices_float64 = np.random.uniform(50, 150, (n_periods, n_assets)).astype(np.float64)
    
    # GOOD: Use appropriate precision
    # Price data: Use float32 if precision allows
    prices_float32 = np.random.uniform(50, 150, (n_periods, n_assets)).astype(np.float32)
    
    # Signal data: Use int8 for small integers
    signals = np.random.choice([-1, 0, 1], size=(n_periods, n_assets), p=[0.1, 0.8, 0.1])
    signals_int8 = signals.astype(np.int8)
    signals_float64 = signals.astype(np.float64)  # Wasteful
    
    print(f"Price data (float64): {prices_float64.nbytes / 1e6:.1f} MB")
    print(f"Price data (float32): {prices_float32.nbytes / 1e6:.1f} MB")
    print(f"Signals (int8):       {signals_int8.nbytes / 1e6:.1f} MB")
    print(f"Signals (float64):    {signals_float64.nbytes / 1e6:.1f} MB")
    
    memory_savings = (1 - prices_float32.nbytes / prices_float64.nbytes) * 100
    signal_savings = (1 - signals_int8.nbytes / signals_float64.nbytes) * 100
    
    print(f"Memory savings: {memory_savings:.1f}% (prices), {signal_savings:.1f}% (signals)")
    
    # 2. In-place operations
    print("\n2. In-place Operations:")
    
    data = np.random.randn(1000, 1000).astype(np.float32)
    
    # BAD: Creates new arrays
    def operations_with_copies(data):
        result1 = data * 2.0
        result2 = result1 + 1.0
        result3 = np.exp(result2)
        return result3
    
    # GOOD: In-place operations where possible
    def operations_inplace(data):
        result = data.copy()  # Only one copy
        result *= 2.0         # In-place multiplication
        result += 1.0         # In-place addition
        np.exp(result, out=result)  # In-place exponential
        return result
    
    import tracemalloc
    
    # Measure memory usage
    tracemalloc.start()
    result_copies = operations_with_copies(data)
    current, peak = tracemalloc.get_traced_memory()
    copies_peak = peak
    tracemalloc.stop()
    
    tracemalloc.start()
    result_inplace = operations_inplace(data)
    current, peak = tracemalloc.get_traced_memory()
    inplace_peak = peak
    tracemalloc.stop()
    
    print(f"Peak memory (copies):  {copies_peak / 1e6:.1f} MB")
    print(f"Peak memory (in-place): {inplace_peak / 1e6:.1f} MB")
    print(f"Memory reduction: {(1 - inplace_peak/copies_peak)*100:.1f}%")
    
    # 3. Memory-mapped arrays for very large datasets
    print("\n3. Memory-Mapped Arrays:")
    
    def create_memory_mapped_array(filename, shape, dtype=np.float32):
        """Create a memory-mapped array for out-of-core computation"""
        return np.memmap(filename, dtype=dtype, mode='w+', shape=shape)
    
    # Example: Large dataset that doesn't fit in memory
    large_shape = (10000, 5000)  # 200MB in float32
    
    # Create memory-mapped array
    mmap_array = create_memory_mapped_array('temp_data.dat', large_shape, np.float32)
    
    # Fill with data (would be done in chunks in practice)
    print(f"Created memory-mapped array: {large_shape} shape")
    print(f"Theoretical memory: {np.prod(large_shape) * 4 / 1e6:.1f} MB")
    
    # Clean up
    del mmap_array
    import os
    if os.path.exists('temp_data.dat'):
        os.remove('temp_data.dat')
    
    # 4. Chunked processing for large datasets
    print("\n4. Chunked Processing:")
    
    def process_large_dataset_chunked(data, chunk_size=1000):
        """Process large dataset in chunks to manage memory"""
        n_samples = len(data)
        results = []
        
        for start_idx in range(0, n_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, n_samples)
            chunk = data[start_idx:end_idx]
            
            # Process chunk (example: standardization)
            chunk_mean = np.mean(chunk, axis=0, keepdims=True)
            chunk_std = np.std(chunk, axis=0, keepdims=True)
            processed_chunk = (chunk - chunk_mean) / (chunk_std + 1e-8)
            
            results.append(processed_chunk)
            
            # Optional: explicit memory cleanup
            del chunk, processed_chunk
        
        return np.vstack(results)
    
    # Demonstrate chunked processing
    large_data = np.random.randn(5000, 100).astype(np.float32)
    
    import time
    start = time.time()
    result_chunked = process_large_dataset_chunked(large_data, chunk_size=500)
    chunked_time = time.time() - start
    
    start = time.time()
    # Regular processing (all at once)
    mean_all = np.mean(large_data, axis=0, keepdims=True)
    std_all = np.std(large_data, axis=0, keepdims=True) 
    result_regular = (large_data - mean_all) / (std_all + 1e-8)
    regular_time = time.time() - start
    
    print(f"Chunked processing: {chunked_time:.4f}s")
    print(f"Regular processing: {regular_time:.4f}s")
    print(f"Results match: {np.allclose(result_chunked, result_regular, rtol=1e-3)}")
    
    return {
        'memory_savings': {
            'price_data': memory_savings,
            'signal_data': signal_savings,
            'inplace_operations': (1 - inplace_peak/copies_peak)*100
        },
        'processing_times': {
            'chunked': chunked_time,
            'regular': regular_time
        }
    }
```

### 2. Advanced Performance Techniques
```python
def advanced_performance_techniques():
    """
    Advanced performance optimization techniques
    """
    import time
    import numpy as np
    
    # Setup
    n_assets = 1000
    n_periods = 2000
    np.random.seed(42)
    
    returns = np.random.normal(0.001, 0.02, (n_periods, n_assets)).astype(np.float32)
    
    print("Advanced Performance Techniques")
    print("=" * 40)
    
    # 1. Using numba for JIT compilation
    print("\n1. JIT Compilation with Numba:")
    
    try:
        from numba import jit, prange
        numba_available = True
    except ImportError:
        print("Numba not available, skipping JIT examples")
        numba_available = False
    
    if numba_available:
        # Regular Python function
        def sharpe_ratio_python(returns, risk_free_rate=0.02):
            """Calculate Sharpe ratio using pure Python/NumPy"""
            mean_returns = np.mean(returns, axis=0)
            std_returns = np.std(returns, axis=0)
            excess_returns = mean_returns * 252 - risk_free_rate
            sharpe_ratios = excess_returns / (std_returns * np.sqrt(252))
            return sharpe_ratios
        
        # JIT-compiled function
        @jit(nopython=True)
        def sharpe_ratio_numba(returns, risk_free_rate=0.02):
            """JIT-compiled Sharpe ratio calculation"""
            n_periods, n_assets = returns.shape
            sharpe_ratios = np.zeros(n_assets)
            
            for i in range(n_assets):
                mean_ret = np.mean(returns[:, i])
                std_ret = np.std(returns[:, i])
                excess_ret = mean_ret * 252 - risk_free_rate
                sharpe_ratios[i] = excess_ret / (std_ret * np.sqrt(252))
            
            return sharpe_ratios
        
        # Parallel JIT-compiled function
        @jit(nopython=True, parallel=True)
        def sharpe_ratio_numba_parallel(returns, risk_free_rate=0.02):
            """Parallel JIT-compiled Sharpe ratio calculation"""
            n_periods, n_assets = returns.shape
            sharpe_ratios = np.zeros(n_assets)
            
            for i in prange(n_assets):  # Parallel loop
                mean_ret = np.mean(returns[:, i])
                std_ret = np.std(returns[:, i])
                excess_ret = mean_ret * 252 - risk_free_rate
                sharpe_ratios[i] = excess_ret / (std_ret * np.sqrt(252))
            
            return sharpe_ratios
        
        # Benchmark
        # Warm up JIT functions
        _ = sharpe_ratio_numba(returns[:100, :10])
        _ = sharpe_ratio_numba_parallel(returns[:100, :10])
        
        start = time.time()
        sharpe_python = sharpe_ratio_python(returns)
        python_time = time.time() - start
        
        start = time.time()
        sharpe_numba = sharpe_ratio_numba(returns)
        numba_time = time.time() - start
        
        start = time.time()
        sharpe_numba_par = sharpe_ratio_numba_parallel(returns)
        numba_parallel_time = time.time() - start
        
        print(f"Python/NumPy:        {python_time:.4f}s")
        print(f"Numba JIT:           {numba_time:.4f}s (Speedup: {python_time/numba_time:.1f}x)")
        print(f"Numba Parallel:      {numba_parallel_time:.4f}s (Speedup: {python_time/numba_parallel_time:.1f}x)")
        
        # Verify results are the same
        print(f"Results match: {np.allclose(sharpe_python, sharpe_numba, rtol=1e-5)}")
    
    # 2. Efficient array operations
    print("\n2. Efficient Array Operations:")
    
    # BAD: Using apply_along_axis for simple operations
    def inefficient_standardization(data):
        """Inefficient way to standardize data"""
        return np.apply_along_axis(lambda x: (x - np.mean(x)) / np.std(x), axis=0, arr=data)
    
    # GOOD: Direct vectorized operations
    def efficient_standardization(data):
        """Efficient vectorized standardization"""
        means = np.mean(data, axis=0, keepdims=True)
        stds = np.std(data, axis=0, keepdims=True)
        return (data - means) / (stds + 1e-8)  # Add small epsilon to avoid division by zero
    
    start = time.time()
    result_inefficient = inefficient_standardization(returns)
    inefficient_time = time.time() - start
    
    start = time.time()
    result_efficient = efficient_standardization(returns)
    efficient_time = time.time() - start
    
    print(f"Inefficient method:  {inefficient_time:.4f}s")
    print(f"Efficient method:    {efficient_time:.4f}s")
    print(f"Speedup:            {inefficient_time/efficient_time:.1f}x")
    print(f"Results match:      {np.allclose(result_inefficient, result_efficient)}")
    
    # 3. Memory layout optimization
    print("\n3. Memory Layout Optimization:")
    
    # C-order vs Fortran-order arrays
    returns_c_order = np.ascontiguousarray(returns)  # C-order (row-major)
    returns_f_order = np.asfortranarray(returns)     # Fortran-order (column-major)
    
    def column_wise_operation(data):
        """Operation that accesses columns (assets)"""
        return np.sum(data**2, axis=0)
    
    def row_wise_operation(data):
        """Operation that accesses rows (time periods)"""
        return np.sum(data**2, axis=1)
    
    # Column-wise operations (better with F-order)
    start = time.time()
    for _ in range(10):
        result_c_col = column_wise_operation(returns_c_order)
    c_order_col_time = time.time() - start
    
    start = time.time()
    for _ in range(10):
        result_f_col = column_wise_operation(returns_f_order)
    f_order_col_time = time.time() - start
    
    # Row-wise operations (better with C-order)
    start = time.time()
    for _ in range(10):
        result_c_row = row_wise_operation(returns_c_order)
    c_order_row_time = time.time() - start
    
    start = time.time()
    for _ in range(10):
        result_f_row = row_wise_operation(returns_f_order)
    f_order_row_time = time.time() - start
    
    print(f"Column operations - C-order: {c_order_col_time:.4f}s")
    print(f"Column operations - F-order: {f_order_col_time:.4f}s (Speedup: {c_order_col_time/f_order_col_time:.1f}x)")
    print(f"Row operations - C-order:    {c_order_row_time:.4f}s")
    print(f"Row operations - F-order:    {f_order_row_time:.4f}s (Speedup: {f_order_row_time/c_order_row_time:.1f}x)")
    
    # 4. Broadcasting optimization
    print("\n4. Broadcasting Optimization:")
    
    weights = np.random.uniform(0, 1, n_assets).astype(np.float32)
    weights = weights / np.sum(weights)
    
    # BAD: Explicit reshaping and tiling
    def portfolio_returns_explicit(returns, weights):
        """Explicitly reshape arrays for multiplication"""
        weights_matrix = np.tile(weights, (len(returns), 1))
        weighted_returns = returns * weights_matrix
        return np.sum(weighted_returns, axis=1)
    
    # GOOD: Let NumPy handle broadcasting
    def portfolio_returns_broadcast(returns, weights):
        """Use broadcasting for efficient computation"""
        return np.sum(returns * weights, axis=1)
    
    # BEST: Matrix multiplication
    def portfolio_returns_matmul(returns, weights):
        """Use matrix multiplication (most efficient)"""
        return returns @ weights
    
    start = time.time()
    port_ret_explicit = portfolio_returns_explicit(returns, weights)
    explicit_time = time.time() - start
    
    start = time.time()
    port_ret_broadcast = portfolio_returns_broadcast(returns, weights)
    broadcast_time = time.time() - start
    
    start = time.time()
    port_ret_matmul = portfolio_returns_matmul(returns, weights)
    matmul_time = time.time() - start
    
    print(f"Explicit reshaping:  {explicit_time:.4f}s")
    print(f"Broadcasting:        {broadcast_time:.4f}s (Speedup: {explicit_time/broadcast_time:.1f}x)")
    print(f"Matrix multiplication: {matmul_time:.4f}s (Speedup: {explicit_time/matmul_time:.1f}x)")
    print(f"Results match:       {np.allclose(port_ret_explicit, port_ret_matmul)}")
    
    return {
        'numba_available': numba_available,
        'speedups': {
            'standardization': inefficient_time/efficient_time,
            'memory_layout_col': c_order_col_time/f_order_col_time,
            'memory_layout_row': f_order_row_time/c_order_row_time,
            'broadcasting': explicit_time/broadcast_time,
            'matrix_multiplication': explicit_time/matmul_time
        }
    }

def profiling_and_debugging():
    """
    Tools and techniques for profiling NumPy code
    """
    print("Profiling and Debugging NumPy Code")
    print("=" * 40)
    
    # Sample computationally intensive function
    def complex_portfolio_analytics(returns, n_portfolios=100):
        """Complex analytics function for profiling"""
        n_periods, n_assets = returns.shape
        
        # Generate random portfolios
        np.random.seed(42)
        weights = np.random.uniform(0, 1, (n_portfolios, n_assets))
        weights = weights / np.sum(weights, axis=1, keepdims=True)
        
        # Calculate portfolio returns
        portfolio_returns = returns @ weights.T
        
        # Calculate various metrics
        results = {}
        
        # 1. Basic statistics
        results['mean_returns'] = np.mean(portfolio_returns, axis=0)
        results['volatilities'] = np.std(portfolio_returns, axis=0)
        results['sharpe_ratios'] = results['mean_returns'] / results['volatilities']
        
        # 2. Rolling metrics
        window = 21
        rolling_vols = np.zeros((n_periods - window + 1, n_portfolios))
        for i in range(n_periods - window + 1):
            window_returns = portfolio_returns[i:i+window]
            rolling_vols[i] = np.std(window_returns, axis=0)
        results['rolling_volatilities'] = rolling_vols
        
        # 3. Correlation matrix
        results['correlation_matrix'] = np.corrcoef(portfolio_returns.T)
        
        # 4. Drawdowns
        cumulative_returns = np.cumprod(1 + portfolio_returns, axis=0)
        running_max = np.maximum.accumulate(cumulative_returns, axis=0)
        drawdowns = (cumulative_returns - running_max) / running_max
        results['max_drawdowns'] = np.min(drawdowns, axis=0)
        
        return results
    
    # 1. Basic timing
    print("\n1. Basic Timing:")
    
    returns = np.random.normal(0.001, 0.02, (1000, 100))
    
    import time
    start = time.time()
    results = complex_portfolio_analytics(returns)
    execution_time = time.time() - start
    
    print(f"Total execution time: {execution_time:.4f}s")
    
    # 2. Line-by-line profiling (simplified version)
    print("\n2. Component Timing:")
    
    def timed_portfolio_analytics(returns, n_portfolios=100):
        """Version with timing for each component"""
        times = {}
        
        start = time.time()
        n_periods, n_assets = returns.shape
        np.random.seed(42)
        weights = np.random.uniform(0, 1, (n_portfolios, n_assets))
        weights = weights / np.sum(weights, axis=1, keepdims=True)
        times['setup'] = time.time() - start
        
        start = time.time()
        portfolio_returns = returns @ weights.T
        times['portfolio_returns'] = time.time() - start
        
        start = time.time()
        mean_returns = np.mean(portfolio_returns, axis=0)
        volatilities = np.std(portfolio_returns, axis=0)
        sharpe_ratios = mean_returns / volatilities
        times['basic_stats'] = time.time() - start
        
        start = time.time()
        window = 21
        rolling_vols = np.zeros((n_periods - window + 1, n_portfolios))
        for i in range(n_periods - window + 1):
            window_returns = portfolio_returns[i:i+window]
            rolling_vols[i] = np.std(window_returns, axis=0)
        times['rolling_stats'] = time.time() - start
        
        start = time.time()
        correlation_matrix = np.corrcoef(portfolio_returns.T)
        times['correlation'] = time.time() - start
        
        start = time.time()
        cumulative_returns = np.cumprod(1 + portfolio_returns, axis=0)
        running_max = np.maximum.accumulate(cumulative_returns, axis=0)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdowns = np.min(drawdowns, axis=0)
        times['drawdowns'] = time.time() - start
        
        return times
    
    component_times = timed_portfolio_analytics(returns)
    total_time = sum(component_times.values())
    
    print("Component timing breakdown:")
    for component, time_taken in component_times.items():
        percentage = (time_taken / total_time) * 100
        print(f"{component:20}: {time_taken:.4f}s ({percentage:5.1f}%)")
    
    # 3. Memory profiling
    print("\n3. Memory Usage Analysis:")
    
    def memory_efficient_analytics(returns, n_portfolios=100):
        """Memory-optimized version"""
        n_periods, n_assets = returns.shape
        
        # Use float32 for intermediate calculations where precision allows
        np.random.seed(42)
        weights = np.random.uniform(0, 1, (n_portfolios, n_assets)).astype(np.float32)
        weights = weights / np.sum(weights, axis=1, keepdims=True)
        
        # Calculate portfolio returns in smaller chunks if necessary
        if n_portfolios > 1000:  # For very large number of portfolios
            chunk_size = 500
            portfolio_returns_list = []
            
            for i in range(0, n_portfolios, chunk_size):
                end_idx = min(i + chunk_size, n_portfolios)
                chunk_weights = weights[i:end_idx]
                chunk_returns = returns @ chunk_weights.T
                portfolio_returns_list.append(chunk_returns)
            
            portfolio_returns = np.hstack(portfolio_returns_list)
        else:
            portfolio_returns = returns @ weights.T
        
        # Use more memory-efficient rolling calculations
        window = 21
        # Instead of storing all rolling values, just compute what we need
        final_rolling_vol = np.std(portfolio_returns[-window:], axis=0)
        
        return {
            'portfolio_returns': portfolio_returns,
            'final_rolling_vol': final_rolling_vol,
            'correlation_subset': np.corrcoef(portfolio_returns[:, :min(20, n_portfolios)].T)
        }
    
    # Compare memory usage
    import tracemalloc
    
    tracemalloc.start()
    regular_result = complex_portfolio_analytics(returns, n_portfolios=50)
    current, peak_regular = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    tracemalloc.start()
    efficient_result = memory_efficient_analytics(returns, n_portfolios=50)
    current, peak_efficient = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"Peak memory (regular):   {peak_regular / 1e6:.1f} MB")
    print(f"Peak memory (efficient): {peak_efficient / 1e6:.1f} MB")
    print(f"Memory reduction:        {(1 - peak_efficient/peak_regular)*100:.1f}%")
    
    # 4. Bottleneck identification
    print("\n4. Common Performance Bottlenecks:")
    
    bottlenecks = {
        "Unnecessary array copies": "Use in-place operations and avoid unnecessary .copy()",
        "Wrong data types": "Use appropriate dtypes (float32 vs float64, int8 vs int64)",
        "Non-vectorized operations": "Replace loops with vectorized NumPy operations",
        "Suboptimal memory access": "Consider array layout (C vs F order) for your access patterns",
        "Redundant calculations": "Cache expensive computations when used multiple times",
        "Large temporary arrays": "Use chunked processing or streaming algorithms"
    }
    
    for bottleneck, solution in bottlenecks.items():
        print(f"• {bottleneck}: {solution}")
    
    return {
        'total_execution_time': execution_time,
        'component_times': component_times,
        'memory_usage': {
            'regular': peak_regular,
            'efficient': peak_efficient,
            'reduction_pct': (1 - peak_efficient/peak_regular)*100
        }
    }
```

---

## Financial Calculations & Vectorization

### 1. Core Financial Formulas
```python
def essential_financial_calculations():
    """
    Essential financial calculations implemented efficiently in NumPy
    
    IMPORTANT: These are the building blocks every quant researcher needs
    """
    # Sample data
    np.random.seed(42)
    prices = np.random.uniform(90, 110, (252, 10))  # Daily prices for 10 assets
    initial_portfolio_value = 1000000  # $1M initial portfolio
    
    print("Essential Financial Calculations")
    print("=" * 40)
    
    # 1. Returns calculation (multiple methods)
    print("\n1. Returns Calculations:")
    
    def calculate_returns_comprehensive(prices):
        """All common return calculation methods"""
        results = {}
        
        # Simple returns: R_t = (P_t - P_{t-1}) / P_{t-1}
        results['simple_returns'] = np.diff(prices, axis=0) / prices[:-1]
        
        # Log returns: R_t = ln(P_t / P_{t-1})
        results['log_returns'] = np.log(prices[1:] / prices[:-1])
        
        # Percentage returns (equivalent to simple returns)
        results['pct_returns'] = (prices[1:] - prices[:-1]) / prices[:-1]
        
        # Multi-period returns (e.g., 5-day returns)
        n_periods = 5
        results[f'{n_periods}_day_returns'] = (prices[n_periods:] - prices[:-n_periods]) / prices[:-n_periods]
        
        # Verify equivalence
        print(f"Simple vs Pct returns match: {np.allclose(results['simple_returns'], results['pct_returns'])}")
        
        return results
    
    returns_dict = calculate_returns_comprehensive(prices)
    simple_returns = returns_dict['simple_returns']
    log_returns = returns_dict['log_returns']
    
    # 2. Cumulative returns and wealth evolution
    print("\n2. Cumulative Returns:")
    
    def wealth_evolution(returns, initial_value=1.0, method='simple'):
        """Calculate wealth evolution from returns"""
        if method == 'simple':
            # Wealth_t = Wealth_0 * ∏(1 + R_i)
            cumulative_returns = np.cumprod(1 + returns, axis=0)
        elif method == 'log':
            # For log returns: Wealth_t = Wealth_0 * exp(∑log_returns)
            cumulative_returns = np.exp(np.cumsum(returns, axis=0))
        else:
            raise ValueError("Method must be 'simple' or 'log'")
        
        return initial_value * cumulative_returns
    
    # Calculate wealth evolution
    wealth_simple = wealth_evolution(simple_returns, initial_portfolio_value, 'simple')
    wealth_log = wealth_evolution(log_returns, initial_portfolio_value, 'log')
    
    # Total returns over the period
    total_returns_simple = wealth_simple[-1] / initial_portfolio_value - 1
    total_returns_log = wealth_log[-1] / initial_portfolio_value - 1
    
    print(f"Total returns (simple method): {np.mean(total_returns_simple):.2%}")
    print(f"Total returns (log method):    {np.mean(total_returns_log):.2%}")
    print(f"Methods approximately equal:   {np.allclose(total_returns_simple, total_returns_log, rtol=0.01)}")
    
    # 3. Annualization
    print("\n3. Annualization:")
    
    def annualize_metrics(returns, periods_per_year=252):
        """Annualize returns, volatility, and Sharpe ratio"""
        # Annualized return: geometric mean approach
        if np.any(returns <= -1):
            # Handle extreme negative returns
            print("Warning: Extreme negative returns detected")
            geometric_mean = np.power(np.prod(1 + np.maximum(returns, -0.99), axis=0), 
                                    periods_per_year/len(returns)) - 1
        else:
            geometric_mean = np.power(np.prod(1 + returns, axis=0), 
                                    periods_per_year/len(returns)) - 1
        
        # Alternative: arithmetic mean * periods_per_year (less precise for compounding)
        arithmetic_annualized = np.mean(returns, axis=0) * periods_per_year
        
        # Annualized volatility
        annualized_vol = np.std(returns, axis=0, ddof=1) * np.sqrt(periods_per_year)
        
        # Sharpe ratio (assuming risk-free rate = 2%)
        risk_free_rate = 0.02
        excess_returns = arithmetic_annualized - risk_free_rate
        sharpe_ratio = excess_returns / annualized_vol
        
        return {
            'geometric_mean': geometric_mean,
            'arithmetic_annualized': arithmetic_annualized,
            'annualized_volatility': annualized_vol,
            'sharpe_ratio': sharpe_ratio
        }
    
    annual_metrics = annualize_metrics(simple_returns)
    
    print("Average across all assets:")
    print(f"Geometric mean return:  {np.mean(annual_metrics['geometric_mean']):.2%}")
    print(f"Arithmetic annualized: {np.mean(annual_metrics['arithmetic_annualized']):.2%}")
    print(f"Annualized volatility:  {np.mean(annual_metrics['annualized_volatility']):.2%}")
    print(f"Average Sharpe ratio:   {np.mean(annual_metrics['sharpe_ratio']):.3f}")
    
    # 4. Compound Annual Growth Rate (CAGR)
    print("\n4. CAGR Calculation:")
    
    def calculate_cagr(beginning_value, ending_value, num_periods, periods_per_year=252):
        """Calculate Compound Annual Growth Rate"""
        total_years = num_periods / periods_per_year
        cagr = np.power(ending_value / beginning_value, 1 / total_years) - 1
        return cagr
    
    # Calculate CAGR for each asset
    beginning_values = prices[0]
    ending_values = prices[-1]
    num_periods = len(prices) - 1
    
    cagr_values = calculate_cagr(beginning_values, ending_values, num_periods)
    
    print(f"Average CAGR: {np.mean(cagr_values):.2%}")
    print(f"CAGR vs Geometric mean close: {np.allclose(cagr_values, annual_metrics['geometric_mean'], rtol=0.01)}")
    
    return {
        'returns': returns_dict,
        'wealth_evolution': {'simple': wealth_simple, 'log': wealth_log},
        'annual_metrics': annual_metrics,
        'cagr': cagr_values
    }

def portfolio_calculations():
    """
    Portfolio-level calculations using NumPy vectorization
    """
    # Setup: 20 assets, 252 trading days, multiple portfolios
    np.random.seed(42)
    n_assets = 20
    n_periods = 252
    n_portfolios = 5
    
    # Generate correlated returns
    correlation_matrix = 0.3 * np.ones((n_assets, n_assets)) + 0.7 * np.eye(n_assets)
    returns = np.random.multivariate_normal(
        mean=np.random.uniform(0.05, 0.15, n_assets) / 252,  # Daily expected returns
        cov=correlation_matrix * (0.2**2 / 252),  # Daily covariance
        size=n_periods
    )
    
    # Different portfolio allocations
    weights_matrix = np.random.uniform(0, 1, (n_portfolios, n_assets))
    weights_matrix = weights_matrix / np.sum(weights_matrix, axis=1, keepdims=True)  # Normalize
    
    print("\nPortfolio Calculations")
    print("=" * 30)
    
    # 1. Portfolio returns (vectorized for multiple portfolios)
    def calculate_portfolio_returns_vectorized(returns, weights_matrix):
        """Calculate returns for multiple portfolios simultaneously"""
        # Each column in result corresponds to one portfolio
        return returns @ weights_matrix.T  # Shape: (n_periods, n_portfolios)
    
    portfolio_returns = calculate_portfolio_returns_vectorized(returns, weights_matrix)
    
    print(f"Portfolio returns shape: {portfolio_returns.shape}")
    print(f"Individual asset returns shape: {returns.shape}")
    
    # 2. Portfolio risk decomposition (vectorized)
    def portfolio_risk_analysis_vectorized(returns, weights_matrix):
        """Comprehensive portfolio risk analysis for multiple portfolios"""
        # Covariance matrix
        cov_matrix = np.cov(returns.T)
        
        # Portfolio variances: w' Σ w for each portfolio
        # Using einsum for efficient computation
        portfolio_vars = np.einsum('pi,ij,pj->p', weights_matrix, cov_matrix, weights_matrix)
        portfolio_vols = np.sqrt(portfolio_vars)
        
        # Marginal contribution to risk: (Σw) / σ_p
        marginal_contrib = (cov_matrix @ weights_matrix.T) / portfolio_vols[np.newaxis, :]
        
        # Component contributions: w_i * MCR_i
        component_contrib = weights_matrix.T * marginal_contrib  # Broadcasting
        
        # Percentage contributions
        pct_contrib = component_contrib / portfolio_vols[np.newaxis, :]
        
        # Verify: sum of component contributions = portfolio volatility
        verification = np.sum(component_contrib, axis=0)
        print(f"Risk decomposition verification (should be ~0): {np.max(np.abs(verification - portfolio_vols)):.8f}")
        
        return {
            'portfolio_volatilities': portfolio_vols,
            'marginal_contributions': marginal_contrib,
            'component_contributions': component_contrib,
            'percentage_contributions': pct_contrib
        }
    
    risk_analysis = portfolio_risk_analysis_vectorized(returns, weights_matrix)
    
    print(f"Portfolio volatilities (annualized): {risk_analysis['portfolio_volatilities'] * np.sqrt(252)}")
    
    # 3. Rebalancing and transaction costs
    def calculate_rebalancing_costs(old_weights, new_weights, transaction_cost_bps=10):
        """Calculate transaction costs for portfolio rebalancing"""
        # Weight changes (absolute)
        weight_changes = np.abs(new_weights - old_weights)
        
        # Transaction costs (in basis points)
        transaction_costs = np.sum(weight_changes, axis=1) * (transaction_cost_bps / 10000)
        
        return {
            'weight_changes': weight_changes,
            'total_turnover': np.sum(weight_changes, axis=1),
            'transaction_costs': transaction_costs
        }
    
    # Simulate rebalancing from equal weight to optimized weights
    equal_weights = np.full((n_portfolios, n_assets), 1/n_assets)
    rebalancing_analysis = calculate_rebalancing_costs(equal_weights, weights_matrix)
    
    print(f"Average turnover: {np.mean(rebalancing_analysis['total_turnover']):.1%}")
    print(f"Average transaction costs: {np.mean(rebalancing_analysis['transaction_costs']):.2%}")
    
    # 4. Performance attribution
    def performance_attribution_vectorized(portfolio_returns, asset_returns, weights_matrix):
        """Decompose portfolio performance by asset contribution"""
        n_periods, n_portfolios = portfolio_returns.shape
        n_assets = asset_returns.shape[1]
        
        # Asset contributions: weight * asset_return for each period
        # Shape: (n_periods, n_portfolios, n_assets)
        asset_contributions = asset_returns[:, np.newaxis, :] * weights_matrix[np.newaxis, :, :]
        
        # Total contribution by asset (sum over time)
        total_asset_contributions = np.sum(asset_contributions, axis=0)  # (n_portfolios, n_assets)
        
        # Verification: sum of asset contributions should equal portfolio performance
        portfolio_total_returns = np.sum(portfolio_returns, axis=0)
        attribution_total = np.sum(total_asset_contributions, axis=1)
        
        print(f"Attribution verification error: {np.max(np.abs(attribution_total - portfolio_total_returns)):.8f}")
        
        return {
            'asset_contributions': asset_contributions,
            'total_asset_contributions': total_asset_contributions,
            'top_contributors': np.argmax(total_asset_contributions, axis=1),
            'bottom_contributors': np.argmin(total_asset_contributions, axis=1)
        }
    
    attribution = performance_attribution_vectorized(portfolio_returns, returns, weights_matrix)
    
    print(f"Top contributing assets by portfolio: {attribution['top_contributors']}")
    print(f"Bottom contributing assets by portfolio: {attribution['bottom_contributors']}")
    
    return {
        'portfolio_returns': portfolio_returns,
        'risk_analysis': risk_analysis,
        'rebalancing_analysis': rebalancing_analysis,
        'attribution': attribution
    }
```

### 2. Advanced Financial Metrics
```python
def advanced_financial_metrics():
    """
    Advanced financial metrics and risk measures
    """
    # Generate realistic financial data
    np.random.seed(42)
    n_assets = 8
    n_periods = 1000  # ~4 years daily data
    
    # Create factor-based returns (3-factor model)
    factors = np.random.multivariate_normal([0.0004, 0.0002, 0.0001], 
                                          [[0.0001, 0.00002, 0.00001],
                                           [0.00002, 0.00008, 0.00001],
                                           [0.00001, 0.00001, 0.00005]], 
                                          size=n_periods)
    
    # Factor loadings (beta, size, value)
    factor_loadings = np.random.normal([[1.0, 0.2, 0.1],    # Large cap growth
                                       [1.2, 0.8, -0.2],   # Small cap growth  
                                       [0.8, -0.3, 0.6],   # Large cap value
                                       [1.1, 0.5, 0.4],    # Mid cap blend
                                       [0.9, -0.1, 0.3],   # Large cap blend
                                       [1.3, 0.9, -0.1],   # Small cap growth
                                       [0.7, -0.2, 0.8],   # Large cap value
                                       [1.0, 0.3, 0.2]])   # Mid cap blend
    
    # Generate asset returns using factor model
    systematic_returns = factors @ factor_loadings.T
    idiosyncratic_returns = np.random.normal(0, 0.01, (n_periods, n_assets))
    returns = systematic_returns + idiosyncratic_returns
    
    print("Advanced Financial Metrics")
    print("=" * 35)
    
    # 1. Drawdown Analysis
    print("\n1. Drawdown Analysis:")
    
    def comprehensive_drawdown_analysis(returns):
        """Complete drawdown analysis"""
        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + returns, axis=0)
        
        # Running maximum (peak values)
        running_max = np.maximum.accumulate(cumulative_returns, axis=0)
        
        # Drawdowns
        drawdowns = (cumulative_returns - running_max) / running_max
        
        # Maximum drawdown
        max_drawdowns = np.min(drawdowns, axis=0)
        
        # Drawdown duration analysis
        def calculate_drawdown_durations(drawdowns_series):
            """Calculate drawdown durations for a single series"""
            durations = []
            current_duration = 0
            
            for dd in drawdowns_series:
                if dd < 0:  # In drawdown
                    current_duration += 1
                else:  # Recovery
                    if current_duration > 0:
                        durations.append(current_duration)
                    current_duration = 0
            
            # Add final duration if still in drawdown
            if current_duration > 0:
                durations.append(current_duration)
            
            return np.array(durations)
        
        # Calculate durations for each asset
        drawdown_durations = []
        for i in range(returns.shape[1]):
            durations = calculate_drawdown_durations(drawdowns[:, i])
            drawdown_durations.append(durations)
        
        # Average and maximum durations
        avg_durations = [np.mean(durations) if len(durations) > 0 else 0 
                        for durations in drawdown_durations]
        max_durations = [np.max(durations) if len(durations) > 0 else 0 
                        for durations in drawdown_durations]
        
        # Recovery times (time to new high after max drawdown)
        recovery_times = []
        for i in range(returns.shape[1]):
            max_dd_idx = np.argmin(drawdowns[:, i])
            
            # Find next time drawdown reaches 0 (new high)
            recovery_idx = None
            for j in range(max_dd_idx, len(drawdowns)):
                if drawdowns[j, i] >= 0:
                    recovery_idx = j
                    break
            
            if recovery_idx is not None:
                recovery_times.append(recovery_idx - max_dd_idx)
            else:
                recovery_times.append(len(drawdowns) - max_dd_idx)  # Still recovering
        
        return {
            'drawdowns': drawdowns,
            'max_drawdowns': max_drawdowns,
            'avg_drawdown_durations': np.array(avg_durations),
            'max_drawdown_durations': np.array(max_durations),
            'recovery_times': np.array(recovery_times),
            'currently_in_drawdown': drawdowns[-1] < 0
        }
    
    drawdown_analysis = comprehensive_drawdown_analysis(returns)
    
    print(f"Average max drawdown: {np.mean(drawdown_analysis['max_drawdowns']):.2%}")
    print(f"Average recovery time: {np.mean(drawdown_analysis['recovery_times']):.1f} days")
    print(f"Assets currently in drawdown: {np.sum(drawdown_analysis['currently_in_drawdown'])}")
    
    # 2. Higher moments and tail risk
    print("\n2. Higher Moments and Tail Risk:")
    
    def calculate_moments_and_tail_risk(returns, confidence_levels=[0.95, 0.99]):
        """Calculate skewness, kurtosis, and tail risk measures"""
        # Standardize returns for moment calculations
        standardized = (returns - np.mean(returns, axis=0)) / np.std(returns, axis=0)
        
        # Third moment (skewness)
        skewness = np.mean(standardized**3, axis=0)
        
        # Fourth moment (excess kurtosis)
        excess_kurtosis = np.mean(standardized**4, axis=0) - 3
        
        # Value at Risk (VaR)
        var_results = {}
        for conf in confidence_levels:
            var_results[f'VaR_{int(conf*100)}'] = np.percentile(returns, (1-conf)*100, axis=0)
        
        # Expected Shortfall (Conditional VaR)
        es_results = {}
        for conf in confidence_levels:
            var_threshold = var_results[f'VaR_{int(conf*100)}']
            # For each asset, calculate ES
            es_values = []
            for i in range(returns.shape[1]):
                tail_losses = returns[returns[:, i] <= var_threshold[i], i]
                es_values.append(np.mean(tail_losses) if len(tail_losses) > 0 else var_threshold[i])
            es_results[f'ES_{int(conf*100)}'] = np.array(es_values)
        
        # Downside deviation (semi-standard deviation)
        downside_returns = np.minimum(returns, 0)
        downside_deviation = np.sqrt(np.mean(downside_returns**2, axis=0))
        
        # Sortino ratio (return vs downside risk)
        annual_returns = np.mean(returns, axis=0) * 252
        sortino_ratio = annual_returns / (downside_deviation * np.sqrt(252))
        
        return {
            'skewness': skewness,
            'excess_kurtosis': excess_kurtosis,
            'var': var_results,
            'expected_shortfall': es_results,
            'downside_deviation': downside_deviation,
            'sortino_ratio': sortino_ratio
        }
    
    moments_analysis = calculate_moments_and_tail_risk(returns)
    
    print(f"Average skewness: {np.mean(moments_analysis['skewness']):.3f}")
    print(f"Average excess kurtosis: {np.mean(moments_analysis['excess_kurtosis']):.3f}")
    print(f"Average VaR (95%): {np.mean(moments_analysis['var']['VaR_95']):.3f}")
    print(f"Average Expected Shortfall (95%): {np.mean(moments_analysis['expected_shortfall']['ES_95']):.3f}")
    print(f"Average Sortino ratio: {np.mean(moments_analysis['sortino_ratio']):.3f}")
    
    # 3. Beta and factor exposures
    print("\n3. Factor Analysis:")
    
    def factor_exposure_analysis(asset_returns, factor_returns):
        """Analyze factor exposures and residual risks"""
        n_periods, n_assets = asset_returns.shape
        n_factors = factor_returns.shape[1]
        
        # Add intercept to factor matrix
        X = np.column_stack([np.ones(n_periods), factor_returns])  # [intercept, factors]
        
        # OLS regression: R = X*β + ε
        # β = (X'X)^(-1) * X' * R
        XtX_inv = np.linalg.inv(X.T @ X)
        coefficients = XtX_inv @ X.T @ asset_returns  # Shape: (n_factors+1, n_assets)
        
        # Extract alphas and factor loadings
        alphas = coefficients[0]  # Intercept terms
        factor_loadings = coefficients[1:]  # Factor loadings
        
        # Predicted returns and residuals
        predicted_returns = X @ coefficients
        residuals = asset_returns - predicted_returns
        
        # R-squared for each asset
        tss = np.sum((asset_returns - np.mean(asset_returns, axis=0))**2, axis=0)
        rss = np.sum(residuals**2, axis=0)
        r_squared = 1 - rss / tss
        
        # Factor-specific risk decomposition
        # Total variance = systematic variance + idiosyncratic variance
        asset_variances = np.var(asset_returns, axis=0, ddof=1)
        residual_variances = np.var(residuals, axis=0, ddof=1)
        systematic_variances = asset_variances - residual_variances
        
        # Factor contribution to variance
        factor_cov = np.cov(factor_returns.T)
        factor_var_contributions = np.zeros((n_factors, n_assets))
        
        for i in range(n_assets):
            # Asset i's factor loadings
            asset_loadings = factor_loadings[:, i]
            
            # Variance contribution from each factor
            for f in range(n_factors):
                # Marginal variance contribution of factor f
                factor_var_contributions[f, i] = (asset_loadings[f]**2 * factor_cov[f, f] + 
                                                np.sum([asset_loadings[f] * asset_loadings[g] * factor_cov[f, g] 
                                                       for g in range(n_factors) if g != f]))
        
        return {
            'alphas': alphas,
            'factor_loadings': factor_loadings,
            'r_squared': r_squared,
            'residual_variances': residual_variances,
            'systematic_variances': systematic_variances,
            'factor_var_contributions': factor_var_contributions,
            'residuals': residuals
        }
    
    factor_analysis = factor_exposure_analysis(returns, factors)
    
    print(f"Average alpha (annualized): {np.mean(factor_analysis['alphas']) * 252:.2%}")
    print(f"Average R-squared: {np.mean(factor_analysis['r_squared']):.3f}")
    print(f"Average market beta: {np.mean(factor_analysis['factor_loadings'][0]):.3f}")
    print(f"Average size factor loading: {np.mean(factor_analysis['factor_loadings'][1]):.3f}")
    print(f"Average value factor loading: {np.mean(factor_analysis['factor_loadings'][2]):.3f}")
    
    # Systematic vs idiosyncratic risk
    systematic_risk_pct = np.mean(factor_analysis['systematic_variances'] / 
                                (factor_analysis['systematic_variances'] + factor_analysis['residual_variances']))
    print(f"Average systematic risk percentage: {systematic_risk_pct:.1%}")
    
    return {
        'drawdown_analysis': drawdown_analysis,
        'moments_analysis': moments_analysis,
        'factor_analysis': factor_analysis,
        'returns': returns,
        'factors': factors
    }
```

---

## Time Series Operations

### 1. Time Series Analysis with NumPy
```python
def time_series_operations():
    """
    Time series operations essential for financial analysis
    
    CRITICAL: Proper handling of time series data is fundamental in finance
    """
    # Create realistic time series data
    np.random.seed(42)
    n_periods = 1000
    n_assets = 5
    
    # Generate dates (business days only)
    import pandas as pd
    dates = pd.date_range('2020-01-01', periods=n_periods, freq='B')
    
    # Generate correlated price series with trends
    base_prices = 100
    drift_rates = np.array([0.08, 0.12, 0.06, 0.10, 0.09])  # Annual drift rates
    volatilities = np.array([0.15, 0.25, 0.12, 0.20, 0.18])  # Annual volatilities
    
    # Correlation matrix
    correlation = np.array([
        [1.00, 0.60, 0.40, 0.30, 0.25],
        [0.60, 1.00, 0.35, 0.40, 0.30],
        [0.40, 0.35, 1.00, 0.25, 0.20],
        [0.30, 0.40, 0.25, 1.00, 0.45],
        [0.25, 0.30, 0.20, 0.45, 1.00]
    ])
    
    # Cholesky decomposition for correlation
    L = np.linalg.cholesky(correlation)
    
    # Generate correlated random shocks
    dt = 1/252  # Daily time step
    random_shocks = np.random.standard_normal((n_periods, n_assets))
    correlated_shocks = random_shocks @ L.T
    
    # Geometric Brownian Motion
    prices = np.zeros((n_periods + 1, n_assets))
    prices[0] = base_prices
    
    for t in range(n_periods):
        drift = (drift_rates - 0.5 * volatilities**2) * dt
        diffusion = volatilities * np.sqrt(dt) * correlated_shocks[t]
        prices[t+1] = prices[t] * np.exp(drift + diffusion)
    
    # Remove initial price for returns calculation
    prices = prices[1:]
    returns = np.log(prices[1:] / prices[:-1])
    
    print("Time Series Operations")
    print("=" * 30)
    
    # 1. Moving averages and smoothing
    print("\n1. Moving Averages and Smoothing:")
    
    def calculate_moving_averages(prices, windows=[5, 10, 20, 50]):
        """Calculate various moving averages efficiently"""
        moving_averages = {}
        
        for window in windows:
            if window <= len(prices):
                # Simple Moving Average using convolution (efficient)
                kernel = np.ones(window) / window
                ma = np.zeros_like(prices)
                
                for i in range(prices.shape[1]):  # For each asset
                    # Use 'valid' mode to avoid edge effects
                    ma_valid = np.convolve(prices[:, i], kernel, mode='valid')
                    
                    # Pad with NaN for first (window-1) observations
                    ma[:, i] = np.concatenate([
                        np.full(window - 1, np.nan),
                        ma_valid
                    ])
                
                moving_averages[f'SMA_{window}'] = ma
                
                # Exponential Moving Average
                ema = np.zeros_like(prices)
                alpha = 2 / (window + 1)  # Smoothing factor
                
                for i in range(prices.shape[1]):
                    ema[0, i] = prices[0, i]  # Initialize with first price
                    for t in range(1, len(prices)):
                        ema[t, i] = alpha * prices[t, i] + (1 - alpha) * ema[t-1, i]
                
                moving_averages[f'EMA_{window}'] = ema
        
        return moving_averages
    
    ma_results = calculate_moving_averages(prices)
    
    # Compare SMA vs EMA responsiveness
    sma_20 = ma_results['SMA_20']
    ema_20 = ma_results['EMA_20']
    
    # Calculate mean absolute deviation from actual prices (measure of smoothing)
    valid_idx = ~np.isnan(sma_20).any(axis=1)
    sma_deviation = np.mean(np.abs(prices[valid_idx] - sma_20[valid_idx]))
    ema_deviation = np.mean(np.abs(prices[valid_idx] - ema_20[valid_idx]))
    
    print(f"SMA vs EMA deviation from prices:")
    print(f"SMA-20 deviation: {sma_deviation:.3f}")
    print(f"EMA-20 deviation: {ema_deviation:.3f} (more responsive)")
    
    # 2. Volatility estimation
    print("\n2. Volatility Estimation:")
    
    def estimate_volatility_methods(returns, windows=[20, 60]):
        """Multiple volatility estimation methods"""
        vol_estimates = {}
        
        for window in windows:
            if window <= len(returns):
                # 1. Simple rolling standard deviation
                rolling_vol = np.zeros_like(returns)
                for i in range(window-1, len(returns)):
                    rolling_vol[i] = np.std(returns[i-window+1:i+1], axis=0, ddof=1)
                
                vol_estimates[f'rolling_vol_{window}'] = rolling_vol * np.sqrt(252)  # Annualized
                
                # 2. EWMA volatility (exponentially weighted moving average)
                ewma_vol = np.zeros_like(returns)
                lambda_ewma = 0.94  # RiskMetrics standard
                
                for asset in range(returns.shape[1]):
                    # Initialize with first squared return
                    ewma_vol[0, asset] = returns[0, asset]**2
                    
                    for t in range(1, len(returns)):
                        ewma_vol[t, asset] = (lambda_ewma * ewma_vol[t-1, asset] + 
                                            (1 - lambda_ewma) * returns[t, asset]**2)
                
                vol_estimates[f'ewma_vol_{window}'] = np.sqrt(ewma_vol * 252)  # Annualized
        
        # 3. Realized volatility (high-frequency proxy)
        # Simulate intraday returns for realized vol calculation
        n_intraday = 24  # Hourly returns within each day
        intraday_vol = np.zeros((len(returns), returns.shape[1]))
        
        for day in range(len(returns)):
            # Simulate hourly returns that sum to daily return
            daily_ret = returns[day]
            hourly_rets = np.random.multivariate_normal(
                daily_ret / n_intraday, 
                np.diag((daily_ret / n_intraday)**2 / 10),  # Scaled variance
                size=n_intraday
            )
            # Realized volatility = sum of squared intraday returns
            intraday_vol[day] = np.sum(hourly_rets**2, axis=0)
        
        vol_estimates['realized_vol'] = np.sqrt(intraday_vol * 252)
        
        return vol_estimates
    
    volatility_estimates = estimate_volatility_methods(returns)
    
    # Compare volatility estimates
    print("Average volatility estimates (annualized):")
    for method, vol_est in volatility_estimates.items():
        avg_vol = np.nanmean(vol_est)
        print(f"{method:20}: {avg_vol:.2%}")
    
    # 3. Trend detection and filtering
    print("\n3. Trend Detection:")
    
    def trend_analysis(prices, returns):
        """Detect trends using multiple methods"""
        trend_results = {}
        
        # 1. Linear trend (OLS regression)
        n_periods = len(prices)
        time_index = np.arange(n_periods)
        
        # Fit linear trend for each asset
        trends = np.zeros(prices.shape[1])
        r_squared = np.zeros(prices.shape[1])
        
        for i in range(prices.shape[1]):
            # Log prices for exponential trend
            log_prices = np.log(prices[:, i])
            
            # OLS: log(P_t) = α + β*t + ε
            X = np.column_stack([np.ones(n_periods), time_index])
            coefficients = np.linalg.lstsq(X, log_prices, rcond=None)[0]
            
            trends[i] = coefficients[1] * 252  # Annualized trend (β)
            
            # R-squared
            predicted = X @ coefficients
            ss_res = np.sum((log_prices - predicted)**2)
            ss_tot = np.sum((log_prices - np.mean(log_prices))**2)
            r_squared[i] = 1 - (ss_res / ss_tot)
        
        trend_results['linear_trends'] = trends
        trend_results['trend_r_squared'] = r_squared
        
        # 2. Moving average crossover signals
        sma_short = ma_results['SMA_10']
        sma_long = ma_results['SMA_20']
        
        # Trend signal: +1 when short MA > long MA, -1 otherwise
        trend_signals = np.zeros_like(prices)
        valid_mask = ~(np.isnan(sma_short) | np.isnan(sma_long))
        
        trend_signals[valid_mask] = np.where(
            sma_short[valid_mask] > sma_long[valid_mask], 1, -1
        )
        
        trend_results['ma_crossover_signals'] = trend_signals
        
        # 3. Momentum indicators
        # Price momentum (return over lookback period)
        lookback_periods = [21, 63, 252]  # 1 month, 3 months, 1 year
        
        momentum_indicators = {}
        for period in lookback_periods:
            if period < len(prices):
                momentum = np.zeros_like(prices)
                momentum[period:] = (prices[period:] / prices[:-period]) - 1
                momentum_indicators[f'momentum_{period}d'] = momentum
        
        trend_results['momentum_indicators'] = momentum_indicators
        
        return trend_results
    
    trend_analysis_results = trend_analysis(prices, returns)
    
    print("Trend Analysis Results:")
    print(f"Average linear trend (annualized): {np.mean(trend_analysis_results['linear_trends']):.2%}")
    print(f"Average trend R-squared: {np.mean(trend_analysis_results['trend_r_squared']):.3f}")
    
    # Current momentum signals
    current_momentum = {
        name: momentum_data[-1] for name, momentum_data 
        in trend_analysis_results['momentum_indicators'].items()
    }
    print("Current momentum (last observation):")
    for period, momentum_val in current_momentum.items():
        print(f"  {period}: {np.mean(momentum_val):.2%}")
    
    return {
        'prices': prices,
        'returns': returns,
        'dates': dates,
        'moving_averages': ma_results,
        'volatility_estimates': volatility_estimates,
        'trend_analysis': trend_analysis_results
    }
```

### 2. Advanced Time Series Techniques
```python
def advanced_time_series_techniques():
    """
    Advanced time series analysis techniques for financial data
    """
    # Generate more complex time series with regime changes
    np.random.seed(42)
    n_periods = 1500
    
    # Regime-switching parameters
    regime_1_periods = 500   # Low volatility regime
    regime_2_periods = 700   # High volatility regime  
    regime_3_periods = 300   # Crisis regime
    
    # Different volatility regimes
    vol_regime_1 = 0.12  # 12% annual volatility
    vol_regime_2 = 0.25  # 25% annual volatility
    vol_regime_3 = 0.45  # 45% annual volatility (crisis)
    
    # Generate returns with regime changes
    returns_series = np.concatenate([
        np.random.normal(0.0008, vol_regime_1/np.sqrt(252), regime_1_periods),  # Bull market
        np.random.normal(0.0003, vol_regime_2/np.sqrt(252), regime_2_periods),  # Neutral
        np.random.normal(-0.0010, vol_regime_3/np.sqrt(252), regime_3_periods)  # Crisis
    ])
    
    print("Advanced Time Series Techniques")
    print("=" * 40)
    
    # 1. Regime detection using rolling statistics
    print("\n1. Regime Detection:")
    
    def detect_regimes_statistical(returns, window=60):
        """Detect regime changes using rolling statistics"""
        n_obs = len(returns)
        
        # Rolling volatility
        rolling_vol = np.zeros(n_obs)
        rolling_mean = np.zeros(n_obs)
        rolling_skew = np.zeros(n_obs)
        
        for i in range(window, n_obs):
            window_data = returns[i-window:i]
            rolling_vol[i] = np.std(window_data) * np.sqrt(252)
            rolling_mean[i] = np.mean(window_data) * 252
            
            # Skewness calculation
            std_data = (window_data - np.mean(window_data)) / np.std(window_data)
            rolling_skew[i] = np.mean(std_data**3)
        
        # Detect regime changes using volatility thresholds
        vol_threshold_high = np.percentile(rolling_vol[window:], 75)
        vol_threshold_low = np.percentile(rolling_vol[window:], 25)
        
        regimes = np.zeros(n_obs, dtype=int)
        regimes[rolling_vol <= vol_threshold_low] = 0  # Low vol
        regimes[(rolling_vol > vol_threshold_low) & (rolling_vol < vol_threshold_high)] = 1  # Medium
        regimes[rolling_vol >= vol_threshold_high] = 2  # High vol
        
        # Regime transition matrix
        n_regimes = 3
        transition_matrix = np.zeros((n_regimes, n_regimes))
        
        for i in range(1, len(regimes)):
            if regimes[i-1] >= 0 and regimes[i] >= 0:  # Valid regimes
                transition_matrix[regimes[i-1], regimes[i]] += 1
        
        # Normalize to probabilities
        for i in range(n_regimes):
            if np.sum(transition_matrix[i]) > 0:
                transition_matrix[i] /= np.sum(transition_matrix[i])
        
        return {
            'regimes': regimes,
            'rolling_volatility': rolling_vol,
            'rolling_mean': rolling_mean,
            'rolling_skewness': rolling_skew,
            'transition_matrix': transition_matrix
        }
    
    regime_analysis = detect_regimes_statistical(returns_series)
    
    # Regime statistics
    regime_counts = np.bincount(regime_analysis['regimes'])
    regime_percentages = regime_counts / len(returns_series) * 100
    
    print("Detected Regimes:")
    print(f"Low volatility:    {regime_percentages[0]:.1f}% of time")
    print(f"Medium volatility: {regime_percentages[1]:.1f}% of time") 
    print(f"High volatility:   {regime_percentages[2]:.1f}% of time")
    
    print("\nTransition Probabilities:")
    regime_names = ['Low Vol', 'Med Vol', 'High Vol']
    for i in range(3):
        for j in range(3):
            prob = regime_analysis['transition_matrix'][i, j]
            print(f"{regime_names[i]} -> {regime_names[j]}: {prob:.3f}")
    
    # 2. Autocorrelation and serial dependence
    print("\n2. Autocorrelation Analysis:")
    
    def autocorrelation_analysis(returns, max_lag=20):
        """Comprehensive autocorrelation analysis"""
        n_obs = len(returns)
        
        # Calculate autocorrelations at different lags
        autocorrelations = np.zeros(max_lag + 1)
        
        for lag in range(max_lag + 1):
            if lag == 0:
                autocorrelations[lag] = 1.0
            else:
                if n_obs > lag:
                    # Pearson correlation between series and lagged series
                    x = returns[:-lag]
                    y = returns[lag:]
                    
                    if len(x) > 1:
                        autocorrelations[lag] = np.corrcoef(x, y)[0, 1]
        
        # Ljung-Box test statistic (simplified)
        # Tests for serial correlation
        ljung_box = 0
        for lag in range(1, min(max_lag + 1, n_obs // 4)):
            if not np.isnan(autocorrelations[lag]):
                ljung_box += autocorrelations[lag]**2 / (n_obs - lag)
        
        ljung_box *= n_obs * (n_obs + 2)
        
        # Partial autocorrelations (simplified calculation)
        partial_autocorrs = np.zeros(max_lag + 1)
        partial_autocorrs[0] = 1.0
        
        if max_lag >= 1:
            partial_autocorrs[1] = autocorrelations[1]
        
        # For higher lags, use Yule-Walker equations (simplified)
        for lag in range(2, max_lag + 1):
            if lag < n_obs // 4:
                # Approximate partial autocorrelation
                numerator = autocorrelations[lag] - np.sum([
                    partial_autocorrs[j] * autocorrelations[lag - j] 
                    for j in range(1, lag)
                ])
                denominator = 1 - np.sum([
                    partial_autocorrs[j] * autocorrelations[j] 
                    for j in range(1, lag)
                ])
                
                if abs(denominator) > 1e-10:
                    partial_autocorrs[lag] = numerator / denominator
        
        return {
            'autocorrelations': autocorrelations,
            'partial_autocorrelations': partial_autocorrs,
            'ljung_box_statistic': ljung_box,
            'significant_lags': np.where(np.abs(autocorrelations) > 2/np.sqrt(n_obs))[0]
        }
    
    autocorr_analysis = autocorrelation_analysis(returns_series)
    
    print(f"Ljung-Box statistic: {autocorr_analysis['ljung_box_statistic']:.2f}")
    print(f"Significant autocorrelation lags: {autocorr_analysis['significant_lags']}")
    
    # First few autocorrelations
    print("Autocorrelations (first 5 lags):")
    for i in range(5):
        print(f"  Lag {i}: {autocorr_analysis['autocorrelations'][i]:.4f}")
    
    # 3. Seasonality detection
    print("\n3. Seasonality Analysis:")
    
    def seasonality_analysis(returns, frequency='daily'):
        """Detect seasonal patterns in financial returns"""
        n_obs = len(returns)
        
        if frequency == 'daily':
            # Weekly seasonality (5 trading days)
            day_of_week_returns = np.zeros(5)
            day_counts = np.zeros(5)
            
            for i in range(n_obs):
                day_of_week = i % 5  # 0=Monday, 4=Friday
                day_of_week_returns[day_of_week] += returns[i]
                day_counts[day_of_week] += 1
            
            # Average return by day of week
            avg_day_returns = day_of_week_returns / np.maximum(day_counts, 1)
            
            # Monthly effect (approximation using 21-day cycles)
            monthly_returns = np.zeros(21)
            monthly_counts = np.zeros(21)
            
            for i in range(n_obs):
                day_of_month = i % 21
                monthly_returns[day_of_month] += returns[i]
                monthly_counts[day_of_month] += 1
            
            avg_monthly_returns = monthly_returns / np.maximum(monthly_counts, 1)
            
        # Statistical test for seasonality (F-test approximation)
        overall_mean = np.mean(returns)
        
        # Day-of-week effect
        dow_variance_explained = np.sum((avg_day_returns - overall_mean)**2 * day_counts)
        total_variance = np.sum((returns - overall_mean)**2)
        
        dow_r_squared = dow_variance_explained / total_variance if total_variance > 0 else 0
        
        return {
            'day_of_week_effects': avg_day_returns,
            'monthly_effects': avg_monthly_returns,
            'day_of_week_r_squared': dow_r_squared,
            'strongest_day': np.argmax(avg_day_returns),
            'weakest_day': np.argmin(avg_day_returns)
        }
    
    seasonality_results = seasonality_analysis(returns_series)
    
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    print("Day-of-week effects (annualized):")
    for i, day in enumerate(day_names):
        effect = seasonality_results['day_of_week_effects'][i] * 252
        print(f"  {day}: {effect:.2%}")
    
    print(f"Strongest day: {day_names[seasonality_results['strongest_day']]}")
    print(f"Weakest day: {day_names[seasonality_results['weakest_day']]}")
    print(f"Day-of-week R²: {seasonality_results['day_of_week_r_squared']:.4f}")
    
    # 4. Filtering and smoothing
    print("\n4. Advanced Filtering:")
    
    def advanced_filtering(returns):
        """Apply various filtering techniques"""
        filtering_results = {}
        
        # 1. Hodrick-Prescott Filter (simplified approximation)
        def hp_filter_simple(data, lambda_param=1600):
            """Simplified HP filter implementation"""
            n = len(data)
            
            # Create penalty matrix (simplified version)
            # In practice, use specialized libraries like statsmodels
            trend = np.zeros(n)
            
            # Simple smoothing approximation
            window = min(int(np.sqrt(lambda_param)), n//4)
            
            for i in range(n):
                start_idx = max(0, i - window)
                end_idx = min(n, i + window + 1)
                trend[i] = np.mean(data[start_idx:end_idx])
            
            cycle = data - trend
            return trend, cycle
        
        # Apply HP filter to cumulative returns
        cumulative_returns = np.cumsum(returns)
        hp_trend, hp_cycle = hp_filter_simple(cumulative_returns)
        
        filtering_results['hp_trend'] = hp_trend
        filtering_results['hp_cycle'] = hp_cycle
        
        # 2. Band-pass filter (approximate)
        # Extract business cycle frequencies
        def bandpass_filter(data, low_freq=6, high_freq=32):
            """Simple band-pass filter"""
            # Use moving averages as approximation
            ma_short = np.convolve(data, np.ones(low_freq)/low_freq, mode='same')
            ma_long = np.convolve(data, np.ones(high_freq)/high_freq, mode='same')
            
            return ma_short - ma_long
        
        bp_filtered = bandpass_filter(cumulative_returns)
        filtering_results['bandpass_cycle'] = bp_filtered
        
        # 3. Kalman Filter (simplified - constant parameters)
        def simple_kalman_filter(observations, process_noise=0.01, observation_noise=0.1):
            """Simplified Kalman filter for trend extraction"""
            n = len(observations)
            
            # State: [level, slope]
            state = np.array([observations[0], 0.0])
            P = np.eye(2) * 1.0  # Initial covariance
            
            # Transition matrix (random walk + drift)
            F = np.array([[1, 1], [0, 1]])
            
            # Observation matrix
            H = np.array([1, 0])
            
            # Noise matrices
            Q = np.eye(2) * process_noise
            R = observation_noise
            
            states = np.zeros((n, 2))
            
            for t in range(n):
                # Predict
                state = F @ state
                P = F @ P @ F.T + Q
                
                # Update
                y = observations[t] - H @ state  # Innovation
                S = H @ P @ H.T + R  # Innovation covariance
                K = P @ H.T / S  # Kalman gain
                
                state = state + K * y
                P = P - K.reshape(-1, 1) @ H.reshape(1, -1) @ P
                
                states[t] = state
            
            return states[:, 0], states[:, 1]  # level, slope
        
        kf_level, kf_slope = simple_kalman_filter(cumulative_returns)
        filtering_results['kalman_trend'] = kf_level
        filtering_results['kalman_slope'] = kf_slope
        
        return filtering_results
    
    filtering_results = advanced_filtering(returns_series)
    
    # Compare filtering methods
    print("Filtering Results (final values):")
    print(f"HP trend (final): {filtering_results['hp_trend'][-1]:.4f}")
    print(f"Kalman trend (final): {filtering_results['kalman_trend'][-1]:.4f}")
    print(f"Current slope estimate: {filtering_results['kalman_slope'][-1]:.6f}")
    
    # Volatility of filtered cycles
    hp_cycle_vol = np.std(filtering_results['hp_cycle']) * np.sqrt(252)
    bp_cycle_vol = np.std(filtering_results['bandpass_cycle']) * np.sqrt(252)
    
    print(f"HP cycle volatility: {hp_cycle_vol:.2%}")
    print(f"Bandpass cycle volatility: {bp_cycle_vol:.2%}")
    
    return {
        'returns_series': returns_series,
        'regime_analysis': regime_analysis,
        'autocorr_analysis': autocorr_analysis,
        'seasonality_results': seasonality_results,
        'filtering_results': filtering_results
    }
```

---

## Risk Calculations & Statistics

### 1. Value at Risk and Risk Metrics
```python
def comprehensive_risk_calculations():
    """
    Comprehensive risk calculations using NumPy
    
    CRITICAL: Risk management is the foundation of quantitative finance
    """
    # Generate realistic portfolio returns with fat tails and volatility clustering
    np.random.seed(42)
    n_periods = 2000
    n_assets = 10
    
    # Generate returns with GARCH-like volatility clustering
    base_vol = 0.02 / np.sqrt(252)  # 2% annual vol, daily
    vol_persistence = 0.9
    vol_shock = 0.1
    
    volatilities = np.zeros((n_periods, n_assets))
    returns = np.zeros((n_periods, n_assets))
    
    # Initialize first period
    volatilities[0] = base_vol
    
    for t in range(n_periods):
        if t > 0:
            # GARCH(1,1) volatility process
            volatilities[t] = np.sqrt(
                base_vol**2 * (1 - vol_persistence - vol_shock) +
                vol_persistence * volatilities[t-1]**2 +
                vol_shock * returns[t-1]**2
            )
        
        # Generate returns with fat tails (Student t-distribution approximation)
        normal_shocks = np.random.standard_normal(n_assets)
        chi2_shocks = np.random.chisquare(6, n_assets)  # 6 degrees of freedom
        fat_tail_shocks = normal_shocks * np.sqrt(6 / chi2_shocks)
        
        returns[t] = volatilities[t] * fat_tail_shocks
    
    print("Comprehensive Risk Calculations")
    print("=" * 40)
    
    # 1. Value at Risk (VaR) - Multiple Methods
    print("\n1. Value at Risk Calculations:")
    
    def calculate_var_comprehensive(returns, confidence_levels=[0.95, 0.99], 
                                  methods=['historical', 'parametric', 'cornish_fisher']):
        """Calculate VaR using multiple methodologies"""
        var_results = {}
        n_periods, n_assets = returns.shape
        
        for conf_level in confidence_levels:
            alpha = 1 - conf_level
            var_results[conf_level] = {}
            
            for method in methods:
                if method == 'historical':
                    # Historical simulation
                    var_values = np.percentile(returns, alpha * 100, axis=0)
                    
                elif method == 'parametric':
                    # Parametric (Gaussian) VaR
                    mean_returns = np.mean(returns, axis=0)
                    std_returns = np.std(returns, axis=0, ddof=1)
                    
                    from scipy import stats
                    z_score = stats.norm.ppf(alpha)
                    var_values = mean_returns + z_score * std_returns
                    
                elif method == 'cornish_fisher':
                    # Modified VaR using Cornish-Fisher expansion (accounts for skew/kurtosis)
                    mean_returns = np.mean(returns, axis=0)
                    std_returns = np.std(returns, axis=0, ddof=1)
                    
                    # Calculate higher moments
                    centered_returns = returns - mean_returns
                    skewness = np.mean((centered_returns / std_returns)**3, axis=0)
                    excess_kurtosis = np.mean((centered_returns / std_returns)**4, axis=0) - 3
                    
                    from scipy import stats
                    z = stats.norm.ppf(alpha)
                    
                    # Cornish-Fisher adjustment
                    z_cf = (z + (z**2 - 1) * skewness / 6 + 
                           (z**3 - 3*z) * excess_kurtosis / 24 - 
                           (2*z**3 - 5*z) * skewness**2 / 36)
                    
                    var_values = mean_returns + z_cf * std_returns
                
                var_results[conf_level][method] = var_values
        
        return var_results
    
    var_results = calculate_var_comprehensive(returns)
    
    # Display VaR results
    for conf_level, methods_dict in var_results.items():
        print(f"\nVaR at {conf_level*100}% confidence level:")
        for method, var_values in methods_dict.items():
            avg_var = np.mean(var_values)
            print(f"  {method:15}: {avg_var:.4f} ({avg_var*np.sqrt(252):.2%} annualized)")
    
    # 2. Expected Shortfall (Conditional VaR)
    print("\n2. Expected Shortfall:")
    
    def calculate_expected_shortfall(returns, confidence_levels=[0.95, 0.99]):
        """Calculate Expected Shortfall (CVaR)"""
        es_results = {}
        
        for conf_level in confidence_levels:
            alpha = 1 - conf_level
            
            # Historical ES
            var_threshold = np.percentile(returns, alpha * 100, axis=0)
            
            es_values = np.zeros(returns.shape[1])
            for i in range(returns.shape[1]):
                tail_losses = returns[returns[:, i] <= var_threshold[i], i]
                es_values[i] = np.mean(tail_losses) if len(tail_losses) > 0 else var_threshold[i]
            
            es_results[conf_level] = es_values
        
        return es_results
    
    es_results = calculate_expected_shortfall(returns)
    
    for conf_level, es_values in es_results.items():
        avg_es = np.mean(es_values)
        print(f"Expected Shortfall ({conf_level*100}%): {avg_es:.4f} ({avg_es*np.sqrt(252):.2%} annualized)")
    
    # 3. Risk-Adjusted Performance Metrics
    print("\n3. Risk-Adjusted Performance Metrics:")
    
    def calculate_risk_adjusted_metrics(returns, risk_free_rate=0.02):
        """Calculate comprehensive risk-adjusted performance metrics"""
        # Annualization factor
        periods_per_year = 252
        
        # Basic statistics
        mean_returns = np.mean(returns, axis=0) * periods_per_year
        std_returns = np.std(returns, axis=0, ddof=1) * np.sqrt(periods_per_year)
        
        # Sharpe Ratio
        excess_returns = mean_returns - risk_free_rate
        sharpe_ratios = excess_returns / std_returns
        
        # Sortino Ratio (downside deviation)
        downside_returns = np.minimum(returns, 0)
        downside_std = np.sqrt(np.mean(downside_returns**2, axis=0)) * np.sqrt(periods_per_year)
        sortino_ratios = excess_returns / downside_std
        
        # Calmar Ratio (return / max drawdown)
        def calculate_max_drawdown(returns_series):
            cumulative = np.cumprod(1 + returns_series, axis=0)
            running_max = np.maximum.accumulate(cumulative, axis=0)
            drawdowns = (cumulative - running_max) / running_max
            return np.min(drawdowns, axis=0)
        
        max_drawdowns = calculate_max_drawdown(returns)
        calmar_ratios = mean_returns / np.abs(max_drawdowns)
        
        # Information Ratio vs equal-weight benchmark
        benchmark_weights = np.ones(returns.shape[1]) / returns.shape[1]
        benchmark_returns = returns @ benchmark_weights
        
        active_returns = returns - benchmark_returns[:, np.newaxis]
        tracking_error = np.std(active_returns, axis=0, ddof=1) * np.sqrt(periods_per_year)
        information_ratios = (mean_returns - np.mean(benchmark_returns) * periods_per_year) / tracking_error
        
        # Treynor Ratio (requires beta calculation)
        market_returns = benchmark_returns  # Use benchmark as market proxy
        betas = np.zeros(returns.shape[1])
        
        for i in range(returns.shape[1]):
            covariance = np.cov(returns[:, i], market_returns)[0, 1]
            market_variance = np.var(market_returns, ddof=1)
            betas[i] = covariance / market_variance if market_variance > 0 else 0
        
        treynor_ratios = excess_returns / betas
        
        return {
            'sharpe_ratios': sharpe_ratios,
            'sortino_ratios': sortino_ratios,
            'calmar_ratios': calmar_ratios,
            'information_ratios': information_ratios,
            'treynor_ratios': treynor_ratios,
            'max_drawdowns': max_drawdowns,
            'betas': betas
        }
    
    risk_metrics = calculate_risk_adjusted_metrics(returns)
    
    print("Average Risk-Adjusted Metrics:")
    for metric_name, metric_values in risk_metrics.items():
        if metric_name not in ['max_drawdowns']:  # Skip drawdowns (already negative)
            avg_metric = np.mean(metric_values)
            print(f"  {metric_name:20}: {avg_metric:.3f}")
    
    print(f"  {'max_drawdowns':20}: {np.mean(risk_metrics['max_drawdowns']):.2%}")
    
    # 4. Portfolio-level risk metrics
    print("\n4. Portfolio Risk Analysis:")
    
    def portfolio_risk_metrics(returns, weights=None):
        """Calculate portfolio-level risk metrics"""
        if weights is None:
            # Equal weight portfolio
            weights = np.ones(returns.shape[1]) / returns.shape[1]
        
        # Portfolio returns
        portfolio_returns = returns @ weights
        
        # Portfolio statistics
        portfolio_mean = np.mean(portfolio_returns) * 252
        portfolio_vol = np.std(portfolio_returns, ddof=1) * np.sqrt(252)
        
        # Portfolio VaR and ES
        portfolio_var_95 = np.percentile(portfolio_returns, 5)
        portfolio_var_99 = np.percentile(portfolio_returns, 1)
        
        # Expected Shortfall
        tail_95 = portfolio_returns[portfolio_returns <= portfolio_var_95]
        tail_99 = portfolio_returns[portfolio_returns <= portfolio_var_99]
        
        portfolio_es_95 = np.mean(tail_95) if len(tail_95) > 0 else portfolio_var_95
        portfolio_es_99 = np.mean(tail_99) if len(tail_99) > 0 else portfolio_var_99
        
        # Component VaR (marginal contribution to portfolio VaR)
        # Approximate using finite differences
        epsilon = 0.01  # 1% weight change
        component_var = np.zeros(len(weights))
        
        base_var = portfolio_var_95
        
        for i in range(len(weights)):
            # Create modified weights
            weights_modified = weights.copy()
            weights_modified[i] += epsilon
            weights_modified = weights_modified / np.sum(weights_modified)  # Renormalize
            
            # Calculate modified portfolio VaR
            modified_returns = returns @ weights_modified
            modified_var = np.percentile(modified_returns, 5)
            
            # Component VaR
            component_var[i] = (modified_var - base_var) / epsilon
        
        return {
            'portfolio_return': portfolio_mean,
            'portfolio_volatility': portfolio_vol,
            'portfolio_var_95': portfolio_var_95,
            'portfolio_var_99': portfolio_var_99,
            'portfolio_es_95': portfolio_es_95,
            'portfolio_es_99': portfolio_es_99,
            'component_var': component_var,
            'portfolio_returns': portfolio_returns
        }
    
    portfolio_metrics = portfolio_risk_metrics(returns)
    
    print("Equal-Weight Portfolio Metrics:")
    print(f"  Expected Return: {portfolio_metrics['portfolio_return']:.2%}")
    print(f"  Volatility:      {portfolio_metrics['portfolio_volatility']:.2%}")
    print(f"  VaR (95%):       {portfolio_metrics['portfolio_var_95']:.4f} daily")
    print(f"  VaR (99%):       {portfolio_metrics['portfolio_var_99']:.4f} daily")
    print(f"  ES (95%):        {portfolio_metrics['portfolio_es_95']:.4f} daily")
    print(f"  ES (99%):        {portfolio_metrics['portfolio_es_99']:.4f} daily")
    
    # Top 3 assets contributing most to portfolio VaR
    top_var_contributors = np.argsort(np.abs(portfolio_metrics['component_var']))[-3:]
    print("Top VaR Contributors (assets):", top_var_contributors[::-1])
    
    return {
        'returns': returns,
        'var_results': var_results,
        'es_results': es_results,
        'risk_metrics': risk_metrics,
        'portfolio_metrics': portfolio_metrics
    }

def stress_testing_and_scenario_analysis():
    """
    Stress testing and scenario analysis frameworks
    """
    # Base portfolio setup
    np.random.seed(42)
    n_assets = 8
    n_periods = 1000
    
    # Generate base returns
    base_returns = np.random.multivariate_normal(
        mean=np.array([0.08, 0.10, 0.06, 0.09, 0.07, 0.11, 0.05, 0.08]) / 252,  # Daily expected returns
        cov=np.array([[0.04, 0.02, 0.01, 0.015, 0.01, 0.02, 0.005, 0.015],
                     [0.02, 0.09, 0.025, 0.03, 0.02, 0.04, 0.01, 0.025],
                     [0.01, 0.025, 0.016, 0.02, 0.015, 0.025, 0.005, 0.015],
                     [0.015, 0.03, 0.02, 0.064, 0.025, 0.035, 0.01, 0.02],
                     [0.01, 0.02, 0.015, 0.025, 0.025, 0.03, 0.008, 0.015],
                     [0.02, 0.04, 0.025, 0.035, 0.03, 0.081, 0.015, 0.03],
                     [0.005, 0.01, 0.005, 0.01, 0.008, 0.015, 0.009, 0.01],
                     [0.015, 0.025, 0.015, 0.02, 0.015, 0.03, 0.01, 0.036]]) / 252,  # Daily covariance
        size=n_periods
    )
    
    # Portfolio weights
    weights = np.array([0.15, 0.12, 0.18, 0.10, 0.20, 0.08, 0.12, 0.05])
    
    print("\nStress Testing and Scenario Analysis")
    print("=" * 45)
    
    # 1. Historical Stress Testing
    print("\n1. Historical Stress Testing:")
    
    def historical_stress_test(returns, weights, stress_periods=None):
        """Apply historical stress scenarios"""
        
        # Define major historical stress periods (approximated)
        if stress_periods is None:
            stress_periods = {
                '2008_crisis': {'mean_shock': -0.008, 'vol_multiplier': 3.0, 'correlation_increase': 0.3},
                '2020_covid': {'mean_shock': -0.012, 'vol_multiplier': 2.5, 'correlation_increase': 0.4},
                '2001_dotcom': {'mean_shock': -0.004, 'vol_multiplier': 1.8, 'correlation_increase': 0.2},
                'flash_crash': {'mean_shock': -0.020, 'vol_multiplier': 5.0, 'correlation_increase': 0.8}
            }
        
        stress_results = {}
        base_portfolio_return = np.mean(returns @ weights) * 252
        base_portfolio_vol = np.std(returns @ weights, ddof=1) * np.sqrt(252)
        
        for period_name, scenario in stress_periods.items():
            # Apply stress scenario
            stressed_mean = np.mean(returns, axis=0) + scenario['mean_shock']
            stressed_std = np.std(returns, axis=0, ddof=1) * scenario['vol_multiplier']
            
            # Increase correlations
            base_corr = np.corrcoef(returns.T)
            stressed_corr = base_corr + scenario['correlation_increase'] * (1 - base_corr)
            np.fill_diagonal(stressed_corr, 1.0)
            
            # Ensure positive semi-definite correlation matrix
            eigenvals, eigenvecs = np.linalg.eigh(stressed_corr)
            eigenvals = np.maximum(eigenvals, 0.001)  # Floor at small positive value
            stressed_corr = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            
            # Create stressed covariance matrix
            std_matrix = np.outer(stressed_std, stressed_std)
            stressed_cov = stressed_corr * std_matrix
            
            # Generate stressed returns
            n_stress_scenarios = 1000
            stressed_returns = np.random.multivariate_normal(
                mean=stressed_mean,
                cov=stressed_cov,
                size=n_stress_scenarios
            )
            
            # Portfolio performance under stress
            stressed_portfolio_returns = stressed_returns @ weights
            
            stress_results[period_name] = {
                'mean_return': np.mean(stressed_portfolio_returns) * 252,
                'volatility': np.std(stressed_portfolio_returns, ddof=1) * np.sqrt(252),
                'var_95': np.percentile(stressed_portfolio_returns, 5),
                'var_99': np.percentile(stressed_portfolio_returns, 1),
                'worst_case': np.min(stressed_portfolio_returns),
                'best_case': np.max(stressed_portfolio_returns)
            }
        
        return stress_results, base_portfolio_return, base_portfolio_vol
    
    stress_results, base_return, base_vol = historical_stress_test(base_returns, weights)
    
    print(f"Base Portfolio - Return: {base_return:.2%}, Volatility: {base_vol:.2%}")
    print("\nStress Test Results:")
    for period, results in stress_results.items():
        print(f"\n{period}:")
        print(f"  Return:     {results['mean_return']:>8.2%} (vs {base_return:.2%} base)")
        print(f"  Volatility: {results['volatility']:>8.2%} (vs {base_vol:.2%} base)")
        print(f"  VaR 95%:    {results['var_95']:>8.4f}")
        print(f"  Worst case: {results['worst_case']:>8.4f}")
    
    # 2. Monte Carlo Stress Testing
    print("\n2. Monte Carlo Stress Testing:")
    
    def monte_carlo_stress_test(returns, weights, n_scenarios=10000, confidence_levels=[0.95, 0.99]):
        """Monte Carlo stress testing with various shock distributions"""
        
        base_mean = np.mean(returns, axis=0)
        base_cov = np.cov(returns.T)
        
        stress_scenarios = {}
        
        # Scenario 1: Fat-tail shocks (Student-t distribution)
        t_df = 3  # Degrees of freedom (lower = fatter tails)
        t_shocks = np.random.standard_t(t_df, size=(n_scenarios, len(weights)))
        t_scaling = np.sqrt((t_df - 2) / t_df) if t_df > 2 else 1.0  # Normalize variance
        
        # Apply Cholesky decomposition for correlation
        L = np.linalg.cholesky(base_cov)
        correlated_t_shocks = (t_shocks / t_scaling) @ L.T
        
        fat_tail_returns = base_mean + correlated_t_shocks
        fat_tail_portfolio = fat_tail_returns @ weights
        
        stress_scenarios['fat_tails'] = {
            'returns': fat_tail_portfolio,
            'var_95': np.percentile(fat_tail_portfolio, 5),
            'var_99': np.percentile(fat_tail_portfolio, 1),
            'expected_shortfall_95': np.mean(fat_tail_portfolio[fat_tail_portfolio <= np.percentile(fat_tail_portfolio, 5)])
        }
        
        # Scenario 2: Jump risk
        jump_probability = 0.05  # 5% chance of jump each day
        jump_size_mean = -0.03   # Average jump is -3%
        jump_size_std = 0.02     # Jump volatility is 2%
        
        normal_returns = np.random.multivariate_normal(base_mean, base_cov, n_scenarios)
        jump_indicators = np.random.binomial(1, jump_probability, n_scenarios)
        jump_sizes = np.random.normal(jump_size_mean, jump_size_std, n_scenarios)
        
        jump_portfolio_returns = (normal_returns @ weights) + jump_indicators * jump_sizes
        
        stress_scenarios['jumps'] = {
            'returns': jump_portfolio_returns,
            'var_95': np.percentile(jump_portfolio_returns, 5),
            'var_99': np.percentile(jump_portfolio_returns, 1),
            'expected_shortfall_95': np.mean(jump_portfolio_returns[jump_portfolio_returns <= np.percentile(jump_portfolio_returns, 5)])
        }
        
        # Scenario 3: Correlation breakdown
        # In stress, correlations tend to go to 1 (everything moves together)
        high_corr_matrix = 0.8 * np.ones_like(base_cov) + 0.2 * np.eye(len(base_cov))
        # Scale to maintain individual asset volatilities
        base_vols = np.sqrt(np.diag(base_cov))
        vol_matrix = np.outer(base_vols, base_vols)
        stressed_cov = high_corr_matrix * vol_matrix
        
        corr_breakdown_returns = np.random.multivariate_normal(base_mean, stressed_cov, n_scenarios)
        corr_breakdown_portfolio = corr_breakdown_returns @ weights
        
        stress_scenarios['correlation_breakdown'] = {
            'returns': corr_breakdown_portfolio,
            'var_95': np.percentile(corr_breakdown_portfolio, 5),
            'var_99': np.percentile(corr_breakdown_portfolio, 1),
            'expected_shortfall_95': np.mean(corr_breakdown_portfolio[corr_breakdown_portfolio <= np.percentile(corr_breakdown_portfolio, 5)])
        }
        
        return stress_scenarios
    
    mc_stress_results = monte_carlo_stress_test(base_returns, weights)
    
    print("Monte Carlo Stress Test Results:")
    base_portfolio_returns = base_returns @ weights
    base_var_95 = np.percentile(base_portfolio_returns, 5)
    base_var_99 = np.percentile(base_portfolio_returns, 1)
    
    print(f"Base Portfolio VaR 95%: {base_var_95:.4f}, VaR 99%: {base_var_99:.4f}")
    
    for scenario_name, results in mc_stress_results.items():
        print(f"\n{scenario_name}:")
        print(f"  VaR 95%: {results['var_95']:>8.4f} (vs {base_var_95:.4f} base)")
        print(f"  VaR 99%: {results['var_99']:>8.4f} (vs {base_var_99:.4f} base)")
        print(f"  ES 95%:  {results['expected_shortfall_95']:>8.4f}")
    
    # 3. Factor-based stress testing
    print("\n3. Factor-Based Stress Testing:")
    
    def factor_stress_test(returns, weights):
        """Stress test based on factor shocks"""
        
        # Simple 3-factor model (Market, Size, Value)
        # Estimate factor loadings using PCA as approximation
        correlation_matrix = np.corrcoef(returns.T)
        eigenvals, eigenvecs = np.linalg.eig(correlation_matrix)
        
        # Sort by eigenvalues (descending)
        idx = np.argsort(eigenvals)[::-1]
        eigenvals_sorted = eigenvals[idx]
        eigenvecs_sorted = eigenvecs[:, idx]
        
        # Use first 3 eigenvectors as factor loadings
        factor_loadings = eigenvecs_sorted[:, :3]
        
        # Project returns onto factor space
        factor_returns = returns @ factor_loadings
        
        # Define factor stress scenarios
        factor_scenarios = {
            'market_crash': np.array([-0.05, 0.001, 0.001]),  # -5% market shock
            'size_crisis': np.array([0.001, -0.03, 0.001]),   # Small cap underperformance
            'value_rotation': np.array([0.001, 0.001, -0.02]), # Value underperformance
            'systemic_crisis': np.array([-0.03, -0.02, -0.015]) # All factors negative
        }
        
        factor_stress_results = {}
        
        for scenario_name, factor_shocks in factor_scenarios.items():
            # Apply factor shocks
            shocked_factor_returns = factor_returns + factor_shocks
            
            # Map back to asset space (approximate)
            shocked_asset_returns = shocked_factor_returns @ factor_loadings.T
            
            # Portfolio impact
            portfolio_shock = np.mean(shocked_asset_returns @ weights, axis=0)
            
            factor_stress_results[scenario_name] = {
                'portfolio_shock': portfolio_shock,
                'factor_shocks': factor_shocks
            }
        
        return factor_stress_results, factor_loadings
    
    factor_results, factor_loadings = factor_stress_test(base_returns, weights)
    
    print("Factor-Based Stress Test Results:")
    for scenario, results in factor_results.items():
        shock = results['portfolio_shock']
        print(f"{scenario:20}: {shock:>8.4f} daily ({shock*252:>6.1%} annualized)")
    
    return {
        'base_returns': base_returns,
        'weights': weights,
        'stress_results': stress_results,
        'mc_stress_results': mc_stress_results,
        'factor_results': factor_results,
        'factor_loadings': factor_loadings
    }
```

---

## Matrix Operations for Portfolio Theory

### 1. Modern Portfolio Theory Implementation
```python
def modern_portfolio_theory_implementation():
    """
    Complete implementation of Modern Portfolio Theory using NumPy
    
    ESSENTIAL: This is the mathematical foundation of portfolio management
    """
    # Generate realistic asset universe
    np.random.seed(42)
    n_assets = 12
    asset_names = [f'Asset_{i+1}' for i in range(n_assets)]
    
    # Create realistic expected returns (annual)
    expected_returns = np.random.uniform(0.05, 0.18, n_assets)
    
    # Create realistic correlation structure
    # Generate correlation matrix using random factor loadings
    n_factors = 3
    factor_loadings = np.random.uniform(-1, 1, (n_assets, n_factors))
    idiosyncratic_var = np.random.uniform(0.01, 0.04, n_assets)
    
    # Correlation matrix from factor model: R = FF' + D
    correlation_matrix = factor_loadings @ factor_loadings.T + np.diag(idiosyncratic_var)
    
    # Normalize to correlation matrix
    vol_matrix = np.sqrt(np.outer(np.diag(correlation_matrix), np.diag(correlation_matrix)))
    correlation_matrix = correlation_matrix / vol_matrix
    np.fill_diagonal(correlation_matrix, 1.0)
    
    # Asset volatilities (annual)
    asset_volatilities = np.random.uniform(0.12, 0.35, n_assets)
    
    # Covariance matrix
    vol_outer = np.outer(asset_volatilities, asset_volatilities)
    covariance_matrix = correlation_matrix * vol_outer
    
    print("Modern Portfolio Theory Implementation")
    print("=" * 45)
    
    # 1. Efficient Frontier Calculation
    print("\n1. Efficient Frontier Construction:")
    
    def calculate_efficient_frontier(expected_returns, covariance_matrix, n_points=50, 
                                   risk_free_rate=0.03):
        """Calculate efficient frontier using quadratic programming approach"""
        
        n_assets = len(expected_returns)
        
        # Portfolio optimization functions
        def portfolio_variance(weights, cov_matrix):
            return np.dot(weights.T, np.dot(cov_matrix, weights))
        
        def portfolio_return(weights, exp_returns):
            return np.dot(weights, exp_returns)
        
        # Key portfolio calculations
        ones = np.ones(n_assets)
        
        try:
            # Inverse covariance matrix
            inv_cov = np.linalg.inv(covariance_matrix)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if singular
            inv_cov = np.linalg.pinv(covariance_matrix)
            print("Warning: Using pseudo-inverse for covariance matrix")
        
        # Portfolio theory constants
        A = ones.T @ inv_cov @ expected_returns
        B = expected_returns.T @ inv_cov @ expected_returns  
        C = ones.T @ inv_cov @ ones
        D = B * C - A**2
        
        # Minimum variance portfolio
        min_var_weights = (inv_cov @ ones) / C
        min_var_return = A / C
        min_var_variance = 1 / C
        
        # Maximum Sharpe ratio portfolio (tangency portfolio)
        excess_returns = expected_returns - risk_free_rate
        tangency_weights_unnorm = inv_cov @ excess_returns
        tangency_weights = tangency_weights_unnorm / np.sum(tangency_weights_unnorm)
        tangency_return = portfolio_return(tangency_weights, expected_returns)
        tangency_variance = portfolio_variance(tangency_weights, covariance_matrix)
        tangency_sharpe = (tangency_return - risk_free_rate) / np.sqrt(tangency_variance)
        
        # Efficient frontier points
        min_return = min_var_return
        max_return = np.max(expected_returns) * 1.2  # 20% above highest individual return
        target_returns = np.linspace(min_return, max_return, n_points)
        
        efficient_weights = np.zeros((n_points, n_assets))
        efficient_variances = np.zeros(n_points)
        efficient_sharpe_ratios = np.zeros(n_points)
        
        for i, target_return in enumerate(target_returns):
            try:
                # Solve constrained optimization: min w'Σw s.t. w'μ = target, w'1 = 1
                # Using method of Lagrange multipliers
                
                # Set up constraint matrix and vector
                constraint_matrix = np.array([expected_returns, ones])
                constraint_vector = np.array([target_return, 1.0])
                
                # Solve: [constraint_matrix @ inv_cov @ constraint_matrix.T] @ λ = constraint_vector
                A_matrix = constraint_matrix @ inv_cov @ constraint_matrix.T
                lambdas = np.linalg.solve(A_matrix, constraint_vector)
                
                # Calculate optimal weights
                weights = inv_cov @ constraint_matrix.T @ lambdas
                
                # Store results
                efficient_weights[i] = weights
                efficient_variances[i] = portfolio_variance(weights, covariance_matrix)
                
                # Calculate Sharpe ratio
                excess_return = target_return - risk_free_rate
                efficient_sharpe_ratios[i] = excess_return / np.sqrt(efficient_variances[i])
                
            except np.linalg.LinAlgError:
                # Handle numerical issues
                efficient_weights[i] = np.full(n_assets, np.nan)
                efficient_variances[i] = np.nan
                efficient_sharpe_ratios[i] = np.nan
        
        return {
            'target_returns': target_returns,
            'efficient_variances': efficient_variances,
            'efficient_volatilities': np.sqrt(efficient_variances),
            'efficient_weights': efficient_weights,
            'efficient_sharpe_ratios': efficient_sharpe_ratios,
            'min_var_portfolio': {
                'weights': min_var_weights,
                'return': min_var_return,
                'variance': min_var_variance,
                'volatility': np.sqrt(min_var_variance)
            },
            'tangency_portfolio': {
                'weights': tangency_weights,
                'return': tangency_return,
                'variance': tangency_variance,
                'volatility': np.sqrt(tangency_variance),
                'sharpe_ratio': tangency_sharpe
            }
        }
    
    efficient_frontier = calculate_efficient_frontier(expected_returns, covariance_matrix)
    
    # Display key portfolio results
    print("Minimum Variance Portfolio:")
    min_var = efficient_frontier['min_var_portfolio']
    print(f"  Expected Return: {min_var['return']:.2%}")
    print(f"  Volatility:      {min_var['volatility']:.2%}")
    print(f"  Top 3 holdings:  {np.argsort(min_var['weights'])[-3:][::-1]}")
    
    print("\nTangency Portfolio (Max Sharpe):")
    tangency = efficient_frontier['tangency_portfolio']
    print(f"  Expected Return: {tangency['return']:.2%}")
    print(f"  Volatility:      {tangency['volatility']:.2%}")
    print(f"  Sharpe Ratio:    {tangency['sharpe_ratio']:.3f}")
    print(f"  Top 3 holdings:  {np.argsort(tangency['weights'])[-3:][::-1]}")
    
    # 2. Black-Litterman Model
    print("\n2. Black-Litterman Model:")
    
    def black_litterman_model(expected_returns, covariance_matrix, 
                            market_cap_weights=None, risk_aversion=3.0,
                            views=None, view_uncertainty=None, tau=0.025):
        """Implement Black-Litterman model for expected return estimation"""
        
        n_assets = len(expected_returns)
        
        if market_cap_weights is None:
            # Use equal weights as proxy for market cap weights
            market_cap_weights = np.ones(n_assets) / n_assets
        
        # Step 1: Implied equilibrium returns (reverse optimization)
        # π = λ * Σ * w_market
        implied_returns = risk_aversion * covariance_matrix @ market_cap_weights
        
        # Step 2: Incorporate investor views
        if views is not None and view_uncertainty is not None:
            # P: picking matrix (which assets the views relate to)
            # Q: view portfolio expected returns
            # Ω: uncertainty in views (diagonal matrix)
            
            P = views['picking_matrix']  # Shape: (n_views, n_assets)
            Q = views['expected_returns']  # Shape: (n_views,)
            Omega = np.diag(view_uncertainty)  # Shape: (n_views, n_views)
            
            # Black-Litterman formula
            # μ_BL = [(τΣ)^-1 + P'Ω^-1P]^-1 * [(τΣ)^-1*π + P'Ω^-1*Q]
            
            tau_cov_inv = np.linalg.inv(tau * covariance_matrix)
            P_omega_inv_P = P.T @ np.linalg.inv(Omega) @ P
            
            # New expected returns
            bl_precision = tau_cov_inv + P_omega_inv_P
            bl_mean = (tau_cov_inv @ implied_returns + 
                      P.T @ np.linalg.inv(Omega) @ Q)
            
            bl_expected_returns = np.linalg.inv(bl_precision) @ bl_mean
            
            # New covariance matrix (incorporates estimation uncertainty)
            bl_covariance = np.linalg.inv(bl_precision)
            
        else:
            # No views: use implied returns
            bl_expected_returns = implied_returns
            bl_covariance = tau * covariance_matrix
        
        return {
            'implied_returns': implied_returns,
            'bl_expected_returns': bl_expected_returns,
            'bl_covariance': bl_covariance,
            'original_expected_returns': expected_returns
        }
    
    # Example: Create some views
    sample_views = {
        'picking_matrix': np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Bullish on Asset_1
            [0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Asset_2 outperforms Asset_3
        ]),
        'expected_returns': np.array([0.15, 0.02])  # 15% return for Asset_1, 2% outperformance
    }
    
    view_uncertainty = np.array([0.001, 0.0005])  # Lower values = more confident
    
    bl_results = black_litterman_model(
        expected_returns, covariance_matrix,
        views=sample_views, view_uncertainty=view_uncertainty
    )
    
    print("Black-Litterman Results:")
    print("Original vs BL Expected Returns (top 5 assets):")
    for i in range(5):
        orig = expected_returns[i]
        bl = bl_results['bl_expected_returns'][i]
        implied = bl_results['implied_returns'][i]
        print(f"  Asset_{i+1}: Original={orig:.2%}, Implied={implied:.2%}, BL={bl:.2%}")
    
    # 3. Risk Budgeting and Equal Risk Contribution
    print("\n3. Risk Budgeting Portfolio:")
    
    def equal_risk_contribution_portfolio(covariance_matrix, max_iterations=1000, tolerance=1e-8):
        """Calculate Equal Risk Contribution (ERC) portfolio using iterative algorithm"""
        
        n_assets = covariance_matrix.shape[0]
        
        # Initialize with equal weights
        weights = np.ones(n_assets) / n_assets
        
        for iteration in range(max_iterations):
            # Calculate risk contributions
            portfolio_variance = weights.T @ covariance_matrix @ weights
            marginal_contrib = covariance_matrix @ weights
            risk_contrib = weights * marginal_contrib / portfolio_variance
            
            # Target risk contribution (equal for all assets)
            target_risk_contrib = 1 / n_assets
            
            # Update weights using Newton-Raphson-like approach
            # Simplified update rule for ERC
            adjustment_factor = target_risk_contrib / risk_contrib
            new_weights = weights * adjustment_factor
            new_weights = new_weights / np.sum(new_weights)  # Renormalize
            
            # Check convergence
            weight_change = np.sum(np.abs(new_weights - weights))
            if weight_change < tolerance:
                print(f"ERC converged in {iteration + 1} iterations")
                break
                
            weights = new_weights
        
        # Final risk contributions
        portfolio_variance = weights.T @ covariance_matrix @ weights
        marginal_contrib = covariance_matrix @ weights
        risk_contrib = weights * marginal_contrib / portfolio_variance
        
        return {
            'weights': weights,
            'risk_contributions': risk_contrib,
            'portfolio_variance': portfolio_variance,
            'converged': iteration < max_iterations - 1
        }
    
    erc_results = equal_risk_contribution_portfolio(covariance_matrix)
    
    print("Equal Risk Contribution Portfolio:")
    print(f"  Converged: {erc_results['converged']}")
    print(f"  Portfolio Volatility: {np.sqrt(erc_results['portfolio_variance']):.2%}")
    print("  Risk Contributions (should be equal):")
    for i in range(min(5, n_assets)):
        weight = erc_results['weights'][i]
        risk_contrib = erc_results['risk_contributions'][i]
        print(f"    Asset_{i+1}: Weight={weight:.1%}, Risk Contrib={risk_contrib:.1%}")
    
    return {
        'expected_returns': expected_returns,
        'covariance_matrix': covariance_matrix,
        'asset_volatilities': asset_volatilities,
        'efficient_frontier': efficient_frontier,
        'bl_results': bl_results,
        'erc_results': erc_results
    }
```

### 2. Advanced Portfolio Optimization
```python
def advanced_portfolio_optimization():
    """
    Advanced portfolio optimization techniques beyond mean-variance
    """
    # Setup: Generate realistic multi-asset universe
    np.random.seed(42)
    n_assets = 15
    n_periods = 1000  # Historical data for robust optimization
    
    # Generate historical returns with realistic properties
    # Use factor model to create realistic correlation structure
    n_factors = 4
    factor_returns = np.random.multivariate_normal(
        mean=[0.0005, 0.0003, 0.0001, 0.0002],  # Factor expected returns
        cov=np.diag([0.0004, 0.0002, 0.0001, 0.0003]),  # Factor variances
        size=n_periods
    )
    
    # Asset factor loadings
    factor_loadings = np.random.normal(0, 0.8, (n_assets, n_factors))
    factor_loadings[:, 0] = np.abs(factor_loadings[:, 0])  # Market factor (positive)
    
    # Generate asset returns
    systematic_returns = factor_returns @ factor_loadings.T
    idiosyncratic_returns = np.random.normal(0, 0.015, (n_periods, n_assets))
    returns = systematic_returns + idiosyncratic_returns
    
    print("Advanced Portfolio Optimization")
    print("=" * 40)
    
    # 1. Robust Portfolio Optimization
    print("\n1. Robust Portfolio Optimization:")
    
    def robust_portfolio_optimization(returns, uncertainty_level=0.1):
        """Implement robust optimization accounting for parameter uncertainty"""
        
        # Sample estimates
        mu_hat = np.mean(returns, axis=0)
        sigma_hat = np.cov(returns.T)
        
        # Uncertainty sets (ellipsoidal uncertainty)
        n_assets = len(mu_hat)
        n_samples = len(returns)
        
        # Standard errors for mean estimates
        mu_std_errors = np.sqrt(np.diag(sigma_hat) / n_samples)
        
        # Create uncertainty ellipsoid for expected returns
        # Worst-case expected returns within confidence ellipsoid
        def robust_mean_variance_optimization(gamma_uncertainty=uncertainty_level):
            """Solve robust mean-variance optimization"""
            
            # This is a simplified robust optimization
            # In practice, would use convex optimization solvers
            
            # Shrink expected returns towards grand mean (robust approach)
            grand_mean = np.mean(mu_hat)
            robust_mu = (1 - gamma_uncertainty) * mu_hat + gamma_uncertainty * grand_mean
            
            # Inflate covariance matrix to account for uncertainty
            robust_sigma = sigma_hat * (1 + gamma_uncertainty)
            
            # Solve standard mean-variance with robust parameters
            ones = np.ones(n_assets)
            
            try:
                inv_sigma = np.linalg.inv(robust_sigma)
            except np.linalg.LinAlgError:
                inv_sigma = np.linalg.pinv(robust_sigma)
            
            # Minimum variance portfolio (robust)
            robust_min_var_weights = (inv_sigma @ ones) / (ones.T @ inv_sigma @ ones)
            
            # Maximum expected return portfolio subject to risk constraint
            # Target volatility = 15% annual
            target_vol = 0.15
            target_variance = target_vol**2
            
            # Optimize: max μ'w subject to w'Σw ≤ target_variance, w'1 = 1
            # Lagrangian approach (simplified)
            
            # Calculate efficient frontier points
            A = ones.T @ inv_sigma @ robust_mu
            B = robust_mu.T @ inv_sigma @ robust_mu
            C = ones.T @ inv_sigma @ ones
            D = B * C - A**2
            
            if D > 1e-10:  # Check for valid discriminant
                # Risk-constrained portfolio
                lambda_opt = (A - np.sqrt(A**2 - C * (B - target_variance * C))) / C
                risk_constrained_weights = (inv_sigma @ robust_mu - lambda_opt * inv_sigma @ ones) / (A - lambda_opt * C)
                
                # Normalize to sum to 1
                risk_constrained_weights = risk_constrained_weights / np.sum(risk_constrained_weights)
            else:
                risk_constrained_weights = robust_min_var_weights
            
            return {
                'robust_min_var_weights': robust_min_var_weights,
                'risk_constrained_weights': risk_constrained_weights,
                'robust_expected_returns': robust_mu,
                'robust_covariance': robust_sigma
            }
        
        return robust_mean_variance_optimization()
    
    robust_results = robust_portfolio_optimization(returns)
    
    # Compare robust vs standard optimization
    standard_mu = np.mean(returns, axis=0)
    standard_sigma = np.cov(returns.T)
    
    # Standard minimum variance portfolio
    ones = np.ones(len(standard_mu))
    inv_sigma_std = np.linalg.inv(standard_sigma)
    standard_min_var_weights = (inv_sigma_std @ ones) / (ones.T @ inv_sigma_std @ ones)
    
    print("Portfolio Comparison (Robust vs Standard):")
    print("Minimum Variance Weights (top 5 assets):")
    for i in range(5):
        robust_w = robust_results['robust_min_var_weights'][i]
        standard_w = standard_min_var_weights[i]
        print(f"  Asset_{i+1}: Robust={robust_w:.1%}, Standard={standard_w:.1%}")
    
    # 2. CVaR Optimization
    print("\n2. Conditional Value at Risk (CVaR) Optimization:")
    
    def cvar_optimization(returns, confidence_level=0.95, n_scenarios=1000):
        """Portfolio optimization using CVaR as risk measure"""
        
        # Use historical returns as scenarios
        scenarios = returns[-n_scenarios:] if len(returns) > n_scenarios else returns
        n_scenarios_actual, n_assets = scenarios.shape
        
        # CVaR optimization using linear programming approximation
        alpha = 1 - confidence_level
        
        def calculate_cvar_portfolio(target_return=None):
            """Calculate CVaR-optimal portfolio"""
            
            # This is a simplified CVaR optimization
            # In practice, would use specialized solvers like CVXPY
            
            # Method: Minimize average of worst alpha% of outcomes
            n_worst = max(1, int(alpha * n_scenarios_actual))
            
            # For each possible weight vector (simplified grid search)
            # In practice, use proper LP/QP solver
            
            best_weights = None
            best_cvar = np.inf
            
            # Generate candidate portfolios
            n_candidates = 500
            candidate_weights = np.random.dirichlet(np.ones(n_assets), n_candidates)
            
            for weights in candidate_weights:
                # Calculate portfolio returns for all scenarios
                portfolio_returns = scenarios @ weights
                
                # Calculate CVaR
                sorted_returns = np.sort(portfolio_returns)
                cvar = np.mean(sorted_returns[:n_worst])
                
                # Check return constraint if specified
                if target_return is not None:
                    expected_portfolio_return = np.mean(portfolio_returns)
                    if expected_portfolio_return < target_return:
                        continue
                
                if cvar > best_cvar:  # We want to maximize CVaR (minimize losses)
                    best_cvar = cvar
                    best_weights = weights
            
            if best_weights is None:
                # Fallback to equal weights
                best_weights = np.ones(n_assets) / n_assets
            
            # Calculate final statistics
            portfolio_returns = scenarios @ best_weights
            var_95 = np.percentile(portfolio_returns, 5)
            cvar_95 = np.mean(portfolio_returns[portfolio_returns <= var_95])
            expected_return = np.mean(portfolio_returns)
            volatility = np.std(portfolio_returns)
            
            return {
                'weights': best_weights,
                'expected_return': expected_return,
                'volatility': volatility,
                'var_95': var_95,
                'cvar_95': cvar_95
            }
        
        # Optimize for CVaR without return constraint
        cvar_optimal = calculate_cvar_portfolio()
        
        return cvar_optimal
    
    cvar_results = cvar_optimization(returns)
    
    print("CVaR-Optimal Portfolio:")
    print(f"  Expected Return: {cvar_results['expected_return'] * 252:.2%}")
    print(f"  Volatility:      {cvar_results['volatility'] * np.sqrt(252):.2%}")
    print(f"  VaR (95%):       {cvar_results['var_95']:.4f}")
    print(f"  CVaR (95%):      {cvar_results['cvar_95']:.4f}")
    print("  Top 3 holdings:", np.argsort(cvar_results['weights'])[-3:][::-1])
    
    # 3. Maximum Diversification Portfolio
    print("\n3. Maximum Diversification Portfolio:")
    
    def maximum_diversification_portfolio(returns):
        """Calculate Maximum Diversification Portfolio"""
        
        # Diversification ratio = (w' * σ) / sqrt(w' * Σ * w)
        # where σ is vector of individual asset volatilities
        # and Σ is covariance matrix
        
        asset_volatilities = np.std(returns, axis=0, ddof=1)
        covariance_matrix = np.cov(returns.T)
        n_assets = len(asset_volatilities)
        
        def diversification_ratio(weights):
            """Calculate diversification ratio for given weights"""
            weighted_vol_sum = np.sum(weights * asset_volatilities)
            portfolio_vol = np.sqrt(weights.T @ covariance_matrix @ weights)
            return weighted_vol_sum / portfolio_vol if portfolio_vol > 0 else 0
        
        # Optimization: maximize diversification ratio
        # This is equivalent to maximizing w'σ subject to w'Σw = constant
        
        # Solve using quadratic programming approximation
        # max w'σ subject to w'Σw ≤ 1, w'1 = 1, w ≥ 0
        
        try:
            inv_cov = np.linalg.inv(covariance_matrix)
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.pinv(covariance_matrix)
        
        # Analytical solution for maximum diversification
        # Proportional to Σ^(-1) * σ
        unnormalized_weights = inv_cov @ asset_volatilities
        max_div_weights = unnormalized_weights / np.sum(unnormalized_weights)
        
        # Calculate diversification ratio
        div_ratio = diversification_ratio(max_div_weights)
        
        # Portfolio statistics
        portfolio_return = np.mean(returns @ max_div_weights)
        portfolio_vol = np.std(returns @ max_div_weights, ddof=1)
        
        return {
            'weights': max_div_weights,
            'diversification_ratio': div_ratio,
            'expected_return': portfolio_return,
            'volatility': portfolio_vol,
            'individual_volatilities': asset_volatilities
        }
    
    max_div_results = maximum_diversification_portfolio(returns)
    
    print("Maximum Diversification Portfolio:")
    print(f"  Diversification Ratio: {max_div_results['diversification_ratio']:.3f}")
    print(f"  Expected Return:       {max_div_results['expected_return'] * 252:.2%}")
    print(f"  Volatility:           {max_div_results['volatility'] * np.sqrt(252):.2%}")
    print("  Top 3 holdings:", np.argsort(max_div_results['weights'])[-3:][::-1])
    
    # Compare concentration across different approaches
    print("\n4. Portfolio Concentration Comparison:")
    
    def calculate_concentration_metrics(weights):
        """Calculate various concentration metrics"""
        # Herfindahl-Hirschman Index (HHI)
        hhi = np.sum(weights**2)
        
        # Effective number of assets
        eff_n_assets = 1 / hhi
        
        # Maximum weight
        max_weight = np.max(weights)
        
        # Entropy (diversification measure)
        entropy = -np.sum(weights * np.log(weights + 1e-10))  # Add small value to avoid log(0)
        
        return {
            'hhi': hhi,
            'effective_n_assets': eff_n_assets,
            'max_weight': max_weight,
            'entropy': entropy
        }
    
    # Equal weight portfolio for comparison
    equal_weights = np.ones(len(returns[0])) / len(returns[0])
    
    portfolios = {
        'Equal Weight': equal_weights,
        'Robust Min Var': robust_results['robust_min_var_weights'],
        'CVaR Optimal': cvar_results['weights'],
        'Max Diversification': max_div_results['weights']
    }
    
    print("Concentration Metrics:")
    print(f"{'Portfolio':20} {'HHI':>6} {'Eff.N':>6} {'MaxW':>6} {'Entropy':>8}")
    print("-" * 50)
    
    for name, weights in portfolios.items():
        metrics = calculate_concentration_metrics(weights)
        print(f"{name:20} {metrics['hhi']:>6.3f} {metrics['effective_n_assets']:>6.1f} "
              f"{metrics['max_weight']:>5.1%} {metrics['entropy']:>8.3f}")
    
    return {
        'returns': returns,
        'robust_results': robust_results,
        'cvar_results': cvar_results,
        'max_div_results': max_div_results,
        'portfolios': portfolios
    }
```

---

## Advanced Numerical Techniques

### 1. Numerical Methods for Finance
```python
def numerical_methods_finance():
    """
    Advanced numerical methods commonly used in quantitative finance
    """
    import numpy as np
    from scipy import optimize, integrate
    
    print("Advanced Numerical Methods for Finance")
    print("=" * 45)
    
    # 1. Root Finding for Implied Volatility
    print("\n1. Implied Volatility Calculation (Newton-Raphson):")
    
    def black_scholes_call_price(S, K, T, r, sigma):
        """Black-Scholes call option price"""
        from scipy.stats import norm
        
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        call_price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        return call_price
    
    def black_scholes_vega(S, K, T, r, sigma):
        """Vega (sensitivity to volatility) of Black-Scholes call"""
        from scipy.stats import norm
        
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T)
        return vega
    
    def implied_volatility_newton(market_price, S, K, T, r, max_iterations=100, tolerance=1e-6):
        """Calculate implied volatility using Newton-Raphson method"""
        
        # Initial guess
        sigma = 0.2  # 20% volatility
        
        for i in range(max_iterations):
            # Calculate price and vega at current sigma
            bs_price = black_scholes_call_price(S, K, T, r, sigma)
            vega = black_scholes_vega(S, K, T, r, sigma)
            
            # Check convergence
            price_diff = bs_price - market_price
            if abs(price_diff) < tolerance:
                return sigma, i + 1
            
            # Newton-Raphson update
            if vega > tolerance:  # Avoid division by very small numbers
                sigma = sigma - price_diff / vega
            else:
                break
            
            # Ensure sigma stays positive
            sigma = max(sigma, 0.001)  # Minimum 0.1% volatility
        
        return sigma, max_iterations  # Return last value if not converged
    
    # Example: Calculate implied volatility
    S0 = 100    # Stock price
    K = 105     # Strike price  
    T = 0.25    # Time to expiration (3 months)
    r = 0.05    # Risk-free rate
    market_price = 2.5  # Observed market price
    
    implied_vol, iterations = implied_volatility_newton(market_price, S0, K, T, r)
    print(f"Implied Volatility: {implied_vol:.2%}")
    print(f"Converged in {iterations} iterations")
    
    # Verify by pricing back
    calculated_price = black_scholes_call_price(S0, K, T, r, implied_vol)
    print(f"Market Price: {market_price:.4f}, Calculated Price: {calculated_price:.4f}")
    
    # 2. Numerical Integration for Path-Dependent Options
    print("\n2. Numerical Integration (Asian Option Pricing):")
    
    def asian_option_integrand(S_avg, S0, K, T, r, sigma, n_steps):
        """Integrand for Asian option pricing (simplified)"""
        # This is a simplified version - full implementation would use 
        # multi-dimensional integration or Monte Carlo
        
        # For demonstration, use approximation based on lognormal assumption
        # for arithmetic average (this is not exact but illustrative)
        
        drift_adjustment = (r - 0.5 * sigma**2) * T
        diffusion_adjustment = sigma * np.sqrt(T / 3)  # Rough adjustment for averaging
        
        adjusted_forward = S0 * np.exp(drift_adjustment)
        adjusted_vol = diffusion_adjustment
        
        from scipy.stats import norm
        d1 = (np.log(adjusted_forward / K)) / adjusted_vol + 0.5 * adjusted_vol
        d2 = d1 - adjusted_vol
        
        asian_price = np.exp(-r * T) * (adjusted_forward * norm.cdf(d1) - K * norm.cdf(d2))
        return max(asian_price, 0)
    
    def asian_option_price_numerical(S0, K, T, r, sigma, n_steps=252):
        """Price Asian option using numerical methods"""
        
        # For simplicity, use closed-form approximation
        # In practice, would use Monte Carlo or PDE methods
        
        # Geometric Asian option (exact formula exists)
        sigma_adj = sigma / np.sqrt(3)
        r_adj = 0.5 * (r + sigma**2 / 6)
        
        geo_asian_price = black_scholes_call_price(S0, K, T, r_adj, sigma_adj)
        
        # Arithmetic Asian is typically higher than geometric
        # Use approximation factor
        arithmetic_adjustment = 1.05  # Rule of thumb
        
        return geo_asian_price * arithmetic_adjustment
    
    asian_price = asian_option_price_numerical(S0, K, T, r, 0.2)
    european_price = black_scholes_call_price(S0, K, T, r, 0.2)
    
    print(f"European Call Price: {european_price:.4f}")
    print(f"Asian Call Price:    {asian_price:.4f}")
    print(f"Asian Premium:       {(asian_price/european_price - 1)*100:.1f}%")
    
    # 3. Optimization for Portfolio Construction
    print("\n3. Constrained Optimization (Portfolio with Constraints):")
    
    def constrained_portfolio_optimization():
        """Portfolio optimization with realistic constraints"""
        
        # Sample data
        np.random.seed(42)
        n_assets = 8
        expected_returns = np.random.uniform(0.05, 0.15, n_assets)
        
        # Create covariance matrix
        A = np.random.normal(0, 1, (n_assets, n_assets))
        covariance_matrix = A @ A.T * 0.01  # Scale to reasonable values
        
        # Objective function: minimize portfolio variance
        def objective(weights):
            return 0.5 * weights.T @ covariance_matrix @ weights
        
        # Gradient of objective
        def gradient(weights):
            return covariance_matrix @ weights
        
        # Constraints
        def return_constraint(weights):
            return weights @ expected_returns - 0.08  # Target 8% return
        
        def weight_sum_constraint(weights):
            return np.sum(weights) - 1.0  # Weights sum to 1
        
        # Bounds: long-only portfolio with max 30% in any asset
        bounds = [(0.0, 0.3) for _ in range(n_assets)]
        
        # Set up constraints
        constraints = [
            {'type': 'eq', 'fun': weight_sum_constraint},
            {'type': 'eq', 'fun': return_constraint}
        ]
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Solve optimization
        result = optimize.minimize(
            objective, x0, method='SLSQP',
            bounds=bounds, constraints=constraints,
            jac=gradient
        )
        
        if result.success:
            optimal_weights = result.x
            optimal_return = optimal_weights @ expected_returns
            optimal_variance = optimal_weights.T @ covariance_matrix @ optimal_weights
            optimal_volatility = np.sqrt(optimal_variance)
            
            return {
                'weights': optimal_weights,
                'expected_return': optimal_return,
                'volatility': optimal_volatility,
                'optimization_success': True,
                'message': result.message
            }
        else:
            return {'optimization_success': False, 'message': result.message}
    
    portfolio_result = constrained_portfolio_optimization()
    
    if portfolio_result['optimization_success']:
        print("Constrained Portfolio Optimization Results:")
        print(f"  Expected Return: {portfolio_result['expected_return']:.2%}")
        print(f"  Volatility:      {portfolio_result['volatility']:.2%}")
        print("  Weights:")
        for i, w in enumerate(portfolio_result['weights']):
            if w > 0.01:  # Only show significant weights
                print(f"    Asset {i+1}: {w:.1%}")
    else:
        print(f"Optimization failed: {portfolio_result['message']}")
    
    # 4. Finite Difference Methods for Greeks
    print("\n4. Finite Difference Greeks Calculation:")
    
    def calculate_greeks_finite_difference(S, K, T, r, sigma, bump_size=0.01):
        """Calculate option Greeks using finite difference methods"""
        
        # Base price
        base_price = black_scholes_call_price(S, K, T, r, sigma)
        
        # Delta: ∂V/∂S
        price_up = black_scholes_call_price(S * (1 + bump_size), K, T, r, sigma)
        price_down = black_scholes_call_price(S * (1 - bump_size), K, T, r, sigma)
        delta = (price_up - price_down) / (2 * S * bump_size)
        
        # Gamma: ∂²V/∂S²
        gamma = (price_up - 2 * base_price + price_down) / (S * bump_size)**2
        
        # Theta: ∂V/∂T (time decay)
        dt = 1/365  # One day
        if T > dt:
            theta_price = black_scholes_call_price(S, K, T - dt, r, sigma)
            theta = (theta_price - base_price) / dt
        else:
            theta = 0
        
        # Vega: ∂V/∂σ
        vol_bump = 0.01  # 1% volatility bump
        vega_price_up = black_scholes_call_price(S, K, T, r, sigma + vol_bump)
        vega_price_down = black_scholes_call_price(S, K, T, r, sigma - vol_bump)
        vega = (vega_price_up - vega_price_down) / (2 * vol_bump)
        
        # Rho: ∂V/∂r
        rate_bump = 0.0001  # 1 basis point
        rho_price_up = black_scholes_call_price(S, K, T, r + rate_bump, sigma)
        rho_price_down = black_scholes_call_price(S, K, T, r - rate_bump, sigma)
        rho = (rho_price_up - rho_price_down) / (2 * rate_bump)
        
        return {
            'price': base_price,
            'delta': delta,
            'gamma': gamma,
            'theta': theta,  # Per day
            'vega': vega,    # Per 1% vol change
            'rho': rho       # Per 1bp rate change
        }
    
    greeks = calculate_greeks_finite_difference(S0, K, T, r, 0.2)
    
    print("Option Greeks (Finite Difference):")
    print(f"  Price: ${greeks['price']:.4f}")
    print(f"  Delta: {greeks['delta']:.4f}")
    print(f"  Gamma: {greeks['gamma']:.4f}")
    print(f"  Theta: ${greeks['theta']:.4f} per day")
    print(f"  Vega:  ${greeks['vega']:.4f} per 1% vol")
    print(f"  Rho:   ${greeks['rho']:.4f} per 1bp rate")
    
    # 5. Interpolation and Extrapolation
    print("\n5. Yield Curve Interpolation:")
    
    def interpolate_yield_curve():
        """Interpolate yield curve using different methods"""
        
        # Market data: maturities and yields
        maturities = np.array([0.25, 0.5, 1, 2, 5, 10, 30])  # Years
        yields = np.array([0.015, 0.018, 0.022, 0.025, 0.028, 0.030, 0.032])  # Yields
        
        # Target maturities for interpolation
        target_maturities = np.array([0.1, 0.75, 1.5, 3, 7, 15, 20])
        
        # Method 1: Linear interpolation
        linear_yields = np.interp(target_maturities, maturities, yields)
        
        # Method 2: Cubic spline
        from scipy.interpolate import CubicSpline
        cubic_spline = CubicSpline(maturities, yields)
        spline_yields = cubic_spline(target_maturities)
        
        # Method 3: Nelson-Siegel parameterization (simplified)
        def nelson_siegel(tau, beta0, beta1, beta2, lambda_param):
            """Nelson-Siegel yield curve model"""
            term1 = beta0
            term2 = beta1 * (1 - np.exp(-lambda_param * tau)) / (lambda_param * tau)
            term3 = beta2 * ((1 - np.exp(-lambda_param * tau)) / (lambda_param * tau) - np.exp(-lambda_param * tau))
            return term1 + term2 + term3
        
        # Fit Nelson-Siegel parameters (simplified using least squares)
        def ns_objective(params):
            beta0, beta1, beta2, lambda_param = params
            model_yields = nelson_siegel(maturities, beta0, beta1, beta2, lambda_param)
            return np.sum((model_yields - yields)**2)
        
        # Initial parameter guess
        initial_params = [0.03, -0.01, 0.005, 1.0]
        
        ns_result = optimize.minimize(ns_objective, initial_params, method='Nelder-Mead')
        
        if ns_result.success:
            ns_params = ns_result.x
            ns_yields = nelson_siegel(target_maturities, *ns_params)
        else:
            ns_yields = np.full_like(target_maturities, np.nan)
        
        return {
            'target_maturities': target_maturities,
            'linear_yields': linear_yields,
            'spline_yields': spline_yields,
            'nelson_siegel_yields': ns_yields,
            'original_maturities': maturities,
            'original_yields': yields
        }
    
    yield_curve_results = interpolate_yield_curve()
    
    print("Yield Curve Interpolation Results:")
    print(f"{'Maturity':>10} {'Linear':>8} {'Spline':>8} {'N-S':>8}")
    print("-" * 40)
    
    for i, maturity in enumerate(yield_curve_results['target_maturities']):
        linear = yield_curve_results['linear_yields'][i]
        spline = yield_curve_results['spline_yields'][i]
        ns = yield_curve_results['nelson_siegel_yields'][i]
        
        print(f"{maturity:>10.1f} {linear:>7.1%} {spline:>7.1%} {ns:>7.1%}")
    
    return {
        'implied_volatility': implied_vol,
        'asian_pricing': {'asian': asian_price, 'european': european_price},
        'portfolio_optimization': portfolio_result,
        'greeks': greeks,
        'yield_curve': yield_curve_results
    }
```

---

## References & Further Reading

### Academic Sources
- **Campbell, J. Y., Lo, A. W., & MacKinlay, A. C.** (1997). *The Econometrics of Financial Markets*. Princeton University Press.
- **Tsay, R. S.** (2010). *Analysis of Financial Time Series*. Wiley.
- **Wilmott, P.** (2007). *Paul Wilmott Introduces Quantitative Finance*. Wiley.
- **Hull, J. C.** (2017). *Options, Futures, and Other Derivatives*. Pearson.
- **Cochrane, J. H.** (2005). *Asset Pricing*. Princeton University Press.

### NumPy Documentation & Resources
- [NumPy Official Documentation](https://numpy.org/doc/stable/) - User Guide and API Reference
- [NumPy Enhancement Proposals (NEPs)](https://numpy.org/neps/) - Technical specifications and improvements
- **VanderPlas, J.** (2016). *Python Data Science Handbook*. O'Reilly Media. - Chapter 2: NumPy
- **McKinney, W.** (2017). *Python for Data Analysis*. O'Reilly Media. - NumPy fundamentals
- [From Python to Numpy](https://www.labri.fr/perso/nrougier/from-python-to-numpy/) by Nicolas P. Rougier

### Performance & Optimization
- [NumPy Performance Tips](https://numpy.org/doc/stable/user/how-to-build.html) - Official performance guide
- **Gorelick, M. & Ozsvald, I.** (2014). *High Performance Python*. O'Reilly Media.
- [Optimizing NumPy](https://ipython-books.github.io/45-accelerating-pure-python-code-with-numba-and-just-in-time-compilation/) - JIT compilation techniques
- [SciPy Lecture Notes](https://scipy-lectures.org/advanced/optimizing/index.html) - Advanced optimization techniques

### Financial Mathematics & Quantitative Methods
- **Joshi, M. S.** (2003). *The Concepts and Practice of Mathematical Finance*. Cambridge University Press.
- **Glasserman, P.** (2003). *Monte Carlo Methods in Financial Engineering*. Springer.
- **Shreve, S. E.** (2004). *Stochastic Calculus for Finance* (Volumes I & II). Springer.
- **Fabozzi, F. J., Focardi, S. M., & Kolm, P. N.** (2010). *Quantitative Equity Investing*. Wiley.

### Portfolio Theory & Risk Management
- **Markowitz, H.** (1952). "Portfolio Selection." *Journal of Finance*, 7(1), 77-91.
- **Black, F. & Litterman, R.** (1992). "Global Portfolio Optimization." *Financial Analysts Journal*, 48(5), 28-43.
- **Jorion, P.** (2006). *Value at Risk: The New Benchmark for Managing Financial Risk*. McGraw-Hill.
- **Rachev, S. T., Stoyanov, S. V., & Fabozzi, F. J.** (2008). *Advanced Stochastic Models, Risk Assessment, and Portfolio Optimization*. Wiley.

### Numerical Methods & Scientific Computing
- **Press, W. H., Teukolsky, S. A., Vetterling, W. T., & Flannery, B. P.** (2007). *Numerical Recipes: The Art of Scientific Computing*. Cambridge University Press.
- **Burden, R. L. & Faires, J. D.** (2010). *Numerical Analysis*. Brooks Cole.
- **Quarteroni, A., Sacco, R., & Saleri, F.** (2006). *Numerical Mathematics*. Springer.

### Python & Scientific Computing Style Guides  
- [PEP 8](https://www.python.org/dev/peps/pep-0008/) - Style Guide for Python Code
- [NumPy Documentation Style Guide](https://numpydoc.readthedocs.io/en/latest/format.html)
- [SciPy Developer Guide](https://docs.scipy.org/doc/scipy/dev/contributor/contributor_toc.html)

### Online Resources & Courses
- [QuantEcon Lectures](https://quantecon.org/lectures/) - Quantitative economics with Python
- [MIT OpenCourseWare - Financial Mathematics](https://ocw.mit.edu/courses/mathematics/)
- [Coursera - Financial Engineering and Risk Management](https://www.coursera.org/specializations/financial-engineering-computational-finance)
- [arXiv Quantitative Finance](https://arxiv.org/list/q-fin/recent) - Latest research papers

---

## Summary Checklist

### Before Starting Analysis:
- [ ] Import NumPy with standard conventions (`import numpy as np`)
- [ ] Set random seeds for reproducible research (`np.random.seed(42)`)
- [ ] Configure NumPy display options for readable output
- [ ] Validate input data types and shapes
- [ ] Check for NaN, infinite, or missing values

### During Development:
- [ ] Use vectorized operations instead of explicit loops
- [ ] Specify data types explicitly to optimize memory usage  
- [ ] Leverage broadcasting for efficient array operations
- [ ] Use appropriate linear algebra functions (`np.linalg.*`)
- [ ] Implement proper error handling and validation
- [ ] Profile performance for computational bottlenecks

### For Financial Calculations:
- [ ] Use appropriate precision (float64 for prices, float32 for returns)
- [ ] Implement multiple calculation methods for validation
- [ ] Handle extreme values and edge cases properly
- [ ] Annualize metrics using correct factors (252 for daily data)
- [ ] Calculate risk-adjusted performance measures
- [ ] Validate results against known analytical solutions when possible

### Code Quality & Documentation:
- [ ] Write comprehensive docstrings with parameters and examples
- [ ] Use descriptive variable names following conventions
- [ ] Implement input validation for all public functions
- [ ] Add unit tests for critical calculations
- [ ] Follow NumPy's broadcasting and indexing best practices
- [ ] Document assumptions and limitations clearly

### Performance Optimization:
- [ ] Use in-place operations where appropriate (`+=`, `*=`, etc.)
- [ ] Minimize array copying and temporary array creation
- [ ] Choose optimal memory layout (C vs Fortran order)
- [ ] Consider using numba for performance-critical loops
- [ ] Profile memory usage for large datasets
- [ ] Use appropriate chunk sizes for large computations

### Risk Management & Validation:
- [ ] Implement multiple VaR calculation methods
- [ ] Calculate both VaR and Expected Shortfall
- [ ] Perform stress testing and scenario analysis
- [ ] Validate portfolio risk decomposition (components sum to total)
- [ ] Check correlation matrices are positive semi-definite
- [ ] Implement robust optimization techniques for parameter uncertainty

### Before Production:
- [ ] Test with realistic market data and edge cases
- [ ] Verify numerical stability with ill-conditioned matrices
- [ ] Implement proper exception handling for matrix operations
- [ ] Add logging for monitoring and debugging
- [ ] Document computational complexity and memory requirements
- [ ] Create comprehensive test suite with known benchmarks

---

## Conclusion

This guide provides a comprehensive foundation for using NumPy effectively in quantitative finance research. The recommendations emphasize:

1. **Computational Efficiency**: Leveraging NumPy's vectorized operations and broadcasting capabilities for high-performance financial calculations

2. **Mathematical Rigor**: Implementing established financial models and risk measures with proper numerical methods and validation

3. **Code Quality**: Writing robust, well-documented, and maintainable code that follows scientific computing best practices

4. **Practical Application**: Focusing on real-world scenarios encountered in quantitative research and portfolio management

5. **Performance Optimization**: Using advanced techniques for memory management, algorithmic efficiency, and computational acceleration

While this guide covers many essential techniques, practitioners should:

- **Validate implementations** thoroughly against established benchmarks and analytical solutions
- **Test edge cases** including market stress scenarios, singular matrices, and extreme parameter values
- **Stay current** with NumPy updates and new scientific computing developments
- **Integrate properly** with other libraries in the PyData ecosystem (pandas, SciPy, scikit-learn)
- **Consider institutional requirements** for risk management, compliance, and performance standards

The financial mathematics and risk management concepts presented here represent industry-standard approaches, but markets evolve continuously. Always combine these technical tools with sound economic reasoning, proper model validation, and appropriate risk controls.

### Key Takeaways

**For New Practitioners:**
- Master the fundamentals of array operations and broadcasting before moving to complex financial models
- Always validate your calculations with simple, known cases first
- Understand the mathematical foundations behind the financial models you implement
- Pay careful attention to data types, precision, and numerical stability

**For Experienced Developers:**
- Focus on performance optimization and memory management for production systems
- Implement multiple calculation methods for critical risk measures
- Use advanced optimization techniques like JIT compilation when appropriate
- Maintain comprehensive test suites and documentation

**For Research Applications:**
- Prioritize reproducibility through proper random seed management
- Document all assumptions and model limitations clearly
- Validate against academic literature and industry benchmarks
- Consider parameter uncertainty and model risk in your analyses

### Disclaimer

This guide is for educational purposes only. All financial calculations should be independently validated and tested thoroughly before use in any investment or risk management decisions. Past performance does not guarantee future results. Always consult with qualified professionals and comply with relevant regulations when making financial decisions.

The code examples provided are for illustration purposes and may require additional error handling, optimization, and validation for production use. Users are responsible for ensuring their implementations meet their specific requirements and institutional standards.

Now I'll complete the remaining sections:

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"id": "1", "content": "Analyze the pandas best practices guide structure and approach", "status": "completed", "priority": "high"}, {"id": "2", "content": "Create comprehensive NumPy best practices guide with detailed sections", "status": "completed", "priority": "high"}, {"id": "3", "content": "Include explicit instructions for inexperienced users", "status": "completed", "priority": "high"}, {"id": "4", "content": "Add quantitative research specific examples and patterns", "status": "completed", "priority": "high"}, {"id": "5", "content": "Include performance optimization and memory management", "status": "completed", "priority": "medium"}, {"id": "6", "content": "Add remaining sections: financial calculations, time series, risk calculations", "status": "in_progress", "priority": "high"}, {"id": "7", "content": "Add conclusion, references, and summary checklist", "status": "pending", "priority": "medium"}]
```

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"id": "1", "content": "Analyze the pandas best practices guide structure and approach", "status": "completed", "priority": "high"}, {"id": "2", "content": "Create comprehensive NumPy best practices guide with detailed sections", "status": "completed", "priority": "high"}, {"id": "3", "content": "Include explicit instructions for inexperienced users", "status": "in_progress", "priority": "high"}, {"id": "4", "content": "Add quantitative research specific examples and patterns", "status": "in_progress", "priority": "high"}, {"id": "5", "content": "Include performance optimization and memory management", "status": "completed", "priority": "medium"}, {"id": "6", "content": "Add remaining sections: financial calculations, time series, risk calculations", "status": "pending", "priority": "high"}, {"id": "7", "content": "Add conclusion, references, and summary checklist", "status": "pending", "priority": "medium"}]