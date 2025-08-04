# Polars Performance & Large Dataset Focus

Master the fundamentals of Polars' performance advantages and leverage them for massive datasets up to hundreds of GB.

## Table of Contents
- [Introduction & Philosophy](#introduction--philosophy)
- [Setup & Environment Configuration](#setup--environment-configuration)
- [Polars Fundamentals & Core Concepts](#polars-fundamentals--core-concepts)
- [Lazy Evaluation & Query Optimization](#lazy-evaluation--query-optimization)
- [Memory Management Strategies](#memory-management-strategies)

---

## Introduction & Philosophy

### Why Polars for Quantitative Research?

Polars is a lightning-fast DataFrame library designed for performance-critical data processing. For quantitative research with large datasets (GB to hundreds of GB), Polars offers:

- **10-100x faster** than pandas for many operations
- **Memory-efficient streaming** for datasets larger than RAM
- **Lazy evaluation** with automatic query optimization
- **Parallel processing** utilizing all CPU cores
- **Apache Arrow backend** for interoperability
- **Rust foundation** providing memory safety and speed

### When to Use Polars vs Pandas

```python
# Use Polars for:
# - Initial data loading and cleaning (large files)
# - Complex filtering and aggregations
# - Time series resampling and window operations
# - Cross-joins and complex merges
# - Data preprocessing pipelines
# - ETL operations

# Use Pandas for:
# - Final analysis and visualization
# - Integration with existing quant libraries (QuantLib, etc.)
# - Statistical functions not yet in Polars
# - Jupyter notebook exploration and plotting
# - Legacy code integration
```

### Philosophy: Lazy-First Approach

**Always start with lazy evaluation** - only collect results when necessary:

```python
import polars as pl
import numpy as np
import pandas as pd

# BAD - Eager evaluation uses more memory
df = pl.read_csv("large_file.csv")
result = df.filter(pl.col("price") > 100).select(["symbol", "price"])

# GOOD - Lazy evaluation with optimization
result = (
    pl.scan_csv("large_file.csv")  # Lazy read
    .filter(pl.col("price") > 100)  # Lazy filter
    .select(["symbol", "price"])    # Lazy select
    .collect()                      # Only execute when needed
)
```

---

## Setup & Environment Configuration

### Installation & Dependencies

```bash
# Core installation
pip install polars[all]

# Additional dependencies for QR work
pip install polars[pandas,numpy,pyarrow,xlsx2csv,fsspec,async]

# For performance monitoring
pip install polars[performance]
```

### Environment Configuration

```python
import polars as pl
import pandas as pd
import numpy as np
import os
from pathlib import Path

# Configure Polars for optimal performance
pl.Config.set_tbl_rows(50)           # Display rows in output
pl.Config.set_tbl_cols(10)           # Display columns in output  
pl.Config.set_fmt_str_lengths(100)   # String display length
pl.Config.set_streaming_chunk_size(10000)  # Streaming chunk size
pl.Config.set_tbl_hide_column_data_types(False)  # Show dtypes

# Set thread count (use all cores for large data)
pl.Config.set_table_cell_alignment("RIGHT")  # Better number display

# Environment variables for large datasets
os.environ['POLARS_MAX_THREADS'] = str(os.cpu_count())
os.environ['POLARS_WARN_UNSTABLE'] = '0'  # Suppress experimental warnings

# Memory management
import gc
gc.set_threshold(700, 10, 10)  # More aggressive garbage collection
```

### Development Environment Setup

```python
def setup_polars_env():
    """
    Standard environment setup for QR work with Polars
    """
    # Enable all optimizations
    pl.Config.activate_decimals(True)
    
    # Performance settings for large datasets
    pl.Config.set_streaming_chunk_size(50000)  # Larger chunks for big data
    
    # Debugging settings (disable in production)
    pl.Config.set_verbose(True)  # Show query plans
    
    print("Polars version:", pl.__version__)
    print("Available threads:", pl.thread_pool_size())
    print("Streaming enabled: True")
    
setup_polars_env()
```

---

## Polars Fundamentals & Core Concepts

### DataFrame vs LazyFrame

```python
# Eager DataFrame - executes immediately
df = pl.DataFrame({
    "symbol": ["AAPL", "GOOGL", "MSFT"],
    "price": [150.0, 2800.0, 300.0],
    "volume": [1000000, 500000, 750000]
})

# Lazy DataFrame - deferred execution with optimization
lazy_df = pl.LazyFrame({
    "symbol": ["AAPL", "GOOGL", "MSFT"], 
    "price": [150.0, 2800.0, 300.0],
    "volume": [1000000, 500000, 750000]
})

# Converting between eager and lazy
eager_to_lazy = df.lazy()
lazy_to_eager = lazy_df.collect()
```

### Expression System - The Heart of Polars

```python
# Polars expressions are the key to performance
# They can be reused, combined, and optimized

# Basic expressions
price_expr = pl.col("price")
volume_expr = pl.col("volume")
symbol_expr = pl.col("symbol")

# Complex expressions
returns_expr = (pl.col("close") / pl.col("close").shift(1) - 1).alias("returns")
volatility_expr = pl.col("returns").rolling_std(window_size=20).alias("volatility_20d")

# Conditional expressions
signal_expr = pl.when(pl.col("returns") > 0.02).then(1).when(pl.col("returns") < -0.02).then(-1).otherwise(0).alias("signal")

# Multiple column operations
market_cap_expr = (pl.col("price") * pl.col("shares_outstanding")).alias("market_cap")

# String operations
clean_symbol_expr = pl.col("symbol").str.strip().str.to_uppercase().alias("clean_symbol")
```

### Data Types & Schema Management

```python
# Polars has rich type system optimized for performance
QUANT_SCHEMA = {
    "date": pl.Date,
    "datetime": pl.Datetime,
    "symbol": pl.Utf8,
    "price": pl.Float64,        # High precision for prices
    "volume": pl.UInt64,        # Unsigned int for volume
    "returns": pl.Float32,      # Lower precision OK for returns
    "signal": pl.Int8,          # Small int for signals (-1, 0, 1)
    "sector": pl.Categorical,   # Categorical for repeated strings
    "is_active": pl.Boolean     # Boolean flags
}

# Apply schema during reading
df = pl.read_csv("data.csv", schema=QUANT_SCHEMA)

# Schema validation function
def validate_quant_schema(df: pl.DataFrame) -> bool:
    """Validate DataFrame matches expected quantitative schema"""
    expected_types = {
        "date": pl.Date,
        "price": pl.Float64,
        "volume": pl.UInt64,
        "returns": pl.Float32
    }
    
    for col, expected_type in expected_types.items():
        if col in df.columns:
            actual_type = df[col].dtype
            if actual_type != expected_type:
                print(f"Warning: {col} is {actual_type}, expected {expected_type}")
                return False
    return True

# Type conversion with error handling
def safe_cast_schema(ldf: pl.LazyFrame) -> pl.LazyFrame:
    """Safely cast DataFrame to expected schema with error handling"""
    return (
        ldf
        .with_columns([
            pl.col("price").cast(pl.Float64, strict=False),
            pl.col("volume").cast(pl.UInt64, strict=False),
            pl.col("symbol").cast(pl.Categorical),
            pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False)
        ])
    )
```

---

## Lazy Evaluation & Query Optimization

### Understanding Lazy Evaluation

Lazy evaluation is Polars' superpower for large datasets. Operations are not executed immediately but built into a query plan that gets optimized before execution.

```python
# Lazy operations build a query plan
lazy_query = (
    pl.scan_csv("large_dataset.csv")
    .filter(pl.col("date") >= "2020-01-01")
    .filter(pl.col("volume") > 1000000) 
    .select(["date", "symbol", "price", "volume"])
    .group_by("symbol")
    .agg([
        pl.col("price").mean().alias("avg_price"),
        pl.col("volume").sum().alias("total_volume")
    ])
    .sort("avg_price", descending=True)
)

# View the optimized query plan
print(lazy_query.explain(optimized=True))

# Execute only when needed
result = lazy_query.collect()
```

### Query Plan Optimization

Polars automatically optimizes queries through:

1. **Predicate Pushdown**: Filters applied as early as possible
2. **Projection Pushdown**: Only read needed columns
3. **Common Subexpression Elimination**: Reuse computed expressions
4. **Constant Folding**: Evaluate constants at compile time

```python
def demonstrate_optimization():
    """Show how Polars optimizes queries automatically"""
    
    # This query will be heavily optimized
    optimized_query = (
        pl.scan_parquet("trades/*.parquet")
        .filter(pl.col("trade_date") >= "2023-01-01")  # Pushed to file read
        .filter(pl.col("symbol").is_in(["AAPL", "GOOGL"]))  # Combined with above
        .select([
            "symbol", 
            "price", 
            "volume",
            (pl.col("price") * pl.col("volume")).alias("notional")  # Computed lazily
        ])
        .group_by("symbol")
        .agg([
            pl.col("notional").sum().alias("total_notional"),
            pl.col("volume").count().alias("trade_count")
        ])
    )
    
    # Compare plans
    print("=== Unoptimized Plan ===")
    print(optimized_query.explain(optimized=False))
    
    print("\n=== Optimized Plan ===")
    print(optimized_query.explain(optimized=True))
    
    return optimized_query.collect()
```

### Query Optimization Strategies

```python
def query_optimization_patterns():
    """Advanced query optimization techniques for maximum performance"""
    
    def optimize_filter_order(df: pl.LazyFrame) -> pl.LazyFrame:
        """Optimize filter order for maximum performance"""
        
        # ✅ Good: Most selective filters first
        optimized_query = (
            df
            .filter(pl.col("symbol") == "AAPL")          # Very selective (1 stock)
            .filter(pl.col("date") >= "2020-01-01")     # Moderately selective (time filter)
            .filter(pl.col("volume") > 1000000)         # Less selective (many high-volume days)
            .filter(pl.col("price") > 0)                # Least selective (almost all valid)
        )
        
        return optimized_query
    
    def efficient_column_operations(df: pl.LazyFrame) -> pl.LazyFrame:
        """Optimize column operations for performance"""
        
        # ✅ Good: Batch column operations together
        optimized = (
            df
            .with_columns([
                # All new columns in one operation
                pl.col("price").pct_change().over("symbol").alias("returns"),
                pl.col("price").rolling_mean(window_size=20).over("symbol").alias("sma_20"),
                pl.col("volume").rolling_mean(window_size=20).over("symbol").alias("volume_sma"),
                (pl.col("price") * pl.col("volume")).alias("notional"),
                pl.col("returns").rolling_std(window_size=252).over("symbol").alias("volatility")
            ])
            .filter(pl.col("returns").is_not_null())  # Filter after computations
        )
        
        return optimized
    
    def optimize_joins(left_df: pl.LazyFrame, right_df: pl.LazyFrame) -> pl.LazyFrame:
        """Optimize join operations for performance"""
        
        # ✅ Good: Filter before joining, smaller tables first
        optimized_join = (
            left_df
            .filter(pl.col("date") >= "2020-01-01")     # Reduce left table size
            .select(["symbol", "date", "price", "volume"])  # Only needed columns
            .join(
                right_df
                .filter(pl.col("market_cap") > 1e9)     # Reduce right table size
                .select(["symbol", "date", "market_cap", "sector"]),  # Only needed columns
                on=["symbol", "date"],
                how="inner"
            )
        )
        
        return optimized_join
    
    return optimize_filter_order, efficient_column_operations, optimize_joins
```

---

## Memory Management Strategies

### Memory Optimization for Large Datasets

```python
def memory_optimization_techniques():
    """Memory optimization for large dataset processing"""
    
    def streaming_aggregations(file_pattern: str) -> pl.LazyFrame:
        """Perform memory-efficient aggregations on large datasets"""
        
        return (
            pl.scan_parquet(file_pattern)
            .filter(pl.col("date") >= "2020-01-01")  # Early filtering
            .select([                                # Column pruning
                "symbol", "date", "close", "volume", "market_cap"
            ])
            .group_by(["symbol", pl.col("date").dt.month_end()])  # Monthly aggregation
            .agg([
                pl.col("close").last().alias("month_end_price"),
                pl.col("volume").sum().alias("monthly_volume"),
                pl.col("market_cap").last().alias("month_end_mcap"),
                pl.col("close").count().alias("trading_days")
            ])
            .filter(pl.col("trading_days") >= 15)    # Filter months with sufficient data
        )
    
    def chunked_processing(large_df: pl.LazyFrame, chunk_size: int = 1000000) -> pl.DataFrame:
        """Process large datasets in chunks to manage memory"""
        
        # Get total row count
        total_rows = large_df.select(pl.len()).collect().item()
        
        results = []
        for start_idx in range(0, total_rows, chunk_size):
            end_idx = min(start_idx + chunk_size, total_rows)
            
            chunk_result = (
                large_df
                .slice(start_idx, end_idx - start_idx)
                .with_columns([
                    pl.col("returns").rolling_std(window_size=20).over("symbol").alias("volatility"),
                    pl.col("price").pct_change().over("symbol").alias("returns")
                ])
                .filter(pl.col("volatility").is_not_null())
                .collect()
            )
            
            results.append(chunk_result)
            
            # Optional: Force garbage collection after each chunk
            import gc
            gc.collect()
        
        return pl.concat(results)
    
    return streaming_aggregations, chunked_processing

def parallel_processing_patterns():
    """Leverage Polars' parallel processing capabilities"""
    
    def parallel_file_processing(file_list: list) -> pl.DataFrame:
        """Process multiple files in parallel efficiently"""
        
        # ✅ Good: Let Polars handle parallelism with glob patterns
        efficient_approach = (
            pl.scan_parquet("data/year_*.parquet")  # Automatically parallel
            .filter(pl.col("date") >= "2020-01-01")
            .group_by("symbol")
            .agg([
                pl.col("returns").mean().alias("avg_return"),
                pl.col("returns").std().alias("volatility"),
                pl.col("market_cap").last().alias("latest_mcap")
            ])
            .collect(streaming=True)
        )
        
        return efficient_approach
    
    def thread_configuration():
        """Optimize thread usage for your hardware"""
        
        import os
        
        # Get hardware info
        cpu_count = os.cpu_count()
        
        # Configure Polars threading
        optimal_threads = min(cpu_count, 16)  # Don't exceed 16 for most workloads
        
        # Set environment variables
        os.environ['POLARS_MAX_THREADS'] = str(optimal_threads)
        
        # For I/O heavy workloads, use fewer threads
        if "io_heavy_workload":
            os.environ['POLARS_MAX_THREADS'] = str(max(4, cpu_count // 2))
        
        print(f"Configured Polars to use {optimal_threads} threads")
        
        return optimal_threads
    
    return parallel_file_processing, thread_configuration
```

### Data Type Optimization

```python
def optimize_data_types():
    """Optimize data types for memory efficiency and performance"""
    
    def optimize_price_data_types(df: pl.LazyFrame) -> pl.LazyFrame:
        """Optimize data types for price/market data"""
        
        return (
            df
            .with_columns([
                # Optimize numeric columns
                pl.col("price").cast(pl.Float64),           # High precision for prices
                pl.col("volume").cast(pl.UInt64),           # Volume always positive
                pl.col("market_cap").cast(pl.UInt64),       # Large integers
                pl.col("shares_outstanding").cast(pl.UInt64),
                
                # Lower precision for derived metrics
                pl.col("returns").cast(pl.Float32),         # Returns don't need high precision
                pl.col("volatility").cast(pl.Float32),
                
                # Categorical for repeated strings
                pl.col("symbol").cast(pl.Categorical),
                pl.col("sector").cast(pl.Categorical),
                pl.col("exchange").cast(pl.Categorical),
                
                # Optimize integer flags
                pl.col("is_active").cast(pl.Boolean),
                pl.col("is_etf").cast(pl.Boolean),
                
                # Date optimization
                pl.col("date").cast(pl.Date),
                pl.col("timestamp").cast(pl.Datetime("us"))  # Microsecond precision
            ])
        )
    
    def memory_usage_analysis(df: pl.DataFrame) -> dict:
        """Analyze memory usage by column and data type"""
        
        memory_info = {}
        
        for col in df.columns:
            col_data = df[col]
            col_type = col_data.dtype
            
            # Estimate memory usage
            if col_type in [pl.Float64, pl.Int64, pl.UInt64]:
                bytes_per_value = 8
            elif col_type in [pl.Float32, pl.Int32, pl.UInt32]:
                bytes_per_value = 4
            elif col_type in [pl.Int16, pl.UInt16]:
                bytes_per_value = 2
            elif col_type in [pl.Int8, pl.UInt8, pl.Boolean]:
                bytes_per_value = 1
            else:
                bytes_per_value = col_data.str.len_bytes().mean() if hasattr(col_data, 'str') else 8
            
            memory_mb = (len(df) * bytes_per_value) / (1024 * 1024)
            
            memory_info[col] = {
                'dtype': str(col_type),
                'memory_mb': memory_mb,
                'null_count': col_data.null_count(),
                'unique_count': col_data.n_unique()
            }
        
        return memory_info
    
    return optimize_price_data_types, memory_usage_analysis
```

## Performance Best Practices Summary

### Essential Performance Patterns

```python
# 1. Always start with lazy evaluation
df = pl.scan_parquet("data.parquet")  # Not pl.read_parquet()

# 2. Filter early and often - most selective filters first
df = (
    pl.scan_parquet("large_data.parquet")
    .filter(pl.col("symbol") == "AAPL")          # Most selective first
    .filter(pl.col("date") >= "2020-01-01")     # Then date filters
    .filter(pl.col("volume") > 1000000)         # Then less selective filters
)

# 3. Use streaming for large datasets
result = df.collect(streaming=True)

# 4. Batch operations together
df = df.with_columns([
    pl.col("price").pct_change().alias("returns"),
    pl.col("returns").rolling_std(window_size=252).alias("volatility"),
    pl.col("market_cap").rank().over("date").alias("mcap_rank")
])

# 5. Optimize data types for memory efficiency
df = df.with_columns([
    pl.col("price").cast(pl.Float64),     # High precision where needed
    pl.col("returns").cast(pl.Float32),   # Lower precision where acceptable
    pl.col("symbol").cast(pl.Categorical) # Categorical for repeated strings
])
```

### Performance Checklist

**Before Running Large Operations:**
- [ ] Use `pl.scan_*` instead of `pl.read_*` 
- [ ] Apply filters as early as possible
- [ ] Select only needed columns
- [ ] Use appropriate data types
- [ ] Enable streaming for large datasets
- [ ] Monitor memory usage

**Query Optimization:**
- [ ] Check query plan with `.explain()`
- [ ] Batch multiple `.with_columns()` operations
- [ ] Use proper join strategies
- [ ] Leverage Polars' built-in functions over custom ones
- [ ] Consider partitioning for very large datasets

**Memory Management:**
- [ ] Use lazy evaluation throughout pipeline
- [ ] Stream results when possible
- [ ] Clean up intermediate variables
- [ ] Monitor memory usage in long-running processes
- [ ] Use appropriate chunk sizes for processing

---

*Next: [Polars-Pandas Integration](polars_pandas_integration.md) - Learn how to strategically combine Polars and Pandas for optimal workflows.*