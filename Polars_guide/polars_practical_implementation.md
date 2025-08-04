# Polars Practical Implementation Guide

A comprehensive, hands-on guide for implementing Polars in quantitative research workflows, covering data loading, cleaning, visualization, and troubleshooting with extensive real-world examples.

## Table of Contents

1. [Data Loading & I/O Best Practices](#data-loading--io-best-practices)
2. [Data Cleaning & Transformation Patterns](#data-cleaning--transformation-patterns)
3. [Common Pitfalls & Solutions](#common-pitfalls--solutions)
4. [Visualization Integration](#visualization-integration)
5. [Troubleshooting Guide](#troubleshooting-guide)
6. [Performance Debugging](#performance-debugging)
7. [Error Handling Patterns](#error-handling-patterns)
8. [Navigation](#navigation)

---

## Data Loading & I/O Best Practices

### File Format Optimization

Understanding which file format to use is crucial for performance in quantitative research:

```python
import polars as pl
import numpy as np
import pandas as pd
from pathlib import Path

def file_format_best_practices():
    """Optimal file formats for different use cases in quantitative research"""
    
    # Format decision matrix
    format_guide = {
        "CSV": {
            "use_for": "Small datasets, data exchange, human-readable",
            "avoid_for": "Large datasets (>100MB), repeated reads",
            "polars_function": "pl.read_csv() / pl.scan_csv()"
        },
        "Parquet": {
            "use_for": "Large datasets, analytical workloads, columnar analysis",
            "advantages": "Compression, column pruning, predicate pushdown",
            "polars_function": "pl.read_parquet() / pl.scan_parquet()"
        },
        "Arrow/Feather": {
            "use_for": "Fast serialization, cross-language compatibility",
            "advantages": "Zero-copy reads, memory mapping",
            "polars_function": "pl.read_ipc() / pl.scan_ipc()"
        },
        "Delta Lake": {
            "use_for": "Versioned datasets, ACID transactions, time travel",
            "advantages": "Schema evolution, concurrent writes",
            "polars_function": "pl.read_delta() / pl.scan_delta()"
        }
    }
    
    return format_guide
```

### Optimized CSV Loading

Loading large CSV files efficiently requires careful configuration:

```python
def optimized_csv_loading():
    """Best practices for loading large CSV files with Polars"""
    
    # Basic optimized CSV reading
    def read_large_csv_optimized(file_path: str):
        return (
            pl.scan_csv(
                file_path,
                # Schema optimization
                dtypes={
                    "date": pl.Date,
                    "symbol": pl.Categorical,  # Repeated strings
                    "price": pl.Float64,
                    "volume": pl.UInt64,
                    "returns": pl.Float32      # Lower precision for returns
                },
                # Performance optimizations
                low_memory=False,          # Faster parsing
                rechunk=True,              # Better memory layout
                skip_rows_after_header=0,  # Skip validation rows if any
                ignore_errors=False,       # Strict parsing for data quality
                # Date parsing optimization
                parse_dates=True
            )
            .collect()
        )
    
    # Reading multiple CSV files efficiently
    def read_multiple_csvs(file_pattern: str):
        """Read and combine multiple CSV files efficiently"""
        return (
            pl.scan_csv(
                file_pattern,  # e.g., "data/trades_*.csv"
                dtypes={
                    "timestamp": pl.Datetime,
                    "symbol": pl.Categorical,
                    "price": pl.Float64,
                    "size": pl.UInt32,
                    "side": pl.Categorical  # Buy/Sell
                }
            )
            .with_columns([
                # Add file source for tracking
                pl.col("timestamp").dt.date().alias("trade_date"),
                # Data quality flags
                (pl.col("price") > 0).alias("valid_price"),
                (pl.col("size") > 0).alias("valid_size")
            ])
            .filter(pl.col("valid_price") & pl.col("valid_size"))  # Filter invalid data early
        )
    
    # Streaming CSV processing for very large files
    def stream_large_csv(file_path: str):
        """Stream process very large CSV files"""
        return (
            pl.scan_csv(
                file_path,
                dtypes={
                    "timestamp": pl.Datetime,
                    "symbol": pl.Categorical,
                    "price": pl.Float64,
                    "volume": pl.UInt64
                }
            )
            .filter(pl.col("timestamp") >= "2020-01-01")  # Filter early
            .group_by([pl.col("timestamp").dt.date(), "symbol"])
            .agg([
                pl.col("price").mean().alias("avg_price"),
                pl.col("volume").sum().alias("total_volume"),
                pl.col("price").count().alias("trade_count")
            ])
            .collect(streaming=True)  # Enable streaming
        )
    
    return read_large_csv_optimized, read_multiple_csvs, stream_large_csv

# Example usage
read_func, multi_func, stream_func = optimized_csv_loading()

# Load a single large CSV file
df = read_func("large_market_data.csv")
print(f"Loaded {len(df):,} rows")

# Load multiple files at once
multi_df = multi_func("data/trades_2023_*.csv").collect()
print(f"Combined {len(multi_df):,} trades")

# Stream process very large file
streamed_df = stream_func("very_large_tick_data.csv")
print(f"Processed into {len(streamed_df):,} daily summaries")
```

### Parquet Optimization Patterns

Parquet is the preferred format for analytical workloads:

```python
def parquet_optimization_patterns():
    """Advanced Parquet usage patterns for quantitative data"""
    
    # Writing optimized Parquet files
    def write_optimized_parquet(df: pl.DataFrame, output_path: str):
        """Write Parquet with optimal settings for analytical workloads"""
        
        df.write_parquet(
            output_path,
            compression="zstd",        # Good compression + speed balance
            compression_level=3,       # Moderate compression
            statistics=True,           # Enable column statistics
            row_group_size=100000,     # Optimize for your typical query size
            data_page_size=1024*1024,  # 1MB pages for better compression
            use_pyarrow=True           # Use PyArrow for advanced features
        )
    
    # Partitioned Parquet for time series data
    def write_partitioned_parquet(df: pl.LazyFrame, output_dir: str):
        """Write time series data with date partitioning"""
        
        (
            df
            .with_columns([
                pl.col("date").dt.year().alias("year"),
                pl.col("date").dt.month().alias("month")
            ])
            .sink_parquet(
                output_dir,
                partition_by=["year", "month"],  # Hierarchical partitioning
                compression="snappy",            # Fast decompression
                maintain_order=True              # Keep sort order
            )
        )
    
    # Reading with optimization
    def read_parquet_optimized(file_pattern: str, columns: list = None, date_filter: str = None):
        """Read Parquet with column pruning and predicate pushdown"""
        
        query = pl.scan_parquet(file_pattern)
        
        # Column pruning (read only needed columns)
        if columns:
            query = query.select(columns)
        
        # Predicate pushdown (filter at file level)
        if date_filter:
            query = query.filter(pl.col("date") >= date_filter)
        
        return query
    
    # Multi-file Parquet reading with metadata
    def read_parquet_with_metadata(data_dir: str):
        """Read partitioned Parquet with file metadata"""
        
        return (
            pl.scan_parquet(f"{data_dir}/**/*.parquet")
            .with_columns([
                # Extract partition information from filename
                pl.col("__path__").str.extract(r"year=(\d{4})", 1).alias("file_year"),
                pl.col("__path__").str.extract(r"month=(\d{1,2})", 1).alias("file_month"),
            ])
            .drop("__path__")  # Remove path column
        )
    
    return write_optimized_parquet, write_partitioned_parquet, read_parquet_optimized, read_parquet_with_metadata

# Example usage
write_func, partition_func, read_func, metadata_func = parquet_optimization_patterns()

# Create sample data
sample_data = pl.DataFrame({
    "date": pl.date_range(start=pl.date(2023, 1, 1), end=pl.date(2023, 12, 31), interval="1d"),
    "symbol": ["AAPL"] * 365,
    "price": np.random.randn(365).cumsum() + 150,
    "volume": np.random.randint(1000000, 10000000, 365)
})

# Write optimized single file
write_func(sample_data, "optimized_data.parquet")

# Write partitioned data
partition_func(sample_data.lazy(), "partitioned_data/")

# Read with optimizations
optimized_read = read_func("optimized_data.parquet", 
                          columns=["date", "symbol", "price"], 
                          date_filter="2023-06-01")
result = optimized_read.collect()
print(f"Optimized read: {len(result):,} rows")
```

### Database Integration Patterns

Connecting Polars to databases for large-scale data extraction:

```python
def database_integration():
    """Integration patterns with databases for quantitative data"""
    
    # SQL database integration (requires connectorx)
    def read_from_sql_optimized():
        """Read large datasets from SQL databases efficiently"""
        
        try:
            # Using connectorx for fast database reads
            connection_string = "postgresql://user:password@localhost:5432/quant_db"
            
            # Read with column selection and filtering
            query = """
                SELECT date, symbol, close_price, volume, market_cap
                FROM daily_prices 
                WHERE date >= '2020-01-01'
                AND market_cap > 1000000000
                ORDER BY symbol, date
            """
            
            df = pl.read_database(
                query,
                connection_string,
                # Optimization options
                partition_on="date",        # Partition for parallel reads
                partition_num=10,           # Number of partitions
                protocol="binary"           # Faster binary protocol
            )
            
            return df
            
        except ImportError:
            print("Install connectorx: pip install connectorx")
            return None
    
    # Incremental data loading pattern
    def incremental_database_load():
        """Load only new data since last update"""
        
        # Check last loaded date
        try:
            last_date = (
                pl.scan_parquet("cache/daily_prices.parquet")
                .select(pl.col("date").max())
                .collect()
                .item()
            )
        except:
            last_date = "1990-01-01"  # Full historical load
        
        # Load only new data
        incremental_query = f"""
            SELECT date, symbol, close_price, volume
            FROM daily_prices 
            WHERE date > '{last_date}'
            ORDER BY symbol, date
        """
        
        new_data = pl.read_database(incremental_query, "your_connection_string")
        
        if len(new_data) > 0:
            # Append to existing data
            existing_data = pl.scan_parquet("cache/daily_prices.parquet")
            
            combined_data = (
                pl.concat([existing_data, new_data.lazy()])
                .unique(subset=["date", "symbol"], maintain_order=True)
                .sort(["symbol", "date"])
            )
            
            # Save updated dataset
            combined_data.sink_parquet("cache/daily_prices.parquet")
            
            return combined_data.collect()
        
        return new_data
    
    return read_from_sql_optimized, incremental_database_load
```

### Cloud Storage Integration

Working with data stored in cloud platforms:

```python
def cloud_storage_patterns():
    """Patterns for reading from cloud storage (S3, GCS, Azure)"""
    
    # AWS S3 integration
    def read_from_s3():
        """Read data from S3 with credentials and optimization"""
        
        # S3 configuration
        s3_options = {
            "aws_access_key_id": "your_access_key",
            "aws_secret_access_key": "your_secret_key", 
            "aws_region": "us-east-1"
        }
        
        # Read partitioned data from S3
        df = (
            pl.scan_parquet(
                "s3://your-bucket/market-data/year=*/month=*/*.parquet",
                storage_options=s3_options
            )
            .filter(pl.col("date") >= "2023-01-01")
            .collect(streaming=True)  # Stream for large S3 datasets
        )
        
        return df
    
    # Multi-cloud data aggregation
    def multi_cloud_aggregation():
        """Aggregate data from multiple cloud sources"""
        
        # Read from different sources
        s3_data = pl.scan_parquet("s3://bucket1/data/*.parquet")
        gcs_data = pl.scan_parquet("gs://bucket2/data/*.parquet") 
        local_data = pl.scan_parquet("local_data/*.parquet")
        
        # Combine and deduplicate
        combined = (
            pl.concat([s3_data, gcs_data, local_data])
            .unique(subset=["date", "symbol"], maintain_order=True)
            .sort(["symbol", "date"])
        )
        
        return combined
    
    return read_from_s3, multi_cloud_aggregation

# Example: Loading from S3 with error handling
def load_s3_data_with_fallback():
    """Load data from S3 with local fallback"""
    
    try:
        # Try S3 first
        s3_data = (
            pl.scan_parquet("s3://market-data/prices/*.parquet")
            .filter(pl.col("date") >= "2023-01-01")
            .collect()
        )
        print(f"Loaded {len(s3_data):,} rows from S3")
        return s3_data
    
    except Exception as e:
        print(f"S3 failed: {e}, trying local backup...")
        
        # Fallback to local data
        local_data = (
            pl.scan_parquet("backup/prices/*.parquet")
            .filter(pl.col("date") >= "2023-01-01")
            .collect()
        )
        print(f"Loaded {len(local_data):,} rows from local backup")
        return local_data
```

### Data Quality Pipeline

Comprehensive data validation during loading:

```python
def data_quality_pipeline():
    """Comprehensive data quality pipeline for quantitative data"""
    
    def validate_price_data(df: pl.LazyFrame) -> pl.LazyFrame:
        """Add data quality flags to price data"""
        
        return (
            df
            .with_columns([
                # Price validation
                (pl.col("price") > 0).alias("valid_price"),
                (pl.col("price").is_finite()).alias("finite_price"),
                
                # Volume validation  
                (pl.col("volume") >= 0).alias("valid_volume"),
                (pl.col("volume").is_not_null()).alias("non_null_volume"),
                
                # Date validation
                (pl.col("date").is_not_null()).alias("valid_date"),
                (pl.col("date") <= pl.date.today()).alias("date_not_future"),
                
                # Cross-validation
                (pl.col("high") >= pl.col("low")).alias("high_ge_low"),
                (pl.col("close").is_between(pl.col("low"), pl.col("high"))).alias("close_in_range"),
                
                # Outlier detection
                (pl.col("price").is_between(
                    pl.col("price").quantile(0.001), 
                    pl.col("price").quantile(0.999)
                )).alias("price_not_outlier")
            ])
            .with_columns([
                # Overall quality score
                (
                    pl.col("valid_price") & 
                    pl.col("finite_price") & 
                    pl.col("valid_volume") & 
                    pl.col("valid_date") & 
                    pl.col("high_ge_low") & 
                    pl.col("close_in_range")
                ).alias("high_quality")
            ])
        )
    
    def quality_report(df: pl.LazyFrame) -> dict:
        """Generate data quality report"""
        
        quality_metrics = (
            df
            .select([
                pl.len().alias("total_records"),
                pl.col("high_quality").sum().alias("high_quality_records"),
                pl.col("valid_price").mean().alias("valid_price_pct"),
                pl.col("valid_volume").mean().alias("valid_volume_pct"),
                pl.col("price_not_outlier").mean().alias("inlier_price_pct"),
                pl.col("date").n_unique().alias("unique_dates"),
                pl.col("symbol").n_unique().alias("unique_symbols")
            ])
            .collect()
        )
        
        return quality_metrics.to_dicts()[0]
    
    return validate_price_data, quality_report

# Example usage
validator, reporter = data_quality_pipeline()

# Load and validate data
raw_data = pl.scan_parquet("market_data.parquet")
validated_data = validator(raw_data)

# Generate quality report
quality_stats = reporter(validated_data)
print("Data Quality Report:")
for metric, value in quality_stats.items():
    if isinstance(value, float):
        print(f"  {metric}: {value:.2%}")
    else:
        print(f"  {metric}: {value:,}")

# Filter for high-quality data only
clean_data = validated_data.filter(pl.col("high_quality")).collect()
print(f"Clean dataset: {len(clean_data):,} rows ({len(clean_data)/quality_stats['total_records']:.1%} of original)")
```

---

## Data Cleaning & Transformation Patterns

### Systematic Data Cleaning Pipeline

A comprehensive approach to cleaning financial datasets:

```python
def comprehensive_cleaning_pipeline():
    """Complete data cleaning pipeline for quantitative datasets"""
    
    def clean_price_data(df: pl.LazyFrame) -> pl.LazyFrame:
        """Standardized price data cleaning"""
        
        return (
            df
            # 1. Remove obvious bad data
            .filter(pl.col("price") > 0)
            .filter(pl.col("volume") >= 0)
            .filter(pl.col("date").is_not_null())
            
            # 2. Handle duplicates (keep last)
            .unique(subset=["date", "symbol"], maintain_order=True, keep="last")
            
            # 3. Sort for time series operations
            .sort(["symbol", "date"])
            
            # 4. Add derived fields
            .with_columns([
                # Price validations
                (pl.col("high") >= pl.col("low")).alias("valid_high_low"),
                (pl.col("close").is_between(pl.col("low"), pl.col("high"))).alias("close_in_range"),
                (pl.col("open").is_between(pl.col("low"), pl.col("high"))).alias("open_in_range"),
                
                # Calculate returns
                pl.col("close").pct_change().over("symbol").alias("returns"),
                
                # Flag weekends/holidays (basic)
                pl.col("date").dt.weekday().alias("weekday"),
                (pl.col("date").dt.weekday() <= 5).alias("is_weekday")
            ])
            
            # 5. Handle missing values intelligently
            .with_columns([
                # Forward fill prices (common in finance)
                pl.col("close").forward_fill().over("symbol"),
                pl.col("volume").fill_null(0),  # Zero volume is meaningful
                
                # Interpolate missing high/low based on close
                pl.when(pl.col("high").is_null())
                .then(pl.col("close"))
                .otherwise(pl.col("high"))
                .alias("high"),
                
                pl.when(pl.col("low").is_null())
                .then(pl.col("close"))
                .otherwise(pl.col("low"))
                .alias("low")
            ])
            
            # 6. Outlier treatment
            .with_columns([
                # Cap extreme returns (winsorize)
                pl.col("returns")
                .clip(
                    pl.col("returns").quantile(0.001),
                    pl.col("returns").quantile(0.999)
                )
                .alias("returns_winsorized")
            ])
        )
    
    def clean_fundamental_data(df: pl.LazyFrame) -> pl.LazyFrame:
        """Clean fundamental/accounting data"""
        
        return (
            df
            # Remove negative market cap, book value, etc.
            .filter(pl.col("market_cap") > 0)
            .filter(pl.col("book_value") > 0)
            
            # Handle missing earnings (could be negative)
            .with_columns([
                pl.col("earnings").fill_null(0),
                pl.col("revenue").fill_null(0)
            ])
            
            # Calculate ratios with safe division
            .with_columns([
                # PE ratio with safe division
                pl.when(pl.col("earnings") > 0)
                .then(pl.col("price") / pl.col("earnings"))
                .otherwise(pl.lit(None))
                .alias("pe_ratio"),
                
                # Book to market
                pl.when(pl.col("market_cap") > 0)
                .then(pl.col("book_value") / pl.col("market_cap"))
                .otherwise(pl.lit(None))
                .alias("book_to_market"),
                
                # Asset turnover
                pl.when(pl.col("total_assets") > 0)
                .then(pl.col("revenue") / pl.col("total_assets"))
                .otherwise(pl.lit(None))
                .alias("asset_turnover")
            ])
            
            # Cap extreme ratios
            .with_columns([
                pl.col("pe_ratio")
                .clip(0, pl.col("pe_ratio").quantile(0.95))
                .alias("pe_ratio_capped")
            ])
        )
    
    return clean_price_data, clean_fundamental_data

# Example usage
price_cleaner, fundamental_cleaner = comprehensive_cleaning_pipeline()

# Clean price data
price_data = pl.scan_parquet("raw_price_data.parquet")
cleaned_prices = price_cleaner(price_data).collect()
print(f"Cleaned price data: {len(cleaned_prices):,} rows")

# Clean fundamental data
fundamental_data = pl.scan_parquet("raw_fundamental_data.parquet") 
cleaned_fundamentals = fundamental_cleaner(fundamental_data).collect()
print(f"Cleaned fundamental data: {len(cleaned_fundamentals):,} rows")
```

### Advanced Outlier Detection & Treatment

Sophisticated methods for handling outliers in financial data:

```python
def advanced_outlier_handling():
    """Sophisticated outlier detection methods for financial data"""
    
    def detect_return_outliers(df: pl.LazyFrame) -> pl.LazyFrame:
        """Multiple methods for return outlier detection"""
        
        return (
            df
            .with_columns([
                # Method 1: Z-score based
                (
                    (pl.col("returns") - pl.col("returns").mean()) / 
                    pl.col("returns").std()
                ).abs().alias("return_zscore"),
                
                # Method 2: IQR based  
                pl.col("returns").quantile(0.25).alias("q25"),
                pl.col("returns").quantile(0.75).alias("q75"),
            ])
            .with_columns([
                (pl.col("q75") - pl.col("q25")).alias("iqr"),
                (pl.col("q25") - 1.5 * (pl.col("q75") - pl.col("q25"))).alias("lower_fence"),
                (pl.col("q75") + 1.5 * (pl.col("q75") - pl.col("q25"))).alias("upper_fence")
            ])
            .with_columns([
                # Outlier flags
                (pl.col("return_zscore") > 3).alias("zscore_outlier"),
                (
                    (pl.col("returns") < pl.col("lower_fence")) |
                    (pl.col("returns") > pl.col("upper_fence"))
                ).alias("iqr_outlier"),
                
                # Method 3: Rolling volatility based
                (
                    pl.col("returns").abs() > 
                    (3 * pl.col("returns").rolling_std(window_size=252).over("symbol"))
                ).alias("vol_outlier")
            ])
            .with_columns([
                # Combined outlier flag
                (
                    pl.col("zscore_outlier") | 
                    pl.col("iqr_outlier") | 
                    pl.col("vol_outlier")
                ).alias("is_outlier")
            ])
        )
    
    def treat_outliers_contextually(df: pl.LazyFrame) -> pl.LazyFrame:
        """Context-aware outlier treatment"""
        
        return (
            df
            .with_columns([
                # Different treatment by market conditions
                pl.col("returns").rolling_std(window_size=20).over("symbol").alias("recent_vol"),
                
                # Market stress indicator (high cross-sectional volatility)
                pl.col("returns").std().over("date").alias("market_stress")
            ])
            .with_columns([
                # Adaptive winsorization based on market conditions
                pl.when(pl.col("market_stress") > pl.col("market_stress").quantile(0.95))
                .then(  # High stress: more lenient outlier treatment
                    pl.col("returns").clip(
                        pl.col("returns").quantile(0.005),
                        pl.col("returns").quantile(0.995)
                    )
                )
                .otherwise(  # Normal times: stricter outlier treatment
                    pl.col("returns").clip(
                        pl.col("returns").quantile(0.01),
                        pl.col("returns").quantile(0.99)
                    )
                )
                .alias("returns_adaptive")
            ])
        )
    
    return detect_return_outliers, treat_outliers_contextually

# Example usage with detailed analysis
outlier_detector, outlier_treater = advanced_outlier_handling()

# Create sample data with outliers
sample_returns = pl.DataFrame({
    "date": pl.date_range(pl.date(2023, 1, 1), pl.date(2023, 12, 31), interval="1d"),
    "symbol": ["AAPL"] * 365,
    "returns": np.random.normal(0.001, 0.02, 365)  # Normal returns
})

# Add some outliers
outlier_indices = [50, 100, 200, 300]
sample_returns = sample_returns.with_row_index("idx")
for idx in outlier_indices:
    sample_returns = sample_returns.with_columns([
        pl.when(pl.col("idx") == idx)
        .then(0.15)  # 15% return outlier
        .otherwise(pl.col("returns"))
        .alias("returns")
    ])

# Detect outliers
outliers_detected = outlier_detector(sample_returns.lazy()).collect()
outlier_count = outliers_detected.select(pl.col("is_outlier").sum()).item()
print(f"Detected {outlier_count} outliers out of {len(outliers_detected)} observations")

# Treat outliers contextually
treated_data = outlier_treater(outliers_detected.lazy()).collect()
print(f"Applied adaptive outlier treatment")

# Compare original vs treated returns
comparison = treated_data.select([
    pl.col("returns").std().alias("original_std"),
    pl.col("returns_adaptive").std().alias("treated_std"),
    pl.col("returns").abs().max().alias("original_max_abs"),
    pl.col("returns_adaptive").abs().max().alias("treated_max_abs")
])
print("\nOutlier Treatment Impact:")
print(comparison)
```

### String Cleaning & Standardization

Cleaning and standardizing text fields in financial data:

```python
def string_cleaning_patterns():
    """Standardize string data for financial datasets"""
    
    def clean_symbol_data(df: pl.LazyFrame) -> pl.LazyFrame:
        """Standardize ticker symbols and company names"""
        
        return (
            df
            .with_columns([
                # Clean ticker symbols
                pl.col("symbol")
                .str.strip()                    # Remove whitespace
                .str.to_uppercase()             # Standardize case
                .str.replace_all(r"[^A-Z0-9]", "")  # Remove special chars
                .alias("clean_symbol"),
                
                # Clean company names
                pl.col("company_name")
                .str.strip()
                .str.to_titlecase()             # Title case
                .str.replace_all(r"\s+", " ")   # Normalize whitespace
                .str.replace_all(r"[^\w\s&.-]", "")  # Keep only valid chars
                .alias("clean_company_name"),
                
                # Standardize sectors/industries
                pl.col("sector")
                .str.strip()
                .str.to_lowercase()
                .str.replace_all("_", " ")
                .str.replace_all(r"\s+", " ")
                .alias("sector_clean")
            ])
            .with_columns([
                # Create sector mapping for consistency
                pl.col("sector_clean")
                .str.replace_all("tech", "technology")
                .str.replace_all("financials", "financial")
                .str.replace_all("healthcare", "health care")
                .alias("sector_standardized")
            ])
        )
    
    def standardize_exchange_data(df: pl.LazyFrame) -> pl.LazyFrame:
        """Standardize exchange and currency information"""
        
        return (
            df
            .with_columns([
                # Standardize exchange names
                pl.col("exchange")
                .str.to_lowercase()
                .str.replace("nasdaq", "NASDAQ")
                .str.replace("nyse", "NYSE")
                .str.replace("amex", "AMEX")
                .str.replace("otc", "OTC")
                .alias("exchange_std"),
                
                # Standardize currency codes
                pl.col("currency")
                .str.to_uppercase()
                .alias("currency_std")
            ])
        )
    
    return clean_symbol_data, standardize_exchange_data

# Example usage
symbol_cleaner, exchange_cleaner = string_cleaning_patterns()

# Create messy sample data
messy_data = pl.DataFrame({
    "symbol": ["  aapl  ", "GOOGL.", "msft!", "TSLA#"],
    "company_name": ["apple    inc.", "Alphabet INC", "microsoft   corp", "Tesla, Inc."],
    "sector": ["Tech", "TECHNOLOGY", "technology", "auto"],
    "exchange": ["nasdaq", "NASDAQ", "nyse", "NASDAQ"],
    "currency": ["usd", "USD", "usd", "USD"]
})

print("Original data:")
print(messy_data)

# Clean the data
cleaned = symbol_cleaner(messy_data.lazy()).collect()
cleaned = exchange_cleaner(cleaned.lazy()).collect()

print("\nCleaned data:")
print(cleaned.select(["clean_symbol", "clean_company_name", "sector_standardized", "exchange_std"]))
```

### Data Type Optimization & Conversion

Optimizing memory usage through proper data types:

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
                # Estimate for strings/categoricals
                bytes_per_value = 8  # Rough estimate
            
            memory_mb = (len(df) * bytes_per_value) / (1024 * 1024)
            
            memory_info[col] = {
                'dtype': str(col_type),
                'memory_mb': memory_mb,
                'null_count': col_data.null_count(),
                'unique_count': col_data.n_unique()
            }
        
        return memory_info
    
    return optimize_price_data_types, memory_usage_analysis

# Example usage with memory comparison
optimizer, analyzer = optimize_data_types()

# Create sample data with suboptimal types
inefficient_data = pl.DataFrame({
    "symbol": ["AAPL", "GOOGL", "MSFT"] * 100000,  # String instead of categorical
    "price": [150.0, 2500.0, 300.0] * 100000,     # Float64 is fine
    "volume": [1000000, 500000, 800000] * 100000,  # Could be UInt32
    "returns": [0.01, -0.02, 0.005] * 100000,     # Could be Float32
    "is_active": [True, True, False] * 100000      # Boolean is optimal
})

print("Memory usage before optimization:")
before_memory = analyzer(inefficient_data)
for col, info in before_memory.items():
    print(f"  {col}: {info['memory_mb']:.1f} MB ({info['dtype']})")

# Optimize data types
optimized_data = optimizer(inefficient_data.lazy()).collect()

print("\nMemory usage after optimization:")
after_memory = analyzer(optimized_data)
for col, info in after_memory.items():
    print(f"  {col}: {info['memory_mb']:.1f} MB ({info['dtype']})")

# Calculate total savings
total_before = sum(info['memory_mb'] for info in before_memory.values())
total_after = sum(info['memory_mb'] for info in after_memory.values())
savings = (total_before - total_after) / total_before * 100
print(f"\nTotal memory savings: {savings:.1f}% ({total_before:.1f} MB → {total_after:.1f} MB)")
```

### Date & Time Handling

Advanced datetime processing for financial data:

```python
def datetime_cleaning_patterns():
    """Advanced date/time cleaning for financial data"""
    
    def standardize_datetime_formats(df: pl.LazyFrame) -> pl.LazyFrame:
        """Handle various datetime formats common in financial data"""
        
        return (
            df
            .with_columns([
                # Handle multiple datetime formats
                pl.when(pl.col("date_str").str.contains(r"\d{4}-\d{2}-\d{2}"))
                .then(pl.col("date_str").str.strptime(pl.Date, format="%Y-%m-%d"))
                .when(pl.col("date_str").str.contains(r"\d{2}/\d{2}/\d{4}"))
                .then(pl.col("date_str").str.strptime(pl.Date, format="%m/%d/%Y"))
                .when(pl.col("date_str").str.contains(r"\d{4}\d{2}\d{2}"))
                .then(pl.col("date_str").str.strptime(pl.Date, format="%Y%m%d"))
                .otherwise(pl.lit(None))
                .alias("parsed_date"),
                
                # Handle timezone-aware timestamps
                pl.col("timestamp_str")
                .str.strptime(pl.Datetime("us", "UTC"), format="%Y-%m-%d %H:%M:%S%z")
                .dt.convert_time_zone("America/New_York")  # Convert to market timezone
                .alias("market_timestamp")
            ])
            .with_columns([
                # Extract useful date components
                pl.col("parsed_date").dt.year().alias("year"),
                pl.col("parsed_date").dt.month().alias("month"),
                pl.col("parsed_date").dt.weekday().alias("weekday"),
                pl.col("parsed_date").dt.quarter().alias("quarter"),
                
                # Market timing features
                pl.col("market_timestamp").dt.hour().alias("hour"),
                (pl.col("market_timestamp").dt.hour().is_between(9, 16)).alias("market_hours"),
                
                # Business day calculations
                pl.col("parsed_date").dt.weekday().le(5).alias("is_weekday")
            ])
        )
    
    def handle_market_calendars(df: pl.LazyFrame) -> pl.LazyFrame:
        """Add market calendar information"""
        
        # Define US market holidays (simplified for 2023)
        market_holidays_2023 = [
            pl.date(2023, 1, 2),   # New Year's Day (observed)
            pl.date(2023, 1, 16),  # MLK Day
            pl.date(2023, 2, 20),  # Presidents Day
            pl.date(2023, 4, 7),   # Good Friday
            pl.date(2023, 5, 29),  # Memorial Day
            pl.date(2023, 6, 19),  # Juneteenth
            pl.date(2023, 7, 4),   # Independence Day
            pl.date(2023, 9, 4),   # Labor Day
            pl.date(2023, 11, 23), # Thanksgiving
            pl.date(2023, 12, 25)  # Christmas
        ]
        
        return (
            df
            .with_columns([
                # Market holiday flag
                pl.col("date").is_in(market_holidays_2023).alias("is_holiday"),
                
                # Trading day flag
                (pl.col("date").dt.weekday() <= 5).alias("is_weekday")
            ])
            .with_columns([
                # Market open flag
                (pl.col("is_weekday") & ~pl.col("is_holiday")).alias("is_market_day")
            ])
        )
    
    return standardize_datetime_formats, handle_market_calendars

# Example usage
datetime_cleaner, calendar_handler = datetime_cleaning_patterns()

# Create sample data with various date formats
mixed_dates = pl.DataFrame({
    "date_str": ["2023-01-15", "01/20/2023", "20230125", "2023-02-01"],
    "timestamp_str": [
        "2023-01-15 09:30:00-05:00",
        "2023-01-20 15:45:00-05:00", 
        "2023-01-25 10:15:00-05:00",
        "2023-02-01 14:20:00-05:00"
    ],
    "price": [150.0, 155.0, 152.0, 158.0]
})

print("Original mixed date formats:")
print(mixed_dates)

# Clean and standardize dates
cleaned_dates = datetime_cleaner(mixed_dates.lazy()).collect()
print("\nStandardized dates:")
print(cleaned_dates.select(["parsed_date", "market_timestamp", "year", "month", "weekday", "market_hours"]))

# Add market calendar information
with_calendar = calendar_handler(cleaned_dates.lazy()).collect()
print("\nWith market calendar:")
print(with_calendar.select(["parsed_date", "is_weekday", "is_holiday", "is_market_day"]))
```

---

## Common Pitfalls & Solutions

### Lazy vs Eager Evaluation Mistakes

Understanding when and how to use lazy evaluation:

```python
def common_pitfalls_and_solutions():
    """Common mistakes when using Polars and how to avoid them"""
    
    def lazy_vs_eager_confusion():
        """Common mistakes with lazy vs eager evaluation"""
        
        # ❌ Common mistake: Forgetting to collect lazy frames
        def mistake_not_collecting():
            lazy_df = pl.scan_csv("data.csv").filter(pl.col("price") > 100)
            # This returns a LazyFrame, not actual data!
            print("Mistake: This won't show the actual data:")
            print(type(lazy_df))  # <class 'polars.LazyFrame'>
            return lazy_df
        
        # ✅ Correct: Collect when you need the results
        def correct_collecting():
            lazy_df = pl.scan_csv("data.csv").filter(pl.col("price") > 100)
            result = lazy_df.collect()  # Now it's a DataFrame
            print("Correct: This shows the actual data:")
            print(type(result))  # <class 'polars.DataFrame'>
            return result
        
        # ❌ Common mistake: Collecting too early
        def mistake_early_collection():
            df = pl.scan_csv("data.csv").collect()  # Collected immediately - loses optimization
            result = df.lazy().filter(pl.col("price") > 100).collect()  # Lost lazy optimization
            return result
        
        # ✅ Correct: Stay lazy until the end
        def correct_lazy_chain():
            result = (
                pl.scan_csv("data.csv")
                .filter(pl.col("price") > 100)
                .group_by("symbol")
                .agg([pl.col("volume").mean()])
                .collect()  # Collect only at the end
            )
            return result
        
        return correct_collecting, correct_lazy_chain
    
    return lazy_vs_eager_confusion

# Example demonstrating the difference
def demonstrate_lazy_vs_eager():
    """Practical demonstration of lazy vs eager evaluation"""
    
    # Create sample data
    sample_data = pl.DataFrame({
        "symbol": ["AAPL", "GOOGL", "MSFT"] * 10000,
        "price": np.random.uniform(100, 300, 30000),
        "volume": np.random.randint(100000, 10000000, 30000),
        "date": [pl.date(2023, 1, 1)] * 30000
    })
    
    # Save to use with scan_csv
    sample_data.write_csv("demo_data.csv")
    
    import time
    
    # ❌ Eager approach (less efficient)
    print("❌ Eager approach:")
    start_time = time.time()
    eager_df = pl.read_csv("demo_data.csv")
    eager_result = (
        eager_df
        .filter(pl.col("price") > 200)
        .filter(pl.col("volume") > 500000)
        .group_by("symbol")
        .agg([
            pl.col("price").mean().alias("avg_price"),
            pl.col("volume").sum().alias("total_volume")
        ])
    )
    eager_time = time.time() - start_time
    print(f"Eager execution time: {eager_time:.4f} seconds")
    
    # ✅ Lazy approach (more efficient)
    print("\n✅ Lazy approach:")
    start_time = time.time()
    lazy_result = (
        pl.scan_csv("demo_data.csv")
        .filter(pl.col("price") > 200)
        .filter(pl.col("volume") > 500000)
        .group_by("symbol")
        .agg([
            pl.col("price").mean().alias("avg_price"),
            pl.col("volume").sum().alias("total_volume")
        ])
        .collect()  # Only collect at the end
    )
    lazy_time = time.time() - start_time
    print(f"Lazy execution time: {lazy_time:.4f} seconds")
    
    print(f"\nSpeedup: {eager_time/lazy_time:.2f}x faster with lazy evaluation")
    
    # Verify results are identical
    print(f"Results identical: {eager_result.equals(lazy_result)}")
    
    return eager_result, lazy_result

# Run demonstration
eager_result, lazy_result = demonstrate_lazy_vs_eager()
```

### Type Errors and Solutions

Handling data type mismatches and conversions:

```python
def type_errors_and_solutions():
    """Common type-related errors and how to fix them"""
    
    # ❌ Common mistake: Type mismatches in operations
    def mistake_type_mismatch():
        df = pl.DataFrame({
            "price": ["100.5", "200.3", "150.0"],  # String instead of float
            "volume": [1000, 2000, 1500]
        })
        
        # This will fail - can't do math on strings
        try:
            result = df.with_columns([
                (pl.col("price") * 1.1).alias("adjusted_price")
            ])
            print("Unexpectedly succeeded!")
            return result
        except Exception as e:
            print(f"❌ Expected error: {e}")
            return None
    
    # ✅ Correct: Explicit type conversion
    def correct_type_handling():
        df = pl.DataFrame({
            "price": ["100.5", "200.3", "150.0"],  # String data
            "volume": [1000, 2000, 1500]
        })
        
        result = df.with_columns([
            # Convert to float first
            pl.col("price").cast(pl.Float64).alias("price_numeric"),
            (pl.col("price").cast(pl.Float64) * 1.1).alias("adjusted_price")
        ])
        print("✅ Successful type conversion:")
        print(result)
        return result
    
    # ✅ Better: Define schema upfront
    def better_schema_definition():
        # When reading from CSV, specify types upfront
        schema = {
            "price": pl.Float64,
            "volume": pl.Int64,
            "symbol": pl.Utf8
        }
        
        print("✅ Schema-based approach prevents type issues")
        return schema
    
    # Practical example with error handling
    def robust_type_conversion():
        """Handle type conversion with error checking"""
        
        df = pl.DataFrame({
            "price": ["100.5", "invalid", "150.0", ""],
            "volume": ["1000", "2000", "invalid", "1500"],
            "date": ["2023-01-01", "not-a-date", "2023-01-03", "2023-01-04"]
        })
        
        print("Original data with mixed/invalid types:")
        print(df)
        
        # Robust conversion with null handling
        converted = (
            df
            .with_columns([
                # Price conversion with error handling
                pl.col("price")
                .str.replace("", None)  # Replace empty strings with null
                .str.to_decimal()
                .cast(pl.Float64, strict=False)  # Non-strict casting
                .alias("price_clean"),
                
                # Volume conversion
                pl.col("volume")
                .str.to_integer(strict=False)
                .cast(pl.Int64, strict=False)
                .alias("volume_clean"),
                
                # Date conversion
                pl.col("date")
                .str.strptime(pl.Date, format="%Y-%m-%d", strict=False)
                .alias("date_clean")
            ])
        )
        
        print("\nAfter robust type conversion:")
        print(converted)
        
        # Show null counts
        null_counts = converted.null_count()
        print("\nNull counts after conversion:")
        print(null_counts)
        
        return converted
    
    return mistake_type_mismatch, correct_type_handling, better_schema_definition, robust_type_conversion

# Demonstrate type handling
type_demo = type_errors_and_solutions()
mistake_func, correct_func, schema_func, robust_func = type_demo

print("=== Type Error Demonstration ===")
mistake_func()
print()
correct_func()
print()
robust_func()
```

### Memory Errors and Solutions

Handling memory issues with large datasets:

```python
def memory_errors_and_solutions():
    """Handle memory errors in large datasets"""
    
    # ❌ Common mistake: Loading everything into memory
    def mistake_memory_intensive():
        """Simulate memory-intensive operations that can fail"""
        
        print("❌ Memory-intensive approach:")
        try:
            # This approach loads everything into memory at once
            large_df = pl.DataFrame({
                "symbol": ["AAPL"] * 1000000,
                "price": np.random.randn(1000000) * 10 + 150,
                "volume": np.random.randint(100000, 10000000, 1000000),
                "date": [pl.date(2023, 1, 1)] * 1000000
            })
            
            # Memory-intensive operations
            processed = large_df.with_columns([
                pl.col("price").pct_change().alias("returns"),
                pl.col("returns").rolling_std(window_size=252).alias("volatility")
            ])
            
            print(f"Processed {len(processed):,} rows (used lots of memory)")
            return processed
            
        except MemoryError as e:
            print(f"Memory error: {e}")
            return None
    
    # ✅ Correct: Use streaming and lazy evaluation
    def correct_streaming_approach():
        """Memory-efficient approach using streaming"""
        
        print("✅ Memory-efficient streaming approach:")
        
        # Create and save sample data
        sample_df = pl.DataFrame({
            "symbol": ["AAPL", "GOOGL", "MSFT"] * 100000,
            "price": np.random.randn(300000) * 10 + 150,
            "volume": np.random.randint(100000, 10000000, 300000),
            "date": pl.date_range(pl.date(2023, 1, 1), pl.date(2023, 12, 31), interval="1d")[:300000]
        })
        sample_df.write_parquet("large_sample.parquet")
        
        try:
            result = (
                pl.scan_parquet("large_sample.parquet")
                .filter(pl.col("date") >= "2023-06-01")    # Reduce data early
                .select(["symbol", "date", "price", "volume"])  # Only needed columns
                .with_columns([
                    pl.col("price").pct_change().over("symbol").alias("returns")
                ])
                .with_columns([
                    pl.col("returns").rolling_std(window_size=30).over("symbol").alias("volatility")
                ])
                .collect(streaming=True)  # Stream processing
            )
            
            print(f"Processed {len(result):,} rows with streaming")
            return result
            
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    # ✅ Alternative: Process in chunks
    def chunk_processing_approach():
        """Process large datasets in manageable chunks"""
        
        print("✅ Chunk processing approach:")
        
        chunk_size = 50000
        results = []
        
        try:
            # Get total rows first
            total_rows = pl.scan_parquet("large_sample.parquet").select(pl.len()).collect().item()
            print(f"Total rows to process: {total_rows:,}")
            
            for start_idx in range(0, total_rows, chunk_size):
                print(f"Processing chunk starting at row {start_idx:,}")
                
                chunk = (
                    pl.scan_parquet("large_sample.parquet")
                    .slice(start_idx, chunk_size)
                    .collect()
                )
                
                processed_chunk = chunk.with_columns([
                    pl.col("price").pct_change().alias("returns")
                ]).with_columns([
                    pl.col("returns").rolling_std(window_size=min(30, len(chunk))).alias("volatility")
                ])
                
                results.append(processed_chunk)
            
            # Combine all chunks
            final_result = pl.concat(results)
            print(f"Combined {len(results)} chunks into {len(final_result):,} total rows")
            return final_result
        
        except Exception as e:
            print(f"Error in chunk processing: {e}")
            return None
    
    # Monitor memory usage during operations
    def memory_monitoring_example():
        """Demonstrate memory monitoring during operations"""
        
        import psutil
        import gc
        
        def get_memory_usage():
            return psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        print("📊 Memory usage monitoring:")
        print(f"Initial memory: {get_memory_usage():.1f} MB")
        
        # Create data
        df = pl.DataFrame({
            "values": np.random.randn(500000),
            "categories": np.random.choice(["A", "B", "C"], 500000)
        })
        print(f"After creating DataFrame: {get_memory_usage():.1f} MB")
        
        # Process data
        processed = df.with_columns([
            pl.col("values").rolling_mean(window_size=100).alias("ma_100"),
            pl.col("values").rolling_std(window_size=100).alias("std_100")
        ])
        print(f"After processing: {get_memory_usage():.1f} MB")
        
        # Clean up intermediate results
        del df
        gc.collect()
        print(f"After cleanup: {get_memory_usage():.1f} MB")
        
        return processed
    
    return mistake_memory_intensive, correct_streaming_approach, chunk_processing_approach, memory_monitoring_example

# Demonstrate memory management approaches
memory_demo = memory_errors_and_solutions()
mistake_func, streaming_func, chunk_func, monitor_func = memory_demo

print("=== Memory Management Demonstration ===")
mistake_func()
print()
streaming_func()
print()
chunk_func()
print()
monitor_func()
```

---

## Visualization Integration

### Seamless Visualization Workflows

Integrating Polars with visualization libraries through Pandas:

```python
def visualization_integration_patterns():
    """Integrate Polars with visualization libraries via Pandas"""
    
    def polars_to_plotting():
        """Convert Polars results to plotting-friendly formats"""
        
        # Process large dataset with Polars, visualize small result with Pandas
        def plot_factor_performance():
            """Calculate factor performance for visualization"""
            
            # Create sample factor data
            sample_factor_data = pl.DataFrame({
                "date": pl.date_range(pl.date(2020, 1, 1), pl.date(2023, 12, 31), interval="1d"),
                "symbol": np.random.choice(["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"], 1461),
                "returns": np.random.normal(0.0005, 0.02, 1461),
                "value_quintile": np.random.randint(1, 6, 1461),  # 1-5 quintiles
                "market_cap": np.random.uniform(1e9, 1e12, 1461)
            })
            
            # Heavy computation in Polars
            factor_returns = (
                sample_factor_data.lazy()
                .filter(pl.col("date") >= "2020-01-01")
                .group_by(["date", "value_quintile"])
                .agg([
                    pl.col("returns").mean().alias("portfolio_return"),
                    pl.col("returns").count().alias("stock_count")
                ])
                .filter(pl.col("stock_count") >= 3)  # Minimum stocks per portfolio
                .collect()
            )
            
            # Convert to Pandas for visualization
            factor_df = factor_returns.to_pandas()
            
            # Create pivot table for plotting
            performance_matrix = factor_df.pivot(
                index='date', 
                columns='value_quintile', 
                values='portfolio_return'
            )
            
            # Calculate cumulative returns
            cumulative_returns = (1 + performance_matrix.fillna(0)).cumprod()
            
            print("Factor performance data prepared:")
            print(f"  Original data: {len(sample_factor_data):,} rows")
            print(f"  Aggregated data: {len(factor_returns):,} rows")
            print(f"  Performance matrix shape: {performance_matrix.shape}")
            
            return factor_returns, performance_matrix, cumulative_returns
        
        # Efficient data preparation for interactive plots
        def prepare_interactive_data():
            """Prepare data for interactive visualizations"""
            
            # Create sample market data
            sectors = ["Technology", "Healthcare", "Financial", "Energy", "Consumer"]
            sample_market_data = pl.DataFrame({
                "date": pl.date_range(pl.date(2023, 1, 1), pl.date(2023, 12, 31), interval="1d"),
                "sector": np.random.choice(sectors, 365),
                "symbol": np.random.choice([f"STOCK{i}" for i in range(100)], 365),
                "returns": np.random.normal(0.001, 0.025, 365),
                "market_cap": np.random.uniform(1e8, 1e11, 365),
                "volume": np.random.randint(100000, 10000000, 365)
            })
            
            # Use Polars to prepare data
            summary_data = (
                sample_market_data.lazy()
                .group_by(["sector", "date"])
                .agg([
                    pl.col("market_cap").sum().alias("sector_mcap"),
                    pl.col("returns").mean().alias("sector_return"),
                    pl.col("returns").std().alias("sector_vol"),
                    pl.col("symbol").n_unique().alias("unique_stocks")
                ])
                .filter(pl.col("unique_stocks") >= 2)
                .collect()
            )
            
            # Convert for plotly/bokeh
            plot_ready = summary_data.to_pandas()
            
            # Add derived columns for plotting
            plot_ready['risk_adjusted_return'] = plot_ready['sector_return'] / plot_ready['sector_vol']
            plot_ready['size_category'] = pd.qcut(
                plot_ready['sector_mcap'], 
                3, 
                labels=['Small', 'Medium', 'Large']
            )
            
            print("Interactive data prepared:")
            print(f"  Summary shape: {plot_ready.shape}")
            print(f"  Sectors: {plot_ready['sector'].nunique()}")
            print(f"  Date range: {plot_ready['date'].min()} to {plot_ready['date'].max()}")
            
            return summary_data, plot_ready
        
        return plot_factor_performance, prepare_interactive_data
    
    return polars_to_plotting

# Example usage
viz_patterns = visualization_integration_patterns()
plotting_funcs = viz_patterns()
factor_func, interactive_func = plotting_funcs

# Prepare data for plotting
factor_returns, performance_matrix, cumulative_returns = factor_func()
summary_data, plot_ready_data = interactive_func()
```

### Matplotlib Integration

Creating publication-quality charts with matplotlib:

```python
def matplotlib_integration():
    """Integration patterns with matplotlib/seaborn"""
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    def plot_rolling_correlations():
        """Plot rolling correlations between assets"""
        
        # Create sample correlation data using Polars
        dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2023, 12, 31), interval="1d")
        n_days = len(dates)
        
        # Simulate correlated returns
        np.random.seed(42)
        returns_data = pl.DataFrame({
            "date": dates,
            "AAPL": np.random.normal(0.001, 0.02, n_days),
            "GOOGL": np.random.normal(0.0008, 0.025, n_days),
            "MSFT": np.random.normal(0.0012, 0.018, n_days),
            "TSLA": np.random.normal(0.002, 0.04, n_days)
        })
        
        # Calculate rolling correlations with Polars
        window_size = 252  # 1 year
        correlations = (
            returns_data.lazy()
            .sort("date")
            .with_columns([
                # Rolling correlation between AAPL and GOOGL
                pl.corr(pl.col("AAPL"), pl.col("GOOGL"))
                .rolling(window_size=window_size)
                .alias("AAPL_GOOGL_corr"),
                
                # Rolling correlation between MSFT and TSLA
                pl.corr(pl.col("MSFT"), pl.col("TSLA"))
                .rolling(window_size=window_size)
                .alias("MSFT_TSLA_corr")
            ])
            .filter(pl.col("AAPL_GOOGL_corr").is_not_null())
            .select(["date", "AAPL_GOOGL_corr", "MSFT_TSLA_corr"])
            .collect()
        )
        
        # Convert to pandas for plotting
        corr_df = correlations.to_pandas()
        corr_df.set_index('date', inplace=True)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(corr_df.index, corr_df['AAPL_GOOGL_corr'], 
               label='AAPL-GOOGL', linewidth=2, color='blue')
        ax.plot(corr_df.index, corr_df['MSFT_TSLA_corr'], 
               label='MSFT-TSLA', linewidth=2, color='red')
        
        ax.set_title('Rolling 1-Year Correlations', fontsize=14, fontweight='bold')
        ax.set_ylabel('Correlation Coefficient')
        ax.set_xlabel('Date')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Format x-axis
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        return fig, ax
    
    def plot_risk_return_scatter():
        """Create risk-return scatter plot"""
        
        # Create sample risk-return data
        symbols = [f"STOCK_{i}" for i in range(50)]
        risk_return_data = pl.DataFrame({
            "symbol": symbols,
            "returns": np.random.normal(0.08, 0.05, 50),  # Annual returns
            "volatility": np.random.uniform(0.1, 0.4, 50),  # Annual volatility
            "market_cap": np.random.uniform(1e9, 1e12, 50)  # Market cap
        })
        
        # Calculate risk-return metrics with Polars
        risk_return = (
            risk_return_data.lazy()
            .with_columns([
                (pl.col("returns") / pl.col("volatility")).alias("sharpe_ratio"),
                pl.col("market_cap").log().alias("log_market_cap")
            ])
            .filter(pl.col("volatility") > 0)
            .collect()
        )
        
        # Convert for plotting
        plot_data = risk_return.to_pandas()
        
        # Create scatter plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create scatter with color coding and size
        scatter = ax.scatter(
            plot_data['volatility'], 
            plot_data['returns'],
            s=plot_data['market_cap'] / 1e9,  # Size by market cap (billions)
            c=plot_data['sharpe_ratio'],      # Color by Sharpe ratio
            alpha=0.6,
            cmap='RdYlGn',
            edgecolors='black',
            linewidth=0.5
        )
        
        # Add quadrant lines
        ax.axhline(y=plot_data['returns'].mean(), color='gray', linestyle='--', alpha=0.7)
        ax.axvline(x=plot_data['volatility'].mean(), color='gray', linestyle='--', alpha=0.7)
        
        # Formatting
        ax.set_xlabel('Annualized Volatility', fontsize=12)
        ax.set_ylabel('Annualized Return', fontsize=12)
        ax.set_title('Risk-Return Profile by Market Cap', fontsize=14, fontweight='bold')
        
        # Format axes as percentages
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        # Add colorbar
        cbar = plt.colorbar(scatter, label='Sharpe Ratio')
        
        # Add legend for size
        legend_elements = [
            plt.scatter([], [], s=50, c='gray', alpha=0.6, label='$50B Market Cap'),
            plt.scatter([], [], s=200, c='gray', alpha=0.6, label='$200B Market Cap'),
            plt.scatter([], [], s=500, c='gray', alpha=0.6, label='$500B Market Cap')
        ]
        ax.legend(handles=legend_elements, loc='upper left', title='Market Cap')
        
        plt.tight_layout()
        plt.show()
        
        return fig, ax
    
    def create_performance_dashboard():
        """Create a multi-panel performance dashboard"""
        
        # Generate sample portfolio data
        dates = pl.date_range(pl.date(2023, 1, 1), pl.date(2023, 12, 31), interval="1d")
        strategies = ["Momentum", "Value", "Growth", "Quality"]
        
        performance_data = []
        for strategy in strategies:
            returns = np.random.normal(0.0008, 0.015, len(dates))
            cumulative = (1 + pd.Series(returns)).cumprod()
            
            for i, date in enumerate(dates):
                performance_data.append({
                    "date": date,
                    "strategy": strategy,
                    "returns": returns[i],
                    "cumulative_return": cumulative.iloc[i],
                    "drawdown": cumulative.iloc[i] / cumulative.iloc[:i+1].max() - 1 if i > 0 else 0
                })
        
        portfolio_df = pl.DataFrame(performance_data)
        plot_data = portfolio_df.to_pandas()
        
        # Create subplot dashboard
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Cumulative Returns
        for strategy in strategies:
            strategy_data = plot_data[plot_data['strategy'] == strategy]
            ax1.plot(strategy_data['date'], strategy_data['cumulative_return'], 
                    label=strategy, linewidth=2)
        
        ax1.set_title('Cumulative Returns by Strategy', fontweight='bold')
        ax1.set_ylabel('Cumulative Return')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Rolling Volatility
        for strategy in strategies:
            strategy_data = plot_data[plot_data['strategy'] == strategy]
            rolling_vol = strategy_data['returns'].rolling(30).std() * np.sqrt(252)
            ax2.plot(strategy_data['date'], rolling_vol, label=strategy, linewidth=2)
        
        ax2.set_title('30-Day Rolling Volatility', fontweight='bold')
        ax2.set_ylabel('Annualized Volatility')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Drawdown
        for strategy in strategies:
            strategy_data = plot_data[plot_data['strategy'] == strategy]
            ax3.fill_between(strategy_data['date'], strategy_data['drawdown'], 0, 
                           alpha=0.3, label=strategy)
        
        ax3.set_title('Drawdowns by Strategy', fontweight='bold')
        ax3.set_ylabel('Drawdown')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Risk-Return Summary
        summary_stats = (
            portfolio_df
            .group_by("strategy")
            .agg([
                pl.col("returns").mean().mul(252).alias("annual_return"),
                pl.col("returns").std().mul(np.sqrt(252)).alias("annual_vol")
            ])
            .with_columns([
                (pl.col("annual_return") / pl.col("annual_vol")).alias("sharpe_ratio")
            ])
            .to_pandas()
        )
        
        bars = ax4.bar(summary_stats['strategy'], summary_stats['sharpe_ratio'], 
                      color=['blue', 'green', 'red', 'orange'], alpha=0.7)
        ax4.set_title('Sharpe Ratios by Strategy', fontweight='bold')
        ax4.set_ylabel('Sharpe Ratio')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        return fig, (ax1, ax2, ax3, ax4)
    
    return plot_rolling_correlations, plot_risk_return_scatter, create_performance_dashboard

# Example usage
matplotlib_funcs = matplotlib_integration()
corr_func, scatter_func, dashboard_func = matplotlib_funcs

print("Creating visualizations...")
print("1. Rolling correlations plot")
fig1, ax1 = corr_func()

print("2. Risk-return scatter plot")
fig2, ax2 = scatter_func()

print("3. Performance dashboard")
fig3, axes = dashboard_func()
```

### Interactive Visualization with Plotly

Creating interactive charts for exploration:

```python
def interactive_visualization_patterns():
    """Patterns for interactive visualizations with Plotly"""
    
    def plotly_integration():
        """Integration with Plotly for interactive charts"""
        
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from plotly.subplots import make_subplots
            
            def create_interactive_performance_chart():
                """Create interactive performance comparison chart"""
                
                # Process data with Polars
                dates = pl.date_range(pl.date(2020, 1, 1), pl.date(2023, 12, 31), interval="1d")
                strategies = ["Strategy A", "Strategy B", "Strategy C"]
                
                # Generate sample performance data
                performance_data = []
                for strategy in strategies:
                    np.random.seed(hash(strategy) % 2**32)  # Consistent random seed per strategy
                    daily_returns = np.random.normal(0.0005, 0.015, len(dates))
                    cumulative_returns = np.cumprod(1 + daily_returns)
                    
                    for i, date in enumerate(dates):
                        performance_data.append({
                            "date": date,
                            "strategy": strategy,
                            "cumulative_return": cumulative_returns[i],
                            "daily_return": daily_returns[i]
                        })
                
                performance_df = pl.DataFrame(performance_data)
                
                # Convert to pandas for Plotly
                plot_data = performance_df.to_pandas()
                
                # Create interactive line chart
                fig = px.line(
                    plot_data, 
                    x='date', 
                    y='cumulative_return',
                    color='strategy',
                    title='Interactive Portfolio Performance Comparison',
                    labels={'cumulative_return': 'Cumulative Return', 'date': 'Date'},
                    hover_data=['daily_return']
                )
                
                # Customize layout
                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Cumulative Return",
                    hovermode='x unified',
                    template='plotly_white',
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    )
                )
                
                # Add range selector
                fig.update_layout(
                    xaxis=dict(
                        rangeselector=dict(
                            buttons=list([
                                dict(count=1, label="1Y", step="year", stepmode="backward"),
                                dict(count=2, label="2Y", step="year", stepmode="backward"),
                                dict(step="all")
                            ])
                        ),
                        rangeslider=dict(visible=True),
                        type="date"
                    )
                )
                
                # Show the plot (in Jupyter this will be interactive)
                print("Interactive performance chart created")
                fig.show()
                
                return fig
            
            def create_factor_heatmap():
                """Create interactive factor correlation heatmap"""
                
                # Create sample factor data
                factors = ["Value", "Momentum", "Quality", "Size", "Profitability", "Investment"]
                n_factors = len(factors)
                
                # Generate correlation matrix
                np.random.seed(42)
                correlation_matrix = np.random.uniform(-0.3, 0.8, (n_factors, n_factors))
                correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2  # Make symmetric
                np.fill_diagonal(correlation_matrix, 1.0)  # Diagonal should be 1
                
                # Create heatmap
                fig = px.imshow(
                    correlation_matrix,
                    x=factors,
                    y=factors,
                    color_continuous_scale='RdBu_r',
                    color_continuous_midpoint=0,
                    title="Factor Correlation Heatmap",
                    text_auto='.2f'
                )
                
                # Customize layout
                fig.update_layout(
                    template='plotly_white',
                    width=600,
                    height=500
                )
                
                fig.show()
                return fig
            
            def create_3d_risk_surface():
                """Create 3D risk surface visualization"""
                
                # Generate sample risk surface data
                returns_range = np.linspace(-0.2, 0.3, 20)
                volatility_range = np.linspace(0.1, 0.5, 20)
                returns_grid, vol_grid = np.meshgrid(returns_range, volatility_range)
                
                # Calculate utility surface (risk-adjusted returns)
                risk_aversion = 3
                utility_surface = returns_grid - 0.5 * risk_aversion * vol_grid**2
                
                # Create 3D surface plot
                fig = go.Figure(data=[go.Surface(
                    z=utility_surface,
                    x=returns_grid,
                    y=vol_grid,
                    colorscale='Viridis',
                    colorbar=dict(title="Utility Score")
                )])
                
                fig.update_layout(
                    title='3D Risk-Return Utility Surface',
                    scene=dict(
                        xaxis_title='Expected Return',
                        yaxis_title='Volatility',
                        zaxis_title='Utility Score'
                    ),
                    template='plotly_white'
                )
                
                fig.show()
                return fig
            
            return create_interactive_performance_chart, create_factor_heatmap, create_3d_risk_surface
            
        except ImportError:
            print("Plotly not available. Install with: pip install plotly")
            return None, None, None
    
    def dashboard_patterns():
        """Patterns for creating analytical dashboards"""
        
        def prepare_dashboard_data():
            """Prepare comprehensive data for dashboard"""
            
            # Create sample market data
            symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
            sectors = ["Technology", "Technology", "Technology", "Auto", "Consumer"]
            
            dates = pl.date_range(pl.date(2023, 11, 1), pl.date(2023, 12, 31), interval="1d")
            
            market_data = []
            for symbol, sector in zip(symbols, sectors):
                for date in dates:
                    market_data.append({
                        "date": date,
                        "symbol": symbol,
                        "sector": sector,
                        "returns": np.random.normal(0.001, 0.02),
                        "volume": np.random.randint(1000000, 50000000),
                        "market_cap": np.random.uniform(1e11, 3e12)
                    })
            
            market_df = pl.DataFrame(market_data)
            
            # Market overview (last 30 days)
            market_summary = (
                market_df
                .filter(pl.col("date") >= (pl.col("date").max() - pl.duration(days=30)))
                .group_by("date")
                .agg([
                    pl.col("returns").mean().alias("market_return"),
                    pl.col("returns").std().alias("market_vol"),
                    pl.col("volume").sum().alias("total_volume"),
                    pl.col("symbol").count().alias("active_stocks")
                ])
            )
            
            # Sector performance (last 7 days)
            sector_performance = (
                market_df
                .filter(pl.col("date") >= (pl.col("date").max() - pl.duration(days=7)))
                .group_by("sector")
                .agg([
                    pl.col("returns").mean().alias("weekly_return"),
                    pl.col("market_cap").sum().alias("sector_mcap"),
                    pl.col("symbol").n_unique().alias("stock_count")
                ])
                .sort("weekly_return", descending=True)
            )
            
            # Top performers (yesterday)
            top_performers = (
                market_df
                .filter(pl.col("date") == market_df.select(pl.col("date").max()).item())
                .sort("returns", descending=True)
                .head(10)
                .select(["symbol", "returns", "volume", "market_cap"])
            )
            
            return {
                'market_summary': market_summary.to_pandas(),
                'sector_performance': sector_performance.to_pandas(), 
                'top_performers': top_performers.to_pandas()
            }
        
        def create_summary_table():
            """Create formatted summary tables"""
            
            # Generate portfolio summary data
            asset_classes = ["Equities", "Bonds", "Commodities", "REITs"]
            portfolio_data = pl.DataFrame({
                "asset_class": asset_classes,
                "weight": [0.6, 0.25, 0.10, 0.05],
                "returns": [0.08, 0.04, 0.06, 0.07],
                "volatility": [0.15, 0.08, 0.20, 0.12],
                "position_count": [150, 20, 15, 25]
            })
            
            # Convert to pandas for formatting
            summary_df = portfolio_data.to_pandas()
            
            # Format for display
            summary_df['allocation'] = summary_df['weight'].apply(lambda x: f"{x:.1%}")
            summary_df['avg_return'] = summary_df['returns'].apply(lambda x: f"{x:.2%}")
            summary_df['volatility_fmt'] = summary_df['volatility'].apply(lambda x: f"{x:.2%}")
            
            print("Portfolio Summary:")
            print(summary_df[['asset_class', 'allocation', 'avg_return', 'volatility_fmt', 'position_count']])
            
            return summary_df
        
        return prepare_dashboard_data, create_summary_table
    
    return plotly_integration, dashboard_patterns

# Example usage
interactive_patterns = interactive_visualization_patterns()
plotly_funcs, dashboard_funcs = interactive_patterns

if plotly_funcs[0] is not None:  # Check if Plotly is available
    perf_func, heatmap_func, surface_func = plotly_funcs
    
    print("Creating interactive visualizations...")
    print("1. Interactive performance chart")
    perf_fig = perf_func()
    
    print("2. Factor correlation heatmap")
    heatmap_fig = heatmap_func()
    
    print("3. 3D risk surface")
    surface_fig = surface_func()

# Dashboard data preparation
data_prep_func, table_func = dashboard_funcs
dashboard_data = data_prep_func()
summary_table = table_func()

print("\nDashboard data prepared:")
for key, df in dashboard_data.items():
    print(f"  {key}: {df.shape}")
```

---

## Troubleshooting Guide

### Performance Debugging

Systematic approach to diagnosing and fixing performance issues:

```python
def performance_debugging():
    """Debug performance issues in Polars queries"""
    
    def query_performance_analysis():
        """Analyze and optimize slow queries"""
        
        def diagnose_slow_query():
            """Step-by-step query diagnosis"""
            
            # Create sample data for testing
            sample_size = 1000000
            sample_data = pl.DataFrame({
                "symbol": np.random.choice(["AAPL", "GOOGL", "MSFT", "TSLA"] * 100, sample_size),
                "date": pl.date_range(pl.date(2020, 1, 1), pl.date(2023, 12, 31), interval="1d")[:sample_size],
                "volume": np.random.randint(100000, 50000000, sample_size),
                "returns": np.random.normal(0.001, 0.02, sample_size),
                "market_cap": np.random.uniform(1e9, 1e12, sample_size)
            })
            sample_data.write_parquet("performance_test.parquet")
            
            # Step 1: Check the query plan
            slow_query = (
                pl.scan_parquet("performance_test.parquet")
                .filter(pl.col("volume") > 1000000)
                .filter(pl.col("symbol").is_in(["AAPL", "GOOGL", "MSFT"]))
                .group_by("symbol")
                .agg([
                    pl.col("returns").mean().alias("avg_return"),
                    pl.col("returns").std().alias("volatility")
                ])
            )
            
            # Examine query plan for optimization opportunities
            print("=== Query Plan Analysis ===")
            print("Unoptimized plan:")
            print(slow_query.explain(optimized=False))
            print("\nOptimized plan:")
            print(slow_query.explain(optimized=True))
            
            # Step 2: Profile the execution
            import time
            start_time = time.time()
            result = slow_query.collect()
            execution_time = time.time() - start_time
            print(f"\nExecution time: {execution_time:.4f} seconds")
            print(f"Result shape: {result.shape}")
            
            return result, execution_time
        
        def optimize_query():
            """Apply optimizations to improve performance"""
            
            print("=== Query Optimization ===")
            
            # Optimization 1: More selective filtering first
            optimized_query = (
                pl.scan_parquet("performance_test.parquet")
                .filter(pl.col("symbol").is_in(["AAPL", "GOOGL", "MSFT"]))  # More selective first
                .filter(pl.col("volume") > 1000000)                        # Less selective after
                .select(["symbol", "returns", "volume"])                   # Project only needed columns
                .group_by("symbol")
                .agg([
                    pl.col("returns").mean().alias("avg_return"),
                    pl.col("returns").std().alias("volatility")
                ])
            )
            
            # Test optimized version
            import time
            start_time = time.time()
            result = optimized_query.collect()
            execution_time = time.time() - start_time
            print(f"Optimized execution time: {execution_time:.4f} seconds")
            
            # Additional optimization: streaming
            streaming_query = (
                pl.scan_parquet("performance_test.parquet")
                .filter(pl.col("symbol").is_in(["AAPL", "GOOGL", "MSFT"]))
                .filter(pl.col("volume") > 1000000)
                .select(["symbol", "returns", "volume"])
                .group_by("symbol")
                .agg([
                    pl.col("returns").mean().alias("avg_return"),
                    pl.col("returns").std().alias("volatility")
                ])
            )
            
            start_time = time.time()
            streaming_result = streaming_query.collect(streaming=True)
            streaming_time = time.time() - start_time
            print(f"Streaming execution time: {streaming_time:.4f} seconds")
            
            return result, execution_time, streaming_result, streaming_time
        
        def performance_comparison():
            """Compare different approaches systematically"""
            
            approaches = {
                "Unoptimized": lambda: (
                    pl.scan_parquet("performance_test.parquet")
                    .filter(pl.col("volume") > 1000000)  # Less selective first
                    .filter(pl.col("symbol").is_in(["AAPL", "GOOGL", "MSFT"]))
                    .group_by("symbol")
                    .agg([pl.col("returns").mean(), pl.col("returns").std()])
                    .collect()
                ),
                "Optimized Order": lambda: (
                    pl.scan_parquet("performance_test.parquet")
                    .filter(pl.col("symbol").is_in(["AAPL", "GOOGL", "MSFT"]))  # More selective first
                    .filter(pl.col("volume") > 1000000)
                    .group_by("symbol")
                    .agg([pl.col("returns").mean(), pl.col("returns").std()])
                    .collect()
                ),
                "With Projection": lambda: (
                    pl.scan_parquet("performance_test.parquet")
                    .filter(pl.col("symbol").is_in(["AAPL", "GOOGL", "MSFT"]))
                    .select(["symbol", "returns", "volume"])  # Early projection
                    .filter(pl.col("volume") > 1000000)
                    .group_by("symbol")
                    .agg([pl.col("returns").mean(), pl.col("returns").std()])
                    .collect()
                ),
                "Streaming": lambda: (
                    pl.scan_parquet("performance_test.parquet")
                    .filter(pl.col("symbol").is_in(["AAPL", "GOOGL", "MSFT"]))
                    .select(["symbol", "returns", "volume"])
                    .filter(pl.col("volume") > 1000000)
                    .group_by("symbol")
                    .agg([pl.col("returns").mean(), pl.col("returns").std()])
                    .collect(streaming=True)
                )
            }
            
            import time
            results = {}
            
            print("=== Performance Comparison ===")
            for name, query_func in approaches.items():
                # Warm up
                query_func()
                
                # Time the query
                start_time = time.time()
                result = query_func()
                execution_time = time.time() - start_time
                
                results[name] = {
                    'time': execution_time,
                    'shape': result.shape
                }
                
                print(f"{name:20s}: {execution_time:.4f}s")
            
            # Find best performance
            best_approach = min(results.keys(), key=lambda k: results[k]['time'])
            best_time = results[best_approach]['time']
            
            print(f"\nBest approach: {best_approach} ({best_time:.4f}s)")
            print("Speedup ratios:")
            for name, stats in results.items():
                speedup = stats['time'] / best_time
                print(f"  {name:20s}: {speedup:.2f}x")
            
            return results
        
        return diagnose_slow_query, optimize_query, performance_comparison
    
    def memory_usage_debugging():
        """Debug memory usage issues"""
        
        def memory_profiling():
            """Profile memory usage during operations"""
            
            import psutil
            import gc
            
            def get_memory_mb():
                return psutil.Process().memory_info().rss / 1024 / 1024
            
            print("=== Memory Usage Profiling ===")
            print(f"Initial memory: {get_memory_mb():.1f} MB")
            
            # Create progressively larger datasets
            sizes = [100000, 500000, 1000000]
            
            for size in sizes:
                print(f"\nTesting with {size:,} rows:")
                
                # Create data
                df = pl.DataFrame({
                    "values": np.random.randn(size),
                    "categories": np.random.choice(["A", "B", "C"], size),
                    "dates": pl.date_range(pl.date(2023, 1, 1), pl.date(2023, 12, 31), interval="1d")[:size]
                })
                print(f"  After creating DataFrame: {get_memory_mb():.1f} MB")
                
                # Process data
                processed = df.with_columns([
                    pl.col("values").rolling_mean(window_size=min(100, size)).alias("ma_100"),
                    pl.col("values").rolling_std(window_size=min(100, size)).alias("std_100")
                ])
                print(f"  After processing: {get_memory_mb():.1f} MB")
                
                # Clean up
                del df, processed
                gc.collect()
                print(f"  After cleanup: {get_memory_mb():.1f} MB")
        
        def identify_memory_hotspots():
            """Identify operations that use most memory"""
            
            import psutil
            import gc
            
            # Create test data
            test_df = pl.DataFrame({
                "symbol": ["AAPL"] * 1000000,
                "price": np.random.randn(1000000) * 10 + 150,
                "volume": np.random.randint(100000, 10000000, 1000000),
                "date": pl.date_range(pl.date(2023, 1, 1), pl.date(2023, 12, 31), interval="1d")[:1000000]
            })
            
            operations = {
                'rolling_mean': lambda df: df.with_columns([pl.col("price").rolling_mean(window_size=252).alias("ma")]),
                'rolling_std': lambda df: df.with_columns([pl.col("price").rolling_std(window_size=252).alias("vol")]),
                'pct_change': lambda df: df.with_columns([pl.col("price").pct_change().alias("returns")]),
                'groupby_agg': lambda df: df.group_by("symbol").agg([pl.col("price").mean(), pl.col("volume").sum()]),
                'sort': lambda df: df.sort(["symbol", "date"]),
                'join_self': lambda df: df.join(df.select(["symbol", "price"]), on="symbol", how="left")
            }
            
            memory_usage = {}
            
            print("=== Memory Hotspot Analysis ===")
            for op_name, operation in operations.items():
                gc.collect()
                start_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                result = operation(test_df)
                
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_usage[op_name] = end_memory - start_memory
                
                print(f"{op_name:15s}: +{memory_usage[op_name]:6.1f} MB")
                
                # Clean up result
                del result
                gc.collect()
            
            # Identify most memory-intensive operations
            max_memory_op = max(memory_usage.keys(), key=lambda k: memory_usage[k])
            print(f"\nMost memory-intensive: {max_memory_op} ({memory_usage[max_memory_op]:.1f} MB)")
            
            return memory_usage
        
        def memory_optimization_strategies():
            """Demonstrate memory optimization strategies"""
            
            print("=== Memory Optimization Strategies ===")
            
            # Strategy 1: Use appropriate data types
            print("1. Data type optimization:")
            
            # Inefficient types
            inefficient_df = pl.DataFrame({
                "price": [100.5] * 100000,           # Float64 (8 bytes)
                "volume": [1000000] * 100000,        # Int64 (8 bytes)  
                "symbol": ["AAPL"] * 100000,         # String (variable)
                "is_active": [True] * 100000         # Boolean (1 byte)
            })
            
            # Efficient types
            efficient_df = inefficient_df.with_columns([
                pl.col("price").cast(pl.Float32),     # Float32 (4 bytes) 
                pl.col("volume").cast(pl.UInt32),     # UInt32 (4 bytes)
                pl.col("symbol").cast(pl.Categorical) # Categorical (efficient for repeated values)
            ])
            
            print(f"  Inefficient memory estimate: ~{len(inefficient_df) * 25 / 1024 / 1024:.1f} MB")
            print(f"  Efficient memory estimate: ~{len(efficient_df) * 13 / 1024 / 1024:.1f} MB")
            
            # Strategy 2: Column selection
            print("\n2. Column selection optimization:")
            large_df = pl.DataFrame({
                f"col_{i}": np.random.randn(100000) for i in range(20)
            })
            
            print(f"  Full dataset: 20 columns")
            print(f"  Selected columns: 3 columns (85% memory reduction)")
            
            # Strategy 3: Lazy evaluation
            print("\n3. Lazy evaluation benefits:")
            print("  - Automatic query optimization")
            print("  - Predicate pushdown")
            print("  - Column pruning")
            print("  - Memory-efficient streaming")
            
        return memory_profiling, identify_memory_hotspots, memory_optimization_strategies
    
    return query_performance_analysis, memory_usage_debugging

# Example usage
perf_debug = performance_debugging()
query_analysis, memory_debug = perf_debug

print("=== Performance Debugging Demo ===")

# Run query analysis
analysis_funcs = query_analysis()
diagnose_func, optimize_func, compare_func = analysis_funcs

print("Running query diagnosis...")
result, exec_time = diagnose_func()

print("\nRunning optimization comparison...")
comparison_results = compare_func()

# Run memory debugging
memory_funcs = memory_debug()
profile_func, hotspot_func, strategy_func = memory_funcs

print("\nRunning memory profiling...")
profile_func()

print("\nIdentifying memory hotspots...")
hotspots = hotspot_func()

print("\nMemory optimization strategies...")
strategy_func()
```

---

## Error Handling Patterns

### Robust File Operations

Comprehensive error handling for file I/O operations:

```python
def error_handling_patterns():
    """Robust error handling for production systems"""
    
    def file_handling_errors():
        """Handle file-related errors gracefully"""
        
        def robust_file_loading(file_path: str):
            """Load files with comprehensive error handling"""
            
            from pathlib import Path
            
            try:
                # Check if file exists
                if not Path(file_path).exists():
                    raise FileNotFoundError(f"File not found: {file_path}")
                
                # Check file size (warn if very large)
                file_size = Path(file_path).stat().st_size / (1024 * 1024)  # MB
                if file_size > 1000:  # > 1GB
                    print(f"⚠️  Warning: Large file ({file_size:.1f} MB). Consider streaming.")
                
                # Try to load with error handling
                if file_path.endswith('.parquet'):
                    df = pl.read_parquet(file_path)
                elif file_path.endswith('.csv'):
                    df = pl.read_csv(file_path)
                else:
                    raise ValueError(f"Unsupported file format: {file_path}")
                
                # Validate loaded data
                if len(df) == 0:
                    raise ValueError("Loaded DataFrame is empty")
                
                print(f"✅ Successfully loaded {len(df):,} rows from {file_path}")
                return df
                
            except FileNotFoundError as e:
                print(f"❌ File error: {e}")
                print("💡 Suggestion: Check file path and permissions")
                return None
            except pl.exceptions.ComputeError as e:
                print(f"❌ Polars compute error: {e}")
                print("💡 Suggestion: Check file format and data quality")
                return None
            except Exception as e:
                print(f"❌ Unexpected error loading {file_path}: {e}")
                print("💡 Suggestion: Check file integrity and available memory")
                return None
        
        def robust_file_writing(df: pl.DataFrame, output_path: str):
            """Write files with error handling and validation"""
            
            from pathlib import Path
            
            try:
                # Validate inputs
                if df is None:
                    raise ValueError("DataFrame is None")
                if len(df) == 0:
                    raise ValueError("Cannot write empty DataFrame")
                
                # Ensure output directory exists
                output_dir = Path(output_path).parent
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Create backup if file exists
                if Path(output_path).exists():
                    backup_path = f"{output_path}.backup"
                    Path(output_path).rename(backup_path)
                    print(f"📁 Created backup: {backup_path}")
                
                # Write the file
                if output_path.endswith('.parquet'):
                    df.write_parquet(output_path)
                elif output_path.endswith('.csv'):
                    df.write_csv(output_path)
                else:
                    raise ValueError(f"Unsupported output format: {output_path}")
                
                # Verify write was successful
                if output_path.endswith('.parquet'):
                    test_read = pl.read_parquet(output_path)
                else:
                    test_read = pl.read_csv(output_path)
                    
                if len(test_read) != len(df):
                    raise ValueError("Write verification failed - row count mismatch")
                
                print(f"✅ Successfully wrote {len(df):,} rows to {output_path}")
                return True
                
            except PermissionError as e:
                print(f"❌ Permission error: {e}")
                print("💡 Suggestion: Check write permissions for the output directory")
                return False
            except OSError as e:
                print(f"❌ OS error: {e}")
                print("💡 Suggestion: Check disk space and file path validity")
                return False
            except Exception as e:
                print(f"❌ Error writing to {output_path}: {e}")
                print("💡 Suggestion: Check DataFrame validity and output path")
                return False
        
        def safe_data_pipeline(input_files: list, output_file: str):
            """Create a robust data processing pipeline"""
            
            successful_loads = []
            failed_loads = []
            
            print(f"🔄 Processing {len(input_files)} files...")
            
            # Load all files with error handling
            for file_path in input_files:
                print(f"\nProcessing: {file_path}")
                df = robust_file_loading(file_path)
                
                if df is not None:
                    successful_loads.append(df)
                else:
                    failed_loads.append(file_path)
            
            # Report loading results
            print(f"\n📊 Loading Summary:")
            print(f"  Successful: {len(successful_loads)}")
            print(f"  Failed: {len(failed_loads)}")
            
            if failed_loads:
                print("  Failed files:")
                for file_path in failed_loads:
                    print(f"    - {file_path}")
            
            # Proceed only if we have some data
            if successful_loads:
                try:
                    # Combine all successful loads
                    combined_df = pl.concat(successful_loads)
                    print(f"📁 Combined {len(combined_df):,} total rows")
                    
                    # Write output with error handling
                    success = robust_file_writing(combined_df, output_file)
                    
                    if success:
                        print(f"✅ Pipeline completed successfully")
                        return combined_df
                    else:
                        print(f"❌ Pipeline failed at output stage")
                        return None
                        
                except Exception as e:
                    print(f"❌ Error combining data: {e}")
                    return None
            else:
                print("❌ No data to process - all files failed to load")
                return None
        
        return robust_file_loading, robust_file_writing, safe_data_pipeline
    
    def data_validation_patterns():
        """Validate data quality and handle issues"""
        
        def validate_financial_data(df: pl.DataFrame):
            """Comprehensive validation for financial data"""
            
            validation_results = {
                'is_valid': True,
                'errors': [],
                'warnings': []
            }
            
            print("🔍 Validating financial data...")
            
            # Check required columns
            required_columns = ['symbol', 'date', 'price', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                validation_results['errors'].append(f"Missing required columns: {missing_columns}")
                validation_results['is_valid'] = False
            
            # Check data types
            if 'price' in df.columns:
                if df['price'].dtype not in [pl.Float32, pl.Float64]:
                    validation_results['errors'].append("Price column must be float type")
                    validation_results['is_valid'] = False
            
            # Check for negative prices
            if 'price' in df.columns:
                negative_prices = (df['price'] <= 0).sum()
                if negative_prices > 0:
                    validation_results['warnings'].append(f"{negative_prices} negative/zero prices found")
            
            # Check for missing dates
            if 'date' in df.columns:
                null_dates = df['date'].null_count()
                if null_dates > 0:
                    validation_results['errors'].append(f"{null_dates} null dates found")
                    validation_results['is_valid'] = False
            
            # Check for duplicates
            if all(col in df.columns for col in ['symbol', 'date']):
                duplicates = df.select(['symbol', 'date']).is_duplicated().sum()
                if duplicates > 0:
                    validation_results['warnings'].append(f"{duplicates} duplicate symbol-date combinations")
            
            # Check date range reasonableness
            if 'date' in df.columns:
                min_date = df['date'].min()
                max_date = df['date'].max()
                today = pl.date.today()
                
                if max_date > today:
                    validation_results['warnings'].append(f"Future dates found (max: {max_date})")
                
                if min_date < pl.date(1900, 1, 1):
                    validation_results['warnings'].append(f"Very old dates found (min: {min_date})")
            
            # Report results
            if not validation_results['is_valid']:
                print("❌ Data validation failed:")
                for error in validation_results['errors']:
                    print(f"  🚫 Error: {error}")
            
            if validation_results['warnings']:
                print("⚠️  Data validation warnings:")
                for warning in validation_results['warnings']:
                    print(f"  ⚠️  Warning: {warning}")
            
            if validation_results['is_valid'] and not validation_results['warnings']:
                print("✅ Data validation passed")
            
            return validation_results
        
        def handle_data_quality_issues(df: pl.DataFrame):
            """Automatically handle common data quality issues"""
            
            print("🔧 Handling data quality issues...")
            
            original_len = len(df)
            
            # Remove rows with negative/zero prices
            if 'price' in df.columns:
                df = df.filter(pl.col("price") > 0)
                removed_negative = original_len - len(df)
                if removed_negative > 0:
                    print(f"  🧹 Removed {removed_negative} rows with negative/zero prices")
                    original_len = len(df)
            
            # Handle duplicates (keep last occurrence)
            if 'symbol' in df.columns and 'date' in df.columns:
                df = df.unique(subset=['symbol', 'date'], keep='last', maintain_order=True)
                removed_duplicates = original_len - len(df)
                if removed_duplicates > 0:
                    print(f"  🧹 Removed {removed_duplicates} duplicate rows")
            
            # Forward fill missing prices within each symbol
            if 'price' in df.columns:
                null_prices_before = df['price'].null_count()
                if null_prices_before > 0:
                    df = df.with_columns([
                        pl.col("price").forward_fill().over("symbol")
                    ])
                    null_prices_after = df['price'].null_count()
                    filled_prices = null_prices_before - null_prices_after
                    if filled_prices > 0:
                        print(f"  🔄 Forward filled {filled_prices} missing prices")
            
            # Handle missing volume (set to 0)
            if 'volume' in df.columns:
                null_volume_before = df['volume'].null_count()
                if null_volume_before > 0:
                    df = df.with_columns([
                        pl.col("volume").fill_null(0)
                    ])
                    print(f"  🔄 Filled {null_volume_before} missing volume values with 0")
            
            print(f"  📊 Final dataset: {len(df):,} rows")
            return df
        
        def create_data_quality_report(df: pl.DataFrame):
            """Generate comprehensive data quality report"""
            
            print("📋 Generating Data Quality Report")
            print("=" * 50)
            
            # Basic statistics
            print(f"Dataset Shape: {df.shape}")
            print(f"Memory Usage: ~{df.estimated_size('mb'):.1f} MB")
            
            # Column information
            print("\nColumn Information:")
            for col in df.columns:
                dtype = df[col].dtype
                null_count = df[col].null_count()
                null_pct = null_count / len(df) * 100
                unique_count = df[col].n_unique()
                
                print(f"  {col:15s}: {str(dtype):15s} | Nulls: {null_count:6,} ({null_pct:5.1f}%) | Unique: {unique_count:6,}")
            
            # Data quality metrics
            if 'price' in df.columns and 'volume' in df.columns:
                print("\nFinancial Data Quality:")
                
                # Price statistics
                price_stats = df.select([
                    pl.col("price").min().alias("min_price"),
                    pl.col("price").max().alias("max_price"),
                    pl.col("price").mean().alias("avg_price"),
                    (pl.col("price") <= 0).sum().alias("invalid_prices")
                ]).to_dicts()[0]
                
                print(f"  Price Range: ${price_stats['min_price']:.2f} - ${price_stats['max_price']:.2f}")
                print(f"  Average Price: ${price_stats['avg_price']:.2f}")
                print(f"  Invalid Prices: {price_stats['invalid_prices']:,}")
                
                # Volume statistics
                volume_stats = df.select([
                    pl.col("volume").min().alias("min_volume"),
                    pl.col("volume").max().alias("max_volume"),
                    pl.col("volume").mean().alias("avg_volume"),
                    (pl.col("volume") < 0).sum().alias("invalid_volumes")
                ]).to_dicts()[0]
                
                print(f"  Volume Range: {volume_stats['min_volume']:,} - {volume_stats['max_volume']:,}")
                print(f"  Average Volume: {volume_stats['avg_volume']:,.0f}")
                print(f"  Invalid Volumes: {volume_stats['invalid_volumes']:,}")
            
            # Date range analysis
            if 'date' in df.columns:
                print("\nTemporal Coverage:")
                date_stats = df.select([
                    pl.col("date").min().alias("start_date"),
                    pl.col("date").max().alias("end_date"),
                    pl.col("date").n_unique().alias("unique_dates")
                ]).to_dicts()[0]
                
                print(f"  Date Range: {date_stats['start_date']} to {date_stats['end_date']}")
                print(f"  Unique Dates: {date_stats['unique_dates']:,}")
                
                # Check for gaps
                expected_days = (date_stats['end_date'] - date_stats['start_date']).days + 1
                missing_days = expected_days - date_stats['unique_dates']
                if missing_days > 0:
                    print(f"  Missing Dates: ~{missing_days} days")
            
            print("=" * 50)
        
        return validate_financial_data, handle_data_quality_issues, create_data_quality_report
    
    return file_handling_errors, data_validation_patterns

# Example usage and demonstration
error_handling = error_handling_patterns()
file_handlers, data_validators = error_handling

# Get the functions
robust_load, robust_write, safe_pipeline = file_handlers
validate_data, clean_data, quality_report = data_validators

# Create sample data for testing
print("🧪 Creating sample data for error handling demonstration...")

# Create valid sample data
valid_data = pl.DataFrame({
    "symbol": ["AAPL", "GOOGL", "MSFT"] * 1000,
    "date": pl.date_range(pl.date(2023, 1, 1), pl.date(2023, 12, 31), interval="1d")[:3000],
    "price": np.random.uniform(100, 300, 3000),
    "volume": np.random.randint(100000, 10000000, 3000)
})

# Create problematic sample data
problematic_data = pl.DataFrame({
    "symbol": ["AAPL", "GOOGL", None, "MSFT"],
    "date": [pl.date(2023, 1, 1), pl.date(2023, 1, 2), None, pl.date(2025, 1, 1)],  # Future date
    "price": [150.0, -50.0, 200.0, None],  # Negative and null prices
    "volume": [1000000, 2000000, -500000, 1500000]  # Negative volume
})

# Save sample files
valid_data.write_parquet("valid_sample.parquet")
problematic_data.write_parquet("problematic_sample.parquet")

print("\n" + "="*60)
print("ERROR HANDLING DEMONSTRATION")
print("="*60)

print("\n1. Testing robust file loading...")
valid_result = robust_load("valid_sample.parquet")
invalid_result = robust_load("nonexistent_file.parquet")

print("\n2. Testing data validation...")
validation_result = validate_data(problematic_data)

print("\n3. Testing data quality handling...")
cleaned_data = clean_data(problematic_data)

print("\n4. Generating data quality report...")
quality_report(valid_data)

print("\n5. Testing safe pipeline...")
test_files = ["valid_sample.parquet", "problematic_sample.parquet", "nonexistent_file.parquet"]
pipeline_result = safe_pipeline(test_files, "pipeline_output.parquet")
```

---

## Navigation

This practical implementation guide is part of the comprehensive Polars guide series:

### Related Guides

- **[Main README](./README.md)** - Overview and getting started
- **[Quantitative Research](./polars_quantitative_research.md)** - Advanced quantitative research techniques  
- **[Performance & Large Datasets](./polars_performance_and_large_datasets.md)** - Performance optimization strategies
- **[Advanced Techniques](./polars_advanced_techniques.md)** - Advanced Polars patterns and techniques
- **[Streaming & Memory](./polars_streaming_and_memory.md)** - Memory-efficient processing strategies
- **[Pandas Integration](./polars_pandas_integration.md)** - Seamless integration with Pandas workflows

### Quick Reference

**Data Loading Best Practices:**
```python
# Always prefer lazy evaluation
df = pl.scan_parquet("data.parquet").filter(...).collect()

# Use appropriate data types
schema = {"price": pl.Float64, "symbol": pl.Categorical}
df = pl.read_csv("data.csv", dtypes=schema)
```

**Error Handling Pattern:**
```python
try:
    result = pl.scan_parquet("file.parquet").collect()
except pl.exceptions.ComputeError as e:
    print(f"Polars error: {e}")
    # Handle gracefully
```

**Memory Optimization:**
```python
# Use streaming for large datasets
result = query.collect(streaming=True)

# Select only needed columns early
df = pl.scan_parquet("large.parquet").select(["col1", "col2"]).collect()
```

**Visualization Integration:**
```python
# Process with Polars, visualize with Pandas/Matplotlib
polars_result = pl.scan_parquet("data.parquet").group_by("category").agg([...]).collect()
pandas_df = polars_result.to_pandas()
pandas_df.plot()
```

This guide provides practical, real-world examples for implementing Polars in quantitative research workflows. Each section includes extensive code examples with error handling and best practices that you can immediately apply to your projects.