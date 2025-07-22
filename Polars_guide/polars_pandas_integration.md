# Polars-Pandas Integration

Master the strategic use of both libraries for optimal workflows, efficient conversions, and seamless integration with the scientific Python ecosystem.

## Table of Contents
- [Strategic Decision Framework](#strategic-decision-framework)
- [Efficient Conversion Patterns](#efficient-conversion-patterns)
- [Hybrid Workflows](#hybrid-workflows)
- [NumPy Integration Patterns](#numpy-integration-patterns)
- [Advanced Integration Patterns](#advanced-integration-patterns)

---

## Strategic Decision Framework

The key to effective Polars-Pandas integration is understanding when to use each library and how to transition between them efficiently.

### Decision Tree for Library Selection

```python
def integration_decision_framework():
    """
    Decision tree for when to use Polars vs Pandas
    
    Use Polars for:
    - Initial data loading (> 10MB files)
    - Data cleaning and preprocessing  
    - Complex filtering and grouping
    - Time series resampling
    - Large joins and aggregations
    - ETL pipelines
    
    Use Pandas for:
    - Exploratory data analysis
    - Statistical analysis and modeling
    - Visualization with matplotlib/seaborn
    - Integration with scikit-learn
    - Legacy code compatibility
    - Interactive Jupyter workflows
    """
    pass
```

### Performance Comparison Guidelines

```python
def performance_guidelines():
    """Guidelines for when each library excels"""
    
    performance_matrix = {
        "Data Loading (>100MB)": "Polars: 5-10x faster",
        "Filtering Operations": "Polars: 10-50x faster", 
        "Group By Aggregations": "Polars: 10-100x faster",
        "Window Functions": "Polars: 5-20x faster",
        "Statistical Functions": "Pandas: More complete",
        "Plotting Integration": "Pandas: Better ecosystem",
        "Machine Learning": "Pandas: sklearn integration",
        "Memory Usage": "Polars: 2-5x more efficient"
    }
    
    return performance_matrix
```

---

## Efficient Conversion Patterns

### Zero-Copy Conversions

```python
def efficient_conversions():
    """Best practices for converting between Polars and Pandas"""
    
    # Polars to Pandas - use Arrow for zero-copy when possible
    pl_df = pl.read_parquet("data.parquet")
    
    # ✅ Efficient conversion via PyArrow (zero-copy)
    pd_df = pl_df.to_pandas(use_pyarrow_extension_array=True)
    
    # ✅ Memory-efficient conversion for large DataFrames
    pd_df_optimized = pl_df.to_pandas(
        use_pyarrow_extension_array=True,
        date_as_object=False  # Keep datetime as proper dtype
    )
    
    # Pandas to Polars - efficient patterns
    pl_from_pd = pl.from_pandas(pd_df)
    
    # ✅ Better - specify schema to avoid inference overhead
    pl_from_pd_schema = pl.from_pandas(
        pd_df,
        schema_overrides={
            "price": pl.Float64,
            "volume": pl.UInt64,
            "symbol": pl.Categorical
        }
    )
    
    return pl_df, pd_df_optimized, pl_from_pd_schema

def memory_efficient_conversions():
    """Utilities for memory-efficient Polars-Pandas conversions"""
    
    def convert_large_polars_to_pandas(pl_df: pl.DataFrame, chunk_size: int = 1000000):
        """Convert large Polars DataFrame to Pandas in chunks"""
        total_rows = len(pl_df)
        chunks = []
        
        for start_idx in range(0, total_rows, chunk_size):
            end_idx = min(start_idx + chunk_size, total_rows)
            chunk = pl_df[start_idx:end_idx].to_pandas()
            chunks.append(chunk)
        
        return pd.concat(chunks, ignore_index=True)
    
    def selective_conversion(pl_df: pl.DataFrame, pandas_columns: list):
        """Convert only specific columns to Pandas, keep rest in Polars"""
        # Keep heavy processing columns in Polars
        polars_part = pl_df.select(pl.exclude(pandas_columns))
        
        # Convert analysis columns to Pandas
        pandas_part = pl_df.select(pandas_columns).to_pandas()
        
        return polars_part, pandas_part
    
    def dtype_optimized_conversion(pl_df: pl.DataFrame):
        """Convert with optimal dtypes for memory efficiency"""
        return pl_df.to_pandas(
            use_pyarrow_extension_array=True,  # Use Arrow extension arrays
            date_as_object=False,              # Keep proper datetime dtypes
        )
    
    return convert_large_polars_to_pandas, selective_conversion, dtype_optimized_conversion
```

---

## Hybrid Workflows

### Common Hybrid Patterns

```python
def hybrid_workflow_patterns():
    """Common hybrid workflows for quantitative research"""
    
    # Pattern 1: Polars for ETL, Pandas for analysis
    def etl_analysis_pattern():
        # Heavy lifting with Polars
        processed_data = (
            pl.scan_parquet("raw_data/*.parquet")
            .filter(pl.col("date") >= "2020-01-01")
            .with_columns([
                pl.col("price").pct_change().alias("returns"),
                pl.col("volume").log().alias("log_volume")
            ])
            .group_by(["symbol", pl.col("date").dt.month().alias("month")])
            .agg([
                pl.col("returns").mean().alias("monthly_return"),
                pl.col("log_volume").mean().alias("avg_log_volume")
            ])
            .collect()
        )
        
        # Switch to Pandas for statistical analysis
        pd_data = processed_data.to_pandas()
        
        # Complex statistical operations in Pandas
        correlation_matrix = pd_data.pivot(
            index='month', 
            columns='symbol', 
            values='monthly_return'
        ).corr()
        
        return processed_data, pd_data, correlation_matrix
    
    # Pattern 2: Polars for data prep, Pandas for modeling
    def data_prep_modeling_pattern():
        # Feature engineering with Polars (fast)
        features = (
            pl.scan_parquet("market_data.parquet")
            .sort(["symbol", "date"])
            .with_columns([
                # Technical indicators
                pl.col("price").rolling_mean(window_size=20).over("symbol").alias("sma_20"),
                pl.col("price").rolling_std(window_size=20).over("symbol").alias("vol_20"),
                pl.col("volume").rolling_mean(window_size=10).over("symbol").alias("avg_volume"),
                
                # Lagged features
                pl.col("returns").shift(1).over("symbol").alias("returns_lag1"),
                pl.col("returns").shift(5).over("symbol").alias("returns_lag5"),
                
                # Cross-sectional ranks
                pl.col("market_cap").rank().over("date").alias("mcap_rank")
            ])
            .drop_nulls()
            .collect()
        )
        
        # Convert to Pandas for sklearn integration
        features_pd = features.to_pandas()
        
        # Machine learning with scikit-learn
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        
        X = features_pd[['sma_20', 'vol_20', 'avg_volume', 'returns_lag1', 'returns_lag5', 'mcap_rank']]
        y = features_pd['returns']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        model = RandomForestRegressor(n_estimators=100)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        
        return features, features_pd, model, predictions
    
    return etl_analysis_pattern(), data_prep_modeling_pattern()
```

### Advanced Integration Patterns

```python
def advanced_integration_patterns():
    """Advanced patterns for seamless Polars-Pandas integration"""
    
    # Pattern 1: Lazy evaluation with Pandas compatibility layer
    class HybridDataFrame:
        """Wrapper that provides lazy Polars operations with Pandas compatibility"""
        
        def __init__(self, data):
            if isinstance(data, pl.LazyFrame):
                self._lazy = data
            elif isinstance(data, pl.DataFrame):
                self._lazy = data.lazy()
            elif isinstance(data, pd.DataFrame):
                self._lazy = pl.from_pandas(data).lazy()
            else:
                raise ValueError("Unsupported data type")
        
        def filter(self, condition):
            """Polars-style filtering"""
            return HybridDataFrame(self._lazy.filter(condition))
        
        def select(self, columns):
            """Polars-style selection"""
            return HybridDataFrame(self._lazy.select(columns))
        
        def to_pandas(self):
            """Convert to Pandas for analysis"""
            return self._lazy.collect().to_pandas()
        
        def to_polars(self):
            """Get Polars DataFrame"""
            return self._lazy.collect()
        
        @property
        def lazy(self):
            """Access underlying LazyFrame"""
            return self._lazy
    
    # Pattern 2: Context managers for efficient data flow
    from contextlib import contextmanager
    
    @contextmanager
    def polars_context(pandas_df):
        """Context manager for temporary Polars operations"""
        # Convert to Polars at entry
        pl_df = pl.from_pandas(pandas_df)
        try:
            yield pl_df
        finally:
            # Cleanup happens automatically
            pass
    
    # Usage example
    def use_polars_context():
        # Start with Pandas DataFrame
        pd_data = pd.read_csv("small_data.csv")  # Small data, pandas is fine
        
        # Use Polars for heavy operations
        with polars_context(pd_data) as pl_data:
            processed = (
                pl_data
                .lazy()
                .filter(pl.col("volume") > 1000000)
                .with_columns([
                    pl.col("price").pct_change().alias("returns"),
                    pl.col("volume").rank().alias("volume_rank")
                ])
                .collect()
            )
        
        # Convert back to Pandas for further analysis
        result_pd = processed.to_pandas()
        return result_pd
    
    # Pattern 3: Streaming integration with Pandas chunks
    def stream_to_pandas_analysis():
        """Stream large Polars data to Pandas for analysis in chunks"""
        
        # Process large dataset in Polars
        large_result = (
            pl.scan_parquet("huge_dataset/*.parquet")
            .filter(pl.col("date") >= "2020-01-01")
            .group_by(["symbol", "date"])
            .agg([
                pl.col("price").mean().alias("avg_price"),
                pl.col("volume").sum().alias("total_volume"),
                pl.col("returns").std().alias("volatility")
            ])
            .collect(streaming=True)
        )
        
        # Process results in Pandas chunks
        chunk_size = 100000
        results = []
        
        for i in range(0, len(large_result), chunk_size):
            chunk = large_result[i:i+chunk_size].to_pandas()
            
            # Pandas-specific analysis on chunk
            chunk_analysis = {
                'correlation': chunk[['avg_price', 'total_volume', 'volatility']].corr(),
                'summary_stats': chunk.describe(),
                'symbol_count': chunk['symbol'].nunique()
            }
            
            results.append(chunk_analysis)
        
        return results
    
    return HybridDataFrame, use_polars_context(), stream_to_pandas_analysis()
```

---

## NumPy Integration Patterns

### Efficient NumPy Array Access

```python
def numpy_integration_patterns():
    """Efficient integration patterns with NumPy for quantitative computations"""
    
    # Direct NumPy array access (zero-copy when possible)
    def polars_to_numpy_efficient():
        pl_df = pl.DataFrame({
            "returns": [0.01, -0.02, 0.005, 0.015, -0.01],
            "volume": [1000000, 2000000, 1500000, 1800000, 1200000]
        })
        
        # Get NumPy arrays directly (zero-copy for simple numeric types)
        returns_array = pl_df["returns"].to_numpy()
        volume_array = pl_df["volume"].to_numpy()
        
        # Use NumPy for computations
        correlation = np.corrcoef(returns_array, volume_array)[0, 1]
        
        # Complex NumPy operations
        rolling_corr = np.correlate(returns_array, volume_array, mode='full')
        
        return returns_array, volume_array, correlation, rolling_corr
    
    # Integration with scipy for statistical functions
    def scipy_integration():
        from scipy import stats
        
        # Generate data with Polars
        data = (
            pl.scan_parquet("returns.parquet")
            .filter(pl.col("symbol").is_in(["AAPL", "GOOGL", "MSFT"]))
            .pivot(index="date", columns="symbol", values="returns")
            .collect()
        )
        
        # Convert to NumPy for scipy operations
        returns_matrix = data.select(pl.exclude("date")).to_numpy()
        
        # Statistical tests
        normality_tests = {}
        for i, symbol in enumerate(["AAPL", "GOOGL", "MSFT"]):
            stat, p_value = stats.jarque_bera(returns_matrix[:, i])
            normality_tests[symbol] = {'statistic': stat, 'p_value': p_value}
        
        return data, returns_matrix, normality_tests
    
    # Custom NumPy functions with Polars
    def custom_numpy_functions():
        """Apply custom NumPy functions to Polars data"""
        
        def rolling_sharpe_numpy(returns: np.ndarray, window: int = 252, rf_rate: float = 0.02):
            """Custom rolling Sharpe ratio using NumPy"""
            sharpe_ratios = np.full(len(returns), np.nan)
            
            for i in range(window, len(returns)):
                window_returns = returns[i-window:i]
                excess_returns = window_returns - rf_rate/252
                sharpe_ratios[i] = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
            
            return sharpe_ratios
        
        # Apply custom function to Polars data
        pl_data = pl.read_parquet("stock_returns.parquet")
        
        # Use map_batches for applying NumPy functions
        result = (
            pl_data
            .sort(["symbol", "date"])
            .with_columns([
                pl.col("returns")
                .map_batches(lambda x: rolling_sharpe_numpy(x.to_numpy()))
                .over("symbol")
                .alias("rolling_sharpe")
            ])
        )
        
        return result
    
    return polars_to_numpy_efficient(), scipy_integration(), custom_numpy_functions()
```

### Scientific Computing Integration

```python
def quantlib_integration():
    """Integration patterns with QuantLib for advanced financial calculations"""
    
    try:
        import QuantLib as ql
        
        def bond_pricing_with_polars():
            """Use Polars for data prep, QuantLib for bond pricing"""
            
            # Prepare bond data with Polars
            bond_data = (
                pl.scan_csv("bond_universe.csv")
                .filter(pl.col("maturity") >= "2025-01-01")
                .filter(pl.col("rating").is_in(["AAA", "AA+", "AA", "AA-"]))
                .with_columns([
                    pl.col("maturity").str.strptime(pl.Date, format="%Y-%m-%d"),
                    pl.col("coupon_rate").cast(pl.Float64) / 100  # Convert to decimal
                ])
                .collect()
            )
            
            # Convert to Pandas for QuantLib integration
            bond_df = bond_data.to_pandas()
            
            # Price bonds using QuantLib
            prices = []
            for _, bond in bond_df.iterrows():
                # Set up QuantLib objects
                maturity_date = ql.Date(bond['maturity'].day, bond['maturity'].month, bond['maturity'].year)
                schedule = ql.MakeSchedule(ql.Date.todaysDate(), maturity_date, ql.Period(6, ql.Months))
                
                bond_instrument = ql.FixedRateBond(
                    2,  # Settlement days
                    100.0,  # Face value
                    schedule,
                    [bond['coupon_rate']],  # Coupon rates
                    ql.Actual360()  # Day count
                )
                
                # Price the bond
                yield_curve = ql.FlatForward(ql.Date.todaysDate(), 0.03, ql.Actual360())
                bond_engine = ql.DiscountingBondEngine(ql.YieldTermStructureHandle(yield_curve))
                bond_instrument.setPricingEngine(bond_engine)
                
                prices.append(bond_instrument.NPV())
            
            # Add prices back to Polars DataFrame
            bond_data_with_prices = bond_data.with_columns([
                pl.Series("theoretical_price", prices)
            ])
            
            return bond_data_with_prices
        
        return bond_pricing_with_polars()
        
    except ImportError:
        print("QuantLib not available - install with: pip install QuantLib-Python")
        return None

def sklearn_integration_patterns():
    """Integration patterns with scikit-learn for machine learning"""
    
    def feature_engineering_pipeline():
        """Complete feature engineering pipeline with Polars + sklearn"""
        
        # Feature engineering with Polars
        features = (
            pl.scan_parquet("market_data.parquet")
            .sort(["symbol", "date"])
            .with_columns([
                # Technical features
                pl.col("close").rolling_mean(window_size=20).over("symbol").alias("sma_20"),
                pl.col("close").rolling_std(window_size=20).over("symbol").alias("volatility"),
                pl.col("volume").rolling_mean(window_size=10).over("symbol").alias("avg_volume"),
                
                # Momentum features
                pl.col("close").pct_change(periods=5).over("symbol").alias("momentum_5d"),
                pl.col("close").pct_change(periods=21).over("symbol").alias("momentum_21d"),
                
                # Cross-sectional features
                pl.col("market_cap").rank().over("date").alias("size_rank"),
                pl.col("close").pct_change().rank().over("date").alias("return_rank")
            ])
            .drop_nulls()
            .collect()
        )
        
        # Convert to pandas for sklearn
        df_pd = features.to_pandas()
        
        # Prepare features and targets
        feature_cols = ['sma_20', 'volatility', 'avg_volume', 'momentum_5d', 'momentum_21d', 'size_rank', 'return_rank']
        target_col = 'forward_return'  # Assuming this exists
        
        # Create forward returns as target
        df_pd['forward_return'] = df_pd.groupby('symbol')['close'].pct_change().shift(-1)
        
        # Remove NaN values
        df_clean = df_pd.dropna()
        
        X = df_clean[feature_cols]
        y = df_clean[target_col]
        
        # Train model
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Feature importance (convert back to Polars for further processing)
        feature_importance = pl.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort('importance', descending=True)
        
        return {
            'model': model,
            'features': features,
            'test_mse': mse,
            'test_r2': r2,
            'feature_importance': feature_importance
        }
    
    return feature_engineering_pipeline()
```

---

## Integration Best Practices

### Workflow Optimization

```python
def integration_best_practices():
    """Best practices for Polars-Pandas integration"""
    
    # Workflow decision matrix
    workflow_guide = {
        "Data Loading (>100MB)": "Start with Polars pl.scan_*",
        "Initial Cleaning": "Use Polars for speed",
        "Complex Aggregations": "Use Polars lazy evaluation", 
        "Statistical Analysis": "Convert to Pandas",
        "Machine Learning": "Use Pandas + sklearn",
        "Visualization": "Convert to Pandas for plotting",
        "Final Storage": "Use Polars for efficient writing"
    }
    
    # Memory management guidelines
    memory_guidelines = {
        "Small Results (<10MB)": "Safe to convert to Pandas",
        "Medium Results (10-100MB)": "Consider chunked conversion",
        "Large Results (>100MB)": "Keep in Polars, convert subsets",
        "Visualization Data": "Always safe to convert (aggregated)",
        "Model Features": "Convert features, keep raw data in Polars"
    }
    
    return workflow_guide, memory_guidelines

def recommended_stack():
    """Recommended technology stack for quantitative research"""
    
    stack_components = {
        "Data Loading": "Polars scan_* functions",
        "Heavy Processing": "Polars lazy operations", 
        "Feature Engineering": "Polars expressions",
        "Statistical Analysis": "Pandas + scipy/statsmodels",
        "Machine Learning": "Pandas + scikit-learn",
        "Visualization": "Matplotlib/Plotly with Pandas data",
        "Storage": "Parquet format via Polars",
        "Memory Management": "Polars streaming + chunked Pandas conversion"
    }
    
    return stack_components
```

### Common Integration Patterns Summary

```python
# Pattern 1: ETL with Polars, Analysis with Pandas
def etl_analysis_workflow():
    # Heavy ETL with Polars
    processed = (
        pl.scan_parquet("large_data.parquet")
        .filter(pl.col("date") >= "2020-01-01")
        .group_by("symbol")
        .agg([pl.col("returns").mean(), pl.col("returns").std()])
        .collect()
    )
    
    # Analysis with Pandas
    analysis_df = processed.to_pandas()
    correlation_matrix = analysis_df.corr()
    return correlation_matrix

# Pattern 2: Feature Engineering with Polars, ML with Pandas
def feature_ml_workflow():
    # Feature engineering with Polars
    features = (
        pl.scan_parquet("market_data.parquet")
        .with_columns([
            pl.col("price").rolling_mean(window_size=20).alias("sma_20"),
            pl.col("returns").rolling_std(window_size=252).alias("volatility")
        ])
        .collect()
    )
    
    # ML with Pandas + sklearn
    ml_data = features.to_pandas()
    # ... sklearn operations
    return ml_data

# Pattern 3: Polars for Processing, Pandas for Visualization
def processing_viz_workflow():
    # Processing with Polars
    summary = (
        pl.scan_parquet("trades.parquet")
        .group_by(["date", "symbol"])
        .agg([pl.col("volume").sum(), pl.col("price").mean()])
        .collect()
    )
    
    # Visualization with Pandas
    viz_data = summary.to_pandas()
    # ... matplotlib/seaborn plotting
    return viz_data
```

---

*Next: [Quantitative Research Specific Content](polars_quantitative_research.md) - Learn specialized patterns for quantitative finance applications.*