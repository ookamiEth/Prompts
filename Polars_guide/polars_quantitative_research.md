# Polars for Quantitative Research

Advanced quantitative research techniques using Polars for time series analysis, factor modeling, event studies, and risk management in financial markets.

## Table of Contents

1. [Time Series Analysis & Technical Indicators](#time-series-analysis--technical-indicators)
2. [Cross-Sectional Analysis & Factor Models](#cross-sectional-analysis--factor-models)
3. [Event Studies & Corporate Actions](#event-studies--corporate-actions)
4. [Risk Modeling & Portfolio Analytics](#risk-modeling--portfolio-analytics)
5. [Complete Factor Research Pipeline](#complete-factor-research-pipeline)
6. [Backtesting Framework](#backtesting-framework)
7. [Performance Attribution](#performance-attribution)
8. [Navigation](#navigation)

---

## Time Series Analysis & Technical Indicators

### High-Performance Time Series Operations

Polars excels at time series operations with its vectorized operations and window functions:

```python
import polars as pl
import numpy as np

def calculate_returns_efficiently(df: pl.LazyFrame) -> pl.LazyFrame:
    """Calculate various types of returns with optimal performance"""
    
    return (
        df
        .sort(["symbol", "date"])  # Essential for time series operations
        .with_columns([
            # Simple returns
            pl.col("close").pct_change().over("symbol").alias("simple_returns"),
            
            # Log returns (more stable for long periods)
            (pl.col("close").log() - pl.col("close").log().shift(1)).over("symbol").alias("log_returns"),
            
            # Multi-period returns
            pl.col("close").pct_change(periods=5).over("symbol").alias("returns_5d"),
            pl.col("close").pct_change(periods=21).over("symbol").alias("returns_21d"),
            
            # Cumulative returns
            (pl.col("simple_returns") + 1).log().cumsum().over("symbol").alias("cumulative_log_returns"),
            
            # Forward returns for prediction tasks
            pl.col("close").pct_change().shift(-1).over("symbol").alias("forward_1d_return"),
            pl.col("close").pct_change(periods=5).shift(-5).over("symbol").alias("forward_5d_return")
        ])
    )

def rolling_metrics_optimized(df: pl.LazyFrame, windows: list = [20, 60, 252]) -> pl.LazyFrame:
    """Calculate rolling metrics with optimal window operations"""
    
    rolling_exprs = []
    
    for window in windows:
        # Volatility (annualized)
        rolling_exprs.extend([
            pl.col("simple_returns").rolling_std(window_size=window).over("symbol")
            .mul(np.sqrt(252)).alias(f"vol_{window}d"),
            
            # Rolling mean (momentum)
            pl.col("simple_returns").rolling_mean(window_size=window).over("symbol")
            .mul(252).alias(f"momentum_{window}d"),
            
            # Rolling Sharpe (assuming 2% risk-free rate)
            (pl.col("simple_returns").rolling_mean(window_size=window) * 252 - 0.02)
            .truediv(pl.col("simple_returns").rolling_std(window_size=window) * np.sqrt(252))
            .over("symbol").alias(f"sharpe_{window}d"),
            
            # Rolling maximum and minimum
            pl.col("close").rolling_max(window_size=window).over("symbol").alias(f"max_price_{window}d"),
            pl.col("close").rolling_min(window_size=window).over("symbol").alias(f"min_price_{window}d"),
            
            # Price position within range
            ((pl.col("close") - pl.col("close").rolling_min(window_size=window)) /
             (pl.col("close").rolling_max(window_size=window) - pl.col("close").rolling_min(window_size=window)))
            .over("symbol").alias(f"price_position_{window}d")
        ])
    
    return df.with_columns(rolling_exprs)
```

### Technical Indicators Implementation

```python
def technical_indicators(df: pl.LazyFrame) -> pl.LazyFrame:
    """Common technical indicators implemented efficiently"""
    
    return (
        df
        .sort(["symbol", "date"])
        .with_columns([
            # Moving averages
            pl.col("close").rolling_mean(window_size=20).over("symbol").alias("sma_20"),
            pl.col("close").rolling_mean(window_size=50).over("symbol").alias("sma_50"),
            pl.col("close").rolling_mean(window_size=200).over("symbol").alias("sma_200"),
            
            # Exponential moving average (approximation)
            pl.col("close").ewm_mean(alpha=2/21, adjust=False).over("symbol").alias("ema_20"),
            
            # Bollinger Bands
            pl.col("close").rolling_mean(window_size=20).over("symbol").alias("bb_middle"),
        ])
        .with_columns([
            # Bollinger Bands continued
            (pl.col("bb_middle") + 2 * pl.col("close").rolling_std(window_size=20).over("symbol")).alias("bb_upper"),
            (pl.col("bb_middle") - 2 * pl.col("close").rolling_std(window_size=20).over("symbol")).alias("bb_lower"),
            
            # RSI (Relative Strength Index) - simplified
            pl.col("simple_returns").map_batches(lambda x: calculate_rsi(x.to_numpy())).over("symbol").alias("rsi_14"),
            
            # MACD components
            (pl.col("ema_20") - pl.col("close").ewm_mean(alpha=2/13, adjust=False).over("symbol")).alias("macd_line"),
            
            # Volume indicators
            pl.col("volume").rolling_mean(window_size=20).over("symbol").alias("volume_sma_20"),
            (pl.col("volume") / pl.col("volume").rolling_mean(window_size=20).over("symbol")).alias("volume_ratio")
        ])
        .with_columns([
            # Bollinger Band position
            ((pl.col("close") - pl.col("bb_lower")) / (pl.col("bb_upper") - pl.col("bb_lower"))).alias("bb_position"),
            
            # MACD signal line
            pl.col("macd_line").ewm_mean(alpha=2/10, adjust=False).over("symbol").alias("macd_signal")
        ])
    )

def calculate_rsi(returns: np.ndarray, window: int = 14) -> np.ndarray:
    """Calculate RSI using NumPy for performance"""
    rsi = np.full_like(returns, np.nan, dtype=float)
    
    if len(returns) < window + 1:
        return rsi
    
    gains = np.where(returns > 0, returns, 0)
    losses = np.where(returns < 0, -returns, 0)
    
    # Initial average
    avg_gain = np.mean(gains[:window])
    avg_loss = np.mean(losses[:window])
    
    for i in range(window, len(returns)):
        avg_gain = (avg_gain * (window - 1) + gains[i]) / window
        avg_loss = (avg_loss * (window - 1) + losses[i]) / window
        
        if avg_loss == 0:
            rsi[i] = 100
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100 - (100 / (1 + rs))
    
    return rsi
```

### Time Series Resampling & Aggregation

```python
def time_series_resampling():
    """Efficient time series resampling for different frequencies"""
    
    def resample_to_monthly(df: pl.LazyFrame) -> pl.LazyFrame:
        """Resample daily data to monthly"""
        
        return (
            df
            .with_columns([
                pl.col("date").dt.year().alias("year"),
                pl.col("date").dt.month().alias("month")
            ])
            .group_by(["symbol", "year", "month"])
            .agg([
                # Price aggregations
                pl.col("close").last().alias("close"),  # Last price of month
                pl.col("close").first().alias("open"),  # First price of month
                pl.col("high").max().alias("high"),     # Highest price
                pl.col("low").min().alias("low"),       # Lowest price
                
                # Volume aggregations
                pl.col("volume").sum().alias("total_volume"),
                pl.col("volume").mean().alias("avg_daily_volume"),
                
                # Return aggregations
                pl.col("simple_returns").sum().alias("monthly_return"),  # Approximate compound return
                ((pl.col("simple_returns") + 1).prod() - 1).alias("monthly_compound_return"),
                pl.col("simple_returns").std().alias("daily_vol_in_month"),
                
                # Trading day count
                pl.col("date").count().alias("trading_days"),
                
                # Date aggregations
                pl.col("date").min().alias("month_start"),
                pl.col("date").max().alias("month_end")
            ])
            .with_columns([
                # Monthly volatility (annualized)
                (pl.col("daily_vol_in_month") * np.sqrt(252)).alias("annualized_vol"),
                
                # Monthly date
                pl.date(year=pl.col("year"), month=pl.col("month"), day=1).alias("month_date")
            ])
        )
    
    def resample_intraday_to_daily(df: pl.LazyFrame) -> pl.LazyFrame:
        """Aggregate intraday data to daily OHLCV"""
        
        return (
            df
            .with_columns([
                pl.col("timestamp").dt.date().alias("date")
            ])
            .group_by(["symbol", "date"])
            .agg([
                # OHLC
                pl.col("price").first().alias("open"),
                pl.col("price").max().alias("high"), 
                pl.col("price").min().alias("low"),
                pl.col("price").last().alias("close"),
                
                # Volume and trades
                pl.col("size").sum().alias("volume"),
                pl.col("size").count().alias("trade_count"),
                
                # Weighted average price
                (pl.col("price") * pl.col("size")).sum().truediv(pl.col("size").sum()).alias("vwap"),
                
                # Price statistics
                pl.col("price").std().alias("intraday_vol"),
                pl.col("price").quantile(0.5).alias("median_price"),
                
                # Time statistics
                pl.col("timestamp").first().alias("first_trade_time"),
                pl.col("timestamp").last().alias("last_trade_time")
            ])
            .with_columns([
                # Intraday return
                (pl.col("close") / pl.col("open") - 1).alias("intraday_return"),
                
                # Trading hours span
                (pl.col("last_trade_time") - pl.col("first_trade_time")).dt.total_seconds().truediv(3600).alias("trading_hours")
            ])
        )
    
    return resample_to_monthly, resample_intraday_to_daily
```

---

## Cross-Sectional Analysis & Factor Models

### Universe Rankings and Factor Scoring

```python
def cross_sectional_analysis():
    """Cross-sectional ranking and scoring for factor models"""
    
    def universe_rankings(df: pl.LazyFrame) -> pl.LazyFrame:
        """Calculate cross-sectional rankings and scores"""
        
        return (
            df
            .with_columns([
                # Market cap ranking (descending - largest gets rank 1)
                pl.col("market_cap").rank(method="ordinal", descending=True).over("date").alias("mcap_rank"),
                
                # Value factor rankings
                pl.col("book_to_market").rank(method="ordinal", descending=True).over("date").alias("value_rank"),
                pl.col("pe_ratio").rank(method="ordinal").over("date").alias("pe_rank"),  # Low PE is better
                
                # Momentum rankings
                pl.col("returns_21d").rank(method="ordinal", descending=True).over("date").alias("momentum_rank"),
                pl.col("returns_252d").rank(method="ordinal", descending=True).over("date").alias("long_momentum_rank"),
                
                # Quality rankings
                pl.col("roe").rank(method="ordinal", descending=True).over("date").alias("quality_rank"),
                pl.col("debt_to_equity").rank(method="ordinal").over("date").alias("leverage_rank"),  # Low leverage is better
                
                # Volatility ranking (low vol is better for risk-adjusted returns)
                pl.col("vol_252d").rank(method="ordinal").over("date").alias("vol_rank")
            ])
            .with_columns([
                # Percentile scores (0-1)
                (pl.col("mcap_rank") / pl.col("mcap_rank").max().over("date")).alias("mcap_percentile"),
                (pl.col("value_rank") / pl.col("value_rank").max().over("date")).alias("value_percentile"),
                (pl.col("momentum_rank") / pl.col("momentum_rank").max().over("date")).alias("momentum_percentile"),
                (pl.col("quality_rank") / pl.col("quality_rank").max().over("date")).alias("quality_percentile"),
                
                # Z-scores for factor exposure
                ((pl.col("book_to_market") - pl.col("book_to_market").mean().over("date")) /
                 pl.col("book_to_market").std().over("date")).alias("value_zscore"),
                
                ((pl.col("returns_21d") - pl.col("returns_21d").mean().over("date")) /
                 pl.col("returns_21d").std().over("date")).alias("momentum_zscore")
            ])
        )
    
    def factor_portfolio_construction(df: pl.LazyFrame) -> pl.LazyFrame:
        """Construct factor-based portfolios"""
        
        return (
            df
            .with_columns([
                # Quintile assignments
                pl.col("value_percentile").qcut(5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"]).alias("value_quintile"),
                pl.col("momentum_percentile").qcut(5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"]).alias("momentum_quintile"),
                pl.col("quality_percentile").qcut(5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"]).alias("quality_quintile"),
                
                # Combined factor scores
                (0.3 * pl.col("value_zscore") + 0.3 * pl.col("momentum_zscore") + 0.4 * pl.col("quality_percentile")).alias("composite_score"),
                
                # Market cap constraints (only include liquid stocks)
                (pl.col("mcap_rank") <= 1000).alias("in_universe")  # Top 1000 by market cap
            ])
            .filter(pl.col("in_universe"))
            .with_columns([
                # Portfolio weights (can be equal-weighted or score-weighted)
                (1.0 / pl.col("symbol").count().over(["date", "value_quintile"])).alias("equal_weight"),
                
                # Score-based weights
                (pl.col("composite_score") / pl.col("composite_score").sum().over("date")).alias("score_weight"),
                
                # Cap-weighted within portfolios
                (pl.col("market_cap") / pl.col("market_cap").sum().over(["date", "value_quintile"])).alias("cap_weight")
            ])
        )
    
    return universe_rankings, factor_portfolio_construction
```

### Multi-Factor Model Implementation

```python
def multi_factor_model():
    """Implementation of Fama-French and custom factor models"""
    
    def calculate_fama_french_factors(df: pl.LazyFrame) -> pl.LazyFrame:
        """Calculate Fama-French HML, SMB, and momentum factors"""
        
        return (
            df
            .with_columns([
                # Size classification (SMB - Small Minus Big)
                pl.when(pl.col("market_cap") <= pl.col("market_cap").quantile(0.5).over("date"))
                .then(pl.lit("Small"))
                .otherwise(pl.lit("Big"))
                .alias("size_group"),
                
                # Value classification (HML - High Minus Low)
                pl.when(pl.col("book_to_market") <= pl.col("book_to_market").quantile(0.3).over("date"))
                .then(pl.lit("Low"))
                .when(pl.col("book_to_market") >= pl.col("book_to_market").quantile(0.7).over("date"))
                .then(pl.lit("High"))
                .otherwise(pl.lit("Medium"))
                .alias("value_group"),
                
                # Momentum classification
                pl.when(pl.col("returns_252d") <= pl.col("returns_252d").quantile(0.3).over("date"))
                .then(pl.lit("Low"))
                .when(pl.col("returns_252d") >= pl.col("returns_252d").quantile(0.7).over("date"))
                .then(pl.lit("High"))
                .otherwise(pl.lit("Medium"))
                .alias("momentum_group")
            ])
            .with_columns([
                # Create portfolio identifiers
                (pl.col("size_group") + "_" + pl.col("value_group")).alias("size_value_portfolio"),
                (pl.col("size_group") + "_" + pl.col("momentum_group")).alias("size_momentum_portfolio")
            ])
        )
    
    def calculate_portfolio_returns(factor_portfolios: pl.LazyFrame) -> pl.LazyFrame:
        """Calculate returns for each factor portfolio"""
        
        return (
            factor_portfolios
            .group_by(["date", "size_value_portfolio"])
            .agg([
                pl.col("simple_returns").mean().alias("portfolio_return"),
                pl.col("market_cap").sum().alias("total_market_cap"),
                pl.col("symbol").count().alias("stock_count")
            ])
            .filter(pl.col("stock_count") >= 5)  # Minimum stocks per portfolio
            .pivot(values="portfolio_return", index="date", columns="size_value_portfolio")
            .with_columns([
                # SMB factor (Small minus Big)
                ((pl.col("Small_High") + pl.col("Small_Medium") + pl.col("Small_Low")) / 3 -
                 (pl.col("Big_High") + pl.col("Big_Medium") + pl.col("Big_Low")) / 3).alias("SMB"),
                
                # HML factor (High minus Low)
                ((pl.col("Small_High") + pl.col("Big_High")) / 2 -
                 (pl.col("Small_Low") + pl.col("Big_Low")) / 2).alias("HML"),
                
                # Market factor (all portfolios average)
                ((pl.col("Small_High") + pl.col("Small_Medium") + pl.col("Small_Low") +
                  pl.col("Big_High") + pl.col("Big_Medium") + pl.col("Big_Low")) / 6).alias("Market")
            ])
        )
    
    return calculate_fama_french_factors, calculate_portfolio_returns
```

---

## Event Studies & Corporate Actions

### Event Study Methodology

```python
def event_study_analysis():
    """Event study methodology for corporate actions and announcements"""
    
    def calculate_abnormal_returns(df: pl.LazyFrame, event_dates: pl.DataFrame) -> pl.LazyFrame:
        """Calculate abnormal returns around event dates"""
        
        # First calculate market returns (equal-weighted)
        market_returns = (
            df
            .group_by("date")
            .agg([
                pl.col("simple_returns").mean().alias("market_return"),
                pl.col("simple_returns").count().alias("stock_count")
            ])
            .filter(pl.col("stock_count") > 100)  # Minimum stocks for reliable market return
        )
        
        # Join with event dates to create event windows
        return (
            df
            .join(event_dates.lazy(), on="symbol", how="inner")
            .join(market_returns.lazy(), on="date", how="left")
            .with_columns([
                # Days relative to event
                (pl.col("date") - pl.col("event_date")).dt.total_days().alias("days_from_event")
            ])
            .filter(pl.col("days_from_event").is_between(-30, 30))  # Event window: -30 to +30 days
            .with_columns([
                # Calculate abnormal returns
                (pl.col("simple_returns") - pl.col("market_return")).alias("abnormal_return"),
                
                # Event window indicators
                pl.col("days_from_event").is_between(-10, -2).alias("pre_event"),
                pl.col("days_from_event").is_between(-1, 1).alias("event_window"),
                pl.col("days_from_event").is_between(2, 10).alias("post_event")
            ])
            .with_columns([
                # Cumulative abnormal returns
                pl.col("abnormal_return").cumsum().over(["symbol", "event_date"]).alias("cumulative_abnormal_return")
            ])
        )
    
    def handle_stock_splits(df: pl.LazyFrame, splits_df: pl.DataFrame) -> pl.LazyFrame:
        """Adjust prices for stock splits"""
        
        return (
            df
            .join(splits_df.lazy(), on=["symbol", "date"], how="left")
            .sort(["symbol", "date"])
            .with_columns([
                # Fill split ratio (1.0 for no split)
                pl.col("split_ratio").fill_null(1.0),
                
                # Calculate cumulative adjustment factor
                pl.col("split_ratio").reverse().cumprod().reverse().over("symbol").alias("adj_factor")
            ])
            .with_columns([
                # Adjust prices
                (pl.col("close") / pl.col("adj_factor")).alias("adj_close"),
                (pl.col("open") / pl.col("adj_factor")).alias("adj_open"),
                (pl.col("high") / pl.col("adj_factor")).alias("adj_high"),
                (pl.col("low") / pl.col("adj_factor")).alias("adj_low"),
                
                # Adjust volume (inverse adjustment)
                (pl.col("volume") * pl.col("adj_factor")).alias("adj_volume")
            ])
        )
    
    def dividend_adjustment(df: pl.LazyFrame, dividends_df: pl.DataFrame) -> pl.LazyFrame:
        """Calculate total returns including dividends"""
        
        return (
            df
            .join(dividends_df.lazy(), on=["symbol", "date"], how="left")
            .sort(["symbol", "date"])
            .with_columns([
                # Fill dividend amount (0 for no dividend)
                pl.col("dividend_amount").fill_null(0.0),
                
                # Calculate dividend yield
                (pl.col("dividend_amount") / pl.col("close").shift(1)).over("symbol").alias("dividend_yield"),
                
                # Total return including dividends
                (pl.col("simple_returns") + pl.col("dividend_yield")).alias("total_return")
            ])
            .with_columns([
                # Cumulative total return
                ((pl.col("total_return") + 1).log().cumsum().exp()).over("symbol").alias("cumulative_total_return")
            ])
        )
    
    return calculate_abnormal_returns, handle_stock_splits, dividend_adjustment
```

---

## Risk Modeling & Portfolio Analytics

### Value at Risk (VaR) and Risk Metrics

```python
def risk_modeling_patterns():
    """Comprehensive risk modeling using Polars"""
    
    def calculate_var_metrics(returns_df: pl.LazyFrame, confidence_levels=[0.01, 0.05, 0.1]) -> pl.LazyFrame:
        """Calculate Value at Risk and Expected Shortfall"""
        
        var_exprs = []
        es_exprs = []
        
        for conf_level in confidence_levels:
            conf_pct = int(conf_level * 100)
            # VaR calculation
            var_exprs.append(
                pl.col("returns").quantile(conf_level).alias(f"var_{conf_pct}pct")
            )
            # Expected Shortfall (Conditional VaR)
            es_exprs.append(
                pl.col("returns").filter(pl.col("returns") <= pl.col("returns").quantile(conf_level))
                .mean().alias(f"es_{conf_pct}pct")
            )
        
        return (
            returns_df
            .group_by("symbol")
            .agg([
                # Basic statistics
                pl.col("returns").mean().alias("mean_return"),
                pl.col("returns").std().alias("volatility"),
                pl.col("returns").min().alias("worst_return"),
                pl.col("returns").max().alias("best_return"),
                
                # VaR metrics
                *var_exprs,
                *es_exprs,
                
                # Higher moments
                pl.col("returns").skewness().alias("skewness"),
                pl.col("returns").kurtosis().alias("kurtosis"),
                
                # Count and dates
                pl.col("returns").count().alias("observation_count"),
                pl.col("date").min().alias("start_date"),
                pl.col("date").max().alias("end_date")
            ])
            .with_columns([
                # Annualized metrics
                (pl.col("mean_return") * 252).alias("annualized_return"),
                (pl.col("volatility") * np.sqrt(252)).alias("annualized_volatility"),
                
                # Risk-adjusted returns
                (pl.col("mean_return") / pl.col("volatility") * np.sqrt(252)).alias("sharpe_ratio"),
                
                # Downside deviation and Sortino ratio
                pl.col("returns").filter(pl.col("returns") < 0).std().alias("downside_deviation")
            ])
            .with_columns([
                (pl.col("annualized_return") / (pl.col("downside_deviation") * np.sqrt(252))).alias("sortino_ratio")
            ])
        )
    
    def calculate_drawdown_metrics(df: pl.LazyFrame) -> pl.LazyFrame:
        """Calculate comprehensive drawdown analysis"""
        
        return (
            df
            .sort(["symbol", "date"])
            .with_columns([
                # Cumulative returns
                ((pl.col("returns") + 1).log().cumsum().exp()).over("symbol").alias("cumulative_return"),
            ])
            .with_columns([
                # Running maximum (peak)
                pl.col("cumulative_return").cummax().over("symbol").alias("peak"),
            ])
            .with_columns([
                # Drawdown calculation
                (pl.col("cumulative_return") / pl.col("peak") - 1).alias("drawdown"),
                
                # Is at new peak?
                (pl.col("cumulative_return") == pl.col("peak")).alias("at_peak")
            ])
            .with_columns([
                # Days since peak
                pl.col("at_peak").cast(pl.Int32).cumsum().over("symbol").alias("recovery_period_id")
            ])
            .group_by(["symbol", "recovery_period_id"])
            .agg([
                pl.col("drawdown").min().alias("max_drawdown_in_period"),
                pl.col("drawdown").count().alias("drawdown_days"),
                pl.col("date").first().alias("peak_date"),
                pl.col("date").last().alias("recovery_date")
            ])
            .group_by("symbol")
            .agg([
                pl.col("max_drawdown_in_period").min().alias("max_drawdown"),
                pl.col("drawdown_days").max().alias("max_drawdown_duration"),
                pl.col("drawdown_days").mean().alias("avg_drawdown_duration"),
                pl.col("max_drawdown_in_period").count().alias("drawdown_periods")
            ])
        )
    
    def calculate_beta_metrics(stock_returns: pl.LazyFrame, market_returns: pl.LazyFrame) -> pl.LazyFrame:
        """Calculate beta and other market-relative metrics"""
        
        # Join stock and market returns
        combined = (
            stock_returns
            .join(market_returns, on="date", how="inner", suffix="_market")
            .filter(pl.col("returns").is_not_null() & pl.col("returns_market").is_not_null())
        )
        
        return (
            combined
            .group_by("symbol")
            .agg([
                # Beta calculation: Cov(stock, market) / Var(market)
                pl.corr("returns", "returns_market").alias("correlation"),
                (pl.cov("returns", "returns_market") / pl.col("returns_market").var()).alias("beta"),
                
                # Alpha calculation
                (pl.col("returns").mean() - pl.col("beta") * pl.col("returns_market").mean()).alias("alpha"),
                
                # Tracking error
                (pl.col("returns") - pl.col("returns_market")).std().alias("tracking_error"),
                
                # Information ratio
                ((pl.col("returns") - pl.col("returns_market")).mean() / 
                 (pl.col("returns") - pl.col("returns_market")).std()).alias("information_ratio"),
                
                # R-squared
                (pl.corr("returns", "returns_market") ** 2).alias("r_squared"),
                
                # Observation count
                pl.col("returns").count().alias("observations")
            ])
            .with_columns([
                # Annualize metrics
                (pl.col("alpha") * 252).alias("annualized_alpha"),
                (pl.col("tracking_error") * np.sqrt(252)).alias("annualized_tracking_error"),
                (pl.col("information_ratio") * np.sqrt(252)).alias("annualized_information_ratio")
            ])
        )
    
    return calculate_var_metrics, calculate_drawdown_metrics, calculate_beta_metrics
```

---

## Complete Factor Research Pipeline

### End-to-End Factor Research Implementation

```python
def complete_factor_research_pipeline():
    """Complete factor research pipeline from data to results"""
    
    # Step 1: Data preparation
    def prepare_research_universe():
        return (
            pl.scan_parquet("universe/*.parquet")
            .filter(pl.col("date") >= "2010-01-01")
            .filter(pl.col("market_cap") > 1e8)  # $100M minimum market cap
            .sort(["symbol", "date"])
            .with_columns([
                # Returns calculation
                pl.col("adj_close").pct_change().over("symbol").alias("returns"),
                
                # Market cap ranking for universe definition
                pl.col("market_cap").rank(descending=True).over("date").alias("mcap_rank")
            ])
            .filter(pl.col("mcap_rank") <= 1500)  # Top 1500 stocks by market cap
        )
    
    # Step 2: Factor calculation
    def calculate_factors(universe: pl.LazyFrame):
        return (
            universe
            .with_columns([
                # Value factors
                pl.col("book_value").truediv(pl.col("market_cap")).alias("book_to_market"),
                pl.col("earnings").truediv(pl.col("market_cap")).alias("earnings_yield"),
                
                # Momentum factors
                pl.col("adj_close").pct_change(periods=21).over("symbol").alias("momentum_1m"),
                pl.col("adj_close").pct_change(periods=252).over("symbol").alias("momentum_12m"),
                
                # Quality factors
                pl.col("roe").alias("quality_roe"),
                pl.col("roa").alias("quality_roa"), 
                pl.col("debt_to_equity").alias("leverage"),
                
                # Size factor
                pl.col("market_cap").log().alias("log_market_cap"),
                
                # Volatility factor
                pl.col("returns").rolling_std(window_size=252).over("symbol").alias("volatility_1y")
            ])
            # Cross-sectional standardization
            .with_columns([
                # Standardize factors within each date
                ((pl.col("book_to_market") - pl.col("book_to_market").mean().over("date")) /
                 pl.col("book_to_market").std().over("date")).alias("value_zscore"),
                
                ((pl.col("momentum_12m") - pl.col("momentum_12m").mean().over("date")) /
                 pl.col("momentum_12m").std().over("date")).alias("momentum_zscore"),
                
                ((pl.col("quality_roe") - pl.col("quality_roe").mean().over("date")) /
                 pl.col("quality_roe").std().over("date")).alias("quality_zscore"),
                
                ((pl.col("log_market_cap") - pl.col("log_market_cap").mean().over("date")) /
                 pl.col("log_market_cap").std().over("date")).alias("size_zscore")
            ])
        )
    
    # Step 3: Portfolio construction
    def construct_portfolios(factor_data: pl.LazyFrame):
        return (
            factor_data
            .with_columns([
                # Factor quintiles
                pl.col("value_zscore").qcut(5, labels=["V1", "V2", "V3", "V4", "V5"]).alias("value_quintile"),
                pl.col("momentum_zscore").qcut(5, labels=["M1", "M2", "M3", "M4", "M5"]).alias("momentum_quintile"),
                pl.col("quality_zscore").qcut(5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"]).alias("quality_quintile"),
                
                # Combined factor score
                (0.3 * pl.col("value_zscore") + 
                 0.3 * pl.col("momentum_zscore") + 
                 0.4 * pl.col("quality_zscore")).alias("composite_score")
            ])
            .with_columns([
                pl.col("composite_score").qcut(10, labels=[f"D{i}" for i in range(1, 11)]).alias("decile")
            ])
            # Portfolio weights (equal weight within each portfolio)
            .with_columns([
                (1.0 / pl.col("symbol").count().over(["date", "decile"])).alias("portfolio_weight")
            ])
        )
    
    # Step 4: Performance calculation
    def calculate_portfolio_performance(portfolio_data: pl.LazyFrame):
        return (
            portfolio_data
            .group_by(["date", "decile"])
            .agg([
                # Portfolio return (weighted average)
                (pl.col("returns") * pl.col("portfolio_weight")).sum().alias("portfolio_return"),
                pl.col("symbol").count().alias("stock_count"),
                pl.col("market_cap").sum().alias("portfolio_mcap")
            ])
            .filter(pl.col("stock_count") >= 10)  # Minimum diversification
            .sort(["decile", "date"])
            .with_columns([
                # Cumulative returns
                ((pl.col("portfolio_return") + 1).log().cumsum().exp()).over("decile").alias("cumulative_return")
            ])
        )
    
    # Step 5: Risk analysis
    def calculate_risk_metrics(performance_data: pl.LazyFrame):
        return (
            performance_data
            .group_by("decile")
            .agg([
                # Return metrics
                pl.col("portfolio_return").mean().mul(252).alias("annualized_return"),
                pl.col("portfolio_return").std().mul(np.sqrt(252)).alias("annualized_vol"),
                
                # Risk metrics
                pl.col("portfolio_return").quantile(0.05).alias("var_5pct"),
                pl.col("portfolio_return").min().alias("worst_return"),
                
                # Performance metrics
                pl.col("cumulative_return").last().alias("total_return"),
                
                # Count
                pl.col("date").count().alias("observation_count")
            ])
            .with_columns([
                # Sharpe ratio (assuming 2% risk-free rate)
                (pl.col("annualized_return") - 0.02).truediv(pl.col("annualized_vol")).alias("sharpe_ratio"),
                
                # Sortino ratio (downside deviation)
                (pl.col("annualized_return") - 0.02).truediv(
                    pl.col("annualized_vol") * 0.8  # Approximation for downside vol
                ).alias("sortino_ratio")
            ])
            .sort("annualized_return", descending=True)
        )
    
    # Execute complete pipeline
    universe = prepare_research_universe()
    factors = calculate_factors(universe)
    portfolios = construct_portfolios(factors)
    performance = calculate_portfolio_performance(portfolios)
    risk_metrics = calculate_risk_metrics(performance)
    
    return {
        'universe': universe,
        'factors': factors, 
        'portfolios': portfolios,
        'performance': performance,
        'risk_metrics': risk_metrics.collect()
    }
```

---

## Backtesting Framework

### Comprehensive Strategy Backtesting

```python
def strategy_backtesting_framework():
    """Generic strategy backtesting framework using Polars"""
    
    class PolarsBacktester:
        def __init__(self, universe_data: pl.LazyFrame, initial_capital: float = 1000000):
            self.universe = universe_data
            self.initial_capital = initial_capital
            self.results = {}
        
        def define_signals(self, signal_logic: callable) -> pl.LazyFrame:
            """Apply signal generation logic to universe"""
            return signal_logic(self.universe)
        
        def calculate_positions(self, signals: pl.LazyFrame, position_logic: callable) -> pl.LazyFrame:
            """Convert signals to position sizes"""
            return position_logic(signals)
        
        def simulate_trading(self, positions: pl.LazyFrame) -> pl.LazyFrame:
            """Simulate trading with transaction costs"""
            
            return (
                positions
                .sort(["symbol", "date"])
                .with_columns([
                    # Position changes
                    (pl.col("position") - pl.col("position").shift(1).fill_null(0)).over("symbol").alias("position_change"),
                    
                    # Trading costs (0.1% per trade)
                    (pl.col("position_change").abs() * pl.col("price") * 0.001).alias("transaction_cost")
                ])
                .with_columns([
                    # PnL calculation
                    (pl.col("position").shift(1) * pl.col("returns") * pl.col("price").shift(1)).over("symbol").alias("pnl"),
                    
                    # Net PnL after costs
                    (pl.col("pnl") - pl.col("transaction_cost")).alias("net_pnl")
                ])
            )
        
        def calculate_performance(self, trading_results: pl.LazyFrame) -> dict:
            """Calculate comprehensive performance metrics"""
            
            # Daily portfolio performance
            daily_performance = (
                trading_results
                .group_by("date")
                .agg([
                    pl.col("net_pnl").sum().alias("daily_pnl"),
                    pl.col("transaction_cost").sum().alias("daily_costs"),
                    pl.col("position").count().alias("position_count")
                ])
                .sort("date")
                .with_columns([
                    # Portfolio value
                    (self.initial_capital + pl.col("daily_pnl").cumsum()).alias("portfolio_value"),
                    
                    # Daily returns
                    pl.col("daily_pnl").truediv(
                        self.initial_capital + pl.col("daily_pnl").shift(1).cumsum().fill_null(0)
                    ).alias("daily_return")
                ])
                .collect()
            )
            
            # Performance statistics
            returns = daily_performance["daily_return"].to_numpy()
            returns_clean = returns[~np.isnan(returns)]
            
            if len(returns_clean) == 0:
                return {"error": "No valid returns data"}
            
            performance_stats = {
                'total_return': float(daily_performance["portfolio_value"][-1] / self.initial_capital - 1),
                'annualized_return': float(np.mean(returns_clean) * 252),
                'volatility': float(np.std(returns_clean) * np.sqrt(252)),
                'sharpe_ratio': float(np.mean(returns_clean) / np.std(returns_clean) * np.sqrt(252)),
                'max_drawdown': float(self.calculate_max_drawdown(daily_performance)),
                'total_trades': int(daily_performance["position_count"].sum()),
                'total_costs': float(daily_performance["daily_costs"].sum())
            }
            
            return {
                'daily_performance': daily_performance,
                'performance_stats': performance_stats
            }
        
        def calculate_max_drawdown(self, daily_performance: pl.DataFrame) -> float:
            """Calculate maximum drawdown"""
            portfolio_values = daily_performance["portfolio_value"].to_numpy()
            running_max = np.maximum.accumulate(portfolio_values)
            drawdowns = (portfolio_values - running_max) / running_max
            return float(np.min(drawdowns))
    
    # Example signal logic
    def momentum_signal_logic(universe: pl.LazyFrame) -> pl.LazyFrame:
        return (
            universe
            .with_columns([
                # 12-1 month momentum
                pl.col("adj_close").pct_change(periods=252).over("symbol").alias("momentum_12m"),
                pl.col("adj_close").pct_change(periods=21).over("symbol").alias("momentum_1m")
            ])
            .with_columns([
                # Momentum signal (long top quintile, short bottom quintile)
                pl.when(pl.col("momentum_12m").qcut(5).over("date") == 4)  # Top quintile (0-indexed)
                .then(pl.lit(1))
                .when(pl.col("momentum_12m").qcut(5).over("date") == 0)    # Bottom quintile
                .then(pl.lit(-1))
                .otherwise(pl.lit(0))
                .alias("signal")
            ])
        )
    
    # Example position logic
    def equal_weight_position_logic(signals: pl.LazyFrame) -> pl.LazyFrame:
        return (
            signals
            .filter(pl.col("signal") != 0)
            .with_columns([
                # Equal weight within each signal group
                (pl.col("signal") / pl.col("signal").abs().sum().over("date")).alias("position")
            ])
        )
    
    return PolarsBacktester, momentum_signal_logic, equal_weight_position_logic
```

---

## Performance Attribution

### Portfolio Performance Attribution

```python
def performance_attribution():
    """Brinson-Hood-Beebower performance attribution model"""
    
    def calculate_attribution(portfolio_returns: pl.LazyFrame, benchmark_returns: pl.LazyFrame) -> pl.LazyFrame:
        """Calculate sector-based performance attribution"""
        
        return (
            portfolio_returns
            .join(benchmark_returns, on=["date", "sector"], how="outer", suffix="_bench")
            .with_columns([
                # Fill missing values
                pl.col("weight").fill_null(0).alias("portfolio_weight"),
                pl.col("weight_bench").fill_null(0).alias("benchmark_weight"),
                pl.col("return").fill_null(0).alias("portfolio_return"),
                pl.col("return_bench").fill_null(0).alias("benchmark_return")
            ])
            .with_columns([
                # Attribution components
                # Asset Allocation Effect: (wp - wb) × rb
                ((pl.col("portfolio_weight") - pl.col("benchmark_weight")) * pl.col("benchmark_return")).alias("allocation_effect"),
                
                # Security Selection Effect: wb × (rp - rb)  
                (pl.col("benchmark_weight") * (pl.col("portfolio_return") - pl.col("benchmark_return"))).alias("selection_effect"),
                
                # Interaction Effect: (wp - wb) × (rp - rb)
                ((pl.col("portfolio_weight") - pl.col("benchmark_weight")) * 
                 (pl.col("portfolio_return") - pl.col("benchmark_return"))).alias("interaction_effect")
            ])
            .with_columns([
                # Total attribution
                (pl.col("allocation_effect") + pl.col("selection_effect") + pl.col("interaction_effect")).alias("total_attribution")
            ])
        )
    
    def rolling_attribution(attribution_data: pl.LazyFrame, window: int = 252) -> pl.LazyFrame:
        """Calculate rolling attribution metrics"""
        
        return (
            attribution_data
            .sort(["sector", "date"])
            .with_columns([
                # Rolling attribution effects
                pl.col("allocation_effect").rolling_sum(window_size=window).over("sector").alias(f"allocation_effect_{window}d"),
                pl.col("selection_effect").rolling_sum(window_size=window).over("sector").alias(f"selection_effect_{window}d"),
                pl.col("interaction_effect").rolling_sum(window_size=window).over("sector").alias(f"interaction_effect_{window}d"),
                
                # Rolling volatility of attribution
                pl.col("allocation_effect").rolling_std(window_size=window).over("sector").alias(f"allocation_vol_{window}d"),
                pl.col("selection_effect").rolling_std(window_size=window).over("sector").alias(f"selection_vol_{window}d")
            ])
        )
    
    return calculate_attribution, rolling_attribution
```

---

## Navigation

### Related Guides

- **[Main README](README.md)** - Overview and quick start guide
- **[Polars vs Pandas Integration](polars_pandas_integration.md)** - Integration strategies and conversion patterns
- **[Performance & Large Datasets](polars_performance_and_large_datasets.md)** - Performance optimization and large-scale processing
- **[Streaming & Memory Management](polars_streaming_and_memory.md)** - Streaming configuration and memory optimization
- **[Advanced Techniques](polars_advanced_techniques.md)** - Complex queries, backtesting, and debugging
- **[Practical Implementation](polars_practical_implementation.md)** - I/O best practices, data cleaning, and troubleshooting

### Key Takeaways

1. **Time Series Excellence**: Polars excels at time series operations with efficient window functions and over() clauses
2. **Cross-Sectional Analysis**: Use lazy evaluation for complex ranking and percentile calculations across large universes
3. **Event Studies**: Implement robust event study methodologies with proper market adjustment and corporate action handling
4. **Risk Modeling**: Calculate comprehensive risk metrics including VaR, drawdowns, and beta using vectorized operations
5. **Factor Research**: Build complete factor research pipelines from data preparation through performance attribution
6. **Backtesting**: Create flexible backtesting frameworks that handle position sizing, transaction costs, and performance analytics

This guide provides the foundation for sophisticated quantitative research workflows using Polars' high-performance capabilities.