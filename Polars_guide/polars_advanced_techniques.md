# Polars Advanced Techniques

Advanced patterns and techniques for complex analytical queries, backtesting frameworks, performance debugging, and production-ready error handling using Polars.

## Table of Contents

1. [Complex Analytical Queries](#complex-analytical-queries)
2. [Advanced Risk Modeling Patterns](#advanced-risk-modeling-patterns)
3. [Backtesting Framework Implementation](#backtesting-framework-implementation)
4. [Performance Debugging & Optimization](#performance-debugging--optimization)
5. [Production Error Handling](#production-error-handling)
6. [Advanced Query Optimization](#advanced-query-optimization)
7. [Data Validation & Quality Control](#data-validation--quality-control)
8. [Navigation](#navigation)

---

## Complex Analytical Queries

### Multi-Factor Analysis

Advanced multi-factor analysis patterns for quantitative research:

```python
import polars as pl
import numpy as np
from contextlib import contextmanager

def advanced_analytical_patterns():
    """Complex analytical patterns for quantitative research"""
    
    def multi_factor_analysis(df: pl.LazyFrame) -> pl.LazyFrame:
        """Multi-factor model analysis with Polars"""
        
        return (
            df
            .sort(["symbol", "date"])
            # Step 1: Calculate factor exposures
            .with_columns([
                # Value factors
                pl.col("book_value").truediv(pl.col("market_cap")).alias("book_to_market"),
                pl.col("earnings").truediv(pl.col("price")).alias("earnings_yield"),
                
                # Momentum factors
                pl.col("close").pct_change(periods=21).over("symbol").alias("mom_1m"),
                pl.col("close").pct_change(periods=252).over("symbol").alias("mom_12m"),
                
                # Quality factors  
                pl.col("roe").alias("quality_roe"),
                pl.col("debt_to_equity").alias("leverage"),
                
                # Size factor
                pl.col("market_cap").log().alias("log_market_cap")
            ])
            # Step 2: Cross-sectional standardization
            .with_columns([
                # Z-scores within each date
                ((pl.col("book_to_market") - pl.col("book_to_market").mean().over("date")) /
                 pl.col("book_to_market").std().over("date")).alias("value_zscore"),
                
                ((pl.col("mom_12m") - pl.col("mom_12m").mean().over("date")) /
                 pl.col("mom_12m").std().over("date")).alias("momentum_zscore"),
                
                ((pl.col("quality_roe") - pl.col("quality_roe").mean().over("date")) /
                 pl.col("quality_roe").std().over("date")).alias("quality_zscore"),
                
                ((pl.col("log_market_cap") - pl.col("log_market_cap").mean().over("date")) /
                 pl.col("log_market_cap").std().over("date")).alias("size_zscore")
            ])
            # Step 3: Composite scores
            .with_columns([
                (0.25 * pl.col("value_zscore") + 
                 0.25 * pl.col("momentum_zscore") + 
                 0.25 * pl.col("quality_zscore") + 
                 0.25 * pl.col("size_zscore")).alias("composite_score")
            ])
            # Step 4: Portfolio assignments
            .with_columns([
                pl.col("composite_score").qcut(10, labels=[f"D{i}" for i in range(1, 11)]).alias("decile")
            ])
        )
    
    def regime_detection_analysis(df: pl.LazyFrame) -> pl.LazyFrame:
        """Detect market regimes using quantiles and rolling statistics"""
        
        return (
            df
            # Calculate market-level indicators
            .group_by("date")
            .agg([
                pl.col("returns").mean().alias("market_return"),
                pl.col("returns").std().alias("cross_sectional_vol"),
                pl.col("volume").sum().alias("total_volume"),
                pl.col("returns").quantile(0.1).alias("return_10th_pct"),
                pl.col("returns").quantile(0.9).alias("return_90th_pct")
            ])
            # Calculate regime indicators
            .sort("date")
            .with_columns([
                # Rolling market characteristics
                pl.col("market_return").rolling_mean(window_size=20).alias("market_trend"),
                pl.col("cross_sectional_vol").rolling_mean(window_size=20).alias("avg_dispersion"),
                pl.col("total_volume").rolling_mean(window_size=20).alias("avg_volume"),
                
                # Volatility regime
                pl.col("cross_sectional_vol").rolling_quantile(quantile=0.8, window_size=252).alias("vol_80th_pct")
            ])
            .with_columns([
                # Define regimes
                pl.when(pl.col("cross_sectional_vol") > pl.col("vol_80th_pct"))
                .then(pl.lit("High Vol"))
                .when(pl.col("market_trend") > 0)
                .then(pl.lit("Bull Market"))
                .otherwise(pl.lit("Bear Market"))
                .alias("market_regime"),
                
                # Risk-on/Risk-off indicator
                pl.when((pl.col("return_90th_pct") - pl.col("return_10th_pct")) > 
                       (pl.col("return_90th_pct") - pl.col("return_10th_pct")).rolling_quantile(0.8, window_size=252))
                .then(pl.lit("Risk-Off"))
                .otherwise(pl.lit("Risk-On"))
                .alias("risk_sentiment")
            ])
        )
    
    def correlation_analysis(df: pl.LazyFrame, window_size: int = 252) -> pl.LazyFrame:
        """Dynamic correlation analysis between assets"""
        
        # Pivot returns to have assets as columns
        returns_matrix = (
            df
            .select(["date", "symbol", "returns"])
            .filter(pl.col("returns").is_not_null())
            .pivot(index="date", columns="symbol", values="returns")
            .sort("date")
        )
        
        # Calculate rolling correlations (simplified - would need custom function for full correlation matrix)
        asset_columns = [col for col in returns_matrix.columns if col != "date"]
        
        if len(asset_columns) >= 2:
            correlation_exprs = []
            
            # Calculate rolling correlations between first few assets (example)
            for i in range(min(5, len(asset_columns))):
                for j in range(i + 1, min(5, len(asset_columns))):
                    asset1, asset2 = asset_columns[i], asset_columns[j]
                    
                    correlation_exprs.append(
                        pl.corr(pl.col(asset1), pl.col(asset2), method="pearson")
                        .rolling_map(lambda x: x, window_size=window_size)
                        .alias(f"corr_{asset1}_{asset2}")
                    )
            
            return returns_matrix.with_columns(correlation_exprs)
        
        return returns_matrix
    
    return multi_factor_analysis, regime_detection_analysis, correlation_analysis

def advanced_time_series_patterns():
    """Advanced time series patterns for quantitative analysis"""
    
    def seasonal_decomposition(df: pl.LazyFrame, frequency: int = 252) -> pl.LazyFrame:
        """Decompose time series into trend, seasonal, and residual components"""
        
        return (
            df
            .sort(["symbol", "date"])
            .with_columns([
                # Trend component using moving average
                pl.col("returns").rolling_mean(window_size=frequency).over("symbol").alias("trend"),
                
                # Seasonal component (day of year effect)
                pl.col("date").dt.ordinal_day().alias("day_of_year"),
                
                # Cyclical patterns (monthly effect)
                pl.col("date").dt.month().alias("month")
            ])
            .with_columns([
                # Detrended series
                (pl.col("returns") - pl.col("trend")).alias("detrended"),
                
                # Seasonal averages
                pl.col("detrended").mean().over(["symbol", "month"]).alias("monthly_seasonal"),
                pl.col("detrended").mean().over(["symbol", "day_of_year"]).alias("daily_seasonal")
            ])
            .with_columns([
                # Residual component
                (pl.col("detrended") - pl.col("monthly_seasonal") - pl.col("daily_seasonal")).alias("residual")
            ])
        )
    
    def change_point_detection(df: pl.LazyFrame, window_size: int = 60) -> pl.LazyFrame:
        """Detect change points in time series using rolling statistics"""
        
        return (
            df
            .sort(["symbol", "date"])
            .with_columns([
                # Rolling statistics for change detection
                pl.col("returns").rolling_mean(window_size=window_size).over("symbol").alias("rolling_mean"),
                pl.col("returns").rolling_std(window_size=window_size).over("symbol").alias("rolling_std"),
                
                # Shifted versions for comparison
                pl.col("returns").rolling_mean(window_size=window_size).shift(window_size).over("symbol").alias("prev_mean"),
                pl.col("returns").rolling_std(window_size=window_size).shift(window_size).over("symbol").alias("prev_std")
            ])
            .with_columns([
                # Change detection metrics
                ((pl.col("rolling_mean") - pl.col("prev_mean")).abs() / 
                 pl.col("prev_std")).alias("mean_change_score"),
                
                ((pl.col("rolling_std") - pl.col("prev_std")).abs() / 
                 pl.col("prev_std")).alias("vol_change_score")
            ])
            .with_columns([
                # Change point flags
                (pl.col("mean_change_score") > 2.0).alias("mean_change_point"),
                (pl.col("vol_change_score") > 1.0).alias("vol_change_point"),
                
                # Combined change score
                (pl.col("mean_change_score") + pl.col("vol_change_score")).alias("combined_change_score")
            ])
        )
    
    def regime_switching_indicators(df: pl.LazyFrame) -> pl.LazyFrame:
        """Calculate regime-switching indicators"""
        
        return (
            df
            .sort(["symbol", "date"])
            .with_columns([
                # Volatility regime indicators
                pl.col("returns").rolling_std(window_size=22).over("symbol").alias("vol_22d"),
                pl.col("returns").rolling_std(window_size=252).over("symbol").alias("vol_252d"),
                
                # Momentum regime indicators
                pl.col("price").pct_change(periods=22).over("symbol").alias("momentum_22d"),
                pl.col("price").pct_change(periods=252).over("symbol").alias("momentum_252d"),
                
                # Trend strength
                pl.col("price").rolling_mean(window_size=50).over("symbol").alias("ma_50"),
                pl.col("price").rolling_mean(window_size=200).over("symbol").alias("ma_200")
            ])
            .with_columns([
                # Regime classifications
                pl.when(pl.col("vol_22d") > pl.col("vol_252d") * 1.5)
                .then(pl.lit("High Vol"))
                .when(pl.col("vol_22d") < pl.col("vol_252d") * 0.7)
                .then(pl.lit("Low Vol"))
                .otherwise(pl.lit("Normal Vol"))
                .alias("vol_regime"),
                
                # Trend regime
                pl.when(pl.col("ma_50") > pl.col("ma_200") * 1.02)
                .then(pl.lit("Strong Uptrend"))
                .when(pl.col("ma_50") < pl.col("ma_200") * 0.98)
                .then(pl.lit("Strong Downtrend"))
                .otherwise(pl.lit("Sideways"))
                .alias("trend_regime"),
                
                # Momentum regime
                pl.when(pl.col("momentum_22d") > 0.1)
                .then(pl.lit("Strong Momentum"))
                .when(pl.col("momentum_22d") < -0.1)
                .then(pl.lit("Strong Reversal"))
                .otherwise(pl.lit("Neutral"))
                .alias("momentum_regime")
            ])
        )
    
    return seasonal_decomposition, change_point_detection, regime_switching_indicators
```

---

## Advanced Risk Modeling Patterns

### Comprehensive Risk Metrics

```python
def advanced_risk_modeling_patterns():
    """Advanced risk modeling using Polars"""
    
    def value_at_risk_calculations(df: pl.LazyFrame) -> pl.LazyFrame:
        """Calculate VaR using multiple methods"""
        
        return (
            df
            .sort(["symbol", "date"])
            .with_columns([
                # Historical VaR (5th percentile)
                pl.col("returns").rolling_quantile(quantile=0.05, window_size=252).over("symbol").alias("var_5pct_hist"),
                pl.col("returns").rolling_quantile(quantile=0.01, window_size=252).over("symbol").alias("var_1pct_hist"),
                
                # Parametric VaR (assuming normal distribution)
                (pl.col("returns").rolling_mean(window_size=252) - 
                 1.645 * pl.col("returns").rolling_std(window_size=252)).over("symbol").alias("var_5pct_param"),
                
                # Expected Shortfall (Conditional VaR)
                pl.col("returns")
                .map_batches(lambda x: calculate_expected_shortfall(x.to_numpy()))
                .over("symbol").alias("expected_shortfall_5pct")
            ])
        )
    
    def beta_calculations(df: pl.LazyFrame, market_returns: pl.LazyFrame) -> pl.LazyFrame:
        """Calculate various beta measures"""
        
        # Join with market returns
        combined_data = df.join(market_returns, on="date", how="inner")
        
        return (
            combined_data
            .sort(["symbol", "date"])
            .with_columns([
                # Rolling beta calculation
                pl.corr(pl.col("returns"), pl.col("market_return"))
                .rolling_map(lambda x: x, window_size=252)
                .mul(
                    pl.col("returns").rolling_std(window_size=252)
                    .truediv(pl.col("market_return").rolling_std(window_size=252))
                )
                .over("symbol").alias("beta_252d"),
                
                # Downside beta (sensitivity during market declines)
                pl.when(pl.col("market_return") < 0)
                .then(pl.col("returns"))
                .otherwise(pl.lit(None))
                .pipe(lambda x: 
                    pl.corr(x, pl.when(pl.col("market_return") < 0).then(pl.col("market_return")))
                    .rolling_map(lambda y: y, window_size=252)
                )
                .over("symbol").alias("downside_beta"),
                
                # Alpha calculation
                (pl.col("returns").rolling_mean(window_size=252) * 252 -
                 pl.col("beta_252d") * pl.col("market_return").rolling_mean(window_size=252) * 252)
                .over("symbol").alias("alpha_annualized")
            ])
        )
    
    def drawdown_analysis(df: pl.LazyFrame) -> pl.LazyFrame:
        """Comprehensive drawdown analysis"""
        
        return (
            df
            .sort(["symbol", "date"])
            .with_columns([
                # Cumulative returns
                ((pl.col("returns") + 1).log().cumsum().exp()).over("symbol").alias("cumulative_return"),
            ])
            .with_columns([
                # Running maximum
                pl.col("cumulative_return").cummax().over("symbol").alias("running_max"),
            ])
            .with_columns([
                # Drawdown
                (pl.col("cumulative_return") / pl.col("running_max") - 1).alias("drawdown"),
                
                # Underwater curve (time since last peak)
                (pl.col("cumulative_return") < pl.col("running_max")).alias("underwater")
            ])
            .with_columns([
                # Maximum drawdown over rolling windows
                pl.col("drawdown").rolling_min(window_size=252).over("symbol").alias("max_dd_252d"),
                pl.col("drawdown").rolling_min(window_size=756).over("symbol").alias("max_dd_3y"),
                
                # Drawdown duration (simplified)
                pl.col("underwater").cast(pl.Int32).rolling_sum(window_size=252).over("symbol").alias("underwater_days_252d")
            ])
        )
    
    def stress_testing_scenarios(df: pl.LazyFrame) -> pl.LazyFrame:
        """Apply stress testing scenarios"""
        
        return (
            df
            .sort(["symbol", "date"])
            .with_columns([
                # Historical stress scenarios
                pl.col("returns").rolling_quantile(quantile=0.01, window_size=252).over("symbol").alias("worst_1pct_scenario"),
                pl.col("returns").rolling_quantile(quantile=0.05, window_size=252).over("symbol").alias("worst_5pct_scenario"),
                
                # Market crash scenarios (fixed shocks)
                (pl.col("returns") - 0.20).alias("market_crash_20pct"),
                (pl.col("returns") - 0.30).alias("market_crash_30pct"),
                
                # Volatility shock scenarios
                pl.col("returns").rolling_std(window_size=22).over("symbol").alias("current_vol"),
                pl.col("returns").rolling_std(window_size=252).over("symbol").alias("long_term_vol")
            ])
            .with_columns([
                # Vol shock impact
                (pl.col("current_vol") * 2 - pl.col("long_term_vol")).alias("vol_shock_2x"),
                (pl.col("current_vol") * 3 - pl.col("long_term_vol")).alias("vol_shock_3x"),
                
                # Combined stress scenarios
                pl.min_horizontal([
                    pl.col("market_crash_20pct"),
                    pl.col("worst_5pct_scenario"),
                    pl.col("vol_shock_2x")
                ]).alias("combined_stress_scenario")
            ])
        )
    
    return value_at_risk_calculations, beta_calculations, drawdown_analysis, stress_testing_scenarios

def calculate_expected_shortfall(returns: np.ndarray, confidence_level: float = 0.05) -> np.ndarray:
    """Calculate Expected Shortfall (Conditional VaR) using NumPy"""
    
    if len(returns) < 20:  # Need minimum observations
        return np.full_like(returns, np.nan)
    
    # Calculate VaR threshold
    var_threshold = np.quantile(returns, confidence_level)
    
    # Expected Shortfall is the mean of returns below VaR
    tail_returns = returns[returns <= var_threshold]
    
    if len(tail_returns) == 0:
        return np.full_like(returns, np.nan)
    
    expected_shortfall = np.mean(tail_returns)
    return np.full_like(returns, expected_shortfall)

def portfolio_risk_decomposition():
    """Decompose portfolio risk into individual contributions"""
    
    def marginal_contribution_to_risk(portfolio_returns: pl.LazyFrame, weights: pl.LazyFrame) -> pl.LazyFrame:
        """Calculate marginal contribution to portfolio risk"""
        
        # Join portfolio returns with weights
        portfolio_data = portfolio_returns.join(weights, on=["date", "symbol"], how="inner")
        
        return (
            portfolio_data
            .group_by("date")
            .agg([
                # Portfolio return
                (pl.col("returns") * pl.col("weight")).sum().alias("portfolio_return"),
                
                # Individual asset statistics
                pl.col("returns").alias("asset_returns"),
                pl.col("weight").alias("asset_weights"),
                pl.col("symbol").alias("assets")
            ])
            .with_columns([
                # Portfolio variance (simplified calculation)
                pl.col("portfolio_return").var().alias("portfolio_variance"),
                
                # Marginal contributions (requires covariance calculation)
                # This is a simplified version - full implementation would need covariance matrix
                (pl.col("asset_returns") * pl.col("asset_weights")).alias("weighted_returns")
            ])
        )
    
    def component_var_decomposition(df: pl.LazyFrame) -> pl.LazyFrame:
        """Component VaR decomposition"""
        
        return (
            df
            .group_by(["date", "portfolio_id"])
            .agg([
                # Portfolio VaR
                pl.col("portfolio_return").quantile(0.05).alias("portfolio_var"),
                
                # Component contributions
                pl.col("component_return").alias("component_returns"),
                pl.col("weight").alias("component_weights")
            ])
            .with_columns([
                # Component VaR contributions (simplified)
                (pl.col("component_returns") * pl.col("component_weights") / 
                 pl.col("portfolio_var")).alias("component_var_contribution")
            ])
        )
    
    return marginal_contribution_to_risk, component_var_decomposition
```

---

## Backtesting Framework Implementation

### Comprehensive Backtesting System

```python
def comprehensive_backtesting_framework():
    """Advanced backtesting framework with realistic trading simulation"""
    
    class AdvancedPolarsBacktester:
        def __init__(self, universe_data: pl.LazyFrame, initial_capital: float = 1000000):
            self.universe = universe_data
            self.initial_capital = initial_capital
            self.transaction_costs = 0.001  # 10 bps
            self.results = {}
            
        def simulate_realistic_trading(self, signals: pl.LazyFrame) -> pl.LazyFrame:
            """Simulate realistic trading with market impact and timing delays"""
            
            return (
                signals
                .sort(["symbol", "date"])
                .with_columns([
                    # Position sizing (risk-adjusted)
                    (pl.col("signal") / pl.col("volatility").clip(0.01, 1.0)).alias("risk_adjusted_signal"),
                    
                    # Market impact (larger trades have higher impact)
                    (0.001 * pl.col("position_size").abs().pow(0.5)).alias("market_impact"),
                    
                    # Execution delay (can't trade on same day as signal)
                    pl.col("close").shift(-1).over("symbol").alias("execution_price"),
                    
                    # Bid-ask spread impact
                    (0.0005 * pl.col("close")).alias("bid_ask_spread")
                ])
                .with_columns([
                    # Total transaction cost
                    (pl.col("market_impact") + pl.col("bid_ask_spread") + self.transaction_costs).alias("total_cost"),
                    
                    # Realistic execution price
                    pl.when(pl.col("risk_adjusted_signal") > 0)
                    .then(pl.col("execution_price") * (1 + pl.col("total_cost")))  # Buy at ask
                    .when(pl.col("risk_adjusted_signal") < 0)
                    .then(pl.col("execution_price") * (1 - pl.col("total_cost")))  # Sell at bid
                    .otherwise(pl.col("execution_price"))
                    .alias("effective_price")
                ])
                .with_columns([
                    # Position changes and PnL
                    (pl.col("risk_adjusted_signal") - pl.col("risk_adjusted_signal").shift(1).fill_null(0))
                    .over("symbol").alias("position_change"),
                    
                    # PnL calculation with realistic pricing
                    (pl.col("position_size") * (pl.col("close") / pl.col("effective_price").shift(1) - 1))
                    .over("symbol").alias("unrealized_pnl")
                ])
            )
        
        def calculate_performance_metrics(self, trading_results: pl.LazyFrame) -> dict:
            """Calculate comprehensive performance metrics"""
            
            # Daily portfolio performance
            daily_perf = (
                trading_results
                .group_by("date")
                .agg([
                    pl.col("unrealized_pnl").sum().alias("daily_pnl"),
                    pl.col("total_cost").sum().alias("daily_costs"),
                    pl.col("position_size").abs().sum().alias("gross_exposure"),
                    pl.col("position_size").sum().alias("net_exposure")
                ])
                .sort("date")
                .with_columns([
                    # Portfolio value evolution
                    (self.initial_capital + pl.col("daily_pnl").cumsum()).alias("portfolio_value"),
                    
                    # Returns
                    pl.col("daily_pnl").truediv(
                        self.initial_capital + pl.col("daily_pnl").shift(1).cumsum().fill_null(0)
                    ).alias("daily_return")
                ])
                .collect()
            )
            
            # Performance statistics
            returns = daily_perf["daily_return"].to_numpy()
            returns_clean = returns[~np.isnan(returns)]
            
            # Calculate comprehensive metrics
            metrics = {
                # Return metrics
                'total_return': float(daily_perf["portfolio_value"][-1] / self.initial_capital - 1),
                'annualized_return': float(np.mean(returns_clean) * 252),
                'volatility': float(np.std(returns_clean) * np.sqrt(252)),
                
                # Risk metrics
                'sharpe_ratio': float(np.mean(returns_clean) / np.std(returns_clean) * np.sqrt(252)) if np.std(returns_clean) > 0 else 0,
                'sortino_ratio': float(self._calculate_sortino_ratio(returns_clean)),
                'max_drawdown': float(self._calculate_max_drawdown(daily_perf["portfolio_value"].to_numpy())),
                
                # Trading metrics
                'total_trades': int(daily_perf["gross_exposure"].sum()),
                'total_costs': float(daily_perf["daily_costs"].sum()),
                'avg_gross_exposure': float(daily_perf["gross_exposure"].mean()),
                'avg_net_exposure': float(daily_perf["net_exposure"].mean()),
                
                # Risk-adjusted metrics
                'calmar_ratio': float(np.mean(returns_clean) * 252 / abs(self._calculate_max_drawdown(daily_perf["portfolio_value"].to_numpy()))),
                'var_5pct': float(np.percentile(returns_clean, 5)),
                'expected_shortfall_5pct': float(np.mean(returns_clean[returns_clean <= np.percentile(returns_clean, 5)]))
            }
            
            return {
                'daily_performance': daily_perf,
                'performance_metrics': metrics
            }
        
        def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
            """Calculate Sortino ratio"""
            downside_returns = returns[returns < 0]
            if len(downside_returns) == 0:
                return 0
            
            downside_deviation = np.std(downside_returns)
            return np.mean(returns) * np.sqrt(252) / (downside_deviation * np.sqrt(252))
        
        def _calculate_max_drawdown(self, portfolio_values: np.ndarray) -> float:
            """Calculate maximum drawdown"""
            running_max = np.maximum.accumulate(portfolio_values)
            drawdowns = (portfolio_values - running_max) / running_max
            return float(np.min(drawdowns))
        
        def run_walk_forward_analysis(self, train_window: int = 252, test_window: int = 63):
            """Run walk-forward backtesting analysis"""
            
            # Get date range
            dates = (
                self.universe
                .select("date")
                .unique()
                .sort("date")
                .collect()
            )["date"].to_list()
            
            results = []
            
            for i in range(train_window, len(dates) - test_window, test_window):
                train_start = dates[i - train_window]
                train_end = dates[i - 1]
                test_start = dates[i]
                test_end = dates[min(i + test_window - 1, len(dates) - 1)]
                
                print(f"Training: {train_start} to {train_end}, Testing: {test_start} to {test_end}")
                
                # Train model on training data
                train_data = (
                    self.universe
                    .filter(pl.col("date").is_between(train_start, train_end))
                )
                
                # Generate signals for test period
                test_data = (
                    self.universe
                    .filter(pl.col("date").is_between(test_start, test_end))
                )
                
                # Run backtest for this period
                # (This would include your specific signal generation and position sizing logic)
                
                results.append({
                    'train_period': (train_start, train_end),
                    'test_period': (test_start, test_end),
                    'performance': {}  # Would contain performance metrics
                })
            
            return results
    
    return AdvancedPolarsBacktester

def strategy_comparison_framework():
    """Framework for comparing multiple strategies"""
    
    def multi_strategy_comparison(strategies: dict, universe: pl.LazyFrame) -> pl.DataFrame:
        """Compare multiple strategies on the same universe"""
        
        results = {}
        
        for strategy_name, strategy_func in strategies.items():
            print(f"Running strategy: {strategy_name}")
            
            # Run strategy
            strategy_signals = strategy_func(universe)
            
            # Calculate performance
            backtester = AdvancedPolarsBacktester(universe)
            trading_results = backtester.simulate_realistic_trading(strategy_signals)
            performance = backtester.calculate_performance_metrics(trading_results)
            
            results[strategy_name] = performance['performance_metrics']
        
        # Convert to comparison DataFrame
        comparison_df = pl.DataFrame(results).transpose()
        
        return comparison_df
    
    def regime_conditional_analysis(strategy_results: pl.DataFrame, regime_data: pl.DataFrame) -> pl.DataFrame:
        """Analyze strategy performance conditional on market regimes"""
        
        # Join strategy results with regime data
        conditional_performance = (
            strategy_results
            .join(regime_data, on="date", how="inner")
            .group_by("market_regime")
            .agg([
                pl.col("daily_return").mean().alias("avg_return"),
                pl.col("daily_return").std().alias("volatility"),
                pl.col("daily_return").count().alias("observations"),
                pl.col("daily_return").quantile(0.05).alias("var_5pct")
            ])
            .with_columns([
                # Risk-adjusted metrics by regime
                (pl.col("avg_return") / pl.col("volatility") * np.sqrt(252)).alias("sharpe_ratio"),
                (pl.col("avg_return") * 252).alias("annualized_return"),
                (pl.col("volatility") * np.sqrt(252)).alias("annualized_volatility")
            ])
        )
        
        return conditional_performance.collect()
    
    return multi_strategy_comparison, regime_conditional_analysis
```

---

## Performance Debugging & Optimization

### Query Performance Analysis

```python
def performance_debugging_toolkit():
    """Debug performance issues in Polars queries"""
    
    def query_performance_analysis():
        """Analyze and optimize slow queries"""
        
        def diagnose_slow_query():
            """Step-by-step query diagnosis"""
            
            # Step 1: Check the query plan
            slow_query = (
                pl.scan_parquet("large_dataset.parquet")
                .filter(pl.col("volume") > 1000000)
                .filter(pl.col("symbol").is_in(["AAPL", "GOOGL", "MSFT"]))
                .group_by("symbol")
                .agg([
                    pl.col("returns").mean().alias("avg_return"),
                    pl.col("returns").std().alias("volatility")
                ])
            )
            
            # Examine query plan for optimization opportunities
            print("=== Query Plan ===")
            print(slow_query.explain(optimized=True))
            
            # Step 2: Profile the execution
            import time
            start_time = time.time()
            result = slow_query.collect()
            execution_time = time.time() - start_time
            print(f"Execution time: {execution_time:.2f} seconds")
            
            return result, execution_time
        
        def optimize_query():
            """Apply optimizations to improve performance"""
            
            # Optimization 1: More selective filtering first
            optimized_query = (
                pl.scan_parquet("large_dataset.parquet")
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
            print(f"Optimized execution time: {execution_time:.2f} seconds")
            
            return result, execution_time
        
        return diagnose_slow_query, optimize_query
    
    def memory_usage_debugging():
        """Debug memory usage issues"""
        
        def memory_profiling():
            """Profile memory usage during operations"""
            
            import psutil
            import gc
            
            def get_memory_mb():
                return psutil.Process().memory_info().rss / 1024 / 1024
            
            print(f"Initial memory: {get_memory_mb():.1f} MB")
            
            # Load data
            df = pl.scan_parquet("large_file.parquet").collect()
            print(f"After loading: {get_memory_mb():.1f} MB")
            
            # Process data
            processed = df.with_columns([
                pl.col("price").pct_change().alias("returns"),
                pl.col("returns").rolling_std(window_size=252).alias("volatility")
            ])
            print(f"After processing: {get_memory_mb():.1f} MB")
            
            # Clean up
            del df
            gc.collect()
            print(f"After cleanup: {get_memory_mb():.1f} MB")
            
            return processed
        
        def identify_memory_hotspots():
            """Identify operations that use most memory"""
            
            operations = {
                'read_data': lambda: pl.read_parquet("large_file.parquet"),
                'calculate_returns': lambda df: df.with_columns([pl.col("price").pct_change().alias("returns")]),
                'rolling_operations': lambda df: df.with_columns([pl.col("returns").rolling_std(window_size=252).alias("vol")]),
                'groupby_operations': lambda df: df.group_by("symbol").agg([pl.col("returns").mean()])
            }
            
            memory_usage = {}
            
            for op_name, operation in operations.items():
                gc.collect()
                start_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                if op_name == 'read_data':
                    result = operation()
                else:
                    # Use result from previous operation
                    result = operation(result)
                
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_usage[op_name] = end_memory - start_memory
                
                print(f"{op_name}: {memory_usage[op_name]:.1f} MB")
            
            return memory_usage
        
        return memory_profiling, identify_memory_hotspots
    
    def query_optimization_analyzer():
        """Analyze query execution plans and suggest optimizations"""
        
        def analyze_query_plan(query: pl.LazyFrame):
            """Analyze query execution plan"""
            
            # Get query plan
            plan = query.explain(optimized=True)
            
            # Analyze plan characteristics
            analysis = {
                'has_streaming': 'STREAMING' in plan,
                'has_joins': 'JOIN' in plan,
                'has_filters': 'FILTER' in plan,
                'has_aggregations': 'AGGREGATE' in plan,
                'has_sorts': 'SORT' in plan,
                'plan_complexity': len(plan.split('\n'))
            }
            
            # Suggest optimizations
            suggestions = []
            
            if analysis['has_joins'] and not analysis['has_filters']:
                suggestions.append("Consider adding filters before joins to reduce data size")
            
            if analysis['has_sorts'] and analysis['has_streaming']:
                suggestions.append("Sorting may disable streaming - consider if sort is necessary")
            
            if not analysis['has_streaming'] and analysis['plan_complexity'] > 10:
                suggestions.append("Complex query may benefit from breaking into smaller parts")
            
            print("Query Analysis:")
            print(f"  Streaming capable: {analysis['has_streaming']}")
            print(f"  Has joins: {analysis['has_joins']}")
            print(f"  Has filters: {analysis['has_filters']}")
            print(f"  Plan complexity: {analysis['plan_complexity']} steps")
            
            if suggestions:
                print("\nOptimization suggestions:")
                for suggestion in suggestions:
                    print(f"  • {suggestion}")
            
            print(f"\nQuery Plan:\n{plan}")
            
            return analysis
        
        def benchmark_query_variations(base_query: pl.LazyFrame, variations: dict) -> dict:
            """Benchmark different variations of a query"""
            
            results = {}
            
            for name, query_func in variations.items():
                modified_query = query_func(base_query)
                
                # Time the execution
                import time
                start_time = time.time()
                result = modified_query.collect()
                execution_time = time.time() - start_time
                
                results[name] = {
                    'execution_time': execution_time,
                    'row_count': len(result),
                    'memory_usage': result.estimated_size()
                }
                
                print(f"{name}: {execution_time:.2f}s, {len(result):,} rows, {result.estimated_size():,} bytes")
            
            return results
        
        return analyze_query_plan, benchmark_query_variations
    
    return query_performance_analysis, memory_usage_debugging, query_optimization_analyzer

@contextmanager
def performance_monitor(operation_name: str):
    """Context manager for monitoring performance"""
    
    import psutil
    import time
    
    process = psutil.Process()
    start_time = time.time()
    start_memory = process.memory_info().rss / 1024 / 1024
    start_cpu = process.cpu_percent()
    
    print(f"[{operation_name}] Starting - Memory: {start_memory:.1f} MB")
    
    try:
        yield
    finally:
        end_time = time.time()
        end_memory = process.memory_info().rss / 1024 / 1024
        end_cpu = process.cpu_percent()
        
        duration = end_time - start_time
        memory_delta = end_memory - start_memory
        
        print(f"[{operation_name}] Completed:")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Memory: {end_memory:.1f} MB (Δ{memory_delta:+.1f} MB)")
        print(f"  CPU: {end_cpu:.1f}%")
```

---

## Production Error Handling

### Robust Error Handling Patterns

```python
def production_error_handling():
    """Production-ready error handling patterns"""
    
    def robust_file_operations():
        """Handle file operations with comprehensive error handling"""
        
        def safe_file_loading(file_path: str, max_retries: int = 3):
            """Load files with retries and comprehensive error handling"""
            
            from pathlib import Path
            import time
            
            for attempt in range(max_retries):
                try:
                    # Check if file exists
                    if not Path(file_path).exists():
                        raise FileNotFoundError(f"File not found: {file_path}")
                    
                    # Check file size (warn if very large)
                    file_size = Path(file_path).stat().st_size / (1024 * 1024)  # MB
                    if file_size > 1000:  # > 1GB
                        print(f"Warning: Large file ({file_size:.1f} MB). Consider streaming.")
                    
                    # Try different loading strategies based on file size
                    if file_size > 5000:  # > 5GB
                        print("Using streaming for very large file")
                        df = pl.scan_parquet(file_path).collect(streaming=True)
                    else:
                        df = pl.read_parquet(file_path)
                    
                    # Validate loaded data
                    if len(df) == 0:
                        raise ValueError("Loaded DataFrame is empty")
                    
                    print(f"Successfully loaded {len(df):,} rows from {file_path}")
                    return df
                    
                except (FileNotFoundError, PermissionError) as e:
                    print(f"File access error (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(1)  # Wait before retry
                    
                except MemoryError as e:
                    print(f"Memory error (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        print("Trying streaming approach...")
                        try:
                            df = pl.scan_parquet(file_path).collect(streaming=True)
                            return df
                        except Exception:
                            continue
                    raise
                    
                except Exception as e:
                    print(f"Unexpected error (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(2 ** attempt)  # Exponential backoff
            
            return None
        
        def safe_file_writing(df: pl.DataFrame, output_path: str, backup: bool = True):
            """Write files with validation and backup"""
            
            from pathlib import Path
            
            try:
                # Ensure output directory exists
                output_dir = Path(output_path).parent
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Validate DataFrame before writing
                if len(df) == 0:
                    raise ValueError("Cannot write empty DataFrame")
                
                # Create backup if file exists
                if backup and Path(output_path).exists():
                    backup_path = f"{output_path}.backup"
                    Path(output_path).rename(backup_path)
                    print(f"Created backup: {backup_path}")
                
                # Write the file
                df.write_parquet(output_path, compression="zstd")
                
                # Verify write was successful
                test_read = pl.read_parquet(output_path)
                if len(test_read) != len(df):
                    raise ValueError("Write verification failed - row count mismatch")
                
                print(f"Successfully wrote {len(df):,} rows to {output_path}")
                return True
                
            except Exception as e:
                print(f"Error writing to {output_path}: {e}")
                
                # Try to restore backup if write failed
                if backup and Path(f"{output_path}.backup").exists():
                    Path(f"{output_path}.backup").rename(output_path)
                    print("Restored backup file")
                
                return False
        
        return safe_file_loading, safe_file_writing
    
    def data_validation_framework():
        """Comprehensive data validation framework"""
        
        def validate_financial_data(df: pl.DataFrame):
            """Comprehensive validation for financial data"""
            
            validation_results = {
                'is_valid': True,
                'errors': [],
                'warnings': [],
                'stats': {}
            }
            
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
                
                validation_results['stats']['price_stats'] = {
                    'min': df['price'].min(),
                    'max': df['price'].max(),
                    'mean': df['price'].mean(),
                    'null_count': df['price'].null_count()
                }
            
            # Check for missing dates
            if 'date' in df.columns:
                null_dates = df['date'].null_count()
                if null_dates > 0:
                    validation_results['errors'].append(f"{null_dates} null dates found")
                    validation_results['is_valid'] = False
                
                validation_results['stats']['date_range'] = {
                    'min_date': df['date'].min(),
                    'max_date': df['date'].max(),
                    'unique_dates': df['date'].n_unique()
                }
            
            # Check for duplicates
            if all(col in df.columns for col in ['symbol', 'date']):
                duplicates = df.select(['symbol', 'date']).is_duplicated().sum()
                if duplicates > 0:
                    validation_results['warnings'].append(f"{duplicates} duplicate symbol-date combinations")
            
            # Statistical outlier detection
            if 'price' in df.columns:
                price_stats = df['price'].describe()
                q1 = df['price'].quantile(0.25)
                q3 = df['price'].quantile(0.75)
                iqr = q3 - q1
                
                outliers = df.filter(
                    (pl.col('price') < (q1 - 1.5 * iqr)) |
                    (pl.col('price') > (q3 + 1.5 * iqr))
                ).height
                
                if outliers > len(df) * 0.05:  # More than 5% outliers
                    validation_results['warnings'].append(f"{outliers} statistical outliers detected ({outliers/len(df):.2%})")
            
            # Report results
            if not validation_results['is_valid']:
                print("❌ Data validation failed:")
                for error in validation_results['errors']:
                    print(f"  Error: {error}")
            
            if validation_results['warnings']:
                print("⚠️  Data validation warnings:")
                for warning in validation_results['warnings']:
                    print(f"  Warning: {warning}")
            
            if validation_results['is_valid'] and not validation_results['warnings']:
                print("✅ Data validation passed")
            
            return validation_results
        
        def handle_data_quality_issues(df: pl.DataFrame, validation_results: dict):
            """Automatically handle common data quality issues"""
            
            print("Handling data quality issues...")
            original_len = len(df)
            
            # Remove rows with negative/zero prices
            if 'price' in df.columns:
                df = df.filter(pl.col("price") > 0)
                removed_negative = original_len - len(df)
                if removed_negative > 0:
                    print(f"Removed {removed_negative} rows with negative/zero prices")
            
            # Handle duplicates (keep last occurrence)
            if 'symbol' in df.columns and 'date' in df.columns:
                df = df.unique(subset=['symbol', 'date'], keep='last', maintain_order=True)
                removed_duplicates = original_len - len(df)
                if removed_duplicates > 0:
                    print(f"Removed {removed_duplicates} duplicate rows")
            
            # Forward fill missing prices
            if 'price' in df.columns:
                df = df.with_columns([
                    pl.col("price").forward_fill().over("symbol")
                ])
                print("Forward filled missing prices")
            
            # Cap extreme outliers
            if 'price' in df.columns:
                q1 = df['price'].quantile(0.01)
                q99 = df['price'].quantile(0.99)
                
                df = df.with_columns([
                    pl.col("price").clip(q1, q99)
                ])
                print(f"Capped extreme prices to [{q1:.2f}, {q99:.2f}] range")
            
            print(f"Final dataset: {len(df):,} rows")
            return df
        
        return validate_financial_data, handle_data_quality_issues
    
    def error_recovery_patterns():
        """Error recovery and fallback patterns"""
        
        class RobustDataProcessor:
            def __init__(self):
                self.error_log = []
                self.recovery_strategies = {
                    'memory_error': self._handle_memory_error,
                    'schema_error': self._handle_schema_error,
                    'data_error': self._handle_data_error,
                    'file_error': self._handle_file_error
                }
            
            def process_with_recovery(self, data_path: str, processing_func: callable):
                """Process data with automatic error recovery"""
                
                try:
                    # Try primary processing
                    return self._primary_processing(data_path, processing_func)
                    
                except MemoryError as e:
                    return self._handle_memory_error(data_path, processing_func, e)
                    
                except pl.SchemaError as e:
                    return self._handle_schema_error(data_path, processing_func, e)
                    
                except Exception as e:
                    return self._handle_general_error(data_path, processing_func, e)
            
            def _primary_processing(self, data_path: str, processing_func: callable):
                """Primary processing approach"""
                df = pl.scan_parquet(data_path)
                result = processing_func(df).collect()
                return result
            
            def _handle_memory_error(self, data_path: str, processing_func: callable, error: Exception):
                """Handle memory errors with streaming"""
                print(f"Memory error encountered: {error}")
                print("Attempting streaming recovery...")
                
                try:
                    df = pl.scan_parquet(data_path)
                    result = processing_func(df).collect(streaming=True)
                    print("✅ Streaming recovery successful")
                    return result
                except Exception as streaming_error:
                    print(f"Streaming recovery failed: {streaming_error}")
                    return self._handle_chunked_processing(data_path, processing_func)
            
            def _handle_chunked_processing(self, data_path: str, processing_func: callable):
                """Fallback to chunked processing"""
                print("Attempting chunked processing...")
                
                try:
                    # Process in smaller chunks
                    chunk_size = 100000
                    results = []
                    
                    total_rows = pl.scan_parquet(data_path).select(pl.len()).collect().item()
                    
                    for start_idx in range(0, total_rows, chunk_size):
                        chunk = (
                            pl.scan_parquet(data_path)
                            .slice(start_idx, chunk_size)
                        )
                        
                        chunk_result = processing_func(chunk).collect()
                        results.append(chunk_result)
                    
                    final_result = pl.concat(results)
                    print("✅ Chunked processing successful")
                    return final_result
                    
                except Exception as chunk_error:
                    print(f"❌ All recovery strategies failed: {chunk_error}")
                    self.error_log.append({
                        'error_type': 'unrecoverable',
                        'original_error': str(error),
                        'recovery_error': str(chunk_error),
                        'data_path': data_path
                    })
                    raise
            
            def _handle_schema_error(self, data_path: str, processing_func: callable, error: Exception):
                """Handle schema-related errors"""
                print(f"Schema error: {error}")
                print("Attempting schema inference recovery...")
                
                try:
                    # Read with flexible schema
                    df = pl.read_parquet(data_path, use_pyarrow=True)
                    result = processing_func(df.lazy()).collect()
                    print("✅ Schema recovery successful")
                    return result
                except Exception as recovery_error:
                    print(f"❌ Schema recovery failed: {recovery_error}")
                    raise
            
            def get_error_summary(self):
                """Get summary of all errors encountered"""
                return {
                    'total_errors': len(self.error_log),
                    'error_types': [e['error_type'] for e in self.error_log],
                    'details': self.error_log
                }
        
        return RobustDataProcessor
    
    return robust_file_operations, data_validation_framework, error_recovery_patterns
```

---

## Advanced Query Optimization

### Query Optimization Strategies

```python
def advanced_query_optimization():
    """Advanced query optimization techniques"""
    
    def predicate_pushdown_optimizer():
        """Optimize queries using predicate pushdown"""
        
        def optimize_filter_placement(query: pl.LazyFrame) -> pl.LazyFrame:
            """Optimize filter placement for maximum efficiency"""
            
            # Example: Move most selective filters to the beginning
            # This is conceptual - actual implementation would analyze filter selectivity
            
            return (
                query
                # Most selective filters first (fewer distinct values)
                .filter(pl.col("symbol") == "AAPL")          # Single value - most selective
                .filter(pl.col("exchange") == "NASDAQ")      # Few values - very selective
                .filter(pl.col("date") >= "2020-01-01")     # Time range - moderately selective  
                .filter(pl.col("volume") > 1000000)         # Numeric range - less selective
                .filter(pl.col("price") > 0)                # Almost all data - least selective
            )
        
        def column_pruning_optimizer(query: pl.LazyFrame, required_columns: list) -> pl.LazyFrame:
            """Optimize column selection to reduce memory usage"""
            
            return (
                query
                .select(required_columns)  # Select only needed columns early
                # ... rest of query operations
            )
        
        return optimize_filter_placement, column_pruning_optimizer
    
    def join_optimization_strategies():
        """Optimize join operations"""
        
        def optimize_join_order(left_df: pl.LazyFrame, right_df: pl.LazyFrame) -> pl.LazyFrame:
            """Optimize join order and strategy"""
            
            # Strategy: Filter both sides before joining
            optimized_join = (
                left_df
                .filter(pl.col("date") >= "2020-01-01")     # Reduce left table size
                .select(["symbol", "date", "price", "volume"])  # Only needed columns
                .join(
                    right_df
                    .filter(pl.col("market_cap") > 1e9)     # Reduce right table size
                    .select(["symbol", "date", "market_cap", "sector"]),  # Only needed columns
                    on=["symbol", "date"],
                    how="inner"  # Use inner join if possible (most efficient)
                )
            )
            
            return optimized_join
        
        def broadcast_join_optimization(small_df: pl.LazyFrame, large_df: pl.LazyFrame) -> pl.LazyFrame:
            """Optimize joins when one table is much smaller"""
            
            # For small lookup tables, collect to memory for broadcast-style join
            if small_df.select(pl.len()).collect().item() < 10000:  # Small table
                small_collected = small_df.collect()
                
                return (
                    large_df
                    .join(small_collected.lazy(), on="symbol", how="left")
                )
            else:
                # Regular join for large tables
                return large_df.join(small_df, on="symbol", how="left")
        
        return optimize_join_order, broadcast_join_optimization
    
    def aggregation_optimization():
        """Optimize aggregation operations"""
        
        def batch_aggregations(df: pl.LazyFrame) -> pl.LazyFrame:
            """Batch multiple aggregations together"""
            
            # ✅ Good: Batch all aggregations in one operation
            return (
                df
                .group_by(["symbol", "date"])
                .agg([
                    pl.col("price").mean().alias("avg_price"),
                    pl.col("price").std().alias("price_vol"),
                    pl.col("volume").sum().alias("total_volume"),
                    pl.col("volume").mean().alias("avg_volume"),
                    pl.col("returns").mean().alias("avg_return"),
                    pl.col("returns").std().alias("return_vol")
                ])
            )
        
        def hierarchical_aggregation(df: pl.LazyFrame) -> pl.LazyFrame:
            """Use hierarchical aggregation for better performance"""
            
            # First aggregate by day, then by symbol
            daily_aggregates = (
                df
                .group_by("date")
                .agg([
                    pl.col("volume").sum().alias("daily_volume"),
                    pl.col("returns").mean().alias("market_return"),
                    pl.col("returns").std().alias("market_vol")
                ])
            )
            
            symbol_aggregates = (
                df
                .group_by("symbol")
                .agg([
                    pl.col("returns").mean().alias("avg_return"),
                    pl.col("returns").std().alias("volatility"),
                    pl.col("volume").mean().alias("avg_volume")
                ])
            )
            
            return daily_aggregates, symbol_aggregates
        
        return batch_aggregations, hierarchical_aggregation
    
    def window_function_optimization():
        """Optimize window functions"""
        
        def efficient_rolling_calculations(df: pl.LazyFrame) -> pl.LazyFrame:
            """Optimize rolling window calculations"""
            
            return (
                df
                .sort(["symbol", "date"])  # Essential for window functions
                # Batch window operations with same window size
                .with_columns([
                    # 20-day windows
                    pl.col("returns").rolling_mean(window_size=20).over("symbol").alias("ma_20"),
                    pl.col("returns").rolling_std(window_size=20).over("symbol").alias("vol_20"),
                    pl.col("volume").rolling_mean(window_size=20).over("symbol").alias("vol_ma_20"),
                ])
                .with_columns([
                    # 252-day windows  
                    pl.col("returns").rolling_mean(window_size=252).over("symbol").alias("ma_252"),
                    pl.col("returns").rolling_std(window_size=252).over("symbol").alias("vol_252"),
                    pl.col("volume").rolling_mean(window_size=252).over("symbol").alias("vol_ma_252"),
                ])
            )
        
        def optimize_multiple_windows(df: pl.LazyFrame, windows: list) -> pl.LazyFrame:
            """Efficiently calculate multiple window sizes"""
            
            # Group window operations by size for better performance
            result_df = df.sort(["symbol", "date"])
            
            for window_size in windows:
                window_exprs = [
                    pl.col("returns").rolling_mean(window_size=window_size).over("symbol").alias(f"ma_{window_size}"),
                    pl.col("returns").rolling_std(window_size=window_size).over("symbol").alias(f"vol_{window_size}"),
                    pl.col("returns").rolling_max(window_size=window_size).over("symbol").alias(f"max_{window_size}"),
                    pl.col("returns").rolling_min(window_size=window_size).over("symbol").alias(f"min_{window_size}"),
                ]
                
                result_df = result_df.with_columns(window_exprs)
            
            return result_df
        
        return efficient_rolling_calculations, optimize_multiple_windows
    
    return predicate_pushdown_optimizer, join_optimization_strategies, aggregation_optimization, window_function_optimization

def query_caching_strategies():
    """Implement intelligent query caching"""
    
    def smart_caching_system():
        """Implement smart caching for frequently used queries"""
        
        from functools import wraps
        import hashlib
        import pickle
        from pathlib import Path
        
        class QueryCache:
            def __init__(self, cache_dir: str = ".polars_cache"):
                self.cache_dir = Path(cache_dir)
                self.cache_dir.mkdir(exist_ok=True)
                self.hits = 0
                self.misses = 0
            
            def _query_hash(self, query: pl.LazyFrame) -> str:
                """Generate hash for query plan"""
                plan = query.explain(optimized=True)
                return hashlib.md5(plan.encode()).hexdigest()
            
            def cached_collect(self, query: pl.LazyFrame, ttl_hours: int = 24) -> pl.DataFrame:
                """Collect query with caching"""
                
                query_hash = self._query_hash(query)
                cache_file = self.cache_dir / f"{query_hash}.parquet"
                
                # Check if cached result exists and is fresh
                if cache_file.exists():
                    file_age = time.time() - cache_file.stat().st_mtime
                    if file_age < ttl_hours * 3600:
                        print(f"Cache hit for query {query_hash[:8]}")
                        self.hits += 1
                        return pl.read_parquet(cache_file)
                
                # Execute query and cache result
                print(f"Cache miss for query {query_hash[:8]}")
                self.misses += 1
                result = query.collect()
                result.write_parquet(cache_file)
                
                return result
            
            def clear_cache(self):
                """Clear all cached results"""
                for cache_file in self.cache_dir.glob("*.parquet"):
                    cache_file.unlink()
                print("Cache cleared")
            
            def cache_stats(self):
                """Get cache statistics"""
                total_requests = self.hits + self.misses
                hit_rate = self.hits / total_requests if total_requests > 0 else 0
                
                return {
                    'hits': self.hits,
                    'misses': self.misses,
                    'hit_rate': hit_rate,
                    'cache_size': len(list(self.cache_dir.glob("*.parquet")))
                }
        
        return QueryCache
    
    def materialized_view_pattern():
        """Implement materialized view pattern for expensive calculations"""
        
        def create_materialized_view(source_query: pl.LazyFrame, view_path: str, 
                                   refresh_condition: callable = None):
            """Create materialized view that refreshes conditionally"""
            
            view_file = Path(view_path)
            
            # Check if refresh is needed
            needs_refresh = True
            if view_file.exists() and refresh_condition:
                needs_refresh = refresh_condition(view_file)
            elif view_file.exists():
                # Default: refresh if older than 1 hour
                file_age = time.time() - view_file.stat().st_mtime
                needs_refresh = file_age > 3600
            
            if needs_refresh:
                print(f"Refreshing materialized view: {view_path}")
                result = source_query.collect()
                result.write_parquet(view_path)
                print(f"Materialized view updated: {len(result):,} rows")
                return result
            else:
                print(f"Using existing materialized view: {view_path}")
                return pl.read_parquet(view_path)
        
        def incremental_materialized_view(source_query: pl.LazyFrame, view_path: str,
                                        date_column: str = "date"):
            """Create incrementally updated materialized view"""
            
            view_file = Path(view_path)
            
            if view_file.exists():
                # Load existing view
                existing_view = pl.read_parquet(view_path)
                last_date = existing_view[date_column].max()
                
                # Get new data since last update
                new_data = source_query.filter(pl.col(date_column) > last_date).collect()
                
                if len(new_data) > 0:
                    print(f"Incrementally updating view with {len(new_data):,} new rows")
                    updated_view = pl.concat([existing_view, new_data])
                    updated_view.write_parquet(view_path)
                    return updated_view
                else:
                    print("No new data to update materialized view")
                    return existing_view
            else:
                # Create initial view
                print("Creating initial materialized view")
                result = source_query.collect()
                result.write_parquet(view_path)
                return result
        
        return create_materialized_view, incremental_materialized_view
    
    return smart_caching_system, materialized_view_pattern
```

---

## Data Validation & Quality Control

### Comprehensive Data Validation

```python
def comprehensive_data_validation():
    """Comprehensive data validation and quality control"""
    
    def statistical_validation():
        """Statistical validation and outlier detection"""
        
        def detect_statistical_outliers(df: pl.DataFrame, columns: list, method: str = "iqr"):
            """Detect statistical outliers using various methods"""
            
            outlier_results = {}
            
            for col in columns:
                if col not in df.columns:
                    continue
                
                if method == "iqr":
                    # Interquartile Range method
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    outliers = df.filter(
                        (pl.col(col) < lower_bound) | (pl.col(col) > upper_bound)
                    )
                    
                elif method == "zscore":
                    # Z-score method
                    mean = df[col].mean()
                    std = df[col].std()
                    
                    outliers = df.filter(
                        ((pl.col(col) - mean) / std).abs() > 3
                    )
                
                elif method == "modified_zscore":
                    # Modified Z-score using median
                    median = df[col].median()
                    mad = (df[col] - median).abs().median()
                    
                    outliers = df.filter(
                        (0.6745 * (pl.col(col) - median) / mad).abs() > 3.5
                    )
                
                outlier_results[col] = {
                    'count': len(outliers),
                    'percentage': len(outliers) / len(df) * 100,
                    'outliers': outliers
                }
                
                print(f"{col}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.2f}%)")
            
            return outlier_results
        
        def validate_data_distributions(df: pl.DataFrame, expected_distributions: dict):
            """Validate that data follows expected distributions"""
            
            validation_results = {}
            
            for col, expected_dist in expected_distributions.items():
                if col not in df.columns:
                    continue
                
                col_data = df[col].drop_nulls()
                
                if expected_dist['type'] == 'normal':
                    # Check if data is approximately normal
                    skewness = col_data.skewness()
                    kurtosis = col_data.kurtosis()
                    
                    is_normal = abs(skewness) < 2 and abs(kurtosis - 3) < 7
                    
                    validation_results[col] = {
                        'type': 'normal',
                        'is_valid': is_normal,
                        'skewness': skewness,
                        'kurtosis': kurtosis,
                        'message': f"Skewness: {skewness:.2f}, Kurtosis: {kurtosis:.2f}"
                    }
                
                elif expected_dist['type'] == 'range':
                    # Check if data is within expected range
                    min_val = col_data.min()
                    max_val = col_data.max()
                    expected_min = expected_dist['min']
                    expected_max = expected_dist['max']
                    
                    is_valid = min_val >= expected_min and max_val <= expected_max
                    
                    validation_results[col] = {
                        'type': 'range',
                        'is_valid': is_valid,
                        'actual_range': (min_val, max_val),
                        'expected_range': (expected_min, expected_max),
                        'message': f"Range: [{min_val:.2f}, {max_val:.2f}], Expected: [{expected_min}, {expected_max}]"
                    }
            
            return validation_results
        
        return detect_statistical_outliers, validate_data_distributions
    
    def business_rule_validation():
        """Validate business rules and logical constraints"""
        
        def validate_financial_rules(df: pl.DataFrame):
            """Validate financial data business rules"""
            
            rule_violations = []
            
            # Rule 1: Price should be positive
            if 'price' in df.columns:
                negative_prices = df.filter(pl.col('price') <= 0).height
                if negative_prices > 0:
                    rule_violations.append(f"Rule violation: {negative_prices} non-positive prices")
            
            # Rule 2: Volume should be non-negative
            if 'volume' in df.columns:
                negative_volume = df.filter(pl.col('volume') < 0).height
                if negative_volume > 0:
                    rule_violations.append(f"Rule violation: {negative_volume} negative volumes")
            
            # Rule 3: Market cap should equal shares * price (if all columns present)
            if all(col in df.columns for col in ['market_cap', 'shares_outstanding', 'price']):
                calculated_mcap = df.with_columns([
                    (pl.col('shares_outstanding') * pl.col('price')).alias('calculated_mcap')
                ])
                
                mcap_mismatches = calculated_mcap.filter(
                    (pl.col('market_cap') - pl.col('calculated_mcap')).abs() > 
                    pl.col('market_cap') * 0.01  # 1% tolerance
                ).height
                
                if mcap_mismatches > 0:
                    rule_violations.append(f"Rule violation: {mcap_mismatches} market cap mismatches")
            
            # Rule 4: Returns should be reasonable (-50% to +100% daily)
            if 'returns' in df.columns:
                extreme_returns = df.filter(
                    (pl.col('returns') < -0.5) | (pl.col('returns') > 1.0)
                ).height
                
                if extreme_returns > 0:
                    rule_violations.append(f"Rule violation: {extreme_returns} extreme returns")
            
            # Rule 5: Dates should be business days (simplified check)
            if 'date' in df.columns:
                weekend_dates = df.filter(
                    pl.col('date').dt.weekday() > 5
                ).height
                
                if weekend_dates > 0:
                    rule_violations.append(f"Rule violation: {weekend_dates} weekend dates")
            
            return rule_violations
        
        def validate_cross_sectional_consistency(df: pl.DataFrame):
            """Validate consistency across cross-sections"""
            
            consistency_issues = []
            
            # Check for extreme values relative to peers
            if all(col in df.columns for col in ['symbol', 'date', 'price', 'market_cap']):
                
                # Calculate sector statistics
                sector_stats = (
                    df
                    .group_by(['date', 'sector'])
                    .agg([
                        pl.col('price').median().alias('median_price'),
                        pl.col('price').std().alias('price_std'),
                        pl.col('market_cap').median().alias('median_mcap')
                    ])
                )
                
                # Join back and check for outliers
                with_sector_stats = df.join(sector_stats, on=['date', 'sector'])
                
                extreme_vs_sector = with_sector_stats.filter(
                    ((pl.col('price') - pl.col('median_price')) / pl.col('price_std')).abs() > 5
                ).height
                
                if extreme_vs_sector > 0:
                    consistency_issues.append(f"Cross-sectional issue: {extreme_vs_sector} extreme prices vs sector")
            
            return consistency_issues
        
        return validate_financial_rules, validate_cross_sectional_consistency
    
    return statistical_validation, business_rule_validation

def data_quality_monitoring():
    """Ongoing data quality monitoring"""
    
    def quality_metrics_dashboard(df: pl.DataFrame):
        """Generate comprehensive data quality dashboard"""
        
        quality_metrics = {}
        
        # Completeness metrics
        total_rows = len(df)
        quality_metrics['completeness'] = {}
        
        for col in df.columns:
            null_count = df[col].null_count()
            completeness = (total_rows - null_count) / total_rows
            quality_metrics['completeness'][col] = {
                'completeness_rate': completeness,
                'null_count': null_count,
                'status': 'good' if completeness > 0.95 else 'warning' if completeness > 0.8 else 'poor'
            }
        
        # Uniqueness metrics (for key columns)
        key_columns = ['symbol', 'date']
        quality_metrics['uniqueness'] = {}
        
        for col in key_columns:
            if col in df.columns:
                unique_count = df[col].n_unique()
                uniqueness = unique_count / total_rows
                quality_metrics['uniqueness'][col] = {
                    'uniqueness_rate': uniqueness,
                    'unique_count': unique_count,
                    'duplicate_count': total_rows - unique_count
                }
        
        # Data freshness
        if 'date' in df.columns:
            latest_date = df['date'].max()
            import datetime
            days_old = (datetime.date.today() - latest_date).days
            
            quality_metrics['freshness'] = {
                'latest_date': latest_date,
                'days_old': days_old,
                'status': 'good' if days_old <= 1 else 'warning' if days_old <= 7 else 'poor'
            }
        
        # Data consistency
        quality_metrics['consistency'] = {}
        
        # Check for consistent data types within columns
        for col in df.columns:
            if df[col].dtype == pl.Utf8:
                # Check for mixed formats in string columns
                sample_values = df[col].drop_nulls().unique().head(100).to_list()
                # This is simplified - would implement format consistency checks
                quality_metrics['consistency'][col] = {
                    'sample_formats': sample_values[:5],
                    'unique_formats': len(sample_values)
                }
        
        return quality_metrics
    
    def automated_quality_alerts(quality_metrics: dict, thresholds: dict = None):
        """Generate automated quality alerts"""
        
        if thresholds is None:
            thresholds = {
                'completeness_threshold': 0.95,
                'freshness_threshold_days': 2,
                'extreme_outlier_threshold': 0.05
            }
        
        alerts = []
        
        # Completeness alerts
        for col, metrics in quality_metrics.get('completeness', {}).items():
            if metrics['completeness_rate'] < thresholds['completeness_threshold']:
                alerts.append({
                    'type': 'completeness',
                    'severity': 'high' if metrics['completeness_rate'] < 0.8 else 'medium',
                    'message': f"Column '{col}' has low completeness: {metrics['completeness_rate']:.2%}",
                    'metric': metrics
                })
        
        # Freshness alerts
        freshness = quality_metrics.get('freshness', {})
        if freshness.get('days_old', 0) > thresholds['freshness_threshold_days']:
            alerts.append({
                'type': 'freshness',
                'severity': 'high' if freshness['days_old'] > 7 else 'medium',
                'message': f"Data is {freshness['days_old']} days old",
                'metric': freshness
            })
        
        return alerts
    
    return quality_metrics_dashboard, automated_quality_alerts
```

---

## Navigation

### Related Guides

- **[Main README](README.md)** - Overview and quick start guide
- **[Quantitative Research](polars_quantitative_research.md)** - Time series analysis, factor modeling, and risk management
- **[Streaming & Memory Management](polars_streaming_and_memory.md)** - Streaming configuration and memory optimization
- **[Polars vs Pandas Integration](polars_pandas_integration.md)** - Integration strategies and conversion patterns
- **[Performance & Large Datasets](polars_performance_and_large_datasets.md)** - Performance optimization and large-scale processing
- **[Practical Implementation](polars_practical_implementation.md)** - I/O best practices, data cleaning, and troubleshooting

### Key Takeaways

1. **Complex Analytics**: Use multi-factor analysis, regime detection, and advanced time series patterns for sophisticated quantitative research
2. **Risk Modeling**: Implement comprehensive risk metrics including VaR, beta calculations, and stress testing scenarios
3. **Backtesting**: Build realistic backtesting frameworks with transaction costs, market impact, and walk-forward validation
4. **Performance Debugging**: Use systematic approaches to identify and resolve performance bottlenecks
5. **Error Handling**: Implement robust error handling with fallback strategies and automatic recovery
6. **Query Optimization**: Apply advanced optimization techniques including predicate pushdown, join optimization, and intelligent caching
7. **Data Quality**: Maintain high data quality with statistical validation, business rule enforcement, and automated monitoring

This guide provides advanced techniques for building production-ready quantitative research systems using Polars' powerful capabilities while maintaining reliability and performance.