# Polars Streaming & Memory Management

Comprehensive guide to streaming operations, memory optimization, and resource control for processing datasets larger than RAM using Polars.

## Table of Contents

1. [Understanding Streaming in Polars](#understanding-streaming-in-polars)
2. [Streaming Configuration & Tuning](#streaming-configuration--tuning)
3. [Memory Management Strategies](#memory-management-strategies)
4. [Chunk Processing Techniques](#chunk-processing-techniques)
5. [Resource Monitoring & Control](#resource-monitoring--control)
6. [Streaming Best Practices & Limitations](#streaming-best-practices--limitations)
7. [Production Memory Management](#production-memory-management)
8. [Navigation](#navigation)

---

## Understanding Streaming in Polars

Streaming allows processing datasets larger than available RAM by processing data in chunks. This is crucial for handling hundreds of GB datasets commonly found in quantitative research.

### Streaming Fundamentals

```python
import polars as pl
import numpy as np
import os
from pathlib import Path

def streaming_fundamentals():
    """Core streaming concepts and setup"""
    
    # Enable streaming for large datasets
    streaming_query = (
        pl.scan_csv("huge_dataset.csv")
        .filter(pl.col("date") >= "2020-01-01")
        .group_by("symbol")
        .agg([
            pl.col("volume").sum(),
            pl.col("price").mean()
        ])
        .collect(streaming=True)  # Enable streaming execution
    )
    
    # Streaming with custom chunk size
    pl.Config.set_streaming_chunk_size(100000)  # 100k rows per chunk
    
    # Check if query supports streaming
    lazy_query = pl.scan_parquet("large_file.parquet").group_by("symbol").agg(pl.col("price").mean())
    
    try:
        result = lazy_query.collect(streaming=True)
        print("Streaming successful")
    except Exception as e:
        print(f"Streaming not supported: {e}")
        result = lazy_query.collect()  # Fallback to eager
        
    return result

def advanced_streaming_patterns():
    """Advanced streaming for quantitative research workflows"""
    
    # Pattern 1: Streaming aggregations across multiple files
    def stream_multi_file_aggregation():
        return (
            pl.scan_parquet("data/year_*.parquet")  # Multiple files
            .filter(pl.col("market_cap") > 1e9)     # Large cap only
            .group_by(["sector", "date"])
            .agg([
                pl.col("returns").mean().alias("sector_return"),
                pl.col("market_cap").sum().alias("sector_mcap"),
                pl.col("symbol").count().alias("stock_count")
            ])
            .collect(streaming=True)
        )
    
    # Pattern 2: Streaming window operations
    def stream_rolling_calculations():
        return (
            pl.scan_parquet("prices/*.parquet")
            .sort(["symbol", "date"])  # Critical for window functions
            .with_columns([
                pl.col("returns").rolling_mean(window_size=252).over("symbol").alias("annual_ma"),
                pl.col("returns").rolling_std(window_size=252).over("symbol").alias("annual_vol"),
                pl.col("volume").rolling_sum(window_size=20).over("symbol").alias("volume_20d")
            ])
            .collect(streaming=True)
        )
    
    # Pattern 3: Streaming joins with large datasets
    def stream_large_joins():
        fundamentals = pl.scan_parquet("fundamentals/*.parquet")
        prices = pl.scan_parquet("prices/*.parquet")
        
        return (
            fundamentals
            .join(prices, on=["symbol", "date"], how="inner")
            .with_columns([
                (pl.col("market_cap") / pl.col("book_value")).alias("market_to_book"),
                (pl.col("price") / pl.col("earnings")).alias("pe_ratio")
            ])
            .filter(pl.col("pe_ratio").is_between(0, 100))
            .collect(streaming=True)
        )
    
    return stream_multi_file_aggregation, stream_rolling_calculations, stream_large_joins
```

### Streaming vs. Eager Execution Comparison

```python
def streaming_vs_eager_comparison():
    """Compare streaming vs eager execution for different scenarios"""
    
    import time
    import psutil
    
    def benchmark_execution_modes(query: pl.LazyFrame, description: str):
        """Benchmark both streaming and eager execution"""
        
        def get_memory_usage():
            return psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        results = {}
        
        # Test streaming execution
        try:
            start_time = time.time()
            start_memory = get_memory_usage()
            
            streaming_result = query.collect(streaming=True)
            
            streaming_time = time.time() - start_time
            streaming_memory = get_memory_usage() - start_memory
            
            results['streaming'] = {
                'time': streaming_time,
                'memory_delta': streaming_memory,
                'success': True,
                'rows': len(streaming_result)
            }
            
        except Exception as e:
            results['streaming'] = {
                'time': None,
                'memory_delta': None,
                'success': False,
                'error': str(e)
            }
        
        # Test eager execution
        try:
            start_time = time.time()
            start_memory = get_memory_usage()
            
            eager_result = query.collect()
            
            eager_time = time.time() - start_time
            eager_memory = get_memory_usage() - start_memory
            
            results['eager'] = {
                'time': eager_time,
                'memory_delta': eager_memory,
                'success': True,
                'rows': len(eager_result)
            }
            
        except Exception as e:
            results['eager'] = {
                'time': None,
                'memory_delta': None,
                'success': False,
                'error': str(e)
            }
        
        # Print comparison
        print(f"\n{description}:")
        for mode, stats in results.items():
            if stats['success']:
                print(f"  {mode:>9}: {stats['time']:.2f}s, {stats['memory_delta']:+.1f}MB memory, {stats['rows']:,} rows")
            else:
                print(f"  {mode:>9}: Failed - {stats['error']}")
        
        return results
    
    # Example benchmarks
    large_aggregation = pl.scan_parquet("large_dataset.parquet").group_by("symbol").agg([
        pl.col("returns").mean(),
        pl.col("volume").sum(),
        pl.col("returns").std()
    ])
    
    large_filter = pl.scan_parquet("large_dataset.parquet").filter(
        pl.col("volume") > 1000000
    ).filter(
        pl.col("date") >= "2020-01-01"
    )
    
    complex_calculation = pl.scan_parquet("large_dataset.parquet").with_columns([
        pl.col("returns").rolling_mean(window_size=252).over("symbol").alias("rolling_mean"),
        pl.col("returns").rolling_std(window_size=252).over("symbol").alias("rolling_std")
    ])
    
    benchmark_execution_modes(large_aggregation, "Large Aggregation")
    benchmark_execution_modes(large_filter, "Large Filter")
    benchmark_execution_modes(complex_calculation, "Complex Rolling Calculation")
```

---

## Streaming Configuration & Tuning

### Performance Optimization

```python
def optimize_streaming_performance():
    """Configure streaming for optimal performance on your system"""
    
    # Memory-based configuration
    available_ram_gb = 64  # Adjust based on your system specs
    chunk_size = min(1000000, available_ram_gb * 10000)  # Scale with RAM
    
    pl.Config.set_streaming_chunk_size(chunk_size)
    
    # Thread configuration for streaming
    max_threads = os.cpu_count()
    
    print(f"Streaming optimized for {available_ram_gb}GB RAM")
    print(f"Chunk size: {chunk_size:,} rows")
    print(f"Using {max_threads} threads")
    
    return chunk_size

def streaming_memory_monitoring():
    """Monitor memory usage during streaming operations"""
    import psutil
    import time
    
    def get_memory_usage():
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB
    
    print(f"Memory before: {get_memory_usage():.1f} MB")
    
    # Large streaming operation
    result = (
        pl.scan_csv("large_file.csv")
        .filter(pl.col("date") >= "2020-01-01")
        .group_by("symbol")
        .agg([
            pl.col("volume").sum(),
            pl.col("price").mean(),
            pl.col("returns").std()
        ])
        .collect(streaming=True)
    )
    
    print(f"Memory after: {get_memory_usage():.1f} MB")
    return result

def streaming_with_progress():
    """Stream large datasets with progress indication"""
    from tqdm import tqdm
    
    # For very large operations, process in batches
    def process_large_dataset_in_batches(file_pattern: str, batch_size: int = 10):
        files = list(Path().glob(file_pattern))
        results = []
        
        for i in tqdm(range(0, len(files), batch_size), desc="Processing batches"):
            batch_files = files[i:i+batch_size]
            
            batch_result = (
                pl.scan_parquet(batch_files)
                .filter(pl.col("volume") > 1000000)
                .group_by("symbol")
                .agg([
                    pl.col("price").mean().alias("avg_price"),
                    pl.col("volume").sum().alias("total_volume")
                ])
                .collect(streaming=True)
            )
            
            results.append(batch_result)
        
        # Combine all batch results
        return pl.concat(results).group_by("symbol").agg([
            pl.col("avg_price").mean(),  # Average of averages (could weight by volume)
            pl.col("total_volume").sum()
        ])
    
    return process_large_dataset_in_batches

def adaptive_streaming_configuration():
    """Automatically configure streaming based on system resources"""
    
    import psutil
    
    def auto_configure_streaming():
        """Auto-configure streaming based on available system resources"""
        
        # Get system information
        memory_info = psutil.virtual_memory()
        cpu_count = psutil.cpu_count()
        available_gb = memory_info.available / (1024**3)
        
        # Conservative chunk sizing based on available memory
        if available_gb > 32:
            chunk_size = 200000
            max_threads = min(cpu_count, 16)
        elif available_gb > 16:
            chunk_size = 100000
            max_threads = min(cpu_count, 8)
        elif available_gb > 8:
            chunk_size = 50000
            max_threads = min(cpu_count, 4)
        else:
            chunk_size = 25000
            max_threads = min(cpu_count, 2)
        
        # Apply configuration
        pl.Config.set_streaming_chunk_size(chunk_size)
        os.environ['POLARS_MAX_THREADS'] = str(max_threads)
        
        print(f"Auto-configured streaming:")
        print(f"  Available memory: {available_gb:.1f} GB")
        print(f"  Chunk size: {chunk_size:,} rows")
        print(f"  Max threads: {max_threads}")
        
        return {
            'chunk_size': chunk_size,
            'max_threads': max_threads,
            'available_memory_gb': available_gb
        }
    
    def environment_specific_config(environment: str):
        """Configure for specific environments"""
        
        configs = {
            'development': {
                'chunk_size': 50000,
                'max_threads': min(8, os.cpu_count()),
                'streaming_pool_size': '1GB'
            },
            'production': {
                'chunk_size': 200000,
                'max_threads': os.cpu_count(),
                'streaming_pool_size': '4GB'
            },
            'testing': {
                'chunk_size': 10000,
                'max_threads': 2,
                'streaming_pool_size': '256MB'
            }
        }
        
        config = configs.get(environment, configs['development'])
        
        pl.Config.set_streaming_chunk_size(config['chunk_size'])
        os.environ['POLARS_MAX_THREADS'] = str(config['max_threads'])
        os.environ['POLARS_STREAMING_POOL_SIZE'] = config['streaming_pool_size']
        
        print(f"Configured for {environment} environment: {config}")
        return config
    
    return auto_configure_streaming, environment_specific_config
```

---

## Memory Management Strategies

### Advanced Memory Management

```python
def memory_management_strategies():
    """Advanced memory management for large-scale quantitative analysis"""
    
    def monitor_memory_usage():
        """Monitor memory usage during Polars operations"""
        
        import psutil
        import gc
        
        def get_memory_info():
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'rss_mb': memory_info.rss / 1024 / 1024,      # Physical memory
                'vms_mb': memory_info.vms / 1024 / 1024,      # Virtual memory
                'percent': process.memory_percent(),           # % of system memory
                'available_mb': psutil.virtual_memory().available / 1024 / 1024
            }
        
        # Memory monitoring context manager
        from contextlib import contextmanager
        
        @contextmanager
        def memory_monitor(operation_name: str):
            gc.collect()  # Clean up before starting
            start_memory = get_memory_info()
            print(f"[{operation_name}] Starting - Memory: {start_memory['rss_mb']:.1f} MB")
            
            try:
                yield start_memory
            finally:
                end_memory = get_memory_info()
                memory_delta = end_memory['rss_mb'] - start_memory['rss_mb']
                print(f"[{operation_name}] Completed - Memory: {end_memory['rss_mb']:.1f} MB (Δ{memory_delta:+.1f} MB)")
        
        return memory_monitor
    
    def optimize_data_types_for_memory(df: pl.LazyFrame) -> pl.LazyFrame:
        """Automatically optimize data types to reduce memory usage"""
        
        return (
            df
            .with_columns([
                # Downcast integers where possible
                pl.col("volume").cast(pl.UInt32),              # Most volumes fit in 32-bit
                pl.col("shares_outstanding").cast(pl.UInt64),   # Keep 64-bit for large numbers
                
                # Use float32 for metrics that don't need high precision
                pl.col("returns").cast(pl.Float32),
                pl.col("volatility").cast(pl.Float32), 
                pl.col("beta").cast(pl.Float32),
                
                # Keep float64 for prices (precision matters)
                pl.col("price").cast(pl.Float64),
                pl.col("market_cap").cast(pl.Float64),
                
                # Use categorical for repeated strings
                pl.col("symbol").cast(pl.Categorical),
                pl.col("sector").cast(pl.Categorical),
                pl.col("exchange").cast(pl.Categorical),
                
                # Boolean for flags
                pl.col("is_active").cast(pl.Boolean),
                pl.col("is_index_member").cast(pl.Boolean)
            ])
        )
    
    def memory_efficient_operations():
        """Patterns for memory-efficient operations"""
        
        # ✅ Good: Process data in streaming fashion
        def streaming_calculation(file_pattern: str):
            return (
                pl.scan_parquet(file_pattern)
                .filter(pl.col("date") >= "2020-01-01")
                .select(["symbol", "date", "close", "volume"])  # Only needed columns
                .with_columns([
                    pl.col("close").pct_change().over("symbol").alias("returns")
                ])
                .group_by("symbol")
                .agg([
                    pl.col("returns").mean().alias("avg_return"),
                    pl.col("returns").std().alias("volatility")
                ])
                .collect(streaming=True)  # Stream processing
            )
        
        # ✅ Good: Use sink operations for large outputs
        def efficient_data_export(df: pl.LazyFrame, output_path: str):
            (
                df
                .sink_parquet(
                    output_path,
                    compression="zstd",      # Good compression ratio
                    row_group_size=100000,   # Reasonable chunk size
                    statistics=True          # For query optimization later
                )
            )
        
        # ❌ Bad: Loading everything into memory
        def memory_intensive_approach():
            # Don't do this with large datasets
            large_df = pl.read_parquet("huge_file.parquet")  # Loads everything
            processed = large_df.with_columns([])            # More memory usage
            result = processed.collect()                     # Even more memory
            return result
        
        return streaming_calculation, efficient_data_export
    
    return monitor_memory_usage, optimize_data_types_for_memory, memory_efficient_operations

def data_type_optimization_guide():
    """Comprehensive guide to data type optimization for memory efficiency"""
    
    def analyze_memory_usage(df: pl.LazyFrame) -> dict:
        """Analyze memory usage by column and data type"""
        
        # Collect small sample to analyze schema
        sample = df.head(1000).collect()
        
        memory_analysis = {}
        total_estimated_memory = 0
        
        for col_name, dtype in sample.schema.items():
            col_data = sample[col_name]
            
            # Estimate memory per value
            if dtype == pl.Float64:
                bytes_per_value = 8
            elif dtype == pl.Float32:
                bytes_per_value = 4
            elif dtype == pl.Int64:
                bytes_per_value = 8
            elif dtype == pl.Int32:
                bytes_per_value = 4
            elif dtype == pl.UInt32:
                bytes_per_value = 4
            elif dtype == pl.Boolean:
                bytes_per_value = 1
            elif dtype == pl.Utf8:
                # Estimate based on average string length
                avg_string_length = col_data.str.n_chars().mean() if col_data.str.n_chars().mean() else 10
                bytes_per_value = avg_string_length * 1.5  # UTF-8 overhead
            elif dtype == pl.Categorical:
                # Categorical uses indices + dictionary
                unique_values = col_data.n_unique()
                bytes_per_value = 4 + (unique_values * 10)  # Rough estimate
            else:
                bytes_per_value = 8  # Default estimate
            
            memory_analysis[col_name] = {
                'dtype': str(dtype),
                'bytes_per_value': bytes_per_value,
                'sample_nulls': col_data.null_count(),
                'sample_unique': col_data.n_unique()
            }
            
            total_estimated_memory += bytes_per_value
        
        return {
            'columns': memory_analysis,
            'total_bytes_per_row': total_estimated_memory,
            'estimated_memory_mb_per_million_rows': total_estimated_memory * 1e6 / (1024**2)
        }
    
    def suggest_optimizations(memory_analysis: dict, row_count_estimate: int) -> dict:
        """Suggest memory optimizations based on data analysis"""
        
        suggestions = {}
        potential_savings = 0
        
        for col_name, stats in memory_analysis['columns'].items():
            dtype = stats['dtype']
            current_bytes = stats['bytes_per_value']
            
            # Suggest optimizations
            if dtype == 'Float64':
                if stats['sample_unique'] < 1000:  # Few unique values
                    suggestions[col_name] = {
                        'current': 'Float64',
                        'suggested': 'Float32 or Categorical',
                        'savings_per_row': 4,
                        'reason': 'Few unique values, consider Float32 or Categorical'
                    }
                    potential_savings += 4
                else:
                    suggestions[col_name] = {
                        'current': 'Float64',
                        'suggested': 'Float32',
                        'savings_per_row': 4,
                        'reason': 'Float32 precision often sufficient for financial data'
                    }
                    potential_savings += 4
            
            elif dtype == 'Int64':
                suggestions[col_name] = {
                    'current': 'Int64',
                    'suggested': 'Int32 or UInt32',
                    'savings_per_row': 4,
                    'reason': 'Most integer values fit in 32 bits'
                }
                potential_savings += 4
            
            elif dtype == 'Utf8' and stats['sample_unique'] < 1000:
                suggestions[col_name] = {
                    'current': 'Utf8',
                    'suggested': 'Categorical',
                    'savings_per_row': max(0, current_bytes - 6),
                    'reason': 'Few unique string values, categorical more efficient'
                }
                potential_savings += max(0, current_bytes - 6)
        
        total_savings_mb = (potential_savings * row_count_estimate) / (1024**2)
        
        return {
            'suggestions': suggestions,
            'potential_savings_bytes_per_row': potential_savings,
            'estimated_total_savings_mb': total_savings_mb
        }
    
    return analyze_memory_usage, suggest_optimizations
```

---

## Chunk Processing Techniques

### Advanced Chunking Strategies

```python
def chunk_processing_strategies():
    """Advanced chunking strategies for datasets larger than RAM"""
    
    def intelligent_chunking():
        """Intelligently determine optimal chunk sizes"""
        
        import psutil
        
        def calculate_optimal_chunk_size(file_path: str, target_memory_mb: int = 1024):
            """Calculate optimal chunk size based on file characteristics"""
            
            # Get file size and estimate row size
            file_size_mb = Path(file_path).stat().st_size / (1024**2)
            
            # Sample a small portion to estimate row size
            sample = pl.scan_parquet(file_path).head(1000).collect()
            estimated_bytes_per_row = sample.estimated_size() / len(sample)
            
            # Calculate chunk size to stay within memory target
            optimal_chunk_size = int((target_memory_mb * 1024 * 1024) / estimated_bytes_per_row)
            
            # Ensure reasonable bounds
            optimal_chunk_size = max(10000, min(optimal_chunk_size, 1000000))
            
            estimated_chunks = file_size_mb / (optimal_chunk_size * estimated_bytes_per_row / (1024**2))
            
            print(f"File analysis:")
            print(f"  File size: {file_size_mb:.1f} MB")
            print(f"  Estimated bytes per row: {estimated_bytes_per_row:.1f}")
            print(f"  Optimal chunk size: {optimal_chunk_size:,} rows")
            print(f"  Estimated chunks: {estimated_chunks:.0f}")
            
            return optimal_chunk_size
        
        return calculate_optimal_chunk_size
    
    def memory_aware_processing():
        """Process data with memory awareness and adaptive chunking"""
        
        def process_with_memory_monitoring(df: pl.LazyFrame, processing_func: callable):
            """Process DataFrame with continuous memory monitoring"""
            
            import psutil
            import gc
            
            def get_available_memory_gb():
                return psutil.virtual_memory().available / (1024**3)
            
            # Start with conservative chunk size
            current_chunk_size = 50000
            max_chunk_size = 500000
            min_chunk_size = 10000
            
            # Get total rows to process
            total_rows = df.select(pl.len()).collect().item()
            processed_rows = 0
            results = []
            
            print(f"Processing {total_rows:,} rows with adaptive chunking")
            
            while processed_rows < total_rows:
                # Check available memory
                available_memory = get_available_memory_gb()
                
                # Adjust chunk size based on available memory
                if available_memory > 8:
                    current_chunk_size = min(current_chunk_size * 1.2, max_chunk_size)
                elif available_memory < 4:
                    current_chunk_size = max(current_chunk_size * 0.8, min_chunk_size)
                
                current_chunk_size = int(current_chunk_size)
                
                # Process chunk
                print(f"Processing chunk {processed_rows:,}-{min(processed_rows + current_chunk_size, total_rows):,} "
                      f"(chunk size: {current_chunk_size:,}, available memory: {available_memory:.1f}GB)")
                
                chunk = (
                    df
                    .slice(processed_rows, current_chunk_size)
                    .pipe(processing_func)  # Apply processing function
                    .collect()
                )
                
                results.append(chunk)
                processed_rows += current_chunk_size
                
                # Force garbage collection after each chunk
                gc.collect()
            
            # Combine results
            print("Combining chunk results...")
            return pl.concat(results)
        
        return process_with_memory_monitoring
    
    def parallel_chunk_processing():
        """Process chunks in parallel with memory management"""
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import queue
        import threading
        
        def process_chunks_parallel(
            file_patterns: list, 
            processing_func: callable,
            max_workers: int = None,
            memory_limit_gb: int = 16
        ):
            """Process multiple file chunks in parallel with memory limits"""
            
            if max_workers is None:
                max_workers = min(4, os.cpu_count())  # Conservative default
            
            # Memory semaphore to limit concurrent memory usage
            memory_semaphore = threading.Semaphore(max_workers)
            
            def process_single_chunk(file_pattern):
                """Process a single file pattern"""
                try:
                    # Acquire memory semaphore
                    memory_semaphore.acquire()
                    
                    result = (
                        pl.scan_parquet(file_pattern)
                        .pipe(processing_func)
                        .collect(streaming=True)
                    )
                    
                    return result
                    
                except Exception as e:
                    print(f"Error processing {file_pattern}: {e}")
                    return None
                    
                finally:
                    # Release memory semaphore
                    memory_semaphore.release()
            
            # Process files in parallel
            results = []
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_pattern = {
                    executor.submit(process_single_chunk, pattern): pattern 
                    for pattern in file_patterns
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_pattern):
                    pattern = future_to_pattern[future]
                    try:
                        result = future.result()
                        if result is not None:
                            results.append(result)
                            print(f"Completed processing: {pattern}")
                    except Exception as e:
                        print(f"Error in {pattern}: {e}")
            
            # Combine all results
            if results:
                return pl.concat(results)
            else:
                return pl.DataFrame()
        
        return process_chunks_parallel
    
    return intelligent_chunking, memory_aware_processing, parallel_chunk_processing

def chunked_operations_patterns():
    """Common patterns for chunked operations in quantitative research"""
    
    def chunked_factor_calculation():
        """Calculate factors across large datasets using chunking"""
        
        def calculate_factors_chunked(data_path: str, chunk_size: int = 100000):
            """Calculate rolling factors in chunks to manage memory"""
            
            # Get total rows for progress tracking
            total_rows = pl.scan_parquet(data_path).select(pl.len()).collect().item()
            
            results = []
            processed_rows = 0
            
            print(f"Calculating factors for {total_rows:,} rows in chunks of {chunk_size:,}")
            
            while processed_rows < total_rows:
                print(f"Processing chunk {processed_rows // chunk_size + 1}")
                
                chunk_result = (
                    pl.scan_parquet(data_path)
                    .slice(processed_rows, chunk_size)
                    .sort(["symbol", "date"])
                    .with_columns([
                        # Factor calculations
                        pl.col("close").pct_change().over("symbol").alias("returns"),
                        pl.col("returns").rolling_mean(window_size=20).over("symbol").alias("momentum_20d"),
                        pl.col("returns").rolling_std(window_size=252).over("symbol").alias("volatility_1y"),
                        (pl.col("book_value") / pl.col("market_cap")).alias("book_to_market")
                    ])
                    .collect()
                )
                
                results.append(chunk_result)
                processed_rows += chunk_size
            
            # Combine and ensure continuity at chunk boundaries
            combined = pl.concat(results)
            
            # Recalculate rolling metrics to ensure continuity across chunks
            final_result = (
                combined
                .lazy()
                .sort(["symbol", "date"])
                .with_columns([
                    # Recalculate rolling metrics for continuity
                    pl.col("returns").rolling_mean(window_size=20).over("symbol").alias("momentum_20d_corrected"),
                    pl.col("returns").rolling_std(window_size=252).over("symbol").alias("volatility_1y_corrected"),
                ])
                .collect()
            )
            
            return final_result
        
        return calculate_factors_chunked
    
    def chunked_portfolio_construction():
        """Construct portfolios using chunked processing"""
        
        def construct_portfolios_chunked(
            factor_data_path: str, 
            rebalance_frequency: str = "monthly",
            chunk_by_date: bool = True
        ):
            """Construct portfolios by processing date chunks"""
            
            if chunk_by_date:
                # Get unique dates
                dates = (
                    pl.scan_parquet(factor_data_path)
                    .select("date")
                    .unique()
                    .sort("date")
                    .collect()
                )
                
                # Process each date separately for cross-sectional ranking
                portfolio_results = []
                
                for date_row in dates.iter_rows():
                    date_value = date_row[0]
                    
                    daily_portfolio = (
                        pl.scan_parquet(factor_data_path)
                        .filter(pl.col("date") == date_value)
                        .with_columns([
                            # Cross-sectional rankings
                            pl.col("momentum_20d").rank(descending=True).alias("momentum_rank"),
                            pl.col("book_to_market").rank(descending=True).alias("value_rank"),
                            pl.col("volatility_1y").rank().alias("vol_rank"),  # Low vol is better
                        ])
                        .with_columns([
                            # Quintile assignments
                            pl.col("momentum_rank").qcut(5).alias("momentum_quintile"),
                            pl.col("value_rank").qcut(5).alias("value_quintile"),
                        ])
                        .with_columns([
                            # Portfolio weights
                            (1.0 / pl.col("symbol").count().over(["momentum_quintile", "value_quintile"])).alias("weight")
                        ])
                        .collect()
                    )
                    
                    portfolio_results.append(daily_portfolio)
                
                return pl.concat(portfolio_results)
            
            else:
                # Use standard chunking approach
                return calculate_factors_chunked(factor_data_path)
        
        return construct_portfolios_chunked
    
    return chunked_factor_calculation, chunked_portfolio_construction
```

---

## Resource Monitoring & Control

### Performance Profiling

```python
def resource_monitoring():
    """Monitor and control resource usage"""
    
    def performance_profiler():
        """Profile Polars operations for performance optimization"""
        
        import time
        from contextlib import contextmanager
        
        @contextmanager
        def profile_operation(operation_name: str):
            """Profile execution time and resource usage"""
            
            import psutil
            
            # Start monitoring
            process = psutil.Process()
            start_time = time.time()
            start_cpu = process.cpu_percent()
            start_memory = process.memory_info().rss / 1024 / 1024
            
            try:
                yield
            finally:
                # End monitoring
                end_time = time.time()
                end_cpu = process.cpu_percent()
                end_memory = process.memory_info().rss / 1024 / 1024
                
                duration = end_time - start_time
                memory_delta = end_memory - start_memory
                
                print(f"[{operation_name}]")
                print(f"  Duration: {duration:.2f} seconds")
                print(f"  Memory: {end_memory:.1f} MB (Δ{memory_delta:+.1f} MB)")
                print(f"  CPU: {end_cpu:.1f}%")
        
        # Usage example
        def profile_query_execution():
            with profile_operation("Large Aggregation"):
                result = (
                    pl.scan_parquet("large_dataset.parquet")
                    .filter(pl.col("date") >= "2020-01-01")
                    .group_by("symbol")
                    .agg([
                        pl.col("returns").mean().alias("avg_return"),
                        pl.col("returns").std().alias("volatility")
                    ])
                    .collect(streaming=True)
                )
            
            return result
        
        return profile_operation, profile_query_execution
    
    def resource_limits():
        """Set resource limits for Polars operations"""
        
        import os
        import resource
        
        def set_memory_limits(max_memory_gb: int):
            """Set memory limits for the process"""
            
            try:
                # Set virtual memory limit
                max_memory_bytes = max_memory_gb * 1024**3
                resource.setrlimit(resource.RLIMIT_AS, (max_memory_bytes, max_memory_bytes))
                
                # Configure Polars for the memory limit
                os.environ['POLARS_MAX_MEMORY'] = str(max_memory_bytes)
                
                print(f"Set memory limit to {max_memory_gb} GB")
                
            except Exception as e:
                print(f"Could not set memory limit: {e}")
        
        def configure_for_environment(environment_type: str):
            """Configure Polars for different environments"""
            
            if environment_type == "development":
                # Development settings - moderate resources
                os.environ['POLARS_MAX_THREADS'] = str(min(8, os.cpu_count()))
                os.environ['POLARS_STREAMING_CHUNK_SIZE'] = '50000'
                set_memory_limits(16)  # 16 GB limit
                
            elif environment_type == "production":
                # Production settings - use more resources
                os.environ['POLARS_MAX_THREADS'] = str(os.cpu_count())
                os.environ['POLARS_STREAMING_CHUNK_SIZE'] = '100000'
                # No memory limits in production (or set very high)
                
            elif environment_type == "limited":
                # Limited resources (e.g., CI/CD, smaller machines)
                os.environ['POLARS_MAX_THREADS'] = '4'
                os.environ['POLARS_STREAMING_CHUNK_SIZE'] = '10000'
                set_memory_limits(4)  # 4 GB limit
            
            print(f"Configured for {environment_type} environment")
        
        return set_memory_limits, configure_for_environment
    
    return performance_profiler, resource_limits

def advanced_monitoring_tools():
    """Advanced monitoring and diagnostics tools"""
    
    def memory_leak_detection():
        """Detect memory leaks in long-running processes"""
        
        import psutil
        import time
        from collections import deque
        
        class MemoryLeakDetector:
            def __init__(self, window_size: int = 100):
                self.window_size = window_size
                self.memory_history = deque(maxlen=window_size)
                self.process = psutil.Process()
            
            def record_memory(self):
                """Record current memory usage"""
                current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
                self.memory_history.append((time.time(), current_memory))
                return current_memory
            
            def detect_leak(self, threshold_mb: float = 100):
                """Detect if memory usage is trending upward"""
                if len(self.memory_history) < self.window_size:
                    return False
                
                # Calculate linear trend
                memory_values = [mem for _, mem in self.memory_history]
                x = range(len(memory_values))
                
                # Simple linear regression
                n = len(memory_values)
                sum_x = sum(x)
                sum_y = sum(memory_values)
                sum_xy = sum(xi * yi for xi, yi in zip(x, memory_values))
                sum_x2 = sum(xi * xi for xi in x)
                
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
                
                # Convert slope to MB per operation
                leak_rate = slope
                
                if leak_rate > threshold_mb / self.window_size:
                    print(f"⚠️ Memory leak detected! Rate: {leak_rate:.2f} MB per operation")
                    return True
                
                return False
        
        return MemoryLeakDetector
    
    def query_performance_analyzer():
        """Analyze query performance and suggest optimizations"""
        
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
        
        return analyze_query_plan
    
    return memory_leak_detection, query_performance_analyzer
```

---

## Streaming Best Practices & Limitations

### Best Practices and Common Patterns

```python
def streaming_best_practices():
    """Best practices and patterns for streaming operations"""
    
    # ✅ Operations that work well with streaming
    good_streaming_ops = [
        "filter",           # Predicate pushdown
        "select", 
        "with_columns",     # Column transformations
        "group_by + agg",   # Aggregations
        "sort",             # External sort
        "join",             # Hash joins
        "unique",           # Deduplication
        "drop_nulls"        # Filtering nulls
    ]
    
    # ❌ Operations that don't support streaming well
    streaming_limitations = [
        "pivot",            # Requires full dataset in memory
        "melt",             # Complex reshaping
        "rolling operations with variable windows",
        "operations requiring global view",
        "complex analytical window functions"
    ]
    
    # Pattern: Hybrid approach for complex operations
    def hybrid_streaming_approach():
        """Use streaming for data preparation, eager for complex analytics"""
        
        # Stream the heavy data processing
        prepared_data = (
            pl.scan_parquet("huge_dataset/*.parquet")
            .filter(pl.col("date") >= "2020-01-01")
            .filter(pl.col("volume") > 1000000)
            .select([
                "symbol", "date", "price", "volume", 
                (pl.col("price") * pl.col("volume")).alias("notional")
            ])
            .collect(streaming=True)  # Streaming for reduction
        )
        
        # Use eager evaluation for complex analytics that need full dataset
        analytical_results = (
            prepared_data
            .lazy()
            .with_columns([
                pl.col("price").pct_change().over("symbol").alias("returns"),
                pl.col("notional").rank().over("date").alias("notional_rank")
            ])
            .collect()  # Eager evaluation for complex operations
        )
        
        return analytical_results
    
    return hybrid_streaming_approach

def robust_streaming_execution(query: pl.LazyFrame):
    """Execute queries with streaming fallback logic"""
    
    try:
        # Try streaming first
        print("Attempting streaming execution...")
        result = query.collect(streaming=True)
        print("✅ Streaming successful")
        return result, "streaming"
        
    except Exception as streaming_error:
        print(f"❌ Streaming failed: {streaming_error}")
        
        try:
            # Fallback to regular execution
            print("Falling back to eager execution...")
            result = query.collect()
            print("✅ Eager execution successful")
            return result, "eager"
            
        except Exception as eager_error:
            print(f"❌ Eager execution also failed: {eager_error}")
            
            # Final fallback: collect with smaller operations
            print("Attempting chunked execution...")
            return execute_in_chunks(query), "chunked"

def execute_in_chunks(query: pl.LazyFrame, chunk_size: int = 1000000):
    """Execute large queries in chunks when streaming fails"""
    # This is a simplified example - real implementation would be more complex
    print(f"Processing in chunks of {chunk_size:,} rows")
    return query.limit(chunk_size).collect()  # Simplified example

def streaming_best_practices_checklist():
    """Checklist for optimizing streaming operations"""
    
    checklist = {
        "Data Preparation": [
            "✓ Use scan_* functions instead of read_* functions",
            "✓ Apply filters as early as possible in the pipeline",
            "✓ Select only necessary columns",
            "✓ Use appropriate data types (smaller types save memory)",
            "✓ Consider chunking very large files into smaller ones"
        ],
        
        "Query Design": [
            "✓ Structure queries to be streaming-compatible",
            "✓ Avoid operations that require full dataset view",
            "✓ Use group_by aggregations instead of complex window functions",
            "✓ Test queries with .explain() to verify streaming capability",
            "✓ Break complex queries into smaller, cacheable steps"
        ],
        
        "Resource Management": [
            "✓ Configure appropriate chunk sizes for your system",
            "✓ Monitor memory usage during execution",
            "✓ Set reasonable thread limits",
            "✓ Use garbage collection in long-running processes",
            "✓ Implement fallback strategies for failed streaming"
        ],
        
        "Performance": [
            "✓ Use efficient file formats (Parquet with good compression)",
            "✓ Partition large datasets by relevant columns (date, symbol)",
            "✓ Use column pruning and predicate pushdown",
            "✓ Profile operations to identify bottlenecks",
            "✓ Consider caching intermediate results for reuse"
        ]
    }
    
    print("Streaming Best Practices Checklist:")
    print("=" * 50)
    
    for category, items in checklist.items():
        print(f"\n{category}:")
        for item in items:
            print(f"  {item}")
    
    return checklist
```

---

## Production Memory Management

### Production-Ready Memory Management

```python
def production_memory_management():
    """Production-ready memory management patterns"""
    
    def memory_pool_configuration():
        """Configure memory pools for production workloads"""
        
        import psutil
        
        def setup_production_memory_pools():
            """Setup optimized memory pools for production"""
            
            # Get system memory
            total_memory_gb = psutil.virtual_memory().total / (1024**3)
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            
            # Conservative memory allocation (use 70% of available)
            polars_memory_gb = min(available_memory_gb * 0.7, total_memory_gb * 0.5)
            
            # Configure environment variables
            os.environ['POLARS_POOL_BYTES'] = str(int(polars_memory_gb * 1024**3))
            os.environ['POLARS_MAX_THREADS'] = str(min(16, os.cpu_count()))
            os.environ['POLARS_STREAMING_CHUNK_SIZE'] = '200000'
            
            print(f"Production memory configuration:")
            print(f"  Total system memory: {total_memory_gb:.1f} GB")
            print(f"  Available memory: {available_memory_gb:.1f} GB")
            print(f"  Polars allocation: {polars_memory_gb:.1f} GB")
            
            return {
                'total_memory_gb': total_memory_gb,
                'allocated_memory_gb': polars_memory_gb,
                'max_threads': min(16, os.cpu_count())
            }
        
        return setup_production_memory_pools
    
    def health_monitoring():
        """Health monitoring for production Polars workloads"""
        
        class PolarsHealthMonitor:
            def __init__(self, check_interval: int = 60):
                self.check_interval = check_interval
                self.metrics = {
                    'memory_usage_history': deque(maxlen=100),
                    'query_times': deque(maxlen=50),
                    'errors': deque(maxlen=20)
                }
            
            def record_query(self, query_name: str, execution_time: float, memory_used: float):
                """Record query metrics"""
                self.metrics['query_times'].append({
                    'query': query_name,
                    'time': execution_time,
                    'timestamp': time.time()
                })
                
                self.metrics['memory_usage_history'].append({
                    'memory_mb': memory_used,
                    'timestamp': time.time()
                })
            
            def record_error(self, error_type: str, error_message: str):
                """Record error for monitoring"""
                self.metrics['errors'].append({
                    'type': error_type,
                    'message': error_message,
                    'timestamp': time.time()
                })
            
            def get_health_status(self):
                """Get current health status"""
                if not self.metrics['memory_usage_history']:
                    return {'status': 'unknown', 'reason': 'no data'}
                
                # Check recent memory usage
                recent_memory = [m['memory_mb'] for m in list(self.metrics['memory_usage_history'])[-10:]]
                avg_memory = sum(recent_memory) / len(recent_memory) if recent_memory else 0
                
                # Check for recent errors
                recent_errors = [e for e in self.metrics['errors'] 
                               if time.time() - e['timestamp'] < 300]  # Last 5 minutes
                
                # Check query performance
                recent_queries = [q for q in self.metrics['query_times']
                                if time.time() - q['timestamp'] < 600]  # Last 10 minutes
                
                avg_query_time = sum(q['time'] for q in recent_queries) / len(recent_queries) if recent_queries else 0
                
                # Determine health status
                if recent_errors:
                    return {
                        'status': 'warning',
                        'reason': f'{len(recent_errors)} recent errors',
                        'avg_memory_mb': avg_memory,
                        'avg_query_time': avg_query_time
                    }
                elif avg_memory > 8000:  # 8GB threshold
                    return {
                        'status': 'warning',
                        'reason': 'high memory usage',
                        'avg_memory_mb': avg_memory,
                        'avg_query_time': avg_query_time
                    }
                else:
                    return {
                        'status': 'healthy',
                        'avg_memory_mb': avg_memory,
                        'avg_query_time': avg_query_time
                    }
        
        return PolarsHealthMonitor
    
    return memory_pool_configuration, health_monitoring
```

---

## Navigation

### Related Guides

- **[Main README](README.md)** - Overview and quick start guide
- **[Quantitative Research](polars_quantitative_research.md)** - Time series analysis, factor modeling, and risk management
- **[Polars vs Pandas Integration](polars_pandas_integration.md)** - Integration strategies and conversion patterns
- **[Performance & Large Datasets](polars_performance_and_large_datasets.md)** - Performance optimization and large-scale processing
- **[Advanced Techniques](polars_advanced_techniques.md)** - Complex queries, backtesting, and debugging
- **[Practical Implementation](polars_practical_implementation.md)** - I/O best practices, data cleaning, and troubleshooting

### Key Takeaways

1. **Streaming Capabilities**: Use streaming for datasets larger than RAM, but understand limitations and fallback strategies
2. **Memory Optimization**: Optimize data types, use appropriate chunk sizes, and monitor memory usage actively
3. **Resource Management**: Configure Polars for your specific environment and implement proper resource limits
4. **Chunk Processing**: Use intelligent chunking strategies for datasets that exceed system capabilities
5. **Production Ready**: Implement health monitoring, error handling, and adaptive resource management for production workloads
6. **Best Practices**: Follow streaming best practices and maintain hybrid approaches for complex operations

This guide provides the foundation for handling large-scale datasets efficiently using Polars' streaming capabilities while maintaining optimal memory usage and system performance.