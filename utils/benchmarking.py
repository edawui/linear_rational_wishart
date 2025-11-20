"""
Performance benchmarking tools for Wishart process simulations.
"""
import time
import jax
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Any
import pandas as pd
from functools import wraps

from ..simulation.jax_implementations import (
    simulate_wishart_jax_scan,
    simulate_wishart_jax_parallel,
    simulate_wishart_floor_jax_scan
)
from ..simulation.euler_maruyama import (
    simulate_wishart_corrected_euler_maruyama_wrapper,
    simulate_wishart_floor_euler_maruyama_wrapper
)


def timer(func: Callable) -> Callable:
    """
    Decorator to time function execution.
    
    Parameters
    ----------
    func : Callable
        Function to time
        
    Returns
    -------
    Callable
        Wrapped function that prints execution time
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper


def benchmark_all_methods(
    x: np.ndarray, 
    alpha: np.ndarray, 
    b: np.ndarray, 
    a: np.ndarray, 
    start_time: float,
    time_list: np.ndarray, 
    num_paths: int = 100,
    dt: float = 1.0/360.0,
    warmup_runs: int = 2,
    benchmark_runs: int = 5
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark all JAX implementations.
    
    Parameters
    ----------
    x : np.ndarray
        Initial state matrix
    alpha : np.ndarray
        Drift parameter (omega)
    b : np.ndarray
        Mean reversion parameter (m)
    a : np.ndarray
        Volatility parameter (sigma)
    start_time : float
        Starting time
    time_list : np.ndarray
        Time points for simulation
    num_paths : int, optional
        Number of paths to simulate
    dt : float, optional
        Time step size
    warmup_runs : int, optional
        Number of warmup runs for JIT compilation
    benchmark_runs : int, optional
        Number of benchmark runs to average
        
    Returns
    -------
    Dict[str, Dict[str, float]]
        Benchmark results with timing statistics
    """
    # Ensure inputs are numpy arrays for fair comparison
    x = np.array(x)
    alpha = np.array(alpha)
    b = np.array(b)
    a = np.array(a)
    time_list = np.array(time_list)
    
    methods = [
        ("JAX Corrected Euler-Maruyama", simulate_wishart_corrected_euler_maruyama_wrapper),
        ("JAX Floor Euler-Maruyama", simulate_wishart_floor_euler_maruyama_wrapper),
        ("JAX Scan", simulate_wishart_jax_scan),
        ("JAX Floor Scan", simulate_wishart_floor_jax_scan),
    ]
    
    # Check if multiple devices available
    if jax.device_count() > 1:
        methods.append(("JAX Parallel (Multi-Device)", simulate_wishart_jax_parallel))
    
    results = {}
    
    print("Benchmarking JAX WIS simulation methods...")
    print(f"Problem size: {num_paths} paths, {len(time_list)} time steps, {x.shape[0]}x{x.shape[1]} matrix")
    print(f"Available devices: {jax.devices()}")
    print(f"Warmup runs: {warmup_runs}, Benchmark runs: {benchmark_runs}")
    print("-" * 80)
    
    for name, method in methods:
        try:
            timings = []
            
            # Warm-up JIT compilation
            if "Parallel" not in name:
                key = jax.random.PRNGKey(42)
                for _ in range(warmup_runs):
                    _ = method(x, alpha, b, a, start_time, time_list, min(10, num_paths), dt, key)
            
            # Actual benchmark
            for run in range(benchmark_runs):
                start = time.time()
                
                if "Parallel" in name:
                    # Special handling for parallel version
                    n_devices = jax.device_count()
                    paths_per_device = num_paths // n_devices
                    keys = jax.random.split(jax.random.PRNGKey(42 + run), n_devices)
                    result = method(keys, x, alpha, b, a, start_time, time_list, paths_per_device, dt)
                else:
                    key = jax.random.PRNGKey(42 + run)
                    result = method(x, alpha, b, a, start_time, time_list, num_paths, dt, key)
                
                # Force computation
                result.block_until_ready() if hasattr(result, 'block_until_ready') else None
                
                end = time.time()
                timings.append(end - start)
            
            # Compute statistics
            timings_array = np.array(timings)
            results[name] = {
                'mean': np.mean(timings_array),
                'std': np.std(timings_array),
                'min': np.min(timings_array),
                'max': np.max(timings_array),
                'median': np.median(timings_array)
            }
            
            print(f"{name:35}: {results[name]['mean']:.4f}s ± {results[name]['std']:.4f}s")
            
        except Exception as e:
            print(f"{name:35}: FAILED - {str(e)}")
            results[name] = {'error': str(e)}
    
    # Find fastest method
    valid_results = {k: v for k, v in results.items() if 'mean' in v}
    if valid_results:
        fastest = min(valid_results.items(), key=lambda x: x[1]['mean'])
        print("-" * 80)
        print(f"Fastest method: {fastest[0]} ({fastest[1]['mean']:.4f}s)")
        
        # Compute relative speeds
        fastest_time = fastest[1]['mean']
        print("\nRelative speeds:")
        for name, stats in valid_results.items():
            speedup = fastest_time / stats['mean']
            print(f"{name:35}: {speedup:.2f}x")
    
    return results


def benchmark_wishart_methods(
    x: np.ndarray, 
    alpha: np.ndarray, 
    b: np.ndarray, 
    a: np.ndarray, 
    start_time: float,
    time_list: np.ndarray, 
    num_paths: int = 100
) -> Dict[str, Dict[str, float]]:
    """
    Comprehensive benchmark of all WIS simulation methods.
    Compatible with original function signature.
    
    Parameters
    ----------
    x : np.ndarray
        Initial state matrix
    alpha : np.ndarray
        Drift parameter (omega)
    b : np.ndarray
        Mean reversion parameter (m)
    a : np.ndarray
        Volatility parameter (sigma)
    start_time : float
        Starting time
    time_list : np.ndarray
        Time points for simulation
    num_paths : int, optional
        Number of paths to simulate
        
    Returns
    -------
    Dict[str, Dict[str, float]]
        Benchmark results
    """
    return benchmark_all_methods(x, alpha, b, a, start_time, time_list, num_paths)


def profile_memory_usage(
    method: Callable,
    x: np.ndarray,
    alpha: np.ndarray,
    b: np.ndarray,
    a: np.ndarray,
    start_time: float,
    time_list: np.ndarray,
    num_paths: int = 100
) -> Dict[str, Any]:
    """
    Profile memory usage of a simulation method.
    
    Parameters
    ----------
    method : Callable
        Simulation method to profile
    x : np.ndarray
        Initial state matrix
    alpha : np.ndarray
        Drift parameter
    b : np.ndarray
        Mean reversion parameter
    a : np.ndarray
        Volatility parameter
    start_time : float
        Starting time
    time_list : np.ndarray
        Time points
    num_paths : int
        Number of paths
        
    Returns
    -------
    Dict[str, Any]
        Memory usage statistics
    """
    try:
        import tracemalloc
    except ImportError:
        return {"error": "tracemalloc not available"}
    
    # Start tracing
    tracemalloc.start()
    
    # Run method
    key = jax.random.PRNGKey(42)
    result = method(x, alpha, b, a, start_time, time_list, num_paths, 1.0/360.0, key)
    
    # Get memory usage
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return {
        "current_memory_mb": current / 1024 / 1024,
        "peak_memory_mb": peak / 1024 / 1024,
        "result_shape": result.shape
    }


def create_benchmark_report(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Create a formatted benchmark report.
    
    Parameters
    ----------
    results : Dict[str, Dict[str, float]]
        Benchmark results from benchmark_all_methods
    save_path : Optional[str]
        Path to save the report
        
    Returns
    -------
    pd.DataFrame
        Formatted benchmark report
    """
    # Convert to DataFrame
    df = pd.DataFrame(results).T
    
    # Add relative performance column if possible
    if 'mean' in df.columns and not df['mean'].isna().all():
        fastest_time = df['mean'].min()
        df['relative_speed'] = fastest_time / df['mean']
        df['percent_slower'] = ((df['mean'] - fastest_time) / fastest_time * 100).round(2)
    
    # Sort by mean time
    if 'mean' in df.columns:
        df = df.sort_values('mean')
    
    # Format numeric columns
    numeric_columns = ['mean', 'std', 'min', 'max', 'median']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].round(4)
    
    if save_path:
        df.to_csv(save_path)
        print(f"Benchmark report saved to {save_path}")
    
    return df


class BenchmarkContext:
    """
    Context manager for benchmarking with automatic timing and resource tracking.
    """
    
    def __init__(self, name: str, track_memory: bool = False):
        self.name = name
        self.track_memory = track_memory
        self.start_time = None
        self.end_time = None
        self.start_memory = None
        self.end_memory = None
        
    def __enter__(self):
        self.start_time = time.time()
        if self.track_memory:
            try:
                import tracemalloc
                if not tracemalloc.is_tracing():
                    tracemalloc.start()
                self.start_memory = tracemalloc.get_traced_memory()[0]
            except ImportError:
                self.track_memory = False
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        if self.track_memory:
            try:
                import tracemalloc
                self.end_memory = tracemalloc.get_traced_memory()[0]
            except ImportError:
                pass
        
        # Print results
        elapsed = self.end_time - self.start_time
        print(f"{self.name}: {elapsed:.4f}s")
        
        if self.track_memory and self.start_memory is not None and self.end_memory is not None:
            memory_used = (self.end_memory - self.start_memory) / 1024 / 1024
            print(f"  Memory used: {memory_used:.2f} MB")
