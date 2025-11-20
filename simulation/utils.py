"""
Utility functions for Wishart process simulations.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from typing import Dict, Union, List, Tuple, Optional


def ensure_start_time_in_list(time_list: jnp.ndarray, start_time: float) -> jnp.ndarray:
    """
    Ensure start_time is in time_list - always returns consistent shape.
    
    Parameters
    ----------
    time_list : jnp.ndarray
        Array of time points
    start_time : float
        Starting time to ensure is in the list
        
    Returns
    -------
    jnp.ndarray
        Time list with start_time included
    """
    # For now, return the original time_list
    # The actual insertion logic is handled in the simulation functions
    return time_list


def convert_jax_results_to_dict(
    jax_results: jnp.ndarray, 
    time_list: Union[jnp.ndarray, np.ndarray, List[float]]
) -> Dict[int, pd.Series]:
    """
    Convert JAX array results to dictionary format matching original interface.
    
    Parameters
    ----------
    jax_results : jnp.ndarray
        Simulation results with shape (num_paths, n_times, dim, dim)
    time_list : Union[jnp.ndarray, np.ndarray, List[float]]
        Time points corresponding to simulation results
        
    Returns
    -------
    Dict[int, pd.Series]
        Dictionary mapping path index to time series of matrices
    """
    num_paths = jax_results.shape[0]
    n_times = jax_results.shape[1]
    
    sim_results = {}
    for path in range(num_paths):
        sim_results[path] = pd.Series(
            {float(time_list[t_idx]): np.array(jax_results[path, t_idx]) 
             for t_idx in range(n_times)}
        )
    
    return sim_results


def find_start_idx(time_list: Union[jnp.ndarray, np.ndarray, List[float]], start_time: float) -> int:
    """
    Find the index of start_time in time_list.
    
    Parameters
    ----------
    time_list : Union[jnp.ndarray, np.ndarray, List[float]]
        Array of time points
    start_time : float
        Time to find
        
    Returns
    -------
    int
        Index of start_time in time_list
    """
    time_list_np = np.array(time_list) if hasattr(time_list, '__array__') else time_list
    mask = np.isclose(time_list_np, start_time)
    return int(np.argmax(mask))


def prepare_simulation_inputs(
    x0: Union[np.ndarray, jnp.ndarray],
    omega: Union[np.ndarray, jnp.ndarray],
    m: Union[np.ndarray, jnp.ndarray],
    sigma: Union[np.ndarray, jnp.ndarray],
    time_list: Union[np.ndarray, jnp.ndarray, List[float]],
    start_time: float
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
    """
    Prepare and validate inputs for simulation.
    
    Parameters
    ----------
    x0 : Union[np.ndarray, jnp.ndarray]
        Initial state matrix
    omega : Union[np.ndarray, jnp.ndarray]
        Drift parameter (alpha)
    m : Union[np.ndarray, jnp.ndarray]
        Mean reversion parameter (b)
    sigma : Union[np.ndarray, jnp.ndarray]
        Volatility parameter (a)
    time_list : Union[np.ndarray, jnp.ndarray, List[float]]
        Time points for simulation
    start_time : float
        Starting time
        
    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int]
        Converted JAX arrays and start index
    """
    # Convert to JAX arrays
    x0_jax = jnp.asarray(x0)
    omega_jax = jnp.asarray(omega)
    m_jax = jnp.asarray(m)
    sigma_jax = jnp.asarray(sigma)
    time_list_jax = jnp.asarray(time_list)
    
    # Find start index
    start_idx = find_start_idx(time_list, start_time)
    
    return x0_jax, omega_jax, m_jax, sigma_jax, time_list_jax, start_idx


def validate_simulation_parameters(
    x0: jnp.ndarray,
    omega: jnp.ndarray,
    m: jnp.ndarray,
    sigma: jnp.ndarray,
    num_paths: int,
    dt: float
) -> None:
    """
    Validate simulation parameters.
    
    Parameters
    ----------
    x0 : jnp.ndarray
        Initial state matrix
    omega : jnp.ndarray
        Drift parameter
    m : jnp.ndarray
        Mean reversion parameter
    sigma : jnp.ndarray
        Volatility parameter
    num_paths : int
        Number of simulation paths
    dt : float
        Time step size
        
    Raises
    ------
    ValueError
        If parameters are invalid
    """
    # Check dimensions
    n = x0.shape[0]
    if x0.shape != (n, n):
        raise ValueError(f"x0 must be square matrix, got shape {x0.shape}")
    
    if omega.shape != (n, n):
        raise ValueError(f"omega must have shape ({n}, {n}), got {omega.shape}")
    
    if m.shape != (n, n):
        raise ValueError(f"m must have shape ({n}, {n}), got {m.shape}")
    
    if sigma.shape != (n, n):
        raise ValueError(f"sigma must have shape ({n}, {n}), got {sigma.shape}")
    
    # Check positive parameters
    if num_paths <= 0:
        raise ValueError(f"num_paths must be positive, got {num_paths}")
    
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")


def create_time_grid(T: float, dt: float, include_zero: bool = True) -> jnp.ndarray:
    """
    Create a time grid for simulation.
    
    Parameters
    ----------
    T : float
        Final time
    dt : float
        Time step size
    include_zero : bool, optional
        Whether to include t=0
        
    Returns
    -------
    jnp.ndarray
        Time grid
    """
    if include_zero:
        return jnp.arange(0, T + dt, dt)
    else:
        return jnp.arange(dt, T + dt, dt)


def extract_path_at_times(
    results: Dict[int, pd.Series],
    path_idx: int,
    times: Union[List[float], np.ndarray]
) -> List[np.ndarray]:
    """
    Extract simulation results at specific times for a given path.
    
    Parameters
    ----------
    results : Dict[int, pd.Series]
        Simulation results dictionary
    path_idx : int
        Path index to extract
    times : Union[List[float], np.ndarray]
        Times to extract
        
    Returns
    -------
    List[np.ndarray]
        List of matrices at requested times
    """
    path_series = results[path_idx]
    extracted = []
    
    for t in times:
        # Find closest time in series
        idx = path_series.index.get_indexer([t], method='nearest')[0]
        extracted.append(path_series.iloc[idx])
    
    return extracted


def aggregate_simulation_statistics(
    results: Dict[int, pd.Series]
) -> Dict[float, Dict[str, np.ndarray]]:
    """
    Compute aggregate statistics across all paths.
    
    Parameters
    ----------
    results : Dict[int, pd.Series]
        Simulation results dictionary
        
    Returns
    -------
    Dict[float, Dict[str, np.ndarray]]
        Dictionary mapping time to statistics (mean, std)
    """
    # Get all unique times
    all_times = set()
    for path_series in results.values():
        all_times.update(path_series.index)
    
    all_times = sorted(all_times)
    
    stats = {}
    for t in all_times:
        values = []
        for path_series in results.values():
            if t in path_series.index:
                values.append(path_series[t])
        
        if values:
            values_array = np.stack(values)
            stats[t] = {
                'mean': np.mean(values_array, axis=0),
                'std': np.std(values_array, axis=0),
                'min': np.min(values_array, axis=0),
                'max': np.max(values_array, axis=0)
            }
    
    return stats
