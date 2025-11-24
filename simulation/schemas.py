import jax
import jax.numpy as jnp
from jax import random, jit
from jax.scipy.linalg import expm
import numpy as np
import pandas as pd
from typing import Optional, Dict, Union, Literal
from functools import partial

from .alfonsi import simulate_fast_second_order_aff
from .utils import convert_jax_results_to_dict
from .euler_maruyama import *

SchemaType = Literal["EULER_CORRECTED", "EULER_FLOORED", "ALFONSI"]


def simulate_wishart(
    x: Union[np.ndarray, jnp.ndarray],
    alpha_bar: Union[np.ndarray, jnp.ndarray],
    b: Union[np.ndarray, jnp.ndarray],
    a: Union[np.ndarray, jnp.ndarray],
    start_time: float,
    time_list: Union[list, tuple, np.ndarray, jnp.ndarray],
    num_paths: int = 10,
    dt_substep: float = 0.01,
    max_k: int = 10,
    schema: SchemaType = "EULER_FLOORED",
    key: Optional[jax.random.PRNGKey] = None
) -> Dict[int, pd.Series]:
    """
    Simulate Wishart process using specified numerical scheme.
    
    This is the main entry point for Wishart process simulation, supporting
    multiple numerical schemes with different accuracy/performance tradeoffs.
    
    Parameters
    ----------
    x : array-like
        Initial state matrix (d x d), must be symmetric positive semidefinite
    alpha_bar : array-like
        Drift parameter matrix (d x d), corresponds to alpha/omega in literature
    b : array-like
        Mean reversion matrix (d x d), corresponds to m in some notations
    a : array-like
        Volatility matrix (d x d), corresponds to sigma in some notations
    start_time : float
        Starting time of simulation
    time_list : array-like
        Times at which to record the process
    num_paths : int, optional
        Number of independent paths to simulate
    dt_substep : float, optional
        Time step size for discretization
    max_k : int, optional
        Maximum order for series expansion (only used in ALFONSI scheme)
    schema : {"EULER_CORRECTED", "EULER_FLOORED", "ALFONSI"}, optional
        Numerical scheme to use:
        - "EULER_CORRECTED": Euler-Maruyama with PSD correction before step
        - "EULER_FLOORED": Euler-Maruyama with PSD correction after step
        - "ALFONSI": Fast second-order scheme from Alfonsi's book
    key : Optional[jax.random.PRNGKey]
        Random key for reproducibility
        
    Returns
    -------
    Dict[int, pd.Series]
        Dictionary mapping path index to time series of matrix values
        
    Raises
    ------
    ValueError
        If unknown schema is specified
        
    Notes
    -----
    - EULER_CORRECTED: More stable but slightly slower
    - EULER_FLOORED: Standard approach, good balance
    - ALFONSI: Higher accuracy but more complex, best for smooth processes
    
    Examples
    --------
    >>> import jax.numpy as jnp
    >>> d = 2
    >>> x0 = jnp.eye(d) * 0.05
    >>> alpha = jnp.eye(d) * 0.01
    >>> b = jnp.eye(d) * 0.1
    >>> a = jnp.eye(d) * 0.2
    >>> times = [0.0, 0.1, 0.2, 0.3]
    >>> results = simulate_wishart(x0, alpha, b, a, 0.0, times)
    """
    if key is None:
        key = random.PRNGKey(42)
    
    # Convert to tuple (hashable) for JAX compatibility
    if isinstance(time_list, (list, np.ndarray, jnp.ndarray)):
        time_list_python = tuple(time_list)
    else:
        time_list_python = time_list

    ##newly added to be debuged
    if time_list_python[0] != start_time:
        time_list_python = (start_time,) + time_list_python


    schema_upper = schema.upper()
    
    if schema_upper == "EULER_CORRECTED":
        # Use the fast JAX implementation
        results = simulate_wishart_corrected_euler_maruyama_wrapper(
            x, alpha_bar, b, a, start_time, time_list_python, 
            num_paths, dt_substep, key
        )
        # Convert to dictionary format
        return convert_jax_results_to_dict(results, time_list_python)
        
    elif schema_upper == "EULER_FLOORED":
        # Use the fast JAX implementation
        results = simulate_wishart_floor_euler_maruyama_wrapper(
            x, alpha_bar, b, a, start_time, time_list_python, 
            num_paths, dt_substep, key
        )
        # Convert to dictionary format
        return convert_jax_results_to_dict(results, time_list_python)
        
    elif schema_upper == "ALFONSI":
        # Alfonsi method returns dictionary directly
        return simulate_fast_second_order_aff(
            x, alpha_bar, b, a, start_time, time_list_python, 
            num_paths, dt_substep, max_k, key
        )
        
    else:
        raise ValueError(
            f"Unknown schema: {schema}. "
            f"Supported schemas are 'EULER_FLOORED', 'EULER_CORRECTED', 'ALFONSI'."
        )


def simulate_wishart_jump(
    x: Union[np.ndarray, jnp.ndarray],
    alpha_bar: Union[np.ndarray, jnp.ndarray],
    b: Union[np.ndarray, jnp.ndarray],
    a: Union[np.ndarray, jnp.ndarray],
    lambda_intensity: float,
    nu: float,
    eta: Union[np.ndarray, jnp.ndarray],
    xi: Union[np.ndarray, jnp.ndarray],
    start_time: float,
    time_list: Union[list, tuple, np.ndarray, jnp.ndarray],
    num_paths: int = 10,
    dt_substep: float = 0.01,
    max_k: int = 10,
    schema: SchemaType = "EULER_FLOORED",
    key: Optional[jax.random.PRNGKey] = None
) -> Dict[int, pd.Series]:
    """
    Simulate Wishart process with jumps.
    
    Extends the basic Wishart process to include jump components, useful
    for modeling sudden changes in volatility or correlation structure.
    
    Parameters
    ----------
    x : array-like
        Initial state matrix (d x d)
    alpha_bar : array-like
        Drift parameter matrix (d x d)
    b : array-like
        Mean reversion matrix (d x d)
    a : array-like
        Volatility matrix (d x d)
    lambda_intensity : float
        Jump intensity (average number of jumps per unit time)
    nu : float
        Jump size scaling parameter
    eta : array-like
        Jump direction matrix (d x d)
    xi : array-like
        Jump offset matrix (d x d)
    start_time : float
        Starting time
    time_list : array-like
        Times at which to record
    num_paths : int, optional
        Number of paths
    dt_substep : float, optional
        Time step size
    max_k : int, optional
        Maximum order for series expansion
    schema : {"EULER_CORRECTED", "EULER_FLOORED", "ALFONSI"}, optional
        Numerical scheme (currently only EULER schemes support jumps)
    key : Optional[jax.random.PRNGKey]
        Random key
        
    Returns
    -------
    Dict[int, pd.Series]
        Dictionary mapping path index to time series
        
    Notes
    -----
    Jump component adds: nu * N_t * (Z @ eta + xi/nu)
    where N_t is a Poisson process and Z is an exponential random matrix.
    
    Currently, only EULER_CORRECTED and EULER_FLOORED schemes support jumps.
    """
    if key is None:
        key = random.PRNGKey(42)
    
    # Convert to tuple for JAX compatibility
    if isinstance(time_list, (list, np.ndarray, jnp.ndarray)):
        time_list_python = tuple(time_list)
    else:
        time_list_python = time_list
        
    schema_upper = schema.upper()

    if schema_upper in ["EULER_CORRECTED", "EULER_FLOORED"]:
        # Both Euler schemes use the same jump implementation
        results = simulate_wishart_jump_euler_maruyama_wrapper(
            x, alpha_bar, b, a,
            lambda_intensity, nu, eta, xi,
            start_time, time_list_python, num_paths, dt_substep, key
        )
        return convert_jax_results_to_dict(results, time_list_python)
        
    elif schema_upper == "ALFONSI":
        # Alfonsi method doesn't currently support jumps
        print("Warning: ALFONSI scheme does not support jumps. Using EULER_FLOORED instead.")
        results = simulate_wishart_jump_euler_maruyama_wrapper(
            x, alpha_bar, b, a,
            lambda_intensity, nu, eta, xi,
            start_time, time_list_python, num_paths, dt_substep, key
        )
        return convert_jax_results_to_dict(results, time_list_python)
        
    else:
        raise ValueError(
            f"Unknown schema: {schema}. "
            f"Supported schemas are 'EULER_FLOORED', 'EULER_CORRECTED', 'ALFONSI'."
        )


def simulate_wishart_bridge(
    x_start: Union[np.ndarray, jnp.ndarray],
    x_end: Union[np.ndarray, jnp.ndarray],
    alpha_bar: Union[np.ndarray, jnp.ndarray],
    b: Union[np.ndarray, jnp.ndarray],
    a: Union[np.ndarray, jnp.ndarray],
    start_time: float,
    end_time: float,
    time_list: Union[list, tuple, np.ndarray, jnp.ndarray],
    num_paths: int = 10,
    dt_substep: float = 0.01,
    key: Optional[jax.random.PRNGKey] = None
) -> Dict[int, pd.Series]:
    """
    Simulate Wishart bridge process conditioned on start and end values.
    
    Parameters
    ----------
    x_start : array-like
        Starting matrix value
    x_end : array-like
        Ending matrix value (condition)
    alpha_bar : array-like
        Drift parameter
    b : array-like
        Mean reversion
    a : array-like
        Volatility
    start_time : float
        Starting time
    end_time : float
        Ending time
    time_list : array-like
        Times at which to record (should be between start_time and end_time)
    num_paths : int, optional
        Number of bridge paths
    dt_substep : float, optional
        Time step size
    key : Optional[jax.random.PRNGKey]
        Random key
        
    Returns
    -------
    Dict[int, pd.Series]
        Dictionary mapping path index to time series
        
    Notes
    -----
    This is a placeholder implementation. Full Wishart bridge sampling
    requires more sophisticated algorithms.
    """
    # For now, use standard simulation and reject paths that don't meet end condition
    # This is inefficient but serves as a placeholder
    
    print("Warning: Wishart bridge sampling not fully implemented. Using approximation.")
    
    # Ensure time_list includes start and end times
    time_list = list(time_list)
    if start_time not in time_list:
        time_list.append(start_time)
    if end_time not in time_list:
        time_list.append(end_time)
    time_list = sorted(time_list)
    
    # Simulate standard paths
    results = simulate_wishart(
        x_start, alpha_bar, b, a, start_time, time_list,
        num_paths, dt_substep, key=key
    )
    
    # Apply simple linear interpolation adjustment (placeholder)
    for path_idx in results:
        path_series = results[path_idx]
        # Adjust final value
        if end_time in path_series.index:
            path_series[end_time] = x_end
            
    return results
#region TO BE REMOVED
# """
# High-level simulation wrappers for different Wishart process schemes.
# """
# import jax
# import jax.numpy as jnp
# from jax import random
# import numpy as np
# import pandas as pd
# from typing import Optional, Union, Dict, Literal

# from .euler_maruyama import (
#     simulate_wishart_corrected_euler_maruyama_wrapper,
#     simulate_wishart_floor_euler_maruyama_wrapper
# )
# from .jax_implementations import (
#     simulate_wishart_jax_scan,
#     simulate_wishart_jump_euler_maruyama_wrapper
# )
# from .alfonsi import simulate_fast_second_order_aff
# from .utils import convert_jax_results_to_dict

#endregion
