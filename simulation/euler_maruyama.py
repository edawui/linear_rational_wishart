"""
Core Euler-Maruyama simulation implementations for Wishart processes.
"""
import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
from typing import Optional, Union
import numpy as np

from ..math.psd_corrections import nearest_psd_jax, sqrtm_real
from ..config.constants import TIMEDECAYVOL
from .utils import ensure_start_time_in_list


@partial(jit, static_argnums=(5, 6, 8))
def simulate_wishart_corrected_euler_maruyama(
    x: jnp.ndarray, 
    alpha: jnp.ndarray, 
    b: jnp.ndarray, 
    a: jnp.ndarray, 
    start_time: float,
    time_list: jnp.ndarray, 
    num_paths: int, 
    dt: float, 
    start_idx: int,
    key: Optional[jax.random.PRNGKey] = None
) -> jnp.ndarray:
    """
    Fast vectorized JAX implementation of WIS simulation with corrected method.
    
    Key optimizations:
    - JIT compilation for massive speedup
    - Vectorized operations using vmap
    - Pre-allocated arrays
    - Efficient JAX random number generation
    
    Parameters
    ----------
    x : jnp.ndarray
        Initial state matrix
    alpha : jnp.ndarray
        Drift parameter (omega)
    b : jnp.ndarray
        Mean reversion parameter (m)
    a : jnp.ndarray
        Volatility parameter (sigma)
    start_time : float
        Starting time
    time_list : jnp.ndarray
        Array of time points
    num_paths : int
        Number of simulation paths
    dt : float
        Time step size
    start_idx : int
        Starting index in time_list
    key : Optional[jax.random.PRNGKey]
        Random key for reproducibility
        
    Returns
    -------
    jnp.ndarray
        Simulated paths with shape (num_paths, n_times, dim, dim)
    """
    if key is None:
        key = jax.random.PRNGKey(42)
    
    # Convert inputs to JAX arrays
    x = jnp.asarray(x)
    alpha = jnp.asarray(alpha)
    b = jnp.asarray(b)
    a = jnp.asarray(a)
    time_list = jnp.asarray(time_list)
    
    # Add startTime if not present
    time_list = ensure_start_time_in_list(time_list, start_time)
    
    # Pre-allocate results array
    n_times = len(time_list)
    dim = x.shape[0]
    results = jnp.zeros((num_paths, n_times, dim, dim))
    
    # Initialize all paths with starting value
    results = results.at[:, start_idx].set(x[jnp.newaxis, :, :])
    
    # Pre-compute constants
    sigma = a
    omega = alpha
    m = b
    min_dt = dt
    
    # Main simulation loop - now with concrete start_idx
    for time_idx in range(start_idx + 1, n_times):
        target_time = time_list[time_idx]
        last_time = time_list[time_idx - 1]
        
        # Get current state for all paths
        current_v = results[:, time_idx - 1]
        
        # Replace while loop with lax.while_loop
        def while_condition(state):
            current_t, _, _ = state
            return current_t < target_time
        
        def while_body(state):
            current_t, current_v, key = state
            
            dt_ = jnp.minimum(min_dt, target_time - current_t)
            new_t = current_t + dt_
            sqrt_dt = jnp.sqrt(dt_)
            
            # Generate random increments for all paths at once
            key, subkey = jax.random.split(key)
            dw = jax.random.normal(subkey, shape=(num_paths, dim, dim)) * sqrt_dt
            
            # Vectorized operations across all paths using vmap
            def single_path_update(v_single, dw_single):
                # PSD correction BEFORE (corrected method)
                corrected_v = nearest_psd_jax(v_single)
                
                # Matrix square root of corrected v
                sqrt_v = jax.scipy.linalg.sqrtm(corrected_v)
                # Ensure real output
                sqrt_v = jnp.real(sqrt_v)
                
                # Diffusion term
                v1 = sqrt_v @ dw_single @ (sigma * jnp.exp(-TIMEDECAYVOL * current_t))
                v1 = v1 + v1.T
                
                # Drift term
                drift = omega + m @ v_single + v_single @ m.T
                
                # Update
                return v_single + dt_ * drift + v1
            
            # Apply to all paths at once
            new_v = vmap(single_path_update)(current_v, dw)
            
            return (new_t, new_v, key)
        
        # Run the while loop
        initial_state = (last_time, current_v, key)
        final_t, final_v, key = jax.lax.while_loop(
            while_condition,
            while_body,
            initial_state
        )
        
        # Store results
        results = results.at[:, time_idx].set(final_v)
    
    return results


@partial(jit, static_argnums=(5, 6, 8))
def simulate_wishart_floor_euler_maruyama(
    x: jnp.ndarray, 
    alpha: jnp.ndarray, 
    b: jnp.ndarray, 
    a: jnp.ndarray, 
    start_time: float,
    time_list: jnp.ndarray, 
    num_paths: int, 
    dt: float, 
    start_idx: int,
    key: Optional[jax.random.PRNGKey] = None
) -> jnp.ndarray:
    """
    Fast vectorized JAX implementation with floor (PSD correction after each step).
    
    Key differences from corrected version:
    - PSD correction applied AFTER each Euler step (not before)
    - Direct application of matrix square root to original v
    
    Parameters
    ----------
    x : jnp.ndarray
        Initial state matrix
    alpha : jnp.ndarray
        Drift parameter (omega)
    b : jnp.ndarray
        Mean reversion parameter (m)
    a : jnp.ndarray
        Volatility parameter (sigma)
    start_time : float
        Starting time
    time_list : jnp.ndarray
        Array of time points
    num_paths : int
        Number of simulation paths
    dt : float
        Time step size
    start_idx : int
        Starting index in time_list
    key : Optional[jax.random.PRNGKey]
        Random key for reproducibility
        
    Returns
    -------
    jnp.ndarray
        Simulated paths with shape (num_paths, n_times, dim, dim)
    """
    if key is None:
        key = jax.random.PRNGKey(42)
    
    # Convert inputs to JAX arrays
    x = jnp.asarray(x)
    alpha = jnp.asarray(alpha)
    b = jnp.asarray(b)
    a = jnp.asarray(a)
    time_list = jnp.asarray(time_list)
    
    # Add startTime if not present
    time_list = ensure_start_time_in_list(time_list, start_time)
    
    # Pre-allocate results array
    n_times = len(time_list)
    dim = x.shape[0]
    results = jnp.zeros((num_paths, n_times, dim, dim))
    
    # Initialize all paths
    results = results.at[:, start_idx].set(x[jnp.newaxis, :, :])
    
    # Pre-compute constants
    sigma = a
    omega = alpha
    m = b
    min_dt = dt
    
    # Main simulation loop - now with concrete start_idx
    for time_idx in range(start_idx + 1, n_times):
        target_time = time_list[time_idx]
        last_time = time_list[time_idx - 1]
        
        current_v = results[:, time_idx - 1].copy()
        
        # Replace while loop with lax.while_loop
        def while_condition(state):
            current_t, _, _ = state
            return current_t < target_time
        
        def while_body(state):
            current_t, current_v, key = state
            
            dt_ = jnp.minimum(min_dt, target_time - current_t)
            new_t = current_t + dt_
            sqrt_dt = jnp.sqrt(dt_)
            
            # Generate random increments
            key, subkey = jax.random.split(key)
            dw = jax.random.normal(subkey, shape=(num_paths, dim, dim)) * sqrt_dt
            
            # Vectorized operations
            def single_path_update_floor(v_single, dw_single):
                # Direct matrix square root (no PSD correction first)
                sqrt_v = jax.scipy.linalg.sqrtm(v_single)
                # Ensure real output by taking real part (imaginary part should be negligible)
                sqrt_v = jnp.real(sqrt_v)
                
                # Diffusion term
                v1 = sqrt_v @ dw_single @ (sigma * jnp.exp(-TIMEDECAYVOL * current_t))
                v1 = v1 + v1.T
                
                # Drift term
                drift = omega + m @ v_single + v_single @ m.T
                
                # Update
                v_new = v_single + dt_ * drift + v1
                
                # PSD correction AFTER the step (floor method)
                return nearest_psd_jax(v_new)
            
            # Apply to all paths
            new_v = vmap(single_path_update_floor)(current_v, dw)
            
            return (new_t, new_v, key)
        
        # Run the while loop
        initial_state = (last_time, current_v, key)
        final_t, final_v, key = jax.lax.while_loop(
            while_condition,
            while_body,
            initial_state
        )
        
        # Store results
        results = results.at[:, time_idx].set(final_v)
    
    return results


# Wrapper functions that handle start_idx computation
def simulate_wishart_corrected_euler_maruyama_wrapper(
    x: Union[np.ndarray, jnp.ndarray], 
    alpha: Union[np.ndarray, jnp.ndarray], 
    b: Union[np.ndarray, jnp.ndarray], 
    a: Union[np.ndarray, jnp.ndarray], 
    start_time: float,
    time_list: Union[np.ndarray, jnp.ndarray], 
    num_paths: int = 10, 
    dt: float = 1.0/360.0, 
    key: Optional[jax.random.PRNGKey] = None
) -> jnp.ndarray:
    """
    Wrapper function that computes start_idx before calling JIT function.
    """
    # Convert to numpy for start_idx computation
    time_list_np = np.array(time_list) if hasattr(time_list, '__array__') else time_list
    mask = np.isclose(time_list_np, start_time)
    start_idx = int(np.argmax(mask))
    
    # Call the JIT-compiled core function
    return simulate_wishart_corrected_euler_maruyama(
        x, alpha, b, a, start_time, time_list, num_paths, dt, start_idx, key
    )


def simulate_wishart_floor_euler_maruyama_wrapper(
    x: Union[np.ndarray, jnp.ndarray], 
    alpha: Union[np.ndarray, jnp.ndarray], 
    b: Union[np.ndarray, jnp.ndarray], 
    a: Union[np.ndarray, jnp.ndarray], 
    start_time: float,
    time_list: Union[np.ndarray, jnp.ndarray], 
    num_paths: int = 10, 
    dt: float = 1.0/360.0, 
    key: Optional[jax.random.PRNGKey] = None
) -> jnp.ndarray:
    """
    Wrapper function that computes start_idx before calling JIT function.
    """
    # Convert to numpy for start_idx computation
    time_list_np = np.array(time_list) if hasattr(time_list, '__array__') else time_list
    mask = np.isclose(time_list_np, start_time)
    start_idx = int(np.argmax(mask))
    
    # Call the JIT-compiled core function
    return simulate_wishart_floor_euler_maruyama(
        x, alpha, b, a, start_time, time_list, num_paths, dt, start_idx, key
    )
