"""
JAX-optimized simulation implementations for Wishart processes.
"""
import jax
import jax.numpy as jnp
from jax import jit, vmap, pmap
from functools import partial
from typing import Optional, Union
import numpy as np

from ..math.psd_corrections import nearest_psd_jax, sqrtm_real
from ..config.constants import TIMEDECAYVOL
from .utils import ensure_start_time_in_list


@partial(jit, static_argnums=(5, 6))
def simulate_wishart_jax_scan(
    x: jnp.ndarray, 
    alpha: jnp.ndarray, 
    b: jnp.ndarray, 
    a: jnp.ndarray, 
    start_time: float,
    time_list: jnp.ndarray, 
    num_paths: int = 10, 
    dt: float = 1.0/360.0, 
    key: Optional[jax.random.PRNGKey] = None
) -> jnp.ndarray:
    """
    Ultra-fast JAX implementation using lax.scan for loop fusion.
    
    This is the fastest pure JAX version using:
    - lax.scan for efficient loop compilation
    - Vectorized operations
    - No Python loops in the core computation
    
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
    if not jnp.any(jnp.isclose(time_list, start_time)):
        insert_index = jnp.searchsorted(time_list, start_time)
        time_list = jnp.insert(time_list, insert_index, start_time)
    
    start_idx = jnp.where(jnp.isclose(time_list, start_time))[0][0]
    
    # Pre-compute constants
    sigma = a
    omega = alpha
    m = b
    dim = x.shape[0]
    
    # Initialize state
    initial_state = jnp.tile(x[jnp.newaxis, :, :], (num_paths, 1, 1))
    
    def scan_fn(carry, time_pair):
        current_v, key = carry
        last_time, target_time = time_pair
        
        # Inner loop for sub-stepping
        def substep_fn(state, _):
            v, t, k = state
            dt_ = jnp.minimum(dt, target_time - t)
            new_t = t + dt_
            sqrt_dt = jnp.sqrt(dt_)
            
            # Random generation
            k, subkey = jax.random.split(k)
            dw = jax.random.normal(subkey, shape=(num_paths, dim, dim)) * sqrt_dt
            
            # Update function for single path
            def update_single(v_single, dw_single):
                corrected_v = nearest_psd_jax(v_single)
                sqrt_v = jax.scipy.linalg.sqrtm(corrected_v)
                v1 = sqrt_v @ dw_single @ (sigma * jnp.exp(-TIMEDECAYVOL * new_t))
                v1 = v1 + v1.T
                drift = omega + m @ v_single + v_single @ m.T
                return v_single + dt_ * drift + v1
            
            # Apply to all paths
            new_v = vmap(update_single)(v, dw)
            
            return (new_v, new_t, k), None
        
        # Determine number of substeps
        n_substeps = jnp.ceil((target_time - last_time) / dt).astype(jnp.int32)
        
        # Run substeps
        (final_v, _, new_key), _ = jax.lax.scan(
            substep_fn, (current_v, last_time, key), None, length=n_substeps
        )
        
        return (final_v, new_key), final_v
    
    # Create time pairs
    time_pairs = jnp.stack([time_list[start_idx:-1], time_list[start_idx+1:]], axis=1)
    
    # Run simulation
    _, results = jax.lax.scan(scan_fn, (initial_state, key), time_pairs)
    
    # Combine initial state with results
    all_results = jnp.concatenate([initial_state[jnp.newaxis, :, :, :], results], axis=0)
    
    # Transpose to match expected format
    return jnp.transpose(all_results, (1, 0, 2, 3))


@partial(pmap, static_broadcasted_argnums=(1, 2, 3, 4, 5, 6, 7))
def simulate_wishart_jax_parallel(
    keys: jax.random.PRNGKey,
    x: jnp.ndarray, 
    alpha: jnp.ndarray, 
    b: jnp.ndarray, 
    a: jnp.ndarray, 
    start_time: float,
    time_list: jnp.ndarray, 
    num_paths: int = 10, 
    dt: float = 1.0/360.0
) -> jnp.ndarray:
    """
    Parallel JAX implementation for multi-device execution.
    
    Designed for:
    - Multi-GPU setups
    - TPU pods
    - Large-scale simulations
    
    Each device simulates a subset of paths.
    
    Parameters
    ----------
    keys : jax.random.PRNGKey
        Array of random keys, one per device
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
        Number of simulation paths per device
    dt : float
        Time step size
        
    Returns
    -------
    jnp.ndarray
        Simulated paths with shape (n_devices, num_paths, n_times, dim, dim)
    """
    # This function runs on each device with its own key
    local_key = keys
    
    # Convert inputs to JAX arrays
    x = jnp.asarray(x)
    alpha = jnp.asarray(alpha)
    b = jnp.asarray(b)
    a = jnp.asarray(a)
    time_list = jnp.asarray(time_list)
    
    # Each device simulates numPaths paths
    return simulate_wishart_jax_scan(x, alpha, b, a, start_time, time_list, num_paths, dt, local_key)


@partial(jit, static_argnums=(10, 11, 12))
def simulate_wishart_jump_euler_maruyama(
    x: jnp.ndarray, 
    alpha: jnp.ndarray, 
    b: jnp.ndarray, 
    a: jnp.ndarray, 
    lambda_intensity: float,
    nu: float,
    eta: jnp.ndarray,
    xi: jnp.ndarray,
    start_time: float,
    time_list: jnp.ndarray, 
    num_paths: int, 
    dt: float, 
    start_idx: int,
    key: Optional[jax.random.PRNGKey] = None
) -> jnp.ndarray:
    """
    Fast vectorized JAX implementation with jumps and floor method.
    
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
    lambda_intensity : float
        Jump intensity
    nu : float
        Jump size parameter
    eta : jnp.ndarray
        Jump direction matrix
    xi : jnp.ndarray
        Jump offset matrix
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
    lambda_intensity = jnp.asarray(lambda_intensity)
    nu = jnp.asarray(nu)
    eta = jnp.asarray(eta)
    xi = jnp.asarray(xi)
    time_list = jnp.asarray(time_list)
    
    # Add startTime if not present
    time_list = ensure_start_time_in_list(time_list, start_time)
    
    # Pre-allocate results
    n_times = len(time_list)
    dim = x.shape[0]
    results = jnp.zeros((num_paths, n_times, dim, dim))
    
    # Initialize
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
            
            # Generate random increments
            key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
            dw = jax.random.normal(subkey1, shape=(num_paths, dim, dim)) * sqrt_dt
            
            # Jump process
            n_jumps = jax.random.poisson(subkey2, lambda_intensity * dt_, shape=(num_paths,))
            jump_sizes = jax.random.exponential(subkey3, shape=(num_paths, dim, dim))
            
            # Vectorized operations with jumps
            def single_path_update_jump(v_single, dw_single, n_jump, jump_size):
                # Direct matrix square root
                sqrt_v = jax.scipy.linalg.sqrtm(v_single)
                # Ensure real output
                sqrt_v = jnp.real(sqrt_v)
                
                # Diffusion term
                v1 = sqrt_v @ dw_single @ (sigma * jnp.exp(-TIMEDECAYVOL * current_t))
                v1 = v1 + v1.T
                
                # Drift term
                drift = omega + m @ v_single + v_single @ m.T
                
                # Jump term - fixed for scalar nu
                # nu is scalar, so we scale the jump matrix
                jump_matrix = jump_size @ eta + xi/nu  # Scale xi by 1/nu if needed
                jump_term = n_jump * nu * jump_matrix  # nu as scalar multiplier
                
                # Update
                v_new = v_single + dt_ * drift + v1 + jump_term
                
                # PSD correction AFTER the step (floor method)
                return nearest_psd_jax(v_new)
            
            # Apply to all paths
            new_v = vmap(single_path_update_jump)(current_v, dw, n_jumps, jump_sizes)
            
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


@partial(jit, static_argnums=(5, 6))
def simulate_wishart_floor_jax_scan(
    x: jnp.ndarray, 
    alpha: jnp.ndarray, 
    b: jnp.ndarray, 
    a: jnp.ndarray, 
    start_time: float,
    time_list: jnp.ndarray, 
    num_paths: int = 10, 
    dt: float = 1.0/360.0, 
    key: Optional[jax.random.PRNGKey] = None
) -> jnp.ndarray:
    """
    Ultra-fast floor method using JAX scan.
    This is typically the fastest implementation.
    
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
    key : Optional[jax.random.PRNGKey]
        Random key for reproducibility
        
    Returns
    -------
    jnp.ndarray
        Simulated paths with shape (num_paths, n_times, dim, dim)
    """
    if key is None:
        key = jax.random.PRNGKey(42)
    
    # Convert inputs
    x = jnp.asarray(x)
    alpha = jnp.asarray(alpha)
    b = jnp.asarray(b)
    a = jnp.asarray(a)
    time_list = jnp.asarray(time_list)
    
    # Add startTime if needed
    if not jnp.any(jnp.isclose(time_list, start_time)):
        insert_index = jnp.searchsorted(time_list, start_time)
        time_list = jnp.insert(time_list, insert_index, start_time)
    
    start_idx = jnp.where(jnp.isclose(time_list, start_time))[0][0]
    
    # Constants
    sigma = a
    omega = alpha
    m = b
    dim = x.shape[0]
    
    # Initialize
    initial_state = jnp.tile(x[jnp.newaxis, :, :], (num_paths, 1, 1))
    
    def scan_fn(carry, time_pair):
        current_v, key = carry
        last_time, target_time = time_pair
        
        def substep_fn(state, _):
            v, t, k = state
            dt_ = jnp.minimum(dt, target_time - t)
            new_t = t + dt_
            sqrt_dt = jnp.sqrt(dt_)
            
            # Random generation
            k, subkey = jax.random.split(k)
            dw = jax.random.normal(subkey, shape=(num_paths, dim, dim)) * sqrt_dt
            
            # Update function
            def update_single_floor(v_single, dw_single):
                sqrt_v = jax.scipy.linalg.sqrtm(v_single)
                v1 = sqrt_v @ dw_single @ (sigma * jnp.exp(-TIMEDECAYVOL * new_t))
                v1 = v1 + v1.T
                drift = omega + m @ v_single + v_single @ m.T
                v_new = v_single + dt_ * drift + v1
                return nearest_psd_jax(v_new)
            
            new_v = vmap(update_single_floor)(v, dw)
            
            return (new_v, new_t, k), None
        
        n_substeps = jnp.ceil((target_time - last_time) / dt).astype(jnp.int32)
        
        (final_v, _, new_key), _ = jax.lax.scan(
            substep_fn, (current_v, last_time, key), None, length=n_substeps
        )
        
        return (final_v, new_key), final_v
    
    # Time pairs
    time_pairs = jnp.stack([time_list[start_idx:-1], time_list[start_idx+1:]], axis=1)
    
    # Run simulation
    _, results = jax.lax.scan(scan_fn, (initial_state, key), time_pairs)
    
    # Combine results
    all_results = jnp.concatenate([initial_state[jnp.newaxis, :, :, :], results], axis=0)
    
    return jnp.transpose(all_results, (1, 0, 2, 3))


# Wrapper function for jump simulation
def simulate_wishart_jump_euler_maruyama_wrapper(
    x: Union[np.ndarray, jnp.ndarray], 
    alpha: Union[np.ndarray, jnp.ndarray], 
    b: Union[np.ndarray, jnp.ndarray], 
    a: Union[np.ndarray, jnp.ndarray], 
    lambda_intensity: float,
    nu: float,
    eta: Union[np.ndarray, jnp.ndarray],
    xi: Union[np.ndarray, jnp.ndarray],
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
    return simulate_wishart_jump_euler_maruyama(
        x, alpha, b, a, lambda_intensity, nu, eta, xi,
        start_time, time_list, num_paths, dt, start_idx, key
    )
