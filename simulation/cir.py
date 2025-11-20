"""
Cox-Ingersoll-Ross (CIR) process simulation.
"""
import jax
import jax.numpy as jnp
from jax import random, jit
from typing import Optional


@jit
def simulate_cir(
    u0: float, 
    alpha: float, 
    t: float, 
    vol_mul: float = 2.0, 
    steps: int = 1000, 
    key: Optional[jax.random.PRNGKey] = None
) -> float:
    """
    Simulate CIR process using Euler-Maruyama method with full truncation.
    
    The CIR process satisfies the SDE:
        du_t = alpha * dt + vol_mul * sqrt(u_t) * dW_t
        
    with full truncation ensuring non-negativity.
    
    Parameters
    ----------
    u0 : float
        Initial value (must be non-negative)
    alpha : float
        Drift parameter
    t : float
        Terminal time
    vol_mul : float, optional
        Volatility multiplier (default is 2.0 for standard CIR)
    steps : int, optional
        Number of time steps for discretization
    key : Optional[jax.random.PRNGKey]
        Random key for reproducibility
        
    Returns
    -------
    float
        Terminal value of the CIR process at time t
        
    Notes
    -----
    The full truncation scheme ensures that the process remains non-negative
    by applying max(u, 0) after each step and in the volatility term.
    
    References
    ----------
    Cox, J.C., Ingersoll, J.E., Ross, S.A. (1985). "A Theory of the Term Structure 
    of Interest Rates". Econometrica 53(2): 385-407.
    """
    if key is None:
        key = random.PRNGKey(42)

    dt = t / steps
    u = u0
    
    def step(u, key):
        """Single step of the Euler-Maruyama scheme."""
        dw = jnp.sqrt(dt) * random.normal(key)
        # Full truncation: ensure non-negativity in volatility term and result
        u_next = jnp.maximum(
            u + (alpha * dt) + vol_mul * jnp.sqrt(jnp.maximum(u, 0)) * dw, 
            0
        )
        return u_next, u_next
    
    # Generate all random keys
    keys = random.split(key, steps)
    
    # Run simulation using scan for efficiency
    _, u_path = jax.lax.scan(step, u, keys)
    
    # Return terminal value
    return u_path[-1]


def simulate_cir_path(
    u0: float, 
    alpha: float, 
    times: jnp.ndarray,
    vol_mul: float = 2.0, 
    dt: float = 0.01,
    key: Optional[jax.random.PRNGKey] = None
) -> jnp.ndarray:
    """
    Simulate full path of CIR process at specified times.
    
    Parameters
    ----------
    u0 : float
        Initial value
    alpha : float
        Drift parameter
    times : jnp.ndarray
        Array of times at which to record the process
    vol_mul : float, optional
        Volatility multiplier
    dt : float, optional
        Time step for discretization
    key : Optional[jax.random.PRNGKey]
        Random key
        
    Returns
    -------
    jnp.ndarray
        Values of the CIR process at specified times
    """
    if key is None:
        key = random.PRNGKey(42)
        
    # Ensure times are sorted
    times = jnp.sort(times)
    values = jnp.zeros_like(times)
    
    # Initial value
    if times[0] == 0:
        values = values.at[0].set(u0)
        start_idx = 1
    else:
        start_idx = 0
        
    current_u = u0
    current_t = 0.0
    
    for i in range(start_idx, len(times)):
        # Simulate from current_t to times[i]
        time_diff = times[i] - current_t
        steps = int(jnp.ceil(time_diff / dt))
        
        key, subkey = random.split(key)
        current_u = simulate_cir(current_u, alpha, time_diff, vol_mul, steps, subkey)
        values = values.at[i].set(current_u)
        current_t = times[i]
        
    return values


def simulate_cir_bridge(
    u_start: float,
    u_end: float,
    t_start: float,
    t_end: float,
    alpha: float,
    vol_mul: float = 2.0,
    n_points: int = 100,
    key: Optional[jax.random.PRNGKey] = None
) -> jnp.ndarray:
    """
    Simulate CIR bridge process conditioned on start and end values.
    
    Parameters
    ----------
    u_start : float
        Starting value
    u_end : float
        Ending value (condition)
    t_start : float
        Starting time
    t_end : float
        Ending time
    alpha : float
        Drift parameter
    vol_mul : float, optional
        Volatility multiplier
    n_points : int, optional
        Number of intermediate points
    key : Optional[jax.random.PRNGKey]
        Random key
        
    Returns
    -------
    jnp.ndarray
        Bridge path values
        
    Notes
    -----
    This is an approximate bridge using rejection sampling or
    importance sampling techniques.
    """
    # Placeholder for more sophisticated bridge sampling
    # For now, return linear interpolation
    times = jnp.linspace(t_start, t_end, n_points)
    weights = (times - t_start) / (t_end - t_start)
    values = (1 - weights) * u_start + weights * u_end
    
    # Add some noise scaled by time
    if key is not None:
        noise = 0.1 * jnp.sqrt(values) * random.normal(key, (n_points,))
        values = jnp.maximum(values + noise, 0)
    
    return values
