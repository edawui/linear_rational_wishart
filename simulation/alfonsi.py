"""
Alfonsi's algorithms for Wishart process simulation.

Based on algorithms from Aurelien Alfonsi's book "Affine Diffusions and Related Processes".
"""
import jax
import jax.numpy as jnp
from jax import random, jit
from jax.scipy.linalg import expm
import numpy as np
import pandas as pd
from typing import Optional, Dict, Union
from functools import partial

from ..math.decompositions import extended_cholesky, safe_cholesky, compute_u_delta
from ..math.operators import truncated_series_bu
from ..math.psd_corrections import nearest_psd
from .cir import simulate_cir
from .sampling import sample_from_eq_227


def simulate_wishart_e1_5_1(
    x: jnp.ndarray, 
    alpha: float, 
    t: float, 
    key: Optional[jax.random.PRNGKey] = None
) -> jnp.ndarray:
    """
    Simulate from WIS_d(x, alpha, 0, e_1; t) using Algorithm 5.1.
    
    This algorithm simulates a Wishart process with specific structure where
    the volatility matrix is e_1 (first canonical basis vector).
    
    Parameters
    ----------
    x : jnp.ndarray
        Initial state matrix (d x d), must be symmetric positive semidefinite
    alpha : float
        Degrees of freedom parameter, must be >= d - 1
    t : float
        Time parameter
    key : Optional[jax.random.PRNGKey]
        Random key for reproducibility
        
    Returns
    -------
    jnp.ndarray
        Simulated matrix at time t
        
    References
    ----------
    A. Alfonsi, "Affine Diffusions and Related Processes", Algorithm 5.1
    """
    if key is None:
        key = random.PRNGKey(42)

    d = x.shape[0]

    # Step 1: Extended Cholesky decomposition
    xx = x.copy()    
    p, k_r, c_r = extended_cholesky(xx[1:, 1:]) 

    # Step 2: Permutation matrix π
    pi_mat = jnp.zeros((d, d))
    pi_mat = pi_mat.at[0, 0].set(1)
    pi_mat = pi_mat.at[1:, 1:].set(p)
    r = c_r.shape[0]

    # Step 3: Tilde x = π x π^T
    x_tilde = pi_mat @ x @ pi_mat.T

    # Step 4: Compute u_tilde values
    u_mat = jnp.linalg.pinv(c_r) @ (x_tilde[0, 1:(r+1)]).T
    u_11_sq = x_tilde[0, 0] + jnp.sum(u_mat[:]**2)
    
    u_tilde = jnp.zeros((1, r+1))
    u_tilde = u_tilde.at[0, 1:r+1].set(u_mat)
    u_tilde = u_tilde.at[0, 0].set(u_11_sq)
    u_tilde = u_tilde.T

    u_11 = jnp.sqrt(jnp.maximum(u_11_sq, 0))

    # Step 5: CIR process
    key, subkey = random.split(key)
    u_11_t = simulate_cir(u_11, alpha - r, t, vol_mul=2.0, steps=1000, key=subkey)

    # Step 6: Generate r standard normals G2..Gr+1
    key, subkey = random.split(key)
    g = random.normal(subkey, (r,))
    u_row = u_mat + jnp.sqrt(t) * g
    u_row = u_row.reshape(1, -1)  # (1, r)

    # Step 7: Build middle symmetric matrix B
    u_11_sq = u_11_t**2
    id_r = jnp.eye(r)

    # Top-left element of B
    top_left = u_11_sq + jnp.sum(u_row**2)

    # Regular case
    first_col = jnp.zeros((d, 1))
    first_col = first_col.at[0, 0].set(top_left)
    first_col = first_col.at[1:(r+1), 0:1].set(u_row.T)

    middle = jnp.zeros((d, r))
    middle = middle.at[0:1, 0:r].set(u_row)
    middle = middle.at[1:(r+1), 0:r].set(id_r)

    last_part = jnp.zeros((d, d - r - 1))
    b = jnp.hstack([first_col, middle, last_part])

    a = jnp.zeros((d, d))
    a = a.at[0, 0].set(1.0)
    a = a.at[1:(r+1), 1:(r + 1)].set(c_r)
    a = a.at[(r+1):, 1:(r + 1)].set(k_r)
    a = a.at[(r+1):, (r + 1):].set(jnp.eye(d - r - 1))

    # Core matrix
    x_core = a @ b @ a.T

    # Final result with permutation
    result = pi_mat.T @ x_core @ pi_mat

    return result


def simulate_wishart_e1_5_4(
    x: jnp.ndarray, 
    alpha: float, 
    t: float, 
    key: Optional[jax.random.PRNGKey] = None
) -> jnp.ndarray:
    """
    Simulate from WIS_d(x, alpha, 0, e_1; t) using Algorithm 5.4.
    
    Alternative implementation of Algorithm 5.1 with different sampling approach.
    
    Parameters
    ----------
    x : jnp.ndarray
        Initial state matrix (d x d), must be symmetric positive semidefinite
    alpha : float
        Degrees of freedom parameter, must be >= d - 1
    t : float
        Time parameter
    key : Optional[jax.random.PRNGKey]
        Random key for reproducibility
        
    Returns
    -------
    jnp.ndarray
        Simulated matrix at time t
        
    References
    ----------
    A. Alfonsi, "Affine Diffusions and Related Processes", Algorithm 5.4
    """
    if key is None:
        key = random.PRNGKey(42)

    d = x.shape[0]

    # Step 1: Extended Cholesky decomposition
    xx = x.copy()    
    p, k_r, c_r = extended_cholesky(xx[1:, 1:]) 

    # Step 2: Permutation matrix π
    pi_mat = jnp.zeros((d, d))
    pi_mat = pi_mat.at[0, 0].set(1)
    pi_mat = pi_mat.at[1:, 1:].set(p)
    r = c_r.shape[0]

    # Step 3: Tilde x = π x π^T
    x_tilde = pi_mat @ x @ pi_mat.T

    # Step 4: Compute u_tilde values
    u_mat = jnp.linalg.pinv(c_r) @ (x_tilde[0, 1:(r+1)]).T
    u_11_sq = x_tilde[0, 0] + jnp.sum(u_mat[:]**2)
    
    u_tilde = jnp.zeros((1, r+1))
    u_tilde = u_tilde.at[0, 1:r+1].set(u_mat)
    u_tilde = u_tilde.at[0, 0].set(u_11_sq)
    u_tilde = u_tilde.T

    u_11 = jnp.sqrt(jnp.maximum(u_11_sq, 0))

    # Step 5: CIR process
    key, subkey = random.split(key)
    u_11_t = simulate_cir(u_11, alpha - r, t, vol_mul=2.0, steps=1000, key=subkey)

    # Step 6: Generate r standard normals G2..Gr+1
    key, subkey = random.split(key)
    g = random.normal(subkey, (r,))
    u_row = u_mat + jnp.sqrt(t) * g
    u_row = u_row.reshape(1, -1)  # (1, r)

    # Step 7: Build middle symmetric matrix B
    u_11_sq = u_11_t**2
    id_r = jnp.eye(r)

    # Top-left element of B
    top_left = u_11_sq + jnp.sum(u_row**2)

    # Regular case
    first_col = jnp.zeros((d, 1))
    first_col = first_col.at[0, 0].set(top_left)
    first_col = first_col.at[1:(r+1), 0:1].set(u_row.T)

    middle = jnp.zeros((d, r))
    middle = middle.at[0:1, 0:r].set(u_row)
    middle = middle.at[1:(r+1), 0:r].set(id_r)

    last_part = jnp.zeros((d, d - r - 1))
    b = jnp.hstack([first_col, middle, last_part])

    a = jnp.zeros((d, d))
    a = a.at[0, 0].set(1.0)
    a = a.at[1:(r+1), 1:(r + 1)].set(c_r)
    a = a.at[(r+1):, 1:(r + 1)].set(k_r)
    a = a.at[(r+1):, (r + 1):].set(jnp.eye(d - r - 1))

    # Core matrix
    x_core = a @ b @ a.T

    # Final result with permutation
    result = pi_mat.T @ x_core @ pi_mat

    return result


def simulate_wishart_idn_5_2(
    x: jnp.ndarray, 
    alpha: float, 
    t: float, 
    key: Optional[jax.random.PRNGKey] = None
) -> jnp.ndarray:
    """
    Simulate from WIS_d(x, alpha, 0, I_d^n; t) using Algorithm 5.2.
    
    This algorithm builds on Algorithm 5.1 to simulate a Wishart process
    with identity volatility structure.
    
    Parameters
    ----------
    x : jnp.ndarray
        Initial state matrix (d x d), must be symmetric positive semidefinite
    alpha : float
        Degrees of freedom parameter, must be >= d - 1
    t : float
        Time parameter
    key : Optional[jax.random.PRNGKey]
        Random key for reproducibility
        
    Returns
    -------
    jnp.ndarray
        Simulated matrix at time t
        
    References
    ----------
    A. Alfonsi, "Affine Diffusions and Related Processes", Algorithm 5.2
    """
    if key is None:
        key = random.PRNGKey(42)

    d = x.shape[0]
    n = d  # Full identity matrix
    y = x.copy()

    for k in range(n):
        # Build permutation matrix that swaps 0 and k
        p = jnp.eye(d)
        p = p.at[0, k].set(1)
        p = p.at[k, 0].set(1)
        p = p.at[0, 0].set(0)

        y_perm = p @ y @ p
        
        # Apply Algorithm 5.1 on permuted matrix
        key, subkey = random.split(key)
        y_new = simulate_wishart_e1_5_1(y_perm, alpha, t, subkey)

        # Reverse permutation
        y = p @ y_new @ p

    return y


def path_fast_second_order_aff_5_6_step(
    start_x: jnp.ndarray, 
    alpha_bar: jnp.ndarray, 
    b: jnp.ndarray, 
    a: jnp.ndarray, 
    t_step: float, 
    max_k: int = 10, 
    key: Optional[jax.random.PRNGKey] = None
) -> jnp.ndarray:
    """
    Simulate a single time step using Algorithm 5.6.
    
    Fast second-order scheme for matrix-valued affine diffusion processes.
    
    Parameters
    ----------
    start_x : jnp.ndarray
        Current state matrix (d x d), symmetric positive semidefinite
    alpha_bar : jnp.ndarray
        Constant drift matrix (d x d)
    b : jnp.ndarray
        Matrix governing the bilinear generator operator (d x d)
    a : jnp.ndarray
        Diffusion scaling matrix (d x d), symmetric positive definite
    t_step : float
        Time increment
    max_k : int, optional
        Maximum number of terms in truncated series expansion
    key : Optional[jax.random.PRNGKey]
        Random key for reproducibility
        
    Returns
    -------
    jnp.ndarray
        Updated matrix value at next time step
        
    References
    ----------
    A. Alfonsi, "Affine Diffusions and Related Processes", Algorithm 5.6
    """
    if key is None:
        key = random.PRNGKey(42)

    d = start_x.shape[0]

    u, delta = compute_u_delta(alpha=alpha_bar, a=a)
    inv_u = jnp.linalg.inv(u)

    x_tilde = inv_u.T @ start_x @ inv_u
    delta_min_eye = jnp.min(jnp.diag(delta)) * jnp.eye(d) 
    delta_min = jnp.min(jnp.diag(delta))

    x_tilde = truncated_series_bu(b, u, inv_u, x_tilde, delta, delta_min_eye, t_step, max_k)
    x_tilde = simulate_wishart_idn_5_2(x_tilde, delta_min, t_step, key) 
    x_tilde = truncated_series_bu(b, u, inv_u, x_tilde, delta, delta_min_eye, t_step, max_k)
    
    x = u.T @ x_tilde @ u
    return x


def path_fast_second_order_aff_5_7_step(
    start_x: jnp.ndarray, 
    alpha_bar: jnp.ndarray, 
    b: jnp.ndarray, 
    a: jnp.ndarray, 
    t_step: float, 
    max_k: int = 10, 
    key: Optional[jax.random.PRNGKey] = None
) -> jnp.ndarray:
    """
    Simulate a single time step using Algorithm 5.7.
    
    Fast second-order scheme with Gaussian approximation.
    
    Parameters
    ----------
    start_x : jnp.ndarray
        Current state matrix (d x d), symmetric positive semidefinite
    alpha_bar : jnp.ndarray
        Constant drift matrix (d x d)
    b : jnp.ndarray
        Matrix governing the bilinear generator operator (d x d)
    a : jnp.ndarray
        Diffusion scaling matrix (d x d), symmetric positive definite
    t_step : float
        Time increment
    max_k : int, optional
        Maximum number of terms in truncated series expansion
    key : Optional[jax.random.PRNGKey]
        Random key for reproducibility
        
    Returns
    -------
    jnp.ndarray
        Updated matrix value at next time step
        
    References
    ----------
    A. Alfonsi, "Affine Diffusions and Related Processes", Algorithm 5.7
    """
    if key is None:
        key = random.PRNGKey(42)

    d = start_x.shape[0]

    u, delta = compute_u_delta(alpha=alpha_bar, a=a)
    inv_u = jnp.linalg.inv(u)

    x_tilde = inv_u.T @ start_x @ inv_u
    delta_min_eye = jnp.min(jnp.diag(delta)) * jnp.eye(d) 
    delta_min = jnp.min(jnp.diag(delta))

    x_tilde = truncated_series_bu(b, u, inv_u, x_tilde, delta, delta_min_eye, t_step, max_k)

    c = safe_cholesky(x_tilde, eps=1e-8, max_tries=10)

    g_hat = random.normal(key, (d, d))
    sqrt_term_g = jnp.sqrt(t_step) * g_hat
    x_tilde = (c + sqrt_term_g).T @ (c + sqrt_term_g)

    x_tilde = truncated_series_bu(b, u, inv_u, x_tilde, delta, delta_min_eye, t_step, max_k)
    x = u.T @ x_tilde @ u
    return x


def simulate_fast_second_order_aff(
    x: jnp.ndarray, 
    alpha_bar: jnp.ndarray, 
    b: jnp.ndarray, 
    a: jnp.ndarray, 
    start_time: float,
    time_list: Union[jnp.ndarray, np.ndarray, list, tuple], 
    num_paths: int = 10, 
    dt_substep: float = 0.01, 
    max_k: int = 10, 
    key: Optional[jax.random.PRNGKey] = None
) -> Dict[int, pd.Series]:
    """
    Simulate multiple paths using fast second-order scheme.
    
    This implementation follows Algorithm 5.7 from Alfonsi's book.
    
    Parameters
    ----------
    x : jnp.ndarray
        Initial matrix state (d x d), symmetric positive semidefinite
    alpha_bar : jnp.ndarray
        Constant drift matrix (d x d)
    b : jnp.ndarray
        Matrix defining the bilinear operator (d x d)
    a : jnp.ndarray
        Diffusion scaling matrix (d x d), symmetric positive definite
    start_time : float
        Starting time of simulation
    time_list : array-like
        Ordered list of time points for recording
    num_paths : int, optional
        Number of independent paths
    dt_substep : float, optional
        Substep size for integration
    max_k : int, optional
        Maximum terms in series expansion
    key : Optional[jax.random.PRNGKey]
        Random key for reproducibility
        
    Returns
    -------
    Dict[int, pd.Series]
        Dictionary mapping path index to time series
        
    References
    ----------
    A. Alfonsi, "Affine Diffusions and Related Processes", Algorithm 5.7
    """
    print(f"simulate_fast_second_order_aff: t = {time_list[0]} dt_substep={dt_substep}")
    
    if key is None:
        key = random.PRNGKey(42)

    time_list = jnp.array(time_list)
    if not jnp.all(jnp.diff(time_list) > 0):
        raise ValueError("time_list must be in increasing order.")

    if not jnp.any(jnp.isclose(time_list, start_time)):
        insert_index = jnp.searchsorted(time_list, start_time)
        time_list = jnp.insert(time_list, insert_index, start_time)

    time_to_index = {float(t): i for i, t in enumerate(time_list)}
    index = time_to_index[float(start_time)]

    shape = x.shape
    sim_results = {
        path: pd.Series({float(t): np.zeros(shape, dtype=np.float64) for t in time_list})
        for path in range(num_paths)
    }

    for path in range(num_paths):
        sim_results[path][float(start_time)][:] = np.array(x)

    # Split key for all paths
    path_keys = random.split(key, num_paths)
   
    for i in range(index + 1, len(time_list)):
        t0 = time_list[i - 1]
        t1 = time_list[i]
        t = t0

        step_sim_results = {path: jnp.array(sim_results[path][float(t0)]) for path in range(num_paths)}

        while t < t1:
            next_t = jnp.minimum(t + dt_substep, t1)
            t_step = next_t - t
            
            # Split keys for this step
            path_keys = [random.split(k)[1] for k in path_keys]
            
            for path in range(num_paths):
                x_path = step_sim_results[path]
                step_sim_results[path] = path_fast_second_order_aff_5_7_step(
                    x_path, alpha_bar, b, a, t_step, max_k, path_keys[path]
                )
                
            t = next_t

        for path in range(num_paths):
            sim_results[path][float(t1)][:] = np.array(step_sim_results[path])

    return sim_results
