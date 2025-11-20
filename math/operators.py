"""
Mathematical operators for Wishart process computations.
"""
import jax.numpy as jnp
from jax.scipy.linalg import expm
from jax.scipy.special import factorial
from scipy.integrate import quad
from typing import Callable


def bu_operator(
    b: jnp.ndarray, 
    u: jnp.ndarray, 
    inv_u: jnp.ndarray, 
    k: int, 
    x: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute the k-th power of the Bu operator.
    
    The Bu operator is defined as:
        Bu(X) = inv_u.T @ (B @ (u.T @ X @ u) + (u.T @ X @ u) @ B.T) @ inv_u
        
    Parameters
    ----------
    b : jnp.ndarray
        Matrix B in the operator definition (d x d)
    u : jnp.ndarray
        Transformation matrix (d x d)
    inv_u : jnp.ndarray
        Inverse of u matrix (d x d)
    k : int
        Power of the operator (non-negative)
    x : jnp.ndarray
        Input matrix (d x d)
        
    Returns
    -------
    jnp.ndarray
        Result of Bu^k(X)
        
    Notes
    -----
    For k=0, returns the transformed X.
    For k=1, applies the Bu operator once.
    For k>1, uses matrix power for efficiency.
    """
    d = x.shape[0]
    if k == 0:
        return x
    elif k == 1:
        u_t = u.T
        temp = u_t @ x @ u
        return inv_u.T @ (b @ temp + temp @ b.T) @ inv_u
    else:
        # Compute B^k using matrix power
        u_t = u.T
        temp = u_t @ x @ u
        b_k = jnp.linalg.matrix_power(b, k)
        return inv_u.T @ (b_k @ temp + temp @ b_k.T) @ inv_u


def initialize_bu(
    b: jnp.ndarray, 
    u: jnp.ndarray, 
    inv_u: jnp.ndarray, 
    x: jnp.ndarray
) -> jnp.ndarray:
    """
    Initialize the Bu operator for matrix exponential computation.
    
    Computes: B @ (u.T @ X @ u) + (u.T @ X @ u) @ B.T
    
    Parameters
    ----------
    b : jnp.ndarray
        Matrix B (d x d)
    u : jnp.ndarray
        Transformation matrix (d x d)
    inv_u : jnp.ndarray
        Inverse of u (d x d)
    x : jnp.ndarray
        Input matrix (d x d)
        
    Returns
    -------
    jnp.ndarray
        Initialized operator result
    """
    u_t = u.T
    mid = u_t @ x @ u
    return b @ mid + mid @ b.T


def x_ode_x(
    t: float, 
    b: jnp.ndarray, 
    u: jnp.ndarray, 
    inv_u: jnp.ndarray, 
    x: jnp.ndarray, 
    delta_bar_delta_min: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute X_t^{ODE, x} solution.
    
    Solves the ODE:
        X_t = exp(t * Bu) * X + integral_0^t exp(s * Bu) * delta_bar ds
        
    Parameters
    ----------
    t : float
        Time parameter
    b : jnp.ndarray
        Matrix B (d x d)
    u : jnp.ndarray
        Transformation matrix (d x d)
    inv_u : jnp.ndarray
        Inverse of u (d x d)
    x : jnp.ndarray
        Initial condition (d x d)
    delta_bar_delta_min : jnp.ndarray
        Drift adjustment matrix (d x d)
        
    Returns
    -------
    jnp.ndarray
        Solution at time t
    """
    d = b.shape[0]
     
    # First term: exp(t * Bu) * x
    initial = initialize_bu(b, u, inv_u, x)
    first_term = expm(t * initial)

    second_part = initialize_bu(b, u, inv_u, delta_bar_delta_min)

    # Second term: integral from 0 to t of exp(s * Bu) * delta_bar ds
    def integrand(s):
        return expm(s * second_part)

    # Integrate each component
    second_term = jnp.zeros((d, d))
    for i in range(d):
        for j in range(d):
            func_i_j = lambda s: integrand(s)[i, j]
            second_term_ij, _ = quad(func_i_j, 0, t)
            second_term = second_term.at[i, j].set(second_term_ij)

    return first_term + second_term


def truncated_series_bu(
    b: jnp.ndarray, 
    u: jnp.ndarray, 
    inv_u: jnp.ndarray, 
    x: jnp.ndarray, 
    delta: jnp.ndarray, 
    delta_min_eye: jnp.ndarray, 
    t: float, 
    max_k: int = 10
) -> jnp.ndarray:
    """
    Compute truncated series expansion of the Bu operator.
    
    Computes the sum:
        Sum_{k=0}^{max_k} [(t/2)^k / k!] * Bu^k(X) + 
        Sum_{k=0}^{max_k} [(t/2)^(k+1) / (k+1)!] * Bu^k(delta - delta_min*I)
        
    Parameters
    ----------
    b : jnp.ndarray
        Matrix B (d x d)
    u : jnp.ndarray
        Transformation matrix (d x d)
    inv_u : jnp.ndarray
        Inverse of u (d x d)
    x : jnp.ndarray
        Input matrix (d x d)
    delta : jnp.ndarray
        Delta matrix (d x d)
    delta_min_eye : jnp.ndarray
        Minimum eigenvalue times identity (d x d)
    t : float
        Time parameter
    max_k : int, optional
        Maximum order of truncation
        
    Returns
    -------
    jnp.ndarray
        Truncated series result
        
    Notes
    -----
    This truncation provides a second-order approximation to the
    exact matrix exponential solution.
    """
    d = x.shape[0]
    delta_minus_delta_min_eye = delta - delta_min_eye
    
    result = jnp.zeros_like(x)
    for k in range(max_k + 1):
        term1 = (((t/2)**k) / factorial(k)) * bu_operator(b, u, inv_u, k, x)
        term2 = (((t/2)**(k + 1)) / factorial(k + 1)) * bu_operator(b, u, inv_u, k, delta_minus_delta_min_eye)
        result += term1 + term2
        
    return result


def compute_generator_matrix(
    alpha_bar: jnp.ndarray,
    b: jnp.ndarray,
    x: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute the infinitesimal generator matrix for the affine diffusion.
    
    Parameters
    ----------
    alpha_bar : jnp.ndarray
        Constant drift matrix (d x d)
    b : jnp.ndarray
        Linear coefficient matrix (d x d)
    x : jnp.ndarray
        Current state matrix (d x d)
        
    Returns
    -------
    jnp.ndarray
        Generator matrix
    """
    return alpha_bar + b @ x + x @ b.T


def matrix_exponential_action(
    a: jnp.ndarray,
    x: jnp.ndarray,
    t: float,
    method: str = "exact"
) -> jnp.ndarray:
    """
    Compute the action of matrix exponential exp(t*A) on matrix X.
    
    Parameters
    ----------
    a : jnp.ndarray
        Generator matrix (d x d)
    x : jnp.ndarray
        Input matrix (d x d)
    t : float
        Time parameter
    method : str, optional
        Method to use: "exact" or "truncated"
        
    Returns
    -------
    jnp.ndarray
        Result of exp(t*A) @ X
    """
    if method == "exact":
        return expm(t * a) @ x
    else:
        # Truncated Taylor series
        result = x
        term = x
        for k in range(1, 20):
            term = (t / k) * (a @ term)
            result += term
        return result
