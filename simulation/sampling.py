"""
Random sampling functions for Wishart process simulations.
"""
import jax
import jax.numpy as jnp
from jax import random
from typing import Optional


def sample_from_eq_227(
    d: int, 
    delta_min: float, 
    key: jax.random.PRNGKey
) -> jnp.ndarray:
    """
    Sample a symmetric matrix with specific distributional properties.
    
    Samples a d-by-d symmetric matrix G_hat with independent elements
    distributed according to the law described in Equation (2.27) of
    Alfonsi's book.
    
    Parameters
    ----------
    d : int
        Dimension of the matrix
    delta_min : float
        Minimum eigenvalue parameter (affects diagonal elements)
    key : jax.random.PRNGKey
        JAX random key for reproducibility
        
    Returns
    -------
    jnp.ndarray
        A d-by-d symmetric matrix with:
        - Diagonal elements ~ Chi distribution with parameter delta_min
        - Off-diagonal elements ~ N(0, 1/2)
        
    Notes
    -----
    The Chi distribution for diagonal elements is implemented as
    sqrt(2 * Gamma(delta_min/2)), which gives the correct distribution.
    
    References
    ----------
    A. Alfonsi, "Affine Diffusions and Related Processes", Equation (2.27)
    """
    g_hat = jnp.zeros((d, d))
    
    # Generate all random values at once for efficiency
    key1, key2 = random.split(key)
    
    # Diagonal elements: Chi distribution with parameter delta_min
    # Chi(delta_min) = sqrt(2 * Gamma(delta_min/2))
    chi_samples = jnp.sqrt(random.gamma(key1, delta_min/2, (d,)) * 2)
    
    # Off-diagonal elements: standard normal / sqrt(2)
    normal_samples = random.normal(key2, (d, d)) / jnp.sqrt(2)
    
    # Build symmetric matrix
    for i in range(d):
        for j in range(i, d):
            if i == j:
                # Diagonal elements
                g_hat = g_hat.at[i, j].set(chi_samples[i])
            else:
                # Off-diagonal elements (ensure symmetry)
                g_hat = g_hat.at[i, j].set(normal_samples[i, j])
                g_hat = g_hat.at[j, i].set(normal_samples[i, j])
                
    return g_hat


def sample_wishart_marginal(
    df: float,
    scale: jnp.ndarray,
    key: jax.random.PRNGKey
) -> jnp.ndarray:
    """
    Sample from Wishart distribution using Bartlett decomposition.
    
    Parameters
    ----------
    df : float
        Degrees of freedom (must be > dimension - 1)
    scale : jnp.ndarray
        Scale matrix (d x d), must be positive definite
    key : jax.random.PRNGKey
        Random key
        
    Returns
    -------
    jnp.ndarray
        Sample from Wishart(df, scale) distribution
        
    Notes
    -----
    Uses the Bartlett decomposition for efficient sampling.
    """
    d = scale.shape[0]
    
    # Cholesky decomposition of scale matrix
    L = jnp.linalg.cholesky(scale)
    
    # Generate Bartlett matrix
    A = jnp.zeros((d, d))
    
    # Split keys for different random components
    key1, key2 = random.split(key)
    
    # Diagonal elements: sqrt of chi-squared
    for i in range(d):
        key1, subkey = random.split(key1)
        A = A.at[i, i].set(jnp.sqrt(random.gamma(subkey, (df - i) / 2) * 2))
    
    # Off-diagonal elements: standard normal
    normal_samples = random.normal(key2, (d, d))
    for i in range(d):
        for j in range(i):
            A = A.at[i, j].set(normal_samples[i, j])
    
    # Compute Wishart sample
    B = L @ A
    return B @ B.T


def sample_matrix_normal(
    mean: jnp.ndarray,
    row_cov: jnp.ndarray,
    col_cov: jnp.ndarray,
    key: jax.random.PRNGKey
) -> jnp.ndarray:
    """
    Sample from matrix normal distribution.
    
    Parameters
    ----------
    mean : jnp.ndarray
        Mean matrix (m x n)
    row_cov : jnp.ndarray
        Row covariance matrix (m x m)
    col_cov : jnp.ndarray
        Column covariance matrix (n x n)
    key : jax.random.PRNGKey
        Random key
        
    Returns
    -------
    jnp.ndarray
        Sample from MatrixNormal(mean, row_cov, col_cov)
    """
    m, n = mean.shape
    
    # Generate standard normal matrix
    Z = random.normal(key, (m, n))
    
    # Apply covariances
    L_row = jnp.linalg.cholesky(row_cov)
    L_col = jnp.linalg.cholesky(col_cov)
    
    return mean + L_row @ Z @ L_col.T


def sample_matrix_t(
    df: float,
    location: jnp.ndarray,
    scale: jnp.ndarray,
    key: jax.random.PRNGKey
) -> jnp.ndarray:
    """
    Sample from matrix t-distribution.
    
    Parameters
    ----------
    df : float
        Degrees of freedom
    location : jnp.ndarray
        Location matrix (d x d)
    scale : jnp.ndarray
        Scale matrix (d x d)
    key : jax.random.PRNGKey
        Random key
        
    Returns
    -------
    jnp.ndarray
        Sample from matrix t-distribution
    """
    d = location.shape[0]
    
    # Split keys
    key1, key2 = random.split(key)
    
    # Sample from Wishart
    W = sample_wishart_marginal(df, jnp.linalg.inv(scale), key1)
    
    # Sample from matrix normal
    Z = sample_matrix_normal(
        jnp.zeros_like(location),
        jnp.eye(d),
        jnp.eye(d),
        key2
    )
    
    # Combine to get matrix t
    W_inv_sqrt = jnp.linalg.inv(jnp.linalg.cholesky(W))
    return location + W_inv_sqrt @ Z


def sample_matrix_beta(
    a: float,
    b: float,
    d: int,
    key: jax.random.PRNGKey
) -> jnp.ndarray:
    """
    Sample from matrix beta distribution.
    
    Parameters
    ----------
    a : float
        First shape parameter
    b : float
        Second shape parameter
    d : int
        Dimension
    key : jax.random.PRNGKey
        Random key
        
    Returns
    -------
    jnp.ndarray
        Sample from matrix beta distribution
    """
    key1, key2 = random.split(key)
    
    # Sample two independent Wisharts
    W1 = sample_wishart_marginal(a, jnp.eye(d), key1)
    W2 = sample_wishart_marginal(b, jnp.eye(d), key2)
    
    # Matrix beta is W1 / (W1 + W2)
    W_sum = W1 + W2
    W_sum_inv_sqrt = jnp.linalg.inv(jnp.linalg.cholesky(W_sum))
    W1_sqrt = jnp.linalg.cholesky(W1)
    
    return W_sum_inv_sqrt @ W1_sqrt @ W1_sqrt.T @ W_sum_inv_sqrt.T
