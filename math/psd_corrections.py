"""
Positive Semi-Definite (PSD) correction methods for matrix operations.
"""
import jax.numpy as jnp
from jax import jit
from jax.scipy.linalg import sqrtm as jax_sqrtm, cholesky
from typing import Optional


@jit
def nearest_psd_jax(matrix: jnp.ndarray, epsilon: float = 1e-8) -> jnp.ndarray:
    """
    Fast PSD correction using JAX operations.
    Optimized with JIT compilation.
    
    Parameters
    ----------
    matrix : jnp.ndarray
        Input matrix to correct
    epsilon : float, optional
        Minimum eigenvalue threshold
        
    Returns
    -------
    jnp.ndarray
        Nearest PSD matrix
    """
    # Ensure symmetry
    matrix = 0.5 * (matrix + matrix.T)
    
    # Eigenvalue decomposition
    eigenvals, eigenvecs = jnp.linalg.eigh(matrix)
    
    # Clip negative eigenvalues
    eigenvals = jnp.maximum(eigenvals, epsilon)
    
    # Reconstruct matrix
    return eigenvecs @ jnp.diag(eigenvals) @ eigenvecs.T


@jit
def matrix_sqrt_jax(matrix: jnp.ndarray) -> jnp.ndarray:
    """
    Fast matrix square root using JAX.
    Falls back to eigenvalue method if Cholesky fails.
    
    Parameters
    ----------
    matrix : jnp.ndarray
        Input matrix
        
    Returns
    -------
    jnp.ndarray
        Matrix square root
    """
    try:
        # Try Cholesky first (fastest for PSD matrices)
        return cholesky(matrix, lower=True)
    except:
        # Fall back to eigenvalue method
        eigenvals, eigenvecs = jnp.linalg.eigh(matrix)
        sqrt_eigenvals = jnp.sqrt(jnp.maximum(eigenvals, 0))
        return eigenvecs @ jnp.diag(sqrt_eigenvals) @ eigenvecs.T


@jit
def sqrtm_real(A: jnp.ndarray) -> jnp.ndarray:
    """
    Matrix square root that ensures real output.
    
    Parameters
    ----------
    A : jnp.ndarray
        Input matrix
        
    Returns
    -------
    jnp.ndarray
        Real matrix square root
    """
    # Eigendecomposition
    eigvals, eigvecs = jnp.linalg.eigh(A)
    
    # Ensure non-negative eigenvalues
    eigvals = jnp.maximum(eigvals, 0.0)
    
    # Reconstruct with square root of eigenvalues
    sqrt_eigvals = jnp.sqrt(eigvals)
    return eigvecs @ jnp.diag(sqrt_eigvals) @ eigvecs.T


@jit
def ensure_psd(matrix: jnp.ndarray, method: str = "eigenvalue", epsilon: float = 1e-8) -> jnp.ndarray:
    """
    Ensure a matrix is positive semi-definite using specified method.
    
    Parameters
    ----------
    matrix : jnp.ndarray
        Input matrix
    method : str, optional
        Method to use: "eigenvalue" or "frobenius"
    epsilon : float, optional
        Minimum eigenvalue threshold
        
    Returns
    -------
    jnp.ndarray
        PSD corrected matrix
    """
    if method == "eigenvalue":
        return nearest_psd_jax(matrix, epsilon)
    elif method == "frobenius":
        # Frobenius norm minimization method
        return nearest_psd_frobenius(matrix, epsilon)
    else:
        raise ValueError(f"Unknown method: {method}")


@jit
def nearest_psd_frobenius(matrix: jnp.ndarray, epsilon: float = 1e-8) -> jnp.ndarray:
    """
    Find nearest PSD matrix in Frobenius norm.
    
    Parameters
    ----------
    matrix : jnp.ndarray
        Input matrix
    epsilon : float, optional
        Minimum eigenvalue threshold
        
    Returns
    -------
    jnp.ndarray
        Nearest PSD matrix in Frobenius norm
    """
    # Ensure symmetry
    B = 0.5 * (matrix + matrix.T)
    
    # Eigenvalue decomposition
    eigenvals, eigenvecs = jnp.linalg.eigh(B)
    
    # Project eigenvalues to non-negative
    eigenvals = jnp.maximum(eigenvals, epsilon)
    
    # Reconstruct
    return eigenvecs @ jnp.diag(eigenvals) @ eigenvecs.T


@jit
def is_psd(matrix: jnp.ndarray, tol: float = 1e-8) -> bool:
    """
    Check if a matrix is positive semi-definite.
    
    Parameters
    ----------
    matrix : jnp.ndarray
        Matrix to check
    tol : float, optional
        Tolerance for eigenvalue check
        
    Returns
    -------
    bool
        True if matrix is PSD
    """
    eigenvals = jnp.linalg.eigvalsh(matrix)
    return jnp.all(eigenvals >= -tol)


def nearest_psd(A: jnp.ndarray, tol: float = 1e-12) -> jnp.ndarray:
    """
    Project a symmetric matrix onto the cone of positive semi-definite matrices.
    
    Projects by zeroing out negative eigenvalues.
    
    Parameters
    ----------
    A : jnp.ndarray
        Input symmetric matrix
    tol : float, optional
        Tolerance for eigenvalue clipping
        
    Returns
    -------
    jnp.ndarray
        Nearest positive semi-definite matrix
    """
    # Ensure symmetry
    A = 0.5 * (A + A.T)
    
    # Eigen-decomposition
    eigvals, eigvecs = jnp.linalg.eigh(A)
    
    # Clip eigenvalues at tolerance level
    eigvals_clipped = jnp.clip(eigvals, tol, None)
    
    # Reconstruct the matrix
    A_psd = eigvecs @ jnp.diag(eigvals_clipped) @ eigvecs.T
    
    return A_psd
    
