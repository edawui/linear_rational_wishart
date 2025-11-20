"""
Matrix decomposition algorithms for Wishart processes.
"""
import jax.numpy as jnp
from jax.scipy.linalg import cholesky, eigh, schur
from typing import Tuple
import warnings

from .psd_corrections import nearest_psd


def extended_cholesky(
    x: jnp.ndarray, 
    tol: float = 1e-15
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Extended Cholesky decomposition for positive semidefinite matrices.
    
    Computes a decomposition that handles rank-deficient matrices by
    identifying the rank and providing a reduced factorization.
    
    Parameters
    ----------
    x : jnp.ndarray
        Symmetric positive semidefinite matrix (n x n)
    tol : float, optional
        Tolerance for considering eigenvalues as zero
        
    Returns
    -------
    perm_matrix : jnp.ndarray
        Permutation matrix such that perm_matrix.T @ x @ perm_matrix 
        has the block form [[x11, 0], [0, 0]]
    k_r : jnp.ndarray
        Lower part of the factorization
    c_r : jnp.ndarray
        Upper triangular matrix such that x = c_r @ c_r.T (after permutation)
        
    Notes
    -----
    This algorithm is useful for singular or near-singular matrices where
    standard Cholesky decomposition would fail.
    """
    x = jnp.array(x, dtype=float)
    d = x.shape[0]
    diag = jnp.copy(jnp.diag(x))
    p = jnp.arange(d)
    c = jnp.zeros((d, d))
    k = 0

    while k < d:
        i = jnp.argmax(diag[k:]) + k
        if diag[i] < tol:
            break

        if i != k:
            x = x.at[[k, i], :].set(x[[i, k], :])
            x = x.at[:, [k, i]].set(x[:, [i, k]])
            diag = diag.at[[k, i]].set(diag[[i, k]])
            p = p.at[[k, i]].set(p[[i, k]])
            c = c.at[[k, i], :].set(c[[i, k], :])

        c = c.at[k, k].set(jnp.sqrt(diag[k]))
        for j in range(k + 1, d):
            c = c.at[j, k].set((x[j, k] - c[j, :k] @ c[k, :k]) / c[k, k])
            diag = diag.at[j].add(-c[j, k] ** 2)

        k += 1

    # Construct the permutation matrix π from the index vector p
    perm_matrix = jnp.eye(d)[p]

    c_ = c[:, :k]
    c_r = c_[0:k, 0:k]
    k_r = c_[k:, 0:k]

    return perm_matrix, k_r, c_r


def safe_cholesky(
    a: jnp.ndarray, 
    eps: float = 1e-10, 
    max_tries: int = 5
) -> jnp.ndarray:
    """
    Perform numerically stable Cholesky decomposition with regularization.
    
    If the matrix is not positive definite, applies regularization by
    projecting onto the nearest positive semidefinite matrix.
    
    Parameters
    ----------
    a : jnp.ndarray
        Input symmetric matrix (n x n)
    eps : float, optional
        Initial regularization parameter
    max_tries : int, optional
        Maximum number of attempts with increasing regularization
        
    Returns
    -------
    jnp.ndarray
        Lower triangular matrix L such that a ≈ L @ L.T
        
    Raises
    ------
    ValueError
        If matrix cannot be made positive definite after max_tries
        
    Notes
    -----
    This function is useful when dealing with matrices that are theoretically
    positive definite but may have numerical issues due to rounding errors.
    """
    for i in range(max_tries):
        try:
            return jnp.linalg.cholesky(a)
        except:
            # Project onto nearest PSD matrix
            a = nearest_psd(a, tol=eps)
            eps *= 10  # Increase regularization if needed

    raise ValueError("Matrix not positive definite even after regularization.")


def simultaneous_diagonalization(
    alpha_bar: jnp.ndarray, 
    a: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute matrix u for simultaneous diagonalization.
    
    Finds u such that:
        alpha_bar = u^T b u
        a = u I_d^n u^T
        
    Using Algorithm 8.7.1 from Golub and Van Loan.
    
    Parameters
    ----------
    alpha_bar : jnp.ndarray
        Symmetric matrix (n x n)
    a : jnp.ndarray
        Symmetric positive definite matrix (n x n)
        
    Returns
    -------
    u : jnp.ndarray
        Transformation matrix
    delta : jnp.ndarray
        Diagonal matrix from the transformation
        
    References
    ----------
    Golub and Van Loan, "Matrix Computations", Algorithm 8.7.1
    """
    # Solve the generalized eigenvalue problem
    eigvals, eigvecs = eigh(a)
    
    # The matrix u is based on the square root of a
    u = jnp.linalg.cholesky(a)
    
    u_inv = jnp.linalg.inv(u)
    delta = u_inv.T @ alpha_bar @ u_inv
    
    return u, delta


def compute_u_delta(
    alpha: jnp.ndarray, 
    a: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute transformation matrices u and delta.
    
    Finds u and delta such that:
        alpha = u.T @ delta @ u
        a.T @ a = u.T @ I_d,n @ u
        
    Parameters
    ----------
    alpha : jnp.ndarray
        Symmetric positive-definite matrix (n x n)
    a : jnp.ndarray
        Full rank matrix (n x n) 
        
    Returns
    -------
    u : jnp.ndarray
        Transformation matrix
    delta : jnp.ndarray
        Diagonal matrix
        
    Notes
    -----
    This decomposition is used in the fast second-order schemes
    for simulating affine diffusions.
    """
    b_mat = a.T @ a
    a_mat = alpha

    # Cholesky decomposition of a.T @ a
    g = cholesky(b_mat, lower=True)
    g_inv = jnp.linalg.inv(g)
    
    # Transform alpha
    c = g_inv @ a_mat @ g_inv.T

    # Schur decomposition
    t, q = schur(c, output='real')
    
    # Compute transformation matrix
    x = g_inv.T @ q
    u = jnp.linalg.inv(x)
    
    # Extract diagonal
    delta = jnp.diag(jnp.diag(q.T @ c @ q))
    
    return u, delta


def compute_delta(u: jnp.ndarray, alpha_bar: jnp.ndarray) -> jnp.ndarray:
    """
    Compute delta matrix from transformation.
    
    Given u and alpha_bar, compute delta such that:
        alpha_bar = u.T @ delta @ u
        
    Parameters
    ----------
    u : jnp.ndarray
        Transformation matrix
    alpha_bar : jnp.ndarray
        Original matrix
        
    Returns
    -------
    jnp.ndarray
        Delta matrix
    """
    u_inv = jnp.linalg.inv(u)
    delta = u_inv.T @ alpha_bar @ u_inv
    return delta
