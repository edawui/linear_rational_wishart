# utils/jdf_functions.py
"""specific utility functions."""

import jax.numpy as jnp
from typing import Tuple

# @jit
def tr_uv(u, v):
    """
    Calculate Tr[uv] where u and v are two matrices - JAX-optimized version
    Returns the trace of the matrix product uv
    """
    # More efficient JAX implementation using direct trace calculation
    return float(jnp.trace(jnp.dot(u, v)))
    
    # Alternative implementation using vectorization (equivalent to original)
    # return jnp.vdot(Vec(u.T), Vec(v))
# @jit
def tr_u(u):
    """
    Calculate Tr[u] where u  matrices - JAX-optimized version
    Returns the trace of the matrix product uv
    """
    # More efficient JAX implementation using direct trace calculation
    return jnp.trace(u)
    
    # Alternative implementation using vectorization (equivalent to original)
    # return jnp.vdot(Vec(u.T), Vec(v))


def eij(i: int, j: int, n: int) -> jnp.ndarray:
    """Create elementary matrix E_ij ."""
    e = jnp.zeros((n, n))
    e = e.at[i, j].set(1.0)
    # if i != j:
    #     e = e.at[j, i].set(1.0)
    #     e = e.at[i,j].set(1.0) ##added later to include the transpose
    # Use JAX's where instead of if
    # If i != j, also set the symmetric element
    e = jnp.where(i != j, 
                  e.at[j, i].set(1.0),
                  e)
    return e

def eij_simple(i: int, j: int, n: int) -> jnp.ndarray:
    """Create elementary matrix E_ij ."""
    e = jnp.zeros((n, n))
    e = e.at[i, j].set(1.0)
    # if i != j:
    #     e = e.at[j, i].set(1.0) ##no transpose 
    # Use JAX's where instead of if
    # If i != j, also set the symmetric element
    e = jnp.where(i != j, 
                  e.at[j, i].set(1.0),
                  e)
    return e


def vec(matrix: jnp.ndarray) -> jnp.ndarray:
    """Vectorization in (may differ from standard)."""
    # JDF convention might use different ordering
    # This is a placeholder - actual implementation depends on JDF paper
    return matrix.flatten()


def vec_inv(vector: jnp.ndarray) -> jnp.ndarray:
    """Inverse vectorization in """
    n = int(jnp.sqrt(len(vector)))
    return vector.reshape((n, n))


def compute_h(m: jnp.ndarray, omega: jnp.ndarray, t: float) -> jnp.ndarray:
    """Compute H matrix in JDF notation."""
    # Placeholder for  computation
    pass

# @jit
def vech(u):
    """
    Vech operator - JAX-optimized version
    Extracts the lower triangular part of a matrix (including diagonal)
    """
    n = u.shape[0]
    
    # Use JAX's advanced indexing for efficient extraction
    indices = jnp.tril_indices(n)
    v = u[indices]
    
    # Reshape to column vector
    return jnp.reshape(v, (-1, 1))

