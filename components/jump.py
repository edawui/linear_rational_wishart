"""
Jump component for Wishart processes.

This module defines the jump component that can be added to
Wishart processes to model discontinuous changes.
"""
# from Sympy.solvers.solvers import det_minor
from functools import partial
from jax import jit, grad, vmap
import jax.numpy as jnp
from jax.scipy import linalg as jlinalg
from typing import Optional
import numpy as np


class JumpComponent:
    """
    Jump component for Wishart processes.
    
    This class models the jump part of a Wishart process with jumps,
    characterized by jump intensity, size distribution parameters.
    
    Parameters
    ----------
    lambda_intensity : float
        Jump intensity (average number of jumps per unit time)
    nu : float
        Degrees of freedom parameter for jump size distribution
    eta : jnp.ndarray
        Jump variance matrix (n x n)
    xi : jnp.ndarray
        Jump mean matrix (n x n)
    
    Attributes
    ----------
    n : int
        Dimension of the process
    expectation_j : jnp.ndarray
        Expected value of jump size
    c : jnp.ndarray
        Vectorized expected jump contribution
    """
    
    def __init__(self, lambda_intensity: float, nu: float, 
                 eta: jnp.ndarray, xi: jnp.ndarray):
        """Initialize jump component."""
        self.nu = nu
        self.eta = jnp.array(eta)
        self.xi = jnp.array(xi)
        self.lambda_intensity = lambda_intensity
        self.n = xi.shape[1]

        self.In = jnp.eye(self.n)
        self.xi_tr =jnp.transpose(self.xi)
        self.xi_eta = self.xi_tr @ self.eta
        self.eta_norm = jnp.linalg.norm(eta)
        # Compute expected jump contribution
        self.c = self.compute_c()
    
    def compute_expectation_j(self) -> jnp.ndarray:
        """
        Compute expected value of jump size.
        
        Returns
        -------
        jnp.ndarray
            Expected jump size matrix
        """
        # Component-wise expectations
        moment_22 = (self.eta[1, 1] * self.nu + 
                     self.eta[0, 1] * self.xi[0, 1] + 
                     self.eta[1, 1] * self.xi[1, 1])
        
        moment_11 = (self.eta[0, 0] * self.nu + 
                     self.eta[0, 0] * self.xi[0, 0] + 
                     self.eta[0, 1] * self.xi[1, 0])
        
        moment_12 = (self.eta[0, 1] * self.nu + 
                     (self.eta[0, 0] * self.xi[0, 1] / 2 + 
                      self.eta[0, 1] * self.xi[0, 0] / 2 +
                      self.eta[0, 1] * self.xi[1, 1] / 2 + 
                      self.eta[1, 1] * self.xi[1, 0] / 2))
        
        self.expectation_j = jnp.array([[moment_11, moment_12], 
                                        [moment_12, moment_22]])
        
        return self.expectation_j
    
    def compute_c(self) -> jnp.ndarray:
        """
        Compute the c vector for jump contribution.
        
        Returns
        -------
        jnp.ndarray
            Vectorized expected jump contribution scaled by intensity
        """
        self.compute_expectation_j()
        return self.lambda_intensity * self._vec(self.expectation_j)
    
    @partial(jit, static_argnums=(0,))
    def compute_mgf_jump_old_fast_ok(self, theta: jnp.ndarray) -> float:
        """
        Compute moment generating function of jump component.
        
        Parameters
        ----------
        theta : jnp.ndarray
            Parameter matrix
            
        Returns
        -------
        float
            MGF value
        """
        
        v0 = self.In - 2 * self.eta @ theta
        det_v0= jnp.linalg.det(v0)
        denom= det_v0* det_v0

        # v1 = jnp.linalg.matrix_power(v0, 2)
        # v2 = jnp.linalg.det(v1)
        v0_inv = jnp.linalg.inv(v0)
        # v3 = self.xi_tr @ self.eta @ theta @ v0_inv
        v3 = self.xi_eta @ theta @ v0_inv
     
        num = jnp.trace(jlinalg.expm(v3))
        # denom = v2
        res = num / denom
        
        return res

    @partial(jit, static_argnums=(0,))
    def compute_mgf_jump(self, theta: jnp.ndarray) -> float:
        """
        Optimized with precomputed values.
        """
        # Use precomputed xi_eta
        v0 = self.In - 2 * self.eta @ theta
        
        det_v0 = jnp.linalg.det(v0)
        denom = det_v0 * det_v0
        
        # Use precomputed xi_eta
        v3 = jnp.linalg.solve(v0.T, (self.xi_eta @ theta).T).T
        
        num = jnp.trace(jlinalg.expm(v3))
        
        return num / denom

    @staticmethod
    def _vec(matrix: jnp.ndarray) -> jnp.ndarray:
        """
        Vectorize a matrix by stacking columns.
        
        Parameters
        ----------
        matrix : jnp.ndarray
            Input matrix
            
        Returns
        -------
        jnp.ndarray
            Vectorized matrix
        """
        return matrix.T.ravel().reshape(-1, 1)
