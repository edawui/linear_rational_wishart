"""
Base class for Wishart process implementations.
"""
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Union
import jax.numpy as jnp
import pandas as pd
from functools import partial
from jax import jit
import jax.scipy.linalg as jsl

# Configuration constants
EPSABS = 1e-7
EPSREL = 1e-04
NMAX = 1000
DEFAULT_INTEGRATION_POINTS = 100


class BaseWishart(ABC):
    """Abstract base class for Wishart processes."""
    
    def __init__(self, n: int, x0: jnp.ndarray, omega: jnp.ndarray, 
                 m: jnp.ndarray, sigma: jnp.ndarray):
        """
        Initialize base Wishart process.
        
        Parameters
        ----------
        n : int
            Size of the Wishart matrix
        x0 : jnp.ndarray
            Initial value matrix
        omega : jnp.ndarray
            Drift matrix
        m : jnp.ndarray
            Mean reversion matrix
        sigma : jnp.ndarray
            Volatility matrix
        """
        self.is_bru_config=False
        self.set_wishart_parameter(n, x0, omega, m, sigma)

    def set_wishart_parameter(self, n: int, x0: jnp.ndarray, omega: jnp.ndarray, 
                 m: jnp.ndarray, sigma: jnp.ndarray):
        self.n = n
        self.x0 = jnp.array(x0)
        self.omega = jnp.array(omega)
        self.m = jnp.array(m)
        self.sigma = jnp.array(sigma)
        self.sigma2 = jnp.matmul(sigma, sigma)
        self.sigma2_inv = jnp.linalg.inv(self.sigma2)
        
        self.eye_n = jnp.eye(self.n)
        self.eye_nn = jnp.eye(self.n * self.n)
        self.vecSigma2 = self._vec(self.sigma2)
        

        # Common attributes
        self.use_range_kutta_for_b = True
        self.A = jnp.add(jnp.kron(jnp.eye(self.n), self.m), 
                        jnp.kron(self.m, jnp.eye(self.n)))
        self.b = self._vec(self.omega)
        self.A_inv = jnp.linalg.inv(self.A)
        self.zero_n = jnp.zeros(self.x0.shape)
         # Eigendecomposition
        self.A_eigvals, self.A_eigvecs = jnp.linalg.eig(self.A)
        self.m_eigvals, self.m_eigvecs = jnp.linalg.eig(self.m)
        # Precompute inverses for efficiency
        self.A_eigvecs_inv = jnp.linalg.inv(self.A_eigvecs)
        self.m_eigvecs_inv = jnp.linalg.inv(self.m_eigvecs)
        # Precompute A_inv @ vecSigma2 since it's used in every integration point
        self.A_inv_vecSigma2 = self.A_inv @ self.vecSigma2
        
        # Moment-related attributes
        self.G = None
        self.exp_gt = None
        self.h = None
        self.g0 = None
        self.h1 = None
        self.poly_prop = None
        self.block_struct = None
        self.pos = None
        self.moments = None
        self.maturity = 0
        self.cd_params_computed=False

        # Derivative attributes
        self.b3_derive_omega = 0
        self.b3_derive_m = 0
        self.a3_derive_m = 0
        self.partial_ij_b3 = jnp.zeros(self.x0.shape)
    
    def reset(self):
        """Reset the model state."""
        pass
    
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
    
    @staticmethod
    def _vec_inv(vector: jnp.ndarray, shape: Tuple[int, int]) -> jnp.ndarray:
        """
        Inverse of vectorization operation.
        
        Parameters
        ----------
        vector : jnp.ndarray
            Vectorized matrix
        shape : Tuple[int, int]
            Target shape (rows, cols)
            
        Returns
        -------
        jnp.ndarray
            Reconstructed matrix
        """
        return vector.reshape(shape[1], shape[0]).T
    
    @staticmethod
    def _vech(matrix: jnp.ndarray) -> jnp.ndarray:
        """
        Half-vectorization of a symmetric matrix.
        
        Parameters
        ----------
        matrix : jnp.ndarray
            Symmetric matrix
            
        Returns
        -------
        jnp.ndarray
            Half-vectorized matrix
        """
        n = matrix.shape[0]
        idx = jnp.triu_indices(n)
        return matrix[idx].reshape(-1, 1)
    
    @partial(jit, static_argnums=(0,))
    def compute_mean_wishart_decomp(self, t: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute e^{A t} and A^-1 (e^{A t}- I_n*n)*b involved in E[vec(x_t)].
        
        Parameters
        ----------
        t : float
            Time parameter
            
        Returns
        -------
        eAt : jnp.ndarray
            Matrix exponential e^{A t}
        v3 : jnp.ndarray
            A^-1 (e^{A t}- I_n*n)*b
        """
        # eAt = jnp.linalg.matrix_exp(jnp.multiply(self.A, t))
        # v2 = jnp.matmul(jnp.subtract(eAt, jnp.eye(self.n * self.n)), self.b)

        eAt = jsl.expm(self.A * t)
        v2 = (eAt -  jnp.eye(self.n * self.n)) @ self.b
        v3 = jnp.linalg.solve(self.A, v2)
        return eAt, v3
    
    @partial(jit, static_argnums=(0,))
    def compute_mean_wishart(self, t: float) -> jnp.ndarray:
        """
        Compute E[vec(x_t)] where x_t is a Wishart process.
        
        Parameters
        ----------
        t : float
            Time parameter
            
        Returns
        -------
        jnp.ndarray
            Expected value of vectorized Wishart process
            
        Notes
        -----
        This implements Equation (12) from the reference paper.
        """
        # print(" ============== compute_mean_wishart CORE-BASE======================")

        vv0 = self._vec(self.x0)
        eAt, v3 = self.compute_mean_wishart_decomp(t)
        v1 = jnp.matmul(eAt, vv0)
        v4 = jnp.add(v1, v3)
        return v4
    
    @partial(jit, static_argnums=(0,))
    def compute_mean(self, t: float, u0: jnp.ndarray) -> float:
        """
        Compute E[tr[u0 x_t]] = vec(u0^top)^top E[vec(x_t)].
        
        Parameters
        ----------
        t : float
            Time parameter
        u0 : jnp.ndarray
            Weight matrix
            
        Returns
        -------
        float
            Expected trace value
        """
        # print(" ============== compute_mean CORE-BASE======================")
        vu0 = self._vec(jnp.transpose(u0))
        v1 = self.compute_mean_wishart(t)
        return jnp.vdot(vu0, v1)
    
    def check_gindikin(self) -> None:
        """
        Check the Gindikin condition for the Wishart process.
        
        Computes the eigenvalues of omega - beta*sigma^2 with beta=n+1
        and prints the eigenvalues that should be positive.
        """
        beta = self.n + 1
        u = self.omega - jnp.multiply(beta, jnp.matmul(self.sigma, self.sigma))
        eig_vals, eig_vecs = jnp.linalg.eig(u)
        print(" ============== check_gindikin ======================")
        print("Eigenvalues of omega - beta sigma^2 with beta = ", beta, " are:")
        print(f"Eigenvalues  = {eig_vals}")
        print(f"Eigenvectors = {eig_vecs}")
        print("=====================================================")
        if jnp.any(eig_vals <= 0):
            return False
        return True
    
    @partial(jit, static_argnums=(0,))
    def compute_exp_mt(self, t: float) -> jnp.ndarray:
        """
        Compute matrix exponential e^{m t}.
        
        Parameters
        ----------
        t : float
            Time parameter
            
        Returns
        -------
        jnp.ndarray
            Matrix exponential
        """
        # return jnp.linalg.matrix_exp(t * self.m)
        return jnp.expm(t * self.m)
    
    @partial(jit, static_argnums=(0,))
    def compute_mean_decompose(self, u0: jnp.ndarray, t: float) -> Tuple[float, jnp.ndarray]:
        """
        Compute decomposition E[tr[u0 x_t]] = b0(t) + tr[a0(t)x_t].
        
        Parameters
        ----------
        u0 : jnp.ndarray
            Weight matrix
        t : float
            Time parameter
            
        Returns
        -------
        b0 : float
            Constant term
        a0 : jnp.ndarray
            Matrix coefficient
        """
        eAt, v3 = self.compute_mean_wishart_decomp(t)
        b0 = jnp.vdot(self._vec(jnp.transpose(u0)), v3)
        a0 = self._vec_inv(
            jnp.matmul(jnp.transpose(eAt), self._vec(jnp.transpose(u0))),
            (self.n, self.n)
        )
        a0 = jnp.transpose(a0)
        return b0, a0
    
    def print_model(self) -> None:
        """Print model parameters."""
        print("\nModel Parameters")
        print(f"n = {self.n}")
        print(f"x0 = {self.x0}")
        print(f"omega = {self.omega}")
        print(f"m = {self.m}")
        print(f"sigma = {self.sigma}")
    
    # Abstract methods that must be implemented by subclasses
    @abstractmethod
    def phi_one(self, z: complex, theta1: jnp.ndarray) -> complex:
        """Compute the first characteristic function."""
        pass
    
    @abstractmethod
    def compute_moments(self, t: float, order: int = 3) -> None:
        """Compute moments of the process."""
        pass
    
    @abstractmethod
    def compute_a(self, t: float, theta1: jnp.ndarray) -> jnp.ndarray:
        """Compute the A matrix function."""
        pass
    
    @abstractmethod
    def compute_b(self, t: float, theta1: jnp.ndarray) -> complex:
        """Compute the B function."""
        pass
