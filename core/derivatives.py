"""
Derivative computations for Wishart processes.

This module contains all derivative calculations organized by type:
- M-derivatives (with respect to mean reversion matrix)
- Omega-derivatives (with respect to drift matrix)
- Vega-derivatives (with respect to volatility)
- Gamma-derivatives
"""
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from typing import Tuple, Optional
import numpy as np
import scipy.integrate
import jax.scipy.linalg as jsl
from ..utils.local_functions import *


class WishartDerivatives:
    """
    Mixin class for derivative computations in Wishart processes.
    
    This class should be mixed into Wishart process implementations
    to provide derivative functionality.
    """
    
    # ========== M-Derivatives ==========
    
    # @partial(jit, static_argnums=(0,))
    def compute_var_sigma_derive_m(self, t: float, i: int, j: int) -> jnp.ndarray:
        """
        Compute derivative of VarSigma with respect to m[i,j].
        
        Parameters
        ----------
        t : float
            Time parameter
        i : int
            Row index
        j : int
            Column index
            
        Returns
        -------
        jnp.ndarray
            Derivative matrix
        """
        n_square = self.n * self.n
        In = jnp.eye(self.n, self.n)
        In_square = jnp.eye(2 * self.n * self.n, 2 * self.n * self.n)
        zero_square = jnp.zeros((self.n * self.n, self.n * self.n))
        eij = self._eij_simple(i, j, self.x0.shape)
        A2 = jnp.zeros((2 * self.n * self.n, 2 * self.n * self.n))
        
        v_0_0_tilde = jnp.zeros((2 * n_square, 1))
        v_0_t_tilde_t = jnp.zeros((2 * n_square, 1))
        
        m_kron_identity_2 = jnp.kron(In, self.m) + jnp.kron(self.m, In)
        id_kron_eij_2 = jnp.kron(In, eij) + jnp.kron(eij, In)
        
        A2 = A2.at[0:n_square, 0:n_square].set(m_kron_identity_2)
        A2 = A2.at[0:n_square, n_square:2*n_square].set(zero_square)
        A2 = A2.at[n_square:2*n_square, 0:n_square].set(id_kron_eij_2)
        A2 = A2.at[n_square:2*n_square, n_square:2*n_square].set(m_kron_identity_2)
        
        b_tilde = jnp.zeros((2 * n_square, 1))
        b_tilde = b_tilde.at[0:n_square, 0:1].set(self._vec(self.sigma2))
        
        exp_t_A2 = jsl.expm(t * A2)
        v_0_t_tilde = (jnp.linalg.inv(A2) @ (exp_t_A2 - In_square)) @ b_tilde
        var_sigma_derive_m_ij_temp = v_0_t_tilde[n_square:2*n_square, 0:1]
        var_sigma_derive_m_ij = self._vec_inv(var_sigma_derive_m_ij_temp, (self.n, self.n))
        
        return var_sigma_derive_m_ij
    
    # @partial(jit, static_argnums=(0,))
    def compute_a_derive_m(self, t: float, i: int, j: int, z: complex,
                          a3: jnp.ndarray, a3_derive_m: jnp.ndarray) -> jnp.ndarray:
        """
        Compute derivative of A with respect to m[i,j].
        
        Parameters
        ----------
        t : float
            Time parameter
        i : int
            Row index
        j : int
            Column index
        z : complex
            Complex parameter
        a3 : jnp.ndarray
            A3 matrix
        a3_derive_m : jnp.ndarray
            Derivative of a3 with respect to m
            
        Returns
        -------
        jnp.ndarray
            Derivative matrix
        """
        emt = jsl.expm(t * self.m)
        emt_t = jnp.transpose(emt)
        
        eAt, var_sigma_init = self.compute_var_sigma(t)
        var_sigma = self._vec_inv(var_sigma_init, (self.n, self.n))
        var_sigma_derive_m = self.compute_var_sigma_derive_m(t, i, j)
        
        In = jnp.eye(self.n, self.n)
        
        temp = jnp.linalg.inv(In - 2 * z * (a3 @ var_sigma))
        z_a3_emt = z * a3 @ emt
        
        a3_derive_m_block = a3_derive_m[i*self.n:(i+1)*self.n, j*self.n:(j+1)*self.n]
        z_a3_derive_m_emt = z * a3_derive_m_block @ emt
        
        eij = self._eij_simple(i, j, self.x0.shape)
        dexp_tm_eij = self._dexp_ax(t * self.m, t * eij)
        dexp_tm_eij_t = jnp.transpose(dexp_tm_eij)
        
        res = dexp_tm_eij_t @ temp @ z_a3_emt
        res += emt_t @ temp @ ((2 * z) * (a3_derive_m_block @ var_sigma + 
                                         a3 @ var_sigma_derive_m)) @ temp @ z_a3_emt
        res += emt_t @ temp @ z_a3_derive_m_emt
        res += emt_t @ temp @ (z * a3) @ dexp_tm_eij
        
        return res
    
    def compute_b_derive_m(self, t: float, i: int, j: int, z: complex,
                          a3: jnp.ndarray, a3_derive_m: jnp.ndarray) -> complex:
        """
        Compute derivative of B with respect to m[i,j].
        
        Parameters
        ----------
        t : float
            Time parameter
        i : int
            Row index
        j : int
            Column index
        z : complex
            Complex parameter
        a3 : jnp.ndarray
            A3 matrix
        a3_derive_m : jnp.ndarray
            Derivative of a3 with respect to m
            
        Returns
        -------
        complex
            Derivative value
        """
        def f1(t1):
            a_derive_m = self.compute_a_derive_m(t1, i, j, z, a3, a3_derive_m)
            z1 = jnp.trace(a_derive_m @ self.omega)
            return z1.real
        
        r1 = scipy.integrate.quad(f1, 0, t)
        c2r = r1[0]
        
        def f3(t1):
            a_derive_m = self.compute_a_derive_m(t1, i, j, z, a3, a3_derive_m)
            z1 = jnp.trace(a_derive_m @ self.omega)
            return z1.imag
        
        r2 = scipy.integrate.quad(f3, 0, t)
        c2i = r2[0]
        b_m_ij = complex(c2r, c2i)
        
        return b_m_ij
    
    # ========== Omega-Derivatives ==========
    
    def compute_b_prime_omega(self, t: float, i: int, j: int, 
                             theta1: jnp.ndarray) -> complex:
        """
        Compute derivative of B with respect to omega[i,j].
        
        Parameters
        ----------
        t : float
            Time parameter
        i : int
            Row index
        j : int
            Column index
        theta1 : jnp.ndarray
            Parameter matrix
            
        Returns
        -------
        complex
            Derivative value
        """
        eij = self._eij(i, j, self.x0.shape)
        
        def f1(t1, theta1):
            a = self.compute_a(t1, theta1)
            z1 = jnp.trace(a @ eij)
            return z1.real
        
        r1 = scipy.integrate.quad(lambda t1: f1(t1, theta1), 0, t)
        c2r = r1[0]
        
        def f3(t1, theta1):
            a = self.compute_a(t1, theta1)
            z1 = jnp.trace(a @ eij)
            return z1.imag
        
        r2 = scipy.integrate.quad(lambda t1: f3(t1, theta1), 0, t)
        c2i = r2[0]
        b_omega_ij = complex(c2r, c2i)
        
        return b_omega_ij
    
    # ========== Vega-Derivatives ==========
    
    @partial(jit, static_argnums=(0,))
    def compute_var_sigma_vega(self, i: int, j: int, t: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute Vega derivative of VarSigma.
        
        Parameters
        ----------
        i : int
            Row index
        j : int
            Column index
        t : float
            Time parameter
            
        Returns
        -------
        eAt : jnp.ndarray
            Matrix exponential
        v3 : jnp.ndarray
            Transformed vector
        """
        eij = self._eij(i, j, self.x0.shape)
        sigma_vega_ij2 = eij @ self.sigma + self.sigma @ eij
        
        eAt = jsl.expm(t * self.A)
        vec_sigma_vega_ij2 = self._vec(sigma_vega_ij2)
        v2 = (eAt - jnp.eye(self.n * self.n)) @ vec_sigma_vega_ij2
        v3 = jnp.linalg.inv(self.A) @ v2
        
        return eAt, v3
    
    @partial(jit, static_argnums=(0,))
    def compute_a_vega(self, i: int, j: int, t: float, theta1: jnp.ndarray) -> jnp.ndarray:
        """
        Compute Vega derivative of A.
        
        Parameters
        ----------
        i : int
            Row index
        j : int
            Column index
        t : float
            Time parameter
        theta1 : jnp.ndarray
            Parameter matrix
            
        Returns
        -------
        jnp.ndarray
            Derivative matrix
        """
        emt = jsl.expm(t * self.m)
        
        eAt, var_sigma_init = self.compute_var_sigma(t)
        var_sigma = self._vec_inv(var_sigma_init, (self.n, self.n))
        m2 = jnp.linalg.inv(jnp.eye(self.n) - 2 * (theta1 @ var_sigma))
        # m2 = jnp.linalg.inv(jnp.eye(self.n) - 2 * (theta1 @ var_sigma))
        
        emt_tr = jnp.transpose(emt)
        
        eAt_temp, var_sigma_vega_ij_init = self.compute_var_sigma_vega(i, j, t)
        var_sigma_vega_ij = self._vec_inv(var_sigma_vega_ij_init, (self.n, self.n))
        
        a = emt_tr @ m2 @ (2 * theta1) @ var_sigma_vega_ij @ m2 @ theta1 @ emt
        return a
    
    def compute_b_vega(self, i: int, j: int, t: float, theta1: jnp.ndarray) -> complex:
        """
        Compute Vega derivative of B.
        
        Parameters
        ----------
        i : int
            Row index
        j : int
            Column index
        t : float
            Time parameter
        theta1 : jnp.ndarray
            Parameter matrix
            
        Returns
        -------
        complex
            Derivative value
        """
        if self.is_bru_config:
            emt = jsl.expm(t * self.m)
            eAt, var_sigma_init = self.compute_var_sigma(t)
            
            var_sigma = self._vec_inv(var_sigma_init, (self.n, self.n))
            eAt_temp, var_sigma_vega_ij_init = self.compute_var_sigma_vega(i, j, t)
            var_sigma_vega_ij = self._vec_inv(var_sigma_vega_ij_init, (self.n, self.n))
            
            m2 = jnp.linalg.inv(jnp.eye(self.n) - 2 * (var_sigma @ theta1))
            temp = m2 @ (var_sigma_vega_ij @ theta1)
            
            b_nu = self.beta * jnp.trace(temp)
            
            return b_nu
        else:
            def f1(t1, theta1):
                a = self.compute_a_vega(i, j, t1, theta1)
                z1 = jnp.trace(a @ self.omega)
                return z1.real
            
            r1 = scipy.integrate.quad(lambda t1: f1(t1, theta1), 0, t)
            c2r = r1[0]
            
            def f3(t1, theta1):
                a = self.compute_a_vega(i, j, t1, theta1)
                z1 = jnp.trace(a @ self.omega)
                return z1.imag
            
            r2 = scipy.integrate.quad(lambda t1: f3(t1, theta1), 0, t)
            c2i = r2[0]
            c3 = complex(c2r, c2i)
            return c3
    
    # ========== Gamma-Derivatives ==========
    
    @partial(jit, static_argnums=(0,))
    def compute_a_j2_gamma_derive(self, t: float, z: complex, a3: jnp.ndarray,
                                  theta1: jnp.ndarray) -> jnp.ndarray:
        """
        Compute J2 Gamma derivative of A.
        
        Parameters
        ----------
        t : float
            Time parameter
        z : complex
            Complex parameter
        a3 : jnp.ndarray
            A3 matrix
        theta1 : jnp.ndarray
            Parameter matrix
            
        Returns
        -------
        jnp.ndarray
            Derivative matrix
        """
        emt = jsl.expm(t * self.m)
        emt_t = jnp.transpose(emt)
        
        eAt, var_sigma_init = self.compute_var_sigma(t)
        var_sigma = self._vec_inv(var_sigma_init, (self.n, self.n))
        
        temp_inv = jnp.linalg.inv(jnp.eye(self.n) - 2 * z * (a3 @ var_sigma))
        
        res = emt_t @ temp_inv @ (2 * z * (theta1 @ var_sigma)) @ temp_inv @ (z * a3) @ emt
        res += emt_t @ temp_inv @ (z * theta1) @ emt
        
        return res
    
    def compute_b_j2_gamma_derive(self, t: float, z: complex, a3: jnp.ndarray,
                                  theta1: jnp.ndarray) -> complex:
        """
        Compute J2 Gamma derivative of B.
        
        Parameters
        ----------
        t : float
            Time parameter
        z : complex
            Complex parameter
        a3 : jnp.ndarray
            A3 matrix
        theta1 : jnp.ndarray
            Parameter matrix
            
        Returns
        -------
        complex
            Derivative value
        """
        def f1(u):
            j2_a_derive = self.compute_a_j2_gamma_derive(u, z, a3, theta1)
            z1 = jnp.trace(self.omega @ j2_a_derive)
            return z1.real
        
        r1 = scipy.integrate.quad(f1, 0, t)
        c2r = r1[0]
        
        def f3(u):
            j2_a_derive = self.compute_a_j2_gamma_derive(u, z, a3, theta1)
            z1 = jnp.trace(self.omega @ j2_a_derive)
            return z1.imag
        
        r2 = scipy.integrate.quad(f3, 0, t)
        c2i = r2[0]
        b_j2_gamma_derive = complex(c2r, c2i)
        
        return b_j2_gamma_derive
    
    @partial(jit, static_argnums=(0,))
    def compute_a_j4_gamma_derive(self, t: float, z: complex, a3: jnp.ndarray,
                                  theta0: jnp.ndarray, theta1: jnp.ndarray) -> jnp.ndarray:
        """
        Compute J4 Gamma derivative of A.
        
        Parameters
        ----------
        t : float
            Time parameter
        z : complex
            Complex parameter
        a3 : jnp.ndarray
            A3 matrix
        theta0 : jnp.ndarray
            First parameter matrix
        theta1 : jnp.ndarray
            Second parameter matrix
            
        Returns
        -------
        jnp.ndarray
            Derivative matrix
        """
        emt = jsl.expm(t * self.m)
        emt_t = jnp.transpose(emt)
        
        eAt, var_sigma_init = self.compute_var_sigma(t)
        var_sigma = self._vec_inv(var_sigma_init, (self.n, self.n))
        
        temp_inv = jnp.linalg.inv(jnp.eye(self.n) - 2 * z * (a3 @ var_sigma))
        
        res = emt_t @ temp_inv @ (2 * z * (theta0 @ var_sigma)) @ temp_inv @ (z * theta1) @ emt
        res += emt_t @ temp_inv @ (2 * z * (theta1 @ var_sigma)) @ temp_inv @ (z * theta0) @ emt
        res += emt_t @ temp_inv @ (2 * z * (theta0 @ var_sigma)) @ temp_inv @ \
               (2 * z * (theta1 @ var_sigma)) @ temp_inv @ (z * a3) @ emt
        res += emt_t @ temp_inv @ (2 * z * (theta1 @ var_sigma)) @ temp_inv @ \
               (2 * z * (theta0 @ var_sigma)) @ temp_inv @ (z * a3) @ emt
        
        return res
    
    def compute_b_j4_gamma_derive(self, t: float, z: complex, a3: jnp.ndarray,
                                  theta0: jnp.ndarray, theta1: jnp.ndarray) -> complex:
        """
        Compute J4 Gamma derivative of B.
        
        Parameters
        ----------
        t : float
            Time parameter
        z : complex
            Complex parameter
        a3 : jnp.ndarray
            A3 matrix
        theta0 : jnp.ndarray
            First parameter matrix
        theta1 : jnp.ndarray
            Second parameter matrix
            
        Returns
        -------
        complex
            Derivative value
        """
        def f1(u):
            j4_a_derive = self.compute_a_j4_gamma_derive(u, z, a3, theta0, theta1)
            z1 = jnp.trace(self.omega @ j4_a_derive)
            return z1.real
        
        r1 = scipy.integrate.quad(f1, 0, t)
        c2r = r1[0]
        
        def f3(u):
            j4_a_derive = self.compute_a_j4_gamma_derive(u, z, a3, theta0, theta1)
            z1 = jnp.trace(self.omega @ j4_a_derive)
            return z1.imag
        
        r2 = scipy.integrate.quad(f3, 0, t)
        c2i = r2[0]
        b_j4_gamma_derive = complex(c2r, c2i)
        
        return b_j4_gamma_derive
    
    # ========== Helper Methods ==========
    
    @staticmethod
    def _eij_simple(i: int, j: int, shape: Tuple[int, int]) -> jnp.ndarray:
        """
        Create elementary matrix with 1 at position (i,j).
        
        Parameters
        ----------
        i : int
            Row index
        j : int
            Column index
        shape : Tuple[int, int]
            Matrix shape
            
        Returns
        -------
        jnp.ndarray
            Elementary matrix
        """
        # eij = jnp.zeros(shape)
        # eij = eij.at[i, j].set(1.0)
        # return eij
        # print(shape)
        return eij_simple(i, j, shape[0])
    
    @staticmethod
    def _eij(i: int, j: int, shape: Tuple[int, int]) -> jnp.ndarray:
        """
        Create symmetric elementary matrix with 1 at positions (i,j) and (j,i).
        
        Parameters
        ----------
        i : int
            Row index
        j : int
            Column index
        shape : Tuple[int, int]
            Matrix shape
            
        Returns
        -------
        jnp.ndarray
            Symmetric elementary matrix
        """
        # eij = jnp.zeros(shape)
        # eij = eij.at[i, j].set(1.0)
        # if i != j:
        #     eij = eij.at[j, i].set(1.0)
        # return eij
        # # print(shape[0])
        return eij(i, j, shape[0])

    
    @staticmethod
    def _dexp_ax(A: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the derivative of exp(A) in direction X.
        
        This computes d/dt exp(A + tX)|_{t=0}.
        
        Parameters
        ----------
        A : jnp.ndarray
            Base matrix
        X : jnp.ndarray
            Direction matrix
            
        Returns
        -------
        jnp.ndarray
            Derivative matrix
        """
        # This is a placeholder - the actual implementation would use
        # the proper matrix exponential derivative formula
        # For now, using a simple approximation
        eps = 1e-8
        exp_plus = jsl.expm(A + eps * X)
        exp_base = jsl.expm(A)
        return (exp_plus - exp_base) / eps