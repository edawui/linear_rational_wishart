"""
Characteristic functions (Phi functions) for Wishart processes.

This module contains the implementation of various characteristic functions
used in option pricing with Wishart processes.
"""
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from typing import Tuple, Optional, Union
import scipy.integrate
import jax.scipy.linalg as jsl
from ..utils.local_functions import tr_uv, tr_u

##Removed from Math

class PhiFunctions:
    """
    Mixin class for characteristic function computations in Wishart processes.
    
    This class provides methods for computing Phi_One, Phi_Two, Phi_Three
    and their various derivatives and variants.
    """
    
    # ========== Phi_One Functions ==========
    
    def phi_one_vega(self, i: int, j: int, z: complex, theta1: jnp.ndarray) -> complex:
        """
        Compute Vega derivative of Phi_One.
        
        Parameters
        ----------
        i : int
            Row index for derivative
        j : int
            Column index for derivative
        z : complex
            Complex parameter
        theta1 : jnp.ndarray
            Parameter matrix
            
        Returns
        -------
        complex
            Vega derivative of Phi_One
            
        Notes
        -----
        Implements Equation (25) from the reference paper.
        """
        theta1 = jnp.multiply(z, theta1)
        
        t = self.maturity
        a_vega_ij = self.compute_a_vega(i, j, t, theta1)
        b_vega_ij = self.compute_b_vega(i, j, t, theta1)
        

        # a2 = tr_uv(self.x0 , a_vega_ij)
        
        # phi_one_vega_ij = (tr_uv(self.x0 @ a_vega_ij) + b_vega_ij) * self.phi_one(z, theta1)
        phi_one_vega_ij = (tr_u(self.x0 @ a_vega_ij) + b_vega_ij) * self.phi_one(z, theta1)
        
        return phi_one_vega_ij
    
    # ========== Phi_Two Functions ==========
    
    # @partial(jit, static_argnums=(0,))
    def compute_a_nu(self, t: float, theta_1: jnp.ndarray, theta_2: jnp.ndarray) -> jnp.ndarray:
        """
        Compute a_nu matrix for Phi_Two.
        
        Parameters
        ----------
        t : float
            Time parameter
        theta_1 : jnp.ndarray
            First parameter matrix
        theta_2 : jnp.ndarray
            Second parameter matrix
            
        Returns
        -------
        jnp.ndarray
            a_nu matrix
            
        Notes
        -----
        Implements Equation (25) from the reference paper.
        """
        # test = jnp.zeros((2,2))
        # return test
        # print("-------------- Debuging ------------------")
        # print(self.m)
        emt = jsl.expm(t * self.m)
        eAt, var_sigma_init = self.compute_var_sigma(t)
        var_sigma = self._vec_inv(var_sigma_init, (self.n, self.n))
        
        m2 = jnp.linalg.inv(jnp.eye(self.n) - 2 * (theta_2 @ var_sigma))
        emt_tr = jnp.transpose(emt)
        
        a_nu = emt_tr @ m2 @ (2 * theta_1 @ var_sigma) @ m2 @ theta_2 @ emt
        a_nu += emt_tr @ m2 @ theta_1 @ emt
        
        return a_nu
    
    def compute_b_nu_old(self, t: float, theta1: jnp.ndarray, theta2: jnp.ndarray) -> complex:
        """
        Compute b_nu for Phi_Two.
        
        Parameters
        ----------
        t : float
            Time parameter
        theta1 : jnp.ndarray
            First parameter matrix
        theta2 : jnp.ndarray
            Second parameter matrix
            
        Returns
        -------
        complex
            b_nu value
        """
        if self.is_bru_config:
            emt = jsl.expm(t * self.m)
            eAt, var_sigma_init = self.compute_var_sigma(t)
            var_sigma = self._vec_inv(var_sigma_init, (self.n, self.n))
            
            m2 = jnp.linalg.inv(jnp.eye(self.n) - 2 * (theta2 @ var_sigma))
            temp = m2 @ (var_sigma @ theta1)
            # tr = tr_uv (m2 , var_sigma @ theta1)
            b_nu = self.beta * tr_u(temp)
            
            return b_nu
        else:
            # @jit
            def f1_jax(t1, theta1, theta2):
                a = self.compute_a_nu(t1, theta1, theta2)
                z1 = jnp.trace(a @ self.omega)
                # z1 = tr_uv(a @ self.omega)
                return z1.real.item()
            
            def f1_scipy(t1, theta1, theta2):
                return float(f1_jax(t1, theta1, theta2))
    
            r1 = scipy.integrate.quad(lambda t1: f1_scipy(t1, theta1, theta2), 0, t)
            c2r = r1[0]
            
            # @jit
            def f3_jax(t1, theta1, theta2):
                a = self.compute_a_nu(t1, theta1, theta2)
                z1 = jnp.trace(a @ self.omega)
                # z1 = tr_uv(a @ self.omega)
                return z1.imag.item()
            
            def f3_scipy(t1, theta1, theta2):
                return float(f3_jax(t1, theta1, theta2))
    
            r2 = scipy.integrate.quad(lambda t1: f3_scipy(t1, theta1, theta2), 0, t)
            c2i = r2[0]
            c3 = complex(c2r, c2i)
            return c3
    
    def compute_b_nu(self, t: float, theta1: jnp.ndarray, theta2: jnp.ndarray) -> complex:
        """
        Simplified compute_b_nu using JAX integration.
        """
        if self.is_bru_config:
            emt = jsl.expm(t * self.m)
            eAt, var_sigma_init = self.compute_var_sigma(t)
            var_sigma = self._vec_inv(var_sigma_init, (self.n, self.n))
        
            m2 = jnp.linalg.inv(jnp.eye(self.n) - 2 * (theta2 @ var_sigma))
            temp = m2 @ (var_sigma @ theta1)
            b_nu = self.beta * jnp.trace(temp)
        
            return b_nu
        else:
            # Use JAX integration instead of scipy
            n_points = 50
            t_vals = jnp.linspace(0, t, n_points)
        
            # Vectorized computation
            def integrand(t1):
                a = self.compute_a_nu(t1, theta1, theta2)
                return jnp.trace(a @ self.omega)
        
            # Compute all values at once
            integrand_vals = jax.vmap(integrand)(t_vals)
        
            # Integrate using trapezoidal rule
            real_part = jnp.trapezoid(integrand_vals.real, t_vals)
            imag_part = jnp.trapezoid(integrand_vals.imag, t_vals)
        
            return real_part + 1j * imag_part


    @partial(jit, static_argnums=(0,))
    def phi_two(self, z: complex, theta_1: jnp.ndarray, theta_2: jnp.ndarray) -> complex:
        """
        Compute second characteristic function Phi_Two.
        
        Parameters
        ----------
        z : complex
            Complex parameter
        theta_1 : jnp.ndarray
            First parameter matrix
        theta_2 : jnp.ndarray
            Second parameter matrix
            
        Returns
        -------
        complex
            Phi_Two value
            
        Notes
        -----
        Implements Equation (24): (tr[a_nu] + b_nu) * Phi_1(theta_2)
        """
        a_nu = self.compute_a_nu(self.maturity, theta_1, theta_2)
        b_nu = self.compute_b_nu(self.maturity, theta_1, theta_2)
        a_nu2 = tr_u(self.x0 @ a_nu)
        phi_1 = self.phi_one(1, theta_2)
        
        phi_two = (a_nu2 + b_nu) * phi_1
        
        return phi_two
    
    def phi_two_nu1(self, theta1: jnp.ndarray, theta2: jnp.ndarray) -> complex:
        """
        Compute Phi_Two_nu1 variant.
        
        Parameters
        ----------
        theta1 : jnp.ndarray
            First parameter matrix
        theta2 : jnp.ndarray
            Second parameter matrix
            
        Returns
        -------
        complex
            Phi_Two_nu1 value
        """
        t = self.maturity
        b_nu = self.compute_b_nu(t, theta1, theta2)
        a_nu = self.compute_a_nu(t, theta1, theta2)
        phi1 = self.phi_one(1, theta2)
        
        # res = (tr_uv(a_nu @ self.x0) + b_nu) * phi1
        res = (tr_u(a_nu @ self.x0) + b_nu) * phi1
        return res
    
    # ========== Phi_Two Vega Functions ==========
    
    @partial(jit, static_argnums=(0,))
    def compute_a_nu_vega(self, i: int, j: int, t: float, 
                         theta_1: jnp.ndarray, theta_2: jnp.ndarray) -> jnp.ndarray:
        """
        Compute Vega derivative of a_nu.
        
        Parameters
        ----------
        i : int
            Row index
        j : int
            Column index
        t : float
            Time parameter
        theta_1 : jnp.ndarray
            First parameter matrix
        theta_2 : jnp.ndarray
            Second parameter matrix
            
        Returns
        -------
        jnp.ndarray
            Vega derivative of a_nu
        """
        emt = jsl.expm(t * self.m)
        eAt, var_sigma_init = self.compute_var_sigma(t)
        var_sigma = self._vec_inv(var_sigma_init, (self.n, self.n))
        
        eAt_temp, var_sigma_vega_ij_init = self.compute_var_sigma_vega(i, j, t)
        var_sigma_vega_ij = self._vec_inv(var_sigma_vega_ij_init, (self.n, self.n))
        
        m2 = jnp.linalg.inv(jnp.eye(self.n) - 2 * (theta_2 @ var_sigma))
        emt_tr = jnp.transpose(emt)
        
        a_nu_ij = emt_tr @ m2 @ (2 * theta_2 @ var_sigma_vega_ij) @ m2 @ \
                  (2 * theta_1 @ var_sigma) @ m2 @ theta_2 @ emt
        a_nu_ij += emt_tr @ m2 @ (2 * theta_1 @ var_sigma_vega_ij) @ m2 @ theta_2 @ emt
        a_nu_ij += emt_tr @ m2 @ (2 * theta_1 @ var_sigma) @ m2 @ \
                   (2 * theta_2 @ var_sigma_vega_ij) @ m2 @ theta_2 @ emt
        a_nu_ij += emt_tr @ m2 @ (2 * theta_2 @ var_sigma_vega_ij) @ m2 @ theta_1 @ emt
        
        return a_nu_ij
    
    def compute_b_nu_vega(self, i: int, j: int, t: float,
                         theta1: jnp.ndarray, theta2: jnp.ndarray) -> complex:
        """
        Compute Vega derivative of b_nu.
        
        Parameters
        ----------
        i : int
            Row index
        j : int
            Column index
        t : float
            Time parameter
        theta1 : jnp.ndarray
            First parameter matrix
        theta2 : jnp.ndarray
            Second parameter matrix
            
        Returns
        -------
        complex
            Vega derivative of b_nu
        """
        if self.is_bru_config:
            emt = jsl.expm(t * self.m)
            eAt, var_sigma_init = self.compute_var_sigma(t)
            
            var_sigma = self._vec_inv(var_sigma_init, (self.n, self.n))
            eAt_temp, var_sigma_vega_ij_init = self.compute_var_sigma_vega(i, j, t)
            var_sigma_vega_ij = self._vec_inv(var_sigma_vega_ij_init, (self.n, self.n))
            
            m2 = jnp.linalg.inv(jnp.eye(self.n) - 2 * (var_sigma @ theta2))
            temp1 = m2 @ var_sigma_vega_ij @ theta2 @ m2 @ var_sigma @ theta1
            temp2 = m2 @ var_sigma_vega_ij @ theta1
            
            b_nu_vega = -2.0 * self.beta * tr_u(temp1) + self.beta * tr_u(temp2)
            
            return b_nu_vega
        else:
            def f1(t1, theta1, theta2):
                a = self.compute_a_nu_vega(i, j, t1, theta1, theta2)
                z1 = tr_u(a @ self.omega)
                return z1.real
            
            r1 = scipy.integrate.quad(lambda t1: f1(t1, theta1, theta2), 0, t)
            c2r = r1[0]
            
            def f3(t1, theta1, theta2):
                a = self.compute_a_nu_vega(i, j, t1, theta1, theta2)
                z1 = tr_u(a @ self.omega)
                return z1.imag
            
            r2 = scipy.integrate.quad(lambda t1: f3(t1, theta1, theta2), 0, t)
            c2i = r2[0]
            c3 = complex(c2r, c2i)
            return c3
    
    def phi_two_vega(self, i: int, j: int, z: complex,
                    theta_1: jnp.ndarray, theta_2: jnp.ndarray) -> complex:
        """
        Compute Vega derivative of Phi_Two.
        
        Parameters
        ----------
        i : int
            Row index
        j : int
            Column index
        z : complex
            Complex parameter
        theta_1 : jnp.ndarray
            First parameter matrix
        theta_2 : jnp.ndarray
            Second parameter matrix
            
        Returns
        -------
        complex
            Vega derivative of Phi_Two
        """
        a_nu = self.compute_a_nu(self.maturity, theta_1, theta_2)
        b_nu = self.compute_b_nu(self.maturity, theta_1, theta_2)
        
        a_nu_vega_ij = self.compute_a_nu_vega(i, j, self.maturity, theta_1, theta_2)
        b_nu_vega_ij = self.compute_b_nu_vega(i, j, self.maturity, theta_1, theta_2)
        
        a_nu_vega_ij_2 = tr_u(self.x0 @ a_nu_vega_ij)
        a_nu2 = tr_u(self.x0 @ a_nu)
        
        phi_1_vega_ij = self.phi_one_vega(i, j, 1, theta_2)
        phi_1 = self.phi_one(1, theta_2)
        
        phi_two_vega_ij = ((a_nu_vega_ij_2 + b_nu_vega_ij) * phi_1 + 
                          (a_nu2 + b_nu) * phi_1_vega_ij)
        
        return phi_two_vega_ij
    
    # ========== Phi_Three Functions ==========
    
    @partial(jit, static_argnums=(0,))
    def phi_three_nu1(self, z: complex, theta_1: jnp.ndarray, theta_2: jnp.ndarray) -> complex:
        """
        Compute Phi_Three_nu1 characteristic function.
        
        Parameters
        ----------
        z : complex
            Complex parameter
        theta_1 : jnp.ndarray
            First parameter matrix
        theta_2 : jnp.ndarray
            Second parameter matrix
            
        Returns
        -------
        complex
            Phi_Three_nu1 value
        """
        a_nu = self.compute_a_nu(self.maturity, theta_1, theta_1 + theta_2)
        b_nu = self.compute_b_nu(self.maturity, theta_1, theta_1 + theta_2)
        
        a_nu2 = tr_u(self.x0 @ a_nu)
        phi_1 = self.phi_one(1, theta_1 + theta_2)
        
        phi_three = (a_nu2 + b_nu) * phi_1
        
        return phi_three
    
    def phi_three_nu1_nu1(self, z: complex, theta_1: jnp.ndarray, theta_2: jnp.ndarray) -> complex:
        """
        Compute Phi_Three_nu1_nu1 characteristic function.
        
        Parameters
        ----------
        z : complex
            Complex parameter
        theta_1 : jnp.ndarray
            First parameter matrix
        theta_2 : jnp.ndarray
            Second parameter matrix
            
        Returns
        -------
        complex
            Phi_Three_nu1_nu1 value
        """
        def compute_a_nu_nu(t, theta1, theta2):
            emt = jsl.expm(t * self.m)
            eAt, var_sigma_init = self.compute_var_sigma(t)
            var_sigma = self._vec_inv(var_sigma_init, (self.n, self.n))
            
            m2 = jnp.linalg.inv(jnp.eye(self.n) - 2 * (theta2 @ var_sigma))
            emt_tr = jnp.transpose(emt)
            
            a_nu_nu = 2.0 * emt_tr @ m2 @ (2 * theta1 @ var_sigma) @ m2 @ theta1 @ emt
            a_nu_nu += 2.0 * emt_tr @ m2 @ (2.0 * theta1 @ var_sigma) @ m2 @ theta1 @ emt
            
            return a_nu_nu
        
        def compute_b_nu_nu(t, theta1, theta2):
            def f1(t1, theta1, theta2):
                a = compute_a_nu_nu(t1, theta1, theta2)
                z1 = tr_u(a @ self.omega)
                return z1.real
            
            r1 = scipy.integrate.quad(lambda t1: f1(t1, theta1, theta2), 0, t)
            c2r = r1[0]
            
            def f3(t1, theta1, theta2):
                a = compute_a_nu_nu(t1, theta1, theta2)
                z1 = tr_u(a @ self.omega)
                return z1.imag
            
            r2 = scipy.integrate.quad(lambda t1: f3(t1, theta1, theta2), 0, t)
            c2i = r2[0]
            b_nu_nu = complex(c2r, c2i)
            return b_nu_nu
        
        a_nu = self.compute_a_nu(self.maturity, theta_1, theta_1 + theta_2)
        b_nu = self.compute_b_nu(self.maturity, theta_1, theta_1 + theta_2)
        a_nu2 = tr_u(self.x0 @ a_nu)
        phi_1 = self.phi_one(1, theta_1 + theta_2)
        phi_three_1 = (a_nu2 + b_nu) * (a_nu2 + b_nu) * phi_1
        
        a_nu_nu = compute_a_nu_nu(self.maturity, theta_1, theta_1 + theta_2)
        b_nu_nu = compute_b_nu_nu(self.maturity, theta_1, theta_1 + theta_2)
        a_nu_nu2 = tr_u(self.x0 @ a_nu_nu)
        phi_three_2 = (a_nu_nu2 + b_nu_nu) * phi_1
        
        phi_three = phi_three_2 + phi_three_1
        
        return phi_three