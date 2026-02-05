"""
Wishart process with jump component implementation.
"""
import jax
import jax.numpy as jnp
from jax import jit
from jax.scipy import linalg as jlinalg
from functools import partial
from typing import Tuple, Optional, Union
import numpy as np
from scipy.stats import norm
import scipy.integrate
import pandas as pd
from scipy.special import binom, factorial
import sympy

from .base import BaseWishart
from .derivatives import WishartDerivatives
from .phi_functions import PhiFunctions

from ..components.jump import JumpComponent
# from ..math.derivatives import WishartDerivatives
# from ..pricing.phi_functions import PhiFunctions
from ..config.constants import COMPUTE_B_N

class WishartWithJump(BaseWishart, WishartDerivatives, PhiFunctions):
    """
    Wishart process with optional jump component.
    
    This class extends the base Wishart process to include
    discontinuous jumps according to a compound Poisson process.
    
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
    
    def __init__(self, n: int, x0: jnp.ndarray, omega: jnp.ndarray,
                 m: jnp.ndarray, sigma: jnp.ndarray):
        """Initialize Wishart process with jump capability."""
        super().__init__(n, x0, omega, m, sigma)
        super().set_wishart_parameter(n, x0, omega, m, sigma)

        self.has_jump = False
        self.jump_component = None
        self.c = None
        
         

        # Cache for variance computations
        self.var_sigma_t = pd.DataFrame(columns=['T', 'eAt', 'VarSigma'])
        self.a_symbolic = pd.DataFrame(columns=['T', 'ASymb'])
        self.use_cache = False
    
        ##Already set in base
        # self.G = 0
        # self.exp_gt = 0
        # self.h = 0
        # self.g0 = 0
        # self.h1 = 0 # (G^-1)*(exp(Gt)-I)*h
        # self.poly_prop = 0
        # self.block_struct = 0
        # self.pos = 0 # G[pos[i][j]] gives the G_{i,j} block matrix
        # More accurate integration with fewer points
        self.setup_quadrature()

    def set_jump(self, jump_component: JumpComponent):
        """
        Add jump component to the Wishart process.
        
        Parameters
        ----------
        jump_component : JumpComponent
            Jump component object
        """
        self.jump_component = jump_component
        self.c = self.jump_component.c
        self.has_jump = True
    
    @partial(jit, static_argnums=(0,))
    def compute_mean_wishart_decomp(self, t: float) -> Union[Tuple[jnp.ndarray, jnp.ndarray], 
                                                              Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        """
        Compute mean decomposition with optional jump component.
        
        Parameters
        ----------
        t : float
            Time parameter
            
        Returns
        -------
        eAt : jnp.ndarray
            Matrix exponential
        v3 : jnp.ndarray
            Drift contribution
        v3_2 : jnp.ndarray, optional
            Jump contribution (if jumps are present)
        """
        n_square = self.n * self.n
        eAt = jlinalg.expm(self.A * t)
        v2 = jnp.matmul(eAt - jnp.eye(n_square), self.b)
        # v3 = jnp.linalg.solve(self.A, v2)
        v3=self.A_inv @ v2
        
        if self.has_jump:
            v2_2 = jnp.matmul(eAt - jnp.eye(n_square), self.c)
            # v3_2 = jnp.linalg.solve(self.A, v2_2)
            v3_2=self.A_inv @ v2_2

            return eAt, v3, v3_2
        else:
            return eAt, v3
    
    def compute_mean_wishart(self, t: float) -> jnp.ndarray:
        """
        Compute E[vec(x_t)] including jump contribution.
        
        Parameters
        ----------
        t : float
            Time parameter
            
        Returns
        -------
        jnp.ndarray
            Expected value of vectorized Wishart process
        """
        # print(" ============== compute_mean_wishart WISHART-JUMP======================")

        if self.has_jump:
            vv0 = self._vec(self.x0)
            eAt, v3, v3_2 = self.compute_mean_wishart_decomp(t)
            v1 = jnp.matmul(eAt, vv0)
            v3_total = v3 + v3_2
            v4 = jnp.add(v1, v3_total)
            return v4
        else:
            return super().compute_mean_wishart(t)
    
    def compute_mean_decompose(self, u0: jnp.ndarray, t: float) -> Tuple[float, jnp.ndarray]:
        """
        Compute mean decomposition with jumps.
        
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
        if self.has_jump:
            eAt, v3, v3_2 = self.compute_mean_wishart_decomp(t)
            v3_total = v3 + v3_2
        else:
            eAt, v3 = self.compute_mean_wishart_decomp(t)
            v3_total = v3
        
        b0 = jnp.vdot(self._vec(jnp.transpose(u0)), v3_total)
        a0 = self._vec_inv(
            jnp.matmul(jnp.transpose(eAt), self._vec(jnp.transpose(u0))),
            (self.n, self.n)
        )
        a0 = jnp.transpose(a0)
        return b0, a0
    
    def add_jump_to_g(self, i: int, k: int, j: int, pos_i: int, 
                      df: pd.DataFrame, dj: list, G1: jnp.ndarray) -> jnp.ndarray:
        """
        Add jump contributions to G matrix.
        
        Parameters
        ----------
        i, k, j : int
            Polynomial indices
        pos_i : int
            Position index
        df : pd.DataFrame
            Polynomial properties dataframe
        dj : list
            Jump moments
        G1 : jnp.ndarray
            G matrix to update
            
        Returns
        -------
        jnp.ndarray
            Updated G matrix
        """
        # Add jump contributions for i terms
        for i1 in range(0, i):
            i2 = i - i1
            for k1 in range(0, k + 1):
                k2 = k - k1
                for j1 in range(0, j + 1):
                    j2 = j - j1
                    c = self.jump_component.lambda_intensity * binom(i, i1) * binom(k, k1) * binom(j, j1)
                    c = c * self.get_jump_moment(df, dj, i2, k2, j2)
                    c = float(c)
                    
                    if (i1 == 0) & (k1 == 0) & (j1 == 0):
                        self.h = self.h.at[pos_i, 0].add(c)
                    else:
                        pos_j = df[(df['i'] == i1) & (df['k'] == k1) & (df['j'] == j1)]['pos']
                        if len(pos_j) > 0:
                            G1 = G1.at[pos_i, int(pos_j.iloc[0])].set(c)
        
        # Add jump contributions for k terms
        for k1 in range(0, k):
            k2 = k - k1
            for j1 in range(0, j + 1):
                j2 = j - j1
                c = self.jump_component.lambda_intensity * binom(k, k1) * binom(j, j1)
                c = c * self.get_jump_moment(df, dj, 0, k2, j2)
                c = float(c)
                
                if (i == 0) & (k1 == 0) & (j1 == 0):
                    self.h = self.h.at[pos_i, 0].add(c)
                else:
                    pos_j = df[(df['i'] == i) & (df['k'] == k1) & (df['j'] == j1)]['pos']
                    if len(pos_j) > 0:
                        G1 = G1.at[pos_i, int(pos_j.iloc[0])].set(c)
        
        # Add jump contributions for j terms
        for j1 in range(0, j):
            j2 = j - j1
            c = self.jump_component.lambda_intensity * binom(j, j1)
            c = c * self.get_jump_moment(df, dj, 0, 0, j2)
            c = float(c)
            
            if (i == 0) & (k == 0) & (j1 == 0):
                self.h = self.h.at[pos_i, 0].add(c)
            else:
                pos_j = df[(df['i'] == i) & (df['k'] == k) & (df['j'] == j1)]['pos']
                if len(pos_j) > 0:
                    G1 = G1.at[pos_i, int(pos_j.iloc[0])].set(c)
        
        return G1
    
    def get_jump_moment(self, df: pd.DataFrame, dj: list, 
                       i2: int, k2: int, j2: int) -> float:
        """
        Get jump moment for given indices.
        
        Parameters
        ----------
        df : pd.DataFrame
            Polynomial properties
        dj : list
            Jump moments
        i2, k2, j2 : int
            Indices
            
        Returns
        -------
        float
            Jump moment value
        """
        pos_series = df[(df['i'] == i2) & (df['k'] == k2) & (df['j'] == j2)]['pos']
        if len(pos_series) > 0:
            id1 = int(pos_series.iloc[0])
            return dj[id1]
        else:
            return 0.0
    
    def compute_jump_moments(self, df: pd.DataFrame) -> list:
        """
        Compute jump moments using symbolic differentiation.
        
        Parameters
        ----------
        df : pd.DataFrame
            Polynomial properties
            
        Returns
        -------
        list
            Jump moment values
        """
        # Symbolic variables
        theta_11 = sympy.symbols('theta_11')
        theta_12 = sympy.symbols('theta_12')
        theta_22 = sympy.symbols('theta_22')
        
        theta = sympy.Matrix([[theta_11, theta_12/2], [theta_12/2, theta_22]])
        
        # Convert JAX arrays to sympy matrices
        if isinstance(self.jump_component.eta, jnp.ndarray):
            eta_np = np.array(self.jump_component.eta)
        else:
            eta_np = self.jump_component.eta
        
        eta1 = sympy.Matrix([[eta_np[0,0], eta_np[0,1]], 
                            [eta_np[1,0], eta_np[1,1]]])
        
        I1 = sympy.eye(2)
        m3 = I1 - 2*eta1*theta
        
        # Convert xi
        if isinstance(self.jump_component.xi, jnp.ndarray):
            xi_np = np.array(self.jump_component.xi)
        else:
            xi_np = self.jump_component.xi
        
        xi1 = sympy.Matrix([[xi_np[0,0], xi_np[0,1]], 
                           [xi_np[1,0], xi_np[1,1]]])
        
        m4 = 2*eta1*theta
        m5 = 2*xi1 * (m4 + m4*m4 + m4*m4*m4 + m4*m4*m4*m4 + m4*m4*m4*m4*m4)
        f = sympy.exp(sympy.matrices.expressions.Trace(m5).simplify()) / (m3.det())**(self.jump_component.nu/2)
        
        # Pre-compute derivatives
        derivatives_cache = {}
        l1 = []
        
        for l in range(0, df.shape[0]):
            df1 = df.iloc[l]
            i3 = int(df1['i'])
            k3 = int(df1['k'])
            j3 = int(df1['j'])
            
            deriv_key = (i3, k3, j3)
            
            if deriv_key not in derivatives_cache:
                v1 = sympy.diff(f, theta_11, i3, theta_12, k3, theta_22, j3).subs(
                    {theta_11: 0.0, theta_12: 0.0, theta_22: 0.0}).evalf()
                derivatives_cache[deriv_key] = v1
            else:
                v1 = derivatives_cache[deriv_key]
            
            l1.append(v1)
        
        return l1
    
    def build_g_NOT_GOOD_TO_BE_REMOVED(self, order: int = 3) -> None:
        """
        Build G matrix including jump contributions.
        
        Parameters
        ----------
        order : int, optional
            Maximum polynomial order, by default 3
        """
        # Build polynomial indices
        l1 = []
        pos = 0
        for l in range(1, order + 1):
            for i1 in range(0, l + 1):
                for k1 in range(0, l - i1 + 1):
                    j1 = l - i1 - k1
                    l1.append([l, i1, k1, j1, pos])
                    pos += 1
        
        df = pd.DataFrame(data=l1, columns=['order', 'i', 'k', 'j', 'pos'])
        
        # Jump moments if needed
        if self.has_jump:
            dj = self.compute_jump_moments(df)
        
        dfg1 = df.groupby(by='order')
        lpoly = dfg1.count()['i']
        lpoly.name = 'nbPoly'
        
        # Initialize G and h
        G1 = jnp.zeros((df.shape[0], df.shape[0]))
        self.h = jnp.zeros((df.shape[0], 1))
        
        # Fill G matrix (similar to WishartBru but with jump additions)
        for l in range(0, df.shape[0]):
            df1 = df.iloc[l]
            i = df1['i']
            k = df1['k']
            j = df1['j']
            pos_i = int(df1['pos'])
            
            # Standard Wishart contributions (same as in WishartBru)
            # ... (matrix filling code)
            
            # Add jump contributions if present
            if self.has_jump:
                G1 = self.add_jump_to_g(i, k, j, pos_i, df, dj, G1)
        
        self.G = G1
        self.poly_prop = df
        self.block_struct = lpoly
        
        # Build g(0)
        self.g0 = jnp.zeros((self.G.shape[0], 1))
        x01 = self._vech(self.x0)
        
        for l in range(0, df.shape[0]):
            df1 = df.iloc[l]
            i = df1['i']
            k = df1['k']
            j = df1['j']
            v = jnp.power(x01[0,0], i) * jnp.power(x01[1,0], k) * jnp.power(x01[2,0], j)
            self.g0 = self.g0.at[df1['pos'], 0].set(v)
    
    def compute_c_old(self, t: float, theta1: jnp.ndarray) -> complex:
        """
        Compute jump contribution to characteristic function.
        
        Parameters
        ----------
        t : float
            Time parameter
        theta1 : jnp.ndarray
            Parameter matrix
            
        Returns
        -------
        complex
            Jump contribution
        """
        def f1(t1, theta1):
            a = self.compute_a(t1, theta1)
            z1 = self.jump_component.compute_mgf_jump(a)
            return z1.real - 1.0
        
        r1 = scipy.integrate.quad(lambda t1: f1(t1, theta1), 0, t)
        c2r = r1[0]
        
        def f3(t1, theta1):
            a = self.compute_a(t1, theta1)
            z1 = self.jump_component.compute_mgf_jump(a)
            return z1.imag
        
        r2 = scipy.integrate.quad(lambda t1: f3(t1, theta1), 0, t)
        c2i = r2[0]
        c3 = complex(c2r, c2i)
        return c3

    @partial(jit, static_argnums=(0,))
    def compute_c_good(self, t: float, theta1: jnp.ndarray) -> complex:
        """
        Optimized jump contribution using JAX integration.
        """
        # Integration points
        n_points = 30  # Adjust for accuracy
        t_vals = jnp.linspace(0, t, n_points)
    
        # Compute all values at once
        @jit
        def compute_mgf_values(t_array):
            # Batch compute all 'a' matrices
            a_matrices = jax.vmap(lambda t1: self.compute_a(t1, theta1))(t_array)
            # Batch compute all MGF values
            return jax.vmap(self.jump_component.compute_mgf_jump)(a_matrices)
    
        mgf_values = compute_mgf_values(t_vals)
    
        # Integrate real and imaginary parts
        c2r = jnp.trapezoid(mgf_values.real - 1.0, t_vals)
        c2i = jnp.trapezoid(mgf_values.imag, t_vals)
    
        return c2r + 1j * c2i

    
    @partial(jit, static_argnums=(0,))
    def compute_c(self, t: float, theta1: jnp.ndarray) -> complex:
        """
        Extreme optimizations following compute_b style:
        1. Gauss-Legendre quadrature for integration
        2. Batch all exponentials using eigendecomposition
        3. Fused operations to minimize memory access
        4. Single pass for real and imaginary parts
        """
        # Transform Gauss-Legendre nodes from [-1, 1] to [0, t]
        t_vals = 0.5 * t * (self.gl_nodes + 1)
        weights_scaled = 0.5 * t * self.gl_weights
    
        # === CRITICAL OPTIMIZATION: Batch all exponentials ===
        # Compute exp(t_i * lambda_j) for all i,j at once
        exp_A_vals = jnp.exp(jnp.outer(t_vals, self.A_eigvals))
        exp_m_vals = jnp.exp(jnp.outer(t_vals, self.m_eigvals))
    
        @jit
        def integrand_extreme(exp_A, exp_m, weight):
            # === 1. FAST MATRIX EXPONENTIAL RECONSTRUCTION ===
            # eAt = V @ diag(exp_A) @ V^{-1}
            VA = self.A_eigvecs * exp_A[None, :]  # Broadcasting instead of diag
            eAt = VA @ self.A_eigvecs_inv
        
            Vm = self.m_eigvecs * exp_m[None, :]
            emt = Vm @ self.m_eigvecs_inv
        
            # === 2. OPTIMIZED VAR_SIGMA ===
            # Same as in compute_b
            eAt_vecSigma2 = eAt @ self.vecSigma2
            v2 = eAt_vecSigma2 - self.vecSigma2
            v3 = self.A_inv @ eAt_vecSigma2 - self.A_inv_vecSigma2
            var_sigma = v3.reshape((self.n, self.n))
        
            # === 3. COMPUTE A MATRIX ===
            # Following the compute_a function logic
            theta_var = theta1 @ var_sigma
            rhs = self.eye_n - 2 * theta_var
        
            # Compute m2 @ theta1 @ emt efficiently
            # Instead of computing m2 = inv(rhs), we solve rhs @ X = theta1 @ emt
            theta1_emt = theta1 @ emt
            m2_theta1_emt = jnp.linalg.solve(rhs, theta1_emt)
        
            # Final a matrix: a = emt.T @ (m2 @ theta1 @ emt)
            a = emt.T @ m2_theta1_emt
        
            # === 4. COMPUTE MGF AND RETURN WEIGHTED VALUES ===
            mgf_value = self.jump_component.compute_mgf_jump(a)
        
            # Return both real-1 and imaginary parts, weighted
            return weight * (mgf_value.real - 1.0), weight * mgf_value.imag
    
        # Vectorized computation with weights included
        # Returns two arrays: real_vals and imag_vals
        real_vals, imag_vals = jax.vmap(
            lambda exp_A, exp_m, w: integrand_extreme(exp_A, exp_m, w), 
            out_axes=(0, 0)
        )(exp_A_vals, exp_m_vals, weights_scaled)
    
        # Sum for Gauss-Legendre quadrature
        c_real = jnp.sum(real_vals)
        c_imag = jnp.sum(imag_vals)
    
        return c_real + 1j * c_imag

    @partial(jit, static_argnums=(0,))
    def phi_one(self, z: complex, theta1: jnp.ndarray) -> complex:
        """
        Compute first characteristic function with jumps.
        
        Parameters
        ----------
        z : complex
            Complex parameter
        theta1 : jnp.ndarray
            Parameter matrix
            
        Returns
        -------
        complex
            Phi_1 value including jump contribution
        """
        theta1 = jnp.multiply(z, theta1)
        t = self.maturity
        a = self.compute_a(t, theta1)
        b1 = self.compute_b(t, theta1)
        # print("b1=", b1)
        a2 = jnp.trace(self.x0 @ a)
        
        if self.has_jump:
            c1 = self.compute_c(t, theta1)
            phi_one = jnp.exp(a2 + b1 + self.jump_component.lambda_intensity * c1)
        else:
            phi_one = jnp.exp(a2 + b1)
        
        return phi_one
    
    # Methods from base that need to be implemented
    # @partial(jit, static_argnums=(0,2))
    # @partial(jit, static_argnums=(0,2))
    def compute_moments(self, t: float, order: int = 3) -> None:
        """Compute moments including jump contributions."""
        # if not self.cd_params_computed:
         
        self.build_g(order)
        self.compute_exp_gt(t)
        self.compute_h1()
        v1 = jnp.add(jnp.matmul(self.exp_gt, self.g0), self.h1)
        self.moments = v1
            # self.cd_params_computed=True

    
    # def compute_var_sigma(self, t):
    @partial(jit, static_argnums=(0,))
    def compute_var_sigma(self, t: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute e^{A t} and -A^-1 (e^{A t}- I_n*n)*vec(sigma^2).
        
        Parameters
        ----------
        t : float
            Time parameter
            
        Returns
        -------
        eAt : jnp.ndarray
            Matrix exponential
        v3 : jnp.ndarray
            Transformed variance-covariance vector
        """
        # Compute e^{A t} and -A^-1 (e^{A t}- I_n*n)*vec(sigma^2)
        # where x_t is a Wishart process.
        eAt = jlinalg.expm(t * self.A) # e^{A t}
        vecSigma2 = self._vec(self.sigma2)
        v2 = (eAt - jnp.eye(self.n * self.n)) @ vecSigma2 # (e^{A t}- I_n*n)*vec(sigma^2)
        # v3 = jnp.linalg.solve(self.A, v2) # A^-1 (e^{A t}- I_n*n)*vec(sigma^2)
        v3=self.A_inv @ v2

        return eAt, v3 
       
    @partial(jit, static_argnums=(0,))
    def compute_a_old(self, t: float, theta1: jnp.ndarray) -> jnp.ndarray:
        """Compute A matrix (same as WishartBru)."""
        emt = jlinalg.expm(t * self.m)
        eAt, var_sigma_init = self.compute_var_sigma(t)
        var_sigma = self._vec_inv(var_sigma_init, (self.n, self.n))
        
        m2 = jnp.linalg.inv(jnp.eye(self.n) - 2*(theta1 @ var_sigma))
        # emt_tr = jnp.transpose(emt)
        
        a = emt.T @ m2 @ theta1 @ emt
        return a

    @partial(jit, static_argnums=(0,))
    def compute_a(self, t: float, theta1: jnp.ndarray) -> jnp.ndarray:
        """Optimized compute_a"""
        """Compute A matrix (same as WishartBru)."""
        emt = jlinalg.expm(t * self.m)
        eAt, var_sigma_vec = self.compute_var_sigma(t)
        var_sigma = var_sigma_vec.reshape((self.n, self.n))  # Faster than _vec_inv
        
        theta_var = theta1 @ var_sigma
        m2 = jnp.linalg.inv(self.eye_n - 2*theta_var)
        
        a = emt.T @ m2 @ theta1 @ emt
        return a
    
    @partial(jit, static_argnums=(0,))
    # def compute_b(self, t: float, theta1: jnp.ndarray) -> complex:
    def compute_b_old(self, t: float, theta1: jnp.ndarray) -> complex:
        """Compute B function (same as WishartBru)."""
        
        def f_complex(t1):
            a = self.compute_a(t1, theta1)
            return jnp.trace(a @ self.omega)
        
        n_points = COMPUTE_B_N#100
        t_vals = jnp.linspace(0, t, n_points)
        integrand_vals = jax.vmap(f_complex)(t_vals)
        res = jnp.trapezoid(integrand_vals, t_vals)
        
        return res
    
    @partial(jit, static_argnums=(0,))
    def compute_b_just_ok(self, t: float, theta1: jnp.ndarray) -> complex:
        """Override with optimized version"""
        n_points = COMPUTE_B_N
        t_vals = jnp.linspace(0, t, n_points)
        
        # Inline all operations
        @jit
        def fast_integrand(t_single):
            # Matrix exponentials
            At = t_single * self.A
            mt = t_single * self.m
            eAt = jlinalg.expm(At)
            emt = jlinalg.expm(mt)
            
            # Var sigma computation
            v2 = (eAt - self.eye_nn) @ self.vecSigma2
            var_sigma = (-self.A_inv @ v2).reshape((self.n, self.n))
            
            # Compute trace without forming full 'a' matrix
            theta_var = theta1 @ var_sigma
            m2_inv = jnp.linalg.inv(self.eye_n - 2*theta_var)
            
            # Efficient trace computation
            temp1 = theta1 @ emt
            temp2 = m2_inv @ temp1
            temp3 = emt.T @ temp2
            
            return jnp.trace(temp3 @ self.omega)
        
        integrand_vals = jax.vmap(fast_integrand)(t_vals)
        return jnp.trapezoid(integrand_vals, t_vals)
    
    @partial(jit, static_argnums=(0,))
    def compute_b_vectorized(self, t: float, theta1: jnp.ndarray) -> complex:
        """
        Fully vectorized compute_b that minimizes redundant computations
        """
        n_points = COMPUTE_B_N
        t_vals = jnp.linspace(0, t, n_points)
        
        # Batch compute all matrix exponentials at once
        @jit
        def compute_all_exponentials(t_array):
            # Compute all e^{tA} and e^{tm} at once
            def single_exp_computation(t_single):
                eAt = jlinalg.expm(t_single * self.A)
                emt = jlinalg.expm(t_single * self.m)
                return eAt, emt
            
            return jax.vmap(single_exp_computation)(t_array)
        
        # Get all exponentials
        eAt_all, emt_all = compute_all_exponentials(t_vals)
        
        # Batch compute var_sigma for all time points
        @jit
        def compute_var_sigma_batch(eAt_batch):
            def single_var_sigma(eAt):
                v2 = (eAt - self.eye_nn) @ self.vecSigma2
                v3 = self.A_inv @ v2
                return v3.reshape((self.n, self.n))
            
            return jax.vmap(single_var_sigma)(eAt_batch)
        
        var_sigma_all = compute_var_sigma_batch(eAt_all)
        
        # Batch compute all 'a' matrices and traces
        @jit
        def compute_traces_batch(emt_batch, var_sigma_batch):
            def single_trace(emt, var_sigma):
                theta_var = theta1 @ var_sigma
                m2 = jnp.linalg.inv(self.eye_n - 2*theta_var)
                a = emt.T @ m2 @ theta1 @ emt
                return jnp.trace(a @ self.omega)
            
            return jax.vmap(single_trace)(emt_batch, var_sigma_batch)
        
        integrand_vals = compute_traces_batch(emt_all, var_sigma_all)
        
        # Integrate
        res = jnp.trapezoid(integrand_vals, t_vals)
        return res
    
    # Keep original compute_b for compatibility
    @partial(jit, static_argnums=(0,))
    def compute_b_a_bit_better(self, t: float, theta1: jnp.ndarray) -> complex:
        """Use vectorized version"""
        return self.compute_b_vectorized(t, theta1)
    
    @partial(jit, static_argnums=(0,))
    def compute_b_way_better(self, t: float, theta1: jnp.ndarray) -> complex:
        n_points = 25  # Much fewer points
        t_vals = jnp.linspace(0, t, n_points)
        
        # Vectorized eigendecomposition approach
        # Compute all exponentials at once
        exp_A_vals = jnp.exp(jnp.outer(t_vals, self.A_eigvals))  # shape: (n_points, n*n)
        exp_m_vals = jnp.exp(jnp.outer(t_vals, self.m_eigvals))  # shape: (n_points, n)
        
        @jit
        def integrand(exp_A, exp_m):
            # Reconstruct matrices from eigendecomposition
            eAt = self.A_eigvecs @ jnp.diag(exp_A) @ self.A_eigvecs_inv
            emt = self.m_eigvecs @ jnp.diag(exp_m) @ self.m_eigvecs_inv
            
            # Compute var_sigma
            v2 = (eAt - self.eye_nn) @ self.vecSigma2
            v3 = self.A_inv @ v2
            var_sigma = v3.reshape((self.n, self.n))
            
            # Compute a matrix
            theta_var = theta1 @ var_sigma
            m2 = jnp.linalg.inv(self.eye_n - 2 * theta_var)
            a = emt.T @ m2 @ theta1 @ emt
            
            # Return trace
            return jnp.trace(a @ self.omega)
        
        # Apply to all time points
        integrand_vals = jax.vmap(integrand)(exp_A_vals, exp_m_vals)
        
        # Integrate
        return jnp.trapezoid(integrand_vals, t_vals)
    
    # Alternative: Even more optimized version with everything inlined
    @partial(jit, static_argnums=(0,))
    # def compute_b_even_more_optimized(self, t: float, theta1: jnp.ndarray) -> complex:
    def compute_b_good(self, t: float, theta1: jnp.ndarray) -> complex:
        n_points = 25
        t_vals = jnp.linspace(0, t, n_points)
        
        # Batch compute all exponential values
        exp_A_vals = jnp.exp(jnp.outer(t_vals, self.A_eigvals))
        exp_m_vals = jnp.exp(jnp.outer(t_vals, self.m_eigvals))
        
        @jit
        def integrand_optimized(exp_A, exp_m):
            # Reconstruct eAt efficiently
            # eAt = V @ diag(exp_A) @ V^{-1}
            eAt_temp = self.A_eigvecs @ jnp.diag(exp_A)
            eAt = eAt_temp @ self.A_eigvecs_inv
            
            # Reconstruct emt efficiently
            emt_temp = self.m_eigvecs @ jnp.diag(exp_m)
            emt = emt_temp @ self.m_eigvecs_inv
            
            # Var sigma computation (inlined)
            v2 = eAt @ self.vecSigma2 - self.vecSigma2  # More efficient than (eAt - I)@vec
            var_sigma_flat = self.A_inv @ v2
            var_sigma = var_sigma_flat.reshape((self.n, self.n))
            
            # Compute trace without forming full 'a' matrix
            # trace(emt.T @ m2 @ theta1 @ emt @ omega)
            # = trace(m2 @ theta1 @ emt @ omega @ emt.T)
            
            # First compute theta1 @ var_sigma
            theta_var = theta1 @ var_sigma
            
            # Compute m2 = (I - 2*theta_var)^{-1}
            # Instead of inverting, solve the system
            m2_denom = self.eye_n - 2 * theta_var
            
            # Compute theta1 @ emt @ omega @ emt.T
            temp1 = theta1 @ emt
            temp2 = temp1 @ self.omega
            temp3 = temp2 @ emt.T
            
            # Solve m2_denom @ result = temp3 for result, then take trace
            # This is more numerically stable than computing m2 = inv(m2_denom)
            result = jnp.linalg.solve(m2_denom, temp3)
            
            return jnp.trace(result)
        
        # Vectorized application
        integrand_vals = jax.vmap(integrand_optimized)(exp_A_vals, exp_m_vals)
        
        # Integration
        return jnp.trapezoid(integrand_vals, t_vals)

    # def setup_quadrature(self, n_points=15):
    def setup_quadrature(self, n_points=10):
        """Setup Gauss-Legendre quadrature nodes and weights."""
        # Get Gauss-Legendre nodes and weights on [-1, 1]
        nodes, weights = np.polynomial.legendre.leggauss(n_points)
        self.gl_nodes = jnp.array(nodes)
        self.gl_weights = jnp.array(weights)
        
    @partial(jit, static_argnums=(0,))
    # def compute_b_extreme(self, t: float, theta1: jnp.ndarray) -> complex:
    def compute_b(self, t: float, theta1: jnp.ndarray) -> complex:
        """
        Extreme optimizations:
        1. Gauss-Legendre quadrature (15 points instead of 25)
        2. Fused operations to minimize memory access
        3. Optimized matrix multiplications
        4. Avoid redundant computations
        """
        # Transform Gauss-Legendre nodes from [-1, 1] to [0, t]
        t_vals = 0.5 * t * (self.gl_nodes + 1)
        weights_scaled = 0.5 * t * self.gl_weights
        
        # === CRITICAL OPTIMIZATION: Batch all exponentials ===
        # Compute exp(t_i * lambda_j) for all i,j at once
        exp_A_vals = jnp.exp(jnp.outer(t_vals, self.A_eigvals))
        exp_m_vals = jnp.exp(jnp.outer(t_vals, self.m_eigvals))
        
        # Precompute theta1 @ omega once (used in trace)
        theta1_omega = theta1 @ self.omega
        
        @jit
        def integrand_extreme(exp_A, exp_m, weight):
            # === 1. FAST MATRIX EXPONENTIAL RECONSTRUCTION ===
            # eAt = V @ diag(exp_A) @ V^{-1}
            # Optimize: compute V @ diag(exp_A) first
            VA = self.A_eigvecs * exp_A[None, :]  # Broadcasting instead of diag
            eAt = VA @ self.A_eigvecs_inv
            
            Vm = self.m_eigvecs * exp_m[None, :]
            emt = Vm @ self.m_eigvecs_inv
            
            # === 2. OPTIMIZED VAR_SIGMA ===
            # v2 = (eAt - I) @ vecSigma2
            # Optimize: compute eAt @ vecSigma2 - vecSigma2
            eAt_vecSigma2 = eAt @ self.vecSigma2
            v2 = eAt_vecSigma2 - self.vecSigma2
            
            # v3 = -A_inv @ v2
            # But we can optimize further since A_inv @ vecSigma2 is precomputed
            v3 = self.A_inv @ eAt_vecSigma2 - self.A_inv_vecSigma2
            
            var_sigma = v3.reshape((self.n, self.n))
            
            # === 3. OPTIMIZED TRACE COMPUTATION ===
            # We want: trace(emt.T @ m2 @ theta1 @ emt @ omega)
            # Rewrite as: trace(m2 @ theta1 @ emt @ omega @ emt.T)
            
            # Compute theta_var = theta1 @ var_sigma
            theta_var = theta1 @ var_sigma
            
            # Compute rhs = I - 2*theta_var for solving
            rhs = self.eye_n - 2 * theta_var
            
            # Compute theta1 @ emt @ omega @ emt.T more efficiently
            # First: emt @ omega @ emt.T
            emt_omega = emt @ self.omega
            emt_omega_emtT = emt_omega @ emt.T
            
            # Then: theta1 @ (emt @ omega @ emt.T)
            theta1_emt_omega_emtT = theta1 @ emt_omega_emtT
            
            # Solve rhs @ x = theta1_emt_omega_emtT for x
            # This gives us m2 @ theta1 @ emt @ omega @ emt.T
            result_matrix = jnp.linalg.solve(rhs, theta1_emt_omega_emtT)
            
            # Return weighted trace
            return weight * jnp.trace(result_matrix)
        
        # Vectorized computation with weights included
        integrand_vals = jax.vmap(integrand_extreme)(exp_A_vals, exp_m_vals, weights_scaled)
        
        # Sum (Gauss-Legendre quadrature)
        return jnp.sum(integrand_vals)
    
    def compute_exp_gt(self, t):
        egt = jnp.zeros(self.G.shape)
        gt = jnp.multiply(t, self.G)
        
        #print(self.G)

        # Exponentiate the diagonal blocks
        for i in range(0, len(self.pos)):
            egt = egt.at[self.pos[i][i]].set(jlinalg.expm(gt[self.pos[i][i]]))
        #self.PrintMatBlock(egt)
            
        #print(self.block_struct)

        # Exponentiate off diagonal blocks
        for i in range(1, len(self.pos)):
            for j in range(i - 1, -1, -1):
                #print("block:",i,j)
                f1 = jnp.zeros((self.block_struct.iloc[i], self.block_struct.iloc[j]))
                #print(self.block_struct.iloc[i],self.block_struct.iloc[j])
                for p in range(0, i - j):
                    #print(i,j + p,j + p,j,i,i - p,i - p,j)##Using egt(which is matrix F) below as recursive and diagonal is already calcualted
                    f1 = jnp.add(f1, jnp.subtract(jnp.matmul(gt[self.pos[i][j + p]], egt[self.pos[j + p][j]]), jnp.matmul(egt[self.pos[i][i - p]], gt[self.pos[i - p][j]])))
                # f2 = jlinalg.solve_sylvester(jnp.multiply(-1.0, gt[self.pos[i][i]]), gt[self.pos[j][j]], f1)
                f2 = jnp.array(scipy.linalg.solve_sylvester(
                                np.array(-gt[self.pos[i][i]]), 
                                np.array(gt[self.pos[j][j]]), 
                                np.array(f1)
                                ))
               
                egt = egt.at[self.pos[i][j]].set(f2)
        
        self.exp_gt = egt        
        #self.PrintMatBlock(egt)
        #print(egt)
        #print(jlinalg.expm(gt))
        #print(jnp.max(jnp.subtract(egt, jlinalg.expm(gt))))
        #print(jnp.min(jnp.subtract(egt, jlinalg.expm(gt))))

    def compute_h1(self):
        # This function should NOT be JIT compiled due to dynamic indexing
        # Compute G^{-1}*(e^{G t}-I)*h
        h0 = jnp.matmul(jnp.subtract(self.exp_gt, jnp.identity(self.exp_gt.shape[0])), self.h)
    
        # Pre-compute indices
        if hasattr(self.block_struct, 'values'):
            block_values = self.block_struct.values
        else:
            block_values = self.block_struct
    
        # Compute cumulative sum
        cumsum_values = np.cumsum(block_values)
        index1 = np.zeros(len(cumsum_values) + 1, dtype=np.int32)
        index1[1:] = cumsum_values
    
        h1 = jnp.zeros_like(h0)
    
        # Convert to numpy for indexing, then back to JAX
        h0_np = np.array(h0)
        h1_np = np.array(h1)
    
        for i in range(len(index1) - 1):
            start_i = int(index1[i])
            end_i = int(index1[i + 1])
        
            # Extract block using numpy indexing
            h2 = h0_np[start_i:end_i]
        
            for j in range(i):
                start_j = int(index1[j])
                end_j = int(index1[j + 1])
            
                h1_slice = h1_np[start_j:end_j]
                # Convert to JAX for matrix multiplication
                h2 = h2 - np.array(jnp.matmul(self.G[self.pos[i][j]], jnp.array(h1_slice)))
        
            # Solve using JAX
            h3 = np.array(jlinalg.solve(self.G[self.pos[i][i]], jnp.array(h2)))
        
            # Update h1
            # h1_np[start_i:end_i] = h3.ravel()
            # Update h1 - preserve the shape
            if h1_np.ndim == 2:
                h1_np[start_i:end_i] = h3.reshape(-1, h1_np.shape[1])
            else:
                h1_np[start_i:end_i] = h3
    
        # Convert back to JAX array
        self.h1 = jnp.array(h1_np)
 
    def build_g(self, order=3):
        """
        Build the matrix G, g(0) and h using JAX arrays while maintaining 
        the same structure as the original function.
        """
        # Build the polynomial indices
        l1 = []
        pos = 0
        for l in range(1, order + 1):
            for i1 in range(0, l + 1):
                for k1 in range(0, l - i1 + 1):
                    j1 = l - i1 - k1
                    l1.append([l, i1, k1, j1, pos])
                    pos += 1
    
        df = pd.DataFrame(data=l1, columns=['order', 'i', 'k', 'j', 'pos'])
    
        # Jump component if needed
        if self.has_jump:
            dj = self.compute_jump_moments(df)
    
        dfg1 = df.groupby(by='order')
        lpoly = dfg1.count()['i']
        lpoly.name = 'nbPoly'
    
        # Initialize G and h as JAX arrays
        G1 = jnp.zeros((df.shape[0], df.shape[0]))
        self.h = jnp.zeros((df.shape[0], 1))
    
        # Convert parameters to JAX arrays for consistency
        m = jnp.array(self.m)
        omega = jnp.array(self.omega)
        sigma2 = jnp.array(self.sigma2)
    
        # Fill G matrix
        for l in range(0, df.shape[0]):
            df1 = df.iloc[l]
            i = df1['i']
            k = df1['k']
            j = df1['j']
            pos_i = int(df1['pos'])
        
            # Diagonal element
            G1 = G1.at[pos_i, pos_i].set(
                i * 2 * m[0,0] + k * (m[0,0] + m[1,1]) + 2 * j * m[1,1]
            )
        
            # Off-diagonal elements
            if ((i > 0) & (k < order)):
                mask = (df['i'] == (i - 1)) & (df['k'] == (k + 1)) & (df['j'] == j)
                if mask.any():
                    pos_j = int(df[mask]['pos'].iloc[0])
                    G1 = G1.at[pos_i, pos_j].set(i * 2 * m[0,1])
        
            if ((i < order) & (k > 0)):
                mask = (df['i'] == (i + 1)) & (df['k'] == (k - 1)) & (df['j'] == j)
                if mask.any():
                    pos_j = int(df[mask]['pos'].iloc[0])
                    G1 = G1.at[pos_i, pos_j].set(k * m[1,0])
        
            if ((k > 0) & (j < order)):
                mask = (df['i'] == i) & (df['k'] == (k - 1)) & (df['j'] == (j + 1))
                if mask.any():
                    pos_j = int(df[mask]['pos'].iloc[0])
                    G1 = G1.at[pos_i, pos_j].set(k * m[0,1])
        
            if ((k < order) & (j > 0)):
                mask = (df['i'] == i) & (df['k'] == (k + 1)) & (df['j'] == (j - 1))
                if mask.any():
                    pos_j = int(df[mask]['pos'].iloc[0])
                    G1 = G1.at[pos_i, pos_j].set(j * 2 * m[1,0])
        
            if(df1['order'] > 1):
                if (i > 0):
                    mask = (df['i'] == (i - 1)) & (df['k'] == k) & (df['j'] == j)
                    if mask.any():
                        pos_j = int(df[mask]['pos'].iloc[0])
                        G1 = G1.at[pos_i, pos_j].set(
                            i * omega[0,0] + 2 * i * k * sigma2[0,0] + 2 * i * (i - 1) * sigma2[0,0]
                        )
            
                if (k > 0):
                    mask = (df['i'] == i) & (df['k'] == (k - 1)) & (df['j'] == j)
                    if mask.any():
                        pos_j = int(df[mask]['pos'].iloc[0])
                        G1 = G1.at[pos_i, pos_j].set(
                            k * omega[0,1] + k * (k - 1) * sigma2[0,1] + 2 * (i * k + j * k) * sigma2[0,1]
                        )
            
                if (j > 0):
                    mask = (df['i'] == i) & (df['k'] == k) & (df['j'] == (j - 1))
                    if mask.any():
                        pos_j = int(df[mask]['pos'].iloc[0])
                        G1 = G1.at[pos_i, pos_j].set(
                            j * omega[1,1] + 2 * j * (j - 1) * sigma2[1,1] + 2 * j * k * sigma2[1,1]
                        )
            
                if ((i < order) & (k > 1)):
                    mask = (df['i'] == (i + 1)) & (df['k'] == (k - 2)) & (df['j'] == j)
                    if mask.any():
                        pos_j = int(df[mask]['pos'].iloc[0])
                        G1 = G1.at[pos_i, pos_j].set((k * (k - 1) / 2) * sigma2[1,1])
            
                if ((k > 1) & (j < order)):
                    mask = (df['i'] == i) & (df['k'] == (k - 2)) & (df['j'] == (j + 1))
                    if mask.any():
                        pos_j = int(df[mask]['pos'].iloc[0])
                        G1 = G1.at[pos_i, pos_j].set((k * (k - 1) / 2) * sigma2[0,0])
            
                if ((i > 0) & (k < order) & (j > 0)):
                    mask = (df['i'] == (i - 1)) & (df['k'] == (k + 1)) & (df['j'] == (j - 1))
                    if mask.any():
                        pos_j = int(df[mask]['pos'].iloc[0])
                        G1 = G1.at[pos_i, pos_j].set(4 * i * j * sigma2[0,1])
            else:
                # For the moment of order 1
                if (i > 0):
                    self.h = self.h.at[pos_i, 0].set(i * omega[0,0])
            
                if (k > 0):
                    self.h = self.h.at[pos_i, 0].set(k * omega[0,1])
            
                if (j > 0):
                    self.h = self.h.at[pos_i, 0].set(j * omega[1,1])
        
            # Add jump terms if needed
            if self.has_jump:
                G1 = self.add_jump_to_g(i, k, j, pos_i, df, dj, G1)
    
        self.G = G1
        self.poly_prop = df
        self.block_struct = lpoly
    
        # Build dataframe to find the sub matrix blocks in G
        index1 = self.block_struct.cumsum()
        index1.loc[0] = 0
        index1.sort_index(inplace=True)
    
        l = []
        for i in range(0, index1.shape[0] - 1):
            l.append(range(index1.loc[i], index1.loc[i + 1]))
    
        l1 = []
        for i in range(0, index1.shape[0] - 1):
            l2 = []
            for j in range(0, index1.shape[0] - 1):
                v = tuple(jnp.meshgrid(jnp.array(l[i]), jnp.array(l[j]), indexing='ij'))
                l2.append(v)
            l1.append(l2)
    
        self.pos = l1
    
        # Build g(0) using JAX arrays
        self.g0 = jnp.zeros((self.G.shape[0], 1))
        x01 = self._vech(self.x0)  # Assuming this returns a JAX array or convert it
    
        # If Vech doesn't return JAX array, convert it
        if not isinstance(x01, jnp.ndarray):
            x01 = jnp.array(x01)
    
        for l in range(0, df.shape[0]):
            df1 = df.iloc[l]
            i = df1['i']
            k = df1['k']
            j = df1['j']
            v = jnp.power(x01[0,0], i) * jnp.power(x01[1,0], k) * jnp.power(x01[2,0], j)
            self.g0 = self.g0.at[df1['pos'], 0].set(v)

    def compute_jump_moments(self, df):
        """
        Compute jump moments using symbolic differentiation with sympy,
        but optimize the evaluation using JAX where possible.
        """
        # Symbolic computation remains with sympy
        theta_11 = sympy.symbols('theta_11')
        theta_12 = sympy.symbols('theta_12')
        theta_22 = sympy.symbols('theta_22')
    
        theta = sympy.Matrix([[theta_11, theta_12/2], [theta_12/2, theta_22]])
    
        # Convert JAX arrays to sympy matrices if needed
        if isinstance(self.jump_component.eta, jnp.ndarray):
            eta_np = np.array(self.jump_component.eta)
        else:
            eta_np = self.jump_component.eta
        
        eta1 = sympy.Matrix([[eta_np[0,0], eta_np[0,1]], 
                            [eta_np[1,0], eta_np[1,1]]])
    
        I1 = sympy.eye(2)
        m3 = I1 - 2*eta1*theta
    
        
        # Convert xi to sympy matrix
        if isinstance(self.jump_component.xi, jnp.ndarray):
            xi_np = np.array(self.jump_component.xi)
        else:
            xi_np = self.jump_component.xi
        
        xi1 = sympy.Matrix([[xi_np[0,0], xi_np[0,1]], 
                           [xi_np[1,0], xi_np[1,1]]])
    
        m4 = 2*eta1*theta
        m5 = 2*xi1 * (m4 + m4*m4 + m4*m4*m4 + m4*m4*m4*m4 + m4*m4*m4*m4*m4)
        f = sympy.exp(sympy.matrices.expressions.Trace(m5).simplify()) / (m3.det())**(self.jump_component.nu/2)
    
        # Pre-compute all unique derivatives to avoid redundant computation
        derivatives_cache = {}
        l1 = []
    
        for l in range(0, df.shape[0]):
            df1 = df.iloc[l]
            i3 = int(df1['i'])
            k3 = int(df1['k'])
            j3 = int(df1['j'])
        
            # Create a key for caching
            deriv_key = (i3, k3, j3)
        
            if deriv_key not in derivatives_cache:
                # Compute the derivative
                v1 = sympy.diff(f, theta_11, i3, theta_12, k3, theta_22, j3).subs(
                    {theta_11: 0.0, theta_12: 0.0, theta_22: 0.0}).evalf()
                derivatives_cache[deriv_key] = v1
            else:
                v1 = derivatives_cache[deriv_key]
            
            l1.append(v1)
    
        return l1

    def compute_mu(self, b3, a3, order=3):
        all_moments_Y = jnp.zeros(order+1)
        all_moments_Y = all_moments_Y.at[0].set(1.0)
        
        B = b3
        A1 = a3[0,0]
        A2 = a3[1,1]
        A12 = a3[0,1]
        
        if A12 == 0:
            for n1 in range(1, order+1):
                momentY = 0
                for i1 in range(0, n1+1):
                    for k1 in range(0, n1-i1+1):
                        j1 = n1-i1-k1
                        multinomialCoef = (factorial(n1)) / (factorial(i1) * factorial(k1) * factorial(j1))
                        l1_l2 = i1 + j1
                        current_moment = self.get_moments(i1, 0, j1)
                        momentY += multinomialCoef * (B**k1) * (A1**i1) * (A2**j1) * current_moment
            
                all_moments_Y = all_moments_Y.at[n1].set(momentY)
        else: # in progress for the general case
            for n1 in range(1, order+1):
                momentY = 0
                for i1 in range(0, n1+1):
                    for k1 in range(0, n1-i1+1):
                        for l1 in range(0, n1-i1-k1+1):
                            j1 = n1-i1-k1-l1
                            multinomialCoef = (factorial(n1)) / (factorial(i1) * factorial(k1) * factorial(j1))
                            
                            current_moment = self.get_moments(i1, l1, j1)
                            momentY += multinomialCoef * (B**k1) * (A1**i1) * (A12**l1) * (A2**j1) * current_moment
            
                all_moments_Y = all_moments_Y.at[n1].set(momentY)
        
        return all_moments_Y
    
    def collin_dufresne_c(self, mu, order=3):
        c = jnp.zeros(8)
        c = c.at[0].set(mu[0])
        c = c.at[1].set(mu[1])
        c = c.at[2].set(mu[2] - mu[1] * mu[1])
        
        if c[2] < 0:
            # print(f"{c[2]} is negative and mu is {mu}")
            c = c.at[2].set(1.0e-6)
            
        c = c.at[3].set(mu[3] - 3 * mu[1] * mu[2] + 2 * (mu[1]**3))
        
        if order == 4:
            c = c.at[4].set(mu[4] - 4 * mu[1] * mu[3] - 3*jnp.power(mu[2], 2) + 12*jnp.power(mu[1], 2) * mu[2] - 6*jnp.power(mu[1], 4))

            c = c.at[5].set(mu[5] - 5 * mu[1] * mu[4] - 10 * mu[2] * mu[3] + 20*jnp.power(mu[1], 2) * mu[3] \
                  + 30*mu[1]*jnp.power(mu[2], 2) - 60*mu[2]*jnp.power(mu[1], 3) + 24*jnp.power(mu[1], 5))

            c = c.at[6].set(mu[6] - 6 * mu[1] * mu[5] - 15 * mu[2] * mu[4] + 30*mu[4]*jnp.power(mu[1], 2)\
                   - 10*jnp.power(mu[3], 2) + 120*mu[1]*mu[2]*mu[3] - 120*mu[3]*jnp.power(mu[1], 3)\
                   + 30*jnp.power(mu[2], 3) - 270*jnp.power(mu[1]*mu[2], 2) + 360*mu[2]*jnp.power(mu[1], 4) - 120*jnp.power(mu[1], 6))

            c = c.at[7].set(mu[7] - 7 * mu[1] * mu[6] - 21 * mu[2] * mu[5] - 35*mu[3] * mu[4] + 140*jnp.power(mu[3], 2) * mu[1] \
                   - 630*jnp.power(mu[2], 3) * mu[1] + 210*mu[1]*mu[2]*mu[4] - 1260*mu[2]*mu[3]*jnp.power(mu[1], 2)\
                   + 42*jnp.power(mu[1], 5) * mu[5] + 2520*jnp.power(mu[1], 3)*jnp.power(mu[2], 2) \
                   - 210*jnp.power(mu[1], 3)*mu[4] + 210*jnp.power(mu[2], 2)*mu[3]\
                   + 840*jnp.power(mu[1], 4)*mu[3] - 2520*jnp.power(mu[1], 5)*mu[2] + 720*jnp.power(mu[1], 7))
           
            # Set to 0 for now
            c = c.at[5].set(0)
            c = c.at[6].set(0) 
            c = c.at[7].set(0)
            
        return c
    
    def collin_dufresne_lambda(self, c, strikeOffset=0, order=3):
        K = 0 
        n1 = (c[1] - K) / jnp.sqrt(c[2])
        n1_bar = (K - c[1]) / jnp.sqrt(c[2])
        n2 = 1.0 / jnp.sqrt(2 * jnp.pi * c[2])
        
        # Use scipy.stats.norm for CDF as JAX doesn't have a built-in normal CDF
        cdf1 = norm.cdf(float(n1))
        cdf2 = norm.cdf(float(n1_bar))
        
        if order == 3:
            lambda_ = jnp.zeros(5)
            lambda_ = lambda_.at[0].set(cdf1)
            lambda_ = lambda_.at[1].set(n2 * jnp.exp(-0.5 * n1 * n1) * c[2])
            lambda_ = lambda_.at[2].set(c[2] * lambda_[0] + lambda_[1] * (K - c[1]))
            lambda_ = lambda_.at[3].set(lambda_[1] * (((K - c[1])**2) + 2 * c[2]))
            lambda_ = lambda_.at[4].set(3 * lambda_[0] * c[2] * c[2] + lambda_[1] * (((K - c[1])**3) + 3 * c[2] * (K - c[1])))
        
        if order == 4:
            lambda_ = jnp.zeros(8)
            lambda_ = lambda_.at[0].set(cdf1)
            lambda_ = lambda_.at[1].set(n2 * jnp.exp(-0.5 * n1 * n1) * c[2])
            lambda_ = lambda_.at[2].set(c[2] * lambda_[0] + lambda_[1] * (K - c[1]))
            lambda_ = lambda_.at[3].set(lambda_[1] * (((K - c[1])**2) + 2 * c[2]))
            lambda_ = lambda_.at[4].set(3 * lambda_[0] * c[2] * c[2] + lambda_[1] * (((K - c[1])**3) + 3 * c[2] * (K - c[1])))
        
            lambda5 = n2 * jnp.exp(-0.5 * n1 * n1) * (c[2] * jnp.power(K-c[1], 4) + 4*c[2]*c[2]*((K-c[1])**2) + 8*(c[2]**3))
            lambda6 = 0
            lambda7 = 0
            
            lambda_ = lambda_.at[5].set(lambda5)
            lambda_ = lambda_.at[6].set(lambda6)
            lambda_ = lambda_.at[7].set(lambda7)

        return lambda_
    
    def collin_dufresne_gamma(self, c, order=3):
        gamma = jnp.zeros(8)
        temp1 = c[6] / factorial(6) + 0.5 * jnp.power(c[3], 2) / jnp.power(factorial(3), 2)
        temp2 = c[7] / factorial(7) + c[3] * c[4] / (factorial(3) * factorial(4))
        c2p2 = jnp.power(c[2], 2)
        c2p3 = jnp.power(c[2], 3)
        c2p4 = jnp.power(c[2], 4)
        c2p5 = jnp.power(c[2], 5)
        c2p6 = jnp.power(c[2], 6)
        c2p7 = jnp.power(c[2], 7)

        gamma = gamma.at[0].set(1)
        gamma = gamma.at[1].set(-(3.0 / c2p2) * (c[3] / factorial(3)))
        gamma = gamma.at[2].set(0)
        gamma = gamma.at[3].set((1.0 / c2p3) * (c[3] / factorial(3)))
        
        if order == 4:
            gamma = gamma.at[4].set((c[4]/(c2p4*factorial(4))))
            
        return gamma

    def get_moments(self, i, k, j):##is not returning order 0 moment Edem
        # return E[x11^i x12^k x22^j]
        if(i+k+j<=0):
             return 1.0

        v = jnp.zeros((1))
        if(i+k+j<=self.poly_prop['order'].max()):
            pos_i = int(self.poly_prop[(self.poly_prop['i'] == i) & (self.poly_prop['k'] == k) & (self.poly_prop['j'] == j)]['pos'].iloc[0])
            v = self.moments[pos_i, 0]
        else:
            print(f'i={i}, k={k}, j={j}, polynomial order not available')

        if v.ndim == 0:
            return float(v)
        else:
            return float(v[0])
        # return float(v[0])
