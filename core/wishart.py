"""
WishartBru implementation - standard Wishart process without jumps.
"""
import jax
import jax.numpy as jnp
from jax import jit, grad, vmap
from jax.scipy import linalg as jlinalg
from functools import partial
from typing import Tuple, Optional, Union, Dict
import pandas as pd
import numpy as np
import scipy.linalg
import scipy.integrate
from scipy.stats import norm
from math import factorial
from scipy.special import binom, factorial

from .base import BaseWishart
from .derivatives import WishartDerivatives
from .phi_functions import PhiFunctions


from ..config.constants import *
from ..utils.local_functions import vech, vec, tr_uv


# class WishartBru(BaseWishart):
class WishartBru(BaseWishart, WishartDerivatives, PhiFunctions):

    """
    Wishart process implementation following Bru specification.
    
    This class implements a standard Wishart process without jump components,
    with support for various moment calculations and derivative computations.
    """
    
    def __init__(self, n: int, x0: jnp.ndarray, omega: jnp.ndarray,
                 m: jnp.ndarray, sigma: jnp.ndarray, is_bru_config: bool = False):
        """
        Initialize Wishart process.
        
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
        is_bru_config : bool, optional
            Whether to use Bru configuration, by default False
        """

        self.is_bru_config =True# is_bru_config
        super().__init__(n, x0, omega, m, sigma)

        super().set_wishart_parameter(n, x0, omega, m, sigma)
        # self.is_bru_config = is_bru_config

    def set_wishart_parameter(self, n: int, x0: jnp.ndarray, omega: jnp.ndarray, m: jnp.ndarray, sigma: jnp.ndarray):
        super().set_wishart_parameter(n, x0, omega, m, sigma)
        # Bru configuration setup
        is_bru_config = self.is_bru_config
        if is_bru_config:
            beta_init = self.omega[0, 0] / self.sigma2[0, 0]
            initial_is_bru_config = True
            
            for i in range(self.n):
                for j in range(self.n):
                    beta = self.omega[i, j] / self.sigma2[i, j]
                    initial_is_bru_config = initial_is_bru_config & (beta == beta_init)
                    
            if initial_is_bru_config:
                self.beta = beta_init
                self.is_bru_config = True
            else:
                self.beta = 0.0
                self.is_bru_config = False
        else:
            self.beta = 0.0
            self.is_bru_config = False

        ##already in base class
        # self.G = 0
        # self.exp_gt = 0
        # self.h = 0
        # self.g0 = 0
        # self.h1 = 0 # (G^-1)*(exp(Gt)-I)*h
        # self.poly_prop = 0
        # self.block_struct = 0
        # self.pos = 0 # G[pos[i][j]] gives the G_{i,j} block matrix

    
    @partial(jit, static_argnums=(0,))
    def mgf(self, t: float, theta1: jnp.ndarray, theta2: jnp.ndarray) -> complex:
        """
        Compute moment generating function.
        
        Parameters
        ----------
        t : float
            Time parameter
        theta1 : jnp.ndarray
            First parameter matrix
        theta2 : jnp.ndarray
            Second parameter matrix (unused in standard implementation)
            
        Returns
        -------
        complex
            MGF value
        """
        return self.phi_one(1, theta1)
    
    @partial(jit, static_argnums=(0,))
    def phi_y(self, u: complex, b3: float, a3: jnp.ndarray) -> complex:
        """
        Compute characteristic function for Y.
        
        Parameters
        ----------
        u : complex
            Complex frequency parameter
        b3 : float
            Scalar parameter
        a3 : jnp.ndarray
            Matrix parameter
            
        Returns
        -------
        complex
            Characteristic function value
        """
        c1 = jnp.exp(1j * u * b3)
        theta1 = jnp.multiply(1j * u, a3)
        theta2 = self.zero_n
        c2 = self.mgf(self.maturity, theta1, theta2)
        return c1 * c2
    
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
        eAt = jlinalg.expm(t * self.A)
        vec_sigma2 = self._vec(self.sigma2)
        
        v2 = (eAt - jnp.eye(self.n * self.n)) @ vec_sigma2 ##next line is faster less operation
        # v2 =  eAt @ vec_sigma2 -vec_sigma2

        v3 = jnp.linalg.solve(self.A, v2)
        return eAt, v3
    
    @partial(jit, static_argnums=(0,))
    def compute_a(self, t: float, theta1: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the A matrix function.
        
        Parameters
        ----------
        t : float
            Time parameter
        theta1 : jnp.ndarray
            Parameter matrix
            
        Returns
        -------
        jnp.ndarray
            A matrix
        """
        emt = jlinalg.expm(t * self.m)
        eAt, var_sigma_init = self.compute_var_sigma(t)
        var_sigma = self._vec_inv(var_sigma_init, (self.n, self.n))
        
        # m2 = jnp.linalg.inv(jnp.eye(self.n) - 2 * (theta1 @ var_sigma))
        # emt_tr = jnp.transpose(emt)
        
        # a = emt_tr @ m2 @ theta1 @ emt

        # Replace inv(I - 2*theta1@var_sigma) with solve
        # Instead of: m2 = inv(B) then m2 @ theta1
        # Do: solve(B, theta1) directly
        B = jnp.eye(self.n) - 2 * (theta1 @ var_sigma)
        m2_theta1 = jnp.linalg.solve(B, theta1)
    
        # Optimize matrix multiplication chain: emt.T @ m2_theta1 @ emt
        # Compute from right to left for better efficiency
        temp = m2_theta1 @ emt
        a = emt.T @ temp
        return a
    
    @partial(jit, static_argnums=(0,))
    def compute_b_bru(self, t: float, theta1: jnp.ndarray) -> complex:
        """
        Compute B function for Bru configuration.
        
        Parameters
        ----------
        t : float
            Time parameter
        theta1 : jnp.ndarray
            Parameter matrix
            
        Returns
        -------
        complex
            B value
        """
        eAt, var_sigma_init = self.compute_var_sigma(t)
        var_sigma = self._vec_inv(var_sigma_init, (self.n, self.n))
        
        m2 = jnp.eye(self.n) - 2 * (var_sigma @ theta1)
        res = -(self.beta / 2.0) * jnp.log(jnp.linalg.det(m2))
        
        return res
    
    @partial(jit, static_argnums=(0,))
    def compute_b_old(self, t: float, theta1: jnp.ndarray) -> complex:
        """
        Compute the B function using integration.
        
        Parameters
        ----------
        t : float
            Time parameter
        theta1 : jnp.ndarray
            Parameter matrix
            
        Returns
        -------
        complex
            B value
        """
        if self.is_bru_config:
            res = self.compute_b_bru(t, theta1)
            return res
        
        # Integration approach for non-Bru configuration
        def f_complex(t1):
            a = self.compute_a(t1, theta1)
            return jnp.trace(a @ self.omega)
        
        # Integration points
        n_points = DEFAULT_INTEGRATION_POINTS
        t_vals = jnp.linspace(0, t, n_points)
        
        # Vectorized evaluation
        integrand_vals = jax.vmap(f_complex)(t_vals)
        
        # Trapezoidal integration
        res = jnp.trapezoid(integrand_vals, t_vals)
        
        return res
    
    @partial(jit, static_argnums=(0,))
    def compute_b_a_bit_faster(self, t: float, theta1: jnp.ndarray) -> complex:
        """
        Compute the B function using integration.
    
        OPTIMIZED: Reduced redundant exponential computations
        """
        if self.is_bru_config:
            return self.compute_b_bru(t, theta1)
    
        # Optimized integration: compute all matrix exponentials at once
        n_points = DEFAULT_INTEGRATION_POINTS
        t_vals = jnp.linspace(0, t, n_points)
    
        # Vectorized computation of all matrix exponentials for self.m
        # This is more efficient than computing them one by one
        emt_all = jax.vmap(lambda t_i: jlinalg.expm(t_i * self.m))(t_vals)
    
        # Vectorized computation of var_sigma for all time points
        def compute_var_sigma_at_t(t_i):
            _, var_sigma_init = self.compute_var_sigma(t_i)
            return self._vec_inv(var_sigma_init, (self.n, self.n))
    
        var_sigma_all = jax.vmap(compute_var_sigma_at_t)(t_vals)
    
        # Vectorized computation of integrand
        def compute_integrand(emt, var_sigma):
            B = jnp.eye(self.n) - 2 * (theta1 @ var_sigma)
            m2_theta1 = jnp.linalg.solve(B, theta1)
            a = emt.T @ (m2_theta1 @ emt)
            return jnp.trace(a @ self.omega)
    
        integrand_vals = jax.vmap(compute_integrand)(emt_all, var_sigma_all)
    
        # Trapezoidal integration
        res = jnp.trapezoid(integrand_vals, t_vals)
    
        return res

    @partial(jit, static_argnums=(0,))
    def compute_b(self, t: float, theta1: jnp.ndarray) -> complex:
        """
        Compute the B function using integration.
    
        OPTIMIZED: Simpson's rule for better accuracy with fewer points
        """
        if self.is_bru_config:
            return self.compute_b_bru(t, theta1)
    
        # Use Simpson's rule (requires odd number of points)
        n_points = DEFAULT_INTEGRATION_POINTS
        if n_points % 2 == 0:
            n_points += 1  # Make odd for Simpson's rule
    
        t_vals = jnp.linspace(0, t, n_points)
    
        # Vectorized computation of integrand
        def f_complex(t1):
            a = self.compute_a(t1, theta1)
            return jnp.trace(a @ self.omega)
    
        integrand_vals = jax.vmap(f_complex)(t_vals)
    
        # Simpson's rule implementation in JAX
        # More accurate than trapezoidal with same number of points
        h = t / (n_points - 1)
        result = integrand_vals[0] + integrand_vals[-1]
        result += 4.0 * jnp.sum(integrand_vals[1:-1:2])  # Odd indices
        result += 2.0 * jnp.sum(integrand_vals[2:-1:2])  # Even indices
        result *= h / 3.0
    
        return result

    @partial(jit, static_argnums=(0,))
    def phi_one(self, z: complex, theta1: jnp.ndarray) -> complex:
        """
        Compute the first characteristic function.
        
        Parameters
        ----------
        z : complex
            Complex parameter
        theta1 : jnp.ndarray
            Parameter matrix
            
        Returns
        -------
        complex
            Phi_1 value
        """
        theta1 = jnp.multiply(z, theta1)
        
        t = self.maturity
        a = self.compute_a(t, theta1)
        b1 = self.compute_b(t, theta1)
        # print("--- b1=",b1)
        # jax.debug.print("--- b1 = {x}", x=b1)
        a2 = jnp.trace(self.x0 @ a)
        
        phi_one = jnp.exp(a2 + b1)
        
        return phi_one
    
    def build_g_DONOTUSE(self, order: int = 3) -> None:
        """
        Build the matrix G, g(0) and h for moment calculations.
        
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
        
        dfg1 = df.groupby(by='order')
        lpoly = dfg1.count()['i']
        lpoly.name = 'nbPoly'
        
        G1 = jnp.zeros((df.shape[0], df.shape[0]))
        self.h = jnp.zeros((df.shape[0], 1))
        
        # Fill G matrix
        for l in range(0, df.shape[0]):
            df1 = df.iloc[l]
            i = df1['i']
            k = df1['k']
            j = df1['j']
            pos_i = int(df1['pos'])
            
            # Diagonal element
            G1 = G1.at[pos_i, pos_i].set(
                i * 2 * self.m[0, 0] + k * (self.m[0, 0] + self.m[1, 1]) + 
                2 * j * self.m[1, 1]
            )
            
            # Off-diagonal elements (code continues as in original)
            # ... (rest of the matrix filling logic)
        
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
            v = jnp.power(x01[0, 0], i) * jnp.power(x01[1, 0], k) * jnp.power(x01[2, 0], j)
            self.g0 = self.g0.at[df1['pos'], 0].set(v)
    
    def build_g(self, order: int = 3) -> None:
        """
        Build the matrix G, g(0) and h for moment calculations.
        
        Parameters
        ----------
        order : int, optional
            Maximum polynomial order, by default 3
        """
        # Build polynomial indices
        # Build the matrix G, g(0) and h
        # order = 4#3
        l1 = []
        pos = 0
        for l in range(1, order + 1):
            for i1 in range(0, l + 1):
                for k1 in range(0, l - i1 + 1):
                    j1 = l - i1 - k1
                    #print(l,i1,k1,j1)
                    l1.append([l,i1,k1,j1,pos])
                    pos += 1

        #print(l1)
        df = pd.DataFrame(data = l1, columns=['order','i','k','j','pos'])
        #print(df.head())
        #print(df)

        dfg1 = df.groupby(by='order')
        lpoly = dfg1.count()['i'] # [['i']] to get a DataFrame
        #lpoly.columns =['nbPoly'] # to get a DataFrame
        lpoly.name = 'nbPoly'
        #print(lpoly)

        G1 = jnp.zeros((df.shape[0], df.shape[0]))
        self.h = jnp.zeros((df.shape[0], 1))

        # filling G
        for l in range(0, df.shape[0]):
            df1 = df.iloc[l]
            i = df1['i']
            k = df1['k']
            j = df1['j']
            pos_i = int(df1['pos'])
            G1 = G1.at[pos_i, pos_i].set(i * 2 * self.m[0,0] + k * (self.m[0,0] + self.m[1,1]) + 2 * j * self.m[1,1])

            if ((i > 0) & (k < order)): ## why do we need to check if k< order
                pos_j = int(df[(df['i'] == (i - 1)) & (df['k'] == (k + 1)) & (df['j'] == j)]['pos'].iloc[0])
                G1 = G1.at[pos_i, pos_j].set(i * 2 * self.m[0,1])
            
            if ((i < order) & (k > 0)): ## why do we need to check if i< order
                pos_j = int(df[(df['i'] == (i + 1)) & (df['k'] == (k - 1)) & (df['j'] == j)]['pos'].iloc[0])
                G1 = G1.at[pos_i, pos_j].set(k * self.m[1,0])

            if ((k > 0) & (j < order)):
                pos_j = int(df[(df['i'] == i) & (df['k'] == (k - 1)) & (df['j'] == (j + 1))]['pos'].iloc[0])
                G1 = G1.at[pos_i, pos_j].set(k * self.m[0,1])

            if ((k < order) & (j > 0)):
                pos_j = int(df[(df['i'] == i) & (df['k'] == (k + 1)) & (df['j'] == (j - 1))]['pos'].iloc[0])
                G1 = G1.at[pos_i, pos_j].set(j * 2 * self.m[1,0])

            if(df1['order'] > 1):
                if (i > 0):
                    pos_j = int(df[(df['i'] == (i - 1)) & (df['k'] == k) & (df['j'] == j)]['pos'].iloc[0])
                    G1 = G1.at[pos_i, pos_j].set(i * self.omega[0,0] + 2 * i * k * self.sigma2[0,0] + 2 * i * (i - 1) * self.sigma2[0,0])
                    
                if (k > 0):
                    pos_j = int(df[(df['i'] == i) & (df['k'] == (k - 1)) & (df['j'] == j)]['pos'].iloc[0])
                    G1 = G1.at[pos_i, pos_j].set(k * self.omega[0,1] + k * (k - 1) * self.sigma2[0,1] + 2 * (i * k + j * k) * self.sigma2[0,1])
            
                if (j > 0):
                    pos_j = int(df[(df['i'] == i) & (df['k'] == k) & (df['j'] == (j - 1))]['pos'].iloc[0])
                    G1 = G1.at[pos_i, pos_j].set(j * self.omega[1,1] + 2 * j * (j - 1) * self.sigma2[1,1] + 2 * j * k * self.sigma2[1,1])

                if ((i < order) & (k > 1)):
                    pos_j = int(df[(df['i'] == (i + 1)) & (df['k'] == (k - 2)) & (df['j'] == j)]['pos'].iloc[0])
                    G1 = G1.at[pos_i, pos_j].set((k * (k - 1) / 2) * self.sigma2[1,1])
            
                if ((k > 1) & (j < order)):
                    pos_j = int(df[(df['i'] == i) & (df['k'] == (k - 2)) & (df['j'] == (j + 1))]['pos'].iloc[0])
                    G1 = G1.at[pos_i, pos_j].set((k * (k - 1) / 2) * self.sigma2[0,0])

                if ((i > 0) & (k < order) & (j > 0)):
                    pos_j = int(df[(df['i'] == (i - 1)) & (df['k'] == (k + 1)) & (df['j'] == (j - 1))]['pos'].iloc[0])
                    G1 = G1.at[pos_i, pos_j].set(4 * i * j * self.sigma2[0,1])
            else:
                # For the moment of order 1 there is a constant term in the ODE
                # that is given by omega.  It goes to h.
                if (i > 0):
                    self.h = self.h.at[pos_i, 0].set(i * self.omega[0,0])
                    
                if (k > 0):
                    self.h = self.h.at[pos_i, 0].set(k * self.omega[0,1])
            
                if (j > 0):
                    self.h = self.h.at[pos_i, 0].set(j * self.omega[1,1])

        self.G = G1
        # self.polyProp = df
        self.poly_prop = df
        self.block_struct = lpoly
        # self.BlockStruct = lpoly # For each order gives the number of polynomes of that order.  The order is
                                 # the index of the pd.Series

        # Build dataframe to find the sub matrix blocks in G
        index1 = self.block_struct.cumsum()
        index1.loc[0] = 0
        index1.sort_index(inplace = True)
        
        l = []
        for i in range(0, index1.shape[0] - 1):
            l.append(range(index1.loc[i], index1.loc[i + 1]))
        
        l1 = []
        for i in range(0, index1.shape[0] - 1):
            l2 = []
            for j in range(0, index1.shape[0] - 1):
                #print(l[i],l[j])
                v = tuple(jnp.meshgrid(jnp.array(l[i]), jnp.array(l[j]), indexing='ij'))
                l2.append(v)
            l1.append(l2)

        self.pos = l1  # G[pos[i][j]] gives the G_{i,j} block matrix
       
        # build g(0)
        self.g0 = jnp.zeros((self.G.shape[0], 1))
        x01 = vech(self.x0)
        for l in range(0, df.shape[0]):
            df1 = df.iloc[l]
            i = df1['i']
            k = df1['k']
            j = df1['j']
            v = jnp.power(x01[0,0], i) * jnp.power(x01[1,0], k) * jnp.power(x01[2,0], j)
            self.g0 = self.g0.at[df1['pos'], 0].set(v)
      

    def compute_moments(self, t: float, order: int = 3) -> None:
        """
        Compute the moments of the process.
        
        Parameters
        ----------
        t : float
            Time parameter
        order : int, optional
            Maximum moment order, by default 3
        """
        self.build_g(order)
        self.compute_exp_gt(t)
        self.compute_h1()
        v1 = jnp.add(jnp.matmul(self.exp_gt, self.g0), self.h1)
        self.moments = v1
    
    def get_moments(self, i: int, k: int, j: int) -> float:
        """
        Return E[x11^i x12^k x22^j].
        
        Parameters
        ----------
        i : int
            Power of x11
        k : int
            Power of x12
        j : int
            Power of x22
            
        Returns
        -------
        float
            Moment value
        """
        if i + k + j <= 0:
            return 1.0
        
        v = jnp.zeros((1))
        if i + k + j <= self.poly_prop['order'].max():
            pos_i = int(self.poly_prop[
                (self.poly_prop['i'] == i) & 
                (self.poly_prop['k'] == k) & 
                (self.poly_prop['j'] == j)
            ]['pos'].iloc[0])
            v = self.moments[pos_i, 0]
        else:
            print(f'i={i}, k={k}, j={j}, polynomial order not available')
        
        if v.ndim == 0:
            return float(v)
        else:
            return float(v[0])
    
    def compute_exp_gt(self, t: float) -> None:
        """
        Compute matrix exponential e^{Gt}.
        
        Parameters
        ----------
        t : float
            Time parameter
        """
        egt = jnp.zeros(self.G.shape)
        gt = jnp.multiply(t, self.G)
        
        # Exponentiate the diagonal blocks
        for i in range(0, len(self.pos)):
            egt = egt.at[self.pos[i][i]].set(jlinalg.expm(gt[self.pos[i][i]]))
        
        # Exponentiate off diagonal blocks
        for i in range(1, len(self.pos)):
            for j in range(i - 1, -1, -1):
                f1 = jnp.zeros((self.block_struct.iloc[i], self.block_struct.iloc[j]))
                for p in range(0, i - j):
                    f1 = jnp.add(f1, jnp.subtract(
                        jnp.matmul(gt[self.pos[i][j + p]], egt[self.pos[j + p][j]]),
                        jnp.matmul(egt[self.pos[i][i - p]], gt[self.pos[i - p][j]])
                    ))
                
                # Use scipy for sylvester equation
                f2 = jnp.array(scipy.linalg.solve_sylvester(
                    np.array(-gt[self.pos[i][i]]),
                    np.array(gt[self.pos[j][j]]),
                    np.array(f1)
                ))
                egt = egt.at[self.pos[i][j]].set(f2)
        
        self.exp_gt = egt
    
    def compute_h1(self) -> None:
        """
        Compute G^{-1}*(e^{G t}-I)*h.
        
        This function handles the block structure of the matrix
        and computes h1 iteratively.
        """
        h0 = jnp.matmul(
            jnp.subtract(self.exp_gt, jnp.identity(self.exp_gt.shape[0])), 
            self.h
        )
        
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
            
            # Update h1 - preserve the shape
            if h1_np.ndim == 2:
                h1_np[start_i:end_i] = h3.reshape(-1, h1_np.shape[1])
            else:
                h1_np[start_i:end_i] = h3
        
        # Convert back to JAX array
        self.h1 = jnp.array(h1_np)
    
    # Additional methods that complete the WishartBru implementation
    
    def compute_exp_gt(self, t: float) -> None:
        """
        Compute matrix exponential e^{Gt}.
        
        Parameters
        ----------
        t : float
            Time parameter
        """
        egt = jnp.zeros(self.G.shape)
        gt = jnp.multiply(t, self.G)
        
        # Exponentiate the diagonal blocks
        for i in range(0, len(self.pos)):
            egt = egt.at[self.pos[i][i]].set(jlinalg.expm(gt[self.pos[i][i]]))
        
        # Exponentiate off diagonal blocks
        for i in range(1, len(self.pos)):
            for j in range(i - 1, -1, -1):
                f1 = jnp.zeros((self.block_struct.iloc[i], self.block_struct.iloc[j]))
                for p in range(0, i - j):
                    f1 = jnp.add(f1, jnp.subtract(
                        jnp.matmul(gt[self.pos[i][j + p]], egt[self.pos[j + p][j]]),
                        jnp.matmul(egt[self.pos[i][i - p]], gt[self.pos[i - p][j]])
                    ))
                
                # Use scipy for sylvester equation
                f2 = jnp.array(scipy.linalg.solve_sylvester(
                    np.array(-gt[self.pos[i][i]]),
                    np.array(gt[self.pos[j][j]]),
                    np.array(f1)
                ))
                egt = egt.at[self.pos[i][j]].set(f2)
        
        self.exp_gt = egt
    
    def partial_sigma_ij_b(self, i: int, j: int, t: float, u_mat: jnp.ndarray) -> float:
        """
        Compute partial derivative of b with respect to sigma[i,j].
        
        Parameters
        ----------
        i : int
            Row index
        j : int
            Column index
        t : float
            Time parameter
        u_mat : jnp.ndarray
            Weight matrix
            
        Returns
        -------
        float
            Partial derivative value
        """
        eij = self._eij_simple(i, j, self.x0.shape)
        b_new = self._vech(self.beta * eij)
        
        eAt = jlinalg.expm(jnp.multiply(self.A, t))
        v2 = jnp.matmul(jnp.subtract(eAt, jnp.eye(self.n * self.n)), b_new)
        v3 = jnp.linalg.solve(self.A, v2)
        vec_u_mat = self._vec(u_mat)
        res = jnp.transpose(vec_u_mat) @ v3
        return res


    def compute_mu(self, b3:float, a3: jnp.ndarray, order: int =3):
        all_moments_Y = jnp.zeros(order+1)
        all_moments_Y = all_moments_Y.at[0].set(1.0)
        
        B = b3
        A1 = a3[0,0]
        A2 = a3[1,1]
        A12 = a3[0,1]
        
        if A12 == 0:
            for n1 in range(1, order+1):
                moment_Y = 0
                for i1 in range(0, n1+1):
                    for k1 in range(0, n1-i1+1):
                        j1 = n1-i1-k1
                        multinomialCoef = (factorial(n1)) / (factorial(i1) * factorial(k1) * factorial(j1))
                        l1_l2 = i1 + j1
                        currentMoment = self.get_moments(i1, 0, j1)
                        moment_Y += multinomialCoef * (B**k1) * (A1**i1) * (A2**j1) * currentMoment
            
                all_moments_Y = all_moments_Y.at[n1].set(moment_Y)
        else: # in progress for the general case
            for n1 in range(1, order+1):
                moment_Y = 0
                for i1 in range(0, n1+1):
                    for k1 in range(0, n1-i1+1):
                        for l1 in range(0, n1-i1-k1+1):
                            j1 = n1-i1-k1-l1
                            multinomialCoef = (factorial(n1)) / (factorial(i1) * factorial(k1) * factorial(j1))
                            
                            currentMoment = self.get_moments(i1, l1, j1)
                            moment_Y += multinomialCoef * (B**k1) * (A1**i1) * (A12**l1) * (A2**j1) * currentMoment
            
                all_moments_Y = all_moments_Y.at[n1].set(moment_Y)
        
        return all_moments_Y
    
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

    def collin_dufresne_c(self, mu:float,  order: int =3):
        c = jnp.zeros(8)
        c = c.at[0].set(mu[0])
        c = c.at[1].set(mu[1])
        c = c.at[2].set(mu[2] - mu[1] * mu[1])
        
        if c[2] < 0:
            print(f"{c[2]} is negative and mu is {mu}")
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
    
    def collin_dufresne_lambda(self, c, strike_offset:float=0, order:int=3):
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
    
    def collin_dufresne_gamma(self, c, order:int=3):
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
