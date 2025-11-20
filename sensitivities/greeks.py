# sensitivities/greeks.py
"""Greeks calculations for LRW model."""

import math
import cmath
from typing import Dict, Tuple, Any
from functools import partial

# from BlackScholes import f
# from WishartBru import NMAX
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jspl
from jax import jit
import scipy.integrate as sp_i
import numpy as np

from ..utils.local_functions import vec, vec_inv, tr_uv, eij, eij_simple
from ..config.constants import  NMAX
from ..pricing.swaption.fourier_pricing import FourierPricer

class GreeksCalculator:
    """Calculate Greeks for LRW interest rate model."""
    
    def __init__(self, model):
        """Initialize Greeks calculator with LRW model."""
        self.model = model
        
    def price_option_sensi_alpha(self, ur: float = 0.5, nmax: int = NMAX, 
                                recompute_a3_b3: bool = False) -> Tuple[float, Dict]:
        """Calculate alpha sensitivity (derivative with respect to alpha)."""
        if recompute_a3_b3:
            self.model.compute_b3_a3()
            
        def f1(ui):
            u = complex(ur, ui)
            z = u
            
            z_a3 = z * self.model.a3
            exp_z_b3 = cmath.exp(z * self.model.b3)
            phi1 = self.model.wishart.phi_one(1, z_a3)
            
            a3_derive_alpha = self._compute_a_derive_alpha(self.model.maturity, z)
            b3_derive_alpha = self._compute_b_derive_alpha(self.model.maturity, z)
            a3_derive_alpha_trace = jnp.trace(a3_derive_alpha @ self.model.x0)
            
            temp1 = exp_z_b3 * (a3_derive_alpha_trace + b3_derive_alpha)
            temp2 = z * self.model.b3_prime_alpha * exp_z_b3
            
            res = ((temp1 + temp2) * phi1) / (z * z)
            return res.real
            
        f2 = lambda x: f1(x)
        r1 = sp_i.quad(f2, 0, nmax)
        res1 = r1[0] / math.pi
        sensi_alpha1 = res1 * math.exp(-self.model.alpha * self.model.maturity) / (1 + tr_uv(self.model.u1, self.model.x0))
        
        fourier_pricer= FourierPricer(self.model)
        option_price = fourier_pricer.price(ur)#self.model.price_option()
        sensi_alpha2 = -self.model.maturity * option_price
        
        sensi_alpha = sensi_alpha1 + sensi_alpha2
        
        alpha_sensi_report = {}
        alpha_sensi_report[f"ALPHASENSI:{self.model.strike}:NA_NA:ALPHASENSIVALUE:NA"] = sensi_alpha
        
        return sensi_alpha, alpha_sensi_report
        
    def price_option_sensi_omega(self, ur: float = 0.5, nmax: int = NMAX) -> Tuple[jnp.ndarray, Dict]:
        """Calculate omega sensitivity (Vomma)."""
        self.model.compute_b3_a3()
        self._compute_b3_derive_omega()
        
        sensi_omega = jnp.zeros((self.model.n, self.model.n))
        omega_sensi_report = {}
        
        for i in range(self.model.n):
            for j in range(self.model.n):
                
                def f1(ui):
                    u = complex(ur, ui)
                    z = u
                    
                    z_a3 = z * self.model.a3
                    exp_z_b3 = cmath.exp(z * self.model.b3)
                    phi1 = self.model.wishart.phi_one(1, z_a3)
                    
                    b3_omega_ij = self.model.b3_derive_omega[i, j]
                    b_derive_omega_ij = self.model.wishart.compute_b_prime_omega(self.model.maturity, i, j, z_a3)
                    
                    temp1 = z * b3_omega_ij * exp_z_b3 * phi1
                    temp2 = b_derive_omega_ij * exp_z_b3 * phi1
                    
                    temp = temp1 + temp2
                    res = temp / (z * z)
                    return res.real
                    
                f2 = lambda x: f1(x)
                r1 = sp_i.quad(f2, 0, nmax)
                res1 = r1[0] / math.pi
                sensi_omega_ij = res1 * math.exp(-self.model.alpha * self.model.maturity) / (1 + tr_uv(self.model.u1, self.model.x0))
                sensi_omega = sensi_omega.at[i, j].set(sensi_omega_ij)
                
                omega_sensi_report[f"OMEGASENSI:{self.model.strike}:{i}_{j}:OMEGASENSIVALUE:ALL:NA"] = sensi_omega[i, j]
                
        return sensi_omega, omega_sensi_report
        
    def price_option_sensi_m(self, ur: float = 0.5, nmax: int = NMAX) -> Tuple[jnp.ndarray, Dict]:
        """Calculate M sensitivity."""
        self.model.compute_b3_a3()
        self._compute_a3_b3_derive_m()
        
        sensi_m = jnp.zeros((self.model.n, self.model.n))
        m_sensi_report = {}
        
        for i in range(self.model.n):
            for j in range(self.model.n):
                
                def f1(ui):
                    u = complex(ur, ui)
                    z = u
                    
                    z_a3 = z * self.model.a3
                    exp_z_b3 = cmath.exp(z * self.model.b3)
                    phi1 = self.model.wishart.phi_one(1, z_a3)
                    
                    b3_m_ij = self.model.b3_derive_m[i, j]
                    a3_m_ij = self.model.a3_derive_m[i*self.model.n:(i+1)*self.model.n, j*self.model.n:(j+1)*self.model.n]
                    
                    b_derive_m_ij = self.model.wishart.compute_b_derive_m(self.model.maturity, i, j, z, self.model.a3, self.model.a3_derive_m)
                    a_derive_m_ij = self.model.wishart.compute_a_derive_m(self.model.maturity, i, j, z, self.model.a3, self.model.a3_derive_m)
                    
                    temp1 = z * b3_m_ij * exp_z_b3 * phi1
                    temp2 = (jnp.trace(a_derive_m_ij @ self.model.x0) + b_derive_m_ij) * exp_z_b3 * phi1
                    
                    temp = temp1 + temp2
                    res = temp / (z * z)
                    return res.real
                    
                f2 = lambda x: f1(x)
                r1 = sp_i.quad(f2, 0, nmax)
                res1 = r1[0] / math.pi
                sensi_m_ij = res1 * math.exp(-self.model.alpha * self.model.maturity) / (1 + tr_uv(self.model.u1, self.model.x0))
                sensi_m = sensi_m.at[i, j].set(sensi_m_ij)
                
                m_sensi_report[f"MSENSI:{self.model.strike}:{i}_{j}:MSENSIVALUE:NA"] = sensi_m[i, j]
                
        return sensi_m, m_sensi_report
        
    def price_option_vega(self, ur: float = 0.5, nmax: int = NMAX) -> Tuple[jnp.ndarray, Dict]:
        """Calculate Vega (sigma sensitivity)."""
        self.model.compute_b3_a3()
        
        sensi_sigma = jnp.zeros((self.model.n, self.model.n))
        sigma_sensi_report = {}
        print("In price_option_vega ")
        for i in range(self.model.n):
            for j in range(self.model.n):
                print(f"Calculating Vega for component {i}, {j}")
                def f1(ui):
                    u = complex(ur, ui)
                    z = u
                    
                    z_a3 = z * self.model.a3
                    exp_z_b3 = cmath.exp(z * self.model.b3)
                    phi1 = self.model.wishart.phi_one_vega(i, j, 1, z_a3)
                    
                    res = (exp_z_b3 * phi1) / (z * z)
                    return res.real
                    
                f2 = lambda x: f1(x)
                r1 = sp_i.quad(f2, 0, nmax)
                res1 = r1[0] / math.pi
                sensi_sigma_ij = res1 * math.exp(-self.model.alpha * self.model.maturity) / (1 + tr_uv(self.model.u1, self.model.x0))
                sensi_sigma = sensi_sigma.at[i, j].set(sensi_sigma_ij)
                
                sigma_sensi_report[f"VEGASENSI:{self.model.strike}:{i}_{j}:VEGAVALUE:ALL:NA"] = sensi_sigma_ij
                
        return sensi_sigma, sigma_sensi_report
        
    def compute_gamma_bond(self, bond_gamma_id0: int, bond_gamma_id1: int, 
                          ur: float = 0.5, nmax: int = 1000) -> Tuple[float, Dict]:
        """Calculate Gamma for bond positions."""
        self.model.compute_b3_a3()
        
        # Prepare all eta, T, mu, and theta values
        all_eta = []
        all_t = []
        all_mu = []
        all_theta = []
        
        # T = T_{n_1} = maturity
        all_t.append(self.model.maturity)
        all_eta.append(1.0)
        b1_bar, a1 = self.model.compute_bar_b1_a1(0)
        delta_t = 0
        all_mu.append(math.exp(-self.model.alpha * delta_t) * b1_bar)
        all_theta.append(math.exp(-self.model.alpha * delta_t) * a1)
        
        # T = T_{n_2} = maturity + tenor
        all_t.append(self.model.maturity + self.model.tenor)
        all_eta.append(-1.0 - self.model.delta_fixed * self.model.strike)
        b1_bar, a1 = self.model.compute_bar_b1_a1(self.model.tenor)
        delta_t = self.model.tenor
        all_mu.append((-1.0 - self.model.delta_fixed * self.model.strike) * math.exp(-self.model.alpha * delta_t) * b1_bar)
        all_theta.append((-1.0 - self.model.delta_fixed * self.model.strike) * math.exp(-self.model.alpha * self.model.tenor) * a1)
        
        # Fixed leg
        for i in range(1, int(self.model.tenor / self.model.delta_fixed)):
            t1 = self.model.maturity + i * self.model.delta_fixed
            all_t.append(t1)
            all_eta.append(-self.model.delta_fixed * self.model.strike)
            
            delta_t = i * self.model.delta_fixed
            b1_bar, a1 = self.model.compute_bar_b1_a1(delta_t)
            all_mu.append((-self.model.delta_fixed * self.model.strike) * math.exp(-self.model.alpha * delta_t) * b1_bar)
            all_theta.append((-self.model.delta_fixed * self.model.strike) * math.exp(-self.model.alpha * delta_t) * a1)
            
        self.all_eta = all_eta
        self.all_t = all_t
        self.all_mu = all_mu
        self.all_theta = all_theta
        
        mu_sum = sum(self.all_mu)
        theta_sum = sum(self.all_theta)
        
        self.mu_sum = mu_sum
        self.theta_sum = theta_sum
        
        mu0 = self.all_mu[bond_gamma_id0]
        theta0 = self.all_theta[bond_gamma_id0]
        
        mu1 = self.all_mu[bond_gamma_id1]
        theta1 = self.all_theta[bond_gamma_id1]
        
        eta0 = self.all_eta[bond_gamma_id0]
        eta1 = self.all_eta[bond_gamma_id1]
        
        p_t0 = self.model.bond(all_t[bond_gamma_id0])
        p_t1 = self.model.bond(all_t[bond_gamma_id1])
        
        divider = p_t0 * p_t1
        
        if bond_gamma_id1 != bond_gamma_id0:
            mu2 = mu_sum - mu0 - mu1
            theta2 = theta_sum - theta0 - theta1
        else:
            mu2 = mu_sum - mu0
            theta2 = theta_sum - theta0
            
        def gamma_integ(ui):
            u = complex(ur, ui)
            z = u
            
            # I1 term
            z_a3 = u * self.model.a3
            phi1 = self.model.wishart.phi_one(1, z_a3)
            exp_z_b3 = cmath.exp(u * self.model.b3)
            res_i1 = mu0 * mu1 * exp_z_b3 * phi1 / (u * u)
            
            # I2 term
            a_j2_gamma = self.model.wishart.compute_a_j2_gamma_derive(self.model.maturity, z, self.model.a3, theta1)
            j2 = jnp.trace(a_j2_gamma @ self.model.x0)
            b_j2_gamma = self.model.wishart.compute_b_j2_gamma_derive(self.model.maturity, z, self.model.a3, theta1)
            j2 = j2 + b_j2_gamma
            j2 = j2 * phi1
            res_i2 = u * mu0 * exp_z_b3 * j2 / (u * u)
            
            # I3 term
            a_j2_gamma = self.model.wishart.compute_a_j2_gamma_derive(self.model.maturity, z, self.model.a3, theta0)
            j2 = jnp.trace(a_j2_gamma @ self.model.x0)
            b_j2_gamma = self.model.wishart.compute_b_j2_gamma_derive(self.model.maturity, z, self.model.a3, theta0)
            j2 = j2 + b_j2_gamma
            j2 = j2 * phi1
            res_i3 = u * mu1 * exp_z_b3 * j2 / (u * u)
            
            # I4 term
            a_j4_gamma = self.model.wishart.compute_a_j4_gamma_derive(self.model.maturity, z, self.model.a3, theta0, theta1)
            j4 = jnp.trace(a_j4_gamma @ self.model.x0)
            b_j4_gamma = self.model.wishart.compute_b_j4_gamma_derive(self.model.maturity, z, self.model.a3, theta0, theta1)
            j4 = j4 + b_j4_gamma
            
            a_j2_gamma_1 = self.model.wishart.compute_a_j2_gamma_derive(self.model.maturity, z, self.model.a3, theta0)
            j2_1 = jnp.trace(a_j2_gamma_1 @ self.model.x0)
            b_j2_gamma_1 = self.model.wishart.compute_b_j2_gamma_derive(self.model.maturity, z, self.model.a3, theta0)
            j2_1 = j2_1 + b_j2_gamma_1
            
            a_j2_gamma_2 = self.model.wishart.compute_a_j2_gamma_derive(self.model.maturity, z, self.model.a3, theta1)
            j2_2 = jnp.trace(a_j2_gamma_2 @ self.model.x0)
            b_j2_gamma_2 = self.model.wishart.compute_b_j2_gamma_derive(self.model.maturity, z, self.model.a3, theta1)
            j2_2 = j2_2 + b_j2_gamma_2
            
            j4 = j4 + j2_2 * j2_1
            j4 = j4 * phi1
            res_i4 = exp_z_b3 * j4 / (u * u)
            
            res = res_i1 + res_i2 + res_i3 + res_i4
            return res.real
            
        f_integ = lambda x: gamma_integ(x)
        r1 = sp_i.quad(f_integ, 0, nmax)
        res1 = r1[0] / math.pi
        expectation = res1
        
        eta_tn1 = math.exp(-self.model.alpha * self.model.maturity) / (1 + jnp.trace(self.model.u1 @ self.model.x0))
        res_gamma = expectation * eta_tn1
        res_gamma /= divider
        
        gamma_result = {}
        gamma_result[f"Gamma:{self.model.strike}:{all_t[bond_gamma_id0]}_{all_t[bond_gamma_id1]}:BOND:NA:NA"] = res_gamma
        
        return res_gamma, gamma_result
        
    def compute_gamma_swap_cross(self, first_component: str, second_component: str,
                                ur: float = 0.5, nmax: int = 1000) -> Tuple[float, Dict]:
        """Calculate cross-Gamma between swap components."""
        self.model.compute_b3_a3()
        
        # Floating leg calculations
        b1_bar, a1 = self.model.compute_bar_b1_a1(0)
        floating_leg_mu = b1_bar
        floating_leg_theta = a1
        
        delta_t = self.model.tenor
        b1_bar, a1 = self.model.compute_bar_b1_a1(delta_t)
        floating_leg_mu = floating_leg_mu - math.exp(-self.model.alpha * delta_t) * b1_bar
        floating_leg_theta = floating_leg_theta - math.exp(-self.model.alpha * delta_t) * a1
        
        # Fixed leg calculations
        fix_leg_mu = 0
        fix_leg_theta = jnp.zeros(self.model.x0.shape)
        
        for i in range(1, int(self.model.tenor / self.model.delta_fixed) + 1):
            t1 = self.model.maturity + i * self.model.delta_fixed
            delta_t = i * self.model.delta_fixed
            b1_bar, a1 = self.model.compute_bar_b1_a1(delta_t)
            
            fix_leg_mu += -self.model.delta_fixed * self.model.strike * math.exp(-self.model.alpha * delta_t) * b1_bar
            fix_leg_theta += -self.model.delta_fixed * self.model.strike * math.exp(-self.model.alpha * delta_t) * a1
            
        # Set up mu and theta based on components
        mu0 = 0
        theta0 = jnp.zeros(self.model.x0.shape)
        mu1 = 0
        theta1 = jnp.zeros(self.model.x0.shape)
        
        divider = 1.0
        if (first_component == "FIXED") & (second_component == "FIXED"):
            divider = 0
            for i in range(1, int(self.model.tenor / self.model.delta_fixed) + 1):
                t1 = self.model.maturity + i * self.model.delta_fixed
                divider += self.model.bond(t1)
            divider = divider * divider
            
            mu0 = fix_leg_mu
            mu1 = fix_leg_mu
            theta0 = fix_leg_theta
            theta1 = fix_leg_theta
            
        elif (first_component == "FLOATING") & (second_component == "FLOATING"):
            divider = self.model.bond(self.model.maturity) - self.model.bond(self.model.maturity + self.model.tenor)
            divider = divider * divider
            
            mu1 = floating_leg_mu
            mu0 = floating_leg_mu
            theta1 = floating_leg_theta
            theta0 = floating_leg_theta
            
        elif ((first_component == "FIXED") & (second_component == "FLOATING")) or \
             ((first_component == "FLOATING") & (second_component == "FIXED")):
            divider = self.model.bond(self.model.maturity) - self.model.bond(self.model.maturity + self.model.tenor)
            temp_div = 0
            for i in range(1, int(self.model.tenor / self.model.delta_fixed) + 1):
                t1 = self.model.maturity + i * self.model.delta_fixed
                temp_div += self.model.bond(t1)
            divider *= temp_div
            
            mu1 = fix_leg_mu
            mu0 = floating_leg_mu
            theta1 = fix_leg_theta
            theta0 = floating_leg_theta
            
        def gamma_integ(ui):
            u = complex(ur, ui)
            z = u
            
            # Similar calculation as compute_gamma_bond but with swap components
            z_a3 = u * self.model.a3
            phi1 = self.model.wishart.phi_one(1, z_a3)
            exp_z_b3 = cmath.exp(u * self.model.b3)
            
            # I1 term
            res_i1 = mu0 * mu1 * exp_z_b3 * phi1 / (u * u)
            
            # I2 term
            a_j2_gamma = self.model.wishart.compute_a_j2_gamma_derive(self.model.maturity, z, self.model.a3, theta1)
            j2 = jnp.trace(a_j2_gamma @ self.model.x0)
            b_j2_gamma = self.model.wishart.compute_b_j2_gamma_derive(self.model.maturity, z, self.model.a3, theta1)
            j2 = j2 + b_j2_gamma
            j2 = j2 * phi1
            res_i2 = u * mu0 * exp_z_b3 * j2 / (u * u)
            
            # I3 term
            a_j2_gamma = self.model.wishart.compute_a_j2_gamma_derive(self.model.maturity, z, self.model.a3, theta0)
            j2 = jnp.trace(a_j2_gamma @ self.model.x0)
            b_j2_gamma = self.model.wishart.compute_b_j2_gamma_derive(self.model.maturity, z, self.model.a3, theta0)
            j2 = j2 + b_j2_gamma
            j2 = j2 * phi1
            res_i3 = u * mu1 * exp_z_b3 * j2 / (u * u)
            
            # I4 term
            a_j4_gamma = self.model.wishart.compute_a_j4_gamma_derive(self.model.maturity, z, self.model.a3, theta0, theta1)
            j4 = jnp.trace(a_j4_gamma @ self.model.x0)
            b_j4_gamma = self.model.wishart.compute_b_j4_gamma_derive(self.model.maturity, z, self.model.a3, theta0, theta1)
            j4 = j4 + b_j4_gamma
            
            a_j2_gamma_1 = self.model.wishart.compute_a_j2_gamma_derive(self.model.maturity, z, self.model.a3, theta0)
            j2_1 = jnp.trace(a_j2_gamma_1 @ self.model.x0)
            b_j2_gamma_1 = self.model.wishart.compute_b_j2_gamma_derive(self.model.maturity, z, self.model.a3, theta0)
            j2_1 = j2_1 + b_j2_gamma_1
            
            a_j2_gamma_2 = self.model.wishart.compute_a_j2_gamma_derive(self.model.maturity, z, self.model.a3, theta1)
            j2_2 = jnp.trace(a_j2_gamma_2 @ self.model.x0)
            b_j2_gamma_2 = self.model.wishart.compute_b_j2_gamma_derive(self.model.maturity, z, self.model.a3, theta1)
            j2_2 = j2_2 + b_j2_gamma_2
            
            j4 = j4 + j2_2 * j2_1
            j4 = j4 * phi1
            res_i4 = exp_z_b3 * j4 / (u * u)
            
            res = res_i1 + res_i2 + res_i3 + res_i4
            return res.real
            
        f_integ = lambda x: gamma_integ(x)
        r1 = sp_i.quad(f_integ, 0, nmax)
        res1 = r1[0] / math.pi
        expectation = res1
        
        eta_tn1 = math.exp(-self.model.alpha * self.model.maturity) / (1 + jnp.trace(self.model.u1 @ self.model.x0))
        res_gamma = expectation * eta_tn1 / divider
        
        gamma_result = {}
        gamma_result[f"Gamma:{self.model.strike}:{first_component}_{second_component}:SWAP:NA:NA"] = res_gamma
        
        return res_gamma, gamma_result
        
    # Private helper methods
    def _compute_a_derive_alpha(self, t: float, z: complex) -> jnp.ndarray:
        """Compute derivative of A with respect to alpha."""
        e_mt = jspl.expm(t * self.model.m)
        e_mt_t = jnp.transpose(e_mt)
        
        e_at, var_sigma_init = self.model.wishart.compute_var_sigma(t)
        var_sigma = vec_inv(np.array(var_sigma_init))
        var_sigma = jnp.array(var_sigma)
        
        i_n = jnp.eye(self.model.n)
        
        temp = jnp.linalg.inv(i_n - 2 * z * (self.model.a3 @ var_sigma))
        
        a_derive_alpha = (e_mt_t @ temp) @ ((2 * z * self.model.a3_prime_alpha) @ var_sigma) @ temp @ (z * self.model.a3 @ e_mt)
        a_derive_alpha += e_mt_t @ temp @ (z * self.model.a3_prime_alpha @ e_mt)
        
        return a_derive_alpha
        
    def _compute_b_derive_alpha(self, t: float, z: complex) -> complex:
        """Compute derivative of B with respect to alpha."""
        def f1(s):
            a_derive_alpha = self._compute_a_derive_alpha(s, z)
            z1 = jnp.trace(a_derive_alpha @ self.model.omega)
            return z1.real
            
        f2 = lambda t1: f1(t1)
        r1 = sp_i.quad(f2, 0, t)
        c2r = r1[0]
        
        def f3(s):
            a_derive_alpha = self._compute_a_derive_alpha(s, z)
            z1 = jnp.trace(a_derive_alpha @ self.model.omega)
            return z1.imag
            
        f4 = lambda t1: f3(t1)
        r2 = sp_i.quad(f4, 0, t)
        c2i = r2[0]
        c3 = complex(c2r, c2i)
        
        return c3
        
    def _compute_b3_derive_omega(self):
        """Compute derivative of b3 with respect to omega."""
        b3_derive_omega = jnp.zeros((self.model.n, self.model.n))
        b1_derive_omega = jnp.zeros((self.model.n, self.model.n))
        
        for ii in range(self.model.n):
            for jj in range(self.model.n):
                b1_derive_omega_ij = self._compute_b1_derive_omega(self.model.tenor, ii, jj)
                b1_derive_omega = b1_derive_omega.at[ii, jj].set(b1_derive_omega_ij)
                b3_derive_omega = b3_derive_omega.at[ii, jj].set(
                    -math.exp(-self.model.alpha * self.model.tenor) * b1_derive_omega_ij)
                    
        # Fixed leg
        for i in range(1, int(self.model.tenor / self.model.delta_fixed) + 1):
            t1 = i * self.model.delta_fixed
            
            for ii in range(self.model.n):
                for jj in range(self.model.n):
                    b1_derive_omega_ij = self._compute_b1_derive_omega(t1, ii, jj)
                    b1_derive_omega = b1_derive_omega.at[ii, jj].set(b1_derive_omega_ij)
                    b3_derive_omega = b3_derive_omega.at[ii, jj].add(
                        -self.model.strike * self.model.delta_fixed * math.exp(-self.model.alpha * t1) * b1_derive_omega_ij)
                        
        self.model.b3_derive_omega = b3_derive_omega
        
    def _compute_b1_derive_omega(self, t: float, i: int, j: int) -> float:
        """Compute derivative of b1 with respect to omega element (i,j)."""
        e_at = jspl.expm(t * self.model.A)
        # eAt = jsl.expm(self.A * t)

        a_inv = self.model.wishart.A_inv#jnp.linalg.inv(self.model.A)
        i_n_square = jnp.eye(self.model.n * self.model.n)
        
        mat = a_inv @ (e_at - i_n_square)
        
        e_ij = eij(i, j, self.model.x0.shape[0])
        e_ij_vec = vec(e_ij)
        
        b_ij_temp = mat @ e_ij_vec
        b_ij = jnp.transpose(vec(np.array(self.model.u1))) @ b_ij_temp
        
        return b_ij#[0, 0]
        
    def _compute_a3_b3_derive_m(self):
        """Compute derivatives of a3 and b3 with respect to m."""
        a3_derive_m = jnp.zeros((self.model.n * self.model.n, self.model.n * self.model.n))
        b3_derive_m = jnp.zeros((self.model.n, self.model.n))
        
        for ii in range(self.model.n):
            for jj in range(self.model.n):
                a_0_t, a_0_t_prime, b_t, b_t_prime = self._compute_a1_b1_derive_m(self.model.tenor, ii, jj)
                
                b3_derive_m = b3_derive_m.at[ii, jj].set(-math.exp(-self.model.alpha * self.model.tenor) * b_t_prime)
                
                a3_derive_m = a3_derive_m.at[ii*self.model.n:(ii+1)*self.model.n, jj*self.model.n:(jj+1)*self.model.n].add(
                    -math.exp(-self.model.alpha * self.model.tenor) * a_0_t_prime)
                    
        # Fixed leg
        for i in range(1, int(self.model.tenor / self.model.delta_fixed) + 1):
            t1 = i * self.model.delta_fixed
            
            for ii in range(self.model.n):
                for jj in range(self.model.n):
                    a_0_t, a_0_t_prime, b_t, b_t_prime = self._compute_a1_b1_derive_m(t1, ii, jj)
                    
                    b3_derive_m = b3_derive_m.at[ii, jj].add(
                        -self.model.strike * self.model.delta_fixed * math.exp(-self.model.alpha * t1) * b_t_prime)
                    a3_derive_m = a3_derive_m.at[ii*self.model.n:(ii+1)*self.model.n, jj*self.model.n:(jj+1)*self.model.n].add(
                        -self.model.strike * self.model.delta_fixed * math.exp(-self.model.alpha * t1) * a_0_t_prime)
                        
        self.model.b3_derive_m = b3_derive_m
        self.model.a3_derive_m = a3_derive_m
        
    def _compute_a1_b1_derive_m(self, t: float, i: int, j: int) -> Tuple[jnp.ndarray, jnp.ndarray, float, float]:
        """Compute derivatives of a1 and b1 with respect to m element (i,j)."""
        n_square = self.model.n * self.model.n
        i_n = jnp.eye(self.model.n)
        i_n_square = jnp.eye(2 * n_square)
        zero_square = jnp.zeros((n_square, n_square))
        
        e_ij = eij_simple(i, j, self.model.x0.shape[0])
        e_ij = jnp.array(e_ij)
        a1 = jnp.zeros((2 * n_square, 2 * n_square))
        v_0_0 = jnp.zeros((2 * n_square, 1))
        v_0_t = jnp.zeros((2 * n_square, 1))
        
        b_0_tilde = jnp.zeros((2 * n_square, 1))
        b_0_tilde_prime = jnp.zeros((2 * n_square, 1))
        
        v_0_0_tilde = jnp.zeros((2 * n_square, 1))
        v_0_t_tilde = jnp.zeros((2 * n_square, 1))
        
        m_kron_identity = jnp.kron(i_n, jnp.transpose(self.model.m)) + jnp.kron(jnp.transpose(self.model.m), i_n)
        id_kron_eij = jnp.kron(i_n, jnp.transpose(e_ij)) + jnp.kron(jnp.transpose(e_ij), i_n)
        
        a1 = a1.at[0:n_square, 0:n_square].set(m_kron_identity)
        a1 = a1.at[0:n_square, n_square:2*n_square].set(zero_square)
        a1 = a1.at[n_square:2*n_square, 0:n_square].set(id_kron_eij)
        a1 = a1.at[n_square:2*n_square, n_square:2*n_square].set(m_kron_identity)
        
        exp_t_a1 = jspl.expm(t * a1)
        temp=vec(np.array(self.model.u1))
        # print(temp)
        # print(temp.reshape(-1, 1))
        # v_0_0 = v_0_0.at[0:n_square, 0:1].set(vec(np.array(self.model.u1)))
        v_0_0 = v_0_0.at[0:n_square, 0:1].set(temp.reshape(-1, 1))
        v_0_t = exp_t_a1 @ v_0_0
        
        vec_a_0_t = v_0_t[0:n_square, 0:1]
        vec_a_0_t_prime = v_0_t[n_square:2*n_square, 0:1]
        
        a_0_t = vec_inv(np.array(vec_a_0_t))
        a_0_t_prime = vec_inv(np.array(vec_a_0_t_prime))
        
        temp2=vec(np.array(self.model.omega))
        # b_0_tilde = b_0_tilde.at[0:n_square, 0:1].set(vec(np.array(self.model.omega)))
        b_0_tilde = b_0_tilde.at[0:n_square, 0:1].set(temp2.reshape(-1, 1))
        v_0_t_tilde = (jnp.linalg.inv(a1) @ (exp_t_a1 - i_n_square)) @ b_0_tilde
        
        vec_a_tilde_0_t = v_0_t_tilde[0:n_square, 0:1]
        vec_a_tilde_0_t_prime = v_0_t_tilde[n_square:2*n_square, 0:1]
        
        a_tilde_0_t = vec_inv(np.array(vec_a_tilde_0_t))
        a_tilde_0_t_prime = vec_inv(np.array(vec_a_tilde_0_t_prime))
        
        b_t = jnp.trace(i_n @ a_tilde_0_t)
        b_t_prime = jnp.trace(i_n @ a_tilde_0_t_prime)
        
        return jnp.array(a_0_t), jnp.array(a_0_t_prime), b_t, b_t_prime

