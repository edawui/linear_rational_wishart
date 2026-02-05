



# utils/density_approximation.py
"""Density approximation methods for option pricing."""

import math
import numpy as np
from typing import List
import scipy.special as sp_special
import scipy.integrate as sp_i


class GammaApproximation:
    """Gamma distribution approximation for density."""
    
    def __init__(self, order: float = 1.0/3.0):
        """Initialize Gamma approximation."""
        self.order = order
        self.mu = []
        self.k = 0.01
        
    def set_parameters(self, mu: List[float], k: float):
        """Set approximation parameters."""
        self.mu = mu
        self.k = k
        
        # Compute Gamma parameters
        self._compute_gamma_params()
        
    def _compute_gamma_params(self):
        """Compute alpha and beta for Gamma distribution."""
        if len(self.mu) < 2:
            raise ValueError("Need at least 2 moments")
            
        # Mean and variance from moments
        mean = self.mu[1]
        variance = self.mu[2] - self.mu[1]**2
        
        # Gamma parameters
        self.beta = mean / variance
        self.alpha = mean * self.beta
        
    def compute_integ(self) -> float:
        """Compute integral for option pricing."""
        def integrand(x):
            if x <= 0:
                return 0
            return max(0, x - self.k) * self.density(x)
            
        # Integrate from 0 to infinity
        result, _ = sp_i.quad(integrand, 0, np.inf)
        return result
        
    def density(self, x: float) -> float:
        """Gamma density function."""
        if x <= 0:
            return 0
            
        return (self.beta**self.alpha / sp_special.gamma(self.alpha)) * \
               x**(self.alpha - 1) * math.exp(-self.beta * x)


# class EdgeworthApproximation:
#     """Edgeworth expansion approximation."""
    
#     def __init__(self, order: int = 4):
#         """Initialize Edgeworth approximation."""
#         self.order = order
#         self.mu = []
#         self.cumulants = []
        
#     def set_parameters(self, mu: List[float]):
#         """Set moments and compute cumulants."""
#         self.mu = mu
#         self._compute_cumulants()
        
#     def _compute_cumulants(self):
#         """Convert moments to cumulants."""
#         # First few cumulants
#         if len(self.mu) > 0:
#             self.cumulants.append(self.mu[0])  # ?1 = µ1
            
#         if len(self.mu) > 1:
#             self.cumulants.append(self.mu[1] - self.mu[0]**2)  # ?2 = µ2 - µ1²
            
#         if len(self.mu) > 2:
#             self.cumulants.append(self.mu[2] - 3*self.mu[1]*self.mu[0] + 2*self.mu[0]**3)
            
#         # Higher order cumulants...
        
#     def density(self, x: float) -> float:
#         """Edgeworth expansion density."""
#         # Standardize
#         if len(self.cumulants) < 2 or self.cumulants[1] <= 0:
#             return 0
            
#         mean = self.cumulants[0]
#         std = math.sqrt(self.cumulants[1])
#         z = (x - mean) / std
        
#         # Standard normal density
#         phi = math.exp(-z**2 / 2) / math.sqrt(2 * math.pi)
        
#         # Edgeworth correction terms
#         correction = 1.0
        
#         if len(self.cumulants) > 2:
#             # Third order term
#             h3 = z**3 - 3*z
#             correction += (self.cumulants[2] / (6 * std**3)) * h3
            
#         if len(self.cumulants) > 3:
#             # Fourth order term
#             h4 = z**4 - 6*z**2 + 3
#             correction += (self.cumulants[3] / (24 * std**4)) * h4
            
#         return phi * correction / std 
    
#         # __init__(self, n: int, x0: jnp.ndarray, omega: jnp.ndarray,
#         #          m: jnp.ndarray, sigma: jnp.ndarray):
#         # """Initialize Wishart process with jump capability."""
#         # super().__init__(n, x0, omega, m, sigma)
#         # self.has_jump = False
#         # self.jump_component = None
        
#     def set_jump(self, jump_component: JumpComponent):
#         """Add jump component to the process."""
#         self.jump_component = jump_component
#         self.has_jump = True
        
#     def compute_mean_decompose_with_jump(self, u: jnp.ndarray, t: float, 
#                                         c: float) -> Tuple[float, float, jnp.ndarray]:
#         """Compute mean decomposition with jump component."""
#         if not self.has_jump:
#             b, a = self.compute_mean_decompose(u, t)
#             return 0.0, b, a
            
#         # Implementation of jump decomposition
#         # This follows the original ComputeMeanDecomposeWithJump logic
#         exp_mt = jspl.expm(t * self.m)
#         a = jnp.transpose(exp_mt) @ u @ exp_mt
        
#         # Compute b component
#         def f1(s):
#             exp_ms = jspl.expm(s * self.m)
#             temp = jnp.transpose(exp_ms) @ u @ exp_ms
#             z1 = jnp.trace(temp @ self.omega)
#             return z1
            
#         f2 = lambda s: f1(s)
#         r1 = sp_i.quad(f2, 0, t)
#         b = r1[0]
        
#         # Compute c component (jump contribution)
#         c_value = self._compute_jump_contribution(u, t, c)
        
#         return c_value, b, a
        
#     def _compute_jump_contribution(self, u: jnp.ndarray, t: float, c: float) -> float:
#         """Compute the jump contribution to the mean."""
#         if not self.has_jump:
#             return 0.0
            
#         # Implementation based on jump parameters
#         lambda_int = self.jump_component.lambda_intensity
#         nu = self.jump_component.nu
#         eta = self.jump_component.eta
#         xi = self.jump_component.xi
        
#         # Jump contribution calculation
#         jump_contrib = lambda_int * t * (jnp.trace(u @ xi) / (1 - jnp.trace(u @ eta)))
        
#         return jump_contrib
        
#     def phi_one(self, z: complex, theta1: jnp.ndarray) -> complex:
#         """Compute Phi_One with jump component."""
#         if not self.has_jump:
#             return super().phi_one(z, theta1)
            
#         # Implementation with jump
#         t = self.maturity
        
#         # Base Wishart contribution
#         base_phi = super().phi_one(z, theta1)
        
#         # Jump contribution
#         if self.has_jump:
#             jump_phi = self._compute_jump_phi(z, theta1, t)
#             return base_phi * jump_phi
            
#         return base_phi
        
#     def _compute_jump_phi(self, z: complex, theta1: jnp.ndarray, t: float) -> complex:
#         """Compute jump contribution to characteristic function."""
#         lambda_int = self.jump_component.lambda_intensity
#         eta = self.jump_component.eta
#         xi = self.jump_component.xi
        
#         # Jump characteristic function
#         det_term = jnp.linalg.det(jnp.eye(self.n) - theta1 @ eta)
#         trace_term = jnp.trace(theta1 @ xi)
        
#         jump_cf = cmath.exp(lambda_int * t * (trace_term / det_term - 1))
        
#         return jump_cf
        
#     def compute_moments_with_jump(self, t: float, order: int = 3):
#         """Compute moments with jump component."""
#         if not self.has_jump:
#             return self.compute_moments(t, order)
            
#         # Extended moment computation including jump effects
#         # Implementation would follow the pattern from the original code
#         pass
