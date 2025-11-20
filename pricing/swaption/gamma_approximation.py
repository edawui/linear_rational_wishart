

# pricing/swaption/gamma_approximation.py
"""Gamma approximation methods for swaption pricing."""

import math
from typing import Dict, Any
import jax.numpy as jnp

from .base import BaseSwaptionPricer
from ...utils.local_functions import tr_uv
from Refactoring_all.refactoring_base.density-approximation.math_density_approximation import GammaApproximation


class GammaApproximationPricer(BaseSwaptionPricer):
    """Gamma approximation based swaption pricing."""
    
    def __init__(self, model, gamma_order: float = 1.0/3.0):
        """Initialize Gamma approximation pricer."""
        super().__init__(model)
        self.gamma_order = gamma_order
        self.k_param = 0.01
        
    def price(self, k_param: float = None) -> float:
        """Price swaption using Gamma approximation."""
        self.validate_inputs()
        
        if k_param is not None:
            self.k_param = k_param
            
        # Compute b3 and a3
        self.model.compute_b3_a3()
        
        # Calculate ATM strike
        atm_strike = self.model.compute_swap_rate()
        strike_offset = atm_strike - self.model.strike
        
        # Extract coefficients
        b_ = self.model.b3
        a1_ = self.model.a3[0, 0]
        a2_ = self.model.a3[1, 1]
        
        # Compute moments up to order 4
        self.model.wishart.compute_moments(self.model.maturity, order=4)
        mu = self.model.wishart.compute_mu(self.model.b3, self.model.a3, approx_order=4)
        
        # Set up Gamma approximation
        density_approximation = GammaApproximation(self.gamma_order)
        density_approximation.set_parameters(mu, self.k_param)
        
        # Compute integral
        price_gamma_temp = density_approximation.compute_integ()
        
        # Apply scaling
        t0 = self.model.maturity
        multiplier = math.exp(-self.model.alpha * t0) / (1 + self.model.x0[0, 0])
        price_gamma = price_gamma_temp * multiplier
        
        # Store approximation details
        self.approximation_details = {
            'mu': mu,
            'gamma_alpha': density_approximation.alpha,
            'gamma_beta': density_approximation.beta
        }
        
        return price_gamma
        
    def get_pricing_info(self) -> Dict[str, Any]:
        """Get detailed pricing information."""
        info = {
            "method": "Gamma Approximation",
            "gamma_order": self.gamma_order,
            "k_parameter": self.k_param
        }
        
        if hasattr(self, 'approximation_details'):
            info.update(self.approximation_details)
            
        return info
        
    def price_with_different_orders(self, orders: list = None) -> Dict[float, float]:
        """Price using different Gamma orders for comparison."""
        if orders is None:
            orders = [1/3, 1/2, 2/3, 1.0]
            
        results = {}
        
        original_order = self.gamma_order
        
        for order in orders:
            self.gamma_order = order
            price = self.price()
            results[order] = price
            
        self.gamma_order = original_order
        
        return results


