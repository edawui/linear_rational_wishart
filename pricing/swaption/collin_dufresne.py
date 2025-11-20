

# pricing/swaption/collin_dufresne.py
"""Collin-Dufresne approximation for swaption pricing."""

import math
from typing import Dict, Any, Optional
import jax.numpy as jnp

from .base import BaseSwaptionPricer
from ...utils.local_functions import tr_uv


class CollinDufresnePricer(BaseSwaptionPricer):
    """Collin-Dufresne approximation based swaption pricing."""
    
    def __init__(self, model, max_order: int = 3):
        """Initialize Collin-Dufresne pricer."""
        super().__init__(model)
        self.max_order = max_order
        
    def price(self, max_order: int = None) -> float:
        """Price swaption using Collin-Dufresne approximation."""
        self.validate_inputs()
        
        if max_order is not None:
            self.max_order = max_order
            
        # Compute b3 and a3
        self.model.compute_b3_a3()
        
        # Calculate ATM strike
        atm_strike = self.model.compute_swap_rate()
        strike_offset = atm_strike - self.model.strike
        
        # Extract coefficients
        b = self.model.b3
        a1 = self.model.a3[0, 0]
        a2 = self.model.a3[1, 1]
        
        # Compute moments
        if self.max_order == 3:
            self.model.wishart.compute_moments(self.model.maturity, 3)
            cd_mu = self.model.wishart.compute_mu(self.model.b3, self.model.a3, 3)
        else:
            self.model.wishart.compute_moments(self.model.maturity, 8)
            cd_mu = self.model.wishart.compute_mu(self.model.b3, self.model.a3, 7)
            
        # Compute Collin-Dufresne coefficients
        cd_c = self.model.wishart.collin_dufresne_c(cd_mu, self.max_order)
        cd_gamma = self.model.wishart.collin_dufresne_gamma(cd_c, self.max_order)
        cd_lambda = self.model.wishart.collin_dufresne_lambda(cd_c, 0.0, self.max_order)
        
        # Calculate annuity
        annuity, _ = self.model.compute_annuity()
        
        # Sum expansion terms
        price = 0
        for j in range(0, self.max_order + 1):
            price += cd_gamma[j] * (cd_lambda[j + 1] + cd_c[1] * cd_lambda[j])
            
        # Apply scaling
        multiplier = jnp.exp(-self.model.alpha * self.model.maturity) / (1 + tr_uv(self.model.u1, self.model.x0))
        price *= multiplier
        
        # Store intermediate results for analysis
        self.cd_coefficients = {
            'mu': cd_mu,
            'c': cd_c,
            'gamma': cd_gamma,
            'lambda': cd_lambda
        }
        
        return price
        
    def get_pricing_info(self) -> Dict[str, Any]:
        """Get detailed pricing information."""
        info = {
            "method": "Collin-Dufresne Approximation",
            "max_order": self.max_order,
        }
        
        if hasattr(self, 'cd_coefficients'):
            info.update(self.cd_coefficients)
            
        return info
        
    def price_with_error_estimate(self) -> tuple[float, float]:
        """Price with error estimate based on last term."""
        price = self.price()
        
        # Estimate error from last term in expansion
        if hasattr(self, 'cd_coefficients'):
            cd_gamma = self.cd_coefficients['gamma']
            cd_lambda = self.cd_coefficients['lambda']
            cd_c = self.cd_coefficients['c']
            
            j = self.max_order
            last_term = cd_gamma[j] * (cd_lambda[j + 1] + cd_c[1] * cd_lambda[j])
            
            # Rough error estimate - assume next term is similar magnitude
            error_estimate = abs(last_term)
        else:
            error_estimate = None
            
        return price, error_estimate
