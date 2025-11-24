

# calibration/pseudo_inverse.py
"""Pseudo-inverse smoothing for model calibration."""

import jax.numpy as jnp
from typing import List, Tuple, Optional
import scipy.optimize as sp_optimize
from functools import partial

from .alpha_curve import AlphaFromInitialCurve


class PseudoInverseCalibrator:
    """Calibrate model using pseudo-inverse smoothing."""
    
    def __init__(self, model):
        """Initialize calibrator with model."""
        self.model = model
        self.alpha_curve = None
        
    def calibrate_alpha_curve(self, market_bonds: List[Tuple[float, float]], 
                            initial_guess: Optional[List[float]] = None) -> AlphaFromInitialCurve:
        """Calibrate alpha curve to match market bond prices."""
        tenors = [t for t, _ in market_bonds]
        market_prices = [p for _, p in market_bonds]
        
        if initial_guess is None:
            # Start with flat alpha adjustment
            initial_guess = [1.0] * len(tenors)
            
        # Objective function
        def objective(alphas):
            # Create alpha curve
            alpha_curve = AlphaFromInitialCurve(tenors, alphas)
            self.model.set_pseudo_inverse_smoothing(alpha_curve)
            
            # Compute model prices
            model_prices = []
            for tenor in tenors:
                price = self.model.bond(tenor)
                model_prices.append(price)
                
            # Sum of squared errors
            errors = jnp.array(model_prices) - jnp.array(market_prices)
            return jnp.sum(errors**2)
            
        # Optimize
        result = sp_optimize.minimize(objective, initial_guess, method='BFGS')
        
        # Create final alpha curve
        self.alpha_curve = AlphaFromInitialCurve(tenors, result.x)
        self.model.set_pseudo_inverse_smoothing(self.alpha_curve)
        
        return self.alpha_curve
        
    def calibrate_with_regularization(self, market_bonds: List[Tuple[float, float]], 
                                    lambda_reg: float = 0.1) -> AlphaFromInitialCurve:
        """Calibrate with smoothness regularization."""
        tenors = [t for t, _ in market_bonds]
        market_prices = [p for _, p in market_bonds]
        
        # Objective with regularization
        def objective(alphas):
            # Pricing error
            alpha_curve = AlphaFromInitialCurve(tenors, alphas)
            self.model.set_pseudo_inverse_smoothing(alpha_curve)
            
            model_prices = []
            for tenor in tenors:
                price = self.model.bond(tenor)
                model_prices.append(price)
                
            errors = jnp.array(model_prices) - jnp.array(market_prices)
            pricing_error = jnp.sum(errors**2)
            
            # Smoothness penalty (second differences)
            if len(alphas) >= 3:
                second_diff = jnp.diff(jnp.diff(alphas))
                smoothness_penalty = lambda_reg * jnp.sum(second_diff**2)
            else:
                smoothness_penalty = 0
                
            return pricing_error + smoothness_penalty
            
        # Initial guess
        initial_guess = [1.0] * len(tenors)
        
        # Optimize
        result = sp_optimize.minimize(objective, initial_guess, method='BFGS')
        
        # Create final alpha curve
        self.alpha_curve = AlphaFromInitialCurve(tenors, result.x)
        self.model.set_pseudo_inverse_smoothing(self.alpha_curve)
        
        return self.alpha_curve
