"""
Calibration methods specific to LRW models.

This module provides calibration functionality for Linear Rational Wishart
models including curve fitting and parameter estimation.
"""

from typing import Optional, Dict, Tuple, List
import jax.numpy as jnp
import numpy as np

from ..models.interest_rate.lrw_model import LRWModel
from .alpha_curve import getInitialAlpha


class LRWCalibrator:
    """
    Calibrator for Linear Rational Wishart models.
    
    Handles calibration to market curves and volatility surfaces.
    """
    
    def __init__(self, lrw_model: LRWModel):
        """
        Initialize the calibrator.
        
        Parameters
        ----------
        lrw_model : LRWModel
            The LRW model to calibrate
        """
        self.model = lrw_model
        
    def calibrate_to_curve(
        self,
        market_dates: jnp.ndarray,
        market_zc_values: jnp.ndarray,
        use_pseudo_inverse: bool = True,
        interpolation: str = 'loglinear',
        extrapolation: str = 'flat'
    ) -> None:
        """
        Calibrate the model to a market zero coupon curve.
        
        Parameters
        ----------
        market_dates : jnp.ndarray
            Maturity dates for market ZC bonds
        market_zc_values : jnp.ndarray
            Market ZC bond prices
        use_pseudo_inverse : bool, default=True
            Whether to use pseudo-inverse smoothing
        interpolation : str, default='loglinear'
            Interpolation method
        extrapolation : str, default='flat'
            Extrapolation method
        """
        # Compute initial model ZC values
        initial_model_values = jnp.array([
            self.model.bond(t) for t in market_dates
            # self.model.Bond(t) for t in market_dates
        ])
        
        if use_pseudo_inverse:
            # Get alpha adjustment curve
            initial_alpha_curve = getInitialAlpha(
                market_dates,
                market_zc_values,
                initial_model_values,
                interpolation=interpolation,
                extrapolation=extrapolation
            )
            
            # Apply to model
            self.model.set_pseudo_inverse_smoothing(initial_alpha_curve)
            
    def calibrate_to_swaptions(
        self,
        swaption_data: List[Dict],
        method: str = "least_squares",
        max_iter: int = 100
    ) -> Dict[str, float]:
        """
        Calibrate model parameters to swaption prices.
        
        Parameters
        ----------
        swaption_data : List[Dict]
            List of swaption market data with keys:
            'maturity', 'tenor', 'strike', 'price'
        method : str, default="least_squares"
            Optimization method
        max_iter : int, default=100
            Maximum iterations
            
        Returns
        -------
        Dict[str, float]
            Calibration results and diagnostics
        """
        # This would implement full parameter calibration
        # For now, return placeholder
        return {
            "status": "not_implemented",
            "message": "Full calibration to be implemented"
        }
        
    def compute_model_zc_curve(
        self,
        dates: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Compute model zero coupon curve for given dates.
        
        Parameters
        ----------
        dates : jnp.ndarray
            Maturity dates
            
        Returns
        -------
        jnp.ndarray
            Model ZC prices
        """
        # return jnp.array([self.model.Bond(t) for t in dates])
        return jnp.array([self.model.bond(t) for t in dates])
    
    def validate_gindikin_condition(self) -> bool:
        """
        Validate that model parameters satisfy Gindikin condition.
        
        Returns
        -------
        bool
            True if condition is satisfied
        """
        return self.model.wishart.check_gindikin()
    
    def adjust_omega_for_gindikin(self, beta: float = 4.0) -> None:
        """
        Adjust omega to satisfy Gindikin condition.
        
        Parameters
        ----------
        beta : float, default=4.0
            Scaling factor for omega = beta * sigma @ sigma
        """
        sigma = self.model.sigma
        omega = beta * sigma @ sigma
        
        # Check if this satisfies Gindikin
        temp = omega - 3.0 * sigma @ sigma
        if jnp.linalg.det(temp) < 0.0:
            print("Warning: Gindikin condition not satisfied with beta =", beta)
            # Try to find a valid beta
            for new_beta in jnp.arange(3.1, 10.0, 0.1):
                omega = new_beta * sigma @ sigma
                temp = omega - 3.0 * sigma @ sigma
                if jnp.linalg.det(temp) >= 0.0:
                    print(f"Using beta = {new_beta} to satisfy Gindikin condition")
                    break
        
        self.model.omega = omega
        self.model.wishart.omega = omega
