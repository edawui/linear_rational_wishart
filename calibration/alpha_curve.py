
# calibration/alpha_curve.py
"""Alpha curve calibration for pseudo-inverse smoothing."""

import numpy as np
import jax.numpy as jnp
from typing import List, Tuple
import scipy.interpolate as sp_interpolate


def getInitialAlpha(zcDates, marketZcValues, modelZcValues,interpolation='loglinear', extrapolation='flat'):

    if len(zcDates) != len(marketZcValues):
            raise ValueError("zcDates and marketZcValues must have the same length")
    
    if len(zcDates) != len(modelZcValues):
            raise ValueError("zcDates and modelZcValues must have the same length")

    marketZcValues = np.asarray(marketZcValues)
    modelZcValues = np.asarray(modelZcValues)
    
    if np.any(modelZcValues == 0):
        raise ValueError("modelZcValues contains zero(s), division by zero is not allowed")
   
    alphaValues =marketZcValues / modelZcValues 
    initialCurveAlpha = AlphaFromInitialCurve(zcDates, alphaValues)
    #, interpolation=interpolation, extrapolation=extrapolation)

    return initialCurveAlpha


class AlphaFromInitialCurve:
    """Alpha curve adjustment from initial curve."""
    
    def __init__(self, tenors: List[float], alphas: List[float]):
        """Initialize with tenor points and alpha values."""
        self.tenors = jnp.array(tenors)
        self.alphas = jnp.array(alphas)
        
        # Create interpolator
        self._create_interpolator()
        
    def _create_interpolator(self):
        """Create smooth interpolator for alpha curve."""
        # Use cubic spline for smooth interpolation
        self.interpolator = sp_interpolate.CubicSpline(
            self.tenors, self.alphas, bc_type='natural'
        )
        
    def get_alpha(self, t: float) -> float:
        """Get alpha adjustment at time t."""
        if t <= self.tenors[0]:
            return float(self.alphas[0])
        elif t >= self.tenors[-1]:
            return float(self.alphas[-1])
        else:
            return float(self.interpolator(t))
            
    def get_alpha_derivative(self, t: float) -> float:
        """Get derivative of alpha adjustment."""
        return float(self.interpolator.derivative()(t))
        
    def update_curve(self, tenors: List[float], alphas: List[float]):
        """Update the alpha curve."""
        self.tenors = jnp.array(tenors)
        self.alphas = jnp.array(alphas)
        self._create_interpolator()


