"""
Interpolation strategies for calibration curves.

This module provides various interpolation methods used in calibration,
including linear, log-linear, and cubic interpolation with configurable
extrapolation behavior.
"""

from abc import ABC, abstractmethod
from typing import Union, Literal, Callable
import numpy as np
from scipy.interpolate import interp1d


class InterpolationStrategy(ABC):
    """
    Abstract base class for interpolation strategies.
    
    All interpolation strategies must implement the interpolate method
    and support both flat and linear extrapolation.
    """
    
    @abstractmethod
    def interpolate(
        self, 
        x_data: np.ndarray,
        y_data: np.ndarray,
        x_new: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Perform interpolation at new x values.
        
        Parameters
        ----------
        x_data : np.ndarray
            Known x values
        y_data : np.ndarray
            Known y values
        x_new : float or np.ndarray
            New x values for interpolation
            
        Returns
        -------
        float or np.ndarray
            Interpolated y values
        """
        pass


class LinearInterpolator(InterpolationStrategy):
    """Linear interpolation strategy."""
    
    def __init__(self, extrapolation: Literal['flat', 'linear'] = 'flat'):
        """
        Initialize linear interpolator.
        
        Parameters
        ----------
        extrapolation : {'flat', 'linear'}
            Extrapolation method
        """
        self.extrapolation = extrapolation
    
    def interpolate(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        x_new: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Perform linear interpolation."""
        fill_value = self._get_fill_value(y_data)
        
        f = interp1d(
            x_data,
            y_data,
            kind='linear',
            fill_value=fill_value,
            bounds_error=False
        )
        
        return f(x_new)
    
    def _get_fill_value(self, y_data: np.ndarray):
        """Get fill value based on extrapolation method."""
        if self.extrapolation == 'flat':
            return (float(y_data[0]), float(y_data[-1]))
        else:  # linear
            return 'extrapolate'


class LogLinearInterpolator(InterpolationStrategy):
    """Log-linear interpolation strategy."""
    
    def __init__(self, extrapolation: Literal['flat', 'linear'] = 'flat'):
        """
        Initialize log-linear interpolator.
        
        Parameters
        ----------
        extrapolation : {'flat', 'linear'}
            Extrapolation method
        """
        self.extrapolation = extrapolation
    
    def interpolate(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        x_new: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Perform log-linear interpolation."""
        if np.any(y_data <= 0):
            raise ValueError("y_data must be positive for log-linear interpolation")
        
        # Transform to log space
        log_y_data = np.log(y_data)
        
        # Interpolate in log space
        fill_value = self._get_fill_value(log_y_data)
        
        f = interp1d(
            x_data,
            log_y_data,
            kind='linear',
            fill_value=fill_value,
            bounds_error=False
        )
        
        # Transform back
        return np.exp(f(x_new))
    
    def _get_fill_value(self, log_y_data: np.ndarray):
        """Get fill value based on extrapolation method."""
        if self.extrapolation == 'flat':
            return (float(log_y_data[0]), float(log_y_data[-1]))
        else:  # linear
            return 'extrapolate'


class CubicInterpolator(InterpolationStrategy):
    """Cubic spline interpolation strategy."""
    
    def __init__(self, extrapolation: Literal['flat', 'linear'] = 'flat'):
        """
        Initialize cubic interpolator.
        
        Parameters
        ----------
        extrapolation : {'flat', 'linear'}
            Extrapolation method
        """
        self.extrapolation = extrapolation
    
    def interpolate(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        x_new: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Perform cubic spline interpolation."""
        if len(x_data) < 4:
            # Fall back to linear for insufficient points
            interpolator = LinearInterpolator(self.extrapolation)
            return interpolator.interpolate(x_data, y_data, x_new)
        
        fill_value = self._get_fill_value(y_data)
        
        f = interp1d(
            x_data,
            y_data,
            kind='cubic',
            fill_value=fill_value,
            bounds_error=False
        )
        
        return f(x_new)
    
    def _get_fill_value(self, y_data: np.ndarray):
        """Get fill value based on extrapolation method."""
        if self.extrapolation == 'flat':
            return (float(y_data[0]), float(y_data[-1]))
        else:  # linear
            return 'extrapolate'


def create_interpolator(
    x_data: np.ndarray,
    y_data: np.ndarray,
    method: Literal['linear', 'loglinear', 'cubic'] = 'linear',
    extrapolation: Literal['flat', 'linear'] = 'flat'
) -> Callable:
    """
    Factory function to create interpolators.
    
    Parameters
    ----------
    x_data : np.ndarray
        X-axis data points
    y_data : np.ndarray
        Y-axis data points
    method : {'linear', 'loglinear', 'cubic'}
        Interpolation method
    extrapolation : {'flat', 'linear'}
        Extrapolation method
        
    Returns
    -------
    Callable
        Interpolation function
        
    Examples
    --------
    >>> x = np.array([0, 1, 2, 3])
    >>> y = np.array([1, 2, 4, 8])
    >>> interp = create_interpolator(x, y, 'linear')
    >>> interp(1.5)
    3.0
    """
    # Validate inputs
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)
    
    if len(x_data) != len(y_data):
        raise ValueError("x_data and y_data must have the same length")
    
    if len(x_data) < 2:
        raise ValueError("At least 2 data points are required")
    
    if np.any(np.diff(x_data) <= 0):
        raise ValueError("x_data must be strictly increasing")
    
    # Create appropriate interpolator
    if method == 'linear':
        strategy = LinearInterpolator(extrapolation)
    elif method == 'loglinear':
        strategy = LogLinearInterpolator(extrapolation)
    elif method == 'cubic':
        strategy = CubicInterpolator(extrapolation)
    else:
        raise ValueError(f"Unknown interpolation method: {method}")
    
    # Return a function that uses the strategy
    def interpolate_function(x_new):
        return strategy.interpolate(x_data, y_data, x_new)
    
    return interpolate_function
