"""
Peter Jäckel's method for Bachelier implied volatility.

This module implements Peter Jäckel's highly accurate and efficient method
for computing implied normal (Bachelier) volatility from option prices.
"""

from typing import Union
import numpy as np
from scipy.stats import norm as scipy_norm

from .bachelier import SQRT_TWO_PI, ONE_OVER_SQRT_TWO_PI, DBL_MIN, DBL_MAX


class JackelImpliedVolatility:
    """
    Peter Jäckel's method for computing Bachelier implied volatility.
    
    This implementation follows the algorithm described in Peter Jäckel's paper
    "Implied Normal Volatility" (2017), providing accurate and robust implied
    volatility calculations for the Bachelier model.
    
    Examples
    --------
    >>> calculator = JackelImpliedVolatility()
    >>> forward, strike, T, price = 100.0, 105.0, 1.0, 8.9595
    >>> iv = calculator.implied_normal_volatility(forward, strike, T, price, 1.0)
    >>> print(f"Implied volatility: {iv:.4f}")
    Implied volatility: 20.0000
    """
    
    def __init__(self):
        """Initialize the Jäckel calculator."""
        # Region boundary for InvPhiTilde approximation
        self._phi_c_tilde = -0.001882039271
    
    def phi_tilde_times_x(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Compute x * Φ(x) + φ(x).
        
        Parameters
        ----------
        x : float or array-like
            Input value(s)
            
        Returns
        -------
        float or ndarray
            x * Φ(x) + φ(x)
        """
        return x * scipy_norm.cdf(x) + scipy_norm.pdf(x)
    
    def inv_phi_tilde(self, phi_tilde_star: float) -> float:
        """
        Inverse of the PhiTilde function.
        
        This function finds x such that PhiTilde(x) = phi_tilde_star,
        where PhiTilde(x) = Φ(x) + φ(x)/x.
        
        Parameters
        ----------
        phi_tilde_star : float
            Target value for PhiTilde
            
        Returns
        -------
        float
            x such that PhiTilde(x) = phi_tilde_star
            
        Notes
        -----
        The implementation uses different approximations for different
        regions of phi_tilde_star to ensure high accuracy.
        """
        # Handle the symmetric case
        if phi_tilde_star > 1:
            return -self.inv_phi_tilde(1 - phi_tilde_star)
        
        if phi_tilde_star >= 0:
            return float('nan')
        
        if phi_tilde_star < self._phi_c_tilde:
            # Region 1: Use rational approximation
            x_bar = self._inv_phi_tilde_region1(phi_tilde_star)
        else:
            # Region 2: Use different rational approximation
            x_bar = self._inv_phi_tilde_region2(phi_tilde_star)
        
        # Refine using Householder's method
        x_star = self._refine_inv_phi_tilde(x_bar, phi_tilde_star)
        
        return x_star
    
    def _inv_phi_tilde_region1(self, phi_tilde_star: float) -> float:
        """
        Compute initial approximation for region 1.
        
        Parameters
        ----------
        phi_tilde_star : float
            Target PhiTilde value (must be < -0.001882039271)
            
        Returns
        -------
        float
            Initial approximation x_bar
        """
        # Equation (2.1)
        g = 1 / (phi_tilde_star - 0.5)
        g2 = g * g
        
        # Equation (2.2) - Rational approximation coefficients
        num = (0.032114372355 - 
               g2 * (0.016969777977 - 
                     g2 * (0.002620733246 - 0.000096066952861 * g2)))
        denom = (1 - 
                 g2 * (0.6635646938 - 
                       g2 * (0.14528712196 - 0.010472855461 * g2)))
        
        xi_bar = num / denom
        
        # Equation (2.3)
        x_bar = g * (ONE_OVER_SQRT_TWO_PI + xi_bar * g2)
        
        return x_bar
    
    def _inv_phi_tilde_region2(self, phi_tilde_star: float) -> float:
        """
        Compute initial approximation for region 2.
        
        Parameters
        ----------
        phi_tilde_star : float
            Target PhiTilde value (must be >= -0.001882039271 and < 0)
            
        Returns
        -------
        float
            Initial approximation x_bar
        """
        # Equation (2.4)
        h = np.sqrt(-np.log(-phi_tilde_star))
        
        # Equation (2.5) - Rational approximation
        num = (9.4883409779 - 
               h * (9.6320903635 - 
                    h * (0.58556997323 + 2.1464093351 * h)))
        denom = (1 - 
                 h * (0.65174820867 + 
                      h * (1.5120247828 + 0.000066437847132 * h)))
        
        x_bar = num / denom
        
        return x_bar
    
    def _refine_inv_phi_tilde(self, x_bar: float, phi_tilde_star: float) -> float:
        """
        Refine the approximation using Householder's method.
        
        Parameters
        ----------
        x_bar : float
            Initial approximation
        phi_tilde_star : float
            Target PhiTilde value
            
        Returns
        -------
        float
            Refined approximation x_star
        """
        # Compute PhiTilde at x_bar
        phi_tilde_x_bar = scipy_norm.cdf(x_bar) + scipy_norm.pdf(x_bar) / x_bar
        
        # Equation (2.7)
        q = (phi_tilde_x_bar - phi_tilde_star) / scipy_norm.pdf(x_bar)
        x2 = x_bar * x_bar
        
        # Equation (2.6) - Householder's method (4th order)
        num = 3 * q * x2 * (2 - q * x_bar * (2 + x2))
        denom = (6 + q * x_bar * (-12 + 
                                  x_bar * (6 * q + 
                                          x_bar * (-6 + 
                                                  q * x_bar * (3 + x2)))))
        
        x_star = x_bar + num / denom
        
        return x_star
    
    def intrinsic_value(
        self, 
        forward: float, 
        strike: float, 
        call_or_put: float
    ) -> float:
        """
        Calculate option intrinsic value.
        
        Parameters
        ----------
        forward : float
            Forward price
        strike : float
            Strike price
        call_or_put : float
            1.0 for call, -1.0 for put
            
        Returns
        -------
        float
            Intrinsic value
        """
        return max(call_or_put * (forward - strike), 0.0)
    
    def implied_normal_volatility(
        self,
        forward: float,
        strike: float,
        time_to_expiry: float,
        price: float,
        call_or_put: float
    ) -> float:
        """
        Calculate implied normal volatility using Jäckel's method.
        
        Parameters
        ----------
        forward : float
            Forward price
        strike : float
            Strike price
        time_to_expiry : float
            Time to expiry in years
        price : float
            Option price
        call_or_put : float
            1.0 for call, -1.0 for put
            
        Returns
        -------
        float
            Implied normal (Bachelier) volatility
            
        Notes
        -----
        Returns 0.0 if price equals intrinsic value, and -DBL_MAX if
        price is less than intrinsic value (indicating arbitrage).
        """
        # Handle ATM case
        if forward == strike:
            return price * SQRT_TWO_PI / np.sqrt(time_to_expiry)
        
        # Calculate intrinsic value
        intrinsic = self.intrinsic_value(forward, strike, call_or_put)
        absolute_moneyness = abs(forward - strike)
        
        # Check for arbitrage
        if price < intrinsic:
            return -DBL_MAX
        
        # Check if at intrinsic value
        if price == intrinsic:
            return 0.0
        
        # Equation (1.6) - Normalized time value
        phi_tilde_star = (intrinsic - price) / absolute_moneyness
        
        # Solve equation (1.7)
        x_star = self.inv_phi_tilde(phi_tilde_star)
        
        # Equation (1.8) - Implied volatility
        implied_vol = absolute_moneyness / abs(x_star * np.sqrt(time_to_expiry))
        
        return implied_vol
    
    def validate_inputs(
        self,
        forward: float,
        strike: float,
        time_to_expiry: float,
        price: float
    ) -> None:
        """
        Validate input parameters.
        
        Parameters
        ----------
        forward : float
            Forward price
        strike : float
            Strike price
        time_to_expiry : float
            Time to expiry
        price : float
            Option price
            
        Raises
        ------
        ValueError
            If inputs are invalid
        """
        if time_to_expiry <= 0:
            raise ValueError("Time to expiry must be positive")
        
        if price < 0:
            raise ValueError("Option price cannot be negative")
        
        if forward <= 0 or strike <= 0:
            raise ValueError("Forward and strike must be positive")


def implied_vol_jackel(
    forward: float,
    strike: float, 
    time_to_expiry: float,
    price: float,
    call_or_put: float = 1.0
) -> float:
    """
    Convenience function for Jäckel implied volatility calculation.
    
    Parameters
    ----------
    forward : float
        Forward price
    strike : float
        Strike price
    time_to_expiry : float
        Time to expiry
    price : float
        Option price
    call_or_put : float
        1.0 for call, -1.0 for put
        
    Returns
    -------
    float
        Implied normal volatility
        
    Examples
    --------
    >>> iv = implied_vol_jackel(100, 105, 1.0, 8.9595, 1.0)
    >>> print(f"Implied volatility: {iv:.4f}")
    Implied volatility: 20.0000
    """
    calculator = JackelImpliedVolatility()
    return calculator.implied_normal_volatility(
        forward, strike, time_to_expiry, price, call_or_put
    )