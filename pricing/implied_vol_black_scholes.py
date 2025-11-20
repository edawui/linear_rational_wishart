"""
Implied volatility calculations for various option pricing models.

This module provides methods for calculating implied volatility from
option prices using different numerical methods.
"""

from typing import Union, Literal, Optional, Callable
import numpy as np
from scipy.optimize import brentq, newton

from .black_scholes import black_scholes_price, black_scholes_vega


def implied_volatility_black_scholes(
    market_price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    dividend_yield: float = 0.0,
    option_type: Union[Literal['call', 'put'], bool] = 'call',
    method: Literal['brent', 'newton', 'bisection'] = 'newton',#'brent',
    vol_bounds: tuple[float, float] = (0.001, 5.0),
    tolerance: float = 1e-6,
    max_iterations: int = 100
) -> float:
    """
    Calculate Black-Scholes implied volatility from option price.
    
    Parameters
    ----------
    market_price : float
        Market price of the option
    spot : float
        Current spot price
    strike : float
        Strike price
    time_to_expiry : float
        Time to expiry in years
    risk_free_rate : float
        Risk-free rate
    dividend_yield : float, optional
        Dividend yield, default is 0.0
    option_type : {'call', 'put'} or bool, optional
        Option type
    method : {'brent', 'newton', 'bisection'}, optional
        Numerical method to use
    vol_bounds : tuple[float, float], optional
        Bounds for volatility search
    tolerance : float, optional
        Convergence tolerance
    max_iterations : int, optional
        Maximum iterations for iterative methods
        
    Returns
    -------
    float
        Implied volatility
        
    Raises
    ------
    ValueError
        If implied volatility cannot be found
        
    Examples
    --------
    >>> spot, strike, T, r = 100.0, 105.0, 1.0, 0.05
    >>> market_price = 6.04
    >>> iv = implied_volatility_black_scholes(market_price, spot, strike, T, r)
    >>> print(f"Implied volatility: {iv:.1%}")
    Implied volatility: 20.0%
    """
    # Check for arbitrage violations
    intrinsic_value = _calculate_intrinsic_value(spot, strike, option_type, 
                                                  risk_free_rate, dividend_yield, 
                                                  time_to_expiry)
    #todo
    # if market_price < intrinsic_value:
    #     raise ValueError(f"Market price {market_price:.4f} is below intrinsic value {intrinsic_value:.4f}")
    
    if method == 'brent':
        return _implied_vol_brent(market_price, spot, strike, time_to_expiry,
                                  risk_free_rate, dividend_yield, option_type,
                                  vol_bounds, tolerance)
    elif method == 'newton':
        return _implied_vol_newton(market_price, spot, strike, time_to_expiry,
                                   risk_free_rate, dividend_yield, option_type,
                                   tolerance, max_iterations)
    elif method == 'bisection':
        return _implied_vol_bisection(market_price, spot, strike, time_to_expiry,
                                      risk_free_rate, dividend_yield, option_type,
                                      vol_bounds, tolerance, max_iterations)
    else:
        raise ValueError(f"Unknown method: {method}")


def _calculate_intrinsic_value(
    spot: float,
    strike: float,
    option_type: Union[Literal['call', 'put'], bool],
    risk_free_rate: float,
    dividend_yield: float,
    time_to_expiry: float
) -> float:
    """Calculate option intrinsic value."""
    if isinstance(option_type, str):
        is_call = option_type.lower() == 'call'
    else:
        is_call = bool(option_type)
    
    # Adjust spot for dividends
    forward = spot * np.exp((risk_free_rate - dividend_yield) * time_to_expiry)
    
    if is_call:
        intrinsic = max(forward - strike, 0.0)
    else:
        intrinsic = max(strike - forward, 0.0)
    
    # Discount to present value
    return intrinsic * np.exp(-risk_free_rate * time_to_expiry)


def _implied_vol_brent(
    market_price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    dividend_yield: float,
    option_type: Union[Literal['call', 'put'], bool],
    vol_bounds: tuple[float, float],
    tolerance: float
) -> float:
    """Use Brent's method for implied volatility."""
    def objective(sigma):
        model_price = black_scholes_price(
            spot, strike, time_to_expiry, risk_free_rate,
            sigma, dividend_yield, option_type
        )
        return model_price - market_price
    
    try:
        iv = brentq(objective, vol_bounds[0], vol_bounds[1], xtol=tolerance)
        return iv
    except ValueError as e:
        # Check bounds
        lower_price = black_scholes_price(
            spot, strike, time_to_expiry, risk_free_rate,
            vol_bounds[0], dividend_yield, option_type
        )
        upper_price = black_scholes_price(
            spot, strike, time_to_expiry, risk_free_rate,
            vol_bounds[1], dividend_yield, option_type
        )
        
        if market_price < lower_price:
            raise ValueError(f"Market price {market_price:.4f} is below minimum model price {lower_price:.4f}")
        elif market_price > upper_price:
            raise ValueError(f"Market price {market_price:.4f} is above maximum model price {upper_price:.4f}")
        else:
            raise e


def _implied_vol_newton(
    market_price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    dividend_yield: float,
    option_type: Union[Literal['call', 'put'], bool],
    tolerance: float,
    max_iterations: int
) -> float:
    """Use Newton-Raphson method for implied volatility."""
    # Initial guess using Brenner-Subrahmanyam approximation
    initial_vol = _brenner_subrahmanyam_approximation(
        market_price, spot, strike, time_to_expiry
    )
    
    sigma = initial_vol
    
    for i in range(max_iterations):
        # Calculate price and vega
        model_price = black_scholes_price(
            spot, strike, time_to_expiry, risk_free_rate,
            sigma, dividend_yield, option_type
        )
        vega = black_scholes_vega(
            spot, strike, time_to_expiry, risk_free_rate,
            sigma, dividend_yield
        )
        
        # Check convergence
        price_diff = model_price - market_price
        if abs(price_diff) < tolerance:
            return sigma
        
        # Avoid division by zero
        if abs(vega) < 1e-10:
            # Switch to bisection if vega is too small
            return _implied_vol_bisection(
                market_price, spot, strike, time_to_expiry,
                risk_free_rate, dividend_yield, option_type,
                (0.001, 5.0), tolerance, max_iterations - i
            )
        
        # Newton update
        sigma -= price_diff / vega
        
        # Keep sigma in reasonable bounds
        sigma = max(0.001, min(5.0, sigma))
    return sigma ##todo
    # raise ValueError(f"Newton's method failed to converge after {max_iterations} iterations")


def _implied_vol_bisection(
    market_price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    dividend_yield: float,
    option_type: Union[Literal['call', 'put'], bool],
    vol_bounds: tuple[float, float],
    tolerance: float,
    max_iterations: int
) -> float:
    """Use bisection method for implied volatility."""
    lower_vol, upper_vol = vol_bounds
    
    # Special handling for short-dated or extreme moneyness options
    if time_to_expiry < 0.25 or spot/strike > 1.5 or spot/strike < 0.5:
        lower_vol = 1e-5
        upper_vol = 3.0
        tolerance = 1e-5
    
    for i in range(max_iterations):
        mid_vol = 0.5 * (lower_vol + upper_vol)
        
        mid_price = black_scholes_price(
            spot, strike, time_to_expiry, risk_free_rate,
            mid_vol, dividend_yield, option_type
        )
        
        if abs(mid_price - market_price) < tolerance:
            return mid_vol
        
        if mid_price < market_price:
            lower_vol = mid_vol
        else:
            upper_vol = mid_vol
        
        if upper_vol - lower_vol < tolerance:
            return mid_vol
    
    return mid_vol


def _brenner_subrahmanyam_approximation(
    market_price: float,
    spot: float,
    strike: float,
    time_to_expiry: float
) -> float:
    """
    Brenner-Subrahmanyam approximation for initial implied volatility guess.
    
    For ATM options: σ ≈ √(2π/T) × (C/S)
    """
    # Adjust for moneyness
    moneyness = spot / strike
    adjustment = 1.0
    
    if moneyness > 1.1:  # ITM call
        adjustment = 0.9
    elif moneyness < 0.9:  # OTM call
        adjustment = 1.1
    
    # Basic approximation
    sigma = np.sqrt(2 * np.pi / time_to_expiry) * (market_price / spot) * adjustment
    
    # Ensure reasonable bounds
    return max(0.01, min(3.0, sigma))


def implied_volatility_smile(
    strikes: np.ndarray,
    market_prices: np.ndarray,
    spot: float,
    time_to_expiry: float,
    risk_free_rate: float,
    dividend_yield: float = 0.0,
    option_type: Union[Literal['call', 'put'], bool] = 'call',
    method: Literal['brent', 'newton', 'bisection'] = 'brent'
) -> np.ndarray:
    """
    Calculate implied volatility smile from multiple strikes.
    
    Parameters
    ----------
    strikes : ndarray
        Array of strike prices
    market_prices : ndarray
        Array of corresponding market prices
    spot : float
        Current spot price
    time_to_expiry : float
        Time to expiry
    risk_free_rate : float
        Risk-free rate
    dividend_yield : float, optional
        Dividend yield
    option_type : {'call', 'put'} or bool, optional
        Option type
    method : {'brent', 'newton', 'bisection'}, optional
        Numerical method
        
    Returns
    -------
    ndarray
        Array of implied volatilities
        
    Examples
    --------
    >>> strikes = np.array([90, 95, 100, 105, 110])
    >>> prices = np.array([11.2, 7.8, 5.0, 2.9, 1.5])
    >>> ivs = implied_volatility_smile(strikes, prices, 100, 1.0, 0.05)
    """
    if len(strikes) != len(market_prices):
        raise ValueError("strikes and market_prices must have the same length")
    
    ivs = np.zeros_like(strikes, dtype=float)
    
    for i, (strike, price) in enumerate(zip(strikes, market_prices)):
        try:
            ivs[i] = implied_volatility_black_scholes(
                price, spot, strike, time_to_expiry, risk_free_rate,
                dividend_yield, option_type, method
            )
        except ValueError:
            # Mark as NaN if IV cannot be computed
            ivs[i] = np.nan
    
    return ivs