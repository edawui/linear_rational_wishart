"""
Black-Scholes option pricing model.

This module provides implementations of the Black-Scholes model for
option pricing, including pricing formulas, Greeks calculations, and
implied volatility methods.
"""

from turtle import forward
from typing import Union, Literal, Optional, Tuple, List
import numpy as np
import jax.numpy as jnp
from scipy.stats import norm as scipy_norm
from scipy.optimize import brentq

from ..utils.jax_utils import is_jax_available, ensure_jax_array

# Constants
SQRT_TWO_PI = np.sqrt(2 * np.pi)
ONE_OVER_SQRT_TWO_PI = 1.0 / np.sqrt(2 * np.pi)
DBL_MIN = 1e-9
DBL_MAX = 1e9

def black_scholes_price(
    spot: Union[float, np.ndarray, jnp.ndarray],
    strike: Union[float, np.ndarray, jnp.ndarray],
    time_to_expiry: Union[float, np.ndarray, jnp.ndarray],
    risk_free_rate: Union[float, np.ndarray, jnp.ndarray],
    volatility: Union[float, np.ndarray, jnp.ndarray],
    dividend_yield: Union[float, np.ndarray, jnp.ndarray] = 0.0,
    option_type: Union[Literal['call', 'put'], bool] = 'call',
    use_jax: bool = False
) -> Union[float, np.ndarray, jnp.ndarray]:
    """
    Calculate option price using the Black-Scholes model.
    
    The Black-Scholes model assumes that the underlying asset follows a
    geometric Brownian motion with constant volatility and drift.
    
    Parameters
    ----------
    spot : float or array-like
        Current spot price of the underlying asset
    strike : float or array-like
        Strike price of the option
    time_to_expiry : float or array-like
        Time to expiry in years
    risk_free_rate : float or array-like
        Risk-free interest rate
    volatility : float or array-like
        Volatility (standard deviation of returns)
    dividend_yield : float or array-like, optional
        Continuous dividend yield, default is 0.0
    option_type : {'call', 'put'} or bool, optional
        Option type: 'call', 'put', or True for call, False for put
    use_jax : bool, optional
        Whether to use JAX for computation, default is False
        
    Returns
    -------
    float or ndarray
        Option price(s)
        
    Examples
    --------
    >>> spot, strike, T, r, sigma = 100.0, 105.0, 1.0, 0.05, 0.2
    >>> price = black_scholes_price(spot, strike, T, r, sigma, option_type='call')
    >>> print(f"Call price: {price:.4f}")
    Call price: 6.0400
    
    Notes
    -----
    The Black-Scholes formula for a call option is:
    C = S₀e^(-qT)N(d₁) - Ke^(-rT)N(d₂)
    where:
    d₁ = [ln(S₀/K) + (r - q + σ²/2)T] / (σ√T)
    d₂ = d₁ - σ√T
    """
    # Convert option type to boolean
    if isinstance(option_type, str):
        is_call = option_type.lower() == 'call'
    else:
        is_call = bool(option_type)
    
    if use_jax and is_jax_available():
        from .jax_implementations import black_scholes_price_jax
        return black_scholes_price_jax(
            spot, strike, time_to_expiry, risk_free_rate, 
            volatility, dividend_yield, is_call
        )
    
    # NumPy implementation
    spot = np.asarray(spot)
    strike = np.asarray(strike)
    time_to_expiry = np.asarray(time_to_expiry)
    risk_free_rate = np.asarray(risk_free_rate)
    volatility = np.asarray(volatility)
    dividend_yield = np.asarray(dividend_yield)
    # print(f"time_to_expiry ={time_to_expiry}")
    sqrt_t = np.sqrt(time_to_expiry)
    
    # Calculate d1 and d2
    d1 = (np.log(spot / strike) + 
          (risk_free_rate - dividend_yield + 0.5 * volatility**2) * time_to_expiry) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t
    
    # Discount factors
    exp_div_t = np.exp(-dividend_yield * time_to_expiry)
    exp_r_t = np.exp(-risk_free_rate * time_to_expiry)
    
    if is_call:
        price = spot * exp_div_t * scipy_norm.cdf(d1) - strike * exp_r_t * scipy_norm.cdf(d2)
    else:
        price = strike * exp_r_t * scipy_norm.cdf(-d2) - spot * exp_div_t * scipy_norm.cdf(-d1)
    
    return price

# def implied_vol_black_scholes_newton(
def implied_vol_black_scholes(#_newton(
    forward: float,
    strike: float,
    time_to_expiry: float,
    price: float,
    call_or_put: float,    
    risk_free_rate: float = 0.0,
    epsilon: float= 1e-9,
    max_iterations: int = 100
) -> float:
    """
    Newton-Raphson method for Black-Scholes implied volatility.
    
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
    epsilon : float
        Convergence tolerance
    risk_free_rate : float
        Risk-free rate (default 0.0 for forward price)
    max_iterations : int
        Maximum number of iterations
        
    Returns
    -------
    float
        Implied volatility
    """
    # Initial guess based on Brenner-Subrahmanyam approximation for Black-Scholes
    intrinsic = max(call_or_put * (forward - strike), 0.0)
    time_value = price - intrinsic
    
    if time_value <= 0:
        return 0.0
    
    # Initial volatility guess - adjusted for Black-Scholes
    # Using at-the-money approximation
    sigma = np.sqrt(2 * np.pi / time_to_expiry) * time_value / forward
    
    # Ensure reasonable initial guess
    sigma = max(0.1, min(sigma, 3.0))
    
    sqrt_t = np.sqrt(time_to_expiry)
    
    for i in range(max_iterations):
        # Calculate option value
        model_price = black_scholes_price(
            forward, strike, time_to_expiry, sigma, call_or_put, risk_free_rate
        )
        
        # Vega for Black-Scholes model
        d1 = (np.log(forward / strike) + 0.5 * sigma * sigma * time_to_expiry) / (sigma * sqrt_t)
        vega = forward * sqrt_t * scipy_norm.pdf(d1) * np.exp(-risk_free_rate * time_to_expiry)
        
        # Newton-Raphson update
        diff = model_price - price
        if abs(diff) < epsilon:
            break
        
        if vega < DBL_MIN:
            break
            
        sigma -= diff / vega
        
        # Ensure sigma stays positive and reasonable
        sigma = max(DBL_MIN, min(sigma, 10.0))
    
    return sigma

def implied_volatility_fx_to_be_removed(
            fx_spot:float,
            r_d:float,
            r_f:float,
            strike:float,
            expiry_maturity:float,
            model_price:float,
            call_or_put:float,
            initial_guess:float =0.0
        ):
     forward = fx_spot * np.exp((r_d - r_f) * expiry_maturity)

     implied_vol= implied_vol_black_scholes( 
                              forward=forward,
                              strike=strike,
                              time_to_expiry=expiry_maturity,
                              price=model_price,
                              call_or_put=call_or_put,    
                              risk_free_rate=r_d )
      
     return implied_vol

def black_scholes_price_forward(
    forward: Union[float, np.ndarray, jnp.ndarray],
    strike: Union[float, np.ndarray, jnp.ndarray],
    time_to_expiry: Union[float, np.ndarray, jnp.ndarray],
    risk_free_rate: Union[float, np.ndarray, jnp.ndarray],
    volatility: Union[float, np.ndarray, jnp.ndarray],
    option_type: Union[Literal['call', 'put'], bool] = 'call',
    use_jax: bool = False
) -> Union[float, np.ndarray, jnp.ndarray]:
    """
    Calculate option price using forward price in the Black-Scholes model.
    
    Parameters
    ----------
    forward : float or array-like
        Forward price of the underlying asset
    strike : float or array-like
        Strike price of the option
    time_to_expiry : float or array-like
        Time to expiry in years
    risk_free_rate : float or array-like
        Risk-free interest rate
    volatility : float or array-like
        Volatility
    option_type : {'call', 'put'} or bool, optional
        Option type
    use_jax : bool, optional
        Whether to use JAX for computation
        
    Returns
    -------
    float or ndarray
        Option price(s)
        
    Examples
    --------
    >>> forward = 105.127  # Forward price
    >>> strike, T, r, sigma = 105.0, 1.0, 0.05, 0.2
    >>> price = black_scholes_price_forward(forward, strike, T, r, sigma, 'call')
    """
    # Convert forward to spot
    discount_factor = np.exp(-risk_free_rate * time_to_expiry)
    spot = forward / discount_factor
    
    return black_scholes_price(
        spot, strike, time_to_expiry, risk_free_rate, 
        volatility, 0.0, option_type, use_jax
    )


def black_scholes_price_fx(
    spot: Union[float, np.ndarray],
    strike: Union[float, np.ndarray],
    time_to_expiry: Union[float, np.ndarray],
    domestic_rate: Union[float, np.ndarray],
    foreign_rate: Union[float, np.ndarray],
    volatility: Union[float, np.ndarray],
    option_type: Union[Literal['call', 'put'], bool] = 'call'
) -> Union[float, np.ndarray]:
    """
    Calculate FX option price using the Black-Scholes model.
    
    For FX options, the foreign interest rate acts like a dividend yield.
    
    Parameters
    ----------
    spot : float or array-like
        Current spot FX rate (domestic per foreign)
    strike : float or array-like
        Strike price
    time_to_expiry : float or array-like
        Time to expiry in years
    domestic_rate : float or array-like
        Domestic risk-free rate
    foreign_rate : float or array-like
        Foreign risk-free rate
    volatility : float or array-like
        FX volatility
    option_type : {'call', 'put'} or bool, optional
        Option type
        
    Returns
    -------
    float or ndarray
        FX option price(s) in domestic currency
        
    Examples
    --------
    >>> spot = 1.2500  # USD/EUR
    >>> strike = 1.2600
    >>> T = 0.25  # 3 months
    >>> r_usd, r_eur = 0.02, 0.01
    >>> sigma = 0.10
    >>> price = black_scholes_price_fx(spot, strike, T, r_usd, r_eur, sigma, 'call')
    """
    return black_scholes_price(
        spot, strike, time_to_expiry, domestic_rate, 
        volatility, foreign_rate, option_type, False
    )


def black_scholes_delta(
    spot: Union[float, np.ndarray],
    strike: Union[float, np.ndarray],
    time_to_expiry: Union[float, np.ndarray],
    risk_free_rate: Union[float, np.ndarray],
    volatility: Union[float, np.ndarray],
    dividend_yield: Union[float, np.ndarray] = 0.0,
    option_type: Union[Literal['call', 'put'], bool] = 'call',
    use_jax: bool = False
) -> Union[float, np.ndarray]:
    """
    Calculate option delta in the Black-Scholes model.
    
    Delta measures the rate of change of option price with respect to
    changes in the underlying asset price.
    
    Parameters
    ----------
    spot : float or array-like
        Current spot price
    strike : float or array-like
        Strike price
    time_to_expiry : float or array-like
        Time to expiry
    risk_free_rate : float or array-like
        Risk-free rate
    volatility : float or array-like
        Volatility
    dividend_yield : float or array-like, optional
        Dividend yield
    option_type : {'call', 'put'} or bool, optional
        Option type
    use_jax : bool, optional
        Whether to use JAX
        
    Returns
    -------
    float or ndarray
        Option delta
        
    Examples
    --------
    >>> delta = black_scholes_delta(100, 105, 1.0, 0.05, 0.2, option_type='call')
    >>> print(f"Delta: {delta:.4f}")
    Delta: 0.4502
    """
    # Convert option type to boolean
    if isinstance(option_type, str):
        is_call = option_type.lower() == 'call'
    else:
        is_call = bool(option_type)
    
    if use_jax and is_jax_available():
        from .jax_implementations import black_scholes_delta_jax
        return black_scholes_delta_jax(
            spot, strike, time_to_expiry, risk_free_rate,
            volatility, dividend_yield, is_call
        )
    
    # NumPy implementation
    sqrt_t = np.sqrt(time_to_expiry)
    d1 = (np.log(spot / strike) + 
          (risk_free_rate - dividend_yield + 0.5 * volatility**2) * time_to_expiry) / (volatility * sqrt_t)
    
    exp_div_t = np.exp(-dividend_yield * time_to_expiry)
    
    if is_call:
        delta = exp_div_t * scipy_norm.cdf(d1)
    else:
        delta = -exp_div_t * scipy_norm.cdf(-d1)
    
    return delta


def black_scholes_vega(
    spot: Union[float, np.ndarray],
    strike: Union[float, np.ndarray],
    time_to_expiry: Union[float, np.ndarray],
    risk_free_rate: Union[float, np.ndarray],
    volatility: Union[float, np.ndarray],
    dividend_yield: Union[float, np.ndarray] = 0.0,
    use_jax: bool = False
) -> Union[float, np.ndarray]:
    """
    Calculate option vega in the Black-Scholes model.
    
    Vega measures the sensitivity of option price to changes in volatility.
    Note: Vega is the same for calls and puts.
    
    Parameters
    ----------
    spot : float or array-like
        Current spot price
    strike : float or array-like
        Strike price
    time_to_expiry : float or array-like
        Time to expiry
    risk_free_rate : float or array-like
        Risk-free rate
    volatility : float or array-like
        Volatility
    dividend_yield : float or array-like, optional
        Dividend yield
    use_jax : bool, optional
        Whether to use JAX
        
    Returns
    -------
    float or ndarray
        Option vega
        
    Examples
    --------
    >>> vega = black_scholes_vega(100, 105, 1.0, 0.05, 0.2)
    >>> print(f"Vega: {vega:.4f}")
    Vega: 37.5248
    """
    if use_jax and is_jax_available():
        from .jax_implementations import black_scholes_vega_jax
        return black_scholes_vega_jax(
            spot, strike, time_to_expiry, risk_free_rate,
            volatility, dividend_yield
        )
    
    # NumPy implementation
    sqrt_t = np.sqrt(time_to_expiry)
    d1 = (np.log(spot / strike) + 
          (risk_free_rate - dividend_yield + 0.5 * volatility**2) * time_to_expiry) / (volatility * sqrt_t)
    
    vega = spot * np.exp(-dividend_yield * time_to_expiry) * scipy_norm.pdf(d1) * sqrt_t
    
    return vega


def black_scholes_vanna(
    spot: Union[float, np.ndarray],
    strike: Union[float, np.ndarray],
    time_to_expiry: Union[float, np.ndarray],
    risk_free_rate: Union[float, np.ndarray],
    volatility: Union[float, np.ndarray],
    dividend_yield: Union[float, np.ndarray] = 0.0,
    use_jax: bool = False
) -> Union[float, np.ndarray]:
    """
    Calculate option vanna in the Black-Scholes model.
    
    Vanna measures the sensitivity of delta to changes in volatility,
    or equivalently, the sensitivity of vega to changes in spot price.
    
    Parameters
    ----------
    spot : float or array-like
        Current spot price
    strike : float or array-like
        Strike price
    time_to_expiry : float or array-like
        Time to expiry
    risk_free_rate : float or array-like
        Risk-free rate
    volatility : float or array-like
        Volatility
    dividend_yield : float or array-like, optional
        Dividend yield
    use_jax : bool, optional
        Whether to use JAX
        
    Returns
    -------
    float or ndarray
        Option vanna
        
    Examples
    --------
    >>> vanna = black_scholes_vanna(100, 105, 1.0, 0.05, 0.2)
    >>> print(f"Vanna: {vanna:.6f}")
    Vanna: -0.356486
    """
    if use_jax and is_jax_available():
        from .jax_implementations import black_scholes_vanna_jax
        return black_scholes_vanna_jax(
            spot, strike, time_to_expiry, risk_free_rate,
            volatility, dividend_yield
        )
    
    # NumPy implementation
    sqrt_t = np.sqrt(time_to_expiry)
    d1 = (np.log(spot / strike) + 
          (risk_free_rate - dividend_yield + 0.5 * volatility**2) * time_to_expiry) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t
    
    vanna = -np.exp(-dividend_yield * time_to_expiry) * scipy_norm.pdf(d1) * (d2 / volatility)
    
    return vanna


def black_scholes_greeks(
    spot: Union[float, np.ndarray],
    strike: Union[float, np.ndarray],
    time_to_expiry: Union[float, np.ndarray],
    risk_free_rate: Union[float, np.ndarray],
    volatility: Union[float, np.ndarray],
    dividend_yield: Union[float, np.ndarray] = 0.0,
    option_type: Union[Literal['call', 'put'], bool] = 'call',
    use_jax: bool = False
) -> Tuple[Union[float, np.ndarray], ...]:
    """
    Calculate all Greeks in a single call for efficiency.
    
    Parameters
    ----------
    spot : float or array-like
        Current spot price
    strike : float or array-like
        Strike price
    time_to_expiry : float or array-like
        Time to expiry
    risk_free_rate : float or array-like
        Risk-free rate
    volatility : float or array-like
        Volatility
    dividend_yield : float or array-like, optional
        Dividend yield
    option_type : {'call', 'put'} or bool, optional
        Option type
    use_jax : bool, optional
        Whether to use JAX
        
    Returns
    -------
    tuple
        (price, delta, vega, vanna)
        
    Examples
    --------
    >>> greeks = black_scholes_greeks(100, 105, 1.0, 0.05, 0.2, option_type='call')
    >>> price, delta, vega, vanna = greeks
    >>> print(f"Price: {price:.4f}, Delta: {delta:.4f}, Vega: {vega:.4f}")
    """
    if use_jax and is_jax_available():
        from .jax_implementations import black_scholes_greeks_batch_jax
        return black_scholes_greeks_batch_jax(
            spot, strike, time_to_expiry, risk_free_rate,
            volatility, dividend_yield, option_type
        )
    
    # Calculate all Greeks
    price = black_scholes_price(
        spot, strike, time_to_expiry, risk_free_rate,
        volatility, dividend_yield, option_type, use_jax
    )
    delta = black_scholes_delta(
        spot, strike, time_to_expiry, risk_free_rate,
        volatility, dividend_yield, option_type, use_jax
    )
    vega = black_scholes_vega(
        spot, strike, time_to_expiry, risk_free_rate,
        volatility, dividend_yield, use_jax
    )
    vanna = black_scholes_vanna(
        spot, strike, time_to_expiry, risk_free_rate,
        volatility, dividend_yield, use_jax
    )
    
    return price, delta, vega, vanna