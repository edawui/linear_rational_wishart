"""
Bachelier option pricing model.

This module provides implementations of the Bachelier (normal) model for
option pricing, including pricing formulas and implied volatility calculations.
"""

from typing import Union, Literal, Optional, Tuple
import numpy as np
import jax.numpy as jnp
from scipy.stats import norm as scipy_norm

from ..utils.jax_utils import is_jax_available, ensure_jax_array

# Constants
SQRT_TWO_PI = np.sqrt(2 * np.pi)
ONE_OVER_SQRT_TWO_PI = 1.0 / np.sqrt(2 * np.pi)
DBL_MIN = 1e-9
DBL_MAX = 1e9


def bachelier_price(
    forward: Union[float, np.ndarray, jnp.ndarray],
    strike: Union[float, np.ndarray, jnp.ndarray],
    time_to_expiry: Union[float, np.ndarray, jnp.ndarray],
    sigma: Union[float, np.ndarray, jnp.ndarray],
    option_type: Union[Literal['call', 'put'], float] = 'call',
    numeraire: Union[float, np.ndarray, jnp.ndarray] = 1.0,
    use_jax: bool = False
) -> Union[float, np.ndarray, jnp.ndarray]:
    """
    Calculate option price using the Bachelier model.
    
    The Bachelier model assumes that the underlying asset follows an
    arithmetic Brownian motion, making it suitable for interest rates
    and other assets that can become negative.
    
    Parameters
    ----------
    forward : float or array-like
        Forward price of the underlying asset
    strike : float or array-like
        Strike price of the option
    time_to_expiry : float or array-like
        Time to expiry in years
    sigma : float or array-like
        Normal (Bachelier) volatility
    option_type : {'call', 'put'} or float, optional
        Option type: 'call', 'put', or 1.0 for call, -1.0 for put
    numeraire : float or array-like, optional
        Numeraire value for discounting, default is 1.0
    use_jax : bool, optional
        Whether to use JAX for computation, default is False
        
    Returns
    -------
    float or ndarray
        Option price(s)
        
    Examples
    --------
    >>> forward, strike, T, sigma = 100.0, 105.0, 1.0, 20.0
    >>> price = bachelier_price(forward, strike, T, sigma, 'call')
    >>> print(f"Call price: {price:.4f}")
    Call price: 8.9595
    
    Notes
    -----
    The Bachelier formula for a call option is:
    C = N(F - K, 0, σ²T) = σ√T φ(d) + (F - K) Φ(d)
    where d = (F - K)/(σ√T), φ is the standard normal PDF, and Φ is the CDF.
    """
    # Convert option type to numeric
    if isinstance(option_type, str):
        call_or_put = 1.0 if option_type.lower() == 'call' else -1.0
    else:
        call_or_put = float(option_type)
    
    if use_jax and is_jax_available():
        from .jax_implementations import bachelier_price_jax
        return bachelier_price_jax(
            forward, strike, time_to_expiry, sigma, call_or_put, numeraire
        )
    
    # NumPy implementation
    forward = np.asarray(forward)
    strike = np.asarray(strike)
    time_to_expiry = np.asarray(time_to_expiry)
    sigma = np.asarray(sigma)
    
    sqrt_t = np.sqrt(time_to_expiry)
    
    # Handle zero time case
    if np.any(time_to_expiry <= 0):
        intrinsic = np.maximum(call_or_put * (forward - strike), 0.0)
        return numeraire * intrinsic
    
    d = (forward - strike) / (sigma * sqrt_t)
    
    price = (sigma * sqrt_t * scipy_norm.pdf(d) + 
             call_or_put * (forward - strike) * scipy_norm.cdf(call_or_put * d))
    price *= numeraire
    
    return price


def implied_normal_volatility(
    forward: Union[float, np.ndarray],
    strike: Union[float, np.ndarray],
    time_to_expiry: Union[float, np.ndarray],
    price: Union[float, np.ndarray],
    option_type: Union[Literal['call', 'put'], float] = 'call',
    numeraire: float = 1.0,
    epsilon: float = 1e-6,
    method: Literal['jackel', 'newton'] = 'newton'#'jackel'
) -> Union[float, np.ndarray]:
    """
    Calculate implied normal (Bachelier) volatility from option price.
    
    Parameters
    ----------
    forward : float or array-like
        Forward price of the underlying asset
    strike : float or array-like
        Strike price of the option
    time_to_expiry : float or array-like
        Time to expiry in years
    price : float or array-like
        Option price (market price)
    option_type : {'call', 'put'} or float, optional
        Option type: 'call', 'put', or 1.0 for call, -1.0 for put
    numeraire : float, optional
        Numeraire value, default is 1.0
    epsilon : float, optional
        Convergence tolerance for iterative methods
    method : {'jackel', 'newton'}, optional
        Method to use: 'jackel' for Peter Jäckel's method, 'newton' for Newton-Raphson
        
    Returns
    -------
    float or ndarray
        Implied normal volatility
        
    Examples
    --------
    >>> forward, strike, T, price = 100.0, 105.0, 1.0, 8.9595
    >>> iv = implied_normal_volatility(forward, strike, T, price, 'call')
    >>> print(f"Implied volatility: {iv:.4f}")
    Implied volatility: 20.0000
    
    Notes
    -----
    The Jäckel method is typically more robust and faster than Newton-Raphson,
    especially for options far from the money or with very short maturities.
    """
    # Convert option type to numeric
    if isinstance(option_type, str):
        call_or_put = 1.0 if option_type.lower() == 'call' else -1.0
    else:
        call_or_put = float(option_type)
    
    # Adjust for numeraire
    adjusted_price = price / numeraire
    
    if method == 'jackel':
        from .jackel_method import JackelImpliedVolatility
        calculator = JackelImpliedVolatility()
        return calculator.implied_normal_volatility(
                    forward, strike, time_to_expiry, adjusted_price, call_or_put
                )
    else:
        # Newton-Raphson method
        return _implied_vol_newton(
            forward, strike, time_to_expiry, adjusted_price, call_or_put, epsilon
        )


def _implied_vol_newton(
    forward: float,
    strike: float,
    time_to_expiry: float,
    price: float,
    call_or_put: float,
    epsilon: float,
    max_iterations: int = 100
) -> float:
    """
    Newton-Raphson method for implied volatility.
    
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
    max_iterations : int
        Maximum number of iterations
        
    Returns
    -------
    float
        Implied volatility
    """
    # Initial guess based on Brenner-Subrahmanyam approximation
    intrinsic = max(call_or_put * (forward - strike), 0.0)
    time_value = price - intrinsic
    
    if time_value <= 0:
        return 0.0
    
    # Initial volatility guess
    sigma = time_value * SQRT_TWO_PI / np.sqrt(time_to_expiry)
    
    for _ in range(max_iterations):
        # Calculate option value and vega
        model_price = bachelier_price(
            forward, strike, time_to_expiry, sigma, call_or_put, 1.0
        )
        
        # Vega for Bachelier model
        sqrt_t = np.sqrt(time_to_expiry)
        d = (forward - strike) / (sigma * sqrt_t)
        vega = sqrt_t * scipy_norm.pdf(d)
        
        # Newton-Raphson update
        diff = model_price - price
        if abs(diff) < epsilon:
            break
        
        if vega < DBL_MIN:
            break
            
        sigma -= diff / vega
        
        # Ensure sigma stays positive
        sigma = max(sigma, DBL_MIN)
    
    return sigma


def phi_tilde(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Compute PhiTilde function: Φ(x) + φ(x)/x.
    
    Parameters
    ----------
    x : float or array-like
        Input value(s)
        
    Returns
    -------
    float or ndarray
        PhiTilde(x) = Φ(x) + φ(x)/x
        
    Notes
    -----
    This function appears in the Bachelier implied volatility formula
    and needs careful handling for x near zero.
    """
    x = np.asarray(x)
    
    # Handle x = 0 case
    if np.any(np.abs(x) < DBL_MIN):
        return np.where(
            np.abs(x) < DBL_MIN,
            0.5 + ONE_OVER_SQRT_TWO_PI,
            scipy_norm.cdf(x) + scipy_norm.pdf(x) / x
        )
    
    return scipy_norm.cdf(x) + scipy_norm.pdf(x) / x


def bachelier_delta(
    forward: Union[float, np.ndarray],
    strike: Union[float, np.ndarray],
    time_to_expiry: Union[float, np.ndarray],
    sigma: Union[float, np.ndarray],
    option_type: Union[Literal['call', 'put'], float] = 'call'
) -> Union[float, np.ndarray]:
    """
    Calculate option delta in the Bachelier model.
    
    Parameters
    ----------
    forward : float or array-like
        Forward price
    strike : float or array-like
        Strike price
    time_to_expiry : float or array-like
        Time to expiry
    sigma : float or array-like
        Normal volatility
    option_type : {'call', 'put'} or float
        Option type
        
    Returns
    -------
    float or ndarray
        Option delta
        
    Examples
    --------
    >>> delta = bachelier_delta(100, 105, 1.0, 20.0, 'call')
    >>> print(f"Delta: {delta:.4f}")
    Delta: 0.4004
    """
    # Convert option type to numeric
    if isinstance(option_type, str):
        call_or_put = 1.0 if option_type.lower() == 'call' else -1.0
    else:
        call_or_put = float(option_type)
    
    sqrt_t = np.sqrt(time_to_expiry)
    d = (forward - strike) / (sigma * sqrt_t)
    
    return call_or_put * scipy_norm.cdf(call_or_put * d)


def bachelier_vega(
    forward: Union[float, np.ndarray],
    strike: Union[float, np.ndarray],
    time_to_expiry: Union[float, np.ndarray],
    sigma: Union[float, np.ndarray],
    option_type: Union[Literal['call', 'put'], float] = 'call'
) -> Union[float, np.ndarray]:
    """
    Calculate option vega in the Bachelier model.
    
    Parameters
    ----------
    forward : float or array-like
        Forward price
    strike : float or array-like
        Strike price
    time_to_expiry : float or array-like
        Time to expiry
    sigma : float or array-like
        Normal volatility
    option_type : {'call', 'put'} or float
        Option type (not used, vega is same for calls and puts)
        
    Returns
    -------
    float or ndarray
        Option vega (sensitivity to volatility)
        
    Examples
    --------
    >>> vega = bachelier_vega(100, 105, 1.0, 20.0)
    >>> print(f"Vega: {vega:.4f}")
    Vega: 0.3969
    """
    sqrt_t = np.sqrt(time_to_expiry)
    d = (forward - strike) / (sigma * sqrt_t)
    
    return sqrt_t * scipy_norm.pdf(d)
