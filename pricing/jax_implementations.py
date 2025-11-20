"""
JAX-optimized implementations for pricing models.

This module contains high-performance JAX implementations of various
pricing algorithms.
"""

from typing import Union
import jax.numpy as jnp
from jax import jit
from jax.scipy.stats import norm

from .bachelier import SQRT_TWO_PI, ONE_OVER_SQRT_TWO_PI, DBL_MIN


@jit
def bachelier_price_jax(
    forward: Union[float, jnp.ndarray],
    strike: Union[float, jnp.ndarray],
    time_to_expiry: Union[float, jnp.ndarray],
    sigma: Union[float, jnp.ndarray],
    call_or_put: float = 1.0,
    numeraire: float = 1.0
) -> Union[float, jnp.ndarray]:
    """
    JAX-optimized Bachelier option pricing formula.
    
    Parameters
    ----------
    forward : float or jnp.ndarray
        Forward price
    strike : float or jnp.ndarray
        Strike price
    time_to_expiry : float or jnp.ndarray
        Time to expiry
    sigma : float or jnp.ndarray
        Normal volatility
    call_or_put : float
        1.0 for call, -1.0 for put
    numeraire : float
        Numeraire value
        
    Returns
    -------
    float or jnp.ndarray
        Option price(s)
    """
    sqrt_t = jnp.sqrt(time_to_expiry)
    d = (forward - strike) / (sigma * sqrt_t)
    
    price = (sigma * sqrt_t * norm.pdf(d) + 
             call_or_put * (forward - strike) * norm.cdf(call_or_put * d))
    price *= numeraire
    
    return price


@jit
def phi_tilde_jax(x: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
    """
    JAX-optimized PhiTilde function.
    
    Parameters
    ----------
    x : float or jnp.ndarray
        Input value(s)
        
    Returns
    -------
    float or jnp.ndarray
        PhiTilde(x) = Φ(x) + φ(x)/x
    """
    return norm.cdf(x) + norm.pdf(x) / x


@jit
def phi_tilde_times_x_jax(x: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
    """
    JAX-optimized PhiTildeTimesX function.
    
    Parameters
    ----------
    x : float or jnp.ndarray
        Input value(s)
        
    Returns
    -------
    float or jnp.ndarray
        x * Φ(x) + φ(x)
    """
    return x * norm.cdf(x) + norm.pdf(x)


@jit
def bachelier_delta_jax(
    forward: Union[float, jnp.ndarray],
    strike: Union[float, jnp.ndarray],
    time_to_expiry: Union[float, jnp.ndarray],
    sigma: Union[float, jnp.ndarray],
    call_or_put: float = 1.0
) -> Union[float, jnp.ndarray]:
    """
    JAX-optimized Bachelier delta calculation.
    
    Parameters
    ----------
    forward : float or jnp.ndarray
        Forward price
    strike : float or jnp.ndarray
        Strike price
    time_to_expiry : float or jnp.ndarray
        Time to expiry
    sigma : float or jnp.ndarray
        Normal volatility
    call_or_put : float
        1.0 for call, -1.0 for put
        
    Returns
    -------
    float or jnp.ndarray
        Option delta
    """
    sqrt_t = jnp.sqrt(time_to_expiry)
    d = (forward - strike) / (sigma * sqrt_t)
    
    return call_or_put * norm.cdf(call_or_put * d)


@jit
def bachelier_vega_jax(
    forward: Union[float, jnp.ndarray],
    strike: Union[float, jnp.ndarray],
    time_to_expiry: Union[float, jnp.ndarray],
    sigma: Union[float, jnp.ndarray]
) -> Union[float, jnp.ndarray]:
    """
    JAX-optimized Bachelier vega calculation.
    
    Parameters
    ----------
    forward : float or jnp.ndarray
        Forward price
    strike : float or jnp.ndarray
        Strike price
    time_to_expiry : float or jnp.ndarray
        Time to expiry
    sigma : float or jnp.ndarray
        Normal volatility
        
    Returns
    -------
    float or jnp.ndarray
        Option vega
    """
    sqrt_t = jnp.sqrt(time_to_expiry)
    d = (forward - strike) / (sigma * sqrt_t)
    
    return sqrt_t * norm.pdf(d)


@jit
def _inv_phi_tilde_region1_jax(phi_star_tilde: float) -> float:
    """
    JAX-optimized InvPhiTilde for region 1.
    
    Parameters
    ----------
    phi_star_tilde : float
        Target PhiTilde value (< -0.001882039271)
        
    Returns
    -------
    float
        Approximation x_bar
    """
    g = 1 / (phi_star_tilde - 0.5)
    g_square = g * g
    
    num = (0.032114372355 - 
           g_square * (0.016969777977 - 
                      g_square * (2.6207332461e-3 - 9.6066952861e-5 * g_square)))
    denom = (1 - 
             g_square * (0.6635646938 - 
                        g_square * (0.14528712196 - 0.010472855461 * g_square)))
    
    xi_bar = num / denom
    x_bar = g * (ONE_OVER_SQRT_TWO_PI + xi_bar * g_square)
    
    return x_bar


@jit 
def _inv_phi_tilde_region2_jax(phi_star_tilde: float) -> float:
    """
    JAX-optimized InvPhiTilde for region 2.
    
    Parameters
    ----------
    phi_star_tilde : float
        Target PhiTilde value
        
    Returns
    -------
    float
        Approximation x_bar
    """
    h = jnp.sqrt(-jnp.log(-phi_star_tilde))
    
    num = (9.4883409779 - 
           h * (9.6320903635 - 
                h * (0.58556997323 + 2.1464093351 * h)))
    denom = (1 - 
             h * (0.65174820867 + 
                  h * (1.5120247828 + 6.6437847132e-5 * h)))
    
    x_bar = num / denom
    return x_bar


@jit
def _refine_x_star_jax(x_bar: float, phi_star_tilde: float) -> float:
    """
    JAX-optimized refinement step for x_star.
    
    Parameters
    ----------
    x_bar : float
        Initial approximation
    phi_star_tilde : float
        Target PhiTilde value
        
    Returns
    -------
    float
        Refined x_star
    """
    q = (phi_tilde_jax(x_bar) - phi_star_tilde) / norm.pdf(x_bar)
    
    num_x = 3 * q * x_bar * x_bar * (2 - q * x_bar * (2 + x_bar * x_bar))
    denom_x = (6 + q * x_bar * (-12.0 + 
                                x_bar * (6 * q + 
                                        x_bar * (-6 + 
                                                q * x_bar * (3 + x_bar * x_bar)))))
    
    x_star = x_bar + num_x / denom_x
    return x_star


@jit
def intrinsic_value_jax(
    forward: Union[float, jnp.ndarray],
    strike: Union[float, jnp.ndarray],
    call_or_put: float
) -> Union[float, jnp.ndarray]:
    """
    JAX-optimized intrinsic value calculation.
    
    Parameters
    ----------
    forward : float or jnp.ndarray
        Forward price
    strike : float or jnp.ndarray
        Strike price
    call_or_put : float
        1.0 for call, -1.0 for put
        
    Returns
    -------
    float or jnp.ndarray
        Intrinsic value
    """
    return jnp.maximum(call_or_put * (forward - strike), 0.0)