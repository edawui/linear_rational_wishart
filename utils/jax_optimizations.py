"""
JAX optimization utilities for LRW Jump model calibration.

This module provides JAX-optimized utility functions for improved
performance in calibration routines.
"""

import jax.numpy as jnp
from jax import jit, vmap
from typing import Union
import numpy as np


@jit
def jax_log(x: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
    """
    JAX-compatible logarithm with safe handling.
    
    Parameters
    ----------
    x : float or jnp.ndarray
        Input value(s)
        
    Returns
    -------
    float or jnp.ndarray
        Log of input with protection against negative/zero values
    """
    return jnp.log(jnp.maximum(x, 1e-15))


@jit  
def jax_exp(x: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
    """
    JAX-compatible exponential with overflow protection.
    
    Parameters
    ----------
    x : float or jnp.ndarray
        Input value(s)
        
    Returns
    -------
    float or jnp.ndarray
        Exponential of input with overflow protection
    """
    return jnp.exp(jnp.minimum(x, 700.0))


@jit
def jax_sqrt(x: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
    """
    JAX-compatible square root with safe handling.
    
    Parameters
    ----------
    x : float or jnp.ndarray
        Input value(s)
        
    Returns
    -------
    float or jnp.ndarray
        Square root of input with protection against negative values
    """
    return jnp.sqrt(jnp.maximum(x, 1e-15))


@jit
def jax_power(
    x: Union[float, jnp.ndarray],
    power: float
) -> Union[float, jnp.ndarray]:
    """
    JAX-compatible power function with safe handling.
    
    Parameters
    ----------
    x : float or jnp.ndarray
        Base value(s)
    power : float
        Exponent
        
    Returns
    -------
    float or jnp.ndarray
        x raised to power with proper sign handling
    """
    abs_x = jnp.maximum(jnp.abs(x), 1e-15)
    return jnp.power(abs_x, power) * jnp.sign(x)**power


@jit
def compute_rmse(errors: Union[np.ndarray, jnp.ndarray]) -> float:
    """
    JAX-compatible RMSE calculation.
    
    Parameters
    ----------
    errors : array-like
        Error values
        
    Returns
    -------
    float
        Root mean square error
    """
    return jax_sqrt(jnp.mean(jnp.square(errors)))


@jit
def bond_price_from_rate(
    rate: Union[float, jnp.ndarray],
    time_to_mat: Union[float, jnp.ndarray]
) -> Union[float, jnp.ndarray]:
    """
    JAX-compatible bond price calculation from rate.
    
    Parameters
    ----------
    rate : float or jnp.ndarray
        Interest rate(s)
    time_to_mat : float or jnp.ndarray
        Time to maturity
        
    Returns
    -------
    float or jnp.ndarray
        Bond price(s)
    """
    return jax_exp(-rate * time_to_mat)


@jit
def rate_from_bond_price(
    bond_price: Union[float, jnp.ndarray],
    time_to_mat: Union[float, jnp.ndarray]
) -> Union[float, jnp.ndarray]:
    """
    JAX-compatible rate calculation from bond price.
    
    Parameters
    ----------
    bond_price : float or jnp.ndarray
        Bond price(s)
    time_to_mat : float or jnp.ndarray
        Time to maturity
        
    Returns
    -------
    float or jnp.ndarray
        Interest rate(s)
    """
    return -jax_log(bond_price) / time_to_mat


@jit
def compute_forward_rate(
    bond_price_t1: Union[float, jnp.ndarray],
    bond_price_t2: Union[float, jnp.ndarray],
    t1: float,
    t2: float
) -> Union[float, jnp.ndarray]:
    """
    Compute forward rate between two times.
    
    Parameters
    ----------
    bond_price_t1 : float or jnp.ndarray
        Bond price at time t1
    bond_price_t2 : float or jnp.ndarray
        Bond price at time t2
    t1 : float
        First time
    t2 : float
        Second time (> t1)
        
    Returns
    -------
    float or jnp.ndarray
        Forward rate between t1 and t2
    """
    return -jax_log(bond_price_t2 / bond_price_t1) / (t2 - t1)


@jit
def compute_swap_rate(
    bond_prices: jnp.ndarray,
    payment_times: jnp.ndarray,
    float_delta: float,
    fixed_delta: float
) -> float:
    """
    Compute swap rate from bond prices.
    
    Parameters
    ----------
    bond_prices : jnp.ndarray
        Array of bond prices at payment times
    payment_times : jnp.ndarray
        Payment times
    float_delta : float
        Floating leg payment frequency
    fixed_delta : float
        Fixed leg payment frequency
        
    Returns
    -------
    float
        Swap rate
    """
    # Floating leg value
    floating_leg = 1.0 - bond_prices[-1]
    
    # Fixed leg annuity
    fixed_leg_annuity = jnp.sum(bond_prices[:-1]) * fixed_delta
    
    return floating_leg / fixed_leg_annuity


# Vectorized versions of key functions
vmap_bond_price_from_rate = vmap(bond_price_from_rate)
vmap_rate_from_bond_price = vmap(rate_from_bond_price)


@jit
def batch_compute_rmse(
    errors_batch: jnp.ndarray,
    axis: int = -1
) -> jnp.ndarray:
    """
    Compute RMSE for batches of errors.
    
    Parameters
    ----------
    errors_batch : jnp.ndarray
        Batch of error arrays
    axis : int, default=-1
        Axis along which to compute RMSE
        
    Returns
    -------
    jnp.ndarray
        RMSE values for each batch
    """
    return jax_sqrt(jnp.mean(jnp.square(errors_batch), axis=axis))


@jit
def safe_divide(
    numerator: Union[float, jnp.ndarray],
    denominator: Union[float, jnp.ndarray],
    min_denominator: float = 1e-15
) -> Union[float, jnp.ndarray]:
    """
    Safe division with protection against division by zero.
    
    Parameters
    ----------
    numerator : float or jnp.ndarray
        Numerator
    denominator : float or jnp.ndarray
        Denominator
    min_denominator : float, default=1e-15
        Minimum denominator value
        
    Returns
    -------
    float or jnp.ndarray
        Result of division
    """
    safe_denominator = jnp.where(
        jnp.abs(denominator) < min_denominator,
        jnp.sign(denominator) * min_denominator,
        denominator
    )
    return numerator / safe_denominator
