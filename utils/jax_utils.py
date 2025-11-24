"""
JAX utility functions for the Wishart process package.

This module provides common utilities for working with JAX arrays and
optimizations throughout the package.
"""

from typing import Union, Any, Optional, Tuple
import numpy as np
import jax
import jax.numpy as jnp
from functools import wraps


def ensure_jax_array(
    x: Union[np.ndarray, jnp.ndarray, list, float, int],
    dtype: Optional[Any] = None
) -> jnp.ndarray:
    """
    Safely convert input to JAX array.
    
    Parameters
    ----------
    x : array-like
        Input data to convert
    dtype : dtype, optional
        Desired data type for the array
        
    Returns
    -------
    jnp.ndarray
        JAX array
        
    Examples
    --------
    >>> ensure_jax_array([1, 2, 3])
    DeviceArray([1, 2, 3], dtype=int32)
    >>> ensure_jax_array(np.array([1.0, 2.0]), dtype=jnp.float32)
    DeviceArray([1., 2.], dtype=float32)
    """
    if isinstance(x, jnp.ndarray):
        if dtype is not None and x.dtype != dtype:
            return x.astype(dtype)
        return x
    
    return jnp.asarray(x, dtype=dtype)


def jax_compatible_wrapper(func):
    """
    Decorator to make functions compatible with both NumPy and JAX arrays.
    
    This decorator ensures that functions can accept both NumPy and JAX
    arrays as input and return the appropriate type based on the input.
    
    Parameters
    ----------
    func : callable
        Function to wrap
        
    Returns
    -------
    callable
        Wrapped function
        
    Examples
    --------
    >>> @jax_compatible_wrapper
    ... def add_one(x):
    ...     return x + 1
    >>> add_one(np.array([1, 2, 3]))
    array([2, 3, 4])
    >>> add_one(jnp.array([1, 2, 3]))
    DeviceArray([2, 3, 4], dtype=int32)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check if any input is a JAX array
        use_jax = any(isinstance(arg, jnp.ndarray) for arg in args)
        use_jax = use_jax or any(isinstance(v, jnp.ndarray) for v in kwargs.values())
        
        if use_jax:
            # Convert all array inputs to JAX
            args = tuple(
                ensure_jax_array(arg) if isinstance(arg, (np.ndarray, list)) else arg
                for arg in args
            )
            kwargs = {
                k: ensure_jax_array(v) if isinstance(v, (np.ndarray, list)) else v
                for k, v in kwargs.items()
            }
        
        return func(*args, **kwargs)
    
    return wrapper


def validate_jax_inputs(
    *arrays: Union[np.ndarray, jnp.ndarray],
    require_same_shape: bool = False,
    require_1d: bool = False,
    min_length: Optional[int] = None
) -> Tuple[jnp.ndarray, ...]:
    """
    Validate and convert multiple arrays for JAX operations.
    
    Parameters
    ----------
    *arrays : array-like
        Arrays to validate and convert
    require_same_shape : bool, optional
        Whether all arrays must have the same shape
    require_1d : bool, optional
        Whether arrays must be 1-dimensional
    min_length : int, optional
        Minimum required length for arrays
        
    Returns
    -------
    tuple of jnp.ndarray
        Validated JAX arrays
        
    Raises
    ------
    ValueError
        If validation fails
        
    Examples
    --------
    >>> x, y = validate_jax_inputs([1, 2, 3], [4, 5, 6], require_same_shape=True)
    >>> x
    DeviceArray([1, 2, 3], dtype=int32)
    """
    if not arrays:
        raise ValueError("At least one array must be provided")
    
    # Convert all to JAX arrays
    jax_arrays = tuple(ensure_jax_array(arr) for arr in arrays)
    
    # Check 1D requirement
    if require_1d:
        for i, arr in enumerate(jax_arrays):
            if arr.ndim != 1:
                raise ValueError(f"Array {i} must be 1-dimensional, got {arr.ndim}D")
    
    # Check same shape requirement
    if require_same_shape and len(jax_arrays) > 1:
        first_shape = jax_arrays[0].shape
        for i, arr in enumerate(jax_arrays[1:], 1):
            if arr.shape != first_shape:
                raise ValueError(
                    f"Array {i} has shape {arr.shape}, expected {first_shape}"
                )
    
    # Check minimum length
    if min_length is not None:
        for i, arr in enumerate(jax_arrays):
            if len(arr) < min_length:
                raise ValueError(
                    f"Array {i} has length {len(arr)}, minimum required is {min_length}"
                )
    
    return jax_arrays


def is_jax_available() -> bool:
    """
    Check if JAX is available and properly configured.
    
    Returns
    -------
    bool
        True if JAX is available and functional
        
    Examples
    --------
    >>> is_jax_available()
    True
    """
    try:
        # Try to create a simple JAX array
        _ = jnp.array([1.0])
        return True
    except Exception:
        return False


def get_jax_device_info() -> dict:
    """
    Get information about available JAX devices.
    
    Returns
    -------
    dict
        Dictionary containing device information
        
    Examples
    --------
    >>> info = get_jax_device_info()
    >>> info['device_count']
    1
    >>> info['default_device']
    'cpu'
    """
    devices = jax.devices()
    default_device = jax.default_backend()
    
    return {
        'device_count': len(devices),
        'devices': [str(d) for d in devices],
        'default_device': default_device,
        'has_gpu': any('gpu' in str(d).lower() for d in devices),
        'has_tpu': any('tpu' in str(d).lower() for d in devices)
    }


def jit_when_available(func):
    """
    Conditionally apply JIT compilation if JAX is available.
    
    This decorator applies JAX JIT compilation to a function if JAX
    is available, otherwise returns the function unchanged.
    
    Parameters
    ----------
    func : callable
        Function to potentially JIT compile
        
    Returns
    -------
    callable
        JIT-compiled function if JAX is available, otherwise original function
        
    Examples
    --------
    >>> @jit_when_available
    ... def compute_sum(x):
    ...     return jnp.sum(x)
    """
    if is_jax_available():
        return jit(func)
    return func
