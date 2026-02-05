"""
Configuration classes for LRW models.

models/interest_rate/config.py
"""
from dataclasses import dataclass
from typing import Optional
import jax.numpy as jnp


@dataclass
class LRWModelConfig:
    """
    Configuration for LRW interest rate model.
    
    Attributes
    ----------
    n : int
        Dimension of the Wishart process
    alpha : float
        Mean reversion level for short rate
    x0 : jnp.ndarray
        Initial value of Wishart process
    omega : jnp.ndarray
        Drift matrix
    m : jnp.ndarray
        Mean reversion matrix
    sigma : jnp.ndarray
        Volatility matrix
    is_bru_config : bool
        Whether to use Bru configuration
    use_range_kutta_for_b : bool
        Whether to use Range-Kutta for B computation
    compute_equity_style_vega : bool
        Whether to compute equity-style vega
    pseudo_inverse_smoothing : bool
        Whether to use pseudo-inverse smoothing
    """
    n: int
    alpha: float
    x0: jnp.ndarray
    omega: jnp.ndarray
    m: jnp.ndarray
    sigma: jnp.ndarray   
    is_bru_config: bool = False
    has_jump:bool=False
    # ##model u1 and u2
    u1: Optional[jnp.ndarray] = None
    u2: Optional[jnp.ndarray] = None
    is_spread: bool = False

    use_range_kutta_for_b: bool = True
    compute_equity_style_vega: bool = True
    pseudo_inverse_smoothing: bool = False
    

    def __post_init__(self):
        """Validate configuration."""
        # Convert to JAX arrays
        self.x0 = jnp.array(self.x0)
        self.omega = jnp.array(self.omega)
        self.m = jnp.array(self.m)
        self.sigma = jnp.array(self.sigma)
        
        # Validate dimensions
        assert self.x0.shape == (self.n, self.n), "x0 must be n x n"
        assert self.omega.shape == (self.n, self.n), "omega must be n x n"
        assert self.m.shape == (self.n, self.n), "m must be n x n"
        assert self.sigma.shape == (self.n, self.n), "sigma must be n x n"
        
        # Check symmetry
        assert jnp.allclose(self.omega, self.omega.T), "omega must be symmetric"
        assert jnp.allclose(self.x0, self.x0.T), "x0 must be symmetric"

        if self.u1 is not None:
            self.u1 = jnp.array(self.u1)
        if self.u2 is not None:
            self.u2 = jnp.array(self.u2)
            self.is_spread = True

@dataclass
class SwaptionConfig:
    """
    Configuration for swaption pricing.
    
    Attributes
    ----------
    maturity : float
        Option maturity
    tenor : float
        Swap tenor
    strike : float
        Strike rate
    delta_float : float
        Floating leg payment frequency
    delta_fixed : float
        Fixed leg payment frequency
    u1 : jnp.ndarray
        First weight matrix
    u2 : Optional[jnp.ndarray]
        Second weight matrix (for spreads)
    is_spread : bool
        Whether this is a spread option
    """
    maturity: float
    tenor: float
    strike: float
    delta_float: float = 0.5
    delta_fixed: float = 1.0
    call:bool=True
    # u1: Optional[jnp.ndarray] = None
    # u2: Optional[jnp.ndarray] = None
    # is_spread: bool = False
    
    def __post_init__(self):
        """Initialize weight matrices if not provided."""
        # if self.u1 is not None:
        #     self.u1 = jnp.array(self.u1)
        # if self.u2 is not None:
        #     self.u2 = jnp.array(self.u2)
        #     self.is_spread = True
        pass

@dataclass
class PricingConfig:
    """
    Configuration for pricing methods.
    
    Attributes
    ----------
    approach : str
        Pricing approach to use
    ur : float
        Real part of integration contour
    nmax : int
        Maximum integration limit
    epsabs : float
        Absolute error tolerance
    epsrel : float
        Relative error tolerance
    """
    approach: str = "range_kutta"
    ur: float = 0.5
    nmax: int = 500
    epsabs: float = 1e-7
    epsrel: float = 1e-4
    
    def __post_init__(self):
        """Validate pricing approach."""
        valid_approaches = ["range_kutta", "collin_dufresne", "gamma_approximation"]
        assert self.approach in valid_approaches, f"approach must be one of {valid_approaches}"


@dataclass
class HedgingConfig:
    """
    Configuration for hedging calculations.
    
    Attributes
    ----------
    compute_delta : bool
        Whether to compute delta hedge
    compute_vega : bool
        Whether to compute vega hedge
    compute_gamma : bool
        Whether to compute gamma
    hedge_with_bonds : bool
        Whether to hedge with zero-coupon bonds
    hedge_with_swaps : bool
        Whether to hedge with swaps
    """
    compute_delta: bool = True
    compute_vega: bool = True
    compute_gamma: bool = False
    hedge_with_bonds: bool = True
    hedge_with_swaps: bool = False
