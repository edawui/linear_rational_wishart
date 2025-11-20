# models/fx/base.py
"""Base class for FX models."""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
import jax.numpy as jnp


class BaseFxModel(ABC):
    """Abstract base class for FX models."""
    
    def __init__(self, domestic_model, foreign_model, fx_spot: float = 1.0):
        """Initialize FX model with domestic and foreign interest rate models."""
        self.domestic_model = domestic_model
        self.foreign_model = foreign_model
        self.fx_spot = fx_spot
        
    @abstractmethod
    def compute_fx_forward(self, t: float) -> float:
        """Compute FX forward rate."""
        pass
        
    @abstractmethod
    def price_fx_option(self, maturity: float, strike: float, 
                       is_call: bool = True) -> float:
        """Price FX option."""
        pass
        
    @abstractmethod
    def compute_fx_vol_correlation(self, t: float) -> float:
        """Compute FX volatility correlation."""
        pass

