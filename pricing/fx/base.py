
# pricing/fx/base.py
"""Base class for FX option pricing."""

from abc import ABC, abstractmethod
from typing import List, Optional
import jax.numpy as jnp


class BaseFxPricer(ABC):
    """Abstract base class for FX option pricing."""
    
    def __init__(self, fx_model):
        """Initialize with FX model."""
        self.fx_model = fx_model
        
    @abstractmethod
    def price_options(self, maturities: List[float], strikes: List[float],
                     is_calls: List[bool], **kwargs) -> List[float]:
        """Price multiple FX options."""
        pass
        
    @abstractmethod
    def get_pricing_method(self) -> str:
        """Get name of pricing method."""
        pass
