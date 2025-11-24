
# pricing/swaption/base.py
"""Base class for swaption pricing methods."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import jax.numpy as jnp
import warnings

class BaseSwaptionPricer(ABC):
    """Abstract base class for swaption pricing."""
    
    def __init__(self, model):
        """Initialize with interest rate model."""
        self.model = model
        
    @abstractmethod
    def price(self, **kwargs) -> float:
        """Price the swaption."""
        pass
        
    @abstractmethod
    def get_pricing_info(self) -> Dict[str, Any]:
        """Get detailed pricing information."""
        pass
        
    def validate_inputs(self):
        """Validate model inputs before pricing."""
        if self.model.maturity <= 0:
            raise ValueError("Maturity must be positive")
            
        if self.model.tenor <= 0:
            raise ValueError("Tenor must be positive")
            
        # if self.model.strike < 0:
        #     raise ValueError("Strike cannot be negative")

        if self.model.strike < 0:
            warnings.warn("Strike cannot be negative", 
                  UserWarning, 
                  stacklevel=2)  # Shows warning at caller's location
