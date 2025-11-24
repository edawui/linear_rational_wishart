"""
Pricing approach enumerations and factory.

enums/pricing_approaches.py
"""
from enum import Enum
from typing import Union

from ..models.interest_rate.lrw_model import LRWModel
# from ..models.interest_rate.base import BaseSwaptionPricer
from ..pricing.swaption.base import BaseSwaptionPricer
from lrw_Jax.wishart_processes.pricing.swaption.lrw-swaption-pricing import FourierSwaptionPricer
from lrw_Jax.wishart_processes.pricing.swaption.lrw-cd-approximation import CollinDufresneSwaptionPricer
from lrw_Jax.wishart_processes.pricing.swaption.lrw-gamma-approximation import GammaApproximationPricer


class PricingApproach(Enum):
    """
    Enumeration of available pricing approaches.
    """
    RANGE_KUTTA = "range_kutta"
    FOURIER = "fourier"
    COLLIN_DUFRESNE = "collin_dufresne"
    GAMMA_APPROXIMATION = "gamma_approximation"
    
    @classmethod
    def from_string(cls, value: str) -> 'PricingApproach':
        """
        Create enum from string value.
        
        Parameters
        ----------
        value : str
            String representation of pricing approach
            
        Returns
        -------
        PricingApproach
            Enum value
            
        Raises
        ------
        ValueError
            If value is not recognized
        """
        value_lower = value.lower()
        for approach in cls:
            if approach.value == value_lower:
                return approach
        raise ValueError(f"Unknown pricing approach: {value}")


class PricerFactory:
    """
    Factory class for creating swaption pricers.
    """
    
    @staticmethod
    def create_pricer(approach: Union[PricingApproach, str], 
                     model: LRWModel) -> BaseSwaptionPricer:
        """
        Create a swaption pricer based on the specified approach.
        
        Parameters
        ----------
        approach : Union[PricingApproach, str]
            Pricing approach to use
        model : LRWModel
            LRW model instance
            
        Returns
        -------
        BaseSwaptionPricer
            Swaption pricer instance
            
        Raises
        ------
        ValueError
            If approach is not supported
        """
        # Convert string to enum if necessary
        if isinstance(approach, str):
            approach = PricingApproach.from_string(approach)
        
        # Create appropriate pricer
        if approach == PricingApproach.RANGE_KUTTA:
            return FourierSwaptionPricer(model)
        elif approach == PricingApproach.FOURIER:
            return FourierSwaptionPricer(model)
        elif approach == PricingApproach.COLLIN_DUFRESNE:
            return CollinDufresneSwaptionPricer(model)
        elif approach == PricingApproach.GAMMA_APPROXIMATION:
            return GammaApproximationPricer(model)
        else:
            raise ValueError(f"Unsupported pricing approach: {approach}")


# Hedging approach enumeration
class HedgingApproach(Enum):
    """
    Enumeration of available hedging approaches.
    """
    ZERO_COUPON_BONDS = "zero_coupon_bonds"
    SWAPS = "swaps"
    SWAPTIONS = "swaptions"
    MIXED = "mixed"


# Greeks calculation methods
class GreeksMethod(Enum):
    """
    Enumeration of Greeks calculation methods.
    """
    ANALYTICAL = "analytical"
    FINITE_DIFFERENCE = "finite_difference"
    COMPLEX_STEP = "complex_step"
    AUTOMATIC_DIFFERENTIATION = "automatic_differentiation"
