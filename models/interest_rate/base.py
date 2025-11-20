"""
Base classes for interest rate models.

models/interest_rate/base.py
"""
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
import jax.numpy as jnp
from dataclasses import dataclass

from .config import LRWModelConfig,SwaptionConfig


@dataclass
class InterestRateModelConfig:
    """Base configuration for interest rate models."""
    maturity: float
    tenor: float
    delta_float: float = 0.5
    delta_fixed: float = 1.0
    strike: float = 0.0

   

class BaseInterestRateModel(ABC):
    """Abstract base class for interest rate models."""
    
    def __init__(self, config: SwaptionConfig):##InterestRateModelConfig):
        """
        Initialize base interest rate model.
        
        Parameters
        ----------
        config : InterestRateModelConfig
            Model configuration
        """
        self.config = config
        self.maturity = config.maturity
        self.tenor = config.tenor
        self.delta_float = config.delta_float
        self.delta_fixed = config.delta_fixed
        self.strike = config.strike
    
    def set_swaption_config(self, swaption_config: SwaptionConfig):
        """
        Set swaption configuration.
        
        Parameters
        ----------
        swaption_config : SwaptionConfig
            Swaption configuration object
        """
        self.config = swaption_config
        self.maturity = swaption_config.maturity
        self.tenor = swaption_config.tenor
        self.strike = swaption_config.strike
        self.delta_float = swaption_config.delta_float
        self.delta_fixed = swaption_config.delta_fixed
        
    @abstractmethod
    def get_short_rate(self) -> float:
        """
        Get current short rate.
        
        Returns
        -------
        float
            Current short rate
        """
        pass
    
    @abstractmethod
    def bond(self, t: float) -> float:
        """
        Compute zero-coupon bond price.
        
        Parameters
        ----------
        t : float
            Time to maturity
            
        Returns
        -------
        float
            Bond price
        """
        pass
    
    @abstractmethod
    def compute_swap_rate(self) -> float:
        """
        Compute swap rate.
        
        Returns
        -------
        float
            Swap rate
        """
        pass
    
    @abstractmethod
    def compute_annuity(self) -> Tuple[float, float]:
        """
        Compute annuity and ATM swap rate.
        
        Returns
        -------
        annuity : float
            Annuity value
        swap_rate : float
            ATM swap rate
        """
        pass
    
    def print_model(self) -> None:
        """Print model parameters."""
        print("\nModel Configuration")
        print(f"Maturity: {self.maturity}")
        print(f"Tenor: {self.tenor}")
        print(f"Strike: {self.strike}")
        print(f"Delta Float: {self.delta_float}")
        print(f"Delta Fixed: {self.delta_fixed}")


class BaseSwaptionPricer(ABC):
    """Abstract base class for swaption pricing."""
    
    def __init__(self, model: BaseInterestRateModel):
        """
        Initialize swaption pricer.
        
        Parameters
        ----------
        model : BaseInterestRateModel
            Interest rate model
        """
        self.model = model
    
    @abstractmethod
    def price_swaption(self, **kwargs) -> float:
        """
        Price a swaption.
        
        Returns
        -------
        float
            Swaption price
        """
        pass
    
    @abstractmethod
    def compute_implied_vol(self, price: float, epsilon: float = 1e-6) -> float:
        """
        Compute implied volatility from price.
        
        Parameters
        ----------
        price : float
            Option price
        epsilon : float, optional
            Convergence tolerance
            
        Returns
        -------
        float
            Implied volatility
        """
        pass
    
    @abstractmethod
    def compute_delta(self, **kwargs) -> Dict[str, float]:
        """
        Compute delta sensitivities.
        
        Returns
        -------
        Dict[str, float]
            Delta values for different instruments
        """
        pass
    
    @abstractmethod
    def compute_vega(self, **kwargs) -> jnp.ndarray:
        """
        Compute vega sensitivities.
        
        Returns
        -------
        jnp.ndarray
            Vega matrix
        """
        pass


class BaseGreeksCalculator(ABC):
    """Abstract base class for Greeks calculations."""
    
    def __init__(self, model: BaseInterestRateModel, pricer: BaseSwaptionPricer):
        """
        Initialize Greeks calculator.
        
        Parameters
        ----------
        model : BaseInterestRateModel
            Interest rate model
        pricer : BaseSwaptionPricer
            Swaption pricer
        """
        self.model = model
        self.pricer = pricer
    
    @abstractmethod
    def compute_delta_hedge(self) -> Dict[str, Any]:
        """
        Compute delta hedging portfolio.
        
        Returns
        -------
        Dict[str, Any]
            Hedging portfolio details
        """
        pass
    
    @abstractmethod
    def compute_gamma(self, **kwargs) -> float:
        """
        Compute gamma sensitivity.
        
        Returns
        -------
        float
            Gamma value
        """
        pass
    
    @abstractmethod
    def compute_vega_hedge(self) -> Dict[str, Any]:
        """
        Compute vega hedging portfolio.
        
        Returns
        -------
        Dict[str, Any]
            Vega hedging details
        """
        pass