"""
Calibration Weights for LRW model calibration.

Simple weighting schemes for different instrument types:
- Single-tenor instruments: OIS, Spreads, FX Options
- Multi-dimensional instruments: Swaptions (expiry, tenor, strike)
"""

import numpy as np
from typing import Dict, List, Callable, Optional, Tuple
from enum import Enum


class WeightScheme(Enum):
    """Available weighting schemes."""
    UNIFORM = "uniform"
    CUSTOM = "custom"
    TENOR_BASED = "tenor_based"  # Function of tenor
    LIQUIDITY = "liquidity"


class SimpleWeights:
    """
    Simple weighting for single-tenor instruments (OIS, Spreads, FX Options).
    
    Examples
    --------
    >>> w = SimpleWeights.uniform()
    >>> w = SimpleWeights.custom({1.0: 1.5, 2.0: 1.0, 5.0: 0.8})
    >>> w = SimpleWeights.short_end_focus(decay=0.1)
    """
    
    def __init__(self, weight_func: Callable[[float], float] = None):
        """
        Parameters
        ----------
        weight_func : callable
            Function that takes tenor and returns weight.
            If None, returns 1.0 (uniform).
        """
        self._weight_func = weight_func or (lambda t: 1.0)
    
    def get_weight(self, tenor: float) -> float:
        """Get weight for a single tenor."""
        return self._weight_func(tenor)
    
    def get_weights(self, tenors: List[float], normalize: bool = False) -> np.ndarray:
        """Get weights for multiple tenors."""
        weights = np.array([self._weight_func(t) for t in tenors])
        if normalize and weights.sum() > 0:
            weights = weights / weights.sum() * len(weights)
        return weights
    
    def apply_to_errors(self, errors: np.ndarray, tenors: List[float], 
                        normalize: bool = False) -> np.ndarray:
        """Apply sqrt(weight) to errors for least-squares."""
        weights = self.get_weights(tenors, normalize)
        return errors * np.sqrt(weights)
    
    # -------------------------------------------------------------------------
    # Factory methods
    # -------------------------------------------------------------------------
    
    @classmethod
    def uniform(cls) -> 'SimpleWeights':
        """All instruments weighted equally."""
        return cls(lambda t: 1.0)
    
    @classmethod
    def custom(cls, weight_dict: Dict[float, float], 
               default: float = 1.0) -> 'SimpleWeights':
        """
        Custom weights from dictionary.
        
        Parameters
        ----------
        weight_dict : dict
            Tenor -> weight mapping
        default : float
            Weight for tenors not in dict
        """
        def func(t):
            # Find closest tenor in dict
            if t in weight_dict:
                return weight_dict[t]
            # Try to find close match (within 0.01)
            for key in weight_dict:
                if abs(key - t) < 0.01:
                    return weight_dict[key]
            return default
        return cls(func)
    
    @classmethod
    def short_end_focus(cls, decay: float = 0.1, 
                        ref_tenor: float = 1.0) -> 'SimpleWeights':
        """
        Higher weight on short tenors.
        weight = exp(-decay * (tenor - ref_tenor))
        
        Parameters
        ----------
        decay : float
            Decay rate. Higher = more focus on short end.
        ref_tenor : float
            Reference tenor where weight = 1.
        """
        return cls(lambda t: np.exp(-decay * (t - ref_tenor)))
    
    @classmethod
    def long_end_focus(cls, growth: float = 0.1,
                       ref_tenor: float = 5.0) -> 'SimpleWeights':
        """Higher weight on long tenors."""
        return cls(lambda t: np.exp(growth * (t - ref_tenor)))
    
    @classmethod
    def inverse_tenor(cls, epsilon: float = 0.5) -> 'SimpleWeights':
        """Weight = 1 / (tenor + epsilon). Strong short-end focus."""
        return cls(lambda t: 1.0 / (t + epsilon))
    
    @classmethod
    def liquidity_profile(cls, profile: Dict[float, float] = None) -> 'SimpleWeights':
        """
        Liquidity-based weights with interpolation.
        
        Parameters
        ----------
        profile : dict, optional
            Tenor -> liquidity score. Uses EUR default if None.
        """
        if profile is None:
            # Default EUR liquidity profile
            profile = {
                0.5: 0.7, 1.0: 0.9, 2.0: 1.0, 3.0: 0.95,
                5.0: 1.0, 7.0: 0.85, 10.0: 0.9, 15.0: 0.7
            }
        
        def func(t):
            tenors = sorted(profile.keys())
            if t <= tenors[0]:
                return profile[tenors[0]]
            if t >= tenors[-1]:
                return profile[tenors[-1]]
            # Linear interpolation
            for i in range(len(tenors) - 1):
                if tenors[i] <= t <= tenors[i + 1]:
                    t1, t2 = tenors[i], tenors[i + 1]
                    w1, w2 = profile[t1], profile[t2]
                    return w1 + (w2 - w1) * (t - t1) / (t2 - t1)
            return 1.0
        
        return cls(func)


class SwaptionWeights:
    """
    Weights for swaptions (multi-dimensional: expiry, tenor, strike).
    
    Can weight by:
    - Expiry only
    - Tenor only  
    - Total maturity (expiry + tenor)
    - Custom (expiry, tenor) pairs
    - Strike/moneyness
    
    Examples
    --------
    >>> w = SwaptionWeights.uniform()
    >>> w = SwaptionWeights.by_expiry(SimpleWeights.short_end_focus())
    >>> w = SwaptionWeights.by_total_maturity(SimpleWeights.short_end_focus())
    >>> w = SwaptionWeights.diagonal_focus()  # expiry ≈ tenor
    """
    
    def __init__(self, weight_func: Callable[[float, float, float], float] = None):
        """
        Parameters
        ----------
        weight_func : callable
            Function(expiry, tenor, strike_offset) -> weight.
            If None, returns 1.0 (uniform).
        """
        self._weight_func = weight_func or (lambda e, t, s: 1.0)
    
    def get_weight(self, expiry: float, tenor: float, 
                   strike_offset: float = 0.0) -> float:
        """Get weight for a single swaption."""
        return self._weight_func(expiry, tenor, strike_offset)
    
    def get_weights(self, swaptions: List[Tuple[float, float, float]],
                    normalize: bool = False) -> np.ndarray:
        """
        Get weights for multiple swaptions.
        
        Parameters
        ----------
        swaptions : list of tuples
            List of (expiry, tenor, strike_offset) tuples
        """
        weights = np.array([self._weight_func(e, t, s) for e, t, s in swaptions])
        if normalize and weights.sum() > 0:
            weights = weights / weights.sum() * len(weights)
        return weights
    
    def apply_to_errors(self, errors: np.ndarray, 
                        swaptions: List[Tuple[float, float, float]],
                        normalize: bool = False) -> np.ndarray:
        """Apply sqrt(weight) to errors for least-squares."""
        weights = self.get_weights(swaptions, normalize)
        return errors * np.sqrt(weights)
    
    # -------------------------------------------------------------------------
    # Factory methods
    # -------------------------------------------------------------------------
    
    @classmethod
    def uniform(cls) -> 'SwaptionWeights':
        """All swaptions weighted equally."""
        return cls(lambda e, t, s: 1.0)
    
    @classmethod
    def by_expiry(cls, expiry_weights: SimpleWeights) -> 'SwaptionWeights':
        """Weight based on expiry only."""
        return cls(lambda e, t, s: expiry_weights.get_weight(e))
    
    @classmethod
    def by_tenor(cls, tenor_weights: SimpleWeights) -> 'SwaptionWeights':
        """Weight based on swap tenor only."""
        return cls(lambda e, t, s: tenor_weights.get_weight(t))
    
    @classmethod
    def by_total_maturity(cls, mat_weights: SimpleWeights) -> 'SwaptionWeights':
        """Weight based on total maturity (expiry + tenor)."""
        return cls(lambda e, t, s: mat_weights.get_weight(e + t))
    
    @classmethod
    def custom(cls, weight_dict: Dict[Tuple[float, float], float],
               default: float = 1.0) -> 'SwaptionWeights':
        """
        Custom weights for (expiry, tenor) pairs.
        
        Parameters
        ----------
        weight_dict : dict
            {(expiry, tenor): weight} mapping
        default : float
            Weight for pairs not in dict
        """
        def func(e, t, s):
            key = (e, t)
            if key in weight_dict:
                return weight_dict[key]
            # Try close match
            for (e2, t2), w in weight_dict.items():
                if abs(e - e2) < 0.01 and abs(t - t2) < 0.01:
                    return w
            return default
        return cls(func)
    
    @classmethod  
    def diagonal_focus(cls, off_diag_discount: float = 0.7) -> 'SwaptionWeights':
        """
        Higher weight on diagonal swaptions (expiry ≈ tenor).
        These are typically most liquid.
        
        Parameters
        ----------
        off_diag_discount : float
            Weight for off-diagonal swaptions (0 to 1)
        """
        def func(e, t, s):
            if abs(e - t) < 0.1:  # On diagonal
                return 1.0
            return off_diag_discount
        return cls(func)
    
    @classmethod
    def atm_focus(cls, otm_discount: float = 0.5) -> 'SwaptionWeights':
        """
        Higher weight on ATM swaptions.
        
        Parameters
        ----------
        otm_discount : float
            Weight for non-ATM swaptions (0 to 1)
        """
        def func(e, t, s):
            if abs(s) < 1:  # ATM (strike_offset near 0)
                return 1.0
            return otm_discount
        return cls(func)
    
    @classmethod
    def benchmark_focus(cls, benchmarks: List[Tuple[float, float]] = None,
                        non_benchmark_weight: float = 0.5) -> 'SwaptionWeights':
        """
        Higher weight on benchmark swaptions.
        
        Parameters
        ----------
        benchmarks : list of tuples
            List of (expiry, tenor) benchmark points.
            Default: [(1,1), (2,2), (5,5), (10,10), (2,5), (5,10)]
        """
        if benchmarks is None:
            benchmarks = [
                (1.0, 1.0), (2.0, 2.0), (5.0, 5.0), (10.0, 10.0),
                (2.0, 5.0), (5.0, 10.0), (1.0, 5.0), (5.0, 5.0)
            ]
        
        def func(e, t, s):
            for be, bt in benchmarks:
                if abs(e - be) < 0.1 and abs(t - bt) < 0.1:
                    return 1.0
            return non_benchmark_weight
        return cls(func)
    
    @classmethod
    def combined(cls, expiry_weights: SimpleWeights = None,
                 tenor_weights: SimpleWeights = None,
                 atm_only: bool = False) -> 'SwaptionWeights':
        """
        Combine expiry and tenor weights (multiplicative).
        
        Parameters
        ----------
        expiry_weights : SimpleWeights
            Weights based on expiry
        tenor_weights : SimpleWeights
            Weights based on tenor
        atm_only : bool
            If True, set weight to 0 for non-ATM
        """
        expiry_w = expiry_weights or SimpleWeights.uniform()
        tenor_w = tenor_weights or SimpleWeights.uniform()
        
        def func(e, t, s):
            if atm_only and abs(s) > 1:
                return 0.0
            return expiry_w.get_weight(e) * tenor_w.get_weight(t)
        return cls(func)


# =============================================================================
# Convenience functions
# =============================================================================

def uniform_weights() -> SimpleWeights:
    """Create uniform weights."""
    return SimpleWeights.uniform()


def create_ois_weights(scheme: str = "uniform", **kwargs) -> SimpleWeights:
    """
    Create weights for OIS calibration.
    
    Parameters
    ----------
    scheme : str
        'uniform', 'short_end', 'long_end', 'custom', 'liquidity'
    **kwargs : 
        Parameters for the scheme
    """
    if scheme == "uniform":
        return SimpleWeights.uniform()
    elif scheme == "short_end":
        return SimpleWeights.short_end_focus(
            decay=kwargs.get('decay', 0.1),
            ref_tenor=kwargs.get('ref_tenor', 1.0)
        )
    elif scheme == "long_end":
        return SimpleWeights.long_end_focus(
            growth=kwargs.get('growth', 0.1),
            ref_tenor=kwargs.get('ref_tenor', 5.0)
        )
    elif scheme == "custom":
        return SimpleWeights.custom(
            kwargs.get('weights', {}),
            kwargs.get('default', 1.0)
        )
    elif scheme == "liquidity":
        return SimpleWeights.liquidity_profile(kwargs.get('profile'))
    else:
        return SimpleWeights.uniform()


def create_swaption_weights(scheme: str = "uniform", **kwargs) -> SwaptionWeights:
    """
    Create weights for swaption calibration.
    
    Parameters
    ----------
    scheme : str
        'uniform', 'by_expiry', 'by_tenor', 'by_total', 
        'diagonal', 'atm', 'benchmark', 'custom'
    **kwargs :
        Parameters for the scheme
    """
    if scheme == "uniform":
        return SwaptionWeights.uniform()
    elif scheme == "by_expiry":
        expiry_w = kwargs.get('expiry_weights', SimpleWeights.uniform())
        return SwaptionWeights.by_expiry(expiry_w)
    elif scheme == "by_tenor":
        tenor_w = kwargs.get('tenor_weights', SimpleWeights.uniform())
        return SwaptionWeights.by_tenor(tenor_w)
    elif scheme == "by_total":
        total_w = kwargs.get('total_weights', SimpleWeights.uniform())
        return SwaptionWeights.by_total_maturity(total_w)
    elif scheme == "diagonal":
        return SwaptionWeights.diagonal_focus(
            kwargs.get('off_diag_discount', 0.7)
        )
    elif scheme == "atm":
        return SwaptionWeights.atm_focus(
            kwargs.get('otm_discount', 0.5)
        )
    elif scheme == "benchmark":
        return SwaptionWeights.benchmark_focus(
            kwargs.get('benchmarks'),
            kwargs.get('non_benchmark_weight', 0.5)
        )
    elif scheme == "custom":
        return SwaptionWeights.custom(
            kwargs.get('weights', {}),
            kwargs.get('default', 1.0)
        )
    else:
        return SwaptionWeights.uniform()
