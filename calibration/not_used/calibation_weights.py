"""
Calibration Weights for LRW model calibration.

This module provides flexible weighting schemes for calibration objectives.
Weights can be uniform, custom, tenor-based, or follow mathematical distributions.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Callable
from enum import Enum
from dataclasses import dataclass, field
import numpy as np


class WeightScheme(Enum):
    """Enumeration of available weighting schemes."""
    UNIFORM = "uniform"
    CUSTOM = "custom"
    TENOR_LINEAR = "tenor_linear"           # Linear function of tenor
    TENOR_EXPONENTIAL = "tenor_exponential" # Exponential decay/growth
    TENOR_INVERSE = "tenor_inverse"         # 1/tenor weighting
    VEGA_WEIGHTED = "vega_weighted"         # Swaption vega weighting
    LIQUIDITY = "liquidity"                 # Market liquidity based
    VARIANCE = "variance"                   # Inverse variance weighting


@dataclass
class WeightConfig:
    """
    Configuration for calibration weights.
    
    Parameters
    ----------
    scheme : WeightScheme
        The weighting scheme to use
    normalize : bool
        Whether to normalize weights to sum to 1
    min_weight : float
        Minimum weight value (floor)
    max_weight : float
        Maximum weight value (cap)
    params : dict
        Scheme-specific parameters
    """
    scheme: WeightScheme = WeightScheme.UNIFORM
    normalize: bool = True
    min_weight: float = 0.0
    max_weight: float = float('inf')
    params: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        if self.params is None:
            self.params = {}


class BaseWeight(ABC):
    """Abstract base class for weight calculators."""
    
    @abstractmethod
    def compute_weights(self, tenors: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute weights for given tenors.
        
        Parameters
        ----------
        tenors : np.ndarray
            Array of tenors/maturities
        **kwargs : dict
            Additional data (e.g., vegas, liquidity scores)
            
        Returns
        -------
        np.ndarray
            Array of weights
        """
        pass


class UniformWeight(BaseWeight):
    """Uniform weighting - all instruments weighted equally."""
    
    def compute_weights(self, tenors: np.ndarray, **kwargs) -> np.ndarray:
        return np.ones(len(tenors))


class CustomWeight(BaseWeight):
    """
    Custom user-defined weights.
    
    Parameters
    ----------
    weights_dict : Dict[float, float]
        Dictionary mapping tenor -> weight
    default_weight : float
        Default weight for tenors not in dict
    """
    
    def __init__(self, weights_dict: Dict[float, float], default_weight: float = 1.0):
        self.weights_dict = weights_dict
        self.default_weight = default_weight
    
    def compute_weights(self, tenors: np.ndarray, **kwargs) -> np.ndarray:
        weights = np.array([
            self.weights_dict.get(t, self.default_weight) 
            for t in tenors
        ])
        return weights


class TenorLinearWeight(BaseWeight):
    """
    Linear weighting based on tenor.
    
    weight(T) = a + b * T
    
    Parameters
    ----------
    slope : float
        Slope of linear function (b)
    intercept : float
        Intercept (a)
    increasing : bool
        If True, weight increases with tenor; if False, decreases
    """
    
    def __init__(self, slope: float = 0.1, intercept: float = 1.0, increasing: bool = True):
        self.slope = slope if increasing else -slope
        self.intercept = intercept
    
    def compute_weights(self, tenors: np.ndarray, **kwargs) -> np.ndarray:
        weights = self.intercept + self.slope * tenors
        return np.maximum(weights, 0)  # Ensure non-negative


class TenorExponentialWeight(BaseWeight):
    """
    Exponential weighting based on tenor.
    
    weight(T) = a * exp(b * T)
    
    Parameters
    ----------
    scale : float
        Scale factor (a)
    decay_rate : float
        Decay/growth rate (b). Negative for decay, positive for growth.
    reference_tenor : float
        Reference tenor for normalization
    """
    
    def __init__(self, scale: float = 1.0, decay_rate: float = -0.1, 
                 reference_tenor: float = 5.0):
        self.scale = scale
        self.decay_rate = decay_rate
        self.reference_tenor = reference_tenor
    
    def compute_weights(self, tenors: np.ndarray, **kwargs) -> np.ndarray:
        # Normalize relative to reference tenor
        weights = self.scale * np.exp(self.decay_rate * (tenors - self.reference_tenor))
        return weights


class TenorInverseWeight(BaseWeight):
    """
    Inverse tenor weighting (short-end focus).
    
    weight(T) = 1 / (T + epsilon)
    
    Parameters
    ----------
    epsilon : float
        Small value to avoid division by zero
    power : float
        Power for inverse (1/T^power)
    """
    
    def __init__(self, epsilon: float = 0.1, power: float = 1.0):
        self.epsilon = epsilon
        self.power = power
    
    def compute_weights(self, tenors: np.ndarray, **kwargs) -> np.ndarray:
        return 1.0 / np.power(tenors + self.epsilon, self.power)


class VegaWeight(BaseWeight):
    """
    Vega-weighted for swaptions (weight by sensitivity to volatility).
    
    Parameters
    ----------
    use_normalized_vega : bool
        If True, normalize vegas by max vega
    """
    
    def __init__(self, use_normalized_vega: bool = True):
        self.use_normalized_vega = use_normalized_vega
    
    def compute_weights(self, tenors: np.ndarray, vegas: np.ndarray = None, 
                       **kwargs) -> np.ndarray:
        if vegas is None:
            raise ValueError("vegas must be provided for VegaWeight")
        
        weights = np.abs(vegas)
        if self.use_normalized_vega and np.max(weights) > 0:
            weights = weights / np.max(weights)
        
        return weights


class LiquidityWeight(BaseWeight):
    """
    Liquidity-based weighting (higher weight for more liquid instruments).
    
    Parameters
    ----------
    liquidity_scores : Dict[float, float]
        Mapping of tenor -> liquidity score
    default_liquidity : float
        Default liquidity for unknown tenors
    """
    
    def __init__(self, liquidity_scores: Dict[float, float] = None,
                 default_liquidity: float = 0.5):
        # Default liquidity profile (typical for EUR swaptions)
        self.default_liquidity_profile = {
            0.5: 0.6,   # 6M
            1.0: 0.8,   # 1Y
            2.0: 1.0,   # 2Y - most liquid
            3.0: 0.95,  # 3Y
            5.0: 1.0,   # 5Y - benchmark
            7.0: 0.85,  # 7Y
            10.0: 0.9,  # 10Y - benchmark
            15.0: 0.7,  # 15Y
            20.0: 0.6,  # 20Y
            30.0: 0.5,  # 30Y
        }
        self.liquidity_scores = liquidity_scores or self.default_liquidity_profile
        self.default_liquidity = default_liquidity
    
    def compute_weights(self, tenors: np.ndarray, **kwargs) -> np.ndarray:
        weights = np.array([
            self._interpolate_liquidity(t) for t in tenors
        ])
        return weights
    
    def _interpolate_liquidity(self, tenor: float) -> float:
        """Interpolate liquidity for a given tenor."""
        sorted_tenors = sorted(self.liquidity_scores.keys())
        
        if tenor <= sorted_tenors[0]:
            return self.liquidity_scores[sorted_tenors[0]]
        if tenor >= sorted_tenors[-1]:
            return self.liquidity_scores[sorted_tenors[-1]]
        
        # Linear interpolation
        for i in range(len(sorted_tenors) - 1):
            if sorted_tenors[i] <= tenor <= sorted_tenors[i + 1]:
                t1, t2 = sorted_tenors[i], sorted_tenors[i + 1]
                w1, w2 = self.liquidity_scores[t1], self.liquidity_scores[t2]
                return w1 + (w2 - w1) * (tenor - t1) / (t2 - t1)
        
        return self.default_liquidity


class InverseVarianceWeight(BaseWeight):
    """
    Inverse variance weighting (statistical optimal weighting).
    
    weight(i) = 1 / variance(i)
    
    Parameters
    ----------
    min_variance : float
        Minimum variance to avoid numerical issues
    """
    
    def __init__(self, min_variance: float = 1e-8):
        self.min_variance = min_variance
    
    def compute_weights(self, tenors: np.ndarray, variances: np.ndarray = None,
                       **kwargs) -> np.ndarray:
        if variances is None:
            raise ValueError("variances must be provided for InverseVarianceWeight")
        
        clipped_var = np.maximum(variances, self.min_variance)
        return 1.0 / clipped_var


class CompositeWeight(BaseWeight):
    """
    Composite weighting - combine multiple weight schemes.
    
    Parameters
    ----------
    weights_list : List[BaseWeight]
        List of weight calculators
    combination_method : str
        How to combine: 'multiply', 'average', 'max', 'min'
    scheme_weights : List[float]
        Weights for each scheme (for 'average' method)
    """
    
    def __init__(self, weights_list: List[BaseWeight],
                 combination_method: str = 'multiply',
                 scheme_weights: List[float] = None):
        self.weights_list = weights_list
        self.combination_method = combination_method
        self.scheme_weights = scheme_weights or [1.0] * len(weights_list)
    
    def compute_weights(self, tenors: np.ndarray, **kwargs) -> np.ndarray:
        all_weights = [w.compute_weights(tenors, **kwargs) for w in self.weights_list]
        
        if self.combination_method == 'multiply':
            result = np.ones(len(tenors))
            for w in all_weights:
                result *= w
            return result
        
        elif self.combination_method == 'average':
            weighted_sum = np.zeros(len(tenors))
            for i, w in enumerate(all_weights):
                weighted_sum += self.scheme_weights[i] * w
            return weighted_sum / sum(self.scheme_weights)
        
        elif self.combination_method == 'max':
            return np.max(all_weights, axis=0)
        
        elif self.combination_method == 'min':
            return np.min(all_weights, axis=0)
        
        else:
            raise ValueError(f"Unknown combination method: {self.combination_method}")


class CalibrationWeights:
    """
    Main class for managing calibration weights.
    
    This class provides a unified interface for computing and applying
    weights to calibration errors.
    
    Parameters
    ----------
    config : WeightConfig
        Configuration for the weighting scheme
    
    Examples
    --------
    >>> # Uniform weights
    >>> weights = CalibrationWeights(WeightConfig(scheme=WeightScheme.UNIFORM))
    >>> w = weights.get_weights(tenors)
    
    >>> # Custom weights
    >>> custom_dict = {1.0: 1.5, 2.0: 1.0, 5.0: 0.8, 10.0: 0.5}
    >>> config = WeightConfig(scheme=WeightScheme.CUSTOM, params={'weights_dict': custom_dict})
    >>> weights = CalibrationWeights(config)
    
    >>> # Exponential decay
    >>> config = WeightConfig(
    ...     scheme=WeightScheme.TENOR_EXPONENTIAL,
    ...     params={'decay_rate': -0.15, 'reference_tenor': 5.0}
    ... )
    >>> weights = CalibrationWeights(config)
    
    >>> # Composite weights
    >>> config = WeightConfig(
    ...     scheme=WeightScheme.CUSTOM,
    ...     params={
    ...         'composite': True,
    ...         'schemes': [WeightScheme.LIQUIDITY, WeightScheme.TENOR_EXPONENTIAL],
    ...         'combination_method': 'multiply'
    ...     }
    ... )
    """
    
    def __init__(self, config: WeightConfig = None):
        self.config = config or WeightConfig()
        self._weight_calculator = self._create_calculator()
        self._cached_weights = None
        self._cached_tenors = None
    
    def _create_calculator(self) -> BaseWeight:
        """Create the appropriate weight calculator based on config."""
        scheme = self.config.scheme
        params = self.config.params
        
        # Check for composite weights
        if params.get('composite', False):
            schemes = params.get('schemes', [])
            calculators = [self._create_single_calculator(s, params) for s in schemes]
            return CompositeWeight(
                calculators,
                combination_method=params.get('combination_method', 'multiply'),
                scheme_weights=params.get('scheme_weights')
            )
        
        return self._create_single_calculator(scheme, params)
    
    def _create_single_calculator(self, scheme: WeightScheme, 
                                   params: Dict) -> BaseWeight:
        """Create a single weight calculator."""
        if scheme == WeightScheme.UNIFORM:
            return UniformWeight()
        
        elif scheme == WeightScheme.CUSTOM:
            return CustomWeight(
                weights_dict=params.get('weights_dict', {}),
                default_weight=params.get('default_weight', 1.0)
            )
        
        elif scheme == WeightScheme.TENOR_LINEAR:
            return TenorLinearWeight(
                slope=params.get('slope', 0.1),
                intercept=params.get('intercept', 1.0),
                increasing=params.get('increasing', True)
            )
        
        elif scheme == WeightScheme.TENOR_EXPONENTIAL:
            return TenorExponentialWeight(
                scale=params.get('scale', 1.0),
                decay_rate=params.get('decay_rate', -0.1),
                reference_tenor=params.get('reference_tenor', 5.0)
            )
        
        elif scheme == WeightScheme.TENOR_INVERSE:
            return TenorInverseWeight(
                epsilon=params.get('epsilon', 0.1),
                power=params.get('power', 1.0)
            )
        
        elif scheme == WeightScheme.VEGA_WEIGHTED:
            return VegaWeight(
                use_normalized_vega=params.get('use_normalized_vega', True)
            )
        
        elif scheme == WeightScheme.LIQUIDITY:
            return LiquidityWeight(
                liquidity_scores=params.get('liquidity_scores'),
                default_liquidity=params.get('default_liquidity', 0.5)
            )
        
        elif scheme == WeightScheme.VARIANCE:
            return InverseVarianceWeight(
                min_variance=params.get('min_variance', 1e-8)
            )
        
        else:
            raise ValueError(f"Unknown weight scheme: {scheme}")
    
    def get_weights(self, tenors: np.ndarray, **kwargs) -> np.ndarray:
        """
        Get weights for given tenors.
        
        Parameters
        ----------
        tenors : np.ndarray
            Array of tenors/maturities
        **kwargs : dict
            Additional data (vegas, variances, etc.)
            
        Returns
        -------
        np.ndarray
            Array of weights
        """
        # Compute raw weights
        weights = self._weight_calculator.compute_weights(tenors, **kwargs)
        
        # Apply bounds
        weights = np.clip(weights, self.config.min_weight, self.config.max_weight)
        
        # Normalize if requested
        if self.config.normalize and np.sum(weights) > 0:
            weights = weights / np.sum(weights) * len(weights)
        
        # Cache for potential reuse
        self._cached_weights = weights
        self._cached_tenors = tenors
        
        return weights
    
    def apply_weights_to_errors(self, errors: np.ndarray, 
                                 tenors: np.ndarray = None,
                                 **kwargs) -> np.ndarray:
        """
        Apply weights to error array.
        
        Parameters
        ----------
        errors : np.ndarray
            Array of calibration errors
        tenors : np.ndarray, optional
            Tenors for weight computation (uses cached if None)
        **kwargs : dict
            Additional data for weight computation
            
        Returns
        -------
        np.ndarray
            Weighted errors
        """
        if tenors is None and self._cached_weights is not None:
            weights = self._cached_weights
        elif tenors is not None:
            weights = self.get_weights(tenors, **kwargs)
        else:
            raise ValueError("Either tenors must be provided or weights must be cached")
        
        if len(weights) != len(errors):
            raise ValueError(f"Weight length ({len(weights)}) != error length ({len(errors)})")
        
        return errors * np.sqrt(weights)
    
    def compute_weighted_rmse(self, errors: np.ndarray, 
                               tenors: np.ndarray = None,
                               **kwargs) -> float:
        """
        Compute weighted RMSE.
        
        Parameters
        ----------
        errors : np.ndarray
            Array of calibration errors
        tenors : np.ndarray, optional
            Tenors for weight computation
        **kwargs : dict
            Additional data
            
        Returns
        -------
        float
            Weighted RMSE
        """
        weighted_errors = self.apply_weights_to_errors(errors, tenors, **kwargs)
        return np.sqrt(np.mean(weighted_errors ** 2))
    
    def summary(self, tenors: np.ndarray = None) -> str:
        """Return a summary of the current weight configuration."""
        lines = [
            f"Calibration Weights Summary",
            f"{'=' * 40}",
            f"Scheme: {self.config.scheme.value}",
            f"Normalize: {self.config.normalize}",
            f"Min weight: {self.config.min_weight}",
            f"Max weight: {self.config.max_weight}",
        ]
        
        if self.config.params:
            lines.append(f"Parameters: {self.config.params}")
        
        if tenors is not None:
            weights = self.get_weights(tenors)
            lines.extend([
                f"\nWeight Distribution:",
                f"Tenors: {tenors}",
                f"Weights: {weights}",
                f"Min: {np.min(weights):.4f}, Max: {np.max(weights):.4f}",
                f"Mean: {np.mean(weights):.4f}, Std: {np.std(weights):.4f}"
            ])
        
        return "\n".join(lines)


# ============================================================================
# Convenience factory functions
# ============================================================================

def uniform_weights() -> CalibrationWeights:
    """Create uniform weights."""
    return CalibrationWeights(WeightConfig(scheme=WeightScheme.UNIFORM))


def custom_weights(weights_dict: Dict[float, float], 
                   default: float = 1.0,
                   normalize: bool = True) -> CalibrationWeights:
    """
    Create custom weights from a dictionary.
    
    Parameters
    ----------
    weights_dict : Dict[float, float]
        Mapping of tenor -> weight
    default : float
        Default weight for unspecified tenors
    normalize : bool
        Whether to normalize weights
    """
    return CalibrationWeights(WeightConfig(
        scheme=WeightScheme.CUSTOM,
        normalize=normalize,
        params={'weights_dict': weights_dict, 'default_weight': default}
    ))


def exponential_decay_weights(decay_rate: float = -0.1,
                               reference_tenor: float = 5.0,
                               normalize: bool = True) -> CalibrationWeights:
    """
    Create exponential decay weights (emphasize short tenors).
    
    Parameters
    ----------
    decay_rate : float
        Negative for decay (short-end focus), positive for growth
    reference_tenor : float
        Reference tenor where weight = 1
    """
    return CalibrationWeights(WeightConfig(
        scheme=WeightScheme.TENOR_EXPONENTIAL,
        normalize=normalize,
        params={'decay_rate': decay_rate, 'reference_tenor': reference_tenor}
    ))


def liquidity_weights(custom_scores: Dict[float, float] = None,
                      normalize: bool = True) -> CalibrationWeights:
    """
    Create liquidity-based weights.
    
    Parameters
    ----------
    custom_scores : Dict[float, float], optional
        Custom liquidity scores. Uses default profile if None.
    """
    return CalibrationWeights(WeightConfig(
        scheme=WeightScheme.LIQUIDITY,
        normalize=normalize,
        params={'liquidity_scores': custom_scores}
    ))


def swaption_vega_weights(normalize: bool = True) -> CalibrationWeights:
    """Create vega-weighted scheme for swaptions."""
    return CalibrationWeights(WeightConfig(
        scheme=WeightScheme.VEGA_WEIGHTED,
        normalize=normalize
    ))


def short_end_focus_weights(power: float = 1.0,
                             normalize: bool = True) -> CalibrationWeights:
    """
    Create weights that focus on short tenors (1/T weighting).
    
    Parameters
    ----------
    power : float
        Power for inverse weighting (1/T^power)
    """
    return CalibrationWeights(WeightConfig(
        scheme=WeightScheme.TENOR_INVERSE,
        normalize=normalize,
        params={'power': power}
    ))


def combined_weights(schemes: List[WeightScheme],
                     combination: str = 'multiply',
                     scheme_weights: List[float] = None,
                     normalize: bool = True,
                     **scheme_params) -> CalibrationWeights:
    """
    Create combined weights from multiple schemes.
    
    Parameters
    ----------
    schemes : List[WeightScheme]
        List of schemes to combine
    combination : str
        How to combine: 'multiply', 'average', 'max', 'min'
    scheme_weights : List[float]
        Weights for averaging (if combination='average')
    **scheme_params : dict
        Parameters for individual schemes
    """
    return CalibrationWeights(WeightConfig(
        scheme=WeightScheme.CUSTOM,
        normalize=normalize,
        params={
            'composite': True,
            'schemes': schemes,
            'combination_method': combination,
            'scheme_weights': scheme_weights,
            **scheme_params
        }
    ))


# ============================================================================
# Integration helper for ObjectiveFunctions class
# ============================================================================

class WeightedObjectiveMixin:
    """
    Mixin class to add weighting capabilities to ObjectiveFunctions.
    
    Add this to your ObjectiveFunctions class:
    
    class ObjectiveFunctions(WeightedObjectiveMixin):
        ...
    
    Or use composition:
    
    self.weight_manager = CalibrationWeights(config)
    """
    
    def set_ois_weights(self, weights: CalibrationWeights):
        """Set weights for OIS calibration."""
        self._ois_weights = weights
    
    def set_spread_weights(self, weights: CalibrationWeights):
        """Set weights for spread calibration."""
        self._spread_weights = weights
    
    def set_swaption_weights(self, weights: CalibrationWeights):
        """Set weights for swaption calibration."""
        self._swaption_weights = weights
    
    def get_weighted_errors(self, errors: np.ndarray, tenors: np.ndarray,
                            instrument_type: str = 'ois', **kwargs) -> np.ndarray:
        """
        Apply appropriate weights to errors based on instrument type.
        
        Parameters
        ----------
        errors : np.ndarray
            Raw calibration errors
        tenors : np.ndarray
            Instrument tenors
        instrument_type : str
            'ois', 'spread', or 'swaption'
        **kwargs : dict
            Additional data (vegas for swaptions, etc.)
        """
        weight_map = {
            'ois': getattr(self, '_ois_weights', None),
            'spread': getattr(self, '_spread_weights', None),
            'swaption': getattr(self, '_swaption_weights', None)
        }
        
        weights = weight_map.get(instrument_type)
        
        if weights is None:
            return errors  # No weights applied
        
        return weights.apply_weights_to_errors(errors, tenors, **kwargs)
