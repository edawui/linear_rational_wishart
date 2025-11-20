"""
Base classes for FX model calibration.

This module provides abstract base classes and interfaces for FX model calibration,
including calibration results and parameter management.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from tabnanny import verbose
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from enum import Enum


from ...models.fx.lrw_fx import LRWFxModel
from ...data.data_fx_market_data import CurrencyPairDailyData,CalibWeightType


class OptimizationMethod(Enum):
    """Available optimization methods for calibration."""
    LEAST_SQUARES = "least_squares"
    DIFFERENTIAL_EVOLUTION = "differential_evolution"
    HYBRID = "hybrid"
    DUAL_ANNEALING = "dual_annealing"


class PricingMethod(Enum):
    """Available pricing methods for calibration."""
    MONTE_CARLO = "MC"
    PDE = "PDE"
    ANALYTICAL = "ANALYTICAL"
    FOURIER = "FOURIER"


@dataclass
class CalibrationConfig:
    """Configuration for FX model calibration."""
    set_same_alpha:bool=False#True
    use_atm_only: bool = False#True
    
    min_maturity_zc: float = 2.0
    max_maturity_zc: float = 11.0
    default_alpha_tenor: float = 20.0

    min_maturity_option: float = 5.0#2.0#7.0#1.0#1.0
    max_maturity_option: float = 5.0#2.0#7.0#5.0#10.0

    pricing_method: PricingMethod = PricingMethod.FOURIER
    # pricing_method: PricingMethod = PricingMethod.MONTE_CARLO
    optimization_method: OptimizationMethod = OptimizationMethod.LEAST_SQUARES
    # optimization_method: OptimizationMethod = OptimizationMethod.HYBRID
    reprice_all_option:bool=True#False

    min_maturity_option_for_chart:float=1.0#2.0#1.0#1.0#1.0
    max_maturity_option_for_chart:float=11.0#2.0#1.0#1.0#1.0

    mc_paths: int = 5000
    mc_timestep: float = 0.125 ##1.0 / 25 #0.125 #
    use_multithreading: bool = False
    calibrate_on_vol: bool = False
    calibrate_based_on_correlation: bool = True
    pseudo_inverse_smoothing: bool = False
    ###Calibration strikes
    selected_based_on_strike:bool=True #False # True # if strikes are selected based on money-ness, False if all strikes are used
    min_option_strike_index:int=0
    max_option_strike_index:int=4

    custom_calibration_option_selection:bool=True
    maturity_max_first_part:float=7

    second_stage_min_option_strike_index:int=1
    second_stage_max_option_strike_index:int=4


    # Weighting options
    weight_type:CalibWeightType= CalibWeightType.UNIFORM

    # weight_type:CalibWeightType= CalibWeightType.MONEYNESS 
    # weight_type:CalibWeightType= CalibWeightType.DISTANCE_FROM_ATM
    # weight_type:CalibWeightType= CalibWeightType.INV_DISTANCE_FROM_ATM
    
    # weight_type:CalibWeightType= CalibWeightType.INV_MONEYNESS

    custom_weight:float=1.0
     
    #Correlation min
    correlation_min: float = -0.75##-0.25#-0.898#9999  # Minimum correlation value for calibration
    #Correlation max 
    correlation_max: float = 0.75##0.25##0.898#9999   # Maximum correlation value for calibration

    # Numerical parameters
    relative_tolerance: float = 1e-7#6 #1e-9#1e-8
    absolute_tolerance: float = 1e-7#6 #1e-9#1e-8
    gradient_tolerance: float = 1e-7#6   # Add gradient tolerance
    max_iterations: int = 50#100#500#1000
    
    # Additional stopping criteria
    min_cost_reduction: float = 1e-8   # Stop if improvement is too small
    consecutive_small_steps: int = 5   # Stop after N small improvements
    verbose: int = 2  # Whether to print detailed logs during calibration
    # JAX optimization
    use_jax: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'use_atm_only': self.use_atm_only,
            'min_maturity_option': self.min_maturity_option,
            'max_maturity_option': self.max_maturity_option,
            'pricing_method': self.pricing_method.value,
            'optimization_method': self.optimization_method.value,
            'mc_paths': self.mc_paths,
            'mc_timestep': self.mc_timestep,
            'use_multithreading': self.use_multithreading,
            'calibrate_on_vol': self.calibrate_on_vol,
            'calibrate_based_on_correlation': self.calibrate_based_on_correlation,
            'pseudo_inverse_smoothing': self.pseudo_inverse_smoothing,
            'relative_tolerance': self.relative_tolerance,
            'absolute_tolerance': self.absolute_tolerance,
            'max_iterations': self.max_iterations,
            'use_jax': self.use_jax
        }


@dataclass
class CalibrationResult:
    """Results from FX model calibration."""
    success: bool=False
    final_parameters: np.ndarray=None
    initial_parameters: np.ndarray=None
    parameter_names: List[str]=None
    
    # Error metrics
    final_error: float=np.finfo(float).max
    initial_error: float=np.finfo(float).max
    rmse_ois_price: Optional[float] = np.finfo(float).max
    rmse_ois_yield: Optional[float] = np.finfo(float).max
    rmse_option_price: Optional[float] = np.finfo(float).max
    rmse_option_vol: Optional[float] = np.finfo(float).max
    
    # Optimization details
    num_iterations: int = 0
    num_function_evaluations: int = 0
    optimization_time: float = 0.0
    optimizer_message: str = ""
    
    # Calibrated model
    calibrated_model: Optional[Any] = None
    
    # Additional diagnostics
    convergence_history: List[float] = field(default_factory=list)
    parameter_history: List[np.ndarray] = field(default_factory=list)
    
    def summary(self) -> str:
        """Generate summary of calibration results."""
        summary_lines = [
            "=" * 80,
            "FX MODEL CALIBRATION RESULTS",
            "=" * 80,
            f"Success: {self.success}",
            f"Final RMSE: {self.final_error:.8f}",
            f"Initial RMSE: {self.initial_error:.8f}",
            f"Improvement: {(1 - self.final_error/self.initial_error)*100:.2f}%",
            "",
            "Error Components:",
            f"  OIS Price RMSE: {self.rmse_ois_price:.8f}" if self.rmse_ois_price else "  OIS Price RMSE: N/A",
            f"  OIS Yield RMSE: {self.rmse_ois_yield:.8f}" if self.rmse_ois_yield else "  OIS Yield RMSE: N/A",
            f"  Option Price RMSE: {self.rmse_option_price:.8f}" if self.rmse_option_price else "  Option Price RMSE: N/A",
            f"  Option Vol RMSE: {self.rmse_option_vol:.8f}" if self.rmse_option_vol else "  Option Vol RMSE: N/A",
            "",
            "Optimization Details:",
            f"  Iterations: {self.num_iterations}",
            f"  Function Evaluations: {self.num_function_evaluations}",
            f"  Time: {self.optimization_time:.2f} seconds",
            f"  Message: {self.optimizer_message}",
            "",
            "Parameter Changes:",
        ]
        
        for i, name in enumerate(self.parameter_names):
            if i < len(self.initial_parameters) and i < len(self.final_parameters):
                initial = self.initial_parameters[i]
                final = self.final_parameters[i]
                change = (final - initial) / abs(initial) * 100 if initial != 0 else float('inf')
                summary_lines.append(f"  {name}: {initial:.6f} → {final:.6f} ({change:+.2f}%)")
        
        summary_lines.append("=" * 80)
        
        return "\n".join(summary_lines)


class ParameterBounds:
    """Parameter bounds for calibration."""
    
    def __init__(self, lower: np.ndarray, upper: np.ndarray):
        """
        Initialize parameter bounds.
        
        Parameters
        ----------
        lower : np.ndarray
            Lower bounds for parameters
        upper : np.ndarray
            Upper bounds for parameters
        """
        if len(lower) != len(upper):
            raise ValueError("Lower and upper bounds must have same length")
        
        self.lower = np.asarray(lower)
        self.upper = np.asarray(upper)
        
        # Validate bounds
        if np.any(self.lower > self.upper):
            raise ValueError("Lower bounds must be less than upper bounds")
    
    def __len__(self) -> int:
        """Return number of parameters."""
        return len(self.lower)
    
    def clip(self, parameters: np.ndarray) -> np.ndarray:
        """Clip parameters to bounds."""
        return np.clip(parameters, self.lower, self.upper)
    
    def is_within_bounds(self, parameters: np.ndarray) -> bool:
        """Check if parameters are within bounds."""
        return np.all(parameters >= self.lower) and np.all(parameters <= self.upper)
    
    def to_scipy_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convert to scipy optimization bounds format."""
        return (self.lower, self.upper)
    
    def to_list_bounds(self) -> List[Tuple[float, float]]:
        """Convert to list of tuples format."""
        return list(zip(self.lower, self.upper))


class FXCalibratorBase(ABC):
    """Abstract base class for FX model calibrators."""
    
    def __init__(
        self,
        daily_data: CurrencyPairDailyData,
        model: LRWFxModel,
        config: Optional[CalibrationConfig] = None
    ):
        """
        Initialize FX calibrator.
        
        Parameters
        ----------
        daily_data : CurrencyPairDailyData
            Market data for calibration
        model : LRWFxModel
            FX model to calibrate
        config : CalibrationConfig, optional
            Calibration configuration
        """
        self.daily_data = daily_data
        self.model = model
        # self.model.temporary_set_model_params()
        self.config = config or CalibrationConfig()
        
        # Initialize curves
        self._initialize_curves()
        
        # Process market data
        self._process_market_data()
    
    @abstractmethod
    def _initialize_curves(self) -> None:
        """Initialize interest rate curves."""
        pass
    
    @abstractmethod
    def _process_market_data(self) -> None:
        """Process raw market data for calibration."""
        pass
    
    @abstractmethod
    def calibrate(self, **kwargs) -> CalibrationResult:
        """
        Perform model calibration.
        
        Returns
        -------
        CalibrationResult
            Results of the calibration
        """
        pass
    
    @abstractmethod
    def get_calibration_instruments(self) -> List[Any]:
        """
        Get instruments used for calibration.
        
        Returns
        -------
        List[Any]
            List of calibration instruments
        """
        pass
    
    @abstractmethod
    def reprice_instruments(self) -> Dict[str, np.ndarray]:
        """
        Reprice all calibration instruments with current model.
        
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary of repriced values by instrument type
        """
        pass
    
    def validate_calibration(self, result: CalibrationResult) -> bool:
        """
        Validate calibration results.
        
        Parameters
        ----------
        result : CalibrationResult
            Calibration results to validate
            
        Returns
        -------
        bool
            True if calibration is valid
        """
        # Check if optimization was successful
        if not result.success:
            return False
        
        # Check if parameters are within bounds
        if hasattr(self, 'parameter_bounds'):
            if not self.parameter_bounds.is_within_bounds(result.final_parameters):
                return False
        
        # Check if error is acceptable
        if result.final_error > 1.0:  # Threshold can be configured
            return False
        
        # Check for NaN or infinite values
        if np.any(np.isnan(result.final_parameters)) or np.any(np.isinf(result.final_parameters)):
            return False
        
        return True
    
    def save_results(self, result: CalibrationResult, filepath: str) -> None:
        """
        Save calibration results to file.
        
        Parameters
        ----------
        result : CalibrationResult
            Calibration results to save
        filepath : str
            Path to save file
        """
        import pickle
        
        with open(filepath, 'wb') as f:
            pickle.dump(result, f)
    
    def load_results(self, filepath: str) -> CalibrationResult:
        """
        Load calibration results from file.
        
        Parameters
        ----------
        filepath : str
            Path to load file
            
        Returns
        -------
        CalibrationResult
            Loaded calibration results
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class ParameterActivation:
    """Manage which parameters are active in calibration."""
    
    def __init__(self, n_dimensions: int = 2):
        """
        Initialize parameter activation flags.
        
        Parameters
        ----------
        n_dimensions : int
            Number of dimensions in the model (default 2 for FX)
        """
        self.n = n_dimensions
        self.flags = self._initialize_flags()
    
    def _initialize_flags(self) -> Dict[str, bool]:
        """Initialize all parameter flags to False."""
        flags = {}
        
        # Alpha parameters
        flags['alpha_i'] = False
        flags['alpha_j'] = False
        
        # X0 matrix
        for i in range(self.n):
            for j in range(self.n):
                flags[f'x0_{i}{j}'] = False
        
        # Omega matrix
        for i in range(self.n):
            for j in range(self.n):
                flags[f'omega_{i}{j}'] = False
        
        # M matrix
        for i in range(self.n):
            for j in range(self.n):
                flags[f'm_{i}{j}'] = False
        
        # Sigma matrix
        for i in range(self.n):
            for j in range(self.n):
                flags[f'sigma_{i}{j}'] = False
        
        return flags
    
    def activate(self, parameter_names: Union[str, List[str]]) -> None:
        """Activate specified parameters."""
        if isinstance(parameter_names, str):
            parameter_names = [parameter_names]
        
        for name in parameter_names:
            if name in self.flags:
                self.flags[name] = True
            else:
                raise ValueError(f"Unknown parameter: {name}")
    
    def deactivate(self, parameter_names: Union[str, List[str]]) -> None:
        """Deactivate specified parameters."""
        if isinstance(parameter_names, str):
            parameter_names = [parameter_names]
        
        for name in parameter_names:
            if name in self.flags:
                self.flags[name] = False
            else:
                raise ValueError(f"Unknown parameter: {name}")
    
    def get_active_count(self) -> int:
        """Get number of active parameters."""
        return sum(self.flags.values())
    
    def get_active_names(self) -> List[str]:
        """Get list of active parameter names."""
        return [name for name, active in self.flags.items() if active]
    
    def to_list(self) -> List[bool]:
        """Convert to ordered list of booleans."""
        # Define parameter order
        ordered_flags = []
        
        # Alphas
        ordered_flags.append(self.flags['alpha_i'])
        ordered_flags.append(self.flags['alpha_j'])
        
        # X0 diagonal
        for i in range(self.n):
            ordered_flags.append(self.flags[f'x0_{i}{i}'])
        
        # X0 off-diagonal
        if self.n == 2:
            ordered_flags.append(self.flags['x0_01'])
        
        # Omega diagonal
        for i in range(self.n):
            ordered_flags.append(self.flags[f'omega_{i}{i}'])
        
        # Omega off-diagonal
        if self.n == 2:
            ordered_flags.append(self.flags['omega_01'])
        
        # M diagonal
        for i in range(self.n):
            ordered_flags.append(self.flags[f'm_{i}{i}'])
        
        # Sigma diagonal
        for i in range(self.n):
            ordered_flags.append(self.flags[f'sigma_{i}{i}'])
        
        # Sigma off-diagonal
        if self.n == 2:
            ordered_flags.append(self.flags['sigma_01'])
        
        return ordered_flags
    
    def from_list(self, flag_list: List[bool]) -> None:
        """Set flags from ordered list of booleans."""
        if len(flag_list) != len(self.to_list()):
            raise ValueError("Flag list length mismatch")
        
        # Would implement the reverse mapping
        # This is a simplified version
        pass
