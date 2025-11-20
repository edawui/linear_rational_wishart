"""
Objective functions for FX model calibration.

This module provides various objective functions for calibrating FX models,
including bond pricing objectives, option pricing objectives, and hybrid objectives.
"""

from abc import ABC, abstractmethod
from lib2to3.fixes.fix_tuple_params import is_docstring
from typing import List, Tuple, Optional, Union, Dict, Any
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap
    from functools import partial
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = np
    jit = lambda f: f
    vmap = lambda f: f
    partial = partial

from ...data.data_fx_market_data import CurrencyPairDailyData
from ...utils.jax_utils import ensure_jax_array #, ensure_numpy_array


class ObjectiveFunction(ABC):
    """Abstract base class for objective functions."""
    
    def __init__(self, power: float = 2.0, use_jax: bool = False):
        """
        Initialize objective function.
        
        Parameters
        ----------
        power : float
            Power for error calculation (default 2.0 for squared errors)
        use_jax : bool
            Whether to use JAX optimizations
        """
        self.power = power
        self.use_jax = use_jax and JAX_AVAILABLE
        self.call_counter = 0
    
    @abstractmethod
    def __call__(self, parameters: np.ndarray) -> Union[float, np.ndarray]:
        """
        Evaluate objective function.
        
        Parameters
        ----------
        parameters : np.ndarray
            Model parameters
            
        Returns
        -------
        float or np.ndarray
            Objective function value(s)
        """
        pass
    
    def reset_counter(self) -> None:
        """Reset function call counter."""
        self.call_counter = 0


class BondPricingObjective(ObjectiveFunction):
    """Objective function for bond pricing calibration."""
    
    def __init__(
        self,
        calibrator: Any,  # Avoid circular import
        min_tenor: float = 1.0,
        max_tenor: float = 11.0,
        calibrate_on_price: bool = True,
        calibrate_on_domestic: bool = True,
        calibrate_on_foreign: bool = True,        
        power: float = 2.0,
        use_jax: bool = False
    ):
        """
        Initialize bond pricing objective.
        
        Parameters
        ----------
        calibrator : FXCalibrator
            Calibrator instance with model and market data
        min_tenor : float
            Minimum tenor for calibration
        max_tenor : float
            Maximum tenor for calibration
        calibrate_on_price : bool
            If True, calibrate on prices; if False, calibrate on yields
        power : float
            Power for error calculation
        use_jax : bool
            Whether to use JAX optimizations
        """
        super().__init__(power, use_jax)
        self.calibrator = calibrator
        self.min_tenor = min_tenor
        self.max_tenor = max_tenor
        self.calibrate_on_price = calibrate_on_price
        
        if not calibrate_on_domestic and not calibrate_on_foreign:
                raise ValueError("At least one of 'calibrate_on_domestic' or 'calibrate_on_foreign' must be True.")
        if calibrate_on_domestic and not calibrate_on_foreign:
                print("Warning: Only domestic currency bonds will be calibrated.")
        if not calibrate_on_domestic and calibrate_on_foreign:
                print("Warning: Only foreign currency bonds will be calibrated.")

        self.calibrate_on_domestic=calibrate_on_domestic
        self.calibrate_on_foreign=calibrate_on_foreign


        if self.use_jax:
            self._setup_jax_functions()
    
    def _setup_jax_functions(self) -> None:
        """Setup JAX-optimized functions."""
        
        @jit
        def compute_bond_errors(market_prices, model_prices, power):
            """JAX-optimized bond error calculation."""
            errors = market_prices - model_prices
            return jnp.power(jnp.abs(errors), power)
        
        self._jax_compute_errors = compute_bond_errors
    
    def __call__(self, parameters: np.ndarray) -> np.ndarray:
        """
        Calculate bond pricing errors.
        
        Parameters
        ----------
        parameters : np.ndarray
            Model parameters
            
        Returns
        -------
        np.ndarray
            Array of pricing errors
        """
        self.call_counter += 1
        # print(f"Bond objective called with counter: {self.call_counter}")
        # print(f"Bond objective called with parameters: {parameters}")
        # Set model parameters
        self.calibrator.set_model_parameters(parameters)
        
        # Reprice bonds
        self.calibrator.reprice_bonds()
        
        errors = []
        if self.calibrate_on_domestic:
            # Domestic currency bonds
            for _, ois_data in self.calibrator.daily_data.domestic_currency_daily_data.ois_rate_data.iterrows():
                tenor = ois_data["TimeToMat"]
                if self.min_tenor <= tenor <= self.max_tenor:
                    obj = ois_data["Object"]
                    if self.calibrate_on_price:
                        error = obj.market_zc_price - obj.model_zc_price
                    else:
                        error = obj.market_zc_rate - obj.model_zc_rate
                    errors.append(error)
        if self.calibrate_on_foreign:
        # Foreign currency bonds
            for _, ois_data in self.calibrator.daily_data.foreign_currency_daily_data.ois_rate_data.iterrows():
                tenor = ois_data["TimeToMat"]
                if self.min_tenor <= tenor <= self.max_tenor:
                    obj = ois_data["Object"]
                    if self.calibrate_on_price:
                        error = obj.market_zc_price - obj.model_zc_price
                    else:
                        error = obj.market_zc_rate - obj.model_zc_rate
                    errors.append(error)
        
        mse= 1e4*np.sum( np.power(errors, 2) / len(errors) if errors else 0.0)
        # print(f"Bond objective - MSE: {mse:.4f}, N: {len(errors)}")
        # print(f"Bond objective - MSE: {mse:.4f}, N: {len(errors)}")
        # print(np.array(errors))
        return np.array(errors)
    
    def compute_rmse(self, parameters: np.ndarray) -> float:
        """
        Compute RMSE for bond pricing.
        
        Parameters
        ----------
        parameters : np.ndarray
            Model parameters
            
        Returns
        -------
        float
            Root mean squared error
        """
        errors = self.__call__(parameters)
        
        if self.use_jax:
            errors_jax = ensure_jax_array(errors)
            squared_errors = self._jax_compute_errors(errors_jax, jnp.zeros_like(errors_jax), self.power)
            mse = jnp.mean(squared_errors)
            print(f"In JAX: Bond objective - MSE: {mse:.4f}, N: {len(errors)}")
            return float(10000 * jnp.sqrt(mse))
        else:
            squared_errors = np.power(np.abs(errors), self.power)
            mse = np.mean(squared_errors)
            print(f"NON JAX: Bond objective - MSE: {mse:.4f}, N: {len(errors)}")
            return 10000 * np.sqrt(mse)


class OptionPricingObjective(ObjectiveFunction):
    """Objective function for option pricing calibration."""
    
    def __init__(
        self,
        calibrator: Any,
        calibrate_on_vol: bool = False,
        power: float = 2.0,
        use_jax: bool = False,
        regularization: float = 1e-6 
    ):
        """
        Initialize option pricing objective.
        
        Parameters
        ----------
        calibrator : FXCalibrator
            Calibrator instance with model and market data
        calibrate_on_vol : bool
            If True, calibrate on implied volatilities
        power : float
            Power for error calculation
        use_jax : bool
            Whether to use JAX optimizations
        """
        super().__init__(power, use_jax)
        self.calibrator = calibrator
        self.calibrate_on_vol = calibrate_on_vol
        self.regularization = regularization
        self.previous_params = None
        self.previous_errors = None

        if self.use_jax:
            self._setup_jax_functions()
    
    def _setup_jax_functions(self) -> None:
        """Setup JAX-optimized functions."""
        
        @jit
        def compute_option_errors(market_values, model_values, power):
            """JAX-optimized option error calculation."""
            errors = market_values - model_values
            return 1e4 * jnp.power(jnp.abs(errors), power)
        
        self._jax_compute_errors = compute_option_errors
    
    def __call__(self, parameters: np.ndarray) -> np.ndarray:
        """
        Calculate option pricing errors.
        
        Parameters
        ----------
        parameters : np.ndarray
            Model parameters
            
        Returns
        -------
        np.ndarray
            Array of pricing errors
        """
        self.call_counter += 1
        
        # Check for NaN/inf in parameters
        if not np.all(np.isfinite(parameters)):
            print(f"WARNING: Non-finite parameters detected: {parameters}")
            # Return large errors to discourage this point
            return np.full(len(self.calibrator.calibration_options), 1e10)

        # Store previous good state
        if self.previous_params is not None and self.previous_errors is not None:
            if np.all(np.isfinite(self.previous_errors)):
                last_good_params = self.previous_params.copy()
                last_good_errors = self.previous_errors.copy()

        try:
            # Debug print
            # print("====================================================================")
            # print(f"\n\nOption objective called with parameters: {parameters}")
        
            # Set model parameters
            self.calibrator.set_model_parameters(parameters)
        
            # self.calibrator.model.print_model()
            # Reprice options
            self.calibrator.reprice_options()
            mul=1e4
            mul=1.0
            errors = []
            # print(self.calibrate_on_vol)
            for option in self.calibrator.calibration_options:
                # print(str(option))
                weight=option.weight if hasattr(option, 'weight') else 1.0
                # if self.calibrate_on_vol:
                #     error = mul * weight * (option.market_vol - option.model_vol)
                # else:
                #     error = mul * weight * (option.market_price - option.model_price)
                # errors.append(error)
                if self.calibrate_on_vol:
                    market_val = option.market_vol
                    model_val = option.model_vol
                else:
                    market_val = option.market_price
                    model_val = option.model_price
                # Check for NaN/inf in prices/vols
                if not np.isfinite(model_val) or not np.isfinite(market_val):
                    print(f"WARNING: Non-finite value - Market: {market_val}, Model: {model_val}")
                    # Use a large error instead of NaN
                    error = 1e10 * weight
                else:
                    error = weight * (market_val - model_val)

                errors.append(error)
            
            errors = np.array(errors)
            # mse= 1e4*np.sum( np.power(errors, 2) / len(errors) if errors else 0.0)
            mse= 1e4*np.mean(np.power(errors, 2))

            # print(f"\nOption objective - MSE: {mse:.4f}, N: {len(errors)}, nb calls:{self.call_counter}")
            # return np.array(errors)
            
            # Check if errors are finite
            if not np.all(np.isfinite(errors)):
                print(f"WARNING: Non-finite errors detected. Using fallback.")
                # Replace NaN/inf with large values
                errors = np.nan_to_num(errors, nan=1e10, posinf=1e10, neginf=-1e10)
            
            # Store good state
            self.previous_params = parameters.copy()
            self.previous_errors = errors.copy()
            
            # Add small regularization to prevent numerical issues
            # if self.regularization > 0:
            #     reg_term = self.regularization * np.linalg.norm(parameters)
            #     errors = np.append(errors, reg_term)
            
            return errors
            
        except Exception as e:
            print(f"ERROR in objective evaluation: {e}")
            # Return large errors on any exception
            return np.full(len(self.calibrator.calibration_options), 1e10)


    def compute_rmse(self, parameters: np.ndarray) -> float:
        """
        Compute RMSE for option pricing.
        
        Parameters
        ----------
        parameters : np.ndarray
            Model parameters
            
        Returns
        -------
        float
            Root mean squared error
        """
        errors = self.__call__(parameters)
        
        if self.use_jax:
            errors_jax = ensure_jax_array(errors)
            squared_errors = jnp.power(jnp.abs(errors_jax), self.power)
            mse = jnp.mean(squared_errors)
            rmse = jnp.sqrt(mse)
        else:
            squared_errors = np.power(np.abs(errors), self.power)
            mse = np.mean(squared_errors)
            rmse = np.sqrt(mse)
        
        n_options = len(self.calibrator.calibration_options)
        error_sum = float(np.sum(squared_errors))
        
        print(f"Option objective - Error: {error_sum:.4f}, RMSE: {rmse:.4f}, N: {n_options}")
        
        return float(rmse)


class HybridObjective(ObjectiveFunction):
    """Combined objective function for bonds and options."""
    
    def __init__(
        self,
        bond_objective: BondPricingObjective,
        option_objective: OptionPricingObjective,
        bond_weight: float = 0.5,
        option_weight: float = 0.5,
        use_jax: bool = False
    ):
        """
        Initialize hybrid objective.
        
        Parameters
        ----------
        bond_objective : BondPricingObjective
            Bond pricing objective
        option_objective : OptionPricingObjective
            Option pricing objective
        bond_weight : float
            Weight for bond errors
        option_weight : float
            Weight for option errors
        use_jax : bool
            Whether to use JAX optimizations
        """
        super().__init__(power=2.0, use_jax=use_jax)
        self.bond_objective = bond_objective
        self.option_objective = option_objective
        self.bond_weight = bond_weight
        self.option_weight = option_weight
        
        # Ensure weights sum to 1
        total_weight = bond_weight + option_weight
        self.bond_weight /= total_weight
        self.option_weight /= total_weight
    
    def __call__(self, parameters: np.ndarray) -> Union[float, np.ndarray]:
        """
        Calculate combined objective.
        
        Parameters
        ----------
        parameters : np.ndarray
            Model parameters
            
        Returns
        -------
        float or np.ndarray
            Combined objective value(s)
        """
        self.call_counter += 1
        
        # Get individual objectives
        bond_errors = self.bond_objective(parameters)
        option_errors = self.option_objective(parameters)
        
        # For least squares, return concatenated errors
        if isinstance(bond_errors, np.ndarray) and isinstance(option_errors, np.ndarray):
            weighted_bond_errors = np.sqrt(self.bond_weight) * bond_errors
            weighted_option_errors = np.sqrt(self.option_weight) * option_errors
            return np.concatenate([weighted_bond_errors, weighted_option_errors])
        
        # For scalar objectives, return weighted sum
        bond_obj = np.sum(bond_errors**2) if isinstance(bond_errors, np.ndarray) else bond_errors
        option_obj = np.sum(option_errors**2) if isinstance(option_errors, np.ndarray) else option_errors
        
        return self.bond_weight * bond_obj + self.option_weight * option_obj
    
    def compute_rmse(self, parameters: np.ndarray) -> Tuple[float, float, float]:
        """
        Compute RMSE for both objectives.
        
        Parameters
        ----------
        parameters : np.ndarray
            Model parameters
            
        Returns
        -------
        Tuple[float, float, float]
            Bond RMSE, Option RMSE, Combined RMSE
        """
        bond_rmse = self.bond_objective.compute_rmse(parameters)
        option_rmse = self.option_objective.compute_rmse(parameters)
        
        # Combined RMSE using weights
        combined_rmse = np.sqrt(
            self.bond_weight * bond_rmse**2 + 
            self.option_weight * option_rmse**2
        )
        
        return bond_rmse, option_rmse, combined_rmse


class RegularizedObjective(ObjectiveFunction):
    """Objective function with regularization."""
    
    def __init__(
        self,
        base_objective: ObjectiveFunction,
        regularization_weight: float = 0.01,
        reference_parameters: Optional[np.ndarray] = None,
        use_jax: bool = False
    ):
        """
        Initialize regularized objective.
        
        Parameters
        ----------
        base_objective : ObjectiveFunction
            Base objective function
        regularization_weight : float
            Weight for regularization term
        reference_parameters : np.ndarray, optional
            Reference parameters for regularization
        use_jax : bool
            Whether to use JAX optimizations
        """
        super().__init__(power=2.0, use_jax=use_jax)
        self.base_objective = base_objective
        self.regularization_weight = regularization_weight
        self.reference_parameters = reference_parameters
    
    def __call__(self, parameters: np.ndarray) -> Union[float, np.ndarray]:
        """
        Calculate regularized objective.
        
        Parameters
        ----------
        parameters : np.ndarray
            Model parameters
            
        Returns
        -------
        float or np.ndarray
            Regularized objective value(s)
        """
        self.call_counter += 1
        
        # Base objective
        base_value = self.base_objective(parameters)
        
        # Regularization term
        if self.reference_parameters is not None:
            diff = parameters - self.reference_parameters
        else:
            diff = parameters
        
        if self.use_jax:
            diff_jax = ensure_jax_array(diff)
            reg_term = self.regularization_weight * jnp.sum(diff_jax**2)
        else:
            reg_term = self.regularization_weight * np.sum(diff**2)
        
        # For array objectives (least squares), append regularization
        if isinstance(base_value, np.ndarray):
            reg_array = np.sqrt(reg_term) * np.ones(len(parameters))
            return np.concatenate([base_value, reg_array])
        
        # For scalar objectives, add regularization
        return base_value + reg_term


# Factory function for creating objectives
def create_objective(
    objective_type: str,
    calibrator: Any,
    **kwargs
) -> ObjectiveFunction:
    """
    Factory function for creating objective functions.
    
    Parameters
    ----------
    objective_type : str
        Type of objective ('bond', 'option', 'hybrid', 'regularized')
    calibrator : FXCalibrator
        Calibrator instance
    **kwargs
        Additional arguments for the objective
        
    Returns
    -------
    ObjectiveFunction
        Created objective function
    """
    if objective_type == 'bond':
        return BondPricingObjective(calibrator, **kwargs)
    
    elif objective_type == 'option':
        return OptionPricingObjective(calibrator, **kwargs)
    
    elif objective_type == 'hybrid':
        bond_obj = BondPricingObjective(calibrator, **kwargs)
        option_obj = OptionPricingObjective(calibrator, **kwargs)
        return HybridObjective(bond_obj, option_obj, **kwargs)
    
    elif objective_type == 'regularized':
        base_type = kwargs.pop('base_type', 'option')
        base_obj = create_objective(base_type, calibrator, **kwargs)
        return RegularizedObjective(base_obj, **kwargs)
    
    else:
        raise ValueError(f"Unknown objective type: {objective_type}")
