"""
Calibration constraints for LRW Jump models.

This module handles parameter bounds, validation, and constraints
for the calibration process.
"""

from typing import List, Tuple, Optional, Dict
import numpy as np
import jax.numpy as jnp

from ..models.interest_rate.lrw_model import LRWModel


class CalibrationConstraints:
    """
    Manages constraints and bounds for LRW Jump model calibration.
    """
    
    def __init__(self, model: LRWModel):
        """
        Initialize constraints manager.
        
        Parameters
        ----------
        model : LRWModel
            The model being calibrated
        """
        self.model = model
        self.n = model.n
        
        # Default bounds
        self.default_lower_bounds = self._initialize_default_lower_bounds()
        self.default_upper_bounds = self._initialize_default_upper_bounds()
        
        # Custom bounds (can be set by user)
        self.lower_bounds = None
        self.upper_bounds = None
        
    def _initialize_default_lower_bounds(self) -> Dict[str, np.ndarray]:
        """Initialize default lower bounds for parameters."""
        n = self.n
        
        bounds = {
            'alpha': 0.001,
            'x0_diag': np.array([0.0001] * n),
            'x0_off_diag': -0.1,
            'omega_diag': np.array([0.0001] * n),
            'omega_off_diag': -0.1,
            'm_diag': np.array([-10.0] * n),
            'sigma_diag': np.array([0.0001] * n),
            'sigma_off_diag': -0.5
        }
        
        return bounds
        
    def _initialize_default_upper_bounds(self) -> Dict[str, np.ndarray]:
        """Initialize default upper bounds for parameters."""
        n = self.n
        
        bounds = {
            'alpha': 0.2,
            'x0_diag': np.array([1.0] * n),
            'x0_off_diag': 0.1,
            'omega_diag': np.array([1.0] * n),
            'omega_off_diag': 0.1,
            'm_diag': np.array([-0.001] * n),
            'sigma_diag': np.array([1.0] * n),
            'sigma_off_diag': 0.5
        }
        
        return bounds
        
    def set_custom_bounds(
        self,
        lower_model: Optional[LRWModel] = None,
        upper_model: Optional[LRWModel] = None
    ):
        """
        Set custom bounds from model instances.
        
        Parameters
        ----------
        lower_model : LRWModel, optional
            Model containing lower bounds
        upper_model : LRWModel, optional
            Model containing upper bounds
        """
        if lower_model:
            self.lower_bounds = self._extract_bounds_from_model(lower_model)
        if upper_model:
            self.upper_bounds = self._extract_bounds_from_model(upper_model)
            
    def _extract_bounds_from_model(self, model: LRWModel) -> Dict[str, np.ndarray]:
        """Extract parameter bounds from a model instance."""
        bounds = {
            'alpha': model.alpha,
            'x0_diag': np.diag(model.x0),
            'x0_off_diag': model.x0[0, 1],
            'omega_diag': np.diag(model.omega),
            'omega_off_diag': model.omega[0, 1],
            'm_diag': np.diag(model.m),
            'sigma_diag': np.diag(model.sigma),
            'sigma_off_diag': model.sigma[0, 1]
        }
        return bounds
        
    def get_parameter_bounds(
        self,
        params_activation: List[bool]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get bounds for active parameters.
        
        Parameters
        ----------
        params_activation : List[bool]
            Activation flags for parameters
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Lower and upper bounds arrays
        """
        lower_bounds = self.lower_bounds or self.default_lower_bounds
        upper_bounds = self.upper_bounds or self.default_upper_bounds
        
        lower_array = []
        upper_array = []
        
        # Alpha
        if params_activation[0]:
            lower_array.append(lower_bounds['alpha'])
            upper_array.append(upper_bounds['alpha'])
            
        # x0 diagonal
        start_idx = 1
        for i in range(self.n):
            if params_activation[start_idx + i]:
                lower_array.append(lower_bounds['x0_diag'][i])
                upper_array.append(upper_bounds['x0_diag'][i])
                
        # x0 off-diagonal
        if params_activation[start_idx + 2]:
            lower_array.append(lower_bounds['x0_off_diag'])
            upper_array.append(upper_bounds['x0_off_diag'])
            
        # omega diagonal
        start_idx = 4
        for i in range(self.n):
            if params_activation[start_idx + i]:
                lower_array.append(lower_bounds['omega_diag'][i])
                upper_array.append(upper_bounds['omega_diag'][i])
                
        # omega off-diagonal
        if params_activation[start_idx + 2]:
            lower_array.append(lower_bounds['omega_off_diag'])
            upper_array.append(upper_bounds['omega_off_diag'])
            
        # m diagonal
        start_idx = 7
        for i in range(self.n):
            if params_activation[start_idx + i]:
                lower_array.append(lower_bounds['m_diag'][i])
                upper_array.append(upper_bounds['m_diag'][i])
                
        # sigma diagonal
        start_idx = 9
        for i in range(self.n):
            if params_activation[start_idx + i]:
                lower_array.append(lower_bounds['sigma_diag'][i])
                upper_array.append(upper_bounds['sigma_diag'][i])
                
        # sigma off-diagonal
        if params_activation[start_idx + 2]:
            lower_array.append(lower_bounds['sigma_off_diag'])
            upper_array.append(upper_bounds['sigma_off_diag'])
            
        return (np.array(lower_array), np.array(upper_array))
        
    def check_gindikin_condition(self, omega: Optional[np.ndarray] = None) -> bool:
        """
        Check if Gindikin condition is satisfied.
        
        Parameters
        ----------
        omega : np.ndarray, optional
            Omega matrix to check. If None, uses model's omega
            
        Returns
        -------
        bool
            True if Gindikin condition is satisfied
        """
        if omega is None:
            omega = self.model.omega
            
        sigma = self.model.sigma
        n = self.n
        
        # Gindikin condition: omega - (n+1) * sigma @ sigma.T > 0
        temp = omega - (n + 1) * sigma @ sigma.T
        
        # Check if positive definite
        try:
            eigenvalues = np.linalg.eigvals(temp)
            return np.all(eigenvalues > 0)
        except:
            return False
            
    def apply_correlation_constraint(
        self,
        value: float,
        diag1: float,
        diag2: float,
        use_correlation: bool = True
    ) -> float:
        """
        Apply correlation constraint to off-diagonal elements.
        
        Parameters
        ----------
        value : float
            Raw parameter value
        diag1 : float
            First diagonal element
        diag2 : float
            Second diagonal element
        use_correlation : bool, default=True
            Whether to interpret value as correlation
            
        Returns
        -------
        float
            Constrained value
        """
        if use_correlation:
            # Interpret as correlation coefficient
            return value * np.sqrt(abs(diag1 * diag2))
        else:
            # Use raw value
            return value
            
    def check_parameter_ratios(
        self,
        calibrated_model: LRWModel,
        base_model: LRWModel,
        param_type: str,
        max_ratio: float
    ) -> bool:
        """
        Check if parameter ratios are within acceptable bounds.
        
        Parameters
        ----------
        calibrated_model : LRWModel
            Calibrated model
        base_model : LRWModel
            Base model for comparison
        param_type : str
            Type of parameters to check ('ois' or 'spread')
        max_ratio : float
            Maximum acceptable ratio
            
        Returns
        -------
        bool
            True if parameters were replaced (ratio exceeded)
        """
        if param_type == 'ois':
            # Check OIS parameters (x0[0,0], omega[0,0], m[0,0])
            params_calib = [
                calibrated_model.x0[0, 0],
                calibrated_model.omega[0, 0],
                calibrated_model.m[0, 0]
            ]
            params_base = [
                base_model.x0[0, 0],
                base_model.omega[0, 0],
                base_model.m[0, 0]
            ]
        elif param_type == 'spread':
            # Check spread parameters (x0[1,1], omega[1,1], m[1,1])
            params_calib = [
                calibrated_model.x0[1, 1],
                calibrated_model.omega[1, 1],
                calibrated_model.m[1, 1]
            ]
            params_base = [
                base_model.x0[1, 1],
                base_model.omega[1, 1],
                base_model.m[1, 1]
            ]
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")
            
        # Check ratios
        for calib, base in zip(params_calib, params_base):
            if abs(base) > 1e-10:  # Avoid division by zero
                ratio = abs(calib / base)
                if ratio > max_ratio or ratio < 1 / max_ratio:
                    return True
                    
        return False
        
    def validate_parameters(self, params: Dict[str, np.ndarray]) -> Dict[str, bool]:
        """
        Validate all parameters.
        
        Parameters
        ----------
        params : Dict[str, np.ndarray]
            Parameters to validate
            
        Returns
        -------
        Dict[str, bool]
            Validation results for each parameter
        """
        results = {}
        
        # Check positivity constraints
        results['alpha_positive'] = params.get('alpha', 0) > 0
        results['x0_diag_positive'] = np.all(params.get('x0_diag', []) > 0)
        results['omega_diag_positive'] = np.all(params.get('omega_diag', []) > 0)
        results['sigma_diag_positive'] = np.all(params.get('sigma_diag', []) > 0)
        
        # Check mean reversion
        results['m_diag_negative'] = np.all(params.get('m_diag', []) < 0)
        
        # Check Gindikin condition
        if 'omega' in params and 'sigma' in params:
            results['gindikin_satisfied'] = self.check_gindikin_condition(params['omega'])
            
        return results
        
    def apply_bounds(
        self,
        params: np.ndarray,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray
    ) -> np.ndarray:
        """
        Apply bounds to parameters.
        
        Parameters
        ----------
        params : np.ndarray
            Parameter values
        lower_bounds : np.ndarray
            Lower bounds
        upper_bounds : np.ndarray
            Upper bounds
            
        Returns
        -------
        np.ndarray
            Bounded parameters
        """
        return np.clip(params, lower_bounds, upper_bounds)
