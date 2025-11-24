"""
Parameter management utilities for calibration.

This module provides utilities for managing model parameters during calibration,
including parameter extraction, setting, and transformation.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import copy

from ..models.fx.lrw_fx import LRWFxModel 


class ParameterManager:
    """Manages model parameters for calibration."""
    
    def __init__(self, model: LRWFxModel):
        """
        Initialize parameter manager.
        
        Parameters
        ----------
        model : LRWFxModel
            FX model instance
        """
        self.model = model
        self.n = model.n  # Dimension
    
    def get_parameters(
        self,
        activation_flags: List[bool],
        as_correlation: bool = True,
        default_correlation:float = None
    ) -> np.ndarray:
        """
        Extract active parameters from model.
        
        Parameters
        ----------
        activation_flags : List[bool]
            Boolean flags indicating which parameters are active
        as_correlation : bool
            If True, extract off-diagonal elements as correlations
            
        Returns
        -------
        np.ndarray
            Array of active parameter values
        """
        if len(activation_flags) != self._get_total_param_count():
            raise ValueError(f"Expected {self._get_total_param_count()} activation flags, got {len(activation_flags)}")
        
        params = []
        idx = 0
        
        # Alpha parameters
        if activation_flags[idx]:
            params.append(self.model.alpha_i)
        idx += 1
        
        if activation_flags[idx]:
            params.append(self.model.alpha_j)
        idx += 1
        
        # X0 matrix
        x0 = self.model.x0
        # Diagonal elements
        for i in range(self.n):
            if activation_flags[idx]:
                params.append(x0[i, i])
            idx += 1
        
        # Off-diagonal elements (correlation)
        if activation_flags[idx]:
            if as_correlation and self.n == 2:
                # Convert covariance to correlation
                corr = x0[0, 1] / np.sqrt(x0[0, 0] * x0[1, 1])
                if default_correlation is not None:
                    corr =default_correlation # np.clip(corr, -default_correlation, default_correlation)
                params.append(corr)
            else:
                params.append(x0[0, 1])
        idx += 1
        
        # Omega matrix
        omega = self.model.omega
        # Diagonal elements
        for i in range(self.n):
            if activation_flags[idx]:
                params.append(omega[i, i])
            idx += 1
        
        # Off-diagonal elements (correlation)
        if activation_flags[idx]:
            if as_correlation and self.n == 2:
                corr = omega[0, 1] / np.sqrt(omega[0, 0] * omega[1, 1])
                if default_correlation is not None:
                    corr =default_correlation # np.clip(corr, -default_correlation, default_correlation)
              
                params.append(corr)
            else:
                params.append(omega[0, 1])
        idx += 1
        
        # M matrix (only diagonal for FX model)
        m = self.model.m
        for i in range(self.n):
            if activation_flags[idx]:
                params.append(m[i, i])
            idx += 1
        
        # Sigma matrix
        sigma = self.model.sigma
        # Diagonal elements
        for i in range(self.n):
            if activation_flags[idx]:
                params.append(sigma[i, i])
            idx += 1
        
        # Off-diagonal elements (correlation)
        if activation_flags[idx]:
            if as_correlation and self.n == 2:
                corr = sigma[0, 1] / np.sqrt(sigma[0, 0] * sigma[1, 1])
                if default_correlation is not None:
                    corr =default_correlation # np.clip(corr, -default_correlation, default_correlation)
                params.append(corr)
            else:
                params.append(sigma[0, 1])
        idx += 1
        
        return np.array(params)
    
    def set_parameters(
        self,
        params: np.ndarray,
        activation_flags: List[bool],
        as_correlation: bool = True
    ) -> None:
        """
        Set model parameters from array.
        
        Parameters
        ----------
        params : np.ndarray
            Parameter values to set
        activation_flags : List[bool]
            Boolean flags indicating which parameters are active
        as_correlation : bool
            If True, interpret off-diagonal elements as correlations
        """
        n_active = sum(activation_flags)
        if len(params) != n_active:
            raise ValueError(f"Expected {n_active} parameters, got {len(params)}")
        
        # Get current matrices
        alpha_i = self.model.alpha_i
        alpha_j = self.model.alpha_j
        x0 = self.model.x0.copy()
        omega = self.model.omega.copy()
        m = self.model.m.copy()
        sigma = self.model.sigma.copy()
        
        param_idx = 0
        flag_idx = 0
        
        # Alpha parameters
        if activation_flags[flag_idx]:
            alpha_i = params[param_idx]
            param_idx += 1
        flag_idx += 1
        
        if activation_flags[flag_idx]:
            alpha_j = params[param_idx]
            param_idx += 1
        flag_idx += 1
        
        # X0 matrix
        # Diagonal elements
        for i in range(self.n):
            if activation_flags[flag_idx]:
                x0=x0.at[i, i].set(params[param_idx])
                # x0[i, i] = params[param_idx]
                param_idx += 1
            flag_idx += 1
        
        # Off-diagonal elements
        if activation_flags[flag_idx]:
            if as_correlation and self.n == 2:
                # Convert correlation to covariance
                corr = params[param_idx]
                cov = corr * np.sqrt(x0[0, 0] * x0[1, 1])
                x0=x0.at[0, 1].set(cov)
                x0=x0.at[1, 0].set(cov)
            else:
                x0=x0.at[0, 1].set( params[param_idx])
                x0=x0.at[1, 0].set( params[param_idx])
                # x0[0, 1] = x0[1, 0] = params[param_idx]
            param_idx += 1
        flag_idx += 1
        
        # Omega matrix
        # Diagonal elements
        for i in range(self.n):
            if activation_flags[flag_idx]:
                omega=omega.at[i, i].set( params[param_idx])
                param_idx += 1
            flag_idx += 1
        
        # Off-diagonal elements
        if activation_flags[flag_idx]:
            if as_correlation and self.n == 2:
                corr = params[param_idx]
                cov = corr * np.sqrt(omega[0, 0] * omega[1, 1])
                omega=omega.at[0, 1].set(cov)
                omega=omega.at[1, 0].set(cov)
                # omega[0, 1] = omega[1, 0] = cov
            else:
                # omega[0, 1] = omega[1, 0] = params[param_idx]
                omega=omega.at[0, 1].set(params[param_idx])
                omega=omega.at[1, 0].set(params[param_idx])
            param_idx += 1
        flag_idx += 1
        
        # M matrix
        for i in range(self.n):
            if activation_flags[flag_idx]:
                m=m.at[i, i].set(params[param_idx])
                param_idx += 1
            flag_idx += 1
        
        # Sigma matrix
        # Diagonal elements
        for i in range(self.n):
            if activation_flags[flag_idx]:
                sigma=sigma.at[i, i].set(params[param_idx])
                param_idx += 1
            flag_idx += 1
        
        # Off-diagonal elements
        if activation_flags[flag_idx]:
            if as_correlation and self.n == 2:
                corr = params[param_idx]
                cov = corr * np.sqrt(sigma[0, 0] * sigma[1, 1])
                sigma=sigma.at[0, 1].set(cov)
                sigma=sigma.at[1, 0].set(cov)
            else:
                # sigma[0, 1] = sigma[1, 0] = params[param_idx]
                sigma=sigma.at[0, 1].set(params[param_idx])
                sigma=sigma.at[1, 0].set(params[param_idx])

            param_idx += 1
        flag_idx += 1
        
        # Update model
        self.model.set_model_params(
            self.n, x0, omega, m, sigma,
            alpha_i, self.model.u_i,
            alpha_j, self.model.u_j,
            self.model.fx_spot
        )
    
    def get_parameters_from_model(
        self,
        model: LRWFxModel,
        activation_flags: List[bool],
        as_correlation: bool = True,
        default_correlation:float = None
    ) -> np.ndarray:
        """
        Extract parameters from a different model instance.
        
        Parameters
        ----------
        model : LRWFxModel
            Model to extract parameters from
        activation_flags : List[bool]
            Boolean flags indicating which parameters to extract
        as_correlation : bool
            If True, extract off-diagonal elements as correlations
            
        Returns
        -------
        np.ndarray
            Array of parameter values
        """
        # Temporarily swap models
        original_model = self.model
        self.model = model
        
        try:
            params = self.get_parameters(activation_flags, as_correlation,default_correlation)
        finally:
            self.model = original_model
        
        return params
    
    def _get_total_param_count(self) -> int:
        """Get total number of parameters in the model."""
        # 2 alphas + n diagonal X0 + 1 off-diag X0 + n diagonal Omega + 1 off-diag Omega
        # + n diagonal M + n diagonal Sigma + 1 off-diag Sigma
        return 2 + self.n + 1 + self.n + 1 + self.n + self.n + 1
    
    def get_parameter_names(self, activation_flags: List[bool]) -> List[str]:
        """
        Get names of active parameters.
        
        Parameters
        ----------
        activation_flags : List[bool]
            Boolean flags indicating which parameters are active
            
        Returns
        -------
        List[str]
            Names of active parameters
        """
        names = []
        idx = 0
        
        # Alpha parameters
        if activation_flags[idx]:
            names.append("alpha_i")
        idx += 1
        
        if activation_flags[idx]:
            names.append("alpha_j")
        idx += 1
        
        # X0 matrix
        for i in range(self.n):
            if activation_flags[idx]:
                names.append(f"x0[{i},{i}]")
            idx += 1
        
        if activation_flags[idx]:
            names.append("x0[0,1]")
        idx += 1
        
        # Omega matrix
        for i in range(self.n):
            if activation_flags[idx]:
                names.append(f"omega[{i},{i}]")
            idx += 1
        
        if activation_flags[idx]:
            names.append("omega[0,1]")
        idx += 1
        
        # M matrix
        for i in range(self.n):
            if activation_flags[idx]:
                names.append(f"m[{i},{i}]")
            idx += 1
        
        # Sigma matrix
        for i in range(self.n):
            if activation_flags[idx]:
                names.append(f"sigma[{i},{i}]")
            idx += 1
        
        if activation_flags[idx]:
            names.append("sigma[0,1]")
        idx += 1
        
        return names
    
    def create_bounds_from_models(
        self,
        lower_model: LRWFxModel,
        upper_model: LRWFxModel,
        activation_flags: List[bool]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create parameter bounds from two model instances.
        
        Parameters
        ----------
        lower_model : LRWFxModel
            Model with lower bound parameters
        upper_model : LRWFxModel
            Model with upper bound parameters
        activation_flags : List[bool]
            Boolean flags indicating which parameters are active
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Lower and upper bounds arrays
        """
        lower_params = self.get_parameters_from_model(lower_model, activation_flags)
        upper_params = self.get_parameters_from_model(upper_model, activation_flags)
        
        return lower_params, upper_params
    
    def check_parameters_moved(
        self,
        base_model: LRWFxModel,
        max_ratio: float,
        activation_flags: List[bool]
    ) -> bool:
        """
        Check if parameters have moved too much from base model.
        
        Parameters
        ----------
        base_model : LRWFxModel
            Base model for comparison
        max_ratio : float
            Maximum allowed ratio between parameters
        activation_flags : List[bool]
            Boolean flags for parameters to check
            
        Returns
        -------
        bool
            True if parameters are within acceptable range
        """
        base_params = self.get_parameters_from_model(base_model, activation_flags)
        current_params = self.get_parameters(activation_flags)
        
        for i in range(len(base_params)):
            base_val = abs(base_params[i])
            current_val = abs(current_params[i])
            
            if base_val > 0 and current_val > 0:
                ratio = max(base_val, current_val) / min(base_val, current_val)
                if ratio > abs(max_ratio):
                    return False
        
        return True
    
    def get_parameter_dict(self) -> Dict[str, Union[float, np.ndarray]]:
        """
        Get all model parameters as a dictionary.
        
        Returns
        -------
        Dict[str, Union[float, np.ndarray]]
            Dictionary of parameter names and values
        """
        return {
            'alpha_i': self.model.alpha_i,
            'alpha_j': self.model.alpha_j,
            'x0': self.model.x0.copy(),
            'omega': self.model.omega.copy(),
            'm': self.model.m.copy(),
            'sigma': self.model.sigma.copy(),
            'u_i': self.model.u_i,
            'u_j': self.model.u_j,
            'fx_spot': self.model.fx_spot
        }
    
    def set_parameter_dict(self, param_dict: Dict[str, Union[float, np.ndarray]]) -> None:
        """
        Set model parameters from dictionary.
        
        Parameters
        ----------
        param_dict : Dict[str, Union[float, np.ndarray]]
            Dictionary of parameter names and values
        """
        self.model.set_model_params(
            self.n,
            param_dict.get('x0', self.model.x0),
            param_dict.get('omega', self.model.omega),
            param_dict.get('m', self.model.m),
            param_dict.get('sigma', self.model.sigma),
            param_dict.get('alpha_i', self.model.alpha_i),
            param_dict.get('u_i', self.model.u_i),
            param_dict.get('alpha_j', self.model.alpha_j),
            param_dict.get('u_j', self.model.u_j),
            param_dict.get('fx_spot', self.model.fx_spot)
        )

    def get_activation_flags(self, parameter_names: List[str]) -> List[bool]:
        """
        Generate activation flags from parameter names.
    
        Parameters
        ----------
        parameter_names : List[str]
            Names of active parameters
        
        Returns
        -------
        List[bool]
            Boolean flags indicating which parameters are active
        """
        # Create a set for O(1) lookup
        name_set = set(parameter_names)
    
        # Initialize flags list
        flags = []
    
        # Alpha parameters
        flags.append("alpha_i" in name_set)
        flags.append("alpha_j" in name_set)
    
        # X0 matrix diagonal
        for i in range(self.n):
            flags.append(f"x0[{i},{i}]" in name_set)
    
        # X0 off-diagonal
        flags.append("x0[0,1]" in name_set)
    
        # Omega matrix diagonal
        for i in range(self.n):
            flags.append(f"omega[{i},{i}]" in name_set)
    
        # Omega off-diagonal
        flags.append("omega[0,1]" in name_set)
    
        # M matrix diagonal
        for i in range(self.n):
            flags.append(f"m[{i},{i}]" in name_set)
    
        # Sigma matrix diagonal
        for i in range(self.n):
            flags.append(f"sigma[{i},{i}]" in name_set)
    
        # Sigma off-diagonal
        flags.append("sigma[0,1]" in name_set)
    
        return flags

    def get_all_parameter_names(self) -> List[str]:
        """
        Get names of all parameters (to activate all flags).
    
        Returns
        -------
        List[str]
            Names of all parameters
        """
        names = []
    
        # Alpha parameters
        names.append("alpha_i")
        names.append("alpha_j")
    
        # X0 matrix diagonal
        for i in range(self.n):
            names.append(f"x0[{i},{i}]")
    
        # X0 off-diagonal
        names.append("x0[0,1]")
    
        # Omega matrix diagonal
        for i in range(self.n):
            names.append(f"omega[{i},{i}]")
    
        # Omega off-diagonal
        names.append("omega[0,1]")
    
        # M matrix diagonal
        for i in range(self.n):
            names.append(f"m[{i},{i}]")
    
        # Sigma matrix diagonal
        for i in range(self.n):
            names.append(f"sigma[{i},{i}]")
    
        # Sigma off-diagonal
        names.append("sigma[0,1]")
    
        return names


class ParameterTransformer:
    """Handles parameter transformations for optimization."""
    
    @staticmethod
    def to_unconstrained(params: np.ndarray, bounds: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        Transform bounded parameters to unconstrained space.
        
        Parameters
        ----------
        params : np.ndarray
            Bounded parameters
        bounds : Tuple[np.ndarray, np.ndarray]
            Lower and upper bounds
            
        Returns
        -------
        np.ndarray
            Unconstrained parameters
        """
        lower, upper = bounds
        
        # Use logit transformation for bounded parameters
        normalized = (params - lower) / (upper - lower)
        # Clip to avoid numerical issues
        normalized = np.clip(normalized, 1e-10, 1 - 1e-10)
        
        return np.log(normalized / (1 - normalized))
    
    @staticmethod
    def from_unconstrained(unconstrained: np.ndarray, bounds: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        Transform unconstrained parameters back to bounded space.
        
        Parameters
        ----------
        unconstrained : np.ndarray
            Unconstrained parameters
        bounds : Tuple[np.ndarray, np.ndarray]
            Lower and upper bounds
            
        Returns
        -------
        np.ndarray
            Bounded parameters
        """
        lower, upper = bounds
        
        # Inverse logit transformation
        normalized = 1 / (1 + np.exp(-unconstrained))
        
        return lower + normalized * (upper - lower)
    
    @staticmethod
    def ensure_positive_definite(matrix: np.ndarray, min_eigenvalue: float = 1e-6) -> np.ndarray:
        """
        Ensure a matrix is positive definite.
        
        Parameters
        ----------
        matrix : np.ndarray
            Input matrix
        min_eigenvalue : float
            Minimum eigenvalue to enforce
            
        Returns
        -------
        np.ndarray
            Positive definite matrix
        """
        # Symmetrize
        matrix = 0.5 * (matrix + matrix.T)
        
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        
        # Clip eigenvalues
        eigenvalues = np.maximum(eigenvalues, min_eigenvalue)
        
        # Reconstruct
        return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    
    @staticmethod
    def correlation_to_covariance(
        correlation: float,
        std1: float,
        std2: float
    ) -> float:
        """
        Convert correlation to covariance.
        
        Parameters
        ----------
        correlation : float
            Correlation coefficient (-1 to 1)
        std1 : float
            Standard deviation of first variable
        std2 : float
            Standard deviation of second variable
            
        Returns
        -------
        float
            Covariance
        """
        return correlation * std1 * std2
    
    @staticmethod
    def covariance_to_correlation(
        covariance: float,
        std1: float,
        std2: float
    ) -> float:
        """
        Convert covariance to correlation.
        
        Parameters
        ----------
        covariance : float
            Covariance
        std1 : float
            Standard deviation of first variable
        std2 : float
            Standard deviation of second variable
            
        Returns
        -------
        float
            Correlation coefficient
        """
        if std1 <= 0 or std2 <= 0:
            return 0.0
        
        correlation = covariance / (std1 * std2)
        return np.clip(correlation, -1.0, 1.0)


class CalibrationStateManager:
    """Manages calibration state and history."""
    
    def __init__(self):
        """Initialize state manager."""
        self.history = []
        self.best_params = None
        self.best_error = float('inf')
        self.iteration_count = 0
    
    def update(
        self,
        params: np.ndarray,
        error: float,
        additional_info: Optional[Dict] = None
    ) -> None:
        """
        Update calibration state.
        
        Parameters
        ----------
        params : np.ndarray
            Current parameters
        error : float
            Current error
        additional_info : Dict, optional
            Additional information to store
        """
        self.iteration_count += 1
        
        state = {
            'iteration': self.iteration_count,
            'params': params.copy(),
            'error': error,
            'timestamp': np.datetime64('now'),
            'is_best': False
        }
        
        if additional_info:
            state.update(additional_info)
        
        # Check if this is the best so far
        if error < self.best_error:
            self.best_error = error
            self.best_params = params.copy()
            state['is_best'] = True
        
        self.history.append(state)
    
    def get_convergence_history(self) -> Tuple[List[int], List[float]]:
        """
        Get convergence history.
        
        Returns
        -------
        Tuple[List[int], List[float]]
            Iterations and corresponding errors
        """
        iterations = [s['iteration'] for s in self.history]
        errors = [s['error'] for s in self.history]
        
        return iterations, errors
    
    def get_parameter_evolution(self, param_index: int) -> Tuple[List[int], List[float]]:
        """
        Get evolution of a specific parameter.
        
        Parameters
        ----------
        param_index : int
            Index of parameter to track
            
        Returns
        -------
        Tuple[List[int], List[float]]
            Iterations and parameter values
        """
        iterations = []
        values = []
        
        for state in self.history:
            if param_index < len(state['params']):
                iterations.append(state['iteration'])
                values.append(state['params'][param_index])
        
        return iterations, values
    
    def save_state(self, filepath: str) -> None:
        """Save calibration state to file."""
        import pickle
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'history': self.history,
                'best_params': self.best_params,
                'best_error': self.best_error,
                'iteration_count': self.iteration_count
            }, f)
    
    def load_state(self, filepath: str) -> None:
        """Load calibration state from file."""
        import pickle
        
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
            
        self.history = state['history']
        self.best_params = state['best_params']
        self.best_error = state['best_error']
        self.iteration_count = state['iteration_count']
    
    def reset(self) -> None:
        """Reset calibration state."""
        self.history = []
        self.best_params = None
        self.best_error = float('inf')
        self.iteration_count = 0
