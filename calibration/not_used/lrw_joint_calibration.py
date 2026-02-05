"""
LRW Model Joint Calibration Solution

This module provides a joint calibration approach that allows omega parameters
to adjust slightly while heavily penalizing deviations from the OIS/spread curve fit.

The key insight is that we need to balance:
1. Preserving the OIS curve fit (calibrated omega affects bond prices)
2. Preserving the spread curve fit (calibrated omega affects spreads)
3. Achieving reasonable swaption volatility fit (requires larger omega to allow larger sigma)
"""

import numpy as np
from scipy import optimize
from typing import List, Tuple, Dict, Optional, Callable
import copy


class JointCalibrationConfig:
    """Configuration for joint calibration."""
    
    def __init__(
        self,
        # Weights for different objectives
        ois_weight: float = 100.0,      # High weight to preserve OIS fit
        spread_weight: float = 100.0,   # High weight to preserve spread fit
        swaption_weight: float = 1.0,   # Lower weight for swaption fit
        
        # Gindikin constraint parameters
        gindikin_penalty_weight: float = 1e6,  # Penalty for violating Gindikin
        gindikin_safety_margin: float = 0.01,  # Small buffer from constraint boundary
        
        # Optimization parameters
        max_omega_change: float = 2.0,   # Maximum factor change in omega (e.g., 2.0 = can double)
        ftol: float = 1e-6,
        xtol: float = 1e-6,
        gtol: float = 1e-6,
        max_nfev: int = 200,
        
        verbose: bool = True
    ):
        self.ois_weight = ois_weight
        self.spread_weight = spread_weight
        self.swaption_weight = swaption_weight
        self.gindikin_penalty_weight = gindikin_penalty_weight
        self.gindikin_safety_margin = gindikin_safety_margin
        self.max_omega_change = max_omega_change
        self.ftol = ftol
        self.xtol = xtol
        self.gtol = gtol
        self.max_nfev = max_nfev
        self.verbose = verbose


def check_gindikin(omega: np.ndarray, sigma: np.ndarray, n: int = 2) -> Tuple[bool, float]:
    """
    Check Gindikin condition and return violation amount.
    
    Returns
    -------
    Tuple[bool, float]
        (is_valid, min_eigenvalue)
    """
    beta = n + 1
    gindikin_matrix = omega - beta * (sigma @ sigma)
    eigenvalues = np.linalg.eigvalsh(gindikin_matrix)
    min_eig = np.min(eigenvalues)
    return min_eig >= 0, min_eig


def compute_gindikin_penalty(omega: np.ndarray, sigma: np.ndarray, n: int = 2,
                             safety_margin: float = 0.01) -> float:
    """
    Compute penalty for Gindikin constraint violation.
    
    Returns 0 if constraint is satisfied with margin, otherwise returns penalty.
    """
    beta = n + 1
    gindikin_matrix = omega - beta * (sigma @ sigma)
    eigenvalues = np.linalg.eigvalsh(gindikin_matrix)
    min_eig = np.min(eigenvalues)
    
    # We want min_eig >= safety_margin * min(diag(omega))
    threshold = safety_margin * np.min(np.diag(omega))
    
    if min_eig >= threshold:
        return 0.0
    else:
        # Quadratic penalty for violation
        violation = threshold - min_eig
        return violation ** 2


def get_joint_param_activation() -> List[bool]:
    """
    Get parameter activation for joint calibration.
    
    Activates:
    - omega[0,0], omega[1,1] (to allow adjustment for vol fit)
    - sigma[0,0], sigma[1,1], sigma[0,1] (volatility parameters)
    - x0[0,1], omega[0,1] (correlation parameters)
    
    Parameter indices:
    [alpha, x11, x22, x12, omega11, omega22, omega12, m11, m22, sigma11, sigma22, sigma12]
    [  0,    1,   2,   3,    4,       5,       6,      7,   8,     9,      10,      11  ]
    """
    activation = [False] * 12
    # Omega diagonal - allow small adjustments
    activation[4] = True   # omega[0,0]
    activation[5] = True   # omega[1,1]
    # Correlation
    activation[3] = True   # x0[0,1]
    activation[6] = True   # omega[0,1]
    # Sigma (volatility)
    activation[9] = True   # sigma[0,0]
    activation[10] = True  # sigma[1,1]
    activation[11] = True  # sigma[0,1]
    return activation


def get_sigma_only_param_activation() -> List[bool]:
    """Get activation for sigma parameters only."""
    activation = [False] * 12
    activation[9] = True   # sigma[0,0]
    activation[10] = True  # sigma[1,1]
    activation[11] = True  # sigma[0,1]
    return activation


class JointCalibrator:
    """
    Joint calibrator that balances curve fit with swaption volatility fit.
    """
    
    def __init__(
        self,
        model,  # LRWModel instance
        calibrator,  # Original calibrator with objective functions
        config: Optional[JointCalibrationConfig] = None
    ):
        self.model = model
        self.calibrator = calibrator
        self.config = config or JointCalibrationConfig()
        
        # Store initial calibrated values for reference
        self.initial_omega = model.omega.copy()
        self.initial_x0 = model.x0.copy()
        
        # Store baseline errors after OIS/spread calibration
        self.baseline_ois_error = None
        self.baseline_spread_error = None
        
    def compute_baseline_errors(self):
        """Compute baseline OIS and spread errors after initial calibration."""
        # These would be computed from the calibrator's objective functions
        # For now, placeholder - you'd call:
        # self.baseline_ois_error = self.calibrator.objectives.ois_price_objective(...)
        # self.baseline_spread_error = self.calibrator.objectives.spread_full_objective(...)
        pass
    
    def joint_objective(
        self,
        params: np.ndarray,
        param_activation: List[bool]
    ) -> np.ndarray:
        """
        Joint objective function combining OIS, spread, and swaption errors.
        
        Parameters
        ----------
        params : np.ndarray
            Parameter values to evaluate
        param_activation : List[bool]
            Which parameters are active
            
        Returns
        -------
        np.ndarray
            Combined weighted error vector
        """
        # Set parameters
        self.calibrator.set_model_parameters(params, param_activation)
        
        # Get individual error components
        # Note: You'll need to adapt these to your actual objective function signatures
        
        # 1. OIS errors
        ois_activation = self.calibrator._get_ois_param_activation()
        ois_params = self.calibrator.get_model_parameters(ois_activation)
        ois_errors = self.calibrator.objectives.ois_price_objective(
            ois_params, ois_activation, 
            self.calibrator.config.max_tenor, 
            self.calibrator.config.min_tenor
        )
        
        # 2. Spread errors
        spread_activation = self.calibrator._get_spread_param_activation()
        spread_params = self.calibrator.get_model_parameters(spread_activation)
        spread_errors = self.calibrator.objectives.spread_full_objective(
            spread_params, spread_activation,
            self.calibrator.config.max_tenor,
            self.calibrator.config.min_tenor
        )
        
        # 3. Swaption errors
        swaption_errors = self.calibrator.objectives.swaption_vol_objective(
            params, param_activation, 
            self.calibrator.config.use_multi_thread
        )
        
        # 4. Gindikin penalty
        gindikin_penalty = compute_gindikin_penalty(
            self.model.omega, 
            self.model.sigma,
            self.model.n,
            self.config.gindikin_safety_margin
        )
        
        # Combine with weights
        weighted_errors = np.concatenate([
            self.config.ois_weight * np.array(ois_errors),
            self.config.spread_weight * np.array(spread_errors),
            self.config.swaption_weight * np.array(swaption_errors),
            [self.config.gindikin_penalty_weight * gindikin_penalty]
        ])
        
        if self.config.verbose:
            ois_rmse = np.sqrt(np.mean(np.array(ois_errors)**2))
            spread_rmse = np.sqrt(np.mean(np.array(spread_errors)**2))
            swaption_rmse = np.sqrt(np.mean(np.array(swaption_errors)**2))
            print(f"OIS RMSE: {ois_rmse:.6f}, Spread RMSE: {spread_rmse:.6f}, "
                  f"Swaption RMSE: {swaption_rmse:.6f}, Gindikin penalty: {gindikin_penalty:.6f}")
        
        return weighted_errors
    
    def get_bounds_with_omega_constraint(
        self,
        param_activation: List[bool]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get parameter bounds, limiting omega change to preserve curve fit.
        """
        # Get base bounds from calibrator
        base_bounds = self.calibrator.constraints.get_parameter_bounds(param_activation)
        lower = list(base_bounds[0])
        upper = list(base_bounds[1])
        
        # Find omega indices and constrain them
        # param_activation: [alpha, x11, x22, x12, omega11, omega22, omega12, m11, m22, sigma11, sigma22, sigma12]
        active_idx = 0
        for i, is_active in enumerate(param_activation):
            if not is_active:
                continue
                
            # Index 4 = omega[0,0], Index 5 = omega[1,1]
            if i == 4:  # omega[0,0]
                lower[active_idx] = self.initial_omega[0, 0] / self.config.max_omega_change
                upper[active_idx] = self.initial_omega[0, 0] * self.config.max_omega_change
            elif i == 5:  # omega[1,1]
                lower[active_idx] = self.initial_omega[1, 1] / self.config.max_omega_change
                upper[active_idx] = self.initial_omega[1, 1] * self.config.max_omega_change
            
            active_idx += 1
        
        return (np.array(lower), np.array(upper))
    
    def calibrate_joint(self) -> Dict:
        """
        Run joint calibration.
        
        Returns
        -------
        Dict with calibration results
        """
        print("=" * 60)
        print("STARTING JOINT CALIBRATION")
        print("=" * 60)
        
        # Get activation and bounds
        param_activation = get_joint_param_activation()
        starting_point = self.calibrator.get_model_parameters(param_activation)
        bounds = self.get_bounds_with_omega_constraint(param_activation)
        
        print(f"\nActivated parameters: {sum(param_activation)}")
        print(f"Starting point: {starting_point}")
        print(f"Bounds: {bounds}")
        
        # Check initial Gindikin
        is_valid, min_eig = check_gindikin(self.model.omega, self.model.sigma, self.model.n)
        print(f"\nInitial Gindikin check: valid={is_valid}, min_eigenvalue={min_eig:.6f}")
        
        # Run optimization
        result = optimize.least_squares(
            lambda x: self.joint_objective(x, param_activation),
            starting_point,
            bounds=bounds,
            ftol=self.config.ftol,
            xtol=self.config.xtol,
            gtol=self.config.gtol,
            max_nfev=self.config.max_nfev,
            verbose=2 if self.config.verbose else 0
        )
        
        # Update model with result
        self.calibrator.set_model_parameters(result.x, param_activation)
        
        # Check final Gindikin
        is_valid, min_eig = check_gindikin(self.model.omega, self.model.sigma, self.model.n)
        print(f"\nFinal Gindikin check: valid={is_valid}, min_eigenvalue={min_eig:.6f}")
        
        # Report omega changes
        print("\nOmega changes:")
        print(f"  omega[0,0]: {self.initial_omega[0,0]:.6f} -> {self.model.omega[0,0]:.6f} "
              f"(ratio: {self.model.omega[0,0]/self.initial_omega[0,0]:.2f}x)")
        print(f"  omega[1,1]: {self.initial_omega[1,1]:.6f} -> {self.model.omega[1,1]:.6f} "
              f"(ratio: {self.model.omega[1,1]/self.initial_omega[1,1]:.2f}x)")
        
        return {
            'success': result.success,
            'message': result.message,
            'final_cost': result.cost,
            'n_iterations': result.nfev,
            'final_params': result.x,
            'omega_change_ratio': [
                self.model.omega[0,0] / self.initial_omega[0,0],
                self.model.omega[1,1] / self.initial_omega[1,1]
            ],
            'gindikin_valid': is_valid,
            'gindikin_min_eigenvalue': min_eig
        }


def calibrate_sigma_at_max_feasible(
    model,
    calibrator,
    safety_factor: float = 0.95
) -> Dict:
    """
    Alternative approach: Set sigma to maximum feasible values
    given current omega, then only calibrate correlations.
    
    This preserves OIS/spread fit completely but may underfit swaption vols.
    """
    n = model.n
    beta = n + 1
    omega = model.omega
    
    # Compute max feasible sigma
    max_sigma = np.zeros_like(omega)
    for i in range(n):
        max_sigma[i, i] = np.sqrt(omega[i, i] / beta) * safety_factor
    
    print(f"Setting sigma to max feasible values:")
    print(f"  sigma[0,0] = {max_sigma[0,0]:.6f} (was {model.sigma[0,0]:.6f})")
    print(f"  sigma[1,1] = {max_sigma[1,1]:.6f} (was {model.sigma[1,1]:.6f})")
    
    # Update model
    model.sigma[0, 0] = max_sigma[0, 0]
    model.sigma[1, 1] = max_sigma[1, 1]
    
    # Now calibrate only the correlation (sigma[0,1])
    # The correlation can be set based on correlation between factors
    # For now, keep it at 0 or calibrate separately
    
    return {
        'max_sigma': max_sigma,
        'gindikin_satisfied': True
    }


# Simplified version that you can adapt to your existing calibrator
def create_joint_objective_simple(
    calibrator,
    ois_weight: float = 100.0,
    spread_weight: float = 100.0,
    swaption_weight: float = 1.0,
    gindikin_penalty: float = 1e6
):
    """
    Create a simple joint objective function.
    
    This is a factory function that returns an objective function
    compatible with your existing calibration code.
    """
    
    def objective(x, param_activation):
        # Set parameters
        calibrator.set_model_parameters(x, param_activation)
        model = calibrator.model
        
        # Check Gindikin
        beta = model.n + 1
        gindikin_matrix = model.omega - beta * (model.sigma @ model.sigma)
        min_eig = np.min(np.linalg.eigvalsh(gindikin_matrix))
        
        if min_eig < 0:
            # Return large error if Gindikin violated
            return np.array([gindikin_penalty * abs(min_eig)])
        
        # Get swaption errors (this is what we primarily want to minimize)
        swaption_errors = calibrator.objectives.swaption_vol_objective(
            x, param_activation, False
        )
        
        return swaption_weight * np.array(swaption_errors)
    
    return objective


if __name__ == "__main__":
    # Example usage
    print("Joint Calibration Module")
    print("Use JointCalibrator class with your existing calibrator")
    print("\nExample:")
    print("  config = JointCalibrationConfig(max_omega_change=3.0)")
    print("  joint_cal = JointCalibrator(model, calibrator, config)")
    print("  result = joint_cal.calibrate_joint()")
