"""
LRW Jump Model Calibrator.

This module provides calibration functionality for Linear Rational Wishart
interest rate models with jump diffusion.
"""

from typing import Optional, Dict, List, Tuple, Union
import numpy as np
import jax.numpy as jnp
from jax import jit
import pandas as pd
from scipy import optimize
import time
import copy
import gc


from ..models.interest_rate.lrw_model import LRWModel
from ..data.data_market_data import *# MarketData
from .objectives import ObjectiveFunctions
from .constraints import CalibrationConstraints
from .market_data_handler import MarketDataHandler
# from ..pricing.jump_pricer import LRWJumpPricer
from ..pricing.swaption_pricer import LRWSwaptionPricer
# from ..pricing.parallel_pricer import ParallelPricer
# from ..utils.calibration_reporting import CalibrationReporter
from .calibration_reporting import IRCalibrationReporter
from .pseudo_inverse import *
from .alpha_curve import *

OIS_FTOL=1e-4  
OIS_XTOL=1e-4   
OIS_GTOL=1e-4   
OIS_NFEV = 30

SPREAD_FTOL=1e-4  
SPREAD_XTOL=1e-4   
SPREAD_GTOL=1e-4   
SPREAD_NFEV = 30

SWAPTION_FTOL=1e-4  
SWAPTION_XTOL=1e-4   
SWAPTION_GTOL=1e-4   
SWAPTION_NFEV = 50
DEFAULT_START_CORRELATION=0.25
class CalibrationConfig:
    """Configuration for LRW Jump model calibration."""
    
    def __init__(
        self,
        max_tenor: float = 11,#15,##11.0,
        min_tenor: float = 1.0,
        alpha_tenor: float =  11,#15,##11.0,
        max_ratio_params: float = 4.0,
        use_multi_thread: bool = False,#True,#False,
        pricing_approach: str = "RangeKutta",
        calibrate_on_swaption: bool = True,#False,#True,
        calibrate_based_on_correl: bool = True,
        use_market_based_strike: bool = False,
        verbose: bool = True,
        calibrate_on_swaption_vol: bool = False#,#True,

    ):
        self.max_tenor = max_tenor
        self.min_tenor = min_tenor
        self.alpha_tenor = alpha_tenor
        self.max_ratio_params = max_ratio_params
        self.use_multi_thread = use_multi_thread
        self.pricing_approach = pricing_approach
        self.calibrate_on_swaption = calibrate_on_swaption
        self.calibrate_based_on_correl = calibrate_based_on_correl
        self.use_market_based_strike = use_market_based_strike
        self.verbose= verbose
        self.calibrate_on_swaption_vol= calibrate_on_swaption_vol

class LRWJumpCalibrator:
    """
    Calibrator for Linear Rational Wishart interest rate models with jumps.
    
    This class handles the calibration of LRW jump models to market data including
    OIS curves, spreads, and swaption prices/volatilities.
    """
    
    def __init__(
        self,
        model: LRWModel,
        daily_data: DailyData,
        config: Optional[CalibrationConfig] = None
    ):
        """
        Initialize the calibrator.
        
        Parameters
        ----------
        model : LRWModel
            The LRW jump model to calibrate
        daily_data : DailyData
            Market data for calibration
        config : CalibrationConfig, optional
            Calibration configuration
        """
        self.model = model
        self.daily_data = daily_data
        self.config = config or CalibrationConfig()
        
        # Initialize components
        self.objectives = ObjectiveFunctions(model, daily_data, config)
        self.constraints = CalibrationConstraints(model)
        self.market_handler = MarketDataHandler(daily_data)
        # self.pricer = LRWJumpPricer(model) FourierPricer
        self.pricer = LRWSwaptionPricer(model)  
        self.reporter = IRCalibrationReporter()
        
        if self.config.use_multi_thread:
            self.parallel_pricer = None ## todo ParallelPricer()
        
        # Initialize curves
        self._initialize_curves()
        
        # Calibration state
        self.spread_params_replaced = False
        self.ois_params_replaced = False
        self.calibration_results = {}
        if self.config.verbose:
            print("LRW Jump Calibrator initialized.")
            self.optimizer_verbose=2
        else:
            self.optimizer_verbose=0
        
        self.use_initial_alpha_curve = False

    def _initialize_curves(self,market_based_strike=True ):
        """Initialize OIS and IBOR curves."""
        self.ois_curve = self.market_handler.create_ois_curve()
        self.ibor_curve = self.market_handler.create_ibor_curve(self.ois_curve)
        self.market_handler.update_swaption_market_data(
                                    model=self.model,
                                    market_based_strike  = market_based_strike)#True)

        # Update market data if needed
        # if self.ibor_curve.has_been_interpolated:
        #     self.daily_data.euribor_rate_data = self.ibor_curve.rate_data_list
            
        # Determine maximum tenor for positive spreads
        #todo

        # self.max_positive_a_tenor = self.config.max_tenor
        self.max_positive_a_tenor = self.market_handler.get_max_positive_spread_tenor()
        print(f"Max positive spread tenor: {self.max_positive_a_tenor:.2f} years")

        self.objectives.ois_curve=self.ois_curve
        self.objectives.ibor_curve=self.ibor_curve
        # self.objectives.precompute_instruments_weights( self.config.max_tenor,   
        #                                                 self.config.min_tenor)

    def calibrate_full(self) -> Dict[str, float]:
        """
        Perform full calibration of the model.
        
        Returns
        -------
        Dict[str, float]
            Calibration results including errors and parameters
        """
        print("Starting full LRW Jump model calibration...")
        
        # Step 1: Calibrate to OIS curve
        self.set_alpha()

        self.objectives.reprice_bond_market_data()
        self.objectives.reprice_spreads_market_data(full_a=True,
                                                    max_tenor = self.config.max_tenor)
        
        # Update market data after calibration
        self.market_handler.update_swaption_market_data(
                                    model=self.model,
                                    market_based_strike  = True)
        self.objectives.precompute_instruments_weights( self.config.min_tenor ,   
                                                        self.config.max_tenor)
        gc.set_threshold(100, 5, 5)
        gc.collect()
        SEQUENTIAL_OIS_SPREAD=True#False#True#False##True#False#True#False
        # if self.model.is_spread==False:
        #     SEQUENTIAL_OIS_SPREAD=False

        if SEQUENTIAL_OIS_SPREAD:
            ois_error =  self.calibrate_ois_curve()
            print(f"OIS calibration completed. RMSE: {ois_error:.4f}")
            
            # Step 2: Calibrate to spreads
            gc.collect()
            if  self.model.is_spread and self.ibor_curve is not None:
                print("Model is a  spread model; now calibrating spread related parameters.")            
                spread_error = self.calibrate_spreads()
                print(f"Spread calibration completed. RMSE: {spread_error:.4f}")
            else:
                print("Model is not a spread model; skipping spread calibration.")
                spread_error = 0.0
        else:
            print(f"============ Joint OIS and Spread calibration ================")
            ois_spread_error =  self.calibrate_ois_and_spreads()
            ois_error=ois_spread_error
            spread_error=ois_spread_error   
            print(f"Joint OIS and Spread calibration completed. RMSE: {ois_error:.4f}")

        self.model.print_model()
        
        if self.use_initial_alpha_curve:
            self.set_initial_alpha_curve()
            # calibrate_with_regularization
        # Step 3: Calibrate to swaptions
        gc.collect()
        if self.config.calibrate_on_swaption:
            # Update market data after calibration of the curves
            self.market_handler.update_swaption_market_data(
                                    model=self.model,
                                    market_based_strike  = self.config.use_market_based_strike)
     

            swaption_error = self.calibrate_swaptions()
            print(f"Swaption calibration completed. RMSE: {swaption_error:.4f}")
        else:
            swaption_error = None
        self.model.print_model()
        # # Update market data after calibration
        # ##no need to do this here again
        # self.market_handler.update_swaption_market_data(
        #                             model=self.model,
        #                             market_based_strike  = self.config.use_market_based_strike)
        # Compile results
        self.calibration_results = {
            'ois_error': ois_error,
            'spread_error': spread_error,
            'swaption_error': swaption_error,
            'parameters': self.get_model_parameters()
        }
        
        return self.calibration_results
     
    def calibrate_swaptions_joint_old(
        self,
        omega_deviation_weight: float = 10.0,
        gindikin_penalty_weight: float = 1e6,
        max_omega_multiplier: float = 25.0
    ) -> float:
        """
        Joint swaption calibration that allows omega to adjust.
        
        This method calibrates sigma AND omega together, with penalties to:
        1. Keep omega close to initial values (preserve OIS/spread fit)
        2. Satisfy the Gindikin condition
        
        Parameters
        ----------
        omega_deviation_weight : float
            Penalty weight for omega deviating from initial values.
            Higher = more preservation of OIS/spread fit.
        gindikin_penalty_weight : float
            Penalty weight for Gindikin violation.
        max_omega_multiplier : float
            Maximum factor by which omega can increase (e.g., 25 = 25x)
            
        Returns
        -------
        float
            Final swaption RMSE
        """
        print("=" * 60)
        print("JOINT SWAPTION CALIBRATION (omega + sigma)")
        print("=" * 60)
        
        # Store initial omega for penalty calculation
        initial_omega = np.array(self.model.omega).copy()
        print(f"Initial omega: [{initial_omega[0,0]:.6f}, {initial_omega[1,1]:.6f}]")
        
        # Parameter activation: omega diagonal + sigma
        param_activation = [False] * 12
        param_activation[4] = True   # omega[0,0]
        param_activation[5] = True   # omega[1,1]
        param_activation[9] = True   # sigma[0,0]
        param_activation[10] = True  # sigma[1,1]
        param_activation[11] = True  # sigma[0,1] (correlation)
        # Optionally add correlation parameters:
        # param_activation[3] = True   # x0[0,1]
        # param_activation[6] = True   # omega[0,1]
        
        def joint_objective(x):
            """Joint objective with Gindikin and omega deviation penalties."""
            # Set parameters
            self.set_model_parameters(x, param_activation)
            
            # Get current model parameters as numpy
            omega_np = np.array(self.model.omega)
            sigma_np = np.array(self.model.sigma)
            
            # 1. Gindikin penalty
            beta = self.model.n + 1
            gindikin_matrix = omega_np - beta * (sigma_np @ sigma_np)
            min_eig = np.min(np.linalg.eigvalsh(gindikin_matrix))
            
            if min_eig < 0:
                gindikin_penalty = gindikin_penalty_weight * (min_eig ** 2)
            else:
                gindikin_penalty = 0.0
            
            # 2. Omega deviation penalty (preserve OIS/spread fit)
            omega_diff = omega_np - initial_omega
            # Relative deviation to handle different scales
            rel_diff_00 = (omega_np[0,0] - initial_omega[0,0]) / initial_omega[0,0]
            rel_diff_11 = (omega_np[1,1] - initial_omega[1,1]) / initial_omega[1,1]
            omega_penalty = omega_deviation_weight * (rel_diff_00**2 + rel_diff_11**2)
            
            # 3. Swaption volatility errors
            if self.config.calibrate_on_swaption_vol:
                swaption_errors = self.objectives.swaption_vol_objective(
                    x, param_activation, self.config.use_multi_thread
                )
            else:
                swaption_errors = self.objectives.swaption_price_objective(
                    x, param_activation, self.config.use_multi_thread
                )
            
            # Combine into error vector for least_squares
            error_list = list(swaption_errors)
            error_list.append(np.sqrt(gindikin_penalty))
            error_list.append(np.sqrt(omega_penalty))
            
            # Print progress
            swaption_rmse = np.sqrt(np.mean(np.array(swaption_errors)**2))
            print(f"  Swaption RMSE: {swaption_rmse:.6f}, "
                f"Omega penalty: {omega_penalty:.4f}, "
                f"Gindikin: {min_eig:.6f} ({'OK' if min_eig >= 0 else 'VIOLATED'})")
            
            return np.array(error_list)
        
        # Get starting point
        starting_point = self.get_model_parameters(param_activation)
        
        # Get bounds
        bounds = self.constraints.get_parameter_bounds(param_activation)
        lower = list(bounds[0])
        upper = list(bounds[1])
        
        # Modify omega bounds to allow larger values
        active_idx = 0
        for i, is_active in enumerate(param_activation):
            if not is_active:
                continue
            if i == 4:  # omega[0,0]
                lower[active_idx] = float(initial_omega[0, 0]) * 0.5  # Can decrease by 50%
                upper[active_idx] = float(initial_omega[0, 0]) * max_omega_multiplier
            elif i == 5:  # omega[1,1]
                lower[active_idx] = float(initial_omega[1, 1]) * 0.5
                upper[active_idx] = float(initial_omega[1, 1]) * max_omega_multiplier
            active_idx += 1
        
        bounds = (np.array(lower), np.array(upper))
        
        print(f"\nStarting point: {starting_point}")
        print(f"Lower bounds: {bounds[0]}")
        print(f"Upper bounds: {bounds[1]}")
        
        # Check initial Gindikin
        beta = self.model.n + 1
        omega_np = np.array(self.model.omega)
        sigma_np = np.array(self.model.sigma)
        gindikin_matrix = omega_np - beta * (sigma_np @ sigma_np)
        init_min_eig = np.min(np.linalg.eigvalsh(gindikin_matrix))
        print(f"\nInitial Gindikin min eigenvalue: {init_min_eig:.6f}")
        
        # Run optimization
        print("\nStarting optimization...")
        result = optimize.least_squares(
            joint_objective,
            starting_point,
            bounds=bounds,
            ftol=SWAPTION_FTOL,
            xtol=SWAPTION_XTOL,
            gtol=SWAPTION_GTOL,
            max_nfev=SWAPTION_NFEV * 2,  # More iterations for joint calibration
            verbose=self.optimizer_verbose
        )
        
        # Apply final result
        self.set_model_parameters(result.x, param_activation)
        
        # Report results
        print("\n" + "=" * 60)
        print("JOINT CALIBRATION RESULTS")
        print("=" * 60)
        print(f"Optimization success: {result.success}")
        print(f"Message: {result.message}")
        print(f"Final cost: {result.cost:.6f}")
        
        # Omega changes
        final_omega = np.array(self.model.omega)
        print(f"\nOmega changes:")
        print(f"  omega[0,0]: {initial_omega[0,0]:.6f} -> {final_omega[0,0]:.6f} "
            f"({final_omega[0,0]/initial_omega[0,0]:.2f}x)")
        print(f"  omega[1,1]: {initial_omega[1,1]:.6f} -> {final_omega[1,1]:.6f} "
            f"({final_omega[1,1]/initial_omega[1,1]:.2f}x)")
        
        # Final Gindikin check
        sigma_np = np.array(self.model.sigma)
        gindikin_matrix = final_omega - beta * (sigma_np @ sigma_np)
        final_min_eig = np.min(np.linalg.eigvalsh(gindikin_matrix))
        print(f"\nFinal Gindikin min eigenvalue: {final_min_eig:.6f}")
        print(f"Gindikin satisfied: {'✅ YES' if final_min_eig >= 0 else '❌ NO'}")
        
        # Calculate final swaption RMSE
        if self.config.calibrate_on_swaption_vol:
            errors = self.objectives.swaption_vol_objective(
                result.x, param_activation, self.config.use_multi_thread
            )
        else:
            errors = self.objectives.swaption_price_objective(
                result.x, param_activation, self.config.use_multi_thread
            )
        rmse = self.objectives.compute_rmse(errors)
        print(f"\nFinal swaption RMSE: {rmse:.6f}")
        
        return rmse

    def calibrate_full_joint_old(self) -> Dict:
        """
        Full calibration using joint approach for swaptions.
        
        This replaces calibrate_full() when you want to use joint calibration.
        """
        print("Starting full LRW Jump model calibration (JOINT mode)...")
        
        # Step 1: Calibrate to OIS curve (same as before)
        self.set_alpha()
        self.objectives.reprice_bond_market_data()
        self.objectives.reprice_spreads_market_data(full_a=True, max_tenor=self.config.max_tenor)
        self.market_handler.update_swaption_market_data(
            model=self.model, market_based_strike=True
        )
        
        import gc
        gc.collect()
        ois_error = self.calibrate_ois_curve()
        print(f"OIS calibration completed. RMSE: {ois_error:.4f}")
        
        # Step 2: Calibrate to spreads (same as before)
        gc.collect()
        if not self.model.is_spread or self.ibor_curve is None:
            # OIS only calibration
            print("Model is not a spread model or no IBOR data; skipping spread calibration.")
            spread_error = 0.0
        else:    
            spread_error = self.calibrate_spreads()
            print(f"Spread calibration completed. RMSE: {spread_error:.4f}")
        
        self.model.print_model()
        
        # Step 3: Joint swaption calibration (NEW)
        gc.collect()
        if self.config.calibrate_on_swaption:
            self.market_handler.update_swaption_market_data(
                model=self.model,
                market_based_strike=self.config.use_market_based_strike
            )
            
            # Use joint calibration instead of standard
            swaption_error = self.calibrate_swaptions_joint(
                omega_deviation_weight=10.0,  # Adjust as needed
                gindikin_penalty_weight=1e6,
                max_omega_multiplier=25.0
            )
            print(f"Joint swaption calibration completed. RMSE: {swaption_error:.4f}")
        else:
            swaption_error = None
        
        self.model.print_model()
        
        # Compile results
        self.calibration_results = {
            'ois_error': ois_error,
            'spread_error': spread_error,
            'swaption_error': swaption_error,
            'parameters': self.get_model_parameters()
        }
        
        return self.calibration_results

    def calibrate_swaptions_joint(
        self,
        omega_deviation_weight: float = 1.0,  # REDUCED from 10.0
        gindikin_penalty_weight: float = 1e8,  # Increased
        max_omega_multiplier: float = 30.0
    ) -> float:
        """
        Joint swaption calibration that allows omega to adjust.
        
        FIXED VERSION: Starts with feasible sigma and lets omega grow.
        """
        print("=" * 60)
        print("JOINT SWAPTION CALIBRATION (omega + sigma) - FIXED")
        if self.config.calibrate_on_swaption_vol:
            print("Calibrating on swaption VOLATILITIES")
        else:
            print("Calibrating on swaption PRICES")
        print("=" * 60)
        
        # Store initial omega
        initial_omega = np.array(self.model.omega).copy()
        print(f"Initial omega: [{initial_omega[0,0]:.6f}, {initial_omega[1,1]:.6f}]")
        
        # =========================================================================
        # FIX 1: Compute feasible starting sigma
        # =========================================================================
        beta = self.model.n + 1
        
        # Max sigma that satisfies Gindikin with current omega
        max_sigma_00 = np.sqrt(initial_omega[0, 0] / beta) * 0.9
        max_sigma_11 = np.sqrt(initial_omega[1, 1] / beta) * 0.9
        
        print(f"Max feasible sigma for current omega: [{max_sigma_00:.6f}, {max_sigma_11:.6f}]")
        print(f"Current sigma: [{self.model.sigma[0,0]:.6f}, {self.model.sigma[1,1]:.6f}]")
        
        # Check if we need to adjust starting sigma
        current_sigma = np.array(self.model.sigma)
        if abs(current_sigma[0,0]) > max_sigma_00 or abs(current_sigma[1,1]) > max_sigma_11:
            print("\n⚠️  Current sigma violates Gindikin. Adjusting starting point...")
            # Set sigma to max feasible as starting point
            self.model.sigma = np.array([
                [max_sigma_00, current_sigma[0,1]],
                [current_sigma[1,0], max_sigma_11]
            ])
            print(f"Adjusted starting sigma: [{self.model.sigma[0,0]:.6f}, {self.model.sigma[1,1]:.6f}]")
        
        # Parameter activation: omega diagonal + sigma
        param_activation = [False] * 12
        param_activation[4] = True   # omega[0,0]
        param_activation[5] = True   # omega[1,1]
        param_activation[9] = True   # sigma[0,0]
        param_activation[10] = True  # sigma[1,1]
        param_activation[11] = True  # sigma[0,1] (correlation)
        
        # Track best result
        best_swaption_rmse = float('inf')
        best_params = None
        
        def joint_objective(x):
            """Joint objective - minimize swaption error while satisfying Gindikin."""
            nonlocal best_swaption_rmse, best_params
            
            # Set parameters
            self.set_model_parameters(x, param_activation)
            
            # Get current model parameters as numpy
            omega_np = np.array(self.model.omega)
            sigma_np = np.array(self.model.sigma)
            
            # =====================================================================
            # FIX 2: Gindikin as hard constraint via barrier
            # =====================================================================
            gindikin_matrix = omega_np - beta * (sigma_np @ sigma_np)
            min_eig = np.min(np.linalg.eigvalsh(gindikin_matrix))
            
            # Barrier function: infinite penalty if violated, small penalty if close to boundary
            if min_eig < 0:
                # Violated - large penalty proportional to violation squared
                gindikin_penalty = gindikin_penalty_weight * (min_eig ** 2)
            elif min_eig < 0.01:
                # Close to boundary - log barrier to push away
                gindikin_penalty = -1000 * np.log(min_eig + 1e-10)
            else:
                gindikin_penalty = 0.0
            
            # =====================================================================
            # FIX 3: Only penalize omega DECREASE, not increase
            # =====================================================================
            # We WANT omega to increase to support larger sigma
            omega_penalty = 0.0
            # Only penalize if omega decreased below initial
            if omega_np[0,0] < initial_omega[0,0]:
                omega_penalty += omega_deviation_weight * ((initial_omega[0,0] - omega_np[0,0]) / initial_omega[0,0]) ** 2
            if omega_np[1,1] < initial_omega[1,1]:
                omega_penalty += omega_deviation_weight * ((initial_omega[1,1] - omega_np[1,1]) / initial_omega[1,1]) ** 2
            
            # =====================================================================
            # Swaption errors (the main objective)
            # =====================================================================
            if self.config.calibrate_on_swaption_vol:
                swaption_errors = self.objectives.swaption_vol_objective(
                    x, param_activation, self.config.use_multi_thread
                )
            else:
                swaption_errors = self.objectives.swaption_price_objective(
                    x, param_activation, self.config.use_multi_thread
                )
            
            swaption_rmse = np.sqrt(np.mean(np.array(swaption_errors)**2))
            
            # Track best feasible solution
            if min_eig >= 0 and swaption_rmse < best_swaption_rmse:
                best_swaption_rmse = swaption_rmse
                best_params = x.copy()
            
            # Combine into error vector
            error_list = list(swaption_errors)
            error_list.append(np.sqrt(gindikin_penalty))
            error_list.append(np.sqrt(omega_penalty))
            
            # Print progress
            status = '✅' if min_eig >= 0 else '❌'
            print(f"  {status} Swaption RMSE: {swaption_rmse:.6f}, "
                f"omega: [{omega_np[0,0]:.4f}, {omega_np[1,1]:.4f}], "
                f"sigma: [{sigma_np[0,0]:.4f}, {sigma_np[1,1]:.4f}], "
                f"Gindikin: {min_eig:.4f}")
            
            return np.array(error_list)
        
        # Get starting point (with adjusted sigma)
        starting_point = self.get_model_parameters(param_activation)
        
        # =========================================================================
        # FIX 4: Set bounds to encourage omega growth
        # =========================================================================
        bounds = self.constraints.get_parameter_bounds(param_activation)
        lower = list(bounds[0])
        upper = list(bounds[1])
        
        active_idx = 0
        for i, is_active in enumerate(param_activation):
            if not is_active:
                continue
            if i == 4:  # omega[0,0]
                lower[active_idx] = float(initial_omega[0, 0]) * 0.8  # Allow small decrease
                upper[active_idx] = float(initial_omega[0, 0]) * max_omega_multiplier
            elif i == 5:  # omega[1,1]
                lower[active_idx] = float(initial_omega[1, 1]) * 0.8
                upper[active_idx] = float(initial_omega[1, 1]) * max_omega_multiplier
            elif i == 9:  # sigma[0,0]
                # Allow sigma to grow beyond current max (omega will grow to accommodate)
                upper[active_idx] = 5.0  # Reasonable upper limit
            elif i == 10:  # sigma[1,1]
                upper[active_idx] = 5.0
            active_idx += 1
        
        bounds = (np.array(lower), np.array(upper))
        
        print(f"\nStarting point: {starting_point}")
        print(f"Lower bounds: {bounds[0]}")
        print(f"Upper bounds: {bounds[1]}")
        
        # Verify starting point satisfies Gindikin
        omega_np = np.array(self.model.omega)
        sigma_np = np.array(self.model.sigma)
        gindikin_matrix = omega_np - beta * (sigma_np @ sigma_np)
        init_min_eig = np.min(np.linalg.eigvalsh(gindikin_matrix))
        print(f"\nStarting Gindikin min eigenvalue: {init_min_eig:.6f} "
            f"({'✅ OK' if init_min_eig >= 0 else '❌ VIOLATED'})")
        
        if init_min_eig < 0:
            print("WARNING: Starting point violates Gindikin! Adjusting sigma further...")
            # Force sigma to be strictly feasible
            safety = 0.8
            self.model.sigma = np.array([
                [max_sigma_00 * safety, 0.0],
                [0.0, max_sigma_11 * safety]
            ])
            starting_point = self.get_model_parameters(param_activation)
            print(f"New starting point: {starting_point}")
        
        # Run optimization
        print("\n" + "-" * 60)
        print("Starting optimization...")
        print("-" * 60)
        
        result = optimize.least_squares(
            joint_objective,
            starting_point,
            bounds=bounds,
            ftol=1e-6,
            xtol=1e-6,
            gtol=1e-6,
            max_nfev=200,
            verbose=2 if self.config.verbose else 0
        )
        
        # =========================================================================
        # FIX 5: Use best feasible solution if final is infeasible
        # =========================================================================
        self.set_model_parameters(result.x, param_activation)
        omega_np = np.array(self.model.omega)
        sigma_np = np.array(self.model.sigma)
        gindikin_matrix = omega_np - beta * (sigma_np @ sigma_np)
        final_min_eig = np.min(np.linalg.eigvalsh(gindikin_matrix))
        
        if final_min_eig < 0 and best_params is not None:
            print("\n⚠️  Final solution violates Gindikin. Using best feasible solution.")
            self.set_model_parameters(best_params, param_activation)
            omega_np = np.array(self.model.omega)
            sigma_np = np.array(self.model.sigma)
            gindikin_matrix = omega_np - beta * (sigma_np @ sigma_np)
            final_min_eig = np.min(np.linalg.eigvalsh(gindikin_matrix))
        
        # Report results
        print("\n" + "=" * 60)
        print("JOINT CALIBRATION RESULTS")
        print("=" * 60)
        print(f"Optimization success: {result.success}")
        print(f"Message: {result.message}")
        
        # Omega changes
        final_omega = np.array(self.model.omega)
        final_sigma = np.array(self.model.sigma)
        print(f"\nOmega changes:")
        print(f"  omega[0,0]: {initial_omega[0,0]:.6f} -> {final_omega[0,0]:.6f} "
            f"({final_omega[0,0]/initial_omega[0,0]:.2f}x)")
        print(f"  omega[1,1]: {initial_omega[1,1]:.6f} -> {final_omega[1,1]:.6f} "
            f"({final_omega[1,1]/initial_omega[1,1]:.2f}x)")
        
        print(f"\nFinal sigma: [{final_sigma[0,0]:.6f}, {final_sigma[1,1]:.6f}]")
        print(f"Final Gindikin min eigenvalue: {final_min_eig:.6f}")
        print(f"Gindikin satisfied: {'✅ YES' if final_min_eig >= 0 else '❌ NO'}")
        
        # Calculate final swaption RMSE
        if self.config.calibrate_on_swaption_vol:
            errors = self.objectives.swaption_vol_objective(
                self.get_model_parameters(param_activation), 
                param_activation, 
                self.config.use_multi_thread
            )
        else:
            errors = self.objectives.swaption_price_objective(
                self.get_model_parameters(param_activation),
                param_activation,
                self.config.use_multi_thread
            )
        rmse = self.objectives.compute_rmse(errors)
        print(f"\nFinal swaption RMSE: {rmse:.6f}")
        
        return rmse

    def calibrate_full_joint(self) -> dict:
        """
        Full calibration using joint approach for swaptions.
        """
        print("Starting full LRW Jump model calibration (JOINT mode)...")
        
        # Step 1: Calibrate to OIS curve
        self.set_alpha()
        self.objectives.reprice_bond_market_data()
        self.objectives.reprice_spreads_market_data(full_a=True, max_tenor=self.config.max_tenor)
        self.market_handler.update_swaption_market_data(
            model=self.model, market_based_strike=True
        )
        
        import gc
        gc.collect()
        ois_error = self.calibrate_ois_curve()
        print(f"OIS calibration completed. RMSE: {ois_error:.4f}")
        
        # Step 2: Calibrate to spreads
        gc.collect()        
        if not self.model.is_spread or self.ibor_curve is None:
            # OIS only calibration
            print("Model is not a spread model or no IBOR data; skipping spread calibration.")
            spread_error = 0.0
        else:    
            spread_error = self.calibrate_spreads()
            print(f"Spread calibration completed. RMSE: {spread_error:.4f}")
        
        print("\nModel after OIS/Spread calibration:")
        self.model.print_model()
        
        # Step 3: Joint swaption calibration
        gc.collect()
        if self.config.calibrate_on_swaption:
            self.market_handler.update_swaption_market_data(
                model=self.model,
                market_based_strike=self.config.use_market_based_strike
            )
            
            swaption_error = self.calibrate_swaptions_joint(
                omega_deviation_weight=1.0,
                gindikin_penalty_weight=1e8,
                max_omega_multiplier=30.0
            )
            print(f"Joint swaption calibration completed. RMSE: {swaption_error:.4f}")
        else:
            swaption_error = None
        
        print("\nFinal model:")
        self.model.print_model()
        
        self.calibration_results = {
            'ois_error': ois_error,
            'spread_error': spread_error,
            'swaption_error': swaption_error,
            'parameters': self.get_model_parameters()
        }
        
        return self.calibration_results

    def set_initial_alpha_curve(self,  max_tenor: Optional[float] = None,
                                        min_tenor: Optional[float] = None):
            max_tenor = max_tenor or self.config.max_tenor
            min_tenor = min_tenor or self.config.min_tenor
            # Step: Set initial alpha curve based on current model fit to OIS
            mkt_maturities = np.array([ ois_data["Object"].time_to_maturity 
                           for _, ois_data in self.daily_data.ois_rate_data.iterrows()
                           if min_tenor <= ois_data["TimeToMat"] <= max_tenor])
            
            mkt_zc_prices = np.array([ ois_data["Object"].market_zc_price 
                           for _, ois_data in self.daily_data.ois_rate_data.iterrows()
                           if min_tenor <= ois_data["TimeToMat"] <= max_tenor])
            model_zc_prices = np.array([ ois_data["Object"].model_zc_price 
                           for _, ois_data in self.daily_data.ois_rate_data.iterrows()
                           if min_tenor <= ois_data["TimeToMat"] <= max_tenor])
            
          
            initial_alpha_curve=getInitialAlpha(mkt_maturities,mkt_zc_prices,model_zc_prices)
            self.model.set_pseudo_inverse_smoothing(initial_alpha_curve)
            print("Initial alpha curve has been set based on current model fit to OIS data.")

            # alpha_compute_input =[ ( ois_data["Object"].time_to_maturity , ois_data["Object"].market_zc_price )
            #                for _, ois_data in self.daily_data.ois_rate_data.iterrows()
            #                if min_tenor <= ois_data["TimeToMat"] <= max_tenor]
            # alpha_pseudo_InverseCalibrator = PseudoInverseCalibrator(self.model)

            # alpha_pseudo_InverseCalibrator.calibrate_alpha_curve(
            #                alpha_compute_input 
            #             )
            
            # alpha_pseudo_InverseCalibrator.calibrate_alpha_curve(alpha_compute_input ## [mkt_maturities , mkt_zc_prices ]                           
            #             )
    def calibrate_ois_curve(
        self,
        on_price: bool = True,
        max_tenor: Optional[float] = None,
        min_tenor: Optional[float] = None
    ) -> float:
        """
        Calibrate model to OIS curve.
        
        Parameters
        ----------
        on_price : bool, default=True
            Whether to calibrate on price (True) or yield (False)
        max_tenor : float, optional
            Maximum tenor for calibration
        min_tenor : float, optional
            Minimum tenor for calibration
            
        Returns
        -------
        float
            Root mean square error of calibration
        """
        max_tenor = max_tenor or self.config.max_tenor
        min_tenor = min_tenor or self.config.min_tenor
        
        # Set up parameter activation for OIS calibration
        param_activation = self._get_ois_param_activation()
        
        # Get bounds and starting point
        starting_point = self.get_model_parameters(param_activation)
        bounds = self.constraints.get_parameter_bounds(param_activation)
        print(f"Starting point: {starting_point},   Bounds: {bounds}")
        
        # Define objective function
        if on_price:
            print("OIS Calibration on Price")
            objective = lambda x: self.objectives.ois_price_objective(
                x, param_activation, max_tenor, min_tenor
            )
        else:
            print("OIS Calibration on Yield")
            objective = lambda x: self.objectives.ois_rate_objective(
                x, param_activation, max_tenor, min_tenor
            )
            
        # Optimize
        result = optimize.least_squares(objective, starting_point, bounds=bounds
                                        , verbose=self.optimizer_verbose
                                        ,ftol=OIS_FTOL,   # Stop when cost reduction < 1e-6
                                        xtol=OIS_XTOL,   # Stop when parameter change < 1e-6  
                                        gtol=OIS_GTOL,   # Stop when gradient norm < 1e-6
                                        max_nfev=OIS_NFEV  # Hard cap on function evaluations
                                        )
        
        # Update model parameters
        self.set_model_parameters(result.x, param_activation)
        
        # Calculate and return RMSE
        errors = objective(result.x)
        rmse = self.objectives.compute_rmse(errors)
        
        return rmse
        
    def calibrate_spreads(
        self,
        on_full_a: bool = True,
        max_tenor: Optional[float] = None,
        min_tenor: Optional[float] = None
    ) -> float:
        """
        Calibrate model to spread data.
        
        Parameters
        ----------
        on_full_a : bool, default=True
            Whether to calibrate on full A or aggregate A
        max_tenor : float, optional
            Maximum tenor for calibration
        min_tenor : float, optional
            Minimum tenor for calibration
            
        Returns
        -------
        float
            Root mean square error of calibration
        """
        # if not self.model.is_spread: 
        #     print("Model is not a spread model; skipping spread calibration.")
        #     return 0.0
        if not self.model.is_spread or self.ibor_curve is None: 
            print("Model is not a spread model or no IBOR data; skipping spread calibration.")
            return 0.0
        max_tenor = min(max_tenor or self.config.max_tenor, self.max_positive_a_tenor)
        min_tenor = min_tenor or self.config.min_tenor
        
        # Set up parameter activation for spread calibration
        param_activation = self._get_spread_param_activation()
        
        # Get bounds and starting point
        starting_point = self.get_model_parameters(param_activation)
        bounds = self.constraints.get_parameter_bounds(param_activation)
        print(f"Starting point: {starting_point},   Bounds: {bounds}")
        # Define objective function
        if on_full_a:
            print("Euribor calibration on Full A")
            objective = lambda x: self.objectives.spread_full_objective(
                x, param_activation, max_tenor, min_tenor
            )
        else:
            print("Euribor calibration on Aggregate A")
            objective = lambda x: self.objectives.spread_aggregate_objective(
                x, param_activation, max_tenor, min_tenor
            )
            
        # Optimize
        result = optimize.least_squares(objective, starting_point, bounds=bounds
                                        , verbose=self.optimizer_verbose
                                        ,ftol=SPREAD_FTOL,   # Stop when cost reduction < 1e-6
                                        xtol=SPREAD_FTOL,   # Stop when parameter change < 1e-6  
                                        gtol=SPREAD_FTOL,   # Stop when gradient norm < 1e-6
                                        max_nfev=SPREAD_NFEV  # Hard cap on function evaluations
                                        )
                                        
        
        # Update model parameters
        self.set_model_parameters(result.x, param_activation)
        
        # Calculate and return RMSE
        errors = self.objectives.spread_aggregate_objective(
            result.x, param_activation, max_tenor, min_tenor
        )
        rmse = self.objectives.compute_rmse(errors)
        
        return rmse

    def calibrate_ois_and_spreads(self,
        on_price: bool = True,
        on_full_a: bool = True,
        max_tenor: Optional[float] = None,
        min_tenor: Optional[float] = None
        )->float:
        
        # max_tenor = max_tenor or self.config.max_tenor
        # min_tenor = min_tenor or self.config.min_tenor

        max_tenor = min(max_tenor or self.config.max_tenor, self.max_positive_a_tenor)
        min_tenor = min_tenor or self.config.min_tenor

        # max_tenor_a = min(max_tenor_a or self.config.max_tenor, self.max_positive_a_tenor)
        # min_tenor_a = min_tenor_a or self.config.min_tenor
        

        # Set up parameter activation for OIS calibration
        ##this is fine also where the spread is involved
        ##becuase the rest of model parameters will then be used for OIS instead of for the spread
        ##the model parameters u1 and u2 will help control that
        param_activation = self._get_ois_and_spread_param_activation()
        # In calibrate_ois_and_spreads, before getting param_activation:
       

        # Get bounds and starting point
        starting_point = self.get_model_parameters(param_activation)
        bounds = self.constraints.get_parameter_bounds(param_activation)
        print(f"Starting point: {starting_point},   Bounds: {bounds}")
        
        if self.model.is_spread and self.ibor_curve is not None:
           
            # Define objective function
            if on_price:
                if on_full_a:
                    print("OIS on price and Euribor Spread on full a calibration")
                    objective = lambda x: self.objectives.ois_price_and_spread_full_a_objective(
                        x, param_activation, max_tenor, min_tenor
                    )
                else:
                    print("OIS on Price and Euribor Spread  on Aggregate a calibration")
                    objective = lambda x: self.objectives.ois_price_and_spread_aggregate_a_objective(
                        x, param_activation, max_tenor, min_tenor
                    )
            else:
                if on_full_a:
                    print("OIS on rate and Euribor Spread on full a calibration")
                    objective = lambda x: self.objectives.ois_rate_and_spread_full_a_objective(
                        x, param_activation, max_tenor, min_tenor
                    )
                else:
                    print("OIS on rate and Euribor Spread  on Aggregate a calibration")
                    objective = lambda x: self.objectives.ois_rate_and_spread_aggregate_a_objective(
                        x, param_activation, max_tenor, min_tenor                
                        )
        else:
            # Define objective function
            if on_price:
                print("OIS Calibration on Price")
                objective = lambda x: self.objectives.ois_price_objective(
                    x, param_activation, max_tenor, min_tenor
                )
            else:
                print("OIS Calibration on Yield")
                objective = lambda x: self.objectives.ois_rate_objective(
                    x, param_activation, max_tenor, min_tenor
                )
        # Optimize
        result = optimize.least_squares(objective, starting_point, bounds=bounds
                                        , verbose=self.optimizer_verbose
                                        ,ftol=OIS_FTOL,   # Stop when cost reduction < 1e-6
                                        xtol=OIS_XTOL,   # Stop when parameter change < 1e-6  
                                        gtol=OIS_GTOL,   # Stop when gradient norm < 1e-6
                                        max_nfev=OIS_NFEV  # Hard cap on function evaluations
                                        )
        
        # Update model parameters
        self.set_model_parameters(result.x, param_activation)
        
        # Calculate and return RMSE
        errors = objective(result.x)
        rmse = self.objectives.compute_rmse(errors)
        
        return rmse
        

    def calibrate_swaptions(self) -> float:
        """
        Calibrate model to swaption data using multi-step approach.
        
        Returns
        -------
        float
            Root mean square error of calibration
        """
        print("Starting multi-step swaption calibration...")
        
        TEST_CASE=1#4
        if  TEST_CASE==1:
            # Step 1: Calibrate volatility parameters
            print("Step 1: Calibrating volatility parameters...")
            param_activation = self._get_vol_param_activation()

            ##todo
            # self.model.set_pricing_approach("CollindufresneApprox")
            
            error1 = self._run_swaption_optimization(param_activation)
            
            # Step 2: Calibrate correlation parameters
            print("Step 2: Calibrating correlation parameters...")
            param_activation = self._get_correl_param_activation()
            error2 = self._run_swaption_optimization(param_activation)
            # Step 3: Fine-tune with full pricing
            print("Step 3: Fine-tuning with full pricing...")
            param_activation = self._get_vol_param_activation()
            # self.model.set_pricing_approach("RangeKutta")
            error3 = self._run_swaption_optimization(param_activation)
            ##temporary
            # TEMP_MANUAL_BUMPED_VOL=True
            # if TEMP_MANUAL_BUMPED_VOL:
            #     param_activation = self._get_vol_param_activation()
            #     calibrated_vol = self.get_model_parameters(param_activation)
            #     bumped_calibrated_vol = calibrated_vol * 2.0
            #     self.set_model_parameters(bumped_calibrated_vol, param_activation)
            #     errors_new_swaption = self.objectives.swaption_vol_objective(
            #             bumped_calibrated_vol, param_activation, self.config.use_multi_thread)
                    
        elif TEST_CASE==2:
            
            # Step 1: Calibrate volatility parameters
            print("Step 1 and 2: Calibrating volatility and correlation parameters...")
            param_activation = self._get_vol_correl_param_activation()

            error1 = self._run_swaption_optimization(param_activation)
            # Step 3: Fine-tune with full pricing
            print("Step 3: Fine-tuning with full pricing...")
            param_activation = self._get_vol_diagonal_param_activation() 
            
            # self.model.set_pricing_approach("RangeKutta")
            error3 = self._run_swaption_optimization(param_activation)
        elif TEST_CASE==3:
            
            # Step 1: Calibrate volatility parameters
            print("Step 1 and 2: Calibrating volatility and correlation parameters...")
            param_activation = self._get_vol_diagonal_param_activation()
            error1 = self._run_swaption_optimization(param_activation)

            # Step 3: Fine-tune with full pricing
            print("Step 3: Fine-tuning with full pricing...")
            param_activation = self._get_all_correl_param_activation()
            error3 = self._run_swaption_optimization(param_activation)
            
        elif TEST_CASE==4:
            
            # Step 1: Calibrate volatility parameters
            print("Step 1 : Calibrating volatility  parameters...")
            param_activation = self._get_vol_diagonal_param_activation()
            error1 = self._run_swaption_optimization(param_activation)
            
            # self.pertubate_parameters=True
            # self.parameter_pertubation_amount=1.1
            # print("Step  2 for Sigma[0, 0]: Calibrating volatility  parameters...")
            # param_activation = self._get_vol_diagonal_by_element_param_activation(0)
            # error2 = self._run_swaption_optimization(param_activation)

            # print("Step  2 for Sigma[1,1]: Calibrating volatility  parameters...")
            # param_activation = self._get_vol_diagonal_by_element_param_activation(1)
            # error2 = self._run_swaption_optimization(param_activation)

            # Step 3: Fine-tune with full pricing
            print("Step 3: Fine-tuning with full pricing...")
            param_activation = self._get_all_correl_param_activation()
            error3 = self._run_swaption_optimization(param_activation)
            
            # param_activation =self._get_vol_param_activation()
            # vol_calibrated  = self.get_model_parameters(param_activation)
            # self.set_model_parameters(result.x, param_activation)

        return error3
        
    def _run_swaption_optimization(self, param_activation: List[bool]) -> float:
        """Run swaption optimization for given parameters."""
        starting_point = self.get_model_parameters(param_activation)
        if hasattr(self, 'pertubate_parameters'):
            if self.pertubate_parameters==True:
                # self.parameter_pertubation_amount=1.1
                starting_point= starting_point * self.parameter_pertubation_amount


        bounds = self.constraints.get_parameter_bounds(param_activation)
        print(f"Starting point: {starting_point},   Bounds: {bounds}")
        if self.config.calibrate_on_swaption_vol:
            objective = lambda x: self.objectives.swaption_vol_objective(
                x, param_activation, self.config.use_multi_thread
            )
        else:
            objective = lambda x: self.objectives.swaption_price_objective(
                x, param_activation, self.config.use_multi_thread
            )
            
        # Optimize
        result = optimize.least_squares(objective, starting_point, bounds=bounds
                                         , verbose=self.optimizer_verbose
                                         ,ftol=SWAPTION_FTOL,   # Stop when cost reduction < 1e-6
                                         xtol =SWAPTION_FTOL,   # Stop when parameter change < 1e-6  
                                         gtol =SWAPTION_FTOL,   # Stop when gradient norm < 1e-6
                                         max_nfev=SWAPTION_NFEV  # Hard cap on function evaluations
                                         )
        print(f"result.x: {result.x}    result.cost: {result.cost} ")
        # Update model parameters
        self.set_model_parameters(result.x, param_activation)
        
        # Calculate and return RMSE
        errors = self.objectives.swaption_vol_objective(
            result.x, param_activation, self.config.use_multi_thread
        )
        rmse = self.objectives.compute_rmse(errors)
        
        return rmse
        
    def set_model_parameters_incomplete(
        self,
        params_list: np.ndarray,
        params_activation: List[bool]
    ):
        """
        Set model parameters based on activation flags.
        
        Parameters
        ----------
        params_list : np.ndarray
            Parameter values
        params_activation : List[bool]
            Activation flags for each parameter
        """
        # Implementation details moved from original SetModelParam
        # This method updates the model's parameters based on the activation flags
        # Code structure preserved but with cleaner interface
        
        n_active = sum(params_activation)
        if len(params_list) != n_active:
            raise ValueError("Number of active parameters doesn't match provided values")
            
        # Extract current parameters
        alpha = self.model.alpha
        x0 = self.model.x0.copy()
        omega = self.model.omega.copy()
        m = self.model.m.copy()
        sigma = self.model.sigma.copy()
        
        # Update parameters based on activation flags
        idx = 0
        
        # Alpha
        if params_activation[0]:
            alpha = params_list[idx]
            idx += 1
            
        # x0 diagonal
        for i in range(self.model.n):
            if params_activation[1 + i]:
                x0[i, i] = params_list[idx]
                idx += 1
                
        # x0 off-diagonal
        if params_activation[3]:
            if self.config.calibrate_based_on_correl:
                correl = params_list[idx] * np.sqrt(x0[0, 0] * x0[1, 1])
                x0[0, 1] = x0[1, 0] = correl
            else:
                x0[0, 1] = x0[1, 0] = params_list[idx]
            idx += 1
            
        # Similar updates for omega, m, and sigma...
        # (Full implementation preserved from original)
        
        # Update model
        self.model.set_model_params(self.model.n, alpha, x0, omega, m, sigma)

    def set_model_parameters(
            self,
            params_list: np.ndarray,
            params_activation: List[bool]
        ):
        """
        Set model parameters based on activation flags.
    
        Parameters
        ----------
        params_list : np.ndarray
            Parameter values
        params_activation : List[bool]
            Activation flags for each parameter
            [alpha, x11, x22, x12, omega11, omega22, omega12, m11, m22, sigma11, sigma22, sigma12]
        """
        n_active = sum(params_activation)
        if len(params_list) != n_active:
            raise ValueError("Number of active parameters doesn't match provided values")
        
        # Extract current parameters
        alpha = self.model.alpha
        x0 = self.model.x0.copy()
        omega = self.model.omega.copy()
        m = self.model.m.copy()
        sigma = self.model.sigma.copy()
    
        idx = 0
    
        # Alpha (index 0)
        if params_activation[0]:
            alpha = params_list[idx]
            idx += 1
        
        # x0 diagonal (indices 1, 2)
        start = 1
        for i in range(self.model.n):
            if params_activation[start + i]:
                x0[i, i] = params_list[idx]
                idx += 1
            
        # x0 off-diagonal (index 3)
        if params_activation[start + 2]:
            if self.config.calibrate_based_on_correl:
                x0[0, 1] = x0[1, 0] = params_list[idx] * np.sqrt(x0[0, 0] * x0[1, 1])
            else:
                x0[0, 1] = x0[1, 0] = params_list[idx]
            idx += 1
        
        # Omega diagonal (indices 4, 5)
        start = 4
        for i in range(self.model.n):
            if params_activation[start + i]:
                omega[i, i] = params_list[idx]
                idx += 1
            
        # Omega off-diagonal (index 6)
        if params_activation[start + 2]:
            if self.config.calibrate_based_on_correl:
                omega[0, 1] = omega[1, 0] = params_list[idx] * np.sqrt(omega[0, 0] * omega[1, 1])
            else:
                omega[0, 1] = omega[1, 0] = params_list[idx]
            idx += 1
        
        # M diagonal (indices 7, 8)
        start = 7
        for i in range(self.model.n):
            if params_activation[start + i]:
                m[i, i] = params_list[idx]
                idx += 1
            
        # Sigma diagonal (indices 9, 10)
        start = 9
        for i in range(self.model.n):
            if params_activation[start + i]:
                sigma[i, i] = params_list[idx]
                idx += 1
            
        # Sigma off-diagonal (index 11)
        if params_activation[start + 2]:
            if self.config.calibrate_based_on_correl:
                sigma[0, 1] = sigma[1, 0] = params_list[idx] * np.sqrt(sigma[0, 0] * sigma[1, 1])
            else:
                sigma[0, 1] = sigma[1, 0] = params_list[idx]
            idx += 1
        
        # Update model
        self.model.set_model_params(self.model.n, alpha, x0, omega, m, sigma)
    
    def get_model_parameters_old(
        self,
        params_activation: Optional[List[bool]] = None
    ) -> np.ndarray:
        """
        Get model parameters based on activation flags.
        
        Parameters
        ----------
        params_activation : List[bool], optional
            Activation flags for each parameter
            
        Returns
        -------
        np.ndarray
            Active parameter values
        """
        if params_activation is None:
            params_activation = [True] * 12  # All parameters
            
        params_list = []
        
        # Extract parameters based on activation flags
        # (Implementation preserved from original GetModelParam)
        
        paramsList=[]
        # nbActive=paramsActivationFlag.count(True)
        
        alpha=self.model.alpha
        x0=self.model.x0
        omega=self.model.omega
        m=self.model.m
        sigma=self.model.sigma
         
        setIndex=0
        ##Setting alpha
        if params_activation[0]:
            paramsList.append(alpha)
            setIndex+=1
        

        ##Getting x0
        startCheck=1          
        ##diagonal of x0
        for i in range(self._model.n ):
            if params_activation[startCheck+i]:
              paramsList.append(x0[i,i])
              setIndex+=1
        ##non-diagonal of x0
        if  params_activation[startCheck+2]:
            paramsList.append(x0[0,1])
            
            setIndex+=1
        
        ##Getting Omega
        startCheck=1 + 3               
        ##diagonal of omega
        for i in range(self._model.n ):
            if params_activation[startCheck+i]:
              paramsList.append(omega[i,i])
              setIndex+=1
        ##non-diagonal of omega
        if  params_activation[startCheck+2]:
            paramsList.append(omega[0,1])
            setIndex+=1
             
        ##Getting m
        startCheck=1 +3  +3              
        for i in range(self._model.n ):
            if params_activation[startCheck+i]:
              paramsList.append(m[i,i])
              setIndex+=1     
         
        ##getting Sigma
        startCheck=1 +3 +3 +2            
        ##diagonal of sigma
        for i in range(self._model.n ):
            if params_activation[startCheck+i]:
              paramsList.append(sigma[i,i])
              setIndex+=1
        ##non-diagonal of sigma
        if  params_activation[startCheck+2]:
            paramsList.append(sigma[0,1])
            setIndex+=1
         
        # return paramsList
        return np.array(paramsList)
    
    def get_model_parameters(
        self,
        params_activation: Optional[List[bool]] = None
        ) -> np.ndarray:
            """
            Get model parameters based on activation flags.
    
            Parameters
            ----------
            params_activation : List[bool], optional
                Activation flags for each parameter
                [alpha, x11, x22, x12, omega11, omega22, omega12, m11, m22, sigma11, sigma22, sigma12]
        
            Returns
            -------
            np.ndarray
                Active parameter values
            """
            if params_activation is None:
                params_activation = [True] * 12  # All parameters
        
            params_list = []
    
            # Extract current parameters
            alpha = self.model.alpha
            x0 = self.model.x0
            omega = self.model.omega
            m = self.model.m
            sigma = self.model.sigma
    
            # Alpha (index 0)
            if params_activation[0]:
                params_list.append(alpha)
    
            # x0 diagonal (indices 1, 2)
            start = 1
            for i in range(self.model.n):
                if params_activation[start + i]:
                    params_list.append(x0[i, i])
            
            # x0 off-diagonal (index 3)
            if params_activation[start + 2]:
                if self.config.calibrate_based_on_correl:
                    correl=x0[0, 1] / np.sqrt(x0[0, 0] * x0[1, 1])
                    if correl==0.0:
                        correl=DEFAULT_START_CORRELATION
                    params_list.append(correl)
                else:
                    params_list.append(x0[0, 1])
    
            # Omega diagonal (indices 4, 5)
            start = 4
            for i in range(self.model.n):
                if params_activation[start + i]:
                    params_list.append(omega[i, i])
            
            # Omega off-diagonal (index 6)
            if params_activation[start + 2]:
                if self.config.calibrate_based_on_correl:
                     
                    correl=omega[0, 1] / np.sqrt(omega[0, 0] * omega[1, 1])
                     
                    if correl==0.0:
                        correl=DEFAULT_START_CORRELATION
                    params_list.append(correl)
                else:
                    params_list.append(omega[0, 1])
         
            # M diagonal (indices 7, 8)
            start = 7
            for i in range(self.model.n):
                if params_activation[start + i]:
                    params_list.append(m[i, i])
     
            # Sigma diagonal (indices 9, 10)
            start = 9
            for i in range(self.model.n):
                if params_activation[start + i]:
                    params_list.append(sigma[i, i])
            
            # Sigma off-diagonal (index 11)
            if params_activation[start + 2]:
                if self.config.calibrate_based_on_correl:
                     
                    correl=sigma[0, 1] / np.sqrt(sigma[0, 0] * sigma[1, 1])
                    
                    if correl==0.0:
                        correl=DEFAULT_START_CORRELATION
                    params_list.append(correl)
                else:
                    params_list.append(sigma[0, 1])
     
            return np.array(params_list)

    def validate_calibration(self, base_model: Optional[LRWModel] = None):
        """
        Validate calibration results.
        
        Parameters
        ----------
        base_model : LRWModel, optional
            Base model for comparison
        """
        if base_model:
            # Check OIS parameters
            self.ois_params_replaced = self.constraints.check_parameter_ratios(
                self.model, base_model, "ois", self.config.max_ratio_params
            )
            
            # Check spread parameters
            self.spread_params_replaced = self.constraints.check_parameter_ratios(
                self.model, base_model, "spread", self.config.max_ratio_params
            )
            
        # Validate Gindikin condition
        if not self.constraints.check_gindikin_condition():
            print("Warning: Gindikin condition not satisfied!")
            
    def generate_report(self, output_dir: str = ".", reprice_instruments: bool = False):
        """
        Generate calibration report.
        
        Parameters
        ----------
        output_dir : str
            Directory for output files
        """
        if  reprice_instruments:
            # Reprice instruments with calibrated model
             
            self.objectives.reprice_bond_market_data()
            if self.ibor_curve is not None:
                self.objectives.reprice_spreads_market_data(full_a=True,
                                                        max_tenor = self.config.max_tenor)
        
            # Update market data after calibration
            self.market_handler.update_swaption_market_data(
                                    model=self.model,
                                    market_based_strike  = True)
            self.objectives._reprice_bonds()#.reprice_bond_market_data()
            if self.ibor_curve is not None:
                self.objectives._reprice_spreads()#.reprice_spreads_market_data(full_a=True, max_tenor=self.config.max_tenor)
            self.objectives._reprice_swaptions()
             
        self.reporter.generate_full_report(
            self.model,
            self.daily_data,
            self.calibration_results,
            output_dir
        )
        
    # Helper methods for parameter activation
    def _get_ois_param_activation(self) -> List[bool]:
        """Get parameter activation for OIS calibration."""
        activation = [False] * 12
        activation[1] = True  # x0[0,0]
        activation[4] = True  # omega[0,0]
        activation[7] = True  # m[0,0]
        if self.model.u1[self.model.n - 1,self.model.n - 1] != 0.0:
            activation[2] = True  # x0[1,1]
            activation[5] = True  # omega[1,1]
            activation[8] = True  # m[1,1]
        if self.model.u1[0,self.model.n - 1] != 0.0: 
            activation[3] = True  # x0[0,1]
            activation[6] = True  # omega[0,1]

        return activation
    
    # Helper methods for parameter activation
    def _get_ois_and_spread_param_activation(self) -> List[bool]:
        """Get parameter activation for OIS calibration."""
        activation = [False] * 12
        activation[1] = True  # x0[0,0]
        activation[4] = True  # omega[0,0]
        activation[7] = True  # m[0,0]
        
        activation[2] = True  # x0[1,1]
        activation[5] = True  # omega[1,1]
        activation[8] = True  # m[1,1]
        if (self.model.u1[0,self.model.n - 1] != 0.0) : 
            activation[3] = True  # x0[0,1]
            activation[6] = True  # omega[0,1]
        ##just in case we have u2 with non-diagonal elements
        if self.model.is_spread:
            if self.model.u2[0,self.model.n - 1] != 0.0: 
                activation[3] = True  # x0[0,1]
                activation[6] = True  # omega[0,1]
        return activation
    
    def _get_spread_param_activation(self) -> List[bool]:
        """Get parameter activation for spread calibration."""
        activation = [False] * 12
        if self.model.is_spread:
            activation[2] = True  # x0[1,1]
            activation[5] = True  # omega[1,1]
            activation[8] = True  # m[1,1]
            if self.model.u2[0,self.model.n - 1] != 0.0: 
                    activation[3] = True  # x0[0,1]
                    activation[6] = True  # omega[0,1]
        return activation
        
    def _get_vol_param_activation(self) -> List[bool]:
        """Get parameter activation for volatility calibration."""
        activation = [False] * 12
        activation[9] = True   # sigma[0,0]
        activation[10] = True  # sigma[1,1]
        activation[11] = True  # sigma[0,1]
        return activation
    
    # _get_vol_diagonal_by_element_param_activation
    def _get_vol_diagonal_by_element_param_activation(self, id) -> List[bool]:
        """Get parameter activation for volatility calibration."""
        activation = [False] * 12
        if id==0:
            activation[9] = True   # sigma[0,0]
        elif id==1:
            activation[10] = True  # sigma[1,1]
        # # activation[11] = True  # sigma[0,1]
        return activation
    
    def _get_vol_diagonal_param_activation(self) -> List[bool]:
        """Get parameter activation for volatility calibration."""
        activation = [False] * 12
        activation[9] = True   # sigma[0,0]
        activation[10] = True  # sigma[1,1]
        # # activation[11] = True  # sigma[0,1]
        return activation
    
    def _get_vol_correl_param_activation(self) -> List[bool]:
        """Get parameter activation for volatility calibration."""
        activation = [False] * 12
        activation[3] = True  # x0[0,1]
        activation[6] = True  # omega[0,1]
        activation[9] = True   # sigma[0,0]
        activation[10] = True  # sigma[1,1]
        activation[11] = True  # sigma[0,1]
        return activation
        
    def _get_correl_param_activation(self) -> List[bool]:
        """Get parameter activation for correlation calibration."""
        activation = [False] * 12
        activation[3] = True  # x0[0,1]
        activation[6] = True  # omega[0,1]
        return activation
    
    def _get_all_correl_param_activation(self) -> List[bool]:
        """Get parameter activation for correlation calibration."""
        activation = [False] * 12
        activation[3] = True  # x0[0,1]
        activation[6] = True  # omega[0,1]
        activation[11] = True  # sigma[0,1]
        return activation
     
    def set_alpha(self, alpha_tenor: float=None) -> None:
        """
        Set alpha parameters from OIS curves.
        
        Parameters
        ----------
        alpha_tenor : float
            Tenor for alpha calculation
        """
        
        # print(f"Setting alpha parameters for tenor: {alpha_tenor} years")
        # print("===================================================================")

        if alpha_tenor is None:
            # Use default tenor if not provided
            if self.config.alpha_tenor is  None:            
                alpha_tenor = self.config.max_tenor
            else:
                alpha_tenor = self.config.alpha_tenor
        
        if not isinstance(alpha_tenor, (int, float)):
           raise ValueError("alpha_tenor must be a number")

        print("===================================================================")
        print(f"Setting alpha parameters for tenor: {alpha_tenor} years")
        print("===================================================================")

        alpha_tenor = float(alpha_tenor)  # Ensure it's a float
        alpha = self.ois_curve.bond_zc_rate(alpha_tenor)
        self.model.set_alpha(alpha)
        
    
