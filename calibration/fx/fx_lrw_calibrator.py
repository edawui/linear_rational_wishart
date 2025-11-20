"""
Main LRW FX model calibrator.

This module provides the main calibration class for the Linear-Rational Wishart
FX model, integrating objectives, optimizers, and market data processing.
"""

from sre_constants import MIN_REPEAT_ONE
from typing import Dict, List, Optional, Tuple, Union, Any, cast
import numpy as np
import pandas as pd
import time
from collections import defaultdict

from .base import (
    FXCalibratorBase, CalibrationConfig, CalibrationResult,
    ParameterBounds, ParameterActivation, OptimizationMethod
)
from .objectives import OptionPricingObjective, create_objective
from .optimizers import create_optimizer

from ...data.market_repricing_option import OptionRepricer
# from ...data import MarketData
from ...data.data_fx_market_data import CurrencyPairDailyData,CalibWeightType
from ...data.market_repricing_bond import BondRepricer


from ...curves.oiscurve_jax import OisCurve
from ...curves.iborcurve_jax import IborCurve
from ...models.fx.lrw_fx import LRWFxModel
from ...pricing.black_scholes import black_scholes_price_fx # black_scholes_call_fx #, implied_volatility_fx
from ..calibration_parameter_utils import ParameterManager


class LrwFxCalibrator(FXCalibratorBase):
    """
    Calibrator for Linear-Rational Wishart FX models.
    
    This class handles the calibration of LRW FX models to market data,
    including bonds and FX options.
    """
    
    def __init__(
        self,
        daily_data: CurrencyPairDailyData,
        model: LRWFxModel,
        config: Optional[CalibrationConfig] = None
    ):
        """
        Initialize LRW FX calibrator.
        
        Parameters
        ----------
        daily_data : CurrencyPairDailyData
            Market data for calibration
        model : LRWFxModel
            LRW FX model instance
        config : CalibrationConfig, optional
            Calibration configuration
        """
        super().__init__(daily_data, model, config)
        
        # Initialize parameter manager
        self.parameter_manager = ParameterManager(model)
        
        # Initialize repricers
        self.bond_repricer = BondRepricer(
            self.domestic_ois_curve,
            self.foreign_ois_curve,
            model.lrw_currency_i,
            model.lrw_currency_j
        )
        
        self.option_repricer = OptionRepricer(
            self.domestic_ois_curve,
            self.foreign_ois_curve,
            model,
            config.pricing_method.value if config else "MC"
        )
        
        # Calibration state
        self.calibration_options = []
        self.parameter_activation = None
        self.parameter_bounds = None
        
        self.daily_data.reprice_all_option = self.config.reprice_all_option
        self.set_option_weight()
        # Get calibration instruments
        self._select_calibration_options()
    
    def _initialize_curves(self) -> None:
        """Initialize interest rate curves."""
        self.domestic_ois_curve = OisCurve(
            self.daily_data.quotation_date,
            self.daily_data.domestic_currency_daily_data.ois_rate_data
        )
        
        self.foreign_ois_curve = OisCurve(
            self.daily_data.quotation_date,
            self.daily_data.foreign_currency_daily_data.ois_rate_data
        )
    
    def _process_market_data(self) -> None:
        """Process raw market data for calibration."""
        # Process FX volatility data
        self.daily_data.process_raw_fx_vol_data(
            self.domestic_ois_curve,
            self.foreign_ois_curve,
            min_maturity=self.config.min_maturity_option,
            max_maturity=self.config.max_maturity_option
        )
        
        # Calculate market prices for options
        self._process_market_prices()
    
    def _process_market_prices(self) -> None:
        """Calculate market prices for FX options."""
        self.fx_spot = self.daily_data.fx_spot
        
        for fx_option in self.daily_data.fx_vol_data:
            T = fx_option.expiry_maturity #expiry_mat
            r_d = self.domestic_ois_curve.bond_zc_rate(T)
            r_f = self.foreign_ois_curve.bond_zc_rate(T)
            
            # Calculate Black-Scholes price
            mkt_price = black_scholes_price_fx(
                self.fx_spot,
                fx_option.strike,
                T,
                r_d,
                r_f,
                fx_option.vol,
                fx_option.call_or_put
            )
            
            fx_option.market_price = mkt_price
    
    def _select_calibration_options(self) -> None:
        """Select options for calibration based on configuration."""
        options_by_maturity = defaultdict(list)
        all_options = []
        
        for option_data in self.daily_data.fx_vol_data:
            mat = option_data.expiry_maturity#_mat
            if (self.config.min_maturity_option <= mat <= self.config.max_maturity_option):
                options_by_maturity[mat].append(option_data)
                all_options.append(option_data)
        
        if self.config.use_atm_only:
            # Select ATM options only
            atm_options = []
            for mat, options in options_by_maturity.items():
                options_sorted = sorted(options, key=lambda x: x.strike)
                if len(options_sorted) >= 3:
                    # Select middle strike (typically ATM)
                    atm_options.append(options_sorted[len(options_sorted) // 2])
                else:
                    print(f"Warning: Not enough strikes for maturity {mat}")
            self.calibration_options = atm_options
        else:

            if self.config.selected_based_on_strike:
                min_option_index=self.config.min_option_strike_index
                max_option_index=self.config.max_option_strike_index
                ##same selction basedon strike for all matrurities
                if not self.config.custom_calibration_option_selection:
                    for mat, options in options_by_maturity.items():
                        options_sorted = sorted(options, key=lambda x: x.strike)
                        strikes= [opt.strike for opt in options_sorted]
                        strike_min_for_calib= strikes[min_option_index]
                        strike_max_for_calib= strikes[max_option_index]
                        for option in options_sorted:
                            if (strike_min_for_calib <= option.strike <= strike_max_for_calib):
                                self.calibration_options.append(option)                
                else:
                    ## Different option selection for option above certain strikes
                    ## Two-stage calibration: select options based on strike range
                    second_part_min_option_index=self.config.second_stage_min_option_strike_index
                    second_part_max_option_index=self.config.second_stage_max_option_strike_index
                    for mat, options in options_by_maturity.items():
                        options_sorted = sorted(options, key=lambda x: x.strike)
                        strikes = [opt.strike for opt in options_sorted]

                        if mat <= self.config.maturity_max_first_part:
                            strike_min_for_calib = strikes[min_option_index]
                            strike_max_for_calib = strikes[max_option_index]
                        else:
                            strike_min_for_calib = strikes[second_part_min_option_index]
                            strike_max_for_calib = strikes[second_part_max_option_index]

                        for option in options_sorted:
                            if (strike_min_for_calib <= option.strike <= strike_max_for_calib):
                               self.calibration_options.append(option)
            
            else:
                self.calibration_options = all_options

        self.daily_data.fx_vol_calib_data = self.calibration_options
        print(f"Selected {len(self.calibration_options)} options for calibration")
    

    def set_option_weight(self):#, weight_tpe:CalibWeightType= CalibWeightType.UNIFORM, custom_weigh:float=1.0):
        self.daily_data.set_weight(self.config.weight_type, self.config.custom_weight)

    def get_calibration_instruments(self) -> List[Any]:
        """Get instruments used for calibration."""
        return self.calibration_options
    
    def set_model_parameters(self, parameters: np.ndarray) -> None:
        """
        Set model parameters from parameter array.
        
        Parameters
        ----------
        parameters : np.ndarray
            Parameter values
        """
        if self.parameter_activation is None:
            raise ValueError("Parameter activation not set")
        
        self.parameter_manager.set_parameters(
            parameters,
            self.parameter_activation,
            self.config.calibrate_based_on_correlation
        )
        # Initialize repricers
        self.bond_repricer = BondRepricer(
            self.domestic_ois_curve,
            self.foreign_ois_curve,
            self.model.lrw_currency_i,
            self.model.lrw_currency_j
        )
        
        self.option_repricer = OptionRepricer(
            self.domestic_ois_curve,
            self.foreign_ois_curve,
            self.model,
            self.config.pricing_method.value if self.config else "MC"
        )
    
    def get_model_parameters(self) -> np.ndarray:
        """
        Get current model parameters.
        
        Returns
        -------
        np.ndarray
            Current parameter values
        """
        if self.parameter_activation is None:
            raise ValueError("Parameter activation not set")
        
        return self.parameter_manager.get_parameters(self.parameter_activation)
    
    def set_boundary_models(
        self,
        lower_bound_model: LRWFxModel,
        upper_bound_model: LRWFxModel
        ) -> None:
        
           self.lower_bound_model=lower_bound_model
           self.upper_bound_model=upper_bound_model
        

    def set_parameter_bounds(
        self
        # ,lower_bound_model: LRWFxModel,
        # upper_bound_model: LRWFxModel
    ) -> None:
        """
        Set parameter bounds from model instances.
        
        Parameters
        ----------
        lower_bound_model : LRWFxModel
            Model with lower bound parameters
        upper_bound_model : LRWFxModel
            Model with upper bound parameters
        """
        if self.parameter_activation is None:
            raise ValueError("Parameter activation must be set first")
        
        lower_params = self.parameter_manager.get_parameters_from_model(
            self.lower_bound_model, #lower_bound_model,
            self.parameter_activation

            , self.config.calibrate_based_on_correlation
            , self.config.correlation_min

        )
        upper_params = self.parameter_manager.get_parameters_from_model(
            self.upper_bound_model,# upper_bound_model,
            self.parameter_activation

            , self.config.calibrate_based_on_correlation
            , self.config.correlation_max
        )
        
        self.parameter_bounds = ParameterBounds(lower_params, upper_params)
    
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
            if self.config.default_alpha_tenor is  None:            
                alpha_tenor = self.config.max_maturity_zc
            else:
                alpha_tenor = self.config.default_alpha_tenor
        
        if not isinstance(alpha_tenor, (int, float)):
           raise ValueError("alpha_tenor must be a number")

        print("===================================================================")
        print(f"Setting alpha parameters for tenor: {alpha_tenor} years")
        print("===================================================================")

        alpha_tenor = float(alpha_tenor)  # Ensure it's a float
        alpha_i = self.domestic_ois_curve.bond_zc_rate(alpha_tenor)
        alpha_j = self.foreign_ois_curve.bond_zc_rate(alpha_tenor)
        
        # self.model.alpha_i = alpha_i
        # self.model.alpha_j = alpha_j
        if not self.config.set_same_alpha:
            self.model.set_alphas(alpha_i, alpha_j)
        else:
            alpha_max = max(alpha_i, alpha_j)
            self.model.set_alphas(alpha_max, alpha_max)

    
    def set_pseudo_inverse_smoothing(self) -> None:
        """Enable pseudo-inverse smoothing for the model."""
        # Implementation would go here
        # This is a placeholder for the actual pseudo-inverse smoothing logic
        if self.config.pseudo_inverse_smoothing:
            print("Pseudo-inverse smoothing enabled")
    
    def reprice_bonds(self) -> None:
        """Reprice all bonds with current model parameters."""
        self.bond_repricer = BondRepricer(
            self.domestic_ois_curve,
            self.foreign_ois_curve,
            self.model.lrw_currency_i,
            self.model.lrw_currency_j
        )
        self.bond_repricer.reprice_all(
            self.daily_data.domestic_currency_daily_data.ois_rate_data,
            self.daily_data.foreign_currency_daily_data.ois_rate_data
        )
    
    def reprice_options(self) -> None:
        """Reprice calibration options with current model parameters."""
        if self.config.use_multithreading:
            self.option_repricer.reprice_options_multithread(self.calibration_options)
        else:
            self.option_repricer.reprice_options(self.calibration_options)
        
        # print(self.config.pricing_method)
        # print(self.daily_data.option_summary())

    
    def reprice_instruments(self) -> Dict[str, np.ndarray]:
        """
        Reprice all calibration instruments.
        
        Returns
        -------
        Dict[str, np.ndarray]
            Repriced values by instrument type
        """
        # Reprice bonds
        self.reprice_bonds()
        
        # # Reprice options
        # self.reprice_options()
           
        # Collect results
        results = {
            'bond_prices': [],
            'bond_yields': [],
            'option_prices': [],
            'option_vols': []
        }
        
        # Bonds
        for _, ois_data in self.daily_data.domestic_currency_daily_data.ois_rate_data.iterrows():
            obj = ois_data["Object"]
            results['bond_prices'].append(obj.model_zc_price)
            results['bond_yields'].append(obj.model_zc_rate)
        
        for _, ois_data in self.daily_data.foreign_currency_daily_data.ois_rate_data.iterrows():
            obj = ois_data["Object"]
            results['bond_prices'].append(obj.model_zc_price)
            results['bond_yields'].append(obj.model_zc_rate)
        
        if self.config.reprice_all_option:
        # Options
         # # Reprice options
            # self.reprice_options()
            if self.config.use_multithreading:
                self.option_repricer.reprice_options_multithread(self.daily_data.fx_vol_data)
            else:
                self.option_repricer.reprice_options(self.daily_data.fx_vol_data)
        
            for option in self.daily_data.fx_vol_data:
                results['option_prices'].append(option.model_price)
                results['option_vols'].append(option.model_vol)
        else:
            # self.reprice_options()
            if self.config.use_multithreading:
                self.option_repricer.reprice_options_multithread(self.calibration_options)
            else:
                self.option_repricer.reprice_options(self.calibration_options)

            for option in self.daily_data.fx_vol_calib_data:
                results['option_prices'].append(option.model_price)
                results['option_vols'].append(option.model_vol)

        return {k: np.array(v) for k, v in results.items()}
    
    def calibrate_ois(
        self,
        calibrate_on_price: bool = True,
        # max_tenor: float = 11.0,
        # min_tenor: float = 1.0,
        joint_calibrate_domestic_and_foreign=True
        , domestic_first=True) -> float:
        """
        Calibrate model to OIS curves.
        
        Parameters
        ----------
        calibrate_on_price : bool
            If True, calibrate on prices; if False, on yields
        max_tenor : float
            Maximum tenor for calibration
        min_tenor : float
            Minimum tenor for calibration
            
        Returns
        -------
        float
            Final RMSE error
        """
        max_tenor=self.config.max_maturity_zc
        min_tenor=self.config.min_maturity_zc

        print(f"\n{'='*80}")
        print("OIS CALIBRATION")
        print(f"{'='*80}")
        print(f"Calibrating on: {'Price' if calibrate_on_price else 'Yield'}")
        print(f"Tenor range: {min_tenor} - {max_tenor} years")
        
        if joint_calibrate_domestic_and_foreign:
            print("joint calibration for domestic and foreign OIS curves")
            # Set up parameter activation for OIS calibration
            param_activation = ParameterActivation(self.model.n)
            param_activation.activate(['x0_00','x0_11', 'omega_00', 'omega_11', 'm_00', 'm_11'])
            self.parameter_activation = param_activation.to_list()
        
            # Get initial parameters
            initial_params = self.get_model_parameters()
            self.set_parameter_bounds()
            # Create objective
            bond_objective = create_objective(
                'bond',
                self,
                min_tenor=min_tenor,
                max_tenor=max_tenor,
                calibrate_on_price=calibrate_on_price,
                calibrate_on_domestic = True,
                calibrate_on_foreign = True,  
                use_jax=self.config.use_jax
            )


            # self.config.max_iterations=50 ##temporary fix for max iterations
            # Create optimizer
            optimizer = create_optimizer(
                'least_squares',
                bond_objective,
                self.parameter_bounds,
                ftol=self.config.relative_tolerance,
                xtol=self.config.absolute_tolerance,
                max_nfev=self.config.max_iterations
               , gtol=self.config.gradient_tolerance
               , verbose= self.config.verbose
                )
            # Run optimization
            result = optimizer.optimize(initial_params, ['x0[0,0]','x0[1,1]', 'omega[0,0]', 'omega[1,1]', 'm[0,0]', 'm[1,1]'])
        
            # Update model with final parameters
            self.set_model_parameters(result.final_parameters)
        
            print(f"\nOIS calibration completed with RMSE: {result.final_error:.8f}")
        
            return result.final_error

        else:
           
            print("separete calibration for domestic and foreign OIS curves")

            if domestic_first:
                print("Calibration for domestic OIS curves")
                # Set up parameter activation for OIS calibration
                param_activation = ParameterActivation(self.model.n)
                param_activation.activate(['x0_00', 'omega_00',  'm_00'])
                self.parameter_activation = param_activation.to_list()
        
                # Get initial parameters
                initial_params = self.get_model_parameters()
                self.set_parameter_bounds()
                # Create objective
                bond_objective_1 = create_objective(
                    'bond',
                    self,
                    min_tenor=min_tenor,
                    max_tenor=max_tenor,
                    calibrate_on_price=calibrate_on_price,
                    calibrate_on_domestic = True,
                    calibrate_on_foreign = False,  
                    use_jax=self.config.use_jax
                )


                # self.config.max_iterations=50 ##temporary fix for max iterations
                # Create optimizer
                optimizer_1 = create_optimizer(
                    'least_squares',
                    bond_objective_1,
                    self.parameter_bounds,
                    ftol=self.config.relative_tolerance,
                    xtol=self.config.absolute_tolerance,
                    max_nfev=self.config.max_iterations
                    , gtol=self.config.gradient_tolerance
                    , verbose= self.config.verbose
                )
        
                # Run optimization
                result_1 =  optimizer_1.optimize(initial_params, ['x0[1,1]', 'omega[0,0]','m[0,0]'])
        
                # Update model with final parameters
                self.set_model_parameters(result_1.final_parameters)
        
                print(f"\nOIS domestic curve calibration completed with RMSE: {result_1.final_error:.8f}")
        

                print("Calibration for foreign OIS curves")
                # Set up parameter activation for OIS calibration
                param_activation = ParameterActivation(self.model.n)
                param_activation.activate(['x0_11', 'omega_11','m_11'])
                self.parameter_activation = param_activation.to_list()
        
                # Get initial parameters
                initial_params = self.get_model_parameters()
                self.set_parameter_bounds()
                # Create objective
                bond_objective_2 = create_objective(
                    'bond',
                    self,
                    min_tenor=min_tenor,
                    max_tenor=max_tenor,
                    calibrate_on_price=calibrate_on_price,
                    calibrate_on_domestic = False,
                    calibrate_on_foreign = True,  
                    use_jax=self.config.use_jax
                )


                # self.config.max_iterations=50 ##temporary fix for max iterations
                # Create optimizer
                optimizer_2 = create_optimizer(
                    'least_squares',
                    bond_objective_2,
                    self.parameter_bounds,
                    ftol=self.config.relative_tolerance,
                    xtol=self.config.absolute_tolerance,
                    max_nfev=self.config.max_iterations
                    , gtol=self.config.gradient_tolerance
                    , verbose= self.config.verbose
                    )
        
                # Run optimization
                result_2 =  optimizer_2.optimize(initial_params, ['x0[1,1]',  'omega[1,1]', 'm[1,1]'])
        
                # Update model with final parameters
                self.set_model_parameters(result_2.final_parameters)
            
                print(f"\nOIS foreign curve calibration completed with RMSE: {result_2.final_error:.8f}")
           
            else:

               

                print("Calibration for foreign OIS curves")
                # Set up parameter activation for OIS calibration
                param_activation = ParameterActivation(self.model.n)
                param_activation.activate(['x0_11', 'omega_11','m_11'])
                self.parameter_activation = param_activation.to_list()
        
                # Get initial parameters
                initial_params = self.get_model_parameters()
                self.set_parameter_bounds()
                # Create objective
                bond_objective_2 = create_objective(
                    'bond',
                    self,
                    min_tenor=min_tenor,
                    max_tenor=max_tenor,
                    calibrate_on_price=calibrate_on_price,
                    calibrate_on_domestic = False,
                    calibrate_on_foreign = True,  
                    use_jax=self.config.use_jax
                )


                # self.config.max_iterations=50 ##temporary fix for max iterations
                # Create optimizer
                optimizer_2 = create_optimizer(
                    'least_squares',
                    bond_objective_2,
                    self.parameter_bounds,
                    ftol=self.config.relative_tolerance,
                    xtol=self.config.absolute_tolerance,
                    max_nfev=self.config.max_iterations
                    , gtol=self.config.gradient_tolerance
                    , verbose= self.config.verbose
                    )
        
                # Run optimization
                result_2 =  optimizer_2.optimize(initial_params, ['x0[1,1]',  'omega[1,1]', 'm[1,1]'])
        
                # Update model with final parameters
                self.set_model_parameters(result_2.final_parameters)
            
                print(f"\nOIS foreign curve calibration completed with RMSE: {result_2.final_error:.8f}")
                # self.model.print_model()

                print("Calibration for domestic OIS curves")
                # Set up parameter activation for OIS calibration
                param_activation = ParameterActivation(self.model.n)
                param_activation.activate(['x0_00', 'omega_00',  'm_00'])
                self.parameter_activation = param_activation.to_list()
        
                # Get initial parameters
                initial_params = self.get_model_parameters()
                self.set_parameter_bounds()
                # Create objective
                bond_objective_1 = create_objective(
                    'bond',
                    self,
                    min_tenor=min_tenor,
                    max_tenor=max_tenor,
                    calibrate_on_price=calibrate_on_price,
                    calibrate_on_domestic = True,
                    calibrate_on_foreign = False,  
                    use_jax=self.config.use_jax
                )


                # self.config.max_iterations=50 ##temporary fix for max iterations
                # Create optimizer
                optimizer_1 = create_optimizer(
                    'least_squares',
                    bond_objective_1,
                    self.parameter_bounds,
                    ftol=self.config.relative_tolerance,
                    xtol=self.config.absolute_tolerance,
                    max_nfev=self.config.max_iterations
                    , gtol=self.config.gradient_tolerance
                    , verbose= self.config.verbose
                    )
        
                # Run optimization
                result_1 =  optimizer_1.optimize(initial_params, ['x0[1,1]', 'omega[0,0]','m[0,0]'])
        
                # Update model with final parameters
                self.set_model_parameters(result_1.final_parameters)
        
                print(f"\nOIS domestic curve calibration completed with RMSE: {result_1.final_error:.8f}")
        
            return   result_1.final_error+result_2.final_error
    
    def calibrate_options(
        self,
        calibrate_on_vol: bool = True,
        calibration_steps: Optional[List[int]] = None
    ) -> CalibrationResult:
        """
        Calibrate model to FX options.
        
        Parameters
        ----------
        calibrate_on_vol : bool
            If True, calibrate on implied volatilities
        calibration_steps : List[int], optional
            Which calibration steps to perform (default: [1, 2])
            
        Returns
        -------
        CalibrationResult
            Calibration results
        """
        if calibration_steps is None:
            calibration_steps = [1, 2]
        
        print(f"\n{'='*80}")
        print(f"FX OPTION CALIBRATION")
        print(f"{'='*80}")
        print(f"Calibrating on: {'Volatility' if calibrate_on_vol else 'Price'}")
        print(f"Optimization method: {self.config.optimization_method.value}")
        print(f"Calibration steps: {calibration_steps}")
        
        final_result = None
        
        for step in calibration_steps:
            print(f"\n{'='*60}")
            print(f"CALIBRATION STEP {step}")
            print(f"{'='*60}")
            
            # Set parameter activation based on step
            param_activation = ParameterActivation(self.model.n)
            
            if step == 1:
                # Calibrate volatility parameters
                param_activation.activate(['sigma_00', 'sigma_11'])
                param_names = ['sigma[0,0]', 'sigma[1,1]']
                print("🎯 Calibrating: Sigma parameters (volatility structure)")
            
            elif step == 2:
                # Calibrate correlation parameters
                param_activation.activate(['x0_01', 'omega_01', 'sigma_01'])
                param_names = ['x0[0,1]', 'omega[0,1]', 'sigma[0,1]']
                print("🎯 Calibrating: Correlation parameters")

            elif step == 3:
                # Calibrate correlation parameters
                # param_activation.activate(['x0_01', 'omega_01', 'sigma_00', 'sigma_11','sigma_01'])
                # param_names = ['x0[0,1]', 'omega[0,1]','sigma[0,0]', 'sigma[1,1]', 'sigma[0,1]']
                # print("🎯 Calibrating: Sigma and Correlation parameters")
            
                param_activation.activate(['sigma_00', 'sigma_11','sigma_01'])
                param_names = ['sigma[0,0]', 'sigma[1,1]', 'sigma[0,1]']
                print("🎯 Calibrating: Sigma and Correlation parameters")
             
            elif step == 4:
                #Calibrate correlation parameters
                param_activation.activate(['x0_01', 'omega_01'])
                param_names = ['x0[0,1]', 'omega[0,1]']
                print("🎯 Calibrating: x0 and omega Correlation parameters")
            elif step == 5:
                #Calibrate correlation parameters
                param_activation.activate(['x0_01'])
                param_names = ['x0[0,1]']
                print("🎯 Calibrating: x0  Correlation parameters")
            elif step == 6:
                #Calibrate correlation parameters
                param_activation.activate(['omega_01'])
                param_names = ['omega[0,1]']
                print("🎯 Calibrating: omega  Correlation parameters")
            
            elif step == 7:
                #Calibrate all options parameters
                
                param_activation.activate(['x0_01', 'omega_01', 'sigma_00', 'sigma_11','sigma_01'])
                param_names = ['x0[0,1]', 'omega[0,1]','sigma[0,0]', 'sigma[1,1]', 'sigma[0,1]']

                print("🎯 Calibrating: Sigma and Correlation parameters")
            elif step == 8:
                #Calibrate all options parameters
                
              
                param_activation.activate(['x0_01', 'sigma_00', 'sigma_11','sigma_01'])
                param_names = ['x0[0,1]', 'sigma[0,0]', 'sigma[1,1]', 'sigma[0,1]']

                print("🎯 Calibrating: Sigma and Correlation parameters[ no Omega Correl")
            
            else:
                raise ValueError(f"Unknown calibration step: {step}")
            
            self.parameter_activation = param_activation.to_list()
            
            # Get initial parameters
            initial_params = self.get_model_parameters()
             
            self.set_parameter_bounds()
            # Create objective
            option_objective = create_objective(
                'option',
                self,
                calibrate_on_vol=calibrate_on_vol,
                use_jax=self.config.use_jax
            )
            
            print(f"initial params:={initial_params}")
            print(f"lower bound:={self.parameter_bounds.lower}")
            print(f"upper bound:={self.parameter_bounds.upper}")

            # Create optimizer
            optimizer = create_optimizer(
                self.config.optimization_method.value,
                option_objective,
                self.parameter_bounds,
                ftol=self.config.relative_tolerance,
                xtol=self.config.absolute_tolerance,
                max_nfev=self.config.max_iterations
                , gtol=self.config.gradient_tolerance
                , verbose= self.config.verbose
            )
            
            # Run optimization
            result = optimizer.optimize(initial_params, param_names)
            
            # Update model with final parameters
            self.set_model_parameters(result.final_parameters)
            
            print(f"\n✅ Step {step} completed with RMSE: {result.final_error:.8f}")
            
            # Store final result
            final_result = result
            
            # Early stopping if convergence is excellent
            if result.final_error < 1e-6:
                print(f"🎉 Excellent convergence achieved (RMSE < 1e-6). Stopping early.")
                break
        
        # Calculate detailed error metrics
        self._calculate_detailed_errors(final_result)
        
        print(f"\n{'='*80}")
        print(f"🏁 CALIBRATION COMPLETED")
        print(final_result.summary())
        
        return final_result
    
    def calibrate(
        self,
        calibrate_ois: bool = True,
        calibrate_options: bool = True,
        **kwargs
    ) -> CalibrationResult:
        """
        Perform full model calibration.
        
        Parameters
        ----------
        calibrate_ois : bool
            Whether to calibrate to OIS curves
        calibrate_options : bool
            Whether to calibrate to options
        **kwargs
            Additional calibration parameters
            
        Returns
        -------
        CalibrationResult
            Final calibration results
        """
        print(f"\n{'='*80}")
        print("STARTING FULL FX MODEL CALIBRATION")
        print(f"{'='*80}")
        
        # Set bounds if not already set
        if self.parameter_bounds is None:
            raise ValueError("Parameter bounds must be set before calibration")
        
        result = None
        
        # Step 1: OIS calibration
        if calibrate_ois:
            ois_rmse = self.calibrate_ois(**kwargs)
        
        # Step 2: Option calibration
        if calibrate_options:
            result = self.calibrate_options(
                calibrate_on_vol=self.config.calibrate_on_vol,
                **kwargs
            )
        
        if result is None:
            # Create a basic result if only OIS was calibrated
            result = CalibrationResult(
                success=True,
                final_parameters=self.get_model_parameters(),
                initial_parameters=self.get_model_parameters(),
                parameter_names=[],
                final_error=ois_rmse if calibrate_ois else 0.0,
                initial_error=0.0,
                calibrated_model=self.model
            )
        
        return result
    
    def _calculate_detailed_errors(self, result: CalibrationResult) -> None:
        """Calculate detailed error metrics for the result."""
        # Reprice everything
        repriced = self.reprice_instruments()
        
        # Calculate bond errors
        bond_price_errors = []
        bond_yield_errors = []
        
        for _, ois_data in self.daily_data.domestic_currency_daily_data.ois_rate_data.iterrows():
            obj = ois_data["Object"]
            bond_price_errors.append(obj.market_zc_price - obj.model_zc_price)
            bond_yield_errors.append(obj.market_zc_rate - obj.model_zc_rate)
        
        for _, ois_data in self.daily_data.foreign_currency_daily_data.ois_rate_data.iterrows():
            obj = ois_data["Object"]
            bond_price_errors.append(obj.market_zc_price - obj.model_zc_price)
            bond_yield_errors.append(obj.market_zc_rate - obj.model_zc_rate)
        
        # Calculate option errors
        option_price_errors = []
        option_vol_errors = []
        
        for option in self.calibration_options:
            option_price_errors.append(option.market_price - option.model_price)
            option_vol_errors.append(option.market_vol - option.model_vol)
        
        # Calculate RMSEs
        if bond_price_errors:
            result.rmse_ois_price = 10000 * np.sqrt(np.mean(np.array(bond_price_errors)**2))
            result.rmse_ois_yield = 10000 * np.sqrt(np.mean(np.array(bond_yield_errors)**2))
        
        if option_price_errors:
            result.rmse_option_price = 10000 * np.sqrt(np.mean(np.array(option_price_errors)**2))
            result.rmse_option_vol = 10000 * np.sqrt(np.mean(np.array(option_vol_errors)**2))
    
    def write_report(
        self,
        result: CalibrationResult,
        ois_file: Optional[str] = None,
        option_file: Optional[str] = None,
        model_file: Optional[str] = None
    ) -> None:
        """
        Write calibration reports to files.
        
        Parameters
        ----------
        result : CalibrationResult
            Calibration results
        ois_file : str, optional
            File path for OIS report
        option_file : str, optional
            File path for option report
        model_file : str, optional
            File path for model report
        """
        # OIS report
        if ois_file:
            ois_report = self.daily_data.ois_summary()
            try:
                with open(ois_file, 'a+') as f:
                    f.write(f"{ois_report}\n")
            except Exception as e:
                print(f"Failed to write OIS report: {e}")
                print(ois_report)
        
        # Option report
        if option_file:
            option_report = self.daily_data.option_summary()
            try:
                with open(option_file, 'a+') as f:
                    f.write(f"{option_report}\n")
            except Exception as e:
                print(f"Failed to write option report: {e}")
                print(option_report)
        
        # Model report
        if model_file:
            model_report = self.model.report() #pretty_print_model()
            error_summary = (
                f"{self.config.min_maturity_option:.4f},"
                f"{self.config.max_maturity_option:.4f},"
                
                f"{result.rmse_ois_price:.4f},"
                f"{result.rmse_ois_yield:.4f},"
                f"{result.rmse_option_price:.4f},"
                f"{result.rmse_option_vol:.4f}"
                f"{self.daily_data.quotation_date}"
            )
            full_report = {
                   'calib_zc_tenor_min': self.config.min_maturity_zc,
                   'calib_zc_tenor_max':self.config.max_maturity_zc,

                   'calib_zc_objective': 'Price', 
                   'calib_option_objective': 'Vol',

                   'calib_option_tenor_min': self.config.min_maturity_option,
                   'calib_option_tenor_max':self.config.max_maturity_option,
                   'rmse_ois_price':    result.rmse_ois_price,
                   'rmse_ois_yield' :   result.rmse_ois_yield,
                   'rmse_option_price': result.rmse_option_price,
                   'rmse_option_vol':   result.rmse_option_vol,
                   'calib_date' :       self.daily_data.quotation_date,
                   'model_parameters':  model_report  
                }

            # full_report = f"{error_summary},{self.daily_data.quotation_date},{model_report}"
            
            try:
                with open(model_file, 'a+') as f:
                    f.write(f"{full_report}\n")
            except Exception as e:
                print(f"Failed to write model report: {e}")
                print(full_report)

    def create_all_options_plots(
                self,
                folder: str,
                file_prefix="",
                maturity_min: float = 1.0,
                maturity_max: float = 5.0
            ) -> None:
        
            self.daily_data.create_all_plots(folder,file_prefix,  maturity_min, maturity_max)

    def create_options_plot(self, folder: str, maturity: float = 1.0) -> None:
         self.daily_data.create_all_plots(folder, maturity)
    

    def create_all_bonds_plots(self, folder, file_prefix="",
                                maturity_min: float = 1.0,
                                maturity_max: float = 5.0) -> None:
        

        self.daily_data.domestic_currency_daily_data.create_ois_bonds_plot( folder, 
                          chart_name=f"{file_prefix}_Domestic",
                          maturity_min= maturity_min,
                          maturity_max= maturity_max)

        self.daily_data.foreign_currency_daily_data.create_ois_bonds_plot( folder,
                          chart_name=f"{file_prefix}_Foreign",
                          maturity_min= maturity_min,
                          maturity_max= maturity_max)

    
    def create_all_plots(self,
                folder: str,
                file_prefix="",
                bonds_maturity_min: float = 1.0,
                bonds_maturity_max: float = 5.0,

                options_maturity_min: float = 1.0,
                options_maturity_max: float = 5.0
                )->None:

                self.create_all_bonds_plots(folder, file_prefix,
                                        maturity_min = bonds_maturity_min,
                                        maturity_max = bonds_maturity_max)

                self.create_all_options_plots(folder, file_prefix,
                                        maturity_min = options_maturity_min,
                                        maturity_max = options_maturity_max)