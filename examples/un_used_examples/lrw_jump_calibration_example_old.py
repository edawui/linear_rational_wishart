"""
Examples of LRW Jump model calibration.

This module demonstrates various calibration scenarios for the
Linear Rational Wishart interest rate model with jumps.
"""

import numpy as np
import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Tuple, Optional, Dict, Any
import os
import shutil
import jax
import gc

from linear_rational_wishart.models.interest_rate.lrw_model import LRWModel
from linear_rational_wishart.calibration.lrw_jump_calibrator import LRWJumpCalibrator, CalibrationConfig
# from linear_rational_wishart.data import MarketData data_market_data.py
from linear_rational_wishart.data.data_market_data import DailyData
from linear_rational_wishart.utils.reporting import print_pretty
from linear_rational_wishart.models.interest_rate.config import SwaptionConfig, LRWModelConfig
# from linear_rational_wishart.models.interest_rate.lrw_model import LRWModel
from linear_rational_wishart.data.data_helpers  import *#get_testing_excel_data #* 


def example_basic_calibration():
    """Basic calibration example for LRW Jump model."""
    print("Example 1: Basic LRW Jump Model Calibration")
    print("-" * 50)
    
    # Create sample market data
    daily_data = create_sample_market_data()
    
    # Initialize model with starting parameters
    n = 2
    alpha = 0.05
    x0 = jnp.array([[0.02, 0.005], [0.005, 0.015]])
    omega = jnp.array([[0.06, 0.001], [0.001, 0.07]])
    m = jnp.array([[-0.29, 0.1], [0.2, -0.5]])
    sigma = jnp.array([[0.03, 0.1], [0.1, 0.1]])
    

    x0      = jnp.array( [[ 0.4, 0.0 ],[ 0.0, 0.22]])
    omega   = jnp.array( [[ 0.5, 0.00 ],[0.00, 0.5]])
    m       = jnp.array( [[-1.4, 0],[ 0, -1.2 ]]);
    sigma   = jnp.array([[ 0.4, 0.02 ],[ 0.02, 0.2]])
    sigma   = jnp.array([[ 0.04, 0.0 ],[ 0.0, 0.02]])
    # sigma   = jnp.array([[0.003, 0.0], [0.0, 0.001]])
  
    alpha = 0.02461063116788864
    x0 = jnp.array( [[4.29816438e-01, 0],[0.0, 1.00000000e-04]])
    omega = jnp.array([[  5.4921864e-1, 0],[0, 1.1299789e-1]])
    m = jnp.array([[-6.1320242e-1,  0], [0, -4.4444444e-1]])


    x0 = jnp.array( [[4.29816438e-01, 2.00000000e-02],[2.00000000e-02, 1.00000000e-02]])
    omega = jnp.array([[  5.4921864e-1, 3.5795735e-3],[3.5795735e-3, 1.1299789e-1]])
    m = jnp.array([[-6.1320242e-1, 0.0], [ 0.0, -4.4444444e-1]])


    alpha = 0.023681147024035454
    x0 = jnp.array([[0.4291363,  0.02      ],   [0.02 ,      0.00264791]])
    omega = jnp.array([[0.54993196, 0.00357957],    [0.00357957 ,0.06779055]])
    m = jnp.array([[-0.60828161,  0.        ],    [ 0. ,        -7.56179852]])
    sigma =jnp.array( [[ 0.00501339, -0.00026962],    [-0.00026962,  0.00721957]])
    
    sigma =jnp.array( [[ 0.0501339, -0.00026962],    [-0.00026962,  0.0721957]])
    sigma =jnp.array( [[ 0.501339, -0.026962],    [-0.026962,  0.721957]])


    ### test case 3
    x0 = jnp.array([[0.4291363,  0.0      ],   [0.0 ,      0.00264791]])
    omega = jnp.array([[0.54993196, 0.0],    [0.00 ,0.06779055]])
    m = jnp.array([[-0.60828161,  0.        ],    [ 0. ,        -7.56179852]])
    sigma =jnp.array( [[ 0.501339, -0.0],    [-0.0,  0.0721957]])
    sigma =jnp.array( [[ 1.501339, -0.0],    [-0.0,  0.721957]])

    # # ###Resultat Paper
    # # x0 = jnp.array( [[0.015, -1.055e-3],[-1.055e-3, 8.25e-4]])
    # # omega = jnp.array([[0.110, -2.974e-3],[-2.974e-3, 1.377e-3]])
    # # m = jnp.array([[-6.642,  0.0], [ 0.0, -0.028]])
    # # sigma = jnp.array([[ 0.165, -0.041],[-0.041, 0.069]])

    # x0 = jnp.array( [[0.015, -1.055e-3],[-1.055e-3, 8.25e-4]])
    # omega = jnp.array([[0.110, -2.974e-3],[-2.974e-3, 1.377e-3]])
    # m = jnp.array([[-6.642,  0.0], [ 0.0, -0.028]])
    # # sigma = jnp.array([[ 0.165, -0.041],[-0.041, 0.069]])
    
    # Configure calibration
    config = CalibrationConfig(
        max_tenor=15,#10.0,
        min_tenor=0.5,
        use_multi_thread=False,#True,#False,
        calibrate_on_swaption=True
    )
    lrw_model_config = LRWModelConfig( n=n,  alpha=alpha,  x0=x0,  omega=omega,  m=m, sigma=sigma ,is_bru_config=False, has_jump=False )
    swaption_config = SwaptionConfig(maturity=1,tenor=1,
                                   strike=0.05, 
                                   delta_float = 0.5,
                                   delta_fixed= 0.5)
    lrw_model = LRWModel(lrw_model_config,swaption_config)
    u1 = jnp.array([[1, 0], [0, 0]])
    u2 = jnp.array([[0, 0], [0, 1]])   
    lrw_model.set_weight_matrices(u1,u2)
    lrw_model.print_model()

    # Create calibrator
    calibrator = LRWJumpCalibrator(lrw_model, daily_data, config)
    
    # Perform calibration
    results = calibrator.calibrate_full()
    
    # Print results
    print("\nCalibration Results:")
    print_pretty(results)
    
    # Generate report
    report_folder=r"E:\OneDrive\Dropbox\LinearRationalWishart_Work\Code\ED\LinearRationalWishart\LinearRationalWishart_NewCode\Output_results"

    calibrator.generate_report(report_folder) #"calibration_results")
    
    return calibrator, results


def example_basic_calibration_old():
    """Basic calibration example for LRW Jump model."""
    print("Example 1: Basic LRW Jump Model Calibration")
    print("-" * 50)
    
    # Create sample market data
    daily_data = create_sample_market_data()
    
    # Initialize model with starting parameters
    n = 2
    alpha = 0.05
    x0 = jnp.array([[0.02, 0.005], [0.005, 0.015]])
    omega = jnp.array([[0.06, 0.001], [0.001, 0.07]])
    m = jnp.array([[-0.29, 0.1], [0.2, -0.5]])
    sigma = jnp.array([[0.03, 0.1], [0.1, 0.1]])
    

    x0      = jnp.array( [[ 0.4, 0.0 ],[ 0.0, 0.22]])
    omega   = jnp.array( [[ 0.5, 0.00 ],[0.00, 0.5]])
    m       = jnp.array( [[-1.4, 0],[ 0, -1.2 ]]);
    sigma   = jnp.array([[ 0.4, 0.02 ],[ 0.02, 0.2]])
    sigma   = jnp.array([[ 0.04, 0.0 ],[ 0.0, 0.02]])
    # sigma   = jnp.array([[0.003, 0.0], [0.0, 0.001]])
  
    alpha = 0.02461063116788864
    x0 = jnp.array( [[4.29816438e-01, 0],[0.0, 1.00000000e-04]])
    omega = jnp.array([[  5.4921864e-1, 0],[0, 1.1299789e-1]])
    m = jnp.array([[-6.1320242e-1,  0], [0, -4.4444444e-1]])


    x0 = jnp.array( [[4.29816438e-01, 2.00000000e-02],[2.00000000e-02, 1.00000000e-02]])
    omega = jnp.array([[  5.4921864e-1, 3.5795735e-3],[3.5795735e-3, 1.1299789e-1]])
    m = jnp.array([[-6.1320242e-1, 0.0], [ 0.0, -4.4444444e-1]])


    alpha = 0.023681147024035454
    x0 = jnp.array([[0.4291363,  0.02      ],   [0.02 ,      0.00264791]])
    omega = jnp.array([[0.54993196, 0.00357957],    [0.00357957 ,0.06779055]])
    m = jnp.array([[-0.60828161,  0.        ],    [ 0. ,        -7.56179852]])
    sigma =jnp.array( [[ 0.00501339, -0.00026962],    [-0.00026962,  0.00721957]])
    
    sigma =jnp.array( [[ 0.0501339, -0.00026962],    [-0.00026962,  0.0721957]])
    sigma =jnp.array( [[ 0.501339, -0.026962],    [-0.026962,  0.721957]])


    ### test case 3
    x0 = jnp.array([[0.4291363,  0.0      ],   [0.0 ,      0.00264791]])
    omega = jnp.array([[0.54993196, 0.0],    [0.00 ,0.06779055]])
    m = jnp.array([[-0.60828161,  0.        ],    [ 0. ,        -7.56179852]])
    sigma =jnp.array( [[ 0.501339, -0.0],    [-0.0,  0.0721957]])
    sigma =jnp.array( [[ 1.501339, -0.0],    [-0.0,  0.721957]])

    # # ###Resultat Paper
    # # x0 = jnp.array( [[0.015, -1.055e-3],[-1.055e-3, 8.25e-4]])
    # # omega = jnp.array([[0.110, -2.974e-3],[-2.974e-3, 1.377e-3]])
    # # m = jnp.array([[-6.642,  0.0], [ 0.0, -0.028]])
    # # sigma = jnp.array([[ 0.165, -0.041],[-0.041, 0.069]])

    # x0 = jnp.array( [[0.015, -1.055e-3],[-1.055e-3, 8.25e-4]])
    # omega = jnp.array([[0.110, -2.974e-3],[-2.974e-3, 1.377e-3]])
    # m = jnp.array([[-6.642,  0.0], [ 0.0, -0.028]])
    # # sigma = jnp.array([[ 0.165, -0.041],[-0.041, 0.069]])
    
    # Configure calibration
    config = CalibrationConfig(
        max_tenor=15,#10.0,
        min_tenor=0.5,
        use_multi_thread=False,#True,#False,
        calibrate_on_swaption=True
    )
    lrw_model_config = LRWModelConfig( n=n,  alpha=alpha,  x0=x0,  omega=omega,  m=m, sigma=sigma ,is_bru_config=False, has_jump=False )
    swaption_config = SwaptionConfig(maturity=1,tenor=1,
                                   strike=0.05, 
                                   delta_float = 0.5,
                                   delta_fixed= 0.5)
    lrw_model = LRWModel(lrw_model_config,swaption_config)
    u1 = jnp.array([[1, 0], [0, 0]])
    u2 = jnp.array([[0, 0], [0, 1]])   
    lrw_model.set_weight_matrices(u1,u2)
    lrw_model.print_model()

    # Create calibrator
    calibrator = LRWJumpCalibrator(lrw_model, daily_data, config)
    
    # Perform calibration
    results = calibrator.calibrate_full()
    
    # Print results
    print("\nCalibration Results:")
    print_pretty(results)
    
    # Generate report
    report_folder=r"E:\OneDrive\Dropbox\LinearRationalWishart_Work\Code\ED\LinearRationalWishart\LinearRationalWishart_NewCode\Output_results"

    calibrator.generate_report(report_folder) #"calibration_results")
    
    return calibrator, results



def example_step_by_step_calibration():
    """Step-by-step calibration example."""
    print("\n\nExample 2: Step-by-Step Calibration")
    print("-" * 50)
    
    # Create market data and model
    daily_data = create_sample_market_data()
    model = create_sample_model()
    
    # Configure calibration
    config = CalibrationConfig(
        max_tenor=10.0,
        min_tenor=0.5,
        calibrate_on_vol=True
    )
    
    # Create calibrator
    calibrator = LRWJumpCalibrator(model, daily_data, config)
    
    # Step 1: Calibrate to OIS curve
    print("\nStep 1: Calibrating to OIS curve...")
    ois_error = calibrator.calibrate_ois_curve(on_price=True)
    print(f"OIS RMSE: {ois_error:.6f}")
    
    # Print OIS parameters
    ois_params = calibrator.get_model_parameters(calibrator._get_ois_param_activation())
    print(f"OIS parameters: {ois_params}")
    
    # Step 2: Calibrate to spreads
    print("\nStep 2: Calibrating to spreads...")
    spread_error = calibrator.calibrate_spreads(on_full_a=False)
    print(f"Spread RMSE: {spread_error:.6f}")
    
    # Print spread parameters
    spread_params = calibrator.get_model_parameters(calibrator._get_spread_param_activation())
    print(f"Spread parameters: {spread_params}")
    
    # Step 3: Calibrate to swaptions
    print("\nStep 3: Calibrating to swaptions...")
    swaption_error = calibrator.calibrate_swaptions()
    print(f"Swaption RMSE: {swaption_error:.6f}")
    
    # Print volatility parameters
    vol_params = calibrator.get_model_parameters(calibrator._get_vol_param_activation())
    print(f"Volatility parameters: {vol_params}")
    
    return calibrator


def example_parameter_sensitivity():
    """Analyze parameter sensitivity in calibration."""
    print("\n\nExample 3: Parameter Sensitivity Analysis")
    print("-" * 50)
    
    # Create base model and data
    daily_data = create_sample_market_data()
    base_model = create_sample_model()
    
    # Test different starting points
    parameter_sets = {
        "Low Vol": {
            "sigma": jnp.array([[0.02, 0.05], [0.05, 0.08]])
        },
        "High Vol": {
            "sigma": jnp.array([[0.05, 0.15], [0.15, 0.15]])
        },
        "High Correlation": {
            "sigma": jnp.array([[0.03, 0.09], [0.09, 0.1]])
        }
    }
    
    results = {}
    
    for name, params in parameter_sets.items():
        print(f"\nCalibrating with {name} starting point...")
        
        # Create model with modified parameters
        model = create_sample_model()
        model.sigma = params["sigma"]
        
        # Calibrate
        config = CalibrationConfig(calibrate_on_vol=True)
        calibrator = LRWJumpCalibrator(model, daily_data, config)
        
        # Only calibrate swaptions to test vol parameters
        swaption_error = calibrator.calibrate_swaptions()
        
        # Store results
        final_params = calibrator.get_model_parameters(calibrator._get_vol_param_activation())
        results[name] = {
            "starting_sigma": params["sigma"],
            "final_params": final_params,
            "error": swaption_error
        }
        
    # Display comparison
    print("\nParameter Sensitivity Results:")
    for name, res in results.items():
        print(f"\n{name}:")
        print(f"  Starting σ₁₁: {res['starting_sigma'][0,0]:.4f}")
        print(f"  Final params: {res['final_params']}")
        print(f"  Error: {res['error']:.6f}")
        
    return results


def example_calibration_with_constraints():
    """Calibration with custom constraints."""
    print("\n\nExample 4: Calibration with Constraints")
    print("-" * 50)
    
    # Create models for bounds
    daily_data = create_sample_market_data()
    
    # Base model
    n = 2
    alpha = 0.05
    x0 = jnp.array([[0.02, 0.005], [0.005, 0.015]])
    omega = jnp.array([[0.06, 0.001], [0.001, 0.07]])
    m = jnp.array([[-0.29, 0.1], [0.2, -0.5]])
    sigma = jnp.array([[0.03, 0.1], [0.1, 0.1]])
    
    model = LRWModel(n, alpha, x0, omega, m, sigma)
    
    # Create bounds
    lower_model = LRWModel(
        n,
        alpha=0.01,  # Lower bound on alpha
        x0=x0 * 0.5,  # 50% of base
        omega=omega * 0.5,
        m=m * 2.0,  # More negative
        sigma=sigma * 0.5
    )
    
    upper_model = LRWModel(
        n,
        alpha=0.10,  # Upper bound on alpha
        x0=x0 * 2.0,  # 200% of base
        omega=omega * 2.0,
        m=m * 0.5,  # Less negative
        sigma=sigma * 2.0
    )
    
    # Configure calibration with constraints
    config = CalibrationConfig()
    calibrator = LRWJumpCalibrator(model, daily_data, config)
    
    # Set custom bounds
    calibrator.constraints.set_custom_bounds(lower_model, upper_model)
    
    # Calibrate
    print("Calibrating with custom bounds...")
    results = calibrator.calibrate_full()
    
    # Validate results
    calibrator.validate_calibration(base_model=model)
    
    print("\nConstrained Calibration Results:")
    print_pretty(results)
    
    return calibrator


def example_multi_currency_calibration():
    """Calibration for multiple currencies."""
    print("\n\nExample 5: Multi-Currency Calibration")
    print("-" * 50)
    
    currencies = ["EUR", "USD", "GBP"]
    calibrators = {}
    
    for currency in currencies:
        print(f"\nCalibrating {currency} model...")
        
        # Create currency-specific data
        daily_data = create_sample_market_data(currency=currency)
        
        # Create model with currency-specific starting values
        model = create_currency_specific_model(currency)
        
        # Configure calibration
        config = CalibrationConfig(
            use_multi_thread=True,  # Use parallel pricing
            calibrate_on_vol=True
        )
        
        # Calibrate
        calibrator = LRWJumpCalibrator(model, daily_data, config)
        results = calibrator.calibrate_full()
        
        calibrators[currency] = {
            "calibrator": calibrator,
            "results": results
        }
        
        print(f"{currency} calibration completed. Swaption error: {results.get('swaption_error', 'N/A'):.6f}")
    
    # Compare results
    print("\n\nCross-Currency Comparison:")
    print("Currency | OIS Error | Spread Error | Swaption Error")
    print("-" * 55)
    
    for currency, data in calibrators.items():
        results = data["results"]
        print(f"{currency:8} | {results['ois_error']:9.6f} | {results['spread_error']:12.6f} | {results.get('swaption_error', 0):14.6f}")
    
    return calibrators


def example_calibration_quality_analysis():
    """Analyze calibration quality and fit."""
    print("\n\nExample 6: Calibration Quality Analysis")
    print("-" * 50)
    
    # Perform calibration
    daily_data = create_sample_market_data()
    model = create_sample_model()
    
    config = CalibrationConfig()
    calibrator = LRWJumpCalibrator(model, daily_data, config)
    results = calibrator.calibrate_full()
    
    # Analyze fit quality
    print("\nAnalyzing calibration quality...")
    
    # Plot OIS curve fit
    plt.figure(figsize=(15, 10))
    
    # OIS fit
    plt.subplot(2, 2, 1)
    plot_ois_fit(calibrator)
    plt.title("OIS Curve Fit")
    plt.xlabel("Maturity (years)")
    plt.ylabel("Rate (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Spread fit
    plt.subplot(2, 2, 2)
    plot_spread_fit(calibrator)
    plt.title("Spread Fit")
    plt.xlabel("Maturity (years)")
    plt.ylabel("Spread (bps)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Swaption vol surface
    plt.subplot(2, 2, 3)
    plot_swaption_vol_surface(calibrator)
    plt.title("Swaption Volatility Surface")
    plt.xlabel("Maturity (years)")
    plt.ylabel("Tenor (years)")
    
    # Error distribution
    plt.subplot(2, 2, 4)
    plot_error_distribution(calibrator)
    plt.title("Calibration Error Distribution")
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    
    plt.tight_layout()
    plt.show()
    
    # Print quality metrics
    print("\nCalibration Quality Metrics:")
    metrics = compute_quality_metrics(calibrator)
    print_pretty(metrics)
    
    return calibrator, metrics


# Helper functions

def create_sample_market_data(currency: str = "EUR") -> DailyData:
    """Create sample market data for testing."""
    # This would create realistic market data
    # Implementation depends on MarketData structure
    # pass
    daily_data= get_ir_testing_excel_data(
        current_date = '20250530'#'20250401' #'20250530'
        , ccy=currency
        , filter_tenors=True)
    return daily_data


def create_sample_model() -> LRWModel:
    """Create sample LRW Jump model."""
    n = 2
    alpha = 0.05
    x0 = jnp.array([[0.02, 0.005], [0.005, 0.015]])
    omega = jnp.array([[0.06, 0.001], [0.001, 0.07]])
    m = jnp.array([[-0.29, 0.1], [0.2, -0.5]])
    sigma = jnp.array([[0.03, 0.1], [0.1, 0.1]])
    
    return LRWModel(n, alpha, x0, omega, m, sigma)


def create_currency_specific_model(currency: str) -> LRWModel:
    """Create model with currency-specific parameters."""
    # Currency-specific starting values
    params = {
        "EUR": {
            "alpha": 0.05,
            "x0_scale": 1.0,
            "vol_scale": 1.0
        },
        "USD": {
            "alpha": 0.06,
            "x0_scale": 1.2,
            "vol_scale": 1.1
        },
        "GBP": {
            "alpha": 0.055,
            "x0_scale": 1.1,
            "vol_scale": 1.05
        }
    }
    
    curr_params = params.get(currency, params["EUR"])
    
    n = 2
    alpha = curr_params["alpha"]
    x0 = jnp.array([[0.02, 0.005], [0.005, 0.015]]) * curr_params["x0_scale"]
    omega = jnp.array([[0.06, 0.001], [0.001, 0.07]])
    m = jnp.array([[-0.29, 0.1], [0.2, -0.5]])
    sigma = jnp.array([[0.03, 0.1], [0.1, 0.1]]) * curr_params["vol_scale"]
    
    return LRWModel(n, alpha, x0, omega, m, sigma)


def plot_ois_fit(calibrator: LRWJumpCalibrator):
    """Plot OIS curve fit."""
    # Implementation for plotting OIS fit
    pass


def plot_spread_fit(calibrator: LRWJumpCalibrator):
    """Plot spread fit."""
    # Implementation for plotting spread fit
    pass


def plot_swaption_vol_surface(calibrator: LRWJumpCalibrator):
    """Plot swaption volatility surface."""
    # Implementation for plotting vol surface
    pass


def plot_error_distribution(calibrator: LRWJumpCalibrator):
    """Plot calibration error distribution."""
    # Implementation for plotting errors
    pass


def compute_quality_metrics(calibrator: LRWJumpCalibrator) -> Dict[str, float]:
    """Compute calibration quality metrics."""
    metrics = {
        "ois_max_error": 0.0,  # To be computed
        "spread_max_error": 0.0,
        "swaption_vol_mean_error": 0.0,
        "gindikin_satisfied": calibrator.constraints.check_gindikin_condition()
    }
    return metrics

def clear_jax_cache():
    jax.clear_caches()
    gc.collect()
    
    jax_cache = os.path.join(os.environ.get('LOCALAPPDATA', ''), 'jax')
    if os.path.exists(jax_cache):
        shutil.rmtree(jax_cache, ignore_errors=True)

if __name__ == "__main__":
    # Run all examples
    print("LRW Jump Model Calibration Examples")
    print("=" * 60)
    
    # from jax.config import config
    # config.update("jax_disable_jit", True)

    import jax
    jax.config.update("jax_disable_jit", True)

    # Basic calibration
    calibrator1, results1 = example_basic_calibration()
    
    # # Step-by-step calibration
    # calibrator2 = example_step_by_step_calibration()
    
    # # Parameter sensitivity
    # sensitivity_results = example_parameter_sensitivity()
    
    # # Constrained calibration
    # calibrator4 = example_calibration_with_constraints()
    
    # # Multi-currency calibration
    # multi_currency_results = example_multi_currency_calibration()
    
    # # Quality analysis
    # calibrator6, quality_metrics = example_calibration_quality_analysis()
    
    clear_jax_cache()

    print("\n\nAll examples completed successfully!")
