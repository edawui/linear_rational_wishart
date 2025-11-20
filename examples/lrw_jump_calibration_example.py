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

from wishart_processes.models.interest_rate import LRWModel
from wishart_processes.calibration import LRWJumpCalibrator, CalibrationConfig
from wishart_processes.data import MarketData
from wishart_processes.utils.reporting import print_pretty


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
    
    model = LRWModel(n, alpha, x0, omega, m, sigma)
    
    # Configure calibration
    config = CalibrationConfig(
        max_tenor=10.0,
        min_tenor=0.5,
        use_multi_thread=False,
        calibrate_on_vol=True
    )
    
    # Create calibrator
    calibrator = LRWJumpCalibrator(model, daily_data, config)
    
    # Perform calibration
    results = calibrator.calibrate_full()
    
    # Print results
    print("\nCalibration Results:")
    print_pretty(results)
    
    # Generate report
    calibrator.generate_report("calibration_results")
    
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

def create_sample_market_data(currency: str = "EUR") -> MarketData.DailyData:
    """Create sample market data for testing."""
    # This would create realistic market data
    # Implementation depends on MarketData structure
    pass


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


if __name__ == "__main__":
    # Run all examples
    print("LRW Jump Model Calibration Examples")
    print("=" * 60)
    
    # Basic calibration
    calibrator1, results1 = example_basic_calibration()
    
    # Step-by-step calibration
    calibrator2 = example_step_by_step_calibration()
    
    # Parameter sensitivity
    sensitivity_results = example_parameter_sensitivity()
    
    # Constrained calibration
    calibrator4 = example_calibration_with_constraints()
    
    # Multi-currency calibration
    multi_currency_results = example_multi_currency_calibration()
    
    # Quality analysis
    calibrator6, quality_metrics = example_calibration_quality_analysis()
    
    print("\n\nAll examples completed successfully!")