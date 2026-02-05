"""
Calibration examples for LRW models.

This module demonstrates calibration to market curves and
various parameter estimation techniques.
"""

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# from lrw_numpy.LrwInterestRateBru import LrwInterestRateBru
from linear_rational_wishart.models.interest_rate.config import SwaptionConfig, LRWModelConfig
from linear_rational_wishart.models.interest_rate.lrw_model import LRWModel
from linear_rational_wishart.calibration.lrw_calibration import LRWCalibrator
# from linear_rational_wishart.pricing import LRWSwaptionPricer
from linear_rational_wishart.pricing.swaption_pricer import LRWSwaptionPricer
from linear_rational_wishart.utils.reporting import print_pretty


def example_curve_calibration():
    """Calibrate LRW model to market yield curve."""
    print("Example 1: Yield Curve Calibration")
    print("-" * 40)
    
    # Initial model parameters
    n = 2
    alpha = 0.05
    x0 = jnp.array([[0.03, -0.01], [-0.01, 0.02]])
    omega = jnp.array([[0.08, 0.002], [0.002, 0.001]])
    m = jnp.array([[-0.3, 0.05], [0.05, -0.25]])
    sigma = jnp.array([[0.04, 0.015], [0.015, 0.035]])
    
    # Create model
    lrw_model_config = LRWModelConfig( n=n,  alpha=alpha,  x0=x0,  omega=omega,  m=m, sigma=sigma    )
    swaption_config = SwaptionConfig(maturity=5,tenor=5,  strike=0.05, delta_float = 0.5,delta_fixed= 1.0)
    lrw_model = LRWModel(lrw_model_config,swaption_config)
    
    # lrw_model = LrwInterestRateBru(n, alpha, x0, omega, m, sigma)
    u1 = jnp.array([[1, 0], [0, 0]])
    u2 = jnp.array([[0, 0], [0, 1]])
    lrw_model.set_weight_matrices(u1,u2)
    # lrw_model.SetU1(u1)
    # lrw_model.SetU2(u2)
    
    # Market data (flat curve at 5%)
    flat_rate = 0.05
    market_dates = jnp.array([0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0])
    market_zc_values = jnp.exp(-flat_rate * market_dates)
    
    # Initial model values
    calibrator = LRWCalibrator(lrw_model)
    initial_model_values = calibrator.compute_model_zc_curve(market_dates)
    
    print("\nBefore Calibration:")
    print("Maturity | Market ZC | Model ZC | Difference")
    print("-" * 45)
    for t, mkt, mdl in zip(market_dates, market_zc_values, initial_model_values):
        diff = mdl - mkt
        print(f"{t:8.1f} | {mkt:9.6f} | {mdl:8.6f} | {diff:+10.6f}")
    
    # Calibrate
    calibrator.calibrate_to_curve(
        market_dates,
        market_zc_values,
        use_pseudo_inverse=True,
        interpolation='loglinear'
    )
    
    # Calibrated values
    calibrated_values = calibrator.compute_model_zc_curve(market_dates)
    
    print("\nAfter Calibration:")
    print("Maturity | Market ZC | Model ZC | Difference")
    print("-" * 45)
    for t, mkt, mdl in zip(market_dates, market_zc_values, calibrated_values):
        diff = mdl - mkt
        print(f"{t:8.1f} | {mkt:9.6f} | {mdl:8.6f} | {diff:+10.6f}")
    
    # Plot comparison
    fine_dates = jnp.linspace(0.1, 20, 200)
    
    # Compute curves
    market_curve = jnp.exp(-flat_rate * fine_dates)
    
    # Initial model curve (need fresh model)
    lrw_model_config = LRWModelConfig( n=n,  alpha=alpha,  x0=x0,  omega=omega,  m=m, sigma=sigma    )
    swaption_config = SwaptionConfig(maturity=5,tenor=5,  strike=0.05, delta_float = 0.5,delta_fixed= 1.0)
    initial_model = LRWModel(lrw_model_config,swaption_config)
    initial_model.set_weight_matrices(u1,u2)
    # initial_model = LrwInterestRateBru(n, alpha, x0, omega, m, sigma)
    # initial_model.SetU1(u1)
    # initial_model.SetU2(u2)
    initial_curve = jnp.array([initial_model.bond(t) for t in fine_dates])
    
    calibrated_curve = calibrator.compute_model_zc_curve(fine_dates)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fine_dates, -jnp.log(market_curve) / fine_dates * 100, 
             'k-', linewidth=2, label='Market')
    plt.plot(fine_dates, -jnp.log(initial_curve) / fine_dates * 100, 
             'r--', linewidth=2, label='Initial Model')
    plt.plot(fine_dates, -jnp.log(calibrated_curve) / fine_dates * 100, 
             'b-', linewidth=2, label='Calibrated Model')
    plt.scatter(market_dates, flat_rate * 100 * np.ones_like(market_dates), 
               c='red', s=50, zorder=5, label='Calibration Points')
    plt.xlabel('Maturity (years)')
    plt.ylabel('Yield (%)')
    plt.title('Yield Curve Calibration')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return lrw_model, calibrator


def example_multiple_curve_shapes():
    """Calibrate to different curve shapes."""
    print("\n\nExample 2: Multiple Curve Shapes")
    print("-" * 40)
    
    # Base model
    n = 2
    alpha = 0.05
    x0 = jnp.array([[0.03, -0.01], [-0.01, 0.02]])
    omega = jnp.array([[0.08, 0.002], [0.002, 0.001]])
    m = jnp.array([[-0.3, 0.05], [0.05, -0.25]])
    sigma = jnp.array([[0.04, 0.015], [0.015, 0.035]])
    
    # Different curve shapes
    market_dates = jnp.array([0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
    
    curve_shapes = {
        'Flat': lambda t: 0.05,
        'Upward': lambda t: 0.03 + 0.02 * (1 - jnp.exp(-0.2 * t)),
        'Inverted': lambda t: 0.07 - 0.03 * (1 - jnp.exp(-0.3 * t)),
        'Humped': lambda t: 0.04 + 0.03 * t * jnp.exp(-0.5 * t)
    }
    
    plt.figure(figsize=(12, 8))
    
    for i, (name, rate_func) in enumerate(curve_shapes.items()):
        # Market values
        market_rates = jnp.array([rate_func(t) for t in market_dates])
        market_zc_values = jnp.exp(-market_rates * market_dates)
        
        # Create and calibrate model
        # lrw_model = LrwInterestRateBru(n, alpha, x0, omega, m, sigma)
        # u1 = jnp.array([[1, 0], [0, 0]])
        # u2 = jnp.array([[0, 0], [0, 1]])
        # lrw_model.SetU1(u1)
        # lrw_model.SetU2(u2)
        
        lrw_model_config = LRWModelConfig( n=n,  alpha=alpha,  x0=x0,  omega=omega,  m=m, sigma=sigma    )
        swaption_config = SwaptionConfig(maturity=5,tenor=5,  strike=0.05, delta_float = 0.5,delta_fixed= 1.0)
        lrw_model = LRWModel(lrw_model_config,swaption_config)
        # lrw_model = LrwInterestRateBru(n, alpha, x0, omega, m, sigma)
        u1 = jnp.array([[1, 0], [0, 0]])
        u2 = jnp.array([[0, 0], [0, 1]])
        lrw_model.set_weight_matrices(u1,u2)
  
        calibrator = LRWCalibrator(lrw_model)
        calibrator.calibrate_to_curve(market_dates, market_zc_values)
        
        # Plot
        plt.subplot(2, 2, i + 1)
        
        fine_dates = jnp.linspace(0.1, 20, 200)
        market_fine = jnp.array([rate_func(t) for t in fine_dates])
        model_zc = calibrator.compute_model_zc_curve(fine_dates)
        model_rates = -jnp.log(model_zc) / fine_dates
        
        plt.plot(fine_dates, market_fine * 100, 'k-', linewidth=2, label='Market')
        plt.plot(fine_dates, model_rates * 100, 'b--', linewidth=2, label='Calibrated')
        plt.scatter(market_dates, market_rates * 100, c='red', s=50, zorder=5)
        
        plt.xlabel('Maturity (years)')
        plt.ylabel('Yield (%)')
        plt.title(f'{name} Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 20)
    
    plt.tight_layout()
    plt.show()


def example_parameter_impact():
    """Show impact of different parameters on calibration."""
    print("\n\nExample 3: Parameter Impact on Calibration")
    print("-" * 40)
    
    # Market curve
    flat_rate = 0.05
    market_dates = jnp.array([0.5, 1.0, 2.0, 5.0, 10.0])
    market_zc_values = jnp.exp(-flat_rate * market_dates)
    
    # Base parameters
    n = 2
    alpha_base = 0.05
    x0_base = jnp.array([[0.03, -0.01], [-0.01, 0.02]])
    
    # Test different omega values
    beta_values = [2.0, 3.0, 4.0, 5.0]
    sigma = jnp.array([[0.04, 0.015], [0.015, 0.035]])
    m = jnp.array([[-0.3, 0.05], [0.05, -0.25]])
    
    plt.figure(figsize=(12, 5))
    
    # Omega impact
    plt.subplot(1, 2, 1)
    for beta in beta_values:
        omega = beta * sigma @ sigma
        
        # Check Gindikin condition
        temp = omega - 3.0 * sigma @ sigma
        if jnp.linalg.det(temp) < 0:
            print(f"β={beta}: Gindikin condition violated")
            continue
            
        lrw_model_config = LRWModelConfig( n=n,  alpha=alpha_base,  x0=x0_base,  omega=omega,  m=m, sigma=sigma    )
        swaption_config = SwaptionConfig(maturity=5,tenor=5,  strike=0.05, delta_float = 0.5,delta_fixed= 1.0)
        lrw_model = LRWModel(lrw_model_config,swaption_config)
        u1 = jnp.array([[1, 0], [0, 0]])
        u2 = jnp.array([[0, 0], [0, 1]])
        lrw_model.set_weight_matrices(u1,u2)
        # lrw_model = LrwInterestRateBru(n, alpha_base, x0_base, omega, m, sigma)
        # u1 = jnp.array([[1, 0], [0, 0]])
        # u2 = jnp.array([[0, 0], [0, 1]])
        # lrw_model.SetU1(u1)
        # lrw_model.SetU2(u2)
        
        calibrator = LRWCalibrator(lrw_model)
        
        # Validate Gindikin
        if calibrator.validate_gindikin_condition():
            calibrator.calibrate_to_curve(market_dates, market_zc_values)
            
            fine_dates = jnp.linspace(0.1, 15, 100)
            model_zc = calibrator.compute_model_zc_curve(fine_dates)
            model_rates = -jnp.log(model_zc) / fine_dates
            
            plt.plot(fine_dates, model_rates * 100, label=f'β={beta}')
        else:
            print(f"β={beta}: Model invalid after calibration")
    
    plt.axhline(y=flat_rate * 100, color='k', linestyle='--', label='Market')
    plt.xlabel('Maturity (years)')
    plt.ylabel('Yield (%)')
    plt.title('Impact of ω = β·σ·σᵀ')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Mean reversion impact
    plt.subplot(1, 2, 2)
    mean_reversion_scales = [0.5, 1.0, 1.5, 2.0]
    omega = 4.0 * sigma @ sigma
    
    for scale in mean_reversion_scales:
        m_scaled = m * scale
        
        # lrw_model = LrwInterestRateBru(n, alpha_base, x0_base, omega, m_scaled, sigma)
        # u1 = jnp.array([[1, 0], [0, 0]])
        # u2 = jnp.array([[0, 0], [0, 1]])
        # lrw_model.SetU1(u1)
        # lrw_model.SetU2(u2)

        lrw_model_config = LRWModelConfig( n=n,  alpha=alpha_base,  x0=x0_base,  omega=omega,  m=m_scaled, sigma=sigma    )
        swaption_config = SwaptionConfig(maturity=5,tenor=5,  strike=0.05, delta_float = 0.5,delta_fixed= 1.0)
        lrw_model = LRWModel(lrw_model_config,swaption_config)
        u1 = jnp.array([[1, 0], [0, 0]])
        u2 = jnp.array([[0, 0], [0, 1]])
        lrw_model.set_weight_matrices(u1,u2)

        calibrator = LRWCalibrator(lrw_model)
        calibrator.calibrate_to_curve(market_dates, market_zc_values)
        
        fine_dates = jnp.linspace(0.1, 15, 100)
        model_zc = calibrator.compute_model_zc_curve(fine_dates)
        model_rates = -jnp.log(model_zc) / fine_dates
        
        plt.plot(fine_dates, model_rates * 100, label=f'M scale={scale}')
    
    plt.axhline(y=flat_rate * 100, color='k', linestyle='--', label='Market')
    plt.xlabel('Maturity (years)')
    plt.ylabel('Yield (%)')
    plt.title('Impact of Mean Reversion M')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def example_swaption_calibration():
    """Example of calibrating to ATM swaption prices."""
    print("\n\nExample 4: Swaption Calibration Setup")
    print("-" * 40)
    
    # Base model with calibrated curve
    n = 2
    alpha = 0.052
    x0 = jnp.array([[0.04, 0.02], [0.02, 0.03]])
    m = jnp.array([[-0.15, 0.06], [0.07, -0.12]])
    sigma = jnp.array([[0.04, 0.015], [0.015, 0.037]])
    omega = 4.0 * sigma @ sigma
    
    lrw_model_config = LRWModelConfig( n, alpha, x0, omega, m, sigma)
    swaption_config = SwaptionConfig(maturity=5,tenor=5,  strike=0.05, delta_float = 0.5,delta_fixed= 1.0)
    lrw_model = LRWModel(lrw_model_config,swaption_config)
    u1 = jnp.array([[1, 0], [0, 0]])
    u2 = jnp.array([[0, 0], [0, 1]])
    lrw_model.set_weight_matrices(u1,u2)

    # lrw_model = LrwInterestRateBru(n, alpha, x0, omega, m, sigma)
    
    # u1 = jnp.array([[1, 0], [0, 1]])
    # u2 = jnp.array([[0, 0], [0, 0]])
    # lrw_model.SetU1(u1)
    # lrw_model.SetU2(u2)
    
    # Define swaption grid
    maturities = [1, 2, 5]
    tenors = [1, 2, 5, 10]
    
    print("\nATM Swaption Prices and Implied Volatilities:")
    print("Maturity | Tenor | ATM Strike | Price   | Implied Vol")
    print("-" * 55)
    
    swaption_data = []
    pricer = LRWSwaptionPricer(lrw_model)
    
    for mat in maturities:
        for ten in tenors:
            # Set swaption
            # lrw_model.SetOptionProperties(ten, mat, 0.5, 0.5, 0.0)
            swaption_config = SwaptionConfig(maturity=mat,tenor=ten,  strike=0.05, delta_float = 0.5,delta_fixed= 0.5)
            lrw_model.set_swaption_config(swaption_config)
            atm = lrw_model.compute_swap_rate()
            # lrw_model.SetOptionProperties(ten, mat, 0.5, 0.5, atm)
            swaption_config = SwaptionConfig(maturity=mat,tenor=ten,  strike=atm, delta_float = 0.5,delta_fixed= 0.5)
            
            # Price
            price, iv = pricer.price_swaption(method="fft", return_implied_vol=True)
            
            print(f"{mat:8}Y | {ten:5}Y | {atm:10.2%} | {price:7.5f} | {iv:11.2%}")
            
            swaption_data.append({
                'maturity': mat,
                'tenor': ten,
                'strike': atm,
                'price': price,
                'implied_vol': iv
            })
    
    # Visualize implied volatility surface
    plt.figure(figsize=(10, 6))
    
    # Create grid for plotting
    for mat in maturities:
        mat_data = [d for d in swaption_data if d['maturity'] == mat]
        tenors_plot = [d['tenor'] for d in mat_data]
        ivs_plot = [d['implied_vol'] * 100 for d in mat_data]
        plt.plot(tenors_plot, ivs_plot, 'o-', label=f'{mat}Y maturity', linewidth=2)
    
    plt.xlabel('Tenor (years)')
    plt.ylabel('ATM Implied Volatility (%)')
    plt.title('ATM Swaption Implied Volatility by Maturity and Tenor')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return lrw_model, swaption_data


def example_gindikin_adjustment():
    """Example of adjusting parameters to satisfy Gindikin condition."""
    print("\n\nExample 5: Gindikin Condition Adjustment")
    print("-" * 40)
    
    # Parameters that might violate Gindikin
    n = 2
    alpha = 0.05
    x0 = jnp.array([[0.02, -0.01], [-0.01, 0.015]])
    m = jnp.array([[-0.4, 0.1], [0.1, -0.5]])
    sigma = jnp.array([[0.08, 0.04], [0.04, 0.07]])
    
    # Try different beta values
    print("Testing different β values for ω = β·σ·σᵀ:")
    print("β    | det(ω - 3σσᵀ) | Gindikin OK")
    print("-" * 35)
    
    for beta in jnp.arange(2.0, 6.0, 0.5):
        omega = beta * sigma @ sigma
        temp = omega - 3.0 * sigma @ sigma
        det_temp = jnp.linalg.det(temp)
        gindikin_ok = det_temp >= 0
        
        print(f"{beta:4.1f} | {det_temp:13.6f} | {gindikin_ok}")
        
        if gindikin_ok:
            # Create model with valid parameters
            lrw_model_config = LRWModelConfig( n, alpha, x0, omega, m, sigma)
            swaption_config = SwaptionConfig(maturity=5,tenor=5,  strike=0.05, delta_float = 0.5,delta_fixed= 1.0)
            lrw_model = LRWModel(lrw_model_config,swaption_config)
            u1 = jnp.array([[1, 0], [0, 0]])
            u2 = jnp.array([[0, 0], [0, 1]])
            lrw_model.set_weight_matrices(u1,u2)
            
            # lrw_model = LrwInterestRateBru(n, alpha, x0, omega, m, sigma)
            # u1 = jnp.array([[1, 0], [0, 0]])
            # u2 = jnp.array([[0, 0], [0, 1]])
            # lrw_model.SetU1(u1)
            # lrw_model.SetU2(u2)
            
            # Use calibrator to verify
            calibrator = LRWCalibrator(lrw_model)
            
            if calibrator.validate_gindikin_condition():
                print(f"\nUsing β = {beta} for valid model")
                
                # Test automatic adjustment
                calibrator.adjust_omega_for_gindikin(beta=2.0)  # Try with low beta
                
                # Verify it's still valid
                if calibrator.validate_gindikin_condition():
                    print("Model remains valid after adjustment")
                else:
                    print("Model became invalid - need higher beta")
                    
                break


if __name__ == "__main__":
    # Run all examples
    example_curve_calibration()
    example_multiple_curve_shapes()
    example_parameter_impact()
    example_swaption_calibration()
    example_gindikin_adjustment()
