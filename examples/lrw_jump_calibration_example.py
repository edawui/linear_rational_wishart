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
from linear_rational_wishart.data.data_helpers  import get_ir_testing_excel_data #*#get_testing_excel_data #* 


# =============================================================================
# GINDIKIN HELPER FUNCTIONS
# =============================================================================

def check_gindikin(omega, sigma, n=2):
    """
    Check Gindikin condition: omega - (n+1)*sigma² > 0
    
    Returns
    -------
    tuple: (is_valid, min_eigenvalue, max_feasible_sigma_diagonal)
    """
    beta = n + 1
    omega_np = np.array(omega)
    sigma_np = np.array(sigma)
    
    sigma2 = sigma_np @ sigma_np
    gindikin_matrix = omega_np - beta * sigma2
    eigenvalues = np.linalg.eigvalsh(gindikin_matrix)
    min_eig = np.min(eigenvalues)
    is_valid = min_eig >= 0
    
    # Max feasible sigma (diagonal)
    max_sigma = np.array([
        np.sqrt(omega_np[i, i] / beta) * 0.95 
        for i in range(n)
    ])
    
    return is_valid, min_eig, max_sigma


def adjust_omega_for_sigma(omega, sigma, n=2, margin=1.1):
    """
    Compute omega needed to support given sigma values.
    
    Returns adjusted omega that satisfies Gindikin with given sigma.
    """
    beta = n + 1
    sigma_np = np.array(sigma)
    omega_np = np.array(omega).copy()
    
    sigma2 = sigma_np @ sigma_np
    required_omega = beta * sigma2 * margin
    
    # Take element-wise maximum
    adjusted_omega = np.maximum(omega_np, required_omega)
    
    return adjusted_omega


def print_gindikin_diagnostic(omega, sigma, n=2):
    """Print Gindikin diagnostic information."""
    is_valid, min_eig, max_sigma = check_gindikin(omega, sigma, n)
    
    print("\n" + "=" * 60)
    print("GINDIKIN CONDITION CHECK")
    print("=" * 60)
    print(f"omega diagonal:      [{omega[0,0]:.6f}, {omega[1,1]:.6f}]")
    print(f"sigma diagonal:      [{sigma[0,0]:.6f}, {sigma[1,1]:.6f}]")
    print(f"max feasible sigma:  [{max_sigma[0]:.6f}, {max_sigma[1]:.6f}]")
    print(f"sigma/max ratio:     [{abs(sigma[0,0])/max_sigma[0]:.1%}, {abs(sigma[1,1])/max_sigma[1]:.1%}]")
    print(f"min eigenvalue:      {min_eig:.6f}")
    print(f"Gindikin valid:      {'✅ YES' if is_valid else '❌ NO - VIOLATED!'}")
    
    if not is_valid:
        adjusted_omega = adjust_omega_for_sigma(omega, sigma, n)
        print(f"\nTo support current sigma, omega needs to be:")
        print(f"  omega[0,0]: {omega[0,0]:.4f} -> {adjusted_omega[0,0]:.4f} ({adjusted_omega[0,0]/omega[0,0]:.1f}x)")
        print(f"  omega[1,1]: {omega[1,1]:.4f} -> {adjusted_omega[1,1]:.4f} ({adjusted_omega[1,1]/omega[1,1]:.1f}x)")
    
    print("=" * 60 + "\n")
    return is_valid, min_eig, max_sigma


# =============================================================================
# SIGMA SENSITIVITY TEST
# =============================================================================

def test_sigma_sensitivity(calibrator, scales=[1.0, 0.1, 0.01, 0.001, 0.0001]):
    """
    Test how model swaption volatility scales with sigma.
    
    This helps diagnose if there's a scaling issue in the model.
    
    Parameters
    ----------
    calibrator : LRWJumpCalibrator
        The calibrator object (after creation, before or after calibration)
    scales : list
        List of scaling factors to test
    """
    print("\n" + "=" * 70)
    print("SIGMA SENSITIVITY TEST")
    print("=" * 70)
    
    model = calibrator.model
    n = model.n
    
    # Store original parameters
    original_alpha = float(model.alpha)
    original_x0 = np.array(model.x0).copy()
    original_omega = np.array(model.omega).copy()
    original_m = np.array(model.m).copy()
    original_sigma = np.array(model.sigma).copy()
    
    print(f"\nOriginal sigma: [{original_sigma[0,0]:.6f}, {original_sigma[1,1]:.6f}]")
    print(f"Original omega: [{original_omega[0,0]:.6f}, {original_omega[1,1]:.6f}]")
    
    # Get market vol for reference (first swaption)
    if hasattr(calibrator, 'daily_data') and hasattr(calibrator.daily_data, 'swaption_data_cube'):
        first_swaption = calibrator.daily_data.swaption_data_cube.iloc[0]["Object"]
        market_vol = first_swaption.vol
        market_price = first_swaption.market_price
        expiry = first_swaption.expiry_maturity #if hasattr(first_swaption, 'expiry_maturity') else first_swaption.expiry
        tenor = first_swaption.swap_tenor_maturity # if hasattr(first_swaption, 'SwapTenorMat') else first_swaption.swapTenor
        print(f"\nReference swaption: {expiry}Y x {tenor}Y")
        print(f"Market vol: {market_vol:.6f} ({market_vol*100:.2f}%)")
        print(f"Market price: {market_price:.6f}")
    else:
        market_vol = None
        market_price = None
        print("\nNo market data available for reference")
    
    print("\n" + "-" * 70)
    print(f"{'Scale':<10} {'Sigma[0,0]':<14} {'Sigma[1,1]':<14} {'Model Vol':<14} {'Model Price':<14} {'Gindikin':<10}")
    print("-" * 70)
    
    results = []
    
    for scale in scales:
        # Scale sigma
        scaled_sigma = original_sigma * scale
        
        # Check Gindikin before setting
        is_valid, min_eig, _ = check_gindikin(original_omega, scaled_sigma, n)
        
        if not is_valid:
            print(f"{scale:<10.4f} {scaled_sigma[0,0]:<14.8f} {scaled_sigma[1,1]:<14.8f} {'N/A':<14} {'N/A':<14} {'❌ VIOLATED':<10}")
            continue
        
        # Set model parameters
        model.set_model_params(
            n=n,
            alpha=original_alpha,
            x0=original_x0,
            omega=original_omega,
            m=original_m,
            sigma=scaled_sigma
        )
        
        # Update swaption market data to ensure model is ready
        calibrator.market_handler.update_swaption_market_data(
            model=model,
            market_based_strike=calibrator.config.use_market_based_strike
        )
        
        # Reprice swaptions
        try:
            # Get the first swaption and reprice it
            if hasattr(calibrator, 'daily_data') and hasattr(calibrator.daily_data, 'swaption_data_cube'):
                # Reprice using the calibrator's method
                # calibrator.objectives.reprice_swaption_market_data() # _reprice_swaptions(s
                calibrator.objectives._reprice_swaptions() # (s
                
                # Get the result
                first_swaption = calibrator.daily_data.swaption_data_cube.iloc[0]["Object"]
                model_vol = first_swaption.model_vol
                model_price = first_swaption.model_price
                
                print(f"{scale:<10.4f} {scaled_sigma[0,0]:<14.8f} {scaled_sigma[1,1]:<14.8f} {model_vol:<14.6f} {model_price:<14.6f} {'✅ OK':<10}")
                
                results.append({
                    'scale': scale,
                    'sigma_00': scaled_sigma[0,0],
                    'sigma_11': scaled_sigma[1,1],
                    'model_vol': model_vol,
                    'model_price': model_price,
                    'gindikin_valid': True
                })
            else:
                print(f"{scale:<10.4f} {scaled_sigma[0,0]:<14.8f} {scaled_sigma[1,1]:<14.8f} {'No data':<14} {'No data':<14} {'✅ OK':<10}")
                
        except Exception as e:
            print(f"{scale:<10.4f} {scaled_sigma[0,0]:<14.8f} {scaled_sigma[1,1]:<14.8f} {'ERROR':<14} {'ERROR':<14} {'✅ OK':<10}")
            print(f"    Error: {str(e)[:50]}")
    
    print("-" * 70)
    
    # Restore original parameters
    model.set_model_params(
        n=n,
        alpha=original_alpha,
        x0=original_x0,
        omega=original_omega,
        m=original_m,
        sigma=original_sigma
    )
    
    print(f"\n✅ Original parameters restored.")
    
    # Analysis
    if results and market_vol is not None:
        print("\n" + "=" * 70)
        print("ANALYSIS")
        print("=" * 70)
        
        for r in results:
            if r['model_vol'] > 0:
                vol_ratio = r['model_vol'] / market_vol
                print(f"Scale {r['scale']:.4f}: Model vol is {vol_ratio:.1f}x market vol")
        
        # Find the scale that gets closest to market vol
        closest = min(results, key=lambda x: abs(x['model_vol'] - market_vol) if x['model_vol'] > 0 else float('inf'))
        print(f"\nClosest to market: scale={closest['scale']}, model_vol={closest['model_vol']:.6f} vs market_vol={market_vol:.6f}")
        
        # Estimate required scale
        if results[0]['model_vol'] > 0:
            # Model vol scales roughly with sigma, so:
            estimated_scale = market_vol / results[0]['model_vol']
            print(f"\nEstimated scale needed: {estimated_scale:.6f}")
            print(f"This means sigma should be approximately:")
            print(f"  sigma[0,0] ≈ {original_sigma[0,0] * estimated_scale:.8f}")
            print(f"  sigma[1,1] ≈ {original_sigma[1,1] * estimated_scale:.8f}")
    
    print("=" * 70 + "\n")
    
    return results


# =============================================================================
# MAIN CALIBRATION EXAMPLE
# =============================================================================

def example_basic_calibration_with_diagnosis():
    """Basic calibration example for LRW Jump model with sigma sensitivity test."""
    print("=" * 70)
    print("Example: LRW Jump Model Calibration")
    print("=" * 70)
    
    # Create sample market data
    daily_data = create_sample_market_data()
    
    # Initialize model with starting parameters
    n = 2

    ### Parameters from your OIS/spread calibration
    alpha = 0.023681147024035454
    x0 = jnp.array([[0.4291363, 0.0], [0.0, 0.00264791]])
    omega = jnp.array([[0.54993196, 0.0], [0.0, 0.06779055]])
    m = jnp.array([[-0.60828161, 0.0], [0.0, -7.56179852]])
    
    # Starting sigma - use smaller values to test
    sigma = jnp.array([[0.1, 0.0], [0.0, 0.05]])  # Start with smaller sigma
    
    # =========================================================================
    # GINDIKIN CHECK
    # =========================================================================
    is_valid, min_eig, max_sigma = print_gindikin_diagnostic(omega, sigma, n)
    
    if not is_valid:
        print("⚠️  Adjusting sigma to satisfy Gindikin...")
        sigma = jnp.array([
            [max_sigma[0] * 0.9, 0.0],
            [0.0, max_sigma[1] * 0.9]
        ])
        print(f"Adjusted sigma: [{sigma[0,0]:.6f}, {sigma[1,1]:.6f}]")
    
    # =========================================================================
    # CREATE MODEL AND CALIBRATOR
    # =========================================================================
    
    config = CalibrationConfig(
        max_tenor=15,
        min_tenor=0.5,
        use_multi_thread=False,
        calibrate_on_swaption=True,
        calibrate_on_swaption_vol=True,
        verbose=True
    )
    
    lrw_model_config = LRWModelConfig(
        n=n, alpha=alpha, x0=x0, omega=omega, m=m, sigma=sigma,
        is_bru_config=False, has_jump=False
    )
    
    swaption_config = SwaptionConfig(
        maturity=1, tenor=1, strike=0.05,
        delta_float=0.5, delta_fixed=0.5
    )
    
    lrw_model = LRWModel(lrw_model_config, swaption_config)
    u1 = jnp.array([[1, 0], [0, 0]])
    u2 = jnp.array([[0, 0], [0, 1]])
    lrw_model.set_weight_matrices(u1, u2)
    
    print("\nInitial Model Parameters:")
    lrw_model.print_model()
    
    # Create calibrator
    calibrator = LRWJumpCalibrator(lrw_model, daily_data, config)
    # After creating calibrator:

    # Run diagnostic
    first_swaption = calibrator.daily_data.swaption_data_cube.iloc[0]["Object"]
    atm_swap_rate = calibrator.model.compute_swap_rate()
    annuity, swap_rate = calibrator.model.compute_annuity()
    print(f"annuity: {annuity}, swap rate: {swap_rate}")
    first_swaption.strike = atm_swap_rate
    print(f"\nSet first swaption strike to ATM swap rate: {atm_swap_rate}")
    
    calibrator.objectives._price_single_swaption(first_swaption)
    
    # # # Run diagnostic
    print("================ RUNNING : Run diagnostic verify_full_pricing_chain ================= ")
    from verify_full_pricing import verify_full_pricing_chain
    verify_full_pricing_chain(calibrator, swaption_index=-1)
    print("="*60)
    print("="*60)
    print("="*60)

    # # # Run diagnostic
    # print("================ RUNNING : Run diagnostic check_state_evolution ================= ")
    # from check_state_evolution import check_state_evolution
    # check_state_evolution(calibrator)
        
    # print("="*60)
    # print("="*60)
    # print("="*60)

    # # # Run diagnostic
    # print("================ RUNNING : Run diagnostic verify_forward_vs_spot_swap ================= ")
    # from verify_forward_swap import verify_forward_vs_spot_swap
    # verify_forward_vs_spot_swap(calibrator)
    # print("="*60)
    # print("="*60)
    # print("="*60)

    # # Run diagnostic
    # print("================ RUNNING : Run diagnostic comprehensive_swaption_diagnostic ================= ")
    # from diagnose_spread_calibration import diagnose_spread_calibration, test_swaption_without_spread
    # diagnose_spread_calibration(calibrator)
    # test_swaption_without_spread(calibrator)
    # print("="*60)
    # print("="*60)
    # print("="*60)

    # # Run diagnostic
    # print("================ RUNNING : Run diagnostic comprehensive_swaption_diagnostic ================= ")    
    # from comprehensive_swaption_diagnostic import comprehensive_swaption_diagnostic
    # result = comprehensive_swaption_diagnostic(calibrator)

    # # Run diagnostic
    # print("================ RUNNING : Run diagnostic test_is_spread_effect ================= ")
    # from test_is_spread_effect import test_is_spread_effect
    # result = test_is_spread_effect(calibrator)

    # # Run diagnostic
    # print("================ RUNNING : Run diagnostic ================= ")
    # from diagnose_swaption_pricing import diagnose_swaption_pricing
    # diagnose_swaption_pricing(calibrator)


    # =========================================================================
    # SIGMA SENSITIVITY TEST (before calibration)
    # =========================================================================
    print("\n" + "=" * 70)
    print("RUNNING SIGMA SENSITIVITY TEST BEFORE CALIBRATION")
    print("=" * 70)
    
    # Test with various scales
    sensitivity_results = test_sigma_sensitivity(
        calibrator, 
        scales=[1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
    )
    
    # # Run diagnostic
    # print("================ RUNNING : Run diagnostic ================= ")
    # from diagnose_swaption_pricing import diagnose_swaption_pricing
    # diagnose_swaption_pricing(calibrator)

    # =========================================================================
    # OPTIONAL: RUN CALIBRATION
    # =========================================================================
    RUN_CALIBRATION = False  # Set to True to run full calibration
    
    if RUN_CALIBRATION:
        print("\n" + "=" * 70)
        print("RUNNING CALIBRATION")
        print("=" * 70)
        
        # Use joint calibration
        results = calibrator.calibrate_full_joint()
        
        print("\nCalibration Results:")
        print_pretty(results)
        
        # Final Gindikin check
        print("\nFinal Gindikin verification:")
        print_gindikin_diagnostic(lrw_model.omega, lrw_model.sigma, n)
        
        # Generate report
        report_folder = r"E:\OneDrive\Dropbox\LinearRationalWishart_Work\Code\ED\LinearRationalWishart\LinearRationalWishart_NewCode\Output_results"
        calibrator.generate_report(report_folder)
        
        return calibrator, results
    else:
        print("\n" + "=" * 70)
        print("CALIBRATION SKIPPED (RUN_CALIBRATION = False)")
        print("Set RUN_CALIBRATION = True to run full calibration")
        print("=" * 70)
        
        return calibrator, sensitivity_results


def example_basic_calibration( data_date:str ='20250224'):
    """Basic calibration example for LRW Jump model with sigma sensitivity test."""
    print("=" * 70)
    print("Example: LRW Jump Model Calibration")
    print("=" * 70)
    
    # Create sample market data
    daily_data = create_sample_market_data( currency="EUR", data_date=data_date)
    
    # Initialize model with starting parameters
    n = 2

    ### Parameters from your OIS/spread calibration
    alpha = 0.023681147024035454
    x0 = jnp.array([[0.4291363, 0.0], [0.0, 0.00264791]])
    omega = jnp.array([[0.54993196, 0.0], [0.0, 0.06779055]])
    m = jnp.array([[-0.60828161, 0.0], [0.0, -7.56179852e-1]])
    # Starting sigma - use smaller values to test
    sigma = jnp.array([[0.1, 0.0], [0.0, 0.05]])  # Start with smaller sigma
    
    # x0= 2.5*x0
    # m= 2.5*m
    # omega= 2.5*omega

    # if case==1:
    
    #     x0= 1.0*x0
    #     m= 1.0*m
    #     omega= 1.0*omega

    # elif case==2:
    
    #     x0= 2.5*x0
    #     m= 2.5*m
    #     omega= 2.5*omega
        

    # elif case==3:
    #     x0= 5.0*x0
    #     m= 5.0*m
    #     omega= 5.0*omega
    # else:
    #     x0= 10.0*x0
    #     m= 10.0*m
    #     omega= 10.0*omega

    
    # =========================================================================
    # GINDIKIN CHECK
    # =========================================================================
    is_valid, min_eig, max_sigma = print_gindikin_diagnostic(omega, sigma, n)
    
    if not is_valid:
        print("⚠️  Adjusting sigma to satisfy Gindikin...")
        sigma = jnp.array([
            [max_sigma[0] * 0.9, 0.0],
            [0.0, max_sigma[1] * 0.9]
        ])
        print(f"Adjusted sigma: [{sigma[0,0]:.6f}, {sigma[1,1]:.6f}]")
    
    # =========================================================================
    # CREATE MODEL AND CALIBRATOR
    # =========================================================================
    
    config = CalibrationConfig(
        max_tenor=11,#15,
        min_tenor=0.5,
        use_multi_thread=False,
        calibrate_on_swaption=True,
        calibrate_on_swaption_vol=True,#False,#True,
        verbose=True
    )
    
    lrw_model_config = LRWModelConfig(
        n=n, alpha=alpha, x0=x0, omega=omega, m=m, sigma=sigma,
        is_bru_config=False, has_jump=False
    )
    
    swaption_config = SwaptionConfig(
        maturity=1, tenor=1, strike=0.05,
        delta_float=0.5, delta_fixed=0.5
    )
    
    lrw_model = LRWModel(lrw_model_config, swaption_config)
    simple_u1_u2=True#False#True#False
    if  simple_u1_u2:
        u1 = jnp.array([[1, 0], [0, 0.5]])#0.0]])
        u2 = jnp.array([[0, 0], [0, 1]])
        # u2 = None
    else:
        u1 = jnp.array([[1.0, 0.150], [0.150, 0.25]])
        # u1 = jnp.array([[1.0, 0.0], [0.0, 0.25]])
        # u1 = jnp.array([[1.0, 0.05], [0.05, 0.0]])
        u1 = jnp.array([[1.0, 0.0], [0.0, 0.50]])
        u2 = jnp.array([[0, 0], [0, 1]])
    
    
    u2=None
    lrw_model.set_weight_matrices(u1, u2)
    
    print("\nInitial Model Parameters:")
    lrw_model.print_model()
    
    # Create calibrator
    calibrator = LRWJumpCalibrator(lrw_model, daily_data, config)
    # After creating calibrator:

    # # Run diagnostic
    # first_swaption = calibrator.daily_data.swaption_data_cube.iloc[0]["Object"]
    # atm_swap_rate = calibrator.model.compute_swap_rate()
    # annuity, swap_rate = calibrator.model.compute_annuity()
    # print(f"annuity: {annuity}, swap rate: {swap_rate}")
    # first_swaption.strike = atm_swap_rate
    # print(f"\nSet first swaption strike to ATM swap rate: {atm_swap_rate}")
    
    # calibrator.objectives._price_single_swaption(first_swaption)
    
    
    # =========================================================================
    # OPTIONAL: RUN CALIBRATION
    # =========================================================================
    RUN_CALIBRATION = True # False  # Set to True to run full calibration
    
    if RUN_CALIBRATION:
        print("\n" + "=" * 70)
        print("RUNNING CALIBRATION")
        print("=" * 70)
        
        results=0
        # Use joint calibration
        calibrator.use_initial_alpha_curve=True
        results = calibrator.calibrate_full()
        # results = calibrator.calibrate_full_joint()
        
        print("\nCalibration Results:")
        print_pretty(results)
        
        # Final Gindikin check
        print("\nFinal Gindikin verification:")
        print_gindikin_diagnostic(lrw_model.omega, lrw_model.sigma, n)
        
        # Generate report
        report_folder = r"E:\OneDrive\Dropbox\LinearRationalWishart_Work\Code\ED\LinearRationalWishart\LinearRationalWishart_NewCode\Output_results"
        calibrator.generate_report(report_folder,reprice_instruments=False)
        # calibrator.generate_report(report_folder,reprice_instruments=True)
        
        return calibrator, results
    else:
        print("\n" + "=" * 70)
        print("CALIBRATION SKIPPED (RUN_CALIBRATION = False)")
        print("Set RUN_CALIBRATION = True to run full calibration")
        print("=" * 70)
        
        return calibrator, None ##sensitivity_results

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_sample_market_data(currency: str = "EUR", data_date:str ='20250224') -> DailyData:
    """Create sample market data for testing."""


    daily_data = get_ir_testing_excel_data(
        current_date=data_date,
        ccy=currency,
        rate_multiplier = 1.0, 
        vol_multiplier = 1.0,
        filter_tenors=True
    )
    return daily_data


def clear_jax_cache():
    jax.clear_caches()
    gc.collect()
    
    jax_cache = os.path.join(os.environ.get('LOCALAPPDATA', ''), 'jax')
    if os.path.exists(jax_cache):
        shutil.rmtree(jax_cache, ignore_errors=True)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("LRW Jump Model Calibration Examples")
    print("=" * 60)
    
    import jax
    jax.config.update("jax_disable_jit", True)

    # Run example with sensitivity test
    dates=[20250401,20250501,20250402,20250602,20250502,20250403,20250303,20250404,20250304,20250305,20250505,20250306,20250506,20250407,20250307,20250507,20250408,20250508,20250409,20250509,20250410,20250310,20250411,20250311,20250312,20250512,20250313,20250513,20250414,20250314,20250514,20250415,20250515,20250416,20250516,20250417,20250317,20250418,20250318,20250319,20250519,20250320,20250520,20250421,20250321,20250521,20250422,20250522,20250423,20250523,20250424,20250224,20250324,20250425,20250225,20250325,20250226,20250326,20250526,20250227,20250327,20250527,20250428,20250228,20250328,20250528,20250429,20250529,20250430,20250530,20250331
            ]
    dates=[20250401
        # ,20250501,20250402,20250602,20250502,20250403,20250303,20250404,20250304,20250305,20250505,20250306
        # ,20250506,20250407,20250307,20250507,20250408,20250508,20250409,20250509,20250410,20250310,20250411
        # ,20250311,20250312,20250512,20250313,20250513,20250414,20250314,20250514,20250415,20250515,20250416
        # ,20250516,20250417,20250317,20250418,20250318,20250319,20250519,20250320,20250520,20250421,20250321
        # ,20250521,20250422,20250522,20250423,20250523,20250424,20250224,20250324,20250425,20250225,20250325
        # ,20250226,20250326,20250526,20250227,20250327,20250527,20250428,20250228,20250328,20250528,20250429
        # ,20250529,20250430,20250530,20250331
            ]
    
    for data_date in dates:
        try:
            calibrator, results = example_basic_calibration( data_date=str(data_date))
        except Exception as e:
            print(f"Error during calibration for date {data_date}: {str(e)}")
    # calibrator, results = example_basic_calibration(case=1)
    # calibrator, results = example_basic_calibration(case=2)
    # calibrator, results = example_basic_calibration(case=3)
    # calibrator, results = example_basic_calibration(case=4)
    
    clear_jax_cache()

    print("\n\nExample completed!")
