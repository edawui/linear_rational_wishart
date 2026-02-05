"""
Sensitivity analysis examples for LRW models.

This module demonstrates comprehensive sensitivity calculations including
Greeks and parameter sensitivities using the new structured results system.
"""

from typing import Dict, List, Optional
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
from pathlib import Path

main_project_root = r"E:\OneDrive\Dropbox\LinearRationalWishart_Work\Code\ED\LinearRationalWishart\LinearRationalWishart_NewCode"

try:
    from ..models.interest_rate.config import SwaptionConfig, LRWModelConfig
    from ..models.interest_rate.lrw_model import LRWModel
    from ..pricing.swaption_pricer import LRWSwaptionPricer
    from ..utils.reporting import print_pretty
    from ..components.jump import JumpComponent
    from ..models.interest_rate.lrw_sensitivities import LRWSensitivityAnalyzer
    from ..sensitivities.results import (
        DeltaHedgingResult,
        VegaHedgingResult,
        GammaResult,
        MatrixResult,
        AlphaSensitivityResult,
        SensitivityReport,
        SensitivityLogger,
        InstrumentType,
        GreekType
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from linear_rational_wishart.components.jump import JumpComponent
    from linear_rational_wishart.models.interest_rate.config import SwaptionConfig, LRWModelConfig
    from linear_rational_wishart.models.interest_rate.lrw_model import LRWModel
    from linear_rational_wishart.pricing.swaption_pricer import LRWSwaptionPricer
    from linear_rational_wishart.models.interest_rate.lrw_sensitivities import LRWSensitivityAnalyzer
    from linear_rational_wishart.utils.reporting import print_pretty, export_results
    from linear_rational_wishart.sensitivities.results import (
        DeltaHedgingResult,
        VegaHedgingResult,
        GammaResult,
        MatrixResult,
        AlphaSensitivityResult,
        SensitivityReport,
        SensitivityLogger,
        InstrumentType,
        GreekType
    )


# =============================================================================
# Model Setup
# =============================================================================

def setup_standard_model() -> LRWModel:
    """Set up a standard LRW model for examples."""
    n = 2
    alpha = 0.05
    x0 = jnp.array([[0.12, -0.01], [-0.01, 0.005]])
    omega = jnp.array([[0.10, 0.002], [0.002, 0.0005]])
    m = jnp.array([[-0.4, 0.01], [0.02, -0.2]])
    sigma = jnp.array([[0.05, 0.02], [0.02, 0.047]])
    
    paper_v16_param=False#True
    paper_v16_param=True
    if paper_v16_param:
        # alpha = 0.024
        # x0 = jnp.array([[0.015,-1.055e-3],[-1.055e-3, 8.25e-4]])
        # omega = jnp.array([[0.110,-2.974e-3],[-2.974e-3, 1.377e-3]])
        # m = jnp.array([[-6.642,0.0],[0.0, -0.028]])
        # sigma = jnp.array([[0.165,-0.041],[-0.0410, 0.069]])


        alpha = 0.052
        x0 = jnp.array([[0.04,0.02],[0.02,0.03]])
        # omega = jnp.array([[0.110,-2.974e-3],[-2.974e-3, 1.377e-3]])
        m = jnp.array([[-0.15,0.06 ],[0.07, -0.12]])
        sigma = jnp.array([[0.04, 0.015],[0.015, 0.037]])
        beta=.0
        omega= beta*omega@omega
        u1 = np.array([[1 ,0.0],[0.0, 0.0]])
        u2 = np.array([[0.0,0.0],[0.0, 1.0]])

    # Initialize model
    lrw_model_config = LRWModelConfig(
        n=n, alpha=alpha, x0=x0, omega=omega, m=m, sigma=sigma
        ,is_bru_config=True
    )
    swaption_config = SwaptionConfig(
        maturity=5, tenor=5, strike=0.05, delta_float=0.5, delta_fixed=1.0
    )
    lrw_model = LRWModel(lrw_model_config, swaption_config)
    
    # Set weight matrices
    u1 = jnp.array([[1, 0], [0, 0]])
    u2 = jnp.array([[0, 0], [0, 1]])
    lrw_model.set_weight_matrices(u1, u2)
    
    return lrw_model


# =============================================================================
# Example 1: Delta Hedging with Structured Results
# =============================================================================

def example_delta_hedging():
    """Delta hedging strategies example using structured results."""
    print("=" * 70)
    print("Example 1: Delta Hedging Strategies (Structured Results)")
    print("=" * 70)
    
    # Set up model
    lrw_model = setup_standard_model()
    
    # Swaption parameters
    maturity = 1.0
    tenor = 2.0
    delta_float = 0.5
    delta_fixed = 0.5
    
    swaption_config = SwaptionConfig(
        maturity=maturity,
        tenor=tenor,
        strike=0.05,
        delta_float=delta_float,
        delta_fixed=delta_fixed
    )
    lrw_model.set_swaption_config(swaption_config)
    atm_strike = lrw_model.compute_swap_rate()
    swaption_config.strike = atm_strike
    lrw_model.set_swaption_config(swaption_config)

    print(f"\nSwaption: {maturity}Y x {tenor}Y")
    print(f"ATM Strike: {atm_strike:.4%}")
    
    # Price the option
    pricer = LRWSwaptionPricer(lrw_model)
    price, iv = pricer.price_swaption(method="fft", return_implied_vol=True)
    print(f"Option Price: {price:.6f}")
    print(f"Implied Vol: {iv:.2%}")
    
    # Compute delta sensitivities using the analyzer
    analyzer = LRWSensitivityAnalyzer(lrw_model)

    # delta_results = analyzer.compute_all_sensitivities(
    #     compute_delta=True,
    #     compute_vega=False,
    #     compute_gamma=False,
    #     compute_parameter_sensi=False
    # )
    # # Get structured delta results
    # delta_zc_result = delta_results
    # print(delta_zc_result)

    delta_zc_result   = analyzer.compute_delta_hedging(instrument_type="ZC")
    delta_swap_result = analyzer.compute_delta_hedging(instrument_type="SWAP")
    
    # Print formatted summaries
    print("\n" + delta_zc_result.summary())
    print("\n" + delta_swap_result.summary())
    
    # Access data programmatically
    print("\n--- Programmatic Access ---")
    print(f"ZC Hedging Price: {delta_zc_result.price:.6f}")
    print(f"Floating Leg Maturities: {list(delta_zc_result.floating_leg.keys())}")
    print(f"Fixed Leg Maturities: {list(delta_zc_result.fixed_leg.keys())}")
    
    # Export to different formats
    print("\n--- Export Formats ---")
    print("As Dictionary:", delta_zc_result.to_dict())
    
    # Visualize hedging portfolio
    _plot_delta_hedging(delta_zc_result)
    
    return delta_zc_result,delta_swap_result


def _plot_delta_hedging(result: DeltaHedgingResult):
    """Plot delta hedging portfolio."""
    # Combine floating and fixed leg data
    all_dates = []
    all_amounts = []
    all_types = []
    
    for t, delta in result.floating_leg.items():
        all_dates.append(t)
        all_amounts.append(delta)
        all_types.append('Floating')
    
    for t, delta in result.fixed_leg.items():
        all_dates.append(t)
        all_amounts.append(delta)
        all_types.append('Fixed')
    
    if not all_dates:
        print("No hedging data to plot.")
        return
    
    # Create DataFrame for easier plotting
    df = pd.DataFrame({
        'Maturity': all_dates,
        'Delta': all_amounts,
        'Leg': all_types
    }).sort_values('Maturity')
    
    plt.figure(figsize=(12, 6))
    
    colors = {'Floating': 'blue', 'Fixed': 'red'}
    for leg_type in ['Floating', 'Fixed']:
        mask = df['Leg'] == leg_type
        if mask.any():
            plt.bar(
                df[mask]['Maturity'],
                df[mask]['Delta'],
                width=0.1,
                alpha=0.7,
                color=colors[leg_type],
                label=f'{leg_type} Leg'
            )
    
    plt.xlabel('Maturity (years)', fontsize=12)
    plt.ylabel('Delta', fontsize=12)
    plt.title(f'Delta Hedging Portfolio ({result.instrument_type.value})\n'
              f'Strike: {result.strike:.4%} | Price: {result.price:.6f}',
              fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add summary box
    total_floating = sum(result.floating_leg.values())
    total_fixed = sum(result.fixed_leg.values())
    summary_text = (f'Total Floating: {total_floating:.4f}\n'
                   f'Total Fixed: {total_fixed:.4f}')
    plt.text(0.02, 0.95, summary_text, transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             verticalalignment='top', fontsize=10)
    
    plt.tight_layout()
    plt.show()


# =============================================================================
# Example 2: Vega Analysis with Structured Results
# =============================================================================

def example_vega_analysis():
    """Vega sensitivity analysis example using structured results."""
    print("\n" + "=" * 70)
    print("Example 2: Vega Sensitivity Analysis (Structured Results)")
    print("=" * 70)
    
    # Set up model
    lrw_model = setup_standard_model()
    
    # Swaption parameters
    maturity = 1.0
    tenor = 2.0
    delta_float = 0.5
    delta_fixed = 0.5
    
    swaption_config = SwaptionConfig(
        maturity=maturity,
        tenor=tenor,
        strike=0.05,
        delta_float=delta_float,
        delta_fixed=delta_fixed
    )
    atm_strike = lrw_model.compute_swap_rate()
    print(f"ATM Strike: {atm_strike:.4%}")
    
    # Test multiple strikes
    strikes = [
        atm_strike * 0.8,
        atm_strike * 0.9,
        atm_strike,
        atm_strike * 1.1,
        atm_strike * 1.2
    ]
    
    vega_results_list = []
    
    for strike in strikes:
        swaption_config.strike = strike
        lrw_model.set_swaption_config(swaption_config)
        
        # Price
        pricer = LRWSwaptionPricer(lrw_model)
        price, iv = pricer.price_swaption(method="fft", return_implied_vol=True)
        
        # Compute Vega using analyzer
        analyzer = LRWSensitivityAnalyzer(lrw_model)
        vega_matrix_result = analyzer.compute_vega_matrix()
        
        # Extract component values from the structured result
        vega_values = vega_matrix_result.values
        
        vega_results_list.append({
            'Strike': strike,
            'Moneyness': strike / atm_strike,
            'Price': price,
            'IV': iv,
            'Vega_11': vega_values[0, 0],
            'Vega_22': vega_values[1, 1],
            'Vega_12': vega_values[0, 1],
            'Result': vega_matrix_result
        })
        
        # Print individual result
        print(f"\nStrike: {strike:.4%} (Moneyness: {strike/atm_strike:.2f})")
        print(vega_matrix_result.summary())
    
    # Create summary DataFrame
    vega_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'Result'} 
                           for r in vega_results_list])
    
    print("\n" + "=" * 50)
    print("VEGA SUMMARY ACROSS STRIKES")
    print("=" * 50)
    # print(vega_df)
    # print("="*60)
    print(vega_df.to_string(index=False,float_format=lambda x: f"{x:.6f}")) ## float_format='%.6f'))
    
    # Plot results
    _plot_vega_analysis(vega_df)
    
    return vega_results_list


def _plot_vega_analysis(vega_df: pd.DataFrame):
    """Plot Vega analysis results."""
    plt.figure(figsize=(14, 10))
    
    # Implied volatility smile
    plt.subplot(2, 2, 1)
    plt.plot(vega_df['Moneyness'], vega_df['IV'] * 100, 'b-o', linewidth=2, markersize=8)
    plt.xlabel('Moneyness', fontsize=11)
    plt.ylabel('Implied Vol (%)', fontsize=11)
    plt.title('Implied Volatility Smile', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, label='ATM')
    plt.legend()
    
    # Vega by component
    plt.subplot(2, 2, 2)
    plt.plot(vega_df['Moneyness'], vega_df['Vega_11'], 'r-o', label='σ₁₁', linewidth=2, markersize=8)
    plt.plot(vega_df['Moneyness'], vega_df['Vega_22'], 'g-o', label='σ₂₂', linewidth=2, markersize=8)
    plt.plot(vega_df['Moneyness'], vega_df['Vega_12'], 'b-o', label='σ₁₂', linewidth=2, markersize=8)
    plt.xlabel('Moneyness', fontsize=11)
    plt.ylabel('Vega', fontsize=11)
    plt.title('Vega by Volatility Component', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    
    # Total Vega
    plt.subplot(2, 2, 3)
    total_vega = vega_df['Vega_11'] + vega_df['Vega_22'] + 2 * vega_df['Vega_12']
    plt.plot(vega_df['Moneyness'], total_vega, 'k-o', linewidth=2, markersize=8)
    
    # print("+"*60)
    # print(vega_df['Moneyness'])
    # print(total_vega)
    
    x = np.asarray(vega_df['Moneyness'], dtype=float)
    y = np.asarray(total_vega, dtype=float)
    plt.fill_between(x, 0.0, y, alpha=0.3)
    # plt.fill_between(vega_df['Moneyness'], 0, total_vega, alpha=0.3)

    plt.xlabel('Moneyness', fontsize=11)
    plt.ylabel('Total Vega', fontsize=11)
    plt.title('Total Vega Profile', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Price vs Moneyness
    plt.subplot(2, 2, 4)
    plt.plot(vega_df['Moneyness'], vega_df['Price'], 'purple', marker='o', linewidth=2, markersize=8)
    plt.xlabel('Moneyness', fontsize=11)
    plt.ylabel('Option Price', fontsize=11)
    plt.title('Swaption Price by Strike', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()


# =============================================================================
# Example 3: Full Sensitivity Report
# =============================================================================

def example_full_sensitivity_report():
    """Generate comprehensive sensitivity report using structured results."""
    print("\n" + "=" * 70)
    print("Example 3: Full Sensitivity Report (Structured Results)")
    print("=" * 70)
    
    # Set up model
    lrw_model = setup_standard_model()
    
    # Swaption parameters
    # maturity = 1.0
    # tenor = 2.0
    # delta_float = 0.5
    # delta_fixed = 0.5

    maturity = 2.0
    tenor = 4.0
    delta_float = 1.0
    delta_fixed = 1.0

    # delta_float = 0.5
    # delta_fixed = 1.0
    # maturity =1 # 
    # tenor = 1 

    swaption_config = SwaptionConfig(
        maturity=maturity,
        tenor=tenor,
        strike=0.052,
        delta_float=delta_float,
        delta_fixed=delta_fixed
    )
    lrw_model.set_swaption_config(swaption_config)
    atm_strike = lrw_model.compute_swap_rate()
    swaption_config.strike = atm_strike
    lrw_model.set_swaption_config(swaption_config)
    
    lrw_model.print_model()
    print(f"\nSwaption: {maturity}Y x {tenor}Y")
    print(f"Strike: {atm_strike:.4%} (ATM)")
    # atm_strike = 0.052##
    print(f"Strike: {atm_strike:.4%} ")
    
    # Price swaption
    pricer = LRWSwaptionPricer(lrw_model)
    price, iv = pricer.price_swaption(method="fft", return_implied_vol=True)
    print(f"Price: {price:.6f}")
    print(f"Implied Vol: {iv:.2%}")
    
    # Create comprehensive sensitivity report
    report = SensitivityReport(
        strike=atm_strike,
        maturity=maturity,
        tenor=tenor
    )
    
    # Initialize analyzer
    analyzer = LRWSensitivityAnalyzer(lrw_model)
    report = analyzer.compute_all_sensitivities(
        compute_delta=True,#True,#False,#True,
        compute_vega= False,#True,#False,#True,#False,#True,
        compute_gamma=False,#True,#False,#False,#True,#False,#True,#False,  # Skip gamma for brevity
        compute_parameter_sensi=False,#True,#False,#True,
        print_intermediate=True,#False
        return_structured = True#False
    )
    
    # Compute all sensitivities and add to report
    print("\nComputing sensitivities...")
    compute_these=False
    if compute_these:
        # Delta hedging (ZC and SWAP)
        print("  - Computing Delta (ZC)...")
        delta_zc = analyzer.compute_delta_hedging(instrument_type="ZC")
        report.add_delta(delta_zc)
    
        print("  - Computing Delta (SWAP)...")
        delta_swap = analyzer.compute_delta_hedging(instrument_type="SWAP")
        report.add_delta(delta_swap)
    
        # Vega matrix
        print("  - Computing Vega matrix...")
        vega_matrix = analyzer.compute_vega_matrix()
        report.add_matrix(vega_matrix)
    
        # Vega hedging for each component
        n = lrw_model.n
        for i in range(n):
            for j in range(n):
                print(f"  - Computing Vega hedging ({i},{j})...")
                vega_hedge = analyzer.compute_vega_hedging(i, j, instrument_type="ZC")
                report.add_vega(vega_hedge)
    
        # Gamma (bond and swap cross)
        print("  - Computing Gamma (Bond)...")
        gamma_bond = analyzer.compute_gamma(0, 1, instrument_type="BOND")
        report.add_gamma(gamma_bond)
    
        print("  - Computing Gamma (Swap Cross)...")
        gamma_swap = analyzer.compute_gamma_swap_cross("FIXED", "FLOATING")
        report.add_gamma(gamma_swap)
    
        # Alpha sensitivity
        print("  - Computing Alpha sensitivity...")
        alpha_result = analyzer.compute_alpha_sensitivity()
        report.set_alpha(alpha_result)
    
        # Omega sensitivity
        print("  - Computing Omega sensitivity...")
        omega_matrix = analyzer.compute_omega_sensitivity()
        report.add_matrix(omega_matrix)
    
        # M sensitivity
        print("  - Computing M sensitivity...")
        m_matrix = analyzer.compute_m_sensitivity()
        report.add_matrix(m_matrix)
    
    # Print full report
    print("\n" + report.summary())
    
    # Export to various formats
    output_folder = Path(main_project_root) / "Output_results" / "Sensitivity"
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # JSON export
    json_path = output_folder / "sensitivity_report.json"
    with open(json_path, 'w') as f:
        f.write(report.to_json())
    print(f"\nJSON report exported to: {json_path}")
    
    # Flat dict for legacy compatibility
    flat_dict = report.to_flat_dict()
    
    # CSV export
    csv_path = output_folder / "sensitivity_flat.csv"
    flat_df = pd.DataFrame([flat_dict])
    flat_df.to_csv(csv_path, index=False)
    print(f"CSV report exported to: {csv_path}")
    
    return report


# =============================================================================
# Example 4: Hedging Strategy Analysis
# =============================================================================

def example_hedging_analysis():
    """Analyze hedging strategies for different market scenarios."""
    print("\n" + "=" * 70)
    print("Example 4: Hedging Strategy Analysis (Structured Results)")
    print("=" * 70)
    
    # Set up model
    lrw_model = setup_standard_model()
    
    # Swaption parameters
    maturity = 2.0
    tenor = 5.0
    delta_float = 0.5
    delta_fixed = 0.5
    
    swaption_config = SwaptionConfig(
        maturity=maturity,
        tenor=tenor,
        strike=0.05,
        delta_float=delta_float,
        delta_fixed=delta_fixed
    )
    atm_strike = lrw_model.compute_swap_rate()
    
    print(f"\nSwaption: {maturity}Y x {tenor}Y")
    print(f"ATM Strike: {atm_strike:.4%}")
    
    # Test different strikes around ATM
    strike_multipliers = [0.9, 0.95, 1.0, 1.05, 1.1]
    
    hedging_results = []
    all_delta_results = []
    
    for mult in strike_multipliers:
        strike = atm_strike * mult
        swaption_config.strike = strike
        lrw_model.set_swaption_config(swaption_config)
        
        # Get hedging strategies using analyzer
        analyzer = LRWSensitivityAnalyzer(lrw_model)
        
        # ZC hedging
        zc_result = analyzer.compute_delta_hedging(instrument_type="ZC")
        
        # Swap hedging
        swap_result = analyzer.compute_delta_hedging(instrument_type="SWAP")
        
        # Price for reference
        pricer = LRWSwaptionPricer(lrw_model)
        price = pricer.price_swaption(method="fft")
        
        # Extract swap hedge amounts
        fixed_delta = sum(swap_result.fixed_leg.values()) if swap_result.fixed_leg else 0
        float_delta = sum(swap_result.floating_leg.values()) if swap_result.floating_leg else 0
        
        hedging_results.append({
            'Strike': strike,
            'Moneyness': mult,
            'Price': price,
            'ZC_Price': zc_result.price,
            'SWAP_Price': swap_result.price,
            'Swap_Fixed_Delta': fixed_delta,
            'Swap_Float_Delta': float_delta,
            'ZC_Floating_Count': len(zc_result.floating_leg),
            'ZC_Fixed_Count': len(zc_result.fixed_leg)
        })
        
        all_delta_results.append({
            'mult': mult,
            'zc': zc_result,
            'swap': swap_result
        })
        
        # Print individual summary
        print(f"\n--- Strike: {strike:.4%} (Moneyness: {mult:.2f}) ---")
        print(f"Price: {price:.6f}")
        print(f"Swap Fixed Delta: {fixed_delta:.6f}")
        print(f"Swap Float Delta: {float_delta:.6f}")
    
    # Display results as DataFrame
    hedge_df = pd.DataFrame(hedging_results)
    
    print("\n" + "=" * 60)
    print("HEDGING STRATEGIES SUMMARY")
    print("=" * 60)
    print(hedge_df.to_string(index=False, float_format='%.6f'))
    
    # Visualize
    _plot_hedging_analysis(hedge_df)
    
    return hedge_df, all_delta_results


def _plot_hedging_analysis(hedge_df: pd.DataFrame):
    """Plot hedging analysis results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Swap hedging by strike
    ax1 = axes[0, 0]
    ax1.plot(hedge_df['Moneyness'], hedge_df['Swap_Fixed_Delta'], 
             'b-o', label='Fixed Leg', linewidth=2, markersize=8)
    ax1.plot(hedge_df['Moneyness'], hedge_df['Swap_Float_Delta'], 
             'r-o', label='Floating Leg', linewidth=2, markersize=8)
    ax1.set_xlabel('Moneyness', fontsize=11)
    ax1.set_ylabel('Delta', fontsize=11)
    ax1.set_title('Swap Hedging by Strike', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    
    # Option price by strike
    ax2 = axes[0, 1]
    ax2.plot(hedge_df['Moneyness'], hedge_df['Price'], 
             'g-o', linewidth=2, markersize=8)
    ax2.set_xlabel('Moneyness', fontsize=11)
    ax2.set_ylabel('Option Price', fontsize=11)
    ax2.set_title('Swaption Price by Strike', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    
    # Price comparison: ZC vs SWAP
    ax3 = axes[1, 0]
    ax3.plot(hedge_df['Moneyness'], hedge_df['ZC_Price'], 
             'b-o', label='ZC Method', linewidth=2, markersize=8)
    ax3.plot(hedge_df['Moneyness'], hedge_df['SWAP_Price'], 
             'r-s', label='SWAP Method', linewidth=2, markersize=8)
    ax3.set_xlabel('Moneyness', fontsize=11)
    ax3.set_ylabel('Hedging Price', fontsize=11)
    ax3.set_title('Hedging Price Comparison', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    
    # Total delta exposure
    ax4 = axes[1, 1]
    total_delta = hedge_df['Swap_Fixed_Delta'] + hedge_df['Swap_Float_Delta']
    ax4.bar(hedge_df['Moneyness'], total_delta, width=0.03, alpha=0.7, color='purple')
    ax4.set_xlabel('Moneyness', fontsize=11)
    ax4.set_ylabel('Total Delta', fontsize=11)
    ax4.set_title('Total Delta Exposure', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()


# =============================================================================
# Example 5: Gamma Analysis
# =============================================================================

def example_gamma_analysis():
    """Gamma sensitivity analysis example."""
    print("\n" + "=" * 70)
    print("Example 5: Gamma Sensitivity Analysis (Structured Results)")
    print("=" * 70)
    
    # Set up model
    lrw_model = setup_standard_model()
    
    # Swaption parameters
    maturity = 1.0
    tenor = 2.0
    
    swaption_config = SwaptionConfig(
        maturity=maturity,
        tenor=tenor,
        strike=0.05,
        delta_float=0.5,
        delta_fixed=0.5
    )
    lrw_model.set_swaption_config(swaption_config)
    atm_strike = lrw_model.compute_swap_rate()
    swaption_config.strike = atm_strike
    lrw_model.set_swaption_config(swaption_config)
    
    print(f"\nSwaption: {maturity}Y x {tenor}Y")
    print(f"Strike: {atm_strike:.4%} (ATM)")
    
    # Initialize analyzer
    analyzer = LRWSensitivityAnalyzer(lrw_model)
    
    # Compute bond gammas
    print("\n--- Bond Gamma Results ---")
    gamma_results = []
    
    # Compute for different bond pairs
    bond_pairs = [(0, 0), (0, 1), (1, 1)]
    
    for id0, id1 in bond_pairs:
        gamma_result = analyzer.compute_gamma(id0, id1, instrument_type="BOND")
        gamma_results.append(gamma_result)
        print(gamma_result.summary())
    
    # Compute swap cross gammas
    print("\n--- Swap Cross Gamma Results ---")
    swap_combos = [
        ("FIXED", "FIXED"),
        ("FLOATING", "FLOATING"),
        ("FIXED", "FLOATING")
    ]
    
    swap_gamma_results = []
    for first, second in swap_combos:
        gamma_result = analyzer.compute_gamma_swap_cross(first, second)
        swap_gamma_results.append(gamma_result)
        print(gamma_result.summary())
    
    return gamma_results, swap_gamma_results


# =============================================================================
# Example 6: Comprehensive Report with Export
# =============================================================================

def example_comprehensive_export():
    """Generate and export comprehensive sensitivity analysis."""
    print("\n" + "=" * 70)
    print("Example 6: Comprehensive Export (All Formats)")
    print("=" * 70)
    
    # Run full analysis
    report = example_full_sensitivity_report()
    
    # Export paths
    output_folder = Path(main_project_root) / "Output_results" / "Sensitivity"
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Export formats
    exports = {}
    
    # 1. JSON (structured)
    json_path = output_folder / "full_report.json"
    with open(json_path, 'w') as f:
        f.write(report.to_json())
    exports['json'] = json_path
    
    # 2. Text summary
    txt_path = output_folder / "full_report.txt"
    with open(txt_path, 'w') as f:
        f.write(report.summary())
    exports['txt'] = txt_path
    
    # 3. CSV (flat)
    csv_path = output_folder / "full_report_flat.csv"
    flat_dict = report.to_flat_dict()
    pd.DataFrame([flat_dict]).to_csv(csv_path, index=False)
    exports['csv_flat'] = csv_path
    
    # 4. CSV (delta details)
    if report.delta_results:
        delta_rows = []
        for delta_result in report.delta_results:
            for t, v in delta_result.floating_leg.items():
                delta_rows.append({
                    'Instrument': delta_result.instrument_type.value,
                    'Leg': 'Floating',
                    'Maturity': t,
                    'Delta': v,
                    'Strike': delta_result.strike
                })
            for t, v in delta_result.fixed_leg.items():
                delta_rows.append({
                    'Instrument': delta_result.instrument_type.value,
                    'Leg': 'Fixed',
                    'Maturity': t,
                    'Delta': v,
                    'Strike': delta_result.strike
                })
        
        delta_csv_path = output_folder / "delta_details.csv"
        pd.DataFrame(delta_rows).to_csv(delta_csv_path, index=False)
        exports['csv_delta'] = delta_csv_path
    
    # 5. CSV (vega details)
    if report.vega_results:
        vega_rows = []
        for vega_result in report.vega_results:
            vega_rows.append({
                'Component_i': vega_result.component_i,
                'Component_j': vega_result.component_j,
                'Instrument': vega_result.instrument_type.value,
                'Vega': vega_result.vega_value,
                'Strike': vega_result.strike
            })
        
        vega_csv_path = output_folder / "vega_details.csv"
        pd.DataFrame(vega_rows).to_csv(vega_csv_path, index=False)
        exports['csv_vega'] = vega_csv_path
    
    print("\n" + "=" * 50)
    print("EXPORT SUMMARY")
    print("=" * 50)
    for format_type, path in exports.items():
        print(f"  {format_type:15}: {path}")
    
    return exports


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("#  LRW SENSITIVITY ANALYSIS EXAMPLES")
    print("#  Using Structured Results System")
    print("#" * 70)
    
    # Uncomment the examples you want to run
    
    # Example 1: Delta hedging
    # example_delta_hedging()
    
    # Example 2: Vega analysis
    # example_vega_analysis()
    
    # Example 3: Full sensitivity report
    example_full_sensitivity_report()
    
    # Example 4: Hedging analysis
    # example_hedging_analysis()
    
    # Example 5: Gamma analysis
    # example_gamma_analysis()
    
    # Example 6: Comprehensive export
    # example_comprehensive_export()
