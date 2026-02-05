"""
Sensitivity analysis examples for LRW models.

This module demonstrates comprehensive sensitivity calculations including
Greeks and parameter sensitivities.
"""

from typing import ValuesView
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
from pathlib import Path

main_project_root = r"E:\OneDrive\Dropbox\LinearRationalWishart_Work\Code\ED\LinearRationalWishart\LinearRationalWishart_NewCode"


# from lrw_numpy.LrwInterestRateBru import LrwInterestRateBru
# from linear_rational_wishart.models.interest_rate.lrw_sensitivities import LRWSensitivityAnalyzer
# from linear_rational_wishart.pricing import LRWSwaptionPricer
# from linear_rational_wishart.utils.reporting import print_pretty, export_results, SensitivityReporter

try:
    from ..models.interest_rate.config import SwaptionConfig, LRWModelConfig
    from ..models.interest_rate.lrw_model import LRWModel
    from ..pricing.swaption_pricer import LRWSwaptionPricer
    from ..utils.reporting import print_pretty
    from ..components.jump  import JumpComponent
    from ..models.interest_rate.lrw_sensitivities import LRWSensitivityAnalyzer
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from linear_rational_wishart.components.jump  import JumpComponent
    from linear_rational_wishart.models.interest_rate.config import SwaptionConfig, LRWModelConfig
    from linear_rational_wishart.models.interest_rate.lrw_model import LRWModel
    from linear_rational_wishart.pricing.swaption_pricer import LRWSwaptionPricer
    from linear_rational_wishart.models.interest_rate.lrw_sensitivities import LRWSensitivityAnalyzer
    from linear_rational_wishart.utils.reporting import print_pretty, export_results, SensitivityReporter



def setup_standard_model():
    """Set up a standard LRW model for examples."""
    # n = 2
    # alpha = 0.05
    # x0 = jnp.array([[0.12, -0.01], [-0.01, 0.005]])
    # omega = jnp.array([[0.10, 0.002], [0.002, 0.0005]])
    # m = jnp.array([[-0.4, 0.01], [0.02, -0.2]])
    # sigma = jnp.array([[0.05, 0.02], [0.02, 0.047]])
    
    # lrw_model = LrwInterestRateBru(n, alpha, x0, omega, m, sigma)
    
    # u1 = jnp.array([[1, 0], [0, 0]])
    # u2 = jnp.array([[0, 0], [0, 1]])
    # lrw_model.SetU1(u1)
    # lrw_model.SetU2(u2)
     # Model parameters
    # n = 2
    # alpha = 0.05
    # x0 = jnp.array([[0.02, 0.005], [0.005, 0.015]])
    # omega = jnp.array([[0.06, 0.001], [0.001, 0.07]])
    # m = jnp.array([[-0.29, 0.1], [0.2, -0.5]])
    # sigma = jnp.array([[0.03, 0.1], [0.1, 0.1]])
    
    # Set up model
    n = 2
    alpha = 0.05
    x0 = jnp.array([[0.12, -0.01], [-0.01, 0.005]])
    omega = jnp.array([[0.10, 0.002], [0.002, 0.0005]])
    m = jnp.array([[-0.4, 0.01], [0.02, -0.2]])
    sigma = jnp.array([[0.05, 0.02], [0.02, 0.047]])
    
    # # Swaption parameters
    # maturity = 1.0
    # tenor = 2.0
    # delta_float = 0.5
    # delta_fixed = 0.5 #1.0
    # Initialize model
    lrw_model_config = LRWModelConfig( n=n,  alpha=alpha,  x0=x0,  omega=omega,  m=m, sigma=sigma    )
    swaption_config = SwaptionConfig(maturity=5,tenor=5,  strike=0.05, delta_float = 0.5,delta_fixed= 1.0)
    lrw_model = LRWModel(lrw_model_config,swaption_config)
    
    # Set  matrices
    u1 = jnp.array([[1, 0], [0, 0]])
    u2 = jnp.array([[0, 0], [0, 1]])   
    lrw_model.set_weight_matrices(u1,u2)
    
    return lrw_model


def example_delta_hedging():
    """Delta hedging strategies example."""
    print("Example 1: Delta Hedging Strategies")
    print("-" * 40)
    
    # Set up model
    lrw_model = setup_standard_model()
    
    # Swaption parameters
    maturity = 1.0
    tenor = 2.0
    delta = 0.5
    delta_float = 0.5
    delta_fixed = 0.5 #1.0
    # Set at ATM
 
    swaption_config = SwaptionConfig(maturity=maturity,tenor=tenor,
                                   strike=0.05, 
                                   delta_float = delta_float,
                                   delta_fixed= delta_fixed)
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
    
    # Compute delta sensitivities
    analyzer = LRWSensitivityAnalyzer(lrw_model)
    delta_results = analyzer.compute_all_sensitivities(
        compute_delta=True,
        compute_vega=False,
        compute_gamma=False,
        compute_parameter_sensi=False
    )
    
    print("\nDelta Hedging Ratios:")
    print_pretty(delta_results)
    
    # Extract and visualize ZC hedging portfolio
    hedge_dates = []
    hedge_amounts = []
    
    for key, value in delta_results.items():
        # if "ZC_Hedge" in key and "date" in key:
        if ("ZC" in key and "FIXEDLEG" in key) or ("ZC" in key and "FLOATINGLEG" in key):
            # Extract date number from key
            parts = key.split(':') #'_')
            if parts[-1].replace('.', '').isdigit():
                date = float(parts[-1])
                hedge_dates.append(date)
                amount_key = key.replace("date", "amount")
                # if amount_key in delta_results:
                #     hedge_amounts.append(delta_results[amount_key])
                hedge_amounts.append(value)
   
    if hedge_dates:
        plt.figure(figsize=(10, 6))
        plt.bar(hedge_dates, hedge_amounts, width=0.1, alpha=0.7, color='blue')
        plt.xlabel('Maturity (years)')
        plt.ylabel('Hedge Amount')
        plt.title('Zero Coupon Bond Hedging Portfolio')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Add total hedge value
        total_hedge = sum(hedge_amounts)
        plt.text(0.02, 0.95, f'Total Hedge: {total_hedge:.6f}', 
                transform=plt.gca().transAxes, bbox=dict(boxstyle='round', facecolor='wheat'))
        
        plt.show()


def example_vega_analysis():
    """Vega sensitivity analysis example."""
    print("\n\nExample 2: Vega Sensitivity Analysis")
    print("-" * 40)
    
    # Set up model
    lrw_model = setup_standard_model()
    
    # # Test multiple strikes
    # maturity = 2.0
    # tenor = 5.0
    # lrw_model.SetOptionProperties(tenor, maturity, 0.5, 0.5, 0.0)
    # atm_strike = lrw_model.ComputeSwapRate()
    
    # Swaption parameters
    maturity = 1.0 #2.0
    tenor = 2.0# 5.0
    delta = 0.5
    delta_float = 0.5
    delta_fixed = 0.5 #1.0
    # Set at ATM
 
    swaption_config = SwaptionConfig(maturity=maturity,tenor=tenor,
                                   strike=0.05, 
                                   delta_float = delta_float,
                                   delta_fixed= delta_fixed)
    # lrw_model.set_swaption_config(swaption_config)
    atm_strike = lrw_model.compute_swap_rate()
    swaption_config.strike = atm_strike   
    lrw_model.set_swaption_config(swaption_config)
    print(f"atm strike={atm_strike:.4%}")
    strikes = [
         # atm_strike

        atm_strike * 0.8,  # 80% of ATM
        atm_strike * 0.9,  # 90% of ATM
        atm_strike,        # ATM
        atm_strike * 1.1,  # 110% of ATM
        atm_strike * 1.2   # 120% of ATM
    ]
    
    vega_results = []
    
    for strike in strikes:
        swaption_config.strike = strike   
        lrw_model.set_swaption_config(swaption_config)
        # lrw_model.SetOptionProperties(tenor, maturity, 0.5, 0.5, strike)
        
        # Price and compute vega
        pricer = LRWSwaptionPricer(lrw_model)
        price, iv = pricer.price_swaption(method="fft", return_implied_vol=True)
        
        # Vega sensitivity
        # vega_sensi, vega_report = lrw_model.PriceOptionVega()
        
          # Compute delta sensitivities
        analyzer = LRWSensitivityAnalyzer(lrw_model)
        vega_report = analyzer.compute_all_sensitivities(
            compute_delta=False,
            compute_vega=True,
            compute_gamma=False,
            compute_parameter_sensi=False

        )
        def get_vega_value(vega_report, strike, i, j):
            """Extract VEGAVALUE for indices i,j"""
            key = f'VEGA_FOR_STRIKE:{strike}:{i}_{j}:VEGAVALUE:ALL:NA' ##'ZC:VEGAVALUE:NA'
            value = vega_report.get(key, 0)
            return float(value) if value != 0 else 0

        # print(f" 'Strike': {strike},  vega_results:={vega_report}")
        vega_results.append({
            'Strike': strike,
            'Moneyness': strike / atm_strike,
            'Price': price,
            'IV': iv,
            'Vega_11': get_vega_value(vega_report, strike, 0, 0),
            'Vega_22': get_vega_value(vega_report, strike, 1, 1),
            'Vega_12': get_vega_value(vega_report, strike, 0, 1)

            # 'Vega_11': vega_report.get('Vega_Sigma_0_0', 0),
            # 'Vega_22': vega_report.get('Vega_Sigma_1_1', 0),
            # 'Vega_12': vega_report.get('Vega_Sigma_0_1', 0)
        })
        print(f"vega_results={vega_results}")
    print("======= VEGA computed =============")
    # Display results
    vega_df = pd.DataFrame(vega_results)
    print(vega_df)
    print("\nVega Sensitivities by Strike:")
    # print(vega_df.to_string(index=False, float_format='%.6f'))
    
    # Plot vega profile
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(vega_df['Moneyness'], vega_df['IV'] * 100, 'b-o', linewidth=2)
    plt.xlabel('Moneyness')
    plt.ylabel('Implied Vol (%)')
    plt.title('Implied Volatility Smile')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(vega_df['Moneyness'], vega_df['Vega_11'], 'r-o', label='σ₁₁', linewidth=2)
    plt.plot(vega_df['Moneyness'], vega_df['Vega_22'], 'g-o', label='σ₂₂', linewidth=2)
    plt.plot(vega_df['Moneyness'], vega_df['Vega_12'], 'b-o', label='σ₁₂', linewidth=2)
    plt.xlabel('Moneyness')
    plt.ylabel('Vega')
    plt.title('Vega by Volatility Component')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    total_vega = vega_df['Vega_11'] + vega_df['Vega_22'] + 2 * vega_df['Vega_12']
    plt.plot(vega_df['Moneyness'], total_vega, 'k-o', linewidth=2)
    plt.xlabel('Moneyness')
    plt.ylabel('Total Vega')
    plt.title('Total Vega Profile')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def example_full_sensitivity_report():
    """Generate comprehensive sensitivity report."""
    print("\n\nExample 3: Full Sensitivity Report")
    print("-" * 40)
    
    # Set up model
    lrw_model = setup_standard_model()
    
    # Swaption parameters
   
    # Swaption parameters
    maturity = 1.0
    tenor = 2.0 #5.0
    delta = 0.5
    delta_float = 0.5
    delta_fixed = 0.5 #1.0
    # Set at ATM
 
    swaption_config = SwaptionConfig(maturity=maturity,tenor=tenor,
                                   strike=0.05, 
                                   delta_float = delta_float,
                                   delta_fixed= delta_fixed)
    lrw_model.set_swaption_config(swaption_config)
    atm_strike = lrw_model.compute_swap_rate()
    swaption_config.strike = atm_strike   
    lrw_model.set_swaption_config(swaption_config)
    print(f"\nSwaption: {maturity}Y x {tenor}Y")
    print(f"Strike: {atm_strike:.4%} (ATM)")
    
    # Price swaption
    pricer = LRWSwaptionPricer(lrw_model)
    price, iv = pricer.price_swaption(method="fft", return_implied_vol=True)
    print(f"Price: {price:.6f}")
    print(f"Implied Vol: {iv:.2%}")
    
    # Compute all sensitivities
    analyzer = LRWSensitivityAnalyzer(lrw_model)
    all_sensitivities = analyzer.compute_all_sensitivities(
        compute_delta=False,#True,#False,#True,
        compute_vega= False,#True,#False,#True,
        compute_gamma=False,#True,#False,#True,#False,  # Skip gamma for brevity
        compute_parameter_sensi=True,
        print_intermediate=True#False
    )
    
    # Add pricing info
    all_sensitivities['Pricing'] = {
        'Maturity': maturity,
        'Tenor': tenor,
        'Strike': atm_strike,
        'Price': price,
        'ImpliedVol': iv
    }
    print("\nFull Sensitivity Report:")
    # Create structured report
    reporter = SensitivityReporter()
    
    # Organize results by category
    delta_results = {k: v for k, v in all_sensitivities.items() if 'Hedge' in k}
    vega_results = {k: v for k, v in all_sensitivities.items() if 'Vega' in k}
    param_results = {k: v for k, v in all_sensitivities.items() if 'Sensi' in k}
    
    reporter.add_results('Pricing', all_sensitivities.get('Pricing', {}))
    reporter.add_results('Delta Sensitivities', delta_results)
    reporter.add_results('Vega Sensitivities', vega_results)
    reporter.add_results('Parameter Sensitivities', param_results)
    
    # Generate and print report
    print("\n" + reporter.generate_report())
    # folder =  main_project_root + r"\wishart_processes\examples\results"
    folder = r"E:\OneDrive\Dropbox\LinearRationalWishart_Work\Code\ED\LinearRationalWishart\LinearRationalWishart_NewCode\Output_results\Sensitivity"

    # Export results
    export_paths = export_results(
        all_sensitivities,
        folder+r'\lrw_sensitivity_report',
        formats=['json', 'csv', 'txt']
    )
    
    print("\nSensitivity report exported to:")
    for format_type, path in export_paths.items():
        print(f"  {format_type}: {path}")
    
    # Print summary statistics
    print("\nSensitivity Summary:")
    print(f"  Alpha sensitivity: {all_sensitivities.get('Sensi_Alpha', 'N/A')}")
    print(f"  Total vega: {sum(v for k, v in all_sensitivities.items() if 'Vega_Sigma' in k):.6f}")
    
    return all_sensitivities


def example_parameter_sensitivity_surface():
    """Plot sensitivity surfaces for different parameters."""
    print("\n\nExample 4: Parameter Sensitivity Surfaces")
    print("-" * 40)
    
    # Base model
    lrw_model = setup_standard_model()
    
    # Parameter ranges
    maturity_range = jnp.linspace(0.5, 5, 10)
    tenor_range = jnp.linspace(1, 10, 10)
    
    # Compute sensitivity surface
    alpha_sensi_surface = np.zeros((len(maturity_range), len(tenor_range)))
    vega_sensi_surface = np.zeros((len(maturity_range), len(tenor_range)))
    price_surface = np.zeros((len(maturity_range), len(tenor_range)))
    
    for i, mat in enumerate(maturity_range):
        for j, ten in enumerate(tenor_range):
            # Set swaption
            lrw_model.SetOptionProperties(ten, mat, 0.5, 0.5, 0.0)
            atm = lrw_model.ComputeSwapRate()
            lrw_model.SetOptionProperties(ten, mat, 0.5, 0.5, atm)
            
            # Price
            price = lrw_model.PriceOption()
            price_surface[i, j] = price
            
            # Alpha sensitivity
            alpha_sensi, _ = lrw_model.PriceOptionSensiAlpha()
            alpha_sensi_surface[i, j] = alpha_sensi
            
            # Total vega
            vega_sensi, vega_report = lrw_model.PriceOptionVega()
            total_vega = sum(v for k, v in vega_report.items() if 'Vega_Sigma' in k)
            vega_sensi_surface[i, j] = total_vega
    
    # Plot surfaces
    fig = plt.figure(figsize=(15, 12))
    
    # Price surface
    ax1 = fig.add_subplot(221, projection='3d')
    X, Y = np.meshgrid(tenor_range, maturity_range)
    surf1 = ax1.plot_surface(X, Y, price_surface, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('Tenor (years)')
    ax1.set_ylabel('Maturity (years)')
    ax1.set_zlabel('Price')
    ax1.set_title('Swaption Price Surface')
    fig.colorbar(surf1, ax=ax1, shrink=0.5)
    
    # Alpha sensitivity
    ax2 = fig.add_subplot(222, projection='3d')
    surf2 = ax2.plot_surface(X, Y, alpha_sensi_surface, cmap='plasma', alpha=0.8)
    ax2.set_xlabel('Tenor (years)')
    ax2.set_ylabel('Maturity (years)')
    ax2.set_zlabel('Alpha Sensitivity')
    ax2.set_title('Alpha Sensitivity Surface')
    fig.colorbar(surf2, ax=ax2, shrink=0.5)
    
    # Vega sensitivity
    ax3 = fig.add_subplot(223, projection='3d')
    surf3 = ax3.plot_surface(X, Y, vega_sensi_surface, cmap='coolwarm', alpha=0.8)
    ax3.set_xlabel('Tenor (years)')
    ax3.set_ylabel('Maturity (years)')
    ax3.set_zlabel('Total Vega')
    ax3.set_title('Total Vega Surface')
    fig.colorbar(surf3, ax=ax3, shrink=0.5)
    
    # Contour plot of vega
    ax4 = fig.add_subplot(224)
    contour = ax4.contourf(X, Y, vega_sensi_surface, levels=20, cmap='coolwarm')
    ax4.set_xlabel('Tenor (years)')
    ax4.set_ylabel('Maturity (years)')
    ax4.set_title('Total Vega Contour Plot')
    fig.colorbar(contour, ax=ax4)
    
    plt.tight_layout()
    plt.show()


def example_hedging_analysis():
    """Analyze hedging strategies for different market scenarios."""
    print("\n\nExample 5: Hedging Strategy Analysis")
    print("-" * 40)
    
    # Set up model
    lrw_model = setup_standard_model()
    
    # Swaption parameters
    maturity = 2.0
    tenor = 5.0
    
    # Test different strikes around ATM
    # Swaption parameters
    # maturity = 1.0 #2.0
    # tenor = 2.0# 5.0
    delta = 0.5
    delta_float = 0.5
    delta_fixed = 0.5 #1.0
    # Set at ATM
 
    swaption_config = SwaptionConfig(maturity=maturity,tenor=tenor,
                                   strike=0.05, 
                                   delta_float = delta_float,
                                   delta_fixed= delta_fixed)
    # lrw_model.set_swaption_config(swaption_config)
    atm_strike = lrw_model.compute_swap_rate()
    swaption_config.strike = atm_strike   
    lrw_model.set_swaption_config(swaption_config)

    # lrw_model.SetOptionProperties(tenor, maturity, 0.5, 0.5, 0.0)
    # atm_strike = lrw_model.ComputeSwapRate()
    
    strike_multipliers = [0.9, 0.95, 1.0, 1.05, 1.1]
    
    hedging_results = []
    
    for mult in strike_multipliers:
        strike = atm_strike * mult
        swaption_config.strike = atm_strike   
        lrw_model.set_swaption_config(swaption_config)

        # lrw_model.SetOptionProperties(tenor, maturity, 0.5, 0.5, strike)
        
        # Get hedging strategies
        analyzer = LRWSensitivityAnalyzer(lrw_model)
        
        # ZC hedging
        zc_hedge = analyzer.compute_hedging_portfolio(hedge_type="zc")
        
        # Swap hedging
        swap_hedge = analyzer.compute_hedging_portfolio(hedge_type="swap")
        fixed_delta=0.0
        float_delta=0.0
        for key, value in swap_hedge.items():
        
            if ("SWAP" in key and "FLOATINGLEG" in key):
                float_delta = value
            elif ("SWAP" in key and "FIXEDLEG" in key):
                fixed_delta = value

        # print("="*60)        
        # print(zc_hedge)

        # print("="*60)
        # print(swap_hedge)

        # Price for reference
        pricer = LRWSwaptionPricer(lrw_model)
        price = pricer.price_swaption(method="fft")
        
        hedging_results.append({
            'Strike': strike,
            'Moneyness': mult,
            'Price': price,
            # 'ZC_Hedge_Count':5,## len([k for k in zc_hedge.keys() if 'amount' in k]),
            'Swap_Fixed_Hedge': fixed_delta,##swap_hedge.get('Swap_Hedge_Fixed_amount', 0),
            'Swap_Float_Hedge': float_delta ## swap_hedge.get('Swap_Hedge_Floating_amount', 0)
        })
    
    # Display results
    hedge_df = pd.DataFrame(hedging_results)
    print("\nHedging Strategies by Strike:")
    print(hedge_df.to_string(index=False, float_format='%.6f'))
    
    # Visualize
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(hedge_df['Moneyness'], hedge_df['Swap_Fixed_Hedge'], 'b-o', label='Fixed Leg', linewidth=2)
    plt.plot(hedge_df['Moneyness'], hedge_df['Swap_Float_Hedge'], 'r-o', label='Floating Leg', linewidth=2)
    plt.xlabel('Moneyness')
    plt.ylabel('Hedge Amount')
    plt.title('Swap Hedging by Strike')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.subplot(1, 2, 2)
    plt.plot(hedge_df['Moneyness'], hedge_df['Price'], 'g-o', linewidth=2)
    plt.xlabel('Moneyness')
    plt.ylabel('Option Price')
    plt.title('Swaption Price by Strike')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def example_monte_carlo_comparison():
    """Compare Monte Carlo schemas and sensitivities."""
    print("\n\nExample 6: Monte Carlo Schema Comparison")
    print("-" * 40)
    
    # Set up model
    lrw_model = setup_standard_model()
    
    # Swaption at ATM
    maturity = 1.0
    tenor = 2.0
    lrw_model.SetOptionProperties(tenor, maturity, 0.5, 0.5, 0.0)
    atm_strike = lrw_model.ComputeSwapRate()
    lrw_model.SetOptionProperties(tenor, maturity, 0.5, 0.5, atm_strike)
    
    print(f"\nSwaption: {maturity}Y x {tenor}Y at {atm_strike:.4%}")
    
    # Initialize pricer
    pricer = LRWSwaptionPricer(lrw_model)
    
    # FFT reference price
    fft_price = pricer.price_swaption(method="fft")
    print(f"\nFFT Reference Price: {fft_price:.6f}")
    
    # Compare MC schemas
    num_paths = 30000
    schemas = ["EULER_CORRECTED", "EULER_FLOORED"]
    
    print(f"\nMonte Carlo Comparison (paths={num_paths}):")
    print("-" * 50)
    
    mc_results = pricer.price_with_schemas(
        num_paths=num_paths,
        dt=0.125,
        schemas=schemas
    )
    
    for schema, price in mc_results.items():
        if isinstance(price, float):
            error = abs(price - fft_price)
            rel_error = error / fft_price * 100
            print(f"{schema:20} Price: {price:.6f}  Error: {error:.6f} ({rel_error:.2f}%)")
        else:
            print(f"{schema:20} {price}")


if __name__ == "__main__":
    # Run all examples
    #This is ok
    # example_delta_hedging()

    ##This is ok
    # example_vega_analysis()
    
    ##This is ok
    example_full_sensitivity_report()

    # example_parameter_sensitivity_surface()
    # example_hedging_analysis()
    # example_monte_carlo_comparison()
