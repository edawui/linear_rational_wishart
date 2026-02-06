"""
Basic examples for Linear Rational Wishart models.

This module demonstrates basic usage of LRW models for interest rate
modeling and swaption pricing.
"""

# from Jax_1.analytic.alpha_curve_jax import compute_adjustment_error
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import matplotlib
import time
import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# from ..models.interest_rate.config import SwaptionConfig, LRWModelConfig
# from ..models.interest_rate.lrw_model import LRWModel
# from ..pricing.swaption_pricer import LRWSwaptionPricer
# from ..utils.reporting import print_pretty

# from ..pricing.mc_pricer import WishartMonteCarloPricer

from linear_rational_wishart.config.constants import Constants_todo
# from linear_rational_wishart.config import constants
from linear_rational_wishart.models.interest_rate.config import SwaptionConfig, LRWModelConfig
from linear_rational_wishart.models.interest_rate.lrw_model import LRWModel
from linear_rational_wishart.pricing.swaption_pricer import LRWSwaptionPricer
from linear_rational_wishart.utils.reporting import print_pretty
from linear_rational_wishart.pricing.mc_pricer import WishartMonteCarloPricer


matplotlib.use('TkAgg')  # or 'Qt5Agg'

# Enable interactive mode globally
plt.ion()

def example_basic_lrw_setup():
    """Basic LRW model setup and bond pricing."""
    print("Example 1: Basic LRW Model Setup")
    print("-" * 40)
    
    # Model parameters
    n = 2
    alpha = 0.05
    x0 = jnp.array([[0.02, 0.005], [0.005, 0.015]])
    omega = jnp.array([[0.06, 0.001], [0.001, 0.07]])
    m = jnp.array([[-0.29, 0.1], [0.2, -0.5]])
    sigma = jnp.array([[0.03, 0.1], [0.1, 0.1]])
    
    # Initialize model
    lrw_model_config = LRWModelConfig( n=n,  alpha=alpha,  x0=x0,  omega=omega,  m=m, sigma=sigma    )
    swaption_config = SwaptionConfig(maturity=5,tenor=5,  strike=0.05, delta_float = 0.5,delta_fixed= 1.0)
    lrw_model = LRWModel(lrw_model_config,swaption_config)
    
    # Set  matrices
    u1 = jnp.array([[1, 0], [0, 0]])
    u2 = jnp.array([[0, 0], [0, 1]])   
    lrw_model.set_weight_matrices(u1,u2)
    
    # Compute bond prices
    maturities = jnp.array([0.5, 1.0, 2.0, 5.0, 10.0])
    bond_prices = [lrw_model.bond(t) for t in maturities]
    
    print("\nZero Coupon Bond Prices:")
    for t, price in zip(maturities, bond_prices):
        print(f"  T={t:4.1f}: {price:.6f}")
    
     # Compute spread
    maturities = jnp.array([0.5, 1.0, 2.0, 5.0, 10.0])
    spreads = [lrw_model.spread(t) for t in maturities]
    
    print("\nSpread:")
    for t, spread in zip(maturities, spreads):
        print(f"  T={t:4.1f}: {spread:.6f}")

    # Compute short rate
    r0 = lrw_model.get_short_rate()
    r_inf = lrw_model.get_short_rate_infinity()
    print(f"\nShort rate r(0): {r0:.4%}")
    print(f"Long-term rate r(∞): {r_inf:.4%}")
    
    # Compute spread
    spread0 = lrw_model.get_spread()
    spread_inf = lrw_model.get_spread_infinity()
    print(f"\nSpread s(0): {spread0:.4%}")
    print(f"Long-term spread s(∞): {spread_inf:.4%}")
    
    return lrw_model


def example_swaption_pricing():
    """Swaption pricing with different methods."""
    print("\n\nExample 2: Swaption Pricing")
    print("-" * 40)
    
    # Set up model
    n = 2
    alpha = 0.05
    x0 = jnp.array([[0.12, -0.01], [-0.01, 0.005]])
    x0 = jnp.array([[0.0012, -0.01], [-0.01, 0.0005]])
    omega = jnp.array([[0.10, 0.002], [0.002, 0.0005]])
    omega = jnp.array([[0.15, 0.0], [0.0, 0.1]])
    omega = jnp.array([[0.0010, 0.0005], [0.0005, 0.001]])
    # m = jnp.array([[-0.4, 0.01], [0.02, -0.2]])
    m = jnp.array([[-0.04, 0.0], [0.0, -0.02]])
    sigma = jnp.array([[0.05, 0.02], [0.02, 0.047]])
    
    # Swaption parameters
    maturity = 1.0
    tenor = 2.0
    delta_float = 0.5
    delta_fixed = 0.5 #1.0
    # tenor = 5.0
    
     
    # Initialize model
    lrw_model_config = LRWModelConfig( n=n,  alpha=alpha,  x0=x0,  omega=omega,  m=m, sigma=sigma ,is_bru_config=True   )
    swaption_config = SwaptionConfig(maturity=maturity,tenor=tenor,
                                   strike=0.05, 
                                   delta_float = delta_float,
                                   delta_fixed= delta_fixed)
    lrw_model = LRWModel(lrw_model_config,swaption_config)
    
    # Set  matrices
    u1 = jnp.array([[1, 0], [0, 0]])
    u2 = jnp.array([[0, 0], [0, 1]])   
    lrw_model.set_weight_matrices(u1,u2)
    
   # Compute ATM strike
    atm_strike = lrw_model.compute_swap_rate()
    print(f"\nATM Strike: {atm_strike:.4%}")
    
    # Set ATM strike
    swaption_config.strike = atm_strike
    strike_lists=[atm_strike+k/1000 for k in range(-10,18)]
    # print(f"\nStrike List: {strike_lists}")
    # strike_lists=[atm_strike]
    # n_prices=  len(strike_lists
    fft_prices= [] #jnp.zeros(n_prices)
    fft_ivs= []

    fft_fast_prices= [] #jnp.zeros(n_prices)
    fft_fast_ivs= []

    mc_prices= []
    mc_ivs= []

    cd_prices= []
    cd_ivs= []

    # strike_lists=[atm_strike]
    nb_run=1#3
    run_fft_fast=False# True#True
    run_fft_nn=True#True
    run_fft=True
    run_mc=True #False
    run_cd=True#False

    # def set_constant_FAST_SWAPTION_PRICING(value):
    #     import sys

    #     for name, module in sys.modules.items():
    #         if name.endswith("constants"):
    #             if hasattr(module, "FAST_SWAPTION_PRICING"):
    #                 module.FAST_SWAPTION_PRICING = value

    for strike in strike_lists:

        swaption_config.strike = strike
        lrw_model.set_swaption_config(swaption_config)

        # lrw_model.SetOptionProperties(tenor, maturity, delta_float, delta_fixed, atm_strike)
        print(f"Updated Swaption Strike: {lrw_model.swaption_config.strike:.4%}")
        # Initialize pricer
        pricer = LRWSwaptionPricer(lrw_model)
    
        # Price with different methods
        print("\nSwaption Prices:")
        if run_fft:
            time_0=time.time()
            for i in range(nb_run):
                # # FFT pricing
                
                # try:
                fft_price, fft_iv = pricer.price_swaption(method="fft", return_implied_vol=True)
            time_1=time.time()
            print(f"  FFT:              Strike={strike:.6f}, Price={fft_price:.6f}, IV={fft_iv:.2%}, average compute time={(time_1-time_0)/nb_run:2f}")
            fft_prices.append(float(fft_price))
            fft_ivs.append(float(fft_iv))

        if run_fft_fast:

            time_0=time.time()
            for i in range(nb_run):
                # # FFT pricing
                # try:
                Constants_todo.FAST_SWAPTION_PRICING= True
                # set_constant_FAST_SWAPTION_PRICING(True)
                fft_fast_price, fft_fast_iv = pricer.price_swaption(method="fft", return_implied_vol=True)
            time_1=time.time()
            print(f"  FFT FAST:         Strike={strike:.6f}, Price={fft_fast_price:.6f}, IV={fft_fast_iv:.2%}, average compute time={(time_1-time_0)/nb_run:2f}")
            fft_fast_prices.append(float(fft_fast_price))
            fft_fast_ivs.append(float(fft_fast_iv))
            Constants_todo.FAST_SWAPTION_PRICING= False
            # set_constant_FAST_SWAPTION_PRICING(False)

        if run_fft_nn:
            time_0=time.time()
            for i in range(nb_run):
                # # FFT pricing
                # time_0=time.time()
                # try:
                fft_price, fft_iv = pricer.price_swaption(method="neural_network", return_implied_vol=True)
            time_1=time.time()
            print(f"  FFT NN:           Strike={strike:.6f}, Price={fft_price:.6f}, IV={fft_iv:.2%}, average compute time={(time_1-time_0)/nb_run:2f}")
            fft_prices.append(float(fft_price))
            fft_ivs.append(float(fft_iv))
        # except:
        #     print("  FFT:  Not available")

        # # # Monte Carlo pricing
        # mc_price = pricer.price_swaption(
        #     method="mc", num_paths=10000, dt=0.125, return_implied_vol=False
        # )
        # print(f"  Monte Carlo:      Price={mc_price:.6f}")

        if run_mc:
            time_0=time.time()

            for i in range(nb_run):
                # time_0=time.time()    
                try:
                    mc_price, mc_iv = pricer.price_swaption(
                        method="mc", num_paths=50000, dt=0.125, return_implied_vol=True
                    )
                    # # # Alternative Monte Carlo pricing with schema
                    # mc_pricer = WishartMonteCarloPricer(lrw_model)
                    # mc_price2  = mc_pricer.price_option_mc_with_schema(
                    #     # nb_mc=50000, dt=0.125,schema  = "EULER_FLOORED")
                    #     nb_mc=5000, dt=0.125,schema  = "EULER_CORRECTED")
                    # print(f"  Monte Carlo:      Strike={strike:.6f}, Price1={mc_price:.6f}, Price2={mc_price2:.6f}")
                    
                   
                except:
                    print("  Monte Carlo:  Not available")
            time_1=time.time()
            print(f"  Monte Carlo:      Strike={strike:.6f}, Price={mc_price:.6f}, IV={mc_iv:.2%}, average compute time={(time_1-time_0)/nb_run:.2f}")
            mc_prices.append(float(mc_price))
            mc_ivs.append(float(mc_iv))
        if run_cd:
            # Approximations
            # cd_price = pricer.price_swaption(
            #         method="collin_dufresne", return_implied_vol=False
            #     )
            # print(f"  Collin-Dufresne:  Price={cd_price:.6f} ")

            time_0=time.time()
            for i in range(nb_run):
                # time_0=time.time()
                try:
                    cd_price, cd_iv = pricer.price_swaption(
                        method="collin_dufresne", return_implied_vol=True
                    )
                    
                except:
                    print("  Collin-Dufresne:  Not available")
            time_1=time.time()
            print(f"  Collin-Dufresne:  Strike={strike:.6f}, Price={cd_price:.6f}, IV={cd_iv:.2%}, average compute time={(time_1-time_0)/nb_run:.2f}")
            cd_prices.append(float(cd_price))
            cd_ivs.append(cd_iv)
        # try:
        #     gamma_price, gamma_iv = pricer.price_swaption(
        #         method="gamma_approx", return_implied_vol=True
        #     )
        #     time_1=time.time()
        #     print(f"  Gamma Approx:     Price={gamma_price:.6f}, IV={gamma_iv:.2%}, compute time={time_1-time_0:.4%}")
        # except:
        #     print("  Gamma Approx:     Not available")
    
    print("="*60)

    plot_fig=True
    if plot_fig:
        # Plot results
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        if run_fft:
            print(f"fft_prices :{fft_prices}")
            plt.plot(strike_lists, fft_prices, label="FFT", marker='o')
        if run_fft_fast:
            print(f"fft_fast_prices :{fft_fast_prices}")

            plt.plot(strike_lists, fft_fast_prices, label="FFT FAST", marker='*')
        if run_mc:
            print(f"mc_prices :{mc_prices}")

            plt.plot(strike_lists, mc_prices, label="Monte Carlo", marker='x')
        if run_cd:
            print(f"cd_prices :{cd_prices}")

            plt.plot(strike_lists, cd_prices, label="Collin-Dufresne", marker='s')
        plt.xlabel("Strike")
        plt.ylabel("Swaption Price")
        plt.title("Swaption Prices vs Strike")
        # plt.legend()
        # plt.grid(True)
        # plt.draw() #plt.show(block=False)

        
    
        plt.subplot(1, 2, 2)
        if run_fft:
            print(f"fft_ivs :{fft_ivs}")
            plt.plot(strike_lists, fft_ivs, label="FFT", marker='o')
        if run_fft_fast:
            print(f"fft_fast_ivs :{fft_fast_ivs}")
            plt.plot(strike_lists, fft_fast_ivs, label="FFT FAST", marker='*')
        if run_mc:
            print(f"mc_ivs :{mc_ivs}")
            plt.plot(strike_lists,  mc_ivs, label="Monte Carlo", marker='x')
        if run_cd:
            print(f"cd_ivs :{cd_ivs}")
            plt.plot(strike_lists,  cd_ivs, label="Collin-Dufresne", marker='s')
        plt.xlabel("Strike")
        plt.ylabel("Swaption implied vol")
        plt.title("Swaption implied vol vs Strike")

        plt.legend()
         
        plt.grid(True, alpha=0.3)
    
        plt.tight_layout()
        plt.draw() #plt.show(block=False)

    return lrw_model, pricer


def example_term_structure():
    """Plot term structure of interest rates."""
    print("\n\nExample 3: Term Structure")
    print("-" * 40)
    
    # Parameters for upward sloping curve
    n = 2
    alpha = 0.052
    x0 = jnp.array([[0.04, 0.02], [0.02, 0.03]])
    m = jnp.array([[-0.15, 0.06], [0.07, -0.12]])
    sigma = jnp.array([[0.04, 0.015], [0.015, 0.037]])
    omega = 4.0 * sigma @ sigma
    
    # # Initialize model
    # lrw_model = LRWModel(n, alpha, x0, omega, m, sigma)
    
    # u1 = jnp.array([[1, 0], [0, 1]])
    # u2 = jnp.array([[0, 0], [0, 0]])
    # lrw_model.SetU1(u1)
    # lrw_model.SetU2(u2)
    
    # Swaption parameters
    maturity = 1.0
    tenor = 2.0
    delta_float = 0.5
    delta_fixed = 0.5 #1.0
    
     
    # Initialize model
    lrw_model_config = LRWModelConfig( n=n,  alpha=alpha,  x0=x0,  omega=omega,  m=m, sigma=sigma ,is_bru_config=True   )
    swaption_config = SwaptionConfig(maturity=maturity,tenor=tenor,
                                   strike=0.05, 
                                   delta_float = delta_float,
                                   delta_fixed= delta_fixed)
    lrw_model = LRWModel(lrw_model_config,swaption_config)
    
    # Set  matrices
    u1 = jnp.array([[1, 0], [0, 0]])
    u2 = jnp.array([[0, 0], [0, 1]])   
    lrw_model.set_weight_matrices(u1,u2)
    
    # Compute term structure
    maturities = jnp.linspace(0.1, 30, 100)
    bond_prices = jnp.array([lrw_model.bond(t) for t in maturities])
    yields = -jnp.log(bond_prices) / maturities
    
    # Compute forward rates
    dt = 0.1
    forward_maturities = maturities[:-1]
    forward_rates = []
    for i in range(len(maturities) - 1):
        p1 = lrw_model.bond(maturities[i])
        p2 = lrw_model.bond(maturities[i+1])
        forward_rate = -jnp.log(p2/p1) / (maturities[i+1] - maturities[i])
        forward_rates.append(forward_rate)
    
    # Plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(maturities, yields * 100, 'b-', linewidth=2)
    plt.xlabel('Maturity (years)')
    plt.ylabel('Yield (%)')
    plt.title('Zero Coupon Yield Curve')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(forward_maturities, np.array(forward_rates) * 100, 'r-', linewidth=2)
    plt.xlabel('Maturity (years)')
    plt.ylabel('Forward Rate (%)')
    plt.title('Instantaneous Forward Curve')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print key rates
    key_maturities = [0.5, 1, 2, 5, 10, 20, 30]
    print("\nKey Rates:")
    print("Maturity | Yield | Forward Rate")
    print("-" * 35)
    for t in key_maturities:
        bond_price = lrw_model.bond(t)
        yield_rate = -np.log(bond_price) / t
        
        # Approximate forward rate
        dt = 0.01
        p1 = lrw_model.bond(t)
        p2 = lrw_model.bond(t + dt)
        forward = -np.log(p2/p1) / dt
        
        print(f"{t:8.1f} | {yield_rate:5.2%} | {forward:5.2%}")


def example_parameter_comparison():
    """Compare different parameter sets."""
    print("\n\nExample 4: Parameter Set Comparison")
    print("-" * 40)
    
    # Define parameter sets
    param_sets = {
        "Low Vol": {
            "sigma": jnp.array([[0.02, 0.01], [0.01, 0.025]]),
            "m": jnp.array([[-0.1, 0.05], [0.05, -0.15]])
        },
        "High Vol": {
            "sigma": jnp.array([[0.06, 0.02], [0.02, 0.05]]),
            "m": jnp.array([[-0.3, 0.1], [0.1, -0.4]])
        },
        "Asymmetric": {
            "sigma": jnp.array([[0.08, 0.0], [0.0, 0.02]]),
            "m": jnp.array([[-0.5, 0.0], [0.0, -0.1]])
        }
    }
    
    # Common parameters
    n = 2
    alpha = 0.05
    x0 = jnp.array([[0.03, 0.01], [0.01, 0.02]])
    
    # Compare swaption prices
    maturity = 1.0
    tenor = 5.0
    delta_float = 0.5
    delta_fixed = 0.5 #1.0
    
    results = {}
    
    for name, params in param_sets.items():
        sigma = params["sigma"]
        m = params["m"]
        omega = 4.0 * sigma @ sigma
        
        # Initialize model
        lrw_model_config = LRWModelConfig( n=n,  alpha=alpha,  x0=x0,  omega=omega,  m=m, sigma=sigma ,is_bru_config=True   )
        swaption_config = SwaptionConfig(maturity=maturity,tenor=tenor,
                                       strike=0.05, 
                                       delta_float = delta_float,
                                       delta_fixed= delta_fixed)
        model = LRWModel(lrw_model_config,swaption_config)
    
        # Set  matrices
        u1 = jnp.array([[1, 0], [0, 0]])
        u2 = jnp.array([[0, 0], [0, 1]])   
        model.set_weight_matrices(u1,u2)
   
        
        # Check Gindikin condition
        gindikin_ok = model.wishart.check_gindikin()
        # def check_gindikin(self) -> None:
    
        # Set swaption
        # atm = model.ComputeSwapRate()
        atm_strike = model.compute_swap_rate()
        model.swaption_config.strike = atm_strike
        
        # Price        
        pricer = LRWSwaptionPricer(model)
        price, iv = pricer.price_swaption(method="fft", return_implied_vol=True)
        
        # Get term structure info
        r0 = model.get_short_rate()
        r_inf = model.get_short_rate_infinity()
        results[name] = {
            "Gindikin OK": gindikin_ok,
            "r(0)": r0,
            "r(∞)": r_inf,
            "ATM Strike": atm_strike,
            "Price": price,
            "Implied Vol": iv
        }
    
    # Display results
    print(f"\nSwaption: {maturity}Y x {tenor}Y")
    print("\nResults by Parameter Set:")
    for name, res in results.items():
        print(f"\n{name}:")
        for key, val in res.items():
            if key == "Price":
                print(f"  {key}: {val:.6f}")
            elif key == "Gindikin OK":
                print(f"  {key}: {val}")
            else:
                print(f"  {key}: {val:.2%}")


def example_wishart_properties():
    """Examine Wishart process properties."""
    print("\n\nExample 5: Wishart Process Properties")
    print("-" * 40)
    
    # Model setup
    n = 2
    alpha = 0.05
    x0 = jnp.array([[0.02, 0.005], [0.005, 0.015]])
    omega = jnp.array([[0.06, 0.001], [0.001, 0.07]])
    m = jnp.array([[-0.29, 0.1], [0.2, -0.5]])
    sigma = jnp.array([[0.03, 0.1], [0.1, 0.1]])
    
    # Compare swaption prices
    maturity = 1.0
    tenor = 5.0
    delta_float = 0.5
    delta_fixed = 0.5 #1.0

    # lrw_model = LRWModel(n, alpha, x0, omega, m, sigma)
    lrw_model_config = LRWModelConfig( n=n,  alpha=alpha,  x0=x0,  omega=omega,  m=m, sigma=sigma ,is_bru_config=True   )
    swaption_config = SwaptionConfig(maturity=maturity,tenor=tenor,
                                       strike=0.05, 
                                       delta_float = delta_float,
                                       delta_fixed= delta_fixed)
    lrw_model = LRWModel(lrw_model_config,swaption_config)
    
    
        # Set  matrices
    u1 = jnp.array([[1, 0], [0, 0]])
    u2 = jnp.array([[0, 0], [0, 1]])   
    lrw_model.set_weight_matrices(u1,u2)
    # Check Gindikin condition
    print(f"Gindikin condition satisfied: {lrw_model.wishart.check_gindikin()}")
    
    # Compute mean at different times
    times = [0.5, 1.0, 2.0, 5.0]
    u = jnp.eye(2)
    
    print("\nWishart Process Mean E[X(t)]:")
    for t in times:
        mean_xt = lrw_model.wishart.compute_mean(t, u)
        print(f"\nt = {t}:")
        print(mean_xt)
    
    # Compute moments
    t = 1.0
    lrw_model.wishart.compute_moments(t)
    
    print(f"\nMoments at t = {t}:")
    print(f"E[X_11]: {lrw_model.wishart.get_moments(1, 1, 0)}")
    print(f"E[X_12]: {lrw_model.wishart.get_moments(1, 2, 0)}")
    print(f"E[X_22]: {lrw_model.wishart.get_moments(2, 2, 0)}")
    
    # Long-term behavior
    print("\nLong-term behavior:")
    mean_inf = omega @ jnp.linalg.inv(-m)
    print("E[X(∞)] =")
    print(mean_inf)

    
def example_swap_exposure():
    """Swaption pricing with different methods."""
    print("\n\nExample 2: Swap Exposure")
    print("-" * 40)
    
    # Set up model
    n = 2
    alpha = 0.05
    x0 = jnp.array([[0.12, -0.01], [-0.01, 0.005]])
    omega = jnp.array([[0.10, 0.002], [0.002, 0.0005]])
    m = jnp.array([[-0.4, 0.01], [0.02, -0.2]])
    sigma = jnp.array([[0.05, 0.02], [0.02, 0.047]])
    
    # Swaption parameters
    maturity = 0.0#2.0
    tenor = 3.0
    delta_float = 0.5
    delta_fixed = 0.5 #1.0
    final_maturity =tenor+maturity
     
    # Initialize model
    lrw_model_config = LRWModelConfig( n=n,  alpha=alpha,  x0=x0,  omega=omega,  m=m, sigma=sigma ,is_bru_config=True   )

    swaption_config = SwaptionConfig(maturity=maturity,tenor=tenor,
                                   strike=0.05, 
                                   delta_float = delta_float,
                                   delta_fixed= delta_fixed)
    lrw_model = LRWModel(lrw_model_config,swaption_config)
    # lrw_model.is_spread=False
    # lrw_model.set_swaption_config(swaption_config)
    # swaption_config = SwaptionConfig(maturity=maturity,tenor=tenor,
    #                                strike=swap_rate, 
    #                                delta_float = delta_float,
    #                                delta_fixed= delta_fixed)

    # Set  matrices
    u1 = jnp.array([[1, 0], [0, 0]])
    u2 = None# jnp.array([[0, 0], [0, 1]])   
    lrw_model.set_weight_matrices(u1,u2)
    swap_rate= lrw_model.compute_swap_rate()
    
    swaption_config.strike=swap_rate
    lrw_model.set_swaption_config(swaption_config)


    mc_pricer = WishartMonteCarloPricer(lrw_model)
    exposure_observation_freq=0.125#/5.0 #1.0/52
    exposure_dates = list(np.arange(0.0, final_maturity+exposure_observation_freq, exposure_observation_freq))
    exposure_dates = [0.0]
    print(f"Exposure observation dates: {exposure_dates}")
    fixed_rate=swap_rate #0.05
    spread=0.0#-0.001
    
    # floating_schedule_trade = list(np.arange(maturity, final_maturity, delta_float)) # 
    # fixed_schedule_trade    = list(np.arange(maturity, final_maturity, delta_fixed)) # 

    floating_schedule_trade = list(np.arange(maturity, final_maturity+delta_float, delta_float)) # 
    fixed_schedule_trade    = list(np.arange(maturity, final_maturity+delta_fixed, delta_fixed)) # 


    print(f"Floating schedule: {floating_schedule_trade}")
    print(f"Fixed schedule: {fixed_schedule_trade}")
    print(f"swap_rate: {swap_rate}, Fixed rate: {fixed_rate}, Spread: {spread}")
    # floating_schedule_trade=[2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0] ##[0.5*i for i in range(9)]
    # fixed_schedule_trade=[2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0] ## [1.0*i for i in range(5)]
    # print(f"Floating schedule: {floating_schedule_trade}")
    # print(f"Fixed schedule: {fixed_schedule_trade}")

    spread=0.0
    nb_mc=15000#00#100#
    dt=exposure_observation_freq
    # dt=0.125/5.0

    # exposureDates = np.arange(0, 6.25, 0.1)

    exposure_profile = mc_pricer.compute_swap_exposure_profile( 
                exposure_dates, 
                fixed_rate,
                spread,
                floating_schedule_trade,
                fixed_schedule_trade,
                nb_mc ,
                dt,
                main_date=0.0,
                # schema = "EULER_FLOORED"
                schema = "EULER_CORRECTED"
            ) 
    mean_profile = np.mean(exposure_profile, axis=1)
    pfe =  np.percentile(exposure_profile, 95, axis=1)

    initial_swap_price= [lrw_model.price_swap()]*len(exposure_dates)

    print(f"Initial swap price: {initial_swap_price}")
     
    print(f"mean_profile: {len(mean_profile)}, {mean_profile}")

    # Plot meanProfile and PFE(95%) vs exposureDates
    plt.figure(figsize=(10, 6))
    plt.plot(exposure_dates, mean_profile, label='Mean Exposure')#, marker='o')
    # plt.plot(exposure_dates, pfe, label='PFE (95%)')#, marker='x')
    plt.plot(exposure_dates, initial_swap_price, label='initial_swap_price')#, marker='x')
    plt.xlabel('Exposure Date')
    plt.ylabel('Exposure')
    plt.title('Exposure Profile: Mean and PFE(95%)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # print(f"Dates: {exposureDates}")
    # print(f"Average Exposure: {meanProfile}")
    # print(f"PFE(95%): {pfe}")
    for i in range(len(exposure_dates)):
        cProfile = exposure_profile[i,:]
        print(f"{exposure_dates[i]}: Average Exposure: {np.mean(cProfile)}, PFE(95%): {np.percentile(cProfile, 95)}") 
  
    
def example_swaption_exposure():
    """Swaption pricing with different methods."""
    print("\n\nExample 2: Swaption Exposure")
    print("-" * 40)
    
    # Set up model
    n = 2
    alpha = 0.05
    # x0 = jnp.array([[0.12, -0.01], [-0.01, 0.005]])
    x0 = jnp.array([[0.14, -0.01], [-0.01, 0.07]])
    omega = jnp.array([[0.10, 0.002], [0.002, 0.0005]])
    m = jnp.array([[-0.4, 0.01], [0.02, -0.2]])
    sigma = jnp.array([[0.05, 0.02], [0.02, 0.047]])
    u1 = jnp.array([[1, 0], [0, 0]])
    u2 = None# jnp.array([[0, 0], [0, 1]])  
    # Swaption parameters
    maturity = 2.0#0.0#2.0
    tenor = 3.0
    delta_float = 0.5
    delta_fixed = 0.5 #1.0
    final_maturity =tenor+maturity
     
    # Initialize model
    lrw_model_config = LRWModelConfig( n=n,  alpha=alpha,  x0=x0,  omega=omega,  m=m, sigma=sigma ,is_bru_config=True
                                       , has_jump=False,  u1=u1,  u2=u2, is_spread=False)

    swaption_config = SwaptionConfig(maturity=maturity,tenor=tenor,
                                   strike=0.05, 
                                   delta_float = delta_float,
                                   delta_fixed= delta_fixed)
    lrw_model = LRWModel(lrw_model_config,swaption_config)
    lrw_model.is_spread=False
    lrw_model.set_swaption_config(swaption_config)
    # swaption_config = SwaptionConfig(maturity=maturity,tenor=tenor,
    #                                strike=swap_rate, 
    #                                delta_float = delta_float,
    #                                delta_fixed= delta_fixed)

    # Set  matrices
    
    lrw_model.set_weight_matrices(u1,u2)
    swap_rate= lrw_model.compute_swap_rate()
    
    swaption_config.strike=swap_rate
    lrw_model.set_swaption_config(swaption_config)


    mc_pricer = WishartMonteCarloPricer(lrw_model)
    exposure_observation_freq=0.5#1.0/12 # 0.5#0.1#50 #1.0/52
    # exposure_dates = list(np.arange(0.0, final_maturity+exposure_observation_freq, exposure_observation_freq))
    exposure_dates = [0.0, 0.5]
    exposure_dates = list(np.arange(0.0, maturity, exposure_observation_freq))
    print(f"Exposure observation dates: {exposure_dates}")
    fixed_rate=swap_rate #0.05
    # spread=0.0#-0.001
    
    # # floating_schedule_trade = list(np.arange(maturity, final_maturity, delta_float)) # 
    # # fixed_schedule_trade    = list(np.arange(maturity, final_maturity, delta_fixed)) # 

    # floating_schedule_trade = list(np.arange(maturity, final_maturity+delta_float, delta_float)) # 
    # fixed_schedule_trade    = list(np.arange(maturity, final_maturity+delta_fixed, delta_fixed)) # 


    # print(f"Floating schedule: {floating_schedule_trade}")
    # print(f"Fixed schedule: {fixed_schedule_trade}")
    # print(f"swap_rate: {swap_rate}, Fixed rate: {fixed_rate}, Spread: {spread}")
    # # floating_schedule_trade=[2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0] ##[0.5*i for i in range(9)]
    # # fixed_schedule_trade=[2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0] ## [1.0*i for i in range(5)]
    # # print(f"Floating schedule: {floating_schedule_trade}")
    # # print(f"Fixed schedule: {fixed_schedule_trade}")

    # spread=0.0
    nb_mc=10#100#100#250#100#15000#00#100#
    dt=exposure_observation_freq
    # dt=0.125/5.0

    # # exposureDates = np.arange(0, 6.25, 0.1)

    # exposure_profile = mc_pricer.compute_swap_exposure_profile( 
    #             exposure_dates, 
    #             fixed_rate,
    #             spread,
    #             floating_schedule_trade,
    #             fixed_schedule_trade,
    #             nb_mc ,
    #             dt,
    #             main_date=0.0,
    #             # schema = "EULER_FLOORED"
    #             schema = "EULER_CORRECTED"
    #         ) 
    
    exposure_profile=[]
    initial_swap_price=[]
    initial_model_price=[]

    mean_profile = []#np.mean(exposure_profile, axis=1)
    pfe = []#np.percentile(exposure_profile, 95, axis=1)
    swaption_pricer = LRWSwaptionPricer(lrw_model)
    # for i, valuation_date in enumerate(exposure_dates[1:], start=1):
    #     print(f"\n--- Valuation Date {i}: {valuation_date} ---")
    import time
    start = time.perf_counter()

    for date in exposure_dates:
        new_maturity = maturity - date
        print(f"maturity = {new_maturity}")
        swaption_config = SwaptionConfig(maturity=new_maturity,
                                         tenor=tenor,
                                       strike=swap_rate, 
                                       delta_float = delta_float,
                                       delta_fixed= delta_fixed)

        lrw_model = LRWModel(lrw_model_config,swaption_config)
        lrw_model.set_weight_matrices(u1,u2)

        # lrw_model.SetOptionProperties(ten, mat, 0.5, 0.5, 0.0)

        swaption_pricer = LRWSwaptionPricer(lrw_model)

        price_value_t=0.0
        price_value_t= swaption_pricer.price_swaption(
                                        method = "fft",
                                        # method = "collin_dufresne",
                                        # method = "gamma_approx",
                                        # method = "mc",
                                        num_paths = nb_mc,
                                        dt=dt,
                                        return_implied_vol= False
                                        )
        initial_model_price.append(price_value_t)
        # mean_profile.append(price_value_t)
    print(f"  fft Time: {time.perf_counter() - start:.2f}s")

    print(f"Initial swaption price: {initial_model_price}")
    
    compute_exposure=True
    if compute_exposure:
        swaption_config = SwaptionConfig(maturity=maturity,
                                           tenor=tenor,
                                           strike=swap_rate, 
                                           delta_float = delta_float,
                                           delta_fixed= delta_fixed)
    
        lrw_model = LRWModel(lrw_model_config,swaption_config)
        lrw_model.set_weight_matrices(u1,u2)
        swaption_pricer = LRWSwaptionPricer(lrw_model)

        # exposure_profile = swaption_pricer.compute_swaption_exposure_profile_vectorized_fast(  
        # exposure_profile = swaption_pricer.compute_swaption_exposure_profile_vectorized_2(              
        # exposure_profile = swaption_pricer.compute_swaption_exposure_profile_vectorized(   
        #       
        exposure_profile = swaption_pricer.compute_swaption_exposure_profile(              

            exposure_dates ,
            swaption_config,
            nb_mc = nb_mc,
            dt=dt,
            main_date = 0.0,
            schema  = "EULER_FLOORED",
            pricing_method = "fft",
            # pricing_method = "collin_dufresne",
            # pricing_method = "gamma_approx",
            # pricing_method = "mc",
            batch_size  = 100#500  # Increased batch size for better performance

        #      # exposure_dates: List[float],
        # # initial_swaption_config,
        # nb_mc: int,
        # dt: float,
        # main_date: float = 0.0,
        # schema: str = "EULER_FLOORED",
        # batch_size: int = 500
            )
        # print(f"exposure_profile shape: {np.array(exposure_profile).shape}")
        # print(f"exposure_profile : {exposure_profile}")
        mean_profile = np.mean(exposure_profile, axis=1)
        pfe =  np.percentile(exposure_profile, 95, axis=1)
         
        print(f"Initial swaption price: {initial_model_price}")
     
        print(f"mean_profile: {len(mean_profile)}, {mean_profile}")

        # Plot meanProfile and PFE(95%) vs exposureDates
        plt.figure(figsize=(10, 6))
        plt.plot(exposure_dates, mean_profile, label='Mean Exposure')#, marker='o')
        plt.plot(exposure_dates, pfe, label='PFE (95%)')#, marker='x')
        plt.plot(exposure_dates, initial_model_price, label='initial_swap_price')#, marker='x')
        plt.xlabel('Exposure Date')
        plt.ylabel('Exposure')
        plt.title('Exposure Profile: Mean and PFE(95%)')
        plt.legend()
        plt.grid(True)
        plt.show()

        # print(f"Dates: {exposureDates}")
        # print(f"Average Exposure: {meanProfile}")
        # print(f"PFE(95%): {pfe}")
        if len(exposure_profile)>0:
            for i in range(len(exposure_dates)):
                cProfile = exposure_profile[i,:]
                print(f"{exposure_dates[i]}: Average Exposure: {np.mean(cProfile)}, PFE(95%): {np.percentile(cProfile, 95)}") 

  

def print_memory_info():
    """Print comprehensive memory information."""
    # Get memory info
    mem = psutil.virtual_memory()
    
    print("="*50)
    print("MEMORY INFORMATION")
    print("="*50)
    print(f"Total RAM:      {mem.total / (1024**3):6.2f} GB")
    print(f"Available RAM:  {mem.available / (1024**3):6.2f} GB")
    print(f"Used RAM:       {mem.used / (1024**3):6.2f} GB")
    print(f"Free RAM:       {mem.free / (1024**3):6.2f} GB")
    print(f"Usage:          {mem.percent:6.2f} %")
    print("="*50)
    
    # Warning if low memory
    if mem.percent > 80:
        print("⚠️  WARNING: Memory usage > 80%!")
    if mem.available < 2 * (1024**3):  # Less than 2GB available
        print("⚠️  WARNING: Less than 2GB available!")
    
    return mem

# Call at the start of your script
print_memory_info()

if __name__ == "__main__":

    import os
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.3'  # Use only 30% of available memory
    os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

    import jax
    jax.config.update('jax_platform_name', 'cpu')
    # jax.config.update('jax_disable_jit', True)  # Disable JIT compilation
    #os.environ['JAX_DISABLE_JIT'] = '1'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.15'

     
    print("JAX JIT disabled?", jax.config.jax_disable_jit)
    print("This should print True ^^^")
    print_memory_info()

    print("Linear Rational Wishart Model Basic Examples")
    # Run all examples
    # # example_basic_lrw_setup()     #ok working
    example_swaption_pricing()    #ok working
    # example_term_structure()        #ok working
    # example_parameter_comparison() #ok working
    # example_wishart_properties() #ok working
    
    # example_swap_exposure()
    # print("=========================Swaption exposure testing======================================")
    # example_swaption_exposure()

    # Keep all plots open
    plt.ioff()  # Turn off interactive mode
    plt.show()  # This will block and keep all windows open
