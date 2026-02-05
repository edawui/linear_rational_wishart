"""
Basic examples for Linear Rational Wishart models.

This module demonstrates basic usage of LRW models for interest rate
modeling and swaption pricing.
"""
import jax
# jax.config.update("jax_enable_x64", False)  # Use float32 instead of float64

from ast import Try
import jax.numpy as jnp

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import time
import matplotlib

# try:
#     from ..models.interest_rate.config import SwaptionConfig, LRWModelConfig
#     from ..models.interest_rate.lrw_model import LRWModel
#     from ..pricing.swaption_pricer import LRWSwaptionPricer
#     from ..utils.reporting import print_pretty
#     from ..components.jump  import JumpComponent
# except ImportError:
#     sys.path.insert(0, str(Path(__file__).parent.parent.parent))

#     from linear_rational_wishart.components.jump  import JumpComponent
#     from linear_rational_wishart.models.interest_rate.config import SwaptionConfig, LRWModelConfig
#     from linear_rational_wishart.models.interest_rate.lrw_model import LRWModel
#     from linear_rational_wishart.pricing.swaption_pricer import LRWSwaptionPricer
#     from linear_rational_wishart.utils.reporting import print_pretty

from linear_rational_wishart.components.jump  import JumpComponent
from linear_rational_wishart.models.interest_rate.config import SwaptionConfig, LRWModelConfig
from linear_rational_wishart.models.interest_rate.lrw_model import LRWModel
from linear_rational_wishart.pricing.swaption_pricer import LRWSwaptionPricer
from linear_rational_wishart.utils.reporting import print_pretty

matplotlib.use('TkAgg')  # or 'Qt5Agg'

# Enable interactive mode globally
plt.ion()

def example_jump_basic_lrw_setup():
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
    lrw_model_config = LRWModelConfig( n=n,  alpha=alpha,  x0=x0,  omega=omega,  m=m, sigma=sigma ,is_bru_config=False, has_jump=True   )
    # lrw_model_config = LRWModelConfig( n=n,  alpha=alpha,  x0=x0,  omega=omega,  m=m, sigma=sigma    )
    swaption_config = SwaptionConfig(maturity=5,tenor=5,  strike=0.05, delta_float = 0.5,delta_fixed= 1.0)
    
    
    lrw_model = LRWModel(lrw_model_config,swaption_config)
    
    ##jump component
    lambda_intensity = jnp.float32(0.1)
    nu = 4
    eta = jnp.array([[0.10, 0.002], [0.002, 0.0005]], dtype=jnp.float32)
    xi = jnp.array([[0.04, 0.004], [0.004, 0.02]], dtype=jnp.float32)
    jump_component =JumpComponent(lambda_intensity, nu, eta, xi)
    lrw_model.wishart.set_jump(jump_component)
    


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


def example_jump_swaption_pricing():
    """Swaption pricing with different methods."""
    print("\n\nExample 2: Swaption Pricing")
    print("-" * 40)
    
    # Set up model
    n = 2
    alpha = 0.05
    x0 = jnp.array([[0.12, -0.01], [-0.01, 0.005]])
    omega = jnp.array([[0.10, 0.002], [0.002, 0.0005]])
    m = jnp.array([[-0.4, 0.01], [0.02, -0.2]])
    sigma = jnp.array([[0.05, 0.02], [0.02, 0.047]])
    
   

    # Swaption parameters
    maturity = 1.0
    tenor = 2.0
    delta_float = 0.5
    delta_fixed = 0.5 #1.0
    
     
    # Initialize model
    lrw_model_config = LRWModelConfig( n=n,  alpha=alpha,  x0=x0,  omega=omega,  m=m, sigma=sigma ,is_bru_config=False, has_jump=False )
    # lrw_model_config = LRWModelConfig( n=n,  alpha=alpha,  x0=x0,  omega=omega,  m=m, sigma=sigma ,is_bru_config=False, has_jump=True )
    swaption_config = SwaptionConfig(maturity=maturity,tenor=tenor,
                                   strike=0.05, 
                                   delta_float = delta_float,
                                   delta_fixed= delta_fixed)
    lrw_model = LRWModel(lrw_model_config,swaption_config)
    
    ##jump component
    lambda_intensity = jnp.float32(0.1)
    nu = 4
    eta = jnp.array([[0.10, 0.002], [0.002, 0.0005]], dtype=jnp.float32)
    xi = jnp.array([[0.04, 0.004], [0.004, 0.02]], dtype=jnp.float32)

    jump_component =JumpComponent(lambda_intensity, nu, eta, xi)
    lrw_model.wishart.set_jump(jump_component)

    
    # Set  matrices
    u1 = jnp.array([[1, 0], [0, 0]])
    u2 = jnp.array([[0, 0], [0, 1]])   
    lrw_model.set_weight_matrices(u1,u2)
    
   
    # Set option properties
    # lrw_model.SetOptionProperties(tenor, maturity, delta_float, delta_fixed, 0.0)
    
    # Compute ATM strike
    atm_strike = lrw_model.compute_swap_rate()
    print(f"\nATM Strike: {atm_strike:.4%}")
    
    # Set ATM strike
    swaption_config.strike = atm_strike
    strike_lists=[atm_strike+k/1000 for k in range(-7,7)]#10,10)]
    # print(f"\nStrike List: {strike_lists}")
    # strike_lists=[atm_strike]
    # n_prices=  len(strike_lists
    fft_prices= [] #jnp.zeros(n_prices)
    fft_ivs= []

    mc_prices= []
    mc_ivs= []

    cd_prices= []
    cd_ivs= []

    # strike_lists=[atm_strike]
    nb_run=1
    run_fft=True
    run_mc=False
    run_cd=False #True#False
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
            for i in range(nb_run):
                # # FFT pricing
                time_0=time.time()
                # try:
                fft_price, fft_iv = pricer.price_swaption(method="fft", return_implied_vol=True)
                time_1=time.time()
                print(f"  FFT:              Strike={strike:.6f}, Price={fft_price:.6f}, IV={fft_iv:.2%}, compute time={time_1-time_0:2f}")
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
            for i in range(nb_run):
                time_0=time.time()    
                try:
                    mc_price, mc_iv = pricer.price_swaption(
                        method="mc", num_paths=50000, dt=0.125, return_implied_vol=True
                    )
                    time_1=time.time()
                    print(f"  Monte Carlo:      Strike={strike:.6f}, Price={mc_price:.6f}, IV={mc_iv:.2%}, compute time={time_1-time_0:.2f}")
                    mc_prices.append(float(mc_price))
                    mc_ivs.append(float(mc_iv))
                except:
                    print("  Monte Carlo:  Not available")
        if run_cd:
            # Approximations
            # cd_price = pricer.price_swaption(
            #         method="collin_dufresne", return_implied_vol=False
            #     )
            # print(f"  Collin-Dufresne:  Price={cd_price:.6f} ")

            for i in range(nb_run):
                time_0=time.time()
                try:
                    cd_price, cd_iv = pricer.price_swaption(
                        method="collin_dufresne", return_implied_vol=True
                    )
                    time_1=time.time()
                    print(f"  Collin-Dufresne:  Strike={strike:.6f}, Price={cd_price:.6f}, IV={cd_iv:.2%}, compute time={time_1-time_0:.2f}")
                    cd_prices.append(float(cd_price))
                    cd_ivs.append(cd_iv)
                except:
                    print("  Collin-Dufresne:  Not available")
    
        # try:
        #     gamma_price, gamma_iv = pricer.price_swaption(
        #         method="gamma_approx", return_implied_vol=True
        #     )
        #     time_1=time.time()
        #     print(f"  Gamma Approx:     Price={gamma_price:.6f}, IV={gamma_iv:.2%}, compute time={time_1-time_0:.4%}")
        # except:
        #     print("  Gamma Approx:     Not available")
    
    plot_fig=True
    if plot_fig:
        # Plot results
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        if run_fft:
            plt.plot(strike_lists, fft_prices, label="FFT", marker='o')
        if run_mc:
            plt.plot(strike_lists, mc_prices, label="Monte Carlo", marker='x')
        if run_cd:
            plt.plot(strike_lists, cd_prices, label="Collin-Dufresne", marker='s')
        plt.xlabel("Strike")
        plt.ylabel("Swaption Price")
        plt.title("Swaption Prices vs Strike")
        # plt.legend()
        # plt.grid(True)
        # plt.draw() #plt.show(block=False)

        
    
        plt.subplot(1, 2, 2)
        if run_fft:
            plt.plot(strike_lists, fft_ivs, label="FFT", marker='o')
        if run_mc:
            plt.plot(strike_lists,  mc_ivs, label="Monte Carlo", marker='x')
        if run_cd:
            plt.plot(strike_lists,  cd_ivs, label="Collin-Dufresne", marker='s')
        plt.xlabel("Strike")
        plt.ylabel("Swaption implied vol")
        plt.title("Swaption implied vol vs Strike")

        plt.legend()
         
        plt.grid(True, alpha=0.3)
    
        plt.tight_layout()
        plt.draw() #plt.show(block=False)

    return lrw_model, pricer


def example_jump_term_structure():
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
    lrw_model_config = LRWModelConfig( n=n,  alpha=alpha,  x0=x0,  omega=omega,  m=m, sigma=sigma ,is_bru_config=False, has_jump=False )
    # lrw_model_config = LRWModelConfig( n=n,  alpha=alpha,  x0=x0,  omega=omega,  m=m, sigma=sigma ,is_bru_config=False, has_jump=True )
    swaption_config = SwaptionConfig(maturity=maturity,tenor=tenor,
                                   strike=0.05, 
                                   delta_float = delta_float,
                                   delta_fixed= delta_fixed)
    lrw_model = LRWModel(lrw_model_config,swaption_config)
    
    ##jump component
    lambda_intensity = jnp.float32(0.1)
    nu = 4
    eta = jnp.array([[0.10, 0.002], [0.002, 0.0005]], dtype=jnp.float32)
    xi = jnp.array([[0.04, 0.004], [0.004, 0.02]], dtype=jnp.float32)

    jump_component =JumpComponent(lambda_intensity, nu, eta, xi)
    lrw_model.wishart.set_jump(jump_component)


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
    plt.draw() #plt.show(block=False)
    
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


def example_jump_parameter_comparison():
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
        # Initialize model
        lrw_model_config = LRWModelConfig( n=n,  alpha=alpha,  x0=x0,  omega=omega,  m=m, sigma=sigma ,is_bru_config=False, has_jump=False )
        # lrw_model_config = LRWModelConfig( n=n,  alpha=alpha,  x0=x0,  omega=omega,  m=m, sigma=sigma ,is_bru_config=False, has_jump=True )
        swaption_config = SwaptionConfig(maturity=maturity,tenor=tenor,
                                       strike=0.05, 
                                       delta_float = delta_float,
                                       delta_fixed= delta_fixed)
        model = LRWModel(lrw_model_config,swaption_config)
    
        ##jump component
        lambda_intensity = jnp.float32(0.1)
        nu = 4
        eta = jnp.array([[0.10, 0.002], [0.002, 0.0005]], dtype=jnp.float32)
        xi = jnp.array([[0.04, 0.004], [0.004, 0.02]], dtype=jnp.float32)

        jump_component =JumpComponent(lambda_intensity, nu, eta, xi)
        model.wishart.set_jump(jump_component)


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


def example_jump_wishart_properties():
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

    # Initialize model
    lrw_model_config = LRWModelConfig( n=n,  alpha=alpha,  x0=x0,  omega=omega,  m=m, sigma=sigma ,is_bru_config=False, has_jump=False )
    # lrw_model_config = LRWModelConfig( n=n,  alpha=alpha,  x0=x0,  omega=omega,  m=m, sigma=sigma ,is_bru_config=False, has_jump=True )
    swaption_config = SwaptionConfig(maturity=maturity,tenor=tenor,
                                   strike=0.05, 
                                   delta_float = delta_float,
                                   delta_fixed= delta_fixed)
    lrw_model = LRWModel(lrw_model_config,swaption_config)
    
    ##jump component
    lambda_intensity = jnp.float32(0.1)
    nu = 4
    eta = jnp.array([[0.10, 0.002], [0.002, 0.0005]], dtype=jnp.float32)
    xi = jnp.array([[0.04, 0.004], [0.004, 0.02]], dtype=jnp.float32)

    jump_component =JumpComponent(lambda_intensity, nu, eta, xi)
    lrw_model.wishart.set_jump(jump_component)


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


if __name__ == "__main__":
    print("Linear Rational Wishart Model with Jump  Basic Examples")
    # Run all examples
    example_jump_basic_lrw_setup()     #ok working
    example_jump_swaption_pricing()    #ok working
    example_jump_term_structure()        #ok working
    example_jump_parameter_comparison() #ok working
    example_jump_wishart_properties() #ok working
    # Keep all plots open
    plt.ioff()  # Turn off interactive mode
    plt.show()  # This will block and keep all windows open
