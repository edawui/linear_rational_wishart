"""
FX volatility smile analysis example.

This script demonstrates how to analyze volatility smiles produced by the
LRW FX model under different parameter configurations.
"""

from itertools import count
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import sys
import os
from pathlib import Path
import matplotlib
import math
import cmath

from linear_rational_wishart.models.fx.base import BaseFxModel
from linear_rational_wishart.models.fx.lrw_fx import LRWFxModel 
from linear_rational_wishart.models.fx.currency_basket import CurrencyBasket
from linear_rational_wishart.pricing.fx.fourier_fx_pricer import FourierFxPricer
from linear_rational_wishart.pricing.fx.mc_fx_pricer import MonteCarloFxPricer   
from linear_rational_wishart.pricing.implied_vol_black_scholes import * 
from linear_rational_wishart.pricing.black_scholes import * 

matplotlib.use('TkAgg')  # or 'Qt5Agg'

# Enable interactive mode globally
plt.ion()

def create_base_fx_model(fx_spot=1.1359) -> LRWFxModel:
    """Create base FX model with standard parameters."""
    n = 2
    
    # Model parameters
    x0 = np.array([[4.99996419e+00, -3.18397238e-04],
                   [-3.18397238e-04, 4.88302000e-04]])
    
    omega = np.array([[2.09114424e+00, -8.04612684e-04],
                      [-8.04612684e-04, 1.92477100e-03]])
    
    m = np.array([[-0.20583617, 0.0],
                  [0.0, -0.02069993]])#0.02
    
    sigma = 0.75*np.array([[0.15871937, 0.10552826],
                      [0.10552826, 0.02298161]])#0.02

    alpha_i = 0.05  # Domestic rate
    alpha_j = 0.04  # Foreign rate
    
    # u_i = np.array([[1.0, 0.0], [0.0, 0.0]])
    # u_j = np.array([[0.0, 0.0], [0.0, 1.0]])
      
    # omega = np.array([[2.09114424e+00, -8.04612684e-04],
    #                   [-8.04612684e-04, 1.92477100e-03]])
    # Model parameters
    # ##parameters from IR model paper.
    # x0 = np.array([[0.15, -1.055e-04],
    #                [-1.055e-04, 8.25e-4]])
    
    # omega = np.array([[0.11 , -2.974e-03],
    #                   [-2.974e-03, 1.3770e-3]])
    
    # m = np.array([[-0.0642, 0.0],
    #               [0.0, -0.028]])
    
    # sigma = np.array([[0.5165, -0.041],
    #                   [-0.041, 0.0669]])
    
    # # omega = 3.0*sigma @ sigma
    # alpha_i = 0.025  # Domestic rate
    # alpha_j = 0.024  # Foreign rate
    
    # ###################################################################################
    
    # x0= np.array([[0.11383552022082195, 0.009504803705570511], [0.009504803705570511, 0.2875820825591115]])
    # omega=np.array([[1.762527884046532e-2, 0.0019615111419316773], [0.0019615111419316773, 0.06463364229290659]])
    # omega=np.array([[1.762527884046532e-4, 0.0019615111419316773], [0.0019615111419316773, 0.06463364229290659]])
    # omega=np.array([[0.5, 0.0019615111419316773], [0.0019615111419316773, 0.06463364229290659]])
    # m= np.array([[-0.011530777480229652, 0.0], [0.0, -0.06148652138127832]] )
    # sigma= 1.5*np.array([[0.2135465714972303, 0.007632468354215629], [0.007632468354215629,1.7074434479097225]])
    # # m=5.0*m
    # ##get initial sigma based on x0 and u_i, u_j
    # # sigma= get_initial_sigma(x0,u_i,u_j,initial_vol=0.09)#85 )
    # rho_x0,rho_omega,rho_sigma= -0.5, 0.5, -0.5
    
    # ##set anti-diagonal elements based on correlation
    # x0[0,1]=x0[1,0]=       rho_x0*math.sqrt(x0[0,0]*x0[1,1])
    # omega[0,1]=omega[1,0]= rho_omega*math.sqrt(omega[0,0]*omega[1,1])
    # sigma[0,1]=sigma[1,0]= rho_sigma*math.sqrt(sigma[0,0]*sigma[1,1])

    # alpha_i = 0.03959478810429573  # Domestic rate
    # alpha_j =  0.01681286096572876  # Foreign rate
    
    ###################################################################

 
    
   

    u_i = np.array([[1.0, 0.0], [0.0, 0.0]])
    u_j = np.array([[0.0, 0.0], [0.0, 1.0]])

    # u_i = np.array([[2.0, 0.02], [0.02, 0.0]])
    # u_j = np.array([[0.00, -0.01], [-0.01, 1.0]])
    # fx_spot = 1.0
    
    return LRWFxModel(n, x0, omega, m, sigma, alpha_i, u_i, alpha_j, u_j, fx_spot)



def compute_moment_generating_function():
#     model: LRWFxModel,
#     maturity: float,
#     strikes: np.ndarray,
#     pricing_method: str = "FOURIER",
#     nb_paths: int = 10000
# ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute implied volatilities for given strikes.
    
    Parameters
    ----------
    model : LRWFxModel
        FX model instance
    maturity : float
        Option maturity
    strikes : np.ndarray
        Array of strike prices
    pricing_method : str
        Pricing method ('FOURIER' or 'MC')
    nb_paths : int
        Number of Monte Carlo paths (if using MC)
        
    Returns
    -------
    strikes : np.ndarray
        Strike prices
    implied_vols : np.ndarray
        Implied volatilities
    """
    fx_spot=1.1359
    
    model = create_base_fx_model(fx_spot=1.1359)
    maturity =1.0  # 1 year maturity
    moneyness =0.75
    # Get forward and discount factors
    fx_fwd = model.compute_fx_forward(maturity)
    df_foreign = model.lrw_currency_i.bond(maturity)
    df_domestic = model.lrw_currency_j.bond(maturity)
    r_f = -np.log(df_foreign) / maturity
    r_d = -np.log(df_domestic) / maturity
    
    print(f"Forward: {fx_fwd:.4f}, r_d: {r_d:.4f}, r_f: {r_f:.4f}")
    print(type(model))


    print("x0")
    print(f"fx model x0 = {model.x0}")
    print(f"model i x0 = {model.lrw_currency_i.x0}")
    print(f"model j x0 = {model.lrw_currency_j.x0}")
    print("==========================================")
    
    print("omega")
    print(f"fx model omega = {model.omega}")
    print(f"model i omega = {model.lrw_currency_i.omega}")
    print(f"model j omega = {model.lrw_currency_j.omega}")
    print("==========================================")

    print("m")
    print(f"fx model m = {model.m}")
    print(f"model i m = {model.lrw_currency_i.m}")
    print(f"model j m = {model.lrw_currency_j.m}")
    print("==========================================")

    
    print("sigma")
    print(f"fx model sigma = {model.sigma}")
    print(f"model i sigma = {model.lrw_currency_i.sigma}")
    print(f"model j sigma = {model.lrw_currency_j.sigma}")
    print("==========================================")

    print("alpha")
    print(f"fx model alpha_i = {model.alpha_i}")
    print(f"model i alpha_i = {model.lrw_currency_i.alpha}")
    print(f"fx model alpha_i = {model.alpha_j}")
    print(f"model j alpha_j = {model.lrw_currency_j.alpha}")
    print("==========================================")
    
    print("u1")
    print(f"fx model u_i = {model.u_i}")
    print(f"model i  u_i = {model.lrw_currency_i.u1}")
    print(f"fx model u_j = {model.u_j}")
    print(f"model j  u_j = {model.lrw_currency_j.u1}")
    print("==========================================")



    def integrand(ui):
            u = complex(0.5, ui)
            z1 = model.lrw_currency_i.wishart.phi_one(1.0, u * model.aij_2)
            z1 *= cmath.exp(u * model.bij_2)
            
            z2 = z1 / (u * u)
            return  z2 #.real
    moneynesses = [0.5,0.75,1.0,1.25,1.5]
    strikes=[ mm*fx_fwd for mm in moneynesses]

    model.set_option_properties(maturity, strikes[0])
    # u = 1.5+1.2j
    ur=0.5
    # ui_range=list(np.arange(0, 0.5, 0.1))
    start=0.0
    end=15#50#60
    nbPoint=1000
    ui_range=list(np.linspace(start, end, nbPoint))
    results_imag=[]
    results_real=[]
    y_values=[[],[]]
    count=0
   
    colors = ['b', 'g', 'r', 'c', 'm']
    plt.figure(figsize=(12, 8))

    fft_pricer =  FourierFxPricer(model, ur=ur, nmax=10)#00)
    # mc_pricer  =  MonteCarloFxPricer(model, nb_mc=10000 , dt=0.1, schema= "EULER_CORRECTED")#"EULER_FLOORED")
    mc_pricer  =  MonteCarloFxPricer(model, nb_mc=10000, dt=0.1, schema= "EULER_FLOORED")
    
    for_df=model.lrw_currency_j.bond(maturity)  # Ensure bond is set for foreign currency
    dom_df=model.lrw_currency_i.bond(maturity)  # Ensure bond is set for foreign currency
    # fx = model.fx_spot
    fx = fx_fwd
    for_check = for_df*fx
    dom_check = dom_df*fx
    # fx_fwd_analytic = model.compute_fx_forward(maturity)

    print(f"Foreign bond value at maturity: {for_df:.6f},  fx fwd time Foreign Df check:  {for_check:.6f}") 
    # print(f"Domestic bond value at maturity: {dom_df:.6f}, domestic check: {dom_check:.6f}")
    model.set_option_properties(maturity, 0.0)
     
    fft_price =fft_pricer.price_options([maturity],[0],[True])
    mc_price, std_error = mc_pricer.price_options([maturity],[0],[True])    
        
    print(f"fx Forward check pricing call at strik zero,  FFT Price: {fft_price[0]:.15f}, MC Price: {mc_price[0]:.15f}, std_error:{std_error[0]:.15f}")
     
    for strike in strikes:
        model.set_option_properties(maturity, strike)
        results_real=[]
        fft_price =fft_pricer.price_options([maturity],[strike],[True])
        # mc_price, std_error = mc_pricer.price_options([maturity],[strike],[True])
      

        moneyness = strike / fx_fwd
        
        # print(f"Strike: {strike:.4f},moneyness={moneyness}, FFT Price: {fft_price[0]:.15f}, MC Price: {mc_price[0]:.15f}, std_error:{std_error[0]:.15f}")
        # print(f" model bij_2= {model.bij_2},  fft bij_2= {fft_pricer.fx_model.bij_2},  MC bij_2= {mc_pricer.fx_model.bij_2}" )
        # print(f" model aij_2= {model.aij_2},  fft aij_2= {fft_pricer.fx_model.aij_2},  MC aij_2= {mc_pricer.fx_model.aij_2}" )
        # print("==========================================================================================================")


        for ui in ui_range:
            u = complex(ur, ui)
            # res= model.lrw_currency_i.wishart.phi_one(1.0, u * model.aij_2)
            res = integrand(ui)
            results_imag.append(res.imag)
            results_real.append(res.real)
        # print(f"Moment generating function at u={u}: {res:.6f}")
            # y_values[count].append(res.real)

        plt.plot(
            ui_range,
            results_real,#y_values[count],
            color=colors[count % len(colors)],
            linestyle='-',  #marker='o',
            # linewidth=2,
            label=f'Moneyness:={moneyness}'
            )

        count+=1
    # print(f"Results for ui={ui_range}:\n")
    # print(f"Real part: {results_real}")
    # print(f"Imaginary part: {results_imag}")
    # fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # colors = ['b', 'g', 'r', 'c', 'm']
    # print(len(ui_range), len(y_values[0]), len(y_values[1]))

    # axes[0].plot(
    #         ui_range,
    #         y_values[0],
    #         color=colors[0],
    #         linestyle='-'  #marker='-'
           
    #     )
        
    #     # Also plot in terms of absolute strikes
    # axes[1].plot(
    #         ui_range,
    #         y_values[1],#results_real,
    #         color=colors[1],
    #         linestyle='-'  #marker='-'
    #     )


    # # Configure plots
    # axes[0].set_xlabel('ui')
    # axes[0].set_ylabel('Moment generating (real part)')
    # axes[0].set_title('Moment generating (real part)')
    # # axes[0].legend()
    # axes[0].grid(True, alpha=0.3)
    
    # axes[1].set_xlabel('ui')
    # axes[1].set_ylabel('Moment generating (imaginary part)')
    # axes[1].set_title('Moment generating (imaginary part)')
    # # axes[1].legend()
    # axes[1].grid(True, alpha=0.3)

    # plt.suptitle(f'Call Integrand')
    # plt.tight_layout()
    # plt.show()

    plt.xlabel('ui')
    plt.ylabel('fx Call option Integrand')
    plt.title('Fx Call Option Integrand')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def compute_implied_volatilities(
    model: LRWFxModel,
    maturity: float,
    strikes: np.ndarray,
    pricing_method: str = "FOURIER",
    nb_paths: int = 10000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute implied volatilities for given strikes.
    
    Parameters
    ----------
    model : LRWFxModel
        FX model instance
    maturity : float
        Option maturity
    strikes : np.ndarray
        Array of strike prices
    pricing_method : str
        Pricing method ('FOURIER' or 'MC')
    nb_paths : int
        Number of Monte Carlo paths (if using MC)
        
    Returns
    -------
    strikes : np.ndarray
        Strike prices
    implied_vols : np.ndarray
        Implied volatilities
    """
    # Get forward and discount factors
    fx_fwd = model.compute_fx_forward(maturity)
    df_foreign = model.lrw_currency_i.bond(maturity)
    df_domestic = model.lrw_currency_j.bond(maturity)
    r_f = -np.log(df_foreign) / maturity
    r_d = -np.log(df_domestic) / maturity
    
    print(f"Forward: {fx_fwd:.4f}, r_d: {r_d:.4f}, r_f: {r_f:.4f}")
    print(type(model))
    # Price options for all strikes
    maturities = [maturity] * len(strikes)
    call_flags = [True] * len(strikes)
    fourier_fx_pricer =FourierFxPricer(model)
    # prices = model.price_fx_option_list(price_options
    prices = fourier_fx_pricer.price_options(
        # pricing_method,
        maturities,
        strikes,
        call_flags
        # nb_mc=nb_paths,
        # dt=1/50.0,
        # schema="EULER_FLOORED",
        # ur=0.5,
        # nmax=100
    )
    
    


    # Compute implied volatilities
    implied_vols = []
    for strike, price in zip(strikes, prices):
        try:

            iv = implied_volatility_black_scholes(price,
                model.fx_spot,strike, maturity, r_d, r_f, True
            )
            # iv = black_scholes.implied_volatility_fx(
            #     model.fx_spot, r_d, r_f, strike, maturity, price, True
            # )
            implied_vols.append(iv)
            print(f"Strike: {strike:.4f}, Price: {price:.6f}, IV: {iv:.4f}")
        except Exception as e:
            print(f"Failed to compute IV for strike {strike}: {e}")
            implied_vols.append(np.nan)
    
    return strikes, np.array(implied_vols)


def analyze_correlation_impact(correlation_on="all"):
    """Analyze how correlation affects the volatility smile."""
    maturity = 1.0
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = ['b', 'g', 'r', 'c', 'm']
    
    # Test different correlation levels
    # if  (correlation_on=="ALL"):
        # correlations = [-0.25, -0.125, 0.0, 0.25, 0.5]
    # if correlation_on=="SIGMA" :
    #     correlations = [ -0.125, 0.0, 0.25, 0.5,0.75]
    # else:
    #     correlations = [-0.25, -0.125, 0.0, 0.25, 0.5] #if correlation_on=="X0" else [ -0.125, 0.0, 0.25, 0.5,0.75]

    correlations = [-0.15, -0.125, 0.0, 0.25, 0.5,0.75]
    correlation_on_title = "x0, omega, sigma" if correlation_on.upper()=="ALL" else correlation_on
    correlation_on=correlation_on.upper()
    for i, corr in enumerate(correlations):
        # Create model with specific correlation
        model = create_base_fx_model()
        x0 = model.x0.copy()
        sigma = model.sigma.copy()
        omega = model.omega.copy()
        
        if correlation_on=="SIGMA" or (correlation_on=="ALL"):
            # Modify correlation in sigma matrix
            sigma_diag = np.sqrt(sigma[0, 0] * sigma[1, 1])
            sigma = sigma.at[1, 0].set(corr * sigma_diag) 
            sigma = sigma.at[1, 0].set(corr * sigma_diag)
        
        if correlation_on=="X0" or (correlation_on=="ALL"):
        
            # Also modify X0 correlation
            x0_diag = np.sqrt(x0[0, 0] * x0[1, 1])
            x0 =x0.at[0, 1].set(corr * x0_diag)
            x0 = x0.at[1, 0].set(corr * x0_diag)
        if (correlation_on=="OMEGA") or (correlation_on=="ALL"):
        
            # Also modify X0 correlation
            omega_diag = np.sqrt(omega[0, 0] * omega[1, 1])
            omega = omega.at[0, 1].set(corr * omega_diag)
            omega = omega.at[1, 0].set(corr * omega_diag)
        
        # Update model
        # model.set_x0(x0)
        # model._sigma = sigma
        
        model.set_model_params(model.n,x0, omega, model.m,sigma,
                               model.alpha_i, model.u_i, model.alpha_j, model.u_j
                               , model.fx_spot)

        # Compute forward
        fx_fwd = model.compute_fx_forward(maturity)
        
        # Define strikes
        moneyness = np.linspace(0.85, 1.15, 11)
        strikes = moneyness * fx_fwd
        
        # Compute implied volatilities
        print(f"\nAnalyzing correlation = {corr}")
        strikes, implied_vols = compute_implied_volatilities(
            model, maturity, strikes, "MC", nb_paths=5000
        )
        
        # Remove NaN values
        valid_mask = ~np.isnan(implied_vols)
        strikes_valid = strikes[valid_mask]
        implied_vols_valid = implied_vols[valid_mask]
        moneyness_valid = strikes_valid / fx_fwd
        
        # Plot results
        axes[0].plot(
            moneyness_valid,
            implied_vols_valid,
            color=colors[i % len(colors)],
            marker='o',
            label=f'ρ = {corr}'
        )
        
        # Also plot in terms of absolute strikes
        axes[1].plot(
            strikes_valid,
            implied_vols_valid,
            color=colors[i % len(colors)],
            marker='o',
            label=f'ρ = {corr}'
        )
    
    # Configure plots
    axes[0].set_xlabel('Moneyness (K/F)')
    axes[0].set_ylabel('Implied Volatility')
    axes[0].set_title('Volatility Smile by Moneyness')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Strike')
    axes[1].set_ylabel('Implied Volatility')
    axes[1].set_title('Volatility Smile by Strike')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f'Impact of Correlation on {correlation_on_title}, on  FX Volatility Smile (T = {maturity}Y)')
    plt.tight_layout()
    plt.show()


def analyze_term_structure():
    """Analyze the term structure of implied volatilities."""
    model = create_base_fx_model()
    
    # Different maturities
    maturities = [0.25, 0.5, 1.0, 2.0, 5.0]
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(maturities)))
    
    for i, maturity in enumerate(maturities):
        # Compute forward
        fx_fwd = model.compute_fx_forward(maturity)
        
        # Define strikes around ATM
        moneyness = np.linspace(0.9, 1.1, 9)
        strikes = moneyness * fx_fwd
        
        print(f"\nAnalyzing maturity = {maturity}Y")
        strikes, implied_vols = compute_implied_volatilities(
            model, maturity, strikes, "FOURIER"
        )
        
        # Remove NaN values
        valid_mask = ~np.isnan(implied_vols)
        moneyness_valid = moneyness[valid_mask]
        implied_vols_valid = implied_vols[valid_mask]
        
        plt.plot(
            moneyness_valid,
            implied_vols_valid,
            color=colors[i],
            marker='o',
            linewidth=2,
            markersize=6,
            label=f'T = {maturity}Y'
        )
    
    plt.xlabel('Moneyness (K/F)')
    plt.ylabel('Implied Volatility')
    plt.title('FX Implied Volatility Term Structure')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def analyze_vol_of_vol_impact():
    """Analyze impact of volatility-of-volatility on smile."""
    maturity = 1.0
    
    # Base model
    base_model = create_base_fx_model()
    fx_fwd = base_model.compute_fx_forward(maturity)
    
    # Define strikes
    moneyness = np.linspace(0.85, 1.15, 11)
    strikes = moneyness * fx_fwd
    
    # Different vol-of-vol levels
    sigma_multipliers = [0.5, 0.75, 1.0, 1.25, 1.5]#2.0, 3.0]
    
    plt.figure(figsize=(10, 6))
    colors = ['b', 'g', 'r', 'm']
    
    for i, multiplier in enumerate(sigma_multipliers):
        # Create model with scaled sigma
        model = create_base_fx_model()
        sigma = base_model.sigma * multiplier
        
        model.set_model_params(model.n,model.x0, model.omega, model.m,sigma,
                               model.alpha_i, model.u_i, model.alpha_j, model.u_j
                               , model.fx_spot)


        print(f"\nAnalyzing sigma multiplier = {multiplier}")
        _, implied_vols = compute_implied_volatilities(
            model, maturity, strikes, "MC", nb_paths=5000
        )
        
        # Remove NaN values
        valid_mask = ~np.isnan(implied_vols)
        moneyness_valid = moneyness[valid_mask]
        implied_vols_valid = implied_vols[valid_mask]
        
        plt.plot(
            moneyness_valid,
            implied_vols_valid,
            color=colors[i % len(colors)],
            marker='o',
            linewidth=2,
            label=f'σ × {multiplier}'
        )
    
    plt.xlabel('Moneyness (K/F)')
    plt.ylabel('Implied Volatility')
    plt.title(f'Impact of Vol-of-Vol on Smile (T = {maturity}Y)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def analyze_interest_rate_differential():
    """Analyze impact of interest rate differential on smile."""
    maturity = 1.0
    
    # Base model
    base_model = create_base_fx_model()
    
    # Different interest rate scenarios
    scenarios = [
        ("Equal rates", 0.05, 0.05),
        ("Positive carry", 0.05, 0.04),
        ("Negative carry", 0.05, 0.055)#,
        # ("Large differential", 0.05, 0.10)
    ]
    
    plt.figure(figsize=(10, 6))
    colors = ['b', 'g', 'r', 'm']
    
    for i, (label, alpha_i, alpha_j) in enumerate(scenarios):
        # Create model with specific rates
        model = create_base_fx_model()
        # model._alpha_i = alpha_i
        # model._alpha_j = alpha_j
        
        model.set_model_params(model.n,model.x0, model.omega, model.m,model.sigma,
                               alpha_i, model.u_i, alpha_j, model.u_j
                               , model.fx_spot)


        # Compute forward
        fx_fwd = model.compute_fx_forward(maturity)
        
        # Define strikes
        moneyness = np.linspace(0.85, 1.15, 11)
        # moneyness = np.linspace(0.9, 1.15, 11)
        strikes = moneyness * fx_fwd
        
        print(f"\nAnalyzing {label}: r_d={alpha_i}, r_f={alpha_j}")
        _, implied_vols = compute_implied_volatilities(
            model, maturity, strikes, "FOURIER"
        )
        
        # Remove NaN values
        valid_mask = ~np.isnan(implied_vols)
        moneyness_valid = moneyness[valid_mask]
        implied_vols_valid = implied_vols[valid_mask]
        
        plt.plot(
            moneyness_valid,
            implied_vols_valid,
            color=colors[i % len(colors)],
            marker='o',
            linewidth=2,
            label=label
        )
    
    plt.xlabel('Moneyness (K/F)')
    plt.ylabel('Implied Volatility')
    plt.title(f'Impact of Interest Rate Differential on Smile (T = {maturity}Y)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    """Run all smile analysis examples."""
    print("FX Volatility Smile Analysis")
    print("=" * 50)

    test_model =create_base_fx_model() 
    
    # print(f"\n Testing to assess the nature/form of the call option integrand:")
    # compute_moment_generating_function()

    # Analyze correlation impact
    print("\n1. Analyzing correlation impact on smile...")
    print(" On all parameters x0, omega, sigma")
    analyze_correlation_impact(correlation_on="all")

    print(" On all parameters x0")
    analyze_correlation_impact(correlation_on="x0")

    print(" On all parameters omega ")
    analyze_correlation_impact(correlation_on="omega")

    print(" On all parameters  sigma")
    analyze_correlation_impact(correlation_on="sigma")

    # Analyze term structure
    print("\n2. Analyzing term structure of implied volatilities...")
    analyze_term_structure()
    
    
    # # Analyze vol-of-vol impact
    # print("\n3. Analyzing volatility-of-volatility impact...")
    analyze_vol_of_vol_impact()

    # Analyze interest rate differential
    print("\n4. Analyzing interest rate differential impact...")
    analyze_interest_rate_differential()
    # return


if __name__ == "__main__":
    main()
    # compute_moment_generating_function()

    # Keep all plots open
    plt.ioff()  # Turn off interactive mode
    plt.show()  # This will block and keep all windows open
