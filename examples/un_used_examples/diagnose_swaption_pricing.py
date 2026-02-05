"""
Diagnostic to understand why model vol doesn't depend on sigma.

Add this function to your calibration example and run it after creating the calibrator.
"""

import numpy as np
import jax.numpy as jnp


from linear_rational_wishart.models.interest_rate.config import SwaptionConfig, LRWModelConfig


def diagnose_swaption_pricing(calibrator):
    """
    Diagnose the swaption pricing to understand volatility components.
    
    Parameters
    ----------
    calibrator : LRWJumpCalibrator
        The calibrator object after creation
    """
    print("\n" + "=" * 70)
    print("SWAPTION PRICING DIAGNOSTIC")
    print("=" * 70)
    
    model = calibrator.model
    
    # Print model parameters
    print("\n1. MODEL PARAMETERS")
    print("-" * 50)
    print(f"  n = {model.n}")
    print(f"  alpha = {model.alpha:.6f}")
    print(f"  x0 =\n{np.array(model.x0)}")
    print(f"  omega =\n{np.array(model.omega)}")
    print(f"  m =\n{np.array(model.m)}")
    print(f"  sigma =\n{np.array(model.sigma)}")
    print(f"  u1 =\n{np.array(model.u1)}")
    print(f"  u2 =\n{np.array(model.u2)}")
    print(f"  is_spread = {model.is_spread}")
    
    # Get first swaption
    print("\n2. FIRST SWAPTION DETAILS")
    print("-" * 50)
    if hasattr(calibrator, 'daily_data') and hasattr(calibrator.daily_data, 'swaption_data_cube'):
        first_swaption = calibrator.daily_data.swaption_data_cube.iloc[0]["Object"]
        for first_swaption in calibrator.daily_data.swaption_data_cube['Object']:
            break  # Get the first swaption object
        expiry = first_swaption.expiry_maturity
        tenor = first_swaption.swap_tenor_maturity
        strike = first_swaption.strike if hasattr(first_swaption, 'strike') else 'N/A'
        market_vol = first_swaption.vol
        market_price = first_swaption.market_price
        
        print(f"  Expiry: {expiry} years")
        print(f"  Tenor: {tenor} years")
        print(f"  Strike: {strike}")
        print(f"  Market Vol: {market_vol:.6f} ({market_vol*100:.2f}%)")
        print(f"  Market Price: {market_price:.8f}")
        
        calibrator.objectives._price_single_swaption(first_swaption)

       
        if hasattr(first_swaption, 'model_vol'):
            print(f"  Model Vol: {first_swaption.model_vol:.6f} ({first_swaption.model_vol*100:.2f}%)")
        if hasattr(first_swaption, 'model_price'):
            print(f"  Model Price: {first_swaption.model_price:.8f}")
        
        

    else:
        print("  No swaption data available")
        first_swaption = None
    
    print("=" * 50  )
    print("\n\n\nChecking the values of a3 and b3 for ALL swaptions in the cube:")
    print("=" * 50  )

    for first_swaption in calibrator.daily_data.swaption_data_cube['Object']:
            # break  # Get the first swaption object
            expiry = first_swaption.expiry_maturity
            tenor = first_swaption.swap_tenor_maturity
            strike = first_swaption.strike if hasattr(first_swaption, 'strike') else 'N/A'
            market_vol = first_swaption.vol
            market_price = first_swaption.market_price
            
            print(f"  Expiry: {expiry} years")
            print(f"  Tenor: {tenor} years")
            print(f"  Strike: {strike}")
            print(f"  Market Vol: {market_vol:.6f} ({market_vol*100:.2f}%)")
            print(f"  Market Price: {market_price:.8f}")
            model.set_swaption_config(
                            SwaptionConfig(
                            maturity=first_swaption.expiry_maturity ,
                            tenor=first_swaption.swap_tenor_maturity,
                            strike=first_swaption.strike,            
                            delta_float=1.0,
                            delta_fixed=0.5
                            #,call=swaption.is_call
                                ))
            # Compute b3 and a3
            model.compute_b3_a3()
            
            print(f"  b3 = {float(model.b3):.8f}")
            print(f"  a3 =\n{np.array(model.a3)}")
            print(f"  Tr(a3) = {np.trace(np.array(model.a3)):.8f}")

            print(f"  Expiry: {expiry} years, \
            Tenor: {tenor} years,\
            Strike: {strike},\
            Market Vol: {market_vol:.6f} ({market_vol*100:.2f}%),\
            Market Price: {market_price:.8f},\
            b3 = {float(model.b3):.8f},\
            a3 =\n{np.array(model.a3)},\
            Tr(a3) = {np.trace(np.array(model.a3)):.8f}")

    print("=" * 50  )
    print("\n\n\nChecking the values of a3 and b3 for ALL swaptions in the cube:")
    print("=" * 50  )
    
    first_swaption = calibrator.daily_data.swaption_data_cube.iloc[0]["Object"]
    
    # Compute b3 and a3
    print("\n3. SWAPTION COEFFICIENTS (a3, b3)")
    print("-" * 50)
    
    # Set swaption parameters on model
    if first_swaption is not None:
        # Update model's swaption config
        # model.maturity = first_swaption.expiry_maturity
        # model.tenor = first_swaption.swap_tenor_maturity
        # model.strike = first_swaption.strike if hasattr(first_swaption, 'strike') else 0.0
        model.set_swaption_config(
                            SwaptionConfig(
                            maturity=first_swaption.expiry_maturity ,
                            tenor=first_swaption.swap_tenor_maturity,
                            strike=first_swaption.strike,            
                            delta_float=1.0,
                            delta_fixed=0.5
                            #,call=swaption.is_call
                                ))
        # Compute b3 and a3
        model.compute_b3_a3()
        
        print(f"  b3 = {float(model.b3):.8f}")
        print(f"  a3 =\n{np.array(model.a3)}")
        print(f"  Tr(a3) = {np.trace(np.array(model.a3)):.8f}")
        
        # Compute Tr(a3 @ x0)
        a3_np = np.array(model.a3)
        x0_np = np.array(model.x0)
        trace_a3_x0 = np.trace(a3_np @ x0_np)
        print(f"  Tr(a3 @ x0) = {trace_a3_x0:.8f}")
        print(f"  b3 + Tr(a3 @ x0) = {float(model.b3) + trace_a3_x0:.8f}")
    
    # Check swap rate and annuity
    print("\n4. SWAP RATE AND ANNUITY")
    print("-" * 50)
    try:
        annuity, swap_rate = model.compute_annuity()
        print(f"  Annuity: {float(annuity):.8f}")
        print(f"  ATM Swap Rate: {float(swap_rate):.6f} ({float(swap_rate)*100:.2f}%)")
        
        # Compare with strike
        if first_swaption is not None and hasattr(first_swaption, 'strike'):
            strike = first_swaption.strike
            moneyness = swap_rate - strike
            print(f"  Strike: {strike:.6f}")
            print(f"  Moneyness (F - K): {float(moneyness):.6f} ({float(moneyness)*10000:.1f} bps)")
    except Exception as e:
        print(f"  Error computing annuity: {e}")
    
    # Check bond prices
    print("\n5. BOND PRICES")
    print("-" * 50)
    try:
        for t in [0.5, 1.0, 2.0, 5.0, 10.0]:
            bond_price = model.bond(t)
            print(f"  P({t}Y) = {float(bond_price):.8f}")
    except Exception as e:
        print(f"  Error computing bonds: {e}")
    
    # Compute E[X_T] for the swaption maturity
    print("\n6. EXPECTED STATE E[X_T]")
    print("-" * 50)
    if first_swaption is not None:
        T = first_swaption.expiry_maturity
        try:
            mean_vec = model.wishart.compute_mean_wishart(T)
            n = model.n
            # mean_vec is a column vector, reshape to matrix
            mean_X_T = np.array(mean_vec).reshape((n, n), order='F')
            print(f"  T = {T} years")
            print(f"  E[X_T] =\n{mean_X_T}")
            
            # Expected trace
            if hasattr(model, 'a3'):
                expected_trace = np.trace(np.array(model.a3) @ mean_X_T)
                print(f"  E[Tr(a3 @ X_T)] = {expected_trace:.8f}")
        except Exception as e:
            print(f"  Error computing E[X_T]: {e}")
    
    # Test with sigma = 0
    print("\n" + "=" * 70)
    print("7. TEST: MODEL WITH SIGMA ≈ 0")
    print("=" * 70)
    
    # Store original parameters
    original_sigma = np.array(model.sigma).copy()
    n = model.n
    
    # Set sigma to near-zero
    zero_sigma = np.eye(n) * 1e-12
    
    print(f"\nSetting sigma to:\n{zero_sigma}")
    
    model.set_model_params(
        n=n,
        alpha=float(model.alpha),
        x0=np.array(model.x0),
        omega=np.array(model.omega),
        m=np.array(model.m),
        sigma=zero_sigma
    )
    
    # Update and reprice
    try:
        calibrator.market_handler.update_swaption_market_data(
            model=model,
            market_based_strike=calibrator.config.use_market_based_strike
        )
        calibrator.objectives._reprice_swaptions()
        
        first_swaption = calibrator.daily_data.swaption_data_cube.iloc[0]["Object"]
        zero_sigma_vol = first_swaption.model_vol
        zero_sigma_price = first_swaption.model_price
        
        print(f"\nResults with sigma ≈ 0:")
        print(f"  Model Vol: {zero_sigma_vol:.6f} ({zero_sigma_vol*100:.2f}%)")
        print(f"  Model Price: {zero_sigma_price:.8f}")
        print(f"  Market Vol: {market_vol:.6f} ({market_vol*100:.2f}%)")
        print(f"  Market Price: {market_price:.8f}")
        
    except Exception as e:
        print(f"Error repricing with sigma=0: {e}")
        zero_sigma_vol = None
    
    # Restore original sigma
    model.set_model_params(
        n=n,
        alpha=float(model.alpha),
        x0=np.array(model.x0),
        omega=np.array(model.omega),
        m=np.array(model.m),
        sigma=original_sigma
    )
    print(f"\n✅ Original sigma restored.")
    
    # Diagnosis
    print("\n" + "=" * 70)
    print("8. DIAGNOSIS")
    print("=" * 70)
    
    if zero_sigma_vol is not None:
        if zero_sigma_vol > 0.01:  # If vol is still > 1% with sigma=0
            print(f"\n⚠️  PROBLEM IDENTIFIED:")
            print(f"   Model produces {zero_sigma_vol*100:.1f}% volatility even with sigma ≈ 0!")
            print(f"   This means volatility is NOT coming from the diffusion (sigma).")
            print(f"\n   Possible causes:")
            print(f"   1. The implied vol calculation may have an issue")
            print(f"   2. The forward rate (b3 + Tr(a3*x)) may be far from the strike")
            print(f"   3. There may be a baseline/floor in the pricing formula")
            print(f"   4. The Bachelier formula may be applied incorrectly")
            
            # Check if it's a moneyness issue
            if first_swaption is not None:
                try:
                    annuity, swap_rate = model.compute_annuity()
                    strike = first_swaption.strike
                    intrinsic = max(0, swap_rate - strike) * annuity
                    print(f"\n   Additional check:")
                    print(f"   Forward swap rate: {float(swap_rate):.6f}")
                    print(f"   Strike: {strike:.6f}")
                    print(f"   Intrinsic value: {float(intrinsic):.8f}")
                    print(f"   Model price: {zero_sigma_price:.8f}")
                    
                    if abs(float(intrinsic) - zero_sigma_price) < 1e-6:
                        print(f"\n   ✅ Model price equals intrinsic value with sigma=0 (correct)")
                        print(f"   The high implied vol is due to moneyness, not a bug.")
                    else:
                        print(f"\n   ❌ Model price differs from intrinsic value!")
                        print(f"   This suggests an issue in the pricing formula.")
                except Exception as e:
                    print(f"   Error in additional check: {e}")
        else:
            print(f"\n✅ Model vol goes to {zero_sigma_vol*100:.2f}% when sigma ≈ 0.")
            print(f"   This is expected behavior - sigma is working correctly.")
    
    print("\n" + "=" * 70 + "\n")


# Usage:
# After creating calibrator in example_basic_calibration():
#
# calibrator = LRWJumpCalibrator(lrw_model, daily_data, config)
# diagnose_swaption_pricing(calibrator)
