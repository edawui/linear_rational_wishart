"""
Test: Check if is_spread = True is causing the a3 issue.

The hypothesis is that a3[1,1] = 1.0 comes from the spread leg,
which shouldn't be part of the swaption payoff structure.
"""

import numpy as np


def test_is_spread_effect(calibrator):
    """
    Test the effect of is_spread on a3 and swaption pricing.
    """
    print("\n" + "=" * 70)
    print("TEST: EFFECT OF is_spread ON SWAPTION PRICING")
    print("=" * 70)
    
    model = calibrator.model
    
    # Store original is_spread
    original_is_spread = model.is_spread
    print(f"\nOriginal is_spread: {original_is_spread}")
    
    # Get first swaption info
    first_swaption = calibrator.daily_data.swaption_data_cube.iloc[0]["Object"]
    model.maturity = first_swaption.expiry_maturity
    model.tenor = first_swaption.swap_tenor_maturity
    model.strike = first_swaption.strike
    
    print(f"\nSwaption: {model.maturity}Y x {model.tenor}Y, Strike: {model.strike:.4f}")
    
    # Test 1: With is_spread = True
    print("\n" + "-" * 50)
    print("TEST 1: is_spread = True")
    print("-" * 50)
    
    model.is_spread = True
    model.compute_b3_a3()
    
    a3_with_spread = np.array(model.a3).copy()
    b3_with_spread = float(model.b3)
    
    print(f"  b3 = {b3_with_spread:.6f}")
    print(f"  a3 =\n{a3_with_spread}")
    print(f"  Tr(a3) = {np.trace(a3_with_spread):.6f}")
    print(f"  a3[0,0] = {a3_with_spread[0,0]:.6f}")
    print(f"  a3[1,1] = {a3_with_spread[1,1]:.6f}")
    
    # Compute forward
    x0_np = np.array(model.x0)
    forward_with_spread = b3_with_spread + np.trace(a3_with_spread @ x0_np)
    print(f"  b3 + Tr(a3 @ x0) = {forward_with_spread:.6f}")
    
    # Reprice
    calibrator.market_handler.update_swaption_market_data(
        model=model,
        market_based_strike=calibrator.config.use_market_based_strike
    )
    calibrator.objectives._reprice_swaptions()
    first_swaption = calibrator.daily_data.swaption_data_cube.iloc[0]["Object"]
    model_vol_with_spread = first_swaption.model_vol
    model_price_with_spread = first_swaption.model_price
    print(f"  Model Vol: {model_vol_with_spread:.6f} ({model_vol_with_spread*100:.2f}%)")
    print(f"  Model Price: {model_price_with_spread:.8f}")
    
    # Test 2: With is_spread = False
    print("\n" + "-" * 50)
    print("TEST 2: is_spread = False")
    print("-" * 50)
    
    model.is_spread = False
    model.compute_b3_a3()
    
    a3_no_spread = np.array(model.a3).copy()
    b3_no_spread = float(model.b3)
    
    print(f"  b3 = {b3_no_spread:.6f}")
    print(f"  a3 =\n{a3_no_spread}")
    print(f"  Tr(a3) = {np.trace(a3_no_spread):.6f}")
    print(f"  a3[0,0] = {a3_no_spread[0,0]:.6f}")
    print(f"  a3[1,1] = {a3_no_spread[1,1]:.6f}")
    
    # Compute forward
    forward_no_spread = b3_no_spread + np.trace(a3_no_spread @ x0_np)
    print(f"  b3 + Tr(a3 @ x0) = {forward_no_spread:.6f}")
    
    # Reprice
    calibrator.market_handler.update_swaption_market_data(
        model=model,
        market_based_strike=calibrator.config.use_market_based_strike
    )
    calibrator.objectives._reprice_swaptions()
    first_swaption = calibrator.daily_data.swaption_data_cube.iloc[0]["Object"]
    model_vol_no_spread = first_swaption.model_vol
    model_price_no_spread = first_swaption.model_price
    print(f"  Model Vol: {model_vol_no_spread:.6f} ({model_vol_no_spread*100:.2f}%)")
    print(f"  Model Price: {model_price_no_spread:.8f}")
    
    # Compare
    print("\n" + "-" * 50)
    print("COMPARISON")
    print("-" * 50)
    print(f"  a3[1,1] with spread:    {a3_with_spread[1,1]:.6f}")
    print(f"  a3[1,1] without spread: {a3_no_spread[1,1]:.6f}")
    print(f"  Difference in a3[1,1]:  {a3_with_spread[1,1] - a3_no_spread[1,1]:.6f}")
    print(f"")
    print(f"  Model vol with spread:    {model_vol_with_spread:.6f} ({model_vol_with_spread*100:.2f}%)")
    print(f"  Model vol without spread: {model_vol_no_spread:.6f} ({model_vol_no_spread*100:.2f}%)")
    print(f"  Market vol:               {first_swaption.vol:.6f} ({first_swaption.vol*100:.2f}%)")
    
    # Restore original
    model.is_spread = original_is_spread
    print(f"\n✅ Restored is_spread to {original_is_spread}")
    
    # Diagnosis
    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)
    
    if abs(a3_with_spread[1,1] - 1.0) < 0.01 and abs(a3_no_spread[1,1]) < 0.01:
        print("✅ CONFIRMED: a3[1,1] = 1.0 comes from is_spread = True")
        print("   The spread leg is adding u2 = [[0,0],[0,1]] to a3")
        print("")
        if model_vol_no_spread < model_vol_with_spread:
            print(f"   Model vol dropped from {model_vol_with_spread*100:.2f}% to {model_vol_no_spread*100:.2f}%")
            print("   when is_spread = False")
        
        print("\n   RECOMMENDATION: For swaption calibration, consider:")
        print("   1. Set is_spread = False if swaptions are on OIS (not IBOR)")
        print("   2. Or check if the spread term should be in the floating leg formula")
    
    print("=" * 70 + "\n")
    
    return {
        'a3_with_spread': a3_with_spread,
        'a3_no_spread': a3_no_spread,
        'model_vol_with_spread': model_vol_with_spread,
        'model_vol_no_spread': model_vol_no_spread
    }


# Usage:
# result = test_is_spread_effect(calibrator)
