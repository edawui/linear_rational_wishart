"""
Diagnostic to check spread calibration and the x0[1,1] value.

The issue: x0[1,1] = 0.00265 seems too small, which means
the spread contribution to swaption payoff is negligible.
"""

import numpy as np


def diagnose_spread_calibration(calibrator):
    """
    Check if x0[1,1] is correctly calibrated to match market spreads.
    """
    model = calibrator.model
    
    print("=" * 70)
    print("SPREAD CALIBRATION DIAGNOSTIC")
    print("=" * 70)
    
    print("\n1. MODEL PARAMETERS FOR SPREAD")
    print("-" * 50)
    x0 = np.array(model.x0)
    u2 = np.array(model.u2)
    alpha = model.alpha
    
    print(f"   x0 = \n{x0}")
    print(f"   u2 = \n{u2}")
    print(f"   α = {alpha:.6f}")
    
    print(f"\n   Tr(u2 @ x0) = {np.trace(u2 @ x0):.8f}")
    
    # The spread at time 0 for tenor Δ is:
    # A(0, 0, Δ) = e^0 * [b2(0) + Tr(a2(0)*x0)] / (1 + Tr(u1*x0))
    # where b2(0) = 0 and a2(0) = u2
    # So A(0,0,Δ) = Tr(u2*x0) / (1 + Tr(u1*x0))
    
    u1 = np.array(model.u1)
    norm = 1 + np.trace(u1 @ x0)
    spot_spread = np.trace(u2 @ x0) / norm
    
    print(f"\n2. SPOT SPREAD CALCULATION")
    print("-" * 50)
    print(f"   1 + Tr(u1*x0) = {norm:.6f}")
    print(f"   Spot spread = Tr(u2*x0) / (1+Tr(u1*x0))")
    print(f"               = {np.trace(u2 @ x0):.8f} / {norm:.6f}")
    print(f"               = {spot_spread:.8f}")
    print(f"               = {spot_spread * 10000:.2f} bps")
    
    print(f"\n3. COMPARISON WITH MARKET SPREAD")
    print("-" * 50)
    
    # Get market spread data if available
    if hasattr(calibrator.daily_data, 'spread_data') and len(calibrator.daily_data.spread_data) > 0:
        market_spreads = calibrator.daily_data.spread_data
        print(f"   Market spreads available: {len(market_spreads)} points")
        for i, row in market_spreads.iterrows():
            maturity = row.get('maturity', row.get('Maturity', i))
            spread = row.get('spread', row.get('Spread', row.get('value', 0)))
            print(f"   {maturity}Y: {spread*10000:.2f} bps")
    else:
        print(f"   No market spread data found in calibrator")
        print(f"   Typical EUR 3M spread: 20-50 bps")
    
    print(f"\n4. CONTRIBUTION TO SWAPTION PAYOFF")
    print("-" * 50)
    
    # For a 1Y swaption on 1Y swap:
    # The floating leg includes spread payments
    # With semi-annual payments, there are 2 spread payments
    
    delta_float = model.delta_float if hasattr(model, 'delta_float') else 0.5
    num_spread_payments = int(model.tenor / delta_float) if hasattr(model, 'tenor') else 2
    
    print(f"   Delta (float leg): {delta_float}")
    print(f"   Number of spread payments: {num_spread_payments}")
    
    # Each spread payment contributes: e^{-α*t} * [b2(t) + Tr(a2(t)*x)]
    # At t=0, a2(0) = u2, b2(0) = 0
    # So first payment contributes: Tr(u2*x0) = x0[1,1] (since u2 = e22)
    
    print(f"\n   First spread payment contribution to a3:")
    print(f"   a2(0) = u2 = [[0,0],[0,1]]")
    print(f"   This adds 1.0 to a3[1,1]")
    print(f"")
    print(f"   But x0[1,1] = {x0[1,1]:.8f}")
    print(f"   So Tr(a3*x0) only gets +{x0[1,1]:.8f} from spread")
    
    # Compare spread contribution to OIS contribution
    ois_contribution = x0[0,0] * 0.7  # approximate a3[0,0] * x0[0,0]
    spread_contribution = x0[1,1] * 1.0  # a3[1,1] * x0[1,1]
    
    print(f"\n5. RELATIVE CONTRIBUTIONS")
    print("-" * 50)
    print(f"   OIS contribution ≈ a3[0,0]*x0[0,0] ≈ 0.7*{x0[0,0]:.4f} = {ois_contribution:.6f}")
    print(f"   Spread contribution = a3[1,1]*x0[1,1] = 1.0*{x0[1,1]:.6f} = {spread_contribution:.6f}")
    print(f"   Ratio (OIS/Spread): {ois_contribution/spread_contribution:.1f}x")
    print(f"")
    print(f"   ⚠️  The spread contribution is {spread_contribution/ois_contribution*100:.1f}% of OIS contribution")
    print(f"   This seems too small for a meaningful multi-curve model!")
    
    print(f"\n6. RECOMMENDATION")
    print("-" * 50)
    print(f"   If market spreads are ~30-50 bps, then x0[1,1] should be larger.")
    print(f"   Current: x0[1,1] = {x0[1,1]:.6f}")
    print(f"")
    print(f"   The spread term in the swap payoff:")
    print(f"   sum_j [e^{{-α*T_j}} * A(T_j)] where A depends on Tr(u2*x)")
    print(f"")
    print(f"   With x0[1,1] so small, spreads contribute almost nothing!")
    
    print(f"\n7. ALTERNATIVE: CHECK IF is_spread SHOULD BE FALSE")
    print("-" * 50)
    print(f"   Current is_spread = {model.is_spread}")
    print(f"")
    print(f"   If swaptions are on OIS swaps (not IBOR), then is_spread should be False")
    print(f"   because OIS swaps don't have IBOR-OIS spread in the floating leg.")
    print(f"")
    print(f"   The ATM swaption market data you have - is it for:")
    print(f"   a) OIS swaptions (floating leg = OIS rate) → is_spread = False")
    print(f"   b) IBOR swaptions (floating leg = IBOR rate) → is_spread = True")
    
    print("=" * 70)
    
    return {
        'x0_11': x0[1,1],
        'spot_spread': spot_spread,
        'ois_contribution': ois_contribution,
        'spread_contribution': spread_contribution
    }


def test_swaption_without_spread(calibrator):
    """
    Test swaption pricing with is_spread = False.
    """
    model = calibrator.model
    
    print("\n" + "=" * 70)
    print("TEST: SWAPTION PRICING WITH is_spread = False")
    print("=" * 70)
    
    # Save original
    original_is_spread = model.is_spread
    
    # Get first swaption
    first_swaption = calibrator.daily_data.swaption_data_cube.iloc[0]["Object"]
    model.maturity = first_swaption.expiry_maturity
    model.tenor = first_swaption.swap_tenor_maturity
    model.strike = first_swaption.strike
    
    # Test with is_spread = True
    model.is_spread = True
    model.compute_b3_a3()
    b3_with = float(model.b3)
    a3_with = np.array(model.a3).copy()
    Y_with = b3_with + np.trace(a3_with @ np.array(model.x0))
    
    print(f"\n1. WITH is_spread = True:")
    print(f"   b3 = {b3_with:.8f}")
    print(f"   a3 = \n{a3_with}")
    print(f"   Y at x0 = {Y_with:.8f}")
    
    # Test with is_spread = False
    model.is_spread = False
    model.compute_b3_a3()
    b3_without = float(model.b3)
    a3_without = np.array(model.a3).copy()
    Y_without = b3_without + np.trace(a3_without @ np.array(model.x0))
    
    print(f"\n2. WITH is_spread = False:")
    print(f"   b3 = {b3_without:.8f}")
    print(f"   a3 = \n{a3_without}")
    print(f"   Y at x0 = {Y_without:.8f}")
    
    print(f"\n3. COMPARISON:")
    print(f"   a3[1,1] with spread:    {a3_with[1,1]:.6f}")
    print(f"   a3[1,1] without spread: {a3_without[1,1]:.6f}")
    print(f"   Difference:             {a3_with[1,1] - a3_without[1,1]:.6f}")
    print(f"")
    print(f"   Y with spread:    {Y_with:.8f}")
    print(f"   Y without spread: {Y_without:.8f}")
    
    # Now check against actual swap value
    annuity, swap_rate = model.compute_annuity()
    expected_swap_value = (float(swap_rate) - model.strike) * float(annuity)
    
    u1 = np.array(model.u1)
    x0 = np.array(model.x0)
    norm = 1 + np.trace(u1 @ x0)
    discount = np.exp(-model.alpha * model.maturity)
    
    adjusted_Y_without = Y_without * discount / norm
    
    print(f"\n4. SWAP VALUE CHECK (is_spread=False):")
    print(f"   Expected swap value = (F-K)*Ann = {expected_swap_value:.8f}")
    print(f"   Y*disc/norm = {Y_without:.8f}*{discount:.6f}/{norm:.6f} = {adjusted_Y_without:.8f}")
    print(f"   Match: {'✅' if abs(adjusted_Y_without - expected_swap_value) < 0.001 else '❌'}")
    
    # Restore
    model.is_spread = original_is_spread
    print(f"\n✅ Restored is_spread to {original_is_spread}")
    
    print("=" * 70)


# Usage:
# diagnose_spread_calibration(calibrator)
# test_swaption_without_spread(calibrator)
