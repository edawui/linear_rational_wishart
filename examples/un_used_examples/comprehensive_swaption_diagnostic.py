"""
Comprehensive diagnostic for LRW swaption pricing.

This diagnostic checks every step of the swaption pricing formula
according to the paper:

From Proposition 3.4 and Corollary 3.5:

Swaption Price = (e^{-α(T0-t)}) / (1 + Tr[u1*x_t]) * E_t[(Y_{T0})^+]

where:
- Y_{T0} = b3 + Tr[a3 * x_{T0}]
- E_t[(Y_{T0})^+] is computed via Fourier transform

The expectation is computed using Eq. (29):
E_t[(Y_{T0})^+] = (1/π) * ∫_0^∞ Re[Φ_Y(z_r - i*z) / (z_r - i*z)^2] dz

where Φ_Y(z) = e^{z*b3} * Φ(T0-t, z*a3, x_t)

and Φ is the moment generating function of the Wishart process (Eq. 9):
Φ(t, θ1, x0) = exp(Tr[a(t,θ1)*x0] + c(t,θ1))
"""

import numpy as np
import cmath
import math
from scipy import integrate as sp_i
from scipy import linalg as sp


def comprehensive_swaption_diagnostic(calibrator, verbose=True):
    """
    Run comprehensive diagnostic on swaption pricing.
    
    Parameters
    ----------
    calibrator : LRWCalibrator
        The calibrator object
    verbose : bool
        Whether to print detailed output
    """
    model = calibrator.model
    
    # Get first swaption
    first_swaption = calibrator.daily_data.swaption_data_cube.iloc[0]["Object"]
    calibrator.objectives._price_single_swaption(first_swaption)
    
    # Set model parameters for this swaption
    model.maturity = first_swaption.expiry_maturity
    model.tenor = first_swaption.swap_tenor_maturity
    model.strike = first_swaption.strike
    
    # Compute b3 and a3
    model.compute_b3_a3()
    
    print("=" * 70)
    print("COMPREHENSIVE SWAPTION PRICING DIAGNOSTIC")
    print("=" * 70)
    
    print(f"\n1. SWAPTION PARAMETERS")
    print("-" * 50)
    print(f"   Expiry (T0): {model.maturity} years")
    print(f"   Tenor: {model.tenor} years")
    print(f"   Strike (K): {model.strike:.6f} ({model.strike*100:.4f}%)")
    
    print(f"\n2. MODEL PARAMETERS")
    print("-" * 50)
    print(f"   α (alpha): {model.alpha:.6f}")
    print(f"   x0 = \n{np.array(model.x0)}")
    print(f"   u1 = \n{np.array(model.u1)}")
    print(f"   u2 = \n{np.array(model.u2)}")
    print(f"   is_spread: {model.is_spread}")
    
    print(f"\n3. COMPUTED COEFFICIENTS (b3, a3)")
    print("-" * 50)
    b3 = float(model.b3)
    a3 = np.array(model.a3)
    x0 = np.array(model.x0)
    u1 = np.array(model.u1)
    
    print(f"   b3 = {b3:.8f}")
    print(f"   a3 = \n{a3}")
    print(f"   Tr(a3) = {np.trace(a3):.8f}")
    print(f"   Tr(a3 @ x0) = {np.trace(a3 @ x0):.8f}")
    
    # Y_T0 at current state (this should relate to swap value)
    Y_at_x0 = b3 + np.trace(a3 @ x0)
    print(f"\n   Y at x0 = b3 + Tr(a3*x0) = {Y_at_x0:.8f}")
    
    print(f"\n4. INTERPRETATION OF Y_{'{T0}'}")
    print("-" * 50)
    print(f"   According to the paper, Y_T0 = b3 + Tr[a3 * x_T0]")
    print(f"   represents the swap value at time T0 (swaption expiry).")
    print(f"")
    print(f"   From Eq. (26-27), this should equal:")
    print(f"   Floating leg - Fixed leg")
    print(f"   = (1 - P(T0,Tn)) + sum(spreads) - K*δ*sum(P(T0,ti))")
    print(f"")
    print(f"   At current time (t=0), with x = x0:")
    print(f"   Y at x0 = {Y_at_x0:.8f}")
    
    # Compare with actual swap value
    annuity, swap_rate = model.compute_annuity()
    swap_value_formula = (swap_rate - model.strike) * annuity
    
    print(f"\n   Computed swap rate: {float(swap_rate):.6f} ({float(swap_rate)*100:.4f}%)")
    print(f"   Strike: {model.strike:.6f} ({model.strike*100:.4f}%)")
    print(f"   Annuity: {float(annuity):.8f}")
    print(f"   Swap value (F-K)*Annuity: {swap_value_formula:.8f}")
    
    print(f"\n5. DISCREPANCY CHECK")
    print("-" * 50)
    
    # The normalization factor
    norm_factor = 1 + np.trace(u1 @ x0)
    discount = math.exp(-model.alpha * model.maturity)
    
    print(f"   1 + Tr(u1*x0) = {norm_factor:.8f}")
    print(f"   exp(-α*T0) = {discount:.8f}")
    
    # According to paper, swap value at t=0 should be:
    # Π_t^swap = [Y / (1 + Tr[u1*x_t])] * exp(-α*T0) evaluated at t=T0
    # But at t=0, the expected swap value is:
    # E[Y_T0] / (1 + Tr[u1*x0]) * exp(-α*T0)
    
    # If E[Y_T0] ≈ Y_x0 (no volatility), then:
    adjusted_Y = Y_at_x0 * discount / norm_factor
    
    print(f"\n   If Y_T0 ≈ Y_x0 (zero vol approximation):")
    print(f"   Adjusted swap value = Y*exp(-αT0)/(1+Tr[u1*x0])")
    print(f"                       = {Y_at_x0:.8f} * {discount:.6f} / {norm_factor:.6f}")
    print(f"                       = {adjusted_Y:.8f}")
    print(f"   Actual swap value   = {swap_value_formula:.8f}")
    
    if abs(adjusted_Y - swap_value_formula) > 0.001:
        print(f"\n   ⚠️  DISCREPANCY: {abs(adjusted_Y - swap_value_formula):.8f}")
        print(f"   This suggests b3/a3 are not correctly representing the swap payoff.")
    else:
        print(f"\n   ✅ Values match closely!")
    
    print(f"\n6. INTRINSIC VALUE CHECK (Swaption with σ=0)")
    print("-" * 50)
    
    # For a payer swaption with zero volatility:
    # Price = max(Y, 0) * exp(-α*T0) / (1 + Tr[u1*x0])
    intrinsic_Y = max(Y_at_x0, 0)
    intrinsic_price = intrinsic_Y * discount / norm_factor
    
    # Or using standard formula:
    intrinsic_standard = max(swap_rate - model.strike, 0) * annuity
    
    print(f"   Y at x0 = {Y_at_x0:.8f}")
    print(f"   max(Y,0) = {intrinsic_Y:.8f}")
    print(f"   Intrinsic (model form) = max(Y,0)*exp(-αT0)/(1+Tr[u1*x0])")
    print(f"                          = {intrinsic_price:.8f}")
    print(f"")
    print(f"   Intrinsic (standard) = max(F-K,0)*Annuity")
    print(f"                        = max({float(swap_rate):.6f}-{model.strike:.6f},0)*{float(annuity):.6f}")
    print(f"                        = {intrinsic_standard:.8f}")
    
    print(f"\n7. SIGN ANALYSIS")
    print("-" * 50)
    
    if Y_at_x0 > 0:
        print(f"   Y > 0: Swaption is in-the-money at current state")
        print(f"   This means swap rate > strike (payer benefits)")
    elif Y_at_x0 < 0:
        print(f"   Y < 0: Swaption is out-of-the-money at current state")
        print(f"   This means swap rate < strike")
    else:
        print(f"   Y ≈ 0: Swaption is at-the-money")
    
    print(f"\n   For ATM swaption, strike should equal forward swap rate:")
    print(f"   Strike = {model.strike:.6f} ({model.strike*100:.4f}%)")
    print(f"   Forward swap rate = {float(swap_rate):.6f} ({float(swap_rate)*100:.4f}%)")
    print(f"   Moneyness (F-K) = {(float(swap_rate)-model.strike)*10000:.2f} bps")
    
    print(f"\n8. MARKET DATA COMPARISON")
    print("-" * 50)
    
    market_vol = first_swaption.vol
    market_price = first_swaption.market_price
    model_vol = first_swaption.model_vol
    model_price = first_swaption.model_price
    
    print(f"   Market vol: {market_vol:.6f} ({market_vol*100:.2f}%)")
    print(f"   Model vol:  {model_vol:.6f} ({model_vol*100:.2f}%)")
    print(f"   Ratio: {model_vol/market_vol:.2f}x")
    print(f"")
    print(f"   Market price: {market_price:.8f}")
    print(f"   Model price:  {model_price:.8f}")
    
    print(f"\n9. DIAGNOSIS")
    print("=" * 70)
    
    # Check if the issue is with b3/a3 representing swap value correctly
    if abs(adjusted_Y - swap_value_formula) > 0.001:
        print(f"   ❌ Issue: b3/a3 don't correctly represent swap value")
        print(f"      Expected swap value: {swap_value_formula:.8f}")
        print(f"      Y*discount/norm: {adjusted_Y:.8f}")
        print(f"")
        print(f"      Possible causes:")
        print(f"      1. Strike (K) not correctly incorporated in b3/a3")
        print(f"      2. Spread terms (if is_spread=True) may be wrong")
        print(f"      3. Normalization factor 1+Tr(u1*x0) issue")
    
    if Y_at_x0 < 0 and (swap_rate - model.strike) > 0:
        print(f"   ❌ Issue: Sign mismatch!")
        print(f"      Y < 0 but (F-K) > 0")
        print(f"      The swap is ITM but Y suggests OTM")
    
    if model_vol > market_vol * 5:
        print(f"   ❌ Issue: Model vol much higher than market")
        print(f"      This could indicate:")
        print(f"      1. Bachelier implied vol inversion issue")
        print(f"      2. Model price includes intrinsic that's being")
        print(f"         interpreted as volatility")
    
    print("=" * 70)
    
    return {
        'b3': b3,
        'a3': a3,
        'Y_at_x0': Y_at_x0,
        'swap_value': swap_value_formula,
        'swap_rate': float(swap_rate),
        'annuity': float(annuity),
        'intrinsic_price': intrinsic_price,
        'market_vol': market_vol,
        'model_vol': model_vol
    }


# Usage:
# result = comprehensive_swaption_diagnostic(calibrator)
