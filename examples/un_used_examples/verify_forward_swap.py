"""
Diagnostic to verify forward vs spot swap calculation.

Key insight: 
- compute_b3_a3() computes coefficients for a swap STARTING at T0 (forward swap)
- compute_annuity() seems to compute for a swap starting NOW (spot swap)

We need to verify this and check consistency.
"""

import numpy as np


def verify_forward_vs_spot_swap(calibrator):
    """
    Verify the relationship between b3/a3 and the forward swap value.
    """
    model = calibrator.model
    
    print("=" * 70)
    print("FORWARD VS SPOT SWAP VERIFICATION")
    print("=" * 70)
    
    # Get first swaption parameters
    first_swaption = calibrator.daily_data.swaption_data_cube.iloc[0]["Object"]
    model.maturity = first_swaption.expiry_maturity
    model.tenor = first_swaption.swap_tenor_maturity
    model.strike = first_swaption.strike
    
    T0 = model.maturity  # swaption expiry
    tenor = model.tenor
    K = model.strike
    alpha = model.alpha
    delta_fixed = model.delta_fixed
    delta_float = model.delta_float if hasattr(model, 'delta_float') else 0.5
    
    x0 = np.array(model.x0)
    u1 = np.array(model.u1)
    u2 = np.array(model.u2)
    
    print(f"\n1. PARAMETERS")
    print("-" * 50)
    print(f"   T0 (swaption expiry): {T0}")
    print(f"   Tenor: {tenor}")
    print(f"   K (strike): {K:.6f}")
    print(f"   α: {alpha:.6f}")
    
    # Compute b3, a3
    model.compute_b3_a3()
    b3 = float(model.b3)
    a3 = np.array(model.a3)
    
    print(f"\n2. COMPUTED b3, a3")
    print("-" * 50)
    print(f"   b3 = {b3:.8f}")
    print(f"   a3[0,0] = {a3[0,0]:.8f}")
    print(f"   a3[1,1] = {a3[1,1]:.8f}")
    
    # Compute Y at x0
    Y = b3 + np.trace(a3 @ x0)
    print(f"   Y = b3 + Tr(a3*x0) = {Y:.8f}")
    
    print(f"\n3. MANUAL FORWARD SWAP VALUE CALCULATION")
    print("-" * 50)
    
    # For a forward-starting swap at T0:
    # At time t=0, the forward value is:
    # E[Swap_T0] discounted = P(0,T0) - P(0,T0+tenor) + spreads_fwd - K*δ*sum(P(0,t_i))
    
    # Bond prices from t=0
    P_T0 = float(model.bond(T0))
    P_Tn = float(model.bond(T0 + tenor))
    
    print(f"   P(0, T0={T0}) = {P_T0:.8f}")
    print(f"   P(0, T0+tenor={T0+tenor}) = {P_Tn:.8f}")
    
    # Floating leg: P(0,T0) - P(0,T0+tenor)
    floating_leg = P_T0 - P_Tn
    print(f"   Floating leg (no spread) = P(0,T0) - P(0,Tn) = {floating_leg:.8f}")
    
    # Add spread if applicable
    spread_sum = 0.0
    if model.is_spread:
        n_float = int(tenor / delta_float)
        print(f"\n   Spread payments ({n_float} payments):")
        for j in range(n_float):
            t_j = T0 + j * delta_float
            spread_tj = float(model.spread(t_j))
            spread_sum += spread_tj
            print(f"   t={t_j}: spread = {spread_tj:.8f}")
        print(f"   Total spread = {spread_sum:.8f}")
    
    floating_leg_with_spread = floating_leg + spread_sum
    print(f"\n   Floating leg (with spread) = {floating_leg_with_spread:.8f}")
    
    # Fixed leg: K*δ*sum(P(0,t_i))
    n_fixed = int(tenor / delta_fixed)
    fixed_leg = 0.0
    annuity = 0.0
    print(f"\n   Fixed leg payments ({n_fixed} payments):")
    for i in range(1, n_fixed + 1):
        t_i = T0 + i * delta_fixed
        P_ti = float(model.bond(t_i))
        fixed_leg += K * delta_fixed * P_ti
        annuity += delta_fixed * P_ti
        print(f"   t={t_i}: K*δ*P = {K * delta_fixed * P_ti:.8f}")
    
    print(f"   Total fixed leg = {fixed_leg:.8f}")
    print(f"   Annuity = {annuity:.8f}")
    
    # Forward swap value
    swap_value_fwd = floating_leg_with_spread - fixed_leg
    print(f"\n   Forward swap value = Floating - Fixed = {swap_value_fwd:.8f}")
    
    # Forward swap rate
    swap_rate_fwd = floating_leg_with_spread / annuity
    print(f"   Forward swap rate = {swap_rate_fwd:.6f} ({swap_rate_fwd*100:.4f}%)")
    
    print(f"\n4. COMPARING WITH MODEL'S compute_annuity()")
    print("-" * 50)
    
    model_annuity, model_swap_rate = model.compute_annuity()
    print(f"   Model annuity: {float(model_annuity):.8f}")
    print(f"   Model swap rate: {float(model_swap_rate):.6f} ({float(model_swap_rate)*100:.4f}%)")
    print(f"")
    print(f"   Manual annuity: {annuity:.8f}")
    print(f"   Manual swap rate: {swap_rate_fwd:.6f}")
    
    if abs(float(model_annuity) - annuity) > 0.0001:
        print(f"\n   ⚠️  Annuity mismatch!")
    if abs(float(model_swap_rate) - swap_rate_fwd) > 0.0001:
        print(f"\n   ⚠️  Swap rate mismatch!")
    
    print(f"\n5. RELATIONSHIP BETWEEN Y AND SWAP VALUE")
    print("-" * 50)
    
    # According to the paper, the swaption price is:
    # Π = exp(-α*T0) / (1 + Tr(u1*x0)) * E[(Y_T0)^+]
    # 
    # And the swap value at T0 is:
    # Swap_T0 = Y_T0 / (1 + Tr(u1*x_T0)) * exp(-α*T0)
    #
    # Wait - this is confusing. Let me re-read the paper.
    #
    # From Eq. (23): The swap value at time t is:
    # Π_t^swap = P(t,T0) - P(t,Tn) + spreads - K*δ*sum(P(t,ti))
    #
    # From Proposition 3.1: P(t,T) = exp(-α(T-t)) * [bar_b1(T-t) + Tr(a1(T-t)*x_t)] / [1 + Tr(u1*x_t)]
    #
    # So the swap value is a combination of such terms.
    #
    # From Proposition 3.4: The swaption price is:
    # Π_t^swaption = exp(-α(T0-t)) / (1+Tr(u1*x_t)) * E_t[(b3 + Tr(a3*x_T0))^+]
    #
    # This means Y_T0 = b3 + Tr(a3*x_T0) is the NUMERATOR of the swap value at T0!
    # i.e., (1+Tr(u1*x_T0)) * exp(α*T0) * Swap_T0 = Y_T0
    
    # At t=0, evaluating at x=x0:
    # Y = b3 + Tr(a3*x0)
    # This should equal the numerator of the swap value if we were at T0 with state x0
    
    # The forward swap value (from time 0) is:
    # swap_value_fwd = Floating - Fixed (computed above)
    
    # The relationship should be:
    # Y = swap_value_fwd * (1 + Tr(u1*x0)) / exp(-α*T0) * normalization_factor
    
    # Wait, let me think about this more carefully...
    # 
    # The key is that b3/a3 are computed as coefficients that give:
    # Y_T0 = b3 + Tr(a3*x_T0) = numerator of swap value at T0
    #
    # When we evaluate at x0 (current state), we get an approximation
    # that equals the numerator of the forward swap value.
    
    norm = 1 + np.trace(u1 @ x0)
    discount = np.exp(-alpha * T0)
    
    # The Y we computed should relate to swap_value_fwd as:
    # Y * discount / norm ≈ swap_value_fwd (if x_T0 ≈ x0)
    Y_adjusted = Y * discount / norm
    
    print(f"   Y = {Y:.8f}")
    print(f"   1 + Tr(u1*x0) = {norm:.6f}")
    print(f"   exp(-α*T0) = {discount:.6f}")
    print(f"")
    print(f"   Y * exp(-α*T0) / norm = {Y_adjusted:.8f}")
    print(f"   Forward swap value = {swap_value_fwd:.8f}")
    print(f"   Ratio: {swap_value_fwd / Y_adjusted:.2f}x")
    
    if abs(swap_value_fwd / Y_adjusted - 1) > 0.1:
        print(f"\n   ❌ SIGNIFICANT MISMATCH!")
        print(f"   The ratio should be ~1.0 if the formula is correct.")
    else:
        print(f"\n   ✅ Values match reasonably well.")
    
    print(f"\n6. CHECKING THE BOND PRICE FORMULA")
    print("-" * 50)
    
    # From Eq. (15): P(t,T) = exp(-α(T-t)) * [bar_b1(T-t) + Tr(a1(T-t)*x_t)] / [1 + Tr(u1*x_t)]
    
    # Let's verify bond(T0) manually
    bb1_T0, a1_T0 = model.compute_bar_b1_a1(T0)
    bb1_T0 = float(bb1_T0)
    a1_T0 = np.array(a1_T0)
    
    numerator = bb1_T0 + np.trace(a1_T0 @ x0)
    P_T0_manual = np.exp(-alpha * T0) * numerator / norm
    
    print(f"   bar_b1({T0}) = {bb1_T0:.8f}")
    print(f"   a1({T0}) = \n{a1_T0}")
    print(f"   Numerator = bar_b1 + Tr(a1*x0) = {numerator:.8f}")
    print(f"   P(0,T0) manual = exp(-α*T0) * num / norm = {P_T0_manual:.8f}")
    print(f"   P(0,T0) from bond() = {P_T0:.8f}")
    
    if abs(P_T0_manual - P_T0) > 1e-6:
        print(f"\n   ⚠️  Bond price mismatch!")
    else:
        print(f"\n   ✅ Bond price formula verified.")
    
    print("=" * 70)
    
    return {
        'Y': Y,
        'swap_value_fwd': swap_value_fwd,
        'Y_adjusted': Y_adjusted,
        'ratio': swap_value_fwd / Y_adjusted if Y_adjusted != 0 else None
    }


# Usage:
# verify_forward_vs_spot_swap(calibrator)
