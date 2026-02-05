"""
Debug the b3/a3 sign issue.

We found:
- Forward swap value = +0.00619647 (positive, swap is ITM)
- Y = b3 + Tr(a3*x0) = -0.00412604 (NEGATIVE!)

This is a sign error somewhere in the formula.
"""

import numpy as np


def debug_b3_a3_sign(calibrator):
    """
    Debug the sign issue in b3/a3.
    """
    model = calibrator.model
    
    print("=" * 70)
    print("DEBUG: b3/a3 SIGN ISSUE")
    print("=" * 70)
    
    # Parameters
    first_swaption = calibrator.daily_data.swaption_data_cube.iloc[0]["Object"]
    model.maturity = first_swaption.expiry_maturity
    model.tenor = first_swaption.swap_tenor_maturity
    
    # Use model's ATM rate as strike
    annuity, swap_rate = model.compute_annuity()
    # model.strike = float(swap_rate)  # Use ATM
    model.strike = 0.020482  # Use the strike from your test
    
    T0 = model.maturity
    tenor = model.tenor
    K = model.strike
    alpha = model.alpha
    delta_fixed = model.delta_fixed
    
    x0 = np.array(model.x0)
    u1 = np.array(model.u1)
    norm = 1 + np.trace(u1 @ x0)
    
    print(f"\n1. PARAMETERS")
    print("-" * 50)
    print(f"   T0={T0}, tenor={tenor}, K={K:.6f}")
    print(f"   Forward rate F = {float(swap_rate):.6f}")
    print(f"   F - K = {(float(swap_rate) - K)*10000:.2f} bps")
    
    print(f"\n2. PAPER'S EQUATION 26-27 (Step by Step)")
    print("-" * 50)
    
    # From Eq. 26:
    # b3 = 1 - exp(-α*Tn)*bar_b1(Tn) + Σ exp(-α*Tj)*b2(Tj) - K*δ*Σ exp(-α*ti)*bar_b1(ti)
    #
    # From Eq. 27:
    # a3 = u1 - exp(-α*Tn)*a1(Tn) + Σ exp(-α*Tj)*a2(Tj) - K*δ*Σ exp(-α*ti)*a1(ti)
    
    # Breaking down: Y = b3 + Tr(a3*x0)
    # Y = [1 + Tr(u1*x0)]  <-- Initial
    #   - [exp(-α*Tn)*bar_b1(Tn) + exp(-α*Tn)*Tr(a1(Tn)*x0)]  <-- Floating end
    #   + [Σ exp(-α*Tj)*b2(Tj) + Σ exp(-α*Tj)*Tr(a2(Tj)*x0)]  <-- Spreads
    #   - [K*δ*Σ exp(-α*ti)*bar_b1(ti) + K*δ*Σ exp(-α*ti)*Tr(a1(ti)*x0)]  <-- Fixed
    
    # Initial term
    Y_init = 1 + np.trace(u1 @ x0)
    print(f"\n   Initial: 1 + Tr(u1*x0) = {Y_init:.8f}")
    
    # Floating leg end: -P_numerator(T0, Tn)
    bb1_tenor, a1_tenor = model.compute_bar_b1_a1(tenor)
    bb1_tenor = float(bb1_tenor)
    a1_tenor = np.array(a1_tenor)
    discount_tenor = np.exp(-alpha * tenor)
    
    float_end_num = discount_tenor * (bb1_tenor + np.trace(a1_tenor @ x0))
    print(f"\n   Floating end: -exp(-α*tenor)*[bar_b1 + Tr(a1*x0)]")
    print(f"   = -{discount_tenor:.6f} * ({bb1_tenor:.6f} + {np.trace(a1_tenor @ x0):.6f})")
    print(f"   = -{float_end_num:.8f}")
    
    # This should represent -P(T0, Tn) in the numerator form
    # P(T0, Tn) = exp(-α*tenor) * [bar_b1 + Tr(a1*x)] / [1+Tr(u1*x)]
    # So numerator = exp(-α*tenor) * [bar_b1 + Tr(a1*x)]
    
    P_Tn = float(model.bond(T0 + tenor))
    P_Tn_num_manual = P_Tn * norm / np.exp(-alpha * T0)
    print(f"\n   Verification:")
    print(f"   P(0, T0+tenor) = {P_Tn:.8f}")
    print(f"   P_numerator = P * norm / exp(-α*T0) = {P_Tn_num_manual:.8f}")
    print(f"   float_end_num = {float_end_num:.8f}")
    
    # Wait - there's a timing issue here!
    # P(0, T0+tenor) is the bond from time 0 to T0+tenor
    # But b3/a3 use bonds from time T0 (expiry) to T0+tenor
    # These are different!
    
    # Let me recalculate using the correct timing
    # P(T0, T0+tenor) with state x at T0 has numerator:
    # exp(-α*tenor) * [bar_b1(tenor) + Tr(a1(tenor)*x)]
    # Note: bar_b1 and a1 are functions of TIME DIFFERENCE (tenor), not absolute time!
    
    print(f"\n   NOTE: bar_b1({tenor}) and a1({tenor}) are functions of tenor, not T0!")
    print(f"   They represent the bond P(T0, T0+tenor) starting from T0")
    
    # Spreads
    spread_num = 0.0
    if model.is_spread:
        n_float = int(tenor / model.delta_float)
        print(f"\n   Spread payments:")
        for j in range(n_float):
            t_j = j * model.delta_float
            b2_tj, a2_tj = model.compute_b2_a2(t_j)
            b2_tj = float(b2_tj)
            a2_tj = np.array(a2_tj)
            discount_tj = np.exp(-alpha * t_j)
            
            spread_j_num = discount_tj * (b2_tj + np.trace(a2_tj @ x0))
            spread_num += spread_j_num
            print(f"   t={t_j}: exp(-α*t)*[b2+Tr(a2*x0)] = {spread_j_num:.8f}")
        print(f"   Total spread_num = {spread_num:.8f}")
    
    # Fixed leg
    n_fixed = int(tenor / delta_fixed)
    fixed_num = 0.0
    print(f"\n   Fixed leg payments:")
    for i in range(1, n_fixed + 1):
        t_i = i * delta_fixed
        bb1_ti, a1_ti = model.compute_bar_b1_a1(t_i)
        bb1_ti = float(bb1_ti)
        a1_ti = np.array(a1_ti)
        discount_ti = np.exp(-alpha * t_i)
        
        fixed_i_num = K * delta_fixed * discount_ti * (bb1_ti + np.trace(a1_ti @ x0))
        fixed_num += fixed_i_num
        print(f"   t={t_i}: K*δ*exp(-α*t)*[bar_b1+Tr(a1*x0)] = {fixed_i_num:.8f}")
    print(f"   Total fixed_num = {fixed_num:.8f}")
    
    # Total Y
    Y_computed = Y_init - float_end_num + spread_num - fixed_num
    
    print(f"\n3. COMPUTED Y")
    print("-" * 50)
    print(f"   Y = Initial - Float_end + Spreads - Fixed")
    print(f"     = {Y_init:.8f} - {float_end_num:.8f} + {spread_num:.8f} - {fixed_num:.8f}")
    print(f"     = {Y_computed:.8f}")
    
    # Compare with model
    model.compute_b3_a3()
    Y_model = float(model.b3) + np.trace(np.array(model.a3) @ x0)
    print(f"\n   Model Y = {Y_model:.8f}")
    print(f"   Match: {'✅' if abs(Y_computed - Y_model) < 1e-6 else '❌'}")
    
    print(f"\n4. WHAT Y SHOULD BE")
    print("-" * 50)
    
    # The swap value at T0 (with state x_T0) is:
    # Swap_T0 = [1 - P(T0,Tn) + Spreads - K*δ*ΣP(T0,ti)] 
    #
    # The numerator form (multiply by [1+Tr(u1*x)]):
    # Y = [1+Tr(u1*x)] * Swap_T0
    #   = [1+Tr(u1*x)] - [1+Tr(u1*x)]*P(T0,Tn) + [1+Tr(u1*x)]*Spreads - K*δ*[1+Tr(u1*x)]*ΣP(T0,ti)
    #
    # Using P(T,T') = exp(-α*(T'-T)) * [bar_b1 + Tr(a1*x)] / [1+Tr(u1*x)]:
    # [1+Tr(u1*x)] * P(T0,T') = exp(-α*(T'-T0)) * [bar_b1(T'-T0) + Tr(a1(T'-T0)*x)]
    
    # So Y = [1+Tr(u1*x)] - exp(-α*tenor)*[bar_b1(tenor)+Tr(a1(tenor)*x)] 
    #      + exp(-α*tj)*[b2(tj)+Tr(a2(tj)*x)] (for spreads)
    #      - K*δ*Σ exp(-α*ti)*[bar_b1(ti)+Tr(a1(ti)*x)]
    
    # Wait, this is exactly what we computed! So the formula seems right...
    
    # Let me verify differently: compute swap value directly
    P_T0 = float(model.bond(T0))  # P(0, T0)
    P_Tn_full = float(model.bond(T0 + tenor))  # P(0, T0+tenor)
    
    # Forward swap value from time 0:
    float_leg_fwd = P_T0 - P_Tn_full  # Without spread
    
    # Add forward spreads
    spread_fwd = 0.0
    if model.is_spread:
        for j in range(int(tenor / model.delta_float)):
            t_j = T0 + j * model.delta_float
            spread_fwd += float(model.spread(t_j))
    
    float_leg_fwd_with_spread = float_leg_fwd + spread_fwd
    
    # Fixed leg
    fixed_fwd = 0.0
    for i in range(1, int(tenor / delta_fixed) + 1):
        t_i = T0 + i * delta_fixed
        fixed_fwd += K * delta_fixed * float(model.bond(t_i))
    
    swap_value_fwd = float_leg_fwd_with_spread - fixed_fwd
    
    print(f"   Forward swap value (from t=0) = {swap_value_fwd:.8f}")
    
    # The relationship between Y and swap value:
    # At time T0 with state x_T0:
    # Swap_T0 = Y_T0 / [1+Tr(u1*x_T0)]
    #
    # The forward swap value at t=0:
    # Swap_fwd = P(0,T0) * E[Swap_T0]
    # 
    # For a first approximation (x_T0 ≈ x0):
    # Swap_fwd ≈ P(0,T0) * Y(x0) / [1+Tr(u1*x0)]
    
    # NO WAIT! The swaption formula is:
    # Π_swaption = exp(-α*T0) / [1+Tr(u1*x0)] * E[(Y_T0)^+]
    #
    # This suggests that the expected swap value at T0 is:
    # E[Swap_T0] = E[Y_T0] / [1+Tr(u1*x_T0)]
    #
    # At t=0, the forward swap value should be:
    # Swap_fwd = P(0,T0) * E[Swap_T0]
    #          = exp(-α*T0) / [1+Tr(u1*x0)] * E[Y_T0 / (1+Tr(u1*x_T0)) * (1+Tr(u1*x_T0))]
    # 
    # Hmm, this is getting complex. Let's just verify numerically.
    
    # If x_T0 = x0 (zero vol):
    swap_T0_zero_vol = Y_model / norm
    swap_fwd_zero_vol = np.exp(-alpha * T0) * swap_T0_zero_vol
    
    print(f"\n   Zero-vol approximation:")
    print(f"   Y = {Y_model:.8f}")
    print(f"   Swap_T0 = Y / norm = {swap_T0_zero_vol:.8f}")
    print(f"   Swap_fwd = exp(-α*T0) * Swap_T0 = {swap_fwd_zero_vol:.8f}")
    print(f"")
    print(f"   Actual forward swap value = {swap_value_fwd:.8f}")
    print(f"   Ratio: {swap_value_fwd / swap_fwd_zero_vol if swap_fwd_zero_vol != 0 else 'N/A'}")
    
    print(f"\n5. THE REAL ISSUE")
    print("-" * 50)
    
    # Let me compute what Y SHOULD be to match the forward swap value
    # Swap_fwd = exp(-α*T0) / norm * Y
    # Y = Swap_fwd * norm / exp(-α*T0)
    
    Y_expected = swap_value_fwd * norm / np.exp(-alpha * T0)
    
    print(f"   For Swap_fwd = {swap_value_fwd:.8f}")
    print(f"   Y should be = Swap_fwd * norm / exp(-α*T0)")
    print(f"               = {swap_value_fwd:.8f} * {norm:.6f} / {np.exp(-alpha*T0):.6f}")
    print(f"               = {Y_expected:.8f}")
    print(f"")
    print(f"   Actual Y = {Y_model:.8f}")
    print(f"   Difference = {Y_expected - Y_model:.8f}")
    
    # Let me check the individual terms
    print(f"\n6. TERM-BY-TERM COMPARISON")
    print("-" * 50)
    
    # Expected floating leg contribution to Y:
    # = norm * (P_T0 - P_Tn) = norm * float_leg_fwd (without spread)
    float_Y_expected = norm * float_leg_fwd
    float_Y_actual = Y_init - float_end_num
    
    print(f"   Floating leg (no spread):")
    print(f"   Expected Y contribution = norm * (P_T0 - P_Tn)")
    print(f"                           = {norm:.6f} * ({P_T0:.8f} - {P_Tn_full:.8f})")
    print(f"                           = {norm:.6f} * {float_leg_fwd:.8f}")
    print(f"                           = {float_Y_expected:.8f}")
    print(f"   Actual Y contribution   = {Y_init:.6f} - {float_end_num:.6f}")
    print(f"                           = {float_Y_actual:.8f}")
    print(f"   Ratio: {float_Y_actual / float_Y_expected:.4f}")
    
    # Hmm, let me think about this differently...
    # P_T0 = P(0, T0) = exp(-α*T0) * bar_b1(T0) / norm
    # P_Tn = P(0, T0+tenor) = exp(-α*(T0+tenor)) * bar_b1(T0+tenor) / norm
    #
    # float_leg_fwd = P_T0 - P_Tn
    #               = [exp(-α*T0)*bar_b1(T0) - exp(-α*(T0+tenor))*bar_b1(T0+tenor)] / norm
    #
    # float_Y_expected = norm * float_leg_fwd
    #                  = exp(-α*T0)*bar_b1(T0) - exp(-α*(T0+tenor))*bar_b1(T0+tenor)
    
    # But in b3/a3 formula:
    # Initial = 1 + Tr(u1*x0) = norm
    # Float_end = exp(-α*tenor) * [bar_b1(tenor) + Tr(a1(tenor)*x0)]
    # 
    # float_Y_actual = norm - exp(-α*tenor)*[bar_b1(tenor)+Tr(a1(tenor)*x0)]
    
    # The issue: tenor vs T0+tenor!
    # In b3/a3: exp(-α*tenor) * bar_b1(tenor)
    # In forward: exp(-α*(T0+tenor)) * bar_b1(T0+tenor) / norm
    
    # WAIT! The b3/a3 formula uses tenor (relative time from T0)
    # But bar_b1(t) is E[x_t | x_0], not E[x_{T0+t} | x_{T0}]
    
    # The b3/a3 formula assumes we're at time T0 looking forward!
    # So bar_b1(tenor) = E[x_{tenor} | x_0] where x_0 is the state at T0
    
    # But when we evaluate at x0 (current state at t=0), we're pretending
    # the state at T0 is also x0, which gives us the zero-vol approximation.
    
    print(f"\n7. KEY INSIGHT")
    print("-" * 50)
    print(f"   The b3/a3 formula computes swap value at T0, not forward swap value!")
    print(f"")
    print(f"   At T0 with state x_T0:")
    print(f"   - Floating leg = P(T0,T0) - P(T0,T0+tenor) = 1 - P(T0,Tn)")
    print(f"   - P(T0,Tn) = exp(-α*tenor) * bar_b1(tenor) / [1+Tr(u1*x_T0)]")
    print(f"")
    print(f"   Y = [1+Tr(u1*x)] * Swap_T0")
    print(f"")
    print(f"   If x_T0 = x0 (zero-vol):")
    print(f"   Swap_T0 = Y(x0) / norm")
    print(f"")
    print(f"   Forward swap value from t=0:")
    print(f"   Swap_fwd = P(0,T0) - P(0,Tn) + spreads - K*δ*ΣP(0,ti)")
    print(f"")
    print(f"   These are DIFFERENT things!")
    print(f"   - Y(x0)/norm = swap value IF state at T0 is x0")
    print(f"   - Swap_fwd = forward swap value observed at t=0")
    
    # The forward swap value uses P(0, T) bonds
    # The b3/a3 formula uses P(T0, T') bonds
    # These differ by a factor related to P(0, T0) and the forward measure
    
    # Let me verify: P(0,T) = P(0,T0) * E^{T0}[P(T0,T)]
    # where E^{T0} is the T0-forward measure
    
    # Under zero-vol, E^{T0}[P(T0,T)] = P(T0,T) evaluated at the expected state
    # The expected state at T0 (under risk-neutral) evolves from x0
    
    print(f"\n   The relationship is:")
    print(f"   P(0,T) = P(0,T0) * P^{fwd}(T0,T)")
    print(f"   where P^{fwd}(T0,T) is the forward bond price")
    print(f"")
    print(f"   P(0,T0) = {P_T0:.8f}")
    print(f"   P(0,Tn) = {P_Tn_full:.8f}")
    print(f"   P^{fwd}(T0,Tn) = P(0,Tn)/P(0,T0) = {P_Tn_full/P_T0:.8f}")
    
    print("=" * 70)


# Usage:
# debug_b3_a3_sign(calibrator)
