"""
Check if the discrepancy is due to state evolution.

Hypothesis: Y(x0) ≠ E[Y(x_T0)] because x_T0 ≠ x0

The Wishart process evolves: E[x_T0] = f(x0, omega, m, T0)
"""

import numpy as np


def check_state_evolution(calibrator):
    """
    Check the expected state at T0 and its impact on Y.
    """
    model = calibrator.model
    
    print("=" * 70)
    print("STATE EVOLUTION CHECK")
    print("=" * 70)
    
    # Parameters
    first_swaption = calibrator.daily_data.swaption_data_cube.iloc[0]["Object"]
    model.maturity = first_swaption.expiry_maturity
    model.tenor = first_swaption.swap_tenor_maturity
    model.strike = 0.020482  # From your test
    
    T0 = model.maturity
    x0 = np.array(model.x0)
    u1 = np.array(model.u1)
    
    print(f"\n1. CURRENT STATE (x0)")
    print("-" * 50)
    print(f"   x0 = \n{x0}")
    print(f"   Tr(u1*x0) = {np.trace(u1 @ x0):.8f}")
    
    print(f"\n2. EXPECTED STATE AT T0")
    print("-" * 50)
    
    # For Wishart: E[x_t | x_0] = vec^{-1}(e^{At} vec(x_0) + A^{-1}(e^{At} - I) vec(omega))
    # This is computed by wishart.compute_mean_wishart(T0)
    
    expected_x_T0_vec = model.wishart.compute_mean_wishart(T0)
    expected_x_T0 = np.array(expected_x_T0_vec).reshape((model.n, model.n)).T
    
    print(f"   E[x_T0 | x0] = \n{expected_x_T0}")
    print(f"   Tr(u1*E[x_T0]) = {np.trace(u1 @ expected_x_T0):.8f}")
    
    # Change in x
    delta_x = expected_x_T0 - x0
    print(f"\n   Change: E[x_T0] - x0 = \n{delta_x}")
    
    print(f"\n3. IMPACT ON Y")
    print("-" * 50)
    
    # Compute b3, a3
    model.compute_b3_a3()
    b3 = float(model.b3)
    a3 = np.array(model.a3)
    
    Y_at_x0 = b3 + np.trace(a3 @ x0)
    Y_at_E_xT0 = b3 + np.trace(a3 @ expected_x_T0)
    
    print(f"   b3 = {b3:.8f}")
    print(f"   a3 = \n{a3}")
    print(f"")
    print(f"   Y(x0) = b3 + Tr(a3*x0) = {Y_at_x0:.8f}")
    print(f"   Y(E[x_T0]) = b3 + Tr(a3*E[x_T0]) = {Y_at_E_xT0:.8f}")
    print(f"   Difference = {Y_at_E_xT0 - Y_at_x0:.8f}")
    
    print(f"\n4. FORWARD SWAP VALUE COMPARISON")
    print("-" * 50)
    
    # Forward swap value (computed earlier)
    annuity, swap_rate = model.compute_annuity()
    K = model.strike
    swap_value_fwd = (float(swap_rate) - K) * float(annuity)
    
    print(f"   Forward swap value = {swap_value_fwd:.8f}")
    
    # What Y should be to match forward swap value:
    # Swaption formula: price = exp(-α*T0) / (1+Tr(u1*x0)) * E[(Y_T0)^+]
    # For ATM-ish swaption, E[(Y_T0)^+] ≈ E[Y_T0] + time_value
    # E[Y_T0] should relate to forward swap value
    
    norm_x0 = 1 + np.trace(u1 @ x0)
    norm_E_xT0 = 1 + np.trace(u1 @ expected_x_T0)
    discount = np.exp(-model.alpha * T0)
    
    # Swap_T0 = Y_T0 / (1 + Tr(u1*x_T0))
    # E[Swap_T0] ≈ E[Y_T0] / E[1 + Tr(u1*x_T0)]  (Jensen's inequality, approximate)
    
    E_swap_T0_approx = Y_at_E_xT0 / norm_E_xT0
    
    print(f"\n   Using expected state E[x_T0]:")
    print(f"   1 + Tr(u1*E[x_T0]) = {norm_E_xT0:.8f}")
    print(f"   Y(E[x_T0]) / norm_E_xT0 = {E_swap_T0_approx:.8f}")
    
    # The forward swap value at t=0 should be:
    # Swap_fwd ≈ P(0,T0) * E[Swap_T0]
    # Let's check if this matches
    
    P_T0 = float(model.bond(T0))
    swap_fwd_from_Y = P_T0 * E_swap_T0_approx
    
    print(f"\n   P(0, T0) = {P_T0:.8f}")
    print(f"   P(0,T0) * E[Swap_T0] ≈ {swap_fwd_from_Y:.8f}")
    print(f"   Actual forward swap value = {swap_value_fwd:.8f}")
    print(f"   Ratio: {swap_value_fwd / swap_fwd_from_Y:.4f}")
    
    print(f"\n5. MORE ACCURATE APPROXIMATION")
    print("-" * 50)
    
    # Actually, the forward swap value is NOT P(0,T0)*E[Swap_T0]!
    # 
    # The forward swap value at t=0 is:
    # Swap_fwd = P(0,T0) - P(0,Tn) + spreads - K*δ*ΣP(0,ti)
    #
    # Each term involves bonds from t=0 to various maturities.
    # These are computed using x0, not E[x_T0].
    #
    # The b3/a3 formula computes the swap payoff at T0 in terms of x_T0.
    # The E[(Y_T0)^+] in the swaption formula is the expected payoff.
    
    # Let me verify the direct relationship:
    # At T0, Swap_T0 = [Float - Fixed] where each term uses P(T0, T)
    # 
    # Float = 1 - P(T0, Tn) = 1 - exp(-α*tenor)*[bar_b1+Tr(a1*x_T0)]/(1+Tr(u1*x_T0))
    #
    # Multiply by (1+Tr(u1*x_T0)):
    # (1+Tr(u1*x_T0)) * Float = (1+Tr(u1*x_T0)) - exp(-α*tenor)*[bar_b1+Tr(a1*x_T0)]
    #
    # This equals the "floating contribution" to Y_T0.
    #
    # So: (1+Tr(u1*x_T0)) * Swap_T0 = Y_T0
    # => Y_T0 = (1+Tr(u1*x_T0)) * Swap_T0
    
    # Now, Swap_fwd (forward swap value at t=0) is:
    # Swap_fwd = P(0,T0) * E^{T0}[Swap_T0] under T0-forward measure
    #
    # Under the forward measure, the numeraire is P(t,T0).
    # This complicates the expectation.
    
    # For now, let's just verify numerically that Y(E[x_T0]) is closer
    # to (1+Tr(u1*E[x_T0])) * expected_swap_value
    
    # Actually, let's compute what happens if we ignore the convexity adjustment:
    # Under "naive" approximation: E[Y] ≈ Y(E[x])
    
    print(f"   Y(E[x_T0]) = {Y_at_E_xT0:.8f}")
    print(f"   norm(E[x_T0]) = {norm_E_xT0:.8f}")
    print(f"   Implied Swap_T0 = Y / norm = {Y_at_E_xT0 / norm_E_xT0:.8f}")
    
    # For the forward swap value, we need to discount:
    # Swap_fwd ≈ exp(-α*T0) * Y / norm_x0  (using current state for norm)
    swap_fwd_approx = discount * Y_at_E_xT0 / norm_x0
    
    print(f"\n   Approximation: Swap_fwd ≈ exp(-α*T0) * Y(E[x_T0]) / norm(x0)")
    print(f"                          = {discount:.6f} * {Y_at_E_xT0:.8f} / {norm_x0:.6f}")
    print(f"                          = {swap_fwd_approx:.8f}")
    print(f"   Actual forward swap value = {swap_value_fwd:.8f}")
    print(f"   Ratio: {swap_value_fwd / swap_fwd_approx:.4f}")
    
    print(f"\n6. CONCLUSION")
    print("-" * 50)
    
    if abs(swap_value_fwd / swap_fwd_approx - 1) < 0.15:
        print(f"   Using E[x_T0] instead of x0 brings Y much closer to expected!")
        print(f"   The remaining difference is likely due to:")
        print(f"   - Convexity adjustment (E[Y/norm] ≠ E[Y]/E[norm])")
        print(f"   - Jensen's inequality effects")
    else:
        print(f"   Using E[x_T0] doesn't fully explain the discrepancy.")
        print(f"   There may be a formula issue in b3/a3.")
    
    print("=" * 70)


# Usage:
# check_state_evolution(calibrator)
