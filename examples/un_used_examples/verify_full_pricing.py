"""
Verify the full swaption pricing chain.

Now that we understand:
- Y(x0) ≠ forward swap value (because x_T0 ≠ x0)
- Y(E[x_T0]) matches the forward swap value exactly!
- The b3/a3 formula IS correct

Let's verify the full pricing and implied vol extraction.
"""

import numpy as np
from scipy.stats import norm


def verify_full_pricing_chain(calibrator, swaption_index=0):
    """
    Verify the full swaption pricing chain.
    """
    model = calibrator.model
    
    print("=" * 70)
    print("FULL SWAPTION PRICING VERIFICATION")
    print("=" * 70)
    
    # Get first swaption
    first_swaption = calibrator.daily_data.swaption_data_cube.iloc[swaption_index]["Object"]
    
    # Set model parameters
    model.maturity = first_swaption.expiry_maturity
    model.tenor = first_swaption.swap_tenor_maturity
    model.strike = first_swaption.strike  # Use market strike
    
    T0 = model.maturity
    K = model.strike
    
    print(f"\n1. SWAPTION PARAMETERS")
    print("-" * 50)
    print(f"   Expiry T0: {T0}")
    print(f"   Tenor: {model.tenor}")
    print(f"   Strike K: {K:.6f} ({K*100:.4f}%)")
    
    # Compute annuity and forward rate
    annuity, swap_rate = model.compute_annuity()
    annuity = float(annuity)
    swap_rate = float(swap_rate)
    
    print(f"   Forward rate F: {swap_rate:.6f} ({swap_rate*100:.4f}%)")
    print(f"   Annuity: {annuity:.8f}")
    print(f"   Moneyness (F-K): {(swap_rate - K)*10000:.2f} bps")
    
    print(f"\n2. MARKET DATA")
    print("-" * 50)
    market_vol = first_swaption.vol
    market_price = first_swaption.market_price
    
    print(f"   Market vol (Bachelier): {market_vol:.6f} ({market_vol*100:.2f}%)")
    print(f"   Market price: {market_price:.8f}")
    
    # Verify market price from vol using Bachelier
    d = (swap_rate - K) / market_vol / np.sqrt(T0) if market_vol > 0 else 0
    bachelier_price = annuity * (
        (swap_rate - K) * norm.cdf(d) + 
        market_vol * np.sqrt(T0) * norm.pdf(d)
    )
    print(f"   Bachelier price (verification): {bachelier_price:.8f}")
    print(f"   Match: {'✅' if abs(bachelier_price - market_price) < 1e-6 else '❌'}")
    
    print(f"\n3. MODEL PRICE")
    print("-" * 50)
    
    # Get model price
    model.compute_b3_a3()
    
    # Price using model's method
    try:
        # Try to call the pricing function
        if hasattr(model, 'wishart') and hasattr(model.wishart, 'Phi_One'):
            # This should be the full pricing
            model_price = first_swaption.model_price
            print(f"   Model price (stored): {model_price:.8f}")
        else:
            model_price = first_swaption.model_price
            print(f"   Model price (from swaption object): {model_price:.8f}")
    except Exception as e:
        print(f"   Error getting model price: {e}")
        model_price = None
    
    # Extract model implied vol
    model_vol = first_swaption.model_vol
    print(f"   Model vol (Bachelier): {model_vol:.6f} ({model_vol*100:.2f}%)")
    
    print(f"\n4. COMPARISON")
    print("-" * 50)
    print(f"   Market vol: {market_vol:.6f} ({market_vol*100:.2f}%)")
    print(f"   Model vol:  {model_vol:.6f} ({model_vol*100:.2f}%)")
    print(f"   Vol ratio:  {model_vol/market_vol:.2f}x")
    print(f"")
    print(f"   Market price: {market_price:.8f}")
    print(f"   Model price:  {model_price:.8f}")
    print(f"   Price ratio:  {model_price/market_price:.2f}x")
    
    print(f"\n5. INTRINSIC VALUE CHECK")
    print("-" * 50)
    
    # Intrinsic value (zero vol)
    intrinsic = max(swap_rate - K, 0) * annuity
    print(f"   Intrinsic value: {intrinsic:.8f}")
    print(f"   Market price:    {market_price:.8f}")
    print(f"   Model price:     {model_price:.8f}")
    print(f"")
    print(f"   Market time value: {market_price - intrinsic:.8f}")
    print(f"   Model time value:  {model_price - intrinsic:.8f}")
    
    if model_price < intrinsic:
        print(f"\n   ⚠️  Model price < intrinsic value!")
        print(f"   This indicates a pricing formula issue.")
    elif model_price > market_price * 5:
        print(f"\n   ⚠️  Model price >> market price")
        print(f"   This could indicate:")
        print(f"   1. Sigma (volatility) parameters too high")
        print(f"   2. Incorrect vol extraction")
    
    print(f"\n6. DIAGNOSIS")
    print("-" * 50)
    
    # Check if the issue is with is_spread
    if model.is_spread:
        print(f"   is_spread = True")
        print(f"   The spread factor contributes to a3[1,1] = 1.0")
        print(f"   This may be adding extra volatility to the swaption")
        print(f"")
        print(f"   Consider: Are these OIS swaptions or IBOR swaptions?")
        print(f"   If OIS: set is_spread = False")
    
    # Check sigma values
    sigma = np.array(model.sigma)
    print(f"\n   Model sigma = \n{sigma}")
    print(f"   sigma * sigma^T = \n{sigma @ sigma.T}")
    
    # The vol of Y is driven by sigma through the Wishart dynamics
    # Higher sigma -> more variance in x_T0 -> more variance in Y -> higher swaption vol
    
    print("=" * 70)
    
    return {
        'market_vol': market_vol,
        'model_vol': model_vol,
        'market_price': market_price,
        'model_price': model_price,
        'intrinsic': intrinsic,
        'forward_rate': swap_rate,
        'strike': K
    }


def check_sigma_sensitivity(calibrator, sigma_factors=[0.001, 0.01, 0.1, 1.0]):
    """
    Check how sensitive the model vol is to sigma.
    """
    model = calibrator.model
    
    print("\n" + "=" * 70)
    print("SIGMA SENSITIVITY CHECK")
    print("=" * 70)
    
    # Get first swaption
    first_swaption = calibrator.daily_data.swaption_data_cube.iloc[0]["Object"]
    
    # Save original sigma
    original_sigma = np.array(model.sigma).copy()
    original_model_vol = first_swaption.model_vol
    
    print(f"\n   Original sigma = \n{original_sigma}")
    print(f"   Original model vol = {original_model_vol:.6f} ({original_model_vol*100:.2f}%)")
    
    print(f"\n   Testing different sigma scales:")
    print(f"   {'Factor':<10} {'sigma[0,0]':<12} {'Model Vol':<12} {'Ratio':<10}")
    print(f"   {'-'*10} {'-'*12} {'-'*12} {'-'*10}")
    
    for factor in sigma_factors:
        # Scale sigma
        new_sigma = original_sigma * factor
        
        # Update model (this is tricky - need to know the right method)
        # For now, just report what we'd test
        print(f"   {factor:<10.4f} {new_sigma[0,0]:<12.6f} {'(test needed)':<12} {'--':<10}")
    
    print(f"\n   To test: reduce sigma and re-run calibration")
    print("=" * 70)


# Usage:
# verify_full_pricing_chain(calibrator)
# check_sigma_sensitivity(calibrator)
