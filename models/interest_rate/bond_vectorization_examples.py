# Usage Examples for Vectorized Bond Pricing

import numpy as np
import matplotlib.pyplot as plt
import time

# Initialize your LRW model (assuming model is already created)
# model = LRWModel(config, swaption_config)

# Example 1: Basic yield curve construction
print("Example 1: Yield Curve Construction")
print("=" * 50)

# Define standard yield curve maturities
yield_curve_maturities = np.array([
    0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0
])

# Compute bond prices and yields vectorized
bond_prices = model.bond_vectorized(yield_curve_maturities)
yields = model.compute_yield_curve(yield_curve_maturities)

print("Maturity | Bond Price | Yield")
print("-" * 35)
for i, (mat, price, yld) in enumerate(zip(yield_curve_maturities, bond_prices, yields)):
    print(f"{mat:8.2f} | {price:10.6f} | {yld:8.4f}")

# Example 2: Performance comparison
print("\nExample 2: Performance Comparison")
print("=" * 50)

# Large number of maturities for performance testing
large_maturities = np.linspace(0.1, 30.0, 1000)

# Time single bond pricing approach
start_time = time.time()
single_prices = []
for maturity in large_maturities:
    single_prices.append(model.bond(maturity))
single_time = time.time() - start_time

# Time vectorized approach
start_time = time.time()
vectorized_prices = model.bond_vectorized(large_maturities)
vectorized_time = time.time() - start_time

print(f"Number of bonds: {len(large_maturities)}")
print(f"Single pricing time: {single_time:.4f}s")
print(f"Vectorized pricing time: {vectorized_time:.4f}s")
print(f"Speedup: {single_time/vectorized_time:.2f}x")

# Verify results are identical
max_difference = np.max(np.abs(np.array(single_prices) - vectorized_prices))
print(f"Maximum difference: {max_difference:.2e}")

# Example 3: Forward rate curve
print("\nExample 3: Forward Rate Curve")
print("=" * 50)

forward_maturities = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
forward_rates = model.compute_forward_rates(forward_maturities)

print("Maturity | Forward Rate")
print("-" * 25)
for mat, fwd in zip(forward_maturities, forward_rates):
    print(f"{mat:8.2f} | {fwd:11.4f}")

# Example 4: Term structure analysis
print("\nExample 4: Term Structure Analysis")
print("=" * 50)

# Fine grid for smooth curves
fine_maturities = np.linspace(0.1, 20.0, 200)
fine_bond_prices = model.bond_vectorized(fine_maturities)
fine_yields = model.compute_yield_curve(fine_maturities)
fine_forwards = model.compute_forward_rates(fine_maturities)

# Find specific points of interest
def find_nearest_index(array, value):
    return np.argmin(np.abs(array - value))

# Key maturity points
key_maturities = [1.0, 5.0, 10.0]
print("Key Points Analysis:")
for key_mat in key_maturities:
    idx = find_nearest_index(fine_maturities, key_mat)
    print(f"{key_mat}Y: Yield={fine_yields[idx]:.4f}, Forward={fine_forwards[idx]:.4f}")

# Example 5: Spread analysis (if model supports spreads)
if hasattr(model, 'is_spread') and model.is_spread:
    print("\nExample 5: Spread Analysis")
    print("=" * 50)
    
    spread_maturities = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0])
    spread_values = model.spread_vectorized(spread_maturities)
    
    print("Maturity | Spread")
    print("-" * 20)
    for mat, spread in zip(spread_maturities, spread_values):
        print(f"{mat:8.2f} | {spread:7.4f}")

# Example 6: Discount factor calculations for cash flow valuation
print("\nExample 6: Cash Flow Valuation")
print("=" * 50)

# Example cash flows at different times
cash_flow_times = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
cash_flows = np.array([100, 100, 100, 100, 100, 1100])  # Coupon bond example

# Get discount factors
discount_factors = model.compute_discount_factors_vectorized(cash_flow_times)

# Calculate present value
present_values = cash_flows * discount_factors
total_pv = np.sum(present_values)

print("Cash Flow Valuation:")
print("Time | Cash Flow | Discount Factor | Present Value")
print("-" * 55)
for t, cf, df, pv in zip(cash_flow_times, cash_flows, discount_factors, present_values):
    print(f"{t:4.1f} | {cf:9.0f} | {df:15.6f} | {pv:12.2f}")
print("-" * 55)
print(f"Total Present Value: {total_pv:.2f}")

# Example 7: Sensitivity analysis
print("\nExample 7: Yield Sensitivity Analysis")
print("=" * 50)

# Base case
base_maturities = np.array([1.0, 5.0, 10.0])
base_yields = model.compute_yield_curve(base_maturities)

# Small parallel shift in alpha (interest rate level)
original_alpha = model.alpha
shift_size = 0.0001  # 1 basis point

model.alpha = original_alpha + shift_size
shifted_yields = model.compute_yield_curve(base_maturities)

# Calculate duration (sensitivity)
yield_changes = shifted_yields - base_yields
durations = yield_changes / shift_size

# Restore original alpha
model.alpha = original_alpha

print("Maturity | Base Yield | Shifted Yield | Duration")
print("-" * 50)
for mat, base_y, shift_y, dur in zip(base_maturities, base_yields, shifted_yields, durations):
    print(f"{mat:8.1f} | {base_y:10.4f} | {shift_y:13.4f} | {dur:8.2f}")

# Example 8: Curve interpolation and extrapolation
print("\nExample 8: Curve Interpolation")
print("=" * 50)

# Market observable points (sparse)
market_maturities = np.array([0.25, 1.0, 5.0, 10.0, 30.0])
market_yields = model.compute_yield_curve(market_maturities)

# Dense interpolation grid
interpolation_maturities = np.linspace(0.1, 35.0, 100)
interpolated_yields = model.compute_yield_curve(interpolation_maturities)

print("Market Points:")
for mat, yld in zip(market_maturities, market_yields):
    print(f"{mat:6.2f}Y: {yld:.4f}")

print(f"\nInterpolated to {len(interpolation_maturities)} points from 0.1Y to 35.0Y")

# Example 9: Bond portfolio valuation
print("\nExample 9: Bond Portfolio Valuation")
print("=" * 50)

# Portfolio of bonds with different maturities and face values
portfolio_maturities = np.array([1.0, 3.0, 5.0, 7.0, 10.0])
portfolio_face_values = np.array([1000000, 500000, 2000000, 750000, 1500000])

# Get bond prices
portfolio_bond_prices = model.bond_vectorized(portfolio_maturities)
portfolio_market_values = portfolio_bond_prices * portfolio_face_values

print("Bond Portfolio Analysis:")
print("Maturity | Face Value | Bond Price | Market Value")
print("-" * 55)
total_face = 0
total_market = 0
for mat, face, price, market in zip(portfolio_maturities, portfolio_face_values, 
                                   portfolio_bond_prices, portfolio_market_values):
    print(f"{mat:8.1f} | {face:10,.0f} | {price:10.6f} | {market:12,.0f}")
    total_face += face
    total_market += market

print("-" * 55)
print(f"Totals   | {total_face:10,.0f} |            | {total_market:12,.0f}")
print(f"Portfolio Yield: {(total_face/total_market - 1):.4f}")

# Example 10: Error handling and edge cases
print("\nExample 10: Error Handling")
print("=" * 50)

try:
    # Test with single maturity
    single_result = model.bond_vectorized(5.0)
    print(f"Single maturity result: {single_result}")
    
    # Test with empty array
    empty_result = model.bond_vectorized([])
    print(f"Empty array result: {empty_result}")
    
    # Test with very small maturities
    small_maturities = np.array([0.001, 0.01, 0.1])
    small_results = model.bond_vectorized(small_maturities)
    print(f"Small maturities: {small_results}")
    
except Exception as e:
    print(f"Error encountered: {e}")

print("\nAll examples completed successfully!")
