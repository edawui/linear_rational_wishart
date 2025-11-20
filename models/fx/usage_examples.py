# Usage Examples for Vectorized FX Option Pricing

# Example 1: Price multiple options with different strikes and maturities
maturities = [0.25, 0.5, 1.0, 2.0]  # 3M, 6M, 1Y, 2Y
strikes = [1.05, 1.10, 1.15, 1.20]  # Different strike levels

# Price all as calls
call_prices = model.price_fx_options_vectorized(maturities, strikes, is_call=True)
print(f"Call prices: {call_prices}")

# Price all as puts
put_prices = model.price_fx_options_vectorized(maturities, strikes, is_call=False)
print(f"Put prices: {put_prices}")

# Example 2: Mixed call/put options
is_call_flags = [True, False, True, False]  # Alternate calls and puts
mixed_prices = model.price_fx_options_vectorized(maturities, strikes, is_call=is_call_flags)
print(f"Mixed prices: {mixed_prices}")

# Example 3: Option chain - same maturity, different strikes
maturity = 1.0
strike_chain = np.linspace(0.95, 1.25, 11)  # Strike range from 0.95 to 1.25
maturities_chain = np.full_like(strike_chain, maturity)

# Price entire option chain
chain_prices = model.price_fx_options_vectorized(maturities_chain, strike_chain, is_call=True)

# Display results
print("\nOption Chain Results:")
for strike, price in zip(strike_chain, chain_prices):
    print(f"Strike: {strike:.3f}, Call Price: {price:.6f}")

# Example 4: Term structure - same strike, different maturities
strike = 1.10
term_maturities = np.array([0.08, 0.17, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])  # Various terms
term_strikes = np.full_like(term_maturities, strike)

term_prices = model.price_fx_options_vectorized(term_maturities, term_strikes, is_call=True)

print("\nTerm Structure Results:")
for mat, price in zip(term_maturities, term_prices):
    print(f"Maturity: {mat:.2f}Y, Call Price: {price:.6f}")

# Example 5: Compare with single option pricing (validation)
# Set properties for single option
model.set_option_properties(maturity=1.0, strike=1.10)
single_price = model.price_fx_option(is_call=True)

# Get same option from vectorized function
vector_price = model.price_fx_options_vectorized([1.0], [1.10], is_call=True)[0]

print(f"\nValidation:")
print(f"Single option price: {single_price:.8f}")
print(f"Vectorized price: {vector_price:.8f}")
print(f"Difference: {abs(single_price - vector_price):.2e}")

# Example 6: Performance comparison
import time

# Time single option pricing
start_time = time.time()
for mat, strike in zip(maturities, strikes):
    model.set_option_properties(mat, strike)
    _ = model.price_fx_option(is_call=True)
single_time = time.time() - start_time

# Time vectorized pricing
start_time = time.time()
_ = model.price_fx_options_vectorized(maturities, strikes, is_call=True)
vector_time = time.time() - start_time

print(f"\nPerformance Comparison:")
print(f"Single pricing time: {single_time:.4f}s")
print(f"Vectorized pricing time: {vector_time:.4f}s")
print(f"Speedup: {single_time/vector_time:.2f}x")

# Example 7: Error handling
try:
    # Mismatched array lengths
    bad_prices = model.price_fx_options_vectorized([1.0, 2.0], [1.05])
except ValueError as e:
    print(f"Expected error: {e}")

try:
    # Wrong is_call array length
    bad_prices = model.price_fx_options_vectorized([1.0, 2.0], [1.05, 1.10], 
                                                  is_call=[True])
except ValueError as e:
    print(f"Expected error: {e}")

# Example 8: Large scale option pricing
print("\nLarge Scale Example:")
n_options = 100
large_maturities = np.random.uniform(0.1, 5.0, n_options)
large_strikes = np.random.uniform(0.8, 1.5, n_options)

start_time = time.time()
large_prices = model.price_fx_options_vectorized(large_maturities, large_strikes)
large_time = time.time() - start_time

print(f"Priced {n_options} options in {large_time:.4f}s")
print(f"Average time per option: {large_time/n_options*1000:.2f}ms")
print(f"Price range: [{large_prices.min():.6f}, {large_prices.max():.6f}]")