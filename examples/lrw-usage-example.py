"""
Example usage of the refactored LRW interest rate model.

examples/lrw_usage.py
"""
import jax.numpy as jnp
import numpy as np

# Import configurations
from wishart_processes.models.interest_rate.config import (
    LRWModelConfig, SwaptionConfig, PricingConfig, HedgingConfig
)

# Import models and pricers
from wishart_processes.models.interest_rate.lrw_model import LRWModel
from lrw_numpy.LrwInterestRateJump import PricingApproach
from ..sensitivities.lrw-greeks-calculator_INCOMPLETE import LRWGreeksCalculator


def example_basic_lrw_swaption():
    """Example: Basic LRW model swaption pricing."""
    print("=== Basic LRW Swaption Pricing Example ===")
    
    # Model parameters
    n = 2
    alpha = 0.03  # 3% mean reversion level
    
    # Initial Wishart process value
    x0 = jnp.array([[0.04, 0.01],
                    [0.01, 0.04]])
    
    # Model matrices
    omega = jnp.array([[0.01, 0.005],
                       [0.005, 0.01]])
    
    m = jnp.array([[0.5, 0.0],
                   [0.0, 0.5]])
    
    sigma = jnp.array([[0.1, 0.05],
                       [0.05, 0.1]])
    
    # Create model configuration
    model_config = LRWModelConfig(
        n=n,
        alpha=alpha,
        x0=x0,
        omega=omega,
        m=m,
        sigma=sigma,
        is_bru_config=False,
        use_range_kutta_for_b=True
    )
    
    # Swaption parameters
    swaption_config = SwaptionConfig(
        maturity=1.0,      # 1 year option
        tenor=5.0,         # 5 year swap
        strike=0.03,       # 3% strike
        delta_float=0.5,   # Semi-annual float
        delta_fixed=1.0,   # Annual fixed
        u1=jnp.eye(2)      # Standard weight matrix
    )
    
    # Create LRW model
    lrw_model = LRWModel(model_config, swaption_config)
    
    # Print model info
    lrw_model.print_model()
    
    # Get current short rate
    short_rate = lrw_model.get_short_rate()
    print(f"\nCurrent short rate: {short_rate:.4f}")
    
    # Compute swap rate
    swap_rate = lrw_model.compute_swap_rate()
    print(f"Current swap rate: {swap_rate:.4f}")
    
    # Price swaption using different methods
    print("\n--- Swaption Pricing ---")
    
    # 1. Fourier pricing
    fourier_pricer = PricerFactory.create_pricer(PricingApproach.FOURIER, lrw_model)
    fourier_price = fourier_pricer.price_swaption()
    print(f"Fourier price: {fourier_price:.6f}")
    
    # 2. Collin-Dufresne approximation
    cd_pricer = PricerFactory.create_pricer(PricingApproach.COLLIN_DUFRESNE, lrw_model)
    cd_price = cd_pricer.price_swaption()
    print(f"Collin-Dufresne price: {cd_price:.6f}")
    
    # 3. Gamma approximation
    gamma_pricer = PricerFactory.create_pricer(PricingApproach.GAMMA_APPROXIMATION, lrw_model)
    gamma_price = gamma_pricer.price_swaption()
    print(f"Gamma approximation price: {gamma_price:.6f}")
    
    # Compute implied volatility
    implied_vol = fourier_pricer.compute_implied_vol(fourier_price)
    print(f"\nImplied volatility: {implied_vol:.2%}")
    
    return lrw_model, fourier_pricer


def example_lrw_greeks():
    """Example: Computing Greeks for LRW swaption."""
    print("\n=== LRW Greeks Calculation Example ===")
    
    # Create model and pricer
    lrw_model, fourier_pricer = example_basic_lrw_swaption()
    
    # Create Greeks calculator
    greeks_calc = LRWGreeksCalculator(lrw_model, fourier_pricer)
    
    # Compute delta hedge
    print("\n--- Delta Hedging ---")
    delta_hedge = greeks_calc.compute_delta_hedge()
    
    print("Zero-coupon bond hedge:")
    print(f"  Floating leg deltas: {delta_hedge['floating_leg']}")
    print(f"  Fixed leg deltas: {delta_hedge['fixed_leg']}")
    print(f"  Total hedge value: {delta_hedge['total_hedge_value']:.6f}")
    
    # Compute swap-based hedge
    swap_hedge = greeks_calc.compute_delta_hedge_swaps()
    print(f"\nSwap hedge:")
    print(f"  Floating leg hedge: {swap_hedge['floating_leg_hedge']:.6f}")
    print(f"  Fixed leg hedge: {swap_hedge['fixed_leg_hedge']:.6f}")
    print(f"  Total hedge value: {swap_hedge['total_hedge_value']:.6f}")
    
    # Compute vega
    print("\n--- Vega Sensitivities ---")
    vega_matrix = fourier_pricer.compute_vega()
    print("Vega matrix:")
    print(vega_matrix)
    print(f"Total vega: {jnp.sum(vega_matrix):.6f}")
    
    # Compute alpha sensitivity
    print("\n--- Alpha Sensitivity ---")
    alpha_sens = greeks_calc.compute_alpha_sensitivity()
    print(f"Alpha sensitivity: {alpha_sens['alpha_sensitivity']:.6f}")
    print(f"  Direct contribution: {alpha_sens['contribution_1']:.6f}")
    print(f"  Maturity contribution: {alpha_sens['contribution_2']:.6f}")
    
    # Compute omega sensitivity
    print("\n--- Omega Sensitivity ---")
    omega_sens = greeks_calc.compute_omega_sensitivity()
    print("Omega sensitivity matrix:")
    print(omega_sens)
    
    return greeks_calc


def example_bru_configuration():
    """Example: LRW model with Bru configuration."""
    print("\n=== LRW Bru Configuration Example ===")
    
    # Model parameters for Bru configuration
    n = 2
    alpha = 0.03
    beta = 2.0  # Bru parameter
    
    # Construct matrices satisfying Bru condition
    sigma = jnp.array([[0.1, 0.05],
                       [0.05, 0.1]])
    sigma2 = sigma @ sigma
    omega = beta * sigma2  # Bru condition: omega = beta * sigma^2
    
    x0 = jnp.array([[0.04, 0.01],
                    [0.01, 0.04]])
    
    m = jnp.array([[0.5, 0.0],
                   [0.0, 0.5]])
    
    # Create Bru configuration
    model_config = LRWModelConfig(
        n=n,
        alpha=alpha,
        x0=x0,
        omega=omega,
        m=m,
        sigma=sigma,
        is_bru_config=True
    )
    
    # Swaption configuration
    swaption_config = SwaptionConfig(
        maturity=2.0,
        tenor=10.0,
        strike=0.025,
        delta_float=0.25,  # Quarterly
        delta_fixed=0.5,   # Semi-annual
        u1=jnp.eye(2)
    )
    
    # Create model
    lrw_bru = LRWModel(model_config, swaption_config)
    
    print(f"Beta parameter: {lrw_bru.beta}")
    print(f"Is Bru configuration: {lrw_bru.is_bru_config}")
    
    # Price swaption
    pricer = PricerFactory.create_pricer(PricingApproach.FOURIER, lrw_bru)
    price = pricer.price_swaption()
    print(f"\nBru model swaption price: {price:.6f}")
    
    # The Bru configuration allows for more efficient computations
    # of certain quantities, especially vegas
    
    return lrw_bru


def example_spread_option():
    """Example: Pricing spread options."""
    print("\n=== Spread Option Example ===")
    
    # Model parameters
    n = 2
    alpha = 0.03
    
    x0 = jnp.array([[0.04, 0.01],
                    [0.01, 0.04]])
    
    omega = jnp.array([[0.01, 0.005],
                       [0.005, 0.01]])
    
    m = jnp.array([[0.5, 0.1],
                   [0.1, 0.5]])
    
    sigma = jnp.array([[0.1, 0.05],
                       [0.05, 0.1]])
    
    # Model configuration
    model_config = LRWModelConfig(
        n=n,
        alpha=alpha,
        x0=x0,
        omega=omega,
        m=m,
        sigma=sigma
    )
    
    # Define weight matrices for spread
    u1 = jnp.array([[1.0, 0.0],
                    [0.0, 0.0]])  # First rate
    
    u2 = jnp.array([[0.0, 0.0],
                    [0.0, 1.0]])  # Second rate (spread)
    
    # Swaption with spread
    swaption_config = SwaptionConfig(
        maturity=1.0,
        tenor=5.0,
        strike=0.03,
        u1=u1,
        u2=u2,
        is_spread=True
    )
    
    # Create model
    lrw_spread = LRWModel(model_config, swaption_config)
    
    # Get current spread
    current_spread = lrw_spread.get_spread()
    print(f"Current spread: {current_spread:.4f}")
    
    # Price spread option
    pricer = PricerFactory.create_pricer(PricingApproach.FOURIER, lrw_spread)
    price = pricer.price_swaption()
    print(f"Spread option price: {price:.6f}")
    
    return lrw_spread


def example_term_structure():
    """Example: Computing term structure of interest rates."""
    print("\n=== Term Structure Example ===")
    
    # Create a simple model
    model_config = LRWModelConfig(
        n=2,
        alpha=0.03,
        x0=jnp.array([[0.04, 0.01], [0.01, 0.04]]),
        omega=jnp.array([[0.01, 0.005], [0.005, 0.01]]),
        m=jnp.array([[0.5, 0.0], [0.0, 0.5]]),
        sigma=jnp.array([[0.1, 0.05], [0.05, 0.1]])
    )
    
    swaption_config = SwaptionConfig(
        maturity=1.0,
        tenor=5.0,
        strike=0.03,
        u1=jnp.eye(2)
    )
    
    lrw_model = LRWModel(model_config, swaption_config)
    
    # Compute zero-coupon bond prices
    maturities = np.linspace(0.25, 10.0, 40)
    bond_prices = []
    yields = []
    
    for t in maturities:
        bond_price = lrw_model.bond(t)
        bond_prices.append(bond_price)
        yields.append(-np.log(bond_price) / t)
    
    print("\nTerm Structure:")
    print("Maturity  |  Bond Price  |  Yield")
    print("-" * 35)
    for i in range(0, len(maturities), 5):
        print(f"{maturities[i]:8.2f}  |  {bond_prices[i]:10.6f}  |  {yields[i]:.4%}")
    
    # Asymptotic short rate
    r_infinity = lrw_model.get_short_rate_infinity()
    print(f"\nAsymptotic short rate: {r_infinity:.4%}")
    
    return lrw_model, maturities, yields


if __name__ == "__main__":
    print("Running LRW Interest Rate Model Examples\n")
    
    # Run all examples
    lrw_model, pricer = example_basic_lrw_swaption()
    greeks_calc = example_lrw_greeks()
    lrw_bru = example_bru_configuration()
    lrw_spread = example_spread_option()
    lrw_ts, maturities, yields = example_term_structure()
    
    print("\nAll examples completed successfully!")