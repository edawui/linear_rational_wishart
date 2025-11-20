"""
wishart_processes/__init__.py

Main package initialization for Wishart processes library.
"""

# Version info
__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import main classes for easy access
from ..core.wishart import WishartBru
from ..core.wishart_jump import WishartWithJump
from ..components.jump import JumpComponent

# Import useful utilities
from lrw_Jax.wishart_processes.pricing.wishart-collin-dufresne import CollinDufresneApproximation
from .pricing.expectations import compute_expectation_xy

# Configure JAX
import jax
jax.config.update("jax_enable_x64", True)

__all__ = [
    "WishartBru",
    "WishartWithJump", 
    "JumpComponent",
    "CollinDufresneApproximation",
    "compute_expectation_xy",
]

# ===== Example usage file: examples/basic_usage.py =====
"""
Basic usage examples for the Wishart processes package.
"""
import numpy as np
import jax.numpy as jnp
from ..core.wishart_jump import WishartWithJump


def example_basic_wishart():
    """Example: Basic Wishart process without jumps."""
    print("=== Basic Wishart Process Example ===")
    
    # Model parameters
    n = 2  # 2x2 Wishart process
    
    # Initial value
    x0 = jnp.array([[0.04, 0.01],
                    [0.01, 0.04]])
    
    # Drift matrix
    omega = jnp.array([[0.01, 0.005],
                       [0.005, 0.01]])
    
    # Mean reversion matrix
    m = jnp.array([[0.5, 0.0],
                   [0.0, 0.5]])
    
    # Volatility matrix
    sigma = jnp.array([[0.1, 0.05],
                       [0.05, 0.1]])
    
    # Create Wishart process
    wishart = WishartBru(n, x0, omega, m, sigma, is_bru_config=False)
    
    # Set maturity
    wishart.maturity = 1.0
    
    # Check Gindikin condition
    wishart.check_gindikin()
    
    # Compute mean at time t=1
    t = 1.0
    u0 = jnp.eye(2)  # Identity matrix for trace
    mean_trace = wishart.compute_mean(t, u0)
    print(f"E[tr(X_t)] at t={t}: {mean_trace}")
    
    # Compute moments up to order 3
    wishart.compute_moments(t, order=3)
    
    # Get specific moments
    moment_100 = wishart.get_moments(1, 0, 0)  # E[X_{11}]
    moment_010 = wishart.get_moments(0, 1, 0)  # E[X_{12}]
    moment_001 = wishart.get_moments(0, 0, 1)  # E[X_{22}]
    
    print(f"E[X_11] = {moment_100}")
    print(f"E[X_12] = {moment_010}")
    print(f"E[X_22] = {moment_001}")
    
    # Compute characteristic function
    theta1 = jnp.array([[0.1, 0.05],
                        [0.05, 0.1]]) * 1j
    phi_value = wishart.phi_one(1.0, theta1)
    print(f"Phi_1(1, theta1) = {phi_value}")
    
    return wishart


def example_wishart_with_jumps():
    """Example: Wishart process with jump component."""
    print("\n=== Wishart Process with Jumps Example ===")
    
    # Model parameters (same as above)
    n = 2
    x0 = jnp.array([[0.04, 0.01],
                    [0.01, 0.04]])
    omega = jnp.array([[0.01, 0.005],
                       [0.005, 0.01]])
    m = jnp.array([[0.5, 0.0],
                   [0.0, 0.5]])
    sigma = jnp.array([[0.1, 0.05],
                       [0.05, 0.1]])
    
    # Create base Wishart process
    wishart_jump = WishartWithJump(n, x0, omega, m, sigma)
    
    # Define jump component
    lambda_intensity = 0.5  # Jump intensity
    nu = 3  # Degrees of freedom for jump size
    eta = jnp.array([[0.02, 0.01],
                     [0.01, 0.02]])  # Jump variance matrix
    xi = jnp.array([[0.01, 0.005],
                    [0.005, 0.01]])  # Jump mean matrix
    
    jump_component = JumpComponent(lambda_intensity, nu, eta, xi)
    
    # Add jump component to Wishart process
    wishart_jump.set_jump(jump_component)
    
    # Set maturity
    wishart_jump.maturity = 1.0
    
    # Compute mean with jumps
    t = 1.0
    u0 = jnp.eye(2)
    mean_trace_jump = wishart_jump.compute_mean(t, u0)
    print(f"E[tr(X_t)] with jumps at t={t}: {mean_trace_jump}")
    
    # Compute characteristic function with jumps
    theta1 = jnp.array([[0.1, 0.05],
                        [0.05, 0.1]]) * 1j
    phi_value_jump = wishart_jump.phi_one(1.0, theta1)
    print(f"Phi_1(1, theta1) with jumps = {phi_value_jump}")
    
    return wishart_jump


def example_option_pricing():
    """Example: Option pricing using Collin-Dufresne approximation."""
    print("\n=== Option Pricing Example ===")
    
    # Create a Wishart process
    wishart = example_basic_wishart()
    
    # Parameters for option pricing
    maturity = 1.0
    wishart.maturity = maturity
    
    # Weight matrices for integrated variance
    a3 = jnp.array([[1.0, 0.0],
                    [0.0, 1.0]])
    b3 = 0.0
    
    # Compute moments of integrated variance
    wishart.compute_moments(maturity, order=3)
    
    # Get moments for Y = tr(a3 * X_t) + b3
    moments_y = wishart.compute_mu(b3, a3, approx_order=3)
    print(f"Moments of Y: {moments_y}")
    
    # Compute Collin-Dufresne coefficients
from lrw_Jax.wishart_processes.pricing.wishart-collin-dufresne import CollinDufresneApproximation
    
    c_coeffs = CollinDufresneApproximation.compute_c(moments_y, approx_order=3)
    print(f"C coefficients: {c_coeffs}")
    
    # Compute lambda coefficients (strike at ATM)
    lambda_coeffs = CollinDufresneApproximation.compute_lambda(c_coeffs, strike_offset=0, approx_order=3)
    print(f"Lambda coefficients: {lambda_coeffs}")
    
    # Compute gamma coefficients
    gamma_coeffs = CollinDufresneApproximation.compute_gamma(c_coeffs, approx_order=3)
    print(f"Gamma coefficients: {gamma_coeffs}")
    
    # Approximate option price
    option_price = CollinDufresneApproximation.compute_price_approximation(
        lambda_coeffs, gamma_coeffs, approx_order=3
    )
    print(f"Approximated option price: {option_price}")
    
    return option_price


def example_sensitivity_analysis():
    """Example: Computing sensitivities (Vegas)."""
    print("\n=== Sensitivity Analysis Example ===")
    
    # Create Wishart process
    n = 2
    x0 = jnp.array([[0.04, 0.01],
                    [0.01, 0.04]])
    omega = jnp.array([[0.01, 0.005],
                       [0.005, 0.01]])
    m = jnp.array([[0.5, 0.0],
                   [0.0, 0.5]])
    sigma = jnp.array([[0.1, 0.05],
                       [0.05, 0.1]])
    
    wishart = WishartBru(n, x0, omega, m, sigma)
    wishart.maturity = 1.0
    
    # Compute Vega sensitivity for sigma[0,0]
    i, j = 0, 0
    theta1 = jnp.array([[0.1, 0.05],
                        [0.05, 0.1]]) * 1j
    
    # Base Phi_One value
    phi_base = wishart.phi_one(1.0, theta1)
    print(f"Base Phi_1 value: {phi_base}")
    
    # Vega derivative
    phi_vega = wishart.phi_one_vega(i, j, 1.0, theta1)
    print(f"Vega[{i},{j}] of Phi_1: {phi_vega}")
    
    # Sensitivity ratio
    if abs(phi_base) > 1e-10:
        sensitivity = phi_vega / phi_base
        print(f"Relative sensitivity: {sensitivity}")
    
    return wishart


if __name__ == "__main__":
    # Run all examples
    print("Running Wishart Process Examples\n")
    
    # Basic Wishart
    wishart_basic = example_basic_wishart()
    
    # Wishart with jumps
    wishart_jump = example_wishart_with_jumps()
    
    # Option pricing
    price = example_option_pricing()
    
    # Sensitivity analysis
    wishart_sens = example_sensitivity_analysis()
    
    print("\nAll examples completed successfully!")


# ===== Setup.py for package installation =====
"""
setup.py

Setup configuration for the Wishart processes package.
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="wishart-processes",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="JAX-based implementation of Wishart processes for financial modeling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/wishart-processes",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "jax>=0.4.0",
        "jaxlib>=0.4.0",
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "sympy>=1.9",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.910",
            "sphinx>=4.0",
        ],
    },
)