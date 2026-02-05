"""
Fourier Pricing Integration with Wishart Neural Operator - Version 2

This module shows how to properly integrate the trained NN with Fourier pricing.

Key fixes from original:
1. Proper handling of JAX vs NumPy arrays
2. Correct complex number handling
3. Support for both REAL and COMPLEX trained models

Author: Da Fonseca, Dawui, Malevergne
"""

import math
import cmath
from typing import Optional, Dict, Any
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
import scipy.integrate as sp_i

from inference import WishartNNInference
from training import TrainingMode
from data_generation import solve_riccati_ode


class FourierPricerNN:
    """
    Fourier transform based swaption pricing using Neural Network.
    
    This class properly interfaces between:
    - The trained Wishart NN (either REAL or COMPLEX mode)
    - The Fourier integration for pricing
    
    Key differences from the original fourier_pricing.py:
    1. Proper array type handling (JAX vs NumPy)
    2. Correct complex theta construction
    3. Support for both training modes
    """
    
    def __init__(
        self,
        inference: WishartNNInference,
        ur: float = 0.5,
        nmax: float = 300.0,
        epsabs: float = 1e-7,
        epsrel: float = 1e-5
    ):
        """
        Initialize Fourier pricer with NN.
        
        Args:
            inference: WishartNNInference instance
            ur: Real part of integration contour
            nmax: Maximum imaginary part for integration
            epsabs: Absolute tolerance for integration
            epsrel: Relative tolerance for integration
        """
        self.inference = inference
        self.ur = ur
        self.nmax = nmax
        self.epsabs = epsabs
        self.epsrel = epsrel
    
    def compute_phi_nn(
        self,
        T: float,
        z: complex,
        a3: np.ndarray,
        m: np.ndarray,
        omega: np.ndarray,
        sigma: np.ndarray,
        x0: np.ndarray
    ) -> complex:
        """
        Compute characteristic function using NN.
        
        This is the main interface to the NN for pricing.
        """
        return self.inference.compute_characteristic_function_for_pricing(
            T, z, a3, m, omega, sigma, x0
        )
    
    def price_simple(
        self,
        T: float,
        a3: np.ndarray,
        b3: float,
        m: np.ndarray,
        omega: np.ndarray,
        sigma: np.ndarray,
        x0: np.ndarray,
        alpha: float = 0.0,
        discount_factor: float = 1.0
    ) -> float:
        """
        Simple Fourier pricing using scipy.integrate.quad.
        
        Args:
            T: Time to maturity
            a3: Base theta matrix (real)
            b3: Scalar for exponential
            m, omega, sigma: Wishart parameters
            x0: Initial state
            alpha: Discount rate
            discount_factor: Additional discount factor
        
        Returns:
            Option price
        """
        def integrand(ui):
            z = complex(self.ur, ui)
            z_a3 = z * a3
            exp_z_b3 = cmath.exp(z * b3)
            
            # Use NN for characteristic function
            phi = self.compute_phi_nn(T, z, a3, m, omega, sigma, x0)
            
            result = exp_z_b3 * phi / (z * z)
            return result.real
        
        integral_result, error = sp_i.quad(
            integrand, 0, self.nmax,
            epsabs=self.epsabs, epsrel=self.epsrel
        )
        
        price = integral_result / math.pi
        price *= math.exp(-alpha * T)
        price *= discount_factor
        
        self.last_integration_error = error
        
        return price
    
    def price_with_intervals(
        self,
        T: float,
        a3: np.ndarray,
        b3: float,
        m: np.ndarray,
        omega: np.ndarray,
        sigma: np.ndarray,
        x0: np.ndarray,
        alpha: float = 0.0,
        discount_factor: float = 1.0,
        n_intervals: int = 5
    ) -> float:
        """
        Fourier pricing with interval-based integration.
        """
        intervals = np.linspace(0.0, self.nmax, n_intervals + 1)
        
        def integrand(ui):
            z = complex(self.ur, ui)
            exp_z_b3 = cmath.exp(z * b3)
            phi = self.compute_phi_nn(T, z, a3, m, omega, sigma, x0)
            result = exp_z_b3 * phi / (z * z)
            return result.real
        
        total_integral = 0.0
        for i in range(len(intervals) - 1):
            result, _ = sp_i.quad(
                integrand, intervals[i], intervals[i+1],
                epsabs=self.epsabs, epsrel=self.epsrel
            )
            total_integral += result
        
        price = total_integral / math.pi
        price *= math.exp(-alpha * T)
        price *= discount_factor
        
        return price
    
    def price_vectorized(
        self,
        T: float,
        a3: np.ndarray,
        b3: float,
        m: np.ndarray,
        omega: np.ndarray,
        sigma: np.ndarray,
        x0: np.ndarray,
        alpha: float = 0.0,
        discount_factor: float = 1.0,
        n_intervals: int = 10,
        n_points: int = 100
    ) -> float:
        """
        Vectorized Fourier pricing (faster but less accurate).
        
        Note: This is faster but may have issues with NN calls
        depending on whether the NN supports vectorized inputs.
        """
        intervals = np.linspace(0.0, self.nmax, n_intervals + 1)
        
        total = 0.0
        
        for i in range(len(intervals) - 1):
            ui_vals = np.linspace(intervals[i], intervals[i+1], n_points)
            
            integrand_vals = []
            for ui in ui_vals:
                z = complex(self.ur, ui)
                exp_z_b3 = cmath.exp(z * b3)
                phi = self.compute_phi_nn(T, z, a3, m, omega, sigma, x0)
                val = (exp_z_b3 * phi / (z * z)).real
                integrand_vals.append(val)
            
            integrand_vals = np.array(integrand_vals)
            total += np.trapz(integrand_vals, ui_vals)
        
        price = total / math.pi
        price *= math.exp(-alpha * T)
        price *= discount_factor
        
        return price


def compare_nn_vs_numerical(
    T: float,
    a3: np.ndarray,
    b3: float,
    m: np.ndarray,
    omega: np.ndarray,
    sigma: np.ndarray,
    x0: np.ndarray,
    inference: WishartNNInference,
    ur: float = 0.5,
    nmax: float = 300.0
) -> Dict[str, Any]:
    """
    Compare NN-based pricing with numerical ODE-based pricing.
    """
    import time
    
    # NN-based pricer
    pricer_nn = FourierPricerNN(inference, ur=ur, nmax=nmax)
    
    t0 = time.time()
    price_nn = pricer_nn.price_simple(T, a3, b3, m, omega, sigma, x0)
    time_nn = time.time() - t0
    
    # Numerical pricer (using ODE solver directly)
    def integrand_numerical(ui):
        z = complex(ur, ui)
        theta = z * a3
        A, B = solve_riccati_ode(T, theta, m, omega, sigma, dt=0.0001)
        phi = np.exp(np.trace(A @ x0) + B)
        exp_z_b3 = cmath.exp(z * b3)
        result = exp_z_b3 * phi / (z * z)
        return result.real
    
    t0 = time.time()
    integral_num, _ = sp_i.quad(integrand_numerical, 0, nmax, epsabs=1e-7, epsrel=1e-5)
    price_num = integral_num / math.pi
    time_num = time.time() - t0
    
    error = abs(price_nn - price_num) / (abs(price_num) + 1e-10)
    
    return {
        'price_nn': price_nn,
        'price_num': price_num,
        'error': error,
        'time_nn': time_nn,
        'time_num': time_num,
        'speedup': time_num / (time_nn + 1e-10)
    }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example: Load a trained model and test pricing
    
    # First, train a model (or load existing)
    import sys
    from pathlib import Path
    
    model_dir = Path("./output_real/models/final")
    norm_stats = Path("./output_real/normalization_stats.npz")
    
    if not model_dir.exists():
        print("No trained model found. Please run main_example.py first.")
        print("Example: python main_example.py --mode real --output ./output_real")
        sys.exit(1)
    
    # Load inference
    inference = WishartNNInference(
        model_path=str(model_dir),
        norm_stats_path=str(norm_stats),
        dim=2,
        hidden_dim=128,
        num_blocks=6
    )
    
    # Test parameters
    T = 2.0
    a3 = np.array([[1.0, 0.0], [0.0, 1.0]])
    b3 = 0.1
    m = np.array([[-0.5, 0.0], [0.0, -0.5]])
    omega = np.array([[0.3, 0.05], [0.05, 0.3]])
    sigma = np.array([[0.5, 0.1], [0.1, 0.5]])
    x0 = np.eye(2)
    
    print("=" * 60)
    print("FOURIER PRICING COMPARISON: NN vs NUMERICAL")
    print("=" * 60)
    
    result = compare_nn_vs_numerical(T, a3, b3, m, omega, sigma, x0, inference)
    
    print(f"\nResults:")
    print(f"  Price (NN):       {result['price_nn']:.6f}")
    print(f"  Price (Num):      {result['price_num']:.6f}")
    print(f"  Relative Error:   {result['error']:.2e}")
    print(f"  Time (NN):        {result['time_nn']*1000:.1f} ms")
    print(f"  Time (Numerical): {result['time_num']*1000:.1f} ms")
    print(f"  Speedup:          {result['speedup']:.1f}x")
