"""
Simple Evaluation Script for Wishart Neural Network Model
==========================================================

This script evaluates how well the neural network approximates
the Wishart characteristic function by comparing with numerical results.

Supports two modes:
    - MODE="complex": theta = (ur + i*ui) * a3  (complex-valued)
    - MODE="real":    theta = ui * a3           (real-valued)

Easy to run:
    python evaluate_nn_model.py

Author: Your Name
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import sys
import jax
import gc
import os
import shutil

# Add your project path
sys.path.insert(0, r"E:\OneDrive\Dropbox\LinearRationalWishart_Work\Code\ED\LinearRationalWishart\LinearRationalWishart_NewCode")

from linear_rational_wishart.neural_operator.inference import WishartPINNInference
from linear_rational_wishart.neural_operator.model import WishartPINNModel

# ============================================================================
# CONFIGURATION - CHANGE THESE TO MATCH YOUR SETUP
# ============================================================================

# Choose mode: "complex" or "real"
MODE = "real"  # or "complex"
MODE =  "complex"

# For real mode, choose test type: "real" or "complex"
# - "real": test with real theta only (theta = ui * a3)
# - "complex": test real-trained model with complex inputs (may not work well)
REAL_CASE_TEST_MODE = "real"
REAL_CASE_TEST_MODE = "complex"

# Base paths
BASE_PATH = r"E:\OneDrive\Dropbox\LinearRationalWishart_Work\Code\ED\LinearRationalWishart\LinearRationalWishart_NewCode\Output_results\neural_operator"

# Derived paths (automatically set based on MODE)
MODEL_PATH = os.path.join(BASE_PATH, f"saved_models/models_{MODE}/final")
NORM_STATS_PATH = os.path.join(BASE_PATH, f"saved_models/normalization_stats_{MODE}.npz")

if MODE == "real":
    OUTPUT_DIR = os.path.join(BASE_PATH, f"evaluation_figures_{MODE}_{REAL_CASE_TEST_MODE}")
else:
    OUTPUT_DIR = os.path.join(BASE_PATH, f"evaluation_figures_{MODE}")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def clear_jax_cache():
    """Clear JAX compilation cache to free memory."""
    jax.clear_caches()
    gc.collect()
    
    local_app_data = os.environ.get('LOCALAPPDATA', '')
    if local_app_data:
        jax_cache = os.path.join(local_app_data, 'jax')
        if os.path.exists(jax_cache):
            shutil.rmtree(jax_cache, ignore_errors=True)


def create_theta(ui, a3, ur=0.5):
    """
    Create theta matrix based on current mode.
    
    Args:
        ui: Imaginary part (or just the scalar for real mode)
        a3: Base matrix (usually identity or diagonal)
        ur: Real part (only used in complex mode)
    
    Returns:
        theta: The theta matrix
    """
    if MODE == "real" and REAL_CASE_TEST_MODE == "real":
        # Real mode: theta = ui * a3 (no complex part)
        return ui * a3
    else:
        # Complex mode: theta = (ur + i*ui) * a3
        z = complex(ur, ui)
        return z * a3


def is_real_mode():
    """Check if we're in real-only mode."""
    return MODE == "real" and REAL_CASE_TEST_MODE == "real"


def load_model_and_inference():
    """Load the trained model."""
    print(f"\nLoading model from: {MODEL_PATH}")
    print(f"Normalization stats: {NORM_STATS_PATH}")
    print(f"Mode: {MODE}, Test mode: {REAL_CASE_TEST_MODE}")
    
    model = WishartPINNModel.load(MODEL_PATH)
    inference = WishartPINNInference(model, normalization_stats_path=NORM_STATS_PATH)
    
    print("\nModel loaded successfully!")
    print(model.summary())
    
    return inference


def get_test_parameters():
    """Define test parameters for the Wishart process."""
    d = 2  # Matrix dimension
    
    params = {
        'T': 1.0,                                      # Maturity
        'm': np.array([[-0.5, 0.0], [0.0, -0.5]]),    # Mean reversion
        'omega': np.array([[0.3, 0.05], [0.05, 0.3]]), # Drift
        'sigma': np.array([[0.5, 0.1], [0.1, 0.5]]),  # Volatility
        'x0': np.eye(d),                              # Initial state
        'ur': 0.5,                                     # Real part (for complex mode)
    }

    
    params = {
        'T': 1.0,                                      # Maturity
        'm': np.array([[-0.4, 0.00], [0.00, -0.2]]),    # Mean reversion
        'omega': np.array([[0.10, 0.002], [0.002, 0.0005]]), # Drift
        'sigma': np.array([[0.05, 0.02], [0.02, 0.047]]),  # Volatility
        'x0': np.array([[0.12, -0.00], [-0.00, 0.5]]),                              # Initial state
        # 'x0': 0.50*np.eye(d),                              # Initial state
        'ur': 0.5,                                     # Real part (for complex mode)
    }

    # params = {
    #     'T': 3.0,                                      # Maturity
    #     'm': np.array([[-0.4, 0.00], [0.00, -0.2]]),    # Mean reversion
    #     'omega': np.array([[0.10, 0.002], [0.002, 0.0005]]), # Drift
    #     'sigma': np.array([[0.5, 0.02], [0.02, 0.47]]),  # Volatility
    #     # 'x0': np.array([[0.12, -0.00], [-0.00, 0.5]]),                              # Initial state
    #     'x0': 0.50*np.eye(d),                              # Initial state
    #     'ur': 0.5,                                     # Real part (for complex mode)
    # }
    
    params = {
   'T': 1.0,                                      # Maturity
    'x0' :np.array([[0.12, -0.01], [-0.01, 0.005]]),
    'omega' :np.array([[0.10, 0.0], [0.00, 0.1]]),
    'm' :np.array([[-0.04, 0.0], [0.0, -0.02]]),
    'sigma' :np.array([[0.05, 0.02], [0.02, 0.047]]),
    'ur': 0.5
    }
    return params


# ============================================================================
# EVALUATION 1: CHARACTERISTIC FUNCTION PROFILE
# ============================================================================

def plot_characteristic_function_profile(inference, params, nb_point=50, save_path=None):
    """
    Plot the characteristic function as ui varies.
    
    This is the most important plot - it shows how well the NN
    approximates Phi(u) = E[exp(Tr(theta * X_T))]
    """
    print("\n" + "="*60)
    print("PLOT 1: Characteristic Function Profile")
    print("="*60)
    
    T = params['T']
    m = params['m']
    omega = params['omega']
    sigma = params['sigma']
    x0 = params['x0']
    ur = params['ur']
    
    # Base matrix for theta
    a3 = np.eye(2)
    
    # Range of values to test
    if is_real_mode():
        # For real mode, use smaller range (real theta can blow up)
        ui_values = np.linspace(0.01, 10.0, nb_point)
        x_label = '$\\theta$ magnitude'
    else:
        ui_values = np.linspace(0, 25, nb_point)
        x_label = '$u_i$ (imaginary part)'
    
    # Storage for results
    phi_nn_real = []
    phi_nn_imag = []
    phi_num_real = []
    phi_num_imag = []
    
    print(f"Computing characteristic functions ({nb_point} points)...")
    for i, ui in enumerate(ui_values):
        if i % 10 == 0:
            print(f"  Progress: {i+1}/{len(ui_values)}")
        
        # Create theta based on mode
        theta = create_theta(ui, a3, ur)
        
        # Neural network prediction
        A_nn, B_nn = inference.compute_A_B(T, theta, m, omega, sigma, denormalize=True)
        phi_nn = np.exp(np.trace(A_nn @ x0) + B_nn)
        phi_nn_real.append(np.real(phi_nn))
        phi_nn_imag.append(np.imag(phi_nn))
        
        # Numerical ground truth
        A_num, B_num = inference.numerical_computer.compute_A_B(T, theta, m, omega, sigma, x0)
        phi_num = np.exp(np.trace(A_num @ x0) + B_num)
        phi_num_real.append(np.real(phi_num))
        phi_num_imag.append(np.imag(phi_num))
    
    # Convert to arrays
    phi_nn_real = np.array(phi_nn_real)
    phi_nn_imag = np.array(phi_nn_imag)
    phi_num_real = np.array(phi_num_real)
    phi_num_imag = np.array(phi_num_imag)
    
    # Compute errors
    phi_nn_complex = phi_nn_real + 1j * phi_nn_imag
    phi_num_complex = phi_num_real + 1j * phi_num_imag
    rel_error = np.abs(phi_nn_complex - phi_num_complex) / (np.abs(phi_num_complex) + 1e-10)
    
    # Create figure
    if is_real_mode():
        # For real mode, only show real part and error (imaginary should be ~0)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Plot 1: Real part (which is the full Phi for real theta)
        axes[0].plot(ui_values, phi_num_real, 'b-', linewidth=2, label='Numerical')
        axes[0].plot(ui_values, phi_nn_real, 'r--', linewidth=2, label='Neural Network')
        axes[0].set_xlabel(x_label, fontsize=12)
        axes[0].set_ylabel('$\Phi$', fontsize=12)
        axes[0].set_title('Characteristic Function (Real Mode)', fontsize=14)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Log scale magnitude
        axes[1].semilogy(ui_values, np.abs(phi_num_complex), 'b-', linewidth=2, label='Numerical')
        axes[1].semilogy(ui_values, np.abs(phi_nn_complex), 'r--', linewidth=2, label='Neural Network')
        axes[1].set_xlabel(x_label, fontsize=12)
        axes[1].set_ylabel('$|\Phi|$', fontsize=12)
        axes[1].set_title('Magnitude (Log Scale)', fontsize=14)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Relative error
        axes[2].semilogy(ui_values, rel_error, 'k-', linewidth=2)
        axes[2].axhline(0.01, color='r', linestyle='--', linewidth=1, label='1% error')
        axes[2].axhline(0.001, color='g', linestyle='--', linewidth=1, label='0.1% error')
        axes[2].set_xlabel(x_label, fontsize=12)
        axes[2].set_ylabel('Relative Error', fontsize=12)
        axes[2].set_title('Relative Error', fontsize=14)
        axes[2].legend(fontsize=10)
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim([1e-6, 10])
        
    else:
        # For complex mode, show all 4 plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Real part
        axes[0, 0].plot(ui_values, phi_num_real, 'b-', linewidth=2, label='Numerical')
        axes[0, 0].plot(ui_values, phi_nn_real, 'r--', linewidth=2, label='Neural Network')
        axes[0, 0].set_xlabel(x_label, fontsize=12)
        axes[0, 0].set_ylabel('Re($\Phi$)', fontsize=12)
        axes[0, 0].set_title('Real Part', fontsize=14)
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Imaginary part
        axes[0, 1].plot(ui_values, phi_num_imag, 'b-', linewidth=2, label='Numerical')
        axes[0, 1].plot(ui_values, phi_nn_imag, 'r--', linewidth=2, label='Neural Network')
        axes[0, 1].set_xlabel(x_label, fontsize=12)
        axes[0, 1].set_ylabel('Im($\Phi$)', fontsize=12)
        axes[0, 1].set_title('Imaginary Part', fontsize=14)
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Magnitude (log scale)
        axes[1, 0].semilogy(ui_values, np.abs(phi_num_complex), 'b-', linewidth=2, label='Numerical')
        axes[1, 0].semilogy(ui_values, np.abs(phi_nn_complex), 'r--', linewidth=2, label='Neural Network')
        axes[1, 0].set_xlabel(x_label, fontsize=12)
        axes[1, 0].set_ylabel('$|\Phi|$', fontsize=12)
        axes[1, 0].set_title('Magnitude (Log Scale)', fontsize=14)
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Relative error
        axes[1, 1].semilogy(ui_values, rel_error, 'k-', linewidth=2)
        axes[1, 1].axhline(0.01, color='r', linestyle='--', linewidth=1, label='1% error')
        axes[1, 1].axhline(0.001, color='g', linestyle='--', linewidth=1, label='0.1% error')
        axes[1, 1].set_xlabel(x_label, fontsize=12)
        axes[1, 1].set_ylabel('Relative Error', fontsize=12)
        axes[1, 1].set_title('Relative Error', fontsize=14)
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim([1e-6, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    # Print summary
    print(f"\n  Summary:")
    print(f"    Mean relative error:   {np.mean(rel_error):.2e}")
    print(f"    Median relative error: {np.median(rel_error):.2e}")
    print(f"    Max relative error:    {np.max(rel_error):.2e}")
    
    return fig


# ============================================================================
# EVALUATION 2: PARITY PLOTS (PREDICTED VS TRUE)
# ============================================================================

def plot_parity(inference, params, n_samples=200, save_path=None):
    """
    Parity plot: compare NN predictions vs numerical ground truth.
    Points on the diagonal = perfect prediction.
    """
    print("\n" + "="*60)
    print("PLOT 2: Parity Plot (Predicted vs True)")
    print("="*60)
    
    m = params['m']
    omega = params['omega']
    sigma = params['sigma']
    x0 = params['x0']
    ur = params['ur']
    
    np.random.seed(42)
    
    A_nn_list = []
    A_num_list = []
    B_nn_list = []
    B_num_list = []
    phi_nn_list = []
    phi_num_list = []
    
    print(f"Computing {n_samples} test cases...")
    for i in range(n_samples):
        if i % 50 == 0:
            print(f"  Progress: {i+1}/{n_samples}")
        
        # Random parameters
        T_test = np.random.uniform(0.5, 5.0)
        a3_diag = np.random.uniform(0.01, 10.0)
        a3 = np.eye(2) * a3_diag
        
        if is_real_mode():
            ui_test = np.random.uniform(0.01, 10.0)
        else:
            ui_test = np.random.uniform(0.0, 25.0)
        
        theta = create_theta(ui_test, a3, ur)
        
        try:
            # NN prediction
            A_nn, B_nn = inference.compute_A_B(T_test, theta, m, omega, sigma, denormalize=True)
            phi_nn = np.exp(np.trace(A_nn @ x0) + B_nn)
            
            # Numerical
            A_num, B_num = inference.numerical_computer.compute_A_B(T_test, theta, m, omega, sigma, x0)
            phi_num = np.exp(np.trace(A_num @ x0) + B_num)
            
            A_nn_list.append(A_nn.flatten())
            A_num_list.append(A_num.flatten())
            B_nn_list.append(B_nn)
            B_num_list.append(B_num)
            phi_nn_list.append(phi_nn)
            phi_num_list.append(phi_num)
        except Exception as e:
            print(f"  Warning: Sample {i} failed: {e}")
            continue
    
    # Convert to arrays
    A_nn_arr = np.array(A_nn_list)
    A_num_arr = np.array(A_num_list)
    B_nn_arr = np.array(B_nn_list)
    B_num_arr = np.array(B_num_list)
    phi_nn_arr = np.array(phi_nn_list)
    phi_num_arr = np.array(phi_num_list)
    
    # Create figure based on mode
    if is_real_mode():
        # Real mode: only real parts matter
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # A matrix - real part
        ax = axes[0]
        ax.scatter(np.real(A_num_arr[:, 0]), np.real(A_nn_arr[:, 0]), alpha=0.5, s=20)
        lims = [min(np.real(A_num_arr[:, 0]).min(), np.real(A_nn_arr[:, 0]).min()),
                max(np.real(A_num_arr[:, 0]).max(), np.real(A_nn_arr[:, 0]).max())]
        ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect')
        ax.set_xlabel('True $A_{11}$', fontsize=11)
        ax.set_ylabel('Predicted $A_{11}$', fontsize=11)
        ax.set_title('A Matrix Element', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # B scalar - real part
        ax = axes[1]
        ax.scatter(np.real(B_num_arr), np.real(B_nn_arr), alpha=0.5, s=20)
        lims = [min(np.real(B_num_arr).min(), np.real(B_nn_arr).min()),
                max(np.real(B_num_arr).max(), np.real(B_nn_arr).max())]
        ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect')
        ax.set_xlabel('True $B$', fontsize=11)
        ax.set_ylabel('Predicted $B$', fontsize=11)
        ax.set_title('B Scalar', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Phi - real part
        ax = axes[2]
        ax.scatter(np.real(phi_num_arr), np.real(phi_nn_arr), alpha=0.5, s=20)
        lims = [min(np.real(phi_num_arr).min(), np.real(phi_nn_arr).min()),
                max(np.real(phi_num_arr).max(), np.real(phi_nn_arr).max())]
        ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect')
        ax.set_xlabel('True $\Phi$', fontsize=11)
        ax.set_ylabel('Predicted $\Phi$', fontsize=11)
        ax.set_title('Characteristic Function', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    else:
        # Complex mode: show real and imaginary parts
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # A matrix - real part
        ax = axes[0, 0]
        ax.scatter(A_num_arr[:, 0].real, A_nn_arr[:, 0].real, alpha=0.5, s=20)
        lims = [min(A_num_arr[:, 0].real.min(), A_nn_arr[:, 0].real.min()),
                max(A_num_arr[:, 0].real.max(), A_nn_arr[:, 0].real.max())]
        ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect')
        ax.set_xlabel('True $A_{11}$ (Real)', fontsize=11)
        ax.set_ylabel('Predicted $A_{11}$ (Real)', fontsize=11)
        ax.set_title('A Matrix (Real)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # A matrix - imaginary part
        ax = axes[0, 1]
        ax.scatter(A_num_arr[:, 0].imag, A_nn_arr[:, 0].imag, alpha=0.5, s=20)
        lims = [min(A_num_arr[:, 0].imag.min(), A_nn_arr[:, 0].imag.min()),
                max(A_num_arr[:, 0].imag.max(), A_nn_arr[:, 0].imag.max())]
        ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect')
        ax.set_xlabel('True $A_{11}$ (Imag)', fontsize=11)
        ax.set_ylabel('Predicted $A_{11}$ (Imag)', fontsize=11)
        ax.set_title('A Matrix (Imag)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # B scalar - real part
        ax = axes[0, 2]
        ax.scatter(np.real(B_num_arr), np.real(B_nn_arr), alpha=0.5, s=20)
        lims = [min(np.real(B_num_arr).min(), np.real(B_nn_arr).min()),
                max(np.real(B_num_arr).max(), np.real(B_nn_arr).max())]
        ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect')
        ax.set_xlabel('True $B$ (Real)', fontsize=11)
        ax.set_ylabel('Predicted $B$ (Real)', fontsize=11)
        ax.set_title('B Scalar (Real)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # B scalar - imaginary part
        ax = axes[1, 0]
        ax.scatter(np.imag(B_num_arr), np.imag(B_nn_arr), alpha=0.5, s=20)
        lims = [min(np.imag(B_num_arr).min(), np.imag(B_nn_arr).min()),
                max(np.imag(B_num_arr).max(), np.imag(B_nn_arr).max())]
        ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect')
        ax.set_xlabel('True $B$ (Imag)', fontsize=11)
        ax.set_ylabel('Predicted $B$ (Imag)', fontsize=11)
        ax.set_title('B Scalar (Imag)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Phi - real part
        ax = axes[1, 1]
        ax.scatter(np.real(phi_num_arr), np.real(phi_nn_arr), alpha=0.5, s=20)
        lims = [min(np.real(phi_num_arr).min(), np.real(phi_nn_arr).min()),
                max(np.real(phi_num_arr).max(), np.real(phi_nn_arr).max())]
        ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect')
        ax.set_xlabel('True $\Phi$ (Real)', fontsize=11)
        ax.set_ylabel('Predicted $\Phi$ (Real)', fontsize=11)
        ax.set_title('Char. Function (Real)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Phi - imaginary part
        ax = axes[1, 2]
        ax.scatter(np.imag(phi_num_arr), np.imag(phi_nn_arr), alpha=0.5, s=20)
        lims = [min(np.imag(phi_num_arr).min(), np.imag(phi_nn_arr).min()),
                max(np.imag(phi_num_arr).max(), np.imag(phi_nn_arr).max())]
        ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect')
        ax.set_xlabel('True $\Phi$ (Imag)', fontsize=11)
        ax.set_ylabel('Predicted $\Phi$ (Imag)', fontsize=11)
        ax.set_title('Char. Function (Imag)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    # Compute R² score
    def r2_score(true, pred):
        ss_res = np.sum(np.abs(true - pred)**2)
        ss_tot = np.sum(np.abs(true - np.mean(true))**2)
        return 1 - ss_res / ss_tot
    
    r2_phi = r2_score(phi_num_arr, phi_nn_arr)
    print(f"\n  R² score for Φ: {r2_phi:.4f}")
    
    return fig


# ============================================================================
# EVALUATION 3: ERROR DISTRIBUTION
# ============================================================================

def plot_error_distribution(inference, params, n_samples=500, save_path=None):
    """Plot histogram of errors across many test cases."""
    print("\n" + "="*60)
    print("PLOT 3: Error Distribution")
    print("="*60)
    
    m = params['m']
    omega = params['omega']
    sigma = params['sigma']
    x0 = params['x0']
    ur = params['ur']
    
    np.random.seed(123)
    
    errors_A = []
    errors_B = []
    errors_phi = []
    
    print(f"Computing {n_samples} test cases...")
    for i in range(n_samples):
        if i % 100 == 0:
            print(f"  Progress: {i+1}/{n_samples}")
        
        # Random parameters
        T_test = np.random.uniform(0.5, 5.0)
        a3_diag = np.random.uniform(0.01, 10.0)
        a3 = np.eye(2) * a3_diag
        
        if is_real_mode():
            ui_test = np.random.uniform(0.01, 10.0)
        else:
            ui_test = np.random.uniform(0.0, 25.0)
        
        theta = create_theta(ui_test, a3, ur)
        
        try:
            # NN prediction
            A_nn, B_nn = inference.compute_A_B(T_test, theta, m, omega, sigma, denormalize=True)
            phi_nn = np.exp(np.trace(A_nn @ x0) + B_nn)
            
            # Numerical
            A_num, B_num = inference.numerical_computer.compute_A_B(T_test, theta, m, omega, sigma, x0)
            phi_num = np.exp(np.trace(A_num @ x0) + B_num)
            
            # Compute errors
            err_A = np.linalg.norm(A_nn - A_num) / (np.linalg.norm(A_num) + 1e-10)
            err_B = np.abs(B_nn - B_num) / (np.abs(B_num) + 1e-10)
            err_phi = np.abs(phi_nn - phi_num) / (np.abs(phi_num) + 1e-10)
            
            errors_A.append(err_A)
            errors_B.append(err_B)
            errors_phi.append(err_phi)
        except:
            continue
    
    errors_A = np.array(errors_A)
    errors_B = np.array(errors_B)
    errors_phi = np.array(errors_phi)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Error A
    ax = axes[0]
    ax.hist(np.log10(errors_A + 1e-12), bins=50, edgecolor='black', alpha=0.7, color='blue')
    ax.axvline(np.log10(np.median(errors_A)), color='r', linestyle='--', 
               linewidth=2, label=f'Median: {np.median(errors_A):.2e}')
    ax.set_xlabel('$\log_{10}$(Relative Error)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('A Matrix Error', fontsize=14)
    ax.legend(fontsize=10)
    
    # Error B
    ax = axes[1]
    ax.hist(np.log10(errors_B + 1e-12), bins=50, edgecolor='black', alpha=0.7, color='green')
    ax.axvline(np.log10(np.median(errors_B)), color='r', linestyle='--', 
               linewidth=2, label=f'Median: {np.median(errors_B):.2e}')
    ax.set_xlabel('$\log_{10}$(Relative Error)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('B Scalar Error', fontsize=14)
    ax.legend(fontsize=10)
    
    # Error Phi
    ax = axes[2]
    ax.hist(np.log10(errors_phi + 1e-12), bins=50, edgecolor='black', alpha=0.7, color='orange')
    ax.axvline(np.log10(np.median(errors_phi)), color='r', linestyle='--', 
               linewidth=2, label=f'Median: {np.median(errors_phi):.2e}')
    ax.set_xlabel('$\log_{10}$(Relative Error)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('$\Phi$ Error', fontsize=14)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    # Print summary
    print(f"\n  Error Statistics:")
    print(f"  {'Metric':<15} {'Mean':>12} {'Median':>12} {'Max':>12}")
    print(f"  {'-'*53}")
    print(f"  {'Error A':<15} {np.mean(errors_A):>12.2e} {np.median(errors_A):>12.2e} {np.max(errors_A):>12.2e}")
    print(f"  {'Error B':<15} {np.mean(errors_B):>12.2e} {np.median(errors_B):>12.2e} {np.max(errors_B):>12.2e}")
    print(f"  {'Error Phi':<15} {np.mean(errors_phi):>12.2e} {np.median(errors_phi):>12.2e} {np.max(errors_phi):>12.2e}")
    
    return fig


# ============================================================================
# EVALUATION 4: ERROR VS PARAMETERS
# ============================================================================

def plot_error_vs_parameters(inference, params, n_samples=300, save_path=None):
    """Show how error varies with input parameters."""
    print("\n" + "="*60)
    print("PLOT 4: Error vs Input Parameters")
    print("="*60)
    
    m = params['m']
    omega = params['omega']
    sigma = params['sigma']
    x0 = params['x0']
    ur = params['ur']
    
    np.random.seed(456)
    
    T_vals = []
    ui_vals = []
    theta_vals = []
    errors_phi = []
    
    print(f"Computing {n_samples} test cases...")
    for i in range(n_samples):
        if i % 100 == 0:
            print(f"  Progress: {i+1}/{n_samples}")
        
        # Random parameters
        T_test = np.random.uniform(0.5, 5.0)
        a3_diag = np.random.uniform(0.01, 10.0)
        a3 = np.eye(2) * a3_diag
        
        if is_real_mode():
            ui_test = np.random.uniform(0.01, 10.0)
        else:
            ui_test = np.random.uniform(0.0, 25.0)
        
        theta = create_theta(ui_test, a3, ur)
        
        try:
            # NN prediction
            A_nn, B_nn = inference.compute_A_B(T_test, theta, m, omega, sigma, denormalize=True)
            phi_nn = np.exp(np.trace(A_nn @ x0) + B_nn)
            
            # Numerical
            A_num, B_num = inference.numerical_computer.compute_A_B(T_test, theta, m, omega, sigma, x0)
            phi_num = np.exp(np.trace(A_num @ x0) + B_num)
            
            err_phi = np.abs(phi_nn - phi_num) / (np.abs(phi_num) + 1e-10)
            
            T_vals.append(T_test)
            ui_vals.append(ui_test)
            theta_vals.append(a3_diag)
            errors_phi.append(err_phi)
        except:
            continue
    
    T_vals = np.array(T_vals)
    ui_vals = np.array(ui_vals)
    theta_vals = np.array(theta_vals)
    errors_phi = np.array(errors_phi)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Axis labels based on mode
    if is_real_mode():
        ui_label = '$\\theta$ scale'
    else:
        ui_label = '$u_i$'
    
    # Error vs T
    ax = axes[0]
    scatter = ax.scatter(T_vals, np.log10(errors_phi + 1e-12), 
                         c=ui_vals, cmap='viridis', alpha=0.6, s=20)
    ax.set_xlabel('Maturity $T$', fontsize=12)
    ax.set_ylabel('$\log_{10}$(Error $\Phi$)', fontsize=12)
    ax.set_title('Error vs Maturity', fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label=ui_label)
    
    # Error vs ui
    ax = axes[1]
    scatter = ax.scatter(ui_vals, np.log10(errors_phi + 1e-12), 
                         c=T_vals, cmap='plasma', alpha=0.6, s=20)
    ax.set_xlabel(ui_label, fontsize=12)
    ax.set_ylabel('$\log_{10}$(Error $\Phi$)', fontsize=12)
    ax.set_title(f'Error vs {ui_label}', fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='$T$')
    
    # Error vs theta magnitude
    ax = axes[2]
    scatter = ax.scatter(theta_vals, np.log10(errors_phi + 1e-12), 
                         c=ui_vals, cmap='viridis', alpha=0.6, s=20)
    ax.set_xlabel('$a_3$ diagonal', fontsize=12)
    ax.set_ylabel('$\log_{10}$(Error $\Phi$)', fontsize=12)
    ax.set_title('Error vs $a_3$ magnitude', fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label=ui_label)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    return fig


# ============================================================================
# EVALUATION 5: SPEED COMPARISON
# ============================================================================

def benchmark_speed(inference, params, n_samples=100, save_path=None):
    """Compare computation speed: NN vs Numerical."""
    print("\n" + "="*60)
    print("PLOT 5: Speed Benchmark")
    print("="*60)
    
    m = params['m']
    omega = params['omega']
    sigma = params['sigma']
    x0 = params['x0']
    ur = params['ur']
    
    np.random.seed(789)
    
    times_nn = []
    times_num = []
    
    print(f"Benchmarking {n_samples} computations...")
    
    for i in range(n_samples):
        # Random parameters
        T_test = np.random.uniform(0.5, 5.0)
        a3 = np.eye(2) * np.random.uniform(0.01, 10.0)
        
        if is_real_mode():
            ui_test = np.random.uniform(0.01, 10.0)
        else:
            ui_test = np.random.uniform(0.0, 25.0)
        
        theta = create_theta(ui_test, a3, ur)
        
        # Time NN
        t0 = time.time()
        A_nn, B_nn = inference.compute_A_B(T_test, theta, m, omega, sigma, denormalize=True)
        phi_nn = np.exp(np.trace(A_nn @ x0) + B_nn)
        times_nn.append(time.time() - t0)
        
        # Time Numerical
        t0 = time.time()
        A_num, B_num = inference.numerical_computer.compute_A_B(T_test, theta, m, omega, sigma, x0)
        phi_num = np.exp(np.trace(A_num @ x0) + B_num)
        times_num.append(time.time() - t0)
    
    times_nn = np.array(times_nn) * 1000  # Convert to ms
    times_num = np.array(times_num) * 1000
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogram
    ax = axes[0]
    ax.hist(times_nn, bins=30, alpha=0.7, label=f'NN: {np.mean(times_nn):.2f} ms', color='red')
    ax.hist(times_num, bins=30, alpha=0.7, label=f'Numerical: {np.mean(times_num):.2f} ms', color='blue')
    ax.set_xlabel('Computation Time (ms)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Time Distribution', fontsize=14)
    ax.legend(fontsize=10)
    
    # Bar chart
    ax = axes[1]
    methods = ['Neural Network', 'Numerical']
    means = [np.mean(times_nn), np.mean(times_num)]
    stds = [np.std(times_nn), np.std(times_num)]
    
    ax.bar(methods, means, yerr=stds, capsize=5, color=['red', 'blue'], alpha=0.7)
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title('Average Time', fontsize=14)
    
    # Speedup annotation
    speedup = np.mean(times_num) / np.mean(times_nn)
    ax.annotate(f'Speedup: {speedup:.1f}x', xy=(0.5, 0.9), xycoords='axes fraction',
                fontsize=14, ha='center', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    print(f"\n  Timing Results:")
    print(f"    Neural Network: {np.mean(times_nn):.2f} ± {np.std(times_nn):.2f} ms")
    print(f"    Numerical:      {np.mean(times_num):.2f} ± {np.std(times_num):.2f} ms")
    print(f"    Speedup:        {speedup:.1f}x")
    
    return fig


# ============================================================================
# EVALUATION 6: MARGINAL PDF
# ============================================================================

def plot_marginal_pdf(inference, params, n_points=50, save_path=None):
    """Plot the marginal PDF using Fourier inversion."""
    print("\n" + "="*60)
    print("PLOT 6: Marginal PDF of Trace(X)")
    print("="*60)
    
    T = params['T']
    m = params['m']
    omega = params['omega']
    sigma = params['sigma']
    x0 = params['x0']
    
    # For trace(X), theta = I
    theta_base = np.eye(2)
    
    # Characteristic function values
    u_max = 50
    u_vals = np.linspace(-u_max, u_max, n_points)
    du = u_vals[1] - u_vals[0]
    
    phi_nn = []
    phi_num = []
    
    print(f"Computing characteristic function ({n_points} points)...")
    for i, u in enumerate(u_vals):
        if i % 20 == 0:
            print(f"  Progress: {i+1}/{n_points}")
        
        # theta = i*u*I for PDF
        theta = 1j * u * theta_base
        
        # NN
        A_nn, B_nn = inference.compute_A_B(T, theta, m, omega, sigma, denormalize=True)
        phi_nn.append(np.exp(np.trace(A_nn @ x0) + B_nn))
        
        # Numerical
        A_num, B_num = inference.numerical_computer.compute_A_B(T, theta, m, omega, sigma, x0)
        phi_num.append(np.exp(np.trace(A_num @ x0) + B_num))
    
    phi_nn = np.array(phi_nn)
    phi_num = np.array(phi_num)
    
    # Inverse FFT
    from scipy import fft
    
    y_max = np.pi / du
    y_vals = np.linspace(-y_max, y_max, n_points)
    
    pdf_nn = np.real(fft.fftshift(fft.ifft(fft.ifftshift(phi_nn)))) * n_points * du / (2 * np.pi)
    pdf_num = np.real(fft.fftshift(fft.ifft(fft.ifftshift(phi_num)))) * n_points * du / (2 * np.pi)
    
    # Only positive values
    mask = (y_vals > 0) & (y_vals < 10)
    y_plot = y_vals[mask]
    pdf_nn_plot = pdf_nn[mask]
    pdf_num_plot = pdf_num[mask]
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # PDF comparison
    ax = axes[0]
    ax.plot(y_plot, pdf_num_plot, 'b-', linewidth=2, label='Numerical')
    ax.plot(y_plot, pdf_nn_plot, 'r--', linewidth=2, label='Neural Network')
    ax.set_xlabel('Trace($X_T$)', fontsize=12)
    ax.set_ylabel('PDF', fontsize=12)
    ax.set_title(f'Marginal PDF at T={T}', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # Difference
    ax = axes[1]
    diff = pdf_nn_plot - pdf_num_plot
    ax.plot(y_plot, diff, 'k-', linewidth=2)
    ax.axhline(0, color='r', linestyle='--', linewidth=1)
    ax.set_xlabel('Trace($X_T$)', fontsize=12)
    ax.set_ylabel('PDF Difference', fontsize=12)
    ax.set_title('Approximation Error', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    return fig

def diagnose_branch_cut_issues(inference, params, T, a3, ui_range, ur=0.5):
    """
    Check if the characteristic function has discontinuities.
    
    Based on Kahl-Jäckel (2005) analysis.
    """
    m = params['m']
    omega = params['omega']
    sigma = params['sigma']
    x0 = params['x0']
    
    phi_values = []
    phase_values = []
    
    for ui in ui_range:
        theta = complex(ur, ui) * a3
        A, B = inference.numerical_computer.compute_A_B(T, theta, m, omega, sigma, x0)
        phi = np.exp(np.trace(A @ x0) + B)
        
        phi_values.append(phi)
        phase_values.append(np.angle(phi))
    
    phi_values = np.array(phi_values)
    phase_values = np.array(phase_values)
    
    # Check for phase jumps
    phase_diff = np.diff(phase_values)
    jumps = np.where(np.abs(phase_diff) > np.pi)[0]
    
    if len(jumps) > 0:
        print(f"WARNING: Found {len(jumps)} potential branch cut crossings!")
        print(f"  At ui values: {ui_range[jumps]}")
        print(f"  Phase jumps: {phase_diff[jumps]}")
        return True, jumps
    else:
        print("No branch cut issues detected.")
        return False, []


def plot_phase_analysis(inference, params, T_values=[1, 5, 10, 20]):
    """
    Visualize phase behavior like Figure 4 in Kahl-Jäckel paper.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    a3 = np.eye(2)
    ui_range = np.linspace(0.01, 50, 200)
    
    for ax, T in zip(axes.flat, T_values):
        phases = []
        for ui in ui_range:
            theta = complex(0.5, ui) * a3
            A, B = inference.numerical_computer.compute_A_B(
                T, theta, params['m'], params['omega'], 
                params['sigma'], params['x0']
            )
            phi = np.exp(np.trace(A @ params['x0']) + B)
            phases.append(np.angle(phi))
        
        ax.plot(ui_range, phases, 'b-', linewidth=1)
        ax.set_xlabel('$u_i$')
        ax.set_ylabel('Phase of $\Phi$')
        ax.set_title(f'T = {T}')
        ax.grid(True, alpha=0.3)
        
        # Mark discontinuities
        phase_diff = np.abs(np.diff(phases))
        jumps = np.where(phase_diff > np.pi)[0]
        for j in jumps:
            ax.axvline(ui_range[j], color='r', linestyle='--', alpha=0.5)
        
        ax.set_ylim([-np.pi - 0.5, np.pi + 0.5])
        clear_jax_cache()
    
    plt.suptitle('Phase Analysis (cf. Kahl-Jäckel Figure 4)', fontsize=14)
    plt.tight_layout()
    return fig

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Run all evaluations."""
    print("="*70)
    print("NEURAL NETWORK MODEL EVALUATION")
    print("="*70)
    print(f"\nMode: {MODE}")
    print(f"Test mode: {REAL_CASE_TEST_MODE}")
    print(f"Real-only mode: {is_real_mode()}")
    
    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Load model
    inference = load_model_and_inference()
    
    # Get parameters
    params = get_test_parameters()
    print("\nTest parameters:")
    print(f"  T = {params['T']}")
    print(f"  m = \n{params['m']}")
    print(f"  omega = \n{params['omega']}")
    print(f"  sigma = \n{params['sigma']}")
    
    # Run evaluations
    print("\n" + "="*70)
    print("RUNNING EVALUATIONS")
    print("="*70)
    
    a3 = np.eye(2)
    ui_range= np.linspace(0, 25, 10)

   

    # 1. Characteristic function profile
    plot_characteristic_function_profile(
        inference, params, nb_point=50,
        save_path=output_dir / "01_char_func_profile.png"
    )
    clear_jax_cache()
    
    # 2. Parity plots
    plot_parity(
        inference, params, n_samples=50,
        save_path=output_dir / "02_parity_plots.png"
    )
    clear_jax_cache()
    
    # 3. Error distribution
    plot_error_distribution(
        inference, params, n_samples=50,
        save_path=output_dir / "03_error_distribution.png"
    )
    clear_jax_cache()
    
    # 4. Error vs parameters
    plot_error_vs_parameters(
        inference, params, n_samples=50,
        save_path=output_dir / "04_error_vs_parameters.png"
    )
    clear_jax_cache()
    
    # 5. Speed benchmark
    benchmark_speed(
        inference, params, n_samples=20,
        save_path=output_dir / "05_speed_benchmark.png"
    )
    clear_jax_cache()
    
    # 6. Marginal PDF
    plot_marginal_pdf(
        inference, params, n_points=50,
        save_path=output_dir / "06_marginal_pdf.png"
    )
    clear_jax_cache()
    
    print("diagnose_branch_cut_issues")    
    diagnose_branch_cut_issues(inference, params, T=1.0, a3=a3, ui_range=ui_range, ur=0.5)
    clear_jax_cache()

    print("plot_phase_analysis")
    plot_phase_analysis(inference, params, T_values=[1, 5, 10, 20])
    clear_jax_cache()
    
    # Summary
    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)
    print(f"\nAll figures saved to: {output_dir}")
    print("\nFigures:")
    print("  1. 01_char_func_profile.png")
    print("  2. 02_parity_plots.png")
    print("  3. 03_error_distribution.png")
    print("  4. 04_error_vs_parameters.png")
    print("  5. 05_speed_benchmark.png")
    print("  6. 06_marginal_pdf.png")
    
    plt.show()


if __name__ == "__main__":
    main()
