#!/usr/bin/env python
"""
Example: Complete Workflow for Wishart Neural Operator
======================================================

This script demonstrates the full workflow:
1. Configure and create model
2. Generate training data using Wishart.py
3. Train the model
4. Save the trained model
5. Load and use for inference
6. Validate against numerical implementation

Handles complex-valued theta, A, and B.

Usage:
    python run_example.py --wishart_path /path/to/Linear_rational_wishart

Author: Da Fonseca, Dawui, Malevergne
"""

import argparse
import numpy as np
from pathlib import Path
import os
import shutil
import jax
import gc
import jax.numpy as jnp



from linear_rational_wishart.core.wishart_jump import WishartWithJump

from linear_rational_wishart.neural_operator.config import (
    WishartPINNConfig
)
from linear_rational_wishart.neural_operator.model import (
    WishartPINNModel,
    WishartCharFuncNetwork,
    HighwayBlock,
    matrix_to_upper_tri,
    upper_tri_to_matrix,
    complex_upper_tri_to_matrix,
    # matrix_to_upper_tri

)

# Data generation
from linear_rational_wishart.neural_operator.data_generation import (
    WishartCharFuncComputer,
    WishartDataGenerator,
    complex_matrix_to_upper_tri
)

# Training
from linear_rational_wishart.neural_operator.training import (
    train_model,
    plot_training_history,
    TrainState
)

# Inference
from linear_rational_wishart.neural_operator.inference import (
    WishartPINNInference,
    validate_model,
    benchmark_throughput
)


def main_old(wishart_path: str,
         output_dir: str = "./output",
         generate_training_data: bool = False):
    """Run complete example workflow."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("WISHART NEURAL OPERATOR - Example Workflow")
    print("=" * 70)
    
    # =========================================================================
    # 1. CONFIGURE
    # =========================================================================
    print("\n[1/6] Configuring model...")
    
    config = WishartPINNConfig(
        dim=2,
        hidden_dim=128,
        num_highway_blocks=6,
        batch_size=512,
        num_epochs=1000,
        learning_rate=3e-4,
        seed=42
    )
    print(config)
    
    # =========================================================================
    # 2. CREATE MODEL
    # =========================================================================
    print("\n[2/6] Creating model...")
    
    model = WishartPINNModel(config)
    print(model.summary())
    
    # =========================================================================
    # 3. GENERATE DATA
    # =========================================================================
    print("\n[3/6] Generating or loading training data...")
    
    if generate_training_data:
        print("Generating training data...")

        generator = WishartDataGenerator(config, wishart_module_path=wishart_path)
    
        n_train = 300
        n_val = 50
    
        print(f"Generating {n_train} training samples...")
        train_data = generator.generate_dataset(n_samples=n_train)
    
        print(f"Generating {n_val} validation samples...")
        val_data = generator.generate_dataset(n_samples=n_val)
    
        generator.save_dataset(train_data, output_dir / "train_data.npz")
        generator.save_dataset(val_data, output_dir / "val_data.npz")
    
    else:
        print("Loading existing datasets...")
        # train_data = WishartDataGenerator.load_dataset(output_dir / "train_data.npz")
        # val_data = WishartDataGenerator.load_dataset(output_dir / "val_data.npz")

        train_data = WishartDataGenerator.load_dataset(output_dir / "normalization_stats.npz")
        val_data = WishartDataGenerator.load_dataset(output_dir / "val_data.npz")

   
    # =========================================================================
    # 4. TRAIN
    # =========================================================================
    print("\n[4/6] Training model...")
    
    save_path = output_dir / "models"
    
    best_params, history = train_model(
        model, config, train_data, val_data,
        save_path=str(save_path),
        use_lr_schedule=True
    )
    
    plot_training_history(history, save_path=str(output_dir / "training_history.png"))
    
    # =========================================================================
    # 5. INFERENCE
    # =========================================================================
    print("\n[5/6] Testing inference...")
    
    inference = WishartPINNInference(
        model, 
        best_params,
        wishart_module_path=wishart_path
    )
    
    # Example computation with COMPLEX theta
    d = 2
    T = 1.0
    
    # Create complex theta = z * a3
    a3 = np.array([[1.0, 0.0], [0.0, 1.0]])
    z = complex(config.ur, 1.0)  # ur from config, ui = 1.0
    theta = z * a3  # Complex theta
    
    m = np.array([[-0.5, 0.0], [0.0, -0.5]])  # Diagonal (as per config)
    omega = np.array([[0.3, 0.05], [0.05, 0.3]])  # Symmetric
    sigma = np.array([[0.5, 0.1], [0.1, 0.5]])  # Symmetric
    x0 = np.eye(d)
    
    T_arr = jnp.array([[T]])
    theta_arr = jnp.array([complex_matrix_to_upper_tri(theta)])
    m_arr = jnp.array([m.flatten()])
    omega_arr = jnp.array([matrix_to_upper_tri(omega)])
    sigma_arr = jnp.array([matrix_to_upper_tri(sigma)])

    inputs = jnp.concatenate([T_arr, theta_arr, m_arr, omega_arr, sigma_arr], axis=-1)


    print("\nExample computation:")
    print(f"  T = {T}")
    print(f"  z = {z}")
    print(f"  a3 = \n{a3}")
    print(f"  theta = z * a3 = \n{theta}")
    
    A_nn, B_nn = inference.compute_A_B(T, theta, m, omega, sigma)
    phi_nn = inference.compute_characteristic_function(T, theta, m, omega, sigma, x0)
    
    print(f"\nNeural Network Results:")
    print(f"  A (complex) = \n{A_nn}")
    print(f"  B (complex) = {B_nn}")
    print(f"  Φ (complex) = {phi_nn}")
    
    # Compare with numerical
    comparison = inference.compare_with_numerical(T, theta, m, omega, sigma, x0)
    
    print(f"\nNumerical (Wishart.py) Results:")
    print(f"  A = \n{comparison['A_num']}")
    print(f"  B = {comparison['B_num']}")
    print(f"  Φ = {comparison['phi_num']}")
    
    print(f"\nComparison:")
    print(f"  Relative Error (A):   {comparison['error_A']:.2e}")
    print(f"  Relative Error (B):   {comparison['error_B']:.2e}")
    print(f"  Relative Error (Φ):   {comparison['error_phi']:.2e}")
    print(f"  Time (NN):            {comparison['time_nn']*1000:.3f} ms")
    print(f"  Time (Numerical):     {comparison['time_num']*1000:.3f} ms")
    print(f"  Speedup:              {comparison['speedup']:.1f}x")
    
    # =========================================================================
    # 6. VALIDATE
    # =========================================================================
    print("\n[6/6] Full validation...")
    
    validation_results = validate_model(inference, config, n_test=100)
    
    # Benchmark throughput
    throughput = benchmark_throughput(inference, config, n_samples=1000)
    
    # =========================================================================
    # DEMONSTRATE SAVE/LOAD
    # =========================================================================
    print("\n" + "-" * 50)
    print("Testing save/load functionality...")
    
    loaded_inference = WishartPINNInference.from_saved_model(
        str(save_path / "final"),
        wishart_module_path=wishart_path
    )
    
    phi_loaded = loaded_inference.compute_characteristic_function(T, theta, m, omega, sigma, x0)
    print(f"Φ from loaded model: {phi_loaded}")
    print(f"Match: {np.isclose(phi_nn, phi_loaded)}")
    
    # =========================================================================
    # DONE
    # =========================================================================
    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print(f"\nOutputs saved to: {output_dir}")
    print(f"  - Training data:    {output_dir}/train_data.npz")
    print(f"  - Validation data:  {output_dir}/val_data.npz")
    print(f"  - Training plot:    {output_dir}/training_history.png")
    print(f"  - Model checkpoints: {save_path}/")
    print(f"  - Final model:      {save_path}/final/")
    
    return inference, validation_results

def main(wishart_path: str,
         output_dir: str = "./output",
         generate_training_data: bool = False):
    """Run complete example workflow with normalization."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("WISHART NEURAL OPERATOR - Example Workflow")
    print("=" * 70)
    
    # =========================================================================
    # 1. CONFIGURE
    # =========================================================================
    print("\n[1/6] Configuring model...")
    
    config = WishartPINNConfig(
        dim=2,
        hidden_dim=128,#256,#128,
        num_highway_blocks=6,#8,#6,
        batch_size=512,
        num_epochs=3000,#1000,
        learning_rate=3e-4,
        seed=42
    )
    print(config)
    
    # =========================================================================
    # 2. CREATE MODEL
    # =========================================================================
    print("\n[2/6] Creating model...")
    
    model = WishartPINNModel(config)
    print(model.summary())
    
    # =========================================================================
    # 3. LOAD AND NORMALIZE DATA
    # =========================================================================
    print("\n[3/6] Loading and normalizing training data...")
    
    if generate_training_data:
        print("Generating training data...")
        generator = WishartDataGenerator(config, wishart_module_path=wishart_path)
    
        n_train = 300
        n_val = 50
    
        print(f"Generating {n_train} training samples...")
        train_data = generator.generate_dataset(n_samples=n_train)
    
        print(f"Generating {n_val} validation samples...")
        val_data = generator.generate_dataset(n_samples=n_val)
    
        generator.save_dataset(train_data, output_dir / "train_data.npz")
        generator.save_dataset(val_data, output_dir / "val_data.npz")
        
        # Convert to numpy dict
        train_data = {k: np.array(v) for k, v in train_data.items()}
        val_data = {k: np.array(v) for k, v in val_data.items()}
    
    else:
        print("Loading existing datasets...")
        # Load as dictionaries
        train_data_raw = np.load(output_dir / "train_data.npz")
        val_data_raw = np.load(output_dir / "val_data.npz")
        
        # Convert to mutable dictionaries
        train_data = {key: train_data_raw[key].copy() for key in train_data_raw.files}
        val_data = {key: val_data_raw[key].copy() for key in val_data_raw.files}
    
    # -------------------------------------------------------------------------
    # NORMALIZATION
    # -------------------------------------------------------------------------
    print("\nComputing normalization statistics...")
    
    # Compute stats from TRAINING data only
    A_mean = np.mean(train_data['A'], axis=0)
    A_std = np.std(train_data['A'], axis=0) + 1e-8
    B_mean = np.mean(train_data['B'], axis=0)
    B_std = np.std(train_data['B'], axis=0) + 1e-8
    
    print(f"  A: mean shape={A_mean.shape}, std shape={A_std.shape}")
    print(f"  B: mean shape={B_mean.shape}, std shape={B_std.shape}")
    
    # Save normalization stats (IMPORTANT: needed for inference!)
    norm_stats_path = output_dir / "normalization_stats.npz"
    np.savez(norm_stats_path, 
             A_mean=A_mean, A_std=A_std, 
             B_mean=B_mean, B_std=B_std)
    print(f"  Saved normalization stats to: {norm_stats_path}")
    
    # Normalize both datasets
    train_data['A'] = (train_data['A'] - A_mean) / A_std
    train_data['B'] = (train_data['B'] - B_mean) / B_std
    val_data['A'] = (val_data['A'] - A_mean) / A_std
    val_data['B'] = (val_data['B'] - B_mean) / B_std
    
    print(f"\nAfter normalization:")
    print(f"  Train A: mean={np.mean(train_data['A']):.4f}, std={np.std(train_data['A']):.4f}")
    print(f"  Train B: mean={np.mean(train_data['B']):.4f}, std={np.std(train_data['B']):.4f}")
    
    # =========================================================================
    # 4. TRAIN
    # =========================================================================
    print("\n[4/6] Training model...")
    
    save_path = output_dir / "models"
    
    best_params, history = train_model(
        model, config, train_data, val_data,
        save_path=str(save_path),
        use_lr_schedule=True
    )
    
    plot_training_history(history, save_path=str(output_dir / "training_history.png"))
    
    # =========================================================================
    # 5. INFERENCE (with denormalization)
    # =========================================================================
    print("\n[5/6] Testing inference...")
    
    # Load normalization stats
    norm_stats = np.load(norm_stats_path)
    
    # Example computation with COMPLEX theta
    d = 2
    T = 1.0
    
    # Create complex theta = z * a3
    a3 = np.array([[1.0, 0.0], [0.0, 1.0]])
    z = complex(config.ur, 1.0)  # ur from config, ui = 1.0
    theta = z * a3  # Complex theta
    
    m = np.array([[-0.5, 0.0], [0.0, -0.5]])  # Diagonal (as per config)
    omega = np.array([[0.3, 0.05], [0.05, 0.3]])  # Symmetric
    sigma = np.array([[0.5, 0.1], [0.1, 0.5]])  # Symmetric
    x0 = np.eye(d)
    
    print("\nExample computation:")
    print(f"  T = {T}")
    print(f"  z = {z}")
    print(f"  a3 = \n{a3}")
    print(f"  theta = z * a3 = \n{theta}")
    
    # -------------------------------------------------------------------------
    # Manual inference with denormalization
    # -------------------------------------------------------------------------
    # Prepare inputs
    T_arr = jnp.array([[T]])
    theta_arr = jnp.array([complex_matrix_to_upper_tri(theta)])
    m_arr = jnp.array([m.flatten()])
    omega_arr = jnp.array([matrix_to_upper_tri(omega)])
    sigma_arr = jnp.array([matrix_to_upper_tri(sigma)])
    
    inputs = jnp.concatenate([T_arr, theta_arr, m_arr, omega_arr, sigma_arr], axis=-1)
    
    # Forward pass (outputs are NORMALIZED)
    A_norm, B_norm = model.network.apply(best_params, inputs)
    
    # DENORMALIZE
    A_denorm = np.array(A_norm) * norm_stats['A_std'] + norm_stats['A_mean']
    B_denorm = np.array(B_norm) * norm_stats['B_std'] + norm_stats['B_mean']
    
    # Convert to complex
    A_nn = complex_upper_tri_to_matrix(A_denorm[0], d)
    B_nn = complex(float(B_denorm[0, 0]), float(B_denorm[0, 1]))
    
    # Compute characteristic function
    phi_nn = np.exp(np.trace(A_nn @ x0) + B_nn)
    
    print(f"\nNeural Network Results (with denormalization):")
    print(f"  A (complex) = \n{A_nn}")
    print(f"  B (complex) = {B_nn}")
    print(f"  Φ (complex) = {phi_nn}")
    
    # -------------------------------------------------------------------------
    # Compare with numerical (Wishart.py)
    # -------------------------------------------------------------------------
    from linear_rational_wishart.core.wishart_jump import WishartWithJump
    
    wishart = WishartWithJump(d, x0, omega, m, sigma)
    wishart.maturity = T
    
    A_num = wishart.compute_a(T, theta)
    B_num = wishart.compute_b(T, theta)
    phi_num = np.exp(np.trace(A_num @ x0) + B_num)
    
    print(f"\nNumerical (Wishart.py) Results:")
    print(f"  A = \n{A_num}")
    print(f"  B = {B_num}")
    print(f"  Φ = {phi_num}")
    
    # Compute errors
    error_A = np.linalg.norm(A_nn - A_num) / (np.linalg.norm(A_num) + 1e-10)
    error_B = np.abs(B_nn - B_num) / (np.abs(B_num) + 1e-10)
    error_phi = np.abs(phi_nn - phi_num) / (np.abs(phi_num) + 1e-10)
    
    print(f"\nComparison:")
    print(f"  Relative Error (A):   {error_A:.2e}")
    print(f"  Relative Error (B):   {error_B:.2e}")
    print(f"  Relative Error (Φ):   {error_phi:.2e}")
    
    # =========================================================================
    # 6. VALIDATE (multiple test cases)
    # =========================================================================
    print("\n[6/6] Full validation...")
    
    n_test = 100
    errors_A = []
    errors_B = []
    errors_phi = []
    
    # Sample random test cases
    key = jax.random.PRNGKey(999)
    
    for i in range(n_test):
        key, *subkeys = jax.random.split(key, 10)
        
        # Random parameters within config ranges
        T_test = float(jax.random.uniform(subkeys[0], minval=config.T_min, maxval=config.T_max))
        
        # Random a3 and ui for theta
        a3_diag = float(jax.random.uniform(subkeys[1], minval=config.theta_min, maxval=config.theta_max))
        a3_test = np.eye(d) * a3_diag
        ui_test = float(jax.random.uniform(subkeys[2], minval=config.ui_min, maxval=config.ui_max))
        z_test = complex(config.ur, ui_test)
        theta_test = z_test * a3_test
        
        # Random m, omega, sigma
        m_diag = float(jax.random.uniform(subkeys[3], minval=config.m_diag_min, maxval=config.m_diag_max))
        m_test = np.eye(d) * m_diag
        
        omega_diag = float(jax.random.uniform(subkeys[4], minval=config.omega_diag_min, maxval=config.omega_diag_max))
        omega_test = np.eye(d) * omega_diag
        
        sigma_diag = float(jax.random.uniform(subkeys[5], minval=config.sigma_diag_min, maxval=config.sigma_diag_max))
        sigma_test = np.eye(d) * sigma_diag
        
        try:
            # Neural network prediction
            T_arr = jnp.array([[T_test]])
            theta_arr = jnp.array([complex_matrix_to_upper_tri(theta_test)])
            m_arr = jnp.array([m_test.flatten()])
            omega_arr = jnp.array([matrix_to_upper_tri(omega_test)])
            sigma_arr = jnp.array([matrix_to_upper_tri(sigma_test)])
            
            inputs = jnp.concatenate([T_arr, theta_arr, m_arr, omega_arr, sigma_arr], axis=-1)
            
            A_norm, B_norm = model.network.apply(best_params, inputs)
            A_denorm = np.array(A_norm) * norm_stats['A_std'] + norm_stats['A_mean']
            B_denorm = np.array(B_norm) * norm_stats['B_std'] + norm_stats['B_mean']
            
            A_nn_test = complex_upper_tri_to_matrix(A_denorm[0], d)
            B_nn_test = complex(float(B_denorm[0, 0]), float(B_denorm[0, 1]))
            phi_nn_test = np.exp(np.trace(A_nn_test @ x0) + B_nn_test)
            
            # Numerical
            wishart = WishartWithJump(d, x0, omega_test, m_test, sigma_test)
            wishart.maturity = T_test
            A_num_test = wishart.compute_a(T_test, theta_test)
            B_num_test = wishart.compute_b(T_test, theta_test)
            phi_num_test = np.exp(np.trace(A_num_test @ x0) + B_num_test)
            
            # Errors
            err_A = np.linalg.norm(A_nn_test - A_num_test) / (np.linalg.norm(A_num_test) + 1e-10)
            err_B = np.abs(B_nn_test - B_num_test) / (np.abs(B_num_test) + 1e-10)
            err_phi = np.abs(phi_nn_test - phi_num_test) / (np.abs(phi_num_test) + 1e-10)
            
            errors_A.append(float(err_A))
            errors_B.append(float(err_B))
            errors_phi.append(float(err_phi))
            
        except Exception as e:
            print(f"  Warning: Test {i} failed: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    print(f"Number of test cases: {len(errors_A)}")
    print(f"\nRelative Errors:")
    print(f"  A matrix:  mean = {np.mean(errors_A):.2e}, max = {np.max(errors_A):.2e}")
    print(f"  B scalar:  mean = {np.mean(errors_B):.2e}, max = {np.max(errors_B):.2e}")
    print(f"  Char func: mean = {np.mean(errors_phi):.2e}, max = {np.max(errors_phi):.2e}")
    print("=" * 60)
    
    # =========================================================================
    # DONE
    # =========================================================================
    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print(f"\nOutputs saved to: {output_dir}")
    print(f"  - Training data:       {output_dir}/train_data.npz")
    print(f"  - Validation data:     {output_dir}/val_data.npz")
    print(f"  - Normalization stats: {output_dir}/normalization_stats.npz")
    print(f"  - Training plot:       {output_dir}/training_history.png")
    print(f"  - Model checkpoints:   {save_path}/")
    print(f"  - Final model:         {save_path}/final/")
    
    return best_params, history, norm_stats


def load_and_compare_characteristic_function(wishart_path: str, output_dir: str = "./output"):
    """Load trained model and compare characteristic functions."""
    
    output_dir = Path(output_dir)
    save_path = output_dir / "models"

    # Load config to get ur value
    config = WishartPINNConfig()

    loaded_inference = WishartPINNInference.from_saved_model(
        str(save_path / "final"),
        wishart_module_path=wishart_path
    )
    
    d = 2
    
    # Test parameters
    T_list = [0.5, 1.0, 2.0]
    
    # a3 base matrices (real, symmetric positive)
    a3_list = [
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([[1.5, 0.0], [0.0, 1.5]]),
        np.array([[2.0, 0.0], [0.0, 2.0]]),
        np.array([[0.5, 0.1], [0.1, 0.5]]),
    ]
    
    # Different ui values to test
    ui_list = [0.0, 1.0, 5.0, 10.0]
    
    # Fixed parameters
    m = np.array([[-0.5, 0.0], [0.0, -0.5]])
    omega = np.array([[0.3, 0.05], [0.05, 0.3]])
    sigma = np.array([[0.5, 0.1], [0.1, 0.5]])
    x0 = np.eye(d)
    
    print("=" * 70)
    print("COMPARING CHARACTERISTIC FUNCTIONS")
    print("=" * 70)
    
    for T in T_list:
        for a3 in a3_list:
            for ui in ui_list:
                # Create complex theta
                z = complex(config.ur, ui)
                theta = z * a3
                
                print("\n" + "=" * 50)
                print(f"T = {T}, z = {z}")
                print(f"a3 = \n{a3}")
                print(f"theta = z * a3 = \n{theta}")
                
                # Neural network prediction
                A_nn, B_nn = loaded_inference.compute_A_B(T, theta, m, omega, sigma)
                phi_nn = loaded_inference.compute_characteristic_function(T, theta, m, omega, sigma, x0)
                
                # Numerical computation using WishartWithJump
                wishart = WishartWithJump(d, x0, omega, m, sigma)
                wishart.maturity = T
                
                A_num = wishart.compute_a(T, theta)
                B_num = wishart.compute_b(T, theta)
                phi_num = wishart.phi_one(1, theta)
                
                print(f"\nNeural Network:")
                print(f"  A = \n{A_nn}")
                print(f"  B = {B_nn}")
                print(f"  Φ = {phi_nn}")
                
                print(f"\nNumerical:")
                print(f"  A = \n{A_num}")
                print(f"  B = {B_num}")
                print(f"  Φ = {phi_num}")
                
                # Compute errors
                error_A = np.linalg.norm(A_nn - A_num) / (np.linalg.norm(A_num) + 1e-10)
                error_B = np.abs(B_nn - B_num) / (np.abs(B_num) + 1e-10)
                error_phi = np.abs(phi_nn - phi_num) / (np.abs(phi_num) + 1e-10)
                
                print(f"\nRelative Errors:")
                print(f"  A: {error_A:.2e}")
                print(f"  B: {error_B:.2e}")
                print(f"  Φ: {error_phi:.2e}")
                print(f"  Match (Φ): {np.isclose(phi_nn, phi_num, rtol=1e-2)}")


def generate_data(wishart_path: str,
                  first_file_id: int = 0,
                  sub_dats_set_output_dir: str = "./output",
                  file_patern: str = "chunk",
                  max_size_per_generation: int = 10,
                  nb_generation: int = 50):
    """Generate training data in chunks."""
    
    output_dir = Path(sub_dats_set_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = WishartPINNConfig(
        dim=2,
        hidden_dim=128,
        num_highway_blocks=6,
        batch_size=256,
        num_epochs=500,
        learning_rate=1e-3,
        seed=42
    )
    
    print("\nGenerating training data...")
    print(f"  Output dir: {output_dir}")
    print(f"  Samples per chunk: {max_size_per_generation}")
    print(f"  Number of chunks: {nb_generation}")

    generator = WishartDataGenerator(config, wishart_module_path=wishart_path)
    
    n_train = max_size_per_generation
    n_chunks = nb_generation // max_size_per_generation if nb_generation > max_size_per_generation else nb_generation

    for i in range(n_chunks):
        print(f"\nChunk {i+1}/{n_chunks}")
        print(f"  Generating {n_train} samples...")
        
        train_data = generator.generate_dataset(n_samples=n_train)
        
        current_file_id = i + first_file_id
        current_file = output_dir / f"{file_patern}_{current_file_id}.npz"
        generator.save_dataset(train_data, current_file)
        
        clear_jax_cache()
    
    print(f"\nData generation complete. {n_chunks} chunks saved to {output_dir}")


def load_merge_data(wishart_path: str,
                    output_dir: str = "./output",
                    chunk_dir: str = "./sub_set_data",
                    file_patern: str = "chunk_*.npz",
                    file_name: str = "train_data.npz"):
    """Load and merge chunked data files."""
    
    merged_data = WishartDataGenerator.load_and_merge_chunks(
        chunk_dir=chunk_dir, 
        pattern=file_patern
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = WishartPINNConfig(dim=2, seed=42)
    generator = WishartDataGenerator(config, wishart_module_path=wishart_path)
    
    generator.save_dataset(merged_data, output_dir / file_name)
    print(f"Merged data saved to {output_dir / file_name}")

def data_statistics(output_dir):
     # import numpy as np

    
    train_data = np.load(output_dir  + r"\train_data.npz")

    # print("A statistics:")
    # print(f"  mean: {np.mean(np.abs(train_data['A'])):.2f}")
    # print(f"  std:  {np.std(train_data['A']):.2f}")
    # print(f"  max:  {np.max(np.abs(train_data['A'])):.2f}")

    # print("\nB statistics:")
    # print(f"  mean: {np.mean(np.abs(train_data['B'])):.2f}")
    # print(f"  std:  {np.std(train_data['B']):.2f}")
    # print(f"  max:  {np.max(np.abs(train_data['B'])):.2f}")


    
    # train_data = dict(np.load("train_data.npz"))
    # val_data = dict(np.load("val_data.npz"))

    train_data_raw = np.load(output_dir  + r"\train_data.npz")
    val_data_raw   = np.load(output_dir  + r"\val_data.npz")
   
    train_data = {key: train_data_raw[key].copy() for key in train_data_raw.files}
    val_data = {key: val_data_raw[key].copy() for key in val_data_raw.files}

    # Compute stats
    A_mean = np.mean(train_data['A'], axis=0)
    A_std = np.std(train_data['A'], axis=0) + 1e-8
    B_mean = np.mean(train_data['B'], axis=0)
    B_std = np.std(train_data['B'], axis=0) + 1e-8

    # Normalize
    train_data['A'] = (train_data['A'] - A_mean) / A_std
    train_data['B'] = (train_data['B'] - B_mean) / B_std
    val_data['A'] = (val_data['A'] - A_mean) / A_std
    val_data['B'] = (val_data['B'] - B_mean) / B_std

    print("After normalization:")
    print(f"A: mean={np.mean(train_data['A']):.4f}, std={np.std(train_data['A']):.4f}")
    print(f"B: mean={np.mean(train_data['B']):.4f}, std={np.std(train_data['B']):.4f}")

    # Save stats for later
    np.savez(output_dir  + r"\normalization_stats.npz", #"normalization_stats.npz", 
             A_mean=A_mean, A_std=A_std, B_mean=B_mean, B_std=B_std)


def clear_jax_cache():
    """Clear JAX compilation cache."""
    jax.clear_caches()
    gc.collect()
    
    jax_cache = os.path.join(os.environ.get('LOCALAPPDATA', ''), 'jax')
    if os.path.exists(jax_cache):
        shutil.rmtree(jax_cache, ignore_errors=True)


if __name__ == "__main__":
    # Configuration
    main_folder = r"E:\OneDrive\Dropbox\LinearRationalWishart_Work\Code\ED\LinearRationalWishart\LinearRationalWishart_NewCode\linear_rational_wishart"
    main_ouput_folder = r"E:\OneDrive\Dropbox\LinearRationalWishart_Work\Code\ED\LinearRationalWishart\LinearRationalWishart_NewCode\Output_results\neural_operator\saved_models"
    ouput_folder = main_ouput_folder
    
    clear_jax_cache()

    # =========================================================================
    # SELECT WHICH TASKS TO RUN
    # =========================================================================
    run_train_model = True#False
    test_model = False #True#False
    run_data_generation = False#True
    merge_data_set = False#True
    check_data_stats=False
    # =========================================================================
    # Data statistic
    # =========================================================================
    if check_data_stats:
        data_statistics(main_ouput_folder)

    # =========================================================================
    # TRAIN MODEL
    # =========================================================================
    if run_train_model:
        generate_training_data = False
        main(main_folder, ouput_folder, generate_training_data)

    # =========================================================================
    # TEST MODEL
    # =========================================================================
    if test_model:
        load_and_compare_characteristic_function(main_folder, ouput_folder)

    # =========================================================================
    # GENERATE DATA
    # =========================================================================
    if run_data_generation:
        train_data_sets_folder = main_ouput_folder + r"\training_data"
        validation_data_sets_folder = main_ouput_folder + r"\validation_data"
        
        # Generate training data
        generate_data(
            main_folder,
            first_file_id=0,
            sub_dats_set_output_dir=train_data_sets_folder,
            file_patern="chunk",
            max_size_per_generation=100,#100,
            nb_generation=10000
        )

        # Generate validation data
        generate_data(
            main_folder,
            first_file_id=0,
            sub_dats_set_output_dir=validation_data_sets_folder,
            file_patern="chunk",
            max_size_per_generation=100,##100,
            nb_generation=2000
        )

    # =========================================================================
    # MERGE DATA
    # =========================================================================
    if merge_data_set:
        print("\nMerging data sets...")
        
        train_data_sets_folder = main_ouput_folder + r"\training_data"
        load_merge_data(
            main_folder,
            output_dir=main_ouput_folder,
            chunk_dir=train_data_sets_folder,
            file_patern="chunk_*.npz",
            file_name="train_data.npz"
        )
        
        validation_data_sets_folder = main_ouput_folder + r"\validation_data"
        load_merge_data(
            main_folder,
            output_dir=main_ouput_folder,
            chunk_dir=validation_data_sets_folder,
            file_patern="chunk_*.npz",
            file_name="val_data.npz"
        )