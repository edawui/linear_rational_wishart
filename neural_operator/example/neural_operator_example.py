#!/usr/bin/env python
"""
Example: Complete Workflow for Wishart Neural Operator
======================================================

This script demonstrates the full workflow for BOTH modes:
1. Configure and create model (REAL or COMPLEX mode)
2. Generate training data using Wishart.py
3. Train the model
4. Save the trained model
5. Load and use for inference
6. Validate against numerical implementation

REAL mode:
    - Train on real theta, A, B
    - Use analytic extension for complex queries
    - Simpler, potentially easier to train

COMPLEX mode:
    - Train directly on complex theta, A, B
    - Direct inference for complex queries

Usage:
    python neural_operator_example.py

Author: Da Fonseca, Dawui, Malevergne
"""

import argparse
import math
import numpy as np
from pathlib import Path
import os
import shutil
import jax
import gc
import jax.numpy as jnp
import math

from linear_rational_wishart.core.wishart_jump import WishartWithJump

from linear_rational_wishart.neural_operator.config import (
    WishartPINNConfig,
    get_real_config,
    get_complex_config,
)
from linear_rational_wishart.neural_operator.model import (
    WishartPINNModel,
    WishartCharFuncNetwork,
    HighwayBlock,
    matrix_to_upper_tri,
    upper_tri_to_matrix,
    complex_upper_tri_to_matrix,
)

# Data generation
from linear_rational_wishart.neural_operator.data_generation import (
    WishartCharFuncComputer,
    WishartDataGenerator,
    complex_matrix_to_upper_tri,
    matrix_to_upper_tri_np,
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
    benchmark_throughput,
)


# =============================================================================
# HELPER FUNCTION FOR INPUT PREPARATION
# =============================================================================

def prepare_nn_inputs(T, theta, m, omega, sigma, mode="complex"):
    """
    Prepare inputs for neural network forward pass.
    
    Args:
        T: Time to maturity (float)
        theta: Symmetric matrix (d x d), real or complex depending on mode
        m: Real matrix (d x d)
        omega: Real symmetric matrix (d x d)
        sigma: Real symmetric matrix (d x d)
        mode: "real" or "complex"
    
    Returns:
        JAX array of shape (1, input_dim)
    """
    theta_np = np.asarray(theta)
    m_np = np.asarray(m)
    omega_np = np.asarray(omega)
    sigma_np = np.asarray(sigma)
    
    T_arr = np.array([[float(T)]])
    m_arr = np.array([m_np.flatten()])
    omega_arr = np.array([matrix_to_upper_tri_np(np.real(omega_np))])
    sigma_arr = np.array([matrix_to_upper_tri_np(np.real(sigma_np))])
    
    if mode == "real":
        theta_arr = np.array([matrix_to_upper_tri_np(np.real(theta_np))])
    else:
        theta_arr = np.array([complex_matrix_to_upper_tri(theta_np)])
    
    inputs_np = np.concatenate([T_arr, theta_arr, m_arr, omega_arr, sigma_arr], axis=-1)
    return jnp.array(inputs_np)


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def main(wishart_path: str,
         output_dir: str = "./output",
         generate_training_data: bool = False,
         mode: str = "real"):
    """
    Run complete example workflow.
    
    Args:
        wishart_path: Path to Wishart module
        output_dir: Output directory for data and models
        generate_training_data: Whether to generate new data
        mode: "real" or "complex"
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print(f"WISHART NEURAL OPERATOR - Example Workflow (MODE: {mode.upper()})")
    print("=" * 70)
    
    # =========================================================================
    # 1. CONFIGURE
    # =========================================================================
    print("\n[1/6] Configuring model...")
    
    if mode == "real":
        config = get_real_config(
            dim=2,
            hidden_dim=128,
            num_highway_blocks=6,
            batch_size=512,
            num_epochs=4000,
            learning_rate=3e-4,
            seed=42
        )
    else:
        config = get_complex_config(
            dim=2,
            hidden_dim=256,
            num_highway_blocks=8,
            batch_size=512,
            num_epochs=5000,
            learning_rate=3e-4,
            ui_max=25.0,
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
    
    # Use mode-specific data files
    train_file = f"train_data_{mode}.npz"
    val_file = f"val_data_{mode}.npz"
    norm_file = f"normalization_stats_{mode}.npz"
    
    if generate_training_data:
        print(f"Generating {mode.upper()} mode training data...")
        generator = WishartDataGenerator(config, wishart_module_path=wishart_path)
    
        n_train = 10000 if mode == "real" else 10000
        n_val = 2000
    
        print(f"Generating {n_train} training samples...")
        train_data = generator.generate_dataset(n_samples=n_train)
    
        print(f"Generating {n_val} validation samples...")
        val_data = generator.generate_dataset(n_samples=n_val)
    
        generator.save_dataset(train_data, output_dir / train_file)
        generator.save_dataset(val_data, output_dir / val_file)
        
        # Convert to numpy dict
        train_data = {k: np.array(v) for k, v in train_data.items()}
        val_data = {k: np.array(v) for k, v in val_data.items()}
    
    else:
        print("Loading existing datasets...")
        train_data_raw = np.load(output_dir / train_file)
        val_data_raw = np.load(output_dir / val_file)
        
        if mode =="complex":
            assert train_data_raw['theta'].shape[1] == 6, "Data must be complex mode (theta dim=6)"
            assert train_data_raw['A'].shape[1] == 6, "Data must be complex mode (A dim=6)"
            assert train_data_raw['B'].shape[1] == 2, "Data must be complex mode (B dim=2)"

        train_data = {key: train_data_raw[key].copy() for key in train_data_raw.files}
        val_data = {key: val_data_raw[key].copy() for key in val_data_raw.files}
    
       
    # -------------------------------------------------------------------------
    # NORMALIZATION
    # -------------------------------------------------------------------------
    print("\nComputing normalization statistics...")
    
    A_mean = np.mean(train_data['A'], axis=0)
    A_std = np.std(train_data['A'], axis=0) + 1e-8
    B_mean = np.mean(train_data['B'], axis=0)
    B_std = np.std(train_data['B'], axis=0) + 1e-8
    
    print(f"  A: mean shape={A_mean.shape}, std shape={A_std.shape}")
    print(f"  B: mean shape={B_mean.shape}, std shape={B_std.shape}")
    
    norm_stats_path = output_dir / norm_file
    np.savez(norm_stats_path, 
             A_mean=A_mean, A_std=A_std, 
             B_mean=B_mean, B_std=B_std)
    print(f"  Saved normalization stats to: {norm_stats_path}")
    
    # Normalize
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
    
    save_path = output_dir / f"models_{mode}"
    
    # Verify input dimension matches data
    expected_input_dim = 1 + 6 + 4 + 3 + 3  # T + theta(complex) + m + omega + sigma = 17
    actual_theta_dim = train_data['theta'].shape[1]
    print(f"\n  Expected theta dim: 6, Actual: {actual_theta_dim}")
    print(f"  Model input dim: {model.input_dim}")


    best_params, history = train_model(
        model, config, train_data, val_data,
        save_path=str(save_path),
        use_lr_schedule=True
       
    )
    
    plot_training_history(
        history, 
        save_path=str(output_dir / f"training_history_{mode}.png"),
        title_suffix=f"({mode.upper()} mode)"
    )
    
    # =========================================================================
    # 5. INFERENCE
    # =========================================================================
    print("\n[5/6] Testing inference...")
    
    norm_stats = dict(np.load(norm_stats_path))
    
    d = 2
    T = 1.0
    x0 = np.eye(d)
    
    # Test parameters
    m = np.array([[-0.5, 0.0], [0.0, -0.5]])
    omega = np.array([[0.3, 0.05], [0.05, 0.3]])
    sigma = np.array([[0.5, 0.1], [0.1, 0.5]])
    
    if mode == "real":
        # Test with real theta first
        theta_real = np.array([[1.0, 0.0], [0.0, 1.0]])
        print(f"\n--- Testing with REAL theta ---")
        print(f"  theta = \n{theta_real}")
        
        inputs = prepare_nn_inputs(T, theta_real, m, omega, sigma, mode="real")
        A_norm, B_norm = model.network.apply(best_params, inputs)
        
        A_denorm = np.array(A_norm) * norm_stats['A_std'] + norm_stats['A_mean']
        B_denorm = np.array(B_norm) * norm_stats['B_std'] + norm_stats['B_mean']
        
        A_nn = upper_tri_to_matrix(jnp.array(A_denorm[0]), d)
        B_nn = float(B_denorm[0, 0])
        phi_nn = np.exp(np.trace(np.array(A_nn) @ x0) + B_nn)
        
        # Numerical
        wishart = WishartWithJump(d, x0, omega, m, sigma)
        wishart.maturity = T
        A_num = wishart.compute_a(T, theta_real)
        B_num = wishart.compute_b(T, theta_real)
        phi_num = np.exp(np.trace(A_num @ x0) + B_num)
        
        print(f"\n  Neural Network: A =\n{np.array(A_nn)}")
        print(f"  Neural Network: B = {B_nn}")
        print(f"  Neural Network: Φ = {phi_nn}")
        print(f"  Numerical:      A =\n{np.real(A_num)}")
        print(f"  Numerical:      B = {np.real(B_num)}")
        print(f"  Numerical:      Φ = {np.real(phi_num)}")
        
        error_phi = np.abs(phi_nn - phi_num) / (np.abs(phi_num) + 1e-10)
        print(f"  Relative Error (Φ): {error_phi:.2e}")
        
        # Now test with complex theta using analytic extension
        print(f"\n--- Testing with COMPLEX theta (analytic extension) ---")
        a3 = np.array([[1.0, 0.0], [0.0, 1.0]])
        z = complex(config.ur, 5.0)
        theta_complex = z * a3
        print(f"  z = {z}")
        print(f"  theta = z * a3 = \n{theta_complex}")
        
        # Use inference class for analytic extension
        inference = WishartPINNInference(
            model, best_params,
            normalization_stats_path=str(norm_stats_path)
        )
        A_nn_c, B_nn_c = inference.compute_A_B(T, theta_complex, m, omega, sigma)
        phi_nn_c = inference.compute_characteristic_function(T, theta_complex, m, omega, sigma, x0)
        
        # Numerical
        A_num_c = wishart.compute_a(T, theta_complex)
        B_num_c = wishart.compute_b(T, theta_complex)
        phi_num_c = np.exp(np.trace(A_num_c @ x0) + B_num_c)
        
        print(f"\n  Neural Network (analytic ext): Φ = {phi_nn_c}")
        print(f"  Numerical:                      Φ = {phi_num_c}")
        error_phi_c = np.abs(phi_nn_c - phi_num_c) / (np.abs(phi_num_c) + 1e-10)
        print(f"  Relative Error (Φ): {error_phi_c:.2e}")
        
    else:  # COMPLEX mode
        a3 = np.array([[1.0, 0.0], [0.0, 1.0]])
        z = complex(config.ur, 1.0)
        theta = z * a3
        
        print(f"\n--- Testing with COMPLEX theta ---")
        print(f"  z = {z}")
        print(f"  theta = z * a3 = \n{theta}")
        
        inputs = prepare_nn_inputs(T, theta, m, omega, sigma, mode="complex")
        A_norm, B_norm = model.network.apply(best_params, inputs)
        
        A_denorm = np.array(A_norm) * norm_stats['A_std'] + norm_stats['A_mean']
        B_denorm = np.array(B_norm) * norm_stats['B_std'] + norm_stats['B_mean']
        
        A_nn = complex_upper_tri_to_matrix(jnp.array(A_denorm[0]), d)
        B_nn = complex(float(B_denorm[0, 0]), float(B_denorm[0, 1]))
        phi_nn = np.exp(np.trace(np.array(A_nn) @ x0) + B_nn)
        
        # Numerical
        wishart = WishartWithJump(d, x0, omega, m, sigma)
        wishart.maturity = T
        A_num = wishart.compute_a(T, theta)
        B_num = wishart.compute_b(T, theta)
        phi_num = np.exp(np.trace(A_num @ x0) + B_num)
        
        print(f"\n  Neural Network: Φ = {phi_nn}")
        print(f"  Numerical:      Φ = {phi_num}")
        
        error_phi = np.abs(phi_nn - phi_num) / (np.abs(phi_num) + 1e-10)
        print(f"  Relative Error (Φ): {error_phi:.2e}")
    
    # =========================================================================
    # 6. VALIDATE
    # =========================================================================
    print("\n[6/6] Full validation...")
    
    inference = WishartPINNInference(
        model, best_params,
        normalization_stats_path=str(norm_stats_path)
    )
    
    validation_results = validate_model(inference, config, n_test=100)
    
    # =========================================================================
    # DONE
    # =========================================================================
    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print(f"\nOutputs saved to: {output_dir}")
    print(f"  - Training data:       {output_dir}/{train_file}")
    print(f"  - Validation data:     {output_dir}/{val_file}")
    print(f"  - Normalization stats: {output_dir}/{norm_file}")
    print(f"  - Training plot:       {output_dir}/training_history_{mode}.png")
    print(f"  - Model checkpoints:   {save_path}/")
    print(f"  - Final model:         {save_path}/final/")
    
    return best_params, history, norm_stats


# =============================================================================
# TEST/COMPARE FUNCTION
# =============================================================================

def load_and_compare_characteristic_function(
    wishart_path: str, 
    output_dir: str = "./output",
    mode: str = "real"
):
    """Load trained model and compare characteristic functions."""
    
    output_dir = Path(output_dir)
    save_path = output_dir / f"models_{mode}"
    norm_stats_path = output_dir / f"normalization_stats_{mode}.npz"

    if mode == "real":
        config = get_real_config()
    else:
        config = get_complex_config()

    loaded_inference = WishartPINNInference.from_saved_model(
        str(save_path / "final"),
        wishart_module_path=wishart_path,
        normalization_stats_path=str(norm_stats_path)
    )
    
    d = 2
    
    # Test parameters
    T_list = [0.5, 1.0, 2.0]
    ui_list = [0.0, 1.0, 5.0, 10.0]
    
    m = np.array([[-0.5, 0.0], [0.0, -0.5]])
    omega = np.array([[0.3, 0.05], [0.05, 0.3]])
    sigma = np.array([[0.5, 0.1], [0.1, 0.5]])
    x0 = np.eye(d)
    
    print("=" * 70)
    print(f"COMPARING CHARACTERISTIC FUNCTIONS (mode={mode.upper()})")
    print("=" * 70)
    
    for T in T_list:
        for ui in ui_list:
            a3 = np.array([[1.0, 0.0], [0.0, 1.0]])
            z = complex(config.ur, ui)
            theta = z * a3
            
            print(f"\nT = {T}, z = {z}")
            
            # Neural network
            phi_nn = loaded_inference.compute_characteristic_function(T, theta, m, omega, sigma, x0)
            
            # Numerical
            wishart = WishartWithJump(d, x0, omega, m, sigma)
            wishart.maturity = T
            phi_num = wishart.phi_one(1, theta)
            
            error_phi = np.abs(phi_nn - phi_num) / (np.abs(phi_num) + 1e-10)
            
            print(f"  NN:  {phi_nn}")
            print(f"  Num: {phi_num}")
            print(f"  Err: {error_phi:.2e}")


# =============================================================================
# DATA GENERATION FUNCTIONS
# =============================================================================

def generate_data(wishart_path: str,
                  first_file_id: int = 0,
                  sub_dats_set_output_dir: str = "./output",
                  file_patern: str = "chunk",
                  max_size_per_generation: int = 100,
                  nb_generation: int = 100,
                  mode: str = "real"):
    """Generate training data in chunks."""
    
    output_dir = Path(sub_dats_set_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if mode == "real":
        config = get_real_config(dim=2, seed=42)
    else:
        config = get_complex_config(dim=2, seed=42)
    
    print(f"\nGenerating {mode.upper()} mode training data...")
    print(f"  Output dir: {output_dir}")
    print(f"  Samples per chunk: {max_size_per_generation}")

    generator = WishartDataGenerator(config, wishart_module_path=wishart_path)
    
    n_train = max_size_per_generation
    # n_chunks = max((int)(nb_generation/max_size_per_generation),1)
    n_chunks = max(nb_generation//max_size_per_generation,1)
    print(f"  Number of chunks: {nb_generation}")

    nb_generation = n_chunks * max_size_per_generation

    for i in range(n_chunks):
        print(f"\nChunk {i+1}/{n_chunks}")
        print(f"  Generating {n_train} samples...")
        nb_train = min(nb_generation- i * max_size_per_generation, max_size_per_generation)

        
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
                    file_name: str = "train_data.npz",
                    mode: str = "real"):
    """Load and merge chunked data files."""
    
    merged_data = WishartDataGenerator.load_and_merge_chunks(
        chunk_dir=chunk_dir, 
        pattern=file_patern
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if mode == "real":
        config = get_real_config(dim=2, seed=42)
    else:
        config = get_complex_config(dim=2, seed=42)
    
    generator = WishartDataGenerator(config, wishart_module_path=wishart_path)
    
    generator.save_dataset(merged_data, output_dir / file_name)
    print(f"Merged data saved to {output_dir / file_name}")


def data_statistics(output_dir: str, mode: str = "real"):
    """Compute and display data statistics."""
    
    train_file = f"train_data_{mode}.npz"
    val_file = f"val_data_{mode}.npz"
    
    train_path = os.path.join(output_dir, train_file)
    val_path = os.path.join(output_dir, val_file)
    
    train_data_raw = np.load(train_path)
    val_data_raw = np.load(val_path)
   
    train_data = {key: train_data_raw[key].copy() for key in train_data_raw.files}
    val_data = {key: val_data_raw[key].copy() for key in val_data_raw.files}

    print(f"Data shapes ({mode.upper()} mode):")
    for key in train_data:
        print(f"  {key}: {train_data[key].shape}")

    print("\nRaw statistics:")
    print(f"  A: mean={np.mean(np.abs(train_data['A'])):.4f}, std={np.std(train_data['A']):.4f}")
    print(f"  B: mean={np.mean(np.abs(train_data['B'])):.4f}, std={np.std(train_data['B']):.4f}")

    # Compute normalization stats
    A_mean = np.mean(train_data['A'], axis=0)
    A_std = np.std(train_data['A'], axis=0) + 1e-8
    B_mean = np.mean(train_data['B'], axis=0)
    B_std = np.std(train_data['B'], axis=0) + 1e-8

    # Normalized
    train_norm_A = (train_data['A'] - A_mean) / A_std
    train_norm_B = (train_data['B'] - B_mean) / B_std

    print("\nAfter normalization:")
    print(f"  A: mean={np.mean(train_norm_A):.4f}, std={np.std(train_norm_A):.4f}")
    print(f"  B: mean={np.mean(train_norm_B):.4f}, std={np.std(train_norm_B):.4f}")

    # Save stats
    norm_file = f"normalization_stats_{mode}.npz"
    norm_stats_path = os.path.join(output_dir, norm_file)
    np.savez(norm_stats_path, 
             A_mean=A_mean, A_std=A_std, B_mean=B_mean, B_std=B_std)
    print(f"\nSaved normalization stats to: {norm_stats_path}")


def clear_jax_cache():
    """Clear JAX compilation cache."""
    jax.clear_caches()
    gc.collect()
    
    local_app_data = os.environ.get('LOCALAPPDATA', '')
    if local_app_data:
        jax_cache = os.path.join(local_app_data, 'jax')
        if os.path.exists(jax_cache):
            shutil.rmtree(jax_cache, ignore_errors=True)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Configuration - UPDATE THESE PATHS FOR YOUR SYSTEM
    main_folder = r"E:\OneDrive\Dropbox\LinearRationalWishart_Work\Code\ED\LinearRationalWishart\LinearRationalWishart_NewCode\linear_rational_wishart"
    main_ouput_folder = r"E:\OneDrive\Dropbox\LinearRationalWishart_Work\Code\ED\LinearRationalWishart\LinearRationalWishart_NewCode\Output_results\neural_operator\saved_models"
    ouput_folder = main_ouput_folder
    
    clear_jax_cache()

    # =========================================================================
    # SELECT MODE: "real" or "complex"
    # =========================================================================
    MODE = "real"  # <-- CHANGE THIS TO "complex" FOR COMPLEX MODE
    MODE = "complex"  # <-- CHANGE THIS TO "complex" FOR COMPLEX MODE
    modes=["complex"]  ##["real","complex"]
    for MODE in modes:
        # =========================================================================
        # SELECT WHICH TASKS TO RUN
        # =========================================================================
        run_train_model = False
        test_model = False
        run_data_generation = True#False
        merge_data_set =      False#True#False
        check_data_stats =    False
    
        # =========================================================================
        # Data statistics
        # =========================================================================
        if check_data_stats:
            data_statistics(main_ouput_folder, mode=MODE)

   
        # =========================================================================
        # TEST MODEL
        # =========================================================================
        if test_model:
            load_and_compare_characteristic_function(main_folder, ouput_folder, mode=MODE)

        # =========================================================================
        # GENERATE DATA
        # =========================================================================
        if run_data_generation:
            train_data_sets_folder = os.path.join(main_ouput_folder, f"training_data_{MODE}")
            validation_data_sets_folder = os.path.join(main_ouput_folder, f"validation_data_{MODE}")
        
            # # Generate training data
            generate_data(
                main_folder,
                first_file_id=0,
                sub_dats_set_output_dir=train_data_sets_folder,
                file_patern="chunk",
                max_size_per_generation=100,
                nb_generation=5000,#15000,
                mode=MODE
            )

            # # Generate validation data
            generate_data(
                main_folder,
                first_file_id=0,
                sub_dats_set_output_dir=validation_data_sets_folder,
                file_patern="chunk",
                max_size_per_generation=100,
                nb_generation=1000,#3000,
                mode=MODE
            )

        # =========================================================================
        # MERGE DATA
        # =========================================================================
        if merge_data_set:
            print("\nMerging data sets...")
        
            train_data_sets_folder = os.path.join(main_ouput_folder, f"training_data_{MODE}")
            load_merge_data(
                main_folder,
                output_dir=main_ouput_folder,
                chunk_dir=train_data_sets_folder,
                file_patern="chunk_*.npz",
                file_name=f"train_data_{MODE}.npz",
                mode=MODE
            )
        
            validation_data_sets_folder = os.path.join(main_ouput_folder, f"validation_data_{MODE}")
            load_merge_data(
                main_folder,
                output_dir=main_ouput_folder,
                chunk_dir=validation_data_sets_folder,
                file_patern="chunk_*.npz",
                file_name=f"val_data_{MODE}.npz",
                mode=MODE
            )

        # =========================================================================
    
        # TRAIN MODEL
        # =========================================================================
        if run_train_model:
            generate_training_data = False#True  # Set to True to generate new data
            main(main_folder, ouput_folder, generate_training_data, mode=MODE)
