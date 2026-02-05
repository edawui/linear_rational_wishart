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

# Import neural_operator package
# from neural_operator import (
#     WishartPINNConfig,
#     WishartPINNModel,
#     WishartDataGenerator,
#     train_model,
#     plot_training_history,
#     WishartPINNInference,
#     validate_model,
#     benchmark_throughput
# )

from linear_rational_wishart.core.wishart_jump import WishartWithJump

from linear_rational_wishart.neural_operator.config import (
    WishartPINNConfig
)
from linear_rational_wishart.neural_operator.model import (
    WishartPINNModel,
    WishartCharFuncNetwork,
    HighwayBlock,
    matrix_to_upper_tri,
    upper_tri_to_matrix
)

# Data generation
from linear_rational_wishart.neural_operator.data_generation import (
    WishartCharFuncComputer,
    WishartDataGenerator
    # ,load_dataset,
    # load_and_merge_chunks
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

def main(wishart_path: str
         , output_dir: str = "./output"
         , generate_training_data=False):
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
    
    # config = WishartPINNConfig(
    #     dim=2,
    #     hidden_dim=128,
    #     num_highway_blocks=6,
    #     batch_size=256,
    #     num_epochs=500,      # Reduce for quick test
    #     learning_rate=1e-3,
    #     seed=42
    # )
    config = WishartPINNConfig(
    dim=2,
    hidden_dim=128,
    num_highway_blocks=6,
    batch_size=512,          # Increase from 256
    num_epochs=1000,         # More epochs with lower LR
    learning_rate=3e-4,      # Lower starting LR (was 1e-3)
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
    # generate_training_data=False #True##False
    if generate_training_data:
        print("\n[3/6] Generating training data...")

        generator = WishartDataGenerator(config, wishart_module_path=wishart_path)
    
        # Generate datasets
        n_train = 300#8000
        n_val = 50#1000
    
        print(f"Generating {n_train} training samples...")
        train_data = generator.generate_dataset(n_samples=n_train)
    
        print(f"Generating {n_val} validation samples...")
        generator.key = generator.key  # Continue with same generator
        val_data = generator.generate_dataset(n_samples=n_val)
    
        # Save datasets for reuse
        generator.save_dataset(train_data, output_dir / "train_data.npz")
        generator.save_dataset(val_data, output_dir / "val_data.npz")
    
    else:
        # Load existing datasets
        print("Loading existing datasets...")
        train_data = np.load(output_dir / "train_data.npz")
        val_data = np.load(output_dir / "val_data.npz")

    # =========================================================================
    # 4. TRAIN
    # =========================================================================
    print("\n[4/6] Training model...")
    
    save_path = output_dir / "models"
    # best_params, history = train_model(
    #     model=model,
    #     config=config,
    #     train_data=train_data,
    #     val_data=val_data,
    #     save_path=str(save_path),
    #     save_every=100
    # )

    # With LR schedule (default)
    best_params, history = train_model(model, config, train_data, val_data)

    # # Without LR schedule (old behavior)
    # best_params, history = train_model(model, config, train_data, val_data, use_lr_schedule=False)
    
    # Plot and save training history
    plot_training_history(history, save_path=str(output_dir / "training_history.png"))
    
    # =========================================================================
    # 5. INFERENCE
    # =========================================================================
    print("\n[5/6] Testing inference...")
    
    # Create inference object with trained parameters
    inference = WishartPINNInference(
        model, 
        best_params,
        wishart_module_path=wishart_path
    )
    
    # Example computation
    T = 1.0
    theta = np.array([[-1.0, 0.0], [0.0, -1.0]])
    m = np.array([[-0.5, 0.1], [0.1, -0.5]])
    omega = np.array([[0.3, 0.05], [0.05, 0.3]])
    sigma = np.array([[0.5, 0.0], [0.0, 0.5]])
    x0 = np.eye(2)
    
    print("\nExample computation:")
    print(f"  T = {T}")
    print(f"  theta = \n{theta}")
    
    A_nn, B_nn = inference.compute_A_B(T, theta, m, omega, sigma)
    phi_nn = inference.compute_characteristic_function(T, theta, m, omega, sigma, x0)
    
    print(f"\nNeural Network Results:")
    print(f"  A = \n{A_nn}")
    print(f"  B = {B_nn:.6f}")
    print(f"  Φ = {phi_nn:.8f}")
    
    # Compare with numerical
    comparison = inference.compare_with_numerical(T, theta, m, omega, sigma, x0)
    
    print(f"\nNumerical (Wishart.py) Results:")
    print(f"  A = \n{comparison['A_num']}")
    print(f"  B = {comparison['B_num']:.6f}")
    print(f"  Φ = {comparison['phi_num']:.8f}")
    
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
    throughput = benchmark_throughput(inference, n_samples=1000)
    
    # =========================================================================
    # DEMONSTRATE SAVE/LOAD
    # =========================================================================
    print("\n" + "-" * 50)
    print("Testing save/load functionality...")
    
    # Load from saved model
    loaded_inference = WishartPINNInference.from_saved_model(
        str(save_path / "final"),
        wishart_module_path=wishart_path
    )
    
    # Verify it works
    phi_loaded = loaded_inference.compute_characteristic_function(T, theta, m, omega, sigma, x0)
    print(f"Φ from loaded model: {phi_loaded:.8f}")
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

def load_and_compare_characteristic_function(wishart_path: str, output_dir: str = "./output"):
    """Run complete example workflow."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "models"

    loaded_inference = WishartPINNInference.from_saved_model(
        str(save_path / "final"),
        wishart_module_path=wishart_path
    )
    # Example computation
    d=2
    T_list=[0.5,1.0,2.0]
    theta_list = [ 
        np.array([[-1.0, 0.0], [0.0, -1.0]])
        ,np.array([[-1.5, 0.0], [0.0, -1.5]])
        ,np.array([[-2.0, 0.0], [0.0, -2.0]])
        ,np.array([[-0.2, 0.0], [0.0, -0.2]])
        ,np.array([[0.20, -0.0], [-0.0, 0.20]])
        ]
    # T = 1.0
    # theta = np.array([[-1.0, 0.0], [0.0, -1.0]])

    m = np.array([[-0.5, 0.1], [0.1, -0.5]])
    omega = np.array([[0.3, 0.05], [0.05, 0.3]])
    sigma = np.array([[0.5, 0.0], [0.0, 0.5]])
    x0 = np.eye(2)
    for T in T_list:
        for theta in theta_list:
             
            print("="*30)
            phi_loaded = loaded_inference.compute_characteristic_function(T, theta, m, omega, sigma, x0)
            print(f"T:{T}, theta:{theta}")
            print(f"Φ from loaded model: {phi_loaded:.8f}")
    


            wishart = WishartWithJump(d, x0, omega, m, sigma)
            wishart.maturity = T
    
            A = wishart.compute_a(T, theta) ##ComputeA(T, theta)
            B = wishart.compute_b(T, theta) ##ComputeB(T, theta)
            phi_code = wishart.phi_one(1,theta) ##ComputeB(T, theta)
            phi_code = np.real(phi_code)
    
            phi_loaded = loaded_inference.compute_characteristic_function(T, theta, m, omega, sigma, x0)
            A_nn, B_nn = loaded_inference.compute_A_B(T, theta, m, omega, sigma)#, x0)
    
            print(f"T:{T}")
            print(f"A:{A}")
            print(f"A_nn:{A_nn}")
            print(f"B:{B}")
            print(f"B_nn:{B_nn}")
            print(f"Φ from loaded model: {phi_loaded:.8f}")
            print(f"Φ computed with code: {phi_code:.8f}")
            print(f"Match: {np.isclose(phi_code, phi_loaded)}")


            

def generate_data(wishart_path: str
                  , first_file_id=0
                  , sub_dats_set_output_dir: str = "./output"
                  , file_patern = "chunk_"
                  , max_size_per_generation=10
                  , nb_generation=50):
    """Run complete example workflow."""
    
    output_dir = Path(sub_dats_set_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    
    config = WishartPINNConfig(
        dim=2,
        hidden_dim=128,
        num_highway_blocks=6,
        batch_size=256,
        num_epochs=500,      # Reduce for quick test
        learning_rate=1e-3,
        seed=42
    )
    
    print("\n[3/6] Generating training data...")

    generator = WishartDataGenerator(config, wishart_module_path=wishart_path)
    
    # Generate datasets
    n_train = max_size_per_generation
    nb_generation=nb_generation//max_size_per_generation

    if nb_generation>1:
        n_train = max_size_per_generation      
    else:
        n_train=nb_generation

    for i in range(nb_generation):
        print(f"Generation {i+1}/{nb_generation}")
        print(f"Generating {n_train} training samples...")
        train_data = generator.generate_dataset(n_samples=n_train)
        

        generator.key = generator.key  # Continue with same generator
  
        current_file_id = i + first_file_id
        # Save datasets for reuse
        curren_file = output_dir / f"{file_patern}_{current_file_id}.npz"
        generator.save_dataset(train_data, curren_file) 
        clear_jax_cache()


def load_merge_data(wishart_path
                    , output_dir: str = "./output"
                    , chunk_dir: str = "./sub_set_data"
                    , file_patern = "chunk_*.npz"
                    , file_name="train_data.npz"):
    
     
    merged_data=WishartDataGenerator.load_and_merge_chunks(chunk_dir=chunk_dir, pattern = file_patern)##"chunk_*.npz")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    
    config = WishartPINNConfig(
        dim=2,
        hidden_dim=128,
        num_highway_blocks=6,
        batch_size=256,
        num_epochs=500,      # Reduce for quick test
        learning_rate=1e-3,
        seed=42
    )

    generator = WishartDataGenerator(config, wishart_module_path=wishart_path)
   
    generator.save_dataset(merged_data, output_dir / file_name)
    


def clear_jax_cache():
    jax.clear_caches()
    gc.collect()
    
    jax_cache = os.path.join(os.environ.get('LOCALAPPDATA', ''), 'jax')
    if os.path.exists(jax_cache):
        shutil.rmtree(jax_cache, ignore_errors=True)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     description="Run Wishart Neural Operator example workflow"
    # )
    # parser.add_argument(
    #     "--wishart_path",
    #     type=str,
    #     default="/mnt/project",
    #     help="Path to directory containing Wishart.py"
    # )
    # parser.add_argument(
    #     "--output_dir",
    #     type=str,
    #     default="./output",
    #     help="Directory to save outputs"
    # )
    
    # args = parser.parse_args()
    
    # main(args.wishart_path, args.output_dir)
    main_folder= r"E:\OneDrive\Dropbox\LinearRationalWishart_Work\Code\ED\LinearRationalWishart\LinearRationalWishart_NewCode\linear_rational_wishart"
    # ouput_folder= r"E:\OneDrive\Dropbox\LinearRationalWishart_Work\Code\ED\LinearRationalWishart\LinearRationalWishart_NewCode\neural_operator\output"
    ouput_folder=r"E:\OneDrive\Dropbox\LinearRationalWishart_Work\Code\ED\LinearRationalWishart\LinearRationalWishart_NewCode\linear_rational_wishart\neural_operator\saved_models"
    ouput_folder=r"E:\OneDrive\Dropbox\LinearRationalWishart_Work\Code\ED\LinearRationalWishart\LinearRationalWishart_NewCode\Output_results\neural_operator\saved_models"
    main_ouput_folder=r"E:\OneDrive\Dropbox\LinearRationalWishart_Work\Code\ED\LinearRationalWishart\LinearRationalWishart_NewCode\Output_results\neural_operator\saved_models"
    
    clear_jax_cache()

    run_train_model=False#True#False
    if run_train_model:
        generate_training_data=False
        main(main_folder, ouput_folder,generate_training_data)

    test_model=False#True#False#True
    if test_model:
        load_and_compare_characteristic_function(main_folder, ouput_folder)

    run_data_generation=True#True
    if run_data_generation:
        train_data_sets_folder       =  main_ouput_folder +r"\training_data"
        validation_data_sets_folder  =  main_ouput_folder +r"\validation_data"
        
        # generate_data(main_folder, ouput_folder)
        generate_data(main_folder
                  , first_file_id=0#67#7#0
                  , sub_dats_set_output_dir=train_data_sets_folder
                  , file_patern = "chunk"
                  , max_size_per_generation=10#0
                  , nb_generation=20)#10000)

         # generate_data(main_folder, ouput_folder)
        generate_data(main_folder
                  , first_file_id=0#12
                  , sub_dats_set_output_dir=validation_data_sets_folder
                  , file_patern = "chunk"
                  , max_size_per_generation=10#0
                  , nb_generation=20)#00)

    merge_data_set=True#False#True#False
    if merge_data_set:
        print("Merging data sets...")
        
        train_data_sets_folder       =  main_ouput_folder +r"\training_data"
        load_merge_data(main_folder
                    , output_dir=main_ouput_folder
                    , chunk_dir=train_data_sets_folder
                    , file_patern = "chunk_*.npz"
                    , file_name="train_data.npz") 
        
        validation_data_sets_folder  =  main_ouput_folder +r"\validation_data"
        load_merge_data(main_folder
                    , output_dir=main_ouput_folder
                    , chunk_dir=validation_data_sets_folder
                    , file_patern = "chunk_*.npz"
                    , file_name="val_data.npz") 