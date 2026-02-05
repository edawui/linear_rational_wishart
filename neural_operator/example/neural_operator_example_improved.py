#!/usr/bin/env python
"""
IMPROVED Example: Complete Workflow for Wishart Neural Operator
===============================================================

This is a modified version of your neural_operator_example.py that integrates
the improvements while keeping your existing infrastructure.

CHANGES FROM ORIGINAL:
1. ImprovedWishartPINNConfig - extended config with new parameters
2. Regime-weighted loss function
3. Curriculum learning
4. Importance sampling for data generation
5. Longer training with better schedule

HOW TO USE:
1. Copy this file to your project
2. Your existing data files (train_data.npz, val_data.npz) are still compatible
3. Run with: python neural_operator_example_improved.py

Author: Modified from Da Fonseca, Dawui, Malevergne
"""

import argparse
import numpy as np
from pathlib import Path
import os
import shutil
import gc
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple, List, Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import random, jit, vmap, value_and_grad, lax
import optax
from tqdm import tqdm
import matplotlib.pyplot as plt
from pprint import pprint

from functools import partial
# import jax

# from jax import jit, value_and_grad
# import jax.numpy as jnp


import optax
# =============================================================================
# YOUR EXISTING IMPORTS (keep these)
# =============================================================================
# Uncomment these when using in your project:
# from linear_rational_wishart.core.wishart_jump import WishartWithJump
# from linear_rational_wishart.neural_operator.model import (
#     WishartPINNModel, WishartCharFuncNetwork, matrix_to_upper_tri,
#     upper_tri_to_matrix, complex_upper_tri_to_matrix
# )
# from linear_rational_wishart.neural_operator.data_generation import (
#     WishartCharFuncComputer, WishartDataGenerator, complex_matrix_to_upper_tri
# )
# from linear_rational_wishart.neural_operator.training import plot_training_history
# from linear_rational_wishart.neural_operator.inference import WishartPINNInference


# =============================================================================
# IMPROVED CONFIG (replaces WishartPINNConfig)
# =============================================================================

MODE="complex"
@dataclass
class ImprovedWishartPINNConfig:
    """
    Extended configuration with improvement parameters.
    
    KEEP all your original parameters, ADD the new ones.
    """
    # === ORIGINAL PARAMETERS (keep these) ===
    dim: int = 2
    hidden_dim: int = 256      # INCREASED from 128
    num_highway_blocks: int = 8  # INCREASED from 6
    batch_size: int = 512
    num_epochs: int = 10000    # INCREASED from 3000-5000
    learning_rate: float = 3e-4
    physics_loss_weight: float = 0.0  # Keep 0 for now
    
    # Parameter ranges (keep your existing values)
    T_min: float = 0.5
    T_max: float = 5.0
    theta_min: float = 0.01
    theta_max: float = 10.0
    theta_offdiag_min: float = -0.9
    theta_offdiag_max: float = 0.9
    ui_min: float = 0.0
    ui_max: float = 25.0
    ur: float = 0.5
    m_diag_min: float = -1.0
    m_diag_max: float = -0.1
    m_offdiag_min: float = 0.0
    m_offdiag_max: float = 0.0
    omega_diag_min: float = 0.0001
    omega_diag_max: float = 0.5
    omega_offdiag_min: float = -0.9
    omega_offdiag_max: float = 0.9
    sigma_diag_min: float = 0.001
    sigma_diag_max: float = 0.5
    sigma_offdiag_min: float = -0.9
    sigma_offdiag_max: float = 0.9
    seed: int = 42
    
    # === NEW PARAMETERS (add these) ===
    # Regime weighting
    low_ui_weight: float = 4.0    # Weight for ui < 5 (most important!)
    mid_ui_weight: float = 2.0    # Weight for 5 <= ui < 15
    high_ui_weight: float = 1.0   # Weight for ui >= 15
    
    # Curriculum learning
    use_curriculum: bool = True
    curriculum_warmup_fraction: float = 0.3  # 30% of epochs for curriculum
    
    # Training improvements
    warmup_epochs: int = 200      # LR warmup
    min_lr_factor: float = 0.001  # Final LR = learning_rate * min_lr_factor
    weight_decay: float = 1e-5    # AdamW weight decay
    
    @property
    def n_upper(self) -> int:
        return self.dim * (self.dim + 1) // 2
    
    @property
    def input_dim(self) -> int:
        d = self.dim
        n_up = self.n_upper
        return 1 + 2*n_up + d*d + n_up + n_up
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'ImprovedWishartPINNConfig':
        # Filter out unknown keys for backward compatibility
        known_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known_keys}
        return cls(**filtered)


# =============================================================================
# TRAINING STATE
# =============================================================================

class TrainState(NamedTuple):
    """Immutable training state."""
    params: Dict
    opt_state: Any
    step: int


# =============================================================================
# IMPROVED LOSS FUNCTION (this is the key change!)
# =============================================================================

def compute_regime_weighted_loss(
    network_fn,
    params: Dict,
    batch: Dict[str, jnp.ndarray],
    config: ImprovedWishartPINNConfig
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Compute loss with different weights for different u_i regimes.
    
    This is the KEY IMPROVEMENT - it focuses training on the important
    low u_i region that matters most for pricing.
    """
    n_upper = config.n_upper
    
    # Prepare inputs (same as original)
    T = batch['T'][:, None] if batch['T'].ndim == 1 else batch['T']
    inputs = jnp.concatenate([
        T, batch['theta'], batch['m'], batch['omega'], batch['sigma']
    ], axis=-1)
    
    # Forward pass
    A_pred, B_pred = network_fn(params, inputs)
    
    # === NEW: Extract u_i and compute weights ===
    # theta is stored as [real_upper, imag_upper]
    # imag_upper contains the u_i information
    theta_imag = batch['theta'][:, n_upper:2*n_upper]
    ui_magnitude = jnp.mean(jnp.abs(theta_imag), axis=-1)
    
    # Compute weights based on u_i regime
    low_mask = ui_magnitude < 5.0
    mid_mask = (ui_magnitude >= 5.0) & (ui_magnitude < 15.0)
    high_mask = ui_magnitude >= 15.0
    
    weights = (
        low_mask.astype(jnp.float32) * config.low_ui_weight +
        mid_mask.astype(jnp.float32) * config.mid_ui_weight +
        high_mask.astype(jnp.float32) * config.high_ui_weight
    )
    
    # Weighted MSE loss
    sq_error_A = (A_pred - batch['A']) ** 2
    sq_error_B = (B_pred - batch['B']) ** 2
    
    # Apply weights (broadcast weights to match dimensions)
    loss_A = jnp.mean(weights[:, None] * sq_error_A)
    loss_B = jnp.mean(weights[:, None] * sq_error_B)
    
    total_loss = loss_A + loss_B
    
    return total_loss, {
        'loss_A': loss_A,
        'loss_B': loss_B,
        'mean_weight': jnp.mean(weights),
        'frac_low_ui': jnp.mean(low_mask.astype(jnp.float32))
    }


def compute_standard_loss(
    network_fn,
    params: Dict,
    batch: Dict[str, jnp.ndarray]
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Standard MSE loss (for validation)."""
    T = batch['T'][:, None] if batch['T'].ndim == 1 else batch['T']
    inputs = jnp.concatenate([
        T, batch['theta'], batch['m'], batch['omega'], batch['sigma']
    ], axis=-1)
    
    A_pred, B_pred = network_fn(params, inputs)
    
    loss_A = jnp.mean((A_pred - batch['A']) ** 2)
    loss_B = jnp.mean((B_pred - batch['B']) ** 2)
    
    return loss_A + loss_B, {'loss_A': loss_A, 'loss_B': loss_B}


def compute_regime_weighted_loss_new(
    network_fn,
    params: Dict,
    batch: Dict[str, jnp.ndarray],
    config: ImprovedWishartPINNConfig
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    
    n_upper = config.n_upper
    
    # 1. Ensure all inputs are at least 2nd order (batch_size, dim)
    # Using jnp.atleast_2d or explicit slicing [:, None]
    T = batch['T']
    if T.ndim == 1: T = T[:, None]
    
    # Ensure params from batch are shaped correctly for concatenation
    inputs = jnp.concatenate([
        T, batch['theta'], batch['m'], batch['omega'], batch['sigma']
    ], axis=-1)
    
    # 2. Forward pass
    A_pred, B_pred = network_fn(params, inputs)
    
    # 3. Extract u_i and compute weights efficiently
    # theta_imag shape is (batch, n_upper)
    theta_imag = batch['theta'][:, n_upper:2*n_upper]
    ui_magnitude = jnp.mean(jnp.abs(theta_imag), axis=-1)
    
    # Cleaner way to assign weights using nested where or select
    weights = jnp.where(
        ui_magnitude < 5.0, 
        config.low_ui_weight,
        jnp.where(ui_magnitude < 15.0, config.mid_ui_weight, config.high_ui_weight)
    )
    
    # 4. Compute Squared Errors
    # Ensure targets from batch match the shape of predictions
    sq_error_A = jnp.square(A_pred - batch['A'])
    sq_error_B = jnp.square(B_pred - batch['B'])
    
    # 5. Apply weights
    # weights is (batch,), sq_error is (batch, dim). 
    # Must use [:, None] to align the batch dimension.
    weighted_sq_error_A = weights[:, None] * sq_error_A
    weighted_sq_error_B = weights[:, None] * sq_error_B
    
    # Use jnp.mean over both batch and feature dimensions
    loss_A = jnp.mean(weighted_sq_error_A)
    loss_B = jnp.mean(weighted_sq_error_B)
    
    total_loss = loss_A + loss_B
    
    # Metrics for debugging
    return total_loss, {
        'loss_A': loss_A,
        'loss_B': loss_B,
        'mean_weight': jnp.mean(weights),
        'frac_low_ui': jnp.mean(ui_magnitude < 5.0)
    }

# =============================================================================
# CURRICULUM LEARNING
# =============================================================================

class CurriculumScheduler:
    """
    Curriculum learning: start with easy samples, gradually include harder ones.
    
    Easy = low u_i (smoother characteristic function)
    Hard = high u_i (more oscillatory)
    """
    
    def __init__(self, total_epochs: int, warmup_fraction: float = 0.3):
        self.total_epochs = total_epochs
        self.warmup_epochs = int(total_epochs * warmup_fraction)
    
    def get_ui_max(self, epoch: int, final_ui_max: float = 25.0) -> float:
        """Get maximum u_i to include at this epoch."""
        if epoch < self.warmup_epochs:
            # Linear ramp from 5 to final_ui_max
            progress = epoch / self.warmup_epochs
            return 5.0 + (final_ui_max - 5.0) * progress
        return final_ui_max
    
    def filter_batch(
        self,
        batch: Dict[str, jnp.ndarray],
        epoch: int,
        n_upper: int
    ) -> Dict[str, jnp.ndarray]:
        """Filter batch to only include samples within current curriculum."""
        ui_max = self.get_ui_max(epoch)
        
        # Extract u_i magnitude
        theta_imag = batch['theta'][:, n_upper:2*n_upper]
        ui_magnitude = jnp.mean(jnp.abs(theta_imag), axis=-1)
        
        # Create mask and filter
        mask = ui_magnitude <= ui_max
        
        # Convert to numpy for boolean indexing, then back to jax
        mask_np = np.array(mask)
        filtered = {k: jnp.array(np.array(v)[mask_np]) for k, v in batch.items()}
        
        return filtered


# =============================================================================
# IMPROVED TRAINING FUNCTION
# =============================================================================

def train_model_improved(
    model,  # Your WishartPINNModel
    config: ImprovedWishartPINNConfig,
    train_data: Dict[str, jnp.ndarray],
    val_data: Optional[Dict[str, jnp.ndarray]] = None,
    save_path: Optional[str] = None,
    save_every: int = 500
) -> Tuple[Dict, Dict[str, List[float]]]:
    """
    Improved training loop with regime weighting and curriculum learning.
    
    This replaces your original train_model function.
    """
    n_samples = train_data['T'].shape[0]
    n_batches = max(1, n_samples // config.batch_size)
    total_steps = config.num_epochs * n_batches
    
    # Print data info
    print(f"Data shapes:")
    for key in train_data:
        print(f"  {key}: {train_data[key].shape}")
    
    # === IMPROVED OPTIMIZER ===
    warmup_steps = config.warmup_epochs * n_batches
    
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=config.learning_rate * 0.01,  # Start very low
        peak_value=config.learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps - warmup_steps,
        end_value=config.learning_rate * config.min_lr_factor
    )
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=config.weight_decay)
    )
    
    # Initialize state
    state = TrainState(
        params=model.params,
        opt_state=optimizer.init(model.params),
        step=0
    )
    
    # Curriculum scheduler
    curriculum = None
    if config.use_curriculum:
        curriculum = CurriculumScheduler(
            config.num_epochs,
            config.curriculum_warmup_fraction
        )
    
    # Create JIT-compiled train step
    @jit
    def train_step(state: TrainState, batch: Dict) -> Tuple[TrainState, jnp.ndarray, Dict]:
        def loss_fn(params):
            return compute_regime_weighted_loss(
                model.network.apply, params, batch, config
            )
        
        (loss, aux), grads = value_and_grad(loss_fn, has_aux=True)(state.params)
        updates, new_opt_state = optimizer.update(grads, state.opt_state, state.params)
        new_params = optax.apply_updates(state.params, updates)
        
        return TrainState(new_params, new_opt_state, state.step + 1), loss, aux
    
    # History
    history = {
        'loss': [], 'loss_A': [], 'loss_B': [],
        'val_loss': [], 'learning_rate': [], 'curriculum_ui_max': []
    }
    
    print(f"\n{'='*60}")
    print(f"IMPROVED TRAINING")
    print(f"{'='*60}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Samples: {n_samples}, Batches/epoch: {n_batches}")
    print(f"Regime weights: low={config.low_ui_weight}, mid={config.mid_ui_weight}, high={config.high_ui_weight}")
    print(f"Curriculum learning: {config.use_curriculum}")
    print(f"LR: {config.learning_rate} -> {config.learning_rate * config.min_lr_factor}")
    print(f"{'='*60}\n")
    
    best_loss = float('inf')
    best_params = None

    key = random.PRNGKey(0)
    for epoch in tqdm(range(config.num_epochs), desc="Training"):
        # Shuffle data
        # key = random.PRNGKey(epoch)
        # perm = random.permutation(key, n_samples)
        
        key, subkey = random.split(key)
        perm = random.permutation(subkey, n_samples)
        # Track curriculum
        if curriculum is not None:
            ui_max = curriculum.get_ui_max(epoch, config.ui_max)
            history['curriculum_ui_max'].append(ui_max)
        
        epoch_loss = 0.0
        epoch_loss_A = 0.0
        epoch_loss_B = 0.0

       
        epoch_loss_init   = jnp.array(0.0)
        epoch_loss_A_init = jnp.array(0.0)
        epoch_loss_B_init = jnp.array(0.0)

        batches_processed = 0
        
        for batch_idx in range(n_batches):
            # Get batch
            start_idx = batch_idx * config.batch_size
            end_idx = min(start_idx + config.batch_size, n_samples)
            idx = perm[start_idx:end_idx]
            # batch = {k: v[idx] for k, v in train_data.items()}
            batch = {   k: v[idx]
                        for k, v in train_data.items()
                        if isinstance(v, (np.ndarray, jnp.ndarray)) and v.ndim > 0
                    }
            
            # Apply curriculum filtering
            if curriculum is not None:
                batch = curriculum.filter_batch(batch, epoch, config.n_upper)
                
                # Skip if batch is too small after filtering
                if len(batch['T']) < 16:
                    continue
            
            # Train step
            state, loss, aux = train_step(state, batch)
            
            # epoch_loss += float(loss)
            # epoch_loss_A += float(aux['loss_A'])
            # epoch_loss_B += float(aux['loss_B'])

            epoch_loss_init += loss
            epoch_loss_A_init += aux['loss_A']
            epoch_loss_B_init += aux['loss_B']

            batches_processed += 1

        epoch_loss, epoch_loss_A, epoch_loss_B = jax.device_get(
            (epoch_loss_init, epoch_loss_A_init, epoch_loss_B_init))
    

        # Record history
        if batches_processed > 0:
            history['loss'].append(epoch_loss / batches_processed)
            history['loss_A'].append(epoch_loss_A / batches_processed)
            history['loss_B'].append(epoch_loss_B / batches_processed)
        
        # Learning rate tracking
        current_step = epoch * n_batches
        if current_step < warmup_steps:
            current_lr = config.learning_rate * 0.01 + (config.learning_rate * 0.99) * current_step / warmup_steps
        else:
            progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
            current_lr = config.learning_rate * config.min_lr_factor + \
                        0.5 * (config.learning_rate - config.learning_rate * config.min_lr_factor) * \
                        (1 + np.cos(np.pi * progress))
        history['learning_rate'].append(float(current_lr))
        
        # Validation
        if val_data is not None:
            val_loss, _ = compute_standard_loss(model.network.apply, state.params, val_data)
            history['val_loss'].append(float(val_loss))
            track_loss = float(val_loss)
        else:
            track_loss = history['loss'][-1] if history['loss'] else float('inf')
        
        # Track best
        if track_loss < best_loss:
            best_loss = track_loss
            best_params = state.params
        
        # Logging
        if (epoch + 1) % 500 == 0:
            msg = f"Epoch {epoch+1}: Loss={history['loss'][-1]:.6f}"
            if val_data is not None:
                msg += f", Val={history['val_loss'][-1]:.6f}"
            if curriculum is not None:
                msg += f", ui_max={history['curriculum_ui_max'][-1]:.1f}"
            msg += f", LR={current_lr:.2e}"
            tqdm.write(msg)
        
        # Save checkpoint
        if save_path is not None and (epoch + 1) % save_every == 0:
            checkpoint_path = Path(save_path) / f"checkpoint_{epoch+1}"
            model.save(str(checkpoint_path), state.params)
    
    # Save final
    if save_path is not None:
        final_path = Path(save_path) / "final"
        model.save(str(final_path), best_params)
    
    print(f"\n{'='*60}")
    print(f"Training complete. Best loss: {best_loss:.6f}")
    print(f"{'='*60}")
    
    return best_params, history


def train_model_optimized(
    model, 
    config: ImprovedWishartPINNConfig,
    train_data: Dict[str, np.ndarray],
    val_data: Optional[Dict[str, np.ndarray]] = None,
    save_path: Optional[str] = None,
    save_every: int = 500
) -> Tuple[Dict, Dict]:

    # --- 1. DATA PREPARATION ---
    # Move all data to the device (GPU/TPU) as JAX arrays immediately.
    # This prevents the TracerArrayConversionError.
    # train_data_jax = {
    #     k: jnp.array(v) for k, v in train_data.items() 
    #     if isinstance(v, (np.ndarray, jnp.ndarray))
    # }
    # --- 1. DATA PREPARATION ---
    # Only convert values that are actually numeric arrays. 
    # We filter out strings or other metadata that JAX cannot process.
    def process_data(data):
        jax_dict = {}
        for k, v in data.items():
            if isinstance(v, (np.ndarray, jnp.ndarray)) and jnp.issubdtype(v.dtype, jnp.number):
                jax_dict[k] = jnp.array(v)
            elif isinstance(v, (float, int, complex)):
                jax_dict[k] = jnp.array(v)
        return jax_dict

    
    train_data_jax = process_data(train_data)
    n_samples = train_data_jax['T'].shape[0]
    n_batches = n_samples // config.batch_size
    total_steps = config.num_epochs * n_batches
  

    val_data_jax = process_data(val_data) if val_data is not None else {}
    # Use .get() to avoid KeyError if val_data is empty/None
    val_samples = val_data_jax['T'].shape[0] if 'T' in val_data_jax else 0
    valid_n_batches = max(1, val_samples // config.batch_size) if val_samples > 0 else 0

    # --- 2. OPTIMIZER SETUP ---
    #This is common in modern deep learning to stabilize training initially and ensure smooth convergence later.
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=config.learning_rate * 0.01,
        peak_value=config.learning_rate,
        warmup_steps=config.warmup_epochs * n_batches,
        decay_steps=total_steps,
        end_value=config.learning_rate * config.min_lr_factor
    )
    

    #This configuration is a standard and robust setup for modern deep learning models.
    # It chains two critical operations: gradient clipping to prevent exploding gradients
    #  and AdamW for decoupled weight decay with a scheduled learning rate
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=config.weight_decay)
    )

    # --- 3. JIT-COMPILED CORE LOGIC ---
    
    @jit  # Compiles the function for high-performance execution on GPU/TPU
    def train_step(state, batch_data):
        # 1. Define a pure function that returns the scalar loss to differentiate
        def loss_fn(params):
            # compute_regime_weighted_loss should use model.network.apply
            return compute_regime_weighted_loss(
                model.network.apply, params, batch_data, config
            )
        
        # 2. Compute gradients of loss_fn with respect to params
        # has_aux=True allows returning extra data (like loss_A, loss_B) without differentiating it   
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        
        # 3. Transform gradients into updates using the Optax optimizer (AdamW + Clipping)
        # This manages optimizer states like momentum and the learning rate schedule
        updates, new_opt_state = optimizer.update(grads, state.opt_state, state.params)
        # 4. Apply updates to parameters: new_params = params + updates
        new_params = optax.apply_updates(state.params, updates)
        
        # 5. Create a new TrainState with updated params, optimizer state, and step count   
        new_state = state._replace(
            params=new_params,
            opt_state=new_opt_state,
            step=state.step + 1
        )
        # Return the new state and a tuple of losses for logging
        return new_state, (loss, aux['loss_A'], aux['loss_B'])

    @jit  # Compiles the function for high-performance execution on GPU/TPU
    def val_step(params, batch_data):
        loss, aux= compute_regime_weighted_loss(
                model.network.apply, params, batch_data, config
            )
        
        return loss, aux['loss_A'], aux['loss_B']

    @partial(jit, static_argnums=(1,)) 
    def val_epoch(params,valid_n_batches, data_dict):

        if valid_n_batches == 0: 
            return 0.0, 0.0, 0.0
        # Assuming valid_n_batches is pre-calculated
        # No permutation needed for validation; indices are sequential
        batch_indices = jnp.arange(valid_n_batches * config.batch_size).reshape((valid_n_batches, config.batch_size))
  
        def scan_body(unused_carry, idx_batch):
            # This slicing works now because data_dict contains JAX arrays
            # Slice dictionary arrays to create the current mini-batch
            batch = {k: v[idx_batch] for k, v in data_dict.items()}
            l_total, l_a, l_b = val_step(params, batch)
            return None, (l_total, l_a, l_b)
            
        _, metrics = lax.scan(scan_body, None, batch_indices)
        return jax.tree_util.tree_map(jnp.mean, metrics)
       

    @partial(jit, static_argnums=(1,)) # static_argnums=1 corresponds to n_batches    
    def train_epoch(state,n_batches, permutation, data_dict):
        # Slice permutation to fit full batches
        # 1. Reshape indices into [num_batches, batch_size] to enable vectorized slicing  
        batch_indices = permutation[:n_batches * config.batch_size].reshape((n_batches, config.batch_size))
        
        def scan_body(carry_state, idx_batch):
            # This slicing works now because data_dict contains JAX arrays
            # Slice dictionary arrays to create the current mini-batch
            batch = {k: v[idx_batch] for k, v in data_dict.items()}
            # Perform the training step; returns (updated_state, metrics)
            return train_step(carry_state, batch)
        
        # Use lax.scan to run the entire epoch on the device
        # 3. Efficiently iterate over all batches on the hardware accelerator
        # final_state: the state after the last batch; losses: arrays of metrics for every step
        # final_state, (losses, losses_A, losses_B) = lax.scan(scan_body, state, batch_indices)
        final_state, metrics = lax.scan(scan_body, state, batch_indices)
        
        # Return averaged metrics for the epoch
        # 4. Aggregate step-wise losses into a single scalar average for the epoch
        # return final_state, (jnp.mean(losses), jnp.mean(losses_A), jnp.mean(losses_B))
        return final_state,jax.tree_util.tree_map(jnp.mean, metrics)

    # --- 4. INITIALIZATION ---
    state = TrainState(
        params=model.params,
        # optimizer.init creates the necessary buffers (momentum, etc.) based on the params
        opt_state=optimizer.init(model.params),
        # Tracks the global training step for the learning rate schedule
        step=0
    )

    # history = {'loss': [], 'loss_A': [], 'loss_B': []}
    history = {
        'loss': [], 'loss_A': [], 'loss_B': [], 
        'val_loss': [], 'val_loss_A': [], 'val_loss_B': [],
        'learning_rate': [], 'curriculum_ui_max': []
        }
    key = random.PRNGKey(0)

    # --- 5. MAIN LOOP ---
    print(f"Starting optimized training: {config.num_epochs} epochs, {n_batches} batches/epoch.")
    
    v_loss = float('nan') # Placeholder

    for epoch in range(config.num_epochs):
        key, subkey = random.split(key)
        perm = random.permutation(subkey, n_samples)
        
        # Pass the state, the permutation, AND the data dictionary explicitly
        state, (m_loss, m_loss_A, m_loss_B) = train_epoch(state, n_batches, perm, train_data_jax)

        # --- New Validation Check ---
        v_log_str = ""
        if val_data is not None:
            v_loss, v_loss_A, v_loss_B = val_epoch(state.params, valid_n_batches, val_data_jax)
            history['val_loss'].append(float(v_loss))
            history['val_loss_A'].append(float(v_loss_A))
            history['val_loss_B'].append(float(v_loss_B))
            v_log_str = f" | Val: {v_loss:.6f}" 
            
        # Record history (casting to float is fine here once per epoch)
        # history['val_loss'].append(float(v_loss))
        history['loss'].append(float(m_loss))
        history['loss_A'].append(float(m_loss_A))
        history['loss_B'].append(float(m_loss_B))

        current_lr = schedule(state.step)
        history['learning_rate'].append(float(current_lr))
        # Logging
        if (epoch + 1) % 100 == 0 or epoch == 0:
        # if (epoch + 1) % 10 == 0 or epoch == 0:
            print("\n\n")
            print(f"Epoch {epoch+1:5d} | Loss: {m_loss:.6f}  | A: {m_loss_A:.6f}  | B: {m_loss_B:.6f}")
            print(f"            | vLoss: {v_loss:.6f} | vA: {v_loss_A:.6f} | vB: {v_loss_B:.6f}")
            
            # Simple Overfitting Warning
            if val_data is not None and v_loss > m_loss * 1.2: 
               print("⚠️ Warning: Potential Overfitting (Val Loss > 1.2x Train Loss)")

        # Checkpoints
        if save_path is not None and (epoch + 1) % save_every == 0:
            checkpoint_path = Path(save_path) / f"checkpoint_{epoch+1}"
            model.save(str(checkpoint_path), state.params)
    
    # Final Save
    if save_path is not None:
        final_path = Path(save_path) / "final"
        model.save(str(final_path), state.params)

    return state.params, history
# =============================================================================
# IMPROVED PLOTTING
# =============================================================================

def plot_training_history_improved(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None
):
    """Plot training history with curriculum info."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss curves
    ax = axes[0, 0]
    epochs = range(len(history['loss']))
    ax.semilogy(history['loss'], alpha=0.5, color='blue', label='Train')
    # if history['val_loss']:
    if 'val_loss' in history and history['val_loss']:
        ax.semilogy(history['val_loss'], alpha=0.5, color='orange', label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Component losses
    ax = axes[0, 1]
    ax.semilogy(history['loss_A'], alpha=0.7, label='A (Train)', color='tab:blue')
    ax.semilogy(history['loss_B'], alpha=0.7, label='B (Train)', color='tab:red')
    # Optional: Add validation components if they exist
    if 'val_loss_A' in history and history['val_loss_A']:
        ax.semilogy(history['val_loss_A'], linestyle='--', alpha=0.5, label='A (Val)', color='tab:blue')
    if 'val_loss_B' in history and history['val_loss_B']:
        ax.semilogy(history['val_loss_B'], linestyle='--', alpha=0.5, label='B (Val)', color='tab:red')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Component Losses')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Learning rate
    ax = axes[1, 0]
    if history.get('learning_rate'):
        ax.plot(history['learning_rate'], color='green')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No learning rate data', ha='center', va='center', fontsize=14)
    
    # Curriculum
    ax = axes[1, 1]
    if history.get('curriculum_ui_max'):
        ax.plot(history['curriculum_ui_max'], color='purple')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Max u_i')
        ax.set_title('Curriculum Schedule')
        ax.axhline(y=5, color='r', linestyle='--', alpha=0.5, label='Critical region')
        ax.axhline(y=15, color='orange', linestyle='--', alpha=0.5, label='Mid region')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No curriculum', ha='center', va='center', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


# =============================================================================
# MAIN FUNCTION (modified from your original)
# =============================================================================

def main_improved(
    wishart_path: str,
    output_dir: str = "./output",
    generate_training_data: bool = False,
    use_existing_data: bool = True
):
    """
    Main function with improvements.
    
    This is a drop-in replacement for your main() function.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("WISHART NEURAL OPERATOR - IMPROVED Workflow")
    print("=" * 70)
    
    # =========================================================================
    # 1. IMPROVED CONFIG
    # =========================================================================
    print("\n[1/6] Configuring model with IMPROVEMENTS...")
    
    config = ImprovedWishartPINNConfig(
        dim=2,
        hidden_dim=128,#256,           # Larger
        num_highway_blocks=6,#8,     # Deeper
        batch_size=512,
        num_epochs= 10000,#5000,#10000,         # Longer training
        learning_rate=3e-4,
        # NEW parameters:
        low_ui_weight=4.0,        # Focus on low u_i
        mid_ui_weight=2.0,
        high_ui_weight=1.0,
        use_curriculum=True,      # Enable curriculum
        curriculum_warmup_fraction=0.3,
        warmup_epochs=200,
        min_lr_factor=0.001,
        seed=42
    )
    print(f"Config: hidden_dim={config.hidden_dim}, blocks={config.num_highway_blocks}")
    print(f"Regime weights: low={config.low_ui_weight}, mid={config.mid_ui_weight}, high={config.high_ui_weight}")
    print(f"Curriculum: {config.use_curriculum}, warmup_fraction={config.curriculum_warmup_fraction}")
    
    # =========================================================================
    # 2. CREATE MODEL (use your existing model)
    # =========================================================================
    print("\n[2/6] Creating model...")
    
    # Import your existing model class
    # from linear_rational_wishart.neural_operator.model import WishartPINNModel
    # from linear_rational_wishart.neural_operator.config import WishartPINNConfig
    
    # For now, create a placeholder - replace with your actual import
    from linear_rational_wishart.neural_operator.config import WishartPINNConfig
    from linear_rational_wishart.neural_operator.model import WishartPINNModel
    
    # Convert improved config to standard config for model creation
    standard_config = WishartPINNConfig(
        dim=config.dim,
        hidden_dim=config.hidden_dim,
        num_highway_blocks=config.num_highway_blocks,
        batch_size=config.batch_size,
        num_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        seed=config.seed
    )
    
    model = WishartPINNModel(standard_config)
    print(model.summary())
    
    print(" Model to be trained Shape:")

    # pprint(jax.tree_util.tree_map(lambda x: x.shape, model.params))
    # model.count_params()

    # =========================================================================
    # 3. LOAD DATA (your existing data works!)
    # =========================================================================
    print("\n[3/6] Loading and normalizing training data...")
    
    if generate_training_data:
        # Your existing data generation code
        from linear_rational_wishart.neural_operator.data_generation import WishartDataGenerator
        
        generator = WishartDataGenerator(standard_config, wishart_module_path=wishart_path)
        
        # IMPROVED: Generate more low u_i samples
        print("Generating training data with importance sampling...")
        
        # Generate 50% with low u_i
        n_train_total = 10000
        n_low_ui = n_train_total // 2
        n_regular = n_train_total - n_low_ui
        
        # Low u_i samples
        original_ui_max = standard_config.ui_max
        standard_config.ui_max = 5.0  # Temporarily limit
        train_data_low = generator.generate_dataset(n_samples=n_low_ui)
        
        # Regular samples
        standard_config.ui_max = original_ui_max
        train_data_regular = generator.generate_dataset(n_samples=n_regular)
        
        # Merge
        train_data = {
            k: jnp.concatenate([train_data_low[k], train_data_regular[k]], axis=0)
            for k in train_data_low.keys()
        }
        
        # Validation data (regular distribution)
        val_data = generator.generate_dataset(n_samples=2000)
        
        generator.save_dataset(train_data, output_dir / f"train_data_{MODE}.npz")
        generator.save_dataset(val_data, output_dir / f"val_data_{MODE}.npz")
        
        train_data = {k: np.array(v) for k, v in train_data.items()}
        val_data = {k: np.array(v) for k, v in val_data.items()}
    else:
        # Load your existing data
        print("Loading existing datasets...")
        train_data_raw = np.load(output_dir / f"train_data_{MODE}.npz")
        val_data_raw = np.load(output_dir / f"val_data_{MODE}.npz")
        
        train_data = {key: train_data_raw[key].copy() for key in train_data_raw.files}
        val_data = {key: val_data_raw[key].copy() for key in val_data_raw.files}
    print(type(train_data))
    # Normalization (same as your original)
    print("\nComputing normalization statistics...")
    A_mean = np.mean(train_data['A'], axis=0)
    A_std = np.std(train_data['A'], axis=0) + 1e-8
    B_mean = np.mean(train_data['B'], axis=0)
    B_std = np.std(train_data['B'], axis=0) + 1e-8
    
    np.savez(output_dir / "normalization_stats.npz",
             A_mean=A_mean, A_std=A_std,
             B_mean=B_mean, B_std=B_std)
    
    train_data['A'] = (train_data['A'] - A_mean) / A_std
    train_data['B'] = (train_data['B'] - B_mean) / B_std
    val_data['A'] = (val_data['A'] - A_mean) / A_std
    val_data['B'] = (val_data['B'] - B_mean) / B_std
    
    print(f"After normalization: A mean={np.mean(train_data['A']):.4f}, std={np.std(train_data['A']):.4f}")
    
    # =========================================================================
    # 4. IMPROVED TRAINING
    # =========================================================================
    print("\n[4/6] Training model with IMPROVEMENTS...")
    
    save_path = output_dir / f"models_{MODE}"
    
    # USE THE IMPROVED TRAINING FUNCTION
    # best_params, history = train_model_improved( 
    best_params, history = train_model_optimized( 
        model, config, train_data, val_data,
        save_path=str(save_path),
        save_every=500##1000
    )
    
    # print("Trained Model Parameters Shape:")
    # pprint(jax.tree_util.tree_map(lambda x: x.shape, model.params))

    # Plot with improved function
    plot_training_history_improved(
        history,
        save_path=str(output_dir / "training_history_improved.png")
    )
    
    # =========================================================================
    # 5-6. VALIDATION (same as your original)
    # =========================================================================
    print("\n[5/6] Validation...")
    
    # Your existing validation code here
    # ...
    
    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print(f"\nOutputs saved to: {output_dir}")
    
    return best_params, history


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Your paths
    main_folder = r"E:\OneDrive\Dropbox\LinearRationalWishart_Work\Code\ED\LinearRationalWishart\LinearRationalWishart_NewCode\linear_rational_wishart"
    output_folder = r"E:\OneDrive\Dropbox\LinearRationalWishart_Work\Code\ED\LinearRationalWishart\LinearRationalWishart_NewCode\Output_results\neural_operator\saved_models"
    
    # Clear JAX cache
    jax.clear_caches()
    gc.collect()
    
    # Run improved training
    main_improved(
        wishart_path=main_folder,
        output_dir=output_folder,
        generate_training_data=False,  # Set True if you want new data
        use_existing_data=True
    )
