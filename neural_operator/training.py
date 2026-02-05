"""
Training module for Wishart Neural Operator.

Supports both REAL and COMPLEX training modes.

Author: Da Fonseca, Dawui, Malevergne
"""

from typing import Dict, Tuple, Optional, NamedTuple, Any, List
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from jax import random, jit, grad, vmap
import optax
from tqdm import tqdm
import matplotlib.pyplot as plt

from .config import WishartPINNConfig
from .model import WishartPINNModel, upper_tri_to_matrix, complex_upper_tri_to_matrix


# =============================================================================
# TRAINING STATE
# =============================================================================

class TrainState(NamedTuple):
    """Immutable training state."""
    params: Dict
    opt_state: Any
    step: int


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

def compute_data_loss(
    network_fn, 
    params: Dict, 
    batch: Dict[str, jnp.ndarray]
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Compute MSE loss on A and B predictions.
    
    Works for both REAL and COMPLEX modes:
    - REAL: A is (batch, n_upper), B is (batch, 1)
    - COMPLEX: A is (batch, 2*n_upper), B is (batch, 2)
    
    Args:
        network_fn: Network apply function
        params: Network parameters
        batch: Dictionary with 'T', 'theta', 'm', 'omega', 'sigma', 'A', 'B'
    
    Returns:
        total_loss: Combined loss
        aux: Dictionary with individual losses
    """
    # Prepare inputs
    T = batch['T'][:, None] if batch['T'].ndim == 1 else batch['T']
    inputs = jnp.concatenate([
        T, batch['theta'], batch['m'], batch['omega'], batch['sigma']
    ], axis=-1)
    
    # Forward pass
    A_pred, B_pred = network_fn(params, inputs)
    
    # Compute losses (works for both real and complex representations)
    loss_A = jnp.mean((A_pred - batch['A']) ** 2)
    loss_B = jnp.mean((B_pred - batch['B']) ** 2)
    
    total_loss = loss_A + loss_B
    
    return total_loss, {'loss_A': loss_A, 'loss_B': loss_B}


def compute_physics_loss(
    network_fn,
    params: Dict,
    batch: Dict[str, jnp.ndarray],
    dim: int,
    mode: str = "complex"
) -> jnp.ndarray:
    """
    Compute physics-informed loss based on Riccati ODE residual.
    
    The Riccati ODE is:
        dA/dt = A M + M^T A + 2 A Σ Σ^T A
    
    Args:
        network_fn: Network apply function
        params: Network parameters
        batch: Dictionary with inputs
        dim: Matrix dimension
        mode: "real" or "complex"
    
    Returns:
        Physics loss (mean squared residual)
    """
    n_upper = dim * (dim + 1) // 2
    
    def single_residual(T_val, theta, m_flat, omega, sigma_flat):
        """Compute residual for a single sample."""
        inputs = jnp.concatenate([
            T_val[None], theta[None], m_flat[None], omega[None], sigma_flat[None]
        ], axis=-1)
        
        # Reconstruct matrices
        m = m_flat.reshape((dim, dim))
        sigma = upper_tri_to_matrix(sigma_flat, dim)
        sigma2 = sigma @ sigma.T
        
        # Get A prediction
        A_flat, _ = network_fn(params, inputs)
        
        if mode == "real":
            A = upper_tri_to_matrix(A_flat[0], dim)
        else:
            A = complex_upper_tri_to_matrix(A_flat[0], dim)
        
        # Compute dA/dT using autodiff
        def A_of_T(t):
            inp = jnp.concatenate([
                t[None, None], theta[None], m_flat[None], omega[None], sigma_flat[None]
            ], axis=-1)
            A_u, _ = network_fn(params, inp)
            if mode == "real":
                return upper_tri_to_matrix(A_u[0], dim)
            else:
                return complex_upper_tri_to_matrix(A_u[0], dim)
        
        dA_dT = jax.jacfwd(A_of_T)(T_val)
        
        # Riccati RHS
        riccati_rhs = A @ m + m.T @ A + 2 * A @ sigma2 @ A
        
        # Residual
        residual = dA_dT - riccati_rhs
        
        if mode == "real":
            return jnp.sum(residual ** 2)
        else:
            return jnp.sum(jnp.abs(residual) ** 2)
    
    # Vectorize over batch
    residuals = vmap(single_residual)(
        batch['T'], batch['theta'], batch['m'], batch['omega'], batch['sigma']
    )
    
    return jnp.mean(residuals)


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def create_train_step(
    network_fn,
    optimizer,
    dim: int,
    mode: str = "complex",
    physics_weight: float = 0.0
):
    """
    Create JIT-compiled training step function.
    
    Args:
        network_fn: Network apply function
        optimizer: Optax optimizer
        dim: Matrix dimension
        mode: "real" or "complex"
        physics_weight: Weight for physics loss (0 = data only)
    
    Returns:
        train_step function
    """
    
    def loss_fn(params, batch):
        data_loss, aux = compute_data_loss(network_fn, params, batch)
        
        if physics_weight > 0:
            physics_loss = compute_physics_loss(network_fn, params, batch, dim, mode)
            total_loss = data_loss + physics_weight * physics_loss
            aux['loss_physics'] = physics_loss
        else:
            total_loss = data_loss
            aux['loss_physics'] = jnp.array(0.0)
        
        return total_loss, aux
    
    @jit
    def train_step(state: TrainState, batch: Dict) -> Tuple[TrainState, jnp.ndarray, Dict]:
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, batch)
        updates, new_opt_state = optimizer.update(grads, state.opt_state, state.params)
        new_params = optax.apply_updates(state.params, updates)
        return TrainState(new_params, new_opt_state, state.step + 1), loss, aux
    
    return train_step


def train_model(
    model: WishartPINNModel,
    config: WishartPINNConfig,
    train_data: Dict[str, jnp.ndarray],
    val_data: Optional[Dict[str, jnp.ndarray]] = None,
    save_path: Optional[str] = None,
    save_every: int = 100,
    physics_weight: float = 0.0,
    use_lr_schedule: bool = True,
    warmup_epochs: int = 100,
    min_lr_factor: float = 0.01
) -> Tuple[Dict, Dict[str, List[float]]]:
    """
    Train the model.
    
    Supports both REAL and COMPLEX modes based on config.
    
    Args:
        model: WishartPINNModel instance
        config: Training configuration (includes mode)
        train_data: Training data dictionary
        val_data: Optional validation data
        save_path: Path to save checkpoints
        save_every: Save checkpoint every N epochs
        physics_weight: Weight for physics loss
        use_lr_schedule: Whether to use cosine learning rate decay
        warmup_epochs: Number of warmup epochs
        min_lr_factor: Minimum LR as fraction of initial
    
    Returns:
        best_params: Parameters with best validation loss
        history: Training history dictionary
    """
    n_samples = train_data['T'].shape[0]
    n_batches = max(1, n_samples // config.batch_size)
    total_steps = config.num_epochs * n_batches
    
    print(f"\n{'='*60}")
    print(f"TRAINING (mode={config.mode.upper()})")
    print(f"{'='*60}")
    
    # Print data shapes for verification
    print(f"Data shapes:")
    print(f"  T:     {train_data['T'].shape}")
    print(f"  theta: {train_data['theta'].shape}")
    print(f"  m:     {train_data['m'].shape}")
    print(f"  omega: {train_data['omega'].shape}")
    print(f"  sigma: {train_data['sigma'].shape}")
    print(f"  A:     {train_data['A'].shape}")
    print(f"  B:     {train_data['B'].shape}")
    
    # Create optimizer with learning rate schedule
    if use_lr_schedule:
        warmup_steps = warmup_epochs * n_batches
        
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=config.learning_rate * 0.1,
            peak_value=config.learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=total_steps - warmup_steps,
            end_value=config.learning_rate * min_lr_factor
        )
        
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=schedule)
        )
        print(f"Using warmup ({warmup_epochs} epochs) + cosine decay schedule")
        print(f"  Peak LR: {config.learning_rate}, End LR: {config.learning_rate * min_lr_factor}")
    else:
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(config.learning_rate)
        )
    
    # Initialize state
    state = TrainState(
        params=model.params,
        opt_state=optimizer.init(model.params),
        step=0
    )
    
    # Create train step with mode
    train_step = create_train_step(
        model.network.apply, 
        optimizer, 
        model.dim,
        mode=config.mode,
        physics_weight=physics_weight
    )
    
    # History tracking
    history = {
        'loss': [],
        'loss_A': [],
        'loss_B': [],
        'loss_physics': [],
        'val_loss': [],
        'learning_rate': []
    }
    
    print(f"\nTraining for {config.num_epochs} epochs on {n_samples} samples")
    print(f"Batch size: {config.batch_size}, Batches per epoch: {n_batches}")
    if physics_weight > 0:
        print(f"Physics loss weight: {physics_weight}")
    
    best_loss = float('inf')
    best_params = None
    
    # For smoothed logging
    smooth_loss = None
    smooth_alpha = 0.9
    # print("\n Type of train_data...")
    # print(type(train_data))

    # for k, v in train_data.items():
    #     print(k, type(v), getattr(v, "shape", None))

    for epoch in tqdm(range(config.num_epochs), desc="Training"):
        # Shuffle data
        key = random.PRNGKey(epoch)
        perm = random.permutation(key, n_samples)
        
        epoch_loss = 0.0
        epoch_loss_A = 0.0
        epoch_loss_B = 0.0
        epoch_loss_physics = 0.0
        
        for batch_idx in range(n_batches):
            # Get batch
            start_idx = batch_idx * config.batch_size
            end_idx = min(start_idx + config.batch_size, n_samples)
            idx = perm[start_idx:end_idx]
            # batch = {k: v[idx] for k, v in train_data.items()}
            # batch = {
            #         k: v[idx] if isinstance(v, np.ndarray) and v.ndim > 0 else v
            #         for k, v in train_data.items()
            #         }

            batch = {   k: v[idx]
                        for k, v in train_data.items()
                        if isinstance(v, np.ndarray) and v.ndim > 0
                    }


            # Train step
            state, loss, aux = train_step(state, batch)
            
            epoch_loss += float(loss)
            epoch_loss_A += float(aux['loss_A'])
            epoch_loss_B += float(aux['loss_B'])
            epoch_loss_physics += float(aux['loss_physics'])
        
        # Average losses
        avg_loss = epoch_loss / n_batches
        history['loss'].append(avg_loss)
        history['loss_A'].append(epoch_loss_A / n_batches)
        history['loss_B'].append(epoch_loss_B / n_batches)
        history['loss_physics'].append(epoch_loss_physics / n_batches)
        
        # Track learning rate (approximate)
        if use_lr_schedule:
            current_step = epoch * n_batches
            if current_step < warmup_steps:
                current_lr = config.learning_rate * 0.1 + (config.learning_rate * 0.9) * current_step / warmup_steps
            else:
                progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
                current_lr = config.learning_rate * min_lr_factor + 0.5 * (config.learning_rate - config.learning_rate * min_lr_factor) * (1 + np.cos(np.pi * progress))
            history['learning_rate'].append(float(current_lr))
        else:
            history['learning_rate'].append(config.learning_rate)
        
        # Validation
        if val_data is not None:
            val_loss, _ = compute_data_loss(model.network.apply, state.params, val_data)
            history['val_loss'].append(float(val_loss))
            track_loss = float(val_loss)
        else:
            track_loss = avg_loss
        
        # Track best model
        if track_loss < best_loss:
            best_loss = track_loss
            best_params = state.params
        
        # Smoothed loss for logging
        if smooth_loss is None:
            smooth_loss = avg_loss
        else:
            smooth_loss = smooth_alpha * smooth_loss + (1 - smooth_alpha) * avg_loss
        
        # Logging
        if (epoch + 1) % 100 == 0:
            msg = f"Epoch {epoch+1}: Loss = {avg_loss:.6f} (smooth: {smooth_loss:.6f})"
            if val_data is not None:
                msg += f", Val = {history['val_loss'][-1]:.6f}"
            if use_lr_schedule:
                msg += f", LR = {history['learning_rate'][-1]:.2e}"
            tqdm.write(msg)
        
        # Save checkpoint
        if save_path is not None and (epoch + 1) % save_every == 0:
            checkpoint_path = Path(save_path) / f"checkpoint_{epoch+1}"
            model.save(str(checkpoint_path), state.params)
    
    # Save final model
    if save_path is not None:
        final_path = Path(save_path) / "final"
        model.save(str(final_path), best_params)
    
    print(f"\nTraining complete. Best loss: {best_loss:.6f}")
    
    return best_params, history


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    smooth_window: int = 20,
    title_suffix: str = ""
):
    """
    Plot training history.
    
    Args:
        history: Training history dictionary
        save_path: Path to save figure (if None, displays)
        smooth_window: Window size for smoothing curves
        title_suffix: Optional suffix for plot titles (e.g., mode info)
    """
    
    def smooth(values, window):
        """Simple moving average smoothing."""
        if len(values) < window:
            return values
        return np.convolve(values, np.ones(window)/window, mode='valid')
    
    has_lr = 'learning_rate' in history and len(history['learning_rate']) > 0
    n_plots = 3 if has_lr else 2
    
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    
    epochs = range(len(history['loss']))
    
    # Total loss (with smoothed version)
    axes[0].semilogy(history['loss'], alpha=0.3, color='blue', label='Train (raw)')
    if len(history['loss']) > smooth_window:
        smoothed = smooth(history['loss'], smooth_window)
        axes[0].semilogy(range(smooth_window-1, len(history['loss'])), smoothed, 
                         color='blue', linewidth=2, label='Train (smooth)')
    
    if history['val_loss']:
        axes[0].semilogy(history['val_loss'], alpha=0.3, color='orange', label='Val (raw)')
        if len(history['val_loss']) > smooth_window:
            smoothed_val = smooth(history['val_loss'], smooth_window)
            axes[0].semilogy(range(smooth_window-1, len(history['val_loss'])), smoothed_val,
                             color='orange', linewidth=2, label='Val (smooth)')
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title(f'Training Progress {title_suffix}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Component losses
    axes[1].semilogy(history['loss_A'], alpha=0.3, color='blue')
    axes[1].semilogy(history['loss_B'], alpha=0.3, color='orange')
    if len(history['loss_A']) > smooth_window:
        axes[1].semilogy(range(smooth_window-1, len(history['loss_A'])), 
                         smooth(history['loss_A'], smooth_window),
                         color='blue', linewidth=2, label='Loss A')
        axes[1].semilogy(range(smooth_window-1, len(history['loss_B'])), 
                         smooth(history['loss_B'], smooth_window),
                         color='orange', linewidth=2, label='Loss B')
    else:
        axes[1].semilogy(history['loss_A'], label='Loss A')
        axes[1].semilogy(history['loss_B'], label='Loss B')
    
    if any(history['loss_physics']):
        axes[1].semilogy(history['loss_physics'], alpha=0.5, label='Loss Physics')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title(f'Component Losses {title_suffix}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Learning rate (if available)
    if has_lr:
        axes[2].plot(history['learning_rate'], color='green', linewidth=2)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_title('Learning Rate Schedule')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_yscale('log')
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
