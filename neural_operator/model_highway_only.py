"""
Neural network architecture for Wishart characteristic function approximation.

Supports both REAL and COMPLEX training modes.

Architecture based on:
- Van Mieghem et al. (2023): Highway networks for option pricing
- Da Fonseca, Dawui, Malevergne: Wishart stochastic volatility framework

Author: Da Fonseca, Dawui, Malevergne
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional, Literal

import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from flax import serialization

from .config import WishartPINNConfig


# =============================================================================
# MATRIX UTILITIES
# =============================================================================

def matrix_to_upper_tri(mat: jnp.ndarray) -> jnp.ndarray:
    """Convert symmetric matrix to upper triangular vector."""
    d = mat.shape[0]
    result = []
    for i in range(d):
        for j in range(i, d):
            result.append(mat[i, j])
    return jnp.array(result)


def upper_tri_to_matrix(upper: jnp.ndarray, d: int) -> jnp.ndarray:
    """Convert upper triangular vector to symmetric matrix."""
    mat = jnp.zeros((d, d), dtype=upper.dtype)
    idx = 0
    for i in range(d):
        for j in range(i, d):
            mat = mat.at[i, j].set(upper[idx])
            if i != j:
                mat = mat.at[j, i].set(upper[idx])
            idx += 1
    return mat


def complex_upper_tri_to_matrix(upper: jnp.ndarray, d: int) -> jnp.ndarray:
    """
    Convert complex upper triangular representation to complex symmetric matrix.
    
    Args:
        upper: Array of shape (2 * n_upper,) with [real_upper, imag_upper]
        d: Matrix dimension
    
    Returns:
        Complex symmetric matrix of shape (d, d)
    """
    n_upper = d * (d + 1) // 2
    real_upper = upper[:n_upper]
    imag_upper = upper[n_upper:]
    
    real_mat = upper_tri_to_matrix(real_upper, d)
    imag_mat = upper_tri_to_matrix(imag_upper, d)
    
    return real_mat + 1j * imag_mat


def matrix_to_complex_upper_tri(mat: jnp.ndarray) -> jnp.ndarray:
    """
    Convert complex symmetric matrix to upper triangular representation.
    
    Args:
        mat: Complex symmetric matrix of shape (d, d)
    
    Returns:
        Array of shape (2 * n_upper,) with [real_upper, imag_upper]
    """
    real_upper = matrix_to_upper_tri(jnp.real(mat))
    imag_upper = matrix_to_upper_tri(jnp.imag(mat))
    return jnp.concatenate([real_upper, imag_upper])


# =============================================================================
# NETWORK LAYERS
# =============================================================================

class HighwayBlock(nn.Module):
    """
    Highway block with gating mechanism.
    
    References:
        Srivastava et al. (2015): Training Very Deep Networks
        Van Mieghem et al. (2023): Machine Learning for Option Pricing
    """
    # def __init__(self,features):
    #         super().__init__()
    #         self.features = features
    #         self.gate_bias = -2.0

    features: int
    gate_bias: float = -4.0#-2.0
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Transform
        h = nn.Dense(self.features)(x)
        h = nn.swish(h)
        
        # Gate (initialized to be mostly closed)
        t = nn.Dense(
            self.features,
            bias_init=nn.initializers.constant(self.gate_bias)
        )(x)
        t = nn.sigmoid(t)
        
        # Highway connection: blend transform with identity
        return h * t + x * (1 - t)


class WishartCharFuncNetwork(nn.Module):
    """
    Neural network for approximating A(T, Θ) and B(T, Θ).
    
    Supports both REAL and COMPLEX modes:
    
    REAL mode:
        - Input theta: (batch, n_upper) - real upper tri
        - Output A: (batch, n_upper) - real upper tri
        - Output B: (batch, 1) - real scalar
    
    COMPLEX mode:
        - Input theta: (batch, 2 * n_upper) - [real_upper, imag_upper]
        - Output A: (batch, 2 * n_upper) - [real_upper, imag_upper]
        - Output B: (batch, 2) - [real, imag]
    
    Attributes:
        dim: Matrix dimension (d for d×d matrices)
        hidden_dim: Width of hidden layers
        num_blocks: Number of highway blocks
        mode: "real" or "complex"
    """
    dim: int = 2
    hidden_dim: int = 128
    num_blocks: int = 6
    mode: str = "complex"

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Forward pass.
        
        Returns:
            A: Upper triangular elements
               - Real mode: shape (batch, n_upper)
               - Complex mode: shape (batch, 2 * n_upper) = [real, imag]
            B: Scalar output
               - Real mode: shape (batch, 1)
               - Complex mode: shape (batch, 2) = [real, imag]
        """
        n_upper = self.dim * (self.dim + 1) // 2
        
        # Input projection
        x = nn.Dense(self.hidden_dim)(inputs)
        x = nn.swish(x)
        
        # Highway blocks
        for _ in range(self.num_blocks):
            x = HighwayBlock(self.hidden_dim)(x)
        
        # # What Flax sees internally:
        # x = HighwayBlock(128)(x)  # Named "HighwayBlock_0"
        # x = HighwayBlock(128)(x)  # Named "HighwayBlock_1"
        # x = HighwayBlock(128)(x)  # Named "HighwayBlock_2"
        # x = HighwayBlock(128)(x)  # Named "HighwayBlock_3"
        # x = HighwayBlock(128)(x)  # Named "HighwayBlock_4"
        # x = HighwayBlock(128)(x)  # Named "HighwayBlock_5"

        # Output heads depend on mode
        if self.mode == "real":
            A = nn.Dense(n_upper, name='head_A')(x)
            B = nn.Dense(1, name='head_B')(x)
        else:
            A = nn.Dense(2 * n_upper, name='head_A')(x)
            B = nn.Dense(2, name='head_B')(x)
        
        return A, B

# =============================================================================
# MODEL WRAPPER
# =============================================================================

class WishartPINNModel:
    """
    Wrapper for the Wishart PINN model with save/load functionality.
    
    Supports both REAL and COMPLEX modes based on config.
    
    Example:
        >>> config = WishartPINNConfig(mode="real", dim=2, hidden_dim=128)
        >>> model = WishartPINNModel(config)
        >>> 
        >>> # Forward pass
        >>> A, B = model(model.params, inputs)
        >>> 
        >>> # Save model
        >>> model.save("./my_model", trained_params)
        >>> 
        >>> # Load model
        >>> loaded_model = WishartPINNModel.load("./my_model")
    """
    
    def __init__(self, config: WishartPINNConfig):
        """
        Initialize model.
        
        Args:
            config: Model configuration (includes mode)
        """
        self.config = config
        self.dim = config.dim
        self.mode = config.mode
        
        # Calculate dimensions
        d = self.dim
        n_upper = d * (d + 1) // 2
        
        self.n_upper = n_upper
        self.n_m = d * d
        self.n_omega = n_upper
        self.n_sigma = n_upper
        
        # Mode-dependent dimensions
        if self.mode == "real":
            self.n_theta = n_upper          # Real theta
            self.n_A = n_upper              # Real A
            self.n_B = 1                    # Real B
        else:
            self.n_theta = 2 * n_upper      # Complex theta: [real, imag]
            self.n_A = 2 * n_upper          # Complex A: [real, imag]
            self.n_B = 2                    # Complex B: [real, imag]
        
        # Total input dimension: T + theta + m + omega + sigma
        self.input_dim = 1 + self.n_theta + self.n_m + self.n_omega + self.n_sigma
        
        # Create network with mode
        self.network = WishartCharFuncNetwork(
            dim=config.dim,
            hidden_dim=config.hidden_dim,
            num_blocks=config.num_highway_blocks,
            mode=self.mode
        )
        
        # Initialize parameters
        key = random.PRNGKey(config.seed)
        dummy_input = jnp.zeros((1, self.input_dim))
        self.params = self.network.init(key, dummy_input)
        
        self.ouput_dim=self.n_A + self.n_B
        # Count parameters
        self.n_params = sum(x.size for x in jax.tree_util.tree_leaves(self.params))
    
    def __call__(
        self, 
        params: Dict, 
        inputs: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass through the network."""
        return self.network.apply(params, inputs)
    
    def save(self, path: str, params: Optional[Dict] = None):
        """
        Save model to disk.
        
        Saves:
            - config.json: Model configuration (includes mode)
            - params.pkl: Serialized parameters
        
        Args:
            path: Directory path to save model
            params: Parameters to save (if None, uses self.params)
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        if params is None:
            params = self.params
        
        # Save config (includes mode)
        config_path = path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        # Save parameters
        params_path = path / "params.pkl"
        with open(params_path, 'wb') as f:
            pickle.dump(serialization.to_bytes(params), f)
        
        print(f"Model saved to {path} (mode={self.mode})")
    
    @classmethod
    def load(cls, path: str) -> 'WishartPINNModel':
        """
        Load model from disk.
        
        Args:
            path: Directory path containing saved model
        
        Returns:
            Loaded WishartPINNModel instance with restored parameters
        """
        path = Path(path)
        
        # Load config (includes mode)
        config_path = path / "config.json"
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = WishartPINNConfig.from_dict(config_dict)
        
        # Create model (this initializes with random params)
        model = cls(config)
        
        # Load saved parameters
        params_path = path / "params.pkl"
        with open(params_path, 'rb') as f:
            params_bytes = pickle.load(f)
        model.params = serialization.from_bytes(model.params, params_bytes)
        
        # print(f"Model loaded from {path} (mode={model.mode})")
        return model
    
    def predict_real(
        self,
        params: Dict,
        inputs: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Forward pass for REAL mode with matrix reconstruction.
        
        Args:
            params: Network parameters
            inputs: Input array of shape (batch, input_dim)
        
        Returns:
            A_matrices: Real matrices of shape (batch, d, d)
            B_scalars: Real scalars of shape (batch,)
        """
        assert self.mode == "real", "predict_real only works in real mode"
        
        A_flat, B_flat = self.network.apply(params, inputs)
        
        batch_size = A_flat.shape[0]
        A_matrices = jnp.array([
            upper_tri_to_matrix(A_flat[i], self.dim)
            for i in range(batch_size)
        ])
        B_scalars = B_flat[:, 0]
        
        return A_matrices, B_scalars
    
    def predict_complex(
        self,
        params: Dict,
        inputs: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Forward pass for COMPLEX mode with matrix reconstruction.
        
        Args:
            params: Network parameters
            inputs: Input array of shape (batch, input_dim)
        
        Returns:
            A_complex: Complex matrices of shape (batch, d, d)
            B_complex: Complex scalars of shape (batch,)
        """
        assert self.mode == "complex", "predict_complex only works in complex mode"
        
        A_flat, B_flat = self.network.apply(params, inputs)
        
        batch_size = A_flat.shape[0]
        A_complex = jnp.array([
            complex_upper_tri_to_matrix(A_flat[i], self.dim)
            for i in range(batch_size)
        ])
        B_complex = B_flat[:, 0] + 1j * B_flat[:, 1]
        
        return A_complex, B_complex
    
    def summary(self) -> str:
        """Return model summary string."""
        lines = [
            "=" * 50,
            f"Wishart PINN Model Summary (MODE: {self.mode.upper()})",
            "=" * 50,
            f"Matrix dimension d:  {self.dim}",
            f"Upper tri elements:  {self.n_upper}",
            f"Input dimension:     {self.input_dim}",
            f"  - T:               1",
            f"  - theta:           {self.n_theta} ({'real' if self.mode == 'real' else 'complex'})",
            f"  - m:               {self.n_m}",
            f"  - omega:           {self.n_omega}",
            f"  - sigma:           {self.n_sigma}",
            f"Output dimension A:  {self.n_A} ({'real' if self.mode == 'real' else 'complex'})",
            f"Output dimension B:  {self.n_B} ({'real' if self.mode == 'real' else 'complex'})",
            f"Hidden dimension:    {self.config.hidden_dim}",
            f"Highway blocks:      {self.config.num_highway_blocks}",
            f"Total parameters:    {self.n_params:,}",
            "=" * 50,
        ]
        return "\n".join(lines)

    def count_params(self):
        #hidden_dim, num_blocks, input_dim=17, output_dim=8):
        # Input projection: input_dim → hidden_dim
        input_proj = self.input_dim * self.config.hidden_dim + self.config.hidden_dim
    
        # Each highway block: 2 dense layers (transform + gate)
        per_block = 2 * (self.config.hidden_dim * self.config.hidden_dim + self.config.hidden_dim)
        blocks_total = self.config.num_highway_blocks * per_block
    
        # Output heads: hidden_dim → output_dim
        output_heads = self.config.hidden_dim * self.ouput_dim + self.ouput_dim
    
        total = input_proj + blocks_total + output_heads
        print(f"{self.config.hidden_dim} × {self.config.num_highway_blocks}: {total:,} params")  
        return total