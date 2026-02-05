"""
Wishart Neural Operator Package

Supports both REAL and COMPLEX training modes.

REAL mode:
    - Train on real-valued theta, A, B
    - Use analytic extension for complex queries
    - Simpler training, smaller network

COMPLEX mode:
    - Train directly on complex-valued theta, A, B
    - Direct inference for complex queries
    - Larger network, potentially harder to train

Usage:
    # REAL mode (recommended)
    from linear_rational_wishart.neural_operator import get_real_config, WishartPINNModel
    config = get_real_config(dim=2, hidden_dim=128, num_epochs=2000)
    model = WishartPINNModel(config)
    
    # COMPLEX mode
    from linear_rational_wishart.neural_operator import get_complex_config, WishartPINNModel
    config = get_complex_config(dim=2, hidden_dim=128, num_epochs=3000)
    model = WishartPINNModel(config)

Author: Da Fonseca, Dawui, Malevergne
"""

from .config import (
    WishartPINNConfig,
    get_real_config,
    get_complex_config,
)

from .model import (
    WishartPINNModel,
    WishartCharFuncNetwork,
    HighwayBlock,
    matrix_to_upper_tri,
    upper_tri_to_matrix,
    complex_upper_tri_to_matrix,
    matrix_to_complex_upper_tri,
)

from .data_generation import (
    WishartCharFuncComputer,
    WishartDataGenerator,
    complex_matrix_to_upper_tri,
    matrix_to_upper_tri_np,
    upper_tri_to_matrix_np,
)

from .training import (
    train_model,
    plot_training_history,
    TrainState,
    compute_data_loss,
)

from .inference import (
    WishartPINNInference,
    validate_model,
    benchmark_throughput,
)

__all__ = [
    # Config
    'WishartPINNConfig',
    'get_real_config',
    'get_complex_config',
    # Model
    'WishartPINNModel',
    'WishartCharFuncNetwork',
    'HighwayBlock',
    'matrix_to_upper_tri',
    'upper_tri_to_matrix',
    'complex_upper_tri_to_matrix',
    'matrix_to_complex_upper_tri',
    # Data generation
    'WishartCharFuncComputer',
    'WishartDataGenerator',
    'complex_matrix_to_upper_tri',
    'matrix_to_upper_tri_np',
    'upper_tri_to_matrix_np',
    # Training
    'train_model',
    'plot_training_history',
    'TrainState',
    'compute_data_loss',
    # Inference
    'WishartPINNInference',
    'validate_model',
    'benchmark_throughput',
]

__version__ = '2.0.0'
