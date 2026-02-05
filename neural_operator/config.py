"""
Configuration classes for Wishart Neural Operator.

Supports both REAL and COMPLEX training modes.

Author: Da Fonseca, Dawui, Malevergne
"""
from dataclasses import dataclass, asdict, field
from typing import Dict, Literal


@dataclass
class WishartPINNConfig:
    """Configuration for the Wishart PINN model."""
    
    # === MODE SELECTION ===
    mode: Literal["real", "complex"] = "complex"  # Default to complex for pricing
    
    # Matrix dimension
    dim: int = 2
    
    # Network architecture
    hidden_dim: int = 128
    num_highway_blocks: int = 6
    
    # Training
    batch_size: int = 256
    num_epochs: int = 1000
    learning_rate: float = 1e-3
    
    # Physics loss weight
    physics_loss_weight: float = 0.1
    
    # Parameter ranges for data generation
    T_min: float = 0.5
    T_max: float = 5.0
    
    # === THETA PARAMETERS ===
    # For REAL mode: theta diagonal and off-diagonal ranges
    # For COMPLEX mode: a3 (base matrix) ranges
    theta_min: float = 0.01
    theta_max: float = 10.0
    theta_offdiag_min: float = -0.9
    theta_offdiag_max: float = 0.9
    
    # === COMPLEX MODE ONLY ===
    # z = ur + i*ui, theta = z * a3
    ui_min: float = 0.0
    ui_max: float = 25.0
    ur: float = 0.5  # Fixed real part (matches Fourier pricing)
    
    # M matrix (mean-reversion, negative diagonal)
    m_diag_min: float = -1.0
    m_diag_max: float = -0.1
    m_offdiag_min: float = 0.0
    m_offdiag_max: float = 0.0
    
    # Omega matrix
    omega_diag_min: float = 0.0001
    omega_diag_max: float = 0.5
    omega_offdiag_min: float = -0.9
    omega_offdiag_max: float = 0.9
    
    # Sigma matrix
    sigma_diag_min: float = 0.001
    sigma_diag_max: float = 0.5
    sigma_offdiag_min: float = -0.9
    sigma_offdiag_max: float = 0.9
    
    # Random seed
    seed: int = 42
    
    # === COMPUTED PROPERTIES ===
    
    @property
    def n_upper(self) -> int:
        """Number of upper triangular elements for a dxd symmetric matrix."""
        return self.dim * (self.dim + 1) // 2
    
    @property
    def input_dim(self) -> int:
        """Total input dimension for the network."""
        d = self.dim
        n_up = self.n_upper
        
        if self.mode == "real":
            # T + theta(real) + m + omega + sigma
            # 1 + n_upper + d*d + n_upper + n_upper
            return 1 + n_up + d*d + n_up + n_up
        else:
            # T + theta(complex) + m + omega + sigma
            # 1 + 2*n_upper + d*d + n_upper + n_upper
            return 1 + 2*n_up + d*d + n_up + n_up
    
    @property
    def output_dim_A(self) -> int:
        """Output dimension for A."""
        if self.mode == "real":
            return self.n_upper  # Real symmetric
        else:
            return 2 * self.n_upper  # Complex symmetric
    
    @property
    def output_dim_B(self) -> int:
        """Output dimension for B."""
        if self.mode == "real":
            return 1  # Real scalar
        else:
            return 2  # Complex scalar [real, imag]
    
    @property
    def is_complex(self) -> bool:
        """Check if in complex mode."""
        return self.mode == "complex"
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'WishartPINNConfig':
        """Create config from dictionary."""
        return cls(**d)
    
    @classmethod
    def for_pricing(cls, **kwargs) -> 'WishartPINNConfig':
        """Create config optimized for Fourier pricing (complex mode)."""
        defaults = {
            'mode': 'complex',
            'ui_min': 0.0,
            'ui_max': 25.0,  # Match integration range
            'ur': 0.5,       # Match Fourier pricing ur parameter
        }
        defaults.update(kwargs)
        return cls(**defaults)
    
    @classmethod
    def for_moments(cls, **kwargs) -> 'WishartPINNConfig':
        """Create config for moment calculations (real mode)."""
        defaults = {
            'mode': 'real',
            'theta_min': 0.01,
            'theta_max': 5.0,
        }
        defaults.update(kwargs)
        return cls(**defaults)
    
    def __repr__(self) -> str:
        lines = [f"WishartPINNConfig(mode='{self.mode}')"]
        for k, v in self.to_dict().items():
            if k != 'mode':
                lines.append(f"    {k}={v},")
        lines.append(")")
        return "\n".join(lines)

# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

def get_real_config(
    dim: int = 2,
    hidden_dim: int = 128,
    num_highway_blocks: int = 6,
    num_epochs: int = 2000,
    learning_rate: float = 3e-4,
    **kwargs
) -> WishartPINNConfig:
    """Get configuration for REAL mode training."""
    return WishartPINNConfig(
        mode="real",
        dim=dim,
        hidden_dim=hidden_dim,
        num_highway_blocks=num_highway_blocks,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        **kwargs
    )


def get_complex_config(
    dim: int = 2,
    hidden_dim: int = 128,
    num_highway_blocks: int = 6,
    num_epochs: int = 3000,
    learning_rate: float = 3e-4,
    ui_max: float = 25.0,
    **kwargs
) -> WishartPINNConfig:
    """Get configuration for COMPLEX mode training."""
    # return WishartPINNConfig(
    #     mode="complex",
    #     dim=dim,
    #     hidden_dim=hidden_dim,
    #     num_highway_blocks=num_highway_blocks,
    #     num_epochs=num_epochs,
    #     learning_rate=learning_rate,
    #     ui_max=ui_max,
    #     **kwargs
    # )

# Create config for COMPLEX mode
    config = WishartPINNConfig(
    # Mode
    mode="complex",  # CRITICAL: Must be complex for pricing!
    
    # Architecture
    dim=2,
    hidden_dim=hidden_dim,        # Increased from 128
    num_highway_blocks=num_highway_blocks,  # Increased from 6
    
    # Training
    batch_size=512,
    num_epochs=num_epochs,       # More epochs for better convergence
    learning_rate=learning_rate,
    
    # Complex theta parameters - MUST MATCH PRICING
    ur=0.5,                # Same as Fourier pricing ur
    ui_min=0.0,
    ui_max=25.0,           # Cover full integration range
    
    # Theta base matrix (a3) ranges
    theta_min=0.001,#0.01,
    theta_max=10.0,
    theta_offdiag_min=0.0,#-0.5,
    theta_offdiag_max=0.0,#0.5,
    
    # Other parameters (keep as before)
    T_min=0.5,
    T_max=5.0,
    
    m_diag_min=-0.25,#-1.0,
    m_diag_max=-0.0001,#-0.1,
    m_offdiag_min=0.0,
    m_offdiag_max=0.0,

    omega_diag_min=0.0001,
    omega_diag_max=0.25,#0.5,
    omega_offdiag_min=-0.5,
    omega_offdiag_max=0.5,

    sigma_diag_min=0.0001,
    sigma_diag_max=0.05,#0.5,
    sigma_offdiag_min=-0.5,
    sigma_offdiag_max=0.5,
    
    seed=42,
    )
    return config