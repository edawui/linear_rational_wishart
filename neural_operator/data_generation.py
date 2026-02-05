"""
Data generation for training using existing Wishart.py implementation.

Supports both REAL and COMPLEX training modes.

This module interfaces with the Da Fonseca-Dawui-Malevergne Wishart
implementation to generate training data for the neural operator.

Author: Da Fonseca, Dawui, Malevergne
"""

from math import sqrt
import sys
from typing import Dict, Optional, Tuple
from pathlib import Path

import numpy as np
import jax.numpy as jnp
from jax import random
from tqdm import tqdm

from .config import WishartPINNConfig
from .model import matrix_to_upper_tri

from ..core.wishart_jump import WishartWithJump


# =============================================================================
# NUMPY MATRIX UTILITIES
# =============================================================================

def matrix_to_upper_tri_np(mat: np.ndarray) -> np.ndarray:
    """
    Convert symmetric matrix to upper triangular vector (NumPy version).
    
    Args:
        mat: Symmetric matrix of shape (d, d)
    
    Returns:
        Array of shape (d*(d+1)//2,) with upper triangular elements
    """
    d = mat.shape[0]
    result = []
    for i in range(d):
        for j in range(i, d):
            result.append(mat[i, j])
    return np.array(result)


def complex_matrix_to_upper_tri(mat: np.ndarray) -> np.ndarray:
    """
    Convert complex symmetric matrix to upper triangular representation.
    
    Uses NumPy operations throughout.
    
    Args:
        mat: Complex symmetric matrix of shape (d, d)
    
    Returns:
        Array of shape (2 * d*(d+1)//2,) with [real_upper, imag_upper]
    """
    mat_np = np.asarray(mat)
    
    real_upper = matrix_to_upper_tri_np(np.real(mat_np))
    imag_upper = matrix_to_upper_tri_np(np.imag(mat_np))
    
    return np.concatenate([real_upper, imag_upper])


def upper_tri_to_complex_matrix(upper_tri: np.ndarray, d: int) -> np.ndarray:
    """
    Convert upper triangular representation back to complex symmetric matrix.
    
    Args:
        upper_tri: Array of shape (2 * d*(d+1)//2,) with [real_upper, imag_upper]
        d: Matrix dimension
    
    Returns:
        Complex symmetric matrix of shape (d, d)
    """
    from .model import upper_tri_to_matrix
    
    n_upper = d * (d + 1) // 2
    real_upper = upper_tri[:n_upper]
    imag_upper = upper_tri[n_upper:]
    
    real_mat = upper_tri_to_matrix(jnp.array(real_upper), d)
    imag_mat = upper_tri_to_matrix(jnp.array(imag_upper), d)
    
    return np.array(real_mat) + 1j * np.array(imag_mat)


def upper_tri_to_matrix_np(upper: np.ndarray, d: int) -> np.ndarray:
    """Convert upper triangular vector to symmetric matrix (NumPy version)."""
    mat = np.zeros((d, d), dtype=upper.dtype)
    idx = 0
    for i in range(d):
        for j in range(i, d):
            mat[i, j] = upper[idx]
            if i != j:
                mat[j, i] = upper[idx]
            idx += 1
    return mat


# =============================================================================
# INTERFACE TO WISHART.PY
# =============================================================================

class WishartCharFuncComputer:
    """
    Interface to compute characteristic functions using existing Wishart.py code.
    
    This class wraps the Da Fonseca-Dawui-Malevergne implementation to provide
    ground truth values for training and validation.
    
    Example:
        >>> computer = WishartCharFuncComputer()
        >>> A, B = computer.compute_A_B(T=1.0, theta=theta, m=m, omega=omega, sigma=sigma)
        >>> phi = computer.compute_characteristic_function(T, theta, m, omega, sigma, x0)
    """
    
    def __init__(self, wishart_module_path: Optional[str] = None):
        """
        Initialize the interface.
        
        Args:
            wishart_module_path: Path to directory containing Wishart.py
                                If None, assumes it's in the Python path
        """
        self.wishart_module_path = wishart_module_path
        self._wishart_class = None
        self._loaded = False

        self._wishart_class = WishartWithJump
        self._loaded = True
        
    def _load_wishart_module(self):
        """Lazily load the Wishart module."""
        if self._loaded:
            return
            
        if self.wishart_module_path is not None:
            if self.wishart_module_path not in sys.path:
                sys.path.insert(0, self.wishart_module_path)
        
        try:
            from Wishart import WishartWithJump 
            self._wishart_class = WishartWithJump
            self._loaded = True
            print(f"Successfully loaded Wishart module from {self.wishart_module_path}")
        except ImportError as e:
            print(f"Warning: Could not import Wishart module: {e}")
            print("Falling back to numerical RK4 solver.")
            self._wishart_class = None
            self._loaded = True
    
    def compute_A_B(
        self,
        T: float,
        theta: np.ndarray,
        m: np.ndarray,
        omega: np.ndarray,
        sigma: np.ndarray,
        x0: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, complex]:
        """
        Compute A(T, Θ) and B(T, Θ) using the existing implementation.
        
        Works for both real and complex theta.
        
        Args:
            T: Time to maturity
            theta: Θ matrix (d x d, symmetric, can be real or complex)
            m: M matrix (d x d, mean-reversion)
            omega: Ω matrix (d x d, symmetric, positive semidefinite)
            sigma: Σ matrix (d x d)
            x0: Initial state (optional)
        
        Returns:
            A: A(T, Θ) matrix (d x d)
            B: B(T, Θ) scalar
        """
        self._load_wishart_module()
        
        d = theta.shape[0]
        if x0 is None:
            x0 = np.eye(d)
        
        if self._wishart_class is not None:
            wishart = self._wishart_class(d, x0, omega, m, sigma)
            wishart.maturity = T
            
            A = wishart.compute_a(T, theta)
            B = wishart.compute_b(T, theta)
            
            return A, B
        else:
            return self._compute_A_B_rk4(T, theta, m, omega, sigma)
    
    def _compute_A_B_rk4(
        self,
        T: float,
        theta: np.ndarray,
        m: np.ndarray,
        omega: np.ndarray,
        sigma: np.ndarray,
        dt: float = 0.001
    ) -> Tuple[np.ndarray, complex]:
        """
        Fallback RK4 solver for Riccati ODEs.
        
        Solves:
            dA/dt = A M + M^T A + 2 A Σ Σ^T A,  A(0) = Θ
            dB/dt = Tr(Ω A),                     B(0) = 0
        """
        sigma2 = sigma @ sigma.T
        
        def dA_dt(A):
            return A @ m + m.T @ A + 2 * A @ sigma2 @ A
        
        def dB_dt(A):
            return np.trace(omega @ A)
        
        n_steps = max(1, int(T / dt))
        h = T / n_steps
        
        # Use complex dtype if theta is complex
        if np.iscomplexobj(theta):
            A = theta.copy().astype(np.complex128)
            B = 0.0 + 0.0j
        else:
            A = theta.copy().astype(np.float64)
            B = 0.0
        
        for _ in range(n_steps):
            k1_A = dA_dt(A)
            k2_A = dA_dt(A + 0.5 * h * k1_A)
            k3_A = dA_dt(A + 0.5 * h * k2_A)
            k4_A = dA_dt(A + h * k3_A)
            A = A + (h / 6) * (k1_A + 2*k2_A + 2*k3_A + k4_A)
            
            A_mid = A - 0.5 * h * k2_A
            k1_B = dB_dt(A - h * k1_A / 6)
            k2_B = dB_dt(A_mid)
            k3_B = dB_dt(A_mid)
            k4_B = dB_dt(A)
            B = B + (h / 6) * (k1_B + 2*k2_B + 2*k3_B + k4_B)
        
        return A, B
    
    def compute_characteristic_function(
        self,
        T: float,
        theta: np.ndarray,
        m: np.ndarray,
        omega: np.ndarray,
        sigma: np.ndarray,
        x0: np.ndarray
    ) -> complex:
        """
        Compute Φ(T, Θ) = E[exp(Tr(Θ X_T)) | X_0 = x0].
        
        Uses the affine formula: Φ = exp(Tr(A x0) + B)
        """
        A, B = self.compute_A_B(T, theta, m, omega, sigma, x0)
        return np.exp(np.trace(A @ x0) + B)


# =============================================================================
# DATA GENERATOR
# =============================================================================

class WishartDataGenerator:
    """
    Generate training data using existing Wishart.py implementation.
    
    Supports both REAL and COMPLEX modes based on config.
    
    REAL mode:
        - theta: real symmetric matrix
        - A: real symmetric matrix
        - B: real scalar
    
    COMPLEX mode:
        - theta = (ur + i*ui) * a3: complex symmetric matrix
        - A: complex symmetric matrix
        - B: complex scalar
    
    Example:
        >>> config = WishartPINNConfig(mode="real", dim=2)
        >>> generator = WishartDataGenerator(config)
        >>> train_data = generator.generate_dataset(n_samples=10000)
    """
    
    def __init__(
        self, 
        config: WishartPINNConfig,
        wishart_module_path: Optional[str] = None
    ):
        """
        Initialize data generator.
        
        Args:
            config: Configuration specifying parameter ranges and mode
            wishart_module_path: Path to directory containing Wishart.py
        """
        self.config = config
        self.dim = config.dim
        self.mode = config.mode
        self.key = random.PRNGKey(config.seed + 1)
        
        self.char_func_computer = WishartCharFuncComputer(wishart_module_path)
    
    def _sample_parameters(self, key) -> Dict[str, np.ndarray]:
        """Sample random Wishart parameters based on mode."""
        d = self.dim
        cfg = self.config
        
        # Calculate number of keys needed
        if self.mode == "real":
            nb_key = 1 + d*d + 3 * ((d*(d+1))//2)  # T + m + [omega, sigma, theta]
        else:
            nb_key = 1 + d*d + 3 * ((d*(d+1))//2) + 1  # T + m + [omega, sigma, a3] + ui
        
        keys = random.split(key, nb_key)
        
        # Sample T
        T = float(random.uniform(keys[0], minval=cfg.T_min, maxval=cfg.T_max))
        key_id = 0
        
        # Sample M (mean-reversion: negative diagonal)
        m = np.zeros((d, d))
        for i in range(d):
            key_id += 1
            m[i, i] = float(random.uniform(
                keys[key_id], minval=cfg.m_diag_min, maxval=cfg.m_diag_max
            ))
     
        for i in range(d):
            for j in range(d):
                if i != j:
                    key_id += 1
                    m[i, j] = float(random.uniform(
                        keys[key_id], minval=cfg.m_offdiag_min, maxval=cfg.m_offdiag_max
                    ))
        
        # Sample Omega (symmetric positive definite)
        omega = np.zeros((d, d))
        for i in range(d):
            key_id += 1
            omega[i, i] = float(random.uniform(
                keys[key_id], minval=cfg.omega_diag_min, maxval=cfg.omega_diag_max
            ))
        for i in range(d):
            for j in range(i + 1, d):
                key_id += 1
                omega[i, j] = float(random.uniform(
                    keys[key_id], minval=cfg.omega_offdiag_min, maxval=cfg.omega_offdiag_max
                )) * sqrt(omega[i, i] * omega[j, j])
                omega[j, i] = omega[i, j]

        # Sample Sigma (symmetric)
        sigma = np.zeros((d, d))
        for i in range(d):
            key_id += 1
            sigma[i, i] = float(random.uniform(
                keys[key_id], minval=cfg.sigma_diag_min, maxval=cfg.sigma_diag_max
            ))
        for i in range(d):
            for j in range(i + 1, d):
                key_id += 1
                sigma[i, j] = float(random.uniform(
                    keys[key_id], minval=cfg.sigma_offdiag_min, maxval=cfg.sigma_offdiag_max
                )) * sqrt(sigma[i, i] * sigma[j, j])
                sigma[j, i] = sigma[i, j]

        # Sample theta based on mode
        if self.mode == "real":
            # REAL MODE: theta is real symmetric
            theta = np.zeros((d, d))
            for i in range(d):
                key_id += 1
                theta[i, i] = float(random.uniform(
                    keys[key_id], minval=cfg.theta_min, maxval=cfg.theta_max
                ))
            for i in range(d):
                for j in range(i + 1, d):
                    key_id += 1
                    theta[i, j] = float(random.uniform(
                        keys[key_id], minval=cfg.theta_offdiag_min, maxval=cfg.theta_offdiag_max
                    )) * sqrt(theta[i, i] * theta[j, j])
                    theta[j, i] = theta[i, j]
            
            return {
                'T': T,
                'm': m,
                'omega': omega,
                'sigma': sigma,
                'theta': theta,
            }
        
        else:
            # COMPLEX MODE: theta = z * a3 where z = ur + i*ui
            # Sample a3 (base matrix for theta, real symmetric)
            a3 = np.zeros((d, d))
            for i in range(d):
                key_id += 1
                a3[i, i] = float(random.uniform(
                    keys[key_id], minval=cfg.theta_min, maxval=cfg.theta_max
                ))
            
            for i in range(d):
                for j in range(i + 1, d):
                    key_id += 1
                    a3[i, j] = float(random.uniform(
                        keys[key_id], minval=cfg.theta_offdiag_min, maxval=cfg.theta_offdiag_max
                    )) ## * sqrt(a3[i, i] * a3[j, j])
                    a3[j, i] = a3[i, j]
            
            # Sample ui for complex part
            key_id += 1
            ui = float(random.uniform(keys[key_id], minval=cfg.ui_min, maxval=cfg.ui_max))
            
            # Create complex theta = z * a3
            z = complex(cfg.ur, ui)
            theta = z * a3
            
            return {
                'T': T,
                'm': m,
                'omega': omega,
                'sigma': sigma,
                'theta': theta,
                'a3': a3,
                'z_real': cfg.ur,
                'z_imag': ui
            }
   
    def generate_dataset(
        self, 
        n_samples: int,
        show_progress: bool = True
    ) -> Dict[str, jnp.ndarray]:
        """
        Generate training dataset based on mode.
        
        Args:
            n_samples: Number of samples to generate
            show_progress: Whether to show progress bar
        
        Returns:
            Dictionary with arrays (format depends on mode)
        """
        d = self.dim
        n_upper = d * (d + 1) // 2
        n_m = d * d
        
        # Pre-allocate arrays based on mode
        T_data = np.zeros(n_samples)
        m_data = np.zeros((n_samples, n_m))
        omega_data = np.zeros((n_samples, n_upper))
        sigma_data = np.zeros((n_samples, n_upper))
        
        if self.mode == "real":
            theta_data = np.zeros((n_samples, n_upper))      # Real
            A_data = np.zeros((n_samples, n_upper))          # Real
            B_data = np.zeros((n_samples, 1))                # Real scalar
            print(f"Generating {n_samples} REAL mode training samples...")
        else:
            theta_data = np.zeros((n_samples, 2 * n_upper))  # Complex
            A_data = np.zeros((n_samples, 2 * n_upper))      # Complex
            B_data = np.zeros((n_samples, 2))                # Complex scalar
            print(f"Generating {n_samples} COMPLEX mode training samples...")
        
        print(f"  Matrix dimension d={d}")
        print(f"  theta shape: (n, {theta_data.shape[1]})")
        print(f"  A shape: (n, {A_data.shape[1]})")
        print(f"  B shape: (n, {B_data.shape[1]})")
        
        iterator = tqdm(range(n_samples)) if show_progress else range(n_samples)
        valid_samples = 0
        
        for i in iterator:
            self.key, subkey = random.split(self.key)
            
            # Sample parameters
            params = self._sample_parameters(subkey)
            
            # Compute A and B using Wishart.py
            try:
                A, B = self.char_func_computer.compute_A_B(
                    params['T'], 
                    params['theta'], 
                    params['m'], 
                    params['omega'], 
                    params['sigma']
                )
                
                # Store data
                T_data[valid_samples] = params['T']
                m_data[valid_samples] = params['m'].flatten()
                omega_data[valid_samples] = matrix_to_upper_tri_np(params['omega'])
                sigma_data[valid_samples] = matrix_to_upper_tri_np(params['sigma'])
                
                if self.mode == "real":
                    # REAL MODE: store real values
                    theta_data[valid_samples] = matrix_to_upper_tri_np(params['theta'])
                    A_data[valid_samples] = matrix_to_upper_tri_np(np.real(A))
                    B_data[valid_samples] = np.array([np.real(B)])
                else:
                    # COMPLEX MODE: store [real, imag]
                    theta_data[valid_samples] = complex_matrix_to_upper_tri(params['theta'])
                    A_data[valid_samples] = complex_matrix_to_upper_tri(A)
                    B_data[valid_samples] = np.array([np.real(B), np.imag(B)])
                
                valid_samples += 1
                
            except Exception as e:
                if show_progress:
                    tqdm.write(f"Warning: Error computing A, B for sample {i}: {e}")
                continue
        
        # Trim to valid samples
        if valid_samples < n_samples:
            print(f"Warning: Only {valid_samples}/{n_samples} samples were valid")
            T_data = T_data[:valid_samples]
            theta_data = theta_data[:valid_samples]
            m_data = m_data[:valid_samples]
            omega_data = omega_data[:valid_samples]
            sigma_data = sigma_data[:valid_samples]
            A_data = A_data[:valid_samples]
            B_data = B_data[:valid_samples]
        
        print(f"Generated {valid_samples} valid samples")
        
        return {
            'T': jnp.array(T_data),
            'theta': jnp.array(theta_data),
            'm': jnp.array(m_data),
            'omega': jnp.array(omega_data),
            'sigma': jnp.array(sigma_data),
            'A': jnp.array(A_data),
            'B': jnp.array(B_data),
        }
    
    def save_dataset(self, data: Dict[str, jnp.ndarray], path: str):
        """Save dataset to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        np.savez(
            path,
            T=np.array(data['T']),
            theta=np.array(data['theta']),
            m=np.array(data['m']),
            omega=np.array(data['omega']),
            sigma=np.array(data['sigma']),
            A=np.array(data['A']),
            B=np.array(data['B']),
            mode=self.mode,  # Save mode info
        )
        print(f"Dataset saved to {path} (mode={self.mode})")
    
    @staticmethod
    def load_dataset(path: str) -> Dict[str, jnp.ndarray]:
        """Load dataset from disk."""
        data = np.load(path, allow_pickle=True)
        result = {
            'T': jnp.array(data['T']),
            'theta': jnp.array(data['theta']),
            'm': jnp.array(data['m']),
            'omega': jnp.array(data['omega']),
            'sigma': jnp.array(data['sigma']),
            'A': jnp.array(data['A']),
            'B': jnp.array(data['B']),
        }
        
        # Check if mode was saved
        if 'mode' in data:
            print(f"Loaded dataset from {path} (mode={data['mode']})")
        else:
            # Infer mode from data shape
            n_upper = 3  # For d=2
            if result['theta'].shape[1] == n_upper:
                print(f"Loaded dataset from {path} (inferred mode=real)")
            else:
                print(f"Loaded dataset from {path} (inferred mode=complex)")
        
        return result
    
    @staticmethod
    def load_and_merge_chunks(chunk_dir: str, pattern: str = "chunk_*.npz") -> Dict[str, jnp.ndarray]:
        """Load and merge multiple chunk files."""
        import glob
        
        chunk_dir = Path(chunk_dir)
        chunk_files = sorted(glob.glob(str(chunk_dir / pattern)))
        
        if not chunk_files:
            raise ValueError(f"No chunk files found matching {chunk_dir / pattern}")
        
        print(f"Found {len(chunk_files)} chunk files to merge...")
        
        all_data = {
            'T': [], 'theta': [], 'm': [], 'omega': [], 
            'sigma': [], 'A': [], 'B': []
        }
        
        for chunk_file in chunk_files:
            data = np.load(chunk_file)
            for key in all_data:
                all_data[key].append(data[key])
            print(f"  Loaded {chunk_file}: {data['T'].shape[0]} samples")
        
        result = {
            key: jnp.array(np.concatenate(all_data[key], axis=0))
            for key in all_data
        }
        
        print(f"Total merged samples: {result['T'].shape[0]}")
        return result
    
    @staticmethod
    def merge_datasets(*datasets: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        """Merge multiple dataset dictionaries."""
        if not datasets:
            raise ValueError("No datasets provided")
        
        keys = datasets[0].keys()
        result = {
            key: jnp.concatenate([np.array(d[key]) for d in datasets], axis=0)
            for key in keys
        }
        
        print(f"Merged {len(datasets)} datasets: total {result['T'].shape[0]} samples")
        return result
