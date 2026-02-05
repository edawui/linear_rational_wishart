"""
Inference module for Wishart Neural Operator.

Supports both REAL and COMPLEX modes:
- REAL mode: Uses analytic extension for complex queries
- COMPLEX mode: Direct neural network inference

Author: Da Fonseca, Dawui, Malevergne
"""

import time
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, random
from jax.scipy import linalg as jlinalg
from tqdm import tqdm

from .config import WishartPINNConfig
from .model import (
    WishartPINNModel, 
    matrix_to_upper_tri, 
    upper_tri_to_matrix,
    complex_upper_tri_to_matrix,
    matrix_to_complex_upper_tri
)
from .data_generation import (
    WishartCharFuncComputer, 
    complex_matrix_to_upper_tri,
    matrix_to_upper_tri_np,
    upper_tri_to_matrix_np
)


# =============================================================================
# NUMPY MATRIX UTILITIES
# =============================================================================

def complex_matrix_to_upper_tri_np(mat: np.ndarray) -> np.ndarray:
    """Convert complex symmetric matrix to [real_upper, imag_upper] (NumPy)."""
    mat_np = np.asarray(mat)
    real_upper = matrix_to_upper_tri_np(np.real(mat_np))
    imag_upper = matrix_to_upper_tri_np(np.imag(mat_np))
    return np.concatenate([real_upper, imag_upper])


# =============================================================================
# INFERENCE CLASS
# =============================================================================

class WishartPINNInference:
    """
    Fast inference using trained PINN model.
    
    Supports both REAL and COMPLEX modes:
    
    REAL mode:
        - Network trained on real theta, A, B
        - For complex queries, uses analytic extension
        - compute_A_B() auto-detects query type
    
    COMPLEX mode:
        - Network trained on complex theta, A, B
        - Direct inference for complex queries
    
    Example:
        >>> model = WishartPINNModel.load("./saved_model")
        >>> inference = WishartPINNInference(model)
        >>> 
        >>> # Works for both real and complex theta
        >>> A, B = inference.compute_A_B(T, theta, m, omega, sigma)
        >>> phi = inference.compute_characteristic_function(T, theta, m, omega, sigma, x0)
    """
    
    def __init__(
        self, 
        model: WishartPINNModel, 
        params: Optional[Dict] = None,
        wishart_module_path: Optional[str] = None,
        normalization_stats_path: Optional[str] = None
    ):
        """
        Initialize inference object.
        
        Args:
            model: WishartPINNModel instance
            params: Model parameters (if None, uses model.params)
            wishart_module_path: Path to Wishart.py for comparison
            normalization_stats_path: Path to normalization stats file
        """
        self.model = model
        self.params = params if params is not None else model.params
        self.dim = model.dim
        self.config = model.config
        self.mode = model.mode
        self.n_upper = model.n_upper
        
        # JIT compile forward pass
        self._forward = jit(model.network.apply)
        
        # For comparison with numerical methods
        self.numerical_computer = WishartCharFuncComputer(wishart_module_path)

        # Load normalization stats once
        self.norm_stats = None
        if normalization_stats_path is not None:
            norm_path = Path(normalization_stats_path)
            if norm_path.exists():
                self.norm_stats = dict(np.load(normalization_stats_path))
                # print(f"Loaded normalization stats from {normalization_stats_path}")
            else:
                print(f"Warning: normalization stats not found at {normalization_stats_path}")
    
    def _prepare_inputs_real(
        self,
        T: float,
        theta: np.ndarray,
        m: np.ndarray,
        omega: np.ndarray,
        sigma: np.ndarray
    ) -> jnp.ndarray:
        """Prepare inputs for REAL mode (theta is real)."""
        theta_np = np.asarray(theta)
        m_np = np.asarray(m)
        omega_np = np.asarray(omega)
        sigma_np = np.asarray(sigma)
        
        T_arr = np.array([[float(T)]])
        theta_arr = np.array([matrix_to_upper_tri_np(np.real(theta_np))])  # Real only
        m_arr = np.array([m_np.flatten()])
        omega_arr = np.array([matrix_to_upper_tri_np(np.real(omega_np))])
        sigma_arr = np.array([matrix_to_upper_tri_np(np.real(sigma_np))])
        
        inputs_np = np.concatenate([T_arr, theta_arr, m_arr, omega_arr, sigma_arr], axis=-1)
        return jnp.array(inputs_np)
    
    def _prepare_inputs_complex(
        self,
        T: float,
        theta: np.ndarray,
        m: np.ndarray,
        omega: np.ndarray,
        sigma: np.ndarray
    ) -> jnp.ndarray:
        """Prepare inputs for COMPLEX mode (theta is complex)."""
        theta_np = np.asarray(theta)
        m_np = np.asarray(m)
        omega_np = np.asarray(omega)
        sigma_np = np.asarray(sigma)
        
        T_arr = np.array([[float(T)]])
        theta_arr = np.array([complex_matrix_to_upper_tri_np(theta_np)])  # Complex
        m_arr = np.array([m_np.flatten()])
        omega_arr = np.array([matrix_to_upper_tri_np(np.real(omega_np))])
        sigma_arr = np.array([matrix_to_upper_tri_np(np.real(sigma_np))])
        
        inputs_np = np.concatenate([T_arr, theta_arr, m_arr, omega_arr, sigma_arr], axis=-1)
        return jnp.array(inputs_np)
     
    def _prepare_inputs(
        self,
        T: float,
        theta: np.ndarray,
        m: np.ndarray,
        omega: np.ndarray,
        sigma: np.ndarray
    ) -> jnp.ndarray:
        """Prepare inputs based on mode."""
        T_arr = jnp.array([[T]])
        
        if self.mode == "real":
            # Real mode: theta is real symmetric
            if np.iscomplexobj(theta):
                raise ValueError("Real mode model cannot accept complex theta!")
            theta_arr = jnp.array([matrix_to_upper_tri(theta)])
        else:
            # Complex mode: theta can be complex
            theta_arr = jnp.array([complex_matrix_to_upper_tri(theta)])
        
        m_arr = jnp.array([m.flatten()])
        omega_arr = jnp.array([matrix_to_upper_tri(omega)])
        sigma_arr = jnp.array([matrix_to_upper_tri(sigma)])
        
        inputs = jnp.concatenate([T_arr, theta_arr, m_arr, omega_arr, sigma_arr], axis=-1)
        return inputs
    
    def compute_A_B(
        self,
        T: float,
        theta: np.ndarray,
        m: np.ndarray,
        omega: np.ndarray,
        sigma: np.ndarray,
        denormalize: bool = True
    ) -> Tuple[np.ndarray, complex]:
        """Compute A and B, handling mode appropriately."""
        d = self.dim
    
        inputs = self._prepare_inputs(T, theta, m, omega, sigma)
        A_flat, B_flat = self._forward(self.params, inputs)
    
        # Denormalize if stats available
        if denormalize and self.norm_stats is not None:
            A_flat = np.array(A_flat) * self.norm_stats['A_std'] + self.norm_stats['A_mean']
            B_flat = np.array(B_flat) * self.norm_stats['B_std'] + self.norm_stats['B_mean']
        else:
            A_flat = np.array(A_flat)
            B_flat = np.array(B_flat)
    
        # Convert based on mode
        if self.mode == "real":
            A = upper_tri_to_matrix_np(A_flat[0], d)
            B = float(B_flat[0, 0])
        else:
            A = complex_upper_tri_to_matrix(A_flat[0], d)
            B = complex(float(B_flat[0, 0]), float(B_flat[0, 1]))
    
        return np.array(A), B
    
    def _compute_var_sigma(
        self,
        T: float,
        m: np.ndarray,
        sigma: np.ndarray
    ) -> np.ndarray:
        """
        Compute var_sigma(T) analytically.
        
        var_sigma = A^{-1} @ (e^{AT} - I) @ vec(Σ²)
        where A = I ⊗ M + M ⊗ I (Kronecker sum)
        
        This is used for analytic complex extension in REAL mode.
        """
        d = self.dim
        sigma2 = sigma @ sigma.T
        
        # Kronecker sum: A = I ⊗ M + M ⊗ I
        I_d = np.eye(d)
        A_kron = np.kron(I_d, m) + np.kron(m, I_d)
        
        # Matrix exponential
        eAt = jlinalg.expm(T * A_kron)
        
        # vec(Σ²)
        vec_sigma2 = sigma2.flatten()
        
        # var_sigma = A^{-1} @ (e^{AT} - I) @ vec(Σ²)
        I_nn = np.eye(d * d)
        diff = np.array(eAt) - I_nn
        v = np.linalg.solve(A_kron, diff @ vec_sigma2)
        
        return v.reshape((d, d))
    
    def _compute_A_B_complex_from_real(
        self,
        T: float,
        theta_complex: np.ndarray,
        m: np.ndarray,
        omega: np.ndarray,
        sigma: np.ndarray
    ) -> Tuple[np.ndarray, complex]:
        """
        Compute A and B for complex theta using REAL mode model.
        
        Uses analytic extension:
        1. Compute var_sigma(T) analytically
        2. Compute A(T, theta_complex) using the formula
        3. Integrate for B
        
        This is mathematically exact because the Riccati solution
        A(t, θ) is analytic in θ.
        """
        d = self.dim
        
        # Compute var_sigma analytically
        var_sigma = self._compute_var_sigma(T, m, sigma)
        
        # Compute e^{mT}
        emt = np.array(jlinalg.expm(T * m))
        
        # Compute A(T, theta_complex)
        # A = e^{mT}^T @ (I - 2*theta*var_sigma)^{-1} @ theta @ e^{mT}
        theta_var = theta_complex @ var_sigma
        I_d = np.eye(d)
        denominator = I_d - 2 * theta_var
        m2 = np.linalg.inv(denominator)  # This works with complex
        
        A = emt.T @ m2 @ theta_complex @ emt
        
        # Compute B by numerical integration
        # B = integral_0^T Tr(Ω @ A(s, theta_complex)) ds
        n_steps = 50
        dt = T / n_steps
        B = 0.0 + 0.0j
        
        for step in range(n_steps):
            s = (step + 0.5) * dt  # Midpoint rule
            
            # Compute var_sigma(s)
            var_sigma_s = self._compute_var_sigma(s, m, sigma)
            
            # Compute e^{ms}
            ems = np.array(jlinalg.expm(s * m))
            
            # Compute A(s, theta_complex)
            theta_var_s = theta_complex @ var_sigma_s
            denom_s = I_d - 2 * theta_var_s
            m2_s = np.linalg.inv(denom_s)
            A_s = ems.T @ m2_s @ theta_complex @ ems
            
            # Add contribution to B
            B += np.trace(omega @ A_s) * dt
        
        return A, B
    
    def compute_A_B_old(
        self,
        T: float,
        theta: np.ndarray,
        m: np.ndarray,
        omega: np.ndarray,
        sigma: np.ndarray,
        denormalize: bool = True
    ) -> Tuple[np.ndarray, complex]:
        """
        Compute A(T, Θ) and B(T, Θ) using neural network.
        
        Automatically handles both real and complex theta based on mode.
        
        For REAL mode with complex theta query:
            Uses analytic extension (mathematically exact)
        
        Args:
            T: Time to maturity
            theta: Θ matrix (d x d, can be real or complex)
            m: M matrix (d x d, real)
            omega: Ω matrix (d x d, real symmetric)
            sigma: Σ matrix (d x d, real symmetric)
            denormalize: Whether to denormalize outputs
        
        Returns:
            A: A(T, Θ) matrix (d x d)
            B: B(T, Θ) scalar
        """
        d = self.dim
        theta_np = np.asarray(theta)
        is_complex_query = np.iscomplexobj(theta_np)
        
        if self.mode == "real":
            if is_complex_query:
                # REAL mode with complex query: use analytic extension
                return self._compute_A_B_complex_from_real(
                    T, theta_np, np.asarray(m), np.asarray(omega), np.asarray(sigma)
                )
            else:
                # REAL mode with real query: direct NN inference
                inputs = self._prepare_inputs_real(T, theta_np, m, omega, sigma)
                A_flat, B_flat = self._forward(self.params, inputs)
                
                A_flat = np.array(A_flat)
                B_flat = np.array(B_flat)
                
                if denormalize and self.norm_stats is not None:
                    A_flat = A_flat * self.norm_stats['A_std'] + self.norm_stats['A_mean']
                    B_flat = B_flat * self.norm_stats['B_std'] + self.norm_stats['B_mean']
                
                # Convert to matrix (real)
                A = upper_tri_to_matrix_np(A_flat[0], d)
                B = float(B_flat[0, 0])
                
                return A, B
        
        else:  # COMPLEX mode
            if not is_complex_query:
                # Convert real theta to complex (with zero imaginary part)
                theta_np = theta_np.astype(np.complex128)
            
            inputs = self._prepare_inputs_complex(T, theta_np, m, omega, sigma)
            A_flat, B_flat = self._forward(self.params, inputs)
            
            A_flat = np.array(A_flat)
            B_flat = np.array(B_flat)
            
            if denormalize and self.norm_stats is not None:
                A_flat = A_flat * self.norm_stats['A_std'] + self.norm_stats['A_mean']
                B_flat = B_flat * self.norm_stats['B_std'] + self.norm_stats['B_mean']
            
            # Convert to complex matrix
            A = complex_upper_tri_to_matrix(jnp.array(A_flat[0]), d)
            B = complex(float(B_flat[0, 0]), float(B_flat[0, 1]))
            
            return np.array(A), B
    
    def compute_A_B_batch(
        self,
        T: np.ndarray,
        theta: np.ndarray,
        m: np.ndarray,
        omega: np.ndarray,
        sigma: np.ndarray,
        denormalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Batch computation of A and B."""
        n_samples = len(T)
        d = self.dim
        
        # For now, loop over samples (can be optimized)
        A_list = []
        B_list = []
        
        for i in range(n_samples):
            A_i, B_i = self.compute_A_B(
                T[i], theta[i], m[i], omega[i], sigma[i], denormalize
            )
            A_list.append(A_i)
            B_list.append(B_i)
        
        return np.array(A_list), np.array(B_list)
    
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
        Compute Φ(T, Θ) = exp(Tr(A x0) + B) using neural network.
        
        Works for both real and complex theta.
        """
        A, B = self.compute_A_B(T, theta, m, omega, sigma)
        return np.exp(np.trace(A @ np.asarray(x0)) + B)
    
    def compute_characteristic_function_for_pricing(
        self,
        T: float,
        theta: np.ndarray,
        m: np.ndarray,
        omega: np.ndarray,
        sigma: np.ndarray,
        x0: np.ndarray
    ) -> complex:
        """
        Compute Φ(T, Θ) optimized for Fourier pricing.
        
        Same as compute_characteristic_function but with a clearer name
        for use in pricing code.
        """
        return self.compute_characteristic_function(T, theta, m, omega, sigma, x0)
    
    def compare_with_numerical(
        self,
        T: float,
        theta: np.ndarray,
        m: np.ndarray,
        omega: np.ndarray,
        sigma: np.ndarray,
        x0: np.ndarray
    ) -> Dict[str, Any]:
        """Compare neural network results with numerical implementation."""
        x0_np = np.asarray(x0)
        
        # Neural network computation
        t0 = time.time()
        A_nn, B_nn = self.compute_A_B(T, theta, m, omega, sigma)
        phi_nn = self.compute_characteristic_function(T, theta, m, omega, sigma, x0_np)
        time_nn = time.time() - t0
        
        # Numerical computation
        t0 = time.time()
        A_num, B_num = self.numerical_computer.compute_A_B(
            T, theta, m, omega, sigma, x0_np
        )
        phi_num = self.numerical_computer.compute_characteristic_function(
            T, theta, m, omega, sigma, x0_np
        )
        time_num = time.time() - t0
        
        # Compute errors
        error_A = np.linalg.norm(A_nn - A_num) / (np.linalg.norm(A_num) + 1e-10)
        error_B = np.abs(B_nn - B_num) / (np.abs(B_num) + 1e-10)
        error_phi = np.abs(phi_nn - phi_num) / (np.abs(phi_num) + 1e-10)
        
        return {
            'A_nn': A_nn,
            'A_num': A_num,
            'B_nn': B_nn,
            'B_num': B_num,
            'phi_nn': phi_nn,
            'phi_num': phi_num,
            'error_A': float(error_A),
            'error_B': float(error_B),
            'error_phi': float(error_phi),
            'time_nn': time_nn,
            'time_num': time_num,
            'speedup': time_num / time_nn if time_nn > 0 else float('inf')
        }
    
    @classmethod
    def from_saved_model(
        cls, 
        path: str,
        wishart_module_path: Optional[str] = None,
        normalization_stats_path: Optional[str] = None
    ) -> 'WishartPINNInference':
        """Load inference object from saved model."""
        model = WishartPINNModel.load(path)
        
        # Try to find normalization stats if not provided
        if normalization_stats_path is None:
            parent_dir = Path(path).parent.parent
            default_path = parent_dir / "normalization_stats.npz"
            if default_path.exists():
                normalization_stats_path = str(default_path)
        
        return cls(model, wishart_module_path=wishart_module_path, 
                   normalization_stats_path=normalization_stats_path)


# =============================================================================
# VALIDATION
# =============================================================================

def validate_model(
    inference: WishartPINNInference,
    config: WishartPINNConfig,
    n_test: int = 100,
    seed: int = 999
) -> Dict[str, Any]:
    """
    Validate model against numerical implementation.
    
    Tests with both real and complex theta based on mode.
    """
    from math import sqrt
    
    print(f"\nValidating on {n_test} test samples (mode={config.mode})...")
    
    d = config.dim
    results = {
        'errors_A': [],
        'errors_B': [],
        'errors_phi': [],
        'times_nn': [],
        'times_num': [],
        'comparisons': []
    }
    
    key = random.PRNGKey(seed)
    
    for i in tqdm(range(n_test), desc="Validating"):
        key, *subkeys = random.split(key, 15)
        
        # Sample T
        T = float(random.uniform(subkeys[0], minval=config.T_min, maxval=config.T_max))
        
        # Sample M
        m = np.zeros((d, d))
        for ii in range(d):
            m[ii, ii] = float(random.uniform(
                subkeys[1], minval=config.m_diag_min, maxval=config.m_diag_max
            ))
        
        # Sample Omega
        omega = np.zeros((d, d))
        for ii in range(d):
            omega[ii, ii] = float(random.uniform(
                subkeys[3], minval=config.omega_diag_min, maxval=config.omega_diag_max
            ))
        for ii in range(d):
            for jj in range(ii + 1, d):
                omega[ii, jj] = float(random.uniform(
                    subkeys[4], minval=config.omega_offdiag_min, maxval=config.omega_offdiag_max
                )) * sqrt(omega[ii, ii] * omega[jj, jj])
                omega[jj, ii] = omega[ii, jj]
        
        # Sample Sigma
        sigma = np.zeros((d, d))
        for ii in range(d):
            sigma[ii, ii] = float(random.uniform(
                subkeys[5], minval=config.sigma_diag_min, maxval=config.sigma_diag_max
            ))
        for ii in range(d):
            for jj in range(ii + 1, d):
                sigma[ii, jj] = float(random.uniform(
                    subkeys[6], minval=config.sigma_offdiag_min, maxval=config.sigma_offdiag_max
                )) * sqrt(sigma[ii, ii] * sigma[jj, jj])
                sigma[jj, ii] = sigma[ii, jj]
        
        # Sample theta based on mode
        if config.mode == "real":
            # Real theta
            theta = np.zeros((d, d))
            for ii in range(d):
                theta[ii, ii] = float(random.uniform(
                    subkeys[7], minval=config.theta_min, maxval=config.theta_max
                ))
            for ii in range(d):
                for jj in range(ii + 1, d):
                    theta[ii, jj] = float(random.uniform(
                        subkeys[8], minval=config.theta_offdiag_min, maxval=config.theta_offdiag_max
                    )) * sqrt(theta[ii, ii] * theta[jj, jj])
                    theta[jj, ii] = theta[ii, jj]
        else:
            # Complex theta = z * a3
            a3 = np.zeros((d, d))
            for ii in range(d):
                a3[ii, ii] = float(random.uniform(
                    subkeys[7], minval=config.theta_min, maxval=config.theta_max
                ))
            for ii in range(d):
                for jj in range(ii + 1, d):
                    a3[ii, jj] = float(random.uniform(
                        subkeys[8], minval=config.theta_offdiag_min, maxval=config.theta_offdiag_max
                    )) * sqrt(a3[ii, ii] * a3[jj, jj])
                    a3[jj, ii] = a3[ii, jj]
            
            ui = float(random.uniform(subkeys[9], minval=config.ui_min, maxval=config.ui_max))
            z = complex(config.ur, ui)
            theta = z * a3
        
        x0 = np.eye(d)
        
        try:
            comparison = inference.compare_with_numerical(T, theta, m, omega, sigma, x0)
            
            results['errors_A'].append(comparison['error_A'])
            results['errors_B'].append(comparison['error_B'])
            results['errors_phi'].append(comparison['error_phi'])
            results['times_nn'].append(comparison['time_nn'])
            results['times_num'].append(comparison['time_num'])
            results['comparisons'].append(comparison)
        except Exception as e:
            tqdm.write(f"Warning: Error in sample {i}: {e}")
            continue
    
    # Compute statistics
    if len(results['errors_A']) > 0:
        results['stats'] = {
            'error_A_mean': np.mean(results['errors_A']),
            'error_A_max': np.max(results['errors_A']),
            'error_A_std': np.std(results['errors_A']),
            'error_B_mean': np.mean(results['errors_B']),
            'error_B_max': np.max(results['errors_B']),
            'error_B_std': np.std(results['errors_B']),
            'error_phi_mean': np.mean(results['errors_phi']),
            'error_phi_max': np.max(results['errors_phi']),
            'error_phi_std': np.std(results['errors_phi']),
            'time_nn_mean': np.mean(results['times_nn']),
            'time_num_mean': np.mean(results['times_num']),
            'speedup_mean': np.mean(results['times_num']) / np.mean(results['times_nn'])
        }
        
        print("\n" + "=" * 60)
        print(f"VALIDATION RESULTS (mode={config.mode})")
        print("=" * 60)
        print(f"Number of test cases: {len(results['errors_A'])}")
        print(f"\nRelative Errors:")
        print(f"  A matrix:  mean = {results['stats']['error_A_mean']:.2e}, "
              f"max = {results['stats']['error_A_max']:.2e}")
        print(f"  B scalar:  mean = {results['stats']['error_B_mean']:.2e}, "
              f"max = {results['stats']['error_B_max']:.2e}")
        print(f"  Char func: mean = {results['stats']['error_phi_mean']:.2e}, "
              f"max = {results['stats']['error_phi_max']:.2e}")
        print(f"\nTiming:")
        print(f"  Neural Network: mean = {results['stats']['time_nn_mean']*1000:.3f} ms")
        print(f"  Numerical:      mean = {results['stats']['time_num_mean']*1000:.3f} ms")
        print(f"  Speedup:        {results['stats']['speedup_mean']:.1f}x")
        print("=" * 60)
    else:
        print("WARNING: No valid test samples!")
        results['stats'] = {}
    
    return results


def benchmark_throughput(
    inference: WishartPINNInference,
    config: WishartPINNConfig,
    n_samples: int = 1000
) -> Dict[str, float]:
    """Benchmark throughput of neural network inference."""
    from math import sqrt
    
    d = inference.dim
    
    key = random.PRNGKey(42)
    keys = random.split(key, 10)
    
    T = np.array(random.uniform(keys[0], shape=(n_samples,), 
                                minval=config.T_min, maxval=config.T_max))
    
    # Generate theta based on mode
    if config.mode == "real":
        theta = np.array([np.eye(d) * float(random.uniform(
            keys[1], minval=config.theta_min, maxval=config.theta_max
        )) for _ in range(n_samples)])
    else:
        a3 = np.array([np.eye(d) * float(random.uniform(
            keys[1], minval=config.theta_min, maxval=config.theta_max
        )) for _ in range(n_samples)])
        ui_vals = np.array(random.uniform(keys[2], shape=(n_samples,), 
                                          minval=config.ui_min, maxval=config.ui_max))
        theta = np.array([(config.ur + 1j * ui_vals[i]) * a3[i] for i in range(n_samples)])
    
    m = np.array([np.diag(np.array(random.uniform(keys[3], shape=(d,), 
                  minval=config.m_diag_min, maxval=config.m_diag_max))) for _ in range(n_samples)])
    omega = np.array([np.eye(d) * float(random.uniform(
        keys[4], minval=config.omega_diag_min, maxval=config.omega_diag_max
    )) for _ in range(n_samples)])
    sigma = np.array([np.eye(d) * float(random.uniform(
        keys[5], minval=config.sigma_diag_min, maxval=config.sigma_diag_max
    )) for _ in range(n_samples)])
    
    # Warmup
    _ = inference.compute_A_B_batch(T[:10], theta[:10], m[:10], omega[:10], sigma[:10])
    
    # Benchmark
    t0 = time.time()
    _ = inference.compute_A_B_batch(T, theta, m, omega, sigma)
    total_time = time.time() - t0
    
    throughput = n_samples / total_time
    latency = total_time / n_samples * 1000
    
    print(f"\nThroughput Benchmark ({n_samples} samples, mode={config.mode}):")
    print(f"  Total time: {total_time:.3f} s")
    print(f"  Throughput: {throughput:.0f} samples/s")
    print(f"  Latency:    {latency:.3f} ms/sample")
    
    return {
        'n_samples': n_samples,
        'total_time': total_time,
        'throughput': throughput,
        'latency_ms': latency
    }
