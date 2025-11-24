"""
Unit tests for alpha curve calibration functionality.
"""

import numpy as np
import pytest
import jax.numpy as jnp

import sys
import os
from pathlib import Path


current_file = os.path.abspath(__file__)
project_root = current_file

# Go up until we find the wishart_processes directory
while os.path.basename(project_root) != "LinearRationalWishart_NewCode" and project_root != os.path.dirname(project_root):
    project_root = os.path.dirname(project_root)

if os.path.basename(project_root) != "LinearRationalWishart_NewCode":
    # Fallback to hardcoded path
    project_root = r"E:\OneDrive\Dropbox\LinearRationalWishart_Work\Code\ED\LinearRationalWishart\LinearRationalWishart_NewCode"

print(f"Using project root: {project_root}")
sys.path.insert(0, project_root)


from linear_rational_wishart.calibration import get_initial_alpha, InitialCurveAlpha
from linear_rational_wishart.calibration.interpolation import create_interpolator
from linear_rational_wishart.utils.jax_utils import is_jax_available


class TestGetInitialAlpha:
    """Test suite for get_initial_alpha function."""
    
    def test_basic_functionality(self):
        """Test basic alpha curve creation."""
        dates = np.array([0.25, 0.5, 1.0, 2.0])
        market = np.array([0.99, 0.98, 0.96, 0.92])
        model = np.array([0.995, 0.985, 0.965, 0.925])
        
        alpha_curve = get_initial_alpha(dates, market, model)
        
        assert isinstance(alpha_curve, InitialCurveAlpha)
        assert alpha_curve.interpolation == 'loglinear'
        assert alpha_curve.extrapolation == 'flat'
    
    def test_input_validation(self):
        """Test input validation."""
        dates = np.array([0.25, 0.5, 1.0])
        market = np.array([0.99, 0.98])  # Wrong length
        model = np.array([0.995, 0.985, 0.965])
        
        with pytest.raises(ValueError, match="same length"):
            get_initial_alpha(dates, market, model)
    
    def test_zero_model_values(self):
        """Test handling of zero model values."""
        dates = np.array([0.25, 0.5, 1.0])
        market = np.array([0.99, 0.98, 0.96])
        model = np.array([0.995, 0.0, 0.965])  # Contains zero
        
        with pytest.raises(ValueError, match="division by zero"):
            get_initial_alpha(dates, market, model)
    
    @pytest.mark.skipif(not is_jax_available(), reason="JAX not available")
    def test_jax_arrays(self):
        """Test with JAX arrays."""
        dates = jnp.array([0.25, 0.5, 1.0, 2.0])
        market = jnp.array([0.99, 0.98, 0.96, 0.92])
        model = jnp.array([0.995, 0.985, 0.965, 0.925])
        
        alpha_curve = get_initial_alpha(dates, market, model, use_jax=True)
        
        assert alpha_curve.is_jax_optimized
        assert isinstance(alpha_curve.alpha_dates, jnp.ndarray)


class TestInitialCurveAlpha:
    """Test suite for InitialCurveAlpha class."""
    
    def test_linear_interpolation(self):
        """Test linear interpolation."""
        dates = np.array([0.0, 1.0, 2.0])
        alphas = np.array([1.0, 1.1, 1.2])
        
        curve = InitialCurveAlpha(dates, alphas, interpolation='linear')
        
        # Test interpolation
        assert np.isclose(curve.get_alpha(0.5), 1.05)
        assert np.isclose(curve.get_alpha(1.5), 1.15)
        
        # Test extrapolation (flat)
        assert np.isclose(curve.get_alpha(-0.5), 1.0)
        assert np.isclose(curve.get_alpha(2.5), 1.2)
    
    def test_loglinear_interpolation(self):
        """Test log-linear interpolation."""
        dates = np.array([0.0, 1.0, 2.0])
        alphas = np.array([1.0, 2.0, 4.0])  # Exponential growth
        
        curve = InitialCurveAlpha(dates, alphas, interpolation='loglinear')
        
        # Test interpolation (should be geometric mean)
        assert np.isclose(curve.get_alpha(0.5), np.sqrt(2.0))
        assert np.isclose(curve.get_alpha(1.5), 2.0 * np.sqrt(2.0))
    
    def test_cubic_interpolation(self):
        """Test cubic interpolation."""
        dates = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        alphas = np.array([1.0, 1.1, 1.15, 1.18, 1.2])
        
        curve = InitialCurveAlpha(dates, alphas, interpolation='cubic', use_jax=False)
        
        # Test smoothness
        result = curve.get_alpha(np.linspace(0, 4, 100))
        assert len(result) == 100
        assert np.all(np.isfinite(result))
    
    def test_linear_extrapolation(self):
        """Test linear extrapolation."""
        dates = np.array([1.0, 2.0, 3.0])
        alphas = np.array([1.0, 1.1, 1.2])
        
        curve = InitialCurveAlpha(dates, alphas, interpolation='linear', extrapolation='linear')
        
        # Test linear extrapolation
        assert np.isclose(curve.get_alpha(0.0), 0.9)  # Extrapolate backwards
        assert np.isclose(curve.get_alpha(4.0), 1.3)  # Extrapolate forwards
    
    def test_array_input(self):
        """Test with array input for evaluation."""
        dates = np.array([0.0, 1.0, 2.0])
        alphas = np.array([1.0, 1.1, 1.2])
        
        curve = InitialCurveAlpha(dates, alphas)
        
        t_eval = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        result = curve.get_alpha(t_eval)
        
        assert len(result) == len(t_eval)
        assert np.allclose(result, [1.0, 1.05, 1.1, 1.15, 1.2])
    
    def test_input_validation(self):
        """Test input validation for InitialCurveAlpha."""
        # Test mismatched lengths
        with pytest.raises(ValueError, match="same length"):
            InitialCurveAlpha([1, 2], [1, 2, 3])
        
        # Test too few points
        with pytest.raises(ValueError, match="at least 2 elements"):
            InitialCurveAlpha([1], [1])
        
        # Test non-ascending dates
        with pytest.raises(ValueError, match="strictly ascending"):
            InitialCurveAlpha([1, 3, 2], [1, 2, 3])
        
        # Test negative values for loglinear
        with pytest.raises(ValueError, match="must be > 0"):
            InitialCurveAlpha([1, 2, 3], [1, -1, 2], interpolation='loglinear')
    
    @pytest.mark.skipif(not is_jax_available(), reason="JAX not available")
    def test_get_alpha_fast(self):
        """Test fast JAX-optimized version."""
        dates = jnp.array([0.0, 1.0, 2.0, 3.0])
        alphas = jnp.array([1.0, 1.1, 1.2, 1.3])
        
        curve = InitialCurveAlpha(dates, alphas, interpolation='linear', use_jax=True)
        
        # Compare regular and fast versions
        t_eval = jnp.linspace(0, 3, 100)
        result_regular = curve.get_alpha(t_eval)
        result_fast = curve.get_alpha_fast(t_eval)
        
        assert jnp.allclose(result_regular, result_fast)


class TestInterpolation:
    """Test suite for interpolation functionality."""
    
    def test_create_interpolator(self):
        """Test interpolator factory function."""
        x = np.array([0, 1, 2, 3])
        y = np.array([1, 2, 4, 8])
        
        # Test linear
        interp_linear = create_interpolator(x, y, 'linear')
        assert np.isclose(interp_linear(1.5), 3.0)
        
        # Test loglinear
        interp_log = create_interpolator(x, y, 'loglinear')
        assert np.isclose(interp_log(1.5), np.sqrt(2 * 4))
        
        # Test cubic
        interp_cubic = create_interpolator(x, y, 'cubic')
        result = interp_cubic(1.5)
        assert np.isfinite(result)
    
    def test_interpolator_validation(self):
        """Test interpolator input validation."""
        # Test mismatched lengths
        with pytest.raises(ValueError, match="same length"):
            create_interpolator([1, 2], [1, 2, 3])
        
        # Test too few points
        with pytest.raises(ValueError, match="at least 2"):
            create_interpolator([1], [1])
        
        # Test non-increasing x
        with pytest.raises(ValueError, match="strictly increasing"):
            create_interpolator([1, 3, 2], [1, 2, 3])
