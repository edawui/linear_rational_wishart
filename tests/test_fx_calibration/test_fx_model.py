"""
Tests for basic FX model functionality.

This module tests the core functionality of the LRW FX model,
including model initialization, parameter setting, and basic calculations.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_almost_equal

from wishart_processes.models.fx import LrwFx
from wishart_processes.models.interest_rate import LrwInterestRateBru


class TestLrwFxModel:
    """Test basic LRW FX model functionality."""
    
    @pytest.fixture
    def model_parameters(self):
        """Standard model parameters for testing."""
        n = 2
        
        # Model parameters
        x0 = np.array([[4.99996419e+00, -3.18397238e-04],
                       [-3.18397238e-04, 4.88302000e-04]])
        
        omega = np.array([[2.09114424e+00, -8.04612684e-04],
                          [-8.04612684e-04, 1.92477100e-03]])
        
        m = np.array([[-0.20583617, 0.0],
                      [0.0, -0.02069993]])
        
        sigma = np.array([[0.15871937, 0.10552826],
                          [0.10552826, 0.02298161]])
        
        alpha_i = 0.05
        alpha_j = 0.04
        
        u_i = np.array([[1.0, 0.0],
                        [0.0, 0.0]])
        
        u_j = np.array([[0.0, 0.0],
                        [0.0, 1.0]])
        
        fx_spot = 1.0
        
        return {
            'n': n,
            'x0': x0,
            'omega': omega,
            'm': m,
            'sigma': sigma,
            'alpha_i': alpha_i,
            'alpha_j': alpha_j,
            'u_i': u_i,
            'u_j': u_j,
            'fx_spot': fx_spot
        }
    
    @pytest.fixture
    def fx_model(self, model_parameters):
        """Create FX model instance."""
        params = model_parameters
        return LrwFx(
            params['n'],
            params['x0'],
            params['omega'],
            params['m'],
            params['sigma'],
            params['alpha_i'],
            params['u_i'],
            params['alpha_j'],
            params['u_j'],
            params['fx_spot']
        )
    
    def test_model_initialization(self, fx_model, model_parameters):
        """Test model initialization."""
        assert fx_model._n == model_parameters['n']
        assert_array_almost_equal(fx_model._x0, model_parameters['x0'])
        assert_array_almost_equal(fx_model._omega, model_parameters['omega'])
        assert_array_almost_equal(fx_model._m, model_parameters['m'])
        assert_array_almost_equal(fx_model._sigma, model_parameters['sigma'])
        assert fx_model._alpha_i == model_parameters['alpha_i']
        assert fx_model._alpha_j == model_parameters['alpha_j']
        assert fx_model.fx_spot == model_parameters['fx_spot']
    
    def test_parameter_setters(self, fx_model):
        """Test parameter setter methods."""
        # Test alpha setters
        new_alpha_i = 0.06
        new_alpha_j = 0.045
        
        fx_model.set_alpha_i(new_alpha_i)
        fx_model.set_alpha_j(new_alpha_j)
        
        assert fx_model._alpha_i == new_alpha_i
        assert fx_model._alpha_j == new_alpha_j
        
        # Test U matrix setters
        new_u_i = np.array([[0.8, 0.0], [0.0, 0.2]])
        new_u_j = np.array([[0.1, 0.0], [0.0, 0.9]])
        
        fx_model.set_u_i(new_u_i)
        fx_model.set_u_j(new_u_j)
        
        assert_array_almost_equal(fx_model.u_i, new_u_i)
        assert_array_almost_equal(fx_model.u_j, new_u_j)
    
    def test_gindikin_condition(self, fx_model):
        """Test Gindikin condition check."""
        # The model should satisfy the Gindikin condition
        result = fx_model.lrw_currency_i.wishart.check_gindikin()
        assert result is True
    
    def test_bond_pricing_consistency(self, fx_model, model_parameters):
        """Test consistency between FX model and IR model bond pricing."""
        # Create equivalent IR model
        ir_model = LrwInterestRateBru(
            model_parameters['n'],
            model_parameters['alpha_i'],
            model_parameters['x0'],
            model_parameters['omega'],
            model_parameters['m'],
            model_parameters['sigma'],
            False
        )
        
        ir_model.set_u1(model_parameters['u_i'])
        ir_model.set_u2(model_parameters['u_j'])
        
        maturities = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        for maturity in maturities:
            # Bond prices should match
            bond_fx = fx_model.lrw_currency_i.bond(maturity)
            bond_ir = ir_model.bond(maturity)
            
            assert_almost_equal(bond_fx, bond_ir, decimal=10,
                              err_msg=f"Bond prices don't match for maturity {maturity}")
            
            # Interest rates should match
            r_fx = -np.log(bond_fx) / maturity
            r_ir = -np.log(bond_ir) / maturity
            
            assert_almost_equal(r_fx, r_ir, decimal=10,
                              err_msg=f"Interest rates don't match for maturity {maturity}")
    
    def test_fx_forward_calculation(self, fx_model):
        """Test FX forward calculation."""
        maturities = [0.5, 1.0, 2.0, 5.0]
        
        for maturity in maturities:
            fx_fwd = fx_model.compute_fx_fwd(maturity)
            
            # Forward should be positive
            assert fx_fwd > 0
            
            # For small maturities, forward should be close to spot
            if maturity < 0.1:
                assert_almost_equal(fx_fwd, fx_model.fx_spot, decimal=2)
    
    def test_option_properties_setting(self, fx_model):
        """Test setting option properties."""
        maturity = 2.0
        strike = 0.9
        
        fx_model.set_option_properties(maturity, strike)
        
        # Check that properties were set correctly
        assert fx_model.maturity == maturity
        assert fx_model.strike == strike
    
    def test_correlation_matrix_properties(self, fx_model):
        """Test properties of correlation matrices."""
        # X0 should be symmetric positive definite
        x0 = fx_model._x0
        assert_array_almost_equal(x0, x0.T)  # Symmetric
        eigenvalues = np.linalg.eigvals(x0)
        assert np.all(eigenvalues > 0)  # Positive definite
        
        # Same for omega
        omega = fx_model._omega
        assert_array_almost_equal(omega, omega.T)
        eigenvalues = np.linalg.eigvals(omega)
        assert np.all(eigenvalues > 0)
        
        # Same for sigma
        sigma = fx_model._sigma
        assert_array_almost_equal(sigma, sigma.T)
    
    def test_model_copy(self, fx_model):
        """Test model copying functionality."""
        # Make a copy
        model_copy = fx_model.copy()
        
        # Modify original
        fx_model._alpha_i = 0.1
        
        # Copy should be unchanged
        assert model_copy._alpha_i != fx_model._alpha_i
        
        # But parameters should initially match
        assert_array_almost_equal(model_copy._x0, fx_model._x0)
    
    def test_print_model(self, fx_model):
        """Test model printing functionality."""
        # Should not raise an error
        model_str = fx_model.print_model()
        
        # Should contain key information
        assert "alpha_i" in model_str
        assert "alpha_j" in model_str
        assert "X0" in model_str
        assert "Omega" in model_str
        assert "Sigma" in model_str


class TestModelParameterVariations:
    """Test model behavior with different parameter configurations."""
    
    def test_zero_correlation_case(self):
        """Test model with zero correlations."""
        n = 2
        
        # Diagonal matrices (zero correlation)
        x0 = np.array([[0.05, 0.0],
                       [0.0, 0.05]])
        
        omega = np.array([[0.02, 0.0],
                          [0.0, 0.02]])
        
        m = np.array([[-0.2, 0.0],
                      [0.0, -0.2]])
        
        sigma = np.array([[0.05, 0.0],
                          [0.0, 0.05]])
        
        alpha_i = 0.05
        alpha_j = 0.04
        u_i = np.array([[1.0, 0.0], [0.0, 0.0]])
        u_j = np.array([[0.0, 0.0], [0.0, 1.0]])
        fx_spot = 1.0
        
        model = LrwFx(n, x0, omega, m, sigma, alpha_i, u_i, alpha_j, u_j, fx_spot)
        
        # Model should initialize without errors
        assert model is not None
        
        # Compute correlation
        fx_vol_corr = model.compute_fx_vol_corr()
        
        # With diagonal matrices, correlation should be low
        assert abs(fx_vol_corr) < 0.1
    
    def test_high_correlation_case(self):
        """Test model with high correlations."""
        n = 2
        correl = 0.8
        
        # Create correlated matrices
        x0_diag = 0.05
        x0 = np.array([[x0_diag, 0.0],
                       [0.0, x0_diag]])
        x0[0, 1] = x0[1, 0] = correl * x0_diag
        
        omega_diag = 0.02
        omega = np.array([[omega_diag, 0.0],
                          [0.0, omega_diag]])
        omega[0, 1] = omega[1, 0] = correl * omega_diag
        
        m = np.array([[-0.2, 0.0],
                      [0.0, -0.2]])
        
        sigma_diag = 0.05
        sigma = np.array([[sigma_diag, 0.0],
                          [0.0, sigma_diag]])
        sigma[0, 1] = sigma[1, 0] = correl * sigma_diag
        
        alpha_i = 0.05
        alpha_j = 0.04
        u_i = np.array([[1.0, 0.0], [0.0, 0.0]])
        u_j = np.array([[0.0, 0.0], [0.0, 1.0]])
        fx_spot = 1.0
        
        model = LrwFx(n, x0, omega, m, sigma, alpha_i, u_i, alpha_j, u_j, fx_spot)
        
        # Model should handle high correlation
        assert model is not None
        
        # Verify matrices are still positive definite
        assert np.all(np.linalg.eigvals(model._x0) > 0)
        assert np.all(np.linalg.eigvals(model._omega) > 0)
    
    def test_asymmetric_parameters(self):
        """Test model with asymmetric parameters between currencies."""
        n = 2
        
        # Asymmetric diagonal values
        x0 = np.array([[0.1, 0.01],
                       [0.01, 0.02]])
        
        omega = np.array([[0.04, 0.002],
                          [0.002, 0.01]])
        
        m = np.array([[-0.3, 0.0],
                      [0.0, -0.1]])
        
        sigma = np.array([[0.08, 0.01],
                          [0.01, 0.03]])
        
        alpha_i = 0.07
        alpha_j = 0.02
        u_i = np.array([[1.0, 0.0], [0.0, 0.0]])
        u_j = np.array([[0.0, 0.0], [0.0, 1.0]])
        fx_spot = 1.2
        
        model = LrwFx(n, x0, omega, m, sigma, alpha_i, u_i, alpha_j, u_j, fx_spot)
        
        # Test forward calculation with asymmetric parameters
        fx_fwd = model.compute_fx_fwd(1.0)
        
        # Forward should reflect the interest rate differential
        expected_direction = np.exp((alpha_j - alpha_i) * 1.0) * fx_spot
        
        # Forward should be in the expected direction
        # (not exact due to stochastic effects)
        assert abs(fx_fwd - expected_direction) / expected_direction < 0.5
