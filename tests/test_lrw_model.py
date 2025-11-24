"""
Unit tests for LRW interest rate models.

Tests core functionality including bond pricing, option valuation,
and model consistency.
"""

import pytest
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose

from linear_rational_wishart.models.interest_rate import LrwInterestRateBru
from linear_rational_wishart.utils.jax_utils import is_jax_available


class TestLRWModel:
    """Test suite for LRW interest rate model."""
    
    @pytest.fixture
    def standard_model(self):
        """Create a standard LRW model for testing."""
        n = 2
        alpha = 0.05
        x0 = jnp.array([[0.02, 0.005], [0.005, 0.015]])
        omega = jnp.array([[0.06, 0.001], [0.001, 0.07]])
        m = jnp.array([[-0.29, 0.1], [0.2, -0.5]])
        sigma = jnp.array([[0.03, 0.1], [0.1, 0.1]])
        
        model = LrwInterestRateBru(n, alpha, x0, omega, m, sigma)
        
        u1 = jnp.array([[1, 0], [0, 0]])
        u2 = jnp.array([[0, 0], [0, 1]])
        model.SetU1(u1)
        model.SetU2(u2)
        
        return model
    
    def test_initialization(self, standard_model):
        """Test model initialization."""
        assert standard_model.n == 2
        assert standard_model.alpha == 0.05
        assert standard_model.x0.shape == (2, 2)
        assert_allclose(standard_model.GetShortRate(), 0.02, rtol=1e-6)
    
    def test_gindikin_condition(self, standard_model):
        """Test Gindikin condition check."""
        # Standard model should satisfy condition
        assert standard_model.Wishart.check_gindikin()
        
        # Create model that violates condition
        n = 2
        alpha = 0.05
        x0 = jnp.array([[0.02, 0.005], [0.005, 0.015]])
        sigma = jnp.array([[0.1, 0.05], [0.05, 0.1]])
        omega = 2.0 * sigma @ sigma  # Too small
        m = jnp.array([[-0.29, 0.1], [0.2, -0.5]])
        
        bad_model = LrwInterestRateBru(n, alpha, x0, omega, m, sigma)
        assert not bad_model.Wishart.check_gindikin()
    
    def test_bond_pricing(self, standard_model):
        """Test zero coupon bond pricing."""
        # Bond price should be 1 at t=0
        assert_allclose(standard_model.Bond(0.0), 1.0, rtol=1e-10)
        
        # Bond prices should decrease with maturity
        maturities = [0.5, 1.0, 2.0, 5.0]
        prices = [standard_model.Bond(t) for t in maturities]
        
        for i in range(len(prices) - 1):
            assert prices[i] > prices[i + 1]
        
        # Long maturity should converge
        p_long = standard_model.Bond(100.0)
        assert p_long > 0 and p_long < 0.01
    
    def test_short_rate_properties(self, standard_model):
        """Test short rate calculations."""
        r0 = standard_model.GetShortRate()
        r_inf = standard_model.GetShortRateInfinity()
        
        # Initial rate should be positive
        assert r0 > 0
        
        # Long-term rate should be positive
        assert r_inf > 0
        
        # Rates should be different (mean reversion)
        assert abs(r0 - r_inf) > 1e-6
    
    def test_spread_calculations(self, standard_model):
        """Test spread calculations."""
        spread0 = standard_model.GetSpread()
        spread_inf = standard_model.GetSpreadInfinity()
        
        # Spreads can be positive or negative
        assert isinstance(spread0, (float, jnp.ndarray))
        assert isinstance(spread_inf, (float, jnp.ndarray))
        
        # Test spread at different times
        spreads = [standard_model.Spread(t) for t in [0.5, 1.0, 2.0]]
        assert all(isinstance(s, (float, jnp.ndarray)) for s in spreads)
    
    def test_swap_rate_calculation(self, standard_model):
        """Test swap rate calculations."""
        standard_model.SetOptionProperties(
            tenor=2.0,
            maturity=1.0,
            Delta=0.5,
            delta=0.5,
            K=0.05
        )
        
        swap_rate = standard_model.ComputeSwapRate()
        
        # Swap rate should be positive and reasonable
        assert swap_rate > 0
        assert swap_rate < 0.2  # Less than 20%
        
        # ATM swaption should have reasonable price
        standard_model.SetOptionProperties(
            tenor=2.0,
            maturity=1.0,
            Delta=0.5,
            delta=0.5,
            K=swap_rate
        )
        
        price = standard_model.PriceOption()
        assert price > 0
        assert price < standard_model.Bond(1.0)  # Less than ZC bond
    
    def test_mean_calculations(self, standard_model):
        """Test Wishart process mean calculations."""
        t = 1.0
        u = jnp.array([[1, 0], [0, 1]])
        
        # Mean should be computed
        mean = standard_model.Wishart.ComputeMean(t, u)
        assert mean.shape == (2, 2)
        
        # Mean should be positive semi-definite
        eigenvals = jnp.linalg.eigvals(mean)
        assert all(eigenvals >= -1e-10)
    
    def test_mgf_computation(self, standard_model):
        """Test moment generating function."""
        standard_model.SetOptionProperties(
            tenor=2.0,
            maturity=1.0,
            Delta=0.5,
            delta=0.5,
            K=0.05
        )
        
        t = 1.0
        theta1 = jnp.array([[0.1, 0.02], [0.02, 0.08]])
        theta2 = jnp.zeros((2, 2))
        
        mgf = standard_model.MGF(t, theta1, theta2)
        
        # MGF should be positive
        assert mgf > 0
        
        # MGF at zero should be 1
        mgf_zero = standard_model.MGF(t, jnp.zeros((2, 2)), jnp.zeros((2, 2)))
        assert_allclose(mgf_zero, 1.0, rtol=1e-6)
    
    def test_option_pricing_consistency(self, standard_model):
        """Test option pricing consistency."""
        # Set up swaption
        tenor = 2.0
        maturity = 1.0
        standard_model.SetOptionProperties(tenor, maturity, 0.5, 0.5, 0.0)
        K = standard_model.ComputeSwapRate()
        standard_model.SetOptionProperties(tenor, maturity, 0.5, 0.5, K)
        
        # Price should be positive
        price = standard_model.PriceOption()
        assert price > 0
        
        # Implied vol should be computable
        iv = standard_model.ImpliedVol(price)
        assert iv > 0
        assert iv < 2.0  # Less than 200%
    
    @pytest.mark.skipif(not is_jax_available(), reason="JAX not available")
    def test_jax_compatibility(self, standard_model):
        """Test JAX functionality."""
        # Bond pricing should work with JAX
        t_jax = jnp.array(1.0)
        bond_price = standard_model.Bond(t_jax)
        assert isinstance(bond_price, jnp.ndarray)
        
        # Array operations should work
        maturities = jnp.array([0.5, 1.0, 2.0])
        prices = jnp.array([standard_model.Bond(float(t)) for t in maturities])
        assert prices.shape == (3,)


class TestLRWBruConfiguration:
    """Test LRW model with Brownian uncertainty configuration."""
    
    @pytest.fixture
    def bru_model(self):
        """Create LRW model with BRU configuration."""
        n = 2
        alpha = 0.05
        x0 = jnp.array([[0.12, -0.01], [-0.01, 0.005]])
        omega = jnp.array([[0.10, 0.002], [0.002, 0.0005]])
        m = jnp.array([[-0.4, 0.01], [0.02, -0.2]])
        sigma = jnp.array([[0.05, 0.02], [0.02, 0.047]])
        
        # Enable BRU configuration
        model = LrwInterestRateBru(n, alpha, x0, omega, m, sigma, isBruConfig=True)
        
        u1 = jnp.array([[1, 0], [0, 0]])
        u2 = jnp.array([[0, 0], [0, 1]])
        model.SetU1(u1)
        model.SetU2(u2)
        
        return model
    
    def test_bru_initialization(self, bru_model):
        """Test BRU model initialization."""
        assert hasattr(bru_model, 'b')
        assert bru_model.b is not None
        
    def test_bru_pricing(self, bru_model):
        """Test pricing with BRU configuration."""
        # Set swaption
        bru_model.SetOptionProperties(
            tenor=2.0,
            maturity=1.0,
            Delta=0.5,
            delta=0.5,
            K=0.05
        )
        
        # Should be able to price
        price = bru_model.PriceOption()
        assert price > 0
        
        # Compare with standard configuration
        standard_model = LrwInterestRateBru(
            bru_model.n,
            bru_model.alpha,
            bru_model.x0,
            bru_model.omega,
            bru_model.m,
            bru_model.sigma,
            isBruConfig=False
        )
        standard_model.SetU1(bru_model.u1)
        standard_model.SetU2(bru_model.u2)
        standard_model.SetOptionProperties(
            tenor=2.0,
            maturity=1.0,
            Delta=0.5,
            delta=0.5,
            K=0.05
        )
        
        standard_price = standard_model.PriceOption()
        
        # Prices should be different but same order of magnitude
        assert abs(price - standard_price) / standard_price < 1.0


class TestLRWUtilities:
    """Test utility functions from original Test1 and Test2."""
    
    def test_vec_operations(self):
        """Test Vec and VecInv operations."""
        from linear_rational_wishart.utils.local_functions import Vec, VecInv
        
        u = jnp.array([[1, 2], [4, 5]])
        vec_u = Vec(u)
        reconstructed = VecInv(vec_u)
        
        assert_allclose(u, reconstructed)
        assert vec_u.shape == (4,)
        
    def test_trace_operations(self):
        """Test trace operations."""
        from linear_rational_wishart.utils.local_functions import TrUV
        
        u = jnp.array([[1, 2], [3, 4]])
        v = jnp.array([[5, 6], [7, 8]])
        
        trace_uv = TrUV(u, v)
        expected = jnp.trace(u @ v)
        
        assert_allclose(trace_uv, expected)
    
    def test_model_components(self):
        """Test various model components from Test2."""
        n = 2
        alpha = 0.05
        x0 = jnp.array([[0.12, -0.01], [-0.01, 0.005]])
        omega = jnp.array([[0.10, 0.002], [0.002, 0.0005]])
        m = jnp.array([[-0.4, 0.01], [0.02, -0.2]])
        sigma = jnp.array([[0.05, 0.02], [0.02, 0.047]])
        
        lrw1 = LrwInterestRateBru(n, alpha, x0, omega, m, sigma)
        
        u1 = jnp.array([[1, 0], [0, 0]])
        u2 = jnp.array([[0, 0], [0, 1]])
        
        lrw1.SetU1(u1)
        lrw1.SetU2(u2)
        
        # Test bond and spread computations
        maturity = 1
        tenor = 2
        Delta = 0.5
        delta = 0.5
        K = 0.055
        
        lrw1.SetOptionProperties(tenor, maturity, Delta, delta, K)
        
        # Compute swap value components
        floating_leg = 1 - lrw1.Bond(tenor)
        fixed_leg = 0.0
        for i in range(1, int(tenor/delta) + 1):
            t1 = i * delta
            fixed_leg += delta * lrw1.Bond(t1)
        
        swap_rate = floating_leg / fixed_leg
        assert swap_rate > 0
        assert swap_rate < 0.2
        
        # Test B3A3 computation
        lrw1.ComputeB3A3()
        assert hasattr(lrw1, 'b3')
        assert hasattr(lrw1, 'a3')
        assert lrw1.b3 is not None
        assert lrw1.a3 is not None


if __name__ == "__main__":
    pytest.main([__file__])
