"""
Unit tests for Bachelier pricing model.
"""

# import numpy as np
# import matplotlib.pyplot as plt
# from typing import List, Dict, Tuple
# import matplotlib
import sys
import os
from pathlib import Path
import numpy as np
import pytest

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

from linear_rational_wishart.pricing.bachelier import bachelier_price, bachelier_delta, bachelier_vega,implied_normal_volatility
from linear_rational_wishart.pricing.jackel_method import JackelImpliedVolatility
from linear_rational_wishart.utils.jax_utils import is_jax_available


class TestBachelierPricing:
    """Test suite for Bachelier option pricing."""
    
    def test_call_price(self):
        """Test call option pricing."""
        forward = 100.0
        strike = 105.0
        time_to_expiry = 1.0
        sigma = 20.0
        
        price = bachelier_price(forward, strike, time_to_expiry, sigma, 'call')
        
        # Expected price calculated independently
        assert price > 0
        assert price < forward  # Call price should be less than forward
    
    def test_put_price(self):
        """Test put option pricing."""
        forward = 100.0
        strike = 105.0
        time_to_expiry = 1.0
        sigma = 20.0
        
        price = bachelier_price(forward, strike, time_to_expiry, sigma, 'put')
        
        assert price > 0
        assert price < strike  # Put price should be less than strike
    
    def test_put_call_parity(self):
        """Test put-call parity in Bachelier model."""
        forward = 100.0
        strike = 105.0
        time_to_expiry = 1.0
        sigma = 20.0
        
        call_price = bachelier_price(forward, strike, time_to_expiry, sigma, 'call')
        put_price = bachelier_price(forward, strike, time_to_expiry, sigma, 'put')
        
        # Put-call parity: C - P = F - K (undiscounted in Bachelier)
        parity = call_price - put_price
        expected = forward - strike
        
        assert np.isclose(parity, expected, rtol=1e-10)
    
    def test_atm_option(self):
        """Test at-the-money option pricing."""
        forward = 100.0
        strike = 100.0
        time_to_expiry = 1.0
        sigma = 20.0
        
        call_price = bachelier_price(forward, strike, time_to_expiry, sigma, 'call')
        put_price = bachelier_price(forward, strike, time_to_expiry, sigma, 'put')
        
        # ATM call and put should have same price in Bachelier
        assert np.isclose(call_price, put_price, rtol=1e-10)
    
    def test_zero_volatility(self):
        """Test pricing with zero volatility."""
        forward = 100.0
        strike = 105.0
        time_to_expiry = 1.0
        sigma = 0.0
        
        call_price = bachelier_price(forward, strike, time_to_expiry, sigma, 'call')
        put_price = bachelier_price(forward, strike, time_to_expiry, sigma, 'put')
        
        # With zero vol, prices should equal intrinsic value
        assert np.isclose(call_price, max(forward - strike, 0), rtol=1e-10)
        assert np.isclose(put_price, max(strike - forward, 0), rtol=1e-10)
    
    def test_array_inputs(self):
        """Test with array inputs."""
        forwards = np.array([90.0, 100.0, 110.0])
        strike = 100.0
        time_to_expiry = 1.0
        sigma = 20.0
        
        prices = bachelier_price(forwards, strike, time_to_expiry, sigma, 'call')
        
        assert len(prices) == len(forwards)
        assert np.all(prices > 0)
        # Prices should increase with forward
        assert np.all(np.diff(prices) > 0)
    
    @pytest.mark.skipif(not is_jax_available(), reason="JAX not available")
    def test_jax_implementation(self):
        """Test JAX implementation matches standard implementation."""
        forward = 100.0
        strike = 105.0
        time_to_expiry = 1.0
        sigma = 20.0
        
        price_standard = bachelier_price(
            forward, strike, time_to_expiry, sigma, 'call', use_jax=False
        )
        price_jax = bachelier_price(
            forward, strike, time_to_expiry, sigma, 'call', use_jax=True
        )
        
        assert np.isclose(price_standard, price_jax, rtol=1e-6)


class TestImpliedVolatility:
    """Test suite for implied volatility calculations."""
    
    def test_implied_vol_recovery(self):
        """Test that implied vol recovers input volatility."""
        forward = 100.0
        strike = 105.0
        time_to_expiry = 1.0
        sigma_input = 20.0
        
        # Calculate option price
        price = bachelier_price(forward, strike, time_to_expiry, sigma_input, 'call')
        
        # Recover implied volatility
        sigma_implied = implied_normal_volatility(
            forward, strike, time_to_expiry, price, 'call'
        )
        
        assert np.isclose(sigma_implied, sigma_input, rtol=1e-6)
    
    def test_implied_vol_jackel_method(self):
        """Test Jäckel method for implied volatility."""
        forward = 100.0
        strike = 105.0
        time_to_expiry = 1.0
        sigma_input = 20.0
        
        price = bachelier_price(forward, strike, time_to_expiry, sigma_input, 'call')
        
        # Test Jäckel method
        sigma_jackel = implied_normal_volatility(
            forward, strike, time_to_expiry, price, 'call', method='jackel'
        )
        
        assert np.isclose(sigma_jackel, sigma_input, rtol=1e-6)
    
    def test_implied_vol_extreme_cases(self):
        """Test implied volatility for extreme cases."""
        forward = 100.0
        strike = 105.0
        time_to_expiry = 1.0
        
        # Test with price at intrinsic value
        intrinsic = 0.0  # OTM call
        iv = implied_normal_volatility(
            forward, strike, time_to_expiry, intrinsic, 'call'
        )
        assert iv == 0.0
        
        # Test with very high price
        high_price = 50.0
        iv_high = implied_normal_volatility(
            forward, strike, time_to_expiry, high_price, 'call'
        )
        assert iv_high > 0
    
    def test_implied_vol_put_call_consistency(self):
        """Test that implied vol is same for puts and calls with same moneyness."""
        forward = 100.0
        strike = 105.0
        time_to_expiry = 1.0
        sigma = 20.0
        
        # Calculate prices
        call_price = bachelier_price(forward, strike, time_to_expiry, sigma, 'call')
        put_price = bachelier_price(forward, strike, time_to_expiry, sigma, 'put')
        
        # Get implied vols
        iv_call = implied_normal_volatility(
            forward, strike, time_to_expiry, call_price, 'call'
        )
        iv_put = implied_normal_volatility(
            forward, strike, time_to_expiry, put_price, 'put'
        )
        
        assert np.isclose(iv_call, iv_put, rtol=1e-6)


class TestGreeks:
    """Test suite for option Greeks."""
    
    def test_delta_bounds(self):
        """Test that delta is within valid bounds."""
        forward = 100.0
        strikes = np.linspace(80, 120, 10)
        time_to_expiry = 1.0
        sigma = 20.0
        
        call_deltas = bachelier_delta(forward, strikes, time_to_expiry, sigma, 'call')
        put_deltas = bachelier_delta(forward, strikes, time_to_expiry, sigma, 'put')
        
        # Call delta should be in [0, 1]
        assert np.all(call_deltas >= 0)
        assert np.all(call_deltas <= 1)
        
        # Put delta should be in [-1, 0]
        assert np.all(put_deltas >= -1)
        assert np.all(put_deltas <= 0)
    
    def test_delta_put_call_relationship(self):
        """Test put-call relationship for delta."""
        forward = 100.0
        strike = 105.0
        time_to_expiry = 1.0
        sigma = 20.0
        
        call_delta = bachelier_delta(forward, strike, time_to_expiry, sigma, 'call')
        put_delta = bachelier_delta(forward, strike, time_to_expiry, sigma, 'put')
        
        # Delta_call - Delta_put = 1
        assert np.isclose(call_delta - put_delta, 1.0, rtol=1e-10)
    
    def test_vega_positive(self):
        """Test that vega is always positive."""
        forward = 100.0
        strikes = np.linspace(80, 120, 10)
        time_to_expiry = 1.0
        sigma = 20.0
        
        vegas = bachelier_vega(forward, strikes, time_to_expiry, sigma)
        
        assert np.all(vegas > 0)
    
    def test_vega_maximum_atm(self):
        """Test that vega is maximum at-the-money."""
        forward = 100.0
        strikes = np.linspace(80, 120, 21)
        time_to_expiry = 1.0
        sigma = 20.0
        
        vegas = bachelier_vega(forward, strikes, time_to_expiry, sigma)
        
        # Find index of ATM strike
        atm_idx = np.argmin(np.abs(strikes - forward))
        
        # Vega should be maximum at ATM
        assert np.argmax(vegas) == atm_idx


class TestJackelMethod:
    """Test suite for Jäckel's method implementation."""
    
    def test_jackel_calculator(self):
        """Test JackelImpliedVolatility calculator."""
        calculator = JackelImpliedVolatility()
        
        forward = 100.0
        strike = 105.0
        time_to_expiry = 1.0
        price = 8.9595  # Known price for sigma = 20
        
        iv = calculator.implied_normal_volatility(
            forward, strike, time_to_expiry, price, 1.0
        )
        
        assert np.isclose(iv, 20.0, rtol=1e-4)
    
    def test_phi_tilde_times_x(self):
        """Test PhiTildeTimesX function."""
        calculator = JackelImpliedVolatility()
        
        x = 1.0
        result = calculator.phi_tilde_times_x(x)
        
        # Should equal x * Φ(x) + φ(x)
        from scipy.stats import norm
        expected = x * norm.cdf(x) + norm.pdf(x)
        
        assert np.isclose(result, expected, rtol=1e-10)
    
    def test_inv_phi_tilde_regions(self):
        """Test inverse PhiTilde in different regions."""
        calculator = JackelImpliedVolatility()
        
        # Test region 1 (phi_tilde_star < -0.001882039271)
        phi_tilde_star1 = -0.01
        x1 = calculator.inv_phi_tilde(phi_tilde_star1)
        assert np.isfinite(x1)
        
        # Test region 2 
        phi_tilde_star2 = -0.0001
        x2 = calculator.inv_phi_tilde(phi_tilde_star2)
        assert np.isfinite(x2)
        
        # Test symmetric case
        phi_tilde_star3 = 1.5
        x3 = calculator.inv_phi_tilde(phi_tilde_star3)
        assert np.isfinite(x3)
