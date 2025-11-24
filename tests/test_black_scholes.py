"""
Unit tests for Black-Scholes pricing model.
"""

import numpy as np
import pytest

from linear_rational_wishart.pricing import (
    black_scholes_price,
    black_scholes_price_forward,
    black_scholes_price_fx,
    black_scholes_delta,
    black_scholes_vega,
    black_scholes_vanna,
    black_scholes_greeks,
    implied_volatility_black_scholes,
    implied_volatility_smile
)
from linear_rational_wishart.utils import is_jax_available


class TestBlackScholesPricing:
    """Test suite for Black-Scholes option pricing."""
    
    def test_call_price(self):
        """Test call option pricing."""
        spot = 100.0
        strike = 105.0
        time_to_expiry = 1.0
        risk_free_rate = 0.05
        volatility = 0.2
        
        price = black_scholes_price(
            spot, strike, time_to_expiry, risk_free_rate, volatility, option_type='call'
        )
        
        # Expected price calculated independently
        expected_price = 6.0400  # Approximate
        assert np.isclose(price, expected_price, rtol=0.01)
    
    def test_put_price(self):
        """Test put option pricing."""
        spot = 100.0
        strike = 105.0
        time_to_expiry = 1.0
        risk_free_rate = 0.05
        volatility = 0.2
        
        price = black_scholes_price(
            spot, strike, time_to_expiry, risk_free_rate, volatility, option_type='put'
        )
        
        assert price > 0
        assert price < strike  # Put price should be less than strike
    
    def test_put_call_parity(self):
        """Test put-call parity in Black-Scholes model."""
        spot = 100.0
        strike = 105.0
        time_to_expiry = 1.0
        risk_free_rate = 0.05
        volatility = 0.2
        
        call_price = black_scholes_price(
            spot, strike, time_to_expiry, risk_free_rate, volatility, option_type='call'
        )
        put_price = black_scholes_price(
            spot, strike, time_to_expiry, risk_free_rate, volatility, option_type='put'
        )
        
        # Put-call parity: C - P = S - K * exp(-rT)
        parity_lhs = call_price - put_price
        parity_rhs = spot - strike * np.exp(-risk_free_rate * time_to_expiry)
        
        assert np.isclose(parity_lhs, parity_rhs, rtol=1e-10)
    
    def test_forward_pricing(self):
        """Test forward pricing consistency."""
        spot = 100.0
        strike = 105.0
        time_to_expiry = 1.0
        risk_free_rate = 0.05
        volatility = 0.2
        
        # Calculate forward price
        forward = spot * np.exp(risk_free_rate * time_to_expiry)
        
        # Price using spot
        price_spot = black_scholes_price(
            spot, strike, time_to_expiry, risk_free_rate, volatility
        )
        
        # Price using forward
        price_forward = black_scholes_price_forward(
            forward, strike, time_to_expiry, risk_free_rate, volatility
        )
        
        assert np.isclose(price_spot, price_forward, rtol=1e-10)
    
    def test_fx_pricing(self):
        """Test FX option pricing."""
        spot = 1.2500  # USD/EUR
        strike = 1.2600
        time_to_expiry = 0.25
        domestic_rate = 0.02  # USD rate
        foreign_rate = 0.01   # EUR rate
        volatility = 0.10
        
        price = black_scholes_price_fx(
            spot, strike, time_to_expiry, domestic_rate, foreign_rate, volatility
        )
        
        # Should be equivalent to regular pricing with dividend yield
        price_regular = black_scholes_price(
            spot, strike, time_to_expiry, domestic_rate, volatility, foreign_rate
        )
        
        assert np.isclose(price, price_regular, rtol=1e-10)
    
    def test_array_inputs(self):
        """Test with array inputs."""
        spots = np.array([90.0, 100.0, 110.0])
        strike = 100.0
        time_to_expiry = 1.0
        risk_free_rate = 0.05
        volatility = 0.2
        
        prices = black_scholes_price(
            spots, strike, time_to_expiry, risk_free_rate, volatility
        )
        
        assert len(prices) == len(spots)
        assert np.all(prices > 0)
        # Call prices should increase with spot
        assert np.all(np.diff(prices) > 0)
    
    @pytest.mark.skipif(not is_jax_available(), reason="JAX not available")
    def test_jax_implementation(self):
        """Test JAX implementation matches standard implementation."""
        spot = 100.0
        strike = 105.0
        time_to_expiry = 1.0
        risk_free_rate = 0.05
        volatility = 0.2
        
        price_standard = black_scholes_price(
            spot, strike, time_to_expiry, risk_free_rate, volatility, use_jax=False
        )
        price_jax = black_scholes_price(
            spot, strike, time_to_expiry, risk_free_rate, volatility, use_jax=True
        )
        
        assert np.isclose(price_standard, price_jax, rtol=1e-6)


class TestBlackScholesGreeks:
    """Test suite for Black-Scholes Greeks."""
    
    def test_delta_bounds(self):
        """Test that delta is within valid bounds."""
        spot = 100.0
        strikes = np.linspace(80, 120, 10)
        time_to_expiry = 1.0
        risk_free_rate = 0.05
        volatility = 0.2
        
        call_deltas = black_scholes_delta(
            spot, strikes, time_to_expiry, risk_free_rate, volatility, option_type='call'
        )
        put_deltas = black_scholes_delta(
            spot, strikes, time_to_expiry, risk_free_rate, volatility, option_type='put'
        )
        
        # Call delta should be in [0, 1]
        assert np.all(call_deltas >= 0)
        assert np.all(call_deltas <= 1)
        
        # Put delta should be in [-1, 0]
        assert np.all(put_deltas >= -1)
        assert np.all(put_deltas <= 0)
    
    def test_delta_monotonicity(self):
        """Test delta monotonicity with respect to spot."""
        spots = np.linspace(80, 120, 20)
        strike = 100.0
        time_to_expiry = 1.0
        risk_free_rate = 0.05
        volatility = 0.2
        
        call_deltas = black_scholes_delta(
            spots, strike, time_to_expiry, risk_free_rate, volatility, option_type='call'
        )
        
        # Call delta should increase with spot
        assert np.all(np.diff(call_deltas) >= 0)
    
    def test_vega_positive(self):
        """Test that vega is always positive."""
        spot = 100.0
        strikes = np.linspace(80, 120, 10)
        time_to_expiry = 1.0
        risk_free_rate = 0.05
        volatility = 0.2
        
        vegas = black_scholes_vega(
            spot, strikes, time_to_expiry, risk_free_rate, volatility
        )
        
        assert np.all(vegas > 0)
    
    def test_vega_maximum_atm(self):
        """Test that vega is maximum at-the-money."""
        spot = 100.0
        strikes = np.linspace(80, 120, 41)
        time_to_expiry = 1.0
        risk_free_rate = 0.05
        volatility = 0.2
        
        vegas = black_scholes_vega(
            spot, strikes, time_to_expiry, risk_free_rate, volatility
        )
        
        # Find index of ATM strike
        atm_idx = np.argmin(np.abs(strikes - spot))
        
        # Vega should be maximum at ATM
        assert np.argmax(vegas) == atm_idx
    
    def test_vanna_sign(self):
        """Test vanna sign properties."""
        spot = 100.0
        strike = 100.0  # ATM
        time_to_expiry = 1.0
        risk_free_rate = 0.05
        volatility = 0.2
        
        vanna = black_scholes_vanna(
            spot, strike, time_to_expiry, risk_free_rate, volatility
        )
        
        # ATM vanna should be close to zero
        assert np.abs(vanna) < 0.1
    
    def test_greeks_batch(self):
        """Test batch Greeks calculation."""
        spots = np.array([90.0, 100.0, 110.0])
        strike = 100.0
        time_to_expiry = 1.0
        risk_free_rate = 0.05
        volatility = 0.2
        
        results = black_scholes_greeks(
            spots, strike, time_to_expiry, risk_free_rate, volatility
        )
        
        prices, deltas, vegas, vannas = results
        
        assert len(prices) == len(spots)
        assert len(deltas) == len(spots)
        assert len(vegas) == len(spots)
        assert len(vannas) == len(spots)
        
        # Check consistency
        assert np.all(prices > 0)
        assert np.all(vegas > 0)


class TestImpliedVolatility:
    """Test suite for implied volatility calculations."""
    
    def test_implied_vol_recovery(self):
        """Test that implied vol recovers input volatility."""
        spot = 100.0
        strike = 105.0
        time_to_expiry = 1.0
        risk_free_rate = 0.05
        volatility_input = 0.2
        
        # Calculate option price
        price = black_scholes_price(
            spot, strike, time_to_expiry, risk_free_rate, volatility_input
        )
        
        # Recover implied volatility
        iv = implied_volatility_black_scholes(
            price, spot, strike, time_to_expiry, risk_free_rate
        )
        
        assert np.isclose(iv, volatility_input, rtol=1e-6)
    
    def test_implied_vol_methods(self):
        """Test different implied volatility methods."""
        spot = 100.0
        strike = 105.0
        time_to_expiry = 1.0
        risk_free_rate = 0.05
        volatility_input = 0.2
        
        price = black_scholes_price(
            spot, strike, time_to_expiry, risk_free_rate, volatility_input
        )
        
        # Test all methods
        iv_brent = implied_volatility_black_scholes(
            price, spot, strike, time_to_expiry, risk_free_rate, method='brent'
        )
        iv_newton = implied_volatility_black_scholes(
            price, spot, strike, time_to_expiry, risk_free_rate, method='newton'
        )
        iv_bisection = implied_volatility_black_scholes(
            price, spot, strike, time_to_expiry, risk_free_rate, method='bisection'
        )
        
        assert np.isclose(iv_brent, volatility_input, rtol=1e-6)
        assert np.isclose(iv_newton, volatility_input, rtol=1e-6)
        assert np.isclose(iv_bisection, volatility_input, rtol=1e-5)
    
    def test_implied_vol_extreme_cases(self):
        """Test implied volatility for extreme cases."""
        spot = 100.0
        strike = 150.0  # Deep OTM
        time_to_expiry = 0.1  # Short maturity
        risk_free_rate = 0.05
        
        # Very low price
        low_price = 0.001
        iv_low = implied_volatility_black_scholes(
            low_price, spot, strike, time_to_expiry, risk_free_rate
        )
        assert iv_low > 0
        
        # High price (but below intrinsic for call)
        intrinsic = max(spot - strike * np.exp(-risk_free_rate * time_to_expiry), 0)
        high_price = intrinsic + 5.0
        iv_high = implied_volatility_black_scholes(
            high_price, spot, strike, time_to_expiry, risk_free_rate
        )
        assert iv_high > 0
    
    def test_implied_vol_smile(self):
        """Test implied volatility smile calculation."""
        spot = 100.0
        strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
        time_to_expiry = 1.0
        risk_free_rate = 0.05
        
        # Generate prices with smile (higher IV for OTM options)
        base_vol = 0.2
        smile_vols = np.array([0.22, 0.21, 0.20, 0.21, 0.22])
        
        prices = np.array([
            black_scholes_price(spot, k, time_to_expiry, risk_free_rate, vol)
            for k, vol in zip(strikes, smile_vols)
        ])
        
        # Calculate implied volatility smile
        ivs = implied_volatility_smile(
            strikes, prices, spot, time_to_expiry, risk_free_rate
        )
        
        assert len(ivs) == len(strikes)
        assert np.all(ivs > 0)
        # Should recover input vols approximately
        assert np.allclose(ivs, smile_vols, rtol=1e-5)
    
    def test_implied_vol_arbitrage(self):
        """Test that arbitrage violations are detected."""
        spot = 100.0
        strike = 105.0
        time_to_expiry = 1.0
        risk_free_rate = 0.05
        
        # Price below intrinsic value
        intrinsic = max(spot - strike * np.exp(-risk_free_rate * time_to_expiry), 0)
        bad_price = intrinsic - 1.0
        
        with pytest.raises(ValueError, match="below intrinsic value"):
            implied_volatility_black_scholes(
                bad_price, spot, strike, time_to_expiry, risk_free_rate
            )


class TestSpecialCases:
    """Test suite for special cases and edge conditions."""
    
    def test_zero_volatility(self):
        """Test pricing with zero volatility."""
        spot = 100.0
        strike = 105.0
        time_to_expiry = 1.0
        risk_free_rate = 0.05
        volatility = 0.0
        
        call_price = black_scholes_price(
            spot, strike, time_to_expiry, risk_free_rate, volatility, option_type='call'
        )
        put_price = black_scholes_price(
            spot, strike, time_to_expiry, risk_free_rate, volatility, option_type='put'
        )
        
        # With zero vol, prices should equal intrinsic value
        forward = spot * np.exp(risk_free_rate * time_to_expiry)
        discount = np.exp(-risk_free_rate * time_to_expiry)
        
        expected_call = max(forward - strike, 0) * discount
        expected_put = max(strike - forward, 0) * discount
        
        assert np.isclose(call_price, expected_call, rtol=1e-10)
        assert np.isclose(put_price, expected_put, rtol=1e-10)
    
    def test_zero_time(self):
        """Test pricing at expiry."""
        spot = 100.0
        strike = 105.0
        time_to_expiry = 0.0
        risk_free_rate = 0.05
        volatility = 0.2
        
        call_price = black_scholes_price(
            spot, strike, time_to_expiry, risk_free_rate, volatility, option_type='call'
        )
        put_price = black_scholes_price(
            spot, strike, time_to_expiry, risk_free_rate, volatility, option_type='put'
        )
        
        # At expiry, should equal intrinsic value
        assert np.isclose(call_price, max(spot - strike, 0), rtol=1e-10)
        assert np.isclose(put_price, max(strike - spot, 0), rtol=1e-10)
    
    def test_deep_itm_otm(self):
        """Test deep in-the-money and out-of-the-money options."""
        spot = 100.0
        time_to_expiry = 1.0
        risk_free_rate = 0.05
        volatility = 0.2
        
        # Deep ITM call
        deep_itm_strike = 50.0
        itm_call = black_scholes_price(
            spot, deep_itm_strike, time_to_expiry, risk_free_rate, volatility, option_type='call'
        )
        # Should be approximately S - K*exp(-rT)
        expected_itm = spot - deep_itm_strike * np.exp(-risk_free_rate * time_to_expiry)
        assert np.isclose(itm_call, expected_itm, rtol=0.01)
        
        # Deep OTM call
        deep_otm_strike = 200.0
        otm_call = black_scholes_price(
            spot, deep_otm_strike, time_to_expiry, risk_free_rate, volatility, option_type='call'
        )
        # Should be very close to zero
        assert otm_call < 0.01
