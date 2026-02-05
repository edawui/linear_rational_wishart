"""
Unit tests for LRW model sensitivities.

Tests Greeks calculations and parameter sensitivities.
"""

import pytest
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose

from linear_rational_wishart.models.interest_rate import LrwInterestRateBru
from linear_rational_wishart.models.interest_rate.lrw_sensitivities import LRWSensitivityAnalyzer
from linear_rational_wishart.pricing import LRWSwaptionPricer


class TestLRWSensitivities:
    """Test suite for LRW sensitivity calculations."""
    
    @pytest.fixture
    def sensitivity_setup(self):
        """Set up model and analyzer for sensitivity tests."""
        n = 2
        alpha = 0.05
        x0 = jnp.array([[0.12, -0.01], [-0.01, 0.005]])
        omega = jnp.array([[0.10, 0.002], [0.002, 0.0005]])
        m = jnp.array([[-0.4, 0.01], [0.02, -0.2]])
        sigma = jnp.array([[0.05, 0.02], [0.02, 0.047]])
        
        model = LrwInterestRateBru(n, alpha, x0, omega, m, sigma)
        
        u1 = jnp.array([[1, 0], [0, 0]])
        u2 = jnp.array([[0, 0], [0, 1]])
        model.SetU1(u1)
        model.SetU2(u2)
        
        # Set swaption
        model.SetOptionProperties(
            tenor=2.0,
            maturity=1.0,
            Delta=0.5,
            delta=0.5,
            K=0.0
        )
        K = model.ComputeSwapRate()
        model.SetOptionProperties(
            tenor=2.0,
            maturity=1.0,
            Delta=0.5,
            delta=0.5,
            K=K
        )
        
        analyzer = LRWSensitivityAnalyzer(model)
        
        return model, analyzer
    
    def test_delta_sensitivities(self, sensitivity_setup):
        """Test delta sensitivity calculations."""
        model, analyzer = sensitivity_setup
        
        # Compute deltas
        delta_results = analyzer.compute_all_sensitivities(
            compute_delta=True,
            compute_vega=False,
            compute_gamma=False,
            compute_parameter_sensi=False
        )
        
        # Should have ZC and swap hedging results
        assert any('ZC_Hedge' in key for key in delta_results)
        assert any('Swap_Hedge' in key for key in delta_results)
        
        # Hedging amounts should sum close to option value
        price = model.PriceOption()
        
        # ZC hedge portfolio value
        zc_hedge_value = 0
        for key, value in delta_results.items():
            if 'ZC_Hedge_amount' in key:
                date_key = key.replace('amount', 'date')
                if date_key in delta_results:
                    maturity = delta_results[date_key]
                    zc_hedge_value += value * model.Bond(maturity)
        
        # Should hedge a significant portion of the option
        assert abs(zc_hedge_value) > 0.1 * price
    
    def test_vega_sensitivities(self, sensitivity_setup):
        """Test vega sensitivity calculations."""
        model, analyzer = sensitivity_setup
        
        # Compute vegas
        vega_results = analyzer.compute_all_sensitivities(
            compute_delta=False,
            compute_vega=True,
            compute_gamma=False,
            compute_parameter_sensi=False
        )
        
        # Should have vega for each sigma component
        n = model.n
        for i in range(n):
            for j in range(n):
                key = f'Vega_Sigma_{i}_{j}'
                assert key in vega_results or f'Vega_Sigma_{j}_{i}' in vega_results
        
        # Total vega should be positive for ATM option
        total_vega = sum(v for k, v in vega_results.items() if 'Vega_Sigma' in k)
        assert total_vega > 0
    
    def test_parameter_sensitivities(self, sensitivity_setup):
        """Test parameter sensitivity calculations."""
        model, analyzer = sensitivity_setup
        
        # Compute parameter sensitivities
        param_results = analyzer.compute_all_sensitivities(
            compute_delta=False,
            compute_vega=False,
            compute_gamma=False,
            compute_parameter_sensi=True
        )
        
        # Should have alpha, omega, and M sensitivities
        assert 'Sensi_Alpha' in param_results
        assert any('Sensi_Omega' in key for key in param_results)
        assert any('Sensi_M' in key for key in param_results)
        
        # Alpha sensitivity should be negative (higher discount -> lower price)
        assert param_results['Sensi_Alpha'] < 0
    
    def test_gamma_sensitivities(self, sensitivity_setup):
        """Test gamma sensitivity calculations."""
        model, analyzer = sensitivity_setup
        
        # Compute gammas (this can be slow)
        gamma_results = analyzer.compute_all_sensitivities(
            compute_delta=False,
            compute_vega=False,
            compute_gamma=True,
            compute_parameter_sensi=False
        )
        
        # Should have bond and swap gammas
        assert any('Gamma_Bond' in key for key in gamma_results)
        assert any('Gamma_Swap' in key for key in gamma_results)
        
        # Gamma values should be finite
        for key, value in gamma_results.items():
            if 'Gamma' in key:
                assert np.isfinite(value)
    
    def test_hedging_portfolio(self, sensitivity_setup):
        """Test hedging portfolio construction."""
        model, analyzer = sensitivity_setup
        
        # Get ZC hedging portfolio
        zc_hedge = analyzer.compute_hedging_portfolio(hedge_type="zc")
        assert isinstance(zc_hedge, dict)
        assert len(zc_hedge) > 0
        
        # Get swap hedging portfolio
        swap_hedge = analyzer.compute_hedging_portfolio(hedge_type="swap")
        assert isinstance(swap_hedge, dict)
        assert len(swap_hedge) > 0
        
        # Invalid hedge type should raise error
        with pytest.raises(ValueError):
            analyzer.compute_hedging_portfolio(hedge_type="invalid")
    
    def test_sensitivity_consistency(self, sensitivity_setup):
        """Test consistency of sensitivity calculations."""
        model, analyzer = sensitivity_setup
        
        # Price bump test for vega
        base_price = model.PriceOption()
        
        # Bump sigma and reprice
        bump_size = 0.001
        original_sigma = model.sigma.copy()
        model.sigma = original_sigma + bump_size
        bumped_price = model.PriceOption()
        model.sigma = original_sigma
        
        # Numerical vega
        numerical_total_vega = (bumped_price - base_price) / bump_size
        
        # Analytical vega
        vega_results = analyzer.compute_all_sensitivities(
            compute_delta=False,
            compute_vega=True,
            compute_gamma=False,
            compute_parameter_sensi=False
        )
        analytical_total_vega = sum(v for k, v in vega_results.items() if 'Vega_Sigma' in k)
        
        # Should be reasonably close
        # Note: This is approximate due to matrix vs scalar bump
        assert abs(numerical_total_vega - analytical_total_vega) / analytical_total_vega < 0.5
    
    def test_full_sensitivity_report(self, sensitivity_setup):
        """Test full sensitivity report generation."""
        model, analyzer = sensitivity_setup
        
        # Generate full report
        all_results = analyzer.compute_all_sensitivities(
            compute_delta=True,
            compute_vega=True,
            compute_gamma=False,  # Skip for speed
            compute_parameter_sensi=True,
            print_intermediate=False
        )
        
        # Should have results from all categories
        assert len(all_results) > 10
        
        # All values should be finite
        for key, value in all_results.items():
            assert np.isfinite(value)
        
        # Check specific results exist
        assert 'Sensi_Alpha' in all_results
        assert any('Vega_Sigma' in key for key in all_results)
        assert any('ZC_Hedge' in key for key in all_results)


class TestSensitivityEdgeCases:
    """Test edge cases in sensitivity calculations."""
    
    def test_zero_volatility(self):
        """Test sensitivities with zero volatility."""
        n = 2
        alpha = 0.05
        x0 = jnp.array([[0.02, 0.0], [0.0, 0.02]])
        omega = jnp.array([[0.0001, 0.0], [0.0, 0.0001]])
        m = jnp.array([[-0.2, 0.0], [0.0, -0.2]])
        sigma = jnp.array([[0.0001, 0.0], [0.0, 0.0001]])  # Near zero
        
        model = LrwInterestRateBru(n, alpha, x0, omega, m, sigma)
        
        u1 = jnp.array([[1, 0], [0, 0]])
        u2 = jnp.array([[0, 0], [0, 1]])
        model.SetU1(u1)
        model.SetU2(u2)
        
        model.SetOptionProperties(1.0, 1.0, 0.5, 0.5, 0.05)
        
        analyzer = LRWSensitivityAnalyzer(model)
        
        # Should still compute without errors
        results = analyzer.compute_all_sensitivities(
            compute_delta=True,
            compute_vega=True,
            compute_gamma=False,
            compute_parameter_sensi=True
        )
        
        assert len(results) > 0
        assert all(np.isfinite(v) for v in results.values())
    
    def test_extreme_moneyness(self):
        """Test sensitivities for extreme moneyness."""
        n = 2
        alpha = 0.05
        x0 = jnp.array([[0.05, -0.01], [-0.01, 0.03]])
        omega = jnp.array([[0.08, 0.002], [0.002, 0.001]])
        m = jnp.array([[-0.3, 0.05], [0.05, -0.25]])
        sigma = jnp.array([[0.04, 0.015], [0.015, 0.035]])
        
        model = LrwInterestRateBru(n, alpha, x0, omega, m, sigma)
        
        u1 = jnp.array([[1, 0], [0, 0]])
        u2 = jnp.array([[0, 0], [0, 1]])
        model.SetU1(u1)
        model.SetU2(u2)
        
        # Get ATM
        model.SetOptionProperties(2.0, 1.0, 0.5, 0.5, 0.0)
        atm = model.ComputeSwapRate()
        
        # Test deep OTM
        model.SetOptionProperties(2.0, 1.0, 0.5, 0.5, atm * 2.0)
        analyzer = LRWSensitivityAnalyzer(model)
        
        otm_results = analyzer.compute_all_sensitivities(
            compute_delta=True,
            compute_vega=True,
            compute_gamma=False,
            compute_parameter_sensi=False
        )
        
        # Test deep ITM
        model.SetOptionProperties(2.0, 1.0, 0.5, 0.5, atm * 0.5)
        
        itm_results = analyzer.compute_all_sensitivities(
            compute_delta=True,
            compute_vega=True,
            compute_gamma=False,
            compute_parameter_sensi=False
        )
        
        # Both should compute successfully
        assert len(otm_results) > 0
        assert len(itm_results) > 0
        
        # Vega should be smaller for extreme moneyness
        otm_vega = sum(v for k, v in otm_results.items() if 'Vega_Sigma' in k)
        itm_vega = sum(v for k, v in itm_results.items() if 'Vega_Sigma' in k)
        
        # ATM vega
        model.SetOptionProperties(2.0, 1.0, 0.5, 0.5, atm)
        atm_results = analyzer.compute_all_sensitivities(
            compute_delta=False,
            compute_vega=True,
            compute_gamma=False,
            compute_parameter_sensi=False
        )
        atm_vega = sum(v for k, v in atm_results.items() if 'Vega_Sigma' in k)
        
        # ATM should have highest vega
        assert atm_vega > otm_vega
        assert atm_vega > itm_vega


class TestMonteCarloSensitivities:
    """Test sensitivity calculations with Monte Carlo pricing."""
    
    @pytest.fixture
    def mc_setup(self):
        """Set up model for Monte Carlo tests."""
        n = 2
        alpha = 0.05
        x0 = jnp.array([[0.12, -0.01], [-0.01, 0.005]])
        omega = jnp.array([[0.10, 0.002], [0.002, 0.0005]])
        m = jnp.array([[-0.4, 0.00], [0.00, -0.2]])
        sigma = jnp.array([[0.05, 0.02], [0.02, 0.047]])
        
        model = LrwInterestRateBru(n, alpha, x0, omega, m, sigma)
        
        u1 = jnp.array([[1, 0], [0, 0]])
        u2 = jnp.array([[0, 0], [0, 1]])
        model.SetU1(u1)
        model.SetU2(u2)
        
        return model
    
    def test_mc_pricing_consistency(self, mc_setup):
        """Test Monte Carlo pricing consistency."""
        model = mc_setup
        
        # Set ATM swaption
        model.SetOptionProperties(2.0, 1.0, 0.5, 0.5, 0.0)
        atm = model.ComputeSwapRate()
        model.SetOptionProperties(2.0, 1.0, 0.5, 0.5, atm)
        
        pricer = LRWSwaptionPricer(model)
        
        # FFT price
        fft_price = pricer.price_swaption(method="fft")
        
        # MC price with different schemas
        mc_prices = pricer.price_with_schemas(
            num_paths=10000,
            dt=0.125,
            schemas=["EULER_CORRECTED", "EULER_FLOORED"]
        )
        
        # All MC prices should be close to FFT
        for schema, price in mc_prices.items():
            if isinstance(price, float):
                rel_error = abs(price - fft_price) / fft_price
                assert rel_error < 0.1  # Within 10%
    
    def test_exposure_profile(self, mc_setup):
        """Test exposure profile calculation."""
        model = mc_setup
        pricer = LRWSwaptionPricer(model)
        
        # Set up swap
        model.SetOptionProperties(5.0, 0.0, 0.5, 0.5, 0.0)
        swap_rate = model.ComputeSwapRate()
        
        # Exposure dates
        exposure_dates = jnp.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Compute exposure
        mean_exposure, pfe_95 = pricer.compute_exposure_profile(
            exposure_dates=exposure_dates,
            fixed_rate=swap_rate,
            spread=0.0,
            floating_schedule=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
            fixed_schedule=[1.0, 2.0, 3.0, 4.0, 5.0],
            num_paths=1000,  # Small for testing
            dt=0.25
        )
        
        # Exposure should start at zero
        assert abs(mean_exposure[0]) < 1e-10
        
        # PFE should be >= mean exposure
        assert all(pfe_95 >= mean_exposure - 1e-10)
        
        # Exposure should end at zero (final payment)
        assert abs(mean_exposure[-1]) < 0.01


class TestComplexParameterSets:
    """Test with various complex parameter configurations."""
    
    def test_paper_v16_parameters(self):
        """Test with Paper V16 parameter set."""
        alpha = 0.024
        x0 = jnp.array([[0.015, -1.055e-3], [-1.055e-3, 8.25e-4]])
        omega = jnp.array([[0.110, -2.974e-3], [-2.974e-3, 1.377e-3]])
        m = jnp.array([[-6.642, 0.0], [0.0, -0.028]])
        sigma = jnp.array([[0.165, -0.041], [-0.041, 0.069]])
        
        model = LrwInterestRateBru(2, alpha, x0, omega, m, sigma)
        
        u1 = jnp.array([[1, 0], [0, 0]])
        u2 = jnp.array([[0, 0], [0, 1]])
        model.SetU1(u1)
        model.SetU2(u2)
        
        # Should satisfy Gindikin
        assert model.Wishart.check_gindikin()
        
        # Test 1Y x 1Y swaption
        model.SetOptionProperties(1.0, 1.0, 0.5, 1.0, 0.0)
        atm = model.ComputeSwapRate()
        model.SetOptionProperties(1.0, 1.0, 0.5, 1.0, atm)
        
        # Should price successfully
        price = model.PriceOption()
        assert price > 0
        
        # Test sensitivities
        analyzer = LRWSensitivityAnalyzer(model)
        results = analyzer.compute_all_sensitivities(
            compute_delta=True,
            compute_vega=True,
            compute_gamma=False,
            compute_parameter_sensi=True
        )
        
        assert len(results) > 0
        assert all(np.isfinite(v) for v in results.values())
    
    def test_multiple_strike_sensitivities(self):
        """Test sensitivities across multiple strikes."""
        n = 2
        alpha = 0.052
        x0 = jnp.array([[0.04, 0.02], [0.02, 0.03]])
        m = jnp.array([[-0.150, 0.06], [0.07, -0.120]])
        sigma = jnp.array([[0.04, 0.015], [0.015, 0.037]])
        omega = 4.0 * sigma @ sigma
        
        model = LrwInterestRateBru(n, alpha, x0, omega, m, sigma)
        
        u1 = jnp.array([[1, 0], [0, 1]])
        u2 = jnp.array([[0, 0], [0, 0]])
        model.SetU1(u1)
        model.SetU2(u2)
        
        # Get ATM
        model.SetOptionProperties(4.0, 2.0, 1.0, 1.0, 0.0)
        atm = model.ComputeSwapRate()
        
        # Test multiple strikes
        strikes = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
        
        vega_by_strike = []
        
        for K in strikes:
            model.SetOptionProperties(4.0, 2.0, 1.0, 1.0, K)
            
            try:
                _, vega_report = model.PriceOptionVega()
                total_vega = sum(v for k, v in vega_report.items() if 'Vega_Sigma' in k)
                vega_by_strike.append((K, total_vega))
            except:
                vega_by_strike.append((K, None))
        
        # Should have computed vega for most strikes
        valid_vegas = [v for k, v in vega_by_strike if v is not None]
        assert len(valid_vegas) >= 5
        
        # Vega should be highest near ATM
        atm_idx = np.argmin([abs(k - atm) for k, v in vega_by_strike if v is not None])
        max_vega_idx = np.argmax([v for k, v in vega_by_strike if v is not None])
        assert abs(atm_idx - max_vega_idx) <= 1  # Should be at or near ATM


class TestCalibrationIntegration:
    """Test sensitivity calculations with calibrated models."""
    
    def test_calibrated_model_sensitivities(self):
        """Test sensitivities after curve calibration."""
        from linear_rational_wishart.calibration.lrw_calibration import LRWCalibrator
        
        # Create model
        n = 2
        alpha = 0.05
        x0 = jnp.array([[0.03, -0.01], [-0.01, 0.02]])
        omega = jnp.array([[0.08, 0.002], [0.002, 0.001]])
        m = jnp.array([[-0.3, 0.05], [0.05, -0.25]])
        sigma = jnp.array([[0.04, 0.015], [0.015, 0.035]])
        
        model = LrwInterestRateBru(n, alpha, x0, omega, m, sigma)
        u1 = jnp.array([[1, 0], [0, 0]])
        u2 = jnp.array([[0, 0], [0, 1]])
        model.SetU1(u1)
        model.SetU2(u2)
        
        # Calibrate to flat curve
        calibrator = LRWCalibrator(model)
        market_dates = jnp.array([0.5, 1.0, 2.0, 5.0, 10.0])
        market_zc_values = jnp.exp(-0.05 * market_dates)
        
        calibrator.calibrate_to_curve(market_dates, market_zc_values)
        
        # Set swaption
        model.SetOptionProperties(2.0, 1.0, 0.5, 0.5, 0.0)
        atm = model.ComputeSwapRate()
        model.SetOptionProperties(2.0, 1.0, 0.5, 0.5, atm)
        
        # Compute sensitivities
        analyzer = LRWSensitivityAnalyzer(model)
        results = analyzer.compute_all_sensitivities(
            compute_delta=True,
            compute_vega=True,
            compute_gamma=False,
            compute_parameter_sensi=True
        )
        
        # Should have all sensitivities
        assert 'Sensi_Alpha' in results
        assert any('Vega_Sigma' in key for key in results)
        assert any('ZC_Hedge' in key for key in results)
        
        # Price should be reasonable
        price = model.PriceOption()
        assert price > 0
        assert price < 0.1  # Reasonable bound for 1Y x 2Y ATM


if __name__ == "__main__":
    pytest.main([__file__])
