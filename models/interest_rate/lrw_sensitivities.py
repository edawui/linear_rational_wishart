"""
Sensitivity calculations for Linear Rational Wishart (LRW) interest rate models.

This module provides comprehensive sensitivity analysis for LRW models including:
- Delta (price sensitivity to initial conditions)
- Vega (volatility sensitivity)
- Gamma (second-order sensitivities)
- Parameter sensitivities (alpha, omega, M)
"""

from typing import Dict, Tuple, Optional, Union, List
import jax.numpy as jnp
import jax
from jax import jit, grad, vmap
import numpy as np

from ...utils.jax_utils import ensure_jax_array, is_jax_available
from .lrw_model import LRWModel
# from .lrw_bru_model import LrwInterestRateBru
from ...sensitivities.delta_hedging import DeltaHedgingCalculator
from ...sensitivities.greeks import  GreeksCalculator
from ...sensitivities.vega_calculations import VegaCalculator



class LRWSensitivityAnalyzer:
    """
    Comprehensive sensitivity analyzer for LRW interest rate models.
    
    This class computes various Greeks and parameter sensitivities for
    swaptions and other interest rate derivatives under the LRW model.
    """
    
    def __init__(self, lrw_model: LRWModel):
        """
        Initialize the sensitivity analyzer.
        
        Parameters
        ----------
        lrw_model : LRWModel
            The LRW model instance to analyze
        """
        self.model = lrw_model
        self._use_jax = is_jax_available()
        self.delta_hedging_calculator= DeltaHedgingCalculator(self.model)
        self.greeks_calculator= GreeksCalculator(self.model)
        self.vega_calculator= VegaCalculator(self.model)


    def compute_all_sensitivities(
        self,
        compute_delta: bool = True,
        compute_vega: bool = True,
        compute_gamma: bool = False,
        compute_parameter_sensi: bool = True,
        print_intermediate: bool = False
    ) -> Dict[str, float]:
        """
        Compute all requested sensitivities.
        
        Parameters
        ----------
        compute_delta : bool, default=True
            Whether to compute delta sensitivities
        compute_vega : bool, default=True
            Whether to compute vega sensitivities
        compute_gamma : bool, default=False
            Whether to compute gamma sensitivities
        compute_parameter_sensi : bool, default=True
            Whether to compute parameter sensitivities (alpha, omega, M)
        print_intermediate : bool, default=False
            Whether to print intermediate results
            
        Returns
        -------
        Dict[str, float]
            Dictionary containing all computed sensitivities
        """
        results = {}
        
        if compute_delta:
            delta_results = self._compute_delta_sensitivities()
            results.update(delta_results)
            if print_intermediate:
                print("Delta sensitivities:", delta_results)
                
        if compute_vega:
            vega_results = self._compute_vega_sensitivities()
            results.update(vega_results)
            print("Vega sensitivities:", vega_results)

            if print_intermediate:
                print("Vega sensitivities:", vega_results)
                
        if compute_gamma:
            gamma_results = self._compute_gamma_sensitivities()
            results.update(gamma_results)
            if print_intermediate:
                print("Gamma sensitivities:", gamma_results)
                
        if compute_parameter_sensi:
            param_results = self._compute_parameter_sensitivities()
            results.update(param_results)
            if print_intermediate:
                print("Parameter sensitivities:", param_results)
                
        return results
    
    def _compute_delta_sensitivities(self) -> Dict[str, float]:
        """Compute delta sensitivities (ZC and swap hedging)."""
        results = {}
        # delta_hedge_calculator = DeltaHedgingCalculator(self.model)
        # Zero coupon hedging strategy
        zc_hedging = self.delta_hedging_calculator.compute_swaption_price_hedging()
        results.update(zc_hedging)
        
        # Swap hedging strategy
        swap_hedging = self.delta_hedging_calculator.compute_swaption_price_hedging_fix_float()
        results.update(swap_hedging)
        
        return results
    
    def _compute_vega_sensitivities(self) -> Dict[str, float]:
        """Compute vega sensitivities."""
        results = {}
        
        # Overall vega
        # vega_calculator = VegaCalculator(self.model)
        vega_sensi, vega_report=self.greeks_calculator.price_option_vega()
      
        # vega_sensi, vega_report = self.model.PriceOptionVega()
        results.update(vega_report)
        
        print("Component-wise vega for hedging")
        # Component-wise vega for hedging
        n = self.model.n
        for i in range(n):
            for j in range(n):
                print(f"Computing vega for components ({i}, {j})")
                # ZC vega hedging
                print("ZC vega hedging")
                # zc_vega = self.model.ComputeSwaptionPriceVegaHedging(i, j)
                zc_vega = self.vega_calculator.compute_swaption_price_vega_hedging(i, j)
                results.update(zc_vega)
                
                # Swap vega hedging
                print("Swap vega hedging")
                # swap_vega = self.model.ComputeSwaptionPriceHedgingFixFloatVega(i, j)
                swap_vega = self.vega_calculator.compute_swaption_price_hedging_fix_float_vega(i, j)
                results.update(swap_vega)
        # print("Finished component-wise vega calculations")
        # print(f"results={results}")
        return results
    
    def _compute_gamma_sensitivities(self) -> Dict[str, float]:
        """Compute gamma sensitivities."""
        results = {}
        
        # Get all relevant dates
        maturity = self.model.maturity
        tenor = self.model.tenor
        delta = self.model.delta
        
        all_dates = [maturity, maturity + tenor]
        for i in range(1, int(tenor / delta)):
            all_dates.append(maturity + i * delta)
        
        # Bond gamma sensitivities
        for i in range(len(all_dates)):
            for j in range(i, len(all_dates)):
                gamma_sensi, gamma_report = self.greeks_calculator.compute_gamma_bond(i, j) #ComputeGammaBond(i, j)
                results.update(gamma_report)
        
        # Swap gamma sensitivities
        leg_types = ["FIXED", "FLOATING"]
        for first_leg in leg_types:
            for second_leg in leg_types:
                gamma_sensi, gamma_report = self.greeks_calculator.compute_gamma_swap_cross(#ComputeGammaSwapCross(
                    first_leg, second_leg
                )
                results.update(gamma_report)
                
        return results
    
    def _compute_parameter_sensitivities(self) -> Dict[str, float]:
        """Compute sensitivities to model parameters."""
        results = {}
        
        # Alpha sensitivity
        print("Computing alpha sensitivity")
        alpha_sensi, alpha_report = self.greeks_calculator.price_option_sensi_alpha()#PriceOptionSensiAlpha()
        results.update(alpha_report)
        
        # Omega sensitivity
        print("Computing omega sensitivity")
        omega_sensi, omega_report = self.greeks_calculator.price_option_sensi_omega()#PriceOptionSensiOmega()
        results.update(omega_report)
        
        # M sensitivity
        #TODO

        print("Computing M sensitivity")
        run_M_sensitivities=False
        print("Todo:  Skipping M sensitivity, as it has bug. To be fixed ")
        if run_M_sensitivities:            
            m_sensi, m_report = self.greeks_calculator.price_option_sensi_m()#PriceOptionSensiM()
            results.update(m_report)
        print("Finished computing parameter sensitivities")
        return results
    
    def compute_hedging_portfolio(
        self,
        hedge_type: str = "zc"
    ) -> Dict[str, float]:
        """
        Compute optimal hedging portfolio.
        
        Parameters
        ----------
        hedge_type : str, default="zc"
            Type of hedging instruments ("zc" for zero coupon, "swap" for swaps)
            
        Returns
        -------
        Dict[str, float]
            Hedging ratios for each instrument
        """
        if hedge_type == "zc":
            return self.delta_hedging_calculator.compute_swaption_price_hedging()#ComputeSwaptionPriceHedging()
        elif hedge_type == "swap":
            return self.delta_hedging_calculator.compute_swaption_price_hedging_fix_float()#ComputeSwaptionPriceHedgingFixFloat()
        else:
            raise ValueError(f"Unknown hedge type: {hedge_type}")
        
        
