"""
Sensitivity calculations for Linear Rational Wishart (LRW) interest rate models.

This module provides comprehensive sensitivity analysis for LRW models including:
- Delta (price sensitivity to initial conditions)
- Vega (volatility sensitivity)
- Gamma (second-order sensitivities)
- Parameter sensitivities (alpha, omega, M)
"""

from typing import Dict, Tuple, Optional, Union, List, overload, Literal
from unittest import result
import jax.numpy as jnp
import jax
from jax import jit, grad, vmap
import numpy as np

from ...utils.jax_utils import ensure_jax_array, is_jax_available
from .lrw_model import LRWModel
from ...config.constants import NMAX, UR
from ...sensitivities.delta_hedging import DeltaHedgingCalculator
from ...sensitivities.greeks import GreeksCalculator
from ...sensitivities.vega_calculations import VegaCalculator
from ...sensitivities.results import (
    DeltaHedgingResult,
    VegaHedgingResult,
    GammaResult,
    MatrixResult,
    AlphaSensitivityResult,
    SensitivityReport,
    SensitivityLogger,
    InstrumentType,
    GreekType,
)


class LRWSensitivityAnalyzer:
    """
    Comprehensive sensitivity analyzer for LRW interest rate models.
    
    This class computes various Greeks and parameter sensitivities for
    swaptions and other interest rate derivatives under the LRW model.
    """
    
    def __init__(self, lrw_model: LRWModel, enable_logging: bool = False):
        """
        Initialize the sensitivity analyzer.
        
        Parameters
        ----------
        lrw_model : LRWModel
            The LRW model instance to analyze
        enable_logging : bool
            Whether to enable logging (default: False)
        """
        self.model = lrw_model
        self._use_jax = is_jax_available()
        self.logger = SensitivityLogger("lrw_analyzer") if enable_logging else None
        
        # Initialize sub-calculators (with logging disabled by default to match original behavior)
        self.delta_hedging_calculator = DeltaHedgingCalculator(self.model, enable_logging=enable_logging)
        self.greeks_calculator = GreeksCalculator(self.model, enable_logging=enable_logging)
        self.vega_calculator = VegaCalculator(self.model, enable_logging=enable_logging)

    # =========================================================================
    # Main entry point - backward compatible
    # =========================================================================

    @overload
    def compute_all_sensitivities(
        self,
        compute_delta: bool = ...,
        compute_vega: bool = ...,
        compute_gamma: bool = ...,
        compute_parameter_sensi: bool = ...,
        print_intermediate: bool = ...,
        return_structured: Literal[False] = ...
    ) -> Dict[str, float]: ...  # <-- ONLY ellipsis, no code!

    @overload
    def compute_all_sensitivities(
        self,
        compute_delta: bool = ...,
        compute_vega: bool = ...,
        compute_gamma: bool = ...,
        compute_parameter_sensi: bool = ...,
        print_intermediate: bool = ...,
        return_structured: Literal[True] = ...
    ) -> SensitivityReport: ...  # <-- ONLY ellipsis, no code!

   
    def _compute_all_sensitivities_legacy(
        self,
        compute_delta: bool = True,
        compute_vega: bool = True,
        compute_gamma: bool = False,
        compute_parameter_sensi: bool = True,
        print_intermediate: bool = False,
        ) -> Dict[str, float]:

        results = {}
        if compute_delta:
            delta_results = self._compute_delta_sensitivities()
            results.update(delta_results)
            if print_intermediate:
                print("Delta sensitivities:", delta_results)
            # print("Delta sensitivities:", delta_results)
                
        if compute_vega:
            vega_results = self._compute_vega_sensitivities()
            results.update(vega_results)
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
            print("Parameter sensitivities:", param_results)
                
        return results
    
   
    def _compute_all_sensitivities_structured(self,compute_delta: bool = True,
                                    compute_vega: bool = True,
                                    compute_gamma: bool = False,
                                    compute_parameter_sensi: bool = True,
                                    print_intermediate: bool = False,
                                    )-> SensitivityReport:
        print("\nComputing sensitivities...")
 
        report = SensitivityReport(
        strike=self.model.swaption_config.strike,
        maturity=self.model.swaption_config.maturity,
        tenor=self.model.swaption_config.tenor
            )
        if compute_delta:
            # Delta hedging (ZC and SWAP)
            print("  - Computing Delta (ZC)...")
            delta_zc = self.compute_delta_hedging(instrument_type="ZC")
            report.add_delta(delta_zc)
    
            print("  - Computing Delta (SWAP)...")
            delta_swap = self.compute_delta_hedging(instrument_type="SWAP")
            report.add_delta(delta_swap)
    
        if compute_vega:
            # Vega matrix
            print("  - Computing Vega matrix...")
            vega_matrix = self.compute_vega_matrix()
            report.add_matrix(vega_matrix)
    
            # Vega hedging for each component
            n = self.model.n
            for i in range(n):
                for j in range(n):
                    print(f"  - Computing Vega hedging ({i},{j})...")
                    vega_hedge = self.compute_vega_hedging(i, j, instrument_type="ZC")
                    report.add_vega(vega_hedge)
    
        if compute_gamma:
            # Gamma (bond and swap cross)
            print("  - Computing Gamma (Bond)...")
            gamma_bond = self.compute_gamma(0, 1, instrument_type="BOND")
            report.add_gamma(gamma_bond)
    
            print("  - Computing Gamma (Swap Cross)...")
            gamma_swap = self.compute_gamma_swap_cross("FIXED", "FLOATING")
            report.add_gamma(gamma_swap)
        
        if compute_parameter_sensi:
            # # M sensitivity
            # print("  - Computing M sensitivity...")
            # m_matrix = self.compute_m_sensitivity()
            # report.add_matrix(m_matrix)

            # Alpha sensitivity
            print("  - Computing Alpha sensitivity...")
            alpha_result = self.compute_alpha_sensitivity()
            report.set_alpha(alpha_result)
    
            # Omega sensitivity
            print("  - Computing Omega sensitivity...")
            omega_matrix = self.compute_omega_sensitivity()
            report.add_matrix(omega_matrix)
    
            # M sensitivity
            print("  - Computing M sensitivity...")
            m_matrix = self.compute_m_sensitivity()
            report.add_matrix(m_matrix)
        return report

      
    def compute_all_sensitivities(
        self,
        compute_delta: bool = True,
        compute_vega: bool = True,
        compute_gamma: bool = False,
        compute_parameter_sensi: bool = True,
        print_intermediate: bool = False,
        return_structured: bool = True
        ) -> Union[Dict[str, float],SensitivityReport]:
        """
        Compute all requested sensitivities.
        
        This method returns legacy dictionary format for backward compatibility.
        
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
        if return_structured:
            print("Using NEW structured output format for sensitivities.")
            return self._compute_all_sensitivities_structured(compute_delta,
                                                    compute_vega,
                                                    compute_gamma,
                                                    compute_parameter_sensi,
                                                    print_intermediate)
        else:
            return self._compute_all_sensitivities_legacy(compute_delta,
                                                        compute_vega,
                                                        compute_gamma,
                                                        compute_parameter_sensi,
                                                        print_intermediate)
       
   
    # =========================================================================
    # Internal methods that return legacy dict format
    # =========================================================================
    
    def _compute_delta_sensitivities(self) -> Dict[str, float]:
        """Compute delta sensitivities (ZC and swap hedging)."""
        results = {}
        
        # Zero coupon hedging strategy - use legacy format
        zc_hedging = self.delta_hedging_calculator.compute_swaption_price_hedging(
            return_legacy_format=True
        )
        results.update(zc_hedging)
        
        # Swap hedging strategy - use legacy format
        swap_hedging = self.delta_hedging_calculator.compute_swaption_price_hedging_fix_float(
            return_legacy_format=True
        )
        results.update(swap_hedging)
        
        return results
    
    def _compute_vega_sensitivities(self) -> Dict[str, float]:
        """Compute vega sensitivities."""
        results = {}
        
        # Overall vega - use legacy format
        vega_sensi, vega_report = self.greeks_calculator.price_option_vega(
            return_legacy_format=True
        )
        results.update(vega_report)
        
        print("Component-wise vega for hedging")
        # Component-wise vega for hedging
        n = self.model.n
        for i in range(n):
            for j in range(n):
                print(f"Computing vega for components ({i}, {j})")
                
                # ZC vega hedging - use legacy format
                print("ZC vega hedging")
                zc_vega = self.vega_calculator.compute_swaption_price_vega_hedging(
                    i, j, return_legacy_format=True
                )
                results.update(zc_vega)
                
                # Swap vega hedging - use legacy format
                print("Swap vega hedging")
                swap_vega = self.vega_calculator.compute_swaption_price_hedging_fix_float_vega(
                    i, j, return_legacy_format=True
                )
                results.update(swap_vega)
        
        return results
    
    def _compute_gamma_sensitivities(self) -> Dict[str, float]:
        """Compute gamma sensitivities."""
        results = {}
        
        # Get all relevant dates
        maturity = self.model.maturity
        tenor = self.model.tenor
        delta = self.model.delta_float
        
        all_dates = [maturity, maturity + tenor]
        for i in range(1, int(tenor / delta)):
            all_dates.append(maturity + i * delta)
        
        # Bond gamma sensitivities - use legacy format
        for i in range(len(all_dates)):
            for j in range(i, len(all_dates)):
                gamma_sensi, gamma_report = self.greeks_calculator.compute_gamma_bond(
                    i, j, return_legacy_format=True
                )
                results.update(gamma_report)
        
        # Swap gamma sensitivities - use legacy format
        leg_types = ["FIXED", "FLOATING"]
        for first_leg in leg_types:
            for second_leg in leg_types:
                gamma_sensi, gamma_report = self.greeks_calculator.compute_gamma_swap_cross(
                    first_leg, second_leg, return_legacy_format=True
                )
                results.update(gamma_report)
                
        return results
    
    def _compute_parameter_sensitivities(self) -> Dict[str, float]:
        """Compute sensitivities to model parameters."""
        results = {}
        
        # Alpha sensitivity - use legacy format
        print("Computing alpha sensitivity")
        alpha_sensi, alpha_report = self.greeks_calculator.price_option_sensi_alpha(
            return_legacy_format=True
        )
        results.update(alpha_report)
        
        # Omega sensitivity - use legacy format
        print("Computing omega sensitivity")
        omega_sensi, omega_report = self.greeks_calculator.price_option_sensi_omega(
            return_legacy_format=True
        )
        results.update(omega_report)
        
        # M sensitivity - use legacy format
        print("Computing M sensitivity")
        run_M_sensitivities = True
        if run_M_sensitivities:
            m_sensi, m_report = self.greeks_calculator.price_option_sensi_m(
                return_legacy_format=True
            )
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
        if hedge_type.lower() == "zc":
            return self.delta_hedging_calculator.compute_swaption_price_hedging(
                return_legacy_format=True
            )
        elif hedge_type.lower() == "swap":
            return self.delta_hedging_calculator.compute_swaption_price_hedging_fix_float(
                return_legacy_format=True
            )
        else:
            raise ValueError(f"Unknown hedge type: {hedge_type}")

    # =========================================================================
    # NEW: Methods that return structured results
    # =========================================================================
  
   
    def compute_delta_hedging(
        self,
        instrument_type: str = "ZC",
        ur: float = UR, #0.5,
        nmax: int = NMAX ##1000
    ) -> DeltaHedgingResult:
        """
        Compute delta hedging strategy with structured result.
        
        Parameters
        ----------
        instrument_type : str
            "ZC" for zero-coupon bonds or "SWAP" for swaps
        ur : float
            Real part of integration contour
        nmax : int
            Maximum integration limit
            
        Returns
        -------
        DeltaHedgingResult
            Structured delta hedging result
        """
        if instrument_type.upper() == "ZC":
            return self.delta_hedging_calculator.compute_swaption_price_hedging(
                ur, nmax, return_legacy_format=False
            )
        elif instrument_type.upper() == "SWAP":
            return self.delta_hedging_calculator.compute_swaption_price_hedging_fix_float(
                ur, nmax, return_legacy_format=False
            )
        else:
            raise ValueError(f"Unknown instrument type: {instrument_type}. Use 'ZC' or 'SWAP'.")
    
    def compute_vega_matrix(
        self,
        ur: float = UR, #0.5,
        nmax: int = NMAX ##1000
    ) -> MatrixResult:
        """
        Compute the full Vega sensitivity matrix with structured result.
        
        Parameters
        ----------
        ur : float
            Real part of integration contour
        nmax : int
            Maximum integration limit
            
        Returns
        -------
        MatrixResult
            Structured Vega matrix result
        """
        vega_matrix, result = self.greeks_calculator.price_option_vega(
            ur, nmax, return_legacy_format=False
        )
        return result
    
    def compute_vega_hedging(
        self,
        i: int,
        j: int,
        instrument_type: str = "ZC",
        ur: float = UR, #0.5,
        nmax: int = NMAX ##1000
    ) -> VegaHedgingResult:
        """
        Compute Vega hedging for a specific sigma component with structured result.
        
        Parameters
        ----------
        i : int
            Row index of sigma component
        j : int
            Column index of sigma component
        instrument_type : str
            "ZC" for zero-coupon bonds or "SWAP" for swaps
        ur : float
            Real part of integration contour
        nmax : int
            Maximum integration limit
            
        Returns
        -------
        VegaHedgingResult
            Structured Vega hedging result
        """
        if instrument_type.upper() == "ZC":
            return self.vega_calculator.compute_swaption_price_vega_hedging(
                i, j, ur, nmax, return_legacy_format=False
            )
        elif instrument_type.upper() == "SWAP":
            return self.vega_calculator.compute_swaption_price_hedging_fix_float_vega(
                i, j, ur, nmax, return_legacy_format=False
            )
        else:
            raise ValueError(f"Unknown instrument type: {instrument_type}. Use 'ZC' or 'SWAP'.")
    
    def compute_gamma(
        self,
        id0: int,
        id1: int,
        instrument_type: str = "BOND",
        ur: float = UR, #0.5,
        nmax: int = NMAX ##1000
    ) -> GammaResult:
        """
        Compute Gamma for bond positions with structured result.
        
        Parameters
        ----------
        id0 : int
            First bond index
        id1 : int
            Second bond index
        instrument_type : str
            Instrument type (default: "BOND")
        ur : float
            Real part of integration contour
        nmax : int
            Maximum integration limit
            
        Returns
        -------
        GammaResult
            Structured Gamma result
        """
        gamma_value, result = self.greeks_calculator.compute_gamma_bond(
            id0, id1, ur, nmax, return_legacy_format=False
        )
        return result
    
    def compute_gamma_swap_cross(
        self,
        first_component: str,
        second_component: str,
        ur: float = UR, #0.5,
        nmax: int = NMAX ##1000
    ) -> GammaResult:
        """
        Compute cross-Gamma between swap components with structured result.
        
        Parameters
        ----------
        first_component : str
            First component ("FIXED" or "FLOATING")
        second_component : str
            Second component ("FIXED" or "FLOATING")
        ur : float
            Real part of integration contour
        nmax : int
            Maximum integration limit
            
        Returns
        -------
        GammaResult
            Structured Gamma result
        """
        gamma_value, result = self.greeks_calculator.compute_gamma_swap_cross(
            first_component, second_component, ur, nmax, return_legacy_format=False
        )
        return result
    
    def compute_alpha_sensitivity(
        self,
        ur: float = UR, #0.5,
        nmax: int = NMAX ##1000
    ) -> AlphaSensitivityResult:
        """
        Compute alpha sensitivity with structured result.
        
        Parameters
        ----------
        ur : float
            Real part of integration contour
        nmax : int
            Maximum integration limit
            
        Returns
        -------
        AlphaSensitivityResult
            Structured alpha sensitivity result
        """
        alpha_value, result = self.greeks_calculator.price_option_sensi_alpha(
            ur, nmax, return_legacy_format=False
        )
        return result
    
    def compute_omega_sensitivity(
        self,
        ur: float = UR, #0.5,
        nmax: int = NMAX ##1000
    ) -> MatrixResult:
        """
        Compute omega sensitivity matrix with structured result.
        
        Parameters
        ----------
        ur : float
            Real part of integration contour
        nmax : int
            Maximum integration limit
            
        Returns
        -------
        MatrixResult
            Structured omega sensitivity result
        """
        omega_matrix, result = self.greeks_calculator.price_option_sensi_omega(
            ur, nmax, return_legacy_format=False
        )
        return result
    
    def compute_m_sensitivity(
        self,
        ur: float = UR, #0.5,
        nmax: int = NMAX ##1000
    ) -> MatrixResult:
        """
        Compute M sensitivity matrix with structured result.
        
        Parameters
        ----------
        ur : float
            Real part of integration contour
        nmax : int
            Maximum integration limit
            
        Returns
        -------
        MatrixResult
            Structured M sensitivity result
        """
        m_matrix, result = self.greeks_calculator.price_option_sensi_m(
            ur, nmax, return_legacy_format=False
        )
        return result
    
    def generate_full_report(
        self,
        compute_delta: bool = True,
        compute_vega: bool = True,
        compute_gamma: bool = True,
        compute_parameter_sensi: bool = True,
        ur: float = UR, #0.5,
        nmax: int = NMAX ##1000
    ) -> SensitivityReport:
        """
        Generate a comprehensive sensitivity report with structured results.
        
        Parameters
        ----------
        compute_delta : bool
            Whether to compute delta hedging
        compute_vega : bool
            Whether to compute vega sensitivities
        compute_gamma : bool
            Whether to compute gamma sensitivities
        compute_parameter_sensi : bool
            Whether to compute parameter sensitivities
        ur : float
            Real part of integration contour
        nmax : int
            Maximum integration limit
            
        Returns
        -------
        SensitivityReport
            Comprehensive structured sensitivity report
        """
        report = SensitivityReport(
            strike=self.model.strike,
            maturity=self.model.maturity,
            tenor=self.model.tenor
        )
        
        if compute_delta:
            report.add_delta(self.compute_delta_hedging("ZC", ur, nmax))
            report.add_delta(self.compute_delta_hedging("SWAP", ur, nmax))
        
        if compute_vega:
            # Add matrix result
            report.add_matrix(self.compute_vega_matrix(ur, nmax))
            
            # Add individual hedging results
            n = self.model.n
            for i in range(n):
                for j in range(n):
                    report.add_vega(self.compute_vega_hedging(i, j, "ZC", ur, nmax))
        
        if compute_gamma:
            # Bond gammas
            report.add_gamma(self.compute_gamma(0, 1, "BOND", ur, nmax))
            
            # Swap cross gammas
            report.add_gamma(self.compute_gamma_swap_cross("FIXED", "FLOATING", ur, nmax))
        
        if compute_parameter_sensi:
            report.set_alpha(self.compute_alpha_sensitivity(ur, nmax))
            report.add_matrix(self.compute_omega_sensitivity(ur, nmax))
            report.add_matrix(self.compute_m_sensitivity(ur, nmax))
        
        return report
