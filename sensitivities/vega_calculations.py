# sensitivities/vega_calculations.py
"""Vega calculations for LRW model."""

import math
import cmath
from typing import Dict, Tuple, Union, List
import jax.numpy as jnp
import scipy.integrate as sp_i
import numpy as np

from ..core.expectations import compute_expectation_xy_vega, compute_expectation_xy
from ..utils.local_functions import tr_uv
from ..config.constants import EPSABS, EPSREL, NMAX, UR
from .results import (
    VegaHedgingResult,
    InstrumentType,
    SensitivityLogger,
)


class VegaCalculator:
    """Calculate Vega sensitivities for swaptions."""
    
    def __init__(self, model, enable_logging: bool = False):
        """Initialize Vega calculator."""
        self.model = model
        self.logger = SensitivityLogger("vega") if enable_logging else None
    
    def compute_swaption_price_vega_hedging(
        self, 
        i: int, 
        j: int, 
        ur: float = UR, #0.5
        nmax: int = NMAX,
        return_legacy_format: bool = False
    ) -> Union[VegaHedgingResult, Dict]:
        """
        Compute Vega hedging strategy using zero-coupon bonds.
        
        Parameters
        ----------
        i : int
            Row index of sigma component
        j : int
            Column index of sigma component
        ur : float
            Real part of integration contour (default: 0.5)
        nmax : int
            Maximum integration limit (default: NMAX)
        return_legacy_format : bool
            If True, return legacy dict format for backward compatibility
            
        Returns
        -------
        VegaHedgingResult or Dict
            Structured Vega hedging result (or legacy dict if return_legacy_format=True)
        """
        if self.logger:
            self.logger.log_calculation_start(
                "Vega (ZC)",
                self.model.strike,
                component=f"({i},{j})",
                maturity=self.model.maturity,
                tenor=self.model.tenor
            )
        
        self.model.compute_b3_a3()
        
        # Initialize legacy format dict (same keys as original)
        vega_zc_hedging_strategy = {}
        hedging_floating_leg = {}
        hedging_fixed_leg = {}
        
        # Initialize containers for structured result
        floating_leg = {}
        fixed_leg = {}
        
        # Floating leg - first maturity
        zero_coupon_tn1 = self.model.bond(self.model.maturity)
        var_rho_tn1 = self.compute_var_rho_t_vega(i, j, self.model.maturity, ur, nmax)
        
        # Store in both formats
        floating_leg[self.model.maturity] = var_rho_tn1
        hedging_floating_leg[f"{self.model.maturity}"] = var_rho_tn1
        vega_zc_hedging_strategy[f"VEGA_FOR_STRIKE:{self.model.strike}:{i}_{j}:ZC:FLOATINGLEG:{self.model.maturity}"] = float(var_rho_tn1)
        
        # Floating leg - second maturity
        zero_coupon_tn2 = self.model.bond(self.model.maturity + self.model.tenor)
        var_rho_tn2 = self.compute_var_rho_t_vega(i, j, self.model.maturity + self.model.tenor, ur, nmax)
        
        # Store in both formats
        floating_leg[self.model.maturity + self.model.tenor] = var_rho_tn2
        hedging_floating_leg[f"{self.model.maturity + self.model.tenor}"] = var_rho_tn2
        vega_zc_hedging_strategy[f"VEGA_FOR_STRIKE:{self.model.strike}:{i}_{j}:ZC:FLOATINGLEG:{self.model.maturity + self.model.tenor}"] = float(var_rho_tn2)
        
        # Fixed leg
        fixed_leg_value = 0
        for k in range(1, int(self.model.tenor / self.model.delta_fixed) + 1):
            t1 = self.model.maturity + k * self.model.delta_fixed
            zero_coupon_t1 = self.model.bond(t1)
            var_rho_t1 = self.compute_var_rho_t_vega(i, j, t1, ur, nmax)
            
            # Store in both formats
            fixed_leg[t1] = var_rho_t1
            hedging_fixed_leg[f"{t1}"] = var_rho_t1
            vega_zc_hedging_strategy[f"VEGA_FOR_STRIKE:{self.model.strike}:{i}_{j}:ZC:FIXEDLEG:{t1}"] = float(var_rho_t1)
            
            fixed_leg_value += zero_coupon_t1 * var_rho_t1
        
        # Compute vega value (same formula as original)
        vega_ij = (zero_coupon_tn1 * var_rho_tn1 
                   - zero_coupon_tn2 * var_rho_tn2 
                   - self.model.delta_fixed * self.model.strike * fixed_leg_value)
        
        vega_zc_hedging_strategy[f"VEGA_FOR_STRIKE:{self.model.strike}:{i}_{j}:ZC:VEGAVALUE:NA"] = float(vega_ij)
        
        if self.logger:
            self.logger.logger.info(f"Vega (ZC) [{i},{j}] complete | Value={vega_ij:+.6f}")
        
        if return_legacy_format:
            return vega_zc_hedging_strategy
        
        # Create structured result
        result = VegaHedgingResult(
            strike=self.model.strike,
            component_i=i,
            component_j=j,
            instrument_type=InstrumentType.ZERO_COUPON,
            vega_value=float(vega_ij),
            floating_leg=floating_leg,
            fixed_leg=fixed_leg
        )
        
        return result
        
    def compute_swaption_price_hedging_fix_float_vega(
        self, 
        i: int, 
        j: int, 
        ur: float = UR, #0.5
        nmax: int = NMAX,
        return_legacy_format: bool = False
    ) -> Union[VegaHedgingResult, Dict]:
        """
        Compute Vega hedging strategy using swaps.
        
        Parameters
        ----------
        i : int
            Row index of sigma component
        j : int
            Column index of sigma component
        ur : float
            Real part of integration contour (default: 0.5)
        nmax : int
            Maximum integration limit (default: NMAX)
        return_legacy_format : bool
            If True, return legacy dict format for backward compatibility
            
        Returns
        -------
        VegaHedgingResult or Dict
            Structured Vega hedging result (or legacy dict if return_legacy_format=True)
        """
        if self.logger:
            self.logger.log_calculation_start(
                "Vega (SWAP)",
                self.model.strike,
                component=f"({i},{j})",
                maturity=self.model.maturity,
                tenor=self.model.tenor
            )
        
        self.model.compute_b3_a3()
        
        # Initialize legacy format dict (same keys as original)
        vega_swap_hedging_strategy = {}
        
        # Initialize containers for structured result
        floating_leg = {}
        fixed_leg = {}
        
        # Floating leg
        zero_coupon_tn1 = self.model.bond(self.model.maturity)
        zero_coupon_tn2 = self.model.bond(self.model.maturity + self.model.tenor)
        var_rho_t1t2 = self.compute_var_rho_t_fix_float_vega(
            i, j, 
            self.model.maturity, 
            self.model.maturity + self.model.tenor, 
            ur, 
            nmax
        )
        
        vega_swap_hedging_strategy[f"VEGA_FOR_STRIKE:{self.model.strike}:{i}_{j}:SWAP:FLOATINGLEG:NA"] = float(var_rho_t1t2)
        floating_leg[self.model.maturity] = var_rho_t1t2
        
        # Fixed leg - accumulate b4, a4, all_zc (same as original)
        b4 = 0
        a4 = jnp.zeros(self.model.x0.shape)
        all_zc = 0
        
        for k in range(1, int(self.model.tenor / self.model.delta_fixed) + 1):
            t1 = self.model.maturity + k * self.model.delta_fixed
            zero_coupon_t1 = self.model.bond(t1)
            b1_bar, a1 = self.model.compute_bar_b1_a1(t1 - self.model.maturity)
            
            b4 += math.exp(-self.model.alpha * (t1 - self.model.maturity)) * b1_bar
            a4 += math.exp(-self.model.alpha * (t1 - self.model.maturity)) * a1
            
            all_zc += zero_coupon_t1
        
        # Compute expectation using CLASS METHOD (same as original)
        expectation_xy = self.compute_expectation_xy_vega_(i, j, b4, a4, ur, nmax)
        
        # Compute var_rho (same formula as original)
        var_rho = (math.exp(-self.model.alpha * self.model.maturity) / 
                   (1 + tr_uv(self.model.u1, self.model.x0))) * (expectation_xy / all_zc)
        
        fixed_leg_value = self.model.delta_fixed * self.model.strike * all_zc * var_rho
        
        # Compute vega value (same formula as original)
        vega_ij = (zero_coupon_tn1 - zero_coupon_tn2) * var_rho_t1t2 - fixed_leg_value
        
        vega_swap_hedging_strategy[f"VEGA_FOR_STRIKE:{self.model.strike}:{i}_{j}:SWAP:FIXEDLEG:NA"] = float(var_rho)
        vega_swap_hedging_strategy[f"VEGA_FOR_STRIKE:{self.model.strike}:{i}_{j}:ZC:VEGAVALUE:NA"] = float(vega_ij)
        
        # Store fixed leg for structured result
        fixed_leg[self.model.maturity + self.model.tenor] = var_rho
        
        if self.logger:
            self.logger.logger.info(f"Vega (SWAP) [{i},{j}] complete | Value={vega_ij:+.6f}")
        
        if return_legacy_format:
            return vega_swap_hedging_strategy
        
        # Create structured result
        result = VegaHedgingResult(
            strike=self.model.strike,
            component_i=i,
            component_j=j,
            instrument_type=InstrumentType.SWAP,
            vega_value=float(vega_ij),
            floating_leg=floating_leg,
            fixed_leg=fixed_leg
        )
        
        return result
        
    def compute_var_rho_t_vega(
        self, 
        i: int, 
        j: int, 
        t_bar: float, 
        ur: float = UR, #0.5
        nmax: int = NMAX
    ) -> float:
        """
        Compute VaR rho Vega sensitivity at time T.
        
        This method uses self.compute_expectation_xy_vega_ (the class method).
        
        Parameters
        ----------
        i : int
            Row index of sigma component
        j : int
            Column index of sigma component
        t_bar : float
            Target time
        ur : float
            Real part of integration contour
        nmax : int
            Maximum integration limit
            
        Returns
        -------
        float
            VaR rho Vega value
        """
        b1_bar, a1 = self.model.compute_bar_b1_a1(t_bar - self.model.maturity)
        
        exp_alpha_t = math.exp(-self.model.alpha * (t_bar - self.model.maturity))
        b4 = exp_alpha_t * b1_bar
        a4 = exp_alpha_t * a1
        
        # IMPORTANT: Use CLASS METHOD compute_expectation_xy_vega_ (same as original)
        expectation_xy = self.compute_expectation_xy_vega_(i, j, b4, a4, ur, nmax)
        
        zero_coupon_tbar = self.model.bond(t_bar)
        
        temp1 = (math.exp(-self.model.alpha * self.model.maturity) / 
                 (1 + tr_uv(self.model.u1, self.model.x0)))
        temp2 = expectation_xy / zero_coupon_tbar
        
        var_rho = temp1 * temp2
        return var_rho
        
    def compute_var_rho_t_fix_float_vega(
        self, 
        i: int, 
        j: int, 
        t1: float, 
        t2: float,
        ur: float = UR, #0.5
        nmax: int = NMAX
    ) -> float:
        """
        Compute VaR rho Vega sensitivity for fixed-floating swap.
        
        This method uses the IMPORTED compute_expectation_xy_vega function (different signature).
        
        Parameters
        ----------
        i : int
            Row index of sigma component
        j : int
            Column index of sigma component
        t1 : float
            First time point
        t2 : float
            Second time point
        ur : float
            Real part of integration contour
        nmax : int
            Maximum integration limit
            
        Returns
        -------
        float
            VaR rho Vega value
        """
        delta_t = t2 - t1
        b1_bar_1, a1_1 = self.model.compute_bar_b1_a1(0)
        b1_bar, a1 = self.model.compute_bar_b1_a1(delta_t)
        
        exp_alpha_t = math.exp(-self.model.alpha * delta_t)
        b4 = b1_bar_1 - exp_alpha_t * b1_bar
        a4 = a1_1 - exp_alpha_t * a1
        
        # IMPORTANT: Use IMPORTED function compute_expectation_xy_vega (same as original)
        # This has different signature: (wishart, i, j, a3, b3, b4, a4, ur, nmax)
        expectation_xy = compute_expectation_xy_vega(
            self.model.wishart,
            i, j,
            self.model.a3, 
            self.model.b3, 
            b4, 
            np.array(a4), 
            ur, 
            nmax
        )
        
        zero_coupon_t2 = self.model.bond(t2)
        zero_coupon_t1 = self.model.bond(t1)
        zero_coupon_tbar = zero_coupon_t1 - zero_coupon_t2
        
        temp1 = (math.exp(-self.model.alpha * self.model.maturity) / 
                 (1 + tr_uv(self.model.u1, self.model.x0)))
        temp2 = expectation_xy / zero_coupon_tbar
        
        var_rho_vega_ij = temp1 * temp2
        return var_rho_vega_ij
        
    def compute_expectation_xy_vega_(
        self, 
        i: int, 
        j: int, 
        b4: float, 
        a4: jnp.ndarray,
        ur: float = UR, #0.5
        nmax: int = NMAX, 
        recompute_a3_b3: bool = False
    ) -> float:
        """
        Compute E[XY] Vega sensitivity expectation.
        
        This is a CLASS METHOD that uses self.model.a3 and self.model.b3.
        Different from the imported compute_expectation_xy_vega function.
        
        Parameters
        ----------
        i : int
            Row index of sigma component
        j : int
            Column index of sigma component
        b4 : float
            Scalar coefficient
        a4 : jnp.ndarray
            Matrix coefficient
        ur : float
            Real part of integration contour
        nmax : int
            Maximum integration limit
        recompute_a3_b3 : bool
            Whether to recompute a3 and b3
            
        Returns
        -------
        float
            Expected value
        """
        if recompute_a3_b3:
            self.model.compute_b3_a3()
            
        def f1(ui):
            u = complex(ur, ui)
            
            z_a3 = u * self.model.a3
            phi1 = self.model.wishart.phi_one_vega(i, j, 1, z_a3)
            phi2 = self.model.wishart.phi_two_vega(i, j, 1, a4, z_a3)
            exp_z_b3 = cmath.exp(u * self.model.b3)
            
            temp1 = b4 * exp_z_b3 * phi1
            temp2 = exp_z_b3 * phi2
            res = temp1 + temp2
            res = res / u
            return res.real
            
        f2 = lambda x: f1(x)
        r1 = sp_i.quad(f2, 0, nmax)
        res1 = r1[0] / math.pi
        expectation = res1
        
        return expectation
    
    # =========================================================================
    # Convenience methods for computing all components
    # =========================================================================
    
    def compute_all_vega_zc(
        self, 
        ur: float = UR, #0.5
        nmax: int = NMAX,
        return_legacy_format: bool = False
    ) -> Union[List[VegaHedgingResult], Dict]:
        """
        Compute Vega hedging for all sigma components using ZC bonds.
        
        Parameters
        ----------
        ur : float
            Real part of integration contour
        nmax : int
            Maximum integration limit
        return_legacy_format : bool
            If True, return combined legacy dict format
            
        Returns
        -------
        List[VegaHedgingResult] or Dict
            List of Vega hedging results for all components (or combined dict)
        """
        results = []
        combined_dict = {}
        n = self.model.n
        
        for i_idx in range(n):
            for j_idx in range(n):
                if return_legacy_format:
                    result_dict = self.compute_swaption_price_vega_hedging(
                        i_idx, j_idx, ur, nmax, return_legacy_format=True
                    )
                    combined_dict.update(result_dict)
                else:
                    result = self.compute_swaption_price_vega_hedging(
                        i_idx, j_idx, ur, nmax, return_legacy_format=False
                    )
                    results.append(result)
        
        if return_legacy_format:
            return combined_dict
        
        return results
    
    def compute_all_vega_swap(
        self, 
        ur: float = UR, #0.5
        nmax: int = NMAX,
        return_legacy_format: bool = False
    ) -> Union[List[VegaHedgingResult], Dict]:
        """
        Compute Vega hedging for all sigma components using swaps.
        
        Parameters
        ----------
        ur : float
            Real part of integration contour
        nmax : int
            Maximum integration limit
        return_legacy_format : bool
            If True, return combined legacy dict format
            
        Returns
        -------
        List[VegaHedgingResult] or Dict
            List of Vega hedging results for all components (or combined dict)
        """
        results = []
        combined_dict = {}
        n = self.model.n
        
        for i_idx in range(n):
            for j_idx in range(n):
                if return_legacy_format:
                    result_dict = self.compute_swaption_price_hedging_fix_float_vega(
                        i_idx, j_idx, ur, nmax, return_legacy_format=True
                    )
                    combined_dict.update(result_dict)
                else:
                    result = self.compute_swaption_price_hedging_fix_float_vega(
                        i_idx, j_idx, ur, nmax, return_legacy_format=False
                    )
                    results.append(result)
        
        if return_legacy_format:
            return combined_dict
        
        return results
    
    def generate_vega_summary(self, results: List[VegaHedgingResult]) -> str:
        """
        Generate a formatted summary of all Vega results.
        
        Parameters
        ----------
        results : List[VegaHedgingResult]
            List of Vega hedging results
            
        Returns
        -------
        str
            Formatted summary string
        """
        if not results:
            return "No Vega results to display."
        
        lines = [
            f"\n{'#'*70}",
            f"#  VEGA SENSITIVITY SUMMARY",
            f"#  Strike: {results[0].strike:.4f}",
            f"{'#'*70}",
            "",
            f"  {'Component':<12} {'Instrument':<12} {'Vega Value':>16}",
            f"  {'-'*42}",
        ]
        
        for result in results:
            lines.append(
                f"  σ({result.component_i},{result.component_j})"
                f"{'':>6} {result.instrument_type.value:<12} "
                f"{result.vega_value:>+16.6f}"
            )
        
        lines.append(f"\n{'#'*70}")
        return "\n".join(lines)
