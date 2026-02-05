

# sensitivities/delta_hedging.py
"""Delta hedging strategies for LRW model."""

import math
import cmath
from typing import Dict, Tuple
import jax.numpy as jnp
import scipy.integrate as sp_i
import numpy as np

from ..utils.local_functions import tr_uv
from ..core.expectations import compute_expectation_xy

class DeltaHedgingCalculator:
    """Calculate delta hedging strategies for swaptions."""
    
    def __init__(self, model):
        """Initialize delta hedging calculator."""
        self.model = model
        
    def compute_swaption_price_hedging(self, ur: float = 0.5, nmax: int = 1000) -> Dict:
        """Compute delta hedging strategy using zero-coupon bonds."""
        self.model.compute_b3_a3()
        
        delta_zc_hedging_strategy = {}
        hedging_floating_leg = {}
        hedging_fixed_leg = {}
        
        # Floating leg
        zero_coupon_tn1 = self.model.bond(self.model.maturity)
        var_rho_tn1 = self.compute_var_rho_t(self.model.maturity, ur, nmax)
        hedging_floating_leg[f"{self.model.maturity}"] = var_rho_tn1
        delta_zc_hedging_strategy[f"DELTA_FOR_STRIKE:{self.model.strike}:NA_NA:ZC:FLOATINGLEG:{self.model.maturity}"] = float(var_rho_tn1)
        
        zero_coupon_tn2 = self.model.bond(self.model.maturity + self.model.tenor)
        var_rho_tn2 = self.compute_var_rho_t(self.model.maturity + self.model.tenor, ur, nmax)
        hedging_floating_leg[f"{self.model.maturity + self.model.tenor}"] = var_rho_tn2
        delta_zc_hedging_strategy[f"DELTA_FOR_STRIKE:{self.model.strike}:NA_NA:ZC:FLOATINGLEG:{self.model.maturity + self.model.tenor}"] = float(var_rho_tn2)
        
        # Fixed leg
        fixed_leg = 0
        for i in range(1, int(self.model.tenor / self.model.delta_fixed) + 1):
            t1 = self.model.maturity + i * self.model.delta_fixed
            zero_coupon_t1 = self.model.bond(t1)
            var_rho_t1 = self.compute_var_rho_t(t1, ur, nmax)
            hedging_fixed_leg[f"{t1}"] = var_rho_t1
            delta_zc_hedging_strategy[f"DELTA_FOR_STRIKE:{self.model.strike}:NA_NA:ZC:FIXEDLEG:{t1}"] = float(var_rho_t1)
            
            fixed_leg += zero_coupon_t1 * var_rho_t1
            
        price = zero_coupon_tn1 * var_rho_tn1 - zero_coupon_tn2 * var_rho_tn2 - self.model.delta_fixed * self.model.strike * fixed_leg
        zc_hedging_strategy = {"DELTA_FOR_STRIKE": {"ZC": {"FloatingLeg": hedging_floating_leg, "FixedLeg": hedging_fixed_leg}}}
        
        delta_zc_hedging_strategy[f"DELTA_FOR_STRIKE:{self.model.strike}:NA_NA:ZC:PRICE:NA"] = float(price)
        
        return delta_zc_hedging_strategy
        
    def compute_swaption_price_hedging_fix_float(self, ur: float = 0.5, nmax: int = 1000) -> Dict:
        """Compute delta hedging strategy using swaps."""
        self.model.compute_b3_a3()
        delta_swap_hedging_strategy = {}
        
        # Floating leg
        zero_coupon_tn1 = self.model.bond(self.model.maturity)
        zero_coupon_tn2 = self.model.bond(self.model.maturity + self.model.tenor)
        var_rho_t1t2 = self.compute_var_rho_t_fix_float(self.model.maturity, self.model.maturity + self.model.tenor, ur, nmax)
        delta_swap_hedging_strategy[f"DELTA_FOR_STRIKE:{self.model.strike}:NA_NA:SWAP:FLOATINGLEG:NA"] = float(var_rho_t1t2)
        
        # Fixed leg
        fixed_leg = 0
        b4 = 0
        a4 = jnp.zeros(self.model.x0.shape)
        all_zc = 0
        
        for i in range(1, int(self.model.tenor / self.model.delta_fixed) + 1):
            t1 = self.model.maturity + i * self.model.delta_fixed
            zero_coupon_t1 = self.model.bond(t1)
            b1_bar, a1 = self.model.compute_bar_b1_a1(t1 - self.model.maturity)
            
            b4 += math.exp(-self.model.alpha * (t1 - self.model.maturity)) * b1_bar
            a4 += math.exp(-self.model.alpha * (t1 - self.model.maturity)) * a1
            
            all_zc += zero_coupon_t1
            
        expectation_xy = self.compute_expectation_xy(b4, a4, ur, nmax)
        var_rho = (math.exp(-self.model.alpha * self.model.maturity) / (1 + tr_uv(self.model.u1, self.model.x0))) * (expectation_xy / all_zc)
        
        fixed_leg = self.model.delta_fixed * self.model.strike * all_zc * var_rho
        
        price = (zero_coupon_tn1 - zero_coupon_tn2) * var_rho_t1t2 - fixed_leg
        
        delta_swap_hedging_strategy[f"DELTA_FOR_STRIKE:{self.model.strike}:NA_NA:SWAP:FIXEDLEG:NA"] = float(var_rho)
        delta_swap_hedging_strategy[f"DELTA_FOR_STRIKE:{self.model.strike}:NA_NA:SWAP:PRICE:NA"] = float(price)
        
        return delta_swap_hedging_strategy
        
    def compute_var_rho_t(self, t_bar: float, ur: float = 0.5, nmax: int = 1000) -> float:
        """Compute VaR rho at time T."""
        b1_bar, a1 = self.model.compute_bar_b1_a1(t_bar - self.model.maturity)
        
        exp_alpha_t = math.exp(-self.model.alpha * (t_bar - self.model.maturity))
        b4 = exp_alpha_t * b1_bar
        a4 = exp_alpha_t * a1
        
        expectation_xy = self.compute_expectation_xy(b4, a4, ur, nmax)
        zero_coupon_tbar = self.model.bond(t_bar)
        
        temp1 = (math.exp(-self.model.alpha * self.model.maturity) / (1 + tr_uv(self.model.u1, self.model.x0)))
        temp2 = expectation_xy / zero_coupon_tbar
        
        var_rho = temp1 * temp2
        return var_rho
        
    def compute_var_rho_t_fix_float(self, t1: float, t2: float, ur: float = 0.5, nmax: int = 1000) -> float:
        """Compute VaR rho for fixed-floating swap."""
        delta_t = t2 - t1
        b1_bar_1, a1_1 = self.model.compute_bar_b1_a1(0)
        b1_bar, a1 = self.model.compute_bar_b1_a1(delta_t)
        
        exp_alpha_t = math.exp(-self.model.alpha * delta_t)
        b4 = b1_bar_1 - exp_alpha_t * b1_bar
        a4 = a1_1 - exp_alpha_t * a1
        
       
        # expectation_xy = self.model.wishart.compute_expectation_xy(self.model.a3, self.model.b3, b4, np.array(a4), ur, nmax)
        expectation_xy = compute_expectation_xy(self.model.wishart, self.model.a3, self.model.b3, b4, np.array(a4), ur, nmax)
        zero_coupon_t2 = self.model.bond(t2)
        zero_coupon_t1 = self.model.bond(t1)
        zero_coupon_tbar = zero_coupon_t1 - zero_coupon_t2
        
        temp1 = (math.exp(-self.model.alpha * self.model.maturity) / (1 + tr_uv(self.model.u1, self.model.x0)))
        temp2 = expectation_xy / zero_coupon_tbar
        
        var_rho = temp1 * temp2
        return var_rho
        
    def compute_expectation_xy(self, b4: float, a4: jnp.ndarray
                               , ur: float = 0.5, nmax: int = 1000 
                               , recompute_a3_b3: bool = False) -> float:
        """Compute E[XY] expectation."""
        if recompute_a3_b3:
            self.model.compute_b3_a3()
            
        def f1(ui):
            u = complex(ur, ui)
            
            z_a3 = u * self.model.a3
            phi1 = self.model.wishart.phi_one(1, z_a3)
            phi2 = self.model.wishart.phi_two(1, a4, z_a3)
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
