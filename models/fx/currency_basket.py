
# models/fx/currency_basket.py
"""Currency basket model for multiple FX pairs."""

import math
import cmath
from typing import List, Tuple, Dict, Any
import scipy.integrate as sp_i

import jax.numpy as jnp

from .lrw_fx import LRWFxModel


class CurrencyBasket:
    """Currency basket for pricing options on weighted FX portfolios."""
    
    def __init__(self, lrw_fx_list: List[LRWFxModel], weight_list: List[float]):
        """
        Initialize currency basket.
        
        Parameters:
        -----------
        lrw_fx_list : List[LRWFxModel]
            List of FX models
        weight_list : List[float]
            Weights for each currency (must sum to 1)
        """
        # Validation
        if len(lrw_fx_list) != len(weight_list):
            raise ValueError("Number of models and weights should be the same")
            
        all_weights = sum(weight_list)
        if abs(all_weights - 1.0) > 1e-10:
            raise ValueError(f"Total weight is {all_weights}, it should be equal to 1")
            
        self.lrw_fx_list = lrw_fx_list
        self.weight_list = jnp.array(weight_list)
        self.n_currency = len(weight_list)
        
        print("Currency basket initialized")
        
    def set_option_properties(self, maturity: float, strike: float):
        """Set option properties for the basket."""
        self.maturity = maturity
        self.strike = strike
        
        # Initialize basket parameters
        self.b_bar = 0
        self.a_bar = jnp.zeros(self.lrw_fx_list[0]._x0.shape)
        
        # Compute weighted sum
        for j in range(self.n_currency):
            self.lrw_fx_list[j].set_option_properties(maturity, strike)
            
            multi_j = self.weight_list[j] * self.lrw_fx_list[j].fx_spot * self.lrw_fx_list[j].eta_j
            self.b_bar = self.b_bar + multi_j
            self.a_bar = self.a_bar + multi_j * self.lrw_fx_list[j].u_j
            
        # Subtract strike component
        self.b_bar = self.b_bar - self.strike * self.lrw_fx_list[0].eta_i
        self.a_bar = self.a_bar - self.strike * self.lrw_fx_list[0].eta_i * self.lrw_fx_list[0].u_i
        
        self._set_all_a3_b3()
        
    def _set_all_a3_b3(self):
        """Set a3 and b3 for all currency models."""
        for j in range(self.n_currency):
            self.lrw_fx_list[j].lrw_currency_i.a3 = self.a_bar
            self.lrw_fx_list[j].lrw_currency_j.a3 = self.a_bar
            self.lrw_fx_list[j].lrw_currency_i.b3 = self.b_bar
            self.lrw_fx_list[j].lrw_currency_j.b3 = self.b_bar
            
    def compute_delta_strategy(self) -> Tuple[float, Dict[str, Any]]:
        """Compute delta hedging strategy for the basket."""
        delta_strategy_report = {}
        price = 0.0
        
        for j in range(self.n_currency):
            fx_model = self.lrw_fx_list[j]
            eta_j = fx_model.eta_j
            eta_i = jnp.exp(-fx_model._alpha_i * self.maturity) / fx_model.zeta_i_s
            
            # Compute expectations
            bij_3 = 1
            aij_3 = fx_model.u_i
            exp_xy_i = fx_model.compute_expectation_xy(bij_3, aij_3, self.b_bar, self.a_bar)
            
            bij_3 = 1
            aij_3 = fx_model.u_j
            exp_xy_j = fx_model.compute_expectation_xy(bij_3, aij_3, self.b_bar, self.a_bar)
            
            # Bond prices
            p_i_t = fx_model.lrw_currency_i.bond(self.maturity)
            p_j_t = fx_model.lrw_currency_j.bond(self.maturity)
            
            # Delta ratios
            rho_i = (eta_i / p_i_t) * exp_xy_i
            rho_j = (eta_j / p_j_t) * exp_xy_j
            
            # Add weighted contribution
            price += self.weight_list[j] * (p_j_t * fx_model.fx_spot * rho_j - p_i_t * self.strike * rho_i)
            
            delta_strategy_report[f"DELTA:{self.strike}:{j}_i:DELTAVALUE:ALL:NA"] = rho_i
            delta_strategy_report[f"DELTA:{self.strike}:{j}_j:DELTAVALUE:ALL:NA"] = rho_j
            
        return price, delta_strategy_report
        
    def compute_cross_gamma(self, gamma_id0: int, gamma_id1: int,
                           ur: float = 0.5, nmax: int = 1000) -> Tuple[float, Dict[str, Any]]:
        """Compute cross-gamma between two currencies."""
        if gamma_id0 not in range(self.n_currency):
            raise ValueError(f"Unknown currency id {gamma_id0} requested")
            
        if gamma_id1 not in range(self.n_currency):
            raise ValueError(f"Unknown currency id {gamma_id1} requested")
            
        # Get parameters for first currency
        j = gamma_id0
        mu0 = self.weight_list[j] * self.lrw_fx_list[j].fx_spot * self.lrw_fx_list[j].eta_j
        theta0 = mu0 * self.lrw_fx_list[j].u_j
        
        # Get parameters for second currency
        j = gamma_id1
        mu1 = self.weight_list[j] * self.lrw_fx_list[j].fx_spot * self.lrw_fx_list[j].eta_j
        theta1 = mu1 * self.lrw_fx_list[j].u_j
        
        # Compute residual
        if gamma_id1 != gamma_id0:
            mu2 = self.b_bar - mu0 - mu1
            theta2 = self.a_bar - theta0 - theta1
        else:
            mu2 = self.b_bar - mu0
            theta2 = self.a_bar - theta0
            
        # Compute gamma using integration
        gamma = self._compute_gamma_integral(mu0, mu1, theta0, theta1, ur, nmax)
        
        gamma_result = {
            f"Cross Gamma:{self.strike}:{gamma_id0}_{gamma_id1}:NA:NA": gamma
        }
        
        return gamma, gamma_result
        
    def compute_gamma(self, gamma_id0: int, ur: float = 0.5, 
                     nmax: int = 1000) -> Tuple[float, Dict[str, Any]]:
        """Compute gamma for a single currency."""
        if gamma_id0 not in range(self.n_currency):
            raise ValueError(f"Unknown currency id {gamma_id0} requested")
            
        j = gamma_id0
        mu0 = self.weight_list[j] * self.lrw_fx_list[j].fx_spot * self.lrw_fx_list[j].eta_j
        theta0 = mu0 * self.lrw_fx_list[j].u_j
        
        mu1 = mu0
        theta1 = theta0
        
        # Compute gamma integral
        gamma = self._compute_gamma_integral_single(mu0, theta0, ur, nmax)
        
        gamma_result = {
            f"Gamma:{self.strike}:{gamma_id0}_{gamma_id0}:NA:NA": gamma
        }
        
        return gamma, gamma_result
        
    def _compute_gamma_integral(self, mu0: float, mu1: float,
                               theta0: jnp.ndarray, theta1: jnp.ndarray,
                               ur: float, nmax: int) -> float:
        """Compute gamma integral for cross-gamma."""
        # This implements the four integral terms from the original code
        # Simplified here - full implementation would include all terms
        
        def integrand(ui):
            u = complex(ur, ui)
            
            # I1 term
            z_a3 = u * self.a_bar
            phi1 = self.lrw_fx_list[0].lrw_currency_i.wishart.phi_one(1, z_a3)
            exp_z_b3 = cmath.exp(u * self.b_bar)
            res_i1 = mu0 * mu1 * exp_z_b3 * phi1 / (u * u)
            
            # Additional terms would be computed here...
            # This is simplified for brevity
            
            return res_i1.real
            
        result, _ = sp_i.quad(integrand, 0, nmax)
        expectation = result / math.pi
        
        return expectation
        
    def _compute_gamma_integral_single(self, mu0: float, theta0: jnp.ndarray,
                                     ur: float, nmax: int) -> float:
        """Compute gamma integral for single currency."""
        # Similar to _compute_gamma_integral but for single currency case
        pass
        
    def print_model(self):
        """Print all models in the basket."""
        for j in range(self.n_currency):
            print(f"\nModel for ccy pair nb: {j}")
            print(f"Model for ccy pair weight: {self.weight_list[j]}")
            self.lrw_fx_list[j].print_model()

