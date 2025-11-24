from CalibrateLrwFx import PRICING
import jax.numpy as jnp
import numpy as np
from typing import List, Union, Tuple
import scipy.integrate as sp_i
import cmath
import math

class LRWFxModel(BaseFxModel):
    # ... existing code ...
    
    #region Vectorize Option pricing

    def price_fx_options_vectorized(self, 
                                   maturities: Union[List[float], np.ndarray], 
                                   strikes: Union[List[float], np.ndarray],
                                   is_call: Union[bool, List[bool]] = True, 
                                   ur: float = 0.5, 
                                   nmax: int = 1000) -> np.ndarray:
        """
        Price multiple FX options efficiently using vectorization.
        
        Parameters:
        -----------
        maturities : array-like
            List or array of option maturities
        strikes : array-like  
            List or array of option strikes
        is_call : bool or array-like
            True for call options, False for puts. Can be single bool or array
        ur : float
            Real part of integration variable
        nmax : int
            Maximum integration limit
            
        Returns:
        --------
        np.ndarray
            Array of option prices corresponding to each (maturity, strike) pair
        """
        maturities = np.asarray(maturities)
        strikes = np.asarray(strikes)
        
        # Handle scalar expansion
        if maturities.shape == ():
            maturities = np.array([maturities])
        if strikes.shape == ():
            strikes = np.array([strikes])
            
        # Ensure same length
        if len(maturities) != len(strikes):
            raise ValueError("maturities and strikes must have the same length")
            
        # Handle is_call parameter
        if isinstance(is_call, bool):
            is_call_array = np.full(len(maturities), is_call)
        else:
            is_call_array = np.asarray(is_call)
            if len(is_call_array) != len(maturities):
                raise ValueError("is_call array must have same length as maturities/strikes")
        
        # Store original state to restore later
        original_state = self._save_option_state()
        
        # Pre-compute all option-specific parameters
        option_params = self._precompute_option_params_vectorized(maturities, strikes)
        
        # Vectorized pricing
        prices = self._price_options_batch(option_params, is_call_array, ur, nmax)
        
        # Restore original state
        self._restore_option_state(original_state)
        
        return prices
    
    def _save_option_state(self) -> dict:
        """Save current option-related state for restoration."""
        state = {}
        if hasattr(self, 'maturity'):
            state['maturity'] = self.maturity
        if hasattr(self, 'strike'):
            state['strike'] = self.strike
        if hasattr(self, 'bij_2'):
            state['bij_2'] = self.bij_2
        if hasattr(self, 'aij_2'):
            state['aij_2'] = self.aij_2
        if hasattr(self, 'eta_j'):
            state['eta_j'] = self.eta_j
        if hasattr(self, 'eta_i'):
            state['eta_i'] = self.eta_i
        return state
    
    def _restore_option_state(self, state: dict):
        """Restore option-related state."""
        for key, value in state.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def _precompute_option_params_vectorized(self, maturities: np.ndarray, 
                                           strikes: np.ndarray) -> List[dict]:
        """Pre-compute all option-specific parameters for vectorized pricing."""
        option_params = []
        
        for maturity, strike in zip(maturities, strikes):
            # Compute parameters that would be set by set_option_properties
            multi1 = self.fx_spot * jnp.exp(-self.alpha_j * maturity) / self.zeta_j_s
            multi2 = strike * jnp.exp(-self.alpha_i * maturity) / self.zeta_i_s
            
            bij_2 = multi1 - multi2
            aij_2 = multi1 * self.u_j - multi2 * self.u_i
            
            eta_j = jnp.exp(-self.alpha_j * maturity) / self.zeta_j_s
            eta_i = jnp.exp(-self.alpha_i * maturity) / self.zeta_i_s
            
            # Pre-compute bond prices for put-call parity
            # Temporarily set maturity for bond computation
            old_maturity = getattr(self, 'maturity', None)
            self.maturity = maturity
            domestic_df = self.lrw_currency_i.bond(maturity)
            foreign_df = self.lrw_currency_j.bond(maturity)
            if old_maturity is not None:
                self.maturity = old_maturity
            
            option_params.append({
                'maturity': maturity,
                'strike': strike,
                'bij_2': bij_2,
                'aij_2': aij_2,
                'eta_j': eta_j,
                'eta_i': eta_i,
                'domestic_df': domestic_df,
                'foreign_df': foreign_df
            })
            
        return option_params
    
    def _price_options_batch(self, option_params: List[dict], 
                           is_call_array: np.ndarray, 
                           ur: float, nmax: int) -> np.ndarray:
        """Price a batch of options with pre-computed parameters."""
        prices = np.zeros(len(option_params))
        
        for i, (params, is_call) in enumerate(zip(option_params, is_call_array)):
            # Create integrand for this specific option
            def integrand(ui):
                u = complex(ur, ui)
                z1 = self.lrw_currency_i.wishart.phi_one(1.0, u * params['aij_2'])
                z1 *= cmath.exp(u * params['bij_2'])
                z2 = z1 / (u * u)
                return z2.real
            
            # Compute integral
            result, _ = sp_i.quad(integrand, 0, nmax)
            price = result / math.pi
            
            # Apply pseudo-inverse smoothing if enabled
            if (hasattr(self.lrw_currency_i, 'pseudo_inverse_smoothing') and 
                self.lrw_currency_i.pseudo_inverse_smoothing):
                curve_alpha_adjustment = self.lrw_currency_i.initial_curve_alpha.get_alpha(params['maturity'])
                price *= curve_alpha_adjustment
            
            # Put-call parity for put options
            if not is_call:
                put_adjustment = (params['domestic_df'] * params['strike'] - 
                                params['foreign_df'] * self.fx_spot)
                price = price + put_adjustment
            
            # Apply minimum price constraint
            price = np.maximum(price, DEFAULT_EPSILON)
            prices[i] = price
            
        return prices
    
    # Enhanced version with better integration handling
    def _price_options_batch_enhanced(self, option_params: List[dict], 
                                    is_call_array: np.ndarray, 
                                    ur: float, nmax: int) -> np.ndarray:
        """Enhanced batch pricing with better integration strategies."""
        prices = np.zeros(len(option_params))
        
        for i, (params, is_call) in enumerate(zip(option_params, is_call_array)):
            # Create integrand for this specific option
            bij_2 = params['bij_2']
            aij_2 = params['aij_2']
            
            def integrand(ui):
                u = complex(ur, ui)
                z1 = self.lrw_currency_i.wishart.phi_one(1.0, u * aij_2)
                z1 *= cmath.exp(u * bij_2)
                z2 = z1 / (u * u)
                return z2.real
            
            # Use the same integration strategy as original
            min_integ = 1e-2
            use_split = False
            
            if use_split:
                # First integration
                result, error = sp_i.quad(integrand, min_integ, nmax)
                
                # Second integration for small values
                if min_integ > 0:
                    result_zero, error_zero = self.trapezoid_integ(integrand, 0.0, min_integ, n_points=30)
                    result += result_zero
            else:
                result, _ = sp_i.quad(integrand, 0, nmax)
            
            price = result / math.pi
            
            # Apply pseudo-inverse smoothing if enabled
            if (hasattr(self.lrw_currency_i, 'pseudo_inverse_smoothing') and 
                self.lrw_currency_i.pseudo_inverse_smoothing):
                curve_alpha_adjustment = self.lrw_currency_i.initial_curve_alpha.get_alpha(params['maturity'])
                price *= curve_alpha_adjustment
            
            # Put-call parity for put options
            if not is_call:
                put_adjustment = (params['domestic_df'] * params['strike'] - 
                                params['foreign_df'] * self.fx_spot)
                price = price + put_adjustment
            
            # Apply minimum price constraint
            price = np.maximum(price, DEFAULT_EPSILON)
            prices[i] = price
            
        return prices
    
    # Convenience method for single option (maintains backward compatibility)
    def price_fx_option_exist_already(self, is_call: bool = True, ur: float = 0.5, 
                       nmax: int = 1000) -> float:
        """
        Price single FX option using Fourier transform.
        
        This method is unchanged to maintain backward compatibility.
        """
        # Original implementation remains the same
        def integrand(ui):
            u = complex(ur, ui)
            z1 = self.lrw_currency_i.wishart.phi_one(1.0, u * self.aij_2)
            z1 *= cmath.exp(u * self.bij_2)
            
            z2 = z1 / (u * u)
            return z2.real
            
        min_integ = 1e-2
        use_split = False
        
        if use_split:
            result, error = sp_i.quad(integrand, min_integ, nmax)
            if min_integ > 0:
                result_zero, error_zero = self.trapezoid_integ(integrand, 0.0, min_integ, n_points=30)
                result += result_zero
        else:
            result, _ = sp_i.quad(integrand, 0, nmax)
           
        price = result / math.pi
        
        # Apply pseudo-inverse smoothing if enabled
        if (hasattr(self.lrw_currency_i, 'pseudo_inverse_smoothing') and 
            self.lrw_currency_i.pseudo_inverse_smoothing):
            curve_alpha_adjustment = self.lrw_currency_i.initial_curve_alpha.get_alpha(self.maturity)
            price *= curve_alpha_adjustment
            
        # Put-call parity for put options
        if not is_call:
            domestic_df = self.lrw_currency_i.bond(self.maturity)
            foreign_df = self.lrw_currency_j.bond(self.maturity)
            put_adjustment = domestic_df * self.strike - foreign_df * self.fx_spot
            price = price + put_adjustment
            
        price = np.maximum(price, DEFAULT_EPSILON)
        return price

    #endregion
