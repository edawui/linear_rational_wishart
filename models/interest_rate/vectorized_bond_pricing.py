import jax.numpy as jnp
import numpy as np
from typing import Union, List
from jax import vmap, jit
from functools import partial

class LRWModel(BaseInterestRateModel):
    # ... existing code ...
    
    #region Vectorize Bond pricing
    def bond_vectorized(self, maturities: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Compute zero-coupon bond prices for multiple maturities efficiently.
        
        Parameters
        ----------
        maturities : array-like
            List or array of times to maturity
            
        Returns
        -------
        np.ndarray
            Array of bond prices corresponding to each maturity
        """
        maturities = np.asarray(maturities)
        
        # Handle scalar input
        if maturities.shape == ():
            maturities = np.array([maturities])
        
        # Use JAX's vmap for vectorization
        vectorized_bond_single = vmap(self._bond_single, in_axes=0)
        
        # Convert to JAX array for computation
        maturities_jax = jnp.array(maturities)
        
        # Compute all bond prices at once
        bond_prices = vectorized_bond_single(maturities_jax)
        
        # Convert back to numpy array
        return np.array(bond_prices)
    
    @partial(jit, static_argnums=(0,))
    def _bond_single(self, t: float) -> float:
        """
        Compute single bond price (internal vectorized version).
        
        This is the core bond pricing logic extracted for vectorization.
        
        Parameters
        ----------
        t : float
            Time to maturity
            
        Returns
        -------
        float
            Bond price
        """
        c1 = 1 + self.wishart.compute_mean(t, self.u1)
        c1 = c1 / (1 + jnp.trace(self.u1 @ self.x0))
        c1 = c1 * jnp.exp(-self.alpha * t)
        
        # Apply pseudo-inverse smoothing if enabled
        if self.pseudo_inverse_smoothing and self.initial_curve_alpha is not None:
            curve_alpha_adjustment = self.initial_curve_alpha.get_alpha(t)
            c1 *= curve_alpha_adjustment
        
        return c1
    
    # Keep original bond method unchanged for backward compatibility
    def bond_existing_already(self, t: float) -> float:
        """
        Compute zero-coupon bond price.
        
        Parameters
        ----------
        t : float
            Time to maturity
            
        Returns
        -------
        float
            Bond price
        """
        c1 = 1 + self.wishart.compute_mean(t, self.u1)
        c1 = c1 / (1 + jnp.trace(self.u1 @ self.x0))
        c1 = c1 * jnp.exp(-self.alpha * t)
        
        if self.pseudo_inverse_smoothing and self.initial_curve_alpha is not None:
            curve_alpha_adjustment = self.initial_curve_alpha.get_alpha(t)
            c1 *= curve_alpha_adjustment
        
        return c1
    
    def spread_vectorized(self, maturities: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Compute spreads for multiple maturities efficiently.
        
        Parameters
        ----------
        maturities : array-like
            List or array of times
            
        Returns
        -------
        np.ndarray
            Array of spread values corresponding to each maturity
        """
        maturities = np.asarray(maturities)
        
        # Handle scalar input
        if maturities.shape == ():
            maturities = np.array([maturities])
        
        # Use JAX's vmap for vectorization
        vectorized_spread_single = vmap(self._spread_single, in_axes=0)
        
        # Convert to JAX array for computation
        maturities_jax = jnp.array(maturities)
        
        # Compute all spread values at once
        spread_values = vectorized_spread_single(maturities_jax)
        
        # Convert back to numpy array
        return np.array(spread_values)
    
    @partial(jit, static_argnums=(0,))
    def _spread_single(self, t: float) -> float:
        """
        Compute single spread value (internal vectorized version).
        
        Parameters
        ----------
        t : float
            Time
            
        Returns
        -------
        float
            Spread value
        """
        c1 = self.wishart.compute_mean(t, self.u2)
        c1 = c1 / (1 + jnp.trace(self.u1 @ self.x0))
        c1 = c1 * jnp.exp(-self.alpha * t)
        return c1
    
    # Keep original spread method unchanged for backward compatibility  
    def spread_existing_already(self, t: float) -> float:
        """
        Compute spread at time t.
        
        Parameters
        ----------
        t : float
            Time
            
        Returns
        -------
        float
            Spread value
        """
        c1 = self.wishart.compute_mean(t, self.u2)
        c1 = c1 / (1 + jnp.trace(self.u1 @ self.x0))
        c1 = c1 * jnp.exp(-self.alpha * t)
        return c1
    
    def compute_yield_curve(self, maturities: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Compute yield curve for multiple maturities.
        
        Parameters
        ----------
        maturities : array-like
            List or array of times to maturity
            
        Returns
        -------
        np.ndarray
            Array of yields corresponding to each maturity
        """
        bond_prices = self.bond_vectorized(maturities)
        maturities = np.asarray(maturities)
        
        # Convert bond prices to yields: y = -ln(P) / T
        yields = -np.log(bond_prices) / maturities
        
        return yields
    
    def compute_forward_rates(self, maturities: Union[List[float], np.ndarray], 
                            tenor: float = 0.01) -> np.ndarray:
        """
        Compute instantaneous forward rates for multiple maturities.
        
        Parameters
        ----------
        maturities : array-like
            List or array of times to maturity
        tenor : float
            Small time increment for numerical differentiation
            
        Returns
        -------
        np.ndarray
            Array of forward rates
        """
        maturities = np.asarray(maturities)
        
        # Compute bond prices at t and t + tenor
        bond_prices_t = self.bond_vectorized(maturities)
        bond_prices_t_plus_tenor = self.bond_vectorized(maturities + tenor)
        
        # Forward rate: f(t) = -d(ln P(t))/dt ≈ -(ln P(t+dt) - ln P(t))/dt
        forward_rates = -(np.log(bond_prices_t_plus_tenor) - np.log(bond_prices_t)) / tenor
        
        return forward_rates
    
    def compute_discount_factors_vectorized(self, maturities: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Compute discount factors for multiple maturities.
        
        This is an alias for bond_vectorized for clarity in some contexts.
        
        Parameters
        ----------
        maturities : array-like
            List or array of times to maturity
            
        Returns
        -------
        np.ndarray
            Array of discount factors (bond prices)
        """
        return self.bond_vectorized(maturities)
    
    def compute_zero_curve_bootstrap(self, market_maturities: np.ndarray, 
                                   market_rates: np.ndarray,
                                   interpolation_maturities: np.ndarray) -> np.ndarray:
        """
        Bootstrap zero curve and interpolate to desired maturities.
        
        Parameters
        ----------
        market_maturities : np.ndarray
            Market observation maturities
        market_rates : np.ndarray
            Market rates for calibration
        interpolation_maturities : np.ndarray
            Maturities where we want interpolated rates
            
        Returns
        -------
        np.ndarray
            Interpolated zero rates
        """
        # This is a placeholder for calibration logic
        # In practice, this would involve fitting model parameters to market data
        
        # For now, just compute model-implied rates
        model_rates = self.compute_yield_curve(interpolation_maturities)
        
        return model_rates

    #endregion