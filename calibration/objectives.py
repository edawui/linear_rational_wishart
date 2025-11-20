"""
Objective functions for LRW Jump model calibration.

This module provides JAX-optimized objective functions for various
calibration targets including OIS curves, spreads, and swaptions.
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial

from ..models.interest_rate.lrw_model import LRWModel
from ..data import MarketData
from ..utils.jax_optimizations import jax_log, jax_exp, jax_sqrt, compute_rmse


class ObjectiveFunctions:
    """
    Collection of objective functions for LRW Jump model calibration.
    
    All functions are JAX-optimized for performance.
    """
    
    def __init__(
        self,
        model: LRWModel,
        daily_data: MarketData.DailyData
    ):
        """
        Initialize objective functions.
        
        Parameters
        ----------
        model : LRWModel
            The model being calibrated
        daily_data : MarketData.DailyData
            Market data for calibration
        """
        self.model = model
        self.daily_data = daily_data
        
    @partial(jit, static_argnums=(0,))
    def _compute_bond_price_errors(
        self,
        market_prices: jnp.ndarray,
        model_prices: jnp.ndarray,
        power: float = 2.0
    ) -> jnp.ndarray:
        """JAX-optimized bond price error calculation."""
        errors = market_prices - model_prices
        return 10000 * jnp.power(jnp.abs(errors), power)
        
    @partial(jit, static_argnums=(0,))
    def _compute_rate_errors(
        self,
        market_rates: jnp.ndarray,
        model_rates: jnp.ndarray,
        power: float = 2.0
    ) -> jnp.ndarray:
        """JAX-optimized rate error calculation."""
        errors = market_rates - model_rates
        return 10000 * jnp.power(jnp.abs(errors), power)
        
    def ois_price_objective(
        self,
        params: np.ndarray,
        params_activation: List[bool],
        max_tenor: float,
        min_tenor: float,
        power: float = 2.0
    ) -> np.ndarray:
        """
        Objective function for OIS price calibration.
        
        Parameters
        ----------
        params : np.ndarray
            Parameter values
        params_activation : List[bool]
            Parameter activation flags
        max_tenor : float
            Maximum tenor to include
        min_tenor : float
            Minimum tenor to include
        power : float, default=2.0
            Power for error calculation
            
        Returns
        -------
        np.ndarray
            Array of errors for least squares optimization
        """
        # Update model parameters
        self._update_model_params(params, params_activation)
        
        # Reprice bonds
        self._reprice_bonds()
        
        # Collect errors
        errors = []
        for _, ois_data in self.daily_data.ois_rate_data.iterrows():
            if min_tenor <= ois_data["TimeToMat"] <= max_tenor:
                market_price = ois_data["Object"].market_zc_price
                model_price = ois_data["Object"].model_zc_price
                error = market_price - model_price
                errors.append(error)
                
        return np.array(errors)
        
    def ois_rate_objective(
        self,
        params: np.ndarray,
        params_activation: List[bool],
        max_tenor: float,
        min_tenor: float
    ) -> np.ndarray:
        """
        Objective function for OIS rate calibration.
        
        Parameters
        ----------
        params : np.ndarray
            Parameter values
        params_activation : List[bool]
            Parameter activation flags
        max_tenor : float
            Maximum tenor to include
        min_tenor : float
            Minimum tenor to include
            
        Returns
        -------
        np.ndarray
            Array of errors for least squares optimization
        """
        # Update model parameters
        self._update_model_params(params, params_activation)
        
        # Reprice bonds
        self._reprice_bonds()
        
        # Collect errors
        errors = []
        for _, ois_data in self.daily_data.ois_rate_data.iterrows():
            if min_tenor <= ois_data["TimeToMat"] <= max_tenor:
                market_rate = ois_data["Object"].market_zc_rate
                model_rate = ois_data["Object"].model_zc_rate
                error = market_rate - model_rate
                errors.append(error)
                
        return np.array(errors)
        
    def spread_aggregate_objective(
        self,
        params: np.ndarray,
        params_activation: List[bool],
        max_tenor: float,
        min_tenor: float
    ) -> np.ndarray:
        """
        Objective function for aggregate spread calibration.
        
        Parameters
        ----------
        params : np.ndarray
            Parameter values
        params_activation : List[bool]
            Parameter activation flags
        max_tenor : float
            Maximum tenor to include
        min_tenor : float
            Minimum tenor to include
            
        Returns
        -------
        np.ndarray
            Array of errors for least squares optimization
        """
        # Update model parameters
        self._update_model_params(params, params_activation)
        
        # Reprice spreads
        self._reprice_spreads(full_a=False)
        
        # Collect errors
        errors = []
        for _, eur_data in self.daily_data.euribor_rate_data.iterrows():
            if min_tenor <= eur_data["TimeToMat"] <= max_tenor:
                time_to_mat = eur_data["TimeToMat"]
                market_a = eur_data["Object"].market_aggregate_a / time_to_mat
                model_a = eur_data["Object"].model_aggregate_a / time_to_mat
                error = market_a - model_a
                errors.append(error)
                
        return np.array(errors)
        
    def spread_full_objective(
        self,
        params: np.ndarray,
        params_activation: List[bool],
        max_tenor: float,
        min_tenor: float
    ) -> np.ndarray:
        """
        Objective function for full spread calibration.
        
        Parameters
        ----------
        params : np.ndarray
            Parameter values
        params_activation : List[bool]
            Parameter activation flags
        max_tenor : float
            Maximum tenor to include
        min_tenor : float
            Minimum tenor to include
            
        Returns
        -------
        np.ndarray
            Array of errors for least squares optimization
        """
        # Update model parameters
        self._update_model_params(params, params_activation)
        
        # Reprice spreads
        self._reprice_spreads(full_a=True)
        
        # Collect errors
        errors = []
        for _, eur_data in self.daily_data.euribor_rate_data.iterrows():
            if min_tenor <= eur_data["TimeToMat"] <= max_tenor:
                time_to_mat = eur_data["TimeToMat"]
                market_a = eur_data["Object"].market_full_a / time_to_mat
                model_a = eur_data["Object"].model_full_a / time_to_mat
                error = market_a - model_a
                errors.append(error)
                
        return np.array(errors)
        
    def swaption_price_objective(
        self,
        params: np.ndarray,
        params_activation: List[bool],
        use_multi_thread: bool = False,
        power: float = 2.0
    ) -> np.ndarray:
        """
        Objective function for swaption price calibration.
        
        Parameters
        ----------
        params : np.ndarray
            Parameter values
        params_activation : List[bool]
            Parameter activation flags
        use_multi_thread : bool, default=False
            Whether to use multi-threading
        power : float, default=2.0
            Power for error calculation
            
        Returns
        -------
        np.ndarray
            Array of errors for least squares optimization
        """
        print(f"Swaption price objective with params: {params}")
        
        # Update model parameters
        self._update_model_params(params, params_activation)
        
        # Reprice swaptions
        if use_multi_thread:
            self._reprice_swaptions_parallel()
        else:
            self._reprice_swaptions()
            
        # Collect errors
        errors = []
        for _, swopt in self.daily_data.swaption_data_cube.iterrows():
            market_price = swopt["Object"].market_price
            model_price = swopt["Object"].model_price
            error = 1e4 * (market_price - model_price)
            errors.append(error)
            
        errors_array = np.array(errors)
        rmse = compute_rmse(errors_array)
        print(f"Swaption Price RMSE: {rmse}")
        
        return errors_array
        
    def swaption_vol_objective(
        self,
        params: np.ndarray,
        params_activation: List[bool],
        use_multi_thread: bool = False,
        power: float = 2.0
    ) -> np.ndarray:
        """
        Objective function for swaption volatility calibration.
        
        Parameters
        ----------
        params : np.ndarray
            Parameter values
        params_activation : List[bool]
            Parameter activation flags
        use_multi_thread : bool, default=False
            Whether to use multi-threading
        power : float, default=2.0
            Power for error calculation
            
        Returns
        -------
        np.ndarray
            Array of errors for least squares optimization
        """
        print(f"Swaption vol objective with params: {params}")
        
        # Update model parameters
        self._update_model_params(params, params_activation)
        
        # Reprice swaptions
        if use_multi_thread:
            self._reprice_swaptions_parallel()
        else:
            self._reprice_swaptions()
            
        # Collect errors
        errors = []
        for _, swopt in self.daily_data.swaption_data_cube.iterrows():
            market_vol = swopt["Object"].market_vol
            model_vol = swopt["Object"].model_vol
            error = 1e4 * (market_vol - model_vol)
            errors.append(error)
            
        errors_array = np.array(errors)
        rmse = compute_rmse(errors_array)
        print(f"Swaption Vol RMSE: {rmse}")
        
        return errors_array
        
    @partial(jit, static_argnums=(0,))
    def compute_rmse(self, errors: Union[np.ndarray, jnp.ndarray]) -> float:
        """
        Compute root mean square error.
        
        Parameters
        ----------
        errors : array-like
            Error values
            
        Returns
        -------
        float
            RMSE value
        """
        return compute_rmse(errors)
        
    # Private helper methods
    def _update_model_params(self, params: np.ndarray, params_activation: List[bool]):
        """Update model parameters (delegates to calibrator)."""
        # This would be implemented to update the model parameters
        # Implementation details depend on the model structure
        pass
        
    def _reprice_bonds(self):
        """Reprice all bonds in the market data."""
        # Implementation for repricing bonds
        pass
        
    def _reprice_spreads(self, full_a: bool = False):
        """Reprice all spreads in the market data."""
        # Implementation for repricing spreads
        pass
        
    def _reprice_swaptions(self):
        """Reprice all swaptions in the market data."""
        # Implementation for repricing swaptions
        pass
        
    def _reprice_swaptions_parallel(self):
        """Reprice all swaptions using parallel processing."""
        # Implementation for parallel repricing
        pass
