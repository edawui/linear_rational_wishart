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
from joblib import Parallel, delayed
from multiprocessing import cpu_count
import time
import gc
import os
import warnings

# import gc

from ..pricing.swaption.fourier_pricing import *
from ..models.interest_rate.config import *
from ..models.interest_rate.lrw_model import LRWModel
from ..data.data_market_data import * #MarketData
from ..utils.jax_optimizations import jax_log, jax_exp, jax_sqrt, compute_rmse

from .calibration_weights_simple import *# SwaptionWeights

NB_UNSED_CPU = 4

class ObjectiveFunctions:
    """
    Collection of objective functions for LRW Jump model calibration.
    
    All functions are JAX-optimized for performance.
    """
    
    def __init__(
        self,
        model: LRWModel,
        daily_data: DailyData,
        config#:CalibrationConfig
    ):
        """
        Initialize objective functions.
        
        Parameters
        ----------
        model : LRWModel
            The model being calibrated
        daily_data : DailyData
            Market data for calibration
        """
        self.model = model
        self.daily_data = daily_data
        self.config= config
        ##These are set later in calibrator
        self.ois_curve=None
        self.ibor_curve=None
        # Precompute weights once
        self._swaption_weights_array = None
        self._ois_weights_array = None
        self._spread_weights_array = None
    
    def precompute_instruments_weights(self, min_tenor , max_tenor, ois_weights: SimpleWeights = None,
                                   spread_weights: SimpleWeights = None,
                                   swaption_weights: SwaptionWeights = None):
        """Call once before calibration starts."""
        self.precompute_ois_weights( min_tenor , max_tenor,ois_weights)
        self.precompute_spread_weights( min_tenor , max_tenor,spread_weights)
        self.precompute_swaption_weights(swaption_weights)
        print("Precomputed instrument weights for calibration objectives.")
        print(f"OIS weights array: {self._ois_weights_array}")
        print(f"Spread weights array: {self._spread_weights_array}")    
        print(f"Swaption weights array: {self._swaption_weights_array}")


    def precompute_ois_weights(self, min_tenor, max_tenor, weights: SimpleWeights = None):
        """Call once before calibration starts."""
        weights = weights or SimpleWeights.uniform()
        tenors = [
            row["TimeToMat"]  # or row["Object"].TimeToMat depending on your structure
            for _, row in self.daily_data.ois_rate_data.iterrows()
            if min_tenor <= row["TimeToMat"] <= max_tenor
        ]
        self._ois_weights_array = np.sqrt(weights.get_weights(tenors))

    def precompute_spread_weights(self, min_tenor, max_tenor, weights: SimpleWeights = None):
        """Call once before calibration starts."""
        weights = weights or SimpleWeights.uniform()
        if self.daily_data.euribor_rate_data.empty:
            self._spread_weights_array = np.array([])
         
        tenors = [
            row["TimeToMat"]
            for _, row in self.daily_data.euribor_rate_data.iterrows()
            if min_tenor <= row["TimeToMat"] <= max_tenor
        ]
        self._spread_weights_array = np.sqrt(weights.get_weights(tenors))

    def precompute_swaption_weights(self, weights: SwaptionWeights = None):
        """Call once before calibration starts."""
        weights = weights or SwaptionWeights.uniform()
        # default_weight=0.01#0.25#0.01#0.25
        # default_tenor=5#2.0# 
        # # w = SwaptionWeights.custom(
        # #     weight_dict={
        # #         (1.0, 5.0): 1.0,  (2.0, 5.0): 1.0,  (3.0, 5.0): 1.0,  (5.0, 5.0): 1.0,  # xxY x 5Y
        # #         (1.0, 2.0): default_weight,  (2.0, 2.0): default_weight,  (3.0, 2.0):default_weight,  (5.0, 2.0): default_weight,  # xxY x 2Y
        # #         # (1.0, 2.0): 0.7,  (2.0, 2.0): 0.7,  (3.0, 2.0): 0.7,  (5.0, 2.0): 0.7,  # xxY x 2Y
        # #     },
        # #     default=default_weight#0.3  # other_weight
        # # )
        # w = SwaptionWeights.custom(
        #     weight_dict={
        #         (1.0, default_tenor): 1.0,  (2.0, default_tenor): 1.0,  (3.0, default_tenor): 1.0,  (5.0, default_tenor): 1.0,  # xxY x default_tenor
        #         (1.0, 2.0): default_weight,  (2.0, 2.0): default_weight,  (3.0, 2.0):default_weight,  (5.0, 2.0): default_weight,  # xxY x 2Y
        #         # (1.0, 2.0): 0.7,  (2.0, 2.0): 0.7,  (3.0, 2.0): 0.7,  (5.0, 2.0): 0.7,  # xxY x 2Y
        #     },
        #     default=default_weight#0.3  # other_weight
        # )

        # # w = SwaptionWeights.custom(
        # #     weight_dict={
        # #         (2.0, 2.0): 1.0,  (2.0, 3.0): 1.0,  (2.0, 4.0): 1.0,  (2.0, 5.0): 1.0,  # xxY x 5Y
        # #         #(1.0, 2.0): default_weight,  (2.0, 2.0): 0.25,  (3.0, 2.0):default_weight,  (5.0, 2.0): default_weight,  # xxY x 2Y
        # #         },
        # #     default=default_weight#0.3  # other_weight
        # # )

        weights = weights or w

        swaption_info = [
            (row["Object"].expiry_maturity, row["Object"].swap_tenor_maturity, row["Object"].strike_offset)
            for _, row in self.daily_data.swaption_data_cube.iterrows()
        ]
        self._swaption_weights_array = np.sqrt(weights.get_weights(swaption_info))

        

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
    
    def _ois_zc_price_errors( self,
                    max_tenor: float,
                    min_tenor: float,
                    multiplier: float = 1.0):
        """Placeholder for OIS error calculation."""
        # errors = []
        # for _, ois_data in self.daily_data.ois_rate_data.iterrows():
        #     if min_tenor <= ois_data["TimeToMat"] <= max_tenor:
        #         market_price = ois_data["Object"].market_zc_price
        #         model_price = ois_data["Object"].model_zc_price
        #         error = (market_price - model_price)*multiplier
        #         errors.append(error)
        
        errors = np.array([ (ois_data["Object"].market_zc_price - ois_data["Object"].model_zc_price)*multiplier
                           for _, ois_data in self.daily_data.ois_rate_data.iterrows()
                           if min_tenor <= ois_data["TimeToMat"] <= max_tenor])

        if self._ois_weights_array is not None:
            errors = errors * self._ois_weights_array
        return errors 
    
    def _ois_zc_rate_errors( self,
                    max_tenor: float,
                    min_tenor: float,
                    multiplier: float = 1.0):
        """Placeholder for OIS error calculation."""
        # errors = []
        # for _, ois_data in self.daily_data.ois_rate_data.iterrows():
        #     if min_tenor <= ois_data["TimeToMat"] <= max_tenor:
        #         market_rate = ois_data["Object"].market_zc_rate
        #         model_rate = ois_data["Object"].model_zc_rate
        #         error = (market_rate - model_rate)*multiplier
        #         errors.append(error)
        errors = np.array([ (ois_data["Object"].market_zc_rate - ois_data["Object"].model_zc_rate)*multiplier
                           for _, ois_data in self.daily_data.ois_rate_data.iterrows()
                           if min_tenor <= ois_data["TimeToMat"] <= max_tenor])
    
        if self._ois_weights_array is not None:
            errors = errors * self._ois_weights_array
        return np.array(errors)
    
    def _spread_full_a_errors( self,
                    max_tenor: float,
                    min_tenor: float,
                    multiplier: float = 1.0):
        """Placeholder for OIS error calculation."""
        # errors = []
        # for _, eur_data in self.daily_data.euribor_rate_data.iterrows():
        #     # print(f"max_tenor={max_tenor}, min_tenor={min_tenor}")
        #     if min_tenor <= eur_data["TimeToMat"] <= max_tenor:
        #         time_to_mat = eur_data["TimeToMat"]
        #         # print(f"time_to_mat={time_to_mat}") 
        #         market_a = eur_data["Object"].market_full_a / time_to_mat
        #         model_a = eur_data["Object"].model_full_a / time_to_mat
        #         if model_a>0.0:
        #             error = (market_a - model_a)*multiplier
        #         errors.append(error)
        if self.daily_data.euribor_rate_data.empty:
            warnings.warn("No IBOR data,", UserWarning)
            return np.array([])  # or appropriate empty result
        
        errors = np.array([(eur_data["Object"].market_full_a / eur_data["TimeToMat"] - eur_data["Object"].model_full_a / eur_data["TimeToMat"])*multiplier
                          for _, eur_data in self.daily_data.euribor_rate_data.iterrows()
                          if (min_tenor <= eur_data["TimeToMat"] <= max_tenor) and (eur_data["Object"].model_full_a >0.0)])

        if self._spread_weights_array is not None:
            errors = errors * self._spread_weights_array
        return np.array(errors)
    
    def _spread_aggregate_a_errors( self,
                    max_tenor: float,
                    min_tenor: float,
                    multiplier: float = 1.0):
        """Placeholder for OIS error calculation."""
        # errors = []
        # for _, eur_data in self.daily_data.euribor_rate_data.iterrows():
        #     if min_tenor <= eur_data["TimeToMat"] <= max_tenor:
        #         time_to_mat = eur_data["TimeToMat"]
        #         market_a = eur_data["Object"].market_aggregate_a / time_to_mat
        #         model_a = eur_data["Object"].model_aggregate_a / time_to_mat
        #         error = (market_a - model_a)*multiplier
        #         errors.append(error)
        if self.daily_data.euribor_rate_data.empty:
            warnings.warn("No IBOR data,", UserWarning)
            return np.array([])  # or appropriate empty result

        errors = np.array([(eur_data["Object"].market_aggregate_a / eur_data["TimeToMat"] - eur_data["Object"].model_aggregate_a / eur_data["TimeToMat"])*multiplier
                          for _, eur_data in self.daily_data.euribor_rate_data.iterrows()
                          if min_tenor <= eur_data["TimeToMat"] <= max_tenor])
        
        if self._spread_weights_array is not None:
            errors = errors * self._spread_weights_array
        return np.array(errors)
    
    def _swaption_price_errors(
            self,
            multiplier: float = 1.0
            ) :
        # errors = []
        # for _, swopt in self.daily_data.swaption_data_cube.iterrows():
        #     market_price = swopt["Object"].market_price
        #     model_price = swopt["Object"].model_price
        #     error =  (market_price - model_price)*multiplier
        #     errors.append(error)
        errors= np.array([ (swopt["Object"].market_price - swopt["Object"].model_price )*multiplier
                          for _, swopt in self.daily_data.swaption_data_cube.iterrows()])
      
        if self._swaption_weights_array is not None:
            errors = errors * self._swaption_weights_array
        return errors
    
    def _swaption_vol_errors(
            self,
            multiplier: float = 1.0
            ):
        
        # errors = []
        # for _, swopt in self.daily_data.swaption_data_cube.iterrows():
        #     market_vol = swopt["Object"].vol
        #     model_vol = swopt["Object"].model_vol
        #     error =  (market_vol - model_vol)*multiplier
        #     errors.append(error)
        #     # print(f"Market Vol={market_vol:.6f}, Model Vol={model_vol:.6f}, Error={error:.6f}")
        #     # print(str(swopt["Object"]))

        errors= np.array([ (swopt["Object"].vol - swopt["Object"].model_vol )*multiplier
                          for _, swopt in self.daily_data.swaption_data_cube.iterrows()])
        if self._swaption_weights_array is not None:
            errors = errors * self._swaption_weights_array
        return errors
      
    def ois_price_and_spread_full_a_objective(self,
        params: np.ndarray,
        params_activation: List[bool],
        max_tenor: float,
        min_tenor: float,
        power: float = 2.0):
        self._update_model_params(params, params_activation)
        # self.model.print_model()
        # print(f"model params={self.model.print_model()}")
        
        # Reprice bonds
        self._reprice_bonds()
        
        # Collect errors
        errors=self._ois_zc_price_errors( 
                    max_tenor,
                    min_tenor,
                    multiplier=1.0)
        # errors = []
        # for _, ois_data in self.daily_data.ois_rate_data.iterrows():
        #     if min_tenor <= ois_data["TimeToMat"] <= max_tenor:
        #         market_price = ois_data["Object"].market_zc_price
        #         model_price = ois_data["Object"].model_zc_price
        #         error = market_price - model_price
        #         errors.append(error)
        
        self._reprice_spreads(full_a=True)
        
        # Collect errors
        
        # for _, eur_data in self.daily_data.euribor_rate_data.iterrows():
        #     # print(f"max_tenor={max_tenor}, min_tenor={min_tenor}")
        #     if min_tenor <= eur_data["TimeToMat"] <= max_tenor:
        #         time_to_mat = eur_data["TimeToMat"]
        #         # print(f"time_to_mat={time_to_mat}") 
        #         market_a = eur_data["Object"].market_full_a / time_to_mat
        #         model_a = eur_data["Object"].model_full_a / time_to_mat
        #         if model_a>0.0:
        #             error = market_a - model_a
        #         errors.append(error)
        errors_spread=self._spread_full_a_errors( 
                    max_tenor,
                    min_tenor,
                    multiplier=1.0)
        errors.append(errors_spread)
        return np.array(errors)

    def ois_price_and_spread_aggregate_a_objective(self,
        params: np.ndarray,
        params_activation: List[bool],
        max_tenor: float,
        min_tenor: float,
        power: float = 2.0):
        self._update_model_params(params, params_activation)
        # self.model.print_model()
        # print(f"model params={self.model.print_model()}")
        
        # Reprice bonds
        self._reprice_bonds()
        
        errors=self._ois_zc_price_errors( 
                    max_tenor,
                    min_tenor,
                    multiplier=1.0)
        # # Collect errors
        # errors = []
        # for _, ois_data in self.daily_data.ois_rate_data.iterrows():
        #     if min_tenor <= ois_data["TimeToMat"] <= max_tenor:
        #         market_price = ois_data["Object"].market_zc_price
        #         model_price = ois_data["Object"].model_zc_price
        #         error = market_price - model_price
        #         errors.append(error)
        
        self._reprice_spreads(full_a=False)
        
        # Collect errors
        
        # for _, eur_data in self.daily_data.euribor_rate_data.iterrows():
        #     if min_tenor <= eur_data["TimeToMat"] <= max_tenor:
        #         time_to_mat = eur_data["TimeToMat"]
        #         market_a = eur_data["Object"].market_aggregate_a / time_to_mat
        #         model_a = eur_data["Object"].model_aggregate_a / time_to_mat
        #         error = market_a - model_a
        #         errors.append(error)
        errors_spread=self._spread_aggregate_a_errors( 
                    max_tenor,
                    min_tenor,
                    multiplier=1.0)
        errors.append(errors_spread)
        return np.array(errors)

    def ois_rate_and_spread_full_a_objective(self,
        params: np.ndarray,
        params_activation: List[bool],
        max_tenor: float,
        min_tenor: float,
        power: float = 2.0):
        self._update_model_params(params, params_activation)
        
        # Reprice bonds
        self._reprice_bonds()
        
        # Collect errors
        errors=self._ois_zc_rate_errors( 
                    max_tenor,
                    min_tenor,
                    multiplier=1.0)
        # errors = []
        # for _, ois_data in self.daily_data.ois_rate_data.iterrows():
        #     if min_tenor <= ois_data["TimeToMat"] <= max_tenor:
        #         market_rate = ois_data["Object"].market_zc_rate
        #         model_rate = ois_data["Object"].model_zc_rate
        #         error = market_rate - model_rate
        #         errors.append(error)
                
        self._reprice_spreads(full_a=True)
        
        # Collect errors
        
        # for _, eur_data in self.daily_data.euribor_rate_data.iterrows():
        #     print(f"max_tenor={max_tenor}, min_tenor={min_tenor}")
        #     if min_tenor <= eur_data["TimeToMat"] <= max_tenor:
        #         time_to_mat = eur_data["TimeToMat"]
        #         market_a = eur_data["Object"].market_full_a / time_to_mat
        #         model_a = eur_data["Object"].model_full_a / time_to_mat
        #         error = market_a - model_a
        #         errors.append(error)
        errors_spread=self._spread_full_a_errors( 
                    max_tenor,
                    min_tenor,
                    multiplier=1.0)
        errors.append(errors_spread)
        return np.array(errors)

    def ois_rate_and_spread_aggregate_a_objective(self,
        params: np.ndarray,
        params_activation: List[bool],
        max_tenor: float,
        min_tenor: float,
        power: float = 2.0):
        self._update_model_params(params, params_activation)
        
        # Reprice bonds
        self._reprice_bonds()
        
        errors=self._ois_zc_rate_errors( 
                    max_tenor,
                    min_tenor,
                    multiplier=1.0)
        # Collect errors
        # errors = []
        # for _, ois_data in self.daily_data.ois_rate_data.iterrows():
        #     if min_tenor <= ois_data["TimeToMat"] <= max_tenor:
        #         market_rate = ois_data["Object"].market_zc_rate
        #         model_rate = ois_data["Object"].model_zc_rate
        #         error = market_rate - model_rate
        #         errors.append(error)

        self._reprice_spreads(full_a=False)
        
        # Collect errors
        
        # for _, eur_data in self.daily_data.euribor_rate_data.iterrows():
        #     if min_tenor <= eur_data["TimeToMat"] <= max_tenor:
        #         time_to_mat = eur_data["TimeToMat"]
        #         market_a = eur_data["Object"].market_aggregate_a / time_to_mat
        #         model_a = eur_data["Object"].model_aggregate_a / time_to_mat
        #         error = market_a - model_a
        #         errors.append(error)
        errors_spread=self._spread_aggregate_a_errors( 
                    max_tenor,
                    min_tenor,
                    multiplier=1.0)
        errors.append(errors_spread)
        return np.array(errors)


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

        # print("*"*60)
        # print(f"params={params}")

        # Update model parameters
        self._update_model_params(params, params_activation)
        # self.model.print_model()
        # print(f"model params={self.model.print_model()}")
        
        # Reprice bonds
        self._reprice_bonds()
        errors=self._ois_zc_price_errors( 
                    max_tenor,
                    min_tenor,
                    multiplier=1.0)
        # # Collect errors
        # errors = []
        # for _, ois_data in self.daily_data.ois_rate_data.iterrows():
        #     if min_tenor <= ois_data["TimeToMat"] <= max_tenor:
        #         market_price = ois_data["Object"].market_zc_price
        #         model_price = ois_data["Object"].model_zc_price
        #         error = market_price - model_price
        #         errors.append(error)
        
        
        # mean_square_errors=  np.sum(np.array(errors)**2)/len(errors)
        # print(f"OIS Price Mean Square Error: {mean_square_errors}")

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
        
        errors=self._ois_zc_rate_errors( 
                    max_tenor,
                    min_tenor,
                    multiplier=1.0)
        # # Collect errors
        # errors = []
        # for _, ois_data in self.daily_data.ois_rate_data.iterrows():
        #     if min_tenor <= ois_data["TimeToMat"] <= max_tenor:
        #         market_rate = ois_data["Object"].market_zc_rate
        #         model_rate = ois_data["Object"].model_zc_rate
        #         error = market_rate - model_rate
        #         errors.append(error)
                
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
        # if not self.model.is_spread: 
        #     print("Model is not a spread model; skipping spread calibration.")
        #     return  np.array([])
        # Update model parameters
        self._update_model_params(params, params_activation)
        
        # Reprice spreads
        self._reprice_spreads(full_a=False)
        
        errors=self._ois_zc_price_errors( 
                    max_tenor,
                    min_tenor,
                    multiplier=1.0)
        # Collect errors
        # errors = []
        # for _, eur_data in self.daily_data.euribor_rate_data.iterrows():
        #     if min_tenor <= eur_data["TimeToMat"] <= max_tenor:
        #         time_to_mat = eur_data["TimeToMat"]
        #         market_a = eur_data["Object"].market_aggregate_a / time_to_mat
        #         model_a = eur_data["Object"].model_aggregate_a / time_to_mat
        #         error = market_a - model_a
        #         errors.append(error)
                
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
        # if not self.model.is_spread: 
        #     print("Model is not a spread model; skipping spread calibration.")
        #     return  np.array([])
        # Update model parameters
        self._update_model_params(params, params_activation)
        
        # Reprice spreads
        self._reprice_spreads(full_a=True)
        
        # Collect errors
        # errors = []
        # for _, eur_data in self.daily_data.euribor_rate_data.iterrows():
        #     if min_tenor <= eur_data["TimeToMat"] <= max_tenor:
        #         time_to_mat = eur_data["TimeToMat"]
        #         market_a = eur_data["Object"].market_full_a / time_to_mat
        #         model_a = eur_data["Object"].model_full_a / time_to_mat
        #         error = market_a - model_a
        #         errors.append(error)
        errors=self._spread_full_a_errors( 
                    max_tenor,
                    min_tenor,
                    multiplier=1.0)
          
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
        # print(f"Swaption price objective with params: {params}")
        
        # Update model parameters
        # print(f"params={params}, params_activation ={params_activation}")
        self._update_model_params(params, params_activation)
        
        # self.model.print_model()
        
        # # Reprice swaptions
        # if use_multi_thread:
        #     self._reprice_swaptions_parallel()
        # else:
        #     self._reprice_swaptions()
        self._reprice_swaptions()

        errors=self._swaption_price_errors(multiplier=1.0) 
        # # Collect errors
        # errors = []
        # for _, swopt in self.daily_data.swaption_data_cube.iterrows():
        #     market_price = swopt["Object"].market_price
        #     model_price = swopt["Object"].model_price
        #     error =  (market_price - model_price)#*1e4
        #     errors.append(error)
        #     # print(str(swopt["Object"]))
            
        errors_array = np.array(errors)
        rmse = compute_rmse(errors_array)
        # print(f"Swaption Price RMSE: {rmse}")
        
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
        # print(f"Swaption vol objective with params: {params}")
        
        # print(f"params={params}, params_activation ={params_activation}")
        # Update model parameters
        self._update_model_params(params, params_activation)
        
        # self.model.print_model()

        # Reprice swaptions
        if use_multi_thread:
            self._reprice_swaptions_parallel()
        else:
            self._reprice_swaptions()
        errors=self._swaption_vol_errors(multiplier=1.0)    
        # # Collect errors
        # errors = []
        # for _, swopt in self.daily_data.swaption_data_cube.iterrows():
        #     market_vol = swopt["Object"].vol
        #     model_vol = swopt["Object"].model_vol
        #     error =  (market_vol - model_vol)#*1e4 
        #     errors.append(error)
        #     # print(f"Market Vol={market_vol:.6f}, Model Vol={model_vol:.6f}, Error={error:.6f}")
        #     # print(str(swopt["Object"]))

        errors_array = np.array(errors)
        rmse = compute_rmse(errors_array)
        # print(f"Swaption Vol RMSE: {rmse}")
        
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
   

    def _update_model_params(self, params: np.ndarray, params_activation: List[bool]):
        """Update model parameters based on activation flags."""
        n_active = sum(params_activation)
        if len(params) != n_active:
            raise ValueError("Number of active parameters doesn't match provided values")
        
        # Extract current parameters
        alpha = self.model.alpha
        x0 = np.array(self.model.x0.copy())
        omega = np.array(self.model.omega.copy())
        m = np.array(self.model.m.copy())
        sigma = np.array(self.model.sigma.copy())
    
        idx = 0
    
        # Alpha (index 0)
        if params_activation[0]:
            alpha = params[idx]
            idx += 1
        
        # x0 diagonal (indices 1, 2)
        start = 1
        for i in range(self.model.n):
            if params_activation[start + i]:
                x0[i, i] = params[idx]
                idx += 1
            
        # x0 off-diagonal (index 3)
        if params_activation[start + 2]:
            if self.config.calibrate_based_on_correl:
                x0[0, 1] = x0[1, 0] = params[idx] * np.sqrt(x0[0, 0] * x0[1, 1])
            else:
                x0[0, 1] = x0[1, 0] = params[idx]
            idx += 1
        
        # Omega diagonal (indices 4, 5)
        start = 4
        for i in range(self.model.n):
            if params_activation[start + i]:
                omega[i, i] = params[idx]
                idx += 1
            
        # Omega off-diagonal (index 6)
        if params_activation[start + 2]:
            if self.config.calibrate_based_on_correl:
                omega[0, 1] = omega[1, 0] = params[idx] * np.sqrt(omega[0, 0] * omega[1, 1])
            else:
                omega[0, 1] = omega[1, 0] = params[idx]
            idx += 1
        
        # M diagonal (indices 7, 8)
        start = 7
        for i in range(self.model.n):
            if params_activation[start + i]:
                m[i, i] = params[idx]
                idx += 1
            
        # Sigma diagonal (indices 9, 10)
        start = 9
        for i in range(self.model.n):
            if params_activation[start + i]:
                sigma[i, i] = params[idx]
                idx += 1
            
        # Sigma off-diagonal (index 11)
        if params_activation[start + 2]:
            if self.config.calibrate_based_on_correl:
                sigma[0, 1] = sigma[1, 0] = params[idx] * np.sqrt(sigma[0, 0] * sigma[1, 1])
            else:
                sigma[0, 1] = sigma[1, 0] = params[idx]
            idx += 1
        
        # Update model
        self.model.set_model_params(self.model.n, alpha, x0, omega, m, sigma)


    def reprice_bond_market_data(self):
        """Reprice all bonds in the market data."""
        for index, ois_data in self.daily_data.ois_rate_data.iterrows():
            rate_data = ois_data["Object"]
            time_to_mat = ois_data["TimeToMat"]
        
            # # Market prices from OIS curve
            rate_data.market_zc_price = self.ois_curve.bond_price(time_to_mat)
            rate_data.market_zc_rate = self.ois_curve.bond_zc_rate(time_to_mat)
     
    def reprice_spreads_market_data(self, full_a: bool = False, max_tenor: float = 11.0):
        """Reprice all spreads in the market data."""
        # if not self.model.is_spread: 
        #     print("Model is not a spread model; skipping spread calibration.")
        #     return  
        if self.daily_data.euribor_rate_data.empty:
            warnings.warn("No IBOR data,", UserWarning)
            return None 
        for index, ibor_data in self.daily_data.euribor_rate_data.iterrows():
            rate_data = ibor_data["Object"]
            time_to_mat = ibor_data["TimeToMat"]
        
            if time_to_mat <= max_tenor:
                # Full A calculation
                if full_a:
                    rate_data.market_full_a = self.ibor_curve.getMktFullA(time_to_mat)
            
                # Aggregate A calculation
                rate_data.market_aggregate_a = self.ibor_curve.getMktA(time_to_mat)
                
 
    def _reprice_bonds_old(self):
        """Reprice all bonds in the market data."""

        # self.model.print_model()

        for index, ois_data in self.daily_data.ois_rate_data.iterrows():
            rate_data = ois_data["Object"]
            time_to_mat = ois_data["TimeToMat"]
        
            # # # Market prices from OIS curve
            # rate_data.market_zc_price = self.ois_curve.bond_price(time_to_mat)
            # rate_data.market_zc_rate = self.ois_curve.bond_zc_rate(time_to_mat)
        
            # Model prices
            rate_data.model_zc_price = self.model.bond(time_to_mat)
            rate_data.model_zc_rate  = self.ois_curve.get_rate_from_zc(
                rate_data.model_zc_price, time_to_mat)
            # print(f"Tenor={time_to_mat:.2f}, Mkt Price={rate_data.market_zc_price:.6f}, Model Price={rate_data.model_zc_price:.6f}, Mkt Rate={rate_data.market_zc_rate:.6f}, Model Rate={rate_data.model_zc_rate:.6f}")



    def _reprice_spreads_old(self, full_a: bool = False, max_tenor: float = 11.0):
        """Reprice all spreads in the market data."""

        if self.daily_data.euribor_rate_data.empty:
            warnings.warn("No IBOR data,", UserWarning)
            return None 
        for index, ibor_data in self.daily_data.euribor_rate_data.iterrows():
            rate_data = ibor_data["Object"]
            time_to_mat = ibor_data["TimeToMat"]
        
            if time_to_mat <= max_tenor:
                # Full A calculation
                if full_a:
                     
                    # rate_data.market_full_a = self.ibor_curve.getMktFullA(time_to_mat)
                    rate_data.model_full_a =  self.ibor_curve.getModelFullA(
                        time_to_mat, self.model
                    )
            
                # Aggregate A calculation
                # rate_data.market_aggregate_a = self.ibor_curve.getMktA(time_to_mat)
                rate_data.model_aggregate_a = self.ibor_curve.getModelA(
                    time_to_mat, self.model
                )

                from joblib import Parallel, delayed
    

    def _reprice_bonds_multithread_not_ok(self):
        """Reprice all bonds in the market data, optionally in parallel."""

        # Function that prices a single bond (joblib-friendly)
        def _price_single_bond(rate_data, time_to_mat):
            rate_data.model_zc_price = self.model.bond(time_to_mat)
            rate_data.model_zc_rate = self.ois_curve.get_rate_from_zc(
                rate_data.model_zc_price, time_to_mat
            )
            return rate_data

        df = self.daily_data.ois_rate_data

        # if self.config.parralel_run:
        if self.config.use_multi_thread:
            # print("Starting parallel bond repricing...")
            n_workers = max(1, os.cpu_count() - NB_UNSED_CPU)
            print(f"Using {n_workers} workers out of {os.cpu_count()} for bond repricing.")
            results = Parallel(
                n_jobs=n_workers,
                backend="loky",
                verbose=0#10 if self.config.verbose else 0
            )(
                delayed(_price_single_bond)(row["Object"], row["TimeToMat"])
                for _, row in df.iterrows()
            )

            # Put results back
            self.daily_data.ois_rate_data["Object"] = results

            gc.collect()
            if hasattr(jax, "clear_caches"):
                jax.clear_caches()

        else:
            # Sequential execution
            for _, row in df.iterrows():
                rate_data = row["Object"]
                t = row["TimeToMat"]

                rate_data.model_zc_price = self.model.bond(t)
                rate_data.model_zc_rate = self.ois_curve.get_rate_from_zc(
                    rate_data.model_zc_price, t
                )
    
    def _reprice_bonds(self):
        """Reprice all bonds using JAX vectorization."""
        df = self.daily_data.ois_rate_data
    
        # Extract all maturities as a JAX array
        maturities = jnp.array(df["TimeToMat"].values)
    
        # Vectorize the bond pricing function
        bond_vmap = jax.vmap(self.model.bond)
        all_prices = bond_vmap(maturities)
    
        # Update objects (sequential, but the heavy lifting is done)
        for i, (_, row) in enumerate(df.iterrows()):
            rate_data = row["Object"]
            rate_data.model_zc_price = float(all_prices[i])
            rate_data.model_zc_rate = self.ois_curve.get_rate_from_zc(
                rate_data.model_zc_price, maturities[i]
            )

    def _reprice_spreads_multithread_not_ok(self, full_a: bool = False, max_tenor: float = 11.0):
        """Reprice all spreads in the market data, optionally in parallel."""
        if self.daily_data.euribor_rate_data.empty:
            warnings.warn("No IBOR data,", UserWarning)
            return None 
        
        def _price_single_spread(rate_data, time_to_mat):
            if time_to_mat <= max_tenor:

                if full_a:
                    rate_data.model_full_a = self.ibor_curve.getModelFullA(
                        time_to_mat, self.model
                    )

                rate_data.model_aggregate_a = self.ibor_curve.getModelA(
                    time_to_mat, self.model
                )

            return rate_data

        df = self.daily_data.euribor_rate_data

        # if self.config.parralel_run:
        if self.config.use_multi_thread:

            # print("Starting parallel Spread repricing...")
            n_workers = max(1, os.cpu_count() - NB_UNSED_CPU)

            results = Parallel(
                n_jobs=n_workers,
                backend="loky",
                verbose=0#10 if self.config.verbose else 0
            )(
                delayed(_price_single_spread)(row["Object"], row["TimeToMat"])
                for _, row in df.iterrows()
            )

            self.daily_data.euribor_rate_data["Object"] = results

            gc.collect()
            if hasattr(jax, "clear_caches"):
                jax.clear_caches()

        else:
            # Sequential path
            for _, row in df.iterrows():
                rate_data = row["Object"]
                t = row["TimeToMat"]

                if t <= max_tenor:

                    if full_a:
                        rate_data.model_full_a = self.ibor_curve.getModelFullA(
                            t, self.model
                        )

                    rate_data.model_aggregate_a = self.ibor_curve.getModelA(
                        t, self.model
                    )

    def _reprice_spreads(self, full_a: bool = False, max_tenor: float = 11.0):
        """Reprice all spreads using JAX vectorization."""
        # if not self.model.is_spread: 
        #     print("Model is not a spread model; skipping spread calibration.")
        #     return  
        if self.daily_data.euribor_rate_data.empty:
            warnings.warn("No IBOR data,", UserWarning)
            return np.array([])  # or appropriate empty result
        
        df = self.daily_data.euribor_rate_data
    
        # Extract maturities and filter by max_tenor
        all_maturities = np.array(df["TimeToMat"].values)
        valid_mask = all_maturities <= max_tenor
    
        if not np.any(valid_mask):
            return
    
        valid_maturities = jnp.array(all_maturities[valid_mask])
    
        # Create vmapped pricing functions
        def compute_aggregate_a(t):
            return self.ibor_curve.getModelA_vmap_compatible(t, self.model)
    
        def compute_full_a(t):
            return self.ibor_curve.getModelFullA_vmap_compatible(t, self.model)
    
        # Vectorized computation
        getModelA_vmap = jax.vmap(compute_aggregate_a)
        all_aggregate_a = getModelA_vmap(valid_maturities)
    
        if full_a:
            getModelFullA_vmap = jax.vmap(compute_full_a)
            all_full_a = getModelFullA_vmap(valid_maturities)
    
        # Update objects (lightweight assignment loop)
        valid_indices = np.where(valid_mask)[0]
        for i, idx in enumerate(valid_indices):
            rate_data = df.iloc[idx]["Object"]
            rate_data.model_aggregate_a = float(all_aggregate_a[i])
            if full_a:
                rate_data.model_full_a = float(all_full_a[i])

    def _reprice_swaptions(self):
        """Reprice all swaptions in the market data."""
        import time
    
        start_time = time.perf_counter()
    
        for index, swaption_data in self.daily_data.swaption_data_cube.iterrows():
            swaption = swaption_data["Object"]
            self._price_single_swaption(swaption)
    
        end_time = time.perf_counter()
        pricing_time = end_time - start_time
    
        # if self.config.verbose:
        #     print(f"All swaptions pricing time = {pricing_time:.2f}s")


    def _reprice_swaptions_parallel_old(self):
        """Reprice all swaptions using parallel processing."""
        import time
        from multiprocessing import Pool, cpu_count
    
        start_time = time.perf_counter()
    
        n_processes = max(1, cpu_count() - 2)
        swaption_list = self.daily_data.swaption_data_cube["Object"].tolist()
    
        with Pool(processes=n_processes) as pool:
            results = pool.map(self._price_single_swaption, swaption_list)
    
        # Update the dataframe with results
        self.daily_data.swaption_data_cube["Object"] = results
    
        end_time = time.perf_counter()
        pricing_time = end_time - start_time
    
        if self.config.verbose:
            print(f"MultiThread: All swaptions pricing time = {pricing_time:.2f}s")

   

    def _reprice_swaptions_parallel(self):
        """Reprice all swaptions using joblib with the loky backend (safe for CPU + JAX)."""

        start_time = time.perf_counter()

        # Use N-2 workers to keep OS responsive
        n_workers = max(1, cpu_count() - 2)

        # Extract objects to price
        swaption_list = self.daily_data.swaption_data_cube["Object"].tolist()

        if self.config.verbose:
            print(f"Starting swaption repricing with {n_workers} workers (loky backend).")

        # Use joblib with loky (process-based parallelism)
        results = Parallel(
            n_jobs=n_workers,
            backend="loky",
            verbose=10 if self.config.verbose else 0
        )(
            delayed(self._price_single_swaption)(swpt)
            for swpt in swaption_list
        )

        # Update results in the dataframe
        self.daily_data.swaption_data_cube["Object"] = results

        # Clean memory after joblib processes
        gc.collect()
        if hasattr(jax, "clear_caches"):
            jax.clear_caches()

        end_time = time.perf_counter()
        pricing_time = end_time - start_time

        if self.config.verbose:
            print(f"Loky Parallel: All swaptions pricing time = {pricing_time:.2f}s")

    def _price_single_swaption(self, swaption) -> object:
        """
        Price a single swaption and update its model values.
    
        Parameters
        ----------
        swaption : SwaptionData
            Swaption data object to price
        
        Returns
        -------
        SwaptionData
            Updated swaption with model price and implied vol
        """
        import time
    
        # # Set option properties on model
        # self.model.set_option_properties(
        #     swaption.swap_tenor_maturity,
        #     swaption.expiry_maturity,
        #     self.daily_data.delta_float_leg,
        #     self.daily_data.delta_fixed_leg,
        #     swaption.strike
        # )
         
        self.model.set_swaption_config(
                            SwaptionConfig(
                            maturity=swaption.expiry_maturity ,
                            tenor=swaption.swap_tenor_maturity,
                            strike=swaption.strike,            
                            delta_float=self.daily_data.delta_float_leg,
                            delta_fixed=self.daily_data.delta_fixed_leg
                            #,call=swaption.is_call
                                ))
        # Price the swaption
        pricer = FourierPricer(self.model)
        swaption.model_price = pricer.price()

        # swaption.model_price = self.model.price_swaption()
    
        # Calculate implied volatility
        model_implied_vol = self._compute_implied_vol(
            forward=swaption.market_forward_swap_rate,##mkt_forward_swap_rate,
            strike=swaption.strike,
            expiry=swaption.expiry_maturity,
            price=swaption.model_price,
            annuity=swaption.model_annuity
        )
    
        swaption.model_vol = model_implied_vol
    
        return swaption


    def _compute_implied_vol(
        self,
        forward: float,
        strike: float,
        expiry: float,
        price: float,
        annuity: float,
        call_or_put: float = 1.0,
        epsilon: float = 1e-6
    ) -> float:
        """
        Compute implied volatility from price using Bachelier model.
    
        Parameters
        ----------
        forward : float
            Forward rate
        strike : float
            Strike price
        expiry : float
            Time to expiry
        price : float
            Option price
        annuity : float
            Annuity factor
        call_or_put : float
            1.0 for call, -1.0 for put
        epsilon : float
            Convergence tolerance
        
        Returns
        -------
        float
            Implied volatility
        """
        # Import or use existing implied vol function
        # from .bachelier_model import implied_vol
    
        # return implied_vol(
        #     forward, strike, expiry, price, annuity,
        #     call_or_put=call_or_put, epsilon=epsilon
        #)
        from ..pricing.bachelier import implied_normal_volatility

        return implied_normal_volatility(
            forward, strike, expiry, price            
            , call_or_put
            , numeraire=annuity
            , epsilon=epsilon
        )