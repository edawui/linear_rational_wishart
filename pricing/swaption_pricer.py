"""
Swaption pricing methods for LRW models.

This module provides various pricing methods for swaptions under the
Linear Rational Wishart model including FFT, Monte Carlo, and approximations.
"""

from logging import config
from typing import Optional, Tuple, Union, Dict, List
 
import jax.numpy as jnp
import numpy as np

import time
import gc
import psutil
import jax
import os
import time
import gc


from joblib import Parallel, delayed
from copy import deepcopy
from typing import Optional


from ..models.interest_rate.lrw_model import LRWModel
from ..pricing.mc_pricer import WishartMonteCarloPricer
from ..pricing.swaption.fourier_pricing import FourierPricer 
from ..pricing.swaption.collin_dufresne import CollinDufresnePricer 
from ..pricing.swaption.gamma_approximation import GammaApproximationPricer 
from ..utils.jax_utils import ensure_jax_array
from ..config.constants import *
from ..config.constants import NMAX
from .bachelier import *

class LRWSwaptionPricer:
    """
    Comprehensive swaption pricer for LRW models.
    
    Supports multiple pricing methods:
    - FFT (Fourier transform)
    - Monte Carlo simulation
    - Collin-Dufresne approximation
    - Gamma approximation
    """
    
    def __init__(self, lrw_model: LRWModel):
        """
        Initialize the swaption pricer.
        
        Parameters
        ----------
        lrw_model : LRWModel
            The LRW model instance
        """
        self.model = lrw_model
        # self.mc_pricer = WishartMonteCarloPricer(lrw_model)
        # self.fft_pricer = FourierPricer(lrw_model)
        # self.collin_dufresne_pricer = CollinDufresnePricer(self.model)
        # self.gamma_approximation_pricer = GammaApproximationPricer(self.model)
        # print(f"Initialized LRWSwaptionPricer with u1={self.model.u1}, u2={self.model.u2}")


    def price_swaption(
        self,
        method: str = "fft",
        num_paths: int = 10000,
        dt: float = 0.125,
        return_implied_vol: bool = False
    ) -> Union[float, Tuple[float, float]]:
        """
        Price a swaption using the specified method.
        
        Parameters
        ----------
        method : str, default="fft"
            Pricing method: "fft", "mc", "collin_dufresne", "gamma_approx"
        num_paths : int, default=10000
            Number of Monte Carlo paths (for MC method)
        dt : float, default=0.125
            Time step for Monte Carlo (for MC method)
        return_implied_vol : bool, default=False
            Whether to also return implied volatility
            
        Returns
        -------
        float or Tuple[float, float]
            Swaption price, or (price, implied_vol) if requested
        """
        if method == "fft":
            price = self._price_fft()
        elif method == "mc":
            price = self._price_monte_carlo(num_paths, dt)
        elif method == "collin_dufresne":
            price = self._price_collin_dufresne()
        elif method == "gamma_approx":
            price = self._price_gamma_approximation()
        elif method == "neural_network":
            price = self._price_fft_nn()
        else:
            raise ValueError(f"Unknown pricing method: {method}")
        # print(price)
        if return_implied_vol:
            # implied_vol =  = self.model.ImpliedVol(price)
            implied_vol = implied_normal_volatility(self.model.compute_swap_rate()
                                      , self.model.swaption_config.strike
                                      , self.model.swaption_config.maturity
                                      , price
                                      , "call" if self.model.swaption_config.call else "put"
                                        )#, 'call')#, self.model.swaption_config. ##todo add this to swaption 
            return price, implied_vol
        else:
            return price
            
    def _price_fft_nn(self) -> float:
        """Price using FFT method."""
        # return self.model.PriceOption()
        self.fft_pricer = FourierPricer(self.model)
        price= self.fft_pricer.price_nn(ur=UR, nmax=NMAX, recompute_a3_b3=True)
        # print(f"Model X0={self.model.x0}, maturity ={self.model.swaption_config.maturity}, price={price}")

        return price
        # return self.fft_pricer.price(ur=UR, nmax=NMAX, recompute_a3_b3=True)
        
    def _price_fft(self) -> float:
        """Price using FFT method."""
        # return self.model.PriceOption()
        self.fft_pricer = FourierPricer(self.model)
        price= self.fft_pricer.price(ur=UR, nmax=NMAX, recompute_a3_b3=True)
        # print(f"Model X0={self.model.x0}, maturity ={self.model.swaption_config.maturity}, price={price}")

        return price
        # return self.fft_pricer.price(ur=UR, nmax=NMAX, recompute_a3_b3=True)
        
    def _price_monte_carlo(self, num_paths: int, dt: float) -> float:
        """Price using Monte Carlo simulation."""
        self.mc_pricer = WishartMonteCarloPricer(self.model)

        return self.mc_pricer.price_option_mc(num_paths, dt)
        
    def _price_collin_dufresne(self) -> float:
        """Price using Collin-Dufresne approximation."""
        self.collin_dufresne_pricer = CollinDufresnePricer(self.model)
        return self.collin_dufresne_pricer.price(max_order=3)
        # return self.model.CollindufresneSwaptionPrice()
        
    def _price_gamma_approximation(self) -> float:
        """Price using Gamma approximation."""
        self.gamma_approximation_pricer = GammaApproximationPricer(self.model)
        self.gamma_approximation_pricer.price(k_param=0.01)
        # return self.model.ComputeSwaptionPrice_Gamma_Approximation()
        
    def price_with_all_methods(
        self,
        num_paths: int = 10000,
        dt: float = 0.125
    ) -> Dict[str, Dict[str, float]]:
        """
        Price swaption using all available methods.
        
        Parameters
        ----------
        num_paths : int, default=10000
            Number of Monte Carlo paths
        dt : float, default=0.125
            Time step for Monte Carlo
            
        Returns
        -------
        Dict[str, Dict[str, float]]
            Results for each method with price and implied vol
        """
        results = {}
        
        methods = ["fft", "mc", "collin_dufresne", "gamma_approx"]
        for method in methods:
            try:
                price, iv = self.price_swaption(
                    method=method,
                    num_paths=num_paths,
                    dt=dt,
                    return_implied_vol=True
                )
                results[method] = {"price": price, "implied_vol": iv}
            except Exception as e:
                results[method] = {"error": str(e)}
                
        return results
        
    def compute_atm_strike(self) -> float:
        """
        Compute the at-the-money strike rate.
        
        Returns
        -------
        float
            ATM strike rate
        """
        return self.model.ComputeSwapRate()
    
    def price_with_schemas(
        self,
        num_paths: int = 10000,
        dt: float = 0.125,
        schemas: List[str] = ["EULER_CORRECTED", "EULER_FLOORED", "ALFONSI"]
    ) -> Dict[str, float]:
        """
        Price using different Monte Carlo schemas.
        
        Parameters
        ----------
        num_paths : int, default=10000
            Number of Monte Carlo paths
        dt : float, default=0.125
            Time step
        schemas : List[str]
            List of schemas to use
            
        Returns
        -------
        Dict[str, float]
            Prices for each schema
        """
        results = {}
        
        for schema in schemas:
            try:
                price = self.mc_pricer.PriceOptionMC_withSchema(
                    num_paths, dt, schema=schema
                )
                results[schema] = price
            except Exception as e:
                results[schema] = f"Error: {str(e)}"
                
        return results
    
    def compute_exposure_profile(
        self,
        exposure_dates: jnp.ndarray,
        fixed_rate: float,
        spread: float,
        floating_schedule: List[float],
        fixed_schedule: List[float],
        num_paths: int = 10000,
        dt: float = 0.125,
        schema: str = "EULER_FLOORED"
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute exposure profile for swap.
        
        Parameters
        ----------
        exposure_dates : jnp.ndarray
            Dates to compute exposure
        fixed_rate : float
            Fixed leg rate
        spread : float
            Floating leg spread
        floating_schedule : List[float]
            Floating leg payment dates
        fixed_schedule : List[float]
            Fixed leg payment dates
        num_paths : int
            Number of Monte Carlo paths
        dt : float
            Time step
        schema : str
            Monte Carlo schema
            
        Returns
        -------
        Tuple[jnp.ndarray, jnp.ndarray]
            Mean exposure profile and PFE (95%)
        """
        exposure_profile = self.mc_pricer.ComputeSwapExposureProfile(
            exposure_dates,
            fixed_rate,
            spread,
            floating_schedule,
            fixed_schedule,
            nbMc=num_paths,
            dt=dt,
            mainDate=0.0,
            schema=schema
        )
        
        mean_profile = jnp.mean(exposure_profile, axis=1)
        pfe_95 = jnp.percentile(exposure_profile, 95, axis=1)
        
        return mean_profile, pfe_95

    def price_swap(
        self#,
        # method: str = "fft",
        # num_paths: int = 10000,
        # dt: float = 0.125#,
        # return_implied_vol: bool = False
    ):

        # print("="*20)
        swap_price = self.model.price_swap()
          
    def compute_swaption_exposure_profile(
        self,
        exposure_dates: List[float],
        initial_swaption_config,
        nb_mc: int,
        dt: float,
        main_date: float = 0.0,
        schema: str = "EULER_FLOORED",
        pricing_method: str = "fft",
        batch_size: int = 50 
        #,n_workers: int = 2  # NEW PARAMETER
    ) -> np.ndarray:
        """
        Memory-optimized with pricer recreation to prevent leaks.
    
        Parameters
        ----------
        n_workers : int, default=1
            Number of parallel workers for pricing. Use 1 for PC, 4+ for Colab
        """
       
        n_workers = EXPOSURE_SWAPTION_WORKERS  # Set to desired number of workers
        print("Starting swaption exposure profile computation...")
        print(f"NMAX={NMAX}, UR={UR}")
        print(f"Using {n_workers} workers for pricing")
    
        # Verify JAX settings
        print("\nJAX Configuration Check:")
        print(f"  JIT disabled: {os.environ.get('JAX_DISABLE_JIT', 'NOT SET')}")
        print(f"  Memory fraction: {os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION', 'NOT SET')}")

        gc.set_threshold(100, 5, 5)

        exposure_profile = np.zeros((len(exposure_dates), nb_mc), dtype=np.float64)
        initial_maturity = initial_swaption_config.maturity

        # MC simulation
        print("\nMC simulation...")
        start = time.perf_counter()
        mc_simulator = WishartMonteCarloPricer(self.model)
        sim_results_dict = mc_simulator.simulate(exposure_dates, nb_mc, dt, schema)
        print(f"  Time: {time.perf_counter() - start:.2f}s")

        del mc_simulator
        gc.collect()

        # Convert
        print("Converting results...")
        start = time.perf_counter()
        sim_results = np.array([
            [sim_results_dict[path][t] for t in exposure_dates]
            for path in range(nb_mc)
        ])
        print(f"  Time: {time.perf_counter() - start:.2f}s")

        del sim_results_dict
        gc.collect()

        # Pricing
        print("Pricing...")
        start = time.perf_counter()

        model_config = self.model.model_config
        swaption_config = initial_swaption_config

        # Handle initial date
        if exposure_dates[0] == 0.0:
            config_copy = deepcopy(model_config)
            swaption_copy = deepcopy(swaption_config)
            config_copy.x0 = sim_results[0, 0]
            swaption_copy.maturity = initial_maturity
    
            lrw_model = LRWModel(config_copy, swaption_copy)
            lrw_model.is_spread = self.model.is_spread
            pricer = LRWSwaptionPricer(lrw_model)
    
            initial_price = pricer.price_swaption(
                method=pricing_method,
                num_paths=nb_mc,
                dt=dt,
                return_implied_vol=False
            )
            exposure_profile[0, :] = initial_price
    
            del lrw_model, pricer
            gc.collect()
    
            enumerate_start = 1
        else:
            enumerate_start = 0

        # Define pricing function for a single path
        def price_single_path(x0_value, config_copy, swaption_config_copy, is_spread):
            """Price a single path - used for both sequential and parallel execution"""
            config_copy.x0 = x0_value
            lrw_model = LRWModel(config_copy, swaption_config_copy)
            lrw_model.is_spread = is_spread
            pricer = LRWSwaptionPricer(lrw_model)
        
            price = pricer.price_swaption(
                method=pricing_method,
                num_paths=nb_mc,
                dt=dt,
                return_implied_vol=False
            )
        
            del lrw_model, pricer
            return price

        # Main pricing loop
        for i, valuation_date in enumerate(exposure_dates[enumerate_start:], start=enumerate_start):
            current_maturity = initial_maturity - valuation_date
            if current_maturity <= 0:
                continue
    
            print(f"  Processing date {i+1}/{len(exposure_dates)}: {valuation_date:.4f}")
    
            # Update maturity
            swaption_config.maturity = current_maturity
        
            if n_workers == 1:
                # Sequential processing - but create fresh pricer for each path to match parallel behavior
                for path in range(nb_mc):
                    # Periodic garbage collection
                    if path % batch_size == 0:
                        gc.collect()
                        if hasattr(jax, 'clear_caches'):
                            jax.clear_caches()
                    
                        mem_pct = psutil.virtual_memory().percent
                        print(f"    Path {path}: {mem_pct:.1f}% RAM")
                
                    # Create fresh copies for each path (same as parallel)
                    config_copy = deepcopy(model_config)
                    swaption_copy = deepcopy(swaption_config)
                
                    exposure_profile[i, path] = price_single_path(
                        sim_results[path, i],
                        config_copy,
                        swaption_copy,
                        self.model.is_spread
                    )
        
            else:
                # Parallel processing
                # Process in batches to manage memory
                n_batches = (nb_mc + batch_size - 1) // batch_size
            
                for batch_idx in range(n_batches):
                    start_path = batch_idx * batch_size
                    end_path = min((batch_idx + 1) * batch_size, nb_mc)
                    batch_paths = range(start_path, end_path)
                
                    print(f"    Batch {batch_idx+1}/{n_batches}: paths {start_path}-{end_path-1}")
                
                    # Parallel pricing for this batch
                    results = Parallel(n_jobs=n_workers, backend='threading')(
                        delayed(price_single_path)(
                            sim_results[path, i],
                            deepcopy(model_config),
                            deepcopy(swaption_config),
                            self.model.is_spread
                        )
                        for path in batch_paths
                    )
                
                    # Store results
                    for idx, path in enumerate(batch_paths):
                        exposure_profile[i, path] = results[idx]
                
                    # Cleanup after batch
                    gc.collect()
                    if hasattr(jax, 'clear_caches'):
                        jax.clear_caches()
                
                    mem_pct = psutil.virtual_memory().percent
                    print(f"    After batch {batch_idx+1}: {mem_pct:.1f}% RAM")
    
            gc.collect()
            mem_pct = psutil.virtual_memory().percent
            print(f"  After date {i+1}: {mem_pct:.1f}% RAM")

        print(f"  Pricing time: {time.perf_counter() - start:.2f}s")

        del sim_results
        gc.collect()

        return exposure_profile



