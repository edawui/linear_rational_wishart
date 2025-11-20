

# pricing/fx/fourier_fx_pricer.py
"""Fourier transform based FX option pricing."""

from typing import List
from .base import BaseFxPricer
from ...config.constants import NMAX
from ...models.fx.lrw_fx import LRWFxModel

from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool
import threading

import copy
import time 
import psutil
import gc
import os
import numpy as np

class FourierFxPricer(BaseFxPricer):
    """Fourier transform based FX option pricer."""
    
    def __init__(self, fx_model:LRWFxModel, ur: float = 0.5, nmax: int = NMAX):
        """Initialize Fourier pricer."""
        super().__init__(fx_model)
        self.ur = ur
        self.nmax = nmax
    

    def price_options_not_ok(self, maturities, strikes, is_calls, n_processes=None, **kwargs):
        n_processes = 1 #2 # max(1, mp.cpu_count() - 2)
        start_time = time.time()
        
        with Pool(processes=n_processes) as pool:
            args = [(self.fx_model, m, s, c, kwargs) for m, s, c in zip(maturities, strikes, is_calls)]
            prices = pool.starmap(self.price_options_simple, args)
        end_time = time.time()
        # print(f"Pricing completed in {end_time - start_time:.2f} seconds with {n_processes} workers.")
        return prices

    def price_options_not_fast(self, maturities: List[float], strikes: List[float],
                        is_calls: List[bool], max_workers: int = None, **kwargs) -> List[float]:
            """Price multiple FX options with multi-threading (if fx_model is thread-safe)."""
            ur = kwargs.get('ur', self.ur)
            nmax = kwargs.get('nmax', self.nmax)
            
            max_workers=4 if max_workers is None else max_workers
            start_time = time.time()
            def price_single(maturity, strike, is_call):
                self.fx_model.set_option_properties(maturity, strike)
                return self.fx_model.price_fx_option(is_call, ur=ur, nmax=nmax)
    
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                prices = list(executor.map(price_single, maturities, strikes, is_calls))
    
            end_time = time.time()
            # print(f"Pricing completed in {end_time - start_time:.2f} seconds with {max_workers} workers.")
            return prices
    
    def price_options(self, maturities: List[float], strikes: List[float],
                     is_calls: List[bool], **kwargs) -> List[float]:
        """Price multiple FX options using Fourier transform."""
        ur = kwargs.get('ur', self.ur)
        nmax = kwargs.get('nmax', self.nmax)
        
        OLD_PRICING=False # Set to True to use old pricing method
        if OLD_PRICING:
            
            start_time = time.time()
        
            # Limit threads
            os.environ['OMP_NUM_THREADS'] = '1'
               

            prices = np.empty(len(maturities), dtype=np.float64)
            for i, (maturity, strike, is_call) in enumerate(zip(maturities, strikes, is_calls)):
                self.fx_model.set_option_properties(maturity, strike)
                prices[i] = self.fx_model.price_fx_option(is_call, ur=ur, nmax=nmax)

                # Periodic cleanup for large batches
                if i % 100 == 0 and i > 0:
                    gc.collect()

            end_time = time.time()
            # print(f"Pricing completed in {end_time - start_time:.2f} seconds.")
     
            return prices
        else:
            prices = self.fx_model.price_fx_options_vectorized(maturities, strikes, is_call=is_calls)
            return prices
    def get_pricing_method(self) -> str:
        """Get pricing method name."""
        return "FOURIER"
