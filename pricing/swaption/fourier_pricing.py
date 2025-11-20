

# pricing/swaption/fourier_pricing.py
"""Fourier transform methods for swaption pricing."""

import math
import cmath
import jax
from typing import Optional, Dict, Any
import scipy.integrate as sp_i
from functools import partial
from jax import jit
import jax.numpy as jnp
from multiprocessing import Pool, cpu_count
import numpy as np

from .base import BaseSwaptionPricer
from ...utils.local_functions import tr_uv

# import numpy as np


import numpy as np

def gauss_legendre_integral(f, a, b, n=32):
    # """
    # Integrate f over [a,b] using n-point Gauss-Legendre quadrature.
    # Calls f only with scalar inputs — safe for non-vectorized integrands.
    # """
    x, w = np.polynomial.legendre.leggauss(n)

    # map nodes from [-1,1] to [a,b]
    t = 0.5 * (x + 1) * (b - a) + a
    scale = 0.5 * (b - a)

    # accumulate sum with scalar evaluations
    total = 0.0
    for ti, wi in zip(t, w):
        total += wi * f(ti)

    return scale * total



# Put this at the module level (outside any class)
def _integrate_chunk_parallel(args):
    """Standalone function that can be pickled."""
    start, end, ur, a3, b3, use_range_kutta, phi_one_func, phi_one_approx_func, epsabs, epsrel = args
    
    def integrand(ui):
        u = complex(ur, ui)
        z = u
        z_a3 = z * a3
        exp_z_b3 = cmath.exp(z * b3)
        
        if use_range_kutta:
            phi1 = phi_one_func(1, z_a3)
        else:
            phi1 = phi_one_approx_func(1, z_a3)
            
        result = exp_z_b3 * phi1 / (z * z)
        return result.real
    
    return sp_i.quad(integrand, start, end, epsabs=epsabs, epsrel=epsrel)


class FourierPricer(BaseSwaptionPricer):
    """Fourier transform based swaption pricing."""
    
    def __init__(self, model, use_range_kutta: bool = True):
        """Initialize Fourier pricer."""
        super().__init__(model)
        self.use_range_kutta = use_range_kutta
        
        # Default integration parameters
        self.ur = 0.5
        self.nmax = 300
        self.epsabs = 1e-7
        self.epsrel = 1e-5
        # print(f"Initialized FourierPricer with u1={self.model.u1}, u2={self.model.u2}")

    

    def price_parallel(self, ur: float = None, nmax: int = None, 
          recompute_a3_b3: bool = True, n_workers: int = None) -> float:
            """Price swaption using parallel integration."""
            self.validate_inputs()
    
            if ur is not None:
                self.ur = ur
            if nmax is not None:
                self.nmax = nmax
            if n_workers is None:
                # n_workers = min(cpu_count() - 1, 8)
                n_workers = max(1, cpu_count() // 2)  # Use half the cores

        
            if recompute_a3_b3:
                self.model.compute_b3_a3()
    
            # Split domain
            splits = np.linspace(0, self.nmax, n_workers + 1)
    
            # Prepare arguments for each worker
            args_list = [
                (splits[i], splits[i+1], 
                 self.ur, self.model.a3, self.model.b3,
                 True,#self.use_range_kutta,
                 self.model.wishart.phi_one,  # Pass method reference
                  self.model.wishart.phi_one,  #self.model.wishart.phi_one_approx_b,  # Pass method reference
                 self.epsabs/n_workers, self.epsrel)
                for i in range(n_workers)
            ]
    
            # Parallel integration
            with Pool(n_workers) as pool:
                results = pool.map(_integrate_chunk_parallel, args_list)
    
            # Combine results
            integral_result = sum(r[0] for r in results)
            total_error = np.sqrt(sum(r[1]**2 for r in results))
    
            # Scale result
            price = integral_result / math.pi
            price *= math.exp(-self.model.alpha * self.model.maturity)
            price /= (1 + tr_uv(self.model.u1, self.model.x0))
    
            self.last_integration_error = total_error
    
            return price

    def price(self, ur: float = None, nmax: int = None, 
              recompute_a3_b3: bool = True) -> float:
        """Price swaption using Fourier transform."""
        self.validate_inputs()
        

        if ur is not None:
            self.ur = ur
        if nmax is not None:
            self.nmax = nmax
        # print(f"self.nmax={self.nmax}")
        
        # return self.price_parallel()
        # return self.price_with_intervals_gauss_legendre()
        # return self.price_with_intervals()
        # return self.price_with_intervals_new()
        return self.price_with_intervals_hybrid()

        if recompute_a3_b3:
            self.model.compute_b3_a3()
        
        # print(f"FourierPricer.price: x0={self.model.x0},a3={self.model.a3}, b3={self.model.b3}")
        # print(f"self.nmax={self.nmax}")
        # Define integrand
        def integrand(ui):
            u = complex(self.ur, ui)
            z = u
            
            z_a3 = z * self.model.a3
            exp_z_b3 = cmath.exp(z * self.model.b3)
            
            if self.use_range_kutta:
                phi1 = self.model.wishart.phi_one(1, z_a3)
            else:
                phi1 = self.model.wishart.phi_one_approx_b(1, z_a3)
                
            result = exp_z_b3 * phi1 / (z * z)
            return result.real
            
        # Numerical integration
        integral_result, error = sp_i.quad(integrand, 0, self.nmax, 
                                          epsabs=self.epsabs, epsrel=self.epsrel)
        
        # Scale result
        price = integral_result / math.pi
        price *= math.exp(-self.model.alpha * self.model.maturity)
        price /= (1 + tr_uv(self.model.u1, self.model.x0))
        
        self.last_integration_error = error
        
        return price

    def price_with_intervals(self, intervals: list = None) -> float:
        """Price using piecewise integration over intervals."""
        self.validate_inputs()
        nb_interval=5##self.nmax/5.0
        # print(f"self.nmax={self.nmax}")
        if intervals is None:
            # intervals = list(np.arange(0.0, self.nmax, nb_interval))
            intervals=np.linspace(0.0, self.nmax, nb_interval + 1).tolist()
            # intervals = [0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, self.nmax]
        
        # print(f"intervals={intervals}")

        self.model.compute_b3_a3()
        
        total_integral = 0.0
        
        for k in range(len(intervals) - 1):
            start = intervals[k]
            end = intervals[k + 1]
            # print(f"start, end={start},{end}")
            def integrand(ui):
                u = complex(self.ur, ui)
                z = u
                
                z_a3 = z * self.model.a3
                exp_z_b3 = cmath.exp(z * self.model.b3)
                phi1 = self.model.wishart.phi_one(1, z_a3)
                
                result = exp_z_b3 * phi1 / (z * z)
                return result.real
                
            integral, _ = sp_i.quad(integrand, start, end, 
                                   epsabs=self.epsabs, epsrel=self.epsrel)
            total_integral += integral
            
        # Scale result
        price = total_integral / math.pi
        price *= math.exp(-self.model.alpha * self.model.maturity)
        price /= (1 + tr_uv(self.model.u1, self.model.x0))
        
        return price
   
    def price_with_intervals_new(self, intervals=None):
        self.validate_inputs()

        # fewer intervals → fewer quad calls
        if intervals is None:
            intervals = np.linspace(0.0, self.nmax, 4).tolist()  
            # only 3 integrals instead of 5 or 6

        self.model.compute_b3_a3()

        a3 = self.model.a3
        b3 = self.model.b3
        ur = self.ur
        phi_one = self.model.wishart.phi_one

        # integrand
        def integrand(ui):
            z = complex(ur, ui)
            return (cmath.exp(z * b3) * phi_one(1, z * a3) / (z * z)).real

        total = 0.0
        for a, b in zip(intervals[:-1], intervals[1:]):
            val, _ = sp_i.quad(
                integrand,
                a, b,
                epsabs=self.epsabs * 0.1,    # relax accuracy slightly
                epsrel=self.epsrel * 0.1
            )
            total += val

        price = total / math.pi
        price *= math.exp(-self.model.alpha * self.model.maturity)
        price /= (1 + tr_uv(self.model.u1, self.model.x0))

        return price

    def price_with_intervals_gauss_legendre(self, intervals=None, n_gauss=64):
        self.validate_inputs()

        # default intervals
        if intervals is None:
            intervals = np.linspace(0.0, self.nmax, 6).tolist()

        # precompute coefficients
        self.model.compute_b3_a3()

        a3 = self.model.a3
        b3 = self.model.b3
        ur = self.ur
        phi_one = self.model.wishart.phi_one

        # integrand
        def integrand(ui):
            # print(f"ui={ui}")
            z = complex (ur, ui)
            return (cmath.exp(z * b3) * phi_one(1, z * a3) / (z * z)).real

        # integrate over all intervals
        total_integral = 0.0
        for a, b in zip(intervals[:-1], intervals[1:]):
            total_integral += gauss_legendre_integral(integrand, a, b, n=n_gauss)

        # scale price
        price = total_integral / math.pi
        price *= math.exp(-self.model.alpha * self.model.maturity)
        price /= (1 + tr_uv(self.model.u1, self.model.x0))

        return price

    def get_pricing_info(self) -> Dict[str, Any]:
        """Get detailed pricing information."""
        return {
            "method": "Fourier Transform",
            "use_range_kutta": self.use_range_kutta,
            "integration_parameter": self.ur,
            "max_integration": self.nmax,
            "last_error": getattr(self, 'last_integration_error', None)
        }

    def price_with_intervals_hybrid(self, intervals=None, n_outer=30):#, n_inner=15):
        """
        Hybrid: Vectorized outer integration + reduced inner integration points
    
        Parameters
        ----------
        n_outer : int
            Number of points for outer (ui) integration
        n_inner : int
            Number of points for inner (time) integration in compute_b
        """
        self.validate_inputs()
    
        if intervals is None:
            intervals = np.linspace(0.0, self.nmax, 4).tolist()
    
        self.model.compute_b3_a3()
        a3 = self.model.a3
        b3 = self.model.b3
        ur = self.ur
    
        # Create vectorized phi_one
        @jax.jit
        def phi_one_batch(z_array):
            """Compute phi_one for multiple z values"""
            return jax.vmap(
                lambda z: self.model.wishart.phi_one(1.0, z * a3)
            )(z_array)
    
        @jax.jit
        def integrand_batch(ui_array):
            """Vectorized integrand"""
            z_array = ur + 1j * ui_array
            exp_zb3 = jnp.exp(z_array * b3)
            phi_values = phi_one_batch(z_array)
            return (exp_zb3 * phi_values / (z_array * z_array)).real
    
        total = 0.0
    
        for a, b in zip(intervals[:-1], intervals[1:]):
            ui_vals = jnp.linspace(a, b, n_outer)
            integrand_vals = integrand_batch(ui_vals)
            val = jnp.trapezoid(integrand_vals, ui_vals)
            total += val
    
        price = float(total) / math.pi
        price *= math.exp(-self.model.alpha * self.model.maturity)
        price /= (1 + tr_uv(self.model.u1, self.model.x0))
    
        return price