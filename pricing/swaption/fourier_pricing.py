

# pricing/swaption/fourier_pricing.py
"""Fourier transform methods for swaption pricing."""

import math
import cmath

from networkx import omega
import jax
from typing import Optional, Dict, Any
import scipy.integrate as sp_i
from functools import partial
from jax import jit
import jax.numpy as jnp
from multiprocessing import Pool, cpu_count
import numpy as np

import multiprocessing as mp
from joblib import Parallel, delayed
import jax
import jax.numpy as jnp
 

from ...config.constants import *
from .base import BaseSwaptionPricer
from ...utils.local_functions import tr_uv

# import numpy as np

# from ...neural_operator.model import (
#     WishartPINNModel,
#     WishartCharFuncNetwork,
#     HighwayBlock,
#     matrix_to_upper_tri,
#     upper_tri_to_matrix
#     )
from ...neural_operator.inference import (
    WishartPINNInference,
    validate_model,
    benchmark_throughput
)

import numpy as np


def gauss_legendre_integral(f, a, b, n=32):
    # """
    # Integrate f over [a,b] using n-point Gauss-Legendre quadrature.
    # Calls f only with scalar inputs � safe for non-vectorized integrands.
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
        self.ur = UR#0.5
        self.nmax = NMAX #300
        self.epsabs =EPSABS# 1e-7
        self.epsrel = EPSREL #1e-5
        # print(f"Initialized FourierPricer with u1={self.model.u1}, u2={self.model.u2}")

        loaded_inference = WishartPINNInference.from_saved_model(
                            NEURAL_NETWORK_MODEL_FOLDER,##str(save_path / "final"),
                            wishart_module_path=PACKAGE_ROOT,
                            normalization_stats_path =NORM_STATS_PATH
                            )
        self.neural_network = loaded_inference


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
            if self.model.pseudo_inverse_smoothing and self.model.initial_curve_alpha is not None:
                curve_alpha_adjustment = self.model.initial_curve_alpha.get_alpha(self.model.maturity)
                price *= curve_alpha_adjustment
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
        
        # Constants_todo.FAST_SWAPTION_PRICING= True#False
        #Checking this hybrid method
        if Constants_todo.FAST_SWAPTION_PRICING:
            # print("Using FAST_SWAPTION_PRICING method")
            return self.price_with_intervals_hybrid()
        else:
            return self.simple_price()
        
        import time as time

        timer_start=time.time()
        simple_price=self.simple_price()
        timer_end_simple= time.time()

        print(f"\n\n\nsimple_price                ={simple_price} , computing time = {timer_end_simple - timer_start}")

        price_interval=  self.price_with_intervals()##ok
        timer_end_interval= time.time()
        print(f"price_interval              ={price_interval} , computing time = {timer_end_interval - timer_end_simple}")

        price_legendre= self.price_with_intervals_gauss_legendre()##ok 
        timer_end_legendre= time.time() 
        print(f"price_legendre              ={price_legendre} , computing time = {timer_end_legendre - timer_end_interval}")

        price_hybrid_simple_integ= self.price_with_intervals_hybrid_simple_integ() ## Ok
        timer_end_hybrid_simple_integ= time.time()
        print(f"price_hybrid_simple_integ   ={price_hybrid_simple_integ} , computing time = {timer_end_hybrid_simple_integ - timer_end_legendre}")

        price_hybrid= self.price_with_intervals_hybrid() ## ok
        timer_end_hybrid= time.time()
        print(f"price_hybrid                ={price_hybrid} , computing time = {timer_end_hybrid - timer_end_hybrid_simple_integ}")  
        
        price_hybrid_Simpson= self.price_with_intervals_hybrid_Simpson() ## ok
        timer_end_hybrid_Simpson= time.time()
        print(f"price_hybrid_Simpson        ={price_hybrid_Simpson} , computing time = {timer_end_hybrid_Simpson - timer_end_hybrid}")
        
        # Swaption 1 Pricing test case
        # simple_price                =0.08927136620889348 , computing time = 27.404237747192383
        # price_interval              =0.08927136620888465 , computing time = 26.33046793937683
        # price_legendre              =0.08927136620888479 , computing time = 58.119983434677124
        # price_hybrid_simple_integ   =0.08927136977245885 , computing time = 4.788143634796143
        # price_hybrid                =0.08927138075555593 , computing time = 2.6972391605377197
        # price_hybrid_Simpson        =0.08927136620888884 , computing time = 4.741518974304199

        # Swaption 2 Pricing test case
        # simple_price                =0.284269908605332 , computing time = 25.602256774902344
        # price_interval              =0.2842699086053232 , computing time = 25.829056978225708
        # price_legendre              =0.28426990860532353 , computing time = 64.53767895698547
        # price_hybrid_simple_integ   =0.28426989612134557 , computing time = 2.316826581954956
        # price_hybrid                =0.2842698576450903 , computing time = 0.8791515827178955
        # price_hybrid_Simpson        =0.2842699086053302 , computing time = 2.3946917057037354

        # Swaption 3 Pricing test case
        # simple_price                =0.5096624846875544 , computing time = 26.720062732696533
        # price_interval              =0.509662484684641 , computing time = 29.773714542388916
        # price_legendre              =0.5096624846875458 , computing time = 64.88547253608704
        # price_hybrid_simple_integ   =0.5096625099802893 , computing time = 2.5049889087677
        # price_hybrid                =0.5096625879340718 , computing time = 0.6985170841217041
        # price_hybrid_Simpson        =0.509662484687453 , computing time = 2.745544195175171

        return simple_price
    
    def price_nn(self, ur: float = None, nmax: int = None, 
              recompute_a3_b3: bool = True) -> float:
        """Price swaption using Fourier transform."""
        self.validate_inputs()
        
        # output_dir = Path(output_dir)
        # output_dir.mkdir(parents=True, exist_ok=True)
        # save_path = output_dir / "models"

        # loaded_inference = WishartPINNInference.from_saved_model(
        #                     NEURAL_NETWORK_MODEL_FOLDER,##str(save_path / "final"),
        #                     #wishart_module_path=PACKAGE_ROOT
        #                     )
        # self.neural_network = loaded_inference
        # print("Neural network model loaded for FourierPricerNN.")
        
        if ur is not None:
            self.ur = ur
        if nmax is not None:
            self.nmax = nmax
        
        # Constants_todo.FAST_SWAPTION_PRICING= True#False
        #Checking this hybrid method
        if Constants_todo.FAST_SWAPTION_PRICING:
            return self.price_with_intervals_hybrid_nn()
        else:
            return self.simple_price_nn()
        
       

        return simple_price
   
    def simple_price(self, ur: float = None, nmax: int = None, 
              recompute_a3_b3: bool = True) -> float:
        """Price swaption using Fourier transform."""
        self.validate_inputs()
        

        if ur is not None:
            self.ur = ur
        if nmax is not None:
            self.nmax = nmax
        if recompute_a3_b3:
            self.model.compute_b3_a3()
        
        # print(f"FourierPricer.price: x0={self.model.x0},a3={self.model.a3}, b3={self.model.b3}")
        # print(f"self.nmax={self.nmax}")
        # Define integrand
        def integrand(ui):
            u = complex(self.ur, ui)
            z = u
            # print(f"Cheking value of  a3 ,{self.model.a3[0,0]}, {self.model.a3[0,1]}, {self.model.a3[1,0]},{self.model.a3[1,1]}")
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
        if self.model.pseudo_inverse_smoothing and self.model.initial_curve_alpha is not None:
            curve_alpha_adjustment = self.model.initial_curve_alpha.get_alpha(self.model.maturity)
            price *= curve_alpha_adjustment
        self.last_integration_error = error
        # print(f" Simple Price: {price}")
        
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
        if self.model.pseudo_inverse_smoothing and self.model.initial_curve_alpha is not None:
            curve_alpha_adjustment = self.model.initial_curve_alpha.get_alpha(self.model.maturity)
            price *= curve_alpha_adjustment
        return price
   
    def price_with_intervals_new(self, intervals=None):
        self.validate_inputs()

        # fewer intervals ? fewer quad calls
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

        if self.model.pseudo_inverse_smoothing and self.model.initial_curve_alpha is not None:
            curve_alpha_adjustment = self.model.initial_curve_alpha.get_alpha(self.model.maturity)
            price *= curve_alpha_adjustment

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

        if self.model.pseudo_inverse_smoothing and self.model.initial_curve_alpha is not None:
            curve_alpha_adjustment = self.model.initial_curve_alpha.get_alpha(self.model.maturity)
            price *= curve_alpha_adjustment

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
    
    def price_with_intervals_hybrid_simple_integ(self, intervals=None, n_outer=100):
        """
        Hybrid: Vectorized outer integration
        """
        self.validate_inputs()

        if intervals is None:
            intervals = np.linspace(0.0, self.nmax, 10).tolist()

        self.model.compute_b3_a3()
        a3 = self.model.a3  # Shape (2, 2)
        b3 = self.model.b3  # Scalar or matrix?
        ur = self.ur

        @jax.jit
        def integrand_batch(ui_array):
            """Vectorized integrand"""
            z_array = ur + 1j * ui_array  # Shape (n_outer,)
            
            def single_integrand(z):
                z_a3 = z * a3  # Scalar * (2,2) matrix = (2,2) matrix
                exp_zb3 = jnp.exp(z * b3)
                phi_val = self.model.wishart.phi_one(1.0, z_a3)
                return (exp_zb3 * phi_val / (z * z)).real
            
            return jax.vmap(single_integrand)(z_array)

        total = 0.0

        for a, b in zip(intervals[:-1], intervals[1:]):
            ui_vals = jnp.linspace(a, b, n_outer)
            integrand_vals = integrand_batch(ui_vals)
            val = jnp.trapezoid(integrand_vals, ui_vals)
            total += val

        price = float(total) / math.pi
        price *= math.exp(-self.model.alpha * self.model.maturity)
        price /= (1 + tr_uv(self.model.u1, self.model.x0))
        if self.model.pseudo_inverse_smoothing and self.model.initial_curve_alpha is not None:
            curve_alpha_adjustment = self.model.initial_curve_alpha.get_alpha(self.model.maturity)
            price *= curve_alpha_adjustment
        return price
  
    def price_with_intervals_hybrid(self, intervals=None, n_outer=None):#100):
        """
        Hybrid: Vectorized outer integration
        """
        self.validate_inputs()

        if intervals is None:
            intervals = np.linspace(0.0, self.nmax, FFT_SWAPTION_NB_INTERVALS).tolist()
        if n_outer is None:
            n_outer = INTEG_NB_POINTS#100
        self.model.compute_b3_a3()
        a3 = self.model.a3
        b3 = self.model.b3
        ur = self.ur

        def single_integrand(z):
            z_a3 = z * a3
            exp_zb3 = jnp.exp(z * b3)
            phi_val = self.model.wishart.phi_one(1.0, z_a3)
            return (exp_zb3 * phi_val / (z * z)).real

        # Pre-compute all ui_vals for all intervals at once
        interval_starts = jnp.array(intervals[:-1])
        interval_ends = jnp.array(intervals[1:])
        n_intervals = len(interval_starts)
        
        # Shape: (n_intervals, n_outer)
        t = jnp.linspace(0.0, 1.0, n_outer)
        ui_vals_all = interval_starts[:, None] + t[None, :] * (interval_ends - interval_starts)[:, None]
        
        # Flatten to (n_intervals * n_outer,)
        ui_flat = ui_vals_all.flatten()
        z_flat = ur + 1j * ui_flat

        @jax.jit
        def compute_all_integrands(z_flat):
            return jax.vmap(single_integrand)(z_flat)

        # Compute all integrand values at once
        integrand_flat = compute_all_integrands(z_flat)
        
        # Reshape back to (n_intervals, n_outer)
        integrand_all = integrand_flat.reshape(n_intervals, n_outer)
        
        # Trapezoid rule for each interval
        @jax.jit
        def trapezoid_all(integrand_all, ui_vals_all):
            return jax.vmap(jnp.trapezoid)(integrand_all, ui_vals_all)
        
        interval_results = trapezoid_all(integrand_all, ui_vals_all)
        total = float(jnp.sum(interval_results))

        price = total / math.pi
        price *= math.exp(-self.model.alpha * self.model.maturity)
        if self.model.pseudo_inverse_smoothing and self.model.initial_curve_alpha is not None:
            curve_alpha_adjustment = self.model.initial_curve_alpha.get_alpha(self.model.maturity)
            price *= curve_alpha_adjustment
        
        price /= (1 + tr_uv(self.model.u1, self.model.x0))

        return price
   
    def price_with_intervals_hybrid_Simpson(self, intervals=None, n_outer=101):  # Must be odd for Simpson
        """
        Hybrid: Using Simpson's rule for better accuracy
        """
        self.validate_inputs()

        if intervals is None:
            intervals = np.linspace(0.0, self.nmax, FFT_SWAPTION_NB_INTERVALS).tolist()

        self.model.compute_b3_a3()
        a3 = self.model.a3
        b3 = self.model.b3
        ur = self.ur

        @jax.jit
        def integrand_batch(ui_array):
            z_array = ur + 1j * ui_array
            
            def single_integrand(z):
                z_a3 = z * a3
                exp_zb3 = jnp.exp(z * b3)
                phi_val = self.model.wishart.phi_one(1.0, z_a3)
                return (exp_zb3 * phi_val / (z * z)).real
            
            return jax.vmap(single_integrand)(z_array)

        @jax.jit
        def simpson(y, x):
            """Simpson's rule integration"""
            n = len(x)
            h = (x[-1] - x[0]) / (n - 1)
            weights = jnp.ones(n)
            weights = weights.at[1:-1:2].set(4)  # Odd indices
            weights = weights.at[2:-1:2].set(2)  # Even indices
            return h / 3 * jnp.sum(weights * y)

        total = 0.0
        for a, b in zip(intervals[:-1], intervals[1:]):
            n_points = n_outer if n_outer % 2 == 1 else n_outer + 1  # Ensure odd
            ui_vals = jnp.linspace(a, b, n_points)
            integrand_vals = integrand_batch(ui_vals)
            val = simpson(integrand_vals, ui_vals)
            total += float(val)

        price = total / math.pi
        price *= math.exp(-self.model.alpha * self.model.maturity)
        price /= (1 + tr_uv(self.model.u1, self.model.x0))

        if self.model.pseudo_inverse_smoothing and self.model.initial_curve_alpha is not None:
            curve_alpha_adjustment = self.model.initial_curve_alpha.get_alpha(self.model.maturity)
            price *= curve_alpha_adjustment

        return price
    
    def price_with_intervals_hybrid_nn(self, intervals=None, n_outer=None):#100):
        """
        Hybrid: Vectorized outer integration
        """
        self.validate_inputs()

        if intervals is None:
            intervals = np.linspace(0.0, self.nmax, FFT_SWAPTION_NB_INTERVALS).tolist()
        if n_outer is None:
            n_outer = INTEG_NB_POINTS#100
        self.model.compute_b3_a3()
        a3 = self.model.a3
        b3 = self.model.b3
        ur = self.ur

        T= self.model.maturity        
        m= self.model.m
        omega= self.model.omega
        sigma= self.model.sigma
        x0= self.model.x0

        def single_integrand(z):
            z_a3 = z * a3
            exp_zb3 = jnp.exp(z * b3)
            # phi_val = self.model.wishart.phi_one(1.0, z_a3)
            theta = z_a3
            # phi_val = self.neural_network.compute_characteristic_function(T, theta, m, omega, sigma, x0)
            phi_val = self.neural_network.compute_characteristic_function_Pricing(T, theta, m, omega, sigma, x0)

            return (exp_zb3 * phi_val / (z * z)).real

        # Pre-compute all ui_vals for all intervals at once
        interval_starts = jnp.array(intervals[:-1])
        interval_ends = jnp.array(intervals[1:])
        n_intervals = len(interval_starts)
        
        # Shape: (n_intervals, n_outer)
        t = jnp.linspace(0.0, 1.0, n_outer)
        ui_vals_all = interval_starts[:, None] + t[None, :] * (interval_ends - interval_starts)[:, None]
        
        # Flatten to (n_intervals * n_outer,)
        ui_flat = ui_vals_all.flatten()
        z_flat = ur + 1j * ui_flat

        @jax.jit
        def compute_all_integrands(z_flat):
            return jax.vmap(single_integrand)(z_flat)

        # Compute all integrand values at once
        integrand_flat = compute_all_integrands(z_flat)
        
        # Reshape back to (n_intervals, n_outer)
        integrand_all = integrand_flat.reshape(n_intervals, n_outer)
        
        # Trapezoid rule for each interval
        @jax.jit
        def trapezoid_all(integrand_all, ui_vals_all):
            return jax.vmap(jnp.trapezoid)(integrand_all, ui_vals_all)
        
        interval_results = trapezoid_all(integrand_all, ui_vals_all)
        total = float(jnp.sum(interval_results))

        price = total / math.pi
        price *= math.exp(-self.model.alpha * self.model.maturity)
        if self.model.pseudo_inverse_smoothing and self.model.initial_curve_alpha is not None:
            curve_alpha_adjustment = self.model.initial_curve_alpha.get_alpha(self.model.maturity)
            price *= curve_alpha_adjustment
        
        price /= (1 + tr_uv(self.model.u1, self.model.x0))

        return price
  
    def simple_price_nn(self, ur: float = None, nmax: int = None, 
              recompute_a3_b3: bool = True) -> float:
        """Price swaption using Fourier transform."""
        self.validate_inputs()
        

        if ur is not None:
            self.ur = ur
        if nmax is not None:
            self.nmax = nmax
        if recompute_a3_b3:
            self.model.compute_b3_a3()
        
        T= self.model.maturity        
        m= self.model.m
        omega= self.model.omega
        sigma= self.model.sigma
        x0= self.model.x0

        # print(f"FourierPricer.price: x0={self.model.x0},a3={self.model.a3}, b3={self.model.b3}")
        # print(f"self.nmax={self.nmax}")
        # Define integrand
        def integrand(ui):
            u = complex(self.ur, ui)
            z = u
            
            z_a3 = z * self.model.a3
            exp_z_b3 = cmath.exp(z * self.model.b3)
            
            if self.use_range_kutta:
                # phi1 = self.model.wishart.phi_one(1, z_a3)
                theta = z_a3
                phi1 = self.neural_network.compute_characteristic_function(T, theta, m, omega, sigma, x0)
                # phi1 = self.neural_network.compute_characteristic_function_Pricing(T, theta, m, omega, sigma, x0)


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
        if self.model.pseudo_inverse_smoothing and self.model.initial_curve_alpha is not None:
            curve_alpha_adjustment = self.model.initial_curve_alpha.get_alpha(self.model.maturity)
            price *= curve_alpha_adjustment
        self.last_integration_error = error
        # print(f" Simple Price: {price}")
        
        return price

   ###end
        