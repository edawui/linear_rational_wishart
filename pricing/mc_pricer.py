"""
Monte Carlo pricer for Wishart-based interest rate models.

This module provides Monte Carlo simulation-based pricing for various
interest rate derivatives using Wishart processes.
"""
import math
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap, lax, random
from functools import partial
from typing import List, Tuple, Dict, Optional, Union, Any

from ..simulation.schemas import simulate_wishart, simulate_wishart_jump
from ..math.psd_corrections import sqrtm_real


@jit
def jax_trace_uv(u: jnp.ndarray, v: jnp.ndarray) -> float:
    """
    Compute trace of matrix product U @ V in JAX.
    
    Parameters
    ----------
    u : jnp.ndarray
        First matrix
    v : jnp.ndarray
        Second matrix
        
    Returns
    -------
    float
        Trace of U @ V
    """
    result = jnp.trace(u @ v)
    return jnp.squeeze(result)


class WishartMonteCarloPricer:
    """
    Monte Carlo pricer for Wishart-based interest rate models.
    
    This class provides pricing functionality for various interest rate
    derivatives including bonds, options, and swaps using Monte Carlo
    simulation of Wishart processes.
    
    Parameters
    ----------
    lrw_model : LRWModel
        The underlying Wishart interest rate model
        
    Attributes
    ----------
    wishart_model : LRWModel
        The model instance
    x0_jax : jnp.ndarray
        Initial state (JAX array)
    omega_jax : jnp.ndarray
        Drift parameter (JAX array)
    m_jax : jnp.ndarray
        Mean reversion (JAX array)
    sigma_jax : jnp.ndarray
        Volatility (JAX array)
    u1_jax : jnp.ndarray
        First transformation matrix
    u2_jax : jnp.ndarray
        Second transformation matrix
    a3_jax : jnp.ndarray
        Option payoff matrix
    """
    
    def __init__(self, lrw_model: Any):
        """Initialize the Monte Carlo pricer."""
        self.wishart_model = lrw_model
        
        # Convert numpy arrays to JAX arrays for better performance
        self.x0_jax = jnp.array(self.wishart_model.x0)
        self.omega_jax = jnp.array(self.wishart_model.omega)
        self.m_jax = jnp.array(self.wishart_model.m)
        self.sigma_jax = jnp.array(self.wishart_model.sigma)
        self.u1_jax = jnp.array(self.wishart_model.u1)
        self.u2_jax = jnp.array(self.wishart_model.u2)
        self.a3_jax = jnp.array(self.wishart_model.a3)

    @partial(jit, static_argnums=(0,))
    def _wishart_step(
        self, 
        v: jnp.ndarray, 
        dt: float, 
        sqrtdt: float, 
        dwt: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Single Wishart process step - JAX optimized.
        
        Parameters
        ----------
        v : jnp.ndarray
            Current state
        dt : float
            Time step
        sqrtdt : float
            Square root of time step
        dwt : jnp.ndarray
            Brownian increment
            
        Returns
        -------
        jnp.ndarray
            Updated state
        """
        v1 = sqrtm_real(v) @ dwt @ self.sigma_jax
        v1 = v1 + v1.T
        drift = self.omega_jax + self.m_jax @ v + v @ self.m_jax
        return v + dt * drift + v1
    
    def simulate(
        self,
        time_list: List[float],
        nb_mc: int,
        dt: float,
        schema: str = "EULER_FLOORED"
    ) -> np.ndarray:
        """
        Simulate Wishart process paths.
        
        Parameters
        ----------
        time_list : List[float]
            Times at which to record the process
        nb_mc : int
            Number of Monte Carlo paths
        dt : float
            Time step for discretization
        schema : str, optional
            Numerical scheme to use
            
        Returns
        -------
        np.ndarray
            Simulated paths with shape (nb_mc, len(time_list), d, d)
        """
        alpha_bar = self.wishart_model.omega
        b = self.wishart_model.m
        a = self.wishart_model.sigma
        x = self.wishart_model.x0
        start_time = 0.0
        
        # Check if model has jumps
        if hasattr(self.wishart_model, 'hasJump') and self.wishart_model.hasJump:
            if hasattr(self.wishart_model, 'JumpComponent'):
                jump_comp = self.wishart_model.JumpComponent
                return simulate_wishart_jump(
                    x, alpha_bar, b, a,
                    jump_comp.lambda_intensity,
                    jump_comp.nu,
                    jump_comp.eta,
                    jump_comp.xi,
                    start_time, time_list,
                    num_paths=nb_mc,
                    dt_substep=dt,
                    schema=schema
                )
        
        # Standard Wishart simulation
        return simulate_wishart(
            x, alpha_bar, b, a,
            start_time, time_list,
            num_paths=nb_mc,
            dt_substep=dt,
            schema=schema
        )

    def price_option_mc(
        self, 
        nb_mc: int, 
        dt: float
    ) -> float:
        """
        Price option using basic Monte Carlo (legacy method).
        
        Parameters
        ----------
        nb_mc : int
            Number of Monte Carlo paths
        dt : float
            Time step
            
        Returns
        -------
        float
            Option price
        """
        maturity = self.wishart_model.maturity
        if nb_mc <= 0:
            raise ValueError("nb_mc must be positive.")
        if dt <= 0:
            raise ValueError("dt must be positive.")
        if maturity <= 0:
            raise ValueError("Maturity must be positive.")

        nb_t = int(self.wishart_model.maturity / dt)
        if nb_t < 1:
            raise ValueError("Time step dt is too large for the given maturity.")

        sqrtdt = math.sqrt(dt)
        key = random.PRNGKey(42)
        
        # Vectorized simulation using JAX
        @jit
        def simulate_path(key):
            keys = random.split(key, nb_t)
            v = self.x0_jax
            
            def step(v, key):
                dwt = random.normal(key, shape=self.x0_jax.shape) * sqrtdt
                v_next = self._wishart_step(v, dt, sqrtdt, dwt)
                return v_next, v_next
            
            _, vs = lax.scan(step, v, keys)
            v_final = vs[-1]
            
            temp = self.wishart_model.b3 + jax_trace_uv(self.a3_jax, v_final)
            payoff = jnp.maximum(temp, 0.0)
            return payoff

        # Vectorize over Monte Carlo paths
        keys = random.split(key, nb_mc)
        payoffs = vmap(simulate_path)(keys)
        price = jnp.mean(payoffs)

        denom = 1 + jax_trace_uv(self.u1_jax, self.x0_jax)
        if denom == 0:
            raise ZeroDivisionError("Denominator in discount factor is zero.")

        price = jnp.exp(-self.wishart_model.alpha * maturity) * price / denom
        
        if self.wishart_model.pseudo_inverse_smoothing:
            curve_alpha_adjustment = self.wishart_model.initial_curve_alpha.get_alpha(maturity)
            price *= curve_alpha_adjustment
        
        return float(price)

    def price_option_mc_with_schema(
        self,
        nb_mc: int,
        dt: float,
        schema: str = "EULER_FLOORED"
    ) -> float:
        """
        Price option using specified numerical scheme.
        
        Parameters
        ----------
        nb_mc : int
            Number of Monte Carlo paths
        dt : float
            Time step
        schema : str, optional
            Numerical scheme
            
        Returns
        -------
        float
            Option price
        """
        maturity = self.wishart_model.maturity
        if nb_mc <= 0:
            raise ValueError("nb_mc must be positive.")
        if maturity <= 0:
            raise ValueError("Maturity must be positive.")

        time_list = [0.0, maturity]

        if schema == "ALFONSI":
            dt = max(maturity / 20, 1e-6)

        sim_results_dict = self.simulate(time_list, nb_mc, dt, schema)
        
        # Convert dict results to array
        sim_results = np.zeros((nb_mc, 2, self.x0_jax.shape[0], self.x0_jax.shape[1]))
        for path_idx in range(nb_mc):
            for t_idx, t in enumerate(time_list):
                sim_results[path_idx, t_idx] = sim_results_dict[path_idx][t]
        
        maturity_idx = 1  # Last time point

        # Vectorized trace computation
        xx_at_maturity = sim_results[:, maturity_idx]
        temp_payoff = vmap(
            lambda xx: jnp.maximum(
                self.wishart_model.b3 + jax_trace_uv(self.a3_jax, xx), 0.0
            )
        )(xx_at_maturity)
        price = jnp.mean(temp_payoff)

        denom = 1 + float(jax_trace_uv(self.u1_jax, self.x0_jax))
        if denom == 0:
            raise ZeroDivisionError("Denominator in discount factor is zero.")

        price = math.exp(-self.wishart_model.alpha * maturity) * price / denom
        
        if self.wishart_model.pseudo_inverse_smoothing:
            curve_alpha_adjustment = self.wishart_model.initial_curve_alpha.get_alpha(maturity)
            price *= curve_alpha_adjustment
            
        return float(price)

    def price_bond_mc_with_schema(
        self,
        t_list: List[float],
        nb_mc: int,
        dt: float,
        schema: str = "EULER_FLOORED"
    ) -> np.ndarray:
        """
        Price zero-coupon bonds for multiple maturities.
        
        Parameters
        ----------
        t_list : List[float]
            Bond maturities
        nb_mc : int
            Number of Monte Carlo paths
        dt : float
            Time step
        schema : str, optional
            Numerical scheme
            
        Returns
        -------
        np.ndarray
            Bond prices for each maturity
        """
        time_list = t_list

        if schema == "ALFONSI":
            diffs = np.diff(np.sort(t_list))
            dt = np.min(diffs[diffs > 0]) if np.any(diffs > 0) else dt

        sim_results_dict = self.simulate(time_list, nb_mc, dt, schema)
        
        # Convert dict results to array
        sim_results = np.zeros((nb_mc, len(time_list), self.x0_jax.shape[0], self.x0_jax.shape[1]))
        for path_idx in range(nb_mc):
            for t_idx, t in enumerate(time_list):
                sim_results[path_idx, t_idx] = sim_results_dict[path_idx][t]
        
        # Compute bond prices
        @jit
        def compute_bond_prices_jit(sim_results, t_list, alpha, u1):
            """Fully JIT-compiled bond price computation."""
            def scan_fn(_, i):
                t = t_list[i]
                xx_all = sim_results[:, i]
                traces = vmap(lambda xx: jax_trace_uv(u1, xx))(xx_all)
                bond_price = jnp.mean(jnp.exp(-alpha * t) * (1 + traces))
                return None, bond_price
    
            _, bond_prices = jax.lax.scan(scan_fn, None, jnp.arange(len(t_list)))
            return bond_prices

        bond_prices = compute_bond_prices_jit(
            jnp.array(sim_results), 
            jnp.array(t_list),
            self.wishart_model.alpha,
            self.u1_jax
        )
      
        # Normalization
        t = 0.0
        denom = jnp.exp(-self.wishart_model.alpha * t) * (1 + jax_trace_uv(self.u1_jax, self.x0_jax))
        bond_prices = bond_prices / denom
    
        # Curve adjustment if needed
        if self.wishart_model.pseudo_inverse_smoothing:
            adjustments = [self.wishart_model.initial_curve_alpha.get_alpha(t) for t in t_list]
            bond_prices = bond_prices * jnp.array(adjustments)
        
        return np.array(bond_prices)

    def compute_floating_mc_coupon(
        self,
        t: float,
        accrual_dcf: float,
        nb_mc: int,
        dt: float,
        schema: str = "EULER_FLOORED"
    ) -> Tuple[float, float, float, float]:
        """
        Compute floating rate coupon value.
        
        Parameters
        ----------
        t : float
            Coupon start time
        accrual_dcf : float
            Accrual day count fraction
        nb_mc : int
            Number of Monte Carlo paths
        dt : float
            Time step
        schema : str, optional
            Numerical scheme
            
        Returns
        -------
        Tuple[float, float, float, float]
            (float_coupon_value, P_T, P_T_Delta, spread_payment)
        """
        if nb_mc <= 0:
            raise ValueError("nb_mc must be positive.")
        if accrual_dcf <= 0:
            raise ValueError("accrual_dcf must be positive.")

        time_list = [t, t + accrual_dcf]
        t_delta = t + accrual_dcf

        if schema == "ALFONSI":
            dt = accrual_dcf

        sim_results_dict = self.simulate(time_list, nb_mc, dt, schema)
        
        # Convert dict results to array
        sim_results = np.zeros((nb_mc, 2, self.x0_jax.shape[0], self.x0_jax.shape[1]))
        for path_idx in range(nb_mc):
            for t_idx, time in enumerate(time_list):
                sim_results[path_idx, t_idx] = sim_results_dict[path_idx][time]
        
        t_idx = 0
        t_delta_idx = 1

        xx_at_t = sim_results[:, t_idx]
        xx_at_t_delta = sim_results[:, t_delta_idx]
        
        spread_payment_all = vmap(lambda xx: jax_trace_uv(self.u2_jax, xx))(xx_at_t)
        p_t_all = vmap(
            lambda xx: jnp.exp(-self.wishart_model.alpha * t) * (1 + jax_trace_uv(self.u1_jax, xx))
        )(xx_at_t)
        p_t_delta_all = vmap(
            lambda xx: jnp.exp(-self.wishart_model.alpha * t_delta) * (1 + jax_trace_uv(self.u1_jax, xx))
        )(xx_at_t_delta)

        spread_payment = jnp.mean(spread_payment_all)
        p_t = jnp.mean(p_t_all)
        p_t_delta = jnp.mean(p_t_delta_all)

        t0 = 0
        denom = math.exp(-self.wishart_model.alpha * t0) * (1 + float(jax_trace_uv(self.u1_jax, self.x0_jax)))
        if denom == 0:
            raise ZeroDivisionError("Denominator in discount factor is zero.")

        spread_payment = math.exp(-self.wishart_model.alpha * t) * spread_payment / denom
        p_t = p_t / denom
        p_t_delta = p_t_delta / denom

        if self.wishart_model.pseudo_inverse_smoothing:
            curve_alpha_adjustment = self.wishart_model.initial_curve_alpha.get_alpha(t_delta)
            spread_payment *= curve_alpha_adjustment
            p_t *= curve_alpha_adjustment
            p_t_delta *= curve_alpha_adjustment

        float_coupon_value = (p_t - p_t_delta) + spread_payment
        return float_coupon_value, p_t, p_t_delta, spread_payment

    def compute_swap_mc_with_schema(
        self,
        fixed_rate: float,
        spread: float,
        floating_schedule: List[float],
        fixed_schedule: List[float],
        nb_mc: int,
        dt: float,
        schema: str = "EULER_FLOORED"
    ) -> Tuple[float, List[float], List[float]]:
        """
        Compute swap value using Monte Carlo.
        
        Parameters
        ----------
        fixed_rate : float
            Fixed leg rate
        spread : float
            Floating leg spread
        floating_schedule : List[float]
            Floating leg payment dates
        fixed_schedule : List[float]
            Fixed leg payment dates
        nb_mc : int
            Number of Monte Carlo paths
        dt : float
            Time step
        schema : str, optional
            Numerical scheme
            
        Returns
        -------
        Tuple[float, List[float], List[float]]
            (swap_value, float_leg_coupons, fixed_leg_coupons)
        """
        if len(floating_schedule) < 2 or len(fixed_schedule) < 2:
            raise ValueError("Schedules must have at least two dates.")

        start_time = 0.0
        all_dates = np.unique(np.concatenate((fixed_schedule, floating_schedule, [start_time])))
        all_dates.sort()
        time_list = all_dates.tolist()

        if schema == "ALFONSI":
            diffs = np.diff(floating_schedule)
            dt = np.min(diffs[diffs > 0]) if np.any(diffs > 0) else dt

        sim_results_dict = self.simulate(time_list, nb_mc, dt, schema)
        
        # Convert dict results to array
        sim_results = np.zeros((nb_mc, len(time_list), self.x0_jax.shape[0], self.x0_jax.shape[1]))
        for path_idx in range(nb_mc):
            for t_idx, t in enumerate(time_list):
                sim_results[path_idx, t_idx] = sim_results_dict[path_idx][t]

        # Process floating leg
        n_float = len(floating_schedule) - 1
        float_leg_coupons = np.zeros(n_float)
        spread_payments = np.zeros(n_float)
        p_ts = np.zeros(n_float + 1)

        # Compute present values
        time_to_idx = {float(time): idx for idx, time in enumerate(time_list)}
        
        for nmc in range(nb_mc):
            t_idx = time_to_idx.get(0.0, 0)
            x_t0 = jnp.array(sim_results[nmc, t_idx])
            
            denom = math.exp(-self.wishart_model.alpha * 0.0) * (1 + float(jax_trace_uv(self.u1_jax, x_t0)))
            if denom == 0:
                raise ZeroDivisionError("Denominator in discount factor is zero.")
                
            for j in range(n_float):
                t = floating_schedule[j]
                t_idx = time_to_idx.get(float(t), 0)
                xx = jnp.array(sim_results[nmc, t_idx])
                spread_payments[j] += float(jax_trace_uv(self.u2_jax, xx)) / denom
                p_ts[j] += float(jnp.exp(-self.wishart_model.alpha * t) * (1 + jax_trace_uv(self.u1_jax, xx))) / denom
                
            last_t = floating_schedule[-1]
            xx_last = jnp.array(sim_results[nmc, time_to_idx.get(float(last_t))])
            p_ts[-1] += float(jnp.exp(-self.wishart_model.alpha * last_t) * (1 + jax_trace_uv(self.u1_jax, xx_last))) / denom

        spread_payments /= nb_mc
        p_ts /= nb_mc

        # Apply curve adjustments
        if self.wishart_model.pseudo_inverse_smoothing:
            for j in range(n_float):
                t = floating_schedule[j]
                adj = self.wishart_model.initial_curve_alpha.get_alpha(t)
                spread_payments[j] *= adj
                p_ts[j] *= adj
            t = floating_schedule[-1]
            adj = self.wishart_model.initial_curve_alpha.get_alpha(t)
            p_ts[-1] *= adj
        
        # Compute floating leg coupons
        for i in range(n_float):
            dcf = floating_schedule[i+1] - floating_schedule[i]
            float_leg_coupons[i] = (p_ts[i] - p_ts[i+1]) + spread_payments[i] + p_ts[i+1] * spread * dcf

        # Process fixed leg
        n_fix = len(fixed_schedule) - 1
        fixed_leg_coupons = np.zeros(n_fix)
        p_ts_fix = np.zeros(len(fixed_schedule))

        for nmc in range(nb_mc):
            x_t0 = jnp.array(sim_results[nmc, time_to_idx.get(0.0)])
            denom = math.exp(-self.wishart_model.alpha * 0.0) * (1 + float(jax_trace_uv(self.u1_jax, x_t0)))
            if denom == 0:
                raise ZeroDivisionError("Denominator in discount factor is zero.")
                
            for j, t in enumerate(fixed_schedule):
                xx = jnp.array(sim_results[nmc, time_to_idx.get(float(t))])
                p_ts_fix[j] += float(jnp.exp(-self.wishart_model.alpha * t) * (1 + jax_trace_uv(self.u1_jax, xx))) / denom

        p_ts_fix /= nb_mc

        # Apply curve adjustments
        for j, t in enumerate(fixed_schedule):
            if self.wishart_model.pseudo_inverse_smoothing:
                adj = self.wishart_model.initial_curve_alpha.get_alpha(t)
                p_ts_fix[j] *= adj

        # Compute fixed leg coupons
        for i in range(n_fix):
            dcf = fixed_schedule[i+1] - fixed_schedule[i]
            fixed_leg_coupons[i] = p_ts_fix[i+1] * fixed_rate * dcf

        swap_value = np.sum(float_leg_coupons) - np.sum(fixed_leg_coupons)
        return swap_value, float_leg_coupons.tolist(), fixed_leg_coupons.tolist()

    def _get_updated_schedule(
        self,
        date_list: List[float],
        valuation_date: float,
        main_date: float
    ) -> Tuple[Optional[float], List[float], List[float]]:
        """
        Update schedule based on valuation date.
        
        Parameters
        ----------
        date_list : List[float]
            Original schedule
        valuation_date : float
            Current valuation date
        main_date : float
            Main reference date
            
        Returns
        -------
        Tuple[Optional[float], List[float], List[float]]
            (last_date, all_dates, new_dates)
        """
        if not date_list:
            raise ValueError("date_list is empty.")
        if valuation_date < main_date:
            raise ValueError("Valuation date is before main date.")
        if valuation_date >= date_list[-1]:
            return None, [], []
    
        idx = next((i for i, val in enumerate(date_list) if val >= valuation_date), len(date_list))
    
        if idx == 0:
            date_list_all = date_list[:]
            last_t = date_list[0]
        else:
            date_list_all = [valuation_date] + date_list[idx:]
            last_t = date_list[idx-1]
            
        date_list_all = np.unique(date_list_all).tolist()
        date_list_all.sort()
    
        date_list_new = [max(valuation_date, main_date)] + [t for t in date_list if t > valuation_date]
        return last_t, date_list_all, date_list_new

    def _compute_swap_mc_path_pricer(
    self,
    valuation_date: float,
    fixed_rate: float,
    spread: float,
    floating_schedule: List[float],
    fixed_schedule: List[float],
    sim_results: np.ndarray,
    path: int,
    time_list: List[float]
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Compute swap value for a single path (optimized version).
        """
        # Convert time_list to numpy array once for fast searching
        time_array = np.array(time_list)
    
        # Helper function to get indices efficiently
        def get_indices(dates):
            """Fast index lookup using searchsorted"""
            indices = np.searchsorted(time_array, dates)
            # Clamp to valid range
            indices = np.clip(indices, 0, len(time_array) - 1)
            return indices
    
        # Get path data once
        path_data = sim_results[path]  # Shape: (n_times, ...)
    
        float_leg_coupons = np.array([])
        fixed_leg_coupons = np.array([])
    
        # Process floating leg
        if len(floating_schedule) >= 2:
            n_float = len(floating_schedule) - 1
            float_leg_coupons = np.zeros(n_float)
        
            # Get all indices at once
            val_idx = get_indices([valuation_date])[0]
            float_indices = get_indices(floating_schedule)
        
            # Get state at valuation date
            x_t0 = path_data[val_idx]
        
            # Pre-compute denominator base - EXPLICITLY convert to float
            trace_t0 = float(jax_trace_uv(self.u1_jax, x_t0))
            denom_base = float(math.exp(-self.wishart_model.alpha * valuation_date) * (1 + trace_t0))
        
            # Vectorize computations - get all states at once
            float_schedule_array = np.array(floating_schedule)
            xx_all = path_data[float_indices]  # Get all states in one operation
        
            # Pre-compute exponential factors - ensure numpy
            exp_factors = np.exp(-self.wishart_model.alpha * float_schedule_array)
        
            # Vectorized trace computations - EXPLICITLY convert each to float
            traces_u1 = np.array([float(jax_trace_uv(self.u1_jax, xx_all[i])) for i in range(len(xx_all))])
            traces_u2 = np.array([float(jax_trace_uv(self.u2_jax, xx_all[i])) for i in range(n_float)])
        
            # Compute p_ts and spread_payments vectorized - ensure all numpy operations
            p_ts = np.array(exp_factors * (1 + traces_u1) / denom_base, dtype=np.float64)
            spread_payments = np.array(traces_u2 / denom_base, dtype=np.float64)
        
            # Apply pseudo inverse smoothing if needed
            if self.wishart_model.pseudo_inverse_smoothing:
                adjustments = np.array([self.wishart_model.initial_curve_alpha.get_alpha(t) 
                                       for t in float_schedule_array], dtype=np.float64)
                p_ts = np.array(p_ts * adjustments, dtype=np.float64)
                spread_payments = np.array(spread_payments * adjustments[:-1], dtype=np.float64)
        
            # Compute coupons vectorized - FORCE numpy array output
            dcf = np.diff(float_schedule_array)
            float_leg_coupons = np.array(
                (p_ts[:-1] - p_ts[1:]) + spread_payments + p_ts[1:] * spread * dcf,
                dtype=np.float64
            )
    
        # Process fixed leg
        if len(fixed_schedule) >= 2:
            n_fix = len(fixed_schedule) - 1
            fixed_leg_coupons = np.zeros(n_fix)
        
            # Get all indices at once
            val_idx = get_indices([valuation_date])[0]
            fixed_indices = get_indices(fixed_schedule)
        
            # Get state at valuation date
            x_t0 = path_data[val_idx]
        
            # Pre-compute denominator base - EXPLICITLY convert to float
            trace_t0 = float(jax_trace_uv(self.u1_jax, x_t0))
            denom_base = float(math.exp(-self.wishart_model.alpha * valuation_date) * (1 + trace_t0))
        
            # Vectorize computations
            fixed_schedule_array = np.array(fixed_schedule)
            xx_all_fix = path_data[fixed_indices]
        
            # Pre-compute exponential factors - ensure numpy
            exp_factors_fix = np.exp(-self.wishart_model.alpha * fixed_schedule_array)
        
            # Vectorized trace computations - EXPLICITLY convert each to float
            traces_u1_fix = np.array([float(jax_trace_uv(self.u1_jax, xx_all_fix[i])) 
                                      for i in range(len(xx_all_fix))])
        
            # Compute p_ts vectorized - ensure all numpy operations
            p_ts_fix = np.array(exp_factors_fix * (1 + traces_u1_fix) / denom_base, dtype=np.float64)
        
            # Apply pseudo inverse smoothing if needed
            if self.wishart_model.pseudo_inverse_smoothing:
                adjustments_fix = np.array([self.wishart_model.initial_curve_alpha.get_alpha(t) 
                                           for t in fixed_schedule_array], dtype=np.float64)
                p_ts_fix = np.array(p_ts_fix * adjustments_fix, dtype=np.float64)
        
            # Compute coupons vectorized - FORCE numpy array output
            dcf_fix = np.diff(fixed_schedule_array)
            fixed_leg_coupons = np.array(
                p_ts_fix[1:] * fixed_rate * dcf_fix,
                dtype=np.float64
            )
    
        swap_value = float(np.sum(float_leg_coupons) - np.sum(fixed_leg_coupons))
    
        # CRITICAL: Ensure we return pure numpy arrays, not JAX arrays
        return swap_value, np.asarray(float_leg_coupons, dtype=np.float64), np.asarray(fixed_leg_coupons, dtype=np.float64)


    def _compute_swap_mc_path_pricer_batched(
    self,
    valuation_date: float,
    fixed_rate: float,
    spread: float,
    floating_schedule: List[float],
    fixed_schedule: List[float],
    sim_results: np.ndarray,
    paths: np.ndarray,
    time_list: List[float]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute swap value for multiple paths simultaneously (batched version).
    
        Parameters
        ----------
        valuation_date : float
            Valuation date
        fixed_rate : float
            Fixed rate
        spread : float
            Spread
        floating_schedule : List[float]
            Floating schedule
        fixed_schedule : List[float]
            Fixed schedule
        sim_results : np.ndarray
            Simulation results with shape (nb_mc, n_times, ...)
        paths : np.ndarray
            Array of path indices to process
        time_list : List[float]
            Time list for indexing
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (swap_values, float_coupons, fixed_coupons)
            All as pure numpy arrays
        """
        n_paths = len(paths)
        time_array = np.array(time_list, dtype=np.float64)
    
        # Helper function for fast index lookup
        def get_indices(dates):
            indices = np.searchsorted(time_array, dates)
            return np.clip(indices, 0, len(time_array) - 1)
    
        # Get data for all paths at once
        paths_data = sim_results[paths]
    
        float_leg_coupons_all = np.array([], dtype=np.float64)
        fixed_leg_coupons_all = np.array([], dtype=np.float64)
    
        # Process floating leg
        if len(floating_schedule) >= 2:
            n_float = len(floating_schedule) - 1
            float_leg_coupons_all = np.zeros((n_paths, n_float), dtype=np.float64)
        
            # Get indices
            val_idx = int(get_indices([valuation_date])[0])
            float_indices = get_indices(floating_schedule)
        
            # Get states at valuation date for all paths
            x_t0_all = paths_data[:, val_idx]
        
            # Compute traces for all paths at valuation date
            traces_t0 = np.zeros(n_paths, dtype=np.float64)
            for i in range(n_paths):
                traces_t0[i] = float(jax_trace_uv(self.u1_jax, x_t0_all[i]))
        
            # Pre-compute denominator base for all paths
            denom_base_all = np.exp(-self.wishart_model.alpha * valuation_date) * (1.0 + traces_t0)
        
            # Get states at all floating schedule dates for all paths
            xx_all = paths_data[:, float_indices]
        
            float_schedule_array = np.array(floating_schedule, dtype=np.float64)
            exp_factors = np.exp(-self.wishart_model.alpha * float_schedule_array)
        
            # Compute traces for all paths and all dates
            traces_u1_all = np.zeros((n_paths, len(floating_schedule)), dtype=np.float64)
            traces_u2_all = np.zeros((n_paths, n_float), dtype=np.float64)
        
            for path_idx in range(n_paths):
                for i in range(len(floating_schedule)):
                    traces_u1_all[path_idx, i] = float(jax_trace_uv(self.u1_jax, xx_all[path_idx, i]))
                    if i < n_float:
                        traces_u2_all[path_idx, i] = float(jax_trace_uv(self.u2_jax, xx_all[path_idx, i]))
        
            # Compute p_ts and spread_payments with broadcasting
            p_ts_all = (exp_factors[np.newaxis, :] * (1.0 + traces_u1_all)) / denom_base_all[:, np.newaxis]
            spread_payments_all = traces_u2_all / denom_base_all[:, np.newaxis]
        
            # Apply pseudo inverse smoothing if needed
            if self.wishart_model.pseudo_inverse_smoothing:
                adjustments = np.zeros(len(float_schedule_array), dtype=np.float64)
                for j, t in enumerate(float_schedule_array):
                    adjustments[j] = self.wishart_model.initial_curve_alpha.get_alpha(t)
                p_ts_all = p_ts_all * adjustments[np.newaxis, :]
                spread_payments_all = spread_payments_all * adjustments[np.newaxis, :-1]
        
            # Compute coupons for all paths
            dcf = np.diff(float_schedule_array)
            float_leg_coupons_all = (
                (p_ts_all[:, :-1] - p_ts_all[:, 1:]) + 
                spread_payments_all + 
                p_ts_all[:, 1:] * spread * dcf[np.newaxis, :]
            )
        
            # Ensure pure numpy array
            float_leg_coupons_all = np.asarray(float_leg_coupons_all, dtype=np.float64)
    
        # Process fixed leg
        if len(fixed_schedule) >= 2:
            n_fix = len(fixed_schedule) - 1
            fixed_leg_coupons_all = np.zeros((n_paths, n_fix), dtype=np.float64)
        
            # Get indices
            val_idx = int(get_indices([valuation_date])[0])
            fixed_indices = get_indices(fixed_schedule)
        
            # Get states at valuation date for all paths
            x_t0_all = paths_data[:, val_idx]
        
            # Compute traces for all paths at valuation date
            traces_t0_fix = np.zeros(n_paths, dtype=np.float64)
            for i in range(n_paths):
                traces_t0_fix[i] = float(jax_trace_uv(self.u1_jax, x_t0_all[i]))
        
            # Pre-compute denominator base for all paths
            denom_base_all_fix = np.exp(-self.wishart_model.alpha * valuation_date) * (1.0 + traces_t0_fix)
        
            # Get states at all fixed schedule dates for all paths
            xx_all_fix = paths_data[:, fixed_indices]
        
            fixed_schedule_array = np.array(fixed_schedule, dtype=np.float64)
            exp_factors_fix = np.exp(-self.wishart_model.alpha * fixed_schedule_array)
        
            # Compute traces for all paths and all dates
            traces_u1_all_fix = np.zeros((n_paths, len(fixed_schedule)), dtype=np.float64)
            for path_idx in range(n_paths):
                for i in range(len(fixed_schedule)):
                    traces_u1_all_fix[path_idx, i] = float(jax_trace_uv(self.u1_jax, xx_all_fix[path_idx, i]))
        
            # Compute p_ts with broadcasting
            p_ts_all_fix = (exp_factors_fix[np.newaxis, :] * (1.0 + traces_u1_all_fix)) / denom_base_all_fix[:, np.newaxis]
        
            # Apply pseudo inverse smoothing if needed
            if self.wishart_model.pseudo_inverse_smoothing:
                adjustments_fix = np.zeros(len(fixed_schedule_array), dtype=np.float64)
                for j, t in enumerate(fixed_schedule_array):
                    adjustments_fix[j] = self.wishart_model.initial_curve_alpha.get_alpha(t)
                p_ts_all_fix = p_ts_all_fix * adjustments_fix[np.newaxis, :]
        
            # Compute coupons for all paths
            dcf_fix = np.diff(fixed_schedule_array)
            fixed_leg_coupons_all = p_ts_all_fix[:, 1:] * fixed_rate * dcf_fix[np.newaxis, :]
        
            # Ensure pure numpy array
            fixed_leg_coupons_all = np.asarray(fixed_leg_coupons_all, dtype=np.float64)
    
        # Compute swap values for all paths
        swap_values = np.sum(float_leg_coupons_all, axis=1) - np.sum(fixed_leg_coupons_all, axis=1)
        swap_values = np.asarray(swap_values, dtype=np.float64)
    
        return swap_values, float_leg_coupons_all, fixed_leg_coupons_all

    
    def compute_swap_exposure_profile(
    self,
    exposure_dates: List[float],
    fixed_rate: float,
    spread: float,
    floating_schedule_trade: List[float],
    fixed_schedule_trade: List[float],
    nb_mc: int,
    dt: float,
    main_date: float = 0.0,
    schema: str = "EULER_FLOORED",
    swap_price_single_path=False#True#
    ) -> np.ndarray:

        if swap_price_single_path:
            return self.compute_swap_exposure_profile_single_path(    
            exposure_dates,
            fixed_rate,
            spread,
            floating_schedule_trade,
            fixed_schedule_trade,
            nb_mc,
            dt,
            main_date,
            schema
            )

        else:
            
            # batch_size = 10#100
                
            return self.compute_swap_exposure_profile_vectorized(    
                exposure_dates,
                fixed_rate,
                spread,
                floating_schedule_trade,
                fixed_schedule_trade,
                nb_mc,
                dt,
                main_date,
                schema
                # , batch_size
                )


    def compute_swap_exposure_profile_single_path(
    self,
    exposure_dates: List[float],
    fixed_rate: float,
    spread: float,
    floating_schedule_trade: List[float],
    fixed_schedule_trade: List[float],
    nb_mc: int,
    dt: float,
    main_date: float = 0.0,
    schema: str = "EULER_FLOORED"
    ) -> np.ndarray:
        """
        Compute swap exposure profile over time.
        """
        exposure_profile = np.zeros((len(exposure_dates), nb_mc))

        all_dates = np.unique(np.concatenate((
            fixed_schedule_trade, 
            floating_schedule_trade, 
            exposure_dates
        )))
        all_dates.sort()
        time_list = all_dates.tolist()

        if schema == "ALFONSI":
            dt = np.min(floating_schedule_trade)

        import time
        start_time = time.perf_counter()

        sim_results_dict = self.simulate(time_list, nb_mc, dt, schema)
        end_time = time.perf_counter()

        end_mc_time = time.perf_counter()
        elapsed_time = end_mc_time - start_time
        print(f"MC simulation time: {elapsed_time:.4f} seconds")
    
        # Convert dict results to array for path pricer
        sim_results = np.zeros((nb_mc, len(time_list), self.x0_jax.shape[0], self.x0_jax.shape[1]))
        for path_idx in range(nb_mc):
            for t_idx, t in enumerate(time_list):
                sim_results[path_idx, t_idx] = sim_results_dict[path_idx][t]
        
        end_mc_time2 = time.perf_counter()
        elapsed_time2 = end_mc_time2 - end_mc_time
        print(f"Simulation swapping time: {elapsed_time2:.4f} seconds")

        for i, valuation_date in enumerate(exposure_dates):
            last_fixed_t, fixed_schedule_all, _ = self._get_updated_schedule(
                fixed_schedule_trade, valuation_date, main_date
            )
            last_floating_t, floating_schedule_all, _ = self._get_updated_schedule(
                floating_schedule_trade, valuation_date, main_date
            )
        
            if len(fixed_schedule_all) >= 2 and len(floating_schedule_all) >= 2:
                for path in range(nb_mc):
                    swap_value, float_leg_coupons, fixed_leg_coupons = self._compute_swap_mc_path_pricer(
                        valuation_date, fixed_rate, spread, 
                        floating_schedule_all, fixed_schedule_all, 
                        sim_results, path, time_list
                    )

                    # Ensure coupons are numpy arrays (not JAX arrays)
                    float_leg_coupons = np.asarray(float_leg_coupons)
                    fixed_leg_coupons = np.asarray(fixed_leg_coupons)

                    # Adjust first fixed coupon for accrued interest
                    if last_fixed_t is not None and last_fixed_t < fixed_schedule_all[0] and len(fixed_schedule_all) > 1:
                        ratio = (fixed_schedule_all[1] - last_fixed_t) / (fixed_schedule_all[1] - fixed_schedule_all[0])
                        fixed_leg_coupons[0] *= ratio

                    # Adjust first floating coupon for accrued interest
                    if last_floating_t is not None and last_floating_t < floating_schedule_all[0] and len(floating_schedule_all) > 1:
                        ratio = (floating_schedule_all[1] - last_floating_t) / (floating_schedule_all[1] - floating_schedule_all[0])
                        float_leg_coupons[0] *= ratio

                    # Recalculate swap value with adjusted coupons
                    swap_value = np.sum(float_leg_coupons) - np.sum(fixed_leg_coupons)
                    exposure_profile[i, path] = swap_value
        pricing_end_time = time.perf_counter()
        pricing_elapsed_time =pricing_end_time 
        print(f" Pricing time: {pricing_elapsed_time:.4f} seconds")
        return exposure_profile

    def _compute_swap_mc_path_pricer_batched_fast(
    self,
    valuation_date: float,
    fixed_rate: float,
    spread: float,
    floating_schedule: List[float],
    fixed_schedule: List[float],
    sim_results: np.ndarray,
    paths: np.ndarray,
    time_list: List[float]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Ultra-fast batched version using JAX vmap for trace computations.
        """
        import jax
        import jax.numpy as jnp
    
        n_paths = len(paths)
        time_array = np.array(time_list, dtype=np.float64)
    
        def get_indices(dates):
            indices = np.searchsorted(time_array, dates)
            return np.clip(indices, 0, len(time_array) - 1)
    
        paths_data = sim_results[paths]
    
        # Pre-compile vectorized trace function
        @jax.jit
        def batch_trace_u1(states):
            """Compute traces for multiple states at once"""
            return jax.vmap(lambda x: jax_trace_uv(self.u1_jax, x))(states)
    
        @jax.jit
        def batch_trace_u2(states):
            """Compute traces for multiple states at once"""
            return jax.vmap(lambda x: jax_trace_uv(self.u2_jax, x))(states)
    
        float_leg_coupons_all = np.array([], dtype=np.float64)
        fixed_leg_coupons_all = np.array([], dtype=np.float64)
    
        # Process floating leg
        if len(floating_schedule) >= 2:
            n_float = len(floating_schedule) - 1
        
            val_idx = int(get_indices([valuation_date])[0])
            float_indices = get_indices(floating_schedule)
        
            # Get all states for all paths at valuation date
            x_t0_all = paths_data[:, val_idx]
        
            # Vectorized trace computation - ALL paths at once
            traces_t0 = np.array(batch_trace_u1(x_t0_all))
        
            denom_base_all = np.exp(-self.wishart_model.alpha * valuation_date) * (1.0 + traces_t0)
        
            # Get all states at all dates for all paths: shape (n_paths, n_dates, ...)
            xx_all = paths_data[:, float_indices]
        
            # Flatten to compute all traces at once: (n_paths * n_dates, ...)
            xx_flat = xx_all.reshape(-1, xx_all.shape[2], xx_all.shape[3])
        
            # Compute ALL traces in one go (this is the key speedup!)
            traces_u1_flat = np.array(batch_trace_u1(xx_flat))
            traces_u1_all = traces_u1_flat.reshape(n_paths, len(floating_schedule))
        
            # For u2, only need n_float traces per path
            xx_flat_u2 = xx_all[:, :-1].reshape(-1, xx_all.shape[2], xx_all.shape[3])
            traces_u2_flat = np.array(batch_trace_u2(xx_flat_u2))
            traces_u2_all = traces_u2_flat.reshape(n_paths, n_float)
        
            float_schedule_array = np.array(floating_schedule, dtype=np.float64)
            exp_factors = np.exp(-self.wishart_model.alpha * float_schedule_array)
        
            p_ts_all = (exp_factors[np.newaxis, :] * (1.0 + traces_u1_all)) / denom_base_all[:, np.newaxis]
            spread_payments_all = traces_u2_all / denom_base_all[:, np.newaxis]
        
            if self.wishart_model.pseudo_inverse_smoothing:
                adjustments = np.array([self.wishart_model.initial_curve_alpha.get_alpha(t) 
                                       for t in float_schedule_array], dtype=np.float64)
                p_ts_all = p_ts_all * adjustments[np.newaxis, :]
                spread_payments_all = spread_payments_all * adjustments[np.newaxis, :-1]
        
            dcf = np.diff(float_schedule_array)
            float_leg_coupons_all = (
                (p_ts_all[:, :-1] - p_ts_all[:, 1:]) + 
                spread_payments_all + 
                p_ts_all[:, 1:] * spread * dcf[np.newaxis, :]
            )
            float_leg_coupons_all = np.asarray(float_leg_coupons_all, dtype=np.float64).copy()
    
        # Process fixed leg
        if len(fixed_schedule) >= 2:
            n_fix = len(fixed_schedule) - 1
        
            val_idx = int(get_indices([valuation_date])[0])
            fixed_indices = get_indices(fixed_schedule)
        
            x_t0_all = paths_data[:, val_idx]
            traces_t0_fix = np.array(batch_trace_u1(x_t0_all))
        
            denom_base_all_fix = np.exp(-self.wishart_model.alpha * valuation_date) * (1.0 + traces_t0_fix)
        
            xx_all_fix = paths_data[:, fixed_indices]
            xx_flat_fix = xx_all_fix.reshape(-1, xx_all_fix.shape[2], xx_all_fix.shape[3])
        
            # Compute ALL traces in one go
            traces_u1_flat_fix = np.array(batch_trace_u1(xx_flat_fix))
            traces_u1_all_fix = traces_u1_flat_fix.reshape(n_paths, len(fixed_schedule))
        
            fixed_schedule_array = np.array(fixed_schedule, dtype=np.float64)
            exp_factors_fix = np.exp(-self.wishart_model.alpha * fixed_schedule_array)
        
            p_ts_all_fix = (exp_factors_fix[np.newaxis, :] * (1.0 + traces_u1_all_fix)) / denom_base_all_fix[:, np.newaxis]
        
            if self.wishart_model.pseudo_inverse_smoothing:
                adjustments_fix = np.array([self.wishart_model.initial_curve_alpha.get_alpha(t) 
                                           for t in fixed_schedule_array], dtype=np.float64)
                p_ts_all_fix = p_ts_all_fix * adjustments_fix[np.newaxis, :]
        
            dcf_fix = np.diff(fixed_schedule_array)
            fixed_leg_coupons_all = p_ts_all_fix[:, 1:] * fixed_rate * dcf_fix[np.newaxis, :]
            fixed_leg_coupons_all = np.asarray(fixed_leg_coupons_all, dtype=np.float64).copy()
    
        swap_values = np.sum(float_leg_coupons_all, axis=1) - np.sum(fixed_leg_coupons_all, axis=1)
    
        return swap_values, float_leg_coupons_all, fixed_leg_coupons_all


    def compute_swap_exposure_profile_vectorized(
        self,
        exposure_dates: List[float],
        fixed_rate: float,
        spread: float,
        floating_schedule_trade: List[float],
        fixed_schedule_trade: List[float],
        nb_mc: int,
        dt: float,
        main_date: float = 0.0,
        schema: str = "EULER_FLOORED",
        batch_size: int = 500  # Increased batch size for better performance
        ) -> np.ndarray:
        """
        Compute swap exposure profile with ultra-fast JAX vectorization.
        """
        import time
    
        exposure_profile = np.zeros((len(exposure_dates), nb_mc), dtype=np.float64)
    
        all_dates = np.unique(np.concatenate((
            fixed_schedule_trade, 
            floating_schedule_trade, 
            exposure_dates
        )))
        all_dates.sort()
        time_list = all_dates.tolist()
    
        # print(f"exposure_dates: {exposure_dates}")

        if schema == "ALFONSI":
            dt = np.min(floating_schedule_trade)
    
        # MC simulation
        start_time = time.perf_counter()
        sim_results_dict = self.simulate(time_list, nb_mc, dt, schema)
        mc_time = time.perf_counter() - start_time
        print(f"MC simulation time: {mc_time:.4f} seconds")
    
        # Convert dict to array
        swap_start = time.perf_counter()
        sim_results = np.zeros((nb_mc, len(time_list), self.x0_jax.shape[0], self.x0_jax.shape[1]))
        for path_idx in range(nb_mc):
            for t_idx, t in enumerate(time_list):
                sim_results[path_idx, t_idx] = sim_results_dict[path_idx][t]
        swap_time = time.perf_counter() - swap_start
        print(f"Simulation swapping time: {swap_time:.4f} seconds")
    
        # Pricing
        pricing_start = time.perf_counter()
        for i, valuation_date in enumerate(exposure_dates):
            last_fixed_t, fixed_schedule_all, _ = self._get_updated_schedule(
                fixed_schedule_trade, valuation_date, main_date
            )
            last_floating_t, floating_schedule_all, _ = self._get_updated_schedule(
                floating_schedule_trade, valuation_date, main_date
            )
        
            if len(fixed_schedule_all) >= 2 and len(floating_schedule_all) >= 2:
                for batch_start in range(0, nb_mc, batch_size):
                    batch_end = min(batch_start + batch_size, nb_mc)
                    batch_paths = np.arange(batch_start, batch_end, dtype=np.int32)
                
                    # Use the ultra-fast version
                    swap_values, float_leg_coupons, fixed_leg_coupons = self._compute_swap_mc_path_pricer_batched_fast(
                        valuation_date, fixed_rate, spread, 
                        floating_schedule_all, fixed_schedule_all, 
                        sim_results, batch_paths, time_list
                    )
                
                    # Adjust coupons
                    if last_fixed_t is not None and last_fixed_t < fixed_schedule_all[0] and len(fixed_schedule_all) > 1:
                        ratio = (fixed_schedule_all[1] - last_fixed_t) / (fixed_schedule_all[1] - fixed_schedule_all[0])
                        fixed_leg_coupons[:, 0] *= ratio
                
                    if last_floating_t is not None and last_floating_t < floating_schedule_all[0] and len(floating_schedule_all) > 1:
                        ratio = (floating_schedule_all[1] - last_floating_t) / (floating_schedule_all[1] - floating_schedule_all[0])
                        float_leg_coupons[:, 0] *= ratio
                
                    swap_values = np.sum(float_leg_coupons, axis=1) - np.sum(fixed_leg_coupons, axis=1)
                    exposure_profile[i, batch_start:batch_end] = swap_values
        
            # Progress tracking
            if (i + 1) % max(1, len(exposure_dates) // 10) == 0:
                elapsed = time.perf_counter() - pricing_start
                remaining = (elapsed / (i + 1)) * (len(exposure_dates) - i - 1)
                print(f"Progress: {i+1}/{len(exposure_dates)} dates, "
                      f"Elapsed: {elapsed:.1f}s, Est. remaining: {remaining:.1f}s")
    
        pricing_time = time.perf_counter() - pricing_start
        print(f"Pricing time: {pricing_time:.4f} seconds")
        # print(f"exposure_dates: {len(exposure_dates)}, {exposure_dates}")
        # print(f"exposure_dates: {len(exposure_profile)},{exposure_profile}")
    
        return exposure_profile