"""
Main LRW interest rate model implementation.

models/interest_rate/lrw_model.py
"""
import jax
import jax.numpy as jnp
from jax import vmap, jit
from jax.scipy import linalg as jlinalg
from functools import partial
from typing import Tuple, Optional, Dict, Any
import numpy as np
from typing import Union, List
import jax.numpy as jnp
# import numpy as np
# from typing import Union, List
# from jax import vmap, jit
# from functools import partial

from .base import BaseInterestRateModel
from .config import LRWModelConfig,SwaptionConfig
from ...core.wishart import WishartBru
from ...core.wishart_jump import WishartWithJump
from ...utils import local_functions as lf

def get_default_lrw_model_config(n=2,
                        alpha=0.03,
                        x0=np.array([[0.02, 0.01], [0.01, 0.02]]),
                        omega=np.array([[0.04, 0.02], [0.02, 0.04]]),
                        m=np.array([[1.0, 0.5], [0.5, 1.0]]),
                        sigma=np.array([[1e-4, 1e-4], [1e-4, 1e-4]]),
                         ):
    """
    Get default LRW model configuration.
    
    Returns
    -------
    LRWModelConfig
        Default LRW model configuration
    """
    return LRWModelConfig(
        n=n,
        alpha=alpha,
        x0=x0,
        omega=omega,
        m=m,
        sigma=sigma,
        is_bru_config=False,
        has_jump=False,
        u1=None,
        u2=None,
        is_spread=False,
        use_range_kutta_for_b=False,
        compute_equity_style_vega=False,
        pseudo_inverse_smoothing=False
    )

def get_default_swaption_config():
    """
    Get default swaption configuration.
    
    Returns
    -------
    SwaptionConfig
        Default swaption configuration
    """
    return SwaptionConfig(
        maturity=5.0,
        tenor=1.0,
        strike=0.02,
        delta_float=0.5,
        delta_fixed=1.0
    )

class LRWModel(BaseInterestRateModel):
    """
    Linear-Rational Wishart (LRW) interest rate model.
    
    This model uses Wishart processes to model interest rates
    with stochastic volatility and correlation.
    """
    
    def __init__(self, config: LRWModelConfig, swaption_config: SwaptionConfig):
        """
        Initialize LRW model.
        
        Parameters
        ----------
        config : LRWModelConfig
            Model configuration
        swaption_config : SwaptionConfig
            Swaption configuration
        """
        # Initialize base class
        super().__init__(swaption_config)
        
        # Store configurations
        self.model_config = config
        self.u1 = jnp.zeros(self.model_config.x0.shape)
        self.u2 = jnp.zeros(self.model_config.x0.shape)
       
        self.swaption_config = swaption_config
        
        # Configuration flags
        self.compute_equity_style_vega = config.compute_equity_style_vega
        self.pseudo_inverse_smoothing = config.pseudo_inverse_smoothing
        self.initial_curve_alpha = None

        self.set_swaption_config(self.swaption_config)
        self.set_wishart_parameter(self.model_config)
        self.wishart.maturity = swaption_config.maturity
    
    def set_swaption_config(self, swaption_config: SwaptionConfig):
        """
        Set swaption configuration.
        
        Parameters
        ----------
        swaption_config : SwaptionConfig
            Swaption configuration object
        """
        self.swaption_config = swaption_config
        
        super().set_swaption_config(swaption_config)
        # Recompute b3 and a3 coefficients
        # # self.compute_b3_a3()
        # self.a3 = self.a3.reshape(self.x0.shape)
        # self.b3 = self.b3.reshape((1, 1))

    def set_wishart_parameter(self,  config: LRWModelConfig):
        """
        Set Wishart parameters.
        
        This method is a placeholder for setting additional Wishart parameters
        if needed in the future.
        """
        self.config = config
          # Model parameters
        self.n = config.n
        self.alpha = config.alpha
        self.x0 = config.x0
        self.omega = config.omega
        self.m = config.m
        self.sigma = config.sigma
        self.sigma2 = jnp.matmul(config.sigma, config.sigma)
        
        # Bru configuration
        self.is_bru_config = config.is_bru_config
        self.beta = self._compute_beta()
        
        # Initialize Wishart process
        
        if config.has_jump:
            self.wishart = WishartWithJump(
                self.n, self.x0, self.omega, self.m, self.sigma
            )
        else:
            if config.is_bru_config:
                self.wishart = WishartBru(
                    self.n, self.x0, self.omega, self.m, self.sigma, 
                    is_bru_config=True
                )
            else:
                self.wishart = WishartWithJump(
                                self.n, self.x0, self.omega, self.m, self.sigma
                            )        
         
        
        self.wishart.use_range_kutta_for_b = config.use_range_kutta_for_b
        
        # Weight matrices
        self.u1 = config.u1 if config.u1 is not None else jnp.zeros(self.x0.shape)
        self.u2 = config.u2 if config.u2 is not None else jnp.zeros(self.x0.shape)
        self.is_spread = config.is_spread
        
        # Common matrices
        self.A = jnp.add(
            jnp.kron(jnp.eye(self.n), self.m), 
            jnp.kron(self.m, jnp.eye(self.n))
        )
        self.b = self._vec(self.omega)
        self.A_inv = jnp.linalg.inv(self.A)
        
        # Initialize computation variables
        self.b3 = 0
        self.a3 = jnp.zeros(self.x0.shape)
        self.partial_ij_b3 = jnp.zeros(self.x0.shape)
        
        # Derivative storage
        self.b3_prime_alpha = 0
        self.a3_prime_alpha = 0
        self.b3_derive_omega = 0
        self.b3_derive_m = 0
        self.a3_derive_m = 0
        
      
   
    def _compute_beta(self) -> float:
        """Compute beta parameter for Bru configuration."""
        if not self.is_bru_config:
            return 0.0
        
        beta_init = self.omega[0, 0] / self.sigma2[0, 0]
        is_consistent = True
        
        for i in range(self.n):
            for j in range(self.n):
                if self.sigma2[i, j] != 0:
                    beta = self.omega[i, j] / self.sigma2[i, j]
                    is_consistent = is_consistent and (beta == beta_init)
        
        return beta_init if is_consistent else 0.0
    
    @staticmethod
    def _vec(matrix: jnp.ndarray) -> jnp.ndarray:
        """Vectorize a matrix."""
        return matrix.T.ravel().reshape(-1, 1)
    
    @staticmethod
    def _vec_inv(vector: jnp.ndarray, shape: Tuple[int, int]) -> jnp.ndarray:
        """Inverse vectorization."""
        return vector.reshape(shape[1], shape[0]).T
    
    def set_pseudo_inverse_smoothing(self, initial_curve_alpha: Any) -> None:
        """
        Enable pseudo-inverse smoothing.
        
        Parameters
        ----------
        initial_curve_alpha : Any
            Initial alpha curve object
        """
        self.initial_curve_alpha = initial_curve_alpha
        self.pseudo_inverse_smoothing = True
    
    def set_weight_matrices(self, u1: jnp.ndarray, u2: Optional[jnp.ndarray] = None) -> None:
        """
        Set weight matrices for the model.
        
        Parameters
        ----------
        u1 : jnp.ndarray
            First weight matrix
        u2 : Optional[jnp.ndarray]
            Second weight matrix (for spreads)
        """
        self.u1 = jnp.array(u1)
        if u2 is not None:
            self.u2 = jnp.array(u2)
            self.is_spread = True
        else:
            self.u2 = jnp.zeros(self.x0.shape)
            self.is_spread = False
    
    @partial(jit, static_argnums=(0,))
    def get_short_rate(self) -> float:
        """
        Get current short rate.
        
        Returns
        -------
        float
            Current short rate
        """
        tr_u1_omega = jnp.trace(self.u1 @ self.omega)
        tr_u1_mx0 = jnp.trace(self.u1 @ jnp.matmul(self.m, self.x0))
        tr_u1_x0 = jnp.trace(self.u1 @ self.x0)
        
        r = tr_u1_omega + 2 * tr_u1_mx0 / (1 + tr_u1_x0)
        r = self.alpha - r
        return r
    
    @partial(jit, static_argnums=(0,))
    def get_short_rate_infinity(self) -> float:
        """
        Get asymptotic short rate.
        
        Returns
        -------
        float
            Asymptotic short rate
        """
        x_infty = self._vec_inv(-jnp.linalg.solve(self.A, self.b), (self.n, self.n))
        
        tr_u1_omega = jnp.trace(self.u1 @ self.omega)
        tr_u1_mx_infty = jnp.trace(self.u1 @ jnp.matmul(self.m, x_infty))
        tr_u1_x_infty = jnp.trace(self.u1 @ x_infty)
        
        r = tr_u1_omega + 2 * tr_u1_mx_infty / (1 + tr_u1_x_infty)
        r = self.alpha - r
        return r
    
    @partial(jit, static_argnums=(0,))
    def get_spread(self) -> float:
        """
        Get current spread.
        
        Returns
        -------
        float
            Current spread
        """
        return jnp.trace(self.u2 @ self.x0)
    
    @partial(jit, static_argnums=(0,))
    def get_spread_infinity(self) -> float:
        """
        Get asymptotic spread.
        
        Returns
        -------
        float
            Asymptotic spread
        """
        x_infty = self._vec_inv(-jnp.linalg.solve(self.A, self.b), (self.n, self.n))
        return jnp.trace(self.u2 @ x_infty)
    
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

    def bond(self, t: float) -> float:
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
    
    def spread(self, t: float) -> float:
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
    
    #endregion

    def compute_bar_b1_a1(self, t: float) -> Tuple[float, jnp.ndarray]:
        """
        Compute adjusted b1 and a1 coefficients.
        
        Parameters
        ----------
        t : float
            Time
            
        Returns
        -------
        b1_bar : float
            Adjusted b1 coefficient
        a1 : jnp.ndarray
            a1 coefficient matrix
        """
        b1, a1 = self.wishart.compute_mean_decompose(self.u1, t)
        
        if self.pseudo_inverse_smoothing and self.initial_curve_alpha is not None:
            curve_alpha_adjustment = self.initial_curve_alpha.get_alpha(t)
            return curve_alpha_adjustment * (1 + b1), curve_alpha_adjustment * a1
        else:
            return 1 + b1, a1
    
    def compute_b2_a2(self, t: float) -> Tuple[float, jnp.ndarray]:
        """
        Compute b2 and a2 coefficients for spread.
        
        Parameters
        ----------
        t : float
            Time
            
        Returns
        -------
        b2 : float
            b2 coefficient
        a2 : jnp.ndarray
            a2 coefficient matrix
        """
        return self.wishart.compute_mean_decompose(self.u2, t)
    
    def compute_swap_rate(self) -> float:
        """
        Compute swap rate.
        
        Returns
        -------
        float
            Swap rate
        """
        self.compute_b3_a3()
        
        # Floating leg
        zero_coupon_tn1 = self.bond(self.maturity)
        zero_coupon_tn2 = self.bond(self.maturity + self.tenor)
        floating_leg_spread = 0
        
        if self.is_spread:
            for i in range(0, int(self.tenor / self.delta_float)):
                ti = self.maturity + i * self.delta_float
                spread_ti = self.spread(ti)
                floating_leg_spread += spread_ti
        floating_leg_spread=0.0
        # Fixed leg
        fixed_leg = 0
        for i in range(1, int(self.tenor / self.delta_fixed) + 1):
            t1 = self.maturity + i * self.delta_fixed
            zero_coupon_t1 = self.bond(t1)
            fixed_leg += zero_coupon_t1
        
        swap_rate = (zero_coupon_tn1 - zero_coupon_tn2 + floating_leg_spread) / (self.delta_fixed * fixed_leg)
        
        return swap_rate

    def price_swap(self) -> float:
        """
        Compute swap price.
        
        Returns
        -------
        float
            Swap price
        """
        self.compute_b3_a3()
        
        # Floating leg
        zero_coupon_tn1 = self.bond(self.maturity)
        zero_coupon_tn2 = self.bond(self.maturity + self.tenor)
                
        fixed_leg=[]
        floating_leg=[]

         
        for i in range(0, int(self.tenor / self.delta_float)):
            ti = self.maturity + i * self.delta_float
            spread_ti=0.0
            # if self.is_spread:
            #     spread_ti = self.spread(ti)                

            forward_rate = self.compute_forward_rates(ti)
            float_coupon = forward_rate * self.delta_float*self.bond(ti+self.delta_float)
            # floating_leg.append(spread_ti + float_coupon)
        floating_leg.append(zero_coupon_tn1 - zero_coupon_tn2)
        # Fixed leg
        # fixed_leg = 0
        for i in range(1, int(self.tenor / self.delta_fixed) + 1):
            t1 = self.maturity + i * self.delta_fixed
            zero_coupon_t1 = self.bond(t1)
           
            fixed_leg.append(zero_coupon_t1*self.delta_fixed*self.strike)
        
        swap_price = sum(floating_leg) - sum(fixed_leg)

          
        return swap_price
    
    def compute_annuity(self) -> Tuple[float, float]:
        """
        Compute annuity and ATM swap rate.
        
        Returns
        -------
        annuity : float
            Annuity value
        swap_rate : float
            ATM swap rate
        """
        self.compute_b3_a3()
        
        # Floating leg
        zero_coupon_tn1 = self.bond(self.maturity)
        zero_coupon_tn2 = self.bond(self.maturity + self.tenor)
        floating_leg_spread = 0
        
        if self.is_spread:
            for i in range(0, int(self.tenor / self.delta_float)):
                ti = self.maturity + i * self.delta_float
                spread_ti = self.spread(ti)
                floating_leg_spread += spread_ti
        
        # Fixed leg
        fixed_leg = 0
        annuity = 0
        for i in range(1, int(self.tenor / self.delta_fixed) + 1):
            t1 = self.maturity + i * self.delta_fixed
            zero_coupon_t1 = self.bond(t1)
            fixed_leg += zero_coupon_t1
            annuity += zero_coupon_t1
        
        swap_rate = (zero_coupon_tn1 - zero_coupon_tn2 + floating_leg_spread) / (self.delta_fixed * fixed_leg)
        annuity = annuity * self.delta_fixed
        
        return annuity, swap_rate
    
    def compute_b3_a3(self) -> int:
        """
        Compute b3 and a3 coefficients for option pricing.
        
        Returns
        -------
        int
            Status (1 for success)
        """
        # Initial values
        b0 = 1.0
        a0 = self.u1.copy()
        
        self.partial_ij_b3 = jnp.zeros(self.x0.shape)
        
        # Compute partial derivatives for Bru configuration
        if self.is_bru_config:
            self._compute_partial_ij_b3()
        
        # Last cash flow
        bb1, a1 = self.compute_bar_b1_a1(self.tenor)
        b0 = b0 - jnp.exp(-self.alpha * self.tenor) * bb1
        a0 = a0 - jnp.exp(-self.alpha * self.tenor) * a1
        
        b0_prime_alpha = self.tenor * jnp.exp(-self.alpha * self.tenor) * bb1
        a0_prime_alpha = self.tenor * jnp.exp(-self.alpha * self.tenor) * a1
        
        # Fixed leg
        for i in range(1, int(self.tenor / self.delta_fixed) + 1):
            t1 = i * self.delta_fixed
            bb1, a1 = self.compute_bar_b1_a1(t1)
            b0 = b0 - self.strike * self.delta_fixed * jnp.exp(-self.alpha * t1) * bb1
            a0 = a0 - self.strike * self.delta_fixed * jnp.exp(-self.alpha * t1) * a1
            
            b0_prime_alpha = b0_prime_alpha + self.strike * self.delta_fixed * t1 * jnp.exp(-self.alpha * t1) * bb1
            a0_prime_alpha = a0_prime_alpha + self.strike * self.delta_fixed * t1 * jnp.exp(-self.alpha * t1) * a1
        
        # Spread (if applicable)
        if self.is_spread:
            for i in range(0, int(self.tenor / self.delta_float)):
                t1 = i * self.delta_float
                b2, a2 = self.compute_b2_a2(t1)
                b0 = b0 + jnp.exp(-self.alpha * t1) * b2
                a0 = a0 + jnp.exp(-self.alpha * t1) * a2
        
        self.b3 = b0
        self.a3 = a0
        self.b3_prime_alpha = b0_prime_alpha
        self.a3_prime_alpha = a0_prime_alpha
        # print(f"LRWModel : a3={self.a3}, b3={self.b3}")
        
        return 1
    
    def _compute_partial_ij_b3(self) -> None:
        """Compute partial derivatives of b3 for Bru configuration."""
        # Time points for derivative computation
        time_points = [0, self.tenor]
        time_points.extend([i * self.delta_fixed for i in range(1, int(self.tenor / self.delta_fixed) + 1)])
        
        for t1 in time_points:
            eAt = jlinalg.expm(self.A * t1)
            v2 = eAt - jnp.eye(self.n * self.n)
            v3 = jnp.transpose(self._vec(jnp.transpose(self.u1))) @ self.A_inv @ v2
            
            sign = -jnp.exp(-self.alpha * t1) if t1 > 0 else 1.0
            if t1 > self.tenor:
                sign *= self.strike * self.delta_fixed
            
            for k in range(self.n):
                for l in range(self.n):
                    sigma2_derive_kl = self._compute_derive_ij_sigma_square(k, l)
                    b_new = self._vec(self.beta * sigma2_derive_kl)
                    b_sigma_ij = v3 @ b_new
                    self.partial_ij_b3 = self.partial_ij_b3.at[k, l].add(sign * b_sigma_ij[0, 0])
    
    # @partial(jit, static_argnums=(0,))
    def _compute_derive_ij_sigma_square(self, i: int, j: int) -> jnp.ndarray:
        """
        Compute derivative of sigma^2 with respect to sigma[i,j].
        
        Parameters
        ----------
        i : int
            Row index
        j : int
            Column index
            
        Returns
        -------
        jnp.ndarray
            Derivative matrix
        """
        d_sigma_ij = jnp.zeros_like(self.sigma)
        d_sigma_ij = d_sigma_ij.at[i, j].set(1.0)
        if i != j:
            d_sigma_ij = d_sigma_ij.at[j, i].set(1.0)
        
        res = d_sigma_ij @ self.sigma + self.sigma @ d_sigma_ij
        return res
    
    def print_model(self) -> None:
        """Print model parameters."""
        super().print_model()
        print(f"\nLRW Model Parameters")
        print(f"n = {self.n}")
        print(f"alpha = {self.alpha}")
        print(f"x0 = {self.x0}")
        print(f"omega = {self.omega}")
        print(f"m = {self.m}")
        print(f"sigma = {self.sigma}")
        print(f"u1 = {self.u1}")
        print(f"u2 = {self.u2}")
        print(f"is_spread = {self.is_spread}")
        print(f"is_bru_config = {self.is_bru_config}")
        print(f"beta = {self.beta}")