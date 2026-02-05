import math  
import cmath
from math import exp, factorial
from pydoc import Helper
import time

import jax.numpy as jnp
import jax
from jax import jit, vmap
import numpy as np
import scipy.linalg as sp
import scipy.integrate as sp_i
import pandas as pd
from QuantLib import *

# from typing import cast
# from ..data.data_helpers import get_daily_data_up_6m_to_5y
# from ..data.data_market_data import convert_tenor_to_years 

import sys
import os
from pathlib import Path

from typing import cast
from ..data.data_helpers import *
from ..data.data_market_data import * 
from ..config import constants

#region To be removed

# try:
#     from typing import cast
#     from ..data.data_helpers import *
#     from ..data.data_market_data import * 

# except ImportError:
#     # sys.path.insert(0, str(Path(__file__).parent.parent.parent))
#     # project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
#     # sys.path.insert(0, project_root)

#     current_file = os.path.abspath(__file__)
#     project_root = current_file

#     # Go up until we find the wishart_processes directory
#     while os.path.basename(project_root) != "LinearRationalWishart_NewCode" and project_root != os.path.dirname(project_root):
#         project_root = os.path.dirname(project_root)

#     if os.path.basename(project_root) != "LinearRationalWishart_NewCode":
#         # Fallback to hardcoded path
#         project_root = r"E:\OneDrive\Dropbox\LinearRationalWishart_Work\Code\ED\LinearRationalWishart\LinearRationalWishart_NewCode"

#     print(f"Using project root: {project_root}")
#     sys.path.insert(0, project_root)

#     from linear_rational_wishart.data.data_helpers import *
#     from linear_rational_wishart.data.data_market_data import * 

  #endregion
   
def forward_swap_rate_calculation_jax(stn, stp, sumTn, sumTp):
        """JAX-compiled forward swap rate calculation"""
        sumTpTn = sumTn - sumTp
        return (stn * sumTn - stp * sumTp) / sumTpTn
# @jit
def interp_jax(x, xp, fp):
    """JAX-compatible interpolation function"""
    return jnp.interp(x, xp, fp)

# @jit
def log_interpolation_jax(timeToMat, T_array, logDf_array):
    """JAX-compiled log-linear interpolation for discount factors"""
    logZc = jnp.interp(timeToMat, T_array, logDf_array)
    return jnp.exp(logZc)

# @jit
def bond_price_from_rate_jax(zcRate, timeToMat):
    """JAX-compiled bond price calculation from zero-coupon rate"""
    return jnp.exp(-zcRate * timeToMat)

# @jit
def zc_rate_from_price_jax(zcPrice, timeToMat):
    """JAX-compiled zero-coupon rate calculation from price"""
    return -jnp.log(zcPrice) / timeToMat

# @jit
def bootstrap_bond_calculation_jax(rate, sum_pv, tau):
    """JAX-compiled bootstrap bond calculation"""
    return (1 - rate * sum_pv) / (1 + rate * tau)

# @jit
def simple_bond_interpolation_jax(t1, T_array, Rate_array):
    """JAX-compiled simple bond price calculation with interpolation"""
    # Find interpolation indices
    i = jnp.searchsorted(T_array, t1, side='right') - 1
    i = jnp.clip(i, 0, len(T_array) - 2)
    
    # Handle extrapolation cases
    is_before_first = t1 < T_array[0]
    is_after_last = t1 > T_array[-1]
    is_in_range = ~(is_before_first | is_after_last)
    
    # Interpolation for in-range values
    alpha = jnp.where(is_in_range, 
                      (t1 - T_array[i]) / (T_array[i+1] - T_array[i]), 
                      0.0)
    r2_interp = jnp.where(is_in_range,
                          (1 - alpha) * Rate_array[i] + alpha * Rate_array[i+1],
                          0.0)
    
    # Extrapolation rates
    r2_before = Rate_array[0]
    r2_after = Rate_array[-1]
    
    # Select appropriate rate
    r2 = jnp.where(is_before_first, r2_before,
                   jnp.where(is_after_last, r2_after, r2_interp))
    
    return 1.0 / (1.0 + t1 * r2)

def GetBootstrapRateCurveQL(T, Tenor, Rate, multiplier=1.0):
    """
    Bootstrap rate curve using QuantLib with JAX-optimized post-processing
    """
    # Convert inputs to JAX arrays for later processing
    T_jax = jnp.array(T)
    Rate_jax = jnp.array(Rate)
    
    # QuantLib bootstrapping (unchanged for compatibility)
    today = Date(31, January, 2022)
    Settings.instance().evaluationDate = today
    tenorTypeMatMapsQL = [Days, Weeks, Months, Years]
    tenorTypes = ['D', 'W', 'M', 'Y']
    
    ois_helpers = [OISRateHelper(0,
                         Period(int(tenor[0:len(tenor)-1]), 
                               tenorTypeMatMapsQL[tenorTypes.index(tenor[len(tenor)-1])]),
                         QuoteHandle(SimpleQuote(rate*multiplier)),
                         Estr(),
                         YieldTermStructureHandle(),
                         True)
           for tenor, rate in zip(Tenor, Rate)]

    discount_curve = PiecewiseLogLinearDiscount(0, TARGET(), ois_helpers, Actual360())
    discount_curve.enableExtrapolation()
    
    # Use vectorized operations for bootstrap curve calculation
    @jit
    def vectorized_discount(t_array):
        return jnp.array([discount_curve.discount(float(t)) for t in t_array])
    
    bootstrapRateCurve = [discount_curve.discount(float(t), True) for t in T]
    
    maxTenor = 20
    allT = list(np.arange(0, maxTenor, 1/12))
    finalT = list(allT)
    finalT.sort()
    
    finalDf = [discount_curve.discount(t) for t in finalT]
    QLdiscount_curve = WrappedQlCurve(finalT, finalDf)
    
    return bootstrapRateCurve, QLdiscount_curve

class WrappedQlCurve(object):
    def __init__(self, allT, allDf):
        self.allT = jnp.array(allT)
        self.allDf = jnp.array(allDf)
        
        # Pre-compute log values for efficient interpolation
        self.logDf = jnp.log(self.allDf)
    
    # @jit
    def _discount_jax(self, t):
        """JAX-compiled discount factor calculation"""
        return jnp.interp(t, self.allT, self.allDf)
    
    def discount(self, t):
        """Main discount method with JAX optimization"""
        if isinstance(t, (list, np.ndarray)):
            t_array = jnp.array(t)
            return jnp.array([self._discount_jax(ti) for ti in t_array])
        # else:
        elif isinstance(t, (int, float)):
            return float(self._discount_jax(jnp.float32(t)))
        else:
            return float(self._discount_jax(jnp.float32(convert_tenor_to_years(t))))

class OisCurve(object):
    def __init__(self, dataDate, rateDataList, interp="LOGLINEARDF") -> None:
        self._dataDate = dataDate
        self._rateDataList = rateDataList
        self.df = pd.DataFrame()
        self.interp = interp
        
        # Extract data and convert to JAX arrays
        self.Tenor = [x.tenor for x in rateDataList["Object"]]
        self.T = jnp.array([convert_tenor_to_years(x.tenor) for x in rateDataList["Object"]])
        self.Rate = jnp.array([x.rate*x.multiplier for x in rateDataList["Object"]])
        
        # Create DataFrame for compatibility
        self.df["TimeToMat"] = [convert_tenor_to_years(x.tenor) for x in rateDataList["Object"]]
        self.df["Rate"] = [x.rate*x.multiplier for x in rateDataList["Object"]]
        self.df = self.df.sort_values(by=['TimeToMat'], ascending=[True])
        
        # Update JAX arrays after sorting
        self.T = jnp.array(self.df["TimeToMat"].values)
        self.Rate = jnp.array(self.df["Rate"].values)
        
        # Bootstrap the curve
        self.bootstrapDfCurve, self.QLdiscount_curve = GetBootstrapRateCurveQL(
            self.T.tolist(), self.Tenor, self.Rate.tolist())
        
        # Convert to JAX arrays
        self.bootstrapDfCurve = jnp.array([self.QLdiscount_curve.discount(float(t)) for t in self.T])
        self.logBootstrapDfCurve = jnp.log(self.bootstrapDfCurve)

        ##todo set these properly from conventions
        self.DeltaFixedLeg =1.0
        self.DeltaFloatingLeg =1.0
        
    def bootstrapRateCurve(self, fixedLegFreq=1.0):
        """Bootstrap rate curve with JAX optimizations"""
        bootstrapRateCurve = self.Rate.copy()
        
        # Find first index where T > 1
        i1 = jnp.where(self.T > 1.0)[0]
        i1 = int(i1[0]) if len(i1) > 0 else len(self.T)
        
        tau = jnp.float32(fixedLegFreq)
        
        for i2 in range(i1, len(self.T)):
            sum_pv = jnp.float32(0.0)
            t1 = tau
            
            # Calculate sum of present values
            while t1 < self.T[i2]:
                bond_price = self._bondSimple(t1)
                sum_pv += tau * bond_price
                t1 += tau
            
            # Handle stub period
            stub = self.T[i2] - (t1 - tau)
            if stub > 0:
                bond_price = self._bondSimple(float(self.T[i2]))
                sum_pv += stub * bond_price
            
            # Bootstrap calculation using JAX
            rate_i2 = float(self.Rate[i2])
            b2 = bootstrap_bond_calculation_jax(rate_i2, sum_pv, tau)
            r2 = (1.0 / b2 - 1) / self.T[i2]
            bootstrapRateCurve = bootstrapRateCurve.at[i2].set(r2)
        
        self.bootstrapRateCurve = bootstrapRateCurve
        self.df["ZcRate"] = self.bootstrapRateCurve.tolist()
     # @jit
    
    def RateInterpolate_jax(self, t, T_array, Rate_array):
        """JAX-compiled rate interpolation"""
        return jnp.interp(t, T_array, Rate_array)
    
    def RateInterpolate(self, t):
        """Rate interpolation with JAX optimization"""
        return float(self.RateInterpolate_jax(jnp.float32(t), self.T, self.Rate))
    def _bondSimple(self, t1):
        """Simple bond price calculation with JAX optimization"""
        return self.QLdiscount_curve.discount(t1)
    
    # @jit
    def _bondSimple_jax(self, t1):
        """JAX-compiled simple bond price calculation"""
        return simple_bond_interpolation_jax(t1, self.T, self.Rate)
    
    def AnnualizingRates(self):
        """Annualize rates with JAX vectorization"""
        # @jit
        def annualize_rate(rate, time):
            return jnp.power(1 + time * rate, 1.0 / time) - 1.0
        
        # Vectorized annualization
        annualized_rates = vmap(annualize_rate)(self.bootstrapRateCurve, self.T)
        
        self.bootstrapRateCurve = annualized_rates
        self.df["ZcRate"] = self.bootstrapRateCurve.tolist()

    # @jit 
    def _bond_zc_rate_jax(self, timeToMat):
        """JAX-compiled zero-coupon rate calculation"""
        if self.interp == "LOGLINEARDF":
            logZc = jnp.interp(timeToMat, self.T, self.logBootstrapDfCurve)
            return -logZc / timeToMat
        else:
            return jnp.interp(timeToMat, self.T, self.bootstrapRateCurve)
    
    def bond_zc_rate(self, timeToMat):
        """Calculate zero-coupon rate with QuantLib fallback and JAX optimization"""
        # Use QuantLib for primary calculation
        zcPrice = self.QLdiscount_curve.discount(timeToMat)
        if isinstance(timeToMat,  float):
            zcRate = zc_rate_from_price_jax(jnp.float32(zcPrice), jnp.float32(timeToMat))
        else:
            zcRate = zc_rate_from_price_jax(jnp.float32(zcPrice), jnp.float32(convert_tenor_to_years(timeToMat)))

        return float(zcRate)
    
    def Bond(self, timeToMat):
        """Alias for bond_price"""
        return self.bond_price(timeToMat)
    
    def bond_price(self, timeToMat):
        """Calculate bond price with JAX optimization"""
        if hasattr(self, 'QLdiscount_curve'):
            return self.QLdiscount_curve.discount(timeToMat)
        else:
            # Fallback to JAX calculation
            zcRate = self.bond_zc_rate(timeToMat)
            return float(bond_price_from_rate_jax(jnp.float32(zcRate), jnp.float32(timeToMat)))
    
    # @jit
    def getZcFromRate_jax(self, zcRate, timetoMat):
        """JAX-compiled zero-coupon price from rate"""
        return bond_price_from_rate_jax(zcRate, timetoMat)
    
    def getZcFromRate(self, zcRate, timetoMat):
        """Calculate zero-coupon price from rate"""
        return float(self.getZcFromRate_jax(jnp.float32(zcRate), jnp.float32(timetoMat)))
    
    # @jit
    def get_rate_from_zc_jax(self, zcPrice, timetoMat):
        """JAX-compiled zero-coupon rate from price"""
        return zc_rate_from_price_jax(zcPrice, timetoMat)
    
    def get_rate_from_zc(self, zcPrice, timetoMat):
        """Calculate zero-coupon rate from price"""
        return float(self.get_rate_from_zc_jax(jnp.float32(zcPrice), jnp.float32(timetoMat)))
    
    # Vectorized methods for batch processing
    # @jit
    def bond_price_vectorized(self, timeToMat_array):
        """Vectorized bond price calculation"""
        return vmap(self.getZcFromRate_jax)(
            vmap(self._bond_zc_rate_jax)(timeToMat_array), 
            timeToMat_array
        )
    
    # @jit 
    def bond_zc_rate_vectorized(self, timeToMat_array):
        """Vectorized zero-coupon rate calculation"""
        return vmap(self._bond_zc_rate_jax)(timeToMat_array)
    
    def forward_swap_rate(self, Tp: float, Tn: float, deltaFixedLeg: float, 
                       deltaFloatingLeg: float):##, oisCurve: OisCurve):
        """
        Compute forward swap rate with JAX optimization
        """
        
        v = 0.0
        check1 = round((Tn - Tp) / deltaFixedLeg) == ((Tn - Tp) / deltaFixedLeg)
        check2 = round((Tn - Tp) / deltaFloatingLeg) == ((Tn - Tp) / deltaFloatingLeg)
         
        if check1 & check2:
            # Calculate bond sums using vectorized operations
            t_values_Tn = jnp.arange(deltaFixedLeg, Tn + deltaFixedLeg/2, deltaFixedLeg)
            t_values_Tp = jnp.arange(deltaFixedLeg, Tp + deltaFixedLeg/2, deltaFixedLeg)
            
            bond_values_Tn = jnp.array([self.Bond(float(t)) for t in t_values_Tn])
            bond_values_Tp = jnp.array([self.Bond(float(t)) for t in t_values_Tp])

            if ((float(self.DeltaFixedLeg) == deltaFixedLeg) & 
                (float(self.DeltaFloatingLeg) == deltaFloatingLeg)):
                stn = self.RateInterpolate(Tn)
                stp = self.RateInterpolate(Tp)
                v = float(self._forward_swap_rate_jax(
                    jnp.float32(Tp), jnp.float32(Tn), jnp.float32(deltaFixedLeg),
                    jnp.float32(stn), jnp.float32(stp), bond_values_Tn, bond_values_Tp))
            else:
                print("The forward swap we want to compute does not have a tenor structure compatible with the tenor structure of the data.")
        else:
            print("The tenor structure of the fixed leg has to be a multiple of deltaFixedLeg, same applies to floating leg. No stub either on the fixed leg or on the floating leg.")
            
        return v
    
    
    def _forward_swap_rate_jax(self, Tp, Tn, deltaFixedLeg, stn, stp, bond_values_Tn, bond_values_Tp):
        """JAX-compiled forward swap rate calculation"""
        sumTn = jnp.sum(bond_values_Tn)
        sumTp = jnp.sum(bond_values_Tp)
        return forward_swap_rate_calculation_jax(stn, stp, sumTn, sumTp)
if __name__ == "__main__":
    print("Testing OIS Curve with JAX optimizations")
    
    folder = constants.mkt_data_folder #r"C:\Users\edem_\Dropbox\LinearRationalWishart\Data"
    oisRateFile = folder + r"\NewData_ESTR_2024_EUR.csv"
    liborRateFile = folder + r"\NewData_EURIBOR_2024_EUR.csv"
    swaptionFile = folder + r"\allDataEurVolData.csv"
    
    currency = "EUR"
    currentDate = '20240131'
    
    # Note: This assumes MarketData_jax is available
    try:
        dailyData6MTo5Y = get_daily_data_up_6m_to_5y(
           # MarketData.GetDailyDataUp6MTo5Y(
            currentDate, currency, oisRateFile, liborRateFile, swaptionFile, 'DummyDataSource')
        print(dailyData6MTo5Y.short_summary())

        # Create OIS curve with JAX optimizations
        oisCurve = OisCurve(dailyData6MTo5Y.quotation_date
                            , dailyData6MTo5Y.ois_rate_data)
        print("=== OIS Curve JAX Optimization Test ===")
        
        # Process OIS data with JAX-optimized calculations
        for index, oisData in dailyData6MTo5Y.ois_rate_data.iterrows():
            cOisData = cast(RateData, oisData["Object"])
            
            # Use JAX-optimized methods
            timeToMat = oisData["TimeToMat"]
            cOisData.marketZcPrice = oisCurve.bond_price(timeToMat)
            cOisData.marketZcRate = oisCurve.bond_zc_rate(timeToMat)
        
        dailyData6MTo5Y.print()
        
        # Demonstrate vectorized operations
        print("\n=== JAX Vectorized Operations Demo ===")
        test_maturities = jnp.array([0.5, 1.0, 2.0, 5.0, 10.0])
        
        # Vectorized bond prices
        bond_prices = oisCurve.bond_price_vectorized(test_maturities)
        print(f"Bond prices: {bond_prices}")
        
        # Vectorized zero-coupon rates  
        zc_rates = oisCurve.bond_zc_rate_vectorized(test_maturities)
        print(f"ZC rates: {zc_rates}")
        
        print("JAX optimization completed successfully")
        
    except ImportError as e:
        print(f"MarketData_jax not available: {e}")
        print("Please ensure MarketData_jax.py is in the correct path")
    except Exception as e:
        print(f"Error during testing: {e}")
