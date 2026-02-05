import math  
import cmath
from math import exp, factorial

import jax.numpy as jnp
import jax
from jax import jit, vmap
import numpy as np
import scipy.linalg as sp
import scipy.integrate as sp_i
import pandas as pd

from typing import cast
from sympy import false

import sys
import os
from pathlib import Path

from ..models.interest_rate.lrw_model import LRWModel 
from typing import cast
from ..data.data_helpers import *
from ..data.data_market_data import * 
from .oiscurve import *
from ..config import constants

#region To be removed
# try:
    
#     from ..models.interest_rate.lrw_model import LRWModel 
#     from typing import cast
#     from ..data.data_helpers import *
#     from ..data.data_market_data import * 
#     from .oiscurve import *

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

#     from linear_rational_wishart.models.interest_rate.lrw_model import LRWModel 
#     from linear_rational_wishart.data.data_helpers import *
#     from linear_rational_wishart.data.data_market_data import * 
#     from linear_rational_wishart.curves.oiscurve import *
#endregion

# JAX-optimized utility functions
# @jit
def interp_jax(x, xp, fp):
    """JAX-compiled interpolation function"""
    return jnp.interp(x, xp, fp)

# @jit
def compute_annuity_jax(schedule, bond_prices):
    """JAX-compiled annuity calculation"""
    deltas = jnp.diff(schedule)
    # Handle the case where bond_prices might be shorter
    relevant_bonds = bond_prices[1:len(deltas)+1]
    return jnp.sum(deltas * relevant_bonds)

# @jit
def compute_swap_pv_jax(rate, annuity, p0, pn):
    """JAX-compiled swap present value calculation"""
    return rate * annuity - (p0 - pn)

# @jit
def compute_aggregate_a_jax(A_values, T_values, DeltaFloatingLeg):
    """JAX-compiled aggregate A calculation"""
    Aaggregate = jnp.zeros_like(A_values)
    Aaggregate = Aaggregate.at[0].set(A_values[0])
    
    for i in range(1, len(A_values)):
        nb1 = (T_values[i] - T_values[i-1]) / DeltaFloatingLeg
        v = (A_values[i] - A_values[i-1]) / nb1
        Aaggregate = Aaggregate.at[i].set(v)
    
    return Aaggregate

# @jit
def forward_swap_rate_calculation_jax(stn, stp, sumTn, sumTp):
    """JAX-compiled forward swap rate calculation"""
    sumTpTn = sumTn - sumTp
    return (stn * sumTn - stp * sumTp) / sumTpTn

# @jit
def generate_schedule_jax(start, end, delta):
    """JAX-compiled schedule generation"""
    n_periods = jnp.floor((end - start) / delta).astype(int)
    schedule = jnp.linspace(start, start + n_periods * delta, n_periods + 1)
    
    # Add final maturity if not already included
    schedule = jnp.where(jnp.abs(schedule[-1] - end) > 1e-10,
                        jnp.append(schedule, end),
                        schedule)
    return schedule

class IborCurve(object):
    def __init__(self, dataDate, rateDataList, oisCurve, deltaFloatLeg=0.5, deltaFixedLeg=1.0, usingFRA=False) -> None:
        self._dataDate = dataDate
        self._rateDataListInitial = rateDataList
        self._rateDataList = rateDataList
        self.df = pd.DataFrame()
        self.HasBeenInterpolated = False
        
        # Extract data and convert to JAX arrays
        self.T = jnp.array([convert_tenor_to_years(x.tenor) for x in rateDataList["Object"]])
        self.Rate = jnp.array([x.rate*x.multiplier for x in rateDataList["Object"]])
        
        # Create DataFrame for compatibility
        self.df["TimeToMat"] = [convert_tenor_to_years(x.tenor) for x in rateDataList["Object"]]
        self.df["Rate"] = [x.rate*x.multiplier for x in rateDataList["Object"]]
        self.df = self.df.sort_values(by=['TimeToMat'], ascending=[True])
        
        # Store base data as JAX arrays
        self._baseT = jnp.array(self.df["TimeToMat"].values)
        self._baseRate = jnp.array(self.df["Rate"].values)

        # Update working arrays
        self.T = jnp.array(self.df["TimeToMat"].values)
        self.Rate = jnp.array(self.df["Rate"].values)
        
        # Convert parameters to JAX arrays
        self.DeltaFloatLeg = jnp.float32(deltaFloatLeg)
        self.deltaFixedLeg = jnp.float32(deltaFixedLeg)
        
        # Generate regularly spaced rates
        self.RegularlySpacedLongRates(float(self.deltaFixedLeg))
        
        self.oisCurve = oisCurve
        self.usingFRA = usingFRA
        startMaturitySwap = float(self.DeltaFloatLeg)
        self.ComputeA(float(self.DeltaFloatLeg), float(self.deltaFixedLeg), self.oisCurve, self.usingFRA, startMaturitySwap)
        
    def RegularlySpacedLongRates(self, deltaFixedLeg: float):
        """Generate regularly spaced long rates with JAX optimization"""
        lr = []
        lt = []
        
        # Collect short-term rates
        for i in range(len(self.T)):
            if self.T[i] < deltaFixedLeg:
                lt.append(float(self.T[i]))
                lr.append(float(self.Rate[i]))
        
        EuriborToInsert = {}
        t1 = deltaFixedLeg
        
        while t1 <= self.T[-1]:
            if t1 in self.T:
                j = jnp.where(self.T == t1)[0][0]
                r = float(self.Rate[j])
            else:   
                r = self.RateInterpolate(t1)
                tenor = str(t1) + "Y"
                self.HasBeenInterpolated = True
                euriborDataobject =RateData(self._dataDate, tenor, r*100, False, "InterpolatedData")
                dateInt = self._dataDate.strftime("%Y%m%d")
                ticker = tenor + "Interpolated"
                self._rateDataList.loc[len(self._rateDataList)] = {
                    'Dates': dateInt, 'Tickers': ticker, 'Tenor': tenor, 'Data': r, 
                    'QuotationDate': self._dataDate, 'Object': euriborDataobject, 'TimeToMat': t1
                }
               
            lt.append(t1)
            lr.append(r)
            t1 += deltaFixedLeg
            
        self._rateDataList = self._rateDataList.sort_values(by=['TimeToMat'], ascending=True)
        
        # Update JAX arrays
        self.Rate = jnp.array(lr)
        self.T = jnp.array(lt)
  
    # @jit
    def RateInterpolate_jax(self, t, T_array, Rate_array):
        """JAX-compiled rate interpolation"""
        return jnp.interp(t, T_array, Rate_array)
    
    def RateInterpolate(self, t):
        """Rate interpolation with JAX optimization"""
        return float(self.RateInterpolate_jax(jnp.float32(t), self.T, self.Rate))
    
    def RateModelInterpolate(self, t):
        """Model rate interpolation"""
        if hasattr(self, 'RateModel'):
            RateModel_jax = jnp.array(self.RateModel)
            return float(self.RateInterpolate_jax(jnp.float32(t), self.T, RateModel_jax))
        else:
            return 0.0
    
    def ComputeA(self, deltaFloatingLeg1: float, deltaFixedLeg1: float, oisCurve: OisCurve, 
                usingFRA: bool, startMaturitySwap: float):
        """
        Compute A values with JAX optimization
        """
        self.DeltaFloatingLeg = jnp.float32(deltaFloatingLeg1)
        self.DeltaFixedLeg = jnp.float32(deltaFixedLeg1)
        a1 = []

        if not usingFRA:
            for i1 in range(len(self.T)):
                t1 = float(self.T[i1])
                
                # Create schedule
                if t1 <= 1.0:
                    schedule = jnp.array([0.0, t1])
                else:
                    # Use JAX for schedule generation
                    n_periods = int(jnp.floor(t1 / deltaFixedLeg1))
                    schedule = jnp.linspace(0.0, n_periods * deltaFixedLeg1, n_periods + 1)
                    
                    # Add final maturity if needed
                    if jnp.abs(schedule[-1] - t1) > 1e-10:
                        schedule = jnp.append(schedule, t1)
                
                # Calculate annuity using JAX
                annuity = 0.0                     
                for j in range(1, len(schedule)):
                    delta_period = schedule[j] - schedule[j-1]
                    bond_price = oisCurve.bond_price(float(schedule[j]))
                    annuity += delta_period * bond_price
                        
                p0 = 1.0
                pn = oisCurve.bond_price(t1)
                sumOfAs = float(self.Rate[i1]) * annuity - (p0 - pn)
                a1.append(sumOfAs)
             
            self.A = a1
            
            # Compute aggregate A using JAX
            A_jax = jnp.array(self.A)
            T_jax = self.T
            Aaggregate_jax = compute_aggregate_a_jax(A_jax, T_jax, float(self.DeltaFloatingLeg))
            self.Aaggregate = Aaggregate_jax.tolist()
        
        else:
            # FRA case - implement with JAX optimizations
            self._ComputeA_FRA(deltaFloatingLeg1, deltaFixedLeg1, oisCurve, startMaturitySwap)
    
    def _ComputeA_FRA(self, deltaFloatingLeg1: float, deltaFixedLeg1: float, 
                     oisCurve: OisCurve, startMaturitySwap: float):
        """FRA-specific A computation with JAX optimization"""
        a1 = []
        tfloat = deltaFloatingLeg1

        # FRA section
        while tfloat < startMaturitySwap:
            p0 = oisCurve.Bond(tfloat)
            p1 = oisCurve.Bond(tfloat + deltaFloatingLeg1)
            l = (deltaFloatingLeg1 * self.RateInterpolate(tfloat + deltaFloatingLeg1) + 1) * p1 - p0
            a1.append(l)
            tfloat += deltaFloatingLeg1
        
        # Swap section
        while tfloat < self.T[-1]:
            pn = oisCurve.Bond(tfloat)
            fixedLegValue = 0.0
            
            # Calculate fixed leg value
            tj1 = tfloat
            while tj1 >= deltaFixedLeg1:
                fixedLegValue += oisCurve.Bond(tj1)
                tj1 -= deltaFixedLeg1
             
            fixedLegValue *= deltaFixedLeg1
            l = self.RateInterpolate(tfloat) * fixedLegValue - 1 + pn
            a1.append(l)
            tfloat += deltaFloatingLeg1
        
        self.A = a1
        
        # Compute aggregate A
        A_jax = jnp.array(self.A)
        T_jax = self.T
        Aaggregate_jax = compute_aggregate_a_jax(A_jax, T_jax, deltaFloatingLeg1)
        self.Aaggregate = Aaggregate_jax.tolist()

    def ComputeRateModel(self, lrw: LRWModel):
        """Compute rate model with JAX optimization"""
        self.RateModel = []
        sumA = 0.0
        
        for i in range(1, len(self.T)):
            s = 0.0
            
            # Sum A values with JAX optimization
            tFloating = float(self.T[i-1])
            local_sumA = sumA
            
            while tFloating <= self.T[i]:
                local_sumA += lrw.spread(tFloating)
                tFloating += float(self.DeltaFloatLeg)
            
            s = 1.0 - lrw.Bond(float(self.T[i])) + local_sumA
            sumA = local_sumA

            tfixed = float(self.DeltaFixedLeg)
            if tfixed > self.T[i]:
                s = s / (float(self.DeltaFloatingLeg) * lrw.Bond(float(self.T[i])))
            else:
                sumFix = 0.0
                j2 = 0
                tfixed_local = tfixed
                while tfixed_local <= self.T[i]:
                    sumFix += lrw.Bond(tfixed_local)
                    tfixed_local += tfixed
                    j2 += 1
                
                s = s / (tfixed * sumFix)
             
            self.RateModel.append(s)

    # @jit
    def _forward_swap_rate_jax(self, Tp, Tn, deltaFixedLeg, stn, stp, bond_values_Tn, bond_values_Tp):
        """JAX-compiled forward swap rate calculation"""
        sumTn = jnp.sum(bond_values_Tn)
        sumTp = jnp.sum(bond_values_Tp)
        return forward_swap_rate_calculation_jax(stn, stp, sumTn, sumTp)

    def forward_swap_rate(self, Tp: float, Tn: float, deltaFixedLeg: float, 
                       deltaFloatingLeg: float, oisCurve: OisCurve):
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
            
            bond_values_Tn = jnp.array([oisCurve.Bond(float(t)) for t in t_values_Tn])
            bond_values_Tp = jnp.array([oisCurve.Bond(float(t)) for t in t_values_Tp])

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
        
    def forward_swap_rateModel(self, Tp: float, Tn: float, deltaFixedLeg: float, 
                           deltaFloatingLeg: float, lrw: LRWModel):
        """
        Compute forward swap rate for model with JAX optimization
        """
        self.ComputeRateModel(lrw)

        v = 0.0
        check1 = round((Tn - Tp) / deltaFixedLeg) == ((Tn - Tp) / deltaFixedLeg)
        check2 = round((Tn - Tp) / deltaFloatingLeg) == ((Tn - Tp) / deltaFloatingLeg)
         
        if check1 & check2:
            # Vectorized bond sum calculations
            t_values_Tn = jnp.arange(deltaFixedLeg, Tn + deltaFixedLeg/2, deltaFixedLeg)
            t_values_Tp = jnp.arange(deltaFixedLeg, Tp + deltaFixedLeg/2, deltaFixedLeg)
            
            bond_values_Tn = jnp.array([lrw.Bond(float(t)) for t in t_values_Tn])
            bond_values_Tp = jnp.array([lrw.Bond(float(t)) for t in t_values_Tp])

            if ((float(self.DeltaFixedLeg) == deltaFixedLeg) & 
                (float(self.DeltaFloatingLeg) == deltaFloatingLeg)):
                
                stn = lrw.RateModelInterpolate(Tn) if hasattr(lrw, 'RateModelInterpolate') else self.RateModelInterpolate(Tn)
                stp = lrw.RateModelInterpolate(Tp) if hasattr(lrw, 'RateModelInterpolate') else self.RateModelInterpolate(Tp)
                
                v = float(self._forward_swap_rate_jax(
                    jnp.float32(Tp), jnp.float32(Tn), jnp.float32(deltaFixedLeg),
                    jnp.float32(stn), jnp.float32(stp), bond_values_Tn, bond_values_Tp))
            else:
                print("The forward swap we want to compute does not have a tenor structure compatible with the tenor structure of the data.")
        else:
            print("The tenor structure of the fixed leg has to be a multiple of deltaFixedLeg, same applies to floating leg. No stub either on the fixed leg or on the floating leg.")
            
        return v
     
    def getMktA(self, mat):
        """Get market A value with JAX array lookup"""
        Avalue = None
        try: 
            T_array = jnp.array(self.T)
            index = jnp.where(T_array == mat)[0][0]
            Avalue = self.Aaggregate[int(index)]
        except Exception:
            print(f"Unknown Tenor: {mat}")
        
        return Avalue
    
    def getMktFullA(self, mat):
        """Get market full A value with JAX array lookup"""
        Avalue = None
        try: 
            T_array = jnp.array(self.T)
            index = jnp.where(T_array == mat)[0][0]
            Avalue = self.A[int(index)]
        except Exception:
            print(f"Unknown Tenor: {mat}")
        
        return Avalue
   
    # @jit
    def _getModelA_jax(self, mat, spreads, DeltaFloatLeg):
        """JAX-compiled model A calculation"""
        # This would need to be adapted based on the LRW model structure
        # For now, return a placeholder
        return spreads[0] / 1.0  # Simplified calculation
   
    def getModelA(self, mat, lrw: LRWModel):
        """Get model A value with JAX optimization"""
        try:
            T_list = list(self.T)
            index = T_list.index(mat)
            tStart = 0
            if index > 0:
                tStart = float(self.T[index-1])
                
            tFloating = tStart
            Avalue = lrw.spread(tFloating)
            tFloating += float(self.DeltaFloatLeg)
            nbA = 1
            
            while ((tFloating < mat) and (math.fabs(mat - tFloating) > (float(self.DeltaFloatLeg)/2.0))):
                Avalue += lrw.spread(tFloating)
                tFloating += float(self.DeltaFloatLeg)
                nbA += 1
            
            # Handle stub
            if tFloating < mat:
                Avalue += lrw.spread(mat)
                nbA += 1
                
            Avalue /= nbA
            return Avalue
            
        except Exception as e:
            print(f"Error in getModelA: {e}")
            return 0.0
    
    def getModelFullA(self, mat, lrw: LRWModel):
        """Get model full A value with JAX optimization"""
        try:
            T_array = jnp.array(self.T)
            index = jnp.where(T_array == mat)[0][0]
            
            tStart = 0
            tFloating = tStart
            Avalue = lrw.spread(tFloating)
            tFloating += float(self.DeltaFloatLeg)
            
            while ((tFloating < mat) and (math.fabs(mat - tFloating) > (float(self.DeltaFloatLeg)/2.0))):
                Avalue += lrw.spread(tFloating)
                tFloating += float(self.DeltaFloatLeg)
            
            return Avalue
            
        except Exception as e:
            print(f"Error in getModelFullA: {e}")
            return 0.0

    # Vectorized operations for batch processing
    # @vmap
    def RateInterpolate_batch(self, t_array):
        """Vectorized rate interpolation"""
        return self.RateInterpolate_jax(t_array, self.T, self.Rate)
    
    # @vmap  
    def compute_bond_prices_batch(self, t_array):
        """Vectorized bond price calculation"""
        return jnp.array([self.oisCurve.bond_price(float(t)) for t in t_array])


    def getModelA_vmap_compatible(self, mat, lrw: LRWModel):
        """
        JAX vmap-compatible version of getModelA.
        Computes average spread over the floating leg period ending at mat.
        """
        delta = float(self.DeltaFloatLeg)
        T_array = jnp.array(self.T)
    
        # Find tStart: the previous maturity in T, or 0 if mat is the first
        # Use jnp.where to find index without dynamic shapes
        mat_idx = jnp.searchsorted(T_array, mat)
        tStart = jnp.where(mat_idx > 0, T_array[jnp.maximum(mat_idx - 1, 0)], 0.0)
    
        # Generate all possible floating times from tStart to mat
        # Use a fixed maximum number of steps (safe upper bound)
        max_steps = int(jnp.ceil(float(T_array[-1]) / delta)) + 2
    
        # Create array of potential floating times
        steps = jnp.arange(max_steps)
        t_floatings = tStart + steps * delta
    
        # Mask: include times that are < mat (with tolerance) or equal to mat for stub
        tolerance = delta / 2.0
        valid_mask = (t_floatings < mat - tolerance) | (jnp.abs(t_floatings - mat) < 1e-10)
        # Also include the stub at mat if last valid t_floating < mat
    
        # For simplicity: include all t <= mat
        valid_mask = t_floatings <= mat + tolerance
        # But exclude times beyond mat
        valid_mask = valid_mask & (t_floatings <= mat + 1e-10)
    
        # Vectorize spread computation
        spread_vmap = jax.vmap(lrw.spread)
        all_spreads = spread_vmap(t_floatings)
    
        # Compute masked sum and count
        masked_spreads = jnp.where(valid_mask, all_spreads, 0.0)
        total = jnp.sum(masked_spreads)
        count = jnp.sum(valid_mask.astype(jnp.float32))
    
        # Avoid division by zero
        Avalue = jnp.where(count > 0, total / count, 0.0)
    
        return Avalue


    def getModelFullA_vmap_compatible(self, mat, lrw: LRWModel):
        """
        JAX vmap-compatible version of getModelFullA.
        Computes sum of spreads from 0 to mat at DeltaFloatLeg intervals.
        """
        delta = float(self.DeltaFloatLeg)
    
        # Generate all floating times from 0 to mat
        max_steps = int(jnp.ceil(float(self.T[-1]) / delta)) + 2
        steps = jnp.arange(max_steps)
        t_floatings = steps * delta
    
        # Mask: times < mat (with tolerance for floating point)
        tolerance = delta / 2.0
        valid_mask = (t_floatings < mat) & (jnp.abs(mat - t_floatings) > tolerance)
        # Include t=0
        valid_mask = valid_mask | (t_floatings == 0.0)
    
        # Vectorize spread computation  
        spread_vmap = jax.vmap(lrw.spread)
        all_spreads = spread_vmap(t_floatings)
    
        # Sum only valid spreads
        Avalue = jnp.sum(jnp.where(valid_mask, all_spreads, 0.0))
    
        return Avalue

if __name__ == "__main__":
    print("Testing IborCurve with JAX optimizations")
   
    folder = constants.mkt_data_folder # r"C:\Users\edem_\Dropbox\LinearRationalWishart\Data"
    oisRateFile = folder + r"\NewData_ESTR_2024_EUR.csv"
    liborRateFile = folder + r"\NewData_EURIBOR_2024_EUR.csv"
    swaptionFile = folder + r"\allDataEurVolData.csv"
    
    currency = "EUR"
    currentDate = '20240131'
    
    try:
        dailyData6MTo5Y = get_daily_data_up_6m_to_5y(
            # MarketData.GetDailyDataUp6MTo5Y(
            currentDate, currency, oisRateFile, liborRateFile, swaptionFile, 'DummyDataSource')
        print(dailyData6MTo5Y.short_summary())

        # Create OIS curve
        oisCurve = OisCurve(dailyData6MTo5Y.quotation_date, dailyData6MTo5Y.ois_rate_data)

        # Create Ibor curve with JAX optimizations
        iborCurve = IborCurve(dailyData6MTo5Y.quotation_date, dailyData6MTo5Y.euribor_rate_data, oisCurve)
        
        print("A values:", iborCurve.A)
        print("Time points:", iborCurve.T.tolist())
        print("Aggregate A values:", iborCurve.Aaggregate)
        
        # Demonstrate JAX array operations
        print("\n=== JAX Optimization Demo ===")
        test_maturities = jnp.array([1.0, 2.0, 3.0, 5.0])
        
        # Vectorized rate interpolation
        interpolated_rates = iborCurve.RateInterpolate_batch(test_maturities)
        print(f"Interpolated rates: {interpolated_rates}")
        
        # Test forward swap rate calculation
        if len(iborCurve.T) >= 2:
            Tp = float(iborCurve.T[0])
            Tn = float(iborCurve.T[1])
            deltaFixed = 1.0
            deltaFloat = 0.5
            
            forward_rate = iborCurve.forward_swap_rate(Tp, Tn, deltaFixed, deltaFloat, oisCurve)
            print(f"Forward swap rate from {Tp} to {Tn}: {forward_rate}")
        
        # Test A value retrieval
        if len(iborCurve.T) > 0:
            test_maturity = float(iborCurve.T[0])
            mkt_A = iborCurve.getMktA(test_maturity)
            print(f"Market A value at {test_maturity}: {mkt_A}")
        
        print("JAX IborCurve testing completed successfully!")
        
    except ImportError as e:
        print(f"MarketData_jax not available: {e}")
        print("Please ensure MarketData_jax.py is in the correct path")
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
