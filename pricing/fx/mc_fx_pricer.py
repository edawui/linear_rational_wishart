

# pricing/fx/mc_fx_pricer.py
"""Monte Carlo based FX option pricing."""


import math
import cmath
import pandas as pd
import numpy as np
from scipy.linalg import cholesky, expm, solve, sqrtm,eigh, schur



from numpy.random import default_rng
import numpy as np
from typing import List, Dict, Any, Optional
from .base import BaseFxPricer
from ..mc_pricer import WishartMonteCarloPricer

class MonteCarloFxPricer(BaseFxPricer):
    """Monte Carlo based FX option pricer."""
    
    def __init__(self, fx_model, nb_mc: int = 1000, dt: float = 1/50.0,
                 schema: str = "EULER_FLOORED"):
        """Initialize Monte Carlo pricer."""
        super().__init__(fx_model)
        self.nb_mc = nb_mc
        self.dt = dt
        self.schema = schema
        
    def price_options(self, maturities: List[float], strikes: List[float],
                     is_calls: List[bool], **kwargs) -> List[float]:
        """Price multiple FX options using Monte Carlo."""
        nb_mc = kwargs.get('nb_mc', self.nb_mc)
        dt = kwargs.get('dt', self.dt)
        schema = kwargs.get('schema', self.schema)
        
        # Get unique maturities for simulation
        unique_maturities = sorted(list(set(maturities)))
        # if unique_maturities[0]!=0.0:
        #     unique_maturities = [0.0] + unique_maturities

        # Simulate Wishart paths
        if self.fx_model.has_jump:
            # Import WishartMC module (would need to be converted to JAX)
            # This is a placeholder - actual implementation would use JAX-based MC
            sim_results = self._simulate_wishart_with_jump(
                unique_maturities, nb_mc, dt, schema
            )
        else:
            sim_results = self._simulate_wishart(
                unique_maturities, nb_mc, dt, schema
            )
        # print(f"nb_mc: {nb_mc}, dt: {dt}, schema: {schema}")
        # print(f"Simulated {len(sim_results)} paths for {len(unique_maturities)} maturities.")
        # Price each option
        prices = []
        std_errors = []
        for maturity, strike, is_call in zip(maturities, strikes, is_calls):
            self.fx_model.set_option_properties(maturity, strike)
            price, std_error = self._price_option_mc(nb_mc, sim_results, maturity, is_call)
            prices.append(price)
            std_errors.append(std_error)
            
        return prices,std_errors
        
    def _simulate_wishart(self, maturities: List[float], nb_mc: int,
                         dt: float, schema: str) -> Dict[int, Dict[float, Any]]:
        """Simulate Wishart paths without jumps."""
        # Placeholder - would implement JAX-based Wishart simulation
        # pass
        mc_pricer = WishartMonteCarloPricer(self.fx_model.lrw_currency_i)
        sim_results = mc_pricer.simulate(maturities, nb_mc=nb_mc
                                         , dt=dt, schema=schema)
        return sim_results


        
       
        
    def _simulate_wishart_with_jump(self, maturities: List[float], nb_mc: int,
                                   dt: float, schema: str) -> Dict[int, Dict[float, Any]]:
        """Simulate Wishart paths with jumps."""
        # Placeholder - would implement JAX-based Wishart simulation with jumps
        pass
    
    def nearest_psd_remove(self, A, tol=1e-12):  
       """  
       Projects a symmetric matrix A onto the cone of positive semi-definite matrices  
       by zeroing out negative eigenvalues.  

       Parameters:  
       - A (ndarray): Input symmetric matrix.  
       - tol (float): Tolerance for eigenvalue clipping. Eigenvalues below this value are set to zero.  

       Returns:  
       - A_psd (ndarray): The nearest positive semi-definite matrix to A.  
       """  
       # Ensure symmetry  
       A = 0.5 * (A + A.T)  

       # Eigen-decomposition  
       eigvals, eigvecs = np.linalg.eigh(A)  

       # Clip eigenvalues at the tolerance level  
       eigvals_clipped = np.clip(eigvals, tol, None)  

       # Reconstruct the matrix  
       A_psd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T  

       # # Check if the matrix was already positive semi-definite  
       # if not np.allclose(A_psd, A):  
       #     print("Matrix is not already positive semi-definite, and was adjusted/changed.")  

       return A_psd
    def simulate_WIS_EulerMaruyama_floor_old_temporary_remove(self, x, alpha, b, a, startTime, timeList, numPaths=10, dt=1.0/360.0, rng=None):
        """
        Simulate WIS_d(x, alpha, b, a; t) using Euler-Maruyama method.
        """
       
        if not np.all(np.diff(timeList) > 0):
            raise ValueError("timeList must be in increasing order.")

        if not np.any(np.isclose(timeList, startTime)):
            insert_index = np.searchsorted(timeList, startTime)
            timeList = np.insert(timeList, insert_index, startTime)

        if rng is None:
            rng = default_rng(42)  # default random number generator with seed 42

        simResults = {}  # key: path index
        for path in range(numPaths):
            simResults[path] = pd.Series(
            {t: np.zeros((x.shape[0], x.shape[1])) for t in timeList}
            )
        index = np.where(np.isclose(timeList, startTime))[0][0]
        for i in range(numPaths):
            # simResults.loc[(i, startTime)] = x.copy()
            simResults[i][startTime] = x.copy()
        
        sigma=a
        omega =  alpha ##*a @ a
        m=b
        min_dt= dt
  
    
        lastTime= startTime
        for i in range(index+1, len(timeList)):
            t = timeList[i] - lastTime                
            for j in range(0, numPaths):
                
                currentT=lastTime
                v= simResults[j][ lastTime].copy()
                while currentT < timeList[i]:
                    if   timeList[i] - currentT  < min_dt:
                        dt_ = timeList[i] - currentT
                    else:
                        dt_ = min_dt

                    currentT += dt_
                    sqrtdt=  math.sqrt(dt_)

                    dwt = rng.normal(0.0, sqrtdt, x.shape)
                
                    v1 = sqrtm(v) @ dwt @ sigma  
              
                    v1 =  v1 + v1.T
                    v = v  + dt*(omega +  m @ v + v @ m.T)
                    v = v + v1 
                
                    v=  self.nearest_psd(v)

                simResults[j][timeList[i]] = v    
                
            lastTime= timeList[i] 


        return simResults


    def _price_option_mc(self, nb_mc: int, sim_results: Dict[int, Dict[float, Any]],
                        maturity: float, is_call: bool) :#-> float:
        """Price single option from Monte Carlo paths."""
        price = 0.0
        payoffs=[]
        
        # print(self.fx_model.x0)
        
        for nmc in range(nb_mc):
            v = sim_results[nmc][maturity]

            # zeta_i_T = math.exp(-self.fx_model.lrw_currency_i.alpha * maturity)*(1.0+np.trace(self.fx_model.lrw_currency_i.u1 @ v))
            # zeta_j_T = math.exp(-self.fx_model.lrw_currency_j.alpha * maturity)*(1.0+np.trace(self.fx_model.lrw_currency_j.u1 @ v))
            
            # zeta_i_t = 1.0+np.trace(self.fx_model.lrw_currency_i.u1 @ self.fx_model.lrw_currency_i.x0)
            # zeta_j_t = 1.0+np.trace(self.fx_model.lrw_currency_j.u1 @ self.fx_model.lrw_currency_j.x0)
            
            zeta_i_T = math.exp(-self.fx_model.alpha_i * maturity)*(1.0+np.trace(self.fx_model.u_i @ v))
            zeta_j_T = math.exp(-self.fx_model.alpha_j * maturity)*(1.0+np.trace(self.fx_model.u_j @ v))
           
            zeta_i_t = 1.0+np.trace(self.fx_model.u_i  @ self.fx_model.x0)
            zeta_j_t = 1.0+np.trace(self.fx_model.u_j  @ self.fx_model.x0)

            S_ij_T= (zeta_j_T/zeta_j_t)*(zeta_i_t/zeta_i_T)*self.fx_model.fx_spot



         
            # y_ij_t =S_ij_T 
            y_ij_t = (zeta_i_T/zeta_i_t) *( S_ij_T - self.fx_model.strike)
            payoffs.append(max(y_ij_t, 0.0))



        # for nmc in range(nb_mc):
        #     v = sim_results[nmc][maturity]
        #     # print(f"path {nmc}")
        #     # print(v)
        #     # Compute payoff
        #     y_ij_t = self.fx_model.bij_2 + np.trace(self.fx_model.aij_2 @ v)
        #     payoffs.append(max(y_ij_t, 0.0))
        #     # if y_ij_t > 0:
        #     #     payoff = y_ij_t
        #     # else:
        #     #     payoff = 0.0
                
        #     # price += payoff
            
        # price = price / nb_mc
        price = np.mean(payoffs)
        std_error = np.std(payoffs) / np.sqrt(nb_mc)

        # Apply pseudo-inverse smoothing if enabled
        if (hasattr(self.fx_model.lrw_currency_i, 'pseudo_inverse_smoothing') and
            self.fx_model.lrw_currency_i.pseudo_inverse_smoothing):
            curve_alpha_adjustment = self.fx_model.lrw_currency_i.initial_curve_alpha.get_alpha(maturity)
            price *= curve_alpha_adjustment
            
        # Put-call parity for put options
        if not is_call:
            domestic_df = self.fx_model.lrw_currency_i.bond(maturity)
            foreign_df = self.fx_model.lrw_currency_j.bond(maturity)
            # price = price + domestic_df * self.fx_model.strike - foreign_df * self.fx_model.fx_spot
            put_adjustment =   domestic_df * self.fx_model.strike - foreign_df * self.fx_model.fx_spot
            
            call_price=price
            
            price = call_price + domestic_df * self.fx_model.strike - foreign_df * self.fx_model.fx_spot
            # print(f"MC: Call Price: {call_price}, Put adjustment: {put_adjustment}, put price ={price}")
            
        return float(price), std_error
        
    def get_pricing_method(self) -> str:
        """Get pricing method name."""
        return "MC"
