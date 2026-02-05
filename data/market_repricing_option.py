"""
Option repricing functionality for calibration.

This module provides utilities for repricing FX options using calibrated models
and calculating pricing errors.
"""

from typing import Any, Dict, List, Tuple, Union, Optional
import numpy as np
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..data.data_fx_market_data import FxVolData
from ..curves.oiscurve import OisCurve
from ..models.fx.lrw_fx import LRWFxModel
from ..pricing import black_scholes as bs
from ..pricing.fx.fourier_fx_pricer import  FourierFxPricer
from ..pricing.fx.mc_fx_pricer import  MonteCarloFxPricer
from ..pricing.implied_vol_black_scholes import implied_volatility_black_scholes
# from ..data import MarketData


class OptionRepricer:
    """Handles option repricing for calibration."""
    
    def __init__(
        self,
        domestic_curve: OisCurve,
        foreign_curve: OisCurve,
        model: LRWFxModel,
        pricing_method: str = "MC",
        mc_paths: int = 2000,
        mc_timestep: float = 1.0/25,
        # # schema: str =  "EULER_FLOORED",
        schema: str = "EULER_CORRECTED",#"EULER_FLOORED",
        use_richardson: float = 0.5,
        n_max: int = 25
    ):
        """
        Initialize option repricer.
        
        Parameters
        ----------
        domestic_curve : OisCurve
            Domestic OIS curve
        foreign_curve : OisCurve
            Foreign OIS curve
        model : LRWFxModel
            FX model for pricing
        pricing_method : str
            Pricing method ('MC', 'PDE', etc.)
        mc_paths : int
            Number of Monte Carlo paths
        mc_timestep : float
            Monte Carlo time step
        schema : str
            Numerical schema for simulation
        use_richardson : float
            Richardson extrapolation parameter
        n_max : int
            Maximum number of time steps
        """
        self.domestic_curve = domestic_curve
        self.foreign_curve = foreign_curve
        self.model = model
        self.pricing_method = pricing_method
        self.mc_paths = mc_paths
        self.mc_timestep = mc_timestep
        self.schema = schema
        self.use_richardson = use_richardson
        self.n_max = n_max
        self.fourier_pricer = FourierFxPricer(self.model)
        self.mc_pricer = MonteCarloFxPricer(self.model,nb_mc= self.mc_paths
                                           ,dt= self.mc_timestep
                                            ,schema=self.schema)


    def reprice_single_option(
        self,
        option_data: FxVolData
    ) -> Tuple[float, float]:
        """
        Reprice a single FX option.
        
        Parameters
        ----------
        option_data : FxVolData
            Option data object
            
        Returns
        -------
        Tuple[float, float]
            Model price and model implied volatility
        """
        # Set option properties in model
        self.model.set_option_properties(option_data.expiry_maturity, option_data.strike)
        
        # Price option
        start_time = time.perf_counter()
        print(self.pricing_method)
        if self.pricing_method == "MC":
            # model_price = self.model.price_fx_option(
            #     pricing_technic=self.pricing_method,
            #     nb_mc=self.mc_paths,
            #     dt=self.mc_timestep,
            #     schema=self.schema,
            #     ur=self.use_richardson,
            #     nmax=self.n_max
            # )
            maturities = [option_data.expiry_maturity]
            strikes = [option_data.strike]
            call_puts = [option_data.call_or_put]
            prices = self.mc_pricer.price_options(maturities=maturities,
                                             strikes=strikes,
                                             is_calls=call_puts,
                                             nb_mc=self.mc_paths,
                                             dt=self.mc_timestep,
                                             schema=self.schema)
            model_price = prices[0] if prices else np.nan
        else:
            model_price = self.model.price_fx_option()## TODO pricing_technic=self.pricing_method)
        
        pricing_time = time.perf_counter() - start_time
        
        # Calculate implied volatility
        start_time = time.perf_counter()
        
        r_d = self.domestic_curve.bond_zc_rate(option_data.expiry_maturity)
        r_f = self.foreign_curve.bond_zc_rate(option_data.expiry_maturity)
        
        # model_vol = bs.implied_volatility_fx(
        #     option_data.fx_spot,
        #     r_d,
        #     r_f,
        #     option_data.strike,
        #     option_data.expiry_maturity,
        #     model_price,
        #     option_data.call_or_put,
        #     initial_guess=option_data.vol
        # )
        
        model_vol = implied_volatility_black_scholes(model_price,
                option_data.fx_spot, r_d, r_f,option_data.strike
                ,  option_data.expiry_maturity, True
                )
        vol_time = time.perf_counter() - start_time
        
        # Update option data
        option_data.model_price = model_price
        option_data.model_vol = model_vol
        
        # Debug output
        print(f"Option {option_data.expiry_maturity:.2f}Y K={option_data.strike:.4f}: "
              f"Price={model_price:.6f} ({pricing_time:.3f}s), "
              f"Vol={model_vol:.4f} ({vol_time:.3f}s)")
        
        return model_price, model_vol
    
    def reprice_options(
        self,
        option_list: List[FxVolData]
    ) -> Dict[str, np.ndarray]:
        """
        Reprice a list of options sequentially.
        
        Parameters
        ----------
        option_list : List[FxVolData]
            List of options to reprice
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary with model prices and volatilities
        """
        # Sort options by maturity and strike
        sorted_options = sorted(option_list, key=lambda x: (x.expiry_maturity, x.strike))
        
        # Prepare for batch pricing if possible
        maturities = []
        strikes = []
        call_puts = []
        
        for option in sorted_options:
            maturities.append(option.expiry_maturity)
            strikes.append(option.strike)
            call_puts.append(option.call_or_put)
        
        use_mkt_curve=False#True
        # Try batch pricing
        if self.pricing_method == "MC":
            if hasattr(self.mc_pricer, 'price_options'):
                # print(f"Batch pricing {len(sorted_options)} options...")
            

                prices = self.mc_pricer.price_options(maturities=maturities,
                                             strikes=strikes,
                                             is_calls=call_puts,
                                             nb_mc=self.mc_paths,
                                             dt=self.mc_timestep,
                                             schema=self.schema)
        else: #self.pricing_method == "FOURIER":
            
            prices = self.fourier_pricer.price_options(maturities,
                                                strikes,
                                                call_puts,)

        # print(f"Fourier batch pricing {prices} ...")
            # Update option data and calculate implied vols
        model_vols = []
        for i, (option, price) in enumerate(zip(sorted_options, prices)):
            option.model_price = price
            
            if use_mkt_curve:
                r_d = self.domestic_curve.bond_zc_rate(option.expiry_maturity)
                r_f = self.foreign_curve.bond_zc_rate(option.expiry_maturity)
            else:
                zc_d=self.fourier_pricer.fx_model.lrw_currency_i.bond(option.expiry_maturity)
                zc_f=self.fourier_pricer.fx_model.lrw_currency_j.bond(option.expiry_maturity)
                
                r_d = self.domestic_curve.get_rate_from_zc(zc_d, option.expiry_maturity)
                r_f = self.foreign_curve.get_rate_from_zc(zc_f, option.expiry_maturity)

            call_put = option.call_or_put

            vol = implied_volatility_black_scholes(price,
                option.fx_spot,option.strike
                , option.expiry_maturity,
                 r_d, r_f,call_put
             )
             

            
            
            # option.model_price = price
            option.model_vol = vol
            model_vols.append(vol)
        
        model_prices = np.array(prices)
        model_vols = np.array(model_vols)
            
        # else:
        #     # Sequential pricing
        #     print(f"Sequential pricing {len(sorted_options)} options...")
        #     model_prices = []
        #     model_vols = []
            
        #     for option in sorted_options:
        #         price, vol = self.reprice_single_option(option)
        #         model_prices.append(price)
        #         model_vols.append(vol)

        #         option.model_price = price
        #         option.model_vol = vol

        #     model_prices = np.array(model_prices)
        #     model_vols = np.array(model_vols)
        
        return {
            'model_prices': model_prices,
            'model_vols': model_vols,
            'maturities': np.array(maturities),
            'strikes': np.array(strikes)
        }
    
    def reprice_options_multithread(
        self,
        option_list: List[FxVolData],
        max_workers: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Reprice options using multiple threads.
        
        Parameters
        ----------
        option_list : List[FxVolData]
            List of options to reprice
        max_workers : int, optional
            Maximum number of worker threads
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary with model prices and volatilities
        """
        if max_workers is None:
            max_workers = min(len(option_list), 4)
        
        print(f"Multi-threaded pricing {len(option_list)} options with {max_workers} workers...")
        
        # Create a copy of the model for each thread
        def price_option_thread(option_data):
            # Each thread needs its own model instance
            thread_model = self.model.copy()
            thread_repricer = OptionRepricer(
                self.domestic_curve,
                self.foreign_curve,
                thread_model,
                self.pricing_method,
                self.mc_paths,
                self.mc_timestep,
                self.schema,
                self.use_richardson,
                self.n_max
            )
            return thread_repricer.reprice_single_option(option_data)
        
        # Execute in parallel
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_option = {
                executor.submit(price_option_thread, option): option
                for option in option_list
            }
            
            # Collect results
            for future in as_completed(future_to_option):
                option = future_to_option[future]
                try:
                    price, vol = future.result()
                    results.append((option, price, vol))
                except Exception as e:
                    print(f"Error pricing option {option}: {e}")
                    results.append((option, np.nan, np.nan))
        
        # Sort results by original order
        results.sort(key=lambda x: option_list.index(x[0]))
        
        # Extract arrays
        model_prices = np.array([r[1] for r in results])
        model_vols = np.array([r[2] for r in results])
        maturities = np.array([r[0].expiry_maturity for r in results])
        strikes = np.array([r[0].strike for r in results])
        
        # Update option objects
        for option, price, vol in results:
            option.model_price = price
            option.model_vol = vol
        
        return {
            'model_prices': model_prices,
            'model_vols': model_vols,
            'maturities': maturities,
            'strikes': strikes
        }
    
    def calculate_rmse(
        self,
        option_list: List[FxVolData],
        use_vol: bool = True
    ) -> float:
        """
        Calculate RMSE for option repricing.
        
        Parameters
        ----------
        option_list : List[FxVolData]
            List of options
        use_vol : bool
            If True, calculate RMSE on vols; else on prices
            
        Returns
        -------
        float
            RMSE value
        """
        errors = []
        
        for option in option_list:
            if use_vol:
                error = option.market_vol - option.model_vol
            else:
                error = option.market_price - option.model_price
            
            if not np.isnan(error):
                errors.append(error)
        
        if not errors:
            return 0.0
        
        errors_array = np.array(errors)
        rmse = np.sqrt(np.mean(errors_array**2))
        
        # Scale by 10000 for basis points if using vol
        if use_vol:
            return 10000 * rmse
        else:
            return rmse
    
    def get_pricing_summary(
        self,
        option_list: List[FxVolData]
    ) -> str:
        """
        Generate summary of option pricing errors.
        
        Parameters
        ----------
        option_list : List[FxVolData]
            List of options
            
        Returns
        -------
        str
            Formatted summary string
        """
        price_rmse = self.calculate_rmse(option_list, use_vol=False)
        vol_rmse = self.calculate_rmse(option_list, use_vol=True)
        
        n_options = len(option_list)
        n_calls = sum(1 for opt in option_list if opt.call_or_put)
        n_puts = n_options - n_calls
        
        summary = [
            "Option Repricing Summary",
            "=" * 50,
            f"Total options: {n_options} ({n_calls} calls, {n_puts} puts)",
            f"Price RMSE: {price_rmse:.6f}",
            f"Volatility RMSE: {vol_rmse:.4f} bps",
            "=" * 50
        ]
        
        return "\n".join(summary)


class OptionErrorAnalyzer:
    """Analyzes option pricing errors for diagnostics."""
    
    def __init__(self, option_repricer: OptionRepricer):
        """
        Initialize error analyzer.
        
        Parameters
        ----------
        option_repricer : OptionRepricer
            Option repricer instance
        """
        self.option_repricer = option_repricer
    
    def analyze_errors_by_maturity(
        self,
        option_list: List[FxVolData],
        maturity_buckets: List[float] = None
    ) -> pd.DataFrame:
        """
        Analyze pricing errors by maturity buckets.
        
        Parameters
        ----------
        option_list : List[FxVolData]
            List of options
        maturity_buckets : List[float], optional
            Maturity bucket boundaries
            
        Returns
        -------
        pd.DataFrame
            Error statistics by maturity bucket
        """
        if maturity_buckets is None:
            maturity_buckets = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
        
        results = []
        
        for i in range(len(maturity_buckets) - 1):
            min_mat = maturity_buckets[i]
            max_mat = maturity_buckets[i + 1]
            
            bucket_options = [
                opt for opt in option_list
                if min_mat <= opt.expiry_maturity < max_mat
            ]
            
            if bucket_options:
                price_rmse = self.option_repricer.calculate_rmse(bucket_options, use_vol=False)
                vol_rmse = self.option_repricer.calculate_rmse(bucket_options, use_vol=True)
                
                results.append({
                    'maturity_bucket': f"{min_mat:.2f}Y-{max_mat:.2f}Y",
                    'n_options': len(bucket_options),
                    'price_rmse': price_rmse,
                    'vol_rmse_bps': vol_rmse
                })
        
        return pd.DataFrame(results)
    
    def analyze_errors_by_moneyness(
        self,
        option_list: List[FxVolData],
        moneyness_buckets: List[float] = None
    ) -> pd.DataFrame:
        """
        Analyze pricing errors by moneyness buckets.
        
        Parameters
        ----------
        option_list : List[FxVolData]
            List of options
        moneyness_buckets : List[float], optional
            Moneyness bucket boundaries (K/S ratios)
            
        Returns
        -------
        pd.DataFrame
            Error statistics by moneyness bucket
        """
        if moneyness_buckets is None:
            moneyness_buckets = [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]
        
        results = []
        
        for i in range(len(moneyness_buckets) - 1):
            min_money = moneyness_buckets[i]
            max_money = moneyness_buckets[i + 1]
            
            bucket_options = [
                opt for opt in option_list
                if min_money <= opt.strike / opt.fx_spot < max_money
            ]
            
            if bucket_options:
                price_rmse = self.option_repricer.calculate_rmse(bucket_options, use_vol=False)
                vol_rmse = self.option_repricer.calculate_rmse(bucket_options, use_vol=True)
                
                avg_moneyness = np.mean([opt.strike / opt.fx_spot for opt in bucket_options])
                
                results.append({
                    'moneyness_bucket': f"{min_money:.2f}-{max_money:.2f}",
                    'avg_moneyness': avg_moneyness,
                    'n_options': len(bucket_options),
                    'price_rmse': price_rmse,
                    'vol_rmse_bps': vol_rmse
                })
        
        return pd.DataFrame(results)
    
    def identify_outliers(
        self,
        option_list: List[FxVolData],
        n_std: float = 3.0
    ) -> pd.DataFrame:
        """
        Identify options with large pricing errors.
        
        Parameters
        ----------
        option_list : List[FxVolData]
            List of options
        n_std : float
            Number of standard deviations for outlier threshold
            
        Returns
        -------
        pd.DataFrame
            DataFrame of outlier options
        """
        # Collect errors
        price_errors = []
        vol_errors = []
        
        for option in option_list:
            price_error = option.market_price - option.model_price
            vol_error = option.market_vol - option.model_vol
            
            if not np.isnan(price_error) and not np.isnan(vol_error):
                price_errors.append(price_error)
                vol_errors.append(vol_error)
        
        if not price_errors:
            return pd.DataFrame()
        
        # Calculate thresholds
        price_errors = np.array(price_errors)
        vol_errors = np.array(vol_errors)
        
        price_mean = np.mean(price_errors)
        price_std = np.std(price_errors)
        vol_mean = np.mean(vol_errors)
        vol_std = np.std(vol_errors)
        
        price_threshold = n_std * price_std
        vol_threshold = n_std * vol_std
        
        # Find outliers
        outliers = []
        
        for option in option_list:
            price_error = option.market_price - option.model_price
            vol_error = option.market_vol - option.model_vol
            
            if (np.isnan(price_error) or np.isnan(vol_error) or
                abs(price_error - price_mean) > price_threshold or
                abs(vol_error - vol_mean) > vol_threshold):
                
                outliers.append({
                    'maturity': option.expiry_maturity,
                    'strike': option.strike,
                    'moneyness': option.strike / option.fx_spot,
                    'call_put': 'Call' if option.call_or_put else 'Put',
                    'price_error': price_error,
                    'vol_error': vol_error * 10000,  # in bps
                    'market_price': option.market_price,
                    'model_price': option.model_price,
                    'market_vol': option.market_vol,
                    'model_vol': option.model_vol
                })
        
        return pd.DataFrame(outliers)
