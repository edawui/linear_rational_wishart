"""
Bond repricing functionality for calibration.

This module provides utilities for repricing bonds using calibrated models
and calculating pricing errors.
"""

from typing import Any, Dict, List, Tuple, Union
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import copy


# from ..curves import OisCurve, IborCurve
from ..curves.oiscurve import OisCurve
from ..models.interest_rate.lrw_model import LRWModel

# from ..data.data_fx_market_data import FxVolData
# from ..models.fx.lrw_fx import LRWFxModel
# from ..pricing import black_scholes as bs
# from ..data import MarketData

class BondRepricer:
    """Handles bond repricing for calibration."""
    
    def __init__(
        self,
        domestic_curve: OisCurve,
        foreign_curve: OisCurve,
        domestic_model: LRWModel,
        foreign_model: LRWModel,

    ):
        """
        Initialize bond repricer.
        
        Parameters
        ----------
        domestic_curve : OisCurve
            Domestic OIS curve
        foreign_curve : OisCurve
            Foreign OIS curve
        model : LrwFx
            FX model for pricing
        """
        self.domestic_curve = domestic_curve
        self.foreign_curve = foreign_curve
        self.domestic_model = domestic_model
        self.foreign_model = foreign_model
    
    def reprice_single_bond(
        self,
        rate_data: Any,
        time_to_maturity: float,
        is_domestic: bool = True
    ) -> Tuple[float, float]:
        """
        Reprice a single bond.
        
        Parameters
        ----------
        rate_data : RateData
            Rate data object to update
        time_to_maturity : float
            Time to maturity in years
        is_domestic : bool
            True for domestic currency, False for foreign
            
        Returns
        -------
        Tuple[float, float]
            Model price and model rate
        """
        # Get market values from curve
        curve = self.domestic_curve if is_domestic else self.foreign_curve
        model = self.domestic_model if is_domestic else self.foreign_model

        market_price = curve.bond_price(time_to_maturity)
        market_rate = curve.bond_zc_rate(time_to_maturity)
        
        model_price = model.bond(time_to_maturity)
        # # Calculate model values
        # if is_domestic:
        #     model_price = self.model.lrw_currency_i.bond(time_to_maturity)
        # else:
        #     model_price = self.model.lrw_currency_j.bond(time_to_maturity)
        
        model_rate = curve.get_rate_from_zc(model_price, time_to_maturity)
        
        # Update rate data object
        rate_data.market_zc_price = market_price
        rate_data.market_zc_rate = market_rate
        rate_data.model_zc_price = model_price
        rate_data.model_zc_rate = model_rate
        
        return model_price, model_rate
    
    def reprice_bond_list(
        self,
        rate_data_df: pd.DataFrame,
        is_domestic: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Reprice a list of bonds.
        
        Parameters
        ----------
        rate_data_df : pd.DataFrame
            DataFrame with rate data
        is_domestic : bool
            True for domestic currency, False for foreign
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary with market and model prices/rates
        """
        results = {
            'market_prices': [],
            'market_rates': [],
            'model_prices': [],
            'model_rates': [],
            'price_errors': [],
            'rate_errors': []
        }
    
         
        yield_curve_maturities = rate_data_df["TimeToMat"].tolist()
        model = self.domestic_model if is_domestic else self.foreign_model
        curve = self.domestic_curve if is_domestic else self.foreign_curve
        
        model_prices = model.bond_vectorized(yield_curve_maturities)
   
        # Vectorized rate calculations
        model_rates = np.array([curve.get_rate_from_zc(price, mat) 
                           for price, mat in zip(model_prices, yield_curve_maturities)])
    
        # # Vectorized market calculations (if curve supports it)
        # if hasattr(curve, 'bond_price_vectorized'):
        #     market_prices = curve.bond_price_vectorized(yield_curve_maturities)
        #     market_rates = curve.bond_zc_rate_vectorized(yield_curve_maturities)
        # else:
            # Fallback to individual calls
        market_prices = np.array([curve.bond_price(t) for t in yield_curve_maturities])
        market_rates = np.array([curve.bond_zc_rate(t) for t in yield_curve_maturities])
    
        # # for index, row in rate_data_df.iterrows():
        # for array_idx, (df_index, row) in enumerate(rate_data_df.iterrows()):
        #     rate_data = row["Object"]
        #     time_to_mat = row["TimeToMat"]
            
        #     # model_price, model_rate = self.reprice_single_bond(
        #     #     rate_data,
        #     #     time_to_mat,
        #     #     is_domestic
        #     # )
        #     model_price = bond_prices[array_idx]

        for i, (index, row) in enumerate(rate_data_df.iterrows()):
            rate_data = row["Object"]
            rate_data.model_zc_price = model_prices[i]
            rate_data.model_zc_rate = model_rates[i]
            rate_data.market_zc_price = market_prices[i]
            rate_data.market_zc_rate = market_rates[i]
     
           
               
            results['market_prices'].append(rate_data.market_zc_price)
            results['market_rates'].append(rate_data.market_zc_rate)
            results['model_prices'].append(rate_data.model_zc_price)
            results['model_rates'].append(rate_data.model_zc_rate)
            results['price_errors'].append(rate_data.market_zc_price - rate_data.model_zc_price)
            results['rate_errors'].append(rate_data.market_zc_rate - rate_data.model_zc_rate)
        


        return {k: np.array(v) for k, v in results.items()}
    
    def reprice_bond_list_not_tested(
            self,
            rate_data_df: pd.DataFrame,
            is_domestic: bool = True,
            max_workers: int = None
            ) -> Dict[str, np.ndarray]:
            """
            Reprice a list of bonds using multi-threading (simpler version).
            """
            def process_row(row_data):
                """Process a single row."""
                rate_data, time_to_mat = row_data
                model_price, model_rate = self.reprice_single_bond(
                    rate_data, time_to_mat, is_domestic
                )
                return {
                    'market_price': rate_data.market_zc_price,
                    'market_rate': rate_data.market_zc_rate,
                    'model_price': model_price,
                    'model_rate': model_rate,
                    'price_error': rate_data.market_zc_price - model_price,
                    'rate_error': rate_data.market_zc_rate - model_rate
                }
            
            max_workers=2 if max_workers is None else max_workers
    
            # Prepare row data
            row_data = [(row["Object"], row["TimeToMat"]) 
                        for _, row in rate_data_df.iterrows()]
    
            # Process in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results_list = list(executor.map(process_row, row_data))
    
            # Reorganize results
            results = {
                'market_prices': [r['market_price'] for r in results_list],
                'market_rates': [r['market_rate'] for r in results_list],
                'model_prices': [r['model_price'] for r in results_list],
                'model_rates': [r['model_rate'] for r in results_list],
                'price_errors': [r['price_error'] for r in results_list],
                'rate_errors': [r['rate_error'] for r in results_list]
            }
    
            return {k: np.array(v) for k, v in results.items()}

    def reprice_all(
        self,
        domestic_rate_data: pd.DataFrame,
        foreign_rate_data: pd.DataFrame
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Reprice all bonds for both currencies.
        
        Parameters
        ----------
        domestic_rate_data : pd.DataFrame
            Domestic currency rate data
        foreign_rate_data : pd.DataFrame
            Foreign currency rate data
            
        Returns
        -------
        Dict[str, Dict[str, np.ndarray]]
            Nested dictionary with results for each currency
        """
        domestic_results = self.reprice_bond_list(domestic_rate_data, is_domestic=True)
        foreign_results = self.reprice_bond_list(foreign_rate_data, is_domestic=False)
        
        return {
            'domestic': domestic_results,
            'foreign': foreign_results
        }
    
    def calculate_rmse(
        self,
        domestic_rate_data: pd.DataFrame,
        foreign_rate_data: pd.DataFrame,
        min_tenor: float = 0.0,
        max_tenor: float = float('inf'),
        use_price: bool = True
    ) -> float:
        """
        Calculate RMSE for bond repricing.
        
        Parameters
        ----------
        domestic_rate_data : pd.DataFrame
            Domestic currency rate data
        foreign_rate_data : pd.DataFrame
            Foreign currency rate data
        min_tenor : float
            Minimum tenor to include
        max_tenor : float
            Maximum tenor to include
        use_price : bool
            If True, calculate RMSE on prices; else on rates
            
        Returns
        -------
        float
            RMSE value
        """
        errors = []
        
        # Domestic bonds
        for _, row in domestic_rate_data.iterrows():
            time_to_mat = row["TimeToMat"]
            if min_tenor <= time_to_mat <= max_tenor:
                rate_data = row["Object"]
                if use_price:
                    error = rate_data.market_zc_price - rate_data.model_zc_price
                else:
                    error = rate_data.market_zc_rate - rate_data.model_zc_rate
                errors.append(error)
        
        # Foreign bonds
        for _, row in foreign_rate_data.iterrows():
            time_to_mat = row["TimeToMat"]
            if min_tenor <= time_to_mat <= max_tenor:
                rate_data = row["Object"]
                if use_price:
                    error = rate_data.market_zc_price - rate_data.model_zc_price
                else:
                    error = rate_data.market_zc_rate - rate_data.model_zc_rate
                errors.append(error)
        
        if not errors:
            return 0.0
        
        errors_array = np.array(errors)
        rmse = np.sqrt(np.mean(errors_array**2))
        
        # Scale by 10000 for basis points
        return 10000 * rmse
    
    def get_pricing_summary(
        self,
        domestic_rate_data: pd.DataFrame,
        foreign_rate_data: pd.DataFrame
    ) -> str:
        """
        Generate summary of bond pricing errors.
        
        Parameters
        ----------
        domestic_rate_data : pd.DataFrame
            Domestic currency rate data
        foreign_rate_data : pd.DataFrame
            Foreign currency rate data
            
        Returns
        -------
        str
            Formatted summary string
        """
        # Calculate RMSEs
        price_rmse = self.calculate_rmse(
            domestic_rate_data,
            foreign_rate_data,
            use_price=True
        )
        rate_rmse = self.calculate_rmse(
            domestic_rate_data,
            foreign_rate_data,
            use_price=False
        )
        
        # Count bonds
        n_domestic = len(domestic_rate_data)
        n_foreign = len(foreign_rate_data)
        n_total = n_domestic + n_foreign
        
        summary = [
            "Bond Repricing Summary",
            "=" * 50,
            f"Total bonds: {n_total} ({n_domestic} domestic, {n_foreign} foreign)",
            f"Price RMSE: {price_rmse:.4f} bps",
            f"Yield RMSE: {rate_rmse:.4f} bps",
            "=" * 50
        ]
        
        return "\n".join(summary)


class BondErrorAnalyzer:
    """Analyzes bond pricing errors for diagnostics."""
    
    def __init__(self, bond_repricer: BondRepricer):
        """
        Initialize error analyzer.
        
        Parameters
        ----------
        bond_repricer : BondRepricer
            Bond repricer instance
        """
        self.bond_repricer = bond_repricer
    
    def analyze_errors_by_tenor(
        self,
        domestic_rate_data: pd.DataFrame,
        foreign_rate_data: pd.DataFrame,
        tenor_buckets: List[float] = None
    ) -> pd.DataFrame:
        """
        Analyze pricing errors by tenor buckets.
        
        Parameters
        ----------
        domestic_rate_data : pd.DataFrame
            Domestic currency rate data
        foreign_rate_data : pd.DataFrame
            Foreign currency rate data
        tenor_buckets : List[float], optional
            Tenor bucket boundaries
            
        Returns
        -------
        pd.DataFrame
            Error statistics by tenor bucket
        """
        if tenor_buckets is None:
            tenor_buckets = [0.0, 1.0, 2.0, 5.0, 10.0, 30.0]
        
        results = []
        
        for i in range(len(tenor_buckets) - 1):
            min_tenor = tenor_buckets[i]
            max_tenor = tenor_buckets[i + 1]
            
            price_rmse = self.bond_repricer.calculate_rmse(
                domestic_rate_data,
                foreign_rate_data,
                min_tenor=min_tenor,
                max_tenor=max_tenor,
                use_price=True
            )
            
            rate_rmse = self.bond_repricer.calculate_rmse(
                domestic_rate_data,
                foreign_rate_data,
                min_tenor=min_tenor,
                max_tenor=max_tenor,
                use_price=False
            )
            
            # Count bonds in bucket
            n_bonds = 0
            for _, row in pd.concat([domestic_rate_data, foreign_rate_data]).iterrows():
                if min_tenor <= row["TimeToMat"] < max_tenor:
                    n_bonds += 1
            
            results.append({
                'tenor_bucket': f"{min_tenor:.0f}Y-{max_tenor:.0f}Y",
                'n_bonds': n_bonds,
                'price_rmse_bps': price_rmse,
                'rate_rmse_bps': rate_rmse
            })
        
        return pd.DataFrame(results)
    
    def identify_outliers(
        self,
        domestic_rate_data: pd.DataFrame,
        foreign_rate_data: pd.DataFrame,
        n_std: float = 3.0
    ) -> pd.DataFrame:
        """
        Identify bonds with large pricing errors.
        
        Parameters
        ----------
        domestic_rate_data : pd.DataFrame
            Domestic currency rate data
        foreign_rate_data : pd.DataFrame
            Foreign currency rate data
        n_std : float
            Number of standard deviations for outlier threshold
            
        Returns
        -------
        pd.DataFrame
            DataFrame of outlier bonds
        """
        outliers = []
        
        # Collect all errors
        all_price_errors = []
        all_rate_errors = []
        
        for currency, rate_data in [('domestic', domestic_rate_data), ('foreign', foreign_rate_data)]:
            for _, row in rate_data.iterrows():
                rate_obj = row["Object"]
                price_error = rate_obj.market_zc_price - rate_obj.model_zc_price
                rate_error = rate_obj.market_zc_rate - rate_obj.model_zc_rate
                
                all_price_errors.append(price_error)
                all_rate_errors.append(rate_error)
        
        # Calculate thresholds
        price_errors = np.array(all_price_errors)
        rate_errors = np.array(all_rate_errors)
        
        price_mean = np.mean(price_errors)
        price_std = np.std(price_errors)
        rate_mean = np.mean(rate_errors)
        rate_std = np.std(rate_errors)
        
        price_threshold = n_std * price_std
        rate_threshold = n_std * rate_std
        
        # Find outliers
        idx = 0
        for currency, rate_data in [('domestic', domestic_rate_data), ('foreign', foreign_rate_data)]:
            for _, row in rate_data.iterrows():
                rate_obj = row["Object"]
                price_error = all_price_errors[idx]
                rate_error = all_rate_errors[idx]
                
                if (abs(price_error - price_mean) > price_threshold or
                    abs(rate_error - rate_mean) > rate_threshold):
                    
                    outliers.append({
                        'currency': currency,
                        'tenor': row["TimeToMat"],
                        'price_error': price_error * 10000,  # in bps
                        'rate_error': rate_error * 10000,  # in bps
                        'market_price': rate_obj.market_zc_price,
                        'model_price': rate_obj.model_zc_price
                    })
                
                idx += 1
        
        return pd.DataFrame(outliers)
