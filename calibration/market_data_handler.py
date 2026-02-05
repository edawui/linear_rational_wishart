"""
Market data handler for LRW Jump model calibration.

This module processes and manages market data for calibration including
OIS curves, IBOR curves, and swaption data.
"""

from typing import Optional, Dict, List, Tuple, cast
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

from ..data.data_market_data import * #MarketData
from ..curves.iborcurve import IborCurve
from ..pricing.bachelier import bachelier_price, implied_normal_volatility ##implied_volatility
from ..curves.oiscurve import OisCurve
from ..models.interest_rate.config import *


class MarketDataHandler:
    """
    Handles market data processing for LRW Jump calibration.
    """
    
    def __init__(self, daily_data: DailyData):
        """
        Initialize market data handler.
        
        Parameters
        ----------
        daily_data : DailyData
            Daily market data
        """
        self.daily_data = daily_data
        self._ois_curve = None
        self._ibor_curve = None
        self.has_libor_data = not daily_data.euribor_rate_data.empty
        
    def create_ois_curve(self) -> OisCurve:
        """
        Create OIS curve from market data.
        
        Returns
        -------
        OisCurve
            OIS curve instance
        """
        if self._ois_curve is None:
            self._ois_curve = OisCurve(
                self.daily_data.quotation_date,
                self.daily_data.ois_rate_data
            )
        return self._ois_curve
        
    def create_ibor_curve(self, ois_curve: OisCurve) -> IborCurve:
        """
        Create IBOR curve from market data.
        
        Parameters
        ----------
        ois_curve : OisCurve
            OIS curve for discounting
            
        Returns
        -------
        IborCurve
            IBOR curve instance
        """
        if self.has_libor_data is False:
            # raise ValueError("No EURIBOR rate data available to create IBOR curve.")
            self._ibor_curve = None
            return None
        if self._ibor_curve is None:
            self._ibor_curve = IborCurve(
                self.daily_data.quotation_date,
                self.daily_data.euribor_rate_data,
                ois_curve,
                self.daily_data.delta_float_leg,
                self.daily_data.delta_fixed_leg,
                self.daily_data.using_fra
            )
        return self._ibor_curve
        
    def get_max_positive_spread_tenor(self) -> float:
        """
        Find maximum tenor with positive spread.
        
        Returns
        -------
        float
            Maximum tenor with positive aggregate A
        """
        if self._ibor_curve is None:
            # raise ValueError("IBOR curve not initialized")
            # print(" Warning no IBOR data, so OIS max tenor is returned")
            # import warnings
            warnings.warn("No IBOR data, so OIS max tenor is returned", UserWarning)
            max_ois_data = self.daily_data.ois_rate_data["TimeToMat"].max()
            return max_ois_data
            
        # Update market data with aggregate A values
        for index, ibor_data in self.daily_data.euribor_rate_data.iterrows():
            rate_data = cast(RateData, ibor_data["Object"])
            rate_data.market_full_a = self._ibor_curve.getMktFullA(ibor_data["TimeToMat"])
            rate_data.market_aggregate_a = self._ibor_curve.getMktA(ibor_data["TimeToMat"])
            # rate_data.market_full_a = self._ibor_curve.get_mkt_full_a(ibor_data["TimeToMat"])
            # rate_data.market_aggregate_a = self._ibor_curve.get_mkt_a(ibor_data["TimeToMat"])
           
        # Find tenors and aggregate A values
        tenors = [x.time_to_maturity for x in self.daily_data.euribor_rate_data["Object"]]
        aggregate_a_list = np.array([x.market_aggregate_a for x in self.daily_data.euribor_rate_data["Object"]])
        
        # Find negative elements
        neg_elements = np.where(aggregate_a_list < 0)[0]
        
        if len(neg_elements) == 0:
            # All positive, return max tenor + buffer
            max_tenor = tenors[-1] + 0.1
        else:
            # Return tenor just before first negative
            index_min = np.min(neg_elements)
            max_tenor = tenors[index_min]
            
        return max_tenor
        
    def update_swaption_market_data(
        self,
        model,
        market_based_strike: bool = True
    ):
        """
        Update swaption market data with current model.
        
        Parameters
        ----------
        model : LRWModel
            Current model
        market_based_strike : bool, default=True
            Whether to use market-based strikes
        """
        for index, swaption_data in self.daily_data.swaption_data_cube.iterrows():
            swaption = cast(SwaptionData, swaption_data["Object"])
            
            # Calculate annuity
            annuity = 0.0
            model_annuity = 0.0
            fixed_leg_delta = self.daily_data.delta_fixed_leg
            
            for i in range(1, int(swaption.swap_tenor_maturity / fixed_leg_delta) + 1):
                t = swaption.expiry_maturity + i * fixed_leg_delta
                annuity += self._ois_curve.bond_price(t)
                model_annuity += model.bond(t)
                
            annuity *= fixed_leg_delta
            model_annuity *= fixed_leg_delta
            
            swaption.annuity = annuity
            swaption.model_annuity = model_annuity
            
            if market_based_strike:
                # Use market forward swap rate
                # swap_rate = self._ibor_curve.forward_swap_rate(
                if self._ibor_curve is None:
                    
                    swap_rate = self._ois_curve.forward_swap_rate(
                        swaption.expiry_maturity,   
                        swaption.expiry_maturity + swaption.swap_tenor_maturity,
                        # 1.0,##self.daily_data.delta_fixed_leg,    
                        # 1.0,##self.daily_data.delta_float_leg
                        self.daily_data.ois_delta_float_leg ,
                        self.daily_data.ois_delta_fixed_leg
                    )   
                else:
                    swap_rate = self._ibor_curve.forward_swap_rate(
                        swaption.expiry_maturity,
                        swaption.expiry_maturity + swaption.swap_tenor_maturity,
                        self.daily_data.delta_fixed_leg,
                        self.daily_data.delta_float_leg,
                        self._ois_curve
                    )
                 
                swaption.market_forward_swap_rate = swap_rate
                swaption.strike = swap_rate + swaption.strike_offset
                swaption_type='call' ##todo if swaption.
                # Calculate market price
                swaption.market_price = bachelier_price(
                    swap_rate,
                    swaption.strike,
                    swaption.expiry_maturity,
                    swaption.vol,#market_vol,
                    option_type = swaption_type, # call_or_put=1.0,
                    numeraire=swaption.annuity
                )
            else:
                # Use model forward swap rate
                # model.SetOptionProperties(
                #     swaption.swap_tenor_mat,
                #     swaption.expiry_mat,
                #     self.daily_data.delta_float_leg,
                #     self.daily_data.delta_fixed_leg,
                #     swaption.strike
                # )
                model.set_swaption_config(
                            SwaptionConfig(
                            swaption.swap_tenor_maturity,
                            swaption.expiry_maturity,
                            swaption.strike,            
                            self.daily_data.delta_float_leg,
                            self.daily_data.delta_fixed_leg
                                ))
                model_swap_rate = model.compute_swap_rate()
                swaption.model_forward_swap_rate = model_swap_rate
                swaption.strike = model_swap_rate + swaption.strike_offset
                
                swaption_type='call' ##todo if swaption.
                
                # Calculate market price with model annuity
                swaption.market_price = bachelier_price(
                    model_swap_rate,
                    swaption.strike,
                    swaption.expiry_maturity,
                    swaption.vol,
                    option_type = swaption_type, # call_or_put=1.0,
                    numeraire=swaption.model_annuity
                )
                
    def validate_market_data(self) -> Dict[str, bool]:
        """
        Validate market data consistency.
        
        Returns
        -------
        Dict[str, bool]
            Validation results
        """
        results = {}
        
        # Check OIS data
        results['ois_data_present'] = len(self.daily_data.ois_rate_data) > 0
        results['ois_rates_positive'] = all(
            data["Object"].market_zc_rate > 0 
            for _, data in self.daily_data.ois_rate_data.iterrows()
        )
        
        # Check IBOR data
        if self.has_libor_data:
            results['ibor_data_present'] = len(self.daily_data.euribor_rate_data) > 0
        else:
            results['ibor_data_present'] = False
        
        # Check swaption data
        results['swaption_data_present'] = len(self.daily_data.swaption_data_cube) > 0
        results['swaption_vols_positive'] = all(
            data["Object"].vol > 0
            for _, data in self.daily_data.swaption_data_cube.iterrows()
        )
        
        # Check curve consistency
        if self._ois_curve and self._ibor_curve:
            results['curves_consistent'] = self._check_curve_consistency()
            
        return results
        
    def _check_curve_consistency(self) -> bool:
        """Check if OIS and IBOR curves are consistent."""
        # Check that IBOR rates are generally higher than OIS
        test_tenors = [1.0, 2.0, 5.0]
        
        for tenor in test_tenors:
            try:
                ois_rate = self._ois_curve.bond_zc_rate(tenor)
                ibor_forward = self._ibor_curve.forward_rate(0, tenor) if self._ibor_curve else ois_rate
                
                if ibor_forward < ois_rate:## - 0.01:  # Allow small negative spread
                    return False
            except:
                pass
                
        return True
        
    def get_swaption_grid(self) -> pd.DataFrame:
        """
        Get swaption data as a grid.
        
        Returns
        -------
        pd.DataFrame
            Swaption data organized by expiry and tenor
        """
        data = []
        
        for _, swaption_data in self.daily_data.swaption_data_cube.iterrows():
            swaption = swaption_data["Object"]
            data.append({
                'expiry': swaption.expiry_mat,
                'tenor': swaption.swap_tenor_mat,
                'strike_offset': swaption.strike_offset,
                'market_vol': swaption.vol,
                'market_price': swaption.market_price
            })
            
        df = pd.DataFrame(data)
        
        # Pivot to create grid
        vol_grid = df.pivot_table(
            values='market_vol',
            index='expiry',
            columns='tenor'
        )
        
        return vol_grid
        
    def compute_market_spreads(self, model) -> pd.DataFrame:
        """
        Compute market spreads for all tenors.
        
        Parameters
        ----------
        model : LRWModel
            Model for spread calculation
            
        Returns
        -------
        pd.DataFrame
            Spread data
        """
        spreads = []
        if self.has_libor_data is False:
            spreads.append({
                'tenor': None,
                'market_spread': None,
                'model_spread': None,
                'spread_error': None
            })
            return pd.DataFrame(spreads)
        
        for _, ibor_data in self.daily_data.euribor_rate_data.iterrows():
            rate_data = ibor_data["Object"]
            tenor = ibor_data["TimeToMat"]
            
            # Market spread
            market_euribor = rate_data.rate / 100.0
            market_ois = self._ois_curve.bond_zc_rate(tenor)
            market_spread = market_euribor - market_ois
            
            # Model spread
            model_ois = -np.log(model.Bond(tenor)) / tenor
            
            # Set up swap
            model.SetOptionProperties(
                tenor, 0,
                self.daily_data.delta_float_leg,
                self.daily_data.delta_fixed_leg,
                0
            )
            model_euribor = model.ComputeSwapRate()
            model_spread = model_euribor - model_ois
            
            spreads.append({
                'tenor': tenor,
                'market_spread': market_spread,
                'model_spread': model_spread,
                'spread_error': market_spread - model_spread
            })
            
        return pd.DataFrame(spreads)
        
    def export_market_data(self, filename: str):
        """
        Export market data to file.
        
        Parameters
        ----------
        filename : str
            Output filename
        """
        with pd.ExcelWriter(filename) as writer:
            # OIS data
            ois_df = self._create_ois_dataframe()
            ois_df.to_excel(writer, sheet_name='OIS', index=False)
            
            # IBOR data
            ibor_df = self._create_ibor_dataframe()
            ibor_df.to_excel(writer, sheet_name='IBOR', index=False)
            
            # Swaption data
            swaption_grid = self.get_swaption_grid()
            swaption_grid.to_excel(writer, sheet_name='Swaptions')
            
    def _create_ois_dataframe(self) -> pd.DataFrame:
        """Create DataFrame from OIS data."""
        data = []
        for _, ois_data in self.daily_data.ois_rate_data.iterrows():
            rate_data = ois_data["Object"]
            data.append({
                'tenor': ois_data["TimeToMat"],
                'market_rate': rate_data.market_zc_rate,
                'market_price': rate_data.market_zc_price
            })
        return pd.DataFrame(data)
        
    def _create_ibor_dataframe(self) -> pd.DataFrame:
        """Create DataFrame from IBOR data."""
        data = []
        if self.has_libor_data is False:
            return pd.DataFrame(data)   
        for _, ibor_data in self.daily_data.euribor_rate_data.iterrows():
            rate_data = ibor_data["Object"]
            data.append({
                'tenor': ibor_data["TimeToMat"],
                'rate': rate_data.rate / 100.0,
                'market_aggregate_a': rate_data.market_aggregate_a,
                'market_full_a': rate_data.market_full_a
            })
        return pd.DataFrame(data)
