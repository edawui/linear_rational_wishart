"""
Market data structures and utilities.

This module provides classes and functions for handling various types of
market data including interest rates, FX rates, and option volatilities.
"""

# from collections import _odict_items
from datetime import datetime, date
from typing import Optional, Union, List, Tuple, Dict, Any
from enum import Enum
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import math
import os

# from ..utils.date_utils import convert_tenor_to_years
from ..utils.jax_utils import is_jax_available, ensure_jax_array


class MarketDataType(Enum):
    """Types of market data."""
    RATE = "rate"
    FX_SPOT = "fx_spot"
    FX_VOLATILITY = "fx_volatility"
    SWAPTION_VOLATILITY = "swaption_volatility"


class QuoteValue:
    """
    Market quote with bid, mid, and ask values.
    
    Parameters
    ----------
    quotation_date : datetime
        Date of the quotation
    bid : float
        Bid price/rate
    mid : float
        Mid price/rate
    ask : float
        Ask price/rate
    tenor : str, optional
        Tenor of the instrument, default is "0D"
        
    Attributes
    ----------
    quotation_date : datetime
        Date of the quotation
    bid : float
        Bid price/rate
    mid : float
        Mid price/rate
    ask : float
        Ask price/rate
    tenor : str
        Tenor of the instrument
    """
    
    def __init__(
        self,
        quotation_date: datetime,
        bid: float,
        mid: float,
        ask: float,
        tenor: str = "0D"
    ):
        """Initialize quote value."""
        self.quotation_date = quotation_date
        self.bid = float(bid)
        self.mid = float(mid)
        self.ask = float(ask)
        self.tenor = tenor
    
    def __repr__(self) -> str:
        """String representation."""
        return f"QuoteValue(date={self.quotation_date}, bid={self.bid:.4f}, mid={self.mid:.4f}, ask={self.ask:.4f}, tenor={self.tenor})"


class RateData:
    """
    Interest rate data point.
    
    Parameters
    ----------
    data_date : datetime
        Date of the data
    tenor : str
        Tenor of the rate (e.g., "3M", "1Y")
    rate : float
        Interest rate value
    is_ois : bool, optional
        Whether this is an OIS rate, default is False
    source : str, optional
        Data source identifier
    multiplier : float, optional
        Rate multiplier (e.g., 0.01 for percentage to decimal)
        
    Attributes
    ----------
    data_date : datetime
        Date of the data
    tenor : str
        Tenor of the rate
    rate : float
        Interest rate value
    is_ois : bool
        Whether this is an OIS rate
    source : str
        Data source identifier
    time_to_maturity : float
        Time to maturity in years
    """
    
    def __init__(
        self,
        data_date: datetime,
        tenor: str,
        rate: float,
        is_ois: bool = False,
        source: str = "DummyDataSource",
        multiplier: float = 1.0/100.0
    ):
        """Initialize rate data."""
        self.data_date = data_date
        self.tenor = tenor
        self._rate = float(rate * multiplier)
        self.is_ois = is_ois
        self.source = source
        self.multiplier = float(multiplier)
        
        # Model values (to be set during calibration)
        self._model_rate: Optional[float] = None
        self._model_zc_price: Optional[float] = None
        self._model_zc_rate: Optional[float] = None
        
        # Market values (to be computed)
        self._market_zc_price: Optional[float] = None
        self._market_zc_rate: Optional[float] = None
        
        # Aggregate values for spread calculations
        self._market_aggregate_a: Optional[float] = None
        self._model_aggregate_a: Optional[float] = None
        
        # Detailed spread components
        self._market_detailed_as: List[float] = []
        self._model_detailed_as: List[float] = []
        
        self.market_full_a: Optional[float] = None
        self.model_full_a: Optional[float] = None
        
        # Calculate time to maturity
        self.time_to_maturity = convert_tenor_to_years(tenor)
    
    def __str__(self) -> str:
        """String representation for reporting."""
        if self.is_ois:
            return f"{self.data_date},{self.tenor},{self.time_to_maturity},{self.rate},{self.model_rate}," \
                   f"{self.market_zc_rate},{self.model_zc_rate},{self.market_zc_price},{self.model_zc_price}"
        else:
            return f"{self.data_date},{self.tenor},{self.time_to_maturity},{self.rate},{self.model_rate}," \
                   f"{self.market_aggregate_a},{self.model_aggregate_a},{self.market_full_a},{self.model_full_a}"
    
    # Properties for rate access
    @property
    def rate(self) -> float:
        """Get the interest rate."""
        return self._rate
    
    @rate.setter
    def rate(self, value: float) -> None:
        """Set the interest rate."""
        self._rate = float(value)
    
    @property
    def model_rate(self) -> Optional[float]:
        """Get the model rate."""
        return self._model_rate
    
    @model_rate.setter
    def model_rate(self, value: Optional[float]) -> None:
        """Set the model rate."""
        self._model_rate = float(value) if value is not None else None
    
    @property
    def model_zc_price(self) -> Optional[float]:
        """Get the model zero coupon price."""
        return self._model_zc_price
    
    @model_zc_price.setter
    def model_zc_price(self, value: Optional[float]) -> None:
        """Set the model zero coupon price."""
        self._model_zc_price = float(value) if value is not None else None
    
    @property
    def model_zc_rate(self) -> Optional[float]:
        """Get the model zero coupon rate."""
        return self._model_zc_rate
    
    @model_zc_rate.setter
    def model_zc_rate(self, value: Optional[float]) -> None:
        """Set the model zero coupon rate."""
        self._model_zc_rate = float(value) if value is not None else None
    
    @property
    def market_zc_rate(self) -> Optional[float]:
        """Get the market zero coupon rate."""
        return self._market_zc_rate
    
    @market_zc_rate.setter
    def market_zc_rate(self, value: Optional[float]) -> None:
        """Set the market zero coupon rate."""
        self._market_zc_rate = float(value) if value is not None else None
    
    @property
    def market_zc_price(self) -> Optional[float]:
        """Get the market zero coupon price."""
        return self._market_zc_price
    
    @market_zc_price.setter
    def market_zc_price(self, value: Optional[float]) -> None:
        """Set the market zero coupon price."""
        self._market_zc_price = float(value) if value is not None else None
    
    @property
    def market_aggregate_a(self) -> Optional[float]:
        """Get the market aggregate A."""
        return self._market_aggregate_a
    
    @market_aggregate_a.setter
    def market_aggregate_a(self, value: Optional[float]) -> None:
        """Set the market aggregate A."""
        self._market_aggregate_a = float(value) if value is not None else None
    
    @property
    def model_aggregate_a(self) -> Optional[float]:
        """Get the model aggregate A."""
        return self._model_aggregate_a
    
    @model_aggregate_a.setter
    def model_aggregate_a(self, value: Optional[float]) -> None:
        """Set the model aggregate A."""
        self._model_aggregate_a = float(value) if value is not None else None


class SwaptionData:
    """
    Swaption volatility data.
    
    Parameters
    ----------
    data_date : datetime
        Date of the data
    expiry : str
        Option expiry tenor
    swap_tenor : str
        Underlying swap tenor
    vol : float
        Implied volatility
    source : str, optional
        Data source identifier
    strike_offset : float, optional
        Strike offset from ATM in basis points
    vol_multiplier : float, optional
        Volatility multiplier
        
    Attributes
    ----------
    data_date : datetime
        Date of the data
    expiry : str
        Option expiry tenor
    swap_tenor : str
        Underlying swap tenor
    vol : float
        Implied volatility
    strike : Optional[float]
        Absolute strike level
    model_vol : Optional[float]
        Model implied volatility
    model_price : Optional[float]
        Model option price
    market_price : Optional[float]
        Market option price
    """
    
    def __init__(
        self,
        data_date: datetime,
        expiry: str,
        swap_tenor: str,
        vol: float,
        source: str = "",
        strike_offset: float = 0,
        vol_multiplier: float = 1.0##/10000.0
    ):
        """Initialize swaption data."""
        self.data_date = data_date
        self.expiry = expiry
        self.swap_tenor = swap_tenor
        self.vol = float(vol * vol_multiplier)
        self.source = source
        self.strike_offset = float(strike_offset)
        
        # To be set during calibration
        self.strike: Optional[float] = None
        self.model_vol: Optional[float] = None
        self.model_price: Optional[float] = None
        self.market_price: Optional[float] = None
        
        # Annuity factor
        self.annuity = 1.0
        
        # Forward swap rates
        self.market_forward_swap_rate: Optional[float] = None
        self.model_forward_swap_rate: Optional[float] = None
        
        # Time to maturity calculations
        self.expiry_maturity = convert_tenor_to_years(expiry)
        self.swap_tenor_maturity = convert_tenor_to_years(swap_tenor)
    
    def __str__(self) -> str:
        """String representation for reporting."""
        return f"{self.data_date},{self.expiry},{self.swap_tenor},{self.strike_offset},{self.strike}," \
               f"{self.vol},{self.model_vol},{self.market_price},{self.model_price}"
    
    def base_rate_data(self) -> RateData:
        """Create a base RateData object for this swaption."""
        return RateData(self.data_date, self.swap_tenor, 0.0, False, self.source)


class DailyData:
    """
    Container for daily market data.
    
    Parameters
    ----------
    quotation_date : datetime
        Date of the market data
    currency : str
        Currency code (e.g., "EUR", "USD")
    ois_rate_data : pd.DataFrame
        OIS rate data
    libor_rate_data : pd.DataFrame
        LIBOR/EURIBOR rate data
    swaption_data_cube : pd.DataFrame
        Swaption volatility data
    source : str, optional
        Data source identifier
        
    Attributes
    ----------
    quotation_date : datetime
        Date of the market data
    currency : str
        Currency code
    ois_rate_data : pd.DataFrame
        OIS rate data
    euribor_rate_data : pd.DataFrame
        LIBOR/EURIBOR rate data
    swaption_data_cube : pd.DataFrame
        Swaption volatility data
    """
    
    def __init__(
        self,
        quotation_date: datetime,
        currency: str,
        ois_rate_data: pd.DataFrame,
        libor_rate_data: pd.DataFrame,
        swaption_data_cube: pd.DataFrame,
        source: str = "DummyDataSource"
    ):
        """Initialize daily data."""
        self.quotation_date = quotation_date
        self.currency = currency
        self.ois_rate_data = ois_rate_data        
        self.euribor_rate_data = libor_rate_data if libor_rate_data is not None else pd.DataFrame()
        self.swaption_data_cube = swaption_data_cube
        self.source = source
        
        # Data counts
        self._nb_ois_data = len(ois_rate_data) 
        self._nb_euribor_data = len(libor_rate_data) if libor_rate_data is not None else 0
        self._nb_swaption_data = len(swaption_data_cube)
        
        # Set currency conventions
        self._set_currency_conventions()
    
    def _set_currency_conventions(self) -> None:
        """Set currency-specific conventions."""
        if self.currency == "EUR":
            self.delta_float_leg = 0.5
            self.delta_fixed_leg = 1.0
            self.using_fra = False
            
            self.libor_delta_float_leg = 0.5
            self.libor_delta_fixed_leg = 1.0
            
            self.ois_delta_float_leg = 1.0
            self.ois_delta_fixed_leg = 1.0
        elif self.currency == "USD":
            self.delta_float_leg = 1.0
            self.delta_fixed_leg = 1.0
            self.using_fra = False
            
            self.libor_delta_float_leg = 1.0
            self.libor_delta_fixed_leg = 1.0
            
            self.ois_delta_float_leg = 1.0
            self.ois_delta_fixed_leg = 1.0
        else:
            raise ValueError(f"Unsupported currency: {self.currency}")
    
    def set_default_convention(
        self,
        delta_float_leg: float = 0.5,
        delta_fixed_leg: float = 1.0,
        using_fra: bool = False
    ) -> None:
        """Set default swap conventions."""
        self.delta_float_leg = float(delta_float_leg)
        self.delta_fixed_leg = float(delta_fixed_leg)
        self.using_fra = using_fra
    
    def short_summary(self) -> str:
        """Get a short summary of the data."""
        return f"DataDate:{self.quotation_date}, number of OIS data:{self._nb_ois_data}, " \
               f"number of Euribor data:{self._nb_euribor_data}, number of Swaption data:{self._nb_swaption_data}"
    
    def print_head(self) -> None:
        """Print the first few rows of each dataset."""
        print("OIS Data:")
        print(self.ois_rate_data.head())
        print("\nEuribor Data:")
        print(self.euribor_rate_data.head())
        print("\nSwaption Data:")
        print(self.swaption_data_cube.head())
    
    def remove_ois_data(self, tenor_list: List[str]) -> None:
        """Remove OIS data for specified tenors."""
        self.ois_rate_data = self.ois_rate_data[~self.ois_rate_data["Tenor"].isin(tenor_list)]
        self._nb_ois_data = len(self.ois_rate_data)
        
    def remove_euribor_data(self, tenor_list: List[str]) -> None:
        if self.euribor_rate_data is None or self.euribor_rate_data.empty:
            return
        self.euribor_rate_data = self.euribor_rate_data[~self.euribor_rate_data["Tenor"].isin(tenor_list)]
        self._nb_euribor_data = len(self.euribor_rate_data)

    def remove_swaption_data(self, expiry_swap_tenor_tuples: List[Tuple[str, str]]) -> None:
        """Remove swaption data for specified expiry/tenor combinations."""
        self.swaption_data_cube = self.swaption_data_cube[
            ~self.swaption_data_cube[['Expiry', 'Tenor']].apply(tuple, 1).isin(expiry_swap_tenor_tuples)
        ]
        self._nb_swaption_data = len(self.swaption_data_cube)
    
    def select_data(
        self,
        ois_tenor_list: List[str],
        euribor_tenor_list: List[str],
        swaption_expiry_swap_tenor_tuples: List[Tuple[str, str]]
    ) -> None:
        """Select only specified data points."""
        self.ois_rate_data = self.ois_rate_data[self.ois_rate_data["Tenor"].isin(ois_tenor_list)]

        if self.euribor_rate_data is None or self.euribor_rate_data.empty:
            self.euribor_rate_data = pd.DataFrame(columns=['Dates', 'Tickers', 'Tenor', 'Data', 'QuotationDate', 'Object'])
        else:
            self.euribor_rate_data = self.euribor_rate_data[self.euribor_rate_data["Tenor"].isin(euribor_tenor_list)]
        self.swaption_data_cube = self.swaption_data_cube[
            self.swaption_data_cube[['Expiry', 'Tenor']].apply(tuple, 1).isin(swaption_expiry_swap_tenor_tuples)
        ]
        
        # Update counts
        self._nb_ois_data = len(self.ois_rate_data)
        self._nb_euribor_data = len(self.euribor_rate_data)
        self._nb_swaption_data = len(self.swaption_data_cube)
    
    def insert_ois_data(self, tenor: str, rate: float, ticker: str = "") -> None:
        """Insert new OIS data point."""
        ois_data_object = RateData(self.quotation_date, tenor, rate, True, "InsertedOISData")
        date_int = self.quotation_date.strftime("%Y%m%d")
        
        new_row = pd.DataFrame([{
            'Dates': date_int,
            'Tickers': ticker,
            'Tenor': tenor,
            'Data': rate,
            'QuotationDate': self.quotation_date,
            'Object': ois_data_object
        }])
        
        self.ois_rate_data = pd.concat([self.ois_rate_data, new_row], ignore_index=True)
        self._nb_ois_data = len(self.ois_rate_data)
    
    def insert_euribor_data(self, tenor: str, rate: float, ticker: str = "") -> None:
        """Insert new Euribor data point."""
        euribor_data_object = RateData(self.quotation_date, tenor, rate, False, "InsertedEuriborData")
        date_int = self.quotation_date.strftime("%Y%m%d")
        
        new_row = pd.DataFrame([{
            'Dates': date_int,
            'Tickers': ticker,
            'Tenor': tenor,
            'Data': rate,
            'QuotationDate': self.quotation_date,
            'Object': euribor_data_object
        }])
        
        if self.euribor_rate_data is None:
            self.euribor_rate_data = new_row
        else:
            self.euribor_rate_data = pd.concat([self.euribor_rate_data, new_row], ignore_index=True)
        self._nb_euribor_data = len(self.euribor_rate_data)
    
    def insert_swaption_data(
        self,
        expiry: str,
        swap_tenor: str,
        vol: float,
        strike_offset: float = 0,
        strike: Optional[float] = None,
        ticker: str = ""
    ) -> None:
        """Insert new swaption data point."""
        swaption_object = SwaptionData(
            self.quotation_date, expiry, swap_tenor, vol, "InsertedSwaptionData", strike_offset
        )
        swaption_object.strike = strike
        date_int = self.quotation_date.strftime("%Y%m%d")
        
        new_row = pd.DataFrame([{
            'Dates': date_int,
            'Tickers': ticker,
            'Expiry': expiry,
            'Tenor': swap_tenor,
            'Data': vol,
            'QuotationDate': self.quotation_date,
            'StrikeOffset': strike_offset,
            'Object': swaption_object
        }])
        
        self.swaption_data_cube = pd.concat([self.swaption_data_cube, new_row], ignore_index=True)
        self._nb_swaption_data = len(self.swaption_data_cube)
    
    def data_summary(self) -> str:
        """Get full data summary."""
        report = self.ois_summary()
        report += self.spread_summary()
        report += self.swaption_summary()
        return report
    
    def ois_summary(self) -> str:
        """Get OIS data summary."""
        report = "\n" + " Printing DailyData Object "
        report += "\n" + "============== OIS data ==============="
        report += "\n" + " dataDate,tenor,TimeToMat,rate,modelRate,marketZcRate,modelZcRate,marketZcPrice,modelZcPrice"
        
        for _, row in self.ois_rate_data.iterrows():
            report += "\n" + str(row["Object"])
        
        return report
    
    def spread_summary(self) -> str:
        report = "\n" + "============== EURIBOR data ==============="
        if self.euribor_rate_data is None or self.euribor_rate_data.empty:
            report += "\n" + "No EURIBOR data available"
            return report
        report += "\n" + " dataDate,tenor,TimeToMat,rate,modelRate,marketAggregateA,modelAggregateA,marketFullA,modelFullA"
        for _, row in self.euribor_rate_data.iterrows():
            report += "\n" + str(row["Object"])
        return report
    
    def swaption_summary(self) -> str:
        """Get swaption data summary."""
        report = "\n" + "============== Swaption data ==============="
        report += "\n" + "dataDate,expiry,swapTenor,strikeOffset,strike,marketVol,ModelVol,MarketPrice,ModelPrice"
        
        for _, row in self.swaption_data_cube.iterrows():
            report += "\n" + str(row["Object"])
        
        report += "\n" + " End printing DailyData Object "
        return report
    
    def print(self) -> None:
        """Print full data summary."""
        print(self.data_summary())

      
    def create_ois_bonds_plot(self, folder: str, 
                          chart_name: str = "OIS",
                          maturity_min: float = 1.0,
                          maturity_max: float = 5.0) -> None:
        """
        Create the chart for the domestic and foreign bonds preice and zero rate.
        
        Parameters
        ----------
        folder : str
            Output folder for plots
        maturity : float
            Maturity to plot
        """
        maturities=[]
        
        bond_mkt_zc_rates=[]
        bond_mkt_prices=[]

        bond_model_zc_rates=[]
        bond_model_prices=[]

        # Filter OIS data for the specified maturity range

        for _, row in self.ois_rate_data.iterrows():

            ois_data = row["Object"]
            maturity = ois_data.time_to_maturity
            if maturity < maturity_min or maturity > maturity_max:
                continue

            maturities.append(ois_data.time_to_maturity)
        
            bond_mkt_prices.append(ois_data._market_zc_price)
            bond_mkt_zc_rates.append(  100.0*ois_data._market_zc_rate)

            bond_model_prices.append( ois_data._model_zc_price)
            bond_model_zc_rates.append(   100.0*ois_data._model_zc_rate)

        # data_points = [
        #    maturities,bond_mkt_prices,bond_mkt_zc_rates
        #    , bond_model_prices, bond_model_zc_rates]
        
        # # Sort by strike
        # data_points.sort(key=lambda x: x[0])
        
        # if not data_points:
        #     return
        
        # maturities,bond_mkt_prices,bond_mkt_zc_rates, bond_model_prices, bond_model_zc_rates = zip(*data_points)
        
        # Convert to numpy arrays
        maturities = np.array(maturities)
        bond_mkt_prices = np.array(bond_mkt_prices)
        bond_mkt_zc_rates = np.array(bond_mkt_zc_rates)
        bond_model_prices = np.array(bond_model_prices)
        bond_model_zc_rates = np.array(bond_model_zc_rates)

        # Get sorting indices
        sort_idx = np.argsort(maturities)

        # Apply sorting to all arrays
        maturities = maturities[sort_idx]
        bond_mkt_prices = bond_mkt_prices[sort_idx]
        bond_mkt_zc_rates = bond_mkt_zc_rates[sort_idx]
        bond_model_prices = bond_model_prices[sort_idx]
        bond_model_zc_rates = bond_model_zc_rates[sort_idx]
        
        filter_short_end_tenors=True
        if filter_short_end_tenors:
            filtered_indices = []
            for i, mat in enumerate(maturities):
                if mat <= 1.0 and i % 3 == 0:  # Every 3rd point for short end
                    filtered_indices.append(i)
                elif mat > 1.0:  # All points for longer maturities
                    filtered_indices.append(i)

            # # Create filtered data
            # plot_maturities = [maturities[i] for i in filtered_indices]
            # plot_mkt_prices = [bond_mkt_prices[i] for i in filtered_indices]
            # plot_model_prices = [bond_model_prices[i] for i in filtered_indices]

            # plot_mkt_zc_rates = [bond_mkt_zc_rates[i] for i in filtered_indices]
            # plot_model_zc_rates = [bond_model_zc_rates[i] for i in filtered_indices]


            maturities = [maturities[i] for i in filtered_indices]
            bond_mkt_prices = [bond_mkt_prices[i] for i in filtered_indices]
            bond_model_prices = [bond_model_prices[i] for i in filtered_indices]

            bond_mkt_zc_rates = [bond_mkt_zc_rates[i] for i in filtered_indices]
            bond_model_zc_rates = [bond_model_zc_rates[i] for i in filtered_indices]

        quote_date_string =self.quotation_date.strftime("%Y%m%d")
        # Create bond price plot
        plt.figure(figsize=(10, 6))
       
        plt.plot(maturities, bond_mkt_prices,   'b-o',  label='Bond Market Price')
        plt.plot(maturities, bond_model_prices, 'r-s',  label='Bond Model Price')

        plt.title(f'{chart_name}: Bonds price for {self.currency} ({quote_date_string})')
        plt.xlabel('Maturity (Years)')
        plt.xlim( max(0.0,maturity_min-0.5), maturity_max+0.5)
        plt.ylabel('price')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        chart_name_file = chart_name.replace(" ", "_")
        quotation_date_str = self.quotation_date.strftime("%Y%m%d_%H%M")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        timestamp = ""
        # plt.savefig(f"{folder}/{chart_name_file}_bonds_price_{quotation_date_str}_{timestamp}.png")
        file_name = folder +f"\{chart_name_file}_bonds_price_{quotation_date_str}_{timestamp}.png"
        plt.savefig(file_name)

        plt.close()

         # Create zero rate plot
        plt.figure(figsize=(10, 6))
       
        plt.plot(maturities, bond_mkt_zc_rates,   'b-o',  label='Bond Market Zero Rate')
        plt.plot(maturities, bond_model_zc_rates, 'r-s',  label='Bond Model Zero Rate')

        plt.title(f'{chart_name}: Bonds zero rates for {self.currency} ({quote_date_string})')
        plt.xlabel('Maturity (Years)')
        plt.xlim( max(0.0,maturity_min-0.5), maturity_max+0.5)
        plt.ylabel('rate(in %)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        chart_name_file = chart_name.replace(" ", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        timestamp=""
        # plt.savefig(f"{folder}/{chart_name_file}_bonds_zcrate_{quotation_date_str}_{timestamp}.png")
        
        file_name = folder +f"\{chart_name_file}_bonds_zcrate_{quotation_date_str}_{timestamp}.png"
        plt.savefig(file_name)

        plt.close()
        


def convert_tenor_to_years(tenor: str) -> float:
    """
    Convert tenor string to years.
    
    Parameters
    ----------
    tenor : str
        Tenor string (e.g., "3M", "1Y", "2W", "30D")
        
    Returns
    -------
    float
        Time in years
        
    Examples
    --------
    >>> convert_tenor_to_years("3M")
    0.25
    >>> convert_tenor_to_years("1Y")
    1.0
    >>> convert_tenor_to_years("2W")
    0.0384615...
    """
    tenor_type_mat_maps = {
        'D': 1.0/360.0,
        'W': 1.0/52.0,
        'M': 1.0/12.0,
        'Y': 1.0
    }
    
    if not tenor or len(tenor) < 2:
        raise ValueError(f"Invalid tenor: {tenor}")
    
    tenor_type = tenor[-1]
    
    if tenor_type not in tenor_type_mat_maps:
        raise ValueError(f"Unknown tenor type: {tenor_type} in tenor {tenor}")
    
    try:
        tenor_length = float(tenor[:-1])
    except ValueError:
        raise ValueError(f"Invalid tenor length in: {tenor}")
    
    time_to_mat = tenor_length * tenor_type_mat_maps[tenor_type]
    
    return time_to_mat


def get_strike_offset(ticker: str) -> float:
    """
    Extract strike offset from ticker string.
    
    Parameters
    ----------
    ticker : str
        Ticker string containing strike information
        
    Returns
    -------
    float
        Strike offset in basis points
    """
    try:
        temp = ticker.split(':')[0].split('.')
        strike = temp[-1]
        
        if strike == 'ATM':
            return 0.0
        else:
            # Extract numeric offset (e.g., from "ATM+100BP")
            offset_str = strike[3:-2]  # Remove "ATM" prefix and "BP" suffix
            return float(offset_str)
    except Exception:
        return 0.0


def read_rate_data(
    current_date: str,
    complete_file_name: str,
    is_ois: bool = True,
    source: str = "DummyDataSource",
    rate_multiplier: float = 1.0/100.0
) -> Optional[pd.DataFrame]:
    """
    Read rate data from CSV file.
    
    Parameters
    ----------
    current_date : str
        Date in YYYYMMDD format
    complete_file_name : str
        Path to CSV file
    is_ois : bool, optional
        Whether this is OIS data
    source : str, optional
        Data source identifier
    rate_multiplier : float, optional
        Multiplier for rate conversion
        
    Returns
    -------
    Optional[pd.DataFrame]
        Rate data for the specified date, or None if not found
    """
    if os.path.exists(complete_file_name)==False:
        print(f'Warning: Rate data file {complete_file_name} does not exist.')
        return None
    
    try:
        all_rate_data = pd.read_csv(complete_file_name)
        all_rate_data['QuotationDate'] = pd.to_datetime(all_rate_data['Dates'].astype(str), format="%Y%m%d")
        all_rate_data = all_rate_data.dropna()
        
        current_date_int = int(current_date)
        current_date_data = all_rate_data[all_rate_data['Dates'] == current_date_int].copy()
        
        if len(current_date_data) == 0:
            print(f'Warning: No data for the date {current_date}')
            return None
        
        # Create RateData objects
        current_date_data['Object'] = current_date_data.apply(
            lambda row: RateData(
                row['QuotationDate'], row['Tenor'], row['Data'],
                is_ois, source, rate_multiplier
            ), axis=1
        )
        
        # Add time to maturity
        current_date_data['TimeToMat'] = current_date_data['Tenor'].apply(convert_tenor_to_years)
        
        # Sort by maturity
        current_date_data = current_date_data.sort_values(by=['TimeToMat'], ascending=True)
        
        return current_date_data
        
    except Exception as e:
        print(f"Error reading rate data: {e}")
        return None


def read_swaption_cube_data(
    current_date: str,
    complete_file_name: str,
    vol_multiplier: float = 1.0,##/10000.0
    source: str = "DummyDataSource"
) -> Optional[pd.DataFrame]:
    """
    Read swaption volatility cube data from CSV file.
    
    Parameters
    ----------
    current_date : str
        Date in YYYYMMDD format
    complete_file_name : str
        Path to CSV file
    source : str, optional
        Data source identifier
        
    Returns
    -------
    Optional[pd.DataFrame]
        Swaption data for the specified date, or None if not found
    """
    try:
        all_data = pd.read_csv(complete_file_name)
        all_data['QuotationDate'] = pd.to_datetime(all_data['Dates'].astype(str), format="%Y%m%d")
        all_data['StrikeOffset'] = all_data['Tickers'].apply(lambda x: get_strike_offset(str(x)))
        all_data = all_data.dropna()
        
        current_date_int = int(current_date)
        current_date_data = all_data[all_data['Dates'] == current_date_int].copy()
        
        if len(current_date_data) == 0:
            print(f'Warning: No data for the date {current_date}')
            return None
        
        # Create SwaptionData objects
        current_date_data['Object'] = current_date_data.apply(
            lambda row: SwaptionData(
                row['QuotationDate'], row['Expiry'], row['Tenor'],
                row['Data'], source, row['StrikeOffset'], vol_multiplier
            ), axis=1
        )
        
        # Add maturities
        current_date_data['ExpiryMat'] = current_date_data['Expiry'].apply(convert_tenor_to_years)
        current_date_data['SwapTenorMat'] = current_date_data['Tenor'].apply(convert_tenor_to_years)
        
        # Sort by date, expiry, and tenor
        current_date_data = current_date_data.sort_values(
            by=['QuotationDate', 'ExpiryMat', 'SwapTenorMat'],
            ascending=[True, True, True]
        )
        
        return current_date_data
        
    except Exception as e:
        print(f"Error reading swaption data: {e}")
        return None
