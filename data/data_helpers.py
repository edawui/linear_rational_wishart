"""
Helper functions and constants for market data handling.

This module provides utility functions and predefined constants for
working with market data, including standard tenor lists and data loaders.
"""

from asyncio import constants
from datetime import datetime
from logging import config
from typing import List, Tuple, Optional
import pandas as pd

from .data_market_data import DailyData
from .data_fx_market_data import CurrencyPairDailyData
from .data_market_data import *
from .data_fx_market_data import *
from ..config import constants

# Default tenor lists
DEFAULT_RATE_TENOR_LIST = [
    "1D", "1W", "2W", "1M", "2M", "3M", "4M", "5M", "6M", "7M", "8M", "9M",
    "10M", "11M", "18M", "1Y", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y",
    "10Y", "12Y", "15Y", "20Y", "25Y", "30Y", "40Y", "50Y"
]

DEFAULT_RATE_TENOR_UP_TO_15Y = [
    "1D", "1W", "2W", "1M", "2M", "3M", "4M", "5M", "6M", "7M", "8M", "9M",
    "10M", "11M", "1Y", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y",
    "10Y", "12Y", "15Y"
]

DEFAULT_EURIBOR_TENOR_UP_TO_15Y = [
    "6M", "1Y", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y", "12Y", "15Y"
]

# Swaption tenor combinations
DEFAULT_SWAPTION_1Y_TO_5Y_TENORS = ["1Y", "2Y", "3Y", "4Y", "5Y"]
DEFAULT_SWAPTION_1Y_TO_5Y = [
    (expiry, tenor) 
    for expiry in DEFAULT_SWAPTION_1Y_TO_5Y_TENORS 
    for tenor in DEFAULT_SWAPTION_1Y_TO_5Y_TENORS
]

DEFAULT_SWAPTION_6M_TO_5Y_TENORS = ["6M", "18M", "1Y", "2Y", "3Y", "4Y", "5Y"]
DEFAULT_SWAPTION_6M_TO_5Y = [
    (expiry, tenor) 
    for expiry in DEFAULT_SWAPTION_6M_TO_5Y_TENORS 
    for tenor in DEFAULT_SWAPTION_6M_TO_5Y_TENORS
]


def get_daily_data(
    current_date: str,
    currency: str,
    ois_rate_data_file: str,
    libor_rate_data_file: str,
    swaption_data_file: str,
    source: str = 'DummyDataSource',
    rate_multiplier: float = 1.0/100.0
) -> Optional[DailyData]:
    """
    Load daily market data from files.
    
    Parameters
    ----------
    current_date : str
        Date in YYYYMMDD format
    currency : str
        Currency code (e.g., "EUR", "USD")
    ois_rate_data_file : str
        Path to OIS rate data file
    libor_rate_data_file : str
        Path to LIBOR/EURIBOR rate data file
    swaption_data_file : str
        Path to swaption volatility data file
    source : str, optional
        Data source identifier
    rate_multiplier : float, optional
        Multiplier for rate conversion
        
    Returns
    -------
    Optional[DailyData]
        Daily market data object, or None if data not found
    """
    # Read OIS data
    ois_data = read_rate_data(
        current_date, ois_rate_data_file, True, source, rate_multiplier
    )
    if ois_data is None:
        return None
    
    # Read LIBOR/EURIBOR data
    euribor_data = read_rate_data(
        current_date, libor_rate_data_file, False, source, rate_multiplier
    )
    if euribor_data is None:
        return None
    
    # Read swaption data
    swaption_data = read_swaption_cube_data(
        current_date, swaption_data_file, source
    )
    if swaption_data is None:
        return None
    
    # Create daily data object
    daily_data = DailyData(
        datetime.strptime(current_date, "%Y%m%d"),
        currency,
        ois_data,
        euribor_data,
        swaption_data,
        source
    )
    
    return daily_data


def get_daily_data_up_6m_to_5y(
    current_date: str,
    currency: str,
    ois_rate_data_file: str,
    libor_rate_data_file: str,
    swaption_data_file: str,
    source: str = 'DummyDataSource',
    rate_multiplier: float = 1.0/100.0
) -> Optional[DailyData]:
    """
    Load daily market data filtered for 6M to 5Y tenors.
    
    Parameters
    ----------
    current_date : str
        Date in YYYYMMDD format
    currency : str
        Currency code
    ois_rate_data_file : str
        Path to OIS rate data file
    libor_rate_data_file : str
        Path to LIBOR rate data file
    swaption_data_file : str
        Path to swaption data file
    source : str, optional
        Data source identifier
    rate_multiplier : float, optional
        Rate conversion multiplier
        
    Returns
    -------
    Optional[DailyData]
        Filtered daily market data
    """
    daily_data = get_daily_data(
        current_date, currency, ois_rate_data_file,
        libor_rate_data_file, swaption_data_file,
        source, rate_multiplier
    )
    
    if daily_data is None:
        return None
    
    # Filter to 6M to 5Y tenors
    daily_data.select_data(
        DEFAULT_RATE_TENOR_UP_TO_15Y,
        DEFAULT_EURIBOR_TENOR_UP_TO_15Y,
        DEFAULT_SWAPTION_6M_TO_5Y
    )
    
    return daily_data


def get_daily_data_up_1y_to_5y(
    current_date: str,
    currency: str,
    ois_rate_data_file: str,
    libor_rate_data_file: str,
    swaption_data_file: str,
    source: str = 'DummyDataSource',
    rate_multiplier: float = 1.0/100.0
) -> Optional[DailyData]:
    """
    Load daily market data filtered for 1Y to 5Y tenors.
    
    Parameters
    ----------
    current_date : str
        Date in YYYYMMDD format
    currency : str
        Currency code
    ois_rate_data_file : str
        Path to OIS rate data file
    libor_rate_data_file : str
        Path to LIBOR rate data file
    swaption_data_file : str
        Path to swaption data file
    source : str, optional
        Data source identifier
    rate_multiplier : float, optional
        Rate conversion multiplier
        
    Returns
    -------
    Optional[DailyData]
        Filtered daily market data
    """
    daily_data = get_daily_data(
        current_date, currency, ois_rate_data_file,
        libor_rate_data_file, swaption_data_file,
        source, rate_multiplier
    )
    
    if daily_data is None:
        return None
    
    # Filter to 1Y to 5Y tenors
    daily_data.select_data(
        DEFAULT_RATE_TENOR_UP_TO_15Y,
        DEFAULT_EURIBOR_TENOR_UP_TO_15Y,
        DEFAULT_SWAPTION_1Y_TO_5Y
    )
    
    return daily_data


def get_currency_pair_data(
    current_date: str,
    domestic_currency: str,
    foreign_currency: str,
    fx_spot: float,
    domestic_ois_file: str,
    domestic_libor_file: str,
    domestic_swaption_file: str,
    foreign_ois_file: str,
    foreign_libor_file: str,
    foreign_swaption_file: str,
    fx_vol_file: str,
    source: str = 'DummyDataSource',
    rate_multiplier: float = 1.0,
    filter_tenors: bool = True
) -> Optional[CurrencyPairDailyData]:
    """
    Load currency pair market data.
    
    Parameters
    ----------
    current_date : str
        Date in YYYYMMDD format
    domestic_currency : str
        Domestic currency code
    foreign_currency : str
        Foreign currency code
    fx_spot : float
        FX spot rate
    domestic_ois_file : str
        Path to domestic OIS data
    domestic_libor_file : str
        Path to domestic LIBOR data
    domestic_swaption_file : str
        Path to domestic swaption data
    foreign_ois_file : str
        Path to foreign OIS data
    foreign_libor_file : str
        Path to foreign LIBOR data
    foreign_swaption_file : str
        Path to foreign swaption data
    fx_vol_file : str
        Path to FX volatility data
    source : str, optional
        Data source identifier
    rate_multiplier : float, optional
        Rate conversion multiplier
    filter_tenors : bool, optional
        Whether to filter to standard tenors
        
    Returns
    -------
    Optional[CurrencyPairDailyData]
        Currency pair market data
    """
    # Load domestic currency data
    print("\nLoading domestic currency data")
    if filter_tenors:
        domestic_data = get_daily_data_up_6m_to_5y(
            current_date, domestic_currency,
            domestic_ois_file, domestic_libor_file,
            domestic_swaption_file, source, rate_multiplier
        )
    else:
        domestic_data = get_daily_data(
            current_date, domestic_currency,
            domestic_ois_file, domestic_libor_file,
            domestic_swaption_file, source, rate_multiplier
        )
    
    if domestic_data is None:
        return None
    
    # Load foreign currency data
    print("\nLoading foreign currency data")
    if filter_tenors:
        foreign_data = get_daily_data_up_6m_to_5y(
            current_date, foreign_currency,
            foreign_ois_file, foreign_libor_file,
            foreign_swaption_file, source, rate_multiplier
        )
    else:
        foreign_data = get_daily_data(
            current_date, foreign_currency,
            foreign_ois_file, foreign_libor_file,
            foreign_swaption_file, source, rate_multiplier
        )
    
    if foreign_data is None:
        return None
    
    # Load FX volatility data
    print("\nLoading FX volatility data")
    raw_fx_vol_data = read_fx_vol_raw_data(
        current_date, fx_spot, fx_vol_file, source
    )
    
    if raw_fx_vol_data is None:
        # Create empty DataFrame if no FX vol data
        raw_fx_vol_data = pd.DataFrame()
    
    # Create currency pair data
    print("\nCreate currency pair data")
    currency_pair_data = CurrencyPairDailyData(
        datetime.strptime(current_date, "%Y%m%d"),
        domestic_data,
        foreign_data,
        raw_fx_vol_data,
        source
    )
    
    return currency_pair_data


def get_testing_excel_data(
        current_date = '20250401' #'20250530'
        , ccy_pair: str = "EURUSD",
        # folder: str = r"C:\Users\edem_\Dropbox\LinearRationalWishart_Work\Data",
        folder: str =  constants.mkt_data_folder ##r"C:\Users\edem_\Dropbox\LinearRationalWishart_Work\Data\Data_new"  
        ) -> Optional[CurrencyPairDailyData]:
    """
    Load test data for a currency pair.
    
    Parameters
    ----------
    folder : str
        Path to data folder
    ccy_pair : str
        Currency pair identifier ("EURUSD" or "USDJPY")
        
    Returns
    -------
    Optional[CurrencyPairDailyData]
        Test currency pair data
    """
    usd_ois_file = folder + r"\USD-SOFR.csv"
    # current_date = '20250530'
    rate_multiplier = 1.0
    
    if ccy_pair == "EURUSD":
        fx_spot = 1.1359
        fx_spot_file =  folder + r"\FxSpot.csv"
        foreign_currency = "EUR"
        foreign_ois_file = folder + r"\EUR-ESTR-SOFR.csv"
        foreign_libor_file = folder + r"\EUR-ESTR-SOFR.csv" ##temporary+ r"\EUR-EURIBOR-6M.csv"
        foreign_swaption_file = folder + r"\EUR-EURIBOR-6M-VOL.csv"
        usd_swaption_file = folder + r"\USD-SOFR-ON-VOL.csv"
        fx_vol_file = folder + r"\EURUSDVOL.csv"
        
        fx_df = pd.read_csv(fx_spot_file, index_col=0, parse_dates=True)
        # fx_spot_value = fx_df[(fx_df.index == current_date) and (fx_df['Ticker'] == 'FX.EURUSD-SPOT.MID')].iloc[0]['Data']
        fx_spot_value =   fx_df[(fx_df.index == current_date) & (fx_df['Tickers'] == 'FX.EURUSD-SPOT.MID')].iloc[0]['Data']
        fx_spot=fx_spot_value
        # For EURUSD, USD is domestic, EUR is foreign
        return get_currency_pair_data(
            current_date,
            "USD",  # Domestic
            "EUR",  # Foreign
            fx_spot,
            usd_ois_file,
            usd_ois_file,  # Using OIS for LIBOR as well
            usd_swaption_file,
            foreign_ois_file,
            foreign_libor_file,
            foreign_swaption_file,
            fx_vol_file,
            "DummyDataSource",
            rate_multiplier,
            True  # Filter tenors
        )
        
    elif ccy_pair == "USDJPY":
        fx_spot = 143.89
        fx_spot_file =  folder + r"\FxSpot.csv"
        foreign_currency = "JPY"
        foreign_ois_file = folder + r"\JPY-TONAR-SOFR.csv"
        foreign_libor_file = folder + r"\JPY-TONAR-SOFR.csv"  # Using EUR as proxy
        foreign_swaption_file = folder + r"\JPY-TONAR-ON-VOL.csv"  # Using EUR as proxy
        usd_swaption_file = folder + r"\USD-SOFR-ON-VOL.csv"
        fx_vol_file = folder + r"\USDJPYVOL.csv"
        
        fx_df = pd.read_csv(fx_spot_file, index_col=0, parse_dates=True)
        # fx_spot_value = fx_df[fx_df.index == current_date and fx_df['Ticker'] == 'FX.USDJPY-SPOT.MID'].iloc[0]['Data']
        fx_spot_value = fx_df[(fx_df.index == current_date) & (fx_df['Tickers'] == 'FX.USDJPY-SPOT.MID')].iloc[0]['Data']
        fx_spot=fx_spot_value


        # For USDJPY, JPY is domestic, USD is foreign
        return get_currency_pair_data(
            current_date,
            "JPY",  # Domestic
            "USD",  # Foreign
            fx_spot,
            foreign_ois_file,
            foreign_libor_file,
            foreign_swaption_file,
            usd_ois_file,
            usd_ois_file,  # Using OIS for LIBOR as well
            usd_swaption_file,
            fx_vol_file,
            "DummyDataSource",
            rate_multiplier,
            True  # Filter tenors
        )
    else:
        raise ValueError(f"Unknown currency pair for testing data: {ccy_pair}")


# Convenience functions for specific data types
def load_rate_curve(
    current_date: str,
    currency: str,
    ois_file: str,
    libor_file: str,
    filter_short_tenors: bool = True,
    source: str = "DummyDataSource"
) -> Optional[DailyData]:
    """
    Load rate curve data without swaption data.
    
    Parameters
    ----------
    current_date : str
        Date in YYYYMMDD format
    currency : str
        Currency code
    ois_file : str
        Path to OIS data
    libor_file : str
        Path to LIBOR data
    filter_short_tenors : bool, optional
        Whether to exclude very short tenors
    source : str, optional
        Data source
        
    Returns
    -------
    Optional[DailyData]
        Rate curve data
    """
    # Create empty swaption DataFrame
    empty_swaption = pd.DataFrame(columns=['Dates', 'Tickers', 'Expiry', 'Tenor', 'Data', 'QuotationDate', 'StrikeOffset', 'Object'])
    
    # Read rate data
    ois_data = read_rate_data(current_date, ois_file, True, source)
    libor_data = read_rate_data(current_date, libor_file, False, source)
    
    if ois_data is None or libor_data is None:
        return None
    
    # Create daily data
    daily_data = DailyData(
        datetime.strptime(current_date, "%Y%m%d"),
        currency,
        ois_data,
        libor_data,
        empty_swaption,
        source
    )
    
    # Filter short tenors if requested
    if filter_short_tenors:
        daily_data.remove_ois_data(["1D", "1W", "2W"])
    
    return daily_data
