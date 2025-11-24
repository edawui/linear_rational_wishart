"""
Foreign exchange market data structures and utilities.

This module provides classes and functions for handling FX spot rates
and FX option volatility data.
"""

from datetime import datetime
from typing import Optional, Union, List, Tuple, Dict, Any
from enum import Enum
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

from .data_market_data import convert_tenor_to_years
from .data_market_data import DailyData
from ..utils.jax_utils import is_jax_available, ensure_jax_array



class CalibWeightType(Enum):
    """Types of Calib Weight for FX vol market data."""
    MONEYNESS = 1
    INV_MONEYNESS = 2
    UNIFORM = 3
    DISTANCE_FROM_ATM=4
    INV_DISTANCE_FROM_ATM=5
    CUSTOM=6


class FxDataType(Enum):
    """Types of FX market data."""
    FX_POINT = 1
    FX_OUTRIGHT = 2


class FxVolDataType(Enum):
    """Types of FX volatility quotes."""
    ATM = 1
    BUTTERFLY = 2
    RISK_REVERSAL = 3
    CALL_PUT = 4


class RawFxVolData:
    """
    Raw FX volatility data as quoted in the market.
    
    Parameters
    ----------
    data_date : datetime
        Date of the data
    fx_spot : float
        FX spot rate
    expiry : str
        Option expiry tenor
    vol : float
        Volatility value
    delta_strike : float
        Delta of the option
    fx_vol_data_type : FxVolDataType
        Type of volatility quote
    source : str, optional
        Data source identifier
        
    Attributes
    ----------
    data_date : datetime
        Date of the data
    fx_spot : float
        FX spot rate
    expiry : str
        Option expiry tenor
    vol : float
        Volatility value
    delta_strike : float
        Delta of the option
    fx_vol_data_type : FxVolDataType
        Type of volatility quote
    expiry_maturity : float
        Time to expiry in years
    """
    
    def __init__(
        self,
        data_date: datetime,
        fx_spot: float,
        expiry: str,
        vol: float,
        delta_strike: str,#float,
        fx_vol_data_type: FxVolDataType,
        source: str = ""
    ):
        """Initialize raw FX volatility data."""
        self.data_date = data_date
        self.fx_spot = float(fx_spot)
        self.expiry = expiry
        self.vol = float(vol)
        self.delta_strike = delta_strike #float(delta_strike)
        self.fx_vol_data_type = fx_vol_data_type
        self.source = source
        self.expiry_maturity = convert_tenor_to_years(expiry)


class FxVolData:
    """
    Processed FX option volatility data.
    
    Parameters
    ----------
    data_date : datetime
        Date of the data
    fx_spot : float
        FX spot rate
    expiry : str
        Option expiry tenor
    vol : float
        Implied volatility
    strike : float
        Option strike
    call_or_put : bool
        True for call, False for put
    source : str, optional
        Data source identifier
    vol_multiplier : float, optional
        Volatility multiplier
        
    Attributes
    ----------
    data_date : datetime
        Date of the data
    fx_spot : float
        FX spot rate
    expiry : str
        Option expiry tenor
    vol : float
        Implied volatility
    strike : float
        Option strike
    call_or_put : bool
        True for call, False for put
    market_vol : float
        Market implied volatility
    model_vol : float
        Model implied volatility
    market_price : float
        Market option price
    model_price : float
        Model option price
    """
    
    def __init__(
        self,
        data_date: datetime,
        fx_spot: float,
        expiry: str,
        vol: float,
        strike: float,
        call_or_put: bool,
        source: str = "",
        vol_multiplier: float = 1.0
    ):
        """Initialize FX volatility data."""
        self.data_date = data_date
        self.fx_spot = float(fx_spot)
        self.expiry = expiry
        self.vol = float(vol_multiplier * vol)
        self.source = source
        self.strike = float(strike)
        self.call_or_put = call_or_put
        
        # Model values
        self.model_vol = 0.0
        self.market_vol = float(vol)
        self.model_price = 0.0
        self.market_price = 0.0
        
        self.expiry_maturity = convert_tenor_to_years(expiry)
        self.moneyness= self.strike/self.fx_spot
        self.weight=1.0
        self.fx_fwd= self.fx_spot 

    def set_fx_fwd(self, fx_fwd):
        self.fx_fwd= fx_fwd 
        self.moneyness= self.strike/self.fx_fwd

    def set_weight(self, weight_tpe:CalibWeightType= CalibWeightType.UNIFORM, custom_weigh:float=1.0):
        self.weight=custom_weigh if weight_tpe == CalibWeightType.CUSTOM else 1.0
        if weight_tpe ==CalibWeightType.UNIFORM:
            self.weight = 1.0 
        elif weight_tpe == CalibWeightType.MONEYNESS:
            self.weight = self.moneyness if self.moneyness != 0 else 1.0
        elif weight_tpe == CalibWeightType.INV_MONEYNESS:
            self.weight = 1.0/self.moneyness if self.moneyness != 0 else 1.0
        elif weight_tpe == CalibWeightType.CUSTOM:
            self.weight = custom_weigh
        elif weight_tpe == CalibWeightType.DISTANCE_FROM_ATM:
            dist= abs(self.strike - self.fx_fwd)
            self.weight = 1.0 + dist if dist != 0 else 1.0
        elif weight_tpe == CalibWeightType.INV_DISTANCE_FROM_ATM:
            dist= abs(self.strike - self.fx_fwd)
            self.weight = 1/(1.0 + dist) if dist != 0 else 1.0
        else:
            self.weight = 1.0
            # raise ValueError(f"Unknown weight type: {weight_tpe}")


    def __str__(self) -> str:
        """String representation."""
        option_type = "call" if self.call_or_put else "put"
        return f"{self.data_date},{self.expiry},{self.expiry_maturity},{self.vol:.4f},{self.strike:.6f}," \
               f"{self.market_vol:.6f},{self.model_vol:.6f},{self.market_price},{self.model_price}," \
               f"{option_type},{self.source}"


class CurrencyPairDailyData:
    """
    Container for currency pair market data.
    
    Parameters
    ----------
    quotation_date : datetime
        Date of the market data
    domestic_currency_daily_data : DailyData
        Domestic currency rate data
    foreign_currency_daily_data : DailyData
        Foreign currency rate data
    raw_fx_vol_data : pd.DataFrame
        Raw FX volatility data
    source : str, optional
        Data source identifier
        
    Attributes
    ----------
    quotation_date : datetime
        Date of the market data
    domestic_currency_daily_data : DailyData
        Domestic currency rate data
    foreign_currency_daily_data : DailyData
        Foreign currency rate data
    fx_spot : float
        FX spot rate
    fx_vol_data : List[FxVolData]
        Processed FX volatility data
    """
    
    def __init__(
        self,
        quotation_date: datetime,
        domestic_currency_daily_data: DailyData,
        foreign_currency_daily_data: DailyData,
        raw_fx_vol_data: pd.DataFrame,
        source: str = "DummyDataSource"
    ):
        """Initialize currency pair data."""
        self.quotation_date = quotation_date
        self.domestic_currency_daily_data = domestic_currency_daily_data
        self.foreign_currency_daily_data = foreign_currency_daily_data
        self.source = source
        self.raw_fx_vol_data = raw_fx_vol_data
        
        # Extract FX spot from raw data
        self.fx_spot = 1.0
        if not raw_fx_vol_data.empty and 'Object' in raw_fx_vol_data.columns:
            self.fx_spot = float(raw_fx_vol_data.iloc[0]['Object'].fx_spot)
        
        self.fx_vol_data: Optional[List[FxVolData]] = None
        self.reprice_all_option= False
    
    def set_weight(self, weight_tpe:CalibWeightType= CalibWeightType.UNIFORM, custom_weigh:float=1.0):
        for fx_vol in self.fx_vol_data:
            fx_vol.set_weight(weight_tpe, custom_weigh)

    def process_raw_fx_vol_data(
        self,
        domestic_currency_curve: Any,
        foreign_currency_curve: Any,
        min_maturity: float = 1.0,
        max_maturity: float = 10.0
    ) -> None:
        """
        Process raw FX volatility data into standard format.
        
        Parameters
        ----------
        domestic_currency_curve : Any
            Domestic currency discount curve
        foreign_currency_curve : Any
            Foreign currency discount curve
        min_maturity : float, optional
            Minimum maturity to include
        max_maturity : float, optional
            Maximum maturity to include
        """
        processed_fx_vol_data = create_fx_vol_data_sheet(
            domestic_currency_curve,
            foreign_currency_curve,
            self.raw_fx_vol_data#,
            # min_maturity,
            # max_maturity
        )
        self.fx_vol_data =  copy.deepcopy(processed_fx_vol_data)  #processed_fx_vol_data
        
        # self.fx_vol_calib_data = processed_fx_vol_data
        # print(type(processed_fx_vol_data))
        # print(type(processed_fx_vol_data[0]))

        # to_calib=[]
        # for fx in self.fx_vol_data:
        #         if min_maturity <= fx.expiry_maturity <= max_maturity:
        #             to_calib.append(fx)

        to_calib = [fx for fx in self.fx_vol_data 
            if min_maturity <= fx.expiry_maturity <= max_maturity]

        self.fx_vol_calib_data = copy.deepcopy(to_calib) 

        print(len(self.fx_vol_data))
        print(len(self.fx_vol_calib_data))
    
    def ois_summary(self) -> str:
        """Get OIS data summary for both currencies."""
        report = "\n" + " Printing CurrencyPairDailyData Object "
        report += "\n" + "============== Domestic OIS data ==============="
        report += "\n" + " dataDate,tenor,TimeToMat,rate,modelRate,marketZcRate,modelZcRate,marketZcPrice,modelZcPrice"
        
        for _, row in self.domestic_currency_daily_data.ois_rate_data.iterrows():
            report += "\n" + str(row["Object"])
        
        report += "\n" + "============== Foreign OIS data ==============="
        report += "\n" + " dataDate,tenor,TimeToMat,rate,modelRate,marketZcRate,modelZcRate,marketZcPrice,modelZcPrice"
        
        for _, row in self.foreign_currency_daily_data.ois_rate_data.iterrows():
            report += "\n" + str(row["Object"])
        
        return report
    
    def option_summary(self) -> str:
        """Get FX option data summary."""
        report = "\n" + "============== Fx Option data ==============="
        report += "\n" + "dataDate,expiry,expiryMat,vol,strike,marketVol,modelVol,marketPrice,modelPrice,callOrPut,source"
        
        if self.fx_vol_data:
            for fx_option in self.fx_vol_data:
                report += "\n" + str(fx_option)
        
        report += "\n" + " End printing CurrencyPairDailyData Object "
        return report
    
    def print_fx_vol_data(self) -> None:
        """Print FX volatility data summary."""
        print(self.option_summary())
    
    def create_all_plots(
        self,
        folder: str,
        file_prefix="",
        maturity_min: float = 1.0,
        maturity_max: float = 5.0
    ) -> None:
        """
        Create volatility smile plots for all maturities.
        
        Parameters
        ----------
        folder : str
            Output folder for plots
        maturity_min : float, optional
            Minimum maturity to plot
        maturity_max : float, optional
            Maximum maturity to plot
        """
       
        if not self.reprice_all_option:
            if not self.fx_vol_calib_data:
                print("No FX vol data to plot")
                return
        
            maturities = sorted(list(set(
                fx.expiry_maturity for fx in self.fx_vol_calib_data
                if maturity_min <= fx.expiry_maturity <= maturity_max
            )))
        
            for maturity in maturities:
                self.create_plot(folder,file_prefix, maturity)
        else:
             # print("+======================================================================+")
             # print(f"self.reprice_all_option= {self.reprice_all_option}")
             # print(f"maturity_min={maturity_min}")
             # print(f"maturity_max={maturity_max}")
             # print("+======================================================================+")

             if not self.fx_vol_data:
                print("No FX vol data to plot")
                return
        
             maturities = sorted(list(set(
                fx.expiry_maturity for fx in self.fx_vol_data
                if maturity_min <= fx.expiry_maturity <= maturity_max
             )))
             # print(f"maturities= {maturities}")
             # print("+======================================================================+")
        
             for maturity in maturities:
                self.create_plot(folder,file_prefix, maturity)

        self.plot_ATM(folder,file_prefix, maturity_min, maturity_max)

    def create_plot(self, folder: str,file_prefix: str="", maturity: float = 1.0) -> None:
        """
        Create volatility smile plot for a specific maturity.
        
        Parameters
        ----------
        folder : str
            Output folder for plots
        maturity : float
            Maturity to plot
        """
        if not self.fx_vol_calib_data:
            print("No FX vol data to plot")
            return
        
        # Filter data for this maturity
        if self.reprice_all_option:
            filtered_data = [fx for fx in self.fx_vol_data if abs(fx.expiry_maturity - maturity) < 1e-6]
        else:
            filtered_data = [fx for fx in self.fx_vol_calib_data if abs(fx.expiry_maturity - maturity) < 1e-6]
        
        if not filtered_data:
            print(f"No FX vol data found for maturity {maturity}")
            return
        
        # Extract data for plotting
        data_points = [
            (fx.strike, fx.market_vol, fx.model_vol, fx.market_price, fx.model_price)
            for fx in filtered_data
        ]
        
        # Sort by strike
        data_points.sort(key=lambda x: x[0])
        
        if not data_points:
            return
        
        strikes, market_vols, model_vols, market_prices, model_prices = zip(*data_points)
        
        quote_date_string =self.quotation_date.strftime("%Y%m%d")

        # Create volatility plot
        plt.figure(figsize=(10, 6))
        plt.plot(strikes, market_vols, 'b-o', label='FX Market Volatility')
        plt.plot(strikes, model_vols, 'r-s', label='FX Model Volatility')
        plt.title(f'FX Option Volatility at Maturity {maturity:.0f}Y ({quote_date_string})')
        plt.xlabel('Strike')
        plt.ylabel('Implied Volatility')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        timestamp = ""
        quotation_date_str = self.quotation_date.strftime("%Y%m%d_%H%M")
        plt.savefig(f"{folder}/{file_prefix}_fx_option_vol_maturity_{maturity:.2f}_{quotation_date_str}_{timestamp}.png")
        plt.close()
        
        # Create price plot
        plt.figure(figsize=(10, 6))
        plt.plot(strikes, market_prices, 'b-o', label='FX Market Prices')
        plt.plot(strikes, model_prices, 'r-s', label='FX Model Prices')
        plt.title(f'FX Option Price at Maturity {maturity:.0f}Y ({quote_date_string})')
        plt.xlabel('Strike')
        plt.ylabel('Price')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # plt.savefig(f"{folder}/{file_prefix}_fx_option_price_maturity_{maturity:.2f}_{quotation_date_str}_{timestamp}.png")
        file_name = folder +f"\{file_prefix}_fx_option_price_maturity_{maturity:.2f}_{quotation_date_str}_{timestamp}.png"
        plt.savefig(file_name)

        plt.close()

    def plot_ATM(self, folder: str,file_prefix: str="",
                 maturity_min: float = 1.0,
                maturity_max: float = 5.0):
        options_by_maturity = defaultdict(list)
        all_options = []
        if not self.reprice_all_option:
        
            for option_data in self.fx_vol_calib_data:
                mat = option_data.expiry_maturity#_mat
                if maturity_min <= mat <= maturity_max:
                    options_by_maturity[mat].append(option_data)
                    all_options.append(option_data)
        else:
            for option_data in self.fx_vol_data:
                mat = option_data.expiry_maturity#_mat
                if maturity_min <= mat <= maturity_max:
                    options_by_maturity[mat].append(option_data)
                    all_options.append(option_data)

        # if True:# self.config.use_atm_only:
            # Select ATM options only
        atm_options = []
        for mat, options in options_by_maturity.items():
                options_sorted = sorted(options, key=lambda x: x.strike)
                if len(options_sorted) >= 3:
                    # Select middle strike (typically ATM)
                    atm_options.append(options_sorted[len(options_sorted) // 2])
                else:
                    print(f"Warning: Not enough strikes for maturity {mat}")
        
        atm_options = sorted(atm_options, key=lambda x: x.expiry_maturity)
        if len(atm_options) >=1:
            # Extract data for plotting
            data_points = [
                (fx.expiry_maturity, fx.market_vol, fx.model_vol, fx.market_price, fx.model_price)
                for fx in atm_options
            ]
        
            # Sort by strike
            data_points.sort(key=lambda x: x[0])
        
            if not data_points:
                return
        
            maturities, market_vols, model_vols, market_prices, model_prices = zip(*data_points)
        
            quote_date_string =self.quotation_date.strftime("%Y%m%d")

            # Create volatility plot
            plt.figure(figsize=(10, 6))
            plt.plot(maturities, market_vols, 'b-o', label='FX Market Volatility')
            plt.plot(maturities, model_vols, 'r-s', label='FX Model Volatility')
            plt.title(f'ATM FX Option Volatility ({quote_date_string})')
            plt.xlabel('Maturity')
            plt.ylabel('Implied Volatility')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
        
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            timestamp = ""
            quotation_date_str = self.quotation_date.strftime("%Y%m%d_%H%M")
            file_name = folder +f"\{file_prefix}_ATM_fx_option_vol_{quotation_date_str}_{timestamp}.png"
            # plt.savefig(f"{folder}/{file_prefix}_ATM_fx_option_vol_{quotation_date_str}_{timestamp}.png")
            plt.savefig(file_name)
            plt.close()
        
            # Create price plot
            plt.figure(figsize=(10, 6))
            plt.plot(maturities, market_prices, 'b-o', label='FX Market Prices')
            plt.plot(maturities, model_prices, 'r-s', label='FX Model Prices')
            plt.title(f'ATM FX Option Volatility ({quote_date_string})')
            plt.xlabel('Maturity')
            plt.ylabel('Price')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
        
            # plt.savefig(f"{folder}/{file_prefix}_ATM_fx_option_price_{quotation_date_str}_{timestamp}.png")
            file_name = folder +f"\{file_prefix}_ATM_fx_option_price_{quotation_date_str}_{timestamp}.png"
            plt.savefig(file_name)

            plt.close()


def get_fx_vol_data_type(string_fx_vol_data_type: str) -> FxVolDataType:
    """
    Convert string to FxVolDataType enum.
    
    Parameters
    ----------
    string_fx_vol_data_type : str
        String representation of volatility type
        
    Returns
    -------
    FxVolDataType
        Corresponding enum value
    """
    type_map = {
        "ATM": FxVolDataType.ATM,
        "BF": FxVolDataType.BUTTERFLY,
        "RR": FxVolDataType.RISK_REVERSAL
    }
    
    if string_fx_vol_data_type not in type_map:
        raise ValueError(f"Unknown FX vol type: {string_fx_vol_data_type}")
    
    return type_map[string_fx_vol_data_type]


def order_fx_vol_data(fx_vol_data_list: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Organize FX vol data by expiry and type.
    
    Parameters
    ----------
    fx_vol_data_list : pd.DataFrame
        Raw FX volatility data
        
    Returns
    -------
    Dict[str, Dict[str, Any]]
        Organized data by expiry
    """
    grouped = defaultdict(lambda: {
        "ATM": None,
        "ButterFly": defaultdict(list),
        "RiskReversals": defaultdict(list)
    })
    
    for _, row in fx_vol_data_list.iterrows():
        fx_option = row['Object']
        expiry = fx_option.expiry
        fx_type = fx_option.fx_vol_data_type
        strike = fx_option.delta_strike
        
        if fx_type == FxVolDataType.ATM:
            grouped[expiry]["ATM"] = fx_option
        elif fx_type == FxVolDataType.BUTTERFLY:
            grouped[expiry]["ButterFly"][strike].append(fx_option)
        elif fx_type == FxVolDataType.RISK_REVERSAL:
            grouped[expiry]["RiskReversals"][strike].append(fx_option)
    
    # Pair butterflies and risk reversals
    result = {}
    for expiry, data in grouped.items():
        pairs = []
        unpaired_bf = []
        unpaired_rr = []
        
        all_strikes = set(data["ButterFly"].keys()) | set(data["RiskReversals"].keys())
        
        for strike in all_strikes:
            bfs = data["ButterFly"].get(strike, [])
            rrs = data["RiskReversals"].get(strike, [])
            
            # Pair up butterflies and risk reversals
            for bf, rr in zip(bfs, rrs):
                pairs.append((bf, rr))
            
            # Track unpaired quotes
            if len(bfs) > len(rrs):
                unpaired_bf.extend(bfs[len(rrs):])
            if len(rrs) > len(bfs):
                unpaired_rr.extend(rrs[len(bfs):])
        
        result[expiry] = {
            "ATM": data["ATM"],
            "Pairs": pairs,
            "UnpairedButterFly": unpaired_bf,
            "UnpairedRiskReversals": unpaired_rr
        }
    
    return result


def create_fx_vol_data_sheet(
    domestic_df_curve: Any,
    foreign_df_curve: Any,
    raw_fx_vol_datasheet: pd.DataFrame #,
    # min_maturity: float = 1.0,
    # max_maturity: float = 10.0
) -> List[FxVolData]:
    """
    Transform raw FX vol data into processed FxVolData objects.
    
    Parameters
    ----------
    domestic_df_curve : Any
        Domestic discount factor curve
    foreign_df_curve : Any
        Foreign discount factor curve
    raw_fx_vol_datasheet : pd.DataFrame
        Raw FX volatility data
    min_maturity : float, optional
        Minimum maturity to include
    max_maturity : float, optional
        Maximum maturity to include
        
    Returns
    -------
    List[FxVolData]
        Processed FX volatility data
    """
    fx_vol_data_sheet = []
    organized = order_fx_vol_data(raw_fx_vol_datasheet)
    
    for expiry, data in organized.items():
        maturity = convert_tenor_to_years(expiry)
        
        # if not (min_maturity <= maturity <= max_maturity):
        #     continue
        
        raw_atm = data["ATM"]
        if raw_atm is None:
            print(f"Warning: No ATM data for expiry {expiry}")
            continue
        
        # Process paired butterfly and risk reversal quotes
        for bf, rr in data["Pairs"]:
            delta_strike = bf.delta_strike
            bf_vol = bf.vol
            rr_vol = rr.vol
            data_scale_factor = getattr(raw_atm, "dataScaleFactor", 1.0)
            
            vanilla_list, atm, call, put = create_call_put_from_fx_vanilla(
                domestic_df_curve,
                foreign_df_curve,
                raw_atm,
                delta_strike,
                rr_vol,
                bf_vol,
                data_scale_factor
            )
            
            fx_vol_data_sheet.extend([call, put])
        
        # Add ATM quote
        if raw_atm:
            atm_strike = calculate_atm_strike(
                raw_atm.fx_spot,
                domestic_df_curve,
                foreign_df_curve,
                maturity
            )
            
            atm_fx_vol = FxVolData(
                raw_atm.data_date,
                raw_atm.fx_spot,
                raw_atm.expiry,
                raw_atm.vol,
                atm_strike,
                True,  # ATM is typically quoted for calls
                raw_atm.source
            )
            fx_vol_data_sheet.append(atm_fx_vol)
        
        # Log unpaired quotes
        for bf in data["UnpairedButterFly"]:
            print(f"  Unpaired Butterfly: {bf}")
        for rr in data["UnpairedRiskReversals"]:
            print(f"  Unpaired RiskReversal: {rr}")
    
    return fx_vol_data_sheet


def calculate_atm_strike(
    fx_spot: float,
    domestic_df_curve: Any,
    foreign_df_curve: Any,
    maturity: float
) -> float:
    """
    Calculate ATM strike (forward).
    
    Parameters
    ----------
    fx_spot : float
        FX spot rate
    domestic_df_curve : Any
        Domestic discount factor curve
    foreign_df_curve : Any
        Foreign discount factor curve
    maturity : float
        Time to maturity
        
    Returns
    -------
    float
        ATM strike (forward rate)
    """
    domestic_df = domestic_df_curve.Bond(maturity)
    foreign_df = foreign_df_curve.Bond(maturity)
    
    forward = fx_spot * foreign_df / domestic_df
    
    return forward


def create_call_put_from_fx_vanilla(
    domestic_df_curve: Any,
    foreign_df_curve: Any,
    atm_vanilla: RawFxVolData,
    delta_strike: float,
    risk_reversal_vol: float,
    butterfly_vol: float,
    data_scale_factor: float,
    data_strike_scale_factor: float = 1.0/100.0
) -> Tuple[List[FxVolData], FxVolData, FxVolData, FxVolData]:
    """
    Create call and put FX options from vanilla quotes.
    
    Parameters
    ----------
    domestic_df_curve : Any
        Domestic discount factor curve
    foreign_df_curve : Any
        Foreign discount factor curve
    atm_vanilla : RawFxVolData
        ATM volatility data
    delta_strike : float
        Delta level for the options
    risk_reversal_vol : float
        Risk reversal volatility
    butterfly_vol : float
        Butterfly volatility
    data_scale_factor : float
        Scale factor for volatility data
    data_strike_scale_factor : float, optional
        Scale factor for strike/delta
        
    Returns
    -------
    Tuple[List[FxVolData], FxVolData, FxVolData, FxVolData]
        List of all options, ATM option, call option, put option
    """
    spot_date = atm_vanilla.data_date
    fx_spot = atm_vanilla.fx_spot
    
    # Scale volatilities
    risk_reversal_vol_scaled = risk_reversal_vol / data_scale_factor
    butterfly_vol_scaled = butterfly_vol / data_scale_factor
    delta_strike_scaled = float(delta_strike) * data_strike_scale_factor
    atm_vol = atm_vanilla.vol / data_scale_factor
    maturity = atm_vanilla.expiry_maturity
    
    # Get discount factors
    domestic_df = domestic_df_curve.Bond(maturity)
    foreign_df = foreign_df_curve.Bond(maturity)
    
    # Calculate forward and ATM strike
    forward = fx_spot * foreign_df / domestic_df
    atm_strike = forward
    
    # Calculate wing volatilities
    sigma_call = atm_vol + butterfly_vol_scaled + 0.5 * risk_reversal_vol_scaled
    sigma_put = atm_vol + butterfly_vol_scaled - 0.5 * risk_reversal_vol_scaled
    
    # Calculate rates for delta calculations
    rd = -np.log(domestic_df) / maturity
    rf = -np.log(foreign_df) / maturity
    
    # Calculate strikes from delta
    strike_call = calculate_strike_from_delta_analytical(
        fx_spot, rd, rf, delta_strike_scaled, sigma_call, maturity, True
    )
    strike_put = calculate_strike_from_delta_analytical(
        fx_spot, rd, rf, delta_strike_scaled, sigma_put, maturity, False
    )
    
    # Validate strike ordering
    if strike_call <= atm_strike or atm_strike <= strike_put:
        print(f"Warning: Strike ordering issue. Maturity:{maturity} Put: {strike_put:.6f}, ATM: {atm_strike:.6f}, Call: {strike_call:.6f}")
    
    # Create FxVolData objects
    atm = FxVolData(
        spot_date, fx_spot, atm_vanilla.expiry, atm_vol, atm_strike, True,
        atm_vanilla.source, vol_multiplier=1.0
    )
    atm.set_fx_fwd(atm_strike)
    call = FxVolData(
        spot_date, fx_spot, atm_vanilla.expiry, sigma_call, strike_call, True,
        atm_vanilla.source, vol_multiplier=1.0
    )
    call.set_fx_fwd(atm_strike)
    
    put = FxVolData(
        spot_date, fx_spot, atm_vanilla.expiry, sigma_put, strike_put, False,
        atm_vanilla.source, vol_multiplier=1.0
    )
    put.set_fx_fwd(atm_strike)
    
    vanilla_list = [atm, call, put]
    
    return vanilla_list, atm, call, put


def calculate_strike_from_delta_analytical(
    fx_spot: float,
    rd: float,
    rf: float,
    target_delta: float,
    vol: float,
    maturity: float,
    is_call: bool
) -> float:
    """
    Calculate strike from delta analytically.
    
    Parameters
    ----------
    fx_spot : float
        FX spot rate
    rd : float
        Domestic interest rate
    rf : float
        Foreign interest rate
    target_delta : float
        Target delta
    vol : float
        Volatility
    maturity : float
        Time to maturity
    is_call : bool
        True for call, False for put
        
    Returns
    -------
    float
        Strike corresponding to the target delta
    """
    from scipy.stats import norm
    
    # Adjust delta for foreign interest rate
    discounted_delta = target_delta / np.exp(-rf * maturity)
    
    # Find d1 corresponding to the delta
    if is_call:
        d1_target = norm.ppf(discounted_delta)
    else:
        d1_target = -norm.ppf(abs(discounted_delta))
    
    # Calculate strike from d1
    drift = (rd - rf + 0.5 * vol * vol) * maturity
    strike = fx_spot * np.exp(-(d1_target * vol * np.sqrt(maturity) - drift))
    
    return strike


def read_fx_vol_raw_data(
    current_date: str,
    fx_spot: float,
    complete_file_name: str,
    source: str = "DummyDataSource"
) -> Optional[pd.DataFrame]:
    """
    Read raw FX volatility data from CSV file.
    
    Parameters
    ----------
    current_date : str
        Date in YYYYMMDD format
    fx_spot : float
        FX spot rate
    complete_file_name : str
        Path to CSV file
    source : str, optional
        Data source identifier
        
    Returns
    -------
    Optional[pd.DataFrame]
        FX volatility data for the specified date, or None if not found
    """
    try:
        all_data = pd.read_csv(complete_file_name)
        all_data['QuotationDate'] = pd.to_datetime(all_data['Dates'].astype(str), format="%Y%m%d")
        all_data = all_data.dropna()
        
        current_date_int = int(current_date)
        current_date_data = all_data[all_data['Dates'] == current_date_int].copy()
        
        if len(current_date_data) == 0:
            print(f'Warning: No data for the date {current_date}')
            return None
        
        # Create RawFxVolData objects
        current_date_data['Object'] = current_date_data.apply(
            lambda row: RawFxVolData(
                row['QuotationDate'],
                fx_spot,
                row['Tenor'],
                row['Data'],
                row['DeltaStrike'],
                get_fx_vol_data_type(row['Type']),
                source
            ), axis=1
        )
        
        # Add expiry maturity
        current_date_data['ExpiryMat'] = current_date_data['Tenor'].apply(convert_tenor_to_years)
        
        # Sort by date and maturity
        current_date_data = current_date_data.sort_values(
            by=['QuotationDate', 'ExpiryMat'],
            ascending=[True, True]
        )
        
        return current_date_data
        
    except Exception as e:
        print(f"Error reading FX vol data: {e}")
        return None
