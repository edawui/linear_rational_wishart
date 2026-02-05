"""
Calibration reporting utilities.

This module provides utilities for generating calibration reports,
including summaries, error analysis, and model parameter reporting.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from datetime import datetime
import json
import csv
from pathlib import Path

from ..models.interest_rate.lrw_model import LRWModel
from ..models.fx.lrw_fx import LRWFxModel
from ..data.data_market_data import  DailyData
from ..data.data_fx_market_data import CurrencyPairDailyData
from .fx.base import CalibrationConfig
from .fx.base import CalibrationResult


class CalibrationReporter:
    """Generates calibration reports in various formats."""
    
    def __init__(
        self,
        model: LRWFxModel,
        daily_data: CurrencyPairDailyData,
        config: CalibrationConfig
    ):
        """
        Initialize calibration reporter.
        
        Parameters
        ----------
        model : LRWFxModel
            Calibrated model
        daily_data : MarketData.CurrencyPairDailyData
            Market data used for calibration
        config : CalibrationConfig
            Calibration configuration
        """
        self.model = model
        self.daily_data = daily_data
        self.config = config
        self.report_timestamp = datetime.now()
    
    def generate_summary_report(
        self,
        result: CalibrationResult,
        include_parameters: bool = True,
        include_errors: bool = True,
        include_config: bool = True
    ) -> str:
        """
        Generate comprehensive summary report.
        
        Parameters
        ----------
        result : CalibrationResult
            Calibration results
        include_parameters : bool
            Include parameter details
        include_errors : bool
            Include error analysis
        include_config : bool
            Include configuration details
            
        Returns
        -------
        str
            Formatted report
        """
        lines = []
        
        # Header
        lines.extend([
            "=" * 80,
            "FX MODEL CALIBRATION REPORT",
            "=" * 80,
            f"Report generated: {self.report_timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Currency pair: {self.daily_data.currency_pair}",
            f"Quotation date: {self.daily_data.quotation_date}",
            ""
        ])
        
        # Calibration results
        lines.extend([
            "CALIBRATION RESULTS",
            "-" * 40,
            f"Success: {result.success}",
            f"Final RMSE: {result.final_error:.8f}",
            f"Improvement: {(1 - result.final_error/result.initial_error)*100:.2f}%",
            f"Iterations: {result.num_iterations}",
            f"Function evaluations: {result.num_function_evaluations}",
            f"Optimization time: {result.optimization_time:.2f} seconds",
            ""
        ])
        
        # Error breakdown
        if include_errors and any([result.rmse_ois_price, result.rmse_ois_yield, 
                                   result.rmse_option_price, result.rmse_option_vol]):
            lines.extend([
                "ERROR BREAKDOWN",
                "-" * 40,
                f"OIS Price RMSE: {result.rmse_ois_price:.4f} bps" if result.rmse_ois_price else "OIS Price RMSE: N/A",
                f"OIS Yield RMSE: {result.rmse_ois_yield:.4f} bps" if result.rmse_ois_yield else "OIS Yield RMSE: N/A",
                f"Option Price RMSE: {result.rmse_option_price:.6f}" if result.rmse_option_price else "Option Price RMSE: N/A",
                f"Option Vol RMSE: {result.rmse_option_vol:.4f} bps" if result.rmse_option_vol else "Option Vol RMSE: N/A",
                ""
            ])
        
        # Model parameters
        if include_parameters:
            lines.extend([
                "MODEL PARAMETERS",
                "-" * 40,
                self._format_model_parameters(),
                ""
            ])
        
        # Configuration
        if include_config:
            lines.extend([
                "CALIBRATION CONFIGURATION",
                "-" * 40,
                f"Optimization method: {self.config.optimization_method.value}",
                f"Pricing method: {self.config.pricing_method.value}",
                f"Use ATM only: {self.config.use_atm_only}",
                f"Option maturity range: {self.config.min_maturity_option:.1f}Y - {self.config.max_maturity_option:.1f}Y",
                f"Calibrate on vol: {self.config.calibrate_on_vol}",
                f"Monte Carlo paths: {self.config.mc_paths}",
                f"Use JAX: {self.config.use_jax}",
                ""
            ])
        
        # Footer
        lines.extend([
            "=" * 80,
            "END OF REPORT",
            "=" * 80
        ])
        
        return "\n".join(lines)
    
    def _format_model_parameters(self) -> str:
        """Format model parameters for display."""
        lines = []
        
        # Alpha parameters
        lines.extend([
            f"Alpha (domestic): {self.model._alpha_i:.6f}",
            f"Alpha (foreign): {self.model._alpha_j:.6f}",
            ""
        ])
        
        # Matrix parameters
        matrices = {
            'X0': self.model._x0,
            'Omega': self.model._omega,
            'M': self.model._m,
            'Sigma': self.model._sigma
        }
        
        for name, matrix in matrices.items():
            lines.append(f"{name} matrix:")
            for i in range(matrix.shape[0]):
                row_str = "  " + " ".join(f"{matrix[i,j]:10.6f}" for j in range(matrix.shape[1]))
                lines.append(row_str)
            
            # Add correlation if 2x2 matrix
            if matrix.shape == (2, 2) and matrix[0,0] > 0 and matrix[1,1] > 0:
                corr = matrix[0,1] / np.sqrt(matrix[0,0] * matrix[1,1])
                lines.append(f"  Correlation: {corr:.6f}")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def generate_csv_report(
        self,
        result: CalibrationResult,
        filepath: str
    ) -> None:
        """
        Generate CSV report for calibration results.
        
        Parameters
        ----------
        result : CalibrationResult
            Calibration results
        filepath : str
            Output file path
        """
        data = {
            'date': self.daily_data.quotation_date,
            'currency_pair': self.daily_data.currency_pair,
            'success': result.success,
            'final_error': result.final_error,
            'initial_error': result.initial_error,
            'improvement_pct': (1 - result.final_error/result.initial_error) * 100,
            'iterations': result.num_iterations,
            'function_evals': result.num_function_evaluations,
            'time_seconds': result.optimization_time,
            'rmse_ois_price': result.rmse_ois_price or np.nan,
            'rmse_ois_yield': result.rmse_ois_yield or np.nan,
            'rmse_option_price': result.rmse_option_price or np.nan,
            'rmse_option_vol': result.rmse_option_vol or np.nan,
            'alpha_i': self.model._alpha_i,
            'alpha_j': self.model._alpha_j,
            'x0_00': self.model._x0[0,0],
            'x0_11': self.model._x0[1,1],
            'x0_01': self.model._x0[0,1],
            'omega_00': self.model._omega[0,0],
            'omega_11': self.model._omega[1,1],
            'omega_01': self.model._omega[0,1],
            'm_00': self.model._m[0,0],
            'm_11': self.model._m[1,1],
            'sigma_00': self.model._sigma[0,0],
            'sigma_11': self.model._sigma[1,1],
            'sigma_01': self.model._sigma[0,1]
        }
        
        # Check if file exists to write header
        file_exists = Path(filepath).exists()
        
        with open(filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(data)
    
    def generate_json_report(
        self,
        result: CalibrationResult,
        filepath: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate JSON report for calibration results.
        
        Parameters
        ----------
        result : CalibrationResult
            Calibration results
        filepath : str, optional
            Output file path (if None, returns dict only)
            
        Returns
        -------
        Dict[str, Any]
            Report data as dictionary
        """
        report = {
            'metadata': {
                'timestamp': self.report_timestamp.isoformat(),
                'currency_pair': self.daily_data.currency_pair,
                'quotation_date': str(self.daily_data.quotation_date),
                'fx_spot': float(self.daily_data.fx_spot)
            },
            'configuration': self.config.to_dict(),
            'results': {
                'success': result.success,
                'final_error': float(result.final_error),
                'initial_error': float(result.initial_error),
                'improvement_percent': float((1 - result.final_error/result.initial_error) * 100),
                'iterations': result.num_iterations,
                'function_evaluations': result.num_function_evaluations,
                'optimization_time_seconds': result.optimization_time,
                'optimizer_message': result.optimizer_message
            },
            'errors': {
                'ois_price_rmse_bps': float(result.rmse_ois_price) if result.rmse_ois_price else None,
                'ois_yield_rmse_bps': float(result.rmse_ois_yield) if result.rmse_ois_yield else None,
                'option_price_rmse': float(result.rmse_option_price) if result.rmse_option_price else None,
                'option_vol_rmse_bps': float(result.rmse_option_vol) if result.rmse_option_vol else None
            },
            'model_parameters': {
                'alpha': {
                    'domestic': float(self.model._alpha_i),
                    'foreign': float(self.model._alpha_j)
                },
                'x0': self._matrix_to_dict(self.model._x0),
                'omega': self._matrix_to_dict(self.model._omega),
                'm': self._matrix_to_dict(self.model._m),
                'sigma': self._matrix_to_dict(self.model._sigma)
            },
            'parameter_changes': self._calculate_parameter_changes(result)
        }
        
        if filepath:
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
        
        return report
    
    def _matrix_to_dict(self, matrix: np.ndarray) -> Dict[str, Any]:
        """Convert matrix to dictionary representation."""
        result = {
            'values': matrix.tolist(),
            'shape': list(matrix.shape)
        }
        
        # Add correlation for 2x2 matrices
        if matrix.shape == (2, 2) and matrix[0,0] > 0 and matrix[1,1] > 0:
            corr = matrix[0,1] / np.sqrt(matrix[0,0] * matrix[1,1])
            result['correlation'] = float(corr)
        
        return result
    
    def _calculate_parameter_changes(self, result: CalibrationResult) -> Dict[str, float]:
        """Calculate parameter changes from calibration."""
        changes = {}
        
        if len(result.initial_parameters) == len(result.final_parameters):
            for i, name in enumerate(result.parameter_names):
                initial = result.initial_parameters[i]
                final = result.final_parameters[i]
                
                if initial != 0:
                    change_pct = (final - initial) / abs(initial) * 100
                else:
                    change_pct = float('inf') if final != 0 else 0.0
                
                changes[name] = {
                    'initial': float(initial),
                    'final': float(final),
                    'change_percent': float(change_pct)
                }
        
        return changes
    
    def generate_market_data_summary(self) -> pd.DataFrame:
        """
        Generate summary of market data used in calibration.
        
        Returns
        -------
        pd.DataFrame
            Market data summary
        """
        summaries = []
        
        # OIS data summary
        for currency, data in [('domestic', self.daily_data.domestic_currency_daily_data),
                              ('foreign', self.daily_data.foreign_currency_daily_data)]:
            
            ois_data = data.ois_rate_data
            
            summary = {
                'instrument_type': 'OIS',
                'currency': currency,
                'count': len(ois_data),
                'min_tenor': ois_data['TimeToMat'].min(),
                'max_tenor': ois_data['TimeToMat'].max(),
                'avg_rate': ois_data.apply(lambda x: x['Object'].market_zc_rate, axis=1).mean()
            }
            
            summaries.append(summary)
        
        # FX option data summary
        fx_options = self.daily_data.fx_vol_data
        
        if fx_options:
            maturities = [opt.expiry_mat for opt in fx_options]
            strikes = [opt.strike for opt in fx_options]
            vols = [opt.vol for opt in fx_options]
            
            summary = {
                'instrument_type': 'FX_Option',
                'currency': 'cross',
                'count': len(fx_options),
                'min_tenor': min(maturities),
                'max_tenor': max(maturities),
                'avg_vol': np.mean(vols),
                'min_strike': min(strikes),
                'max_strike': max(strikes),
                'spot': self.daily_data.fx_spot
            }
            
            summaries.append(summary)
        
        return pd.DataFrame(summaries)
    
    def save_all_reports(
        self,
        result: CalibrationResult,
        output_dir: str,
        base_filename: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Save all report types to directory.
        
        Parameters
        ----------
        result : CalibrationResult
            Calibration results
        output_dir : str
            Output directory
        base_filename : str, optional
            Base filename (default: uses timestamp)
            
        Returns
        -------
        Dict[str, str]
            Dictionary of report types and file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if base_filename is None:
            base_filename = f"calibration_{self.daily_data.currency_pair}_{self.report_timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        files_created = {}
        
        # Text report
        text_path = output_path / f"{base_filename}.txt"
        with open(text_path, 'w') as f:
            f.write(self.generate_summary_report(result))
        files_created['text'] = str(text_path)
        
        # CSV report
        csv_path = output_path / f"{base_filename}.csv"
        self.generate_csv_report(result, str(csv_path))
        files_created['csv'] = str(csv_path)
        
        # JSON report
        json_path = output_path / f"{base_filename}.json"
        self.generate_json_report(result, str(json_path))
        files_created['json'] = str(json_path)
        
        # Market data summary
        market_path = output_path / f"{base_filename}_market_data.csv"
        self.generate_market_data_summary().to_csv(market_path, index=False)
        files_created['market_data'] = str(market_path)
        
        return files_created


    
class IRCalibrationReporter:
    """
    Generates comprehensive calibration reports for LRW Jump models.
    """
    
    def __init__(self):
        """Initialize the reporter."""
        self.report_data = {}
        self.summary_stats = {}
        
    def generate_full_report(
        self,
        model: LRWModel,
        daily_data: DailyData,
        calibration_results: Dict[str, Any],
        output_dir: str = ".",
        report_name: Optional[str] = None
    ):
        """
        Generate comprehensive calibration report.
        
        Parameters
        ----------
        model : LRWModel
            Calibrated model
        daily_data : MarketData.DailyData
            Market data used for calibration
        calibration_results : Dict[str, Any]
            Calibration results
        output_dir : str, default="."
            Output directory
        report_name : str, optional
            Report name prefix
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if report_name is None:
            report_name = f"lrw_jump_calibration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        # Generate individual reports
        self.write_ois_report(model, daily_data, output_path / f"{report_name}_ois.csv")
        self.write_spread_report(model, daily_data, output_path / f"{report_name}_spread.csv")
        self.write_swaption_report(model, daily_data, output_path / f"{report_name}_swaption.csv")
        self.write_model_report(model, calibration_results, output_path / f"{report_name}_model.txt")
        
        # Generate summary report
        self.write_summary_report(
            model, daily_data, calibration_results,
            output_path / f"{report_name}_summary.json"
        )
        
        print(f"Calibration reports generated in: {output_path}")
        
    def write_ois_report(
        self,
        model: LRWModel,
        daily_data: DailyData,
        filename: Path
    ):
        """
        Write OIS calibration report.
        
        Parameters
        ----------
        model : LRWModel
            Calibrated model
        daily_data : MarketData.DailyData
            Market data
        filename : Path
            Output file path
        """
        ois_data = []
        
        for _, ois_row in daily_data.ois_rate_data.iterrows():
            rate_data = ois_row["Object"]
            tenor = rate_data.time_to_maturity ##ois_row["TimeToMat"]
            
            # Model values
            model_price = model.bond(tenor)
            model_rate = -np.log(model_price) / tenor if tenor > 0 else 0
            
            ois_data.append({
                'date': daily_data.quotation_date,
                'tenor': tenor,
                'market_rate': rate_data.market_zc_rate,
                'market_price': rate_data.market_zc_price,
                'model_rate': model_rate,
                'model_price': model_price,
                'rate_error': rate_data.market_zc_rate - model_rate,
                'price_error': rate_data.market_zc_price - model_price,
                'rate_error_bps': (rate_data.market_zc_rate - model_rate) * 10000,
                'price_error_bps': (rate_data.market_zc_price - model_price) * 10000
            })
            
        df = pd.DataFrame(ois_data)
        df.to_csv(filename, index=False)
        
        # Store summary statistics
        self.summary_stats['ois'] = {
            'rmse_rate': np.sqrt(np.mean(df['rate_error']**2)),
            'rmse_price': np.sqrt(np.mean(df['price_error']**2)),
            'max_rate_error': df['rate_error'].abs().max(),
            'max_price_error': df['price_error'].abs().max(),
            'mean_rate_error': df['rate_error'].mean()
        }
        
    def write_spread_report(
        self,
        model: LRWModel,
        daily_data: DailyData,
        filename: Path
    ):
        """
        Write spread calibration report.
        
        Parameters
        ----------
        model : LRWModel
            Calibrated model
        daily_data : MarketData.DailyData
            Market data
        filename : Path
            Output file path
        """
        spread_data = []
        
        for _, ibor_row in daily_data.euribor_rate_data.iterrows():
            rate_data = ibor_row["Object"]
            tenor = rate_data.time_to_maturity # ibor_row["TimeToMat"]
            
            # Model spread calculation would be implemented here
            # This is a placeholder structure
            spread_data.append({
                'date': daily_data.quotation_date,
                'tenor': tenor,
                'market_rate': rate_data.rate,
                'market_aggregate_a': getattr(rate_data, 'market_aggregate_a', 0),
                'model_aggregate_a': getattr(rate_data, 'model_aggregate_a', 0)
                #, 'aggregate_a_error': getattr(rate_data, 'market_aggregate_a', 0) - 
                #                    getattr(rate_data, 'model_aggregate_a', 0)
            })
            
        df = pd.DataFrame(spread_data)
        df.to_csv(filename, index=False)
        
        # Store summary statistics
        if 'aggregate_a_error' in df.columns:
            self.summary_stats['spread'] = {
                'rmse_aggregate_a': np.sqrt(np.mean(df['aggregate_a_error']**2)),
                'max_aggregate_a_error': df['aggregate_a_error'].abs().max()
            }
            
    def write_swaption_report(
        self,
        model: LRWModel,
        daily_data: DailyData,
        filename: Path
    ):
        """
        Write swaption calibration report.
        
        Parameters
        ----------
        model : LRWModel
            Calibrated model
        daily_data : MarketData.DailyData
            Market data
        filename : Path
            Output file path
        """
        swaption_data = []
        
        for _, swaption_row in daily_data.swaption_data_cube.iterrows():
            swaption = swaption_row["Object"]
            
            swaption_data.append({
                'date': daily_data.quotation_date,
                'expiry': swaption.expiry_maturity,
                'tenor': swaption.swap_tenor_maturity,
                'strike_offset': swaption.strike_offset,
                'market_vol': swaption.vol,
                'market_price': swaption.market_price,
                'model_vol': getattr(swaption, 'model_vol', 0),
                'model_price': getattr(swaption, 'model_price', 0),
                'vol_error': swaption.vol - getattr(swaption, 'model_vol', 0),
                'price_error': swaption.market_price - getattr(swaption, 'model_price', 0),
                'vol_error_bps': (swaption.vol - getattr(swaption, 'model_vol', 0)) * 10000,
                'price_error_bps': (swaption.market_price - getattr(swaption, 'model_price', 0)) * 10000
            })
            
        df = pd.DataFrame(swaption_data)
        df.to_csv(filename, index=False)
        
        # Store summary statistics
        self.summary_stats['swaption'] = {
            'rmse_vol': np.sqrt(np.mean(df['vol_error']**2)),
            'rmse_price': np.sqrt(np.mean(df['price_error']**2)),
            'max_vol_error': df['vol_error'].abs().max(),
            'max_price_error': df['price_error'].abs().max(),
            'mean_vol_error': df['vol_error'].mean()
        }
        
    def write_model_report(
        self,
        model: LRWModel,
        calibration_results: Dict[str, Any],
        filename: Path
    ):
        """
        Write model parameters report.
        
        Parameters
        ----------
        model : LRWModel
            Calibrated model
        calibration_results : Dict[str, Any]
            Calibration results
        filename : Path
            Output file path
        """
        with open(filename, 'w') as f:
            f.write("LRW Jump Model Calibration Report\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Model parameters
            f.write("Model Parameters:\n")
            f.write("-" * 30 + "\n")
            f.write(f"n (dimension): {model.n}\n")
            f.write(f"alpha: {model.alpha:.6f}\n\n")
            
            f.write("x0 matrix:\n")
            f.write(self._format_matrix(model.x0))
            f.write("\n")
            
            f.write("omega matrix:\n")
            f.write(self._format_matrix(model.omega))
            f.write("\n")
            
            f.write("m matrix:\n")
            f.write(self._format_matrix(model.m))
            f.write("\n")
            
            f.write("sigma matrix:\n")
            f.write(self._format_matrix(model.sigma))
            f.write("\n")

            f.write("u1 matrix:\n")
            f.write(self._format_matrix(model.u1))
            f.write("\n")

            f.write("u2 matrix:\n")
            f.write(self._format_matrix(model.u2))
            f.write("\n")
            
            # Calibration results
            f.write("\nCalibration Results:\n")
            f.write("-" * 30 + "\n")
            
            if 'ois_error' in calibration_results:
                f.write(f"OIS RMSE: {calibration_results['ois_error']:.6f}\n")
            if 'spread_error' in calibration_results:
                f.write(f"Spread RMSE: {calibration_results['spread_error']:.6f}\n")
            if 'swaption_error' in calibration_results:
                f.write(f"Swaption RMSE: {calibration_results['swaption_error']:.6f}\n")
                
            # Additional information
            f.write("\nModel Validation:\n")
            f.write("-" * 30 + "\n")
            gindikin = self._check_gindikin_condition(model)
            f.write(f"Gindikin condition satisfied: {gindikin}\n")
            
    def write_summary_report(
        self,
        model: LRWModel,
        daily_data: DailyData,
        calibration_results: Dict[str, Any],
        filename: Path
    ):
        """
        Write summary report in JSON format.
        
        Parameters
        ----------
        model : LRWModel
            Calibrated model
        daily_data : MarketData.DailyData
            Market data
        calibration_results : Dict[str, Any]
            Calibration results
        filename : Path
            Output file path
        """
        summary = {
            'report_date': datetime.now().isoformat(),
            'market_date': str(daily_data.quotation_date),
            'model_parameters': {
                'n': model.n,
                'alpha': float(model.alpha),
                'x0': model.x0.tolist(),
                'omega': model.omega.tolist(),
                'm': model.m.tolist(),
                'sigma': model.sigma.tolist(),
                'u1': model.u1.tolist(),
                'u2': model.u2.tolist()
            },
            'calibration_errors': calibration_results,
            'summary_statistics': self.summary_stats,
            'model_validation': {
                'gindikin_satisfied': self._check_gindikin_condition(model)
            }
        }
         
        # print(f"filename:={filename}")
        # print(f"summary:={summary}")

        def convert_to_serializable(obj):
            """Recursively convert arrays to Python native types."""
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(obj, 'tolist'):  # Catches JAX arrays, PyTorch tensors, etc.
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif hasattr(obj, 'item'):  # Scalar arrays
                return obj.item()
            else:
                return obj

        # Use it before json.dump
        summary_serializable = convert_to_serializable(summary)
        # json.dump(summary_serializable, f, indent=2)

        with open(filename, 'w') as f:
            # json.dump(summary, f, indent=2)
            json.dump(summary_serializable, f, indent=2)
            
    def create_latex_report(
        self,
        model: LRWModel,
        calibration_results: Dict[str, Any],
        filename: Path
    ):
        """
        Create LaTeX-formatted calibration report.
        
        Parameters
        ----------
        model : LRWModel
            Calibrated model
        calibration_results : Dict[str, Any]
            Calibration results
        filename : Path
            Output file path
        """
        with open(filename, 'w') as f:
            f.write("\\documentclass{article}\n")
            f.write("\\usepackage{amsmath}\n")
            f.write("\\usepackage{booktabs}\n")
            f.write("\\begin{document}\n\n")
            
            f.write("\\section*{LRW Jump Model Calibration Report}\n\n")
            
            # Model parameters
            f.write("\\subsection*{Model Parameters}\n")
            f.write("\\begin{align}\n")
            f.write(f"\\alpha &= {model.alpha:.4f} \\\\\n")
            f.write("x_0 &= " + self._format_matrix_latex(model.x0) + " \\\\\n")
            f.write("\\omega &= " + self._format_matrix_latex(model.omega) + " \\\\\n")
            f.write("m &= " + self._format_matrix_latex(model.m) + " \\\\\n")
            f.write("\\sigma &= " + self._format_matrix_latex(model.sigma) + "\n")
            f.write("\\end{align}\n\n")
            
            # Calibration results table
            f.write("\\subsection*{Calibration Results}\n")
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\begin{tabular}{lr}\n")
            f.write("\\toprule\n")
            f.write("Metric & Value \\\\\n")
            f.write("\\midrule\n")
            
            if 'ois_error' in calibration_results:
                f.write(f"OIS RMSE & {calibration_results['ois_error']:.4f} \\\\\n")
            if 'spread_error' in calibration_results:
                f.write(f"Spread RMSE & {calibration_results['spread_error']:.4f} \\\\\n")
            if 'swaption_error' in calibration_results:
                f.write(f"Swaption RMSE & {calibration_results['swaption_error']:.4f} \\\\\n")
                
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n\n")
            
            f.write("\\end{document}\n")
            
    def _format_matrix(self, matrix: np.ndarray) -> str:
        """Format matrix for text output."""
        lines = []
        for row in matrix:
            formatted_row = "  ".join(f"{val:10.6f}" for val in row)
            lines.append(f"  [{formatted_row}]")
        return "\n".join(lines)
        
    def _format_matrix_latex(self, matrix: np.ndarray) -> str:
        """Format matrix for LaTeX output."""
        rows = []
        for row in matrix:
            formatted_row = " & ".join(f"{val:.4f}" for val in row)
            rows.append(formatted_row)
        matrix_str = " \\\\ ".join(rows)
        return f"\\begin{{pmatrix}} {matrix_str} \\end{{pmatrix}}"
        
    def _check_gindikin_condition(self, model: LRWModel) -> bool:
        """Check if model satisfies Gindikin condition."""
        n = model.n
        omega = model.omega
        sigma = model.sigma
        
        # Gindikin: omega - (n+1) * sigma @ sigma.T > 0
        temp = omega - (n + 1) * sigma @ sigma.T
        
        try:
            eigenvalues = np.linalg.eigvals(temp)
            return bool(np.all(eigenvalues > 0))
        except:
            return False
