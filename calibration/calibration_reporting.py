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

from ..models.fx.lrw_fx import LRWFxModel
from ..data.data_fx_market_data import CurrencyPairDailyData
from .fx.base import CalibrationConfig


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