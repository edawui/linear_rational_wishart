"""
Calibration reporting utilities for LRW Jump models.

This module provides comprehensive reporting functionality for
calibration results, including error analysis and model parameters.
"""

from typing import Dict, Optional, List, Any
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import json

from ..models.interest_rate.lrw_model import LRWModel
from ..data import MarketData


class CalibrationReporter:
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
        daily_data: MarketData.DailyData,
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
        daily_data: MarketData.DailyData,
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
            tenor = ois_row["TimeToMat"]
            
            # Model values
            model_price = model.Bond(tenor)
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
        daily_data: MarketData.DailyData,
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
            tenor = ibor_row["TimeToMat"]
            
            # Model spread calculation would be implemented here
            # This is a placeholder structure
            spread_data.append({
                'date': daily_data.quotation_date,
                'tenor': tenor,
                'market_rate': rate_data.rate / 100.0,
                'market_aggregate_a': getattr(rate_data, 'market_aggregate_a', 0),
                'model_aggregate_a': getattr(rate_data, 'model_aggregate_a', 0),
                'aggregate_a_error': getattr(rate_data, 'market_aggregate_a', 0) - 
                                   getattr(rate_data, 'model_aggregate_a', 0)
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
        daily_data: MarketData.DailyData,
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
                'expiry': swaption.expiry_mat,
                'tenor': swaption.swap_tenor_mat,
                'strike_offset': swaption.strike_offset,
                'market_vol': swaption.market_vol,
                'market_price': swaption.market_price,
                'model_vol': getattr(swaption, 'model_vol', 0),
                'model_price': getattr(swaption, 'model_price', 0),
                'vol_error': swaption.market_vol - getattr(swaption, 'model_vol', 0),
                'price_error': swaption.market_price - getattr(swaption, 'model_price', 0),
                'vol_error_bps': (swaption.market_vol - getattr(swaption, 'model_vol', 0)) * 10000,
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
        daily_data: MarketData.DailyData,
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
                'sigma': model.sigma.tolist()
            },
            'calibration_errors': calibration_results,
            'summary_statistics': self.summary_stats,
            'model_validation': {
                'gindikin_satisfied': self._check_gindikin_condition(model)
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
            
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
