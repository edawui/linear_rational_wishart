"""
Reporting utilities for model outputs and sensitivity analysis.

This module provides functions for formatting and outputting results
from pricing and sensitivity calculations.
"""

from typing import Dict, Any, Optional, Union, List
import json
import pandas as pd
from pathlib import Path


def print_pretty(results: Dict[str, Any], indent: int = 2) -> None:
    """
    Pretty print a dictionary of results.
    
    Parameters
    ----------
    results : Dict[str, Any]
        Results dictionary to print
    indent : int, default=2
        Indentation level
    """
    for key, value in results.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            print_pretty(value, indent + 2)
        else:
            print(f"{' ' * indent}{key}: {value}")


def print_pretty_to_file(
    results: Dict[str, Any],
    filename: Union[str, Path],
    mode: str = 'a'
) -> None:
    """
    Write results dictionary to file in readable format.
    
    Parameters
    ----------
    results : Dict[str, Any]
        Results to write
    filename : Union[str, Path]
        Output file path
    mode : str, default='a'
        File open mode
    """
    with open(filename, mode) as f:
        for key, value in results.items():
            f.write(f"\n{key},{value}")


def results_to_dataframe(results: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert results dictionary to pandas DataFrame.
    
    Parameters
    ----------
    results : Dict[str, Any]
        Results dictionary
        
    Returns
    -------
    pd.DataFrame
        Results as DataFrame
    """
    # Flatten nested dictionaries
    flat_results = {}
    
    def flatten(d: Dict, parent_key: str = ''):
        for k, v in d.items():
            new_key = f"{parent_key}_{k}" if parent_key else k
            if isinstance(v, dict):
                flatten(v, new_key)
            else:
                flat_results[new_key] = v
                
    flatten(results)
    
    # Convert to DataFrame
    df = pd.DataFrame([flat_results])
    return df


def export_results(
    results: Dict[str, Any],
    base_filename: str,
    formats: List[str] = ['csv', 'json']
) -> Dict[str, Path]:
    """
    Export results in multiple formats.
    
    Parameters
    ----------
    results : Dict[str, Any]
        Results to export
    base_filename : str
        Base filename without extension
    formats : List[str], default=['csv', 'json']
        Export formats
        
    Returns
    -------
    Dict[str, Path]
        Paths to exported files
    """
    exported_files = {}
    
    if 'csv' in formats:
        csv_path = Path(f"{base_filename}.csv")
        df = results_to_dataframe(results)
        df.to_csv(csv_path, index=False)
        exported_files['csv'] = csv_path
        
    if 'json' in formats:
        json_path = Path(f"{base_filename}.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        exported_files['json'] = json_path
        
    if 'txt' in formats:
        txt_path = Path(f"{base_filename}.txt")
        print_pretty_to_file(results, txt_path, mode='w')
        exported_files['txt'] = txt_path
        
    return exported_files


class SensitivityReporter:
    """
    Specialized reporter for sensitivity analysis results.
    """
    
    def __init__(self):
        self.results = {}
        
    def add_results(
        self,
        category: str,
        results: Dict[str, Any]
    ) -> None:
        """
        Add results to a category.
        
        Parameters
        ----------
        category : str
            Category name
        results : Dict[str, Any]
            Results to add
        """
        if category not in self.results:
            self.results[category] = {}
        self.results[category].update(results)
        
    def generate_report(self) -> str:
        """
        Generate a formatted text report.
        
        Returns
        -------
        str
            Formatted report
        """
        report_lines = ["Sensitivity Analysis Report", "=" * 50]
        
        for category, cat_results in self.results.items():
            report_lines.append(f"\n{category}")
            report_lines.append("-" * len(category))
            
            for key, value in cat_results.items():
                report_lines.append(f"  {key}: {value}")
                
        return "\n".join(report_lines)
    
    def to_latex_table(self, category: Optional[str] = None) -> str:
        """
        Convert results to LaTeX table format.
        
        Parameters
        ----------
        category : Optional[str]
            Specific category to export, or None for all
            
        Returns
        -------
        str
            LaTeX table code
        """
        if category:
            data = self.results.get(category, {})
        else:
            data = {}
            for cat_results in self.results.values():
                data.update(cat_results)
        
        if not data:
            return ""
        
        latex_lines = [
            "\\begin{table}[h]",
            "\\centering",
            "\\begin{tabular}{ll}",
            "\\hline",
            "Parameter & Value \\\\",
            "\\hline"
        ]
        
        for key, value in data.items():
            # Format value
            if isinstance(value, float):
                formatted_value = f"{value:.6f}"
            else:
                formatted_value = str(value)
            
            # Escape underscores in key
            key_latex = key.replace("_", "\\_")
            
            latex_lines.append(f"{key_latex} & {formatted_value} \\\\")
        
        latex_lines.extend([
            "\\hline",
            "\\end{tabular}",
            "\\caption{Sensitivity Analysis Results}",
            "\\end{table}"
        ])
        
        return "\n".join(latex_lines)
    
    def to_markdown_table(self, category: Optional[str] = None) -> str:
        """
        Convert results to Markdown table format.
        
        Parameters
        ----------
        category : Optional[str]
            Specific category to export, or None for all
            
        Returns
        -------
        str
            Markdown table
        """
        if category:
            data = self.results.get(category, {})
        else:
            data = {}
            for cat_results in self.results.values():
                data.update(cat_results)
        
        if not data:
            return ""
        
        lines = ["| Parameter | Value |", "|-----------|-------|"]
        
        for key, value in data.items():
            if isinstance(value, float):
                formatted_value = f"{value:.6f}"
            else:
                formatted_value = str(value)
            lines.append(f"| {key} | {formatted_value} |")
        
        return "\n".join(lines)
