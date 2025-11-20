import json
import numpy as np
from datetime import datetime

import json
import numpy as np
from datetime import datetime
from collections import defaultdict

def load_model_data(json_file):
    """Load model data from JSON file"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def format_matrix_latex(matrix, precision=4):
    """Format a matrix for LaTeX display"""
    if not matrix:
        return ""
    
    matrix = np.array(matrix)
    if matrix.ndim == 1:
        # Vector
        formatted_elements = [f"{x:.{precision}f}" for x in matrix]
        return "\\begin{pmatrix} " + " \\\\ ".join(formatted_elements) + " \\end{pmatrix}"
    else:
        # Matrix
        rows = []
        for row in matrix:
            formatted_row = " & ".join([f"{x:.{precision}f}" for x in row])
            rows.append(formatted_row)
        return "\\begin{pmatrix}\n" + " \\\\\n".join(rows) + "\n\\end{pmatrix}"

def organize_data_by_date_and_tenor(model_data):
    """Organize the data by calibration date and option tenor"""
    organized_data = defaultdict(dict)
    
    for entry in model_data:
        calib_date = entry['calib_date']
        tenor_min = entry['calib_option_tenor_min']
        tenor_max = entry['calib_option_tenor_max']
        
        # Create tenor key
        if tenor_min == tenor_max:
            tenor_key = f"{tenor_min:.0f}Y"
        else:
            tenor_key = f"{tenor_min:.0f}Y-{tenor_max:.0f}Y"
        
        organized_data[calib_date][tenor_key] = entry
    
    return organized_data

def get_unique_tenors(model_data):
    """Get all unique tenor configurations"""
    tenors = set()
    for entry in model_data:
        tenor_min = entry['calib_option_tenor_min']
        tenor_max = entry['calib_option_tenor_max']
        
        if tenor_min == tenor_max:
            tenors.add(f"{tenor_min:.0f}Y")
        else:
            tenors.add(f"{tenor_min:.0f}Y-{tenor_max:.0f}Y")
    
    return sorted(tenors)

def generate_latex_report(model_data):
    """Generate LaTeX code for the multi-maturity model calibration report"""
    
    # Organize data
    organized_data = organize_data_by_date_and_tenor(model_data)
    unique_tenors = get_unique_tenors(model_data)
    unique_dates = sorted(organized_data.keys())
    unique_tenors=[ '1Y', '2Y', '3Y', '4Y', '5Y', '7Y','10Y', '3Y-7Y']
    print(unique_tenors)

    latex_code = r"""
\documentclass{article}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{geometry}
\usepackage{graphicx}
\usepackage{float}
\usepackage{subcaption}
\usepackage{rotating}
\usepackage{pdflscape}
\geometry{margin=0.8in}

\title{LRW FX Model Multi-Maturity Calibration Report}
\author{Model Calibration Team}
\date{\today}

\begin{document}

\graphicspath{{charts/}}

\maketitle

\section{Executive Summary}
This report presents the calibrated parameters for the LRW (Linear-Rational-Wishart) FX Model across multiple calibration dates and option maturities. The model is calibrated on domestic and foreign curves with various option tenor configurations.

\subsection{Calibration Overview}
\begin{itemize}
\item \textbf{Model Type:} LRW FX Model
\item \textbf{Dimensions:} 2-factor model
\item \textbf{Calibration Dates:} """ + f"{len(unique_dates)} dates from {min(unique_dates)} to {max(unique_dates)}" + r"""
\item \textbf{Option Tenors:} """ + ", ".join(unique_tenors) + r"""
\item \textbf{Zero Coupon Curve Range:} 2Y to 11Y
\item \textbf{Calibration Objectives:} Price (bonds), Volatility (options)
\end{itemize}

\begin{table}[h!]
\centering
\begin{tabular}{ll}
\toprule
Parameter & Value \\
\midrule
Model Type & LRW FX Model \\
Dimensions (n) & 2 \\
Jump Component & No Jump \\
$\alpha_i   \neq  \alpha_j$ relationship & Varies by date \\
\bottomrule
\end{tabular}
\caption{General Model Settings}
\end{table}

\begin{itemize}

\item \textbf{The matrix $u_i$ for the domestic curve:}
$$u_i = \begin{pmatrix}
1.0000 & 0.0000 \\
0.0000 & 0.1250
\end{pmatrix}$$

\item \textbf{The matrix $u_j$  for foreign curve:}
$$u_j = \begin{pmatrix}
0.1000 & 0.0000 \\
0.0000 & 1.0000
\end{pmatrix}$$
\end{itemize}
\section{Calibration Quality Summary}

"""
    
    # Add calibration quality table
    latex_code += r"""
%\begin{landscape}
\begin{longtable}{lcc}
\caption{Calibration on yield data quality metrics by date (RMSE in b.p.)} \\
\toprule
Date &  RMSE OIS Price & RMSE OIS Yield   \\
\midrule
\endfirsthead
\multicolumn{3}{c}%
{{\bfseries \tablename\ \thetable{} -- continued from previous page}} \\
\toprule
Date &  RMSE OIS Price & RMSE OIS Yield  \\
\midrule
\endhead
\midrule \multicolumn{3}{r}{{Continued on next page}} \\ \midrule
\endfoot
\bottomrule
\endlastfoot
"""

    for date in unique_dates:
        # for tenor in unique_tenors:
            tenor = unique_tenors[0]
            if tenor in organized_data[date]:
                entry = organized_data[date][tenor]
                params = entry['model_parameters']
                date_formatted = datetime.strptime(date, "%Y-%m-%d").strftime("%m/%d/%Y")
                
                latex_code += f"""{date_formatted}  & {entry['rmse_ois_price']:.2f} & {entry['rmse_ois_yield']:.2f}   \\\\
"""

    latex_code += r"""
\end{longtable}
%\end{landscape}

"""
  # Add calibration quality table
    latex_code += r"""
    %%%%%%%%%%%%%%%%%%%%%%%%%% Calibration results on Vol%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%\begin{landscape}
\begin{longtable}{lccc}
\caption{Calibration on Vol data quality metrics by date and tenor (RMSE in b.p.)} \\
\toprule
Date & Tenor & RMSE Option Price & RMSE Option Vol  \\
\midrule
\endfirsthead
\multicolumn{4}{c}%
{{\bfseries \tablename\ \thetable{} -- continued from previous page}} \\
\toprule
Date & Tenor & RMSE Option Price & RMSE Option Vol \\
\midrule
\endhead
\midrule \multicolumn{4}{r}{{Continued on next page}} \\ \midrule
\endfoot
\bottomrule
\endlastfoot
"""

    for date in unique_dates:
        for tenor in unique_tenors:
            if tenor in organized_data[date]:
                entry = organized_data[date][tenor]
                params = entry['model_parameters']
                date_formatted = datetime.strptime(date, "%Y-%m-%d").strftime("%m/%d/%Y")
                
                latex_code += f"""{date_formatted} & {tenor} & {entry['rmse_option_price']:.2f} & {entry['rmse_option_vol']:.2f}  \\\\
"""

    latex_code += r"""
\end{longtable}
%\end{landscape}

"""


 # Add calibration quality table
    latex_code += r"""
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%% Model parameters on Yield curves %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%\begin{landscape}
\begin{longtable}{lccccccccc}
\caption{Model paramters calibrated on yield curve by date} \\
\toprule
Date & FX Spot & $alpha_i$ & $alpha_j$ & ${x_0}_{11}$ & ${x_0}_{22}$ & $\omega_{11}$ & $\omega_{22}$ & $m_{11}$ & $m_{22}$  \\

\midrule
\endfirsthead
\multicolumn{10}{c}%
{{\bfseries \tablename\ \thetable{} -- continued from previous page}} \\
\toprule
Date & FX Spot & $alpha_i$ & $alpha_j$ & ${x_0}_{11}$ & ${x_0}_{22}$ & $\omega_{11}$ & $\omega_{22}$ & $m_{11}$ & $m_{22}$  \\
\midrule
\endhead
\midrule \multicolumn{10}{r}{{Continued on next page}} \\ \midrule
\endfoot
\bottomrule
\endlastfoot
"""

    for date in unique_dates:
        # for tenor in unique_tenors:
            tenor = unique_tenors[0] ##Taking the fisrt as these parameters are the same for all tenors
            if tenor in organized_data[date]:
                entry = organized_data[date][tenor]
                params = entry['model_parameters']
                date_formatted = datetime.strptime(date, "%Y-%m-%d").strftime("%m/%d/%Y")
                
                latex_code += f"""{date_formatted}& {params['fx_spot']:.5f} & {params['alpha_i']:.6f} & {params['alpha_j']:.6f} & {params['x0'][0][0]:.6f} & {params['x0'][1][1]:.6f} & {params['omega'][0][0]:.6f} & {params['omega'][1][1]:.6f} & {params['m'][0][0]:.6f} & {params['m'][1][1]:.6f}   \\\\
"""

    latex_code += r"""
\end{longtable}
%\end{landscape}

\section{Model Parameters by Calibration Date and Tenor}

"""

    # Generate detailed sections for each date and tenor
    for date in unique_dates:
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        formatted_date = date_obj.strftime("%B %d, %Y")
        
        latex_code += f"""
\\subsection{{{formatted_date}}}

"""
        
        for tenor in unique_tenors:
            if tenor in organized_data[date]:
                entry = organized_data[date][tenor]
                params = entry['model_parameters']
                
                latex_code += f"""
\\subsubsection{{Tenor: {tenor}}}

\\begin{{table}}[h!]
\\centering
\\begin{{tabular}}{{ll}}
\\toprule
Parameter & Value \\\\
\\midrule
Option Tenor & {tenor} \\\\
FX Spot Rate & {params['fx_spot']:.5f} \\\\
$\\alpha_i$ & {params['alpha_i']:.6f} \\\\
$\\alpha_j$ & {params['alpha_j']:.6f} \\\\
RMSE OIS Price & {entry['rmse_ois_price']:.2f} \\\\
RMSE OIS Yield & {entry['rmse_ois_yield']:.2f} \\\\
RMSE Option Price & {entry['rmse_option_price']:.2f} \\\\
RMSE Option Vol & {entry['rmse_option_vol']:.2f} \\\\
\\bottomrule
\\end{{tabular}}
\\caption{{Parameters for {formatted_date} - {tenor}}}
\\end{{table}}

\\textbf{{Initial State Vector ($x_0$):}}
$$x_0 = {format_matrix_latex(params['x0'])}$$

\\textbf{{Omega Matrix ($\\omega$):}}
$$\\omega = {format_matrix_latex(params['omega'])}$$

\\textbf{{Mean Reversion Matrix ($m$):}}
$$m = {format_matrix_latex(params['m'])}$$

\\textbf{{Sigma Matrix ($\\sigma$):}}
$$\\sigma = {format_matrix_latex(params['sigma'])}$$

\\textbf{{u for domestic curve:}}
$$u_i = {format_matrix_latex(params['u_i'])}$$

\\textbf{{u for foreign curve:}}
$$u_j = {format_matrix_latex(params['u_j'])}$$

\\vspace{{0.5cm}}

"""

    # Add parameter evolution analysis
    latex_code += r"""
\section{Parameter Evolution Analysis}

\subsection{FX Spot Rate Evolution}
"""
    
    # Create evolution tables for key parameters
    for tenor in ["1Y", "4Y", "10Y"]:  # Focus on key tenors
        if any(tenor in organized_data[date] for date in unique_dates):
            latex_code += f"""
\\subsubsection{{Key Parameters Evolution - {tenor} Tenor}}

\\begin{{table}}[h!]
\\centering
\\begin{{tabular}}{{lcccccc}}
\\toprule
Date & FX Spot & $\\alpha_i$ & $\\alpha_j$ & $\\sigma_{{11}}$ & $\\sigma_{{22}}$ & Option RMSE \\\\
\\midrule
"""
            
            for date in unique_dates:
                if tenor in organized_data[date]:
                    entry = organized_data[date][tenor]
                    params = entry['model_parameters']
                    date_formatted = datetime.strptime(date, "%Y-%m-%d").strftime("%m/%d/%Y")
                    
                    sigma_11 = params['sigma'][0][0]
                    sigma_22 = params['sigma'][1][1]
                    
                    latex_code += f"""{date_formatted} & {params['fx_spot']:.5f} & {params['alpha_i']:.5f} & {params['alpha_j']:.5f} & {sigma_11:.4f} & {sigma_22:.4f} & {entry['rmse_option_vol']:.2f} \\\\
"""
            
            latex_code += f"""\\bottomrule
\\end{{tabular}}
\\caption{{Parameter Evolution for {tenor} Tenor}}
\\end{{table}}

"""

    # Add tenor comparison section
    latex_code += r"""
\section{Cross-Tenor Analysis}

\subsection{Calibration Quality by Tenor}

The following analysis compares calibration quality across different option tenors:

"""

    # Create a summary table of average RMSE by tenor
    latex_code += r"""
\begin{table}[h!]
\centering
\begin{tabular}{lcccc}
\toprule
Tenor & Avg RMSE OIS Price & Avg RMSE OIS Yield & Avg RMSE Option Price & Avg RMSE Option Vol \\
\midrule
"""

    for tenor in unique_tenors:
        rmse_ois_price = []
        rmse_ois_yield = []
        rmse_option_price = []
        rmse_option_vol = []
        
        for date in unique_dates:
            if tenor in organized_data[date]:
                entry = organized_data[date][tenor]
                rmse_ois_price.append(entry['rmse_ois_price'])
                rmse_ois_yield.append(entry['rmse_ois_yield'])
                rmse_option_price.append(entry['rmse_option_price'])
                rmse_option_vol.append(entry['rmse_option_vol'])
        
        if rmse_ois_price:  # Only add if we have data
            avg_ois_price = np.mean(rmse_ois_price)
            avg_ois_yield = np.mean(rmse_ois_yield)
            avg_option_price = np.mean(rmse_option_price)
            avg_option_vol = np.mean(rmse_option_vol)
            
            latex_code += f"""{tenor} & {avg_ois_price:.2f} & {avg_ois_yield:.2f} & {avg_option_price:.2f} & {avg_option_vol:.2f} \\\\
"""

    latex_code += r"""
\bottomrule
\end{tabular}
\caption{Average Calibration Quality Metrics by Tenor}
\end{table}

\section{Calibration Charts}

Charts showing the calibration results are organized by date and tenor. Each calibration includes:
\begin{itemize}
\item Domestic and foreign yield curve fits
\item Zero coupon bond price comparisons
\item FX implied volatility surface calibration
\end{itemize}

Note: Chart file naming convention follows the pattern:
\begin{itemize}
\item Domestic curves: \texttt{Domestic\_bonds\_[zcrate/price]\_[YYYYMMDD].png}
\item Foreign curves: \texttt{Foreign\_bonds\_[zcrate/price]\_[YYYYMMDD].png}
\item FX volatility: \texttt{fx\_option\_vol\_maturity\_[tenor]\_[YYYYMMDD].png}
\end{itemize}

\section{Key Findings and Conclusions}

\subsection{Model Performance}
\begin{itemize}
\item The LRW model demonstrates varying calibration quality across different option tenors
\item Shorter tenor options (1Y-2Y) generally show better volatility fits
\item Longer tenor options (7Y-10Y) exhibit higher calibration errors, suggesting potential model limitations for long-term volatility structures
\item FX spot rates show realistic market movements across the calibration period
\end{itemize}

\subsection{Parameter Stability}
\begin{itemize}
\item $\alpha$ parameters show tenor-dependent behavior
\item Volatility matrix ($\sigma$) parameters adapt significantly to different tenor requirements
\item Mean reversion parameters remain relatively stable across tenors
\item Cross-correlation structures vary with option maturity
\end{itemize}

\subsection{Recommendations}
\begin{itemize}
\item Focus on 1Y-5Y tenors for optimal model performance
\item Consider tenor-specific parameter constraints for better stability
\item Investigate alternative volatility specifications for long-term options
\item Monitor parameter evolution over time for early warning of model degradation
\end{itemize}

\end{document}
"""

    return latex_code

def main():
    # File paths
    main_project_root = r"E:\OneDrive\Dropbox\LinearRationalWishart_Work\Code\ED\LinearRationalWishart\LinearRationalWishart_NewCode"
    
    json_file = main_project_root + r"\wishart_processes\reporting\model_summary_all_report_all.json"
    # latex_report_file = main_project_root + r"\wishart_processes\reporting\multi_maturity_model_report.tex"
    latex_report_file = r"E:\OneDrive\Dropbox\LinearRationalWishart_Work\Code\ED\LinearRationalWishart\LinearRationalWishart_NewCode\wishart_processes\reporting\report_files_v2\multi_maturity_model_report.tex"
    
    # Load the model data
    model_data = load_model_data(json_file)
    
    # Generate LaTeX report
    latex_report = generate_latex_report(model_data)
    
    # Save to file
    with open(latex_report_file, "w") as f:
        f.write(latex_report)
    
    print("Multi-maturity LaTeX report generated successfully!")
    print(f"Report saved to: {latex_report_file}")
    print("Compile with: pdflatex multi_maturity_model_report.tex")
    
    # Print summary statistics
    organized_data = organize_data_by_date_and_tenor(model_data)
    unique_tenors = get_unique_tenors(model_data)
    unique_dates = sorted(organized_data.keys())
    
    print(f"\nReport Summary:")
    print(f"- Calibration dates: {len(unique_dates)}")
    print(f"- Option tenors: {len(unique_tenors)}")
    print(f"- Total calibrations: {len(model_data)}")
    print(f"- Tenors covered: {', '.join(unique_tenors)}")

if __name__ == "__main__":
    main()
