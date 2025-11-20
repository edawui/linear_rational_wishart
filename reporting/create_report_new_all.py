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
    date_colors = [
        #"white",           # First date - no color
        "gray!15",         # Second date - light gray
        "blue!10",         # Third date - light blue
        "green!10",        # Fourth date - light green
        "yellow!15",       # Fifth date - light yellow
        "red!10",          # Sixth date - light red
        "purple!10",       # Seventh date - light purple
        "orange!10",       # Eighth date - light orange
    ]
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
\usepackage{xcolor}
\usepackage{colortbl}

\title{LRW FX Model Calibration Report}
\author{Model Calibration Team}
\date{\today}

\begin{document}

\graphicspath{{charts/}}

\maketitle

\section{Summary}
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
\caption{Calibration on yield data quality metrics by date (RMSE in b.p. - calibrated on price, on 2Y-10Y instruments)} \\
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

    # for date in unique_dates:
    for date_idx, date in enumerate(unique_dates):
        color = date_colors[date_idx % 2] #len(date_colors)]
        for tenor in unique_tenors:
            if tenor in organized_data[date]:
                entry = organized_data[date][tenor]
                params = entry['model_parameters']
                date_formatted = datetime.strptime(date, "%Y-%m-%d").strftime("%m/%d/%Y")
                if color != "white":
                    latex_code += f"\\rowcolor{{{color}}}\n"
                latex_code += f"""{date_formatted} & {tenor} & {entry['rmse_option_price']:.2f} & {entry['rmse_option_vol']:.2f}  \\\\
"""

    latex_code += r"""
\end{longtable}
%\end{landscape}

"""


 # Add calibration result on yield curve
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
%\end{landscape}"""

 # Add calibration result on vol data
    latex_code += r"""
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%% Model parameters on vol data curves %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%\begin{landscape}
\begin{longtable}{lcccccc}
\caption{Model paramters calibrated on Vol data by date and by tenor} \\
\toprule
Date &  Tenor & $\sigma_{11}$ & $\sigma_{22}$ & Correl-$\sigma$  & Correl-$x_0$   & Correl-$\omega$   \\

\midrule
\endfirsthead
\multicolumn{7}{c}%
{{\bfseries \tablename\ \thetable{} -- continued from previous page}} \\
\toprule
Date &  Tenor & $\sigma_{11}$ & $\sigma_{22}$ & Correl-$\sigma$  & Correl-$x_0$   & Correl-$\omega$   \\
\midrule
\endhead
\midrule \multicolumn{7}{r}{{Continued on next page}} \\ \midrule
\endfoot
\bottomrule
\endlastfoot
"""

    for date_idx, date in enumerate(unique_dates):
        color = date_colors[date_idx % 2] #len(date_colors)]
        # for date in unique_dates:
        for tenor in unique_tenors: 
            if tenor in organized_data[date]:
                entry = organized_data[date][tenor]
                params = entry['model_parameters']
                date_formatted = datetime.strptime(date, "%Y-%m-%d").strftime("%m/%d/%Y")
                sigma_correl=  params['sigma'][0][1] / np.sqrt(params['sigma'][0][0]*params['sigma'][1][1])
                x0_correl= params['x0'][0][1] / np.sqrt(params['x0'][0][0]*params['x0'][1][1])
                omega_correl= params['omega'][0][1] / np.sqrt(params['omega'][0][0]*params['omega'][1][1])
                if color != "white":
                    latex_code += f"\\rowcolor{{{color}}}\n"
                latex_code += f"""{date_formatted} & {tenor} &  {params['sigma'][0][0]:.6f} & {params['sigma'][1][1]:.6f} & {sigma_correl:.6f} & {x0_correl:.6f} & {omega_correl:.6f}   \\\\
"""




    latex_code += r"""
\end{longtable}
%\end{landscape}
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
\begin{tabular}{lcc}
\toprule
Tenor & Avg RMSE Option Price & Avg RMSE Option Vol \\
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
            
            # latex_code += f"""{tenor} & {avg_ois_price:.2f} & {avg_ois_yield:.2f} & {avg_option_price:.2f} & {avg_option_vol:.2f} \\\\
            latex_code += f"""{tenor}  & {avg_option_price:.2f} & {avg_option_vol:.2f} \\\\
"""

    latex_code += r"""
\bottomrule
\end{tabular}
\caption{Average Calibration Quality Metrics by Tenor.}
\end{table}
"""
    rmse_curve_text=f""" Note, for the yield curve, the average RMSE on price is {avg_ois_price:.2f}, and the average RMSE on yield is {avg_ois_yield:.2f}."""

    latex_code += rmse_curve_text


#########################Calibration Charts Section##########################################
   
    tenors = ['1', '2', '3', '4', '5', '7']
    tenor_labels = ['1Y', '2Y', '3Y', '4Y', '5Y', '7Y']

    tenors_all = [ '3', '4', '5', '7']
    tenor_labels_all = ['3Y', '4Y', '5Y', '7Y']

    latex_code += r"""
 
\section{Calibration Result Charts}

This section presents the calibration results through various charts showing the yield curves and implied volatility surfaces for each calibration date.

"""

    # Add charts for each date
    # for date, params in model_data.items():
    for date in unique_dates:
        # date_obj = datetime.strptime(date, "%Y%m%d")
        date_obj = datetime.strptime(date, "%Y-%m-%d") #.strftime("%m/%d/%Y")

        formatted_date = date_obj.strftime("%B %d, %Y")
        short_date = date_obj.strftime("%Y%m%d")
        
        latex_code += f"""
\\subsection{{Calibration Results for {formatted_date}}}

\\subsubsection{{Domestic Curve Analysis}}

\\begin{{figure}}[H]
\\centering
\\begin{{subfigure}}{{0.48\\textwidth}}
\\centering
\\includegraphics[width=\\textwidth]{{calibrate_on_vol_4__4_Domestic_bonds_zcrate_{short_date}_0000_.png}}
\\caption{{Zero Rate Curve}}
\\label{{fig:Domestic_bonds_zcrate_{short_date}}}
\\end{{subfigure}}
\\hfill
\\begin{{subfigure}}{{0.48\\textwidth}}
\\centering
\\includegraphics[width=\\textwidth]{{calibrate_on_vol_4__4_Domestic_bonds_price_{short_date}_0000_.png}}
\\caption{{Zero Coupon Bond Price Curve}}
\\label{{fig:Domestic_bonds_price_{short_date}}}
\\end{{subfigure}}
\\caption{{Domestic Curve Calibration Results for {formatted_date}}}
\\label{{fig:domestic_curves_{short_date}}}
\\end{{figure}}

\\subsubsection{{Foreign Curve Analysis}}

\\begin{{figure}}[H]
\\centering
\\begin{{subfigure}}{{0.48\\textwidth}}
\\centering
\\includegraphics[width=\\textwidth]{{calibrate_on_vol_4__4_Foreign_bonds_zcrate_{short_date}_0000_.png}}
\\caption{{Zero Rate Curve}}
\\label{{fig:Foreign_bonds_zcrate_{short_date}}}
\\end{{subfigure}}
\\hfill
\\begin{{subfigure}}{{0.48\\textwidth}}
\\centering
\\includegraphics[width=\\textwidth]{{calibrate_on_vol_4__4_Foreign_bonds_price_{short_date}_0000_.png}}
\\caption{{Zero Coupon Bond Price Curve}}
\\label{{fig:Foreign_bonds_price_{short_date}}}
\\end{{subfigure}}
\\caption{{Foreign Curve Calibration Results for {formatted_date}}}
\\label{{fig:foreign_curves_{short_date}}}
\\end{{figure}}



"""
###############################
   
    
        latex_code += f"""
    \\subsubsection{{FX Volatility Surface - Calibration by tenor}}

    \\begin{{figure}}[H]
    \\centering
    """

        for i, (tenor, label) in enumerate(zip(tenors, tenor_labels)):
            # 2 rows, 3 columns layout
            latex_code += f"""\\begin{{subfigure}}{{0.32\\textwidth}}
    \\centering
    \\includegraphics[width=\\textwidth]{{calibrate_on_vol_{tenor}__{tenor}_fx_option_vol_maturity_{tenor}.00_{short_date}_0000_.png}}
    \\caption{{{label}}}
    \\label{{fig:fx_option_vol_{tenor}Y_{short_date}}}
    \\end{{subfigure}}"""
        
            # Add spacing
            if (i + 1) % 3 == 0 and i < len(tenors) - 1:  # End of row, not last item
                latex_code += "\n\n"
            elif (i + 1) % 3 != 0:  # Not end of row
                latex_code += "\\hfill\n"
    
        latex_code += f"""
    \\caption{{FX Implied Volatility Surfaces for All Tenors (separate calibration by tenor) - {formatted_date}}}
    \\label{{fig:fx_option_vol_all_tenors_{short_date}}}
    \\end{{figure}}

    \\vspace{{1cm}}
    """
        latex_code += f"""
    \\subsubsection{{FX Volatility Surface - joint calibration for tenors 3Y, 4Y, 5Y, and 7Y}}

    \\begin{{figure}}[H]
    \\centering
    """
     
        for i, (tenor, label) in enumerate(zip(tenors_all, tenor_labels_all)):
            # 2 rows, 3 columns layout
            latex_code += f"""\\begin{{subfigure}}{{0.32\\textwidth}}
    \\centering
    \\includegraphics[width=\\textwidth]{{calibrate_on_vol_ALL__all_tenors_fx_option_vol_maturity_{tenor}.00_{short_date}_0000_.png}}
    \\caption{{{label}}}
    \\label{{fig:fx_option_vol_{tenor}Y_{short_date}}}
    \\end{{subfigure}}"""
        
            # Add spacing
            if (i + 1) % 2 == 0 and i < len(tenors_all) - 1:  # End of row, not last item
                latex_code += "\n\n"
            elif (i + 1) % 2 != 0:  # Not end of row
                latex_code += "\\hfill\n"
    
        latex_code += f"""
    \\caption{{FX Implied Volatility Surfaces for joint calibration for 3Y, 4Y, 5Y, and 7Y - {formatted_date}}}
    \\label{{fig:fx_option_vol_all_tenors_{short_date}}}
    \\end{{figure}}

    \\vspace{{1cm}}
    """
##################################


    latex_code += r"""

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
