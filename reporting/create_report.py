import json
import numpy as np
from datetime import datetime

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

def generate_latex_report(model_data):
    """Generate LaTeX code for the model calibration report"""
    
    latex_code = r"""
\documentclass{article}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{geometry}
\usepackage{graphicx}
\usepackage{float}
\usepackage{subcaption}
\geometry{margin=1in}



\title{LRW FX Model Calibration Report}
\author{Model Calibration Team}
\date{\today}

\begin{document}

\graphicspath{{charts/}}

\maketitle

\section{Summary}
This report presents the calibrated parameters for the LRW (Linear-Rational-Wishart) FX Model across multiple calibration dates. The model is calibrated on domestic and foreign curves. For the options, 4Y tenor option data is used for the calibration.

\begin{itemize}
	\item Model details:
	

\begin{table}[h!]
\centering
\begin{tabular}{ll}
\toprule
Parameter & Value \\
\midrule
Model Type & LRW FX Model \\
Dimensions (n) & 2 \\
Jump Component & No Jump \\
\alpha_i is equal to \alpha_j  \\
\bottomrule
\end{tabular}
\caption{General model setting}
\end{table}


	\item details on matrix $u_i$ and $u_j$

\textbf{u for domestic curve:}
$$u_i = \begin{pmatrix}
1.0000 & 0.0000 \\
0.0000 & 0.1250
\end{pmatrix}$$

\textbf{u for foreign curve:}
$$u_j = \begin{pmatrix}
0.1000 & 0.0000 \\
0.0000 & 1.0000
\end{pmatrix}$$

\item 
$\alpha_i$ and $\alpha_j$ are the same.

\end{itemize}
\section{Model Parameters by Calibration Date}

\section{Model Parameters by Calibration Date}

"""
    
    for date, params in model_data.items():
        # Convert date format
        date_obj = datetime.strptime(date, "%Y%m%d")
        formatted_date = date_obj.strftime("%B %d, %Y")
        
        latex_code += f"""
\\subsection{{{formatted_date}}}

\\begin{{table}}[h!]
\\centering
\\begin{{tabular}}{{ll}}
\\toprule
Parameter & Value \\\\
\\midrule
Model Type & {params['model_type']} \\\\
%Dimensions (n) & {params['n']} \\\\
FX Spot Rate & {params['fx_spot']:.5f} \\\\
%Jump Component & {params['jump']} \\\\
$\\alpha_i$ & {params['alpha_i']:.6f} \\\\
$\\alpha_j$ & {params['alpha_j']:.6f} \\\\
\\bottomrule
\\end{{tabular}}
\\caption{{Basic Parameters for {formatted_date}}}
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

\\newpage
"""
    
    # Add summary table
    latex_code += r"""
\section{Parameter Evolution Summary}

\begin{longtable}{lccccccccccc}
\caption{Key Parameter Evolution Over Time} \\
\toprule
Date & FX Spot  & $alpha_i$ & $alpha_j$ & ${x_0}_{11}$ & ${x_0}_{22}$ & $\omega_{11}$ & $\omega_{22}$ & $m_{11}$ & $m_{22}$ & $\sigma_{11}$ & $\sigma_{22}$ \\
\midrule
\endfirsthead
\multicolumn{12}{c}%
{{\bfseries \tablename\ \thetable{} -- continued from previous page}} \\
\toprule
Date & FX Spot & $alpha_i$ & $alpha_j$ & ${x_0}_{11}$ & ${x_0}_{22}$ & $\omega_{11}$ & $\omega_{22}$ & $m_{11}$ & $m_{22}$ & $\sigma_{11}$ & $\sigma_{22}$ \\
\midrule
\endhead
\midrule \multicolumn{12}{r}{{Continued on next page}} \\ \midrule
\endfoot
\bottomrule
\endlastfoot
"""
    
    for date, params in model_data.items():
        date_obj = datetime.strptime(date, "%Y%m%d")
        formatted_date = date_obj.strftime("%m/%d/%Y")
        
        alpha_i=params['alpha_i']
        alpha_j=params['alpha_j']
        x0_11 = params['x0'][0][0] if 'x0' in params else params.get('x_0', [[0,0],[0,0]])[0][0]
        x0_22 = params['x0'][1][1] if 'x0' in params else params.get('x_0', [[0,0],[0,0]])[1][1]

        omega_11 = params['omega'][0][0]
        omega_22 = params['omega'][1][1]

        m_11 = params['m'][0][0]
        m_22 = params['m'][1][1]

        sigma_11 = params['sigma'][0][0]
        sigma_22 = params['sigma'][1][1]
        
        latex_code += f"{formatted_date} & {params['fx_spot']:.5f}& {alpha_i:.5f}& {alpha_j:.5f} & {x0_11:.4f} & {x0_22:.4f} & {omega_11:.4f} & {omega_22:.4f} & {m_11:.4f} & {m_22:.4f} & {sigma_11:.4f} & {sigma_22:.4f} \\\\\n"
    
    latex_code += r"""
\end{longtable}

\section{Calibration Result Charts}

This section presents the calibration results through various charts showing the yield curves and implied volatility surfaces for each calibration date.

"""

    # Add charts for each date
    for date, params in model_data.items():
        date_obj = datetime.strptime(date, "%Y%m%d")
        formatted_date = date_obj.strftime("%B %d, %Y")
        short_date = date_obj.strftime("%Y%m%d")
        
        latex_code += f"""
\\subsection{{Calibration Results for {formatted_date}}}

\\subsubsection{{Domestic Curve Analysis}}

\\begin{{figure}}[H]
\\centering
\\begin{{subfigure}}{{0.48\\textwidth}}
\\centering
\\includegraphics[width=\\textwidth]{{Domestic_bonds_zcrate_{short_date}.png}}
\\caption{{Zero Rate Curve}}
\\label{{fig:Domestic_bonds_zcrate_{short_date}}}
\\end{{subfigure}}
\\hfill
\\begin{{subfigure}}{{0.48\\textwidth}}
\\centering
\\includegraphics[width=\\textwidth]{{Domestic_bonds_price_{short_date}.png}}
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
\\includegraphics[width=\\textwidth]{{Foreign_bonds_zcrate_{short_date}.png}}
\\caption{{Zero Rate Curve}}
\\label{{fig:Foreign_bonds_zcrate_{short_date}}}
\\end{{subfigure}}
\\hfill
\\begin{{subfigure}}{{0.48\\textwidth}}
\\centering
\\includegraphics[width=\\textwidth]{{Foreign_bonds_price_{short_date}.png}}
\\caption{{Zero Coupon Bond Price Curve}}
\\label{{fig:Foreign_bonds_price_{short_date}}}
\\end{{subfigure}}
\\caption{{Foreign Curve Calibration Results for {formatted_date}}}
\\label{{fig:foreign_curves_{short_date}}}
\\end{{figure}}

\\subsubsection{{FX Volatility Surface}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.8\\textwidth]{{fx_option_vol_maturity_4.00_{short_date}.png}}
\\caption{{FX Implied Volatility Surface for {formatted_date}}}
\\label{{fig:fx_option_vol_maturity_4.00_{short_date}}}
\\end{{figure}}

\\vspace{{1cm}}

"""

    latex_code += r"""


\end{document}
"""
    

# \section{Model Specification}

# The LRW FX Model is specified as a 2-factor stochastic volatility model with the following key characteristics:

# \begin{itemize}
# \item \textbf{Domestic Interest Rate Process}: Follows a Wishart-driven affine process
# \item \textbf{Foreign Interest Rate Process}: Follows a Wishart-driven affine process  
# \item \textbf{FX Rate Process}: Coupled with both interest rate processes through correlation structure
# \item \textbf{Volatility Structure}: Time-varying volatility driven by the Wishart process
# \end{itemize}

# The model parameters are calibrated to market data on each calibration date to ensure accurate pricing of FX derivatives and consistency with observed market curves and volatility surfaces.

# \section{Conclusion}

# The calibration results demonstrate the model's ability to fit market data across different time periods. The parameter evolution shows the dynamic nature of the market conditions and the model's adaptability to changing market environments.

# Key observations from the calibration:
# \begin{itemize}
# \item FX spot rates show typical market movements over the calibration period
# \item Volatility parameters ($\sigma$ matrix) adapt to market conditions
# \item Mean reversion parameters ($m$ matrix) remain relatively stable
# \item The model successfully captures both curve dynamics and volatility structure
# \end{itemize}

    return latex_code
main_project_root = r"E:\OneDrive\Dropbox\LinearRationalWishart_Work\Code\ED\LinearRationalWishart\LinearRationalWishart_NewCode"


def main():
    # Example usage
    # json_file = r"E:\OneDrive\Dropbox\LinearRationalWishart_Work\Code\ED\LinearRationalWishart\LinearRationalWishart_NewCode\wishart_processes\reporting\curve_calibrated_model_report_4Y.json"
    # latex_report_file = r"E:\OneDrive\Dropbox\LinearRationalWishart_Work\Code\ED\LinearRationalWishart\LinearRationalWishart_NewCode\wishart_processes\reporting\model_report_4Y.tex"

    json_file = main_project_root +r"\wishart_processes\reporting\curve_calibrated_model_report_4Y.json"
    latex_report_file = main_project_root+ r"\wishart_processes\reporting\model_report_4Y.tex"


    #"curve_calibrated_model_report_4Y.json"
    
    # Load the model data
    model_data = load_model_data(json_file)
    
    # Generate LaTeX report
    latex_report = generate_latex_report(model_data)
    
    # Save to file
    with open(latex_report_file, "w") as f: # "model_calibration_report.tex", "w") as f:
        f.write(latex_report)
    
    print("LaTeX report generated successfully!")
    print("Compile with: pdflatex model_calibration_report.tex")
    
    # Also print a sample of the LaTeX code
    print("\nSample LaTeX output:")
    print(latex_report[:1000] + "...")

if __name__ == "__main__":
    main()
