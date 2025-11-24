# from optparse import Option
# from Sympy.logic.boolalg import true
# from Sympy.core.cache import USE_CACHE
# from Sympy.plotting.pygletplot.util import model_to_screen
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Union, Literal, Optional, Tuple, List
import sys
import os
from pathlib import Path
import matplotlib
import math
import os
import gc
import jax
from contextlib import contextmanager

import pandas as pd
# import numpy as np
import re
import json
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
# from set_path import *

# main_project_root = r"E:\OneDrive\Dropbox\LinearRationalWishart_Work\Code\ED\LinearRationalWishart\LinearRationalWishart_NewCode"


# def complete_setup():
"""Setup that forces constants to update everywhere"""
    
# 1. Clear constants cache if it exists
if 'constants' in sys.modules:
    del sys.modules['constants']

# 2. Detect environment
if 'google.colab' in sys.modules:
    from google.colab import drive
    drive.mount('/content/drive')
    project_root = "/content/drive/MyDrive/LinearRationalWishart_Work/Code/ED/LinearRationalWishart/LinearRationalWishart_NewCode"
else:
    project_root = r"E:\OneDrive\Dropbox\LinearRationalWishart_Work\Code\ED\LinearRationalWishart\LinearRationalWishart_NewCode"

mkt_data_folder = os.path.join(project_root, "wishart_processes", "mkt_data", "Data_new")

# 3. Set environment variables BEFORE importing constants
os.environ['FORCE_PROJECT_ROOT'] = project_root
os.environ['FORCE_MKT_DATA_FOLDER'] = mkt_data_folder

# 4. Create directories and set working directory
os.makedirs(project_root, exist_ok=True)
os.makedirs(mkt_data_folder, exist_ok=True)
os.chdir(project_root)

main_project_root = project_root  # Ensure main_project_root is set correctly





try:
   
    from ...data.data_helpers  import * 
    from ...data.data_fx_market_data  import CurrencyPairDailyData
    from ...calibration.fx.base import CalibrationConfig, OptimizationMethod, CalibrationResult
    from ...calibration.fx.fx_lrw_calibrator import LrwFxCalibrator
    from ...models.fx.base import BaseFxModel
    from ...models.fx.lrw_fx import LRWFxModel
    from ...models.fx.currency_basket import CurrencyBasket
    from ...pricing.fx.fourier_fx_pricer import FourierFxPricer
    from ...pricing.fx.mc_fx_pricer import MonteCarloFxPricer
    from ...pricing.implied_vol_black_scholes import * 
    from ...pricing.black_scholes import * 
  
except ImportError:
    # sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    # project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
    # sys.path.insert(0, project_root)

    current_file = os.path.abspath(__file__)
    project_root = current_file

    # Go up until we find the wishart_processes directory
    while os.path.basename(project_root) != "LinearRationalWishart_NewCode" and project_root != os.path.dirname(project_root):
        project_root = os.path.dirname(project_root)

    if os.path.basename(project_root) != "LinearRationalWishart_NewCode":
        # Fallback to hardcoded path
        project_root = main_project_root #r"E:\OneDrive\Dropbox\LinearRationalWishart_Work\Code\ED\LinearRationalWishart\LinearRationalWishart_NewCode"

    print(f"Using project root: {project_root}")
    sys.path.insert(0, project_root)


    from linear_rational_wishart.data.data_helpers  import * 
    from linear_rational_wishart.data.data_fx_market_data  import CurrencyPairDailyData
    from linear_rational_wishart.calibration.fx.fx_lrw_calibrator import LrwFxCalibrator
    from linear_rational_wishart.calibration.fx.base import CalibrationConfig,OptimizationMethod, CalibrationResult

    from linear_rational_wishart.models.fx.base import BaseFxModel
    from linear_rational_wishart.models.fx.lrw_fx import LRWFxModel 
    from linear_rational_wishart.models.fx.currency_basket import CurrencyBasket
    from linear_rational_wishart.pricing.fx.fourier_fx_pricer import FourierFxPricer
    from linear_rational_wishart.pricing.fx.mc_fx_pricer import MonteCarloFxPricer   
    from linear_rational_wishart.pricing.implied_vol_black_scholes import * 
    from linear_rational_wishart.pricing.black_scholes import * 

# matplotlib.use('TkAgg')  # or 'Qt5Agg'
# # Enable interactive mode globally
# plt.ion()

matplotlib.use('Agg')  # or 'Qt5Agg'
# Enable interactive mode globally
plt.ioff()

# curve_calibrated_model_report = r"E:\OneDrive\Dropbox\LinearRationalWishart_Work\Code\ED\LinearRationalWishart\LinearRationalWishart_NewCode\wishart_processes\curve_calibrated_model_report.csv"
curve_calibrated_model_report =  main_project_root+r"\wishart_processes\curve_calibrated_model_report.csv"


@contextmanager
def no_display():
    """Context manager to temporarily disable plot display."""
    old_backend = matplotlib.get_backend()
    matplotlib.use('Agg')
    try:
        yield
    finally:
        matplotlib.use(old_backend)

def get_initial_sigma(x0:np.ndarray,u_i:np.ndarray,u_j:np.ndarray                      
                      ,initial_vol:Union[float, np.ndarray] )-> np.ndarray:

    """Calculate initial sigma matrix based on x0 and volatilities."""
    initial_vol_mean= 0.0
    if isinstance(initial_vol, float):
        initial_vol_mean= initial_vol
    elif isinstance(initial_vol, np.ndarray):
        initial_vol_mean = np.mean(initial_vol)
    else:
        raise ValueError("initial_vol must be a float or numpy array")
    
    temp1 =4*np.trace( u_i @ x0 @ u_j)
    temp2 =4*np.trace( u_i @ x0 @ u_i)
    temp3 =8*np.trace( u_j @ x0 @ u_i)
    temp4 =np.trace( u_i @ x0 )
    temp5 =np.trace( u_j @ x0)

    temp = temp1/((1+temp4)**2) + temp2/((1+temp4)**2) - temp3/(1+temp4)/(1+temp5)
    initial_sigma = initial_vol_mean /temp
    initial_sigma = np.sqrt(initial_sigma) * np.array([[1.0, 0.0], [0.0, 1.0]])
    return initial_sigma




def get_curve_calibrated_model(date, file=curve_calibrated_model_report):
    """
    Read calibrated model parameters from CSV file and create LRWFxModel instance.
    
    Parameters:
    date (str): Date in format 'YYYYMMDD' (e.g., '20250314')
    
    Returns:
    LRWFxModel: Instance of the calibrated model
    """
    
    # Read the CSV file


    
    # Handle case where date might be passed as a list
    if isinstance(date, list):
        if len(date) == 0:
            raise ValueError("Date list is empty")
        date = str(date[0])
    elif not isinstance(date, str):
        date = str(date)
    
    # Remove any whitespace and ensure it's a string
    date = date.strip()
    
    # Read the CSV file
    # file_path = 'curve_calibrated_model_report.csv'
    file_path = file
    
    try:
        with open(file_path, 'r') as file:
            content = file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Model data file not found: {file_path}")
    
    # Parse the data from each line
    model_data = {}
    lines = content.strip().split('\n')
    
    def parse_matrix(matrix_str):
        """Helper function to parse matrix strings like '[[1.0, 2.0], [3.0, 4.0]]'"""
        try:
            # Clean up the string and parse as JSON
            cleaned = matrix_str.replace("'", '"')
            return json.loads(cleaned)
        except:
            return None
    
    for line in lines:
        if line.strip():
            # Extract date
            date_match = re.search(r'^(\d{8})', line)
            if date_match:
                line_date = date_match.group(1)
                params = {}
                
                # Extract n
                n_match = re.search(r"'n':\s*(\d+)", line)
                if n_match:
                    params['n'] = int(n_match.group(1))
                
                # Extract x0 matrix
                x0_match = re.search(r"'x0':\s*(\[\[.*?\]\])", line)
                if x0_match:
                    params['x0'] = np.array(parse_matrix(x0_match.group(1)))
                
                # Extract omega matrix
                omega_match = re.search(r"'omega':\s*(\[\[.*?\]\])", line)
                if omega_match:
                    params['omega'] = np.array(parse_matrix(omega_match.group(1)))
                
                # Extract m matrix
                m_match = re.search(r"'m':\s*(\[\[.*?\]\])", line)
                if m_match:
                    params['m'] = np.array(parse_matrix(m_match.group(1)))
                
                # Extract sigma matrix
                sigma_match = re.search(r"'sigma':\s*(\[\[.*?\]\])", line)
                if sigma_match:
                    params['sigma'] = np.array(parse_matrix(sigma_match.group(1)))
                
                # Extract alpha_i
                alpha_i_match = re.search(r"'alpha_i':\s*([\d\.-]+)", line)
                if alpha_i_match:
                    params['alpha_i'] = float(alpha_i_match.group(1))
                
                # Extract u_i matrix (remove _1, _2 suffixes)
                u_i_match = re.search(r"'u_i':\s*(\[\[[^\]]*\]\s*,\s*\[[^\]]*\]\])", line)
                if u_i_match:
                    ui_str = re.sub(r'_\d+', '', u_i_match.group(1))
                    params['u_i'] = np.array(parse_matrix(ui_str))
                
                # Extract alpha_j
                alpha_j_match = re.search(r"'alpha_j':\s*([\d\.-]+)", line)
                if alpha_j_match:
                    params['alpha_j'] = float(alpha_j_match.group(1))
                
                # Extract u_j matrix (remove _1, _2 suffixes)
                u_j_match = re.search(r"'u_j':\s*(\[\[[^\]]*\]\s*,\s*\[[^\]]*\]\])", line)
                if u_j_match:
                    uj_str = re.sub(r'_\d+', '', u_j_match.group(1))
                    params['u_j'] = np.array(parse_matrix(uj_str))
                
                # Extract fx_spot
                fx_spot_match = re.search(r"'fx_spot':\s*([\d\.]+)", line)
                if fx_spot_match:
                    params['fx_spot'] = float(fx_spot_match.group(1))
                
                model_data[line_date] = params
    
    # Check if the requested date exists
    # if date not in model_data:
    #     available_dates = list(model_data.keys())
    #     raise ValueError(f"Date '{date}' not found in model data. Available dates: {available_dates}")
    
    # # Debug print to help troubleshoot
    # print(f"Loading model for date: {date} (type: {type(date)})")
    # print(f"Available dates: {list(model_data.keys())}")
    
    # Get parameters for the requested date
    params = model_data[date]
    
    # Extract all required parameters
    n = params['n']
    x0 = params['x0']
    omega = params['omega']
    m = params['m']
    sigma = params['sigma']
    alpha_i = params['alpha_i']
    u_i = params['u_i']
    alpha_j = params['alpha_j']
    u_j = params['u_j']
    fx_spot = params['fx_spot']
    
    # Create and return the LRWFxModel instance
    # Note: You'll need to import your LRWFxModel class
    # from your_module import LRWFxModel
    
    rho_x0,rho_omega,rho_sigma=0.10,0.10,0.10

    ##get initial sigma based on x0 and u_i, u_j
    # sigma= get_initial_sigma(x0,u_i,u_j,initial_vol=0.09)#85 )
    
    x0= np.array(x0)
    omega= np.array(omega)
    sigma= np.array(sigma)
    ##set anti-diagonal elements based on correlation
    x0[0,1]=x0[1,0]=       rho_x0*math.sqrt(x0[0,0]*x0[1,1])
    omega[0,1]=omega[1,0]= rho_omega*math.sqrt(omega[0,0]*omega[1,1])
    sigma[0,1]=sigma[1,0]= rho_sigma*math.sqrt(sigma[0,0]*sigma[1,1])

    lrwfx1 = LRWFxModel(
        n=n,
        x0=x0,
        omega=omega,
        m=m,
        sigma=sigma,
        alpha_i=alpha_i,
        u_i=u_i,
        alpha_j=alpha_j,
        u_j=u_j,
        fx_spot=fx_spot
    )
    
    return lrwfx1


def get_curve_calibrated_model_old():
         x0= np.array([[0.5236196019081235, 0.004530853181353052], [0.004530853181353052, 1.0006205529436851e-06]] )
         omega= np.array([[0.2871528799671776, 0.001762527884046532], [0.001762527884046532, 0.02174181052462134]])
         m= np.array([[-0.27129944342978357, 0.0], [0.0, -0.03984080543418016]])
         sigma= np.array([[0.2135465714972303, 0.038867997482414846], [0.038867997482414846, 0.7074434479097225]] )
         alpha_i= 0.03959478810429573
         u_i= [[1.0, 0.0], [0.0, 0.125]] 
         alpha_j= 0.03959478810429573
         u_j= [[0.1, 0.0], [0.0, 1.0]]
         fx_spot= 1.088
         n=2

         sigma= 0.3*np.array([[0.2135465714972303, 0.038867997482414846], [0.038867997482414846, 0.7074434479097225]] )

         rho_x0,rho_omega,rho_sigma=0.10,0.10,0.10

         ##get initial sigma based on x0 and u_i, u_j
         # sigma= get_initial_sigma(x0,u_i,u_j,initial_vol=0.09)#85 )
    
         ##set anti-diagonal elements based on correlation
         x0[0,1]=x0[1,0]=       rho_x0*math.sqrt(x0[0,0]*x0[1,1])
         omega[0,1]=omega[1,0]= rho_omega*math.sqrt(omega[0,0]*omega[1,1])
         sigma[0,1]=sigma[1,0]= rho_sigma*math.sqrt(sigma[0,0]*sigma[1,1])

         lrwfx1    = LRWFxModel(n, x0, omega, m, sigma,alpha_i, u_i, alpha_j, u_j,fx_spot)
         return lrwfx1

def create_base_fx_model(fx_spot=1.0) -> LRWFxModel:
    """Create base FX model with standard parameters."""
   #region Remove this later
   #  n = 2
    
   #  # Model parameters
   #  x0 = np.array([[4.99996419e+00, -3.18397238e-04],
   #                 [-3.18397238e-04, 4.88302000e-04]])
    
   #  omega = np.array([[2.09114424e+00, -8.04612684e-04],
   #                    [-8.04612684e-04, 1.92477100e-03]])
    
   #  m = np.array([[-0.20583617, 0.0],
   #                [0.0, -0.02069993]])
    
   #  sigma = np.array([[0.15871937, 0.10552826],
   #                    [0.10552826, 0.02298161]])
    
   #  alpha_i = 0.05  # Domestic rate
   #  alpha_j = 0.04  # Foreign rate
    
   #  # u_i = np.array([[1.0, 0.0], [0.0, 0.0]])
   #  # u_i = np.array([[1.0, -0.01], [-0.01, 0.5]])
   #  # u_i = np.array([[1.0, 0.50], [0.50, 0.00]])
   #  u_i = np.array([[1.0, 0.0], [0.0, 0.25]])
   #  u_i = np.array([[1.0, 0.0], [0.0, 0.25]])
   #  u_j = np.array([[0.1, 0.0], [0.0, 1.0]])

   #  u_i = np.array([[1.0, 0.0], [0.0, 0.25]])
   #  u_j = np.array([[0.1, 0.0], [0.0, 1.0]])

   #  # u_i = np.array([[1.0, 0.0], [0.0, 0.0]])
   #  # u_j = np.array([[0.0, 0.0], [0.0, 1.0]])
    
   #  # fx_spot = 1.0
   #  # return LRWFxModel(n, x0, omega, m, sigma, alpha_i, u_i, alpha_j, u_j, fx_spot)
   #  ################################################################
   #  # The following parameters are based on the original code provided
   #  ################################################################
   # #  x0_ST = np.array( [[ 0.17771204, -0.002749317 ],[ -0.002749317, 0.000488302]])
   # #  omega_ST = np.array( [[ 0.31866193, 0.003327917 ],[0.003327917, 0.001924771]])
   # #  m_ST = np.array( [[-0.822496255, 0],[ 0, -0.773125807 ]]);
   # #  sigma_ST= np.array([[ 0.134722479, 1.30e-06 ],[ 1.30e-06, 0.006223976]])
    
   # #  sigma_ST= np.array([[ 1.21641381, 1.30e-06 ],[ 1.30e-06, 0.28050888]])
   # #  x0Rho=-0.69173367
   # #  omegaRho=-0.51209522
   # #  sigmaRho=0.30927215

   # # # [-0.69173367 -0.51209522  1.21641381  0.28050888  0.30927215]
   # # # [0.31807089 0.70445911 1.6596622  1.87673194 0.74272099]
   # #  x0_ST[0,1]=x0_ST[1,0]=       x0Rho*math.sqrt(x0_ST[0,0]*x0_ST[1,1])
   # #  omega_ST[0,1]=omega_ST[1,0]= omegaRho*math.sqrt(omega_ST[0,0]*omega_ST[1,1])
   # #  sigma_ST[0,1]=sigma_ST[1,0]= sigmaRho*math.sqrt(sigma_ST[0,0]*sigma_ST[1,1])
    
     
   #  x0 = np.array( [[ 0.17771204, -0.002749317 ],[ -0.002749317, 0.000488302]])

   #  omega = np.array([[ 2.09114424e+00,-8.04612684e-04]
   #                  , [-8.04612684e-04, 1.92477100e-03]])

   #  m = np.array([[-0.20583617, 0.0        ]
   #                ,[ 0.0  ,      -0.02069993]])

   #  sigma = np.array([[0.15871937, 0.10552826]
   #                  , [0.10552826, 0.02298161]])
    
   #  sigma[1,1] =  sigma[0,0]
   #  rho_x0,rho_omega,rho_sigma=-0.89999272, 0.89760242, -0.88379066
   #  rho_x0,rho_omega,rho_sigma=0.5,0.5,0.5
   #  rho_x0,rho_omega,rho_sigma=-0.10704539,-0.10704539,-0.10704539 #not moving
   #  rho_x0,rho_omega,rho_sigma=-0.10704539,0.10704539,-0.10704539#not moving
   #  rho_x0,rho_omega,rho_sigma=-0.10704539,0.10704539,0.10704539#not moving
   #  rho_x0,rho_omega,rho_sigma=0.10704539,0.10704539,0.10704539#not moving
   #  rho_x0,rho_omega,rho_sigma=-0.5,-0.5,0.5#not moving
   #  rho_x0,rho_omega,rho_sigma=0,0,0
   #  # rho_x0=0.0


   #  x0[0,1]=x0[1,0]=       rho_x0*math.sqrt(x0[0,0]*x0[1,1])
   #  omega[0,1]=omega[1,0]= rho_omega*math.sqrt(omega[0,0]*omega[1,1])
   #  sigma[0,1]=sigma[1,0]= rho_sigma*math.sqrt(sigma[0,0]*sigma[1,1])
    
   #  sigma=4.0*sigma  # Scale sigma to match the original model

   #  #############################result after OIS calibration
    
   #  n = 2
   #  alpha = 0.03820131719112396
   #  x0 = [[2.77417019e+00, 0.00000000e+00],
   #   [0.00000000e+00, 3.27986190e-04]]
   #  omega = [[1.52072488, 0.        ],
   #   [0.,         0.02425804]]
   #  m = [[-0.27059843,  0.        ],
   #   [ 0.,         -0.04255721]]
    
   

   #  sigma = [[0.63487748, 0.        ],
   #   [0.,         0.63487748]]
   #  sigma=np.array( [[0.2135465714972303, 0.007632468354215629], [0.007632468354215629, 1.7074434479097225]])
   #  # for 1 to 5y
   #  sigma= 0.5*np.array([[0.2135465714972303, 0.007632468354215629], [0.007632468354215629, 1.7074434479097225]])

   #  ##### From bond calibration 0.5Y up to 11y
   #  x0= np.array([[4.557792523291154, 0.07445096114255088], [0.07445096114255088, 0.17297996909848626]])
   #  omega=np.array( [[0.7906565704207942, 0.021743759506268705], [0.021743759506268705, 0.04246760183899087]])
   #  m= np.array([[-0.08399813930807837, 0.0], [0.0, -0.036763824449248556]])
    
   #  x0= np.array([[4.557792523291154, 0.07445096114255088], [0.07445096114255088, 1.0]])
   #  omega=np.array( [[0.7906565704207942, 0.021743759506268705], [0.021743759506268705, 1.0]])
   #  m= np.array([[-0.08399813930807837, 0.0], [0.0, -1.0]])
    
   #  x0= np.array([[1.1383552022082195, 0.009504803705570511], [0.009504803705570511, 0.2875820825591115]])
   #  omega=np.array([[1.762527884046532e-06, 0.0019615111419316773], [0.0019615111419316773, 0.06463364229290659]])
   #  m= np.array([[-0.011530777480229652, 0.0], [0.0, -0.06148652138127832]] )
    
   #  x0= np.array([[0.11383552022082195, 0.009504803705570511], [0.009504803705570511, 0.2875820825591115]])
   #  omega=np.array([[1.762527884046532e-2, 0.0019615111419316773], [0.0019615111419316773, 0.06463364229290659]])
   #  omega=np.array([[1.762527884046532e-4, 0.0019615111419316773], [0.0019615111419316773, 0.06463364229290659]])
    
   #endregion
    n = 2  # Number of factors

    ###For the new calibration
    x0= np.array([[1.1383552022082195, 0.009504803705570511], [0.009504803705570511, 0.2875820825591115]])
    omega=np.array([[1.762527884046532e-2, 0.0019615111419316773], [0.0019615111419316773, 0.06463364229290659]])

    x0= 0.1*np.array([[0.71383552022082195, 0.009504803705570511], [0.009504803705570511, 0.2875820825591115]])
    omega=np.array([[1.762527884046532e-2, 0.0019615111419316773], [0.0019615111419316773, 1.762527884046532e-2]])
    m= np.array([[-0.011530777480229652, 0.0], [0.0, -0.06148652138127832]] )

    ##Too low sigma
    sigma= 0.25*np.array([[0.2135465714972303, 0.007632468354215629], [0.007632468354215629, 0.7074434479097225]])
    
    ##Too high sigma and abnormal smile
    sigma= 1.0*np.array([[0.2135465714972303, 0.007632468354215629], [0.007632468354215629, 0.7074434479097225]])
    
    # sigma= 0.35*np.array([[0.2135465714972303, 0.007632468354215629], [0.007632468354215629, 0.7074434479097225]])

    rho_x0,rho_omega,rho_sigma=0.10,0.10,0.10

    ##get initial sigma based on x0 and u_i, u_j
    # sigma= get_initial_sigma(x0,u_i,u_j,initial_vol=0.09)#85 )
    
    ##set anti-diagonal elements based on correlation
    x0[0,1]=x0[1,0]=       rho_x0*math.sqrt(x0[0,0]*x0[1,1])
    omega[0,1]=omega[1,0]= rho_omega*math.sqrt(omega[0,0]*omega[1,1])
    sigma[0,1]=sigma[1,0]= rho_sigma*math.sqrt(sigma[0,0]*sigma[1,1])
   
    u_i = np.array([[1.0, 0.0], [0.0, 0.0]])
    u_j = np.array([[0.0, 0.0], [0.0, 1.0]])

    u_i= [[1.0, 0.0], [0.0, 0.125]]       
    u_j= [[0.10, 0.0], [0.0, 1.0]]

    alpha_i = 0.03959478810429573  # Domestic rate
    alpha_j = 0.03959478810429573  # Domestic rate

    # alpha_i = 0.02959478810429573  # Domestic rate
    # alpha_j = 0.02959478810429573  # Domestic rate

    # alpha_j =  0.01681286096572876  # Foreign rate

    fxSpot = fx_spot  # Default FX spot rate
    lrwfx1    = LRWFxModel(n, x0, omega, m, sigma,alpha_i, u_i, alpha_j, u_j,fxSpot)
    return lrwfx1
    # return LRWFxModel(n, x0, omega, m, sigma, alpha_i, u_i, alpha_j, u_j, fx_spot)

def create_default_boudary_fx_model(fxSpot=1.0) -> LRWFxModel:
  
    n=2
    nonDiagLB = -0.9
    nonDiagUB = 0.9
    sigma_max=5.0

    sigma_min=1e-4
    x0_min=1e-4
    omega_min=1e-4

    x0_LB =  np.array(  [[ x0_min, nonDiagLB ], [ nonDiagLB, x0_min ] ])
    omega_LB =  np.array(  [[ omega_min, nonDiagLB ], [ nonDiagLB,  omega_min] ])
    m_LB =  np.array(  [[ -6, -2 ], [ -2, -4 ] ])
    sigma_LB =  np.array(  [[ sigma_min,-0.9 ], [ -0.9, sigma_min ]] )
    alpha_LB = 0.001

    x0_max=10.0
    omega_max=10.0
    sigma_max=5.0
    x0_UB =  np.array(  [[ x0_max, nonDiagUB ], [ nonDiagUB, x0_max ] ])
    omega_UB =  np.array(  [[ omega_max, nonDiagUB ], [ nonDiagUB, omega_max ] ])
    m_UB =  np.array(  [[ -1e-5, 1e-6 ], [ 1e-6, -1e-5 ] ])
    sigma_UB =  np.array(  [[ sigma_max, 0.9 ], [ 0.9, sigma_max ] ])
    alpha_UB = 0.1 
 
    #######################################################################
    #######################################################################
    
    alpha_i = 0.02
    alpha_j = 0.02
    # u_i = np.array([[1.0 ,0.0],[0.0, 0.0]])
    # u_j = np.array([[0.00,0.0],[0.0, 1.0]])
    u_i = np.array([[1.0, 0.0],[0.0, 0.4]])
    u_j = np.array([[0.5, 0.0],[0.0, 1.0]])
    lrwfx1_LB = LRWFxModel(n, x0_LB, omega_LB, m_LB, sigma_LB, alpha_LB, u_i, alpha_LB, u_j, fxSpot)
    lrwfx1_UB = LRWFxModel(n, x0_UB, omega_UB, m_UB, sigma_UB, alpha_UB, u_i, alpha_UB, u_j, fxSpot)
    return lrwfx1_LB, lrwfx1_UB

def example_lrw_fx_calibration(tenor,current_date = '20250401', suffix="calibrate_on_vol", on_vol=True, correl="YES",option_calibration_steps=None):
    print("Creating base FX model and calibrator...")
    
    

    print("Getting market data")

    test_fx_data = get_testing_excel_data(current_date = current_date #'20250530'
                                        , ccy_pair = "EURUSD" )
    fx_spot = test_fx_data.fx_spot ##get_spot_rate("EURUSD")
    print("Creating base FX model...")

    # use_curve_calibrated_model = True
    use_curve_calibrated_model=False #True

    if use_curve_calibrated_model:
        # lrw_fx_model = get_curve_calibrated_model_old()
        lrw_fx_model = get_curve_calibrated_model(current_date)

    else:
        lrw_fx_model =create_base_fx_model(fx_spot)

    zc_mat_min=2.0#3.0#2.0#3.0#4.0#2.0
    
    print("Creating calibration configuration...")
    calibration_config= CalibrationConfig()
    calibration_config.calibrate_based_on_correlation=True

    if on_vol:
        calibration_config.calibrate_on_vol=True
    else:
       calibration_config.calibrate_on_vol=False

    # optimization_method: OptimizationMethod = OptimizationMethod.HYBRID
    if tenor=="ALL":
        calibration_config.min_maturity_option = 3.0 #3.0#3.0  # zc_mat_min +1# 2.0
        calibration_config.max_maturity_option = 7.00#5.0  #7.0  # Maximum maturity for calibration
        
        calibration_config.min_maturity_option_for_chart = 3.0#3.0#3.0  
        calibration_config.max_maturity_option_for_chart = 7.0#5.0 
        
        # calibration_config.use_atm_only =True
        
        suffix += "_all_tenors"
    else:
        calibration_config.min_maturity_option = float(tenor)  # Minimum maturity for calibration
        calibration_config.max_maturity_option = float(tenor)  # Maximum maturity for calibration

        calibration_config.min_maturity_option_for_chart= float(tenor)  
        calibration_config.max_maturity_option_for_chart = float(tenor)   
        suffix += f"_{tenor}"

    
    print("Creating LRW FX calibrator...")
    lrw_fx_calibrator =LrwFxCalibrator(test_fx_data,lrw_fx_model,calibration_config)
    # return 
    print("creating calibration boundary models...")
    lrwfx1_LB, lrwfx1_UB =create_default_boudary_fx_model(fx_spot) 

    print("setting calibration boundary models...")
    lrw_fx_calibrator.set_boundary_models(lrwfx1_LB, lrwfx1_UB)

    print("setting alpha...")
    # lrw_fx_calibrator.set_alpha(alpha_tenor=5.0)
    # lrw_fx_calibrator.set_alpha(alpha_tenor=lrw_fx_calibrator.config.max_maturity_zc)
    lrw_fx_calibrator.set_alpha()

    # lrw_fx_calibrator.model.temporary_set_model_params()

    
    print("Repricing bonds...")
    lrw_fx_calibrator.reprice_bonds()

    print(" Calibration starting point")
    lrw_fx_calibrator.model.print_model()
        
    if not use_curve_calibrated_model:
        print("Calibration on OIS Zero coupon bond ") 
    
        error_OIS=0.0
        error_OIS=lrw_fx_calibrator.calibrate_ois(calibrate_on_price=True,
                                                  #max_tenor=11,min_tenor=zc_mat_min,
                                                  joint_calibrate_domestic_and_foreign=True
                                                  # joint_calibrate_domestic_and_foreign=False
                                                  # , domestic_first=False #True
                                                  , domestic_first=True
                                                  )
    
        jax.clear_caches()
        gc.collect()
                                         
    ##########################################################################
    set_sigma= True
    if set_sigma:
        sigma= get_initial_sigma(lrw_fx_calibrator.model.x0,lrw_fx_calibrator.model.u_i,lrw_fx_calibrator.model.u_j,initial_vol=0.09)#85 )
        x0=np.array(lrw_fx_calibrator.model.x0)
        m= np.array(lrw_fx_calibrator.model.m)
        omega= np.array(lrw_fx_calibrator.model.omega)

       
        # # x0= [[0.43448948892668604, 0.0], [0.0, 0.027443388711797644]]
        # # omega= [[1.0000000000000003e-05, 0.0], [0.0, 0.052958311797696375]]
        # # m= [[-0.006086909713659209, 0.0], [0.0, -0.8264698793462261]]
        # # sigma= [[0.10677328574861515, 0.0], [0.0, 0.35372172395486123]]
        # # alpha_i= 0.03761078789830208
        # # u_i= [[1.0, 0.0], [0.0, 0.25]]
        # # alpha_j= 0.024520594626665115
        # # u_j= [[0.1, 0.0], [0.0, 1.0]]
        # # x_spot= 1.088
       

        # # # # # sigma     =  np.array([[0.03238411, 0.0], [0.0,0.03632396]])     
        # # rho_sigma=-0.8969805 
        # # rho_x0=-0.5
        # # rho_omega=-0.5
        sigma =0.5*sigma
        rho_sigma,rho_x0,rho_omega=0.25,-0.25,0.25
        rho_sigma,rho_x0,rho_omega = -0.10, -0.1, -0.10

        ##From Tenor=4Y calibration
        rho_sigma,rho_x0,rho_omega = 0.17, 0.025, 0.25
        # rho_sigma,rho_x0,rho_omega = 0.0, 0.0, 0.0

        x0[0,1]=x0[1,0]=rho_x0*math.sqrt(x0[0,0]*x0[1,1]) # Set correlation to 0.5)
        omega[0,1]=omega[1,0]=rho_omega*math.sqrt(omega[0,0]*omega[1,1]) # Set correlation to 0.5)
        sigma[0,1]=sigma[1,0]=rho_sigma*math.sqrt(sigma[0,0]*sigma[1,1]) # Set correlation to 0.5)


        
         # # Calibration on Option, with alpha based on maturity =5
         # # Model Report: {'model_type': 'LRW FX Model', 'n': 2, 'x0': [[1.7931206094918175, 0.022654265906765263], [0.022654265906765263, 5.7917332163852855]], 'omega': [[0.08208115307433547, 0.01762527884046532], [0.01762527884046532, 0.003306558645118938]], 'm': [[-0.02337262748056307, 0.0], [0.0, -0.0012725382807416497]], 'sigma': [[0.2135465714972303, 0.038867997482414846], [0.038867997482414846, 0.7074434479097225]], 'alpha_i': 0.03722203150391579, 'u_i': [[1.0, 0.0], [0.0, 0.0]], 'alpha_j': 0.02318561263382435, 'u_j': [[0.0, 0.0], [0.0, 1.0]], 'fx_spot': 1.088, 'has_jump': False, 'jump': 'No Jump'}

         # #Domestic is Good
         # # Model Report: {'model_type': 'LRW FX Model', 'n': 2, 'x0': [[1.0108356689623457, 0.004530853181353052], [0.004530853181353052, 9.99483353177249]], 'omega': [[0.11590137179499428, 0.001762527884046532], [0.001762527884046532, 1.0000000000000003e-05]], 'm': [[-0.04566353860663734, 0.0], [0.0, -0.004584016096228337]], 'sigma': [[0.2135465714972303, 0.038867997482414846], [0.038867997482414846, 0.7074434479097225]], 'alpha_i': 0.03959478810429573, 'u_i': [[1.0, 0.0], [0.0, 0.125]], 'alpha_j': 0.01681286096572876, 'u_j': [[0.1, 0.0], [0.0, 1.0]], 'fx_spot': 1.088, 'has_jump': False, 'jump': 'No Jump'}

         # #Domestic and foreign are good
         # # Model Report: {'model_type': 'LRW FX Model', 'n': 2, 'x0': [[0.5236196019081235, 0.004530853181353052], [0.004530853181353052, 1.0006205529436851e-06]], 'omega': [[0.2871528799671776, 0.001762527884046532], [0.001762527884046532, 0.02174181052462134]], 'm': [[-0.27129944342978357, 0.0], [0.0, -0.03984080543418016]], 'sigma': [[0.2135465714972303, 0.038867997482414846], [0.038867997482414846, 0.7074434479097225]], 'alpha_i': 0.03959478810429573, 'u_i': [[1.0, 0.0], [0.0, 0.125]], 'alpha_j': 0.03959478810429573, 'u_j': [[0.1, 0.0], [0.0, 1.0]], 'fx_spot': 1.088, 'has_jump': False, 'jump': 'No Jump'}
         # # Model Report: {'model_type': 'LRW FX Model', 'n': 2, 
         # # x0= [[0.5236196019081235, 0.004530853181353052], [0.004530853181353052, 1.0006205529436851e-06]] 
         # # omega= [[0.2871528799671776, 0.001762527884046532], [0.001762527884046532, 0.02174181052462134]]
         # # m= [[-0.27129944342978357, 0.0], [0.0, -0.03984080543418016]]
         # # sigma= [[0.2135465714972303, 0.038867997482414846], [0.038867997482414846, 0.7074434479097225]] 
         # # alpha_i= 0.03959478810429573
         # # u_i= [[1.0, 0.0], [0.0, 0.125]] 
         # # alpha_j= 0.03959478810429573
         # # u_j= [[0.1, 0.0], [0.0, 1.0]]
         # # fx_spot= 1.088
         # #  'has_jump': False, 'jump': 'No Jump'}

        print("set sigma based on initial parameters...")
        print(f"Initial sigma={sigma}")
    
        lrw_fx_calibrator.model.set_model_params(2,x0
                                                ,omega
                                                ,m
                                                ,sigma
                                                ,lrw_fx_calibrator.model.alpha_i
                                                ,lrw_fx_calibrator.model.u_i
                                                ,lrw_fx_calibrator.model.alpha_j
                                                ,lrw_fx_calibrator.model.u_j
                                                ,fx_spot=fx_spot)
        #########################################################################
   
    lrw_fx_calibrator.model.print_model()
    
    # print(lrw_fx_calibrator.daily_data.ois_summary())

    # print("Calibration on Euribor Spread A")    
              
    # print(f"RMSE Error OIS ={errorOIS}, RMSE Error on Spread A={errorA}")
    print("Calibration on Option ")  
    # lrw_fx_calibrator.config.optimization_method=OptimizationMethod.DUAL_ANNEALING
    # lrw_fx_calibrator.config.optimization_method=OptimizationMethod.HYBRID
    lrw_fx_calibrator.config.optimization_method=OptimizationMethod.LEAST_SQUARES
    # lrw_fx_calibrator.config.optimization_method=OptimizationMethod.DIFFERENTIAL_EVOLUTION


    calibration_result=CalibrationResult()
 
    if option_calibration_steps==None:
        # option_calibration_steps=[1]
        # option_calibration_steps=[1,2]
        # option_calibration_steps=[3]
        # option_calibration_steps=[3,4]   
        # option_calibration_steps=[3,5,6]
        option_calibration_steps=[7]






    calibration_result =lrw_fx_calibrator.calibrate_options(calibrate_on_vol=on_vol
                                                           , calibration_steps =option_calibration_steps)
                                                          

  
    repricing_results= lrw_fx_calibrator.reprice_instruments()
    # print(lrw_fx_calibrator.daily_data.ois_summary())
   
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    file_suffix = suffix + "_"+timestamp
    report_folder = Path(__file__).parent ## / "reports"
    # report_folder=r"E:\OneDrive\Dropbox\LinearRationalWishart_Work\Code\ED\LinearRationalWishart\LinearRationalWishart_NewCode\wishart_processes\results"
    report_folder =main_project_root + r"\wishart_processes\results"
    ois_file    =  "ois_report_"+suffix+"_.csv"
    option_file =  "option_report_"+suffix+"_.csv"
    model_file  =  "model_report_"+suffix+"_.csv"

    ois_file   = report_folder + f"\{ois_file}"
    option_file= report_folder + f"\{option_file}"
    model_file = report_folder + f"\{model_file}"

    lrw_fx_calibrator.write_report(calibration_result
                                    ,ois_file=ois_file
                                    ,option_file=option_file
                                    ,model_file=model_file
                                    )

    chart_folder = report_folder + f"\charts"
    lrw_fx_calibrator.create_all_plots( 
                folder=chart_folder,file_prefix=suffix,
                bonds_maturity_min = 0.0,#lrw_fx_calibrator.config.min_maturity_zc,
                bonds_maturity_max = lrw_fx_calibrator.config.max_maturity_zc,    

                options_maturity_min = lrw_fx_calibrator.config.min_maturity_option_for_chart,
                options_maturity_max = lrw_fx_calibrator.config.max_maturity_option_for_chart                
                )
   
#region Old To be removed
# def example_lrw_fx_calibration_check_correl(tenor,current_date = '20250401', suffix="calibrate_on_vol", on_vol=True, correl="YES"):
#     print("Creating base FX model and calibrator...")
    
#     print("Getting market data")
#     test_fx_data = get_testing_excel_data(current_date = current_date #'20250530'
#                                         , ccy_pair = "EURUSD" )
#     fx_spot = test_fx_data.fx_spot ##get_spot_rate("EURUSD")
#     print("Creating base FX model...")
#     lrw_fx_model =create_base_fx_model(fx_spot)

#     zc_mat_min=3.0#4.0#2.0
    
#     print("Creating calibration configuration...")
#     calibration_config= CalibrationConfig()
#     calibration_config.calibrate_based_on_correlation=True
#     if on_vol:
#         calibration_config.calibrate_on_vol=True
#     else:
#        calibration_config.calibrate_on_vol=False

#     # optimization_method: OptimizationMethod = OptimizationMethod.HYBRID
#     if tenor=="ALL":
#         calibration_config.min_maturity_option = zc_mat_min +1# 2.0
#         calibration_config.max_maturity_option =7.0  # Maximum maturity for calibration
#         suffix += "_all_tenors"
#     else:
#         calibration_config.min_maturity_option = float(tenor)  # Minimum maturity for calibration
#         calibration_config.max_maturity_option = float(tenor)  # Maximum maturity for calibration
#         suffix += f"_{tenor}"

    
#     print("Creating LRW FX calibrator...")
#     lrw_fx_calibrator =LrwFxCalibrator(test_fx_data,lrw_fx_model,calibration_config)
#     # return 
#     print("creating calibration boundary models...")
#     lrwfx1_LB, lrwfx1_UB =create_default_boudary_fx_model(fx_spot) 

#     print("setting calibration boundary models...")
#     lrw_fx_calibrator.set_boundary_models(lrwfx1_LB, lrwfx1_UB)

#     print("setting alpha...")
#     lrw_fx_calibrator.set_alpha(alpha_tenor=lrw_fx_calibrator.config.max_maturity_option)

#     # lrw_fx_calibrator.model.temporary_set_model_params()

    
#     print("Repricing bonds...")
#     lrw_fx_calibrator.reprice_bonds()

#     print(" Calibration starting point")
#     lrw_fx_calibrator.model.print_model()
        
#     print("Calibration on OIS Zero coupon bond ") 
#     sigma= get_initial_sigma(lrw_fx_calibrator.model.x0,lrw_fx_calibrator.model.u_i,lrw_fx_calibrator.model.u_j,initial_vol=0.09)#85 )
#     x0=lrw_fx_calibrator.model.x0
#     m= lrw_fx_calibrator.model.m
#     omega= lrw_fx_calibrator.model.omega

#     x0    =  np.array([[1.7923196952996328, -0.28608152672928666], [-0.28608152672928666, 0.9597425644752376]])
#     omega =  np.array([[0.03696160867402621, -0.0533663275832095], [-0.0533663275832095, 0.1350820488602678]])
#     m     =  np.array([[-0.012507429102419991, 0.0], [0.0, -0.050797316415513376]])
    
#     sigma     =  np.array([[0.03238411, 0.0], [0.0,0.03632396]])     
#     sigma= 2.0*sigma
#     rho_sigma=-0.8969805 
#     rho_x0=-0.5
#     rho_omega=-0.5
#     correls=[-0.8,-0.75,-0.5,0,0.5,0.75,0.8]
#     sigma_muls=[1.0,1.5,2.0]##0.1,0.5,0.8,1.0,2,3.0]
#     sigma_init=sigma
#     # for sigma_mul  in sigma_muls:
#     #     sigma= sigma_mul*sigma_init
#     #     correl=sigma_mul
#     for correl  in correls:
#         # rho_x0=correl/10 # Set correlation to 0.5
#         rho_omega=-correl/10
#         rho_sigma=correl
#         print(f"Running testing with parameters {correl}...")

#         x0[0,1]=x0[1,0]=rho_x0*math.sqrt(x0[0,0]*x0[1,1]) # Set correlation to 0.5)
#         omega[0,1]=omega[1,0]=rho_omega*math.sqrt(omega[0,0]*omega[1,1]) # Set correlation to 0.5)
#         sigma[0,1]=sigma[1,0]=rho_sigma*math.sqrt(sigma[0,0]*sigma[1,1]) # Set correlation to 0.5)

#         print("set sigma based on initial parameters...")
#         print(f"Initial sigma={sigma}")
    
#         lrw_fx_calibrator.model.set_model_params(2,x0
#                                                 ,omega
#                                                 ,m
#                                                 ,sigma
#                                                 ,lrw_fx_calibrator.model.alpha_i
#                                                 ,lrw_fx_calibrator.model.u_i
#                                                 ,lrw_fx_calibrator.model.alpha_j
#                                                 ,lrw_fx_calibrator.model.u_j
#                                                 ,fx_spot=fx_spot)
#         #########################################################################
   
#         lrw_fx_calibrator.model.print_model()
    
   
#         repricing_results= lrw_fx_calibrator.reprice_instruments()
#         # print(lrw_fx_calibrator.daily_data.ois_summary())
   
#         report_folder = Path(__file__).parent / "reports"
#         # report_folder=r"E:\OneDrive\Dropbox\LinearRationalWishart_Work\Code\ED\LinearRationalWishart\LinearRationalWishart_NewCode\wishart_processes\results"
#         report_folder =main_project_root + "\wishart_processes\results"
#         ois_file    =  "ois_report_"+suffix+"_.csv"
#         option_file =  "option_report_"+suffix+"_.csv"
#         model_file  =  "model_report_"+suffix+"_.csv"

#         ois_file   = report_folder + "/"+ ois_file   
#         option_file= report_folder + "/"+ option_file
#         model_file = report_folder + "/"+ model_file 
    
#         calibration_result=CalibrationResult()

#         lrw_fx_calibrator.write_report(calibration_result
#                                         ,ois_file=ois_file
#                                         ,option_file=option_file
#                                         ,model_file=model_file
#                                         )

#         chart_folder = report_folder + "/charts"
#         # lrw_fx_calibrator.create_all_plots( 
#         #             folder=chart_folder,
#         #             bonds_maturity_min = zc_mat_min,
#         #             bonds_maturity_max = 11.0,

#         #             options_maturity_min = lrw_fx_calibrator.config.min_maturity_option,
#         #             options_maturity_max =lrw_fx_calibrator.config.max_maturity_option                
#         #             )
#         lrw_fx_calibrator.create_all_options_plots(folder=chart_folder, file_prefix=f"correl_{correl}_",
#                                             maturity_min = lrw_fx_calibrator.config.min_maturity_option,
#                                             maturity_max =lrw_fx_calibrator.config.max_maturity_option                
#                                             )
#endregion


def restart_jax():
    """Force restart JAX backend to clear all memory"""
    try:
        # Clear all caches
        jax.clear_caches()
        
        # Force garbage collection
        gc.collect()
        
        # Try to reset the backend (this might not work in all versions)
        if hasattr(jax._src.lib, 'xla_bridge'):
            jax._src.lib.xla_bridge.get_backend().shutdown()
    except:
        pass

if __name__=="__main__":



    
    print("Testing calibration and pricing functionality...")
    # tenors=[1, 2, 3, 4, 5, 7, 10,  "ALL"] # in years
    # tenors=[5]# 2, 3, 4, 5, 7, 10,  "ALL"] # in years
    # for tenor in tenors: ##range(1):
    #     print(f"================tenor {tenor}=======================")
    #     # example_lrw_fx_calibration(suffix="calibrate_on_vol", on_vol=True)
    #     try:
    #         suffix="calibrate_on_price"
    #         suffix += f"_{tenor}"
    #         # print(f"Running calibration for tenor {tenor} with suffix {suffix}")
    #         example_lrw_fx_calibration(tenor,suffix="calibrate_on_price", on_vol=False)        
    #     except :
    #         pass

    # # example_lrw_fx_calibration(suffix="calibrate_on_price", on_vol=False)
    # # # example_lrw_fx_calibration(suffix="calibrate_on_vol", on_vol=True)
    # # # example_lrw_fx_calibration(suffix="calibrate_on_price", on_vol=False)

   
    
    # Set XLA to release memory back to the system
    os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.7'  # Use max 70% of memory

    # Optional: Preallocate memory to avoid fragmentation
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

    # tenors = [ "ALL"]#, 3.0, 4.0]#1, 2, 3, 4, 5, 7, 10, "ALL"]
    tenors = [1.0, 2, 3, 4, 5, 7, 10, "ALL"]
    current_dates = ['20250314' ,'20250401','20250415','20250516','20250530']
    # current_dates = ['20250314', '20250530']##'20250314']#,'20250401','20250415','20250516','20250530']


    option_calibration_steps=[]
    # option_calibration_steps.append([1])## Seems ok for tenor 4.0 with start correl  rho_sigma,rho_x0,rho_omega=0.25,-0.25,0.25
    # option_calibration_steps.append([1,2]) 
    # option_calibration_steps.append([3])
    # option_calibration_steps.append([3,4])
    # option_calibration_steps.append([3,5,6])    ## Seems ok for tenor 4.0 with start correl  rho_sigma,rho_x0,rho_omega=0.25,-0.25,0.25 uniform weigh
    option_calibration_steps.append([7])        ## Seems ok for tenor 4.0 with start correl  rho_sigma,rho_x0,rho_omega=0.25,-0.25,0.25 uniform weigh
    # option_calibration_steps.append([8])        ## Seems ok for tenor 4.0 with start correl  rho_sigma,rho_x0,rho_omega=0.25,-0.25,0.25 uniform weigh


    # for option in option_calibration_steps:
    #     print(option)
    calib_option_nb=0
    calib_on_vol=True
    for tenor in tenors:
        print(f"================tenor {tenor}=======================")
        for current_date in current_dates:   
            calib_option_nb=0
            for option_calibration_step in option_calibration_steps:
                calib_option_nb+=1
                try:
                    print(f"Running calibration for tenor {tenor} on date {current_date}")
                    if calib_on_vol:
                        suffix = f"calibrate_on_vol_{tenor}_"
                        # suffix = f"calibrate_on_vol_{tenor}_{current_date}_Option_step_{calib_option_nb}"
                    else:
                        suffix = f"calibrate_on_price_{tenor}_"
                        # suffix = f"calibrate_on_price_{tenor}_{current_date}_Option_step_{calib_option_nb}"
                    print(f"Running calibration for tenor {tenor} on date {current_date} with option calibration steps {option_calibration_step}")
                    example_lrw_fx_calibration(tenor,current_date = current_date, suffix=suffix, on_vol=calib_on_vol, option_calibration_steps=option_calibration_step)
                    # # # example_lrw_fx_calibration(tenor,current_date = current_date, suffix="calibrate_on_price", on_vol=True,correl="No")
               
                    restart_jax()
               
                except Exception as e:
                    print(f"Error in tenor {tenor}: {e}")
                finally:
                    jax.clear_caches()
                    gc.collect()
    no_display()

     # example_lrw_fx_calibration_check_correl(tenor,current_date = current_date, suffix="calibrate_on_price", on_vol=False,correl="Yes")

                # example_lrw_fx_calibration(tenor, suffix="calibrate_on_price", on_vol=False)
                # example_lrw_fx_calibration(tenor, suffix="calibrate_on_price", on_vol=True, correl="No")
                # example_lrw_fx_calibration(tenor, suffix="calibrate_on_price", on_vol=False, correl="No")

    ## for 20250530 curve model from the file and sigma =[0.12033281 0.1614329 ] with correl rho_sigma,rho_x0,rho_omega=0.25,-0.25,0.25 for tenor =4.0--> Good result
    ## other sigma and correl [0.11491546 0.19085651 0.25  
    # case option_calib_set --> 7
    #--Resutl  x0[0,1]: -0.250000 ? -0.211286 (+15.49%)
      # omega[0,1]: 0.250000 ? -0.228871 (-191.55%)
      # sigma[0,0]: 0.344538 ? 0.093263 (-72.93%)
      # sigma[1,1]: 0.344538 ? 0.179188 (-47.99%)
      # sigma[0,1]: 0.250000 ? 0.250000 (-0.00%)
