"""
Global constants and configuration for the Wishart processes package.
"""
import jax
import os
import sys

# Enable 64-bit precision for JAX (needed for financial applications)
jax.config.update("jax_enable_x64", True)

# project_root = r"Test_E:\OneDrive\Dropbox\LinearRationalWishart_Work\Code\ED\LinearRationalWishart\LinearRationalWishart_NewCode"
# mkt_data_folder = r"Test_E:\OneDrive\Dropbox\LinearRationalWishart_Work\Code\ED\LinearRationalWishart\LinearRationalWishart_NewCode\wishart_processes\mkt_data\Data_new"



# Check for forced paths from environment
forced_root = os.environ.get('FORCE_PROJECT_ROOT')
if forced_root:
    project_root = forced_root
    mkt_data_folder = os.environ.get('FORCE_MKT_DATA_FOLDER')
else:
    # Your existing auto-detection code
    if 'google.colab' in sys.modules:
         try:
            from google.colab import drive
            drive.mount('/content/drive')
         except:
            pass
        
         project_root = "/content/drive/MyDrive/LinearRationalWishart_Work/Code/ED/LinearRationalWishart/LinearRationalWishart_NewCode"

    else:
        project_root = r"TEST E:\OneDrive\Dropbox\LinearRationalWishart_Work\Code\ED\LinearRationalWishart\LinearRationalWishart_NewCode"
    mkt_data_folder = os.path.join(project_root, "wishart_processes", "mkt_data", "Data_new")

        # ... rest of your code

# # WITH THESE:
# def _get_environment_paths():
#     """Get paths based on current environment"""
#     if 'google.colab' in sys.modules:
#         try:
#             from google.colab import drive
#             drive.mount('/content/drive')
#         except:
#             pass
        
#         project_root = "/content/drive/MyDrive/LinearRationalWishart_Work/Code/ED/LinearRationalWishart/LinearRationalWishart_NewCode"
#     else:
#         project_root = r"TEST E:\OneDrive\Dropbox\LinearRationalWishart_Work\Code\ED\LinearRationalWishart\LinearRationalWishart_NewCode"
    
#     mkt_data_folder = os.path.join(project_root, "wishart_processes", "mkt_data", "Data_new")
#     return project_root, mkt_data_folder

# project_root, mkt_data_folder = _get_environment_paths()

# def update_paths(new_project_root, new_mkt_data_folder):
#     """Update paths programmatically"""
#     global project_root, mkt_data_folder
#     project_root = new_project_root
#     mkt_data_folder = new_mkt_data_folder


# mkt_data_folder = r"C:\Users\edem_\Dropbox\LinearRationalWishart_Work\Data\Data_new"  

# Numerical constants
EPSABS = 1e-7
EPSREL = 1e-04
NMAX = 5 #10#1000#100#1000#25#50#
DEFAULT_INTEGRATION_POINTS =15#20#50## 100
DEFAULT_EPSILON = 1e-8
UR=0.5
COMPUTE_B_N=25

# Simulation constants
TIMEDECAYVOL = 0.0  # Time decay volatility factor
DEFAULT_DT = 1.0 / 360.0  # Default time step (daily)
DEFAULT_NUM_PATHS = 1000  # Default number of simulation paths

# JAX configuration
JAX_ENABLE_X64 = True
JAX_DEFAULT_DEVICE = None  # None means use default device

# Pricing constants
DEFAULT_TENOR = 1.0  # Default swap tenor in years
DEFAULT_MATURITY = 0.25  # Default option maturity in years
DEFAULT_STRIKE = 0.0  # Default ATM strike

# Calibration constants
MAX_CALIBRATION_ITER = 100
CALIBRATION_TOL = 1e-6
CALIBRATION_LEARNING_RATE = 0.1

# Model constants
DEFAULT_DIMENSION = 2  # Default matrix dimension for Wishart process

# Numerical integration constants
INTEGRATION_METHOD = "adaptive"  # adaptive, fixed, gauss-kronrod
MAX_INTEGRATION_SUBDIVISIONS = 50

# Random seed
DEFAULT_RANDOM_SEED = 42

# Plotting constants (if visualization is added)
PLOT_DPI = 150
PLOT_FIGSIZE = (10, 6)
PLOT_STYLE = "seaborn"

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Performance settings
USE_PARALLEL = False  # Whether to use parallel processing by default
CHUNK_SIZE = 1000  # Chunk size for batch processing

# Validation settings
STRICT_VALIDATION = True  # Whether to perform strict parameter validation
VALIDATION_RTOL = 1e-5  # Relative tolerance for validation
VALIDATION_ATOL = 1e-8  # Absolute tolerance for validation

# Cache settings
ENABLE_CACHING = True  # Whether to cache computed values
CACHE_SIZE = 1000  # Maximum cache size

# Export formats
SUPPORTED_EXPORT_FORMATS = ["csv", "hdf5", "pickle", "json"]
DEFAULT_EXPORT_FORMAT = "pickle"
