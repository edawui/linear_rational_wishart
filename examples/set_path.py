import os
import sys
from pathlib import Path

# Just add this at the top of your notebooks
def setup_paths():
    if 'google.colab' in sys.modules:
        from google.colab import drive
        drive.mount('/content/drive')
        project_root = "/content/drive/MyDrive/LinearRationalWishart_Work/Code/ED/LinearRationalWishart/LinearRationalWishart_NewCode"
    else:
        project_root = r"E:\OneDrive\Dropbox\LinearRationalWishart_Work\Code\ED\LinearRationalWishart\LinearRationalWishart_NewCode"
    
    mkt_data_folder = os.path.join(project_root, "wishart_processes", "mkt_data", "Data_new")
    os.makedirs(project_root, exist_ok=True)
    os.makedirs(mkt_data_folder, exist_ok=True)
    os.chdir(project_root)
    
    return project_root, mkt_data_folder

# project_root, mkt_data_folder = setup_paths()

# import os
# import sys

def complete_setup():
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
    
    # 5. NOW import constants (will read env vars)
    import constants
    
    print(f"✅ Setup complete - constants.project_root: {constants.project_root}")
    
    return project_root, mkt_data_folder

# Run this FIRST, before importing any other modules
# project_root, mkt_data_folder = complete_setup()

# NOW import your other modules (they'll see the correct constants)
# import your_other_modules