import csv
import json
import pandas as pd
import math
import json
import ast
from datetime import datetime
import matplotlib.pyplot as plt
import os
import ast

input_file=r"E:\OneDrive\Dropbox\LinearRationalWishart_Work\Code\ED\LinearRationalWishart\LinearRationalWishart_NewCode\wishart_processes\results\Result_all_tenors\model_summary_all.csv"
# output_file=r"E:\OneDrive\Dropbox\LinearRationalWishart_Work\Code\ED\LinearRationalWishart\LinearRationalWishart_NewCode\wishart_processes\results\Result_all_tenors\model_summary_all_report.json"
# folder=r"E:\OneDrive\Dropbox\LinearRationalWishart_Work\Code\ED\LinearRationalWishart\LinearRationalWishart_NewCode\wishart_processes\results\Result_all_tenors\model_summary_charts"
# format_string = "%Y-%m-%d"
# format_string = '%m/%d/%Y %H:%M'

model_report_file=r"E:\OneDrive\Dropbox\LinearRationalWishart_Work\Code\ED\LinearRationalWishart\LinearRationalWishart_NewCode\wishart_processes\results\Result_all_tenors_v2\model_summary_all_report.json"
folder           =r"E:\OneDrive\Dropbox\LinearRationalWishart_Work\Code\ED\LinearRationalWishart\LinearRationalWishart_NewCode\wishart_processes\results\Result_all_tenors_v2\model_summary_charts"
output_file=r"E:\OneDrive\Dropbox\LinearRationalWishart_Work\Code\ED\LinearRationalWishart\LinearRationalWishart_NewCode\wishart_processes\results\Result_all_tenors_v2\model_summary_all_report.json"
format_string = "%Y-%m-%d" #2025-03-14T00:00:00

def lower_tenor(tenor):
    if tenor[0]=="A":
        return 3
    else:
        return int(tenor)
 
def max_tenor(tenor):
    if tenor[0]=="A":
        return 7
    else:
        return int(tenor)

def csv_to_json():
    columns = [
                 'calib_option_tenor_min'
               , 'calib_option_tenor_max'
               , 'rmse_ois_price'
               , 'rmse_ois_yield' 
               , 'rmse_option_price'
               , 'rmse_option_vol'
               , 'calib_date' 
               , 'model_parameters'
               ]
    print(f"Converting {input_file} to {output_file}")

    data = []
    with open(input_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            model_params_str = ','.join(row[6:])
            if len(row) >= 6:
                record = {
                     'calib_zc_objective': 'Price', 
                     'calib_zc_tenor_min':3,
                     'calib_zc_tenor_max': 11,
                      columns[2]: float(row[1]),
                      columns[3]: float(row[2]),
                     'calib_option_objective': 'Vol', 
                    columns[0]: lower_tenor(row[0]),
                    columns[1]: max_tenor(row[0]),
               
                    columns[4]: float(row[3]),
                    columns[5]: float(row[4]),
                    columns[6]: row[5].strip(),
                    columns[7]: model_params_str # row[6:]  # Assuming the rest are model parameters
                }

                data.append(record)
    
    with open(output_file, 'w') as jsonfile:
        json.dump(data, jsonfile, indent=2)
    
    print(f"✅ Converted {len(data)} records")

def list_files_starting_with(  
                            folder =r"E:\OneDrive\Dropbox\LinearRationalWishart_Work\Code\ED\LinearRationalWishart\LinearRationalWishart_NewCode\wishart_processes\results\Result_all_tenors_v2"
                            ,prefix='model_report'
                            ,suffix='.csv'):

       files =[]
       for file in os.listdir(folder):
             if file.endswith(suffix) and file.startswith(prefix):
                  # files.append(file)
                file_path = os.path.join(folder, file)
                if os.path.isfile(file_path):
                    files.append(file_path)

       return files

def convert_csv_to_json_new_notok():
   model_files = list_files_starting_with()
   print(f"Found {len(model_files)} files starting with 'model_report' and ending with '.csv':")
   safe_dict = {
        "datetime": datetime,
        "__builtins__": {}
    }
   data = []
   for file in model_files:
         print(file)
         with open(file, 'r') as current_model_file:
            for row in current_model_file:
                row= row.strip()
                if row:
                    # try:
                        # model_params_record =row
                        # model_params_record =ast.literal_eval(row)
                        model_params_record =eval(row, safe_dict)
                        data.append(model_params_record)
                        print(row)
                    # except:
                        print(f"Error parsing row")#": {row}")
   # with open(model_report_file,'w') as json_report_file:
   #       json.dump(data, json_report_file, indent=2)

def convert_csv_to_json_new():
    model_files = list_files_starting_with()
    print(f"Found {len(model_files)} files starting with 'model_report' and ending with '.csv':")
    data = []
    
    for file in model_files:
        print(file)
        with open(file, 'r') as current_model_file:
            for row in current_model_file:
                row = row.strip()
                if row:
                    try:
                        # Replace datetime.datetime(2025, 5, 30, 0, 0) with "2025-05-30T00:00:00"
                        import re
                        
                        def replace_datetime(match):
                            # Extract year, month, day, hour, minute from the match
                            params = match.group(1).split(', ')
                            year = int(params[0])
                            month = int(params[1])
                            day = int(params[2])
                            hour = int(params[3]) if len(params) > 3 else 0
                            minute = int(params[4]) if len(params) > 4 else 0
                            
                            # Create ISO format string
                            # return f'"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:00"'
                            return f'"{year:04d}-{month:02d}-{day:02d}"'
                        
                        # Replace all datetime.datetime(...) with ISO strings
                        modified_row = re.sub(r'datetime\.datetime\(([^)]+)\)', replace_datetime, row)
                        
                        # Now parse with ast.literal_eval (safe)
                        model_params_record = ast.literal_eval(modified_row)
                        data.append(model_params_record)
                        print(f"Successfully parsed row from {file}")
                        
                    except Exception as e:
                        print(f"Error parsing row in {file}: {e}")
                        print(f"Original row: {row[:100]}...")
                        print(f"Modified row: {modified_row[:100]}...")
    
    with open(model_report_file, 'w') as json_report_file:
        json.dump(data, json_report_file, indent=2)


# Load and parse the JSON data
def load_data():
    with open(output_file, 'r') as f:
        data = json.load(f)
    
    # Parse model_parameters strings
    for record in data:
        if isinstance(record['model_parameters'], str):
            record['model_parameters'] = ast.literal_eval(record['model_parameters'])
    
    return data

# Your specific query
def get_x0_for_single_tenors( tenor, parameter, is_matrix=True, i=0 , j=0):
    data = load_data()
    
    date_list = []
    tenor_list = []
    calibrated_parameter_values = []
    tenor_min= tenor_max = tenor
    if tenor == "ALL":
        tenor_min = 3
        tenor_max = 7
    for record in data:
        # Check if tenor_min == tenor_max (single tenor)
        if (record['calib_option_tenor_min'] == tenor_min) and (record['calib_option_tenor_max'] ==tenor_max):
            # if record['calib_option_tenor_min'] == tenor:
                tenor = f"{record['calib_option_tenor_min'] }Y:{record['calib_option_tenor_max'] }Y"
                if is_matrix:
                    calibrated_parameter = record['model_parameters'][parameter][i][j]
                    if i != j:
                        calibrated_parameter_0 = record['model_parameters'][parameter][0][0]
                        calibrated_parameter_1 = record['model_parameters'][parameter][1][1]
                        calibrated_parameter =calibrated_parameter/math.sqrt(calibrated_parameter_0 * calibrated_parameter_1)
                else:
                    calibrated_parameter = record['model_parameters'][parameter]
                    # x0_00 = record['model_parameters']['x0'][0][0]  # x0[0,0]
                date_value = record['calib_date']
                datetime_object = datetime.strptime(date_value, format_string)
                date_list.append(datetime_object)
                tenor_list.append(tenor)
                calibrated_parameter_values.append(calibrated_parameter)
    
    return tenor_list, date_list, calibrated_parameter_values

# # Run the query
# tenor_list, x0_values = get_x0_for_single_tenors()
# print(f"Tenors: {tenor_list}")
# print(f"X0[0,0]: {x0_values}")

def plot_calibrated_parameter():
    # Example usage
    calibrated_tenors=[1, 2, 3, 4, 5, 7]
    calibrated_tenors=[ 2, 3, 4, 5, 7, "ALL"]
    parameter_names=    ['x0', 'omega', 'm', 'sigma', 'alpha_i', 'alpha_j']
    # parameter_names=    ['sigma','alpha_j']
    # parameter_names=    ['alpha_j']
    # parameter_names=    ['sigma']
    n=2
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    option_calibrated_paramters=['x0_0_1','omega_0_1','sigma_0_0','sigma_1_1','sigma_0_1']  
    for parameter in parameter_names:
       if parameter  in ['alpha_i', 'alpha_j']:
                is_matrix_params= False
                n=1
       else:
                is_matrix_params= True
                n=2
    
       for i in range(n):
           for j in range(i, n):                  
                 plt.figure(figsize=(12, 8))  
                 # for tenor in calibrated_ternors:
                 if is_matrix_params:
                     parameter_name= f"{parameter}_{i}_{j}"
                 else:
                     parameter_name= f"{parameter}"
                 if  parameter_name in option_calibrated_paramters:
                     for idx, tenor in enumerate(calibrated_tenors):  
                        
    
                            tenor_list, date_values, calibrated_parameter_values = get_x0_for_single_tenors(tenor, parameter=parameter, is_matrix=is_matrix_params, i=i , j=j)
                            # print(f"tenor:{tenor}, calibrated_parameter_values:{parameter_name}: {calibrated_parameter_values}")
         
                            chart_tenor =tenor_list[0]
                            color = colors[idx % len(colors)]
                            plt.plot(date_values, calibrated_parameter_values, '-o', color=color, label=f"Tenor {chart_tenor}", linewidth=2, markersize=6)
                 else:
                     tenor_list, date_values, calibrated_parameter_values = get_x0_for_single_tenors("ALL", parameter=parameter, is_matrix=is_matrix_params, i=i , j=j)
                     # print(f"calibrated_parameter_values:{parameter_name}: {calibrated_parameter_values}")
                     plt.plot(date_values, calibrated_parameter_values, '-o', color='blue', label=f"All Tenors", linewidth=2, markersize=6)
                 if i!=j:
                     # print(f"Checked i!=j: {i} == {j}")
                     print(f'Plotting {parameter_name} Correlation Evolution Across All Tenors')
                     plt.title(f'{parameter_name} Correlation Evolution Across All Tenors')
                 else:
                     # print(f"Checked i==j: {i} == {j}")
                     print(f'Plotting {parameter_name}: Evolution Across All Tenors')
                     plt.title(f'{parameter_name}: Evolution Across All Tenors')

                 plt.xlabel('Dates')
                 plt.ylabel(f'{parameter_name}')
                 plt.grid(True, alpha=0.3)
                 plt.legend()
                 plt.xticks(rotation=45)
                 plt.tight_layout()


                 # plt.ylabel(f'{parameter_name}') 
                 # plt.grid(True, alpha=0.3)
                 # plt.legend()
                 # plt.tight_layout()

                 plt.savefig(f"{folder}/{parameter_name}.png")
                 plt.close()
if __name__ == "__main__":
   
  
    # convert_csv_to_json_new()
    plot_calibrated_parameter()
  