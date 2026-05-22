

# 1. SYSTEM IMPORTS (Crucial to define 'os' and 'sys' first)
import os
import sys
import time
from os.path import sep
# Plotting
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Math + stats
import numpy as np
import pandas as pd
from scipy.stats import binom
from scipy.stats import t
import scipy.stats as stats

# Logistics
from os.path import sep
from tqdm import tqdm
import sys
import time

# Local modules
# ==============================================================================
# SYSTEM COMPATIBILITY SECTION
# ==============================================================================
# PASTE YOUR FOLDER PATH HERE (Keep the 'r' before the quotes)
# file_path_manual = r"C:\Users\CEGIS FOUNDATION\New folder\GitHub\discsim\api\utils\Pre Survey Nested Simulation"
# new loc in new lap
file_path_manual=r"C:\Users\CEGIS\Desktop\work\GitHub\discsim\api\utils\Pre Survey Nested Simulation"
# "C:\Users\CEGIS\Desktop\work\GitHub\discsim\api\utils\Pre Survey Nested Simulation"

# 1. Register the path
if os.path.exists(file_path_manual):
    if file_path_manual not in sys.path:
        sys.path.append(file_path_manual)
    print(f"✅ Path Registered: {file_path_manual}")
else:
    print(f"❌ ERROR: Path not found! Please check the folder path.")

# 2. Import Local Modules using the correct subfolder names
try:
    # We must import from the 'ecd_nested_simulation_functions' subfolder
    import ecd_nested_simulation_functions.generate_ecd_dummy_data as generate_ecd_dummy_data
    import ecd_nested_simulation_functions.ecd_sampling_strategy as ecd_sampling_strategy
    #import ecd_nested_simulation_functions.generate_ecd_dummy_data2 as generate_ecd_dummy_data2
    # Try to import the DTA extractor (it's usually in the main folder)
    try:
        from AS_extract_data_from_dta_file import extract_data_from_dta
    except ImportError:
        # Fallback to standard pandas reader if specific file is missing
        def extract_data_from_dta(path): return pd.read_stata(path)
        
    print("✅ ECD Simulation modules imported successfully.")
    
except ImportError as e:
    print(f"❌ IMPORT ERROR: {e}")
    print("Hint: Check if there is a folder named 'ecd_nested_simulation_functions' inside your manual path.")

# Enable re-load of local modules every time they are called
%load_ext autoreload
%autoreload 2
%aimport numpy 
%aimport pandas


# Loading the anthro lms who data 
# 11;15 am jan 30 friday good
import pandas as pd
import os

# 1. SETUP PATHS
# We define the base folder to keep the code clean
# BASE_DIR = r"C:\Users\CEGIS FOUNDATION\New folder\GitHub\igrowup_update"

BASE_DIR=r"C:\Users\CEGIS\Desktop\work\GitHub\igrowup_update"

def extract_data_from_dta(filename):
    """
    Helper function to join path and read Stata (.dta) files
    """
    full_path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"File not found: {full_path}")
    print(f"Loading: {filename}...")
    return pd.read_stata(full_path)

# 2. SET PARAMETERS
num_children = 10000
girl_ratio = 0.5
num_timepoints = 1
time_lags = []
min_age = 0 * 365
max_age = 5 * 365

# 3. LOAD WHO STANDARDS
print("--- Loading WHO Growth Standards ---")

# Load HAZ parameters (Length/Height-for-age)
# Corresponds to: lenanthro.dta
haz_params = extract_data_from_dta('lenanthro.dta')

# Load WAZ parameters (Weight-for-age)
# Corresponds to: weianthro.dta
waz_params = extract_data_from_dta('weianthro.dta')

# Load WHZ parameters (Weight-for-Length -> Lying)
# Corresponds to: wflanthro.dta
whz_params_lying = extract_data_from_dta('wflanthro.dta')

# Load WHZ parameters (Weight-for-Height -> Standing)
# Corresponds to: wfhanthro.dta
whz_params_standing = extract_data_from_dta('wfhanthro.dta')

print("\nSuccess! All parameters loaded.")
print(f"HAZ Shape: {haz_params.shape}")
print(f"WAZ Shape: {waz_params.shape}")



import pandas as pd
import numpy as np
import time
import os
from tqdm import tqdm
import ecd_nested_simulation_functions.generate_ecd_dummy_data as generate_ecd_dummy_data

# Create the dedicated subfolder for the output files
output_dir = "Generated_Universes_10_Factory"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


N_L1_SUPERVISORS, N_L0_CENTERS_PER_L1, N_CHILDREN = 334, 25, 15

# Define your 5 Scenarios with the explicit Good/Bad naming convention
scenarios = {
    "1_Good_L0_Good_L1": {
        "stunting_under_report": 5, "bunch_factor": 0.05, 
        "copy": 5, "collusion": 0.05
    },
    "2_Bad_L0_Lazy_L1": {
        "stunting_under_report": 30, "bunch_factor": 0.20, 
        "copy": 40, "collusion": 0.50  
    },
    "3_Bad_L0_Good_L1": {
        "stunting_under_report": 10, "bunch_factor": 0.05, 
        "copy": 5, "collusion": 0.10
    },
    "4_Bad_L0_Corrupt_L1": {
        # FIXED: Capped at 33 to prevent exceeding the real_percent_underweight limit
        "stunting_under_report": 33, "bunch_factor": 0.50, 
        "copy": 50, "collusion": 0.90
    },
    "5_Normal_L0_Normal_L1": {
        "stunting_under_report": 30, "bunch_factor": 0.20, 
        "copy": 20, "collusion": 0.50
    }
}

rho_values = [0.0, 0.7]

print("INITIATING THE 10-UNIVERSE MEGA FACTORY...")
start_time = time.time()

for rho in rho_values:
    for scene_name, params in scenarios.items():
        print(f"\n=======================================================")
        print(f"Generating: {scene_name} | rho = {rho}")
        print(f"=======================================================")
        
        # Determine Measurement Error based on Scenario
        if "5_Normal" in scene_name:
            # Normal Universe: Realistic human clumsiness at ALL levels (L0, L1, L2)
            err_ht_l0, err_wt_l0 = 1.0, 0.1
            err_ht_l1, err_wt_l1 = 1.0, 0.1
            err_ht_l2, err_wt_l2 = 1.0, 0.1
        else:
            # All other Universes: Pure fraud isolation (Measurement Error OFF)
            err_ht_l0, err_wt_l0 = 0.0, 0.0
            err_ht_l1, err_wt_l1 = 0.0, 0.0
            err_ht_l2, err_wt_l2 = 0.0, 0.0
        
        common_real_params = {
            'girl_ratio': 0.5, 'min_age': 0, 'max_age': 1790, 'num_timepoints': 1, 'time_lags': [],
            'percent_stunting': 35, 'percent_underweight': 33,
            'rho': rho
        }
        
        L0_params_list, L1_params_list, L2_params_dict = generate_ecd_dummy_data.generate_nested_distortion_parameters(
            n_L1s=N_L1_SUPERVISORS, n_L0s_per_L1=N_L0_CENTERS_PER_L1,
            real_percent_stunting=35, real_percent_underweight=33,
            
            # Scenario Injections
            mean_percent_under_reporting_stunting=params["stunting_under_report"], 
            mean_percent_under_reporting_underweight=params["stunting_under_report"], 
            mean_bunch_factor_haz=params["bunch_factor"], 
            mean_bunch_factor_waz=params["bunch_factor"], 
            mean_bunch_factor_whz=params["bunch_factor"],
            
            mean_percent_copy=params["copy"], 
            mean_collusion_index=params["collusion"],
            
            # Base Variances
            sd_across_units_percent_under_reporting_stunting=2, sd_across_units_percent_under_reporting_underweight=2,
            sd_within_units_percent_under_reporting_stunting=1, sd_within_units_percent_under_reporting_underweight=1,
            sd_across_units_bunch_factor_haz=0.01, sd_across_units_bunch_factor_waz=0.01, sd_across_units_bunch_factor_whz=0.01,
            sd_within_units_bunch_factor_haz=0.01, sd_within_units_bunch_factor_waz=0.01, sd_within_units_bunch_factor_whz=0.01,
            sd_percent_copy=2, sd_collusion_index=0.02,
            
            # MEASUREMENT ERRORS (Dynamically injected)
            error_sd_height_all_L0s=err_ht_l0, error_sd_weight_all_L0s=err_wt_l0,
            error_sd_height_L1=err_ht_l1, error_sd_weight_L1=err_wt_l1, 
            error_sd_height_L2=err_ht_l2, error_sd_weight_L2=err_wt_l2, 
            
            mean_time_lag_L1=15, mean_time_lag_L2=30
        )

        np.random.seed(42) # Ensure biology is identical across matching rho scenarios
        nested_measurements = generate_ecd_dummy_data.generate_nested_measurements(
            real_params=common_real_params, L0_params_list=L0_params_list, L1_params_list=L1_params_list, L2_params_dict=L2_params_dict,
            n_L1s=N_L1_SUPERVISORS, n_L0s_per_L1=N_L0_CENTERS_PER_L1, n_children_per_L0=N_CHILDREN,
            n_children_L1=N_CHILDREN, n_children_L2=N_CHILDREN,
            haz_params=haz_params, waz_params=waz_params, whz_params_lying=whz_params_lying, whz_params_standing=whz_params_standing, make_plots=False
        )

        all_children_records = []
        for L1_id, L1_data in tqdm(nested_measurements.items(), desc="Flattening Data"):
            if L1_id == 'metadata': continue
            for L0_id, L0_data in L1_data.items():
                if L0_id == 'L1_info': continue
                df_real = L0_data['real']['data'].copy().rename(columns={'haz': 'real_haz', 'waz': 'real_waz', 'whz': 'real_whz', 'height': 'real_height', 'weight': 'real_weight', 'age': 'real_age_days'})
                df_L0 = L0_data['L0']['data'].copy().rename(columns={'haz': 'L0_haz', 'waz': 'L0_waz', 'whz': 'L0_whz', 'height': 'L0_height', 'weight': 'L0_weight'})
                df_L1 = L0_data['L1']['data'].copy().rename(columns={'haz': 'L1_haz', 'waz': 'L1_waz', 'whz': 'L1_whz', 'height': 'L1_height', 'weight': 'L1_weight', 'age': 'L1_age_days'})
                df_L2 = L0_data['L2']['data'].copy().rename(columns={'haz': 'L2_haz', 'waz': 'L2_waz', 'whz': 'L2_whz', 'height': 'L2_height', 'weight': 'L2_weight', 'age': 'L2_age_days'})
                
                merged_df = df_real[['child_id', 'gender', 'real_age_days', 'real_height', 'real_weight', 'real_haz', 'real_waz', 'real_whz', 'loh']].merge(
                    df_L0[['child_id', 'L0_height', 'L0_weight', 'L0_haz', 'L0_waz', 'L0_whz']], on='child_id', how='left').merge(
                    df_L1[['child_id', 'L1_age_days', 'L1_height', 'L1_weight', 'L1_haz', 'L1_waz', 'L1_whz']], on='child_id', how='left').merge(
                    df_L2[['child_id', 'L2_age_days', 'L2_height', 'L2_weight', 'L2_haz', 'L2_waz', 'L2_whz']], on='child_id', how='left')
                merged_df['L1_id'], merged_df['L0_id'] = L1_id, L0_id
                all_children_records.append(merged_df)

        golden_df = pd.concat(all_children_records, ignore_index=True)
        
        # Save dynamically to the new subfolder
        output_filename = os.path.join(output_dir, f'sensitivity_dataset_{scene_name}_rho_{rho}.parquet')
        golden_df.to_parquet(output_filename, engine='pyarrow')
        print(f"Saved as {output_filename}")

print(f"\nAll 10 Universes Generated in {(time.time() - start_time) / 60:.2f} minutes!")