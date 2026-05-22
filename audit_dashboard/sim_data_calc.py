

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
import os
from tqdm.auto import tqdm
from datetime import datetime

# ==============================================================================
# CHUNK 0: INITIALIZATION & CONFIGURATION
# ==============================================================================
print("BOOTING UP: Grand Master Engine (Height & Weight Processing)...")

input_dir = "Generated_Universes_10_Factory"
output_dir = "Calculated_Tracer_Metricswh"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

universe_files = {
    # Univariate (rho=0.0)
    "1_Good_L0_Good_L1 (rho=0.0)": f"{input_dir}/sensitivity_dataset_1_Good_L0_Good_L1_rho_0.0.parquet",
    "2_Bad_L0_Lazy_L1 (rho=0.0)": f"{input_dir}/sensitivity_dataset_2_Bad_L0_Lazy_L1_rho_0.0.parquet",
    "3_Bad_L0_Good_L1 (rho=0.0)": f"{input_dir}/sensitivity_dataset_3_Bad_L0_Good_L1_rho_0.0.parquet",
    "4_Bad_L0_Corrupt_L1 (rho=0.0)": f"{input_dir}/sensitivity_dataset_4_Bad_L0_Corrupt_L1_rho_0.0.parquet",
    "5_Normal_L0_Normal_L1 (rho=0.0)": f"{input_dir}/sensitivity_dataset_5_Normal_L0_Normal_L1_rho_0.0.parquet",
    
    # Bivariate (rho=0.7)
    "1_Good_L0_Good_L1 (rho=0.7)": f"{input_dir}/sensitivity_dataset_1_Good_L0_Good_L1_rho_0.7.parquet",
    "2_Bad_L0_Lazy_L1 (rho=0.7)": f"{input_dir}/sensitivity_dataset_2_Bad_L0_Lazy_L1_rho_0.7.parquet",
    "3_Bad_L0_Good_L1 (rho=0.7)": f"{input_dir}/sensitivity_dataset_3_Bad_L0_Good_L1_rho_0.7.parquet",
    "4_Bad_L0_Corrupt_L1 (rho=0.7)": f"{input_dir}/sensitivity_dataset_4_Bad_L0_Corrupt_L1_rho_0.7.parquet",
    "5_Normal_L0_Normal_L1 (rho=0.7)": f"{input_dir}/sensitivity_dataset_5_Normal_L0_Normal_L1_rho_0.7.parquet"
}

TARGET_L1_COUNT = 100
TARGET_L0_PERCENTILE = 0.30 
N_SIMULATIONS = 1           

MAX_L1_CLINICS = 25
MAX_KIDS_PER_CLINIC = 15
TOTAL_L1_KIDS = MAX_L1_CLINICS * MAX_KIDS_PER_CLINIC  

PERCENTAGES = [0.20, 0.40, 0.60, 0.80, 1.00]
master_database = []

# Define the columns we are looking for based on the Indicator
INDICATORS = {
    'Height': {'l0': 'L0_haz', 'l1': 'L1_haz', 'l2': 'L2_haz', 'real': 'real_haz'},
    'Weight': {'l0': 'L0_waz', 'l1': 'L1_waz', 'l2': 'L2_waz', 'real': 'real_waz'}
}

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def calculate_metrics(df, group_cols, col_meas, col_baseline):
    temp_df = df.copy()
    temp_df['_err'] = np.abs(temp_df[col_meas] - temp_df[col_baseline])
    temp_df['_sq_err'] = temp_df['_err'] ** 2
    
    agg = temp_df.groupby(group_cols).agg(
        MAE=('_err', 'mean'),
        RMSE=('_sq_err', lambda x: np.sqrt(np.mean(x))),
        P90=('_err', lambda x: np.quantile(x, 0.90))
    ).reset_index()
    return agg

def get_top_k_overlap(ranked_df, god_set, rank_col, target_k, tie_breaker_col='L1_id'):
    ranked_df = ranked_df.sort_values(by=[rank_col, tie_breaker_col], ascending=[False, True]).reset_index(drop=True)
    caught_set = set(ranked_df.head(target_k)[tie_breaker_col])
    if target_k == 0: return 0.0
    return (len(caught_set & god_set) / target_k) * 100

def generate_dynamic_strategies(budget, max_c, max_k, target_qty=6):
    strats = []
    min_k = max(1, int(np.floor(budget / max_c)))
    max_possible_k = min(max_k, budget)
    
    if min_k > max_possible_k: return [(max_c, max_k)]
        
    raw_ks = np.unique(np.round(np.linspace(min_k, max_possible_k, target_qty * 2)).astype(int))
    for k in raw_ks:
        c = int(np.round(budget / k))
        if 1 <= c <= max_c and c * k <= max_c * max_k:
            if (c, k) not in strats: strats.append((c, k))
                
    strats = sorted(strats, key=lambda x: x[0], reverse=True)
    if len(strats) > target_qty:
        indices = np.round(np.linspace(0, len(strats)-1, target_qty)).astype(int)
        strats = [strats[i] for i in indices]
    while 0 < len(strats) < target_qty:
        strats.append(strats[-1])
    return strats
# ==============================================================================
# MAIN ENGINE LOOP (OPTIMIZED)
# ==============================================================================
for uni_name, file_name in universe_files.items():
    if not os.path.exists(file_name):
        print(f"Skipping {uni_name}: '{file_name}' not found.")
        continue
        
    print(f"\nPROCESSING UNIVERSE: {uni_name.upper()}")
    df_pop = pd.read_parquet(file_name)
    
    # --------------------------------------------------------------------------
    # CHUNK 1: PRE-CALCULATE ALL "GOD MODES" FOR BOTH HEIGHT & WEIGHT
    # --------------------------------------------------------------------------
    god_data = {'Height': {}, 'Weight': {}}
    
    for ind_name, cols in INDICATORS.items():
        god_l1 = calculate_metrics(df_pop, ['L1_id'], cols['l0'], cols['real'])
        god_data[ind_name]['L1_MAE'] = set(god_l1.sort_values(by=['MAE', 'L1_id'], ascending=[False, True]).head(TARGET_L1_COUNT)['L1_id'])
        god_data[ind_name]['L1_RMSE'] = set(god_l1.sort_values(by=['RMSE', 'L1_id'], ascending=[False, True]).head(TARGET_L1_COUNT)['L1_id'])
        god_data[ind_name]['L1_P90'] = set(god_l1.sort_values(by=['P90', 'L1_id'], ascending=[False, True]).head(TARGET_L1_COUNT)['L1_id'])
        
        god_l0 = calculate_metrics(df_pop, ['L1_id', 'L0_id'], cols['l0'], cols['real'])
        l0_targets = god_l0.groupby('L1_id').size().apply(lambda x: int(np.floor(x * TARGET_L0_PERCENTILE))).to_dict()
        
        god_data[ind_name]['L0_Targets'] = l0_targets
        god_data[ind_name]['L0_Sets'] = {'MAE': {}, 'RMSE': {}, 'P90': {}}
        
        for l1_id in god_l0['L1_id'].unique():
            target = l0_targets.get(l1_id, 0)
            l1_subset = god_l0[god_l0['L1_id'] == l1_id]
            god_data[ind_name]['L0_Sets']['MAE'][l1_id] = set(l1_subset.sort_values(by=['MAE', 'L0_id'], ascending=[False, True]).head(target)['L0_id'])
            god_data[ind_name]['L0_Sets']['RMSE'][l1_id] = set(l1_subset.sort_values(by=['RMSE', 'L0_id'], ascending=[False, True]).head(target)['L0_id'])
            god_data[ind_name]['L0_Sets']['P90'][l1_id] = set(l1_subset.sort_values(by=['P90', 'L0_id'], ascending=[False, True]).head(target)['L0_id'])

    # --------------------------------------------------------------------------
    # CHUNK 2: MULTI-LEVEL SAMPLING
    # --------------------------------------------------------------------------
    for l1_pct in tqdm(PERCENTAGES, desc=f"   Sweeping L1 Budgets", leave=False):
        l1_budget = int(TOTAL_L1_KIDS * l1_pct)
        l1_strategies = generate_dynamic_strategies(l1_budget, MAX_L1_CLINICS, MAX_KIDS_PER_CLINIC, 6)
        
        for l1_c, l1_k in l1_strategies:
            for sim_id in range(N_SIMULATIONS):
                
                # 1. Sample L1 Spreadsheet 
                l1_clinics = df_pop[['L1_id', 'L0_id']].drop_duplicates().groupby('L1_id').sample(n=l1_c, replace=False)
                df_l1_sheet = df_pop.merge(l1_clinics, on=['L1_id', 'L0_id']).groupby(['L1_id', 'L0_id']).sample(n=l1_k, replace=False)
                
                # PRE-CALCULATE L1 METRICS (Do this ONLY ONCE per L1 sample)
                l1_eval = {'Height': {}, 'Weight': {}}
                
                for ind_name, cols in INDICATORS.items():
                    # Spreadsheet truth for L2 comparison
                    sheet_truth = calculate_metrics(df_l1_sheet, ['L1_id'], cols['l1'], cols['real'])
                    l1_eval[ind_name]['sheet_MAE'] = set(sheet_truth.sort_values(by=['MAE', 'L1_id'], ascending=[False, True]).head(TARGET_L1_COUNT)['L1_id'])
                    l1_eval[ind_name]['sheet_RMSE'] = set(sheet_truth.sort_values(by=['RMSE', 'L1_id'], ascending=[False, True]).head(TARGET_L1_COUNT)['L1_id'])
                    l1_eval[ind_name]['sheet_P90'] = set(sheet_truth.sort_values(by=['P90', 'L1_id'], ascending=[False, True]).head(TARGET_L1_COUNT)['L1_id'])

                    # V1 Accuracy
                    l1_diagnosis = calculate_metrics(df_l1_sheet, ['L1_id'], cols['l1'], cols['l0'])
                    l1_eval[ind_name]['v1_mae_acc'] = get_top_k_overlap(l1_diagnosis, god_data[ind_name]['L1_MAE'], 'MAE', TARGET_L1_COUNT, 'L1_id')
                    l1_eval[ind_name]['v1_rmse_acc'] = get_top_k_overlap(l1_diagnosis, god_data[ind_name]['L1_RMSE'], 'RMSE', TARGET_L1_COUNT, 'L1_id')
                    l1_eval[ind_name]['v1_p90_acc'] = get_top_k_overlap(l1_diagnosis, god_data[ind_name]['L1_P90'], 'P90', TARGET_L1_COUNT, 'L1_id')
                    
                    # V2 Accuracy
                    l1_vs_l0_clinic = calculate_metrics(df_l1_sheet, ['L1_id', 'L0_id'], cols['l1'], cols['l0'])
                    v2_acc = {'MAE': [], 'RMSE': [], 'P90': []}
                    
                    # Group by dict for lightning fast Pandas lookups
                    grouped_l1_vs_l0 = dict(tuple(l1_vs_l0_clinic.groupby('L1_id')))
                    
                    for l1_id in df_pop['L1_id'].unique():
                        targ = god_data[ind_name]['L0_Targets'].get(l1_id, 0)
                        if targ == 0: continue
                        
                        if l1_id in grouped_l1_vs_l0:
                            subset_l1 = grouped_l1_vs_l0[l1_id]
                            l1_caught_mae = set(subset_l1.sort_values(by=['MAE', 'L0_id'], ascending=[False, True]).head(targ)['L0_id'])
                            l1_caught_rmse = set(subset_l1.sort_values(by=['RMSE', 'L0_id'], ascending=[False, True]).head(targ)['L0_id'])
                            l1_caught_p90 = set(subset_l1.sort_values(by=['P90', 'L0_id'], ascending=[False, True]).head(targ)['L0_id'])
                        else:
                            l1_caught_mae, l1_caught_rmse, l1_caught_p90 = set(), set(), set()
                        
                        v2_acc['MAE'].append(len(l1_caught_mae & god_data[ind_name]['L0_Sets']['MAE'].get(l1_id, set())) / targ)
                        v2_acc['RMSE'].append(len(l1_caught_rmse & god_data[ind_name]['L0_Sets']['RMSE'].get(l1_id, set())) / targ)
                        v2_acc['P90'].append(len(l1_caught_p90 & god_data[ind_name]['L0_Sets']['P90'].get(l1_id, set())) / targ)
                    
                    l1_eval[ind_name]['v2_mae_acc'] = np.mean(v2_acc['MAE'])*100 if v2_acc['MAE'] else 0
                    l1_eval[ind_name]['v2_rmse_acc'] = np.mean(v2_acc['RMSE'])*100 if v2_acc['RMSE'] else 0
                    l1_eval[ind_name]['v2_p90_acc'] = np.mean(v2_acc['P90'])*100 if v2_acc['P90'] else 0

                # --------------------------------------------------------------
                # CHUNK 3: L2 AUDIT (Fast inner loop)
                # --------------------------------------------------------------
                for l2_pct in PERCENTAGES:
                    l2_budget = int((l1_c * l1_k) * l2_pct)
                    l2_strategies = generate_dynamic_strategies(l2_budget, max_c=l1_c, max_k=l1_k, target_qty=6)
                    
                    for l2_c, l2_k in l2_strategies:
                        l2_clinics = df_l1_sheet[['L1_id', 'L0_id']].drop_duplicates().groupby('L1_id').sample(n=l2_c, replace=False)
                        df_l2_audit = df_l1_sheet.merge(l2_clinics, on=['L1_id', 'L0_id']).groupby(['L1_id', 'L0_id']).sample(n=l2_k, replace=False)
                        
                        for ind_name, cols in INDICATORS.items():
                            # V3 Accuracy
                            l2_vs_l1 = calculate_metrics(df_l2_audit, ['L1_id'], cols['l2'], cols['l1'])
                            
                            v3_mae_acc = get_top_k_overlap(l2_vs_l1, l1_eval[ind_name]['sheet_MAE'], 'MAE', TARGET_L1_COUNT, 'L1_id')
                            v3_rmse_acc = get_top_k_overlap(l2_vs_l1, l1_eval[ind_name]['sheet_RMSE'], 'RMSE', TARGET_L1_COUNT, 'L1_id')
                            v3_p90_acc = get_top_k_overlap(l2_vs_l1, l1_eval[ind_name]['sheet_P90'], 'P90', TARGET_L1_COUNT, 'L1_id')

                            master_database.append({
                                'Universe': uni_name,
                                'Indicator': ind_name,
                                'Sim_ID': sim_id,
                                'L1_Budget_Pct': f"{int(l1_pct*100)}%",
                                'L1_C': l1_c, 'L1_K': l1_k, 'L1_Label': f"{l1_c}C x {l1_k}K",
                                'L2_Budget_Pct': f"{int(l2_pct*100)}%",
                                'L2_C': l2_c, 'L2_K': l2_k, 'L2_Label': f"{l2_c}C x {l2_k}K",
                                
                                'V1_MAE_Acc': l1_eval[ind_name]['v1_mae_acc'], 
                                'V1_RMSE_Acc': l1_eval[ind_name]['v1_rmse_acc'], 
                                'V1_P90_Acc': l1_eval[ind_name]['v1_p90_acc'],
                                'V2_MAE_Acc': l1_eval[ind_name]['v2_mae_acc'], 
                                'V2_RMSE_Acc': l1_eval[ind_name]['v2_rmse_acc'], 
                                'V2_P90_Acc': l1_eval[ind_name]['v2_p90_acc'],
                                'V3_MAE_Acc': v3_mae_acc, 
                                'V3_RMSE_Acc': v3_rmse_acc, 
                                'V3_P90_Acc': v3_p90_acc
                            })

print("\nCompilation complete. Building Unified Master Database...")
final_df = pd.DataFrame(master_database)

# Split out the Universe name and Rho value for Dashboard filtering
final_df[['Scenario', 'Rho_Model']] = final_df['Universe'].str.split(r' \(rho=', regex=True, expand=True)
final_df['Rho_Model'] = final_df['Rho_Model'].str.replace(')', '')

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
export_name = f"{output_dir}/Tracer_Master_DB_Height_Weight_{timestamp}.csv"
final_df.to_csv(export_name, index=False)

print("="*80)
print(f"SUCCESS! Height & Weight accurately calculated side-by-side.")
print(f"Exported {len(final_df)} combinatorial rows to: {export_name}")
print("="*80)