import os
import time
import random
import numpy as np
import pandas as pd

# Import your local ECD modules
import sys
# Make sure the path to the simulation functions is registered
sim_path = r"C:\Users\CEGIS\Desktop\work\GitHub\discsim\api\utils\Pre Survey Nested Simulation"
if sim_path not in sys.path:
    sys.path.append(sim_path)

import ecd_nested_simulation_functions.generate_ecd_dummy_data as generate_ecd_dummy_data

# NOTE: You must load your WHO DTA tables (haz_params, etc.) here or inside the function!
# Assuming they are loaded or available via your AS_extract_data_from_dta_file utility.

def build_new_universe(output_dir, scene_name, n_l1, n_l0, n_kids, 
                       stunting_fraud, bunch_factor, copy_pct, collusion, 
                       err_ht, err_wt, rho_val):
    
    start_time = time.time()
    
    # 1. Base Parameters
    common_real_params = {
        'girl_ratio': 0.5, 'min_age': 0, 'max_age': 1790, 'num_timepoints': 1, 'time_lags': [],
        'percent_stunting': 35, 'percent_underweight': 33, 'rho': rho_val
    }
    
    np.random.seed(42)
    random.seed(42)

    # 2. Get Instructions
    L0_p, L1_p, L2_p = generate_ecd_dummy_data.generate_nested_distortion_parameters(
        n_L1s=n_l1, n_L0s_per_L1=n_l0,
        real_percent_stunting=35, real_percent_underweight=33,
        mean_percent_under_reporting_stunting=stunting_fraud, 
        mean_percent_under_reporting_underweight=stunting_fraud, 
        mean_bunch_factor_haz=bunch_factor, mean_bunch_factor_waz=bunch_factor, mean_bunch_factor_whz=bunch_factor,
        mean_percent_copy=copy_pct, mean_collusion_index=collusion,
        sd_across_units_percent_under_reporting_stunting=2, sd_across_units_percent_under_reporting_underweight=2,
        sd_within_units_percent_under_reporting_stunting=1, sd_within_units_percent_under_reporting_underweight=1,
        sd_across_units_bunch_factor_haz=0.01, sd_across_units_bunch_factor_waz=0.01, sd_across_units_bunch_factor_whz=0.01,
        sd_within_units_bunch_factor_haz=0.01, sd_within_units_bunch_factor_waz=0.01, sd_within_units_bunch_factor_whz=0.01,
        sd_percent_copy=2, sd_collusion_index=0.02,
        error_sd_height_all_L0s=err_ht, error_sd_weight_all_L0s=err_wt,
        error_sd_height_L1=err_ht, error_sd_weight_L1=err_wt, 
        error_sd_height_L2=err_ht, error_sd_weight_L2=err_wt, 
        mean_time_lag_L1=15, mean_time_lag_L2=30
    )

    # 3. Generate Data (Assuming haz_params are globally available from your environment)
    nested_measurements = generate_ecd_dummy_data.generate_nested_measurements(
        real_params=common_real_params, L0_params_list=L0_p, L1_params_list=L1_p, L2_params_dict=L2_p,
        n_L1s=n_l1, n_L0s_per_L1=n_l0, n_children_per_L0=n_kids, n_children_L1=n_kids, n_children_L2=n_kids,
        haz_params=haz_params, waz_params=waz_params, whz_params_lying=whz_params_lying, whz_params_standing=whz_params_standing, make_plots=False
    )

    # 4. Flatten Data
    all_children_records = []
    for L1_id, L1_data in nested_measurements.items():
        if L1_id == 'metadata': continue
        for L0_id, L0_data in L1_data.items():
            if L0_id == 'L1_info': continue
            
            df_real = L0_data['real']['data'].copy().rename(columns={'haz': 'real_haz', 'waz': 'real_waz', 'whz': 'real_whz', 'height': 'real_height', 'weight': 'real_weight', 'age': 'real_age_days'})
            df_L0 = L0_data['L0']['data'].copy().rename(columns={'haz': 'L0_haz', 'waz': 'L0_waz', 'whz': 'L0_whz', 'height': 'L0_height', 'weight': 'L0_weight'})
            df_L1 = L0_data['L1']['data'].copy().rename(columns={'haz': 'L1_haz', 'waz': 'L1_waz', 'whz': 'L1_whz', 'height': 'L1_height', 'weight': 'L1_weight', 'age': 'L1_age_days'})
            df_L2 = L0_data['L2']['data'].copy().rename(columns={'haz': 'L2_haz', 'waz': 'L2_waz', 'whz': 'L2_whz', 'height': 'L2_height', 'weight': 'L2_weight', 'age': 'L2_age_days'})
            
            merged_df = df_real[['child_id', 'gender', 'real_age_days', 'real_height', 'real_weight', 'real_haz', 'real_waz']].merge(
                df_L0[['child_id', 'L0_height', 'L0_weight', 'L0_haz', 'L0_waz']], on='child_id', how='left').merge(
                df_L1[['child_id', 'L1_age_days', 'L1_height', 'L1_weight', 'L1_haz', 'L1_waz']], on='child_id', how='left').merge(
                df_L2[['child_id', 'L2_age_days', 'L2_height', 'L2_weight', 'L2_haz', 'L2_waz']], on='child_id', how='left')
            
            merged_df['L1_id'] = L1_id
            merged_df['L0_id'] = L0_id
            all_children_records.append(merged_df)

    golden_df = pd.concat(all_children_records, ignore_index=True)
    
    # Save the file
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    file_path = os.path.join(output_dir, f'single_universe_{scene_name}.parquet')
    golden_df.to_parquet(file_path, engine='pyarrow')
    
    return file_path