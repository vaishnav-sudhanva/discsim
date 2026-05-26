import os
import sys
import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
API_DIR = os.path.dirname(CURRENT_DIR)
ECD_DIR = os.path.join(API_DIR, "ecd_nested_simulation_functions")
if API_DIR not in sys.path: sys.path.append(API_DIR)
if ECD_DIR not in sys.path: sys.path.append(ECD_DIR)

from ecd_nested_simulation_functions import generate_ecd_dummy_data

def build_universe(params, task_id, output_dir):
    # =========================================================
    # FIXED: Unpack EXACTLY the keys sent by 1_generate_universes.py
    # =========================================================
    n_l1 = int(params.get("n_L1s", 334))
    n_l0 = int(params.get("n_L0s_per_L1", 25))
    n_kids = int(params.get("n_children_per_L0", 15))
    
    stunting_ur = params.get("mean_percent_under_reporting_stunting", 5.0)
    underweight_ur = params.get("mean_percent_under_reporting_underweight", 5.0)
    
    bunch_haz = params.get("mean_bunch_factor_haz", 0.05)
    bunch_waz = params.get("mean_bunch_factor_waz", 0.05)
    bunch_whz = params.get("mean_bunch_factor_whz", 0.05)
    
    copy_pct = params.get("mean_percent_copy", 5.0)
    collusion = params.get("mean_collusion_index", 0.05)
    
    err_ht = params.get("error_sd_height_all_L0s", 0.0)
    err_wt = params.get("error_sd_weight_all_L0s", 0.0)
    
    rho_val = params.get("rho", 0.7)

    # ---------------------------------------------------------
    # BULLETPROOF FILE FINDER
    # ---------------------------------------------------------
    def get_dta(file):
        who_dir = r"C:\Users\CEGIS\Documents\GitHub\discsim\may18_validata_id_dashboard\api\services\who"
        target_path = os.path.join(who_dir, file)
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"CRITICAL ERROR: Could not find {file} precisely at {target_path}")
        return pd.read_stata(target_path)
    
    haz_p, waz_p = get_dta('lenanthro.dta'), get_dta('weianthro.dta')
    whz_ly, whz_st = get_dta('wflanthro.dta'), get_dta('wfhanthro.dta')

    common_params = {
        'girl_ratio': 0.5, 'min_age': 0, 'max_age': 1790, 'num_timepoints': 1, 'time_lags': [],
        'percent_stunting': 35, 'percent_underweight': 33, 'rho': rho_val
    }

    # Pass the correctly unpacked variables into the generator
    L0_p, L1_p, L2_p = generate_ecd_dummy_data.generate_nested_distortion_parameters(
        n_L1s=n_l1, n_L0s_per_L1=n_l0, real_percent_stunting=35, real_percent_underweight=33,
        
        mean_percent_under_reporting_stunting=stunting_ur, 
        mean_percent_under_reporting_underweight=underweight_ur, 
        
        mean_bunch_factor_haz=bunch_haz, 
        mean_bunch_factor_waz=bunch_waz, 
        mean_bunch_factor_whz=bunch_whz,
        
        mean_percent_copy=copy_pct, 
        mean_collusion_index=collusion,
        
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

    np.random.seed(42)
    nested_measurements = generate_ecd_dummy_data.generate_nested_measurements(
        real_params=common_params, L0_params_list=L0_p, L1_params_list=L1_p, L2_params_dict=L2_p,
        n_L1s=n_l1, n_L0s_per_L1=n_l0, n_children_per_L0=n_kids, n_children_L1=n_kids, n_children_L2=n_kids,
        haz_params=haz_p, waz_params=waz_p, whz_params_lying=whz_ly, whz_params_standing=whz_st, make_plots=False
    )

    records = []
    for L1_id, L1_data in nested_measurements.items():
        if L1_id == 'metadata': continue
        for L0_id, L0_data in L1_data.items():
            if L0_id == 'L1_info': continue
            
            # FIXED: Restore the full raw columns from the old code
            df_real = L0_data['real']['data'].copy().rename(columns={'haz': 'real_haz', 'waz': 'real_waz', 'whz': 'real_whz', 'height': 'real_height', 'weight': 'real_weight', 'age': 'real_age_days'})
            df_L0 = L0_data['L0']['data'].copy().rename(columns={'haz': 'L0_haz', 'waz': 'L0_waz', 'whz': 'L0_whz', 'height': 'L0_height', 'weight': 'L0_weight'})
            df_L1 = L0_data['L1']['data'].copy().rename(columns={'haz': 'L1_haz', 'waz': 'L1_waz', 'whz': 'L1_whz', 'height': 'L1_height', 'weight': 'L1_weight', 'age': 'L1_age_days'})
            df_L2 = L0_data['L2']['data'].copy().rename(columns={'haz': 'L2_haz', 'waz': 'L2_waz', 'whz': 'L2_whz', 'height': 'L2_height', 'weight': 'L2_weight', 'age': 'L2_age_days'})
            
            merged = df_real[['child_id', 'gender', 'real_age_days', 'real_height', 'real_weight', 'real_haz', 'real_waz', 'real_whz', 'loh']].merge(
                df_L0[['child_id', 'L0_height', 'L0_weight', 'L0_haz', 'L0_waz', 'L0_whz']], on='child_id', how='left').merge(
                df_L1[['child_id', 'L1_age_days', 'L1_height', 'L1_weight', 'L1_haz', 'L1_waz', 'L1_whz']], on='child_id', how='left').merge(
                df_L2[['child_id', 'L2_age_days', 'L2_height', 'L2_weight', 'L2_haz', 'L2_waz', 'L2_whz']], on='child_id', how='left')
            
            merged['L1_id'], merged['L0_id'] = L1_id, L0_id
            records.append(merged)

    df_pop = pd.concat(records, ignore_index=True)
    child_path = os.path.join(output_dir, f"child_data_{task_id}.parquet")
    df_pop.to_parquet(child_path, engine='pyarrow')
    return child_path, df_pop